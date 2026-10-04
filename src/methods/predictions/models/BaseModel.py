"""BaseModel: the fit / predict / cross-validation machinery every prediction
model shares (fold generation, L1 augmentation, imputation, and the
Pool x FoldMethod segment runner). Subclasses supply fit / _predict and,
optionally, preprocessing and closed-form leave-one-out shortcuts.
"""
import copy
import os
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from tqdm import tqdm

from src.constants import (
    Fields, FoldsMethods, TEST_COLS_IN_L1, ALL_TESTS_MAX_SCORES,
    SEEN_CV_ONLY_PREFIXES, AggTypes, TRANSITIONS_FEATURE_PREFIX, WFC_FEATURE_PREFIX,
    Pool, FoldMethod, InnerValidation, INNER_KFOLD_K, INNER_HOLDOUT_FRAC
)


def _fold_tqdm(iterable, total, desc):
    """Per-fold progress bar, opt-in via PROF_FOLD_TQDM=1.

    Off by default so existing runs are byte-identical in behaviour and log
    output. When several jobs run in one process pool they all write to the
    same stream, so `desc` carries the segment id and mininterval is high --
    the point is a periodic "n/total" line in the log, not a smooth bar.
    """
    if os.environ.get("PROF_FOLD_TQDM", "") not in ("1", "true", "yes"):
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="fold",
                mininterval=30, miniters=1, dynamic_ncols=False, ncols=0)


def validate_seen_only_features(feature_cols, pool, df):
    """Hard error if seen-only features (TRANSITIONS / WFC) are used outside
    a regime that guarantees train and test share content_group.

    Only Pool.SEEN guarantees same-context train/test. OneStop is detected by Fields.CONTENT_GROUP; MECO (which
    lacks it) trivially satisfies the constraint for any pool because every
    participant reads all 12 articles.
    """
    if not feature_cols:
        return
    if not any(
        any(col.startswith(prefix) for prefix in SEEN_CV_ONLY_PREFIXES)
        for col in feature_cols
    ):
        return
    is_onestop = Fields.CONTENT_GROUP in df.columns
    if not is_onestop:
        return
    if pool == Pool.SEEN:
        return
    raise ValueError(
        "TRANSITIONS / WFC features require Pool.SEEN on OneStop, "
        f"because per-text features are only comparable across participants "
        f"who read the same texts. Got pool={pool}."
    )


class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def _predict(self, X):
        pass

    def _l2_rows(self, df):
        """The L2 (non-native) rows of a training frame, or all rows when the
        cohort can't be identified.

        Mirrors RidgeRegression._fit_scaler's language mask (same column
        preference, same all-rows fallback) so both model families derive their
        imputation statistics from the same population.
        """
        native = getattr(self, "native_language", "English")
        for col in ("language", Fields.L1):
            if col in df.columns:
                mask = df[col] != native
                return df[mask] if mask.any() else df
        return df

    def prepare_training_data(self, df, feature_cols, target_col):
        """Default preprocessing for models without their own transform (the
        tree models, TabPFN, TabStar): impute NaN feature cells with the
        training fold's column means.

        Undefined cells occur in every feature set — per-text columns for text a
        participant did not read, RR for the MECO ch_s site, a POS cluster for a
        tag the participant never read. The mean is taken over L2 rows only,
        mirroring RidgeRegression._fit_scaler: Ridge centres on L2, so its
        NaN→0-after-z-score fill puts an imputed cell exactly at the L2 mean, and
        both families should impute the same value. L2 is also the population
        every test row comes from, while the L1 rows are augmentation carrying
        imputed targets.

        Ridge / LogRidge override this (they z-score + fill in `_transform`).
        """
        X = df[feature_cols]
        # `df` is the training fold, so these means are fold-scoped. The fill is
        # done on the numpy matrix: DataFrame.fillna(Series) is quadratic in the
        # column count, which is minutes per call on the ~670k-column per-text
        # frames. Same result (per-column train mean; all-NaN column -> 0.0).
        vals = X.to_numpy(dtype=np.float64, copy=True)
        self._impute_means = self._l2_rows(df)[feature_cols].mean(axis=0)  # skipna, L2 only
        col_means = self._impute_means.to_numpy(dtype=np.float64)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)  # all-NaN col -> 0.0
        nan_r, nan_c = np.nonzero(np.isnan(vals))
        vals[nan_r, nan_c] = col_means[nan_c]
        return pd.DataFrame(vals, index=X.index, columns=X.columns), df[target_col]

    def prepare_test_data(self, df, feature_cols):
        X = df[feature_cols]
        means = getattr(self, "_impute_means", None)
        if means is None:                            # non-per-text: untouched
            return X
        vals = X.to_numpy(dtype=np.float64, copy=True)
        m = means.reindex(X.columns).to_numpy(dtype=np.float64)
        m = np.where(np.isfinite(m), m, 0.0)
        nan_r, nan_c = np.nonzero(np.isnan(vals))
        vals[nan_r, nan_c] = m[nan_c]
        return pd.DataFrame(vals, index=X.index, columns=X.columns)

    def fit_df(self, df, feature_cols=None, target_col=None):
        # Target only: undefined feature cells are imputed from training-fold
        # means in prepare_training_data, so dropping on them here would delete
        # rows the model can still use (see drop_missing_and_negative_ones).
        df = df.dropna(subset=[target_col])
        df = df[df[target_col] != -1]
        self.clip_min = 0
        self.clip_max = ALL_TESTS_MAX_SCORES.get(target_col, 100)
        X, y = self.prepare_training_data(df, feature_cols, target_col)
        return self.fit(X, y)

    def predict_df(self, df, feature_cols=None):
        cols = feature_cols if feature_cols is not None else self.feature_cols
        if cols is None:
            raise ValueError("feature_cols must be provided or set during fit_df")
        X = self.prepare_test_data(df, cols)
        predictions = self._predict(X)
        return np.clip(predictions, self.clip_min, self.clip_max)

    def fit_with_augmentation(self, df_l1, df_l2, aug_precentage, feature_cols=None, target_col=None, random_state=42):
        df_l1 = self.l1_imputation(df_l1, target_col)
        if feature_cols is None or target_col is None:
            raise ValueError("feature_cols and target_col must be provided")
        if (aug_precentage < 0 or aug_precentage > 1):
            raise ValueError("aug_precentage must be in the range (0, 1]")

        # Slim both frames first, then build the combined frame via numpy stack rather than pd.concat.
        # On 100K-col per-text frames pandas concat was ~1.7s/fold (per-column block-merge overhead) and
        # left a multi-block result that made downstream df[feature_cols].to_numpy() slow too. Rebuilding
        # from a single np.ndarray gives a one-block DataFrame so all downstream slicing is cheap.
        feature_cols = list(feature_cols)
        meta_cols = [c for c in ("language", Fields.L1) if c in df_l2.columns and c not in feature_cols and c != target_col]

        if aug_precentage == 1:
            df_l1_sample = df_l1
        elif aug_precentage == 0 or aug_precentage is None:
            df_l1_sample = None
        else:
            df_l1_sample = df_l1.sample(frac=aug_precentage, replace=True, random_state=random_state)

        if df_l1_sample is None:
            feat_arr = df_l2[feature_cols].to_numpy(dtype=np.float64, copy=False)
            target_arr = df_l2[target_col].to_numpy()
            meta_arrs = {c: df_l2[c].to_numpy() for c in meta_cols if c in df_l2.columns}
        else:
            feat_arr = np.vstack([
                df_l2[feature_cols].to_numpy(dtype=np.float64, copy=False),
                df_l1_sample[feature_cols].to_numpy(dtype=np.float64, copy=False),
            ])
            target_arr = np.concatenate([df_l2[target_col].to_numpy(), df_l1_sample[target_col].to_numpy()])
            meta_arrs = {}
            for c in meta_cols:
                if c in df_l2.columns and c in df_l1_sample.columns:
                    meta_arrs[c] = np.concatenate([df_l2[c].to_numpy(), df_l1_sample[c].to_numpy()])
        df = pd.DataFrame(feat_arr, columns=feature_cols)
        df[target_col] = target_arr
        for c, arr in meta_arrs.items():
            df[c] = arr

        self.clip_min = 0
        self.clip_max = ALL_TESTS_MAX_SCORES.get(target_col, 100)
        X, y = self.prepare_training_data(df, feature_cols, target_col)
        return self.fit(X, y)

    # ----- inner-validation hyperparameter selection (tree models) -----------
    # Same default-no-op override pattern as try_loocv_segment / the scaler
    # hooks below: a model opts in by returning a non-empty grid, everything
    # else keeps the existing single-fit behaviour untouched.

    def hyperparameter_grid(self) -> list[dict]:
        """Configs to search inside each outer fold. Empty = no search.

        Ridge/LogRidge/Linear stay empty on purpose — they pick alpha via
        RidgeCV's closed-form LOO, which is free and a better estimator than an
        explicit inner split.
        """
        return []

    def apply_config(self, cfg: dict) -> None:
        """Push one grid config onto the underlying estimator.

        Works unmodified for every tree model: DecisionTreeRegressor,
        RandomForestRegressor, LGBMRegressor and XGBRegressor are all
        sklearn-compatible. This is why the tree classes need no rebuild path
        despite constructing their estimator eagerly in __init__.
        """
        self.model.set_params(**cfg)

    def _inner_splits(self, df_l2, strategy, random_state=42):
        """Yield (inner_train_idx, inner_val_idx) positional index arrays.

        Splits the OUTER fold's L2 training rows only. In fully_agg each row is
        one participant, so a row-level split is already participant-level.
        Stratified by L1 when the strata allow it, falling back to unstratified
        rather than failing — small OneStop content groups (21-28 people) can
        easily have singleton languages.
        """
        n = len(df_l2)
        strat = df_l2[Fields.L1].to_numpy() if Fields.L1 in df_l2.columns else None
        if strat is not None:
            _, counts = np.unique(strat, return_counts=True)

        if strategy == InnerValidation.HOLDOUT:
            idx = np.arange(n)
            can_strat = strat is not None and counts.min() >= 2
            try:
                tr, va = train_test_split(
                    idx, test_size=INNER_HOLDOUT_FRAC, random_state=random_state,
                    stratify=strat if can_strat else None,
                )
            except ValueError:
                tr, va = train_test_split(idx, test_size=INNER_HOLDOUT_FRAC,
                                          random_state=random_state)
            yield tr, va
            return

        k = min(INNER_KFOLD_K, n)
        can_strat = strat is not None and counts.min() >= k
        splitter = (StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
                    if can_strat else
                    KFold(n_splits=k, shuffle=True, random_state=random_state))
        y_for_split = strat if can_strat else np.zeros(n)
        for tr, va in splitter.split(np.zeros(n), y_for_split):
            yield tr, va

    @staticmethod
    def _inner_score(true, pred):
        """Pearson r over POOLED out-of-fold predictions.

        Pooled, not the mean of per-fold r: an inner 5-fold on a 22-row OneStop
        segment leaves 4-5 validation rows per fold, where a per-fold
        correlation is noise. Degenerate cases (constant predictions, n < 2)
        score -1.0 so they lose to anything real rather than propagating NaN.
        """
        if len(true) < 2:
            return -1.0
        sd_t, sd_p = np.std(true), np.std(pred)
        if not np.isfinite(sd_t) or not np.isfinite(sd_p) or sd_t == 0 or sd_p == 0:
            return -1.0
        r = float(np.corrcoef(true, pred)[0, 1])
        return r if np.isfinite(r) else -1.0

    def fit_with_augmentation_tuned(self, df_l1, df_l2, aug_precentage,
                                    feature_cols=None, target_col=None,
                                    random_state=42,
                                    inner_validation=InnerValidation.NONE):
        """fit_with_augmentation, preceded by an inner hyperparameter search.

        With an empty grid or inner_validation=NONE this is exactly
        fit_with_augmentation — no extra fits, no behaviour change. That is the
        path Ridge and every non-tree model take.

        The search sees only `df_l2` (this outer fold's training rows) and the
        L1 augmentation block. The outer held-out participant was already
        removed by the caller. L1 rows are used for inner TRAINING only and
        never scored against: they carry imputed targets (see l1_imputation),
        so validating on them would be validating against fabricated values.
        """
        grid = self.hyperparameter_grid()
        strategy = InnerValidation(inner_validation) if inner_validation else InnerValidation.NONE
        if not grid or strategy == InnerValidation.NONE:
            return self.fit_with_augmentation(
                df_l1, df_l2, aug_precentage, feature_cols, target_col, random_state)

        # Too few rows to hold anything out — fall back to defaults rather than
        # scoring on one or two points.
        if len(df_l2) < 4:
            logger.warning(
                f"{self.model_name}: only {len(df_l2)} training rows; skipping the "
                f"inner {strategy} search and using constructor defaults.")
            return self.fit_with_augmentation(
                df_l1, df_l2, aug_precentage, feature_cols, target_col, random_state)

        splits = list(self._inner_splits(df_l2, strategy, random_state))
        best_cfg, best_score = None, -np.inf
        for cfg in grid:
            oof_true, oof_pred = [], []
            for tr, va in splits:
                inner_train, inner_val = df_l2.iloc[tr], df_l2.iloc[va]
                self.apply_config(cfg)
                self.fit_with_augmentation(df_l1, inner_train, aug_precentage,
                                           feature_cols, target_col, random_state)
                oof_pred.append(np.asarray(self.predict_df(inner_val, feature_cols),
                                           dtype=float))
                oof_true.append(inner_val[target_col].to_numpy(dtype=float))
            score = self._inner_score(np.concatenate(oof_true), np.concatenate(oof_pred))
            if score > best_score:
                best_cfg, best_score = cfg, score

        self._chosen_config = best_cfg
        logger.debug(f"{self.model_name}: inner {strategy} over {len(grid)} configs "
                     f"picked {best_cfg} (score={best_score:+.4f})")
        self.apply_config(best_cfg)
        return self.fit_with_augmentation(
            df_l1, df_l2, aug_precentage, feature_cols, target_col, random_state)

    def drop_missing_and_negative_ones(self, df, feature_cols, target_col):
        """Drop rows with a missing or -1 target, and rows whose features are all
        NaN. Partly-NaN feature rows are kept: prepare_training_data /
        prepare_test_data impute them from training-fold means (for per-text
        sets, NaN just means an unread paragraph)."""
        is_per_text = any(
            c.startswith(TRANSITIONS_FEATURE_PREFIX) or c.startswith(WFC_FEATURE_PREFIX)
            for c in feature_cols
        )
        check_cols = [target_col]
        feat_nan = df[feature_cols].isna()
        rows_any_nan = feat_nan.any(axis=1)
        if not is_per_text and rows_any_nan.any():
            rows_all_nan = feat_nan.all(axis=1)
            partial = rows_any_nan & ~rows_all_nan
            if partial.any():
                bad_cols = feat_nan.loc[partial].any()
                logger.info(
                    f"{int(partial.sum())}/{len(df)} rows have undefined values in some "
                    f"feature columns for target_col={target_col}; imputing from "
                    f"training-fold means. Affected columns (first 20): "
                    f"{list(bad_cols[bad_cols].index[:20])}."
                )
            if rows_all_nan.any():
                # No usable data at all — nothing to impute from, so drop.
                logger.warning(
                    f"Dropping {int(rows_all_nan.sum())} rows whose feature columns are all "
                    f"NaN (participants with no data in this slice) for target_col={target_col}."
                )
                df = df[~rows_all_nan]
        if df[check_cols].isnull().any().any():
            num_missing = df[check_cols].isnull().sum().sum()
            logger.debug(f"Dropping {num_missing} rows with missing values in {'target' if is_per_text else 'feature/target'} columns for target_col={target_col}.")
        if (df[target_col] == -1).any():
            num_negative_ones = (df[target_col] == -1).sum()
            logger.debug(f"Dropping {num_negative_ones} rows with target value of -1 for target_col={target_col}.")
        df = df.dropna(subset=check_cols)
        df = df[df[target_col] != -1]
        return df

    def create_folds_2_dfs(self, df_train, df_test, method:str ,n_splits=5, shuffle=True, random_state=42, allow_small_strata: bool = True):
        fold_assignments_train = list(self.create_folds(df_train, method=method, n_splits=n_splits, shuffle=shuffle, random_state=random_state, allow_small_strata=allow_small_strata))
        # assign fold ids to df_train and df_test based on participant ids
        fold_assignments_test = list(self.copy_folds_to_other_df(df_train, df_test, fold_assignments_train))
        # create a joined df where fold indices are test and train indices are the same for the same fold id
        return fold_assignments_train, fold_assignments_test

    def create_folds(self, df, method:str ,n_splits=5, shuffle=True, random_state=42, allow_small_strata: bool = True):
        """Yield (fold_id, train_idx, test_idx) for the legacy single-frame path."""
        if method == FoldsMethods.RANDOM:
            # L1-stratified k-fold with k = #L1 languages, so the train-size
            # scale matches LOLO (which holds out one L1 per fold). The
            # supplied n_splits is ignored here. StratifiedKFold needs ≥ k
            # rows per class, so smaller strata are either:
            #   allow_small_strata=True  -> force-scattered: each row is
            #     assigned a distinct random fold id (sampled w/o
            #     replacement from {0..k-1}), so the stratum still appears
            #     in train/test.
            #   allow_small_strata=False -> dropped from both train and test.
            n_splits_l1 = df[Fields.L1].nunique()
            counts = df[Fields.L1].value_counts()
            too_small = counts[counts < n_splits_l1]
            valid_pos = np.flatnonzero(
                (~df[Fields.L1].isin(too_small.index)).to_numpy()
            )

            if not too_small.empty:
                action = "force-scattering" if allow_small_strata else "dropping"
                logger.warning(
                    f"random (L1-stratified): {action} L1 groups smaller than "
                    f"n_splits={n_splits_l1}: {too_small.to_dict()}"
                )

            df_v = df.iloc[valid_pos]
            skf = StratifiedKFold(
                n_splits=n_splits_l1, shuffle=shuffle, random_state=random_state
            )

            # -1 marks rows excluded from every fold (drop case).
            fold_of = np.full(len(df), -1, dtype=int)
            for fold_id, (_, test_pos) in enumerate(
                skf.split(df_v, df_v[Fields.L1])
            ):
                fold_of[valid_pos[test_pos]] = fold_id

            if allow_small_strata and not too_small.empty:
                rng = np.random.RandomState(random_state)
                for lang in too_small.index:
                    pos = np.flatnonzero((df[Fields.L1] == lang).to_numpy())
                    fold_of[pos] = rng.choice(
                        n_splits_l1, size=len(pos), replace=False
                    )

            for fold_id in range(n_splits_l1):
                test_idx = np.flatnonzero(fold_of == fold_id)
                train_idx = np.flatnonzero(
                    (fold_of != fold_id) & (fold_of != -1)
                )
                yield fold_id, train_idx, test_idx
        elif method == FoldsMethods.LEAVE_ONE_LANGUAGE_OUT:
            languages = df[Fields.L1].unique()
            for language in languages:
                is_test = (df[Fields.L1] == language).to_numpy()
                test_idx = np.flatnonzero(is_test)
                train_idx = np.flatnonzero(~is_test)
                yield language, train_idx, test_idx
        elif method == FoldsMethods.LEAVE_ONE_PARTICIPANT_OUT:
            participants = df[Fields.SUBJECT_ID].unique()
            for participant in participants:
                is_test = (df[Fields.SUBJECT_ID] == participant).to_numpy()
                test_idx = np.flatnonzero(is_test)
                train_idx = np.flatnonzero(~is_test)
                yield participant, train_idx, test_idx
        else:
            raise ValueError(f"Unknown fold creation method: {method}")

    # ------------------------------------------------------------------
    # Two-axis dispatch: Pool × FoldMethod.
    #
    # create_folds above yields (fold_id, train_idx, test_idx) into ONE df.
    # This dispatch separates the two axes: a PoolSegment (from
    # pool_resolver.py) supplies train_pool + test_pool dataframes, and
    # _iter_test_groups_by_fold_method decides which test participants belong
    # to each fold. Train rows are then derived per fold by excluding those
    # participants from train_pool (a no-op when train_pool and test_pool are
    # disjoint, as in OneStop UNSEEN).
    # ------------------------------------------------------------------
    def _iter_test_groups_by_fold_method(
        self,
        test_pool,
        fold_method: FoldMethod,
        random_state: int = 42,
        allow_small_strata: bool = True,
    ):
        """Yield (fold_id, train_exclude_col, train_exclude_vals, test_col, test_vals).

        - train_exclude_col / vals: which rows to drop from the train pool
          (e.g. for LOLO this is L1-based so that an UNSEEN-pool fold still
          drops same-L1 participants from train even though they're disjoint
          from test by participant ID).
        - test_col / vals: which rows of the test pool form this fold's test set.
        """
        if fold_method == FoldMethod.POOL_ALL:
            for p in test_pool[Fields.SUBJECT_ID].unique():
                yield p, Fields.SUBJECT_ID, [p], Fields.SUBJECT_ID, [p]
            return

        if fold_method == FoldMethod.LOLO:
            # Exclude by L1 (not participant id): under UNSEEN the test
            # participants aren't in train, but same-L1 participants are —
            # those must still be held out to make LOLO meaningful.
            for lang in test_pool[Fields.L1].unique():
                yield lang, Fields.L1, [lang], Fields.L1, [lang]
            return

        if fold_method == FoldMethod.L1_STRAT_KFOLD:
            # K-fold L1-stratified at the participant level (not row level): one
            # participant contributes one stratum-label = their L1, so all of
            # their rows land in the same fold. k = #L1 to match LOLO's scale.
            participants = test_pool[[Fields.SUBJECT_ID, Fields.L1]].drop_duplicates()
            n_splits = participants[Fields.L1].nunique()
            counts = participants[Fields.L1].value_counts()
            too_small = counts[counts < n_splits]
            valid = participants[~participants[Fields.L1].isin(too_small.index)]

            if not too_small.empty:
                action = "force-scattering" if allow_small_strata else "dropping"
                logger.warning(
                    f"L1_STRAT_KFOLD: {action} L1 groups smaller than "
                    f"n_splits={n_splits}: {too_small.to_dict()}"
                )

            fold_of = {}
            if not valid.empty:
                skf = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=random_state
                )
                for fid, (_, test_pos) in enumerate(skf.split(valid, valid[Fields.L1])):
                    for pid in valid.iloc[test_pos][Fields.SUBJECT_ID].to_numpy():
                        fold_of[pid] = fid

            if allow_small_strata and not too_small.empty:
                rng = np.random.RandomState(random_state)
                for lang in too_small.index:
                    pids = participants.loc[participants[Fields.L1] == lang, Fields.SUBJECT_ID].to_numpy()
                    assigned = rng.choice(n_splits, size=len(pids), replace=False)
                    for pid, fid in zip(pids, assigned):
                        fold_of[pid] = fid

            by_fold: dict[int, list] = {fid: [] for fid in range(n_splits)}
            for pid, fid in fold_of.items():
                by_fold[fid].append(pid)
            for fid in range(n_splits):
                if by_fold[fid]:
                    yield fid, Fields.SUBJECT_ID, by_fold[fid], Fields.SUBJECT_ID, by_fold[fid]
            return

        raise ValueError(f"Unknown fold method: {fold_method}")

    # ----- per-segment / per-fold scaler-caching hooks (default no-op) -----
    # Subclasses can override these to skip expensive per-fold work that only
    # changes by a small leave-out from the segment-level pool. RidgeRegression
    # uses this to precompute sufficient statistics (Σx, Σx², count) once per
    # segment and derive per-fold scaler stats by subtracting the held-out
    # contribution — the original _fit_scaler runs nanmean+nanstd over 80k cols
    # × ~1100 rows every fold, which is ~33% of wall on MECO TRANSITIONS.
    def precompute_segment_scaler(self, train_pool, df_l1, feature_cols,
                                  target_col, aug_precentage, narrow_l1):
        pass

    def derive_fold_scaler(self, train_mask):
        pass

    def clear_fold_scaler(self):
        pass

    def clear_segment_scaler(self):
        pass

    # ----- closed-form LOOCV shortcut (default: not supported) -----
    # Returns a list of result rows (same format as the per-fold loop) when
    # the model can produce all leave-one-out predictions for the segment via
    # a closed-form/cross-fit shortcut. Returns None to indicate fall-through
    # to the per-fold loop. RidgeRegression implements this with a cross-fit
    # scheme (α/scaler from one half, LOO via RidgeCV on the full data with
    # the other half's α/scaler) — leak-free per participant, ~hundreds-of-
    # times faster than per-fold LOPO on per-text frames.
    def try_loocv_segment(self, segment, df_l1, *, fold_method, pool,
                          feature_cols, target_col, aug_precentage, narrow_l1):
        return None

    def run_pool_fold_segment(
        self,
        segment,
        df_l1,
        pool: Pool,
        fold_method: FoldMethod,
        feature_cols,
        target_col,
        aug_precentage=None,
        random_state: int = 42,
        narrow_l1: bool = True,
        inner_validation=InnerValidation.NONE,
    ):
        """Run one PoolSegment under the chosen FoldMethod.

        Train-invariance dedup: when every fold in this segment would yield the
        same train subset (i.e. the fold's exclude is a no-op on train_pool),
        fit ONE model on the full train_pool and predict each fold's test
        slice. This applies in practice only to OneStop UNSEEN × POOL_ALL,
        where train_pool = other-CG participants and test_pool = this-CG
        participants — disjoint by SUBJECT_ID, so holding out a test
        participant from train is a no-op. Without dedup that cell would fit
        N times with identical data.

        `narrow_l1` toggles the (list, batch, preview) L1 augmentation filter
        — on for OneStop, off for MECO (no batch column).

        `inner_validation` selects how tree models pick a hyperparameter config
        inside each outer fold (NONE / HOLDOUT / KFOLD). Models with an empty
        hyperparameter_grid() — the ridge family and everything else — ignore
        it entirely and behave exactly as before.
        """
        train_pool = segment.train_pool
        test_pool = segment.test_pool

        # Detect train-invariance once per segment. The fold method tells us
        # which column the exclude is against; if train and test pools are
        # disjoint along that column, every fold's exclude_vals (drawn from
        # test_pool) is guaranteed not to match anything in train_pool, so
        # the train subset is invariant across folds in this segment.
        invariance_col = {
            FoldMethod.POOL_ALL: Fields.SUBJECT_ID,
            FoldMethod.LOLO: Fields.L1,
            FoldMethod.L1_STRAT_KFOLD: Fields.SUBJECT_ID,
        }.get(fold_method)
        train_invariant = (
            invariance_col is not None
            and set(train_pool[invariance_col].unique()).isdisjoint(
                set(test_pool[invariance_col].unique())
            )
        )
        fold_iter = self._iter_test_groups_by_fold_method(
            test_pool, fold_method, random_state=random_state
        )

        # LOOCV shortcut: tried BEFORE the train_invariant branch so models
        # that have a dedicated fast path for POOL_ALL (currently Ridge — both
        # cross-fit for SEEN/ALL and direct RidgeCV for UNSEEN) take it instead
        # of going through the more general _run_invariant_train_folds. Models
        # that don't override try_loocv_segment return None and fall through.
        loocv_results = self.try_loocv_segment(
            segment, df_l1,
            fold_method=fold_method, pool=pool,
            feature_cols=feature_cols, target_col=target_col,
            aug_precentage=aug_precentage, narrow_l1=narrow_l1,
        )
        if loocv_results is not None:
            if not loocv_results:
                return None
            return pd.DataFrame(loocv_results)

        if train_invariant:
            logger.debug(
                f"Train-invariant segment {segment.segment_id}: fitting once on "
                f"|train_pool|={len(train_pool)} rows, predicting per fold."
            )
            return self._run_invariant_train_folds(
                segment, df_l1, pool, fold_iter,
                feature_cols=feature_cols, target_col=target_col,
                aug_precentage=aug_precentage, narrow_l1=narrow_l1,
                inner_validation=inner_validation,
            )

        # Per-fold fit (train genuinely varies). Cache live-column scans per
        # content_group so the per-text NaN scan only runs once per CG.
        fold_feature_cols_cache: dict = {}
        results = []

        self.precompute_segment_scaler(
            train_pool, df_l1, feature_cols, target_col,
            aug_precentage=aug_precentage, narrow_l1=narrow_l1,
        )

        # FOLD_N_JOBS>1 runs the fold loop concurrently. The fits (BLAS /
        # lightgbm / xgboost / sklearn trees) release the GIL, so threads give
        # near-linear speedup on jobs dominated by fit time — the intended use
        # is focusing the whole machine on ONE huge segment (MECO per-text:
        # 2210 folds × minutes-long fits) with --n-jobs 1. Each fold gets a
        # deepcopy of the model (fit mutates self.model / scaler state) and
        # its own df_l1 copy (l1_imputation mutates the target column of the
        # frame it is handed). Not used by Ridge, whose LOOCV shortcut
        # returns before this loop.
        fold_workers = int(os.environ.get("FOLD_N_JOBS", "1"))
        if fold_workers > 1:
            try:
                return self._run_folds_threaded(
                    segment, df_l1, pool, list(fold_iter),
                    feature_cols=feature_cols, target_col=target_col,
                    aug_precentage=aug_precentage, narrow_l1=narrow_l1,
                    n_workers=fold_workers,
                    inner_validation=inner_validation,
                )
            finally:
                self.clear_segment_scaler()

        try:
            fold_list = list(fold_iter)
            for fold_id, exclude_col, exclude_vals, test_col, test_vals in _fold_tqdm(
                    fold_list, len(fold_list), f"folds {segment.segment_id}"):
                train_mask = ~train_pool[exclude_col].isin(set(exclude_vals)).to_numpy()
                test_mask = test_pool[test_col].isin(set(test_vals)).to_numpy()
                train_df = train_pool.iloc[train_mask]
                test_df = test_pool.iloc[test_mask]
                full_fold_id = f"{segment.segment_id}/{fold_id}"
                self.derive_fold_scaler(train_mask)
                try:
                    fold_results, fold_feature_cols_cache = self._run_one_fold_pool_aware(
                        full_fold_id,
                        df_l1,
                        train_df,
                        test_df,
                        pool=pool,
                        narrow_l1=narrow_l1,
                        aug_precentage=aug_precentage,
                        feature_cols=feature_cols,
                        target_col=target_col,
                        fold_feature_cols_cache=fold_feature_cols_cache,
                        inner_validation=inner_validation,
                    )
                finally:
                    self.clear_fold_scaler()
                if fold_results is not None:
                    results.extend(fold_results)
        finally:
            self.clear_segment_scaler()

        if not results:
            return None
        return pd.DataFrame(results)

    def _run_folds_threaded(
        self, segment, df_l1, pool, folds, *,
        feature_cols, target_col, aug_precentage, narrow_l1, n_workers,
        inner_validation=InnerValidation.NONE,
    ):
        """Run the per-fold loop of run_pool_fold_segment across threads.

        Semantics match the sequential loop with two deliberate differences:
        (1) each fold fits a deepcopy of the model instead of reusing one
        mutable instance, and (2) each fold's l1_imputation works on a fresh
        copy of df_l1, so no fold sees a previous fold's imputed values
        (strictly cleaner than the sequential in-place mutation). The
        per-CG live-column cache is skipped — each fold rescans, which is
        noise next to the minutes-long fits this path is meant for.
        """
        train_pool = segment.train_pool
        test_pool = segment.test_pool

        def _one_fold(fold):
            fold_id, exclude_col, exclude_vals, test_col, test_vals = fold
            train_mask = ~train_pool[exclude_col].isin(set(exclude_vals)).to_numpy()
            test_mask = test_pool[test_col].isin(set(test_vals)).to_numpy()
            train_df = train_pool.iloc[train_mask]
            test_df = test_pool.iloc[test_mask]
            clone = copy.deepcopy(self)
            clone.derive_fold_scaler(train_mask)
            try:
                fold_results, _ = clone._run_one_fold_pool_aware(
                    f"{segment.segment_id}/{fold_id}",
                    df_l1.copy(),
                    train_df,
                    test_df,
                    pool=pool,
                    narrow_l1=narrow_l1,
                    aug_precentage=aug_precentage,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    fold_feature_cols_cache={},
                    inner_validation=inner_validation,
                )
            finally:
                clone.clear_fold_scaler()
            return fold_results

        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for fold_results in _fold_tqdm(
                    executor.map(_one_fold, folds), len(folds),
                    f"folds {segment.segment_id}"):
                if fold_results:
                    results.extend(fold_results)

        if not results:
            return None
        return pd.DataFrame(results)

    def _run_invariant_train_folds(
        self, segment, df_l1, pool, fold_iter, *,
        feature_cols, target_col, aug_precentage, narrow_l1,
        inner_validation=InnerValidation.NONE,
    ):
        """Fit one model on segment.train_pool, then predict each fold's test slice.

        Used when train is provably invariant across folds in this segment
        (detected in run_pool_fold_segment). Mirrors _run_one_fold_pool_aware
        but pulls the L1-narrow, live-column-scan, and fit steps out of the
        loop. Predict happens per fold so per-fold result rows still carry
        the fold id.
        """
        train_pool = segment.train_pool
        test_pool = segment.test_pool

        if len(train_pool) == 0 or len(test_pool) == 0:
            logger.warning(f"Skipping invariant segment {segment.segment_id}: train={len(train_pool)} test={len(test_pool)}")
            return None

        # L1 narrowing (once)
        if narrow_l1 and pool != Pool.ALL and Fields.BATCH in df_l1.columns and Fields.BATCH in train_pool.columns:
            df_l1_aug = self._narrow_l1_by_combo(train_pool, df_l1)
        else:
            df_l1_aug = df_l1
        if len(df_l1_aug) == 0:
            logger.warning(f"Skipping invariant segment {segment.segment_id}: no relevant L1 data")
            return None

        # Per-text gating is enforced upstream (per-text requires Pool.SEEN,
        # which can never be train-invariant since train_pool == test_pool
        # there). The is_per_text branch is kept for symmetry with the
        # per-fold path but will not fire in practice.
        is_per_text = any(
            any(c.startswith(p) for p in SEEN_CV_ONLY_PREFIXES) for c in feature_cols
        )
        cgs = pd.unique(train_pool[Fields.CONTENT_GROUP]) if Fields.CONTENT_GROUP in train_pool.columns else []
        if is_per_text and len(cgs) == 1 and Fields.CONTENT_GROUP in df_l1_aug.columns:
            df_l1_aug = df_l1_aug[df_l1_aug[Fields.CONTENT_GROUP] == cgs[0]]
            if len(df_l1_aug) == 0:
                logger.warning(f"Skipping invariant segment {segment.segment_id}: no L1 rows in content_group {cgs[0]}")
                return None

        # Live-column scan over train_pool ∪ test_pool (the union of every
        # possible per-fold fold_l2). Stable because train is invariant and
        # any test slice is a subset of test_pool.
        fold_l2 = pd.concat([train_pool, test_pool], axis=0)
        l2_live_mask = fold_l2[feature_cols].notna().all()
        if is_per_text:
            l1_live_mask = df_l1_aug[feature_cols].notna().all()
            fold_feature_cols = [c for c in feature_cols if l2_live_mask[c] and l1_live_mask[c]]
        else:
            fold_feature_cols = [c for c in feature_cols if l2_live_mask[c]]
        if not fold_feature_cols:
            logger.warning(f"Skipping invariant segment {segment.segment_id}: no feature columns with full coverage")
            return None

        # Fit ONCE on the entire train_pool. The inner search (if any) also runs
        # once here rather than per participant, which is correct: this branch
        # is only taken when the train set is invariant across the segment's
        # folds, so a per-fold search would repeat identical work.
        self.fit_with_augmentation_tuned(
            df_l1_aug, train_pool, aug_precentage=aug_precentage,
            feature_cols=fold_feature_cols, target_col=target_col,
            inner_validation=inner_validation,
        )

        # Predict per fold-group; each fold supplies the test slice and the
        # fold id that lands in the results column.
        results = []
        for fold_id, _exclude_col, _exclude_vals, test_col, test_vals in fold_iter:
            test_mask = test_pool[test_col].isin(set(test_vals)).to_numpy()
            test_df = test_pool.iloc[test_mask]
            if len(test_df) == 0:
                continue
            predictions = self.predict_df(test_df, feature_cols=fold_feature_cols)
            full_fold_id = f"{segment.segment_id}/{fold_id}"
            for pid, l1v, tv, pv in zip(
                test_df[Fields.SUBJECT_ID].values,
                test_df[Fields.L1].values,
                test_df[target_col].values,
                predictions,
            ):
                results.append({
                    Fields.SUBJECT_ID: pid,
                    Fields.L1: l1v,
                    'true': tv,
                    'pred': pv,
                    'fold': full_fold_id,
                })
        if not results:
            return None
        return pd.DataFrame(results)

    def _run_one_fold_pool_aware(
        self, fold, df_l1, train_df, test_df, *, pool: Pool, narrow_l1: bool,
        aug_precentage, feature_cols, target_col, fold_feature_cols_cache,
        inner_validation=InnerValidation.NONE,
    ):
        """Pool-aware variant of run_one_fold_basic.

        Mirrors run_one_fold_basic but routes L1 narrowing through the pool
        axis instead of the legacy `method == BATCH_CROSS_VALIDATION` check.
        Everything else (per-text NaN handling, live-column caching, model
        fit + predict) matches the legacy path exactly.
        """
        if fold_feature_cols_cache is None:
            fold_feature_cols_cache = {}
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning(f"Skipping fold {fold}: train={len(train_df)} test={len(test_df)}")
            return None, fold_feature_cols_cache

        if narrow_l1 and pool != Pool.ALL and Fields.BATCH in df_l1.columns and Fields.BATCH in train_df.columns:
            df_l1_aug = self._narrow_l1_by_combo(train_df, df_l1)
        else:
            df_l1_aug = df_l1
        if len(df_l1_aug) == 0:
            logger.warning(f"Skipping fold {fold}: no relevant L1 data")
            return None, fold_feature_cols_cache

        cache_key = None
        if Fields.CONTENT_GROUP in train_df.columns:
            cgs = pd.unique(train_df[Fields.CONTENT_GROUP])
            if len(cgs) == 1:
                cache_key = cgs[0]
        is_per_text = any(
            any(c.startswith(p) for p in SEEN_CV_ONLY_PREFIXES) for c in feature_cols
        )
        if is_per_text and cache_key is not None and Fields.CONTENT_GROUP in df_l1_aug.columns:
            df_l1_aug = df_l1_aug[df_l1_aug[Fields.CONTENT_GROUP] == cache_key]
            if len(df_l1_aug) == 0:
                logger.warning(f"Skipping fold {fold}: no L1 rows in content_group {cache_key}")
                return None, fold_feature_cols_cache
        if cache_key is not None and cache_key in fold_feature_cols_cache:
            fold_feature_cols = fold_feature_cols_cache[cache_key]
        elif is_per_text and cache_key is None:
            # MECO per-text (no content_group): NaN is per-participant — every
            # column has ~28% NaN and none is fully covered, so we CANNOT drop
            # columns. Keep them all; the model imputes NaN → cohort mean (Ridge
            # in _transform; tree models in prepare_training_data). This branch
            # is MECO-only, so OneStop's structural NaN never lands here.
            fold_feature_cols = list(feature_cols)
        else:
            fold_l2 = pd.concat([train_df, test_df], axis=0)
            l2_live_mask = fold_l2[feature_cols].notna().all()
            # OneStop per-text reaches here (cache_key set). Its NaN is structural
            # per content group — a per-text column is either present for every
            # participant in the group or NaN for all of them (verified: zero
            # within-group mixing), so the standard full-coverage filter drops
            # exactly the other-group columns losslessly and the data reaches the
            # model NaN-free (no imputation needed / applied for OneStop).
            if is_per_text:
                # Require coverage in the L1 augmentation too, matching
                # RidgeRegression._nested_loocv_segment and the invariant-train
                # path above. load_per_text_features_pair aligns both cohorts to
                # the column union with fill_value=0, which materialises ~188k
                # transitions columns L2 never observed as all-ZERO rather than
                # NaN — invisible to the L2 mask. They are NaN in the same-CG L1
                # rows (those participants never read that batch), so this clause
                # removes exactly them: measured 0 genuine columns lost across all
                # six content groups, 252k -> ~96k per fold. Without it TabPFN's
                # 2000-feature budget lands entirely on that dead block and its
                # constant-feature step fails the fit.
                l1_live_mask = df_l1_aug[feature_cols].notna().all()
                live = [c for c in feature_cols
                        if l2_live_mask[c] and l1_live_mask[c]]
                # A segment with no fully-covered column would otherwise fit on an
                # empty design; fall back to the L2-only set.
                fold_feature_cols = live or [c for c in feature_cols if l2_live_mask[c]]
            else:
                fold_feature_cols = [c for c in feature_cols if l2_live_mask[c]]
            if cache_key is not None:
                fold_feature_cols_cache[cache_key] = fold_feature_cols
        if not fold_feature_cols:
            logger.warning(f"Skipping fold {fold}: no feature columns with full coverage in L2 fold")
            return None, fold_feature_cols_cache

        self.fit_with_augmentation_tuned(
            df_l1_aug, train_df, aug_precentage=aug_precentage,
            feature_cols=fold_feature_cols, target_col=target_col,
            inner_validation=inner_validation,
        )
        predictions = self.predict_df(test_df, feature_cols=fold_feature_cols)
        results = [
            {
                Fields.SUBJECT_ID: pid,
                Fields.L1: l1v,
                'true': tv,
                'pred': pv,
                'fold': fold,
            }
            for pid, l1v, tv, pv in zip(
                test_df[Fields.SUBJECT_ID].values,
                test_df[Fields.L1].values,
                test_df[target_col].values,
                predictions,
            )
        ]
        return (results or None), fold_feature_cols_cache

    def _narrow_l1_by_combo(self, l2_train, df_l1):
        """Restrict L1 augmentation to readers sharing (list, batch, preview)
        combos with the L2 training set. Same logic as
        `take_relevant_l1_rows` for BATCH_CROSS_VALIDATION but reusable from
        the new dispatch without depending on the legacy fold-method enum.
        """
        def _combo(df):
            return (
                df[Fields.SUBJECT_ID].str.split('_').str[0]
                + "_" + df[Fields.BATCH].astype(int).astype(str)
                + "_" + df[Fields.HAS_PREVIEW].astype(str)
            )
        allowed = set(_combo(l2_train).unique())
        return df_l1[_combo(df_l1).isin(allowed)]

    def copy_folds_to_other_df(self, df_origin, df_target, fold_assignments_first_window):
        """For moving_p_agg, copy fold assignments from one window to the next.
        fold assignments is the output of create_folds for the first window, which we want to reuse for subsequent windows: (fold_id, train_idx, test_idx)
        """
        # find participants in train
        for fold_id, train_idx, test_idx in fold_assignments_first_window:
            # find train_idx in new df
            train_participants = df_origin.iloc[train_idx][Fields.SUBJECT_ID].unique()
            test_participants = df_origin.iloc[test_idx][Fields.SUBJECT_ID].unique()
            train_idx_target = np.where(df_target[Fields.SUBJECT_ID].isin(train_participants))[0]
            test_idx_next_target = np.where(df_target[Fields.SUBJECT_ID].isin(test_participants))[0]

            yield fold_id, train_idx_target, test_idx_next_target

    def take_relevant_l1_rows(self, l2_train, df_l1, method:str):
        """Under BATCH_CROSS_VALIDATION (OneStop), restrict the L1 augmentation
        to readers sharing (list, batch, preview) combos with the L2 train set;
        otherwise return df_l1 unchanged."""
        if method == FoldsMethods.BATCH_CROSS_VALIDATION:
            # Compute the (list, batch, has_preview) keys without copying the full DataFrames — for per-text
            # frames df_l1 is hundreds of thousands of columns wide and a per-fold .copy() dominated runtime.
            def _combo(df):
                return (
                    df[Fields.SUBJECT_ID].str.split('_').str[0]
                    + "_" + df[Fields.BATCH].astype(int).astype(str)
                    + "_" + df[Fields.HAS_PREVIEW].astype(str)
                )
            allowed = set(_combo(l2_train).unique())
            return df_l1[_combo(df_l1).isin(allowed)]
        else:
            return df_l1

    def run_one_fold_basic(self, fold, df_l1, train_df, test_df, fold_method, aug_precentage=None, feature_cols=None, target_col=None, fold_feature_cols_cache: dict = None):
        """Fit on one fold's training rows (plus L1 augmentation) and predict its
        test rows. Returns (result rows or None, fold_feature_cols_cache)."""
        results = []
        if fold_feature_cols_cache is None:
            fold_feature_cols_cache = {}
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning(f"Skipping fold {fold} due to empty train set ({len(train_df)} rows) or test set ({len(test_df)} rows)")
            return None, fold_feature_cols_cache
        df_l1_aug = self.take_relevant_l1_rows(train_df, df_l1, method=fold_method)
        if len(df_l1_aug) == 0:
            logger.warning(f"Skipping fold {fold} due to no relevant L1 data")
            return None, fold_feature_cols_cache
        # Detect per-text feature sets. The same-CG L1 filter + L1-intersection in fold_feature_cols
        # are needed only for these (TRANSITIONS / WFC), where pair-alignment 0-fills cohort-missing
        # cols in df_l2 and leaks other-CGs' paragraphs into fold_feature_cols. For non-per-text
        # features there is no NaN at all and L1 augmentation should keep its broader pre-patch
        # set (same list+batch+preview, any content_group) to preserve historical behavior.
        cache_key = None
        if Fields.CONTENT_GROUP in train_df.columns:
            cgs = pd.unique(train_df[Fields.CONTENT_GROUP])
            if len(cgs) == 1:
                cache_key = cgs[0]
        is_per_text = any(
            any(c.startswith(p) for p in SEEN_CV_ONLY_PREFIXES) for c in feature_cols
        )
        if is_per_text and cache_key is not None and Fields.CONTENT_GROUP in df_l1_aug.columns:
            df_l1_aug = df_l1_aug[df_l1_aug[Fields.CONTENT_GROUP] == cache_key]
            if len(df_l1_aug) == 0:
                logger.warning(f"Skipping fold {fold} due to no L1 rows in content_group {cache_key}")
                return None, fold_feature_cols_cache
        if cache_key is not None and cache_key in fold_feature_cols_cache:
            fold_feature_cols = fold_feature_cols_cache[cache_key]
        else:
            fold_l2 = pd.concat([train_df, test_df], axis=0)
            l2_live_mask = fold_l2[feature_cols].notna().all()
            if is_per_text and cache_key is None:
                # MECO per-text: per-participant NaN, no fully-covered columns to
                # keep — keep all and let the model impute. OneStop per-text
                # (cache_key set) drops its structural-NaN columns via the filter
                # below, exactly like _run_one_fold_pool_aware. (Legacy path:
                # per-text is Pool.SEEN only, so this is effectively unreached;
                # kept consistent to avoid a latent trap.)
                fold_feature_cols = list(feature_cols)
            else:
                fold_feature_cols = [c for c in feature_cols if l2_live_mask[c]]
            if cache_key is not None:
                fold_feature_cols_cache[cache_key] = fold_feature_cols
        if not fold_feature_cols:
            logger.warning(f"Skipping fold {fold}: no feature columns with full coverage in L2 fold")
            return None, fold_feature_cols_cache
        self.fit_with_augmentation(df_l1_aug, train_df, aug_precentage=aug_precentage, feature_cols=fold_feature_cols, target_col=target_col)
        predictions = self.predict_df(test_df, feature_cols=fold_feature_cols)
        true_values = test_df[target_col].values
        participant_ids = test_df[Fields.SUBJECT_ID].values
        l1_values = test_df[Fields.L1].values
        window_indices = test_df['window_index'].values if 'window_index' in test_df.columns else [None] * len(test_df)
        # Store each prediction individually
        for participant_id, l1_val, true_val, pred_val, win_idx in zip(participant_ids, l1_values, true_values, predictions, window_indices):
            row = {
                Fields.SUBJECT_ID: participant_id,
                Fields.L1: l1_val,
                'true': true_val,
                'pred': pred_val,
                'fold': fold,
            }
            if win_idx is not None:
                row['window_index'] = win_idx
            results.append(row)

        if results == []:
            return None, fold_feature_cols_cache

        return results, fold_feature_cols_cache

    def run_cross_validation_2_dfs(self, df_train_origin, df_test_origin, df_l1, aug_precentage=None, feature_cols=None, target_col=None, fold_method="random", n_splits=5):
        # notice: df_l1 needs to come from the same level df_train_origin is from, cause it is only used to test
        fold_assignments_train, fold_assignments_test = self.create_folds_2_dfs(df_train_origin, df_test_origin, method=fold_method, n_splits=n_splits)
        results = []
        fold_feature_cols_cache = {}

        for (fold_id_train, train_idx, _), (fold_id_test, _, test_idx) in zip(fold_assignments_train, fold_assignments_test):
            assert fold_id_train == fold_id_test, f"Fold IDs do not match between train and test: {fold_id_train} vs {fold_id_test}"
            train_df = df_train_origin.iloc[train_idx]
            test_df = df_test_origin.iloc[test_idx]
            fold_results, fold_feature_cols_cache = self.run_one_fold_basic(fold_id_train, df_l1, train_df, test_df, fold_method, aug_precentage=aug_precentage, feature_cols=feature_cols, target_col=target_col, fold_feature_cols_cache=fold_feature_cols_cache)
            if fold_results is not None:
                results.extend(fold_results)
        results_df = pd.DataFrame(results)
        return results_df

    def run_cross_validation_basic(self, df_l1, df_l2, aug_precentage=None, feature_cols=None, target_col=None, fold_method="random", n_splits=5, l2_folds=None, disable_tqdm=True):
        # Multi-seed re-runs existed only for the now-dead BCV random
        # frameworks (SEEN_LARGE, UNSEEN_SMALL, UNSEEN_SMALL_BY_BATCH). The
        # remaining fold methods (LOPO, LOLO, RANDOM/L1-strat) are all
        # deterministic given a fixed random_state, so single-seed is correct.
        results = []
        if l2_folds is None:
            l2_folds = list(self.create_folds(df_l2, method=fold_method, n_splits=n_splits))
        # Cache live feature columns by content_group: within a seen-CV CG every fold shares the same NaN pattern,
        # so the notna scan over hundreds of thousands of per-text columns only needs to run once per CG.
        fold_feature_cols_cache: dict = {}
        for fold, train_idx, test_idx in tqdm(l2_folds, desc="Running CV folds", unit="fold", disable=disable_tqdm):
            train_df = df_l2.iloc[train_idx]
            test_df = df_l2.iloc[test_idx]
            fold_results, fold_feature_cols_cache = self.run_one_fold_basic(fold, df_l1, train_df, test_df, fold_method, aug_precentage=aug_precentage, feature_cols=feature_cols, target_col=target_col, fold_feature_cols_cache=fold_feature_cols_cache)
            if fold_results is not None:
                results.extend(fold_results)

        if results == []:
            return None
        return pd.DataFrame(results)

    def run_cross_validation_moving_p_agg(self, df_l1, df_l2, aug_precentage=None, feature_cols=None, target_col=None, fold_method="random", n_splits=5, disable_tqdm=True):
        results = []
        sorted_indices = sorted(df_l2['window_index'].unique())
        first_window_df = df_l2[df_l2['window_index'] == sorted_indices[0]]
        expected_test_participants = set()

        l2_folds = list(self.create_folds(first_window_df, method=fold_method, n_splits=n_splits))
        for _, _, test_idx in l2_folds:
            expected_test_participants.update(first_window_df.iloc[test_idx][Fields.SUBJECT_ID])
        for win_idx in sorted_indices:
            df_l1_win = df_l1[df_l1['window_index'] == win_idx]
            df_l2_win = df_l2[df_l2['window_index'] == win_idx]
            l2_folds_win = list(self.copy_folds_to_other_df(first_window_df, df_l2_win, l2_folds))
            fold_results = self.run_cross_validation_basic(df_l1_win, df_l2_win, aug_precentage=aug_precentage, feature_cols=feature_cols,
                                                           target_col=target_col, fold_method=fold_method, n_splits=n_splits,
                                                           l2_folds=l2_folds_win, disable_tqdm=disable_tqdm) # we use the same folds for all windows, which means that the same participants are in the train and test sets across windows (e.g. in leave one participant out, the same participant is held out across all windows)
            if fold_results is None:
                continue
            fold_results['window_index'] = win_idx
            actual_test_participants = set(fold_results[Fields.SUBJECT_ID].unique())
            assert actual_test_participants == expected_test_participants, \
                f"Expected {len(expected_test_participants)} test participants but got {len(actual_test_participants)} in window {win_idx}"
            results.append(fold_results)
        if results == []:
            return None
        results_df = pd.concat(results, ignore_index=True)
        return results_df

    def run_cross_validation_moving_p_agg_2_dfs(self, df_l1, df_l2_moving, df_l2_full, aug_precentage=None, feature_cols=None, target_col=None, fold_method="random", n_splits=5):
        """
        This function runs moving p agg in a way where we train on the full db and test on a specific moving window each time
        """
        results = []
        sorted_indices = sorted(df_l2_moving['window_index'].unique())
        # take only participants from l2 full that exist in l2 moving, to avoid issues with participants that only exist in l2 full and not in l2 moving (e.g. because they were filtered out due to missing values in the moving window)
        df_l2_full = df_l2_full[df_l2_full[Fields.SUBJECT_ID].isin(df_l2_moving[Fields.SUBJECT_ID].unique())]

        # get folds from the full df
        l2_folds = list(self.create_folds(df_l2_full, method=fold_method, n_splits=n_splits))
        expected_test_participants = set()
        for _, _, test_idx in l2_folds:
            expected_test_participants.update(df_l2_full.iloc[test_idx][Fields.SUBJECT_ID])

        for win_idx in sorted_indices:
            df_l2_win = df_l2_moving[df_l2_moving['window_index'] == win_idx]

            l2_folds_win = list(self.copy_folds_to_other_df(df_l2_full, df_l2_win, l2_folds))
            # we insert df_l1 here so that in train we have the full l1 as well
            fold_results = self.run_cross_validation_basic(df_l1, df_l2_win, aug_precentage=aug_precentage, feature_cols=feature_cols,
                                                           target_col=target_col, fold_method=fold_method, n_splits=n_splits,
                                                           l2_folds=l2_folds_win) # we use the same folds for all windows, which means that the same participants are in the train and test sets across windows (e.g. in leave one participant out, the same participant is held out across all windows)
            if fold_results is None:
                continue
            fold_results['window_index'] = win_idx
            actual_test_participants = set(fold_results[Fields.SUBJECT_ID].unique())
            assert actual_test_participants == expected_test_participants, \
                f"Expected {len(expected_test_participants)} test participants but got {len(actual_test_participants)}"

            results.append(fold_results)
        if results == []:
            return None
        results_df = pd.concat(results, ignore_index=True)
        return results_df

    def run_cross_validation(self, df_l1, df_l2, aug_precentage=None,
                             feature_cols=None, target_col=None, fold_method="random",
                             n_splits=5, save_path=None,
                             agg_type=None, df_l2_full=None, train_on_full=False, disable_tqdm=True,
                             pool=None):
        # This single-frame path is the ALL pool unless the caller says
        # otherwise (the per-item runners pass their pool); per-text features
        # need Pool.SEEN, so the validator rejects them under ALL.
        validation_pool = pool if pool is not None else Pool.ALL
        validate_seen_only_features(feature_cols, validation_pool, df_l2)
        df_l2 = self.drop_missing_and_negative_ones(df_l2, feature_cols, target_col)

        if train_on_full:
            if df_l2_full is None:
                raise ValueError("df_l2_full must be provided when train_on_full_l2 is True")
            df_l2_full = self.drop_missing_and_negative_ones(df_l2_full, feature_cols, target_col)
            if agg_type == AggTypes.MOVING_P:
                results_df = self.run_cross_validation_moving_p_agg_2_dfs(df_l1, df_l2, df_l2_full, aug_precentage=aug_precentage, feature_cols=feature_cols, target_col=target_col, fold_method=fold_method, n_splits=n_splits)
            else:
                results_df = self.run_cross_validation_2_dfs(df_l2_full, df_l2, df_l1, aug_precentage=aug_precentage, feature_cols=feature_cols, target_col=target_col, fold_method=fold_method, n_splits=n_splits)
        else:
            if agg_type == AggTypes.MOVING_P:
                results_df = self.run_cross_validation_moving_p_agg(df_l1, df_l2, aug_precentage=aug_precentage, feature_cols=feature_cols, target_col=target_col, fold_method=fold_method, n_splits=n_splits, disable_tqdm=disable_tqdm)
            else:
                results_df = self.run_cross_validation_basic(df_l1, df_l2, aug_precentage=aug_precentage, feature_cols=feature_cols, target_col=target_col, fold_method=fold_method, n_splits=n_splits, disable_tqdm=disable_tqdm)

        if results_df is None:
            logger.warning("No cross-validation results produced (all folds failed or had no feature coverage)")
            return None

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(save_path, index=False)
        return results_df

    def l1_imputation(self, l1_df, target_col):
        """Fill the L1 rows' missing (-1) scores: the L1 subset mean for tests
        some L1 readers took (TEST_COLS_IN_L1), else the test's maximum score."""
        if l1_df[target_col].isna().sum() != 0:
            raise ValueError(f"Expected no missing values in L1 data for column '{target_col}' but found {l1_df[target_col].isna().sum()}")
        if target_col in TEST_COLS_IN_L1:
            valid = l1_df[target_col][l1_df[target_col] != -1]
            if len(valid) > 0:
                impute_value = valid.mean()
            else:
                # No L1 participant in this subset took the test (common for
                # cg 1_0 in Hunting filter: list-prefix matches yield zero
                # valid LexTALE scores). Fall back to the test's max so we
                # don't propagate NaN into y.
                impute_value = ALL_TESTS_MAX_SCORES.get(target_col, 100)
                logger.warning(
                    f"l1_imputation: no valid L1 values for '{target_col}' in this subset "
                    f"({len(l1_df)} rows, all -1); falling back to {impute_value}"
                )
            l1_df[target_col] = l1_df[target_col].replace(-1, impute_value)
        elif target_col in ALL_TESTS_MAX_SCORES.keys():
            max_score = ALL_TESTS_MAX_SCORES[target_col]
            l1_df[target_col] = l1_df[target_col].replace(-1, max_score)
        else:
            raise ValueError(f"Unknown target column for L1 imputation: {target_col}")
        return l1_df
