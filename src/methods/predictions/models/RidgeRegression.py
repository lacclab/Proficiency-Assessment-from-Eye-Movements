import os

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge, RidgeCV

from src.constants import (Fields, FoldMethod, Pool, ALL_TESTS_MAX_SCORES,
                           TRANSITIONS_FEATURE_PREFIX, WFC_FEATURE_PREFIX)
from src.methods.predictions.models.BaseModel import BaseModel


# Shared α grid for RidgeCV-based α selection (per-fold loop, cross-fit,
# unseen-direct). Dropped from 30 to 10 alphas because RidgeCV(cv=None)'s
# per-α score is O(n) and scales with grid size — coarser is fine in the
# p ≫ n regime where the LOO score surface is shallow.
_DEFAULT_ALPHA_GRID = np.logspace(-3, 4, 10).tolist()


def _alpha_grid():
    """RidgeCV alpha grid, overridable via RIDGE_ALPHA_GRID="lo,hi,n" (logspace
    exponents). The default floor of 1e-3 leaves wide per-text feature sets
    (p >> n) essentially unregularized: RidgeCV pins alpha at the floor, the fit
    interpolates, and the reconstructed LOO predictions blow past the score
    bounds. See the LOOCV_DIAG instrumentation."""
    spec = os.environ.get("RIDGE_ALPHA_GRID")
    if spec:
        lo, hi, n = spec.split(",")
        return np.logspace(float(lo), float(hi), int(n)).tolist()
    return _DEFAULT_ALPHA_GRID


LOO_MODES = ("nested", "crossfit", "auto")
_DEFAULT_LOO_MODE = "nested"
# Below this many participants in a segment the cross-fit's half-split becomes
# unreliable (α and the scaler get picked on ~n/2 rows), so "auto" routes small
# segments to nested LOO. Measured: at n≈25 (OneStop SEEN content groups) the
# cross-fit loses up to 0.26 Pearson r against nested LOO and swings 0.20 across
# RIDGE_CF_SEED values; at n≈1100 (MECO paragraph halves) the two agree to
# within 0.002 and the cross-fit is ~300x cheaper.
_AUTO_NESTED_BELOW_N = 100


def _loo_mode():
    """Which leave-one-out implementation the POOL_ALL SEEN/ALL fast path uses.

    "nested"   — one RidgeCV per held-out participant; α and the scaler are fit
                 on that fold's own n-1 rows. The default: no split seed, no
                 half-sized α estimate, and it is what the per-item pipeline
                 already does via the per-fold loop.
    "crossfit" — the legacy stratified A/B half-split (α/scaler from one half,
                 LOO reconstruction on the full data with the other half's).
                 Kept for reproducing previously published numbers.
    "auto"     — nested below _AUTO_NESTED_BELOW_N participants, crossfit above.

    Set via --loo-mode on the runner, which exports PROF_LOO_MODE.
    """
    mode = os.environ.get("PROF_LOO_MODE", _DEFAULT_LOO_MODE).lower()
    if mode not in LOO_MODES:
        raise ValueError(f"PROF_LOO_MODE must be one of {LOO_MODES}, got {mode!r}")
    return mode


def _cf_reconcile_alphas(alpha_A, alpha_B):
    """Cross-fit predicts each half with the OTHER half's alpha. When the two
    halves disagree by orders of magnitude the LOO criterion is unreliable, and
    cross-applying leaves one half effectively unregularized. RIDGE_CF_GUARD=<r>
    collapses both to the larger alpha once the ratio exceeds r (default off);
    preferring more regularization trades variance for bias."""
    thresh = os.environ.get("RIDGE_CF_GUARD")
    if not thresh:
        return alpha_A, alpha_B
    lo, hi = min(alpha_A, alpha_B), max(alpha_A, alpha_B)
    if lo > 0 and hi / lo > float(thresh):
        return hi, hi
    return alpha_A, alpha_B


class RidgeRegression(BaseModel):
    """Ridge regression on z-scored features (scaler fit on the L2 rows,
    NaN -> 0 after scaling), with closed-form leave-one-out fast paths for the
    POOL_ALL segments. `choose_alpha` picks α per fit with RidgeCV;
    `test_size_alpha` is accepted for compatibility and not used."""

    def __init__(
        self,
        alpha: float = 1.0,
        choose_alpha: bool = False,
        test_size_alpha: float = 0.2,
        random_state: int = 42,
        native_language: str = "English",
        loo_mode: str | None = None,
    ):
        super().__init__(model_name="RidgeRegression")
        self.alpha = alpha
        self.choose_alpha = choose_alpha
        self.test_size_alpha = test_size_alpha
        self.random_state = random_state
        self.native_language = native_language
        # None defers to PROF_LOO_MODE (see _loo_mode); an explicit value wins,
        # which is what the tests and the equivalence harness use.
        if loo_mode is not None and loo_mode not in LOO_MODES:
            raise ValueError(f"loo_mode must be one of {LOO_MODES}, got {loo_mode!r}")
        self.loo_mode = loo_mode

        self.model = Ridge(alpha=self.alpha)
        self.feature_cols = None
        self.target_col = None
        self.scaler_mean_ = None
        self.scaler_std_ = None

    def _get_language_col(self, df: pd.DataFrame):
        if "language" in df.columns:
            return "language"
        if Fields.L1 in df.columns:
            return Fields.L1
        return None

    def _fit_scaler(self, train_df: pd.DataFrame, feature_cols):
        # Operate on numpy: pandas .mean()/.std() iterate per column, which dominates runtime on the
        # 100K-column slim frames produced by per-text feature sets. Keep mean/std as pandas Series so
        # downstream column alignment via .reindex() still works.
        override = getattr(self, '_fold_scaler_override', None)
        if override is not None:
            means, stds, ctx_cols = override
            if list(feature_cols) == ctx_cols:
                self.scaler_mean_ = pd.Series(means, index=feature_cols)
                self.scaler_std_ = pd.Series(stds, index=feature_cols)
                return
        language_col = self._get_language_col(train_df)
        if language_col is not None:
            mask = train_df[language_col] != self.native_language
            l2_df = train_df[mask] if mask.any() else train_df
            fit_on_arr = l2_df[feature_cols].to_numpy(dtype=np.float64, copy=False)
        else:
            fit_on_arr = train_df[feature_cols].to_numpy(dtype=np.float64, copy=False)

        means = np.nanmean(fit_on_arr, axis=0)
        stds = np.nanstd(fit_on_arr, axis=0, ddof=1)  # match pandas .std() default
        stds = np.where(stds == 0, 1.0, stds)
        self.scaler_mean_ = pd.Series(means, index=feature_cols)
        self.scaler_std_ = pd.Series(stds, index=feature_cols)

    # ----- per-segment scaler precomputation (incremental LOPO) -----
    # Skips the per-fold nanmean+nanstd over 80k cols × ~1100 rows by caching
    # Σx, Σx², count once per segment and deriving per-fold stats by
    # subtracting the held-out L2 rows' contribution. Only safe when the L1
    # augmentation rows are stable across folds — currently MECO (narrow_l1=
    # False). OneStop's _narrow_l1_by_combo changes the L1 sample per fold,
    # so we no-op there.

    def precompute_segment_scaler(self, train_pool, df_l1, feature_cols,
                                  target_col, aug_precentage, narrow_l1):
        self._seg_scaler_ctx = None
        if narrow_l1:
            return
        language_col = self._get_language_col(train_pool)
        if language_col is None:
            return
        cols = list(feature_cols)

        # L2 contribution: rows in train_pool with language != native (varies
        # per fold via the train_mask).
        l2_pool_mask = (train_pool[language_col] != self.native_language).to_numpy()
        l2_arr = train_pool[cols].to_numpy(dtype=np.float64, copy=False)[l2_pool_mask]
        l2_not_nan = ~np.isnan(l2_arr)
        l2_safe = np.where(l2_not_nan, l2_arr, 0.0)
        S_l2 = l2_safe.sum(axis=0)
        SS_l2 = (l2_safe * l2_safe).sum(axis=0)
        n_l2 = l2_not_nan.sum(axis=0)

        # L1 contribution: fixed across folds. l1_imputation is idempotent so
        # .copy() just guards against mutating shared state.
        df_l1_local = self.l1_imputation(df_l1.copy(), target_col)
        if aug_precentage == 1:
            l1_sample = df_l1_local
        elif not aug_precentage:
            l1_sample = None
        else:
            l1_sample = df_l1_local.sample(frac=aug_precentage, replace=True, random_state=42)

        if l1_sample is not None and language_col in l1_sample.columns:
            l1_mask = (l1_sample[language_col] != self.native_language).to_numpy()
            l1_arr = l1_sample[cols].to_numpy(dtype=np.float64, copy=False)[l1_mask]
            l1_not_nan = ~np.isnan(l1_arr)
            l1_safe = np.where(l1_not_nan, l1_arr, 0.0)
            S_l1 = l1_safe.sum(axis=0)
            SS_l1 = (l1_safe * l1_safe).sum(axis=0)
            n_l1 = l1_not_nan.sum(axis=0)
        else:
            S_l1 = np.zeros(len(cols))
            SS_l1 = np.zeros(len(cols))
            n_l1 = np.zeros(len(cols), dtype=np.int64)

        self._seg_scaler_ctx = {
            'feature_cols': cols,
            'l2_pool_mask': l2_pool_mask,
            'l2_arr': l2_arr,
            'l2_not_nan': l2_not_nan,
            'S_l2_full': S_l2, 'SS_l2_full': SS_l2, 'n_l2_full': n_l2,
            'S_l1': S_l1, 'SS_l1': SS_l1, 'n_l1': n_l1,
        }

    def derive_fold_scaler(self, train_mask):
        ctx = getattr(self, '_seg_scaler_ctx', None)
        if ctx is None:
            return
        keep_in_l2 = train_mask[ctx['l2_pool_mask']]
        if keep_in_l2.all():
            S_l2, SS_l2, n_l2 = ctx['S_l2_full'], ctx['SS_l2_full'], ctx['n_l2_full']
        else:
            out = ~keep_in_l2
            out_arr = ctx['l2_arr'][out]
            out_not_nan = ctx['l2_not_nan'][out]
            out_safe = np.where(out_not_nan, out_arr, 0.0)
            S_l2 = ctx['S_l2_full'] - out_safe.sum(axis=0)
            SS_l2 = ctx['SS_l2_full'] - (out_safe * out_safe).sum(axis=0)
            n_l2 = ctx['n_l2_full'] - out_not_nan.sum(axis=0)

        S = S_l2 + ctx['S_l1']
        SS = SS_l2 + ctx['SS_l1']
        n = n_l2 + ctx['n_l1']
        n_safe = np.maximum(n, 1)
        means = np.where(n > 0, S / n_safe, 0.0)
        var = np.maximum((SS - n * means * means) / np.maximum(n - 1, 1), 0.0)
        stds = np.sqrt(var)
        stds = np.where(stds == 0, 1.0, stds)
        self._fold_scaler_override = (means, stds, ctx['feature_cols'])

    def clear_fold_scaler(self):
        self._fold_scaler_override = None

    def clear_segment_scaler(self):
        self._seg_scaler_ctx = None

    # ----- closed-form cross-fit LOOCV shortcut -----
    # Replace the per-fold LOPO loop with two RidgeCV(cv=None) passes whose
    # α and scaler come from a stratified A/B half-split, with the LOO
    # computation itself done on the FULL augmented set. Each participant's
    # LOO prediction is leak-free: the α and scaler used for their model
    # were chosen on the OTHER half (so never saw their row), and the
    # Ridge fit excluded them (standard LOO). Per-participant training data
    # is (n-1) — same as full LOPO — not (n/2-1). ~hundreds of times faster
    # than per-fold RidgeCV LOPO on per-text frames.
    #
    # Gated to POOL_ALL × narrow_l1=False × test_pool participants ⊆
    # train_pool, and fully_agg (one row per participant in train_pool —
    # the single-row leave-out identity doesn't apply to first_p/moving_p
    # with multiple rows per participant).
    def try_loocv_segment(self, segment, df_l1, *, fold_method, pool,
                          feature_cols, target_col, aug_precentage, narrow_l1):
        if fold_method != FoldMethod.POOL_ALL:
            return None
        train_pool = segment.train_pool
        test_pool = segment.test_pool
        if train_pool[Fields.SUBJECT_ID].duplicated().any():
            return None

        train_pids = set(train_pool[Fields.SUBJECT_ID])
        test_pids = set(test_pool[Fields.SUBJECT_ID])

        # Two distinct fast paths under POOL_ALL:
        # - test ⊆ train  → leave-one-out on the train pool (SEEN/ALL pools)
        # - test ∩ train = ∅  → one RidgeCV on train, predict test (UNSEEN)
        # Partial overlap → fall through to per-fold loop.
        if test_pids.isdisjoint(train_pids):
            return self._unseen_ridgecv_fast(
                segment, df_l1, target_col, aug_precentage,
                narrow_l1=narrow_l1, pool=pool, feature_cols=feature_cols,
            )
        # Cross-fit LOO predicts train_pool's OWN rows; it never reads
        # test_pool's features. That is what we want when test_pool IS the
        # train rows (SEEN/ALL), but under UNSEEN the prediction must come from
        # the other text half — taking this branch would silently reproduce the
        # SEEN result. MECO UNSEEN reaches here (same participants in both
        # halves), so it is excluded explicitly.
        if pool == Pool.UNSEEN:
            return None
        if not test_pids.issubset(train_pids):
            return None

        # Route to the requested LOO implementation. Both produce one row per
        # test participant in the same format; they differ only in where α and
        # the scaler come from.
        mode = self.loo_mode or _loo_mode()
        if mode == "auto":
            mode = "nested" if len(train_pool) < _AUTO_NESTED_BELOW_N else "crossfit"
        kwargs = dict(pool=pool, feature_cols=feature_cols, target_col=target_col,
                      aug_precentage=aug_precentage, narrow_l1=narrow_l1)
        if mode == "crossfit":
            return self._crossfit_loocv_segment(segment, df_l1, **kwargs)
        return self._nested_loocv_segment(segment, df_l1, **kwargs)

    def _crossfit_loocv_segment(self, segment, df_l1, *, pool, feature_cols,
                                target_col, aug_precentage, narrow_l1):
        """Legacy stratified A/B half-split cross-fit (see the block comment
        above try_loocv_segment). Unchanged from the implementation that
        produced the previously published numbers."""
        train_pool = segment.train_pool
        test_pool = segment.test_pool
        feature_cols = list(feature_cols)
        self.clip_min = 0
        self.clip_max = ALL_TESTS_MAX_SCORES.get(target_col, 100)

        # Build augmented set: L2 train pool + L1 sample (deterministic).
        # OneStop SEEN narrows L1 to readers sharing (list, batch, preview)
        # combos with the L2 train pool (per-fold narrowing collapses to
        # segment-level here since the per-fold combo set differs from the
        # segment combo set only when the held-out participant was the only
        # L2 reader in a combo — rare with multiple readers per combo).
        if (narrow_l1 and pool != Pool.ALL
                and Fields.BATCH in df_l1.columns
                and Fields.BATCH in train_pool.columns):
            df_l1 = self._narrow_l1_by_combo(train_pool, df_l1)
            if len(df_l1) == 0:
                return None
        df_l1_local = self.l1_imputation(df_l1.copy(), target_col)
        if aug_precentage == 1:
            l1_sample = df_l1_local
        elif not aug_precentage:
            l1_sample = None
        else:
            l1_sample = df_l1_local.sample(frac=aug_precentage, replace=True, random_state=42)

        X_l2 = train_pool[feature_cols].to_numpy(dtype=np.float64, copy=False)
        y_l2 = self._target_forward(train_pool[target_col].to_numpy(dtype=np.float64))
        lang_l2 = train_pool[Fields.L1].to_numpy() if Fields.L1 in train_pool.columns else None

        if l1_sample is not None:
            X_l1 = l1_sample[feature_cols].to_numpy(dtype=np.float64, copy=False)
            y_l1 = self._target_forward(l1_sample[target_col].to_numpy(dtype=np.float64))
            lang_l1 = (l1_sample[Fields.L1].to_numpy()
                       if Fields.L1 in l1_sample.columns else None)
            X_aug = np.vstack([X_l2, X_l1])
            y_aug = np.concatenate([y_l2, y_l1])
            if lang_l2 is not None and lang_l1 is not None:
                lang_aug = np.concatenate([lang_l2, lang_l1])
            else:
                lang_aug = None
        else:
            X_aug = X_l2; y_aug = y_l2; lang_aug = lang_l2
        n_l2 = len(X_l2)

        # Stratified A/B split of L2 participants by L1 — within each L1
        # group alternate participants to the two halves.
        # Seed overridable via RIDGE_CF_SEED purely for stability diagnostics:
        # holding the data fixed and varying only this seed isolates how much of
        # a run-to-run difference is the A/B split rather than the data.
        rng = np.random.default_rng(int(os.environ.get("RIDGE_CF_SEED", 42)))
        A_mask_l2 = np.zeros(n_l2, dtype=bool)
        if lang_l2 is not None:
            for grp in pd.unique(lang_l2):
                idxs = np.where(lang_l2 == grp)[0]
                rng.shuffle(idxs)
                A_mask_l2[idxs[:len(idxs) // 2]] = True
        else:
            idxs = np.arange(n_l2)
            rng.shuffle(idxs)
            A_mask_l2[idxs[:n_l2 // 2]] = True
        B_mask_l2 = ~A_mask_l2
        if A_mask_l2.sum() < 2 or B_mask_l2.sum() < 2:
            return None  # too small to split meaningfully

        alphas = _alpha_grid()

        # PHASE 1: per half, fit scaler + pick α on (half_L2 + L1).
        def _half_alpha_and_scaler(half_mask):
            X_half = np.vstack([X_l2[half_mask], X_l1]) if l1_sample is not None else X_l2[half_mask]
            y_half = np.concatenate([y_l2[half_mask], y_l1]) if l1_sample is not None else y_l2[half_mask]
            lang_half = (np.concatenate([lang_l2[half_mask], lang_l1])
                         if (lang_aug is not None and l1_sample is not None) else
                         (lang_l2[half_mask] if lang_l2 is not None else None))
            means, stds = self._cf_scaler(X_half, lang_half)
            X_half_s = self._cf_apply_scaler(X_half, means, stds)
            try:
                rcv = RidgeCV(alphas=alphas, cv=None, fit_intercept=True)
                rcv.fit(X_half_s, y_half)
            except (np.linalg.LinAlgError, ValueError):
                return None
            return float(rcv.alpha_), means, stds

        a_pack = _half_alpha_and_scaler(A_mask_l2)
        b_pack = _half_alpha_and_scaler(B_mask_l2)
        if a_pack is None or b_pack is None:
            return None
        alpha_A, means_A, stds_A = a_pack
        alpha_B, means_B, stds_B = b_pack
        alpha_A, alpha_B = _cf_reconcile_alphas(alpha_A, alpha_B)

        # PHASE 2: full-data RidgeCV with each half's α+scaler → LOO preds.
        def _full_loo(means, stds, alpha):
            X_full_s = self._cf_apply_scaler(X_aug, means, stds)
            try:
                rcv = RidgeCV(alphas=[alpha], cv=None, store_cv_results=True, fit_intercept=True)
                rcv.fit(X_full_s, y_aug)
            except (np.linalg.LinAlgError, ValueError):
                return None
            sq_err = rcv.cv_results_[:, 0]
            full_pred = rcv.predict(X_full_s)
            loo_resid = np.sign(y_aug - full_pred) * np.sqrt(np.maximum(sq_err, 0.0))
            return y_aug - loo_resid

        loo_via_A = _full_loo(means_A, stds_A, alpha_A)
        loo_via_B = _full_loo(means_B, stds_B, alpha_B)
        if loo_via_A is None or loo_via_B is None:
            return None

        # Opt-in diagnostic (LOOCV_DIAG=1): the cross-fit LOO reconstruction is
        # only meaningful while the fit is not interpolating. For per-text sets
        # p can exceed n by >100x, and then RidgeCV can pin alpha at the grid
        # floor, the in-sample residual collapses to ~0, and the reconstructed
        # LOO prediction is dominated by 1/(1-h). Log the inputs so that regime
        # is visible instead of silently clipping at the score bounds.
        if os.environ.get("LOOCV_DIAG"):
            _pre = self._target_inverse(np.where(A_mask_l2, loo_via_B[:n_l2], loo_via_A[:n_l2]))
            _oob = int(((_pre < self.clip_min) | (_pre > self.clip_max)).sum())
            logger.warning(
                f"LOOCV_DIAG n_l2={n_l2} n_aug={len(y_aug)} p={X_aug.shape[1]} "
                f"alpha_A={alpha_A:.3g} alpha_B={alpha_B:.3g} "
                f"grid=[{min(alphas):.3g}..{max(alphas):.3g}] "
                f"pred_pre_clip=[{float(np.min(_pre)):.1f}..{float(np.max(_pre)):.1f}] "
                f"out_of_range={_oob}/{n_l2}"
            )

        # Compose: A's preds from Run Y (using B's α+scaler), B's preds from
        # Run X (using A's α+scaler). Restrict to L2 portion.
        preds_l2 = np.empty(n_l2)
        preds_l2[A_mask_l2] = loo_via_B[:n_l2][A_mask_l2]
        preds_l2[B_mask_l2] = loo_via_A[:n_l2][B_mask_l2]
        # LOO preds are in the fit target space (log space for LogRidge);
        # invert before clipping to the raw score range.
        preds_l2 = np.clip(self._target_inverse(preds_l2), self.clip_min, self.clip_max)

        # Filter to participants in test_pool (usually a no-op now that the
        # alpha-holdout drop is gone).
        test_pids = set(test_pool[Fields.SUBJECT_ID])
        pids = train_pool[Fields.SUBJECT_ID].to_numpy()
        l1_vals = train_pool[Fields.L1].to_numpy()
        true_vals = train_pool[target_col].to_numpy()
        results = []
        for i in range(n_l2):
            if pids[i] not in test_pids:
                continue
            results.append({
                Fields.SUBJECT_ID: pids[i],
                Fields.L1: l1_vals[i],
                'true': true_vals[i],
                'pred': float(preds_l2[i]),
                'fold': f"{segment.segment_id}/{pids[i]}",
            })
        logger.info(f"cross-fit LOOCV: segment={segment.segment_id} α_A={alpha_A:.3g} α_B={alpha_B:.3g} n={len(results)}")
        return results

    def _nested_loocv_segment(self, segment, df_l1, *, pool, feature_cols,
                              target_col, aug_precentage, narrow_l1):
        """Proper nested leave-one-participant-out: one RidgeCV per held-out
        participant, with α AND the scaler fit on that fold's own n-1 rows.

        Semantically identical to what the per-fold loop in
        run_pool_fold_segment would produce (and to what the per-item pipeline
        already does), but kept on numpy arrays rather than re-slicing wide
        pandas frames per fold — that slicing, not the linear algebra, is where
        the generic per-fold loop spends its time.

        No split seed and no half-sized α estimate, so unlike the cross-fit the
        result is deterministic and does not degrade on small segments.

        Per-text sets additionally narrow to fully-covered columns first (~6x
        saving). That is result-neutral: dropped columns are either all-NaN —
        they standardise to all-zero, which a ridge fit ignores — or partially
        observed, and both give predictions identical to 4 decimal places.
        """
        train_pool = segment.train_pool
        test_pool = segment.test_pool
        feature_cols = list(feature_cols)
        self.clip_min = 0
        self.clip_max = ALL_TESTS_MAX_SCORES.get(target_col, 100)

        # L1 narrowing + imputation + augmentation sample: identical to the
        # cross-fit path so the two modes see the same training data.
        if (narrow_l1 and pool != Pool.ALL
                and Fields.BATCH in df_l1.columns
                and Fields.BATCH in train_pool.columns):
            df_l1 = self._narrow_l1_by_combo(train_pool, df_l1)
            if len(df_l1) == 0:
                return None
        df_l1_local = self.l1_imputation(df_l1.copy(), target_col)
        if aug_precentage == 1:
            l1_sample = df_l1_local
        elif not aug_precentage:
            l1_sample = None
        else:
            l1_sample = df_l1_local.sample(frac=aug_precentage, replace=True, random_state=42)

        is_per_text = any(
            c.startswith(TRANSITIONS_FEATURE_PREFIX) or c.startswith(WFC_FEATURE_PREFIX)
            for c in feature_cols
        )
        if is_per_text:
            live_mask = train_pool[feature_cols].notna().all()
            if l1_sample is not None:
                live_mask &= l1_sample[feature_cols].notna().all()
            live = [c for c in feature_cols if live_mask[c]]
            # A segment with no fully-covered column would otherwise fit on an
            # empty design; fall back to the full set and let NaN->0 handle it.
            if live:
                feature_cols = live

        X_l2 = train_pool[feature_cols].to_numpy(dtype=np.float64, copy=False)
        y_l2 = self._target_forward(train_pool[target_col].to_numpy(dtype=np.float64))
        lang_l2 = train_pool[Fields.L1].to_numpy() if Fields.L1 in train_pool.columns else None

        if l1_sample is not None:
            X_l1 = l1_sample[feature_cols].to_numpy(dtype=np.float64, copy=False)
            y_l1 = self._target_forward(l1_sample[target_col].to_numpy(dtype=np.float64))
            lang_l1 = (l1_sample[Fields.L1].to_numpy()
                       if Fields.L1 in l1_sample.columns else None)
        else:
            X_l1 = y_l1 = lang_l1 = None

        n_l2 = len(X_l2)
        pids = train_pool[Fields.SUBJECT_ID].to_numpy()
        test_pids = set(test_pool[Fields.SUBJECT_ID])
        alphas = _alpha_grid()

        # Only participants that are actually scored need a fold. Under
        # SEEN/ALL test_pool == train_pool, so this is normally every row.
        wanted = np.flatnonzero(np.isin(pids, list(test_pids)))
        preds = np.full(n_l2, np.nan)
        picked = np.full(n_l2, np.nan)
        for i in wanted:
            keep = np.ones(n_l2, dtype=bool)
            keep[i] = False
            if l1_sample is not None:
                X_tr = np.vstack([X_l2[keep], X_l1])
                y_tr = np.concatenate([y_l2[keep], y_l1])
                lang_tr = (np.concatenate([lang_l2[keep], lang_l1])
                           if (lang_l2 is not None and lang_l1 is not None) else None)
            else:
                X_tr = X_l2[keep]
                y_tr = y_l2[keep]
                lang_tr = lang_l2[keep] if lang_l2 is not None else None
            means, stds = self._cf_scaler(X_tr, lang_tr)
            try:
                rcv = RidgeCV(alphas=alphas, cv=None, fit_intercept=True)
                rcv.fit(self._cf_apply_scaler(X_tr, means, stds), y_tr)
            except (np.linalg.LinAlgError, ValueError):
                return None
            preds[i] = rcv.predict(self._cf_apply_scaler(X_l2[i:i + 1], means, stds))[0]
            picked[i] = float(rcv.alpha_)

        # LOO preds are in the fit target space (log space for LogRidge);
        # invert before clipping to the raw score range.
        out = np.clip(self._target_inverse(preds), self.clip_min, self.clip_max)

        l1_vals = train_pool[Fields.L1].to_numpy()
        true_vals = train_pool[target_col].to_numpy()
        results = []
        for i in wanted:
            results.append({
                Fields.SUBJECT_ID: pids[i],
                Fields.L1: l1_vals[i],
                'true': true_vals[i],
                'pred': float(out[i]),
                'fold': f"{segment.segment_id}/{pids[i]}",
            })
        if results:
            sel = picked[wanted]
            logger.info(
                f"nested LOOCV: segment={segment.segment_id} n={len(results)} "
                f"p={len(feature_cols)} α=[{np.nanmin(sel):.3g}..{np.nanmax(sel):.3g}] "
                f"({len(np.unique(sel[~np.isnan(sel)]))} distinct)"
            )
        return results

    def _unseen_ridgecv_fast(self, segment, df_l1, target_col, aug_precentage,
                             *, narrow_l1, pool, feature_cols):
        """Direct RidgeCV path for UNSEEN POOL_ALL (test disjoint from train).

        Standardizes the augmented training set once, calls RidgeCV(cv=None)
        which picks α AND fits the final Ridge model in a single pass, then
        predicts test_pool using the same scaler + fitted model.

        Leak-free: train and test participants are disjoint, so test rows
        never inform α selection, scaler stats, or the Ridge fit.
        """
        train_pool = segment.train_pool
        test_pool = segment.test_pool
        feature_cols = list(feature_cols)
        self.clip_min = 0
        self.clip_max = ALL_TESTS_MAX_SCORES.get(target_col, 100)

        # OneStop SEEN/UNSEEN narrows L1 by (list, batch, preview) combo. Since
        # train is invariant across folds here, segment-level narrowing matches
        # the per-fold version exactly.
        if (narrow_l1 and pool != Pool.ALL
                and Fields.BATCH in df_l1.columns
                and Fields.BATCH in train_pool.columns):
            df_l1 = self._narrow_l1_by_combo(train_pool, df_l1)
            if len(df_l1) == 0:
                return None

        df_l1_local = self.l1_imputation(df_l1.copy(), target_col)
        if aug_precentage == 1:
            l1_sample = df_l1_local
        elif not aug_precentage:
            l1_sample = None
        else:
            l1_sample = df_l1_local.sample(frac=aug_precentage, replace=True, random_state=42)

        X_train = train_pool[feature_cols].to_numpy(dtype=np.float64, copy=False)
        y_train = self._target_forward(train_pool[target_col].to_numpy(dtype=np.float64))
        lang_train = (train_pool[Fields.L1].to_numpy()
                      if Fields.L1 in train_pool.columns else None)

        if l1_sample is not None:
            X_l1 = l1_sample[feature_cols].to_numpy(dtype=np.float64, copy=False)
            y_l1 = self._target_forward(l1_sample[target_col].to_numpy(dtype=np.float64))
            lang_l1 = (l1_sample[Fields.L1].to_numpy()
                       if Fields.L1 in l1_sample.columns else None)
            X_aug = np.vstack([X_train, X_l1])
            y_aug = np.concatenate([y_train, y_l1])
            if lang_train is not None and lang_l1 is not None:
                lang_aug = np.concatenate([lang_train, lang_l1])
            else:
                lang_aug = None
        else:
            X_aug = X_train
            y_aug = y_train
            lang_aug = lang_train

        # Segment-level scaler from (language != native) subset.
        means, stds = self._cf_scaler(X_aug, lang_aug)
        X_aug_s = self._cf_apply_scaler(X_aug, means, stds)

        # RidgeCV picks α and fits the final Ridge model in one call.
        alphas = _alpha_grid()
        try:
            rcv = RidgeCV(alphas=alphas, cv=None, fit_intercept=True)
            rcv.fit(X_aug_s, y_aug)
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Predict test_pool with the same segment scaler.
        X_test = test_pool[feature_cols].to_numpy(dtype=np.float64, copy=False)
        X_test_s = self._cf_apply_scaler(X_test, means, stds)
        # RidgeCV was fit in the target space (log space for LogRidge); invert
        # before clipping to the raw score range.
        preds = np.clip(self._target_inverse(rcv.predict(X_test_s)), self.clip_min, self.clip_max)

        # Each test participant becomes one fold row (matches the POOL_ALL
        # fold_iter output that _run_invariant_train_folds would have emitted).
        pids = test_pool[Fields.SUBJECT_ID].to_numpy()
        l1_vals = test_pool[Fields.L1].to_numpy()
        true_vals = test_pool[target_col].to_numpy()
        results = []
        for i in range(len(test_pool)):
            results.append({
                Fields.SUBJECT_ID: pids[i],
                Fields.L1: l1_vals[i],
                'true': true_vals[i],
                'pred': float(preds[i]),
                'fold': f"{segment.segment_id}/{pids[i]}",
            })
        logger.info(
            f"unseen RidgeCV: segment={segment.segment_id} α={rcv.alpha_:.3g} n={len(results)}"
        )
        return results

    def _cf_scaler(self, X_arr, lang_arr):
        """Compute z-score (μ, σ) over the language!=native subset, NaN-aware."""
        if lang_arr is not None:
            fit_mask = lang_arr != self.native_language
            if not fit_mask.any():
                fit_mask = np.ones(len(X_arr), dtype=bool)
        else:
            fit_mask = np.ones(len(X_arr), dtype=bool)
        fit_arr = X_arr if fit_mask.all() else X_arr[fit_mask]
        not_nan = ~np.isnan(fit_arr)
        safe = np.where(not_nan, fit_arr, 0.0)
        n_pc = not_nan.sum(axis=0)
        S = safe.sum(axis=0); SS = (safe * safe).sum(axis=0)
        means = np.where(n_pc > 0, S / np.maximum(n_pc, 1), 0.0)
        var = np.maximum((SS - n_pc * means * means) / np.maximum(n_pc - 1, 1), 0.0)
        stds = np.sqrt(var); stds = np.where(stds == 0, 1.0, stds)
        return means, stds

    def _cf_apply_scaler(self, X_arr, means, stds):
        X = (X_arr - means) / stds
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # ----- target-space hooks for the closed-form fast paths -----
    # The cross-fit LOOCV and unseen-RidgeCV shortcuts run RidgeCV directly on
    # the target and reconstruct predictions from its internals, bypassing
    # fit()/_predict(). Subclasses that transform the target (LogRidge) must
    # apply the transform to y BEFORE RidgeCV sees it and invert it on the
    # predictions, or the fast path silently regresses on the raw scale.
    # Identity here; LogRidgeRegression overrides with log / exp.
    def _target_forward(self, y):
        return y

    def _target_inverse(self, pred):
        return pred

    def _transform(self, X) -> np.ndarray:
        # Reindex means/stds to X's column order, then do all arithmetic in numpy. Pandas (X - series) /
        # series + .replace + .fillna iterate per column and were the per-fold hot path on wide frames.
        cols = getattr(X, 'columns', None)
        X_arr: np.ndarray = X.to_numpy(dtype=np.float64, copy=False) if hasattr(X, 'to_numpy') else X
        if self.scaler_mean_ is None or self.scaler_std_ is None:
            return X_arr
        means = self.scaler_mean_.reindex(cols).to_numpy() if cols is not None else self.scaler_mean_.to_numpy()
        stds = self.scaler_std_.reindex(cols).to_numpy() if cols is not None else self.scaler_std_.to_numpy()
        X_scaled = (X_arr - means) / stds
        X_scaled[~np.isfinite(X_scaled)] = 0.0
        return X_scaled

    def fit(self, X, y):
        y = y.squeeze() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        self.model.fit(X, y)
        # If the model was RidgeCV (choose_alpha=True path), expose the chosen
        # α so logging and downstream consumers can see what was picked.
        if hasattr(self.model, 'alpha_'):
            self.alpha = float(self.model.alpha_)
            logger.info(f"RidgeCV picked alpha={self.alpha:.3g}")
        return self

    def _predict(self, X):
        return self.model.predict(X)

    def prepare_training_data(self, df, feature_cols, target_col):
        # With choose_alpha the model IS a RidgeCV, so the later self.fit(X, y)
        # picks α and fits the final estimator in one RidgeCV.fit.
        if self.choose_alpha:
            self.model = RidgeCV(alphas=_DEFAULT_ALPHA_GRID, cv=None,
                                 fit_intercept=True)
        else:
            self.model = Ridge(alpha=self.alpha)
        self._fit_scaler(df, feature_cols)  # has access to language column here
        X = self._transform(df[feature_cols])
        y = df[target_col]
        return X, y

    def prepare_test_data(self, df, feature_cols):
        X = self._transform(df[feature_cols])
        return X



class LogRidgeRegression(RidgeRegression):
    """Ridge fit on log(target); predictions are exponentiated back."""

    def fit(self, X, y):
        y_log = np.log(y)
        super().fit(X, y_log)
        return self

    def _predict(self, X):
        pred_log = super()._predict(X)
        return np.exp(pred_log)

    # Fast paths (try_loocv_segment / _unseen_ridgecv_fast) regress in log
    # space and invert on the predictions — matching fit()/_predict() above.
    def _target_forward(self, y):
        return np.log(y)

    def _target_inverse(self, pred):
        return np.exp(pred)
