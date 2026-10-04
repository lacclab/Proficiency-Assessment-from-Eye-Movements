import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from sklearn.linear_model import LinearRegression as SkLinearRegression
from statsmodels.stats.outliers_influence import OLSInfluence

from src.constants import Fields, FoldMethod, Pool, ALL_TESTS_MAX_SCORES
from src.methods.predictions.models.RidgeRegression import RidgeRegression


class LinearRegression(RidgeRegression):
    """Ordinary least squares linear regression (sklearn ``LinearRegression``).

    Subclasses ``RidgeRegression`` only to reuse its feature-scaling pipeline
    (``_fit_scaler`` / ``_transform``: z-score on the ``language != native``
    subset with NaN → 0), which every linear model in this repo shares. The
    model itself is unregularized OLS — there is no alpha and nothing to
    select.

    ``try_loocv_segment`` below replaces Ridge's RidgeCV-based shortcut with
    the OLS leave-one-out (PRESS) identity, via statsmodels' ``OLSInfluence``.
    Where its exactness guards don't hold it returns None and the standard
    per-fold fit/predict loop takes over.

    Note: on feature sets with more columns than training rows (the per-text
    WFC/TRANSITIONS frames), OLS has no unique solution; sklearn returns the
    minimum-norm least-squares fit, which interpolates the training data.
    """

    def __init__(
        self,
        random_state: int = 42,
        native_language: str = "English",
    ):
        # choose_alpha stays False: OLS has no regularization strength, so the
        # RidgeCV alpha-selection path is never used.
        super().__init__(
            choose_alpha=False,
            random_state=random_state,
            native_language=native_language,
        )
        self.model_name = "LinearRegression"
        self.model = SkLinearRegression()

    def prepare_training_data(self, df, feature_cols, target_col):
        # Fresh estimator per fold (mirrors RidgeRegression, keeps parallel
        # folds from sharing fitted state), then the shared scaler + transform.
        self.model = SkLinearRegression()
        self._fit_scaler(df, feature_cols)  # language column is available here
        X = self._transform(df[feature_cols])
        y = df[target_col]
        return X, y

    def fit(self, X, y):
        y = y.squeeze() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        self.model.fit(X, y)
        return self

    def try_loocv_segment(self, segment, df_l1, *, fold_method, pool,
                          feature_cols, target_col, aug_precentage, narrow_l1):
        """Closed-form OLS leave-one-out for POOL_ALL segments (test ⊆ train).

        The PRESS identity y_i − ŷ_i^LOO = e_i / (1 − h_ii) is exact for OLS
        (Ridge's closed-form LOO is its regularized generalization), so the
        per-fold LOPO loop — one fit per held-out participant — collapses to
        a single OLS fit; statsmodels' OLSInfluence.resid_press does the
        per-row computation. L1 augmentation rows stay in every fold's
        training set, matching the identity (only row i leaves). Unlike
        Ridge's shortcut no cross-fit split is needed: there is no α to
        select, and full-rank OLS predictions are invariant to affine feature
        scaling, so the full-pool scaler stats leak nothing.

        Exactness guards — fall back to the per-fold loop (return None) when:
        - any feature cell is NaN (the per-fold loop drops uncovered columns
          per fold; imputing from full-pool stats would both change semantics
          and leak the held-out row into the imputed values), or
        - rows ≤ columns + intercept, or the design is rank-deficient / has
          h_ii ≈ 1: no unique OLS solution and the identity degenerates.

        UNSEEN segments are excluded outright. OneStop's are participant-
        disjoint, so train is fold-invariant and the orchestrator already fits
        exactly once via _run_invariant_train_folds. MECO's overlap by
        participant, and there the identity would return the held-out reader's
        TRAIN-half row — the wrong text half for this regime.
        """
        if fold_method != FoldMethod.POOL_ALL:
            return None
        train_pool = segment.train_pool
        test_pool = segment.test_pool
        if train_pool[Fields.SUBJECT_ID].duplicated().any():
            return None
        train_pids = set(train_pool[Fields.SUBJECT_ID])
        test_pids = set(test_pool[Fields.SUBJECT_ID])
        # The PRESS identity gives LOO predictions for train_pool's own rows,
        # which under UNSEEN are the wrong text half (see the docstring).
        if pool == Pool.UNSEEN:
            return None
        if not test_pids.issubset(train_pids):
            return None

        feature_cols = list(feature_cols)
        self.clip_min = 0
        self.clip_max = ALL_TESTS_MAX_SCORES.get(target_col, 100)

        # Build augmented set: L2 train pool + L1 sample (mirrors
        # RidgeRegression.try_loocv_segment, including segment-level combo
        # narrowing for OneStop SEEN).
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
        y_l2 = train_pool[target_col].to_numpy(dtype=np.float64)
        lang_l2 = train_pool[Fields.L1].to_numpy() if Fields.L1 in train_pool.columns else None
        if l1_sample is not None:
            X_l1 = l1_sample[feature_cols].to_numpy(dtype=np.float64, copy=False)
            y_l1 = l1_sample[target_col].to_numpy(dtype=np.float64)
            lang_l1 = (l1_sample[Fields.L1].to_numpy()
                       if Fields.L1 in l1_sample.columns else None)
            X_aug = np.vstack([X_l2, X_l1])
            y_aug = np.concatenate([y_l2, y_l1])
            lang_aug = (np.concatenate([lang_l2, lang_l1])
                        if lang_l2 is not None and lang_l1 is not None else None)
        else:
            X_aug, y_aug, lang_aug = X_l2, y_l2, lang_l2
        n_l2 = len(X_l2)

        if np.isnan(X_aug).any() or np.isnan(y_aug).any():
            return None
        if X_aug.shape[0] <= X_aug.shape[1] + 1:
            return None

        # Scaling cannot change full-rank OLS predictions; keep it anyway so
        # the fit is well-conditioned regardless of feature units.
        means, stds = self._cf_scaler(X_aug, lang_aug)
        X_s = self._cf_apply_scaler(X_aug, means, stds)

        ols_res = sm.OLS(y_aug, sm.add_constant(X_s, has_constant="add")).fit()
        if ols_res.df_resid <= 0 or ols_res.model.exog.shape[1] > np.linalg.matrix_rank(ols_res.model.exog):
            return None  # rank-deficient design → no unique OLS fit
        influence = OLSInfluence(ols_res)
        if (influence.hat_matrix_diag >= 1.0 - 1e-10).any():
            return None  # leverage 1 → LOO identity degenerates
        loo = y_aug - influence.resid_press
        preds_l2 = np.clip(loo[:n_l2], self.clip_min, self.clip_max)

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
        logger.info(f"OLS closed-form LOO (PRESS): segment={segment.segment_id} n={len(results)}")
        return results
