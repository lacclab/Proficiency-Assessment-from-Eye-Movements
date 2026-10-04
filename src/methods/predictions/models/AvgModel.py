import numpy as np
import pandas as pd

from src.constants import ALL_TESTS_MAX_SCORES
from src.methods.predictions.models.BaseModel import BaseModel


class AvgModel(BaseModel):
    """Featureless baseline: predict the mean target of the current train set.

    Every participant in a fold's test slice gets the same number — the mean of
    ``target_col`` over that fold's L2 training rows. There is nothing to tune
    and nothing to fit beyond that one scalar, so this is the floor any
    eye-movement model has to beat: it is what you can predict knowing only the
    cohort, not the reader.

    It runs through the *unchanged* orchestration — the same PoolSegments, the
    same LOPO/LOLO/k-fold folds, the same ``drop_missing_and_negative_ones``
    row filtering, the same clipping — so its MAE is directly comparable to any
    other model's cell in the same (dataset, preview, pool, fold method,
    target) position.

    Two deliberate deviations from the other models, both about *what the mean
    is taken over* rather than about the CV structure:

    - **Features are ignored.** A feature set still has to be named on the
      command line, because the orchestrator resolves columns, drops rows with
      all-NaN features and skips cells with no covered column before the model
      is ever called. Pass ``--feature-sets READING_SPEED``: it is one always-
      present, never-NaN column (``reading_speed``, the WPM baseline), so the
      surviving row set is identical to the WPM cell's and the two are exactly
      comparable. The values themselves never reach ``fit``/``_predict``.

    - **L1 augmentation rows are excluded from the mean** (``use_l1_aug=False``,
      the default). L1 readers carry *imputed* targets — ``l1_imputation``
      replaces a missing score with the test max, or with the L1 subset mean —
      so averaging them in would drag every prediction toward a fabricated
      ceiling and measure the imputation rule rather than the cohort. The other
      models use those rows as regularizing signal, where a wrong-but-typical
      target costs little; here the target *is* the prediction. Set
      ``use_l1_aug=True`` to reproduce the augmented-mean variant.
    """

    def __init__(self, use_l1_aug: bool = False):
        super().__init__(model_name="Average")
        self.use_l1_aug = use_l1_aug
        self.train_mean_ = None

    # --- training -----------------------------------------------------------

    def prepare_training_data(self, df, feature_cols, target_col):
        """Pass the frame through untouched — no imputation, no scaling.

        BaseModel's default fills feature NaN from training-fold means, which
        costs a full pass over the design matrix for a model that never looks
        at it. The target is cleaned here instead: callers reach ``fit`` via
        several routes (``fit_df`` drops NaN/-1 targets, ``fit_with_augmentation``
        does not), so the mean is taken over valid targets in every one of them.
        """
        y = df[target_col]
        self._fit_y = y[y.notna() & (y != -1)]
        return df[feature_cols], y

    def prepare_test_data(self, df, feature_cols):
        return df[feature_cols]

    def fit(self, X, y):
        y_fit = getattr(self, "_fit_y", None)
        if y_fit is None:
            y_fit = pd.Series(np.asarray(y, dtype=float).ravel())
            y_fit = y_fit[y_fit.notna() & (y_fit != -1)]
        self._fit_y = None
        if len(y_fit) == 0:
            raise ValueError(
                f"{self.model_name}: no valid target values in the training fold "
                "(all NaN or -1); cannot form a mean."
            )
        self.train_mean_ = float(np.mean(np.asarray(y_fit, dtype=float)))
        return self

    def fit_with_augmentation(self, df_l1, df_l2, aug_precentage, feature_cols=None,
                              target_col=None, random_state=42):
        """Fit on the L2 training fold alone, ignoring the L1 augmentation.

        See the class docstring: L1 targets are imputed, and for a mean-only
        model an imputed target is not weak signal but a directly injected
        wrong answer. ``use_l1_aug=True`` restores BaseModel's behaviour (the
        L1 block is vstacked in and averaged along with the L2 rows).
        """
        if target_col is None or feature_cols is None:
            raise ValueError("feature_cols and target_col must be provided")
        if self.use_l1_aug:
            return super().fit_with_augmentation(
                df_l1, df_l2, aug_precentage, feature_cols, target_col, random_state)
        self.clip_min = 0
        self.clip_max = ALL_TESTS_MAX_SCORES.get(target_col, 100)
        X, y = self.prepare_training_data(df_l2, feature_cols, target_col)
        return self.fit(X, y)

    # --- prediction ---------------------------------------------------------

    def _predict(self, X):
        if self.train_mean_ is None:
            raise ValueError(f"{self.model_name}: predict called before fit.")
        return np.full(len(X), self.train_mean_, dtype=float)
