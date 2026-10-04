"""Every model must impute undefined feature cells, not just the ridge family.

compute_grouped_features no longer zero-fills, so NaN reaches the models and
means "undefined" (RR for the ch_s site, a POS cluster for a tag the participant
never read). Two independent mechanisms cover that:

  • the ridge family (Ridge / LogRidge / Linear) z-scores with
    fold-scoped nanmean/nanstd, then maps non-finite to 0 — the fold mean;
  • everything else falls through to BaseModel.prepare_training_data, which
    imputes to the training-fold column mean.

This walks the model registry and checks the property that matters at the
boundary — no NaN survives into the fitted matrix, and the partially-undefined
participant is still predicted — so a new model that overrides the hooks and
forgets NaN fails here rather than in a sweep.
"""
import unittest

import numpy as np
import pandas as pd

from src.configs import ModelNames
from src.constants import Fields
from src.methods.predictions.run import _create_model

# TabPFN / TabStar are excluded: heavyweight optional deps (and TabPFN does its
# own constant-column pruning), so they are not import-safe in a unit run.
MODELS = [
    ModelNames.RIDGE_CLASSIFIER,
    ModelNames.LOG_RIDGE_REGRESSION,
    ModelNames.LINEAR_REGRESSION,
    ModelNames.RANDOM_FOREST,
    ModelNames.DECISION_TREE,
    ModelNames.LIGHTGBM,
    ModelNames.XGBOOST,
]

# The mean baseline never reads a feature cell, so it is exempt from the
# imputation property below (it hands the frame to .fit untouched) but must
# still survive undefined cells end to end. See AvgModel.
FEATURELESS_MODELS = [ModelNames.AVERAGE]

FEATS = ["reading_speed", "mean_FF", "rr_like"]
TARGET = "lextale_score"


def _frame(n=40):
    """n participants; the last 6 have an undefined `rr_like`, the ch_s shape."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        Fields.SUBJECT_ID: [f"p{i}" for i in range(n)],
        Fields.L1: ["Spanish"] * n,
        "language": ["Spanish"] * n,
        "reading_speed": rng.normal(200, 30, n),
        "mean_FF": rng.normal(210, 20, n),
        "rr_like": rng.normal(0.2, 0.05, n),
        TARGET: rng.uniform(30, 90, n),
    })
    df.loc[df.index[-6:], "rr_like"] = np.nan
    return df


class TestEveryModelImputes(unittest.TestCase):
    def test_no_nan_reaches_the_estimator(self):
        """The matrix handed to .fit must be finite for every model."""
        for name in MODELS:
            with self.subTest(model=str(name)):
                model = _create_model(name)
                df = _frame()
                X, _ = model.prepare_training_data(df, FEATS, TARGET)
                arr = np.asarray(X, dtype=float)
                self.assertFalse(np.isnan(arr).any(),
                                 f"{name} passed NaN through to the estimator")

    def test_fit_and_predict_survive_undefined_cells(self):
        """End to end: undefined cells must not crash or drop the participant."""
        for name in MODELS + FEATURELESS_MODELS:
            with self.subTest(model=str(name)):
                model = _create_model(name)
                df = _frame()
                model.fit_df(df, feature_cols=FEATS, target_col=TARGET)
                preds = model.predict_df(df, feature_cols=FEATS)
                self.assertEqual(len(preds), len(df),
                                 f"{name} lost rows with undefined features")
                self.assertTrue(np.isfinite(np.asarray(preds, dtype=float)).all(),
                                f"{name} produced non-finite predictions")

    def test_undefined_cell_lands_on_the_training_mean(self):
        """Not merely finite — the imputed value must be the fold's own mean.

        Checked through predictions so it holds for both mechanisms: a row whose
        only difference is an undefined cell must score the same as one holding
        that column's training mean explicitly.
        """
        for name in (ModelNames.RIDGE_CLASSIFIER, ModelNames.LINEAR_REGRESSION,
                     ModelNames.RANDOM_FOREST):
            with self.subTest(model=str(name)):
                model = _create_model(name)
                train = _frame()
                model.fit_df(train, feature_cols=FEATS, target_col=TARGET)

                defined = train["rr_like"].dropna()
                probe = pd.DataFrame({
                    Fields.L1: ["Spanish"] * 2, "language": ["Spanish"] * 2,
                    "reading_speed": [200.0, 200.0], "mean_FF": [210.0, 210.0],
                    "rr_like": [np.nan, float(defined.mean())],
                })
                got = np.asarray(model.predict_df(probe, feature_cols=FEATS), dtype=float)
                self.assertAlmostEqual(got[0], got[1], places=6,
                                       msg=f"{name} did not impute to the training mean")


if __name__ == "__main__":
    unittest.main()
