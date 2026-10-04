"""The mean baseline predicts the training fold's mean and nothing else.

AvgModel opts out of the feature pipeline entirely, so the properties worth
pinning are about *which rows the mean is taken over* — the part that decides
whether the number is a floor or an artefact of the L1 imputation rule.
"""
import unittest

import numpy as np
import pandas as pd

from src.configs import ModelNames
from src.constants import Fields
from src.methods.predictions.models.AvgModel import AvgModel
from src.methods.predictions.run import _create_model

FEATS = ["reading_speed", "mean_FF"]
TARGET = "lextale_score"


def _l2(n=20, start=30.0):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        Fields.SUBJECT_ID: [f"p{i}" for i in range(n)],
        Fields.L1: ["Spanish"] * n,
        "language": ["Spanish"] * n,
        "reading_speed": rng.normal(200, 30, n),
        "mean_FF": rng.normal(210, 20, n),
        TARGET: np.linspace(start, start + n - 1, n),
    })


def _l1(n=50):
    """L1 augmentation block, all at the test ceiling — what l1_imputation
    produces for readers who never took the test."""
    return pd.DataFrame({
        Fields.SUBJECT_ID: [f"n{i}" for i in range(n)],
        Fields.L1: ["English"] * n,
        "language": ["English"] * n,
        "reading_speed": np.full(n, 300.0),
        "mean_FF": np.full(n, 190.0),
        TARGET: np.full(n, 100.0),
    })


class TestAvgModel(unittest.TestCase):
    def test_factory_returns_a_fresh_instance(self):
        """_create_model must not hand out the shared MODELS singleton: the
        parallel runner fits one model object per job."""
        a, b = _create_model(ModelNames.AVERAGE), _create_model(ModelNames.AVERAGE)
        self.assertIsInstance(a, AvgModel)
        self.assertIsNot(a, b)

    def test_prediction_is_the_constant_train_mean(self):
        df = _l2()
        model = _create_model(ModelNames.AVERAGE)
        model.fit_df(df, feature_cols=FEATS, target_col=TARGET)
        preds = np.asarray(model.predict_df(df, feature_cols=FEATS), dtype=float)
        self.assertEqual(len(preds), len(df))
        self.assertAlmostEqual(preds[0], df[TARGET].mean(), places=9)
        self.assertEqual(len(np.unique(preds)), 1)

    def test_features_are_ignored(self):
        """Scrambling the features cannot move the prediction."""
        df = _l2()
        scrambled = df.copy()
        scrambled[FEATS] = scrambled[FEATS].to_numpy()[::-1]
        scrambled.loc[scrambled.index[:5], "mean_FF"] = np.nan

        base = _create_model(ModelNames.AVERAGE)
        base.fit_df(df, feature_cols=FEATS, target_col=TARGET)
        other = _create_model(ModelNames.AVERAGE)
        other.fit_df(scrambled, feature_cols=FEATS, target_col=TARGET)
        self.assertAlmostEqual(base.train_mean_, other.train_mean_, places=9)

        preds = np.asarray(other.predict_df(scrambled, feature_cols=FEATS), dtype=float)
        self.assertTrue(np.isfinite(preds).all())

    def test_l1_augmentation_is_excluded_by_default(self):
        """The imputed L1 targets must not drag the mean toward the ceiling."""
        df, l1 = _l2(), _l1()
        model = _create_model(ModelNames.AVERAGE)
        model.fit_with_augmentation(l1, df, 1.0, feature_cols=FEATS, target_col=TARGET)
        self.assertAlmostEqual(model.train_mean_, df[TARGET].mean(), places=9)

        opted_in = AvgModel(use_l1_aug=True)
        opted_in.fit_with_augmentation(l1, df, 1.0, feature_cols=FEATS, target_col=TARGET)
        pooled = pd.concat([df[TARGET], l1[TARGET]]).mean()
        self.assertAlmostEqual(opted_in.train_mean_, pooled, places=9)

    def test_missing_and_sentinel_targets_are_excluded(self):
        """-1 is the 'test not taken' sentinel; averaging it in would be a
        silent 30-point pull on a 0-100 target."""
        df = _l2()
        df.loc[df.index[0], TARGET] = -1
        df.loc[df.index[1], TARGET] = np.nan
        expected = df[TARGET][2:].mean()

        model = _create_model(ModelNames.AVERAGE)
        # fit_with_augmentation, unlike fit_df, does no target cleaning of its
        # own — the model has to do it.
        model.fit_with_augmentation(_l1(), df, 1.0, feature_cols=FEATS, target_col=TARGET)
        self.assertAlmostEqual(model.train_mean_, expected, places=9)

    def test_predictions_are_clipped_to_the_target_range(self):
        model = _create_model(ModelNames.AVERAGE)
        df = _l2()
        df[TARGET] = 150.0  # above the LexTALE ceiling
        model.fit_df(df, feature_cols=FEATS, target_col=TARGET)
        preds = np.asarray(model.predict_df(df, feature_cols=FEATS), dtype=float)
        self.assertTrue((preds <= 100.0).all())


if __name__ == "__main__":
    unittest.main()
