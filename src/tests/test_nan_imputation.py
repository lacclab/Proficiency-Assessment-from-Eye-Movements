"""Undefined feature cells must be imputed from training-fold means, not dropped.

compute_grouped_features used to zero-fill every undefined cell, so fully-agg
frames never carried NaN and the prediction path could afford to drop or reject
NaN feature rows. With that fill removed, NaN reaches the models and means
"undefined": RR for the ch_s site (firstfix.sac.in is 100% NaN upstream, so the
first-pass denominator is unknown), and POS clusters for a tag the participant
never read.

Guards that the row survives, that the imputed value is the *training* mean, and
that a row with no usable features at all is still dropped.
"""
import unittest

import numpy as np
import pandas as pd

from src.constants import Fields
from src.methods.predictions.models.BaseModel import BaseModel


class _StubModel(BaseModel):
    """Concrete BaseModel; the tests exercise preprocessing, not fitting."""

    def fit(self, X, y):
        self.seen_X = X
        return self

    def _predict(self, X):
        return np.zeros(len(X))


def _frame():
    """4 usable participants + one all-NaN + one bad target.

    p3 is the ch_s shape: defined elsewhere, undefined in `rr`.
    """
    return pd.DataFrame({
        Fields.SUBJECT_ID: ["p0", "p1", "p2", "p3", "p_empty", "p_notarget"],
        Fields.L1: ["Spanish"] * 6,
        "reading_speed": [100.0, 200.0, 300.0, 400.0, np.nan, 150.0],
        "rr": [0.1, 0.2, 0.3, np.nan, np.nan, 0.4],
        "lextale_score": [50.0, 60.0, 70.0, 80.0, 55.0, -1.0],
    })


FEATS = ["reading_speed", "rr"]
TARGET = "lextale_score"


class TestRowSurvival(unittest.TestCase):
    def setUp(self):
        self.model = _StubModel("stub")

    def test_partial_nan_row_is_kept(self):
        """p3 has an undefined rr but real data elsewhere — it must survive."""
        out = self.model.drop_missing_and_negative_ones(_frame(), FEATS, TARGET)
        self.assertIn("p3", set(out[Fields.SUBJECT_ID]))

    def test_all_nan_row_is_dropped(self):
        """Nothing to impute from, so the row goes."""
        out = self.model.drop_missing_and_negative_ones(_frame(), FEATS, TARGET)
        self.assertNotIn("p_empty", set(out[Fields.SUBJECT_ID]))

    def test_sentinel_target_still_dropped(self):
        out = self.model.drop_missing_and_negative_ones(_frame(), FEATS, TARGET)
        self.assertNotIn("p_notarget", set(out[Fields.SUBJECT_ID]))

    def test_cohort_is_not_silently_shrunk(self):
        """Dropping on feature NaN would have removed 2 of 4 usable rows."""
        out = self.model.drop_missing_and_negative_ones(_frame(), FEATS, TARGET)
        self.assertEqual(len(out), 4)


class TestImputation(unittest.TestCase):
    def setUp(self):
        self.model = _StubModel("stub")

    def test_nan_becomes_the_training_mean(self):
        df = _frame().iloc[:4]           # p0..p3, p3's rr undefined
        X, _ = self.model.prepare_training_data(df, FEATS, TARGET)
        self.assertFalse(X.isna().any().any(), "NaN survived preprocessing")
        # mean of the defined rr values, 0.1/0.2/0.3
        self.assertAlmostEqual(float(X.loc[3, "rr"]), 0.2, places=9)

    def test_defined_cells_are_untouched(self):
        df = _frame().iloc[:4]
        X, _ = self.model.prepare_training_data(df, FEATS, TARGET)
        np.testing.assert_allclose(X["reading_speed"].to_numpy(),
                                   [100.0, 200.0, 300.0, 400.0])
        np.testing.assert_allclose(X["rr"].to_numpy()[:3], [0.1, 0.2, 0.3])

    def test_test_rows_use_training_means_not_their_own(self):
        """Otherwise the imputed value would leak the test fold's distribution."""
        train = _frame().iloc[:4]
        self.model.prepare_training_data(train, FEATS, TARGET)
        test = pd.DataFrame({
            Fields.L1: ["Spanish", "Spanish"],
            "reading_speed": [500.0, 600.0],
            "rr": [np.nan, 9.9],          # a test-fold mean would be ~9.9
        })
        Xt = self.model.prepare_test_data(test, FEATS)
        self.assertAlmostEqual(float(Xt.iloc[0]["rr"]), 0.2, places=9)

    def test_fit_df_keeps_the_partially_undefined_participant(self):
        model = _StubModel("stub")
        model.fit_df(_frame(), feature_cols=FEATS, target_col=TARGET)
        self.assertEqual(len(model.seen_X), 5)   # 6 rows minus the -1 target
        self.assertFalse(np.isnan(np.asarray(model.seen_X, dtype=float)).any())


if __name__ == "__main__":
    unittest.main()
