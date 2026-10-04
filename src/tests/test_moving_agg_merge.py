"""A top-up moving-agg build must not multiply its own rows.

add_moving_agg_features merges the freshly computed window features into
`features_and_md_path`. On a fresh build the left side is the metadata — one row
per participant — so merging on the participant alone is right, and the result
fans out to one row per window.

On a top-up build (replace_file=False, file present) the left side is the
function's OWN prior output, already one row per (participant, window). Merging
that on the participant alone is a cartesian product: n_windows x n_windows rows
per participant, which is 144 for MECO's 12 windows. The merge key has to include
the window.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.constants import FeatureGroups, Fields
from src.preprocessing import extract_features as ef

N_PARTICIPANTS, N_WINDOWS = 3, 12          # 12 windows -> 144 if it regresses


def _windows(feature_name):
    """Stand-in for create_moving_agg_features: one row per (participant, window)."""
    return pd.DataFrame([
        {Fields.SUBJECT_ID: f"p{p}", "window_index": w, feature_name: float(w)}
        for p in range(N_PARTICIPANTS) for w in range(N_WINDOWS)
    ])


def _metadata():
    return pd.DataFrame({
        Fields.SUBJECT_ID: [f"p{p}" for p in range(N_PARTICIPANTS)],
        "lextale_score": [50.0, 60.0, 70.0],
    })


class TestMovingAggTopUpMerge(unittest.TestCase):
    def _build(self, path, feature_name, replace_file):
        with patch.object(ef, "create_moving_agg_features",
                          return_value=_windows(feature_name)):
            return ef.add_moving_agg_features(
                None, pd.DataFrame(), _metadata(), [Fields.SUBJECT_ID],
                "MecoL1", path, p_agg_level="paragraph", window_size=1,
                feature_groups_to_add=[FeatureGroups.FIXATION_METRICS],
                replace_file=replace_file,
            )

    def test_fresh_build_fans_out_over_windows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "features_and_metadata.csv"
            out = self._build(path, "mean_FF", replace_file=True)
            self.assertEqual(len(out), N_PARTICIPANTS * N_WINDOWS)

    def test_top_up_does_not_multiply_rows(self):
        """The regression: 3x12 = 36 rows must stay 36, not become 432."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "features_and_metadata.csv"
            self._build(path, "mean_FF", replace_file=True)
            out = self._build(path, "mean_TF", replace_file=False)

            self.assertEqual(len(out), N_PARTICIPANTS * N_WINDOWS,
                             f"row count multiplied: {len(out)} rows for "
                             f"{N_PARTICIPANTS} participants x {N_WINDOWS} windows")
            counts = out.groupby([Fields.SUBJECT_ID, "window_index"]).size()
            self.assertTrue((counts == 1).all(),
                            f"duplicate (participant, window) rows: max {counts.max()}")

    def test_top_up_keeps_both_feature_generations(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "features_and_metadata.csv"
            self._build(path, "mean_FF", replace_file=True)
            out = self._build(path, "mean_TF", replace_file=False)
            for col in ("mean_FF", "mean_TF", "lextale_score", "window_index"):
                self.assertIn(col, out.columns)
            # Values must still line up with their own window, not a cartesian partner.
            np.testing.assert_allclose(
                out.sort_values([Fields.SUBJECT_ID, "window_index"])["mean_TF"].to_numpy(),
                np.tile(np.arange(N_WINDOWS, dtype=float), N_PARTICIPANTS))


if __name__ == "__main__":
    unittest.main()
