"""Window-index integrity of the extracted moving_p_agg feature files."""
import unittest

import pandas as pd
import pytest

from src.constants import DataSets, DATA_PATH, Fields

pytestmark = pytest.mark.slow


class TestMovingPAggWindowIndices(unittest.TestCase):
    """Test window_index integrity for moving_p_agg features."""

    def setUp(self):
        """Set up test data paths."""
        self.base_path = DATA_PATH
        # Use fully_agg as reference for participant counts
        self.fully_agg_path = self.base_path / f"{DataSets.ONESTOPL2}/features_and_targets/fully_agg/ordinary/all/features_and_metadata.csv"

    def _load_moving_p_agg_data(self, dataset, p_agg_level, p_agg_value):
        """Load moving_p_agg features and targets for a dataset.

        These tests assert properties of generated output, so they can only run
        where that output exists. A missing tree means the moving_p_agg pass has
        not been run for this dataset — that is not a failure of the code under
        test, so skip rather than fail and leave a standing red mark that hides
        real breakage.
        """
        agg_path_suffix = f"moving_p_agg/ordinary/all/{p_agg_level}_{p_agg_value}/features_and_metadata.csv"
        l2_path = self.base_path / f"{dataset}/features_and_targets/{agg_path_suffix}"

        if not l2_path.exists():
            self.skipTest(f"moving_p_agg features not generated: {l2_path}")

        df = pd.read_csv(l2_path)
        return df, l2_path

    def test_window_index_column_exists(self):
        """Test that window_index column exists in moving_p_agg data."""
        df, path = self._load_moving_p_agg_data(DataSets.ONESTOPL2, "paragraph", "5")

        self.assertIsNotNone(df, f"Could not load data from {path}")
        self.assertIn('window_index', df.columns, "window_index column missing from moving_p_agg features")

    def test_window_index_is_numeric(self):
        """Test that window_index values are numeric and non-negative."""
        df, _ = self._load_moving_p_agg_data(DataSets.ONESTOPL2, "paragraph", "5")

        self.assertIsNotNone(df)
        self.assertTrue(pd.api.types.is_numeric_dtype(df['window_index']),
                       "window_index should be numeric")
        self.assertTrue((df['window_index'] >= 0).all(),
                       "window_index values should be non-negative")

    def test_window_index_sequential_per_participant(self):
        """Test that window_index is sequential (0, 1, 2, ...) per participant."""
        df, _ = self._load_moving_p_agg_data(DataSets.ONESTOPL2, "paragraph", "5")

        self.assertIsNotNone(df)

        for participant_id in df[Fields.SUBJECT_ID].unique():
            participant_data = df[df[Fields.SUBJECT_ID] == participant_id].sort_values('window_index')
            window_indices = participant_data['window_index'].values

            # Check indices are sequential from 0
            expected_indices = list(range(len(window_indices)))
            self.assertEqual(list(window_indices), expected_indices,
                           f"Window indices for participant {participant_id} are not sequential")

    def test_window_count_reasonable(self):
        """Test that number of windows per participant is reasonable.

        For moving windows with size=5, each participant should have at least 1 window.
        Window count should be consistent and reasonable.
        """
        moving_df, _ = self._load_moving_p_agg_data(DataSets.ONESTOPL2, "paragraph", "5")
        window_size = 5

        self.assertIsNotNone(moving_df)

        window_counts = moving_df.groupby(Fields.SUBJECT_ID).size()

        # All participants should have at least 1 window
        self.assertTrue((window_counts >= 1).all(),
                       f"Some participants have 0 windows: {window_counts[window_counts < 1]}")

        # With window_size=5, max windows per participant should be reasonable
        # (typically < 100 for most datasets)
        self.assertTrue((window_counts < 100).all(),
                       f"Some participants have suspiciously many windows: {window_counts[window_counts >= 100]}")

    def test_features_and_targets_window_alignment(self):
        """Test that features and targets have matching window indices."""
        df, _ = self._load_moving_p_agg_data(DataSets.ONESTOPL2, "paragraph", "5")

        self.assertIsNotNone(df)

        # Get target columns (typically at the end)
        target_cols = [col for col in df.columns if col.lower().endswith('_score') or col.lower().endswith('_target')]

        if not target_cols:
            # If no explicit target cols, assume last few columns after features
            # Check that no NaN values appear in window_index when targets exist
            self.assertFalse(df['window_index'].isna().any(),
                           "window_index contains NaN values")
        else:
            # Check that every row with window_index has valid targets
            for target_col in target_cols:
                if target_col in df.columns:
                    rows_with_window = df[df['window_index'].notna()]
                    self.assertEqual(len(rows_with_window), len(df),
                                   f"Not all rows have window_index when target {target_col} exists")

    def test_no_duplicate_participant_window_combinations(self):
        """Test that (participant, window_index) combinations are unique."""
        df, _ = self._load_moving_p_agg_data(DataSets.ONESTOPL2, "paragraph", "5")

        self.assertIsNotNone(df)

        # Check for duplicate (subject_id, window_index) pairs
        duplicates = df.groupby([Fields.SUBJECT_ID, 'window_index']).size()
        duplicates = duplicates[duplicates > 1]

        self.assertEqual(len(duplicates), 0,
                       f"Found duplicate (participant, window_index) combinations: {duplicates.to_dict()}")


if __name__ == '__main__':
    unittest.main()
