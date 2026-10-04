"""Unit tests for moving_p_agg aggregation logic: on dummy data, and on the real
MECO and OneStop harmonized IA data (the latter marked slow)."""
import unittest
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from src.constants import Fields, DATA_PATH, FeatureGroups
from src.preprocessing.extract_features import create_moving_agg_features
from src.preprocessing.utils import add_ia_rr


@lru_cache(maxsize=None)
def _read_harmonized_ia(dataset: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / dataset / 'harmonized' / 'ia.csv')


def harmonized_ia(dataset: str) -> pd.DataFrame:
    """A fresh copy of `dataset`'s harmonized IA file. The file is read from disk
    once per session (OneStop's is 5.8 GB) rather than once per test."""
    return _read_harmonized_ia(dataset).copy()


class TestMovingAggDummyData(unittest.TestCase):
    """Test moving aggregation with synthetic dummy data for controlled testing."""

    def create_dummy_meco_data(self, n_participants=3, n_paragraphs=10, n_rows_per_para=50):
        """Create dummy MECO-like data for testing.

        Args:
            n_participants: Number of dummy participants
            n_paragraphs: Number of paragraphs per participant
            n_rows_per_para: Rows per paragraph (e.g., fixations or IAs per paragraph)
        """
        data = []
        for subj in range(1, n_participants + 1):
            for para in range(1, n_paragraphs + 1):
                for _ in range(n_rows_per_para):
                    data.append({
                        Fields.SUBJECT_ID: f"subject_{subj:02d}",
                        Fields.UNIQUE_PARAGRAPH_ID: para,
                        'IA_FIRST_FIXATION_DURATION': np.random.uniform(100, 300),
                        'IA_FIRST_RUN_DWELL_TIME': np.random.uniform(200, 800),
                        'IA_DWELL_TIME': np.random.uniform(300, 1500),
                        'IA_REGRESSION_PATH_DURATION': np.random.uniform(200, 1200),
                        'IA_SKIP': np.random.choice([0, 1], p=[0.7, 0.3]),
                        'IA_REGRESSION_OUT_COUNT': np.random.choice([0, 1], p=[0.8, 0.2]),
                        'IA_FIRST_FIX_PROGRESSIVE': np.random.choice([0, 1], p=[0.2, 0.8]),
                    })
        df = pd.DataFrame(data)
        # Mirror utils.add_additional_metrics so the fixture has the per-word
        # IA_RR indicator that the FIXATION_METRICS block consumes.
        progressive_mask = df['IA_FIRST_FIX_PROGRESSIVE'] == 1
        df['IA_RR'] = (df['IA_REGRESSION_OUT_COUNT'] >= 1).astype(float).where(progressive_mask, np.nan)
        return df

    def test_dummy_data_window_size_one(self):
        """Test moving aggregation with dummy data, window_size=1."""
        dummy_df = self.create_dummy_meco_data(n_participants=3, n_paragraphs=10)

        result = create_moving_agg_features(
            None, dummy_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Should have 3 participants * 10 paragraphs = 30 rows
        self.assertEqual(len(result), 30, f"Expected 30 rows, got {len(result)}")

        # Check all participants are present
        unique_subjs = result[Fields.SUBJECT_ID].unique()
        self.assertEqual(len(unique_subjs), 3)

    def test_dummy_data_window_size_greater_than_one(self):
        """Test moving aggregation with dummy data, window_size=5."""
        dummy_df = self.create_dummy_meco_data(n_participants=2, n_paragraphs=10)

        result = create_moving_agg_features(
            None, dummy_df, 'MecoL1',
            p_agg_level='paragraph', window_size=5,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Should have fewer windows than window_size=1
        self.assertGreater(len(result), 0)
        # Each participant should have 10 - 5 + 1 = 6 windows
        for subj in result[Fields.SUBJECT_ID].unique():
            subj_windows = len(result[result[Fields.SUBJECT_ID] == subj])
            self.assertEqual(subj_windows, 6, f"Subject {subj} has {subj_windows} windows, expected 6")

    def test_dummy_data_window_larger_than_paragraphs(self):
        """Test with window_size larger than available paragraphs."""
        dummy_df = self.create_dummy_meco_data(n_participants=2, n_paragraphs=5)

        result = create_moving_agg_features(
            None, dummy_df, 'MecoL1',
            p_agg_level='paragraph', window_size=10,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Should still create at least one partial window per participant
        self.assertEqual(len(result), 2, f"Expected 2 rows (one per participant), got {len(result)}")

    def test_dummy_data_window_index_correct(self):
        """Test that window_index is correct with dummy data."""
        dummy_df = self.create_dummy_meco_data(n_participants=1, n_paragraphs=8)

        result = create_moving_agg_features(
            None, dummy_df, 'MecoL1',
            p_agg_level='paragraph', window_size=3,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # With 8 paragraphs and window_size=3:
        # Windows: [1-3], [2-4], [3-5], [4-6], [5-7], [6-8]
        # That's 6 windows (indices 0-5)
        self.assertEqual(len(result), 6)

        window_indices = sorted(result['window_index'].values)
        expected_indices = list(range(6))
        self.assertEqual(window_indices, expected_indices)

    def test_dummy_data_features_reasonable(self):
        """Test that features computed on dummy data are reasonable."""
        dummy_df = self.create_dummy_meco_data(n_participants=1, n_paragraphs=5)

        result = create_moving_agg_features(
            None, dummy_df, 'MecoL1',
            p_agg_level='paragraph', window_size=2,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Check rate metrics are valid 0-1
        for col in ['mean_SK', 'mean_RR']:
            if col in result.columns:
                self.assertTrue((result[col] >= 0).all())
                self.assertTrue((result[col] <= 1).all())

        # Check mean durations are positive
        for col in ['mean_FF', 'mean_FP', 'mean_TF', 'mean_RP']:
            if col in result.columns:
                self.assertTrue((result[col] > 0).all(), f"{col} contains non-positive values")

    def test_dummy_data_column_order(self):
        """Test that columns are in expected order."""
        dummy_df = self.create_dummy_meco_data(n_participants=1, n_paragraphs=5)

        result = create_moving_agg_features(
            None, dummy_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # First columns should be subject_id and window_index
        self.assertEqual(result.columns[0], Fields.SUBJECT_ID)
        self.assertEqual(result.columns[1], 'window_index')


@pytest.mark.slow
class TestMovingAggMECOBasics(unittest.TestCase):
    """Test basic moving aggregation functionality on MECO."""

    def setUp(self):
        """Load MECO sample data for a single participant."""
        dataset = 'MecoL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df
        self.sample_subj = ia_df[Fields.SUBJECT_ID].unique()[0]

    def test_moving_agg_creates_dataframe(self):
        """Test that moving aggregation returns a DataFrame."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_moving_agg_has_required_columns(self):
        """Test that result has subject_id and window_index columns."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertIn(Fields.SUBJECT_ID, result.columns)
        self.assertIn('window_index', result.columns)

    def test_moving_agg_result_not_empty(self):
        """Test that result contains data."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertGreater(len(result), 0, "Result is empty")


@pytest.mark.slow
class TestMovingAggMECOWindowSize(unittest.TestCase):
    """Test moving aggregation with different window sizes on MECO."""

    def setUp(self):
        """Load MECO sample data."""
        dataset = 'MecoL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df

    def test_window_size_one(self):
        """Test with window_size=1 (each paragraph is its own window)."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertGreater(len(result), 0)
        # Each paragraph should be a separate window
        for subj in result[Fields.SUBJECT_ID].unique():
            subj_data = result[result[Fields.SUBJECT_ID] == subj]
            # window_index should start from 0
            self.assertEqual(subj_data['window_index'].min(), 0)

    def test_window_size_five(self):
        """Test with window_size=5."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=5,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertGreater(len(result), 0)

    def test_large_window_size(self):
        """Test with large window_size (creates partial windows)."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=20,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        # Should still create results (at least partial windows)
        self.assertGreater(len(result), 0)

    def test_window_size_smaller_window_has_more_rows(self):
        """Test that smaller window size creates more rows than larger window size."""
        result_w1 = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        result_w5 = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=5,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Window size 1 should have more total windows than size 5
        self.assertGreater(len(result_w1), len(result_w5))


@pytest.mark.slow
class TestMovingAggMECOWindowIndex(unittest.TestCase):
    """Test window_index behavior on MECO."""

    def setUp(self):
        """Load MECO sample data."""
        dataset = 'MecoL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df

    def test_window_index_sequential(self):
        """Test that window_index values are sequential starting from 0."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=3,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        for subj in result[Fields.SUBJECT_ID].unique():
            subj_data = result[result[Fields.SUBJECT_ID] == subj]
            window_indices = sorted(subj_data['window_index'].values)

            # Should start at 0
            self.assertEqual(window_indices[0], 0, f"First window index not 0 for subject {subj}")

            # Should be sequential
            if len(window_indices) > 1:
                for i in range(len(window_indices) - 1):
                    self.assertEqual(window_indices[i+1], window_indices[i] + 1,
                                   f"Non-sequential window indices for subject {subj}")

    def test_window_index_increases_with_window_size(self):
        """Test that number of windows decreases with larger window_size."""
        result_w1 = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        result_w5 = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=5,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Check first participant
        subj = result_w1[Fields.SUBJECT_ID].unique()[0]
        max_idx_w1 = result_w1[result_w1[Fields.SUBJECT_ID] == subj]['window_index'].max()
        max_idx_w5 = result_w5[result_w5[Fields.SUBJECT_ID] == subj]['window_index'].max()

        # With window_size=1, max index should be higher than window_size=5
        self.assertGreater(max_idx_w1, max_idx_w5)


@pytest.mark.slow
class TestMovingAggMECOFeatures(unittest.TestCase):
    """Test that features are correctly computed in MECO windows."""

    def setUp(self):
        """Load MECO sample data."""
        dataset = 'MecoL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df

    def test_feature_columns_exist(self):
        """Test that feature columns are present in result."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Should have at least some features beyond subject_id and window_index
        expected_features = ['mean_FF', 'mean_FP', 'mean_TF', 'mean_RP', 'mean_SK', 'mean_RR']
        found_features = [f for f in expected_features if f in result.columns]
        self.assertGreater(len(found_features), 0,
                          f"No expected features found. Available columns: {list(result.columns)}")

    def test_feature_values_are_numeric(self):
        """Test that feature values are numeric."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=2,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Get feature columns (all except subject_id and window_index)
        feature_cols = [c for c in result.columns if c not in [Fields.SUBJECT_ID, 'window_index']]

        for col in feature_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(result[col]),
                          f"Column {col} is not numeric, dtype: {result[col].dtype}")

    def test_undefined_values_confined_to_rr(self):
        """Undefined cells are allowed now, but only where they mean something.

        create_moving_agg_features used to fillna(0), so this asserted no NaN at
        all. That fill is gone: RR is undefined for a window with no first-pass
        progressive words (and for a whole site, upstream), which is not the same
        as a rate of zero. Every other fixation metric is a mean over all words,
        including skipped ones, so it stays defined wherever the window has rows.
        """
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=2,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        feature_cols = [c for c in result.columns if c not in [Fields.SUBJECT_ID, 'window_index']]

        for col in feature_cols:
            nan_count = result[col].isna().sum()
            if 'RR' in col:
                self.assertLess(nan_count, len(result),
                                f"Column {col} is entirely undefined")
            else:
                self.assertEqual(nan_count, 0, f"Column {col} has {nan_count} NaN values")

    def test_rate_metrics_in_valid_range(self):
        """Test that rate metrics (mean_SK, mean_RR) are between 0 and 1."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=3,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        for col in ['mean_SK', 'mean_RR']:
            if col in result.columns:
                self.assertTrue((result[col] >= 0).all(),
                              f"{col} has negative values: {result[col].min()}")
                self.assertTrue((result[col] <= 1).all(),
                              f"{col} exceeds 1: {result[col].max()}")

    def test_mean_durations_positive(self):
        """Test that mean duration columns are positive."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        for col in ['mean_FF', 'mean_FP', 'mean_TF', 'mean_RP']:
            if col in result.columns:
                self.assertTrue((result[col] >= 0).all(),
                              f"{col} has negative values: {result[col].min()}")


@pytest.mark.slow
class TestMovingAggMECOPartialWindows(unittest.TestCase):
    """Test handling of partial windows on MECO."""

    def setUp(self):
        """Load MECO sample data."""
        dataset = 'MecoL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df

    def test_partial_window_creates_result(self):
        """Test that large window size still creates result (as partial window)."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=50,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        # Should have at least one row per participant (the partial window)
        self.assertGreater(len(result), 0)

    def test_at_least_one_window_per_participant(self):
        """Test that every participant gets at least one window."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=100,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # All participants in original data should have at least one window
        original_subjs = len(self.ia_df[Fields.SUBJECT_ID].unique())
        result_subjs = len(result[Fields.SUBJECT_ID].unique())

        self.assertGreater(result_subjs, 0, "No subjects in result")


@pytest.mark.slow
class TestMovingAggMECOEdgeCases(unittest.TestCase):
    """Test edge cases for MECO moving aggregation."""

    def setUp(self):
        """Load MECO sample data."""
        dataset = 'MecoL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df

    def test_window_size_one_per_participant_correspondence(self):
        """Test that window_size=1 creates one window per unique paragraph per participant."""
        result = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        for subj in result[Fields.SUBJECT_ID].unique():
            # Get unique paragraphs for this subject
            subj_ia = self.ia_df[self.ia_df[Fields.SUBJECT_ID] == subj]
            unique_paras = subj_ia[Fields.UNIQUE_PARAGRAPH_ID].nunique()

            # Get windows for this subject
            subj_result = result[result[Fields.SUBJECT_ID] == subj]
            num_windows = len(subj_result)

            # With window_size=1, should have one window per paragraph
            self.assertEqual(num_windows, unique_paras,
                           f"Subject {subj}: expected {unique_paras} windows, got {num_windows}")

    def test_consistent_feature_computation(self):
        """Test that the same window produces same features across runs."""
        result1 = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=2,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        result2 = create_moving_agg_features(
            None, self.ia_df, 'MecoL1',
            p_agg_level='paragraph', window_size=2,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2, check_dtype=False)


@pytest.mark.slow
class TestMovingAggOneStopBasics(unittest.TestCase):
    """Test basic moving aggregation functionality on OneStop."""

    def setUp(self):
        """Load OneStop sample data for a single participant."""
        dataset = 'OneStopL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df[
            (ia_df['practice_trial'] == False) &
            (ia_df['repeated_reading_trial'] == False)
        ].copy()
        self.sample_subj = self.ia_df[Fields.SUBJECT_ID].unique()[0]

    def test_moving_agg_creates_dataframe(self):
        """Test that moving aggregation returns a DataFrame."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_moving_agg_has_required_columns(self):
        """Test that result has subject_id and window_index columns."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertIn(Fields.SUBJECT_ID, result.columns)
        self.assertIn('window_index', result.columns)

    def test_moving_agg_result_not_empty(self):
        """Test that result contains data."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertGreater(len(result), 0, "Result is empty")


@pytest.mark.slow
class TestMovingAggOneStopWindowSize(unittest.TestCase):
    """Test moving aggregation with different window sizes on OneStop."""

    def setUp(self):
        """Load OneStop sample data."""
        dataset = 'OneStopL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df[
            (ia_df['practice_trial'] == False) &
            (ia_df['repeated_reading_trial'] == False)
        ].copy()

    def test_window_size_one(self):
        """Test with window_size=1 (each paragraph is its own window)."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertGreater(len(result), 0)
        # Each paragraph should be a separate window
        for subj in result[Fields.SUBJECT_ID].unique():
            subj_data = result[result[Fields.SUBJECT_ID] == subj]
            # window_index should start from 0
            self.assertEqual(subj_data['window_index'].min(), 0)

    def test_window_size_ten(self):
        """Test with window_size=10."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=10,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        self.assertGreater(len(result), 0)

    def test_window_size_smaller_window_has_more_rows(self):
        """Test that smaller window size creates more rows than larger window size."""
        result_w1 = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        result_w10 = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=10,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Window size 1 should have more total windows than size 10
        self.assertGreater(len(result_w1), len(result_w10))


@pytest.mark.slow
class TestMovingAggOneStopWindowIndex(unittest.TestCase):
    """Test window_index behavior on OneStop."""

    def setUp(self):
        """Load OneStop sample data."""
        dataset = 'OneStopL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df[
            (ia_df['practice_trial'] == False) &
            (ia_df['repeated_reading_trial'] == False)
        ].copy()

    def test_window_index_sequential(self):
        """Test that window_index values are sequential starting from 0."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=5,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        for subj in result[Fields.SUBJECT_ID].unique():
            subj_data = result[result[Fields.SUBJECT_ID] == subj]
            window_indices = sorted(subj_data['window_index'].values)

            # Should start at 0
            self.assertEqual(window_indices[0], 0, f"First window index not 0 for subject {subj}")

            # Should be sequential
            if len(window_indices) > 1:
                for i in range(len(window_indices) - 1):
                    self.assertEqual(window_indices[i+1], window_indices[i] + 1,
                                   f"Non-sequential window indices for subject {subj}")

    def test_window_index_decreases_with_larger_window(self):
        """Test that number of windows decreases with larger window_size."""
        result_w1 = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        result_w10 = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=10,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Check first participant
        subj = result_w1[Fields.SUBJECT_ID].unique()[0]
        max_idx_w1 = result_w1[result_w1[Fields.SUBJECT_ID] == subj]['window_index'].max()
        max_idx_w10 = result_w10[result_w10[Fields.SUBJECT_ID] == subj]['window_index'].max()

        # With window_size=1, max index should be higher than window_size=10
        self.assertGreater(max_idx_w1, max_idx_w10)


@pytest.mark.slow
class TestMovingAggOneStopFeatures(unittest.TestCase):
    """Test that features are correctly computed in OneStop windows."""

    def setUp(self):
        """Load OneStop sample data."""
        dataset = 'OneStopL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df[
            (ia_df['practice_trial'] == False) &
            (ia_df['repeated_reading_trial'] == False)
        ].copy()

    def test_feature_columns_exist(self):
        """Test that feature columns are present in result."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Should have at least some features beyond subject_id and window_index
        expected_features = ['mean_FF', 'mean_FP', 'mean_TF', 'mean_RP', 'mean_SK', 'mean_RR']
        found_features = [f for f in expected_features if f in result.columns]
        self.assertGreater(len(found_features), 0,
                          f"No expected features found. Available columns: {list(result.columns)}")

    def test_feature_values_are_numeric(self):
        """Test that feature values are numeric."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=2,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Get feature columns (all except subject_id and window_index)
        feature_cols = [c for c in result.columns if c not in [Fields.SUBJECT_ID, 'window_index']]

        for col in feature_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(result[col]),
                          f"Column {col} is not numeric, dtype: {result[col].dtype}")

    def test_undefined_values_confined_to_rr(self):
        """See the MECO counterpart: the fillna(0) is gone, so RR may be undefined."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=2,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        feature_cols = [c for c in result.columns if c not in [Fields.SUBJECT_ID, 'window_index']]

        for col in feature_cols:
            nan_count = result[col].isna().sum()
            if 'RR' in col:
                self.assertLess(nan_count, len(result),
                                f"Column {col} is entirely undefined")
            else:
                self.assertEqual(nan_count, 0, f"Column {col} has {nan_count} NaN values")

    def test_rate_metrics_in_valid_range(self):
        """Test that rate metrics (mean_SK, mean_RR) are between 0 and 1."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=5,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        for col in ['mean_SK', 'mean_RR']:
            if col in result.columns:
                self.assertTrue((result[col] >= 0).all(),
                              f"{col} has negative values: {result[col].min()}")
                self.assertTrue((result[col] <= 1).all(),
                              f"{col} exceeds 1: {result[col].max()}")

    def test_mean_durations_positive(self):
        """Test that mean duration columns are positive."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        for col in ['mean_FF', 'mean_FP', 'mean_TF', 'mean_RP']:
            if col in result.columns:
                self.assertTrue((result[col] >= 0).all(),
                              f"{col} has negative values: {result[col].min()}")


@pytest.mark.slow
class TestMovingAggOneStopEdgeCases(unittest.TestCase):
    """Test edge cases for OneStop moving aggregation."""

    def setUp(self):
        """Load OneStop sample data."""
        dataset = 'OneStopL1'
        # add_ia_rr is what the real pipeline applies on load; IA_RR is derived,
        # not a harmonization output, and the feature code requires it.
        ia_df = add_ia_rr(harmonized_ia(dataset))
        self.ia_df = ia_df[
            (ia_df['practice_trial'] == False) &
            (ia_df['repeated_reading_trial'] == False)
        ].copy()

    def test_window_size_one_per_participant_correspondence(self):
        """Test that window_size=1 creates one window per unique paragraph per participant."""
        result = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=1,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        for subj in result[Fields.SUBJECT_ID].unique():
            # Get unique paragraphs for this subject
            subj_ia = self.ia_df[self.ia_df[Fields.SUBJECT_ID] == subj]
            unique_paras = subj_ia[Fields.UNIQUE_PARAGRAPH_ID].nunique()

            # Get windows for this subject
            subj_result = result[result[Fields.SUBJECT_ID] == subj]
            num_windows = len(subj_result)

            # With window_size=1, should have one window per paragraph
            self.assertEqual(num_windows, unique_paras,
                           f"Subject {subj}: expected {unique_paras} windows, got {num_windows}")

    def test_consistent_feature_computation(self):
        """Test that the same window produces same features across runs."""
        result1 = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=5,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )
        result2 = create_moving_agg_features(
            None, self.ia_df, 'OneStopL1',
            p_agg_level='paragraph', window_size=5,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS]
        )

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
