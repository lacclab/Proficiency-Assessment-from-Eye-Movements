"""Tests for BaseModel.take_relevant_l1_rows: under batch cross-validation the
L1 augmentation is restricted to readers sharing (list, batch, preview) with
the L2 training set. On synthetic frames, and on the real OneStop features
(slow)."""
import unittest
from pathlib import Path

import pandas as pd
import pytest

from src.constants import AggTypes, Fields, FoldsMethods
from src.methods.predictions.models.BaseModel import BaseModel


class DummyModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class TestTakeRelevantL1Rows(unittest.TestCase):
    """Test the take_relevant_l1_rows method for batch cross-validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DummyModel("test_model")

    def test_batch_cross_validation_with_dummy_data(self):
        """Test filtering L1 data to match L2 training data combinations.

        This test creates small dummy dataframes to verify that:
        - L1 rows are kept only when their (list, batch, has_preview) combination exists in L2
        - L1 rows are dropped when their combination is not in L2
        - Temporary columns are properly cleaned up
        """
        # Create dummy L2 training data with 3 samples
        l2_train = pd.DataFrame({
            Fields.SUBJECT_ID: ['l1_100', 'l1_101', 'l2_200'],
            Fields.BATCH: [1, 2, 1],
            Fields.HAS_PREVIEW: [True, False, True],
            'feature1': [1.0, 2.0, 3.0],
        })

        # Create dummy L1 data with 5 samples
        # Should keep: rows with (l1, batch=1, preview=True), (l1, batch=2, preview=False), (l2, batch=1, preview=True)
        # Should drop: rows with (l1, batch=3, preview=True), (l2, batch=2, preview=False)
        df_l1 = pd.DataFrame({
            Fields.SUBJECT_ID: ['l1_102', 'l1_103', 'l1_104', 'l2_201', 'l2_202'],
            Fields.BATCH: [1, 2, 3, 1, 2],
            Fields.HAS_PREVIEW: [True, False, True, True, False],
            'feature1': [1.5, 2.5, 3.5, 4.0, 5.0],
        })

        # Call the method with BATCH_CROSS_VALIDATION
        result = self.model.take_relevant_l1_rows(l2_train, df_l1, FoldsMethods.BATCH_CROSS_VALIDATION)

        # Expected: keep rows where (l1, batch=1, preview=True), (l1, batch=2, preview=False), (l2, batch=1, preview=True)
        # That's indices 0, 1, 3 from df_l1
        expected_subject_ids = {'l1_102', 'l1_103', 'l2_201'}
        result_subject_ids = set(result[Fields.SUBJECT_ID].values)

        self.assertEqual(result_subject_ids, expected_subject_ids)
        self.assertEqual(len(result), 3)

        # Verify temporary columns were dropped
        self.assertNotIn('list', result.columns)
        self.assertNotIn('list_batch_preview', result.columns)

        # Verify original L1 and L2 dataframes were not modified
        self.assertNotIn('list', df_l1.columns)
        self.assertNotIn('list_batch_preview', df_l1.columns)
        self.assertNotIn('list', l2_train.columns)
        self.assertNotIn('list_batch_preview', l2_train.columns)

    def test_non_batch_cross_validation_returns_all_l1(self):
        """Test that non-batch methods return the entire L1 dataframe."""
        l2_train = pd.DataFrame({
            Fields.SUBJECT_ID: ['l1_100', 'l1_101'],
            Fields.BATCH: [1, 2],
            Fields.HAS_PREVIEW: [True, False],
        })

        df_l1 = pd.DataFrame({
            Fields.SUBJECT_ID: ['l1_102', 'l1_103', 'l1_104'],
            Fields.BATCH: [3, 4, 5],
            Fields.HAS_PREVIEW: [True, False, True],
        })

        # Call with a different method
        result = self.model.take_relevant_l1_rows(l2_train, df_l1, FoldsMethods.RANDOM)

        # Should return all of df_l1
        pd.testing.assert_frame_equal(result, df_l1)
        self.assertEqual(len(result), 3)


@pytest.mark.slow
class TestTakeRelevantL1RowsWithRealData(unittest.TestCase):
    """Test with actual data from OneStop L1 and L2 datasets."""

    def setUp(self):
        """Load real data from CSV files."""
        self.model = DummyModel("test_model")

        # Load the real data files
        data_path = Path('data')
        l2_path = data_path / 'OneStopL2' / 'features_and_targets' / AggTypes.FULL / 'ordinary' / 'all' / 'features_and_metadata.csv'
        l1_path = data_path / 'OneStopL1' / 'features_and_targets' / AggTypes.FULL / 'ordinary' / 'all' / 'features_and_metadata.csv'

        if not l2_path.exists() or not l1_path.exists():
            self.skipTest(f"Test data files not found. Expected:\n  {l2_path}\n  {l1_path}")

        self.df_l2 = pd.read_csv(l2_path)
        self.df_l1 = pd.read_csv(l1_path)

    def test_real_data_filtering(self):
        """Test that L1 data is correctly filtered based on L2 train data combinations.

        This verifies:
        - The filtering logic works with real data
        - Result has fewer or equal rows than input L1 data
        - Result only contains (list, batch, has_preview) combinations from L2 train
        """
        # Use the full loaded data
        l2_train = self.df_l2.copy()
        df_l1 = self.df_l1.copy()

        initial_l1_count = len(df_l1)

        # Apply the filtering
        result = self.model.take_relevant_l1_rows(l2_train, df_l1, FoldsMethods.BATCH_CROSS_VALIDATION)

        # Verify result has same columns as input (temporary columns removed)
        self.assertEqual(set(result.columns), set(df_l1.columns))

        # Verify result is subset or equal
        self.assertLessEqual(len(result), initial_l1_count)

        # Verify each result row's (list, batch, preview) combo exists in L2
        for _, result_row in result.iterrows():
            result_list = result_row[Fields.SUBJECT_ID].split('_')[0]
            result_batch = result_row[Fields.BATCH]
            result_preview = result_row[Fields.HAS_PREVIEW]

            # Check if this exact combo exists in L2
            l2_list = l2_train[Fields.SUBJECT_ID].str.split('_').str[0]
            matching_in_l2 = l2_train[
                (l2_list == result_list) &
                (l2_train[Fields.BATCH] == result_batch) &
                (l2_train[Fields.HAS_PREVIEW] == result_preview)
            ]

            self.assertGreater(len(matching_in_l2), 0,
                              f"Result row has combo ({result_list}, {result_batch}, {result_preview}) "
                              f"not found in L2 train data")

        # Verify original dataframes were not modified
        self.assertNotIn('list', df_l1.columns)
        self.assertNotIn('list_batch_preview', df_l1.columns)

    def test_filtered_train_data_subset(self):
        """Test with a specific subset: batch 2, preview False, lists l11, l12, l14.

        This verifies:
        - Result only contains rows matching the filtered L2 train conditions
        - All result rows have batch=2, preview=False, and list in [l11, l12, l14]
        """
        # Load and rename columns as needed
        l2_train = self.df_l2.copy()
        df_l1 = self.df_l1.copy()

        if 'participant_id' in l2_train.columns:
            l2_train = l2_train.rename(columns={'participant_id': Fields.SUBJECT_ID})
        if 'participant_id' in df_l1.columns:
            df_l1 = df_l1.rename(columns={'participant_id': Fields.SUBJECT_ID})

        # Filter L2 train to only batch 2, preview False, lists l11/l12/l14
        l2_train['list'] = l2_train[Fields.SUBJECT_ID].str.split('_').str[0]
        filtered_l2_train = l2_train[
            (l2_train[Fields.BATCH] == 2) &
            (l2_train[Fields.HAS_PREVIEW] == False) &
            (l2_train['list'].isin(['l11', 'l12', 'l14']))
        ].drop(columns=['list'])

        # Apply the filtering
        result = self.model.take_relevant_l1_rows(filtered_l2_train, df_l1, FoldsMethods.BATCH_CROSS_VALIDATION)

        # Verify result has the expected columns (no temp columns)
        self.assertEqual(set(result.columns), set(df_l1.columns))

        # Verify all result rows match the filter criteria
        result['list'] = result[Fields.SUBJECT_ID].str.split('_').str[0]

        # Check batch = 2
        self.assertTrue((result[Fields.BATCH] == 2).all(),
                       f"All result rows should have batch=2, but found: {result[Fields.BATCH].unique()}")

        # Check preview = False
        self.assertTrue((result[Fields.HAS_PREVIEW] == False).all(),
                       f"All result rows should have preview=False, but found: {result[Fields.HAS_PREVIEW].unique()}")

        # Check list in [11, 12, 14]
        valid_lists = {'l11', 'l12', 'l14'}
        result_lists = set(result['list'].unique())
        self.assertTrue(result_lists.issubset(valid_lists),
                       f"All result lists should be in {valid_lists}, but found: {result_lists}")

        self.assertTrue(len(result) > 0, "Expected at least one row in the result, but got an empty dataframe.")
        # Clean up temp column
        result = result.drop(columns=['list'])



if __name__ == '__main__':
    unittest.main()
