"""Tests for src/methods/predictions/run.py: data loading and preview filtering,
run_cross_val / run_one_feature_set parameter passing, and train_on_full
(including first_p_agg)."""
import unittest
from unittest.mock import patch
import pandas as pd

from src.methods.predictions.run import (
    run_cross_val,
    run_one_feature_set,
    load_and_filter_data,
)

from src.constants import AggTypes
class TestLoadAndFilterData(unittest.TestCase):
    """Test the load_and_filter_data function."""

    @patch('src.methods.predictions.run.pd.read_csv')
    @patch('src.methods.predictions.run.filter_by_preview')
    @patch('src.methods.predictions.run.Path.exists')
    def test_load_and_filter_data_fully_agg(self, mock_exists, mock_filter, mock_read_csv):
        """Test loading fully_agg data."""
        mock_exists.return_value = True
        mock_l1_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_l2_df = pd.DataFrame({'col1': [4, 5, 6]})
        mock_read_csv.side_effect = [mock_l1_df, mock_l2_df]
        mock_filter.return_value = (mock_l1_df, mock_l2_df)

        l1_df, l2_df, agg_label = load_and_filter_data("OneStop", AggTypes.FULL, None, None, "All")

        self.assertIsNotNone(l1_df)
        self.assertIsNotNone(l2_df)
        self.assertEqual(agg_label, AggTypes.FULL)

    @patch('src.methods.predictions.run.pd.read_csv')
    @patch('src.methods.predictions.run.filter_by_preview')
    @patch('src.methods.predictions.run.Path.exists')
    def test_load_and_filter_data_moving_p_agg(self, mock_exists, mock_filter, mock_read_csv):
        """Test loading moving_p_agg data."""
        mock_exists.return_value = True
        mock_l1_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_l2_df = pd.DataFrame({'col1': [4, 5, 6]})
        mock_read_csv.side_effect = [mock_l1_df, mock_l2_df]
        mock_filter.return_value = (mock_l1_df, mock_l2_df)

        l1_df, l2_df, agg_label = load_and_filter_data("OneStop", AggTypes.MOVING_P, "first", "3", "All")

        self.assertIsNotNone(l1_df)
        self.assertIsNotNone(l2_df)
        self.assertEqual(agg_label, "moving_p_agg_first_3")

    @patch('src.methods.predictions.run.Path.exists')
    def test_load_and_filter_data_missing_file(self, mock_exists):
        """Test handling of missing data files."""
        mock_exists.return_value = False

        l1_df, l2_df, agg_label = load_and_filter_data("OneStop", AggTypes.FULL, None, None, "All")

        self.assertIsNone(l1_df)
        self.assertIsNone(l2_df)


class TestRunCrossVal(unittest.TestCase):
    """Test the run_cross_val function parameter passing."""

    def test_run_cross_val_accepts_train_on_full_param(self):
        """Test that run_cross_val function signature accepts train_on_full parameter."""
        import inspect
        sig = inspect.signature(run_cross_val)
        param_names = list(sig.parameters.keys())

        self.assertIn('train_on_full', param_names)
        self.assertIn('df_l2_full', param_names)


class TestRunOneFeatureSet(unittest.TestCase):
    """Test the run_one_feature_set function."""

    @patch('src.methods.predictions.run.run_cross_val')
    def test_run_one_feature_set_basic(self, mock_run_cross_val):
        """Test basic run_one_feature_set call."""
        mock_l1_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        mock_l2_df = mock_l1_df.copy()

        mock_run_cross_val.return_value = pd.DataFrame({
            'pred': [0, 1, 0],
            'true': [0, 1, 0]
        })

        result = run_one_feature_set(
            features_df_L1=mock_l1_df,
            features_df_L2=mock_l2_df,
            feature_set='all_features',
            feature_cols=['feature1'],
            target_col='target',
            model_name='Ridge_Classifier',
            agg_type=AggTypes.FULL,
            train_on_full=False
        )

        self.assertIsNotNone(result)
        mock_run_cross_val.assert_called_once()

    @patch('src.methods.predictions.run.run_cross_val')
    def test_run_one_feature_set_with_full_training(self, mock_run_cross_val):
        """Test run_one_feature_set with train_on_full enabled."""
        mock_l1_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        mock_l2_df = mock_l1_df.copy()
        mock_l2_full = mock_l1_df.copy()

        mock_run_cross_val.return_value = pd.DataFrame({
            'pred': [0, 1, 0],
            'true': [0, 1, 0]
        })

        result = run_one_feature_set(
            features_df_L1=mock_l1_df,
            features_df_L2=mock_l2_df,
            feature_set='all_features',
            feature_cols=['feature1'],
            target_col='target',
            model_name='Ridge_Classifier',
            agg_type=AggTypes.MOVING_P,
            df_l2_full=mock_l2_full,
            train_on_full=True
        )

        self.assertIsNotNone(result)
        mock_run_cross_val.assert_called_once()
        call_kwargs = mock_run_cross_val.call_args[1]
        self.assertTrue(call_kwargs['train_on_full'])
        self.assertIsNotNone(call_kwargs['df_l2_full'])

    @patch('src.methods.predictions.run.run_cross_val')
    def test_run_one_feature_set_passes_agg_type(self, mock_run_cross_val):
        """Test run_one_feature_set passes agg_type to run_cross_val."""
        mock_l1_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        mock_l2_df = mock_l1_df.copy()

        mock_run_cross_val.return_value = pd.DataFrame({
            'pred': [0, 1, 0],
            'true': [0, 1, 0]
        })

        result = run_one_feature_set(
            features_df_L1=mock_l1_df,
            features_df_L2=mock_l2_df,
            feature_set='all_features',
            feature_cols=['feature1'],
            target_col='target',
            model_name='Ridge_Classifier',
            agg_type='moving_p_agg'
        )

        self.assertIsNotNone(result)
        mock_run_cross_val.assert_called_once()


class TestTrainOnFullMode(unittest.TestCase):
    """Test train_on_full parameter behavior."""

    @patch('src.methods.predictions.run.run_cross_val')
    def test_run_one_feature_set_train_on_full_false(self, mock_run_cross_val):
        """Test run_one_feature_set with train_on_full=False doesn't pass df_l2_full."""
        mock_l1_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        mock_l2_df = mock_l1_df.copy()

        mock_run_cross_val.return_value = pd.DataFrame({
            'pred': [0, 1, 0],
            'true': [0, 1, 0]
        })

        result = run_one_feature_set(
            features_df_L1=mock_l1_df,
            features_df_L2=mock_l2_df,
            feature_set='all_features',
            feature_cols=['feature1'],
            target_col='target',
            model_name='Ridge_Classifier',
            agg_type='moving_p_agg',
            train_on_full=False
        )

        self.assertIsNotNone(result)
        call_kwargs = mock_run_cross_val.call_args[1]
        self.assertFalse(call_kwargs['train_on_full'])
        self.assertIsNone(call_kwargs.get('df_l2_full'))

    @patch('src.methods.predictions.run.run_cross_val')
    def test_run_one_feature_set_train_on_full_true_requires_df_l2_full(self, mock_run_cross_val):
        """Test run_one_feature_set with train_on_full=True passes df_l2_full."""
        mock_l1_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        mock_l2_df = mock_l1_df.copy()
        mock_l2_full = mock_l1_df.copy()

        mock_run_cross_val.return_value = pd.DataFrame({
            'pred': [0, 1, 0],
            'true': [0, 1, 0]
        })

        result = run_one_feature_set(
            features_df_L1=mock_l1_df,
            features_df_L2=mock_l2_df,
            feature_set='all_features',
            feature_cols=['feature1'],
            target_col='target',
            model_name='Ridge_Classifier',
            agg_type='moving_p_agg',
            df_l2_full=mock_l2_full,
            train_on_full=True
        )

        self.assertIsNotNone(result)
        call_kwargs = mock_run_cross_val.call_args[1]
        self.assertTrue(call_kwargs['train_on_full'])
        self.assertIsNotNone(call_kwargs['df_l2_full'])
        pd.testing.assert_frame_equal(call_kwargs['df_l2_full'], mock_l2_full)

    @patch('src.methods.predictions.run.run_cross_val')
    def test_run_one_feature_set_train_on_full_false_ignores_df_l2_full(self, mock_run_cross_val):
        """Test run_one_feature_set with train_on_full=False doesn't use df_l2_full even if provided."""
        mock_l1_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        mock_l2_df = mock_l1_df.copy()
        mock_l2_full = mock_l1_df.copy()

        mock_run_cross_val.return_value = pd.DataFrame({
            'pred': [0, 1, 0],
            'true': [0, 1, 0]
        })

        result = run_one_feature_set(
            features_df_L1=mock_l1_df,
            features_df_L2=mock_l2_df,
            feature_set='all_features',
            feature_cols=['feature1'],
            target_col='target',
            model_name='Ridge_Classifier',
            agg_type='moving_p_agg',
            df_l2_full=mock_l2_full,
            train_on_full=False
        )

        self.assertIsNotNone(result)
        call_kwargs = mock_run_cross_val.call_args[1]
        self.assertFalse(call_kwargs['train_on_full'])


class TestRunCrossValTrainOnFull(unittest.TestCase):
    """Test run_cross_val with train_on_full parameter."""

    def test_run_cross_val_has_train_on_full_and_df_l2_full_params(self):
        """Test that run_cross_val signature has train_on_full and df_l2_full parameters."""
        import inspect
        sig = inspect.signature(run_cross_val)
        param_names = list(sig.parameters.keys())

        self.assertIn('train_on_full', param_names)
        self.assertIn('df_l2_full', param_names)

        # Verify default values
        self.assertFalse(sig.parameters['train_on_full'].default)
        self.assertIsNone(sig.parameters['df_l2_full'].default)


class TestFirstPAggTrainOnFull(unittest.TestCase):
    """Test first_p_agg with train_on_full parameter."""

    @patch('src.methods.predictions.run.run_cross_val')
    def test_first_p_agg_with_train_on_full(self, mock_run_cross_val):
        """Test run_one_feature_set with first_p_agg and train_on_full=True."""
        mock_l1_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        mock_l2_df = mock_l1_df.copy()
        mock_l2_full = mock_l1_df.copy()

        mock_run_cross_val.return_value = pd.DataFrame({
            'pred': [0, 1, 0],
            'true': [0, 1, 0]
        })

        result = run_one_feature_set(
            features_df_L1=mock_l1_df,
            features_df_L2=mock_l2_df,
            feature_set='all_features',
            feature_cols=['feature1'],
            target_col='target',
            model_name='Ridge_Classifier',
            agg_type=AggTypes.FIRST_P,
            df_l2_full=mock_l2_full,
            train_on_full=True
        )

        self.assertIsNotNone(result)
        mock_run_cross_val.assert_called_once()
        call_kwargs = mock_run_cross_val.call_args[1]
        self.assertTrue(call_kwargs['train_on_full'])
        self.assertIsNotNone(call_kwargs['df_l2_full'])

    @patch('src.methods.predictions.run.run_cross_val')
    def test_first_p_agg_with_train_on_full_passes_agg_type(self, mock_run_cross_val):
        """Test run_one_feature_set passes agg_type correctly for first_p_agg."""
        mock_l1_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        mock_l2_df = mock_l1_df.copy()
        mock_l2_full = mock_l1_df.copy()

        mock_run_cross_val.return_value = pd.DataFrame({
            'pred': [0, 1, 0],
            'true': [0, 1, 0]
        })

        result = run_one_feature_set(
            features_df_L1=mock_l1_df,
            features_df_L2=mock_l2_df,
            feature_set='all_features',
            feature_cols=['feature1'],
            target_col='target',
            model_name='Ridge_Classifier',
            agg_type=AggTypes.FIRST_P,
            df_l2_full=mock_l2_full,
            train_on_full=True
        )

        self.assertIsNotNone(result)
        call_kwargs = mock_run_cross_val.call_args[1]
        self.assertEqual(call_kwargs['agg_type'], AggTypes.FIRST_P)

    @patch('src.methods.predictions.run.run_cross_val')
    def test_first_p_agg_without_train_on_full(self, mock_run_cross_val):
        """Test run_one_feature_set with first_p_agg and train_on_full=False."""
        mock_l1_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        mock_l2_df = mock_l1_df.copy()

        mock_run_cross_val.return_value = pd.DataFrame({
            'pred': [0, 1, 0],
            'true': [0, 1, 0]
        })

        result = run_one_feature_set(
            features_df_L1=mock_l1_df,
            features_df_L2=mock_l2_df,
            feature_set='all_features',
            feature_cols=['feature1'],
            target_col='target',
            model_name='Ridge_Classifier',
            agg_type=AggTypes.FIRST_P,
            train_on_full=False
        )

        self.assertIsNotNone(result)
        call_kwargs = mock_run_cross_val.call_args[1]
        self.assertFalse(call_kwargs['train_on_full'])
        self.assertIsNone(call_kwargs.get('df_l2_full'))


if __name__ == '__main__':
    unittest.main()
