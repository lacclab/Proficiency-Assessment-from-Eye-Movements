"""Tests for the EyeScore correlation helpers in
src/correlations/between_measures/eyescore.py: discovering and loading the
per-feature-set score files, merging them, the (per-language) correlation
matrices, and the heatmap / CSV outputs."""
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.correlations.between_measures.eyescore import (
    discover_version_paths,
    load_eyescore_files,
    merge_eyescores,
    compute_correlations,
    compute_correlations_per_language,
    _shorten_feature_name,
    plot_correlation_heatmap,
    save_correlations_to_csv,
)


class TestShortenFeatureName(unittest.TestCase):
    """Test the _shorten_feature_name helper function."""

    def test_reading_speed_abbrevation(self):
        """Test that READING_SPEED is shortened to RS."""
        result = _shorten_feature_name("READING_SPEED")
        self.assertEqual(result, "RS")

    def test_fixation_metrics_abbrevation(self):
        """Test that FIXATION_METRICS is shortened to FM."""
        result = _shorten_feature_name("FIXATION_METRICS")
        self.assertEqual(result, "FM")

    def test_s_clusters_abbrevation(self):
        """Test that S_CLUSTERS_NO_NORM is shortened to SC.

        The abbreviation table maps the whole NO_NORM name, so the plain
        S_CLUSTERS keeps its own distinct form (SCLUSTERS) and the two cannot
        collide in a plot legend.
        """
        result = _shorten_feature_name("S_CLUSTERS_NO_NORM")
        self.assertEqual(result, "SC")
        self.assertNotEqual(_shorten_feature_name("S_CLUSTERS"), result)

    def test_wp_coefs_abbrevation(self):
        """Test that WP_COEFS_NO_NORM is shortened to WP, distinctly from WP_COEFS."""
        result = _shorten_feature_name("WP_COEFS_NO_NORM")
        self.assertEqual(result, "WP")
        self.assertNotEqual(_shorten_feature_name("WP_COEFS"), result)

    def test_underscores_removed(self):
        """Test that underscores are removed from the output."""
        result = _shorten_feature_name("TEST_NAME")
        self.assertNotIn("_", result)

    def test_preserves_unknown_names(self):
        """Test that unknown names are handled (underscores removed)."""
        result = _shorten_feature_name("UNKNOWN_FEATURE")
        self.assertEqual(result, "UNKNOWNFEATURE")


class TestMergeEyescores(unittest.TestCase):
    """Test merging eyescore dataframes."""

    def setUp(self):
        """Set up test data."""
        self.eyescore_data1 = pd.DataFrame({
            "participant_id": ["p1", "p2", "p3"],
            "L1": ["English", "Spanish", "French"],
            "eye_score": [0.5, 0.6, 0.7],
        })
        self.eyescore_data2 = pd.DataFrame({
            "participant_id": ["p1", "p2", "p3"],
            "L1": ["English", "Spanish", "French"],
            "eye_score": [0.4, 0.55, 0.65],
        })

    def test_merge_basic(self):
        """Test basic merging of two eyescore dataframes."""
        eyescores = {
            "READING_SPEED": self.eyescore_data1,
            "FIXATION_METRICS": self.eyescore_data2,
        }
        result = merge_eyescores(eyescores)

        # Should have all 3 participants
        self.assertEqual(len(result), 3)
        # Should have participant_id, L1, and 2 feature groups
        self.assertEqual(len(result.columns), 4)
        self.assertIn("READING_SPEED", result.columns)
        self.assertIn("FIXATION_METRICS", result.columns)

    def test_merge_removes_nan(self):
        """Test that rows with NaN in eyescore columns are removed."""
        data_with_nan = self.eyescore_data2.copy()
        data_with_nan.loc[0, "eye_score"] = np.nan

        eyescores = {
            "READING_SPEED": self.eyescore_data1,
            "FIXATION_METRICS": data_with_nan,
        }
        result = merge_eyescores(eyescores)

        # p1 should be removed due to NaN in FIXATION_METRICS
        self.assertEqual(len(result), 2)
        self.assertNotIn("p1", result["participant_id"].values)

    def test_merge_preserves_l1(self):
        """Test that L1 column is preserved in merged data."""
        eyescores = {
            "READING_SPEED": self.eyescore_data1,
            "FIXATION_METRICS": self.eyescore_data2,
        }
        result = merge_eyescores(eyescores)

        self.assertIn("L1", result.columns)
        self.assertEqual(result["L1"].tolist(), ["English", "Spanish", "French"])

    def test_merge_empty_dict(self):
        """Test that empty dict returns empty DataFrame."""
        result = merge_eyescores({})
        self.assertTrue(result.empty)

    def test_merge_single_dataframe(self):
        """Test merging with only one eyescore dataframe."""
        eyescores = {"READING_SPEED": self.eyescore_data1}
        result = merge_eyescores(eyescores)

        self.assertEqual(len(result), 3)
        self.assertIn("READING_SPEED", result.columns)
        self.assertIn("L1", result.columns)


class TestComputeCorrelations(unittest.TestCase):
    """Test correlation computation functions."""

    def setUp(self):
        """Set up test data with multiple eyescore columns."""
        np.random.seed(42)
        self.merged_df = pd.DataFrame({
            "participant_id": [f"p{i}" for i in range(50)],
            "L1": np.random.choice(["English", "Spanish", "French"], 50),
            "READING_SPEED": np.random.rand(50),
            "FIXATION_METRICS": np.random.rand(50),
            "S_CLUSTERS": np.random.rand(50),
        })

    def test_compute_correlations_returns_two_matrices(self):
        """Test that compute_correlations returns both Pearson and Spearman."""
        pearson, spearman = compute_correlations(self.merged_df)

        self.assertIsInstance(pearson, pd.DataFrame)
        self.assertIsInstance(spearman, pd.DataFrame)

    def test_correlation_matrix_shape(self):
        """Test that correlation matrices have correct shape."""
        pearson, spearman = compute_correlations(self.merged_df)

        expected_size = 3  # 3 eyescore columns
        self.assertEqual(pearson.shape, (expected_size, expected_size))
        self.assertEqual(spearman.shape, (expected_size, expected_size))

    def test_correlation_matrix_diagonal_is_one(self):
        """Test that diagonal of correlation matrix is 1."""
        pearson, spearman = compute_correlations(self.merged_df)

        # Diagonal should be 1 (perfect self-correlation)
        np.testing.assert_array_almost_equal(np.diag(pearson), 1.0)
        np.testing.assert_array_almost_equal(np.diag(spearman), 1.0)

    def test_correlation_matrix_symmetric(self):
        """Test that correlation matrices are symmetric."""
        pearson, spearman = compute_correlations(self.merged_df)

        # Should be symmetric
        pd.testing.assert_frame_equal(pearson, pearson.T)
        pd.testing.assert_frame_equal(spearman, spearman.T)

    def test_correlation_values_in_range(self):
        """Test that correlation values are in [-1, 1]."""
        pearson, spearman = compute_correlations(self.merged_df)

        # All values should be in [-1, 1]
        self.assertTrue((pearson >= -1).all().all())
        self.assertTrue((pearson <= 1).all().all())
        self.assertTrue((spearman >= -1).all().all())
        self.assertTrue((spearman <= 1).all().all())

    def test_less_than_two_features_returns_empty(self):
        """Test that less than 2 feature groups returns empty DataFrames."""
        df = self.merged_df[["participant_id", "L1", "READING_SPEED"]].copy()
        pearson, spearman = compute_correlations(df)

        self.assertTrue(pearson.empty)
        self.assertTrue(spearman.empty)


class TestComputeCorrelationsPerLanguage(unittest.TestCase):
    """Test per-language correlation computation."""

    def setUp(self):
        """Set up test data with language groups."""
        np.random.seed(42)
        self.merged_df = pd.DataFrame({
            "participant_id": [f"p{i}" for i in range(50)],
            "L1": ["English"] * 20 + ["Spanish"] * 20 + ["French"] * 10,
            "READING_SPEED": np.random.rand(50),
            "FIXATION_METRICS": np.random.rand(50),
            "S_CLUSTERS": np.random.rand(50),
        })

    def test_returns_dictionaries(self):
        """Test that function returns two dictionaries."""
        pearson_dict, spearman_dict = compute_correlations_per_language(self.merged_df)

        self.assertIsInstance(pearson_dict, dict)
        self.assertIsInstance(spearman_dict, dict)

    def test_includes_all_languages(self):
        """Test that all languages with sufficient data are included."""
        pearson_dict, spearman_dict = compute_correlations_per_language(self.merged_df)

        # English 20, Spanish 20, French 10 — all clear the 3-participant
        # minimum, so all three belong here.
        self.assertIn("English", pearson_dict)
        self.assertIn("Spanish", pearson_dict)
        self.assertIn("French", pearson_dict)

    def test_correlation_matrices_per_language(self):
        """Test that each language has correct correlation matrices."""
        pearson_dict, spearman_dict = compute_correlations_per_language(self.merged_df)

        for lang in ["English", "Spanish"]:
            self.assertIsInstance(pearson_dict[lang], pd.DataFrame)
            self.assertIsInstance(spearman_dict[lang], pd.DataFrame)
            # Should have 3x3 matrices
            self.assertEqual(pearson_dict[lang].shape, (3, 3))

    def test_skips_small_groups(self):
        """Test that language groups with < 3 participants are skipped."""
        # Add tiny language group
        self.merged_df = pd.concat([
            self.merged_df,
            pd.DataFrame({
                "participant_id": ["p51"],
                "L1": ["German"],
                "READING_SPEED": [0.5],
                "FIXATION_METRICS": [0.6],
                "S_CLUSTERS": [0.7],
            })
        ], ignore_index=True)

        pearson_dict, spearman_dict = compute_correlations_per_language(self.merged_df)

        # German should be excluded (only 1 participant)
        self.assertNotIn("German", pearson_dict)


class TestPlotCorrelationHeatmap(unittest.TestCase):
    """Test correlation heatmap plotting."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        data = np.random.rand(3, 3)
        self.corr_matrix = pd.DataFrame(
            data,
            columns=["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS"],
            index=["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS"],
        )

    def tearDown(self):
        """Clean up matplotlib figures."""
        plt.close('all')

    def test_plot_creates_figure(self):
        """Test that plot creates a figure."""
        fig_count_before = len(plt.get_fignums())

        plot_correlation_heatmap(self.corr_matrix, "Test Plot")

        # Figure should have been created and closed
        fig_count_after = len(plt.get_fignums())
        self.assertEqual(fig_count_after, fig_count_before)

    @patch('matplotlib.pyplot.savefig')
    def test_plot_saves_to_file(self, mock_savefig):
        """Test that plot saves to file when save_path is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            plot_correlation_heatmap(self.corr_matrix, "Test Plot", save_path=save_path)

            mock_savefig.assert_called_once()
            call_args = mock_savefig.call_args
            self.assertIsNotNone(call_args)

    @patch('matplotlib.pyplot.show')
    def test_plot_shows_without_save_path(self, mock_show):
        """Test that plot shows when no save_path provided."""
        plot_correlation_heatmap(self.corr_matrix, "Test Plot", save_path=None)
        mock_show.assert_called_once()

    def test_plot_with_custom_figsize(self):
        """Test that custom figsize is used."""
        figsize = (10, 10)
        plot_correlation_heatmap(self.corr_matrix, "Test Plot", figsize=figsize)
        # Just verify it doesn't raise an error

    def test_plot_auto_scales_figsize(self):
        """Test that figsize is auto-scaled based on number of features."""
        # Create larger correlation matrix
        large_corr = pd.DataFrame(
            np.random.rand(10, 10),
            columns=[f"feat_{i}" for i in range(10)],
            index=[f"feat_{i}" for i in range(10)],
        )
        plot_correlation_heatmap(large_corr, "Test Plot")
        # Just verify it doesn't raise an error


class TestSaveCorrelationsToCSV(unittest.TestCase):
    """Test saving correlation matrices to CSV."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        data = np.random.rand(3, 3)
        self.pearson_corr = pd.DataFrame(
            data,
            columns=["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS"],
            index=["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS"],
        )
        self.spearman_corr = self.pearson_corr.copy()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_saves_both_correlation_files(self):
        """Test that both Pearson and Spearman files are saved."""
        save_correlations_to_csv(
            self.pearson_corr,
            self.spearman_corr,
            dataset="TestDataset",
            preview="All",
            save_dir=Path(self.temp_dir),
        )

        pearson_path = Path(self.temp_dir) / "TestDataset" / "All" / "eyescore_correlations_pearson.csv"
        spearman_path = Path(self.temp_dir) / "TestDataset" / "All" / "eyescore_correlations_spearman.csv"

        self.assertTrue(pearson_path.exists())
        self.assertTrue(spearman_path.exists())

    def test_saves_with_version_path(self):
        """Test that files are saved in correct directory with version_path."""
        save_correlations_to_csv(
            self.pearson_corr,
            self.spearman_corr,
            dataset="TestDataset",
            preview="All",
            version_path="fully_agg/all/all",
            save_dir=Path(self.temp_dir),
        )

        pearson_path = (
            Path(self.temp_dir)
            / "TestDataset"
            / "All"
            / "fully_agg/all/all"
            / "eyescore_correlations_pearson.csv"
        )
        self.assertTrue(pearson_path.exists())

    def test_saved_files_are_readable(self):
        """Test that saved files can be read back."""
        save_correlations_to_csv(
            self.pearson_corr,
            self.spearman_corr,
            dataset="TestDataset",
            preview="All",
            save_dir=Path(self.temp_dir),
        )

        pearson_path = Path(self.temp_dir) / "TestDataset" / "All" / "eyescore_correlations_pearson.csv"
        loaded = pd.read_csv(pearson_path, index_col=0)

        self.assertEqual(loaded.shape, self.pearson_corr.shape)


class TestLoadEyescoreFiles(unittest.TestCase):
    """Test loading eyescore files."""

    def setUp(self):
        """Set up temporary directory with test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "OneStop" / "All" / "fully_agg/all/all"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

        # Create sample eyescore files
        self.sample_df1 = pd.DataFrame({
            "participant_id": ["p1", "p2", "p3"],
            "L1": ["English", "Spanish", "French"],
            "eye_score": [0.5, 0.6, 0.7],
        })
        self.sample_df2 = pd.DataFrame({
            "participant_id": ["p1", "p2", "p3"],
            "L1": ["English", "Spanish", "French"],
            "eye_score": [0.4, 0.55, 0.65],
        })

        self.sample_df1.to_csv(self.test_data_dir / "READING_SPEED.csv", index=False)
        self.sample_df2.to_csv(self.test_data_dir / "FIXATION_METRICS.csv", index=False)

    def tearDown(self):
        """Clean up temporary directory."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.correlations.between_measures.eyescore.EYESCORE_RESULTS_DIR')
    def test_load_eyescore_files(self, mock_results_dir):
        """Test loading eyescore files from a directory."""
        mock_results_dir.return_value = Path(self.temp_dir)

        # We need to patch the module-level EYESCORE_RESULTS_DIR
        with patch('src.correlations.between_measures.eyescore.EYESCORE_RESULTS_DIR', Path(self.temp_dir)):
            eyescores = load_eyescore_files("OneStop", "All", "fully_agg/all/all")

            self.assertEqual(len(eyescores), 2)
            self.assertIn("READING_SPEED", eyescores)
            self.assertIn("FIXATION_METRICS", eyescores)

    @patch('src.correlations.between_measures.eyescore.EYESCORE_RESULTS_DIR')
    def test_load_specific_feature_groups(self, mock_results_dir):
        """Test loading only specific feature groups."""
        with patch('src.correlations.between_measures.eyescore.EYESCORE_RESULTS_DIR', Path(self.temp_dir)):
            eyescores = load_eyescore_files(
                "OneStop",
                "All",
                "fully_agg/all/all",
                feature_groups=["READING_SPEED"],
            )

            self.assertEqual(len(eyescores), 1)
            self.assertIn("READING_SPEED", eyescores)
            self.assertNotIn("FIXATION_METRICS", eyescores)


class TestDiscoverVersionPaths(unittest.TestCase):
    """Test discovering available version paths."""

    def setUp(self):
        """Set up temporary directory structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir) / "OneStop" / "All"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create version subdirectories with CSV files
        version1 = self.base_dir / "fully_agg/all/all"
        version2 = self.base_dir / "fully_agg/ordinary/all"
        version1.mkdir(parents=True, exist_ok=True)
        version2.mkdir(parents=True, exist_ok=True)

        # Create dummy CSV files
        (version1 / "READING_SPEED.csv").touch()
        (version2 / "READING_SPEED.csv").touch()

    def tearDown(self):
        """Clean up temporary directory."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.correlations.between_measures.eyescore.EYESCORE_RESULTS_DIR')
    def test_discover_version_paths(self, mock_results_dir):
        """Test discovering available version paths."""
        with patch('src.correlations.between_measures.eyescore.EYESCORE_RESULTS_DIR', Path(self.temp_dir)):
            paths = discover_version_paths("OneStop", "All")

            self.assertEqual(len(paths), 2)
            self.assertIn("fully_agg/all/all", paths)
            self.assertIn("fully_agg/ordinary/all", paths)

    @patch('src.correlations.between_measures.eyescore.EYESCORE_RESULTS_DIR')
    def test_discover_nonexistent_dataset(self, mock_results_dir):
        """Test discovering paths for nonexistent dataset returns empty list."""
        with patch('src.correlations.between_measures.eyescore.EYESCORE_RESULTS_DIR', Path(self.temp_dir)):
            paths = discover_version_paths("NonExistent", "All")

            self.assertEqual(len(paths), 0)


if __name__ == "__main__":
    unittest.main()
