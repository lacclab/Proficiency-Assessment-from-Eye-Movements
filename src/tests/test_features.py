"""Tests for the feature extraction in src/preprocessing (reading speed,
S-clusters, WP-coefficients, participant-level / grouped / transitions / WFC
features, IA-id alignment, the OneStop cascade realignment) and for
validate_seen_only_features."""
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (
    FIXATIONS_METRICS_CLUSTERS,
    FIXATIONS_METRICS_COEFS,
    WORD_PROPERTY_COLUMNS,
    FeatureGroups,
    Fields,
    PosCols,
)
from src.constants import DataSets, Pool
from src.preprocessing.extract_features import (
    add_participant_level_features,
    calc_reading_speed,
    compute_grouped_features,
    compute_ia_participant_level_features,
    compute_transitions_features,
    compute_wfc_features,
    create_s_clusters_dict,
    create_s_clusters_dict_inner,
    find_wp_coefs,
    find_wp_coefs_inner,
    flip_group_to_features,
    validate_ia_id_alignment,
)
from src.preprocessing.utils import realign_ia_cascade_onestop
from src.methods.predictions.models.BaseModel import validate_seen_only_features


def _make_ia_df(n_words=20, n_trials=2, participant_id="p1"):
    """Create a minimal IA DataFrame with required columns for testing."""
    rng = np.random.RandomState(42)
    rows = []
    for trial_idx in range(n_trials):
        for word_idx in range(n_words):
            rows.append({
                Fields.SUBJECT_ID: participant_id,
                Fields.UNIQUE_PARAGRAPH_ID: f"para_{trial_idx}",
                Fields.UNIQUE_TRIAL_ID: f"trial_{trial_idx}",
                "IA_ID": word_idx,
                "IA_FIRST_FIXATION_DURATION": rng.uniform(100, 400),
                "IA_FIRST_RUN_DWELL_TIME": rng.uniform(100, 500),
                "IA_DWELL_TIME": rng.uniform(100, 600),
                "IA_REGRESSION_PATH_DURATION": rng.uniform(100, 700),
                "IA_SKIP": rng.choice([0, 1], p=[0.8, 0.2]),
                "IA_REGRESSION_OUT_COUNT": rng.choice([0, 1]),
                "IA_FIRST_FIX_PROGRESSIVE": rng.choice([0, 1], p=[0.3, 0.7]),
                "total_skip": rng.choice([0, 1], p=[0.8, 0.2]),
                "PARAGRAPH_RT": rng.uniform(5000, 15000),
                "normalized_ID": word_idx / n_words,
                # word properties
                "word_length": rng.randint(1, 15),
                "wordfreq_frequency": rng.uniform(-10, -1),
                "gpt2_surprisal": rng.uniform(0, 20),
                "ptb_pos": rng.choice(["NN", "VB", "JJ", "DT"]),
                "universal_pos": rng.choice(["VERB", "NOUN", "ADJ", "ADV", "DET"]),
                "is_content_word": rng.choice([0, 1]),
            })
    df = pd.DataFrame(rows)
    # Mirror utils.add_additional_metrics so tests that bypass the
    # harmonization pipeline still get the per-word IA_RR indicator.
    progressive_mask = df['IA_FIRST_FIX_PROGRESSIVE'] == 1
    df['IA_RR'] = (df['IA_REGRESSION_OUT_COUNT'] >= 1).astype(float).where(progressive_mask, np.nan)
    return df


def _make_fixation_df(n_fixations=30, n_trials=2, participant_id="p1"):
    """Create a minimal fixation DataFrame for testing."""
    rng = np.random.RandomState(42)
    rows = []
    for trial_idx in range(n_trials):
        for _ in range(n_fixations):
            rows.append({
                Fields.SUBJECT_ID: participant_id,
                Fields.UNIQUE_PARAGRAPH_ID: f"para_{trial_idx}",
                Fields.UNIQUE_TRIAL_ID: f"trial_{trial_idx}",
                "CURRENT_FIX_INTEREST_AREA_INDEX": rng.randint(0, 10),
                "CURRENT_FIX_DURATION": rng.uniform(50, 500),
            })
    return pd.DataFrame(rows)


class TestCalcReadingSpeed(unittest.TestCase):
    def test_basic_reading_speed(self):
        """Reading speed = num_words / sum(paragraph reading times per trial)."""
        df = pd.DataFrame({
            Fields.UNIQUE_TRIAL_ID: ["t1"] * 5 + ["t2"] * 5,
            "PARAGRAPH_RT": [1000] * 5 + [2000] * 5,
        })
        # 10 words, trial t1 RT=1000ms, trial t2 RT=2000ms → 10 words / 3s = 200 WPM
        speed = calc_reading_speed(df)
        self.assertAlmostEqual(speed, 200.0)

    def test_single_trial(self):
        df = pd.DataFrame({
            Fields.UNIQUE_TRIAL_ID: ["t1"] * 3,
            "PARAGRAPH_RT": [500, 500, 500],
        })
        # 3 words, single trial RT=500ms → 3 words / 0.5s = 360 WPM
        speed = calc_reading_speed(df)
        self.assertAlmostEqual(speed, 360.0)


class TestFlipGroupToFeatures(unittest.TestCase):
    def test_pivot_produces_flat_columns(self):
        criterion = "ptb_pos"
        fixation_metrics = ["FF", "TF"]
        df = pd.DataFrame({
            criterion: ["NN", "VB"],
            f"{criterion}_FF": [100.0, 200.0],
            f"{criterion}_TF": [300.0, 400.0],
        })
        result = flip_group_to_features(df, criterion, fixation_metrics)
        self.assertEqual(result.shape[0], 1)
        # Should have 4 columns: NN_FF, NN_TF, VB_FF, VB_TF
        self.assertEqual(result.shape[1], 4)
        for pos in ["NN", "VB"]:
            for met in fixation_metrics:
                col = f"{pos}_{criterion}_{met}"
                self.assertIn(col, result.columns)


class TestCreateSClustersDictInner(unittest.TestCase):
    def test_returns_dict_with_expected_keys(self):
        rng = np.random.RandomState(0)
        n = 50
        df = pd.DataFrame({
            "ptb_pos": rng.choice(["NN", "VB"], size=n),
            "FF": rng.uniform(100, 400, n),
            "TF": rng.uniform(100, 600, n),
        })
        result = create_s_clusters_dict_inner(df, ["FF", "TF"], "ptb_pos", normalize_RS=True)
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)
        # Normalized values should be close to 1 on average
        for key in result:
            self.assertNotIn("no_norm", key)

    def test_no_norm_suffix(self):
        rng = np.random.RandomState(0)
        n = 50
        df = pd.DataFrame({
            "ptb_pos": rng.choice(["NN", "VB"], size=n),
            "FF": rng.uniform(100, 400, n),
        })
        result = create_s_clusters_dict_inner(df, ["FF"], "ptb_pos", normalize_RS=False)
        for key in result:
            self.assertTrue(key.endswith("_no_norm"))


class TestCreateSClustersDict(unittest.TestCase):
    def test_renames_columns_before_computing(self):
        """create_s_clusters_dict should rename IA columns to nicknames before computing."""
        rng = np.random.RandomState(0)
        n = 50
        # Use original IA column names
        df = pd.DataFrame({
            "IA_FIRST_FIXATION_DURATION": rng.uniform(100, 400, n),
            "IA_FIRST_RUN_DWELL_TIME": rng.uniform(100, 500, n),
            "IA_DWELL_TIME": rng.uniform(100, 600, n),
            "IA_REGRESSION_PATH_DURATION": rng.uniform(100, 700, n),
            "IA_SKIP": rng.choice([0, 1], size=n),
            "IA_RR": rng.choice([0.0, 1.0], size=n),
            "ptb_pos": rng.choice(["NN", "VB", "IN"], size=n),
        })
        result = create_s_clusters_dict(df, FIXATIONS_METRICS_CLUSTERS, PosCols.PTB_POS, normalize_RS=True)
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)


class TestFindWpCoefsInner(unittest.TestCase):
    def test_returns_coefficients_for_each_metric(self):
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame({
            "word_length": rng.randint(1, 15, n),
            "wordfreq_frequency": rng.uniform(-10, -1, n),
            "gpt2_surprisal": rng.uniform(0, 20, n),
            "FF": rng.uniform(100, 400, n),
            "FP": rng.uniform(100, 500, n),
        })
        result = find_wp_coefs_inner(df, ["FF", "FP"], normalize_RS=True)
        self.assertIsInstance(result, dict)
        # Should have intercept + 3 word properties per metric = 4 * 2 = 8
        self.assertEqual(len(result), 8)
        for metric in ["FF", "FP"]:
            self.assertIn(f"{metric}_intercept", result)
            for wp in WORD_PROPERTY_COLUMNS:
                self.assertIn(f"{metric}_{wp}_coef", result)

    def test_no_norm_suffix(self):
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame({
            "word_length": rng.randint(1, 15, n),
            "wordfreq_frequency": rng.uniform(-10, -1, n),
            "gpt2_surprisal": rng.uniform(0, 20, n),
            "FF": rng.uniform(100, 400, n),
        })
        result = find_wp_coefs_inner(df, ["FF"], normalize_RS=False)
        for key in result:
            self.assertTrue(key.endswith("_no_norm"))


class TestFindWpCoefs(unittest.TestCase):
    def test_renames_columns_before_regression(self):
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame({
            "word_length": rng.randint(1, 15, n),
            "wordfreq_frequency": rng.uniform(-10, -1, n),
            "gpt2_surprisal": rng.uniform(0, 20, n),
            "IA_FIRST_FIXATION_DURATION": rng.uniform(100, 400, n),
            "IA_FIRST_RUN_DWELL_TIME": rng.uniform(100, 500, n),
            "IA_DWELL_TIME": rng.uniform(100, 600, n),
            "IA_REGRESSION_PATH_DURATION": rng.uniform(100, 700, n),
            "IA_SKIP": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            "IA_RR": rng.choice([0.0, 1.0], size=n, p=[0.7, 0.3]),
        })
        result = find_wp_coefs(df, FIXATIONS_METRICS_COEFS, normalize_RS=True)
        self.assertEqual(len(result), len(FIXATIONS_METRICS_COEFS) * (len(WORD_PROPERTY_COLUMNS) + 1))


class TestComputeIaParticipantLevelFeatures(unittest.TestCase):
    def test_reading_speed_feature(self):
        df = _make_ia_df()
        result = compute_ia_participant_level_features(df, [FeatureGroups.READING_SPEED])
        self.assertIn("reading_speed", result)
        self.assertGreater(result["reading_speed"], 0)

    def test_fixation_metrics_features(self):
        df = _make_ia_df()
        result = compute_ia_participant_level_features(df, [FeatureGroups.FIXATION_METRICS])
        expected_keys = {"mean_FF", "mean_FP", "mean_TF", "mean_RP", "mean_SK", "mean_RR"}
        self.assertEqual(expected_keys, set(result.keys()))
        for v in result.values():
            self.assertIsNotNone(v)

    def test_multiple_feature_groups(self):
        df = _make_ia_df()
        result = compute_ia_participant_level_features(
            df, [FeatureGroups.READING_SPEED, FeatureGroups.FIXATION_METRICS]
        )
        self.assertIn("reading_speed", result)
        self.assertIn("mean_SK", result)

    def test_s_clusters_feature(self):
        df = _make_ia_df()
        result = compute_ia_participant_level_features(df, [FeatureGroups.S_CLUSTERS])
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)

    def test_wp_coefs_feature(self):
        df = _make_ia_df(n_words=50, n_trials=3)
        result = compute_ia_participant_level_features(df, [FeatureGroups.WP_COEFS])
        self.assertIsInstance(result, dict)
        self.assertTrue(any("intercept" in k for k in result))
        self.assertTrue(any("_coef" in k for k in result))

    def test_wp_coefs_no_intercept(self):
        df = _make_ia_df(n_words=50, n_trials=3)
        result = compute_ia_participant_level_features(df, [FeatureGroups.WP_COEFS_NO_INTERCEPT])
        self.assertIsInstance(result, dict)
        self.assertFalse(any("intercept" in k for k in result))


class TestComputeGroupedFeatures(unittest.TestCase):
    def test_groups_by_participant(self):
        # Create data for 2 participants
        df1 = _make_ia_df(participant_id="p1")
        df2 = _make_ia_df(participant_id="p2")
        df = pd.concat([df1, df2], ignore_index=True)

        grouped_partial = partial(
            compute_ia_participant_level_features,
            feature_groups=[FeatureGroups.READING_SPEED],
        )
        result = compute_grouped_features(
            df, grouped_partial, label="test",
            participant_groupby_columns=[Fields.SUBJECT_ID],
        )
        self.assertEqual(len(result), 2)
        self.assertIn("reading_speed", result.columns)

    def test_fills_nan_with_zero(self):
        df = _make_ia_df(participant_id="p1")
        grouped_partial = partial(
            compute_ia_participant_level_features,
            feature_groups=[FeatureGroups.READING_SPEED],
        )
        result = compute_grouped_features(
            df, grouped_partial, label="test",
            participant_groupby_columns=[Fields.SUBJECT_ID],
        )
        self.assertFalse(result.isna().any().any())


class TestAddParticipantLevelFeatures(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp_dir.name) / "test_features_output.csv"

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_creates_new_file(self):
        ia_df = _make_ia_df(participant_id="p1")
        metadata = pd.DataFrame({Fields.SUBJECT_ID: ["p1"]})
        result = add_participant_level_features(
            harmonized_fixation_data=None,
            harmonized_ia_data=ia_df,
            metadata_df=metadata,
            participant_groupby_columns=[Fields.SUBJECT_ID],
            features_and_md_path=self.tmp_path,
            feature_groups_to_add=[FeatureGroups.READING_SPEED],
            replace_file=True,
        )
        self.assertTrue(self.tmp_path.exists())
        self.assertIn("reading_speed", result.columns)
        self.assertEqual(len(result), 1)

    def test_appends_features_to_existing(self):
        ia_df = _make_ia_df(participant_id="p1")
        metadata = pd.DataFrame({Fields.SUBJECT_ID: ["p1"]})
        # First call: create with reading speed
        add_participant_level_features(
            harmonized_fixation_data=None,
            harmonized_ia_data=ia_df,
            metadata_df=metadata,
            participant_groupby_columns=[Fields.SUBJECT_ID],
            features_and_md_path=self.tmp_path,
            feature_groups_to_add=[FeatureGroups.READING_SPEED],
            replace_file=True,
        )
        # Second call: add fixation metrics
        result = add_participant_level_features(
            harmonized_fixation_data=None,
            harmonized_ia_data=ia_df,
            metadata_df=None,
            participant_groupby_columns=[Fields.SUBJECT_ID],
            features_and_md_path=self.tmp_path,
            feature_groups_to_add=[FeatureGroups.FIXATION_METRICS],
        )
        self.assertIn("reading_speed", result.columns)
        self.assertIn("mean_SK", result.columns)

    def test_replace_existing_features(self):
        ia_df = _make_ia_df(participant_id="p1")
        metadata = pd.DataFrame({Fields.SUBJECT_ID: ["p1"]})
        # Create initial
        result1 = add_participant_level_features(
            harmonized_fixation_data=None,
            harmonized_ia_data=ia_df,
            metadata_df=metadata,
            participant_groupby_columns=[Fields.SUBJECT_ID],
            features_and_md_path=self.tmp_path,
            feature_groups_to_add=[FeatureGroups.READING_SPEED],
            replace_file=True,
        )
        speed1 = result1["reading_speed"].iloc[0]
        # Re-add same feature (should replace, not duplicate)
        result2 = add_participant_level_features(
            harmonized_fixation_data=None,
            harmonized_ia_data=ia_df,
            metadata_df=None,
            participant_groupby_columns=[Fields.SUBJECT_ID],
            features_and_md_path=self.tmp_path,
            feature_groups_to_add=[FeatureGroups.READING_SPEED],
        )
        # Should still have exactly one reading_speed column
        self.assertEqual(list(result2.columns).count("reading_speed"), 1)
        self.assertAlmostEqual(result2["reading_speed"].iloc[0], speed1)

    def test_raises_without_metadata_on_new_file(self):
        ia_df = _make_ia_df(participant_id="p1")
        with self.assertRaises(ValueError):
            add_participant_level_features(
                harmonized_fixation_data=None,
                harmonized_ia_data=ia_df,
                metadata_df=None,
                participant_groupby_columns=[Fields.SUBJECT_ID],
                features_and_md_path=self.tmp_path,
                feature_groups_to_add=[FeatureGroups.READING_SPEED],
                replace_file=True,
            )

    def test_multiple_participants(self):
        ia_df1 = _make_ia_df(participant_id="p1")
        ia_df2 = _make_ia_df(participant_id="p2")
        ia_df = pd.concat([ia_df1, ia_df2], ignore_index=True)
        metadata = pd.DataFrame({Fields.SUBJECT_ID: ["p1", "p2"]})
        result = add_participant_level_features(
            harmonized_fixation_data=None,
            harmonized_ia_data=ia_df,
            metadata_df=metadata,
            participant_groupby_columns=[Fields.SUBJECT_ID],
            features_and_md_path=self.tmp_path,
            feature_groups_to_add=[FeatureGroups.READING_SPEED, FeatureGroups.FIXATION_METRICS],
            replace_file=True,
        )
        self.assertEqual(len(result), 2)
        self.assertIn("reading_speed", result.columns)
        self.assertIn("mean_SK", result.columns)


class TestComputeTransitionsFeatures(unittest.TestCase):
    def test_counts_match_manual_pairs(self):
        # p1 in paragraph para_0: saccades 0->1, 1->2, 2->1, 1->2  ->  (0,1):1, (1,2):2, (2,1):1
        # p1 in paragraph para_1: saccade 0->0  ->  (0,0):1
        # p2 in paragraph para_0: saccade 0->2  ->  (0,2):1
        rows = [
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "CURRENT_FIX_INTEREST_AREA_INDEX": 0, "NEXT_FIX_INTEREST_AREA_INDEX": 1},
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "CURRENT_FIX_INTEREST_AREA_INDEX": 1, "NEXT_FIX_INTEREST_AREA_INDEX": 2},
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "CURRENT_FIX_INTEREST_AREA_INDEX": 2, "NEXT_FIX_INTEREST_AREA_INDEX": 1},
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "CURRENT_FIX_INTEREST_AREA_INDEX": 1, "NEXT_FIX_INTEREST_AREA_INDEX": 2},
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_1",
             "CURRENT_FIX_INTEREST_AREA_INDEX": 0, "NEXT_FIX_INTEREST_AREA_INDEX": 0},
            {Fields.SUBJECT_ID: "p2", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "CURRENT_FIX_INTEREST_AREA_INDEX": 0, "NEXT_FIX_INTEREST_AREA_INDEX": 2},
        ]
        fix_df = pd.DataFrame(rows)
        wide = compute_transitions_features(fix_df, [Fields.SUBJECT_ID])
        self.assertEqual(wide.loc["p1", "transitions_para_0_0_1"], 1)
        self.assertEqual(wide.loc["p1", "transitions_para_0_1_2"], 2)
        self.assertEqual(wide.loc["p1", "transitions_para_0_2_1"], 1)
        self.assertEqual(wide.loc["p1", "transitions_para_1_0_0"], 1)
        self.assertEqual(wide.loc["p2", "transitions_para_0_0_2"], 1)
        # Absent transitions in the wide table fill to 0
        self.assertEqual(wide.loc["p2", "transitions_para_0_0_1"], 0)
        self.assertEqual(wide.loc["p1", "transitions_para_0_0_2"], 0)

    def test_drops_invalid_indices(self):
        rows = [
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "CURRENT_FIX_INTEREST_AREA_INDEX": 0, "NEXT_FIX_INTEREST_AREA_INDEX": 1},
            # off-IA fixations should be dropped
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "CURRENT_FIX_INTEREST_AREA_INDEX": -1, "NEXT_FIX_INTEREST_AREA_INDEX": 1},
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "CURRENT_FIX_INTEREST_AREA_INDEX": 0, "NEXT_FIX_INTEREST_AREA_INDEX": np.nan},
        ]
        fix_df = pd.DataFrame(rows)
        wide = compute_transitions_features(fix_df, [Fields.SUBJECT_ID])
        # only one valid pair: (0, 1)
        self.assertEqual(wide.shape[1], 1)
        self.assertEqual(wide.loc["p1", "transitions_para_0_0_1"], 1)


class TestComputeWfcFeatures(unittest.TestCase):
    def test_columns_match_ia_values(self):
        ia_df = pd.DataFrame([
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_FIRST_RUN_DWELL_TIME": 200.0, "IA_DWELL_TIME": 350.0},
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 1, "IA_FIRST_RUN_DWELL_TIME": 100.0, "IA_DWELL_TIME": 150.0},
            {Fields.SUBJECT_ID: "p2", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_FIRST_RUN_DWELL_TIME": 250.0, "IA_DWELL_TIME": 400.0},
        ])
        wide = compute_wfc_features(ia_df, [Fields.SUBJECT_ID])
        self.assertEqual(wide.loc["p1", "wfc_para_0_0_FP"], 200.0)
        self.assertEqual(wide.loc["p1", "wfc_para_0_0_TF"], 350.0)
        self.assertEqual(wide.loc["p1", "wfc_para_0_1_FP"], 100.0)
        self.assertEqual(wide.loc["p2", "wfc_para_0_0_FP"], 250.0)
        # p2 didn't read word 1 → 0 fill
        self.assertEqual(wide.loc["p2", "wfc_para_0_1_FP"], 0)


class TestValidateIaIdAlignment(unittest.TestCase):
    def test_passes_when_aligned_meco(self):
        ia_df = pd.DataFrame([
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_LABEL": "the"},
            {Fields.SUBJECT_ID: "p2", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_LABEL": "the"},
        ])
        # Should not raise
        validate_ia_id_alignment(ia_df, metadata_df=None, dataset_name=DataSets.MECOL1)

    def test_raises_on_mismatch_meco(self):
        ia_df = pd.DataFrame([
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_LABEL": "the"},
            {Fields.SUBJECT_ID: "p2", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_LABEL": "a"},  # mismatch
        ])
        with self.assertRaises(ValueError) as ctx:
            validate_ia_id_alignment(ia_df, metadata_df=None, dataset_name=DataSets.MECOL1)
        self.assertIn("IA_ID alignment broken", str(ctx.exception))

    def test_passes_within_content_group_onestop(self):
        ia_df = pd.DataFrame([
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_LABEL": "the"},
            {Fields.SUBJECT_ID: "p2", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_LABEL": "the"},
            # different content_group, different word at same IA_ID is fine
            {Fields.SUBJECT_ID: "p3", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_LABEL": "a"},
        ])
        metadata = pd.DataFrame({
            Fields.SUBJECT_ID: ["p1", "p2", "p3"],
            Fields.CONTENT_GROUP: ["1_0", "1_0", "1_1"],
        })
        validate_ia_id_alignment(ia_df, metadata_df=metadata, dataset_name=DataSets.ONESTOPL2)

    def test_raises_within_content_group_onestop(self):
        ia_df = pd.DataFrame([
            {Fields.SUBJECT_ID: "p1", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_LABEL": "the"},
            # same content_group, different word at same IA_ID -> error
            {Fields.SUBJECT_ID: "p2", Fields.UNIQUE_PARAGRAPH_ID: "para_0",
             "IA_ID": 0, "IA_LABEL": "a"},
        ])
        metadata = pd.DataFrame({
            Fields.SUBJECT_ID: ["p1", "p2"],
            Fields.CONTENT_GROUP: ["1_0", "1_0"],
        })
        with self.assertRaises(ValueError):
            validate_ia_id_alignment(ia_df, metadata_df=metadata, dataset_name=DataSets.ONESTOPL2)


class TestValidateSeenOnlyFeatures(unittest.TestCase):
    """validate_seen_only_features(feature_cols, pool, df): OneStop allows per-text
    (transitions_ / wfc_) features in the Seen pool only; MECO, which has no
    content groups, passes in any pool."""
    def test_no_per_text_features_skips_check(self):
        df = pd.DataFrame({Fields.CONTENT_GROUP: ["1_0", "1_1"]})
        # no transitions_/wfc_ columns -> should pass regardless of pool
        validate_seen_only_features(["reading_speed", "mean_FF"], Pool.ALL, df)

    def test_meco_allows_any_pool(self):
        # MECO has no CONTENT_GROUP column -> trivially passes
        df = pd.DataFrame({Fields.SUBJECT_ID: ["p1"]})
        validate_seen_only_features(["transitions_para_0_0_1"], Pool.ALL, df)
        validate_seen_only_features(["transitions_para_0_0_1"], Pool.UNSEEN, df)

    def test_onestop_all_pool_with_transitions_raises(self):
        df = pd.DataFrame({Fields.CONTENT_GROUP: ["1_0", "1_1"]})
        with self.assertRaises(ValueError):
            validate_seen_only_features(["transitions_para_0_0_1"], Pool.ALL, df)

    def test_onestop_seen_pool_with_wfc_passes(self):
        df = pd.DataFrame({Fields.CONTENT_GROUP: ["1_0", "1_1"]})
        validate_seen_only_features(["wfc_para_0_0_FP"], Pool.SEEN, df)

    def test_onestop_unseen_pool_with_wfc_raises(self):
        df = pd.DataFrame({Fields.CONTENT_GROUP: ["1_0", "1_1"]})
        with self.assertRaises(ValueError):
            validate_seen_only_features(["wfc_para_0_0_FP"], Pool.UNSEEN, df)


class TestRealignIaCascadeOnestop(unittest.TestCase):
    @staticmethod
    def _make_paragraph(pid, words, paragraph='para_A'):
        return [
            {Fields.SUBJECT_ID: pid, Fields.UNIQUE_PARAGRAPH_ID: paragraph,
             "IA_ID": i, "IA_LABEL": w}
            for i, w in enumerate(words)
        ]

    @staticmethod
    def _make_fixations(pid, current_to_next, paragraph='para_A'):
        return [
            {Fields.SUBJECT_ID: pid, Fields.UNIQUE_PARAGRAPH_ID: paragraph,
             "CURRENT_FIX_INTEREST_AREA_INDEX": cur,
             "NEXT_FIX_INTEREST_AREA_INDEX": nxt}
            for cur, nxt in current_to_next
        ]

    def test_simple_split_join_realigns(self):
        # joined participant: "another", "credit-card", "with", "the"
        # split  participant: "another", "credit-", "card", "with", "the"
        ia = pd.DataFrame(
            self._make_paragraph('p_joined', ['another', 'credit-card', 'with', 'the'])
            + self._make_paragraph('p_split',  ['another', 'credit-', 'card', 'with', 'the'])
        )
        # Fixations: each pid does a few saccades across the boundary.
        fix = pd.DataFrame(
            self._make_fixations('p_joined', [(0, 1), (1, 2), (2, 3)])
            + self._make_fixations('p_split',  [(0, 1), (1, 2), (2, 3), (3, 4)])
        )
        md = pd.DataFrame({
            Fields.SUBJECT_ID: ['p_joined', 'p_split'],
            Fields.CONTENT_GROUP: ['1_0', '1_0'],
        })

        ia_new, fix_new = realign_ia_cascade_onestop(ia, fix, md)

        # Both participants should now share IA layout.  The hyphenated
        # word is gone, surviving IAs are contiguously renumbered.
        joined_after = ia_new[ia_new[Fields.SUBJECT_ID] == 'p_joined'].sort_values('IA_ID')
        split_after  = ia_new[ia_new[Fields.SUBJECT_ID] == 'p_split'].sort_values('IA_ID')
        self.assertEqual(joined_after['IA_ID'].tolist(), [0, 1, 2])
        self.assertEqual(joined_after['IA_LABEL'].tolist(), ['another', 'with', 'the'])
        self.assertEqual(split_after['IA_ID'].tolist(), [0, 1, 2])
        self.assertEqual(split_after['IA_LABEL'].tolist(), ['another', 'with', 'the'])

        # Validator should now pass.
        validate_ia_id_alignment(ia_new, md, DataSets.ONESTOPL1)

        # Every surviving saccade endpoint must still correspond to a real IA
        # in the post-fix IA frame for that (participant, paragraph).
        for pid in ['p_joined', 'p_split']:
            valid_ids = set(ia_new.loc[ia_new[Fields.SUBJECT_ID] == pid, 'IA_ID'])
            sub = fix_new[fix_new[Fields.SUBJECT_ID] == pid]
            self.assertTrue(set(sub['CURRENT_FIX_INTEREST_AREA_INDEX']).issubset(valid_ids))
            self.assertTrue(set(sub['NEXT_FIX_INTEREST_AREA_INDEX']).issubset(valid_ids))

    def test_clean_data_is_unchanged(self):
        ia = pd.DataFrame(
            self._make_paragraph('p1', ['a', 'b', 'c'])
            + self._make_paragraph('p2', ['a', 'b', 'c'])
        )
        md = pd.DataFrame({
            Fields.SUBJECT_ID: ['p1', 'p2'],
            Fields.CONTENT_GROUP: ['1_0', '1_0'],
        })
        ia_new, fix_new = realign_ia_cascade_onestop(ia, None, md)
        pd.testing.assert_frame_equal(
            ia_new.reset_index(drop=True), ia.reset_index(drop=True), check_like=True,
        )
        self.assertIsNone(fix_new)

    def test_iterates_for_two_split_points(self):
        # Joined: "a", "blue-jay", "sat", "near-me", "today"  (5 IAs)
        # Split:  "a", "blue-", "jay", "sat", "near-", "me", "today"  (7 IAs)
        ia = pd.DataFrame(
            self._make_paragraph('p_joined', ['a', 'blue-jay', 'sat', 'near-me', 'today'])
            + self._make_paragraph('p_split',  ['a', 'blue-', 'jay', 'sat', 'near-', 'me', 'today'])
        )
        md = pd.DataFrame({
            Fields.SUBJECT_ID: ['p_joined', 'p_split'],
            Fields.CONTENT_GROUP: ['1_0', '1_0'],
        })
        ia_new, _ = realign_ia_cascade_onestop(ia, None, md)
        joined_after = ia_new[ia_new[Fields.SUBJECT_ID] == 'p_joined'].sort_values('IA_ID')
        split_after  = ia_new[ia_new[Fields.SUBJECT_ID] == 'p_split'].sort_values('IA_ID')
        # After both splits are resolved, both should agree on every remaining IA.
        self.assertEqual(
            joined_after[['IA_ID', 'IA_LABEL']].values.tolist(),
            split_after[['IA_ID', 'IA_LABEL']].values.tolist(),
        )
        validate_ia_id_alignment(ia_new, md, DataSets.ONESTOPL1)


if __name__ == "__main__":
    unittest.main()
