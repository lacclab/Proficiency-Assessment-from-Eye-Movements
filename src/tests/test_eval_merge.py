"""Tests for save_preserving_unevaluated and the two evaluation scope builders.

Guards the rule that a filtered evaluation run must replace only what it
actually evaluated:
  • a run narrowed by model, target, or agg type keeps every row outside that
    filter (the target-filtered case is the regression — the old merge keyed on
    model_names alone, so any other filter took the wholesale-overwrite path);
  • a fully unfiltered run still overwrites, so stale rows cannot accumulate;
  • EyeScore's preview rule is dataset-dependent, so it is applied per row.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.utils import save_preserving_unevaluated
from src.evaluation.predictions.evaluation import _evaluation_scope as pred_scope
from src.evaluation.EyeScore.evaluation import _evaluation_scope as eye_scope


def _prediction_rows() -> pd.DataFrame:
    """Stand-in for all_evaluation_results.csv: 3 models x 2 targets x 2 sets."""
    rows = [
        {"dataset": "OneStop", "preview": "Gathering",
         "version_path": "fully_agg/ordinary/all", "target_col": target,
         "feature_set": fs, "model_name": model, "pearson_r": 0.5}
        for model in ("Ridge_Classifier", "LightGBM", "XGBoost")
        for target in ("lextale_score", "michtest_score")
        for fs in ("READING_SPEED", "WFC")
    ]
    return pd.DataFrame(rows)


def _eyescore_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"dataset": "OneStop", "preview": "Gathering",
         "version_path": "fully_agg/ordinary/all", "target_col": "lextale_score",
         "feature_set": "WFC", "pearson_r": 0.5},
        {"dataset": "OneStop", "preview": "Hunting",
         "version_path": "fully_agg/ordinary/all", "target_col": "lextale_score",
         "feature_set": "WFC", "pearson_r": 0.5},
        {"dataset": "Meco", "preview": "All",
         "version_path": "fully_agg/all/all", "target_col": "lextale_score",
         "feature_set": "WFC", "pearson_r": 0.5},
    ])


class _TmpCsv(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "all_evaluation_results.csv"

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, df):
        df.to_csv(self.path, index=False)


def _prediction_rows_two_datasets() -> pd.DataFrame:
    """Same cell (target/model) present under two datasets."""
    return pd.DataFrame([
        {"dataset": dataset, "preview": "Gathering" if dataset == "OneStop" else "All",
         "version_path": "fully_agg/ordinary/all", "target_col": "lextale_score",
         "feature_set": "WFC", "model_name": model, "pearson_r": 0.5}
        for dataset in ("OneStop", "Meco")
        for model in ("Ridge_Classifier", "LightGBM")
    ])


class TestPredictionScope(_TmpCsv):
    def test_dataset_scope_protects_rows_from_a_dataset_not_on_disk(self):
        """Regression: a rerun whose results_dir only ever held Meco (e.g. a
        Meco-only rerun) still matches OneStop's rows on target_col/model_name.
        Without a dataset key in scope those OneStop rows looked "covered" and
        were silently deleted even though this run never touched OneStop."""
        self.write(_prediction_rows_two_datasets())
        new = _prediction_rows_two_datasets().query("dataset == 'Meco'").assign(pearson_r=0.9)
        out = save_preserving_unevaluated(
            new, self.path,
            pred_scope(target_cols=["lextale_score"],
                      model_names=["Ridge_Classifier", "LightGBM"],
                      datasets=["Meco"]))
        self.assertEqual(len(out), 4)
        self.assertEqual(sorted(out.dataset.unique()), ["Meco", "OneStop"])
        self.assertEqual(sorted(out.query("dataset == 'OneStop'").pearson_r.unique()), [0.5])
        self.assertEqual(sorted(out.query("dataset == 'Meco'").pearson_r.unique()), [0.9])

    def test_model_filter_keeps_other_models(self):
        self.write(_prediction_rows())
        new = _prediction_rows().query("model_name == 'Ridge_Classifier'").assign(pearson_r=0.9)
        out = save_preserving_unevaluated(
            new, self.path, pred_scope(model_names=["Ridge_Classifier"]))
        self.assertEqual(len(out), 12)
        self.assertEqual(sorted(out.model_name.unique()),
                         ["LightGBM", "Ridge_Classifier", "XGBoost"])
        # The evaluated model's rows are the fresh ones, not the stale copies.
        self.assertEqual(sorted(out.query("model_name=='Ridge_Classifier'").pearson_r.unique()),
                         [0.9])

    def test_target_filter_keeps_other_targets_and_models(self):
        """The regression: no model filter, so the old code overwrote wholesale."""
        self.write(_prediction_rows())
        new = _prediction_rows().query("target_col == 'lextale_score'").assign(pearson_r=0.9)
        out = save_preserving_unevaluated(
            new, self.path, pred_scope(target_cols=["lextale_score"]))
        self.assertEqual(len(out), 12)
        self.assertEqual(sorted(out.target_col.unique()),
                         ["lextale_score", "michtest_score"])
        self.assertEqual(sorted(out.model_name.unique()),
                         ["LightGBM", "Ridge_Classifier", "XGBoost"])

    def test_agg_type_filter_matches_version_path_prefix(self):
        df = _prediction_rows()
        df.loc[df.index[:4], "version_path"] = "first_p_agg/ordinary/all"
        self.write(df)
        new = df[df.version_path.str.startswith("fully_agg")].assign(pearson_r=0.9)
        out = save_preserving_unevaluated(new, self.path, pred_scope(agg_type="fully_agg"))
        self.assertEqual(len(out), 12)
        self.assertEqual(int((out.version_path == "first_p_agg/ordinary/all").sum()), 4)

    def test_unfiltered_run_overwrites_wholesale(self):
        """No filters means the run regenerates everything; nothing may linger."""
        self.write(_prediction_rows())
        new = _prediction_rows().query("model_name == 'Ridge_Classifier'")
        out = save_preserving_unevaluated(new, self.path, pred_scope())
        self.assertEqual(len(out), 4)


class TestEyeScoreScope(_TmpCsv):
    def test_single_preview_run_leaves_other_previews_alone(self):
        self.write(_eyescore_rows())
        scope, preview_covered = eye_scope(previews=["Hunting"])
        new = _eyescore_rows().query("preview == 'Hunting'").assign(pearson_r=0.9)
        out = save_preserving_unevaluated(new, self.path, scope,
                                          row_predicate=preview_covered)
        self.assertEqual(len(out), 3)
        self.assertEqual(sorted(out.query("pearson_r == 0.9").preview.unique()), ["Hunting"])

    def test_default_previews_cover_every_allowed_row(self):
        self.write(_eyescore_rows())
        scope, preview_covered = eye_scope()
        out = save_preserving_unevaluated(_eyescore_rows().assign(pearson_r=0.9),
                                          self.path, scope, row_predicate=preview_covered)
        self.assertEqual(len(out), 3)

    def test_preview_outside_the_allow_list_survives(self):
        """A default run never evaluates it, so it must not be deleted either."""
        stale = pd.concat([_eyescore_rows(), pd.DataFrame([{
            "dataset": "OneStop", "preview": "Retired",
            "version_path": "fully_agg/ordinary/all", "target_col": "lextale_score",
            "feature_set": "WFC", "pearson_r": 0.5}])], ignore_index=True)
        self.write(stale)
        scope, preview_covered = eye_scope()
        out = save_preserving_unevaluated(_eyescore_rows().assign(pearson_r=0.9),
                                          self.path, scope, row_predicate=preview_covered)
        self.assertEqual(len(out), 4)
        self.assertEqual(int((out.preview == "Retired").sum()), 1)


if __name__ == "__main__":
    unittest.main()
