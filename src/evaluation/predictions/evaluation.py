"""Prediction evaluation: score every prediction CSV and test models against
each other.

evaluate_all_results writes all_evaluation_results.csv (Pearson/Spearman r,
MAE, RMSE, R² per cell, with bootstrap CIs); compute_pairwise_significance
writes the paired-bootstrap comparisons the table stars come from. Run by
`python -m src.run.predictions --stage evaluation` (run_evaluation_pipeline).
"""
import os
import pandas as pd
import numpy as np
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.configs import MODELS, ModelNames
from src.constants import ALL_FEATURE_SETS_DICT, ALL_TESTS, TestCols
from src.evaluation.stats_utils import (
    N_BOOTSTRAP,
    bootstrap_correlation_ci,
    pairwise_dependent_tests,
)
from src.evaluation.utils import save_preserving_unevaluated


logger = logging.getLogger(__name__)

# Markers of superseded result trees kept beside the live ones for comparison.
# Always skipped by collect_result_csvs: they are duplicates of the live tree at
# an older code revision.
_STALE_TREE_MARKERS = ("_prerun_", "backup", "_stale", "_rrfix_",
                       "_unseenfix_", "_prefix_")


def _evaluation_scope(target_cols=None, feature_sets=None, model_names=None,
                      agg_type=None, datasets=None, previews=None) -> dict:
    """Map this run's filters onto the columns of the saved evaluation CSVs.

    Mirrors collect_result_csvs: membership tests for the list filters, and a
    prefix test for agg_type, which is the leading segment of version_path
    rather than a column of its own. Both the aggregate and per-language CSVs
    carry all four columns, so one scope serves both.

    `datasets`/`previews` are not user-facing filters (evaluate_all_results
    has no --dataset/--preview flag) -- callers pass the set actually found in
    `entries`, i.e. what results_dir had on disk for this run. Without them, a
    run whose results_dir happens to hold only one dataset (e.g. a MECO-only
    rerun) has no column protecting the other dataset's rows, and a
    target_col/feature_set/model_name match that spans both datasets silently
    replaces them with nothing. Always pass the collected set here (never
    None) so it stays active even on an otherwise-unfiltered run.
    """
    return {
        "target_col": target_cols,
        "feature_set": feature_sets,
        "model_name": model_names,
        "dataset": datasets,
        "preview": previews,
        "version_path": (lambda vp: vp.startswith(agg_type)) if agg_type else None,
    }


def _eval_max_workers() -> int:
    """Worker count for the bootstrap/pairwise process pools.

    Capped at 8 by default — the box is shared with the long-running
    prediction pipelines, and uncapped (cores-2) pools were starving them.
    Set PRED_EVAL_N_JOBS to override in either direction.
    """
    env = int(os.environ.get("PRED_EVAL_N_JOBS", 0))
    if env > 0:
        return env
    return max(1, min(8, (os.cpu_count() or 4) - 2))


# --require-l1-lextale variant. When REQUIRE_L1_LEXTALE=1 is set in the
# environment BEFORE this module is imported (src.run.predictions
# --require-l1-lextale does this), every root shifts to the require_l1_lextale/
# subtree: prediction CSVs are read from results/require_l1_lextale/, and the
# evaluation CSVs / tables are written to matching require_l1_lextale/ subtrees
# so the LexTALE-only-L1 runs never mix with the default runs.
REQUIRE_L1_LEXTALE_SUBTREE = "require_l1_lextale"
REQUIRE_L1_LEXTALE = os.environ.get("REQUIRE_L1_LEXTALE", "") == "1"

RESULTS_DIR = Path("src/methods/predictions/results")
EVAL_SAVE_DIR = Path("src/evaluation/predictions/results")
if REQUIRE_L1_LEXTALE:
    RESULTS_DIR = RESULTS_DIR / REQUIRE_L1_LEXTALE_SUBTREE
    EVAL_SAVE_DIR = EVAL_SAVE_DIR / REQUIRE_L1_LEXTALE_SUBTREE

# Full all-pairs significance CSV (written only by unrestricted runs).
PAIRWISE_CSV = "pairwise_significance.csv"
# WPM-baseline subset — what the bootstrap combined tables read for the stars
# against the WPM row. Refreshed by every run that computes baseline pairs.
PAIRWISE_WPM_CSV = "pairwise_significance_wpm.csv"
# Model-vs-Ridge subset: same feature set / target / method, one model against
# the Ridge baseline. One row per cell of a non-Ridge model's results table,
# giving that cell's difference from Ridge and the bootstrap significance of
# the difference. Refreshed by every run that computes ridge pairs.
PAIRWISE_RIDGE_CSV = "pairwise_significance_ridge.csv"

# Bootstrap columns appended to every evaluation row: the reported correlation
# is the bootstrap mean (<m>_r_boot), with the resample std / normal-approx 95%
# CI as the error bar. The exact point estimates (pearson_r / spearman_r) are
# kept alongside so existing consumers keep working.
BOOTSTRAP_METRICS = [
    "pearson_r_boot", "pearson_r_boot_std", "pearson_ci_low_boot", "pearson_ci_high_boot",
    "spearman_r_boot", "spearman_r_boot_std", "spearman_ci_low_boot", "spearman_ci_high_boot",
    "mae_boot", "mae_boot_std", "mae_ci_low_boot", "mae_ci_high_boot",
    "n_boot",
]

EVALUATION_METRICS = ["r2", "adjusted_r2", "rmse", "mae", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n"] + BOOTSTRAP_METRICS


# Older prediction filenames → the canonical {pool}__{fold_method} name, so
# result trees written under the earlier naming (the per-batch OneStop
# batch_cross_validation_* and the MECO per-half leave_one_participant_out_*
# files) still evaluate. Size variants (SEEN_LARGE, UNSEEN_LARGE, UNSEEN_SMALL)
# all collapse to the same cell; when several exist on disk, the dedup pass
# below prefers the canonical filename, then the older default.
LEGACY_METHOD_NORMALIZATION = {
    # OneStop legacy
    "leave_one_participant_out": "all__pool_all",
    "leave_one_language_out": "all__lolo",
    "random": "all__l1_strat_kfold",
    "batch_cross_validation_seen_small": "seen__pool_all",
    "batch_cross_validation_seen_large": "seen__pool_all",
    "batch_cross_validation_unseen_large": "unseen__pool_all",
    "batch_cross_validation_unseen_small": "unseen__pool_all",
    "batch_cross_validation_unseen_small_by_batch": "unseen__pool_all",
    # MECO legacy
    "leave_one_participant_out_seen": "seen__pool_all",
    "leave_one_participant_out_unseen": "unseen__pool_all",
}


def normalize_method(raw_method: str) -> str:
    """Map a CSV filename stem to its canonical Pool × FoldMethod name.
    New-convention stems pass through unchanged."""
    return LEGACY_METHOD_NORMALIZATION.get(raw_method, raw_method)


def compute_regression_metrics(true: np.ndarray, pred: np.ndarray, num_features: int = 1,
                               include_bootstrap: bool = True,
                               n_bootstrap: int = N_BOOTSTRAP) -> Dict[str, float]:
    """Compute regression evaluation metrics between true and predicted values.

    When `include_bootstrap` is set, also bootstrap the correlation between
    `true` and `pred`: the reported <m>_r_boot is the mean over resamples and
    <m>_r_boot_std / <m>_ci_*_boot give the error bar (see stats_utils).
    """
    mask = ~(np.isnan(true) | np.isnan(pred))
    true, pred = true[mask], pred[mask]
    n = len(true)
    if n < 3:
        logger.warning("Not enough valid data points to compute metrics (n=%d). Returning NaN for all metrics.", n)
        return {m: np.nan for m in EVALUATION_METRICS}

    r2 = r2_score(true, pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - num_features - 1) if n > num_features + 1 else np.nan
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    try:
        pearson_r, pearson_p = stats.pearsonr(true, pred)
        spearman_r, spearman_p = stats.spearmanr(true, pred)
    except Exception as e:
        logger.warning("Failed to compute Pearson/Spearman correlation: %s. Setting to NaN.", e)
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan

    out = {
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "n": n,
    }
    if include_bootstrap:
        boot = bootstrap_correlation_ci(true, pred, n_bootstrap=n_bootstrap, include_mae=True)
        out.update({m: boot[m] for m in BOOTSTRAP_METRICS})
    else:
        out.update({m: np.nan for m in BOOTSTRAP_METRICS})
    return out


def evaluate_one_result(pred_df: pd.DataFrame, num_features: int = 1,
                        include_bootstrap: bool = True) -> Dict[str, float]:
    """Evaluate a single result DataFrame with columns: participant_id, L1, true, pred, fold."""
    if pred_df["true"].unique().size == 1:
        logger.warning("Only one unique true value in the data. Metrics may be unreliable.")
    return compute_regression_metrics(pred_df["true"].values, pred_df["pred"].values,
                                      num_features, include_bootstrap=include_bootstrap)


def evaluate_one_result_per_lang(pred_df: pd.DataFrame, num_features: int = 1,
                                 include_bootstrap: bool = True) -> Dict[str, Dict[str, float]]:
    """Evaluate a single result DataFrame per language."""
    results = {}
    for lang, group in pred_df.groupby("L1"):
        results[lang] = compute_regression_metrics(
            group["true"].values, group["pred"].values, num_features,
            include_bootstrap=include_bootstrap)
    return results


# Path segment written by tuned tree runs (see src.run.predictions:
# `inner_{strategy}` appended to version_suffix). Kept as data rather than a
# hardcoded string so adding an inner-validation strategy needs no edit here.
INNER_VALIDATION_PREFIX = "inner_"


def _split_inner_validation(version_path: str) -> tuple:
    """Split a trailing `inner_<scheme>` segment off a version_path.

    "fully_agg/inner_kfold" -> ("fully_agg", "kfold")
    "fully_agg"             -> ("fully_agg", "")
    """
    parts = [p for p in version_path.split("/") if p]
    if parts and parts[-1].startswith(INNER_VALIDATION_PREFIX):
        return "/".join(parts[:-1]), parts[-1][len(INNER_VALIDATION_PREFIX):]
    return version_path, ""


def collect_result_csvs(results_dir: Path, target_cols: List[str] = None,
                        feature_sets: List[str] = None, model_names: List[str] = None,
                        datasets: List[str] = None, agg_type: str = None) -> List[dict]:
    """Walk the results directory and collect all result CSV paths with their metadata.

    Handles both structures:
    1. Old: {dataset}/{aug_p}/{preview}/{agg_type}/{p_level_p}/{target_col}/{feature_set}/{model_name}/{method}.csv
    2. New: {dataset}/{aug_p}/{preview}/{agg_type}/{p_level_p}/{target_col}/{feature_set}/{model_name}/{train_type}/{method}.csv

    Args:
        agg_type: Filter by aggregation type (e.g., "fully_agg", "first_p_agg", "moving_p_agg")
    """
    entries = []
    for csv_path in results_dir.rglob("*.csv"):
        rel = csv_path.relative_to(results_dir)
        parts = rel.parts
        # Skip superseded trees kept beside the live ones for comparison. Only
        # "backup" was matched before, which left ten other dated trees on disk
        # (…_prerun_<stamp>, …_rrfix_<stamp>, …_unseenfix_<stamp>,
        # …_prefix_<stamp>, _combined_set_stale_<stamp>) being evaluated: 9,176
        # CSVs, ~17% of an unfiltered Ridge sweep, and their rows then collide
        # with the live ones in tables.py, whose per-item filter is a
        # startswith("per_item_agg") that matches per_item_agg_prerun_* too.
        # Mirrors _STALE_TREE_MARKERS in src/evaluation/EyeScore/evaluation.py.
        if any(m in part for part in parts for m in _STALE_TREE_MARKERS):
            continue
        # The require_l1_lextale/ variant subtree nests inside the default
        # results root; when walking the default root it must not be picked up
        # (parts[0] would be parsed as a dataset). When results_dir IS the
        # subtree, rel paths never start with it, so this never misfires.
        if parts[0] == REQUIRE_L1_LEXTALE_SUBTREE:
            continue
        # minimum: dataset/aug_p/preview/.../target_col/feature_set/model_name/method.csv (length >= 7)
        if len(parts) < 7:
            continue

        raw_method = csv_path.stem
        # Skip per-item intermediate files (only evaluate aggregated)
        if raw_method.endswith("_items"):
            continue
        method = normalize_method(raw_method)
        # `__` is the Pool × FoldMethod separator; new-convention stems contain it.
        is_legacy_name = raw_method != method
        dataset = parts[0]
        aug_p = parts[1]
        preview = parts[2]

        # Work backwards from the CSV to extract metadata
        # The structure is: .../{target_col}/{feature_set}/{model_name}/[train_type]/{method}.csv
        # Check if there's a train_type folder between model_name and method
        train_type_folder = None
        if len(parts) >= 8:  # Could have train_type folder
            potential_train_type = parts[-2]
            if potential_train_type in ["train_on_full", "partial_train"]:
                train_type_folder = potential_train_type
                model_name = parts[-3]
                feature_set = parts[-4]
                target_col = parts[-5]
                version_path = "/".join(parts[3:-5])
            else:
                # No train_type folder
                model_name = parts[-2]
                feature_set = parts[-3]
                target_col = parts[-4]
                version_path = "/".join(parts[3:-4])
        else:
            # Old structure
            model_name = parts[-2]
            feature_set = parts[-3]
            target_col = parts[-4]
            version_path = "/".join(parts[3:-4])

        if datasets is not None and dataset not in datasets:
            continue
        if target_cols is not None and target_col not in target_cols:
            continue
        if feature_sets is not None and feature_set not in feature_sets:
            continue
        if model_names is not None and model_name not in model_names:
            continue
        if agg_type is not None and not version_path.startswith(agg_type):
            continue
        # Skip per-text features (TRANSITIONS, WFC) with train_on_full: they were skipped during prediction
        if train_type_folder == "train_on_full" and feature_set in ("TRANSITIONS", "WFC"):
            continue

        # Tuned tree runs write an extra `inner_<scheme>` path segment so a
        # tuned cell cannot overwrite an untuned one on disk. The table layer
        # tests `version_path == "fully_agg"` in ~18 places and groups pairwise
        # comparisons by version_path, so leaving the segment in place would
        # (a) silently drop every tuned result from every paper table and
        # (b) put the trees in a different comparison group from Ridge, which
        # writes plain `fully_agg` -- killing the vs-Ridge subscripts.
        #
        # So the segment is stripped here and folded into the model name
        # instead: version_path stays `fully_agg`, and the models stay
        # distinguishable as Decision_Tree_holdout / Decision_Tree_kfold, with
        # Ridge_Classifier unchanged so it is still PAIRWISE_BASELINE_MODEL.
        version_path, inner_scheme = _split_inner_validation(version_path)
        if inner_scheme:
            model_name = f"{model_name}_{inner_scheme}"

        entry = {
            "csv_path": csv_path,
            "dataset": dataset,
            "aug_p": aug_p,
            "preview": preview,
            "version_path": version_path,
            "inner_validation": inner_scheme or "none",
            "target_col": target_col,
            "feature_set": feature_set,
            "model_name": model_name,
            "method": method,
            "_raw_method": raw_method,
            "_is_legacy_name": is_legacy_name,
        }
        if train_type_folder:
            entry["train_type"] = train_type_folder

        entries.append(entry)

    # When an older and a canonical filename map to the same method for the
    # same (dataset, preview, version_path, target, feature, model, train_type)
    # combo, prefer the canonical one, so a cell is never evaluated twice.
    by_key: dict[tuple, dict] = {}
    for e in entries:
        key = (e["dataset"], e["aug_p"], e["preview"], e["version_path"],
               e["target_col"], e["feature_set"], e["model_name"],
               e["method"], e.get("train_type"))
        existing = by_key.get(key)
        if existing is None or (existing["_is_legacy_name"] and not e["_is_legacy_name"]):
            by_key[key] = e
    return [
        {k: v for k, v in e.items() if not k.startswith("_")}
        for e in by_key.values()
    ]


def _evaluate_entry(entry: dict, include_bootstrap: bool = True) -> tuple:
    """Evaluate one collected result CSV. Returns (rows, lang_rows).

    Module-level so ProcessPoolExecutor workers can pickle and run it.
    With `include_bootstrap=False` the bootstrap columns are left NaN; that
    resample loop is essentially the whole cost of this function.
    """
    rows = []
    lang_rows = []

    try:
        pred_df = pd.read_csv(entry["csv_path"])
    except Exception as e:
        logger.warning("Failed to read %s: %s", entry["csv_path"], e)
        return rows, lang_rows

    if pred_df.empty or "true" not in pred_df.columns or "pred" not in pred_df.columns:
        logger.warning("Skipping %s: missing required columns", entry["csv_path"])
        return rows, lang_rows

    # Get number of features for adjusted R²
    num_features = len(ALL_FEATURE_SETS_DICT.get(entry["feature_set"], [None]))

    # Check if this is moving_p_agg
    agg_type = entry["version_path"].split('/')[0] if entry["version_path"] else ""
    is_moving_p_agg = agg_type == "moving_p_agg"

    if is_moving_p_agg and "window_index" in pred_df.columns:
        # Evaluate per window_index for moving_p_agg
        for window_idx, window_df in pred_df.groupby("window_index"):
            metrics = evaluate_one_result(window_df, num_features=num_features,
                                          include_bootstrap=include_bootstrap)
            row = {
                "dataset": entry["dataset"],
                "aug_p": entry["aug_p"],
                "preview": entry["preview"],
                "version_path": entry["version_path"],
                "target_col": entry["target_col"],
                "feature_set": entry["feature_set"],
                "model_name": entry["model_name"],
                "method": entry["method"],
                "window_index": window_idx,
                **metrics,
            }
            if "train_type" in entry:
                row["train_type"] = entry["train_type"]
            rows.append(row)
    else:
        # Overall evaluation for non-moving_p_agg
        metrics = evaluate_one_result(pred_df, num_features=num_features,
                                      include_bootstrap=include_bootstrap)
        row = {
            "dataset": entry["dataset"],
            "aug_p": entry["aug_p"],
            "preview": entry["preview"],
            "version_path": entry["version_path"],
            "target_col": entry["target_col"],
            "feature_set": entry["feature_set"],
            "model_name": entry["model_name"],
            "method": entry["method"],
            **metrics,
        }
        if "train_type" in entry:
            row["train_type"] = entry["train_type"]
        rows.append(row)

    # Per-language evaluation - only for fully_agg and first_p_agg with the
    # cross-validation methods that produce per-language signal. Method
    # names here are the canonical Pool × FoldMethod form after
    # normalization in collect_result_csvs.
    allowed_agg_types = ["fully_agg", "first_p_agg"]
    allowed_methods = {"all__lolo", "all__l1_strat_kfold", "all__pool_all"}

    if agg_type in allowed_agg_types and entry["method"] in allowed_methods:
        lang_metrics = evaluate_one_result_per_lang(
            pred_df, num_features=num_features, include_bootstrap=include_bootstrap)
        for lang, lang_met in lang_metrics.items():
            lang_row = {
                "dataset": entry["dataset"],
                "aug_p": entry["aug_p"],
                "preview": entry["preview"],
                "version_path": entry["version_path"],
                "target_col": entry["target_col"],
                "feature_set": entry["feature_set"],
                "model_name": entry["model_name"],
                "method": entry["method"],
                "language": lang,
                **lang_met,
            }
            if "train_type" in entry:
                lang_row["train_type"] = entry["train_type"]
            lang_rows.append(lang_row)

    return rows, lang_rows


def evaluate_all_results(results_dir: Path = None, save_dir: Path = None,
                         target_cols: List[str] = None, feature_sets: List[str] = None,
                         model_names: List[str] = None, agg_type: str = None,
                         include_bootstrap: bool = True) -> pd.DataFrame:
    """Evaluate all prediction results and save evaluation CSVs.

    Saves:
        - One aggregated CSV per (dataset, preview, version_path, target_col) with all feature sets as rows
        - One per-language CSV alongside each aggregated CSV
        - One combined CSV with all results

    Args:
        agg_type: Filter by aggregation type (e.g., "fully_agg", "first_p_agg", "moving_p_agg")
        include_bootstrap: when False, skip the per-row bootstrap and leave the
            BOOTSTRAP_METRICS columns NaN. The resample loop is ~99.9% of this
            stage's cost, so this turns ~20 minutes into seconds — but the
            written rows then carry no <m>_r_boot / CI values, which is what
            the bootstrap tables print. Point-estimate refreshes only.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    if save_dir is None:
        save_dir = EVAL_SAVE_DIR

    combined_path = save_dir / "all_evaluation_results.csv"

    entries = collect_result_csvs(results_dir, target_cols=target_cols,
                                  feature_sets=feature_sets, model_names=model_names,
                                  agg_type=agg_type)
    if not entries:
        logger.warning("No result CSVs found in %s", results_dir)
        return pd.DataFrame()

    # When specific models were requested, report which of them actually have
    # result CSVs on disk: models still running are simply absent from this
    # round of evaluation and can be re-run later with --models.
    if model_names is not None:
        found_models = {e["model_name"] for e in entries}
        for m in model_names:
            n_files = sum(1 for e in entries if e["model_name"] == m)
            logger.info("Model %s: %d result files found", m, n_files)
        missing = [m for m in model_names if m not in found_models]
        if missing:
            logger.warning("No result CSVs on disk yet for model(s): %s — "
                           "they will not appear in the merged evaluation.", missing)

    logger.info("Found %d result files to evaluate", len(entries))

    all_rows = []
    all_lang_rows = []

    # Evaluate files in parallel: each entry is independent, and the bootstrap
    # makes single files expensive enough that process workers pay off.
    # PRED_EVAL_N_JOBS caps the worker count (default: min(8, cores-2) — the
    # machine is shared with the prediction runs, so don't flood it).
    max_workers = _eval_max_workers()
    if not include_bootstrap:
        logger.warning(
            "Bootstrap disabled: %s will be written as NaN for the %d rows "
            "this run replaces, so the bootstrap tables lose their values for "
            "them until a full run restores the columns.",
            ", ".join(BOOTSTRAP_METRICS[:4]) + ", ...", len(entries))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(partial(_evaluate_entry,
                                            include_bootstrap=include_bootstrap),
                                    entries, chunksize=8)
        for rows, lang_rows in tqdm(results_iter, total=len(entries),
                                    desc="Evaluating predictions"):
            all_rows.extend(rows)
            all_lang_rows.extend(lang_rows)

    # Save combined results
    all_results_df = pd.DataFrame(all_rows)
    all_lang_results_df = pd.DataFrame(all_lang_rows)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Merge into the existing combined CSVs instead of overwriting them:
    # replace only what this run actually evaluated and keep every other row
    # intact. A fully unfiltered run still overwrites wholesale.
    #
    # Every filter this function accepts has to appear here: a filter left out
    # would send a narrowed run down the overwrite path and delete every row it
    # had not evaluated.
    # `model_names` is matched by collect_result_csvs against the model
    # DIRECTORY name ("LightGBM"), but the saved rows carry the inner-validation
    # suffix ("LightGBM_kfold") -- the two are not the same string, and the
    # scope is compared against the saved column. Passing the raw filter here
    # deletes rows (a name that matches no directory collects nothing yet marks
    # every existing row covered) or duplicates them (a bare name collects rows
    # but marks none covered). Derive the scope from what was actually
    # collected so both halves always agree.
    scope_models = (sorted({e["model_name"] for e in entries})
                    if model_names is not None else None)
    if model_names is not None:
        # A name that matched nothing is almost always the directory-vs-label
        # confusion above, and is otherwise silent: the run "succeeds" having
        # evaluated nothing for that model.
        unmatched = [n for n in model_names
                     if not any(m == n or m.startswith(n + "_") for m in scope_models)]
        if unmatched:
            logger.warning(
                "--models %s matched no result directories and were skipped. "
                "Model filters name the directory (e.g. 'LightGBM'), not the "
                "suffixed row label ('LightGBM_kfold'). Their existing rows are "
                "left untouched.", ", ".join(unmatched))
    if model_names is not None and not scope_models:
        raise ValueError(
            f"--models {model_names} matched no result directories. Model "
            "filters name the directory (e.g. 'LightGBM'), not the suffixed "
            "row label ('LightGBM_kfold'). Refusing to save: an empty match "
            "would delete every row the scope covers.")
    # Always derived from what results_dir actually held for this run (see
    # _evaluation_scope docstring) -- unlike scope_models this is never None.
    scope_datasets = sorted({e["dataset"] for e in entries})
    scope_previews = sorted({e["preview"] for e in entries})
    scope = _evaluation_scope(target_cols, feature_sets, scope_models, agg_type,
                              datasets=scope_datasets, previews=scope_previews)

    def _save(df: pd.DataFrame, path: Path) -> None:
        save_preserving_unevaluated(df, path, scope)

    _save(all_results_df, combined_path)
    logger.info("Saved combined evaluation results to %s", combined_path)

    combined_lang_path = save_dir / "all_evaluation_results_per_language.csv"
    _save(all_lang_results_df, combined_lang_path)
    logger.info("Saved combined per-language evaluation results to %s", combined_lang_path)
    return all_results_df


def _read_pred_series(csv_path: Path) -> Optional[pd.DataFrame]:
    """Read a prediction CSV down to one (true, pred) row per participant.

    Returns None if the file lacks the required columns or has multiple rows
    per participant (e.g. moving_p_agg window rows), which can't be aligned
    across models for a pairwise dependent-correlation test.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("Failed to read %s: %s", csv_path, e)
        return None
    if df.empty or not {"participant_id", "true", "pred"}.issubset(df.columns):
        return None
    if "window_index" in df.columns or df["participant_id"].duplicated().any():
        return None
    return df[["participant_id", "true", "pred"]].dropna()


# Baseline feature set for --pairwise-baseline-only runs (WPM; same value as
# tables.BOOTSTRAP_BASELINE_FS, which is what the bootstrap tables star against).
PAIRWISE_BASELINE_FS = "READING_SPEED"
# Baseline *model* for the vs-Ridge comparison: every other model is compared
# against Ridge cell by cell — same feature set, target, method and split, only
# the model differs (same value as tables.BOOTSTRAP_BASELINE_MODEL).
PAIRWISE_BASELINE_MODEL = str(ModelNames.RIDGE_CLASSIFIER)


def _pair_kinds(ea: dict, eb: dict) -> set:
    """Which restricted comparison sets this pair of entries belongs to.

    "baseline" — one side is the WPM baseline feature set: the stars the
    bootstrap combined tables put on every non-WPM row.
    "ridge" — same feature set, exactly one side is the Ridge baseline model:
    one table cell against the same cell of the Ridge table.
    """
    kinds = set()
    if PAIRWISE_BASELINE_FS in (ea["feature_set"], eb["feature_set"]):
        kinds.add("baseline")
    if ea["feature_set"] == eb["feature_set"] and (
            (ea["model_name"] == PAIRWISE_BASELINE_MODEL)
            != (eb["model_name"] == PAIRWISE_BASELINE_MODEL)):
        kinds.add("ridge")
    return kinds


def _pairwise_group(item: tuple, pair_kinds: Optional[set] = None) -> List[dict]:
    """Run all pairwise dependent-correlation tests for one group.

    Module-level so ProcessPoolExecutor can pickle it. `item` is one
    (key, group) pair from the grouped dict in compute_pairwise_significance.
    `pair_kinds` (see `_pair_kinds`) restricts which pairs are tested; None
    tests every pair in the group.
    """
    key, group = item
    dataset, aug_p, preview, version_path, target_col, method, train_type = key
    # executor.map yields in submission order, so the tqdm bar stalls on a slow
    # first group even when later ones have finished. These two lines report
    # what each worker actually starts and finishes, with its own timing.
    _t0 = time.time()
    logger.info("pairwise START  %s/%s/%s/%s/%s (%d entries)",
                dataset, preview, target_col, method, version_path, len(group))

    # Select the pairs first, so a restricted run only reads the CSVs of the
    # entries that actually take part.
    selected = []
    for i, j in itertools.combinations(range(len(group)), 2):
        kinds = _pair_kinds(group[i], group[j])
        if pair_kinds is not None and not (kinds & pair_kinds):
            continue
        # Orient model-vs-Ridge pairs so Ridge is always side b: every
        # difference in the ridge CSV then reads as "this model minus Ridge",
        # positive meaning the model beats Ridge (for MAE, negative does).
        if "ridge" in kinds and group[i]["model_name"] == PAIRWISE_BASELINE_MODEL:
            i, j = j, i
        selected.append((i, j))
    if not selected:
        logger.info("pairwise SKIP   %s/%s/%s/%s (no pairs selected)",
                    dataset, preview, target_col, method)
        return []

    # Load each model's per-participant (true, pred), keeping only alignable ones.
    loaded = {}
    for i in {i for pair in selected for i in pair}:
        df = _read_pred_series(group[i]["csv_path"])
        if df is not None and len(df) >= 4:
            loaded[i] = df

    rows = []
    for i, j in selected:
        if i not in loaded or j not in loaded:
            continue
        (ea, da), (eb, db) = (group[i], loaded[i]), (group[j], loaded[j])
        merged = da.merge(db, on="participant_id", suffixes=("_a", "_b"))
        merged = merged.dropna(subset=["true_a", "pred_a", "pred_b"])
        if len(merged) < 4:
            continue
        stats_row = pairwise_dependent_tests(
            merged["true_a"].values, merged["pred_a"].values, merged["pred_b"].values,
            include_mae=True,
        )
        rows.append({
            "dataset": dataset, "aug_p": aug_p, "preview": preview,
            "version_path": version_path, "target_col": target_col,
            "method": method, "train_type": train_type,
            "feature_set_a": ea["feature_set"], "model_name_a": ea["model_name"],
            "feature_set_b": eb["feature_set"], "model_name_b": eb["model_name"],
            "model_a": f"{ea['feature_set']}|{ea['model_name']}",
            "model_b": f"{eb['feature_set']}|{eb['model_name']}",
            **stats_row,
        })
    logger.info("pairwise DONE   %s/%s/%s/%s (%.1f min)",
                dataset, preview, target_col, method, (time.time()-_t0)/60)
    return rows


def _subset_by_kind(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """The `kind` ("baseline" / "ridge") slice of a pairwise result frame."""
    if df.empty:
        return df
    if kind == "baseline":
        return df[(df["feature_set_a"] == PAIRWISE_BASELINE_FS)
                  | (df["feature_set_b"] == PAIRWISE_BASELINE_FS)]
    return df[(df["feature_set_a"] == df["feature_set_b"])
              & ((df["model_name_a"] == PAIRWISE_BASELINE_MODEL)
                 != (df["model_name_b"] == PAIRWISE_BASELINE_MODEL))]


def extract_subset_from_full(kind: str = "ridge", save_dir: Path = None) -> pd.DataFrame:
    """Carve a subset CSV out of the existing all-pairs CSV — no recompute.

    The full `pairwise_significance.csv` already holds every pair, so the
    model-vs-Ridge rows an earlier unrestricted run produced can simply be
    extracted instead of bootstrapped again. Use this when the existing
    comparisons are the ones the tables should show: a fresh run computes at
    the current N_BOOTSTRAP, which mixes eras if the file on disk was written
    at a different resample count (the `n_boot_diff` column records it).

    Covers only what that run covered — result CSVs added since are missing
    from it and will show no subscript until the comparison is computed.
    """
    if save_dir is None:
        save_dir = EVAL_SAVE_DIR
    full_path = save_dir / PAIRWISE_CSV
    if not full_path.exists():
        raise FileNotFoundError(f"No all-pairs CSV to extract from: {full_path}")
    out_name = {"ridge": PAIRWISE_RIDGE_CSV, "baseline": PAIRWISE_WPM_CSV}[kind]
    # Read in chunks: the all-pairs file runs to hundreds of MB.
    chunks = [_subset_by_kind(chunk, kind)
              for chunk in pd.read_csv(full_path, chunksize=500_000)]
    subset = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    out_path = save_dir / out_name
    subset.to_csv(out_path, index=False)
    if "n_boot_diff" in subset.columns and not subset.empty:
        logger.info("Resample counts in the extracted rows: %s",
                    subset["n_boot_diff"].value_counts().to_dict())
    logger.info("Extracted %d %s rows from %s to %s",
                len(subset), kind, full_path.name, out_path)
    return subset


def compute_pairwise_significance(results_dir: Path = None, save_dir: Path = None,
                                  target_cols: List[str] = None, feature_sets: List[str] = None,
                                  model_names: List[str] = None, agg_type: str = None,
                                  baseline_only: bool = False,
                                  vs_ridge: bool = False) -> pd.DataFrame:
    """Pairwise paired-bootstrap test between every two prediction models sharing
    the same target.

    Two models are "dependent overlapping" correlations: each model's `pred`
    correlates with the shared `true` target, and the two models' preds are
    themselves correlated. Models are grouped by
    (dataset, aug_p, preview, version_path, target_col, method[, train_type]);
    within a group every (feature_set, model_name) pair is compared on the
    participants they share.

    Two restricted comparison sets can be requested instead of all pairs:

    `baseline_only` — only pairs where one side is the WPM baseline feature
    set (PAIRWISE_BASELINE_FS); ~10x fewer pairs than the full sweep.
    `vs_ridge` — only model-vs-Ridge pairs (same feature set, target and
    method; PAIRWISE_BASELINE_MODEL on one side), i.e. one row per cell of a
    non-Ridge model's results table. ~40x fewer pairs again.
    Setting both computes the union.

    Output files: an unrestricted run writes `pairwise_significance.csv` (all
    pairs) and also refreshes both subsets, `pairwise_significance_wpm.csv`
    (what the bootstrap combined tables star against) and
    `pairwise_significance_ridge.csv` (the per-cell difference from Ridge).
    A restricted run writes only the subset files it actually computed and
    leaves the full all-pairs CSV untouched.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    if save_dir is None:
        save_dir = EVAL_SAVE_DIR

    entries = collect_result_csvs(results_dir, target_cols=target_cols,
                                  feature_sets=feature_sets, model_names=model_names,
                                  agg_type=agg_type)
    if not entries:
        logger.warning("No result CSVs found in %s for pairwise significance", results_dir)
        return pd.DataFrame()

    # Group models that share a target (and everything else except the model).
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for e in entries:
        key = (e["dataset"], e["aug_p"], e["preview"], e["version_path"],
               e["target_col"], e["method"], e.get("train_type"))
        grouped[key].append(e)

    # Each group is independent (load CSVs, run the dependent-correlation
    # tests), so groups are farmed out to process workers — same pattern and
    # PRED_EVAL_N_JOBS cap as evaluate_all_results.
    max_workers = _eval_max_workers()
    kinds = set()
    if baseline_only:
        kinds.add("baseline")
    if vs_ridge:
        kinds.add("ridge")
    pair_kinds = kinds or None
    if pair_kinds:
        logger.info("Pairwise restricted to %s pairs (%s)",
                    " + ".join(sorted(pair_kinds)),
                    f"vs the {PAIRWISE_BASELINE_FS} feature set / "
                    f"the {PAIRWISE_BASELINE_MODEL} model")
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(
            partial(_pairwise_group, pair_kinds=pair_kinds),
            # chunksize=1: with ~12 groups a larger chunksize leaves most
            # workers idle while one works through its chunk serially.
            grouped.items(), chunksize=1)
        for group_rows in tqdm(results_iter, total=len(grouped),
                               desc="Pairwise model significance"):
            rows.extend(group_rows)

    result_df = pd.DataFrame(rows)
    save_dir.mkdir(parents=True, exist_ok=True)

    # A comparison is regenerated only when BOTH of its endpoints were in
    # scope, so the feature-set and model filters apply to the _a and _b
    # columns alike; the scope entries are ANDed, which is exactly that.
    base = _evaluation_scope(target_cols, agg_type=agg_type)
    # Same directory-name vs suffixed-row-label mismatch as in
    # evaluate_all_results: model_name_a/_b carry the inner-validation suffix,
    # so scope on the collected names rather than the raw --models filter.
    scope_models = (sorted({e["model_name"] for e in entries})
                    if model_names is not None else None)
    # Always derived from what results_dir actually held (see
    # _evaluation_scope docstring) so an other-dataset/-preview row is never
    # mistaken for covered just because target_col/feature_set/model matched.
    scope_datasets = sorted({e["dataset"] for e in entries})
    scope_previews = sorted({e["preview"] for e in entries})
    scope = {
        "target_col": base["target_col"],
        "version_path": base["version_path"],
        "dataset": scope_datasets,
        "preview": scope_previews,
        "feature_set_a": feature_sets, "feature_set_b": feature_sets,
        "model_name_a": scope_models, "model_name_b": scope_models,
    }

    # A restricted run leaves the full all-pairs CSV untouched and refreshes
    # only the subset files it computed; an unrestricted run refreshes all three.
    if pair_kinds is None:
        save_preserving_unevaluated(result_df, save_dir / PAIRWISE_CSV, scope)
    for kind, filename in [("baseline", PAIRWISE_WPM_CSV),
                           ("ridge", PAIRWISE_RIDGE_CSV)]:
        if pair_kinds is not None and kind not in pair_kinds:
            continue
        if result_df.empty:
            # Nothing was computed, so there is nothing to replace — writing
            # the empty frame would wipe the subset file instead.
            logger.warning("No pairwise rows computed; leaving %s untouched", filename)
            continue
        save_preserving_unevaluated(_subset_by_kind(result_df, kind),
                                    save_dir / filename, scope)
    return result_df


def run_evaluation_pipeline(model_names: Optional[List[str]] = None,
                            skip_pairwise: bool = False,
                            pairwise_baseline_only: bool = False,
                            pairwise_vs_ridge: bool = False,
                            comparisons_only: bool = False,
                            include_bootstrap: bool = True,
                            agg_type: Optional[str] = None) -> pd.DataFrame:
    """Run the full evaluation stage (metrics + pairwise significance).

    With `model_names=None`, every model found on disk is evaluated and the
    combined CSVs are fully overwritten. With a list of names, only those
    models' result CSVs are (re-)evaluated and their rows are merged into the
    existing combined CSVs (other models' rows are kept). Names outside the
    configured MODELS registry are allowed — any model directory with result
    CSVs under RESULTS_DIR is picked up, and submodels/configurations that
    have results already are evaluated even if the model run is incomplete.

    `agg_type` restricts both stages to result CSVs whose version_path starts
    with it (e.g. "fully_agg"), leaving every other aggregation's rows in the
    combined CSVs untouched. Unfiltered, the sweep also picks up first_p_agg,
    moving_p_agg, split-half, per-item and every *_backup / *_prerun / *_stale
    directory on disk — an order of magnitude more files than the paper tables
    read.

    `comparisons_only` skips the per-model metrics stage entirely — the
    evaluation CSVs already hold those numbers — and runs only the requested
    comparison sets. It computes whichever of `pairwise_baseline_only` /
    `pairwise_vs_ridge` are set, and both if neither is: the vs-Ridge set
    alone is a fraction of the cost of the WPM-baseline set, so asking for
    just it (`comparisons_only=True, pairwise_vs_ridge=True`) is the cheap
    refresh when the WPM pairs are already up to date.
    """
    if comparisons_only:
        if skip_pairwise:
            raise ValueError("comparisons_only and skip_pairwise are contradictory")
        if not (pairwise_baseline_only or pairwise_vs_ridge):
            pairwise_baseline_only = True
            pairwise_vs_ridge = True
    if model_names:
        unknown = [m for m in model_names if m not in MODELS]
        if unknown:
            logger.warning(
                "Model(s) %s are not in the configured MODELS registry; they "
                "will still be evaluated from any result CSVs found on disk. "
                "Known: %s", unknown, list(MODELS.keys()))

    feature_sets = list(ALL_FEATURE_SETS_DICT.keys())

    # Ensure proficiency_agg is included for Meco datasets
    target_cols_with_proficiency = list(ALL_TESTS)
    if TestCols.PROFICIENCY_AGG_COL not in target_cols_with_proficiency:
        target_cols_with_proficiency.append(TestCols.PROFICIENCY_AGG_COL)

    if comparisons_only:
        result = pd.DataFrame()
    else:
        result = evaluate_all_results(target_cols=target_cols_with_proficiency,
                                      model_names=model_names, feature_sets=feature_sets,
                                      agg_type=agg_type,
                                      include_bootstrap=include_bootstrap)

    if not skip_pairwise:
        # Pairwise tests compare models against each other, so a
        # single-model evaluation cannot compute its cross-model pairs from a
        # filtered collection. Always run pairwise across ALL models (full
        # overwrite of pairwise_significance.csv) so e.g. a newly added
        # model's pairs against the existing models are included.
        compute_pairwise_significance(target_cols=target_cols_with_proficiency,
                                      model_names=None, feature_sets=feature_sets,
                                      agg_type=agg_type,
                                      baseline_only=pairwise_baseline_only,
                                      vs_ridge=pairwise_vs_ridge)
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate prediction results")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names to (re-)evaluate. Their rows are "
             "merged into the existing combined CSVs; other models' rows are "
             "kept. Names not in the MODELS registry are allowed (evaluated "
             "from whatever result CSVs exist on disk). Default: all models "
             f"(full overwrite). Known: {', '.join(MODELS.keys())}")
    parser.add_argument(
        "--skip-pairwise", action="store_true",
        help="Skip the pairwise significance stage.")
    parser.add_argument(
        "--pairwise-baseline-only", action="store_true",
        help="Only compute pairwise tests involving the WPM baseline feature "
             "set (the pairs the bootstrap tables use); ~10x fewer pairs. "
             "Writes only pairwise_significance_wpm.csv; the full all-pairs "
             "pairwise_significance.csv is left untouched.")
    parser.add_argument(
        "--pairwise-vs-ridge", action="store_true",
        help="Compute the model-vs-Ridge comparison: every cell of a "
             "non-Ridge model against the same cell of Ridge (same feature "
             f"set, target and method). Writes {PAIRWISE_RIDGE_CSV}. "
             "Combine with --pairwise-baseline-only for both restricted sets.")
    parser.add_argument(
        "--no-bootstrap-ci", action="store_true",
        help="Skip the per-row bootstrap in the metrics stage (point "
             "estimates only). That resample loop is ~99.9% of the stage's "
             "cost — fully_agg drops from ~20 min to seconds — but the rows "
             "this run writes carry NaN in every *_boot / *_ci_*_boot column, "
             "so the bootstrap tables have nothing to print for them. For "
             "quick point-estimate refreshes, not for producing paper tables. "
             "Does not affect the pairwise stage, where the bootstrap IS the "
             "significance test (use --skip-pairwise there).")
    parser.add_argument(
        "--extract-vs-ridge", action="store_true",
        help=f"Do not compute anything: carve {PAIRWISE_RIDGE_CSV} out of the "
             "model-vs-Ridge rows an earlier all-pairs run already wrote to "
             f"{PAIRWISE_CSV}. Instant, and keeps the difference subscripts "
             "in the same bootstrap era as the stars already in the tables. "
             "Cells whose result CSVs postdate that run stay blank.")
    parser.add_argument(
        "--comparisons-only", action="store_true",
        help="Shorthand for --pairwise-baseline-only --pairwise-vs-ridge with "
             "the per-model metrics stage skipped: refreshes only the two "
             "comparison CSVs the tables read (WPM stars and vs-Ridge "
             "differences), leaving all_evaluation_results.csv untouched.")
    parser.add_argument(
        "--agg-type", type=str, default=None,
        help="Only evaluate result CSVs whose version_path starts with this "
             "(e.g. fully_agg) — other aggregations' rows are kept as-is. "
             "Default: sweep everything under the results tree, which also "
             "picks up first_p_agg, moving_p_agg, split-half, per-item and "
             "any *_backup / *_prerun / *_stale directories.")
    args = parser.parse_args()

    eval_model_names = ([m.strip() for m in args.models.split(",")]
                        if args.models else None)
    if args.extract_vs_ridge:
        extract_subset_from_full("ridge")
        raise SystemExit(0)
    run_evaluation_pipeline(model_names=eval_model_names,
                            skip_pairwise=args.skip_pairwise,
                            pairwise_baseline_only=args.pairwise_baseline_only,
                            pairwise_vs_ridge=args.pairwise_vs_ridge,
                            comparisons_only=args.comparisons_only,
                            include_bootstrap=not args.no_bootstrap_ci,
                            agg_type=args.agg_type)
