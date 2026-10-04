"""
Generate LaTeX results tables for the predictions pipeline.

Produces one .tex table per (dataset, preview, version_path) combination.
Each table has:
  - Rows: feature sets (single features + combined, separated by \\hline)
  - Columns: CV method (top) → target columns (sub) → r, MAE (sub-sub)
  - Best value per column bolded

Usage:
    python -m src.evaluation.predictions.tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

from src.constants import MICH_TEST_PARTS_EXTENDED, TestCols
from src.evaluation.predictions.evaluation import (
    EVAL_SAVE_DIR, REQUIRE_L1_LEXTALE, REQUIRE_L1_LEXTALE_SUBTREE,
)
from src.evaluation.stats_utils import N_BOOTSTRAP
from src.configs import ModelNames
from src.latex_captions import rename_datasets_in_tables

logger = logging.getLogger(__name__)

def write_table(table_path, latex: str) -> None:
    """Write a finished table, applying the shared dataset-naming rules."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(rename_datasets_in_tables(latex))


TABLES_SAVE_DIR = Path("src/evaluation/predictions/tables")
if REQUIRE_L1_LEXTALE:
    # Mirror the eval-side redirection (see evaluation.py): tables for the
    # LexTALE-only-L1 runs land in their own subtree, same layout below it.
    TABLES_SAVE_DIR = TABLES_SAVE_DIR / REQUIRE_L1_LEXTALE_SUBTREE

# ── Display name mappings ────────────────────────────────────────────────────

FEATURE_SET_DISPLAY = {
    "READING_SPEED": "WPM",
    "FIXATION_METRICS": "Avg. Fix.",
    "S_CLUSTERS_NO_NORM": "S-Clusters",
    "WP_COEFS_NO_NORM": "WP-Coefs",
    "READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM": "Combined",
    "TRANSITIONS": "Transitions",
    "WFC": "Word Fix.",
}

# Per-dataset target column display labels.
DATASET_TARGET_COL_DISPLAY = {
    "OneStop": {
        "michtest_score": "Michigan",
        "lextale_score": "LexTALE",
        "comprehension_score-regular_trials": "Comprehension",
    },
    "Meco": {
        "proficiency_agg": "Composite",
        "lextale_score": "LexTALE",
        # MECO gained comprehension once the MECO L2 OSF accuracy release was
        # merged into the metadata; OneStop has had this entry all along.
        "comprehension_score-regular_trials": "Comprehension",
    },
}

# Michigan test parts display labels
MICHIGAN_PARTS_DISPLAY = {
    TestCols.MICHIGEN_TEST_COL: "MichTest (Overall)",
    "MPT_listening_score": "Listening",
    "MPT_grammar_score": "Grammar",
    "MPT_vocabulary_score": "Vocabulary",
    "MPT_reading_score": "Reading",
    "MPT_listen_grammar": "Listen+Grammar",
    "MPT_vocab_read": "Vocab+Reading",
    "MPT_grammar_vocab_read": "Grammar+Vocab+Reading",
}

# Mapping from raw method name in the eval CSV to the scope label shown in
# the table. Only methods listed here are emitted as columns.
DATASET_METHOD_TO_SCOPE = {
    # Canonical Pool × FoldMethod names after evaluation.normalize_method;
    # legacy stems (batch_cross_validation_seen_small, leave_one_participant_out_seen, …)
    # are mapped to these at the discovery layer in evaluation.py.
    "OneStop": {
        "seen__pool_all": "Fixed",
        "unseen__pool_all": "Any",
    },
    "Meco": {
        "seen__pool_all": "Fixed",
        "unseen__pool_all": "Any",
    },
}

DATASET_SCOPES = {
    "OneStop": ["Fixed", "Any"],
    "Meco": ["Fixed", "Any"],
}

# Feature sets that are inherently per-text — they only have a "Seen" cell.
FIXED_TEXT_ONLY_FEATURE_SETS = {"TRANSITIONS", "WFC"}

# Model display names
MODEL_DISPLAY = {
    "Ridge_Classifier": "Ridge",
    "Log_Ridge_Regression": "Log Ridge",
    "Linear_Regression": "Linear",
    "Random_Forest": "Random Forest",
    "Decision_Tree": "Decision Tree",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "TabStar": "TabStar",
    "TabPFN": "TabPFN-3",
    "Average": "Avg. Train",
}

# Tuned tree runs arrive as "<Model>_<scheme>" (Decision_Tree_kfold): the eval
# layer folds the inner-validation scheme into the model name so the trees share
# a version_path -- and therefore a comparison group -- with the untuned Ridge
# baseline. Expand the display maps over the schemes rather than hardcoding each
# combination, so a new strategy needs no edit here.
INNER_VALIDATION_DISPLAY = {"holdout": "holdout", "kfold": "k-fold"}
for _base, _disp in list(MODEL_DISPLAY.items()):
    for _sch, _sdisp in INNER_VALIDATION_DISPLAY.items():
        MODEL_DISPLAY[f"{_base}_{_sch}"] = f"{_disp} ({_sdisp})"

# Model phrase used inside combined-table captions ("... using eye movements
# with <phrase>."). Kept separate from MODEL_DISPLAY so the Ridge paper
# caption stays exactly "Ridge regression".
MODEL_CAPTION_DISPLAY = {
    "Ridge_Classifier": "Ridge regression",
    "Log_Ridge_Regression": "log-transformed Ridge regression",
    "Linear_Regression": "linear regression",
    "Random_Forest": "a Random Forest",
    "Decision_Tree": "a decision tree",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "TabStar": "TabStar",
    "TabPFN": "TabPFN-3",
    "Average": "a train-set-mean baseline",
}
for _base, _disp in list(MODEL_CAPTION_DISPLAY.items()):
    for _sch, _sdisp in INNER_VALIDATION_DISPLAY.items():
        MODEL_CAPTION_DISPLAY[f"{_base}_{_sch}"] = f"{_disp}, tuned by {_sdisp}"


def caption_model_bold(model_name: str) -> str:
    """``LightGBM_holdout`` -> ``\\textbf{LightGBM}``.

    The inner-validation scheme is dropped: it is carried by the \\label and the
    surrounding text, and in the caption it only competes with the model name.
    """
    for sch in INNER_VALIDATION_DISPLAY:
        if model_name.endswith(f"_{sch}"):
            model_name = model_name[: -len(sch) - 1]
            break
    disp = MODEL_DISPLAY.get(model_name, model_name.replace("_", " "))
    return f"\\textbf{{{disp}}}"


def _caption_model_phrase(model_name: str) -> str:
    """Caption-ready phrase for a model; unknown models fall back to the raw
    name with underscores replaced (so newly added models never break)."""
    return MODEL_CAPTION_DISPLAY.get(model_name, model_name.replace("_", " "))

# Feature set rows grouped for the table; an \hline is inserted between groups.
FEATURE_SET_GROUPS = [
    ["READING_SPEED"],
    [
        "FIXATION_METRICS",
        "S_CLUSTERS_NO_NORM",
        "WP_COEFS_NO_NORM",
    ],
    ["TRANSITIONS", "WFC"],
]

ALL_FEATURE_SETS_ORDERED = [fs for group in FEATURE_SET_GROUPS for fs in group]
FEATURE_SET_GROUP_INDEX = {
    fs: i for i, group in enumerate(FEATURE_SET_GROUPS) for fs in group
}

# Target columns per dataset
DATASET_TARGET_COLS = {
    "OneStop": ["lextale_score", "michtest_score"],
    "Meco": ["lextale_score", "proficiency_agg"],
}

# OneStop comprehension target columns (lextale, michtest, comprehension from regular trials)
ONESTOP_COMPREHENSION_TARGET_COLS = ["lextale_score", "michtest_score", "comprehension_score-regular_trials"]


def _display(key: str, mapping: dict) -> str:
    result = mapping.get(key, key)
    # If not in mapping, escape underscores for LaTeX
    if key not in mapping:
        result = result.replace("_", r"\_")
    return result


def _fmt_val(val: float, bold: bool, decimals: int = 2) -> str:
    """Format a numeric value for the table, bolding if best in column."""
    if pd.isna(val):
        return "NA"
    formatted = f"{val:.{decimals}f}"
    if bold:
        return f"\\textbf{{{formatted}}}"
    return formatted


def generate_latex_table(
    eval_df: pd.DataFrame,
    target_cols: List[str],
    scopes: List[str],
    method_to_scope: dict,
    target_col_display: dict,
    dataset: str = "",
    version_path: str = "",
    model_name: str = "",
    feature_sets_ordered: Optional[List[str]] = None,
    feature_set_group_index: Optional[dict] = None,
) -> str:
    """Generate a LaTeX predictions table in the MichTest/LexTALE × Seen/Unseen × {r, MAE} layout.

    Args:
        eval_df: DataFrame filtered to a single (dataset, aug, preview, version_path, model).
        target_cols: Ordered list of target column names (outer column group).
        scopes: Ordered list of scope labels (middle column group), e.g. ["Seen","Unseen"].
        method_to_scope: Mapping from raw method name -> scope label.
        target_col_display: Mapping from raw target_col -> display label.
        dataset, version_path: Used to build the table caption.
        model_name: Model name to include in the table caption.

    Returns:
        LaTeX table string, or empty string if no data.
    """
    ordered = feature_sets_ordered if feature_sets_ordered is not None else ALL_FEATURE_SETS_ORDERED
    group_index = feature_set_group_index if feature_set_group_index is not None else FEATURE_SET_GROUP_INDEX
    feature_sets = [fs for fs in ordered
                    if fs in eval_df["feature_set"].unique()]
    if not feature_sets:
        return ""

    metrics = ["pearson_r", "mae"]
    metric_labels = {"pearson_r": "$r$", "mae": "MAE"}

    # Build a lookup: (feature_set, target_col, scope) -> {pearson_r, mae}
    lookup: dict = {}
    for _, row in eval_df.iterrows():
        scope = method_to_scope.get(row["method"])
        if scope is None:
            continue
        key = (row["feature_set"], row["target_col"], scope)
        lookup[key] = {m: row[m] for m in metrics}

    # Force fixed-text-only feature sets to NA in non-Fixed scopes.
    for fs in feature_sets:
        if fs in FIXED_TEXT_ONLY_FEATURE_SETS:
            for tc in target_cols:
                for scope in scopes:
                    if scope != "Fixed":
                        lookup[(fs, tc, scope)] = {"pearson_r": np.nan, "mae": np.nan}

    # Find best (max r, min mae) per (target_col, scope) across feature sets
    best: dict = {}
    for tc in target_cols:
        for scope in scopes:
            r_vals, mae_vals = [], []
            for fs in feature_sets:
                vals = lookup.get((fs, tc, scope))
                if vals:
                    if not pd.isna(vals["pearson_r"]):
                        r_vals.append(vals["pearson_r"])
                    if not pd.isna(vals["mae"]):
                        mae_vals.append(vals["mae"])
            best[(tc, scope, "pearson_r")] = max(r_vals) if r_vals else None
            best[(tc, scope, "mae")] = min(mae_vals) if mae_vals else None

    n_scopes = len(scopes)
    n_metrics = len(metrics)

    lines = []
    lines.append("\\begin{table*}[ht!]")
    lines.append("\\centering")
    lines.append("\\small")

    if dataset or model_name:
        display_model_name = MODEL_DISPLAY.get(model_name, model_name) if model_name else ""
        caption_parts = [p for p in (dataset, display_model_name) if p]
        lines.append(f"\\caption{{{' -- '.join(caption_parts)}}}")

    # Column spec: features col, then per-target block of (per-scope (r, MAE))
    col_groups = []
    for _ in target_cols:
        scope_block = "|".join(["cc"] * n_scopes)
        col_groups.append(scope_block)
    col_spec = "l||" + "||".join(col_groups)

    # Wrap in resizebox if many columns to prevent overfull hbox
    if len(target_cols) * n_scopes > 4:
        lines.append("\\resizebox{\\textwidth}{!}{")

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header row 1: target_col display names spanning scope * metric
    header1_parts = [""]
    for tc in target_cols:
        span = n_scopes * n_metrics
        display = _display(tc, target_col_display)
        header1_parts.append(f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{display}}}}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    # Header row 2: scope labels spanning r + MAE
    header2_parts = [""]
    for _tc in target_cols:
        for scope in scopes:
            header2_parts.append(f"\\multicolumn{{{n_metrics}}}{{c}}{{{scope}}}")
    lines.append(" & ".join(header2_parts) + " \\\\")

    # Header row 3: metric labels (r, MAE repeated)
    header3_parts = ["\\textbf{Features}"]
    for _tc in target_cols:
        for _scope in scopes:
            for m in metrics:
                header3_parts.append(metric_labels[m])
    lines.append(" & ".join(header3_parts) + " \\\\")
    lines.append("\\hline")

    for i, fs in enumerate(feature_sets):
        if i > 0 and group_index[fs] != group_index[feature_sets[i - 1]]:
            lines.append("\\hline")

        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for tc in target_cols:
            for scope in scopes:
                vals = lookup.get((fs, tc, scope))
                for m in metrics:
                    if vals and not pd.isna(vals[m]):
                        best_val = best.get((tc, scope, m))
                        is_best = (best_val is not None and
                                   np.isclose(vals[m], best_val, atol=1e-6))
                        row_parts.append(_fmt_val(vals[m], is_best))
                    else:
                        row_parts.append("NA")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")

    # Close resizebox if it was opened
    if len(target_cols) * n_scopes > 4:
        lines.append("}")

    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_all_tables(
    eval_csv: Path = None,
    save_dir: Path = None,
    model_name: str = None,
    feature_sets: Optional[List[str]] = None,
):
    """Generate all LaTeX tables from the combined evaluation CSV.

    Creates one .tex file per (dataset, aug_p, preview, version_path, model).
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR
    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_ORDERED

    df = pd.read_csv(eval_csv)

    # Filter to feature sets and optionally to a specific model
    df = df[df["feature_set"].isin(feature_sets)]
    if model_name is not None:
        df = df[df["model_name"] == model_name]

    if df.empty:
        logger.warning("No data found after filtering for model=%s", model_name)
        return

    # Group by (dataset, aug_p, preview, version_path, model_name)
    group_cols = ["dataset", "aug_p", "preview", "version_path", "model_name"]
    for group_key, group_df in df.groupby(group_cols):
        dataset, aug_p, preview, version_path, curr_model_name = group_key

        # Determine target columns for this dataset
        target_cols = DATASET_TARGET_COLS.get(dataset)
        if target_cols is None:
            logger.warning("No target columns configured for dataset=%s, skipping", dataset)
            continue

        # Filter to relevant target columns
        group_df = group_df[group_df["target_col"].isin(target_cols)]
        if group_df.empty:
            continue

        method_to_scope = DATASET_METHOD_TO_SCOPE.get(dataset)
        scopes = DATASET_SCOPES.get(dataset)
        target_col_display = DATASET_TARGET_COL_DISPLAY.get(dataset)
        if method_to_scope is None or scopes is None or target_col_display is None:
            logger.warning("No scope mapping configured for dataset=%s, skipping", dataset)
            continue

        # For per_item_agg, use ["Fixed", "Any"] scopes split by seen/unseen
        if version_path.startswith("per_item_agg"):
            scopes = ["Fixed", "Any"]
            method_to_scope = {
                "seen__pool_all": "Fixed",
                "unseen__pool_all": "Any",
            }

        latex = generate_latex_table(
            group_df,
            target_cols=target_cols,
            scopes=scopes,
            method_to_scope=method_to_scope,
            target_col_display=target_col_display,
            dataset=dataset,
            version_path=version_path,
            model_name=curr_model_name,
        )
        if not latex:
            continue

        # Save to directory: dataset/aug_p/preview/version_path/model/table.tex
        table_dir = save_dir / dataset / aug_p / preview / version_path / curr_model_name
        table_dir.mkdir(parents=True, exist_ok=True)
        table_path = table_dir / "main.tex"
        write_table(table_path, latex)
        logger.info("Saved table: %s", table_path)

    logger.info("All tables generated under %s", save_dir)


# ── LOLO vs L1-stratified k-fold table ──────────────────────────────────────

LOLO_VS_STRAT_METHOD_TO_SCOPE = {
    "all__lolo": "LOLO",
    "all__l1_strat_kfold": "L1-Strat.",
}
LOLO_VS_STRAT_SCOPES = ["LOLO", "L1-Strat."]

LOLO_VS_STRAT_FEATURE_GROUPS = [
    ["READING_SPEED"],
    ["FIXATION_METRICS", "WP_COEFS_NO_NORM", "S_CLUSTERS_NO_NORM"],
    ["READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM"],
]
LOLO_VS_STRAT_FEATURE_SETS = [fs for g in LOLO_VS_STRAT_FEATURE_GROUPS for fs in g]
LOLO_VS_STRAT_GROUP_INDEX = {
    fs: i for i, g in enumerate(LOLO_VS_STRAT_FEATURE_GROUPS) for fs in g
}


def generate_lolo_vs_stratified_tables(
    eval_csv: Path = None,
    save_dir: Path = None,
    model_name: str = None,
    feature_sets: Optional[List[str]] = None,
):
    """One .tex per (dataset, aug_p, preview, version_path, model) comparing LOLO vs L1-stratified k-fold."""
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR  # dataset-first; filename "lolo_vs_stratified.tex" carries the variant
    if feature_sets is None:
        feature_sets = LOLO_VS_STRAT_FEATURE_SETS

    df = pd.read_csv(eval_csv)
    df = df[
        (df["feature_set"].isin(feature_sets))
        & (df["method"].isin(LOLO_VS_STRAT_METHOD_TO_SCOPE.keys()))
    ]
    if model_name is not None:
        df = df[df["model_name"] == model_name]

    if df.empty:
        logger.warning("No LOLO/random rows for model=%s", model_name)
        return

    for group_key, group_df in df.groupby(["dataset", "aug_p", "preview", "version_path", "model_name"]):
        dataset, aug_p, preview, version_path, curr_model_name = group_key
        target_cols = DATASET_TARGET_COLS.get(dataset)
        target_col_display = DATASET_TARGET_COL_DISPLAY.get(dataset)
        if target_cols is None or target_col_display is None:
            continue

        group_df = group_df[group_df["target_col"].isin(target_cols)]
        if group_df.empty:
            continue

        latex = generate_latex_table(
            group_df,
            target_cols=target_cols,
            scopes=LOLO_VS_STRAT_SCOPES,
            method_to_scope=LOLO_VS_STRAT_METHOD_TO_SCOPE,
            target_col_display=target_col_display,
            dataset=dataset,
            version_path=version_path,
            model_name=curr_model_name,
            feature_sets_ordered=LOLO_VS_STRAT_FEATURE_SETS,
            feature_set_group_index=LOLO_VS_STRAT_GROUP_INDEX,
        )
        if not latex:
            continue

        table_dir = save_dir / dataset / aug_p / preview / version_path / curr_model_name
        table_dir.mkdir(parents=True, exist_ok=True)
        table_path = table_dir / "lolo_vs_stratified.tex"
        write_table(table_path, latex)
        logger.info("Saved table: %s", table_path)


# ── OneStop comprehension tables ───────────────────────────────────────────────

def generate_comprehension_tables(
    eval_csv: Path = None,
    save_dir: Path = None,
    model_name: str = None,
    feature_sets: Optional[List[str]] = None,
):
    """Generate comprehension tables for OneStop (michtest + lextale + comprehension).

    Creates one .tex file per (dataset, aug_p, preview, version_path, model).
    Only generates tables for fully_agg version_path.
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR  # dataset-first; filename "comprehension.tex"
    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_ORDERED

    df = pd.read_csv(eval_csv)
    df = df[df["feature_set"].isin(feature_sets)]
    if model_name is not None:
        df = df[df["model_name"] == model_name]

    if df.empty:
        logger.warning("No comprehension data found after filtering for model=%s", model_name)
        return

    # Filter to OneStop, fully_agg, and comprehension target columns
    df = df[
        (df["dataset"] == "OneStop")
        & (df["version_path"] == "fully_agg")
        & (df["target_col"].isin(ONESTOP_COMPREHENSION_TARGET_COLS))
    ]
    if df.empty:
        logger.warning("No OneStop comprehension data found for model=%s", model_name)
        return

    # Group by (dataset, aug_p, preview, version_path, model_name)
    group_cols = ["dataset", "aug_p", "preview", "version_path", "model_name"]
    for group_key, group_df in df.groupby(group_cols):
        dataset, aug_p, preview, version_path, curr_model_name = group_key

        # Filter to relevant target columns
        group_df = group_df[group_df["target_col"].isin(ONESTOP_COMPREHENSION_TARGET_COLS)]
        if group_df.empty:
            continue

        method_to_scope = {
            "seen__pool_all": "Fixed",
            "unseen__pool_all": "Any",
        }
        scopes = ["Fixed", "Any"]

        latex = generate_latex_table(
            group_df,
            target_cols=ONESTOP_COMPREHENSION_TARGET_COLS,
            scopes=scopes,
            method_to_scope=method_to_scope,
            target_col_display=DATASET_TARGET_COL_DISPLAY[dataset],
            dataset=dataset,
            version_path=version_path,
            model_name=curr_model_name,
        )
        if not latex:
            continue

        # Save to directory
        table_dir = save_dir / dataset / aug_p / preview / version_path / curr_model_name
        table_dir.mkdir(parents=True, exist_ok=True)
        table_path = table_dir / "comprehension.tex"
        write_table(table_path, latex)
        logger.info("Saved table: %s", table_path)

    logger.info("Comprehension tables generated under %s", save_dir)


# ── Michigan test parts tables ─────────────────────────────────────────────

MICHIGAN_PARTS_FEATURE_GROUPS = [
    ["READING_SPEED"],
    ["FIXATION_METRICS", "WP_COEFS_NO_NORM", "S_CLUSTERS_NO_NORM"],
    ["READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM"],
]
MICHIGAN_PARTS_FEATURE_SETS = [fs for g in MICHIGAN_PARTS_FEATURE_GROUPS for fs in g]
MICHIGAN_PARTS_GROUP_INDEX = {
    fs: i for i, g in enumerate(MICHIGAN_PARTS_FEATURE_GROUPS) for fs in g
}

# Michigan test parts target columns (overall score first, then individual parts + combinations)
MICHIGAN_PARTS_TARGET_COLS = [TestCols.MICHIGEN_TEST_COL] + MICH_TEST_PARTS_EXTENDED

# Michigan test parts simplified (first and second parts)
MICHIGAN_PARTS_SIMPLIFIED_TARGET_COLS = [TestCols.MICHIGEN_TEST_COL, "MPT_listen_grammar", "MPT_vocab_read"]


# ── OneStop comprehension + LOLO vs stratified ────────────────────────────────

def generate_comprehension_lolo_vs_stratified_tables(
    eval_csv: Path = None,
    save_dir: Path = None,
    model_name: str = None,
    feature_sets: Optional[List[str]] = None,
):
    """LOLO vs stratified tables for OneStop comprehension (fully_agg only)."""
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR  # dataset-first; filename "comprehension_lolo_vs_stratified.tex"
    if feature_sets is None:
        feature_sets = LOLO_VS_STRAT_FEATURE_SETS

    df = pd.read_csv(eval_csv)
    df = df[
        (df["dataset"] == "OneStop")
        & (df["version_path"] == "fully_agg")
        & (df["feature_set"].isin(feature_sets))
        & (df["method"].isin(LOLO_VS_STRAT_METHOD_TO_SCOPE.keys()))
        & (df["target_col"].isin(ONESTOP_COMPREHENSION_TARGET_COLS))
    ]
    if model_name is not None:
        df = df[df["model_name"] == model_name]

    if df.empty:
        logger.warning("No comprehension LOLO/stratified data found for model=%s", model_name)
        return

    for group_key, group_df in df.groupby(["aug_p", "preview", "version_path", "model_name"]):
        aug_p, preview, version_path, curr_model_name = group_key

        group_df = group_df[group_df["target_col"].isin(ONESTOP_COMPREHENSION_TARGET_COLS)]
        if group_df.empty:
            continue

        latex = generate_latex_table(
            group_df,
            target_cols=ONESTOP_COMPREHENSION_TARGET_COLS,
            scopes=LOLO_VS_STRAT_SCOPES,
            method_to_scope=LOLO_VS_STRAT_METHOD_TO_SCOPE,
            target_col_display=DATASET_TARGET_COL_DISPLAY["OneStop"],
            dataset="OneStop",
            version_path=version_path,
            model_name=curr_model_name,
            feature_sets_ordered=LOLO_VS_STRAT_FEATURE_SETS,
            feature_set_group_index=LOLO_VS_STRAT_GROUP_INDEX,
        )  # Note: Uses LOLO and L1-Strat scopes (unchanged)
        if not latex:
            continue

        table_dir = save_dir / "OneStop" / aug_p / preview / version_path / curr_model_name
        table_dir.mkdir(parents=True, exist_ok=True)
        table_path = table_dir / "comprehension_lolo_vs_stratified.tex"
        write_table(table_path, latex)
        logger.info("Saved table: %s", table_path)


# ── OneStop comprehension + Michigan test parts ─────────────────────────────────

# The Michigan-parts table family, one table per (aug_p, preview, model):
# {comprehension_}michigan_parts{_simplified}_{cv}.tex. cv -> (label used in log
# messages, label used in run_all_tables' progress lines, scopes, method -> scope).
MICHIGAN_PARTS_CV = {
    "lolo": ("LOLO", "LOLO", ["Fixed"], {"all__lolo": "Fixed"}),
    "lopo": ("LOPO", "LOPO", ["Overall"], {"all__pool_all": "Overall"}),
    "batch_cv": ("batch CV", "Batch CV", ["Fixed", "Any"],
                 {"seen__pool_all": "Fixed", "unseen__pool_all": "Any"}),
}


def generate_michigan_parts_tables(
    cv: str,
    comprehension: bool = False,
    simplified: bool = False,
    eval_csv: Path = None,
    save_dir: Path = None,
    model_name: str = None,
    feature_sets: Optional[List[str]] = None,
):
    """Michigan test parts tables for OneStop fully_agg, one per (aug_p, preview, model).

    Columns are the Michigan test (overall + individual parts + combinations,
    or with `simplified` just overall + the listen+grammar and vocab+reading
    halves), plus comprehension accuracy when `comprehension` is set; rows are
    feature sets. `cv` is a MICHIGAN_PARTS_CV key selecting the fold method.
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR
    if feature_sets is None:
        feature_sets = MICHIGAN_PARTS_FEATURE_SETS
    log_label, _, scopes, method_to_scope = MICHIGAN_PARTS_CV[cv]

    target_cols = MICHIGAN_PARTS_SIMPLIFIED_TARGET_COLS if simplified else MICHIGAN_PARTS_TARGET_COLS
    target_col_display = MICHIGAN_PARTS_DISPLAY
    if comprehension:
        target_cols = target_cols + ["comprehension_score-regular_trials"]
        target_col_display = {**MICHIGAN_PARTS_DISPLAY,
                              "comprehension_score-regular_trials": "Comprehension"}
    stem = (f"{'comprehension_' if comprehension else ''}michigan_parts"
            f"{'_simplified' if simplified else ''}_{cv}")

    df = pd.read_csv(eval_csv)
    df = df[
        (df["dataset"] == "OneStop")
        & (df["feature_set"].isin(feature_sets))
        & (df["method"].isin(list(method_to_scope)))
        & (df["version_path"] == "fully_agg")
        & (df["target_col"].isin(target_cols))
    ]
    if model_name is not None:
        df = df[df["model_name"] == model_name]

    if df.empty:
        logger.warning("No %s%s %s data for model=%s",
                       "comprehension+michigan parts" if comprehension else "Michigan parts",
                       " simplified" if simplified else "", log_label, model_name)
        return

    for (aug_p, preview, model), model_df in df.groupby(["aug_p", "preview", "model_name"]):
        latex = generate_latex_table(
            model_df,
            target_cols=target_cols,
            scopes=scopes,
            method_to_scope=method_to_scope,
            target_col_display=target_col_display,
            dataset="OneStop",
            version_path="fully_agg",
            model_name=model,
            feature_sets_ordered=MICHIGAN_PARTS_FEATURE_SETS,
            feature_set_group_index=MICHIGAN_PARTS_GROUP_INDEX,
        )
        if not latex:
            continue

        table_dir = save_dir / "OneStop" / aug_p / preview / "fully_agg" / model
        table_dir.mkdir(parents=True, exist_ok=True)
        table_path = table_dir / f"{stem}.tex"
        write_table(table_path, latex)
        logger.info("Saved table: %s", table_path)


def _check_model_has_results(df: pd.DataFrame, model_name: str, version_path: str = None) -> bool:
    """Check if a model has evaluation results in the DataFrame.

    Args:
        df: Evaluation results DataFrame
        model_name: Model name to check
        version_path: Optional version path filter (if None, checks all versions)

    Returns:
        True if model has results, False otherwise
    """
    model_df = df[df["model_name"] == model_name]
    if version_path:
        model_df = model_df[model_df["version_path"] == version_path]
    return not model_df.empty


def generate_all_tables_for_all_models(
    eval_csv: Path = None,
    save_dir: Path = None,
    model_names: Optional[List[str]] = None,
    feature_sets: Optional[List[str]] = None,
    version_path_filter: Optional[str] = None,
):
    """Generate tables for all specified models, with validation.

    Args:
        eval_csv: Path to evaluation CSV
        save_dir: Directory to save tables
        model_names: List of model names to generate tables for. If None, uses all ModelNames.
        feature_sets: Feature sets to include. If None, uses all.
        version_path_filter: Optional version path to filter (e.g., "fully_agg")
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR
    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_ORDERED

    df = pd.read_csv(eval_csv)
    if model_names is None:
        # Default to every model actually present in the evaluation CSV (the
        # ModelNames registry plus any extra model evaluated from disk), so
        # newly added models get their per-model table folders automatically.
        known = [name.value for name in ModelNames]
        extra = sorted(set(df["model_name"].unique()) - set(known))
        model_names = known + extra

    for model_name in model_names:
        # Check if model has results
        if not _check_model_has_results(df, model_name, version_path_filter):
            logger.warning(
                "No results found for model=%s%s. Skipping table generation.",
                model_name,
                f" with version_path={version_path_filter}" if version_path_filter else ""
            )
            continue

        # Filter by version_path if specified
        model_df = df[df["model_name"] == model_name]
        if version_path_filter:
            model_df = model_df[model_df["version_path"] == version_path_filter]

        # Check that we have all target columns and methods for this model
        required_targets = set()
        for dataset in DATASET_TARGET_COLS.keys():
            required_targets.update(DATASET_TARGET_COLS[dataset])

        model_targets = set(model_df["target_col"].unique())
        missing_targets = required_targets - model_targets

        if missing_targets:
            logger.warning(
                "Model=%s is missing target columns: %s. Table generation may be incomplete.",
                model_name, missing_targets
            )

        logger.info("Generating tables for model=%s...", model_name)
        generate_all_tables(
            eval_csv=eval_csv,
            save_dir=save_dir,
            model_name=model_name,
            feature_sets=feature_sets,
        )


# ── Combined OneStop × Meco tables ───────────────────────────────────────────


def generate_latex_combined_table(
    eval_dfs: dict,  # {dataset: filtered_eval_df}
    target_cols: dict,  # {dataset: [target_cols]}
    scopes: List[str],
    method_to_scope: dict,
    target_col_display: dict,  # {dataset: {target_col: display_name}}
    dataset_display: dict,  # {dataset: display_name}
    version_path: str = "",
    model_name: str = "",
    show_scope_row: bool = True,
    scope_label: str = "",
    multi_version_pairs: Optional[dict] = None,
    paper_style: bool = False,
    caption_override: Optional[str] = None,
    label: Optional[str] = None,
    decimals: int = 2,
) -> str:
    """Generate a LaTeX table combining multiple datasets side-by-side.

    Columns structure: Dataset (OneStop | Meco) → Target columns → Scopes → Metrics
    Rows: Feature sets

    Args:
        show_scope_row: If False, hide the scope row and add scope_label to caption instead
        scope_label: Text to add to caption when show_scope_row=False (e.g., "LOLO", "L1-Stratified")
    """
    feature_sets = []
    for dataset, df in eval_dfs.items():
        # A dataset whose result CSVs are all missing upstream arrives as an
        # empty frame with no columns at all -- skip it rather than raising a
        # KeyError on the column that was never created.
        if df.empty or "feature_set" not in df.columns:
            continue
        feature_sets.extend(df["feature_set"].unique())
    feature_sets = list(set(feature_sets))
    feature_sets = [fs for fs in ALL_FEATURE_SETS_ORDERED if fs in feature_sets]

    if not feature_sets:
        return ""

    metrics = ["pearson_r", "mae"]
    metric_labels = {"pearson_r": "$r$", "mae": "MAE"}

    # Build lookups per dataset: (feature_set, target_col, scope) -> {pearson_r, mae}
    lookups = {}
    for dataset, df in eval_dfs.items():
        lookup = {}
        for _, row in df.iterrows():
            scope = method_to_scope.get(row["method"])
            if scope is None:
                continue
            key = (row["feature_set"], row["target_col"], scope)
            lookup[key] = {m: row[m] for m in metrics}
        lookups[dataset] = lookup

    # Force fixed-text-only feature sets to NA in non-Fixed scopes
    for dataset in eval_dfs.keys():
        for fs in feature_sets:
            if fs in FIXED_TEXT_ONLY_FEATURE_SETS:
                for tc in target_cols.get(dataset, []):
                    for scope in scopes:
                        if scope != "Fixed":
                            lookups[dataset][(fs, tc, scope)] = {"pearson_r": np.nan, "mae": np.nan}

    # Find best values per (dataset, target_col, scope) across feature sets.
    # Pair cells (n1/n6) contribute both halves to the pool so the bolding
    # logic can highlight whichever side of the pair beats the scalar
    # competition.
    best = {}
    for dataset in eval_dfs.keys():
        for tc in target_cols.get(dataset, []):
            for scope in scopes:
                r_vals, mae_vals = [], []
                for fs in feature_sets:
                    pair_r = multi_version_pairs.get((dataset, fs, tc, scope, "pearson_r")) if multi_version_pairs else None
                    pair_mae = multi_version_pairs.get((dataset, fs, tc, scope, "mae")) if multi_version_pairs else None
                    if pair_r is not None:
                        for v in pair_r:
                            if not pd.isna(v):
                                r_vals.append(v)
                    if pair_mae is not None:
                        for v in pair_mae:
                            if not pd.isna(v):
                                mae_vals.append(v)
                    if pair_r is None and pair_mae is None:
                        vals = lookups[dataset].get((fs, tc, scope))
                        if vals:
                            if not pd.isna(vals["pearson_r"]):
                                r_vals.append(vals["pearson_r"])
                            if not pd.isna(vals["mae"]):
                                mae_vals.append(vals["mae"])
                best[(dataset, tc, scope, "pearson_r")] = max(r_vals) if r_vals else None
                best[(dataset, tc, scope, "mae")] = min(mae_vals) if mae_vals else None

    n_scopes = len(scopes)
    n_metrics = len(metrics)
    datasets = list(eval_dfs.keys())

    n_data_cols = sum(len(target_cols.get(d, [])) * n_scopes * n_metrics for d in datasets)
    n_total_cols = 1 + n_data_cols

    if paper_style:
        # Template format: booktabs rules, one `cc` (c * n_metrics) per
        # (target, scope) pair, `|` between every such pair; dataset/target
        # boundaries are marked via the c|/c alignment on multicolumn headers,
        # not via `||` in the tabular spec.
        target_scope_blocks = []
        for ds in datasets:
            for _tc in target_cols.get(ds, []):
                for _ in scopes:
                    target_scope_blocks.append("c" * n_metrics)
        col_spec = "l" + "|".join(target_scope_blocks)

        # Single-dataset paper tables (e.g. OneStop/Hunting) are narrow enough
        # for one column; multi-dataset tables span both columns.
        use_full_width = len(datasets) > 1
        table_env = "table*" if use_full_width else "table"
        resize_width = "\\textwidth" if use_full_width else "\\columnwidth"

        lines = [
            f"\\begin{{{table_env}}}[ht!]",
            "\\centering",
            "% \\small",
            f"\\resizebox{{{resize_width}}}{{!}}{{",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
        ]

        hdr_empty = "\\multicolumn{1}{l}{}"

        # Header 1: datasets. c| on every dataset except the last → boundary rule.
        h1 = [hdr_empty]
        for i, ds in enumerate(datasets):
            span = len(target_cols.get(ds, [])) * n_scopes * n_metrics
            is_last_ds = (i == len(datasets) - 1)
            align = "c" if is_last_ds else "c|"
            display = dataset_display.get(ds, ds)
            h1.append(f"\\multicolumn{{{span}}}{{{align}}}{{\\textbf{{{display}}}}}")
        lines.append(" & ".join(h1) + " \\\\")

        # Header 2: target labels. | only on the last target of a non-last
        # dataset (dataset boundary).
        h2 = [hdr_empty]
        for i, ds in enumerate(datasets):
            tcs = target_cols.get(ds, [])
            is_last_ds = (i == len(datasets) - 1)
            for j, tc in enumerate(tcs):
                is_last_tc = (j == len(tcs) - 1)
                align = "c|" if (is_last_tc and not is_last_ds) else "c"
                label_text = _display(tc, target_col_display.get(ds, {}))
                h2.append(
                    f"\\multicolumn{{{n_scopes * n_metrics}}}{{{align}}}{{{label_text}}}"
                )
        lines.append(" & ".join(h2) + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

        # Header 3: scope labels. | on the last scope of every (dataset, target)
        # except the very last one (target-pair boundary, including the dataset
        # boundary).
        h3 = [hdr_empty]
        total_targets = sum(len(target_cols.get(d, [])) for d in datasets)
        target_index = 0
        for ds in datasets:
            tcs = target_cols.get(ds, [])
            for _tc in tcs:
                is_last_target = (target_index == total_targets - 1)
                for k, scope in enumerate(scopes):
                    is_last_scope = (k == n_scopes - 1)
                    add_bar = is_last_scope and not is_last_target
                    align = "c|" if add_bar else "c"
                    h3.append(f"\\multicolumn{{{n_metrics}}}{{{align}}}{{{scope}}}")
                target_index += 1
        if show_scope_row:
            lines.append(" & ".join(h3) + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

        # Header 4: Features label + metric labels (plain cells — col_spec |s show).
        h4 = ["\\textbf{Features}"]
        for ds in datasets:
            for _tc in target_cols.get(ds, []):
                for _scope in scopes:
                    for m in metrics:
                        h4.append(metric_labels[m])
        lines.append(" & ".join(h4) + " \\\\")
        lines.append(f"\\cmidrule{{1-{n_total_cols}}}")

        for i, fs in enumerate(feature_sets):
            if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[feature_sets[i - 1]]:
                lines.append("\\midrule")
            row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
            for ds in datasets:
                for tc in target_cols.get(ds, []):
                    for scope in scopes:
                        vals = lookups[ds].get((fs, tc, scope))
                        for m in metrics:
                            if vals and not pd.isna(vals[m]):
                                best_val = best.get((ds, tc, scope, m))
                                is_best = (best_val is not None and
                                           np.isclose(vals[m], best_val, atol=1e-6))
                                row_parts.append(_fmt_val(vals[m], is_best, decimals))
                            else:
                                row_parts.append("-")
            lines.append(" & ".join(row_parts) + " \\\\")
        lines.append("\\bottomrule")

        lines.append("\\end{tabular}")
        lines.append("}")

        if caption_override is not None:
            lines.append(f"\\caption{{{caption_override}}}")
        else:
            caption_parts = []
            if scope_label:
                caption_parts.append(scope_label)
            if model_name:
                caption_parts.append(MODEL_DISPLAY.get(model_name, model_name))
            if caption_parts:
                lines.append(f"\\caption{{{' -- '.join(caption_parts)}}}")
        if label is not None:
            lines.append(f"\\label{{{label}}}")
        lines.append(f"\\end{{{table_env}}}")
        return "\n".join(lines)

    # ── Non-paper (standard) style ──────────────────────────────────────────
    hdr_empty = ""
    pairs_for_cells = multi_version_pairs

    lines = ["\\begin{table*}[ht!]", "\\centering", "\\small"]

    col_groups = []
    for dataset in datasets:
        dataset_targets = target_cols.get(dataset, [])
        scope_block = "|".join(["c"] * (len(dataset_targets) * n_scopes * n_metrics))
        col_groups.append(scope_block)
    col_spec = "l||" + "||".join(col_groups)

    if len(eval_dfs) * max(len(target_cols.get(d, [])) for d in datasets) * n_scopes > 4:
        lines.append("\\resizebox{\\textwidth}{!}{")

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    header1_parts = [hdr_empty]
    for dataset in datasets:
        dataset_targets = target_cols.get(dataset, [])
        span = len(dataset_targets) * n_scopes * n_metrics
        display = dataset_display.get(dataset, dataset)
        header1_parts.append(f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{display}}}}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    header2_parts = [hdr_empty]
    for dataset in datasets:
        for tc in target_cols.get(dataset, []):
            span = n_scopes * n_metrics
            display = _display(tc, target_col_display.get(dataset, {}))
            header2_parts.append(f"\\multicolumn{{{span}}}{{c}}{{{display}}}")
    lines.append(" & ".join(header2_parts) + " \\\\")

    if show_scope_row:
        header3_parts = [hdr_empty]
        for dataset in datasets:
            for _tc in target_cols.get(dataset, []):
                for scope in scopes:
                    header3_parts.append(f"\\multicolumn{{{n_metrics}}}{{c}}{{{scope}}}")
        lines.append(" & ".join(header3_parts) + " \\\\")

    header4_parts = ["\\textbf{Features}"]
    for dataset in datasets:
        for _tc in target_cols.get(dataset, []):
            for _scope in scopes:
                for m in metrics:
                    header4_parts.append(metric_labels[m])
    lines.append(" & ".join(header4_parts) + " \\\\")
    lines.append("\\hline")

    for i, fs in enumerate(feature_sets):
        if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[feature_sets[i - 1]]:
            lines.append("\\hline")
        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for dataset in datasets:
            for tc in target_cols.get(dataset, []):
                for scope in scopes:
                    vals = lookups[dataset].get((fs, tc, scope))
                    for m in metrics:
                        pair = None
                        if pairs_for_cells is not None:
                            pair = pairs_for_cells.get((dataset, fs, tc, scope, m))
                        if pair is not None:
                            n1, n6 = pair
                            best_val = best.get((dataset, tc, scope, m))
                            def _fmt(v):
                                if pd.isna(v):
                                    return "NA"
                                is_best = (best_val is not None and
                                           np.isclose(v, best_val, atol=1e-6))
                                return _fmt_val(v, is_best, decimals)
                            row_parts.append(f"{_fmt(n1)}/{_fmt(n6)}")
                        elif vals and not pd.isna(vals[m]):
                            best_val = best.get((dataset, tc, scope, m))
                            is_best = (best_val is not None and
                                       np.isclose(vals[m], best_val, atol=1e-6))
                            row_parts.append(_fmt_val(vals[m], is_best, decimals))
                        else:
                            row_parts.append("NA")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    if len(eval_dfs) * max(len(target_cols.get(d, [])) for d in datasets) * n_scopes > 4:
        lines.append("}")

    if caption_override is not None:
        lines.append(f"\\caption{{{caption_override}}}")
    else:
        caption_parts = []
        if scope_label:
            caption_parts.append(scope_label)
        if model_name:
            caption_parts.append(MODEL_DISPLAY.get(model_name, model_name))
        if caption_parts:
            lines.append(f"\\caption{{{' -- '.join(caption_parts)}}}")
    if label is not None:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# ── Bootstrap combined table (mean ± std, bootstrap stars vs WPM) ────────────

BOOTSTRAP_COMBINED_OUTPUT = "combined_bootstrap.tex"
# Spearman variants of the two tables above (identical layout, rank correlation).
BOOTSTRAP_COMBINED_SPEARMAN_OUTPUT = "combined_bootstrap_spearman.tex"

# Baseline feature set every other row is tested against (WPM).
BOOTSTRAP_BASELINE_FS = "READING_SPEED"
# Baseline model every other model's table is compared against, cell by cell.
# Non-Ridge models' cells carry the difference from the same Ridge cell as a
# subscript; Ridge's own tables carry no subscript. Same value as
# evaluation.PAIRWISE_BASELINE_MODEL, which produces the comparison CSV.
BOOTSTRAP_BASELINE_MODEL = "Ridge_Classifier"
PAIRWISE_RIDGE_CSV = "pairwise_significance_ridge.csv"

# Horizontal gap between the r and the MAE column of the same (dataset, target,
# regime) block. It replaces both \tabcolsep gaps inside the pair, so the two
# numbers read as one block while the default spacing still separates blocks.
METRIC_PAIR_SEP = "4pt"

# Colours for the vs-baseline-model difference subscript: green when this model
# is ahead of the baseline (a "+" difference), red when it is behind ("-").
# The definitions are emitted above the table so each .tex file is
# self-contained; the document only needs \usepackage{xcolor}.
# Dataset names as they appear in the paper tables. OneStop is labelled with its
# cohort because every target in these tables (LexTALE, Michigan) is an L2 test.
BOOTSTRAP_DATASET_DISPLAY = {"OneStop": "OneStopL2", "Meco": "MECO"}


DIFF_COLOR_BETTER = "diffbetter"
DIFF_COLOR_WORSE = "diffworse"
DIFF_COLOR_DEFS = (
    "% Requires \\usepackage{xcolor} in the document preamble.\n"
    f"\\definecolor{{{DIFF_COLOR_BETTER}}}{{rgb}}{{0.00,0.45,0.70}}\n"
    f"\\definecolor{{{DIFF_COLOR_WORSE}}}{{rgb}}{{0.70,0.00,0.00}}"
)

# Dataset -> file-name / label slug for the per-dataset split of the combined
# bootstrap tables.
DATASET_FILE_SLUG = {"Meco": "meco", "OneStop": "onestop"}

# The featureless mean baseline (src/methods/predictions/models/AvgModel.py):
# predicts the training fold's mean target, so it is the floor every row below
# it has to beat. Its model_name in the evaluation CSV, the row label it gets
# in the tables, and the dashed rule that fences it off from the feature-set
# rows -- it is a reference line, not a competitor, so it is kept out of the
# significance markers and out of the per-column bolding.
BASELINE_ROW_MODEL = "Average"
BASELINE_ROW_LABEL = "Avg. Train"
# The rule is emitted under our own name rather than as a bare \hdashline.
# A real dashed rule across a tabular needs arydshln (the tabular's width is
# not available to a \noalign-level macro, so \leaders/\dotfill tricks either
# collapse to zero width or fight the cell's own \hfil). When the document
# loads arydshln we use its \hdashline, at half the default dash pitch --
# \resizebox shrinks the whole table, and at the default 4pt/4pt the dashes
# come out sparse. Otherwise the row is fenced off by a solid \midrule. Either
# way the rule gets \addlinespace on both sides: without it the rule sits hard
# against the two rows it separates and reads as a smudge rather than a
# divider.
BASELINE_ROW_RULE_NAME = "\\meanbaselinerule"


# Vertical breathing room either side of the rule, and the arydshln dash
# pattern. Both are knobs: raise BASELINE_ROW_RULE_PAD to open the gap up,
# raise the dash/gap lengths to make the dashes longer and sparser.
BASELINE_ROW_RULE_PAD = "2pt"
BASELINE_ROW_DASH = "2pt"
BASELINE_ROW_DASH_GAP = "2pt"


def baseline_row_rule_def(n_total_cols: int) -> str:
    r"""Definition of the rule that fences the reference row off from the rest.

    Emitted inside each float so that the .tex stays self-contained when a
    single table is pasted into a document; \providecommand, not \newcommand,
    so several such tables in one document do not clash.

    The dash lengths are set only when arydshln is what defines \hdashline --
    another package could define the macro without them, and \setlength on an
    undefined length is an error, not a no-op.
    """
    del n_total_cols  # unused; kept in the signature for callers
    pad = "\\addlinespace[" + BASELINE_ROW_RULE_PAD + "]"
    return (
        "\\makeatletter\n"
        "\\@ifundefined{hdashline}\n"
        "  {\\providecommand{" + BASELINE_ROW_RULE_NAME + "}{"
        + pad + "\\midrule" + pad + "}}\n"
        "  {\\@ifundefined{dashlinedash}{}{\\setlength\\dashlinedash{"
        + BASELINE_ROW_DASH + "}\\setlength\\dashlinegap{"
        + BASELINE_ROW_DASH_GAP + "}}%\n"
        "   \\providecommand{" + BASELINE_ROW_RULE_NAME + "}{"
        + pad + "\\hdashline" + pad + "}}\n"
        "\\makeatother")


BASELINE_ROW_LEGEND = (
    "The first row, \\emph{" + BASELINE_ROW_LABEL + "}, is a featureless "
    "baseline that predicts the mean target of the training fold under the "
    "same cross-validation as every other row. It depends only on the cohort, "
    "so it is identical for every model and is shown for reference only: it "
    "takes no part in the significance tests and is excluded from the "
    "bolding. Its prediction is constant within a fold, so a correlation is "
    "not defined for it -- its left-hand cell reports $R^2$ $\\uparrow$ "
    "instead, and its right-hand cell the MAE $\\downarrow$ on the same "
    "scale as the rows below.")

# stars_from -> (pairwise p-value column template, caption fragment describing
# the test); {m} in the template is filled with the table's correlation metric.
#
# The paired bootstrap is the only test behind any reported result. Steiger's
# columns are still written to pairwise_significance.csv as a cross-check (see
# stats_utils) but are deliberately not selectable here, so no table can be
# generated from them by accident.
BOOTSTRAP_STARS_SOURCES = {
    "bootstrap": (
        "boot_diff_{m}_p",
        "a paired-bootstrap test of the correlation difference "
        "(two-sided null-centred $p$ over the same paired resamples)"),
}

# corr_metric -> (eval CSV column prefix, LaTeX display name)
BOOTSTRAP_CORR_METRICS = {
    "pearson": ("pearson", "Pearson $r$"),
    "spearman": ("spearman", "Spearman $\\rho$"),
}


def _p_to_n_stars(p) -> int:
    """Significance level → number of stars (0-3)."""
    if p is None or pd.isna(p):
        return 0
    if p < 0.001:
        return 3
    if p < 0.01:
        return 2
    if p < 0.05:
        return 1
    return 0


# One glyph, one meaning: stars mark the significance level and nothing else.
# The direction is not encoded in the marker -- a cell that is significantly
# worse than the baseline carries the same stars as one that is better, and the
# printed value says which way the difference runs.
SIGNIFICANCE_LEGEND = (
    "Stars show the significance of the difference from the performance of "
    "the WPM baseline for the corresponding regime and proficiency test, in "
    "either direction: no marker $p\\geq 0.05$, $^{*}$ $p<0.05$, "
    "$^{**}$ $p<0.01$, $^{***}$ $p<0.001$."
)

# Comparison tables carry a second annotation the main tables do not: the
# vs-Ridge subscript. Both of its parts need naming, and the significance
# legend has to say which comparison each marker belongs to -- the value's
# markers test against WPM, the subscript's against Ridge.
RIDGE_SUBSCRIPT_LEGEND = (
    "The subscript is this cell's difference from the corresponding Ridge "
    "cell, \\textcolor{" + DIFF_COLOR_BETTER + "}{blue} when this model is "
    "ahead of Ridge and \\textcolor{" + DIFF_COLOR_WORSE + "}{red} when "
    "behind (for MAE lower is better, so a negative difference is an "
    "improvement)."
)
# Value and subscript are marked the same way: stars for the level in both
# directions. On the subscript the printed sign and the colour already say
# which way the difference runs; on the value the printed number does.
COMPARISON_SIGNIFICANCE_LEGEND = (
    "On the value, stars give the significance of the comparison against "
    "the WPM baseline in either direction: no marker $p\\geq 0.05$, "
    "$^{*}$ $p<0.05$, $^{**}$ $p<0.01$, $^{***}$ $p<0.001$. In the subscript "
    "the comparison is against Ridge and the level is marked the same way "
    "($^{*}$ $p<0.05$, $^{**}$ $p<0.01$, $^{***}$ $p<0.001$); the "
    "sign and the colour of the subscript say which way the difference runs."
)


# Both directions use stars: the marker means "significant at this level",
# and the value itself says whether the cell is above or below the baseline.
SIG_GLYPH_BETTER = "*"
SIG_GLYPH_WORSE = "*"


def _directional_stars(p, value_fs, value_base, higher_is_better: bool) -> int:
    """Signed significance level for a two-sided test against the baseline.

    Every p-value in pairwise_significance.csv is two-sided, so a feature set
    that is significantly *worse* than WPM carries the same p as one that is
    significantly better. The magnitude is the significance level (0-3) and the
    **sign** carries the direction: positive when the difference runs the
    favourable way (higher r, lower MAE), negative when it runs against.
    ``_sig_marker`` renders both directions as stars, so a significantly worse
    cell is marked rather than silently left blank next to a merely
    non-significant one. Zero = not significant, or direction undeterminable --
    which includes an exact tie: equal values have no direction, so marking one
    as the worse of the two would be an artefact of the comparison operator.
    """
    if p is None or pd.isna(p) or pd.isna(value_fs) or pd.isna(value_base):
        return 0
    n = _p_to_n_stars(p)
    if not n or value_fs == value_base:
        return 0
    wins = value_fs > value_base if higher_is_better else value_fs < value_base
    return n if wins else -n


def _sig_marker(ns: int) -> str:
    """Superscript for a signed significance level from ``_directional_stars``:
    stars in either direction, nothing when non-significant. Only the level is
    marked; the direction is read off the value."""
    if not ns:
        return ""
    glyph = SIG_GLYPH_BETTER if ns > 0 else SIG_GLYPH_WORSE
    return "^{" + glyph * abs(ns) + "}"


def generate_bootstrap_combined_latex(
    eval_dfs: dict,  # {dataset: filtered_eval_df} — the fully_agg group slice
    target_cols: dict,  # {dataset: [target_cols]}
    scopes: List[str],
    pairwise_df: pd.DataFrame,
    model_name: str,
    aug_p: str,
    decimals: Optional[int] = None,
    stars_from: str = "bootstrap",
    corr_metric: str = "pearson",
    ridge_df: Optional[pd.DataFrame] = None,
    show_dispersion: bool = True,
    include_model_in_title: bool = True,
    baseline_mae_note: str = "",
    caption_override: str = "",
    single_column: bool = False,
    baseline_df: Optional[pd.DataFrame] = None,
) -> str:
    """Combined OneStop × MECO prediction table with bootstrap point estimates.

    Same layout as ``combined.tex`` (rows = feature sets, cols = dataset ×
    target × Fixed/Any regime × {r, MAE}), but each correlation cell shows
    the bootstrap **mean** ± one bootstrap resample **std**
    (``<metric>_r_boot`` / ``<metric>_r_boot_std``; a normal-approx 95% CI is
    the 2.5/97.5 percentile interval), annotated with significance stars for
    the paired-bootstrap test of that feature set's correlation against WPM
    (from pairwise_significance.csv): * p<.05, ** p<.01, *** p<.001; no
    stars = not significant. The WPM baseline row carries no stars. MAE is
    the bootstrap mean ± std on the same resamples, annotated with stars
    for the paired-bootstrap test of the MAE difference vs WPM
    (``boot_diff_mae_p``). Best mean r / lowest MAE per column is bolded.
    Stars are shown only where the feature set beats WPM (higher r, lower MAE).

    ``stars_from`` is retained for signature stability but has one valid value,
    "bootstrap" — the paired-bootstrap difference test is the only significance
    test behind any reported result. ``corr_metric`` picks the correlation
    ("pearson" or "spearman") shown in the r cells and used for the stars.

    ``ridge_df`` (pairwise_significance_ridge.csv) adds, for every model other
    than the Ridge baseline, that cell's difference from the same Ridge cell as
    a subscript: ``0.340^{***}_{(+0.020^{*})}`` — Ridge's value subtracted from
    this model's, signed so ``+`` means this model is ahead of Ridge on the
    metric shown, with stars for the paired-bootstrap test of that difference.
    Unlike the stars against WPM, these are not direction-suppressed: the sign
    already says which way the difference runs, so a significantly *worse*
    cell is marked too. The subscript is coloured green when this model is
    ahead of the baseline and red when it is behind, so the .tex needs
    ``xcolor`` (the two colours are defined at the top of the emitted file).
    Omitted when the CSV is missing.

    ``baseline_df`` is the evaluation slice for the featureless mean baseline
    (``model_name == "Average"``, any feature set), covering the same datasets,
    previews, targets and methods as ``eval_dfs``. When given, its cells are
    drawn as a single first row above the feature sets, fenced off by a dashed
    rule: $R^2$ in the correlation column (the baseline's prediction is
    constant within a fold, so a correlation is degenerate) and MAE in the MAE
    column. The row carries no stars, no vs-Ridge subscript and no bolding --
    it is a reference line, identical for every model, and deliberately outside
    every comparison in the table. It also supersedes ``baseline_mae_note``,
    which reports the same numbers in prose.

    Passing a single-dataset ``eval_dfs`` renders that dataset alone; the
    dataset then also appears in the caption and is appended to the
    ``\\label`` (``..._meco`` / ``..._onestop``).

    ``show_dispersion=False`` drops the ``± std`` from every r and MAE cell —
    same bootstrap means, same stars, same vs-Ridge subscripts, same bolding,
    just the point estimate — and labels the table ``..._noci`` so it can sit
    in the same document as the dispersion version.

    Every cell is reported to two decimals; pass ``decimals`` to override.
    """
    if decimals is None:
        decimals = 2
    if stars_from not in BOOTSTRAP_STARS_SOURCES:
        raise ValueError(f"stars_from must be one of {list(BOOTSTRAP_STARS_SOURCES)}")
    if corr_metric not in BOOTSTRAP_CORR_METRICS:
        raise ValueError(f"corr_metric must be one of {list(BOOTSTRAP_CORR_METRICS)}")
    star_p_col, star_test_desc = BOOTSTRAP_STARS_SOURCES[stars_from]
    star_p_col = star_p_col.format(m=corr_metric)
    metric_col, metric_display = BOOTSTRAP_CORR_METRICS[corr_metric]

    datasets = [d for d in ["Meco", "OneStop"] if d in eval_dfs]
    # One dataset is enough: nothing below assumes a pair (the header spans,
    # the pairwise preview filter and the bolding all iterate `datasets`), and
    # the single-dataset form is what the OneStop/Hunting tables need.
    if len(datasets) < 1:
        return ""

    sample = next(iter(eval_dfs.values()))
    if f"{metric_col}_r_boot" not in sample.columns:
        logger.warning(
            "No bootstrap columns in evaluation results — rerun evaluation.py "
            "to add them; skipping bootstrap combined table")
        return ""

    # (fs, ds, tc, scope) -> (r_boot_mean, r_boot_std, mae_boot_mean, mae_boot_std)
    values = {}
    for ds in datasets:
        method_to_scope = DATASET_METHOD_TO_SCOPE[ds]
        for _, row in eval_dfs[ds].iterrows():
            scope = method_to_scope.get(row["method"])
            if scope is None or scope not in scopes:
                continue
            values[(row["feature_set"], ds, row["target_col"], scope)] = (
                row[f"{metric_col}_r_boot"], row[f"{metric_col}_r_boot_std"],
                row.get("mae_boot", row["mae"]), row.get("mae_boot_std", np.nan))
    if not values:
        logger.warning("No data for bootstrap combined table; skipping")
        return ""

    # (fs, ds, tc, scope) -> n_stars from the test of fs vs WPM (correlation
    # test picked by stars_from; MAE always the paired-bootstrap difference).
    # pairwise_significance.csv stores each pair once, in either a/b order.
    stars, mae_stars = {}, {}
    if pairwise_df is not None and not pairwise_df.empty:
        for ds in datasets:
            method_to_scope = DATASET_METHOD_TO_SCOPE[ds]
            # Match each dataset's actual preview(s) from its eval slice: in
            # pair_across_preview mode the group preview is the synthetic
            # "combined_preview" while the rows are pinned per dataset
            # (Meco: all, OneStop: gathering).
            ds_previews = eval_dfs[ds]["preview"].unique()
            sub = pairwise_df[
                (pairwise_df["dataset"] == ds)
                & (pairwise_df["aug_p"] == aug_p)
                & (pairwise_df["preview"].isin(ds_previews))
                & (pairwise_df["version_path"] == "fully_agg")
                & (pairwise_df["target_col"].isin(target_cols.get(ds, [])))
                & (pairwise_df["model_name_a"] == model_name)
                & (pairwise_df["model_name_b"] == model_name)
                & ((pairwise_df["feature_set_a"] == BOOTSTRAP_BASELINE_FS)
                   | (pairwise_df["feature_set_b"] == BOOTSTRAP_BASELINE_FS))]
            for _, row in sub.iterrows():
                scope = method_to_scope.get(row["method"])
                if scope is None or scope not in scopes:
                    continue
                # pairwise rows store the pair in either order, so orient each
                # metric onto (feature set, baseline) before judging direction.
                a_is_baseline = row["feature_set_a"] == BOOTSTRAP_BASELINE_FS
                fs = (row["feature_set_b"] if a_is_baseline
                      else row["feature_set_a"])

                def _oriented(col_a: str, col_b: str):
                    return ((row.get(col_b), row.get(col_a)) if a_is_baseline
                            else (row.get(col_a), row.get(col_b)))

                r_fs, r_base = _oriented(f"{metric_col}_r_a", f"{metric_col}_r_b")
                mae_fs, mae_base = _oriented("mae_a", "mae_b")

                key = (fs, ds, row["target_col"], scope)
                stars[key] = _directional_stars(
                    row.get(star_p_col), r_fs, r_base, higher_is_better=True)
                mae_stars[key] = _directional_stars(
                    row.get("boot_diff_mae_p"), mae_fs, mae_base,
                    higher_is_better=False)
    else:
        logger.warning("Pairwise significance data missing — stars vs WPM omitted")

    # (fs, ds, tc, scope) -> (advantage over Ridge, n_stars) for the same cell
    # of the Ridge table, for the correlation and for MAE. The stored value is
    # this model's *advantage*: r_model - r_ridge for the correlation, and
    # mae_ridge - mae_model for MAE (sign flipped, since lower MAE is better),
    # so a positive subscript always means "better than Ridge".
    ridge_diffs, ridge_mae_diffs = {}, {}
    show_ridge_diff = (model_name != BOOTSTRAP_BASELINE_MODEL
                       and ridge_df is not None and not ridge_df.empty)
    if show_ridge_diff:
        for ds in datasets:
            method_to_scope = DATASET_METHOD_TO_SCOPE[ds]
            ds_previews = eval_dfs[ds]["preview"].unique()
            is_pair = (
                ((ridge_df["model_name_a"] == model_name)
                 & (ridge_df["model_name_b"] == BOOTSTRAP_BASELINE_MODEL))
                | ((ridge_df["model_name_b"] == model_name)
                   & (ridge_df["model_name_a"] == BOOTSTRAP_BASELINE_MODEL)))
            sub = ridge_df[
                (ridge_df["dataset"] == ds)
                & (ridge_df["aug_p"] == aug_p)
                & (ridge_df["preview"].isin(ds_previews))
                & (ridge_df["version_path"] == "fully_agg")
                & (ridge_df["target_col"].isin(target_cols.get(ds, [])))
                & (ridge_df["feature_set_a"] == ridge_df["feature_set_b"])
                & is_pair]
            for _, row in sub.iterrows():
                scope = method_to_scope.get(row["method"])
                if scope is None or scope not in scopes:
                    continue
                # Rows are written with Ridge on side b, but orient anyway so a
                # differently-ordered row (e.g. from the full all-pairs CSV)
                # is not reported with the sign reversed.
                sign = 1.0 if row["model_name_a"] == model_name else -1.0
                key = (row["feature_set_a"], ds, row["target_col"], scope)
                d_r = row.get(f"boot_diff_{corr_metric}")
                if pd.notna(d_r):
                    ridge_diffs[key] = (sign * d_r,
                                        _p_to_n_stars(row.get(star_p_col)))
                d_mae = row.get("boot_diff_mae")
                if pd.notna(d_mae):
                    ridge_mae_diffs[key] = (-sign * d_mae,
                                            _p_to_n_stars(row.get("boot_diff_mae_p")))
        if not ridge_diffs:
            logger.warning(
                "No vs-Ridge comparisons found for model %s — subscripts "
                "omitted; run the evaluation with --pairwise-vs-ridge", model_name)
    elif model_name != BOOTSTRAP_BASELINE_MODEL:
        logger.warning(
            "No vs-Ridge comparison data for model %s — run the evaluation "
            "with --pairwise-vs-ridge to add the difference subscripts", model_name)

    # (ds, tc, scope) -> (R^2, MAE) for the featureless mean baseline. Keyed
    # without a feature set: the baseline ignores the features entirely, so a
    # single row covers the whole table however many feature sets it has.
    baseline_cells = {}
    if baseline_df is not None and not baseline_df.empty:
        for ds in datasets:
            method_to_scope = DATASET_METHOD_TO_SCOPE[ds]
            sub = baseline_df[
                (baseline_df["dataset"] == ds)
                & (baseline_df["target_col"].isin(target_cols.get(ds, [])))]
            for _, row in sub.iterrows():
                scope = method_to_scope.get(row["method"])
                if scope is None or scope not in scopes:
                    continue
                mae = row.get("mae_boot")
                if mae is None or pd.isna(mae):
                    mae = row.get("mae")
                baseline_cells[(ds, row["target_col"], scope)] = (
                    row.get("r2"), mae)
        if not baseline_cells:
            logger.warning(
                "Mean-baseline rows were supplied but none matched this "
                "table's datasets/targets/methods — baseline row omitted")

    # The vs-baseline-model subscript makes every cell a different width, and
    # centring then scatters the numbers across the column. Left-aligning the
    # metric columns lines the values up on their first digit; the phantom
    # star padding that centring needed is dropped with it (a shorter cell
    # just ends earlier). Tables without subscripts keep the centred layout.
    left_align = bool(ridge_diffs or ridge_mae_diffs)

    feature_sets = []
    for ds in datasets:
        # Same guard as generate_latex_combined_table: a dataset with no
        # result CSVs on disk is an empty, column-less frame.
        df_ds = eval_dfs[ds]
        if df_ds.empty or "feature_set" not in df_ds.columns:
            continue
        feature_sets.extend(df_ds["feature_set"].unique())
    fs_present = [fs for fs in ALL_FEATURE_SETS_ORDERED if fs in set(feature_sets)]
    if not fs_present:
        return ""

    # Force fixed-text-only feature sets to NA in non-Fixed scopes.
    for fs in fs_present:
        if fs in FIXED_TEXT_ONLY_FEATURE_SETS:
            for ds in datasets:
                for tc in target_cols.get(ds, []):
                    for scope in scopes:
                        if scope != "Fixed":
                            values[(fs, ds, tc, scope)] = (np.nan, np.nan, np.nan, np.nan)

    columns = [(ds, tc, scope)
               for ds in datasets
               for tc in target_cols.get(ds, [])
               for scope in scopes]

    # Best boot mean (max), best MAE (min) and max star count per column
    # (star deficit is phantom-padded so numbers stay vertically aligned).
    best_r, best_mae, max_stars, max_mae_stars = {}, {}, {}, {}
    for col in columns:
        means = [values[(fs, *col)][0] for fs in fs_present
                 if (fs, *col) in values and not pd.isna(values[(fs, *col)][0])]
        maes = [values[(fs, *col)][2] for fs in fs_present
                if (fs, *col) in values and not pd.isna(values[(fs, *col)][2])]
        best_r[col] = max(means) if means else None
        best_mae[col] = min(maes) if maes else None
        # Magnitude, not the signed value: the sign is not rendered, so a
        # significantly-worse cell needs the same phantom width as a better one.
        max_stars[col] = max(
            (abs(stars.get((fs, *col), 0)) for fs in fs_present), default=0)
        max_mae_stars[col] = max(
            (abs(mae_stars.get((fs, *col), 0)) for fs in fs_present), default=0)

    def _ridge_sub(diffs: dict, fs, col,
                   lower_is_better: bool = False) -> str:
        """LaTeX subscript with this cell's difference from the Ridge cell.

        ``diffs`` stores the *advantage* (positive = this model is better) for
        both metrics. The subscript prints the plain metric difference, so for
        a lower-is-better metric such as MAE the printed sign is flipped: an
        improvement shows as a negative number, still coloured green.

        Significance is marked with stars in *both* directions -- the printed
        sign and the colour already say which way the difference runs, so the
        marker is left to mean only the level. No phantom padding either: the
        columns are left-aligned (see ``col_spec``), so a cell with fewer
        stars simply ends earlier instead of holding open a gap before the
        closing parenthesis.
        """
        entry = diffs.get((fs, *col))
        if entry is None or pd.isna(entry[0]):
            return ""
        diff, ns = entry
        printed = -diff if lower_is_better else diff
        printed = printed if printed != 0 else 0.0  # never print "-0.00"
        shown = f"{printed:+.{decimals}f}"
        # Colour from the sign that is actually printed, not from the raw
        # value: a difference that rounds to "+0.00" (or "-0.00" for MAE)
        # reads as a no-advantage cell.
        ahead = shown.startswith("-") if lower_is_better else not shown.startswith("-")
        color = DIFF_COLOR_BETTER if ahead else DIFF_COLOR_WORSE
        star_str = _sig_marker(ns)
        body = "(" + shown + star_str + ")"
        return "_{{\\color{" + color + "}" + body + "}}"

    # A missing cell is a dash padded to a number's own width, split either
    # side, so under the left-aligned column spec (the one the vs-Ridge
    # subscripts switch the table to) it sits where the value's middle would
    # be rather than against the left edge. The padding is symmetric, so the
    # centred layout is unaffected.
    na_cell = "\\phantom{0.}-\\phantom{" + "0" * decimals + "}"

    def _r_cell(fs, col):
        v = values.get((fs, *col))
        if v is None or pd.isna(v[0]):
            return na_cell
        mean, std = v[0], v[1]
        body = f"{mean:.{decimals}f}"
        if show_dispersion and not pd.isna(std):
            body += f" \\pm {std:.{decimals}f}"
        if best_r[col] is not None and np.isclose(mean, best_r[col], atol=1e-6):
            body = f"\\mathbf{{{body}}}"
        ns = 0 if fs == BOOTSTRAP_BASELINE_FS else stars.get((fs, *col), 0)
        star_str = _sig_marker(ns)
        deficit = max_stars[col] - abs(ns)
        pad = ("\\phantom{^{" + "*" * deficit + "}}"
               if deficit > 0 and not left_align else "")
        sub = _ridge_sub(ridge_diffs, fs, col)
        return f"${body}{star_str}{sub}{pad}$"

    def _mae_cell(fs, col):
        v = values.get((fs, *col))
        if v is None or pd.isna(v[2]):
            return na_cell
        mae, mae_std = v[2], v[3]
        body = f"{mae:.{decimals}f}"
        if show_dispersion and not pd.isna(mae_std):
            body += f" \\pm {mae_std:.{decimals}f}"
        if best_mae[col] is not None and np.isclose(mae, best_mae[col], atol=1e-6):
            body = f"\\mathbf{{{body}}}"
        ns = 0 if fs == BOOTSTRAP_BASELINE_FS else mae_stars.get((fs, *col), 0)
        star_str = _sig_marker(ns)
        deficit = max_mae_stars[col] - abs(ns)
        pad = ("\\phantom{^{" + "*" * deficit + "}}"
               if deficit > 0 and not left_align else "")
        sub = _ridge_sub(ridge_mae_diffs, fs, col, lower_is_better=True)
        return f"${body}{star_str}{sub}{pad}$"

    n_scopes = len(scopes)
    n_metrics = 2  # r, MAE
    a = "l" if left_align else "c"
    col_spec = "l" + "|".join(
        a + "@{\\hspace{" + METRIC_PAIR_SEP + "}}" + a for _ in columns)
    n_total_cols = 1 + len(columns) * n_metrics

    hdr_empty = "\\multicolumn{1}{l}{}"
    dataset_display = BOOTSTRAP_DATASET_DISPLAY
    # A single-dataset table is half as wide, and a plain
    # \resizebox{\textwidth} would blow it *up*; the \ifdim form scales down
    # only, leaving a narrow table at its natural size.
    # single_column renders into one column (table + \\columnwidth) instead of
    # spanning the page, matching the EyeScore renderer's flag of the same name.
    _env = "table" if single_column else "table*"
    if single_column:
        resize = "\\resizebox{\\columnwidth}{!}{"
    else:
        resize = ("\\resizebox{\\textwidth}{!}{" if len(datasets) > 1 else
                  "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth"
                  "\\else\\width\\fi}{!}{")
    lines = []
    if ridge_diffs or ridge_mae_diffs:
        lines.append(DIFF_COLOR_DEFS)
    lines += [
        f"\\begin{{{_env}}}[ht!]",
    ]
    # Inside the float, not above it: these .tex files are pasted into the
    # paper a float at a time, and a definition sitting above \begin{table*}
    # is left behind by that paste -- the macro is then undefined at use.
    # Defining it here makes the float self-contained; the definition is local
    # to the float's group, so several such tables in one document cannot
    # clash.
    if baseline_cells:
        lines.append(baseline_row_rule_def(n_total_cols))
    lines += [
        "\\centering",
        resize,
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header 1: dataset names (c| on all but the last for the boundary rule).
    # Only earns its row when there is more than one dataset to tell apart --
    # on a single-dataset table the name is already in the title, the caption
    # and the \label, and the banner spanning every column just adds a header
    # level to read past.
    if len(datasets) > 1:
        h1 = [hdr_empty]
        for i, ds in enumerate(datasets):
            span = len(target_cols.get(ds, [])) * n_scopes * n_metrics
            align = "c" if i == len(datasets) - 1 else "c|"
            h1.append(
                f"\\multicolumn{{{span}}}{{{align}}}"
                f"{{\\textbf{{{dataset_display[ds]}}}}}")
        lines.append(" & ".join(h1) + " \\\\")

    # Header 2: target labels (| only at the dataset boundary).
    h2 = [hdr_empty]
    for i, ds in enumerate(datasets):
        tcs = target_cols.get(ds, [])
        for j, tc in enumerate(tcs):
            is_boundary = (j == len(tcs) - 1 and i != len(datasets) - 1)
            align = "c|" if is_boundary else "c"
            label_text = _display(tc, DATASET_TARGET_COL_DISPLAY.get(ds, {}))
            h2.append(f"\\multicolumn{{{n_scopes * n_metrics}}}{{{align}}}{{{label_text}}}")
    lines.append(" & ".join(h2) + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

    # Header 3: scope labels. | on the last scope of a non-last dataset, so the
    # dataset-boundary rule runs unbroken from \toprule to \bottomrule instead
    # of stopping at this row and picking up again in the body.
    h3 = [hdr_empty]
    for i, ds in enumerate(datasets):
        tcs = target_cols.get(ds, [])
        is_last_ds = (i == len(datasets) - 1)
        for j, _tc in enumerate(tcs):
            is_last_tc = (j == len(tcs) - 1)
            for k, scope in enumerate(scopes):
                is_last_scope = (k == n_scopes - 1)
                align = ("c|" if (is_last_tc and is_last_scope and not is_last_ds)
                         else "c")
                h3.append(f"\\multicolumn{{{n_metrics}}}{{{align}}}{{{scope}}}")
    lines.append(" & ".join(h3) + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

    # Header 4: Features label + metric labels.
    metric_labels = ["$r$" if corr_metric == "pearson" else "$\\rho$", "MAE"]
    h4 = ["\\textbf{Features}"]
    for _ in columns:
        h4.extend(metric_labels)
    lines.append(" & ".join(h4) + " \\\\")
    lines.append("\\midrule")

    # Reference row, above the dashed rule: same columns, different metric on
    # the left (R^2, not a correlation) and no markers of any kind.
    if baseline_cells:
        brow = [BASELINE_ROW_LABEL]
        for col in columns:
            cell = baseline_cells.get(col)
            # Every cell below carries \phantom{^{***}} padding out to the
            # column's widest marker, so under a centred column spec they all
            # occupy the same width and their digits line up. This row has no
            # markers to pad, and without the same phantom its numbers were
            # centred on a narrower box -- visibly offset from the column.
            pads = [
                "\\phantom{^{" + "*" * max_ns.get(col, 0) + "}}"
                if max_ns.get(col, 0) > 0 and not left_align else ""
                for max_ns in (max_stars, max_mae_stars)]
            if cell is None:
                brow += [na_cell, na_cell]
                continue
            for v, pad in zip(cell, pads):
                if v is None or pd.isna(v):
                    brow.append(na_cell)
                    continue
                # An R^2 a hair below zero is the expected value for this
                # baseline; printing it as "-0.00" reads as a typo.
                v = 0.0 if round(float(v), decimals) == 0 else float(v)
                brow.append(f"${v:.{decimals}f}{pad}$")
        lines.append(" & ".join(brow) + " \\\\")
        lines.append(BASELINE_ROW_RULE_NAME)

    for i, fs in enumerate(fs_present):
        if (i > 0 and FEATURE_SET_GROUP_INDEX.get(fs)
                != FEATURE_SET_GROUP_INDEX.get(fs_present[i - 1])):
            lines.append("\\midrule")
        row = [_display(fs, FEATURE_SET_DISPLAY)]
        for col in columns:
            row.append(_r_cell(fs, col))
            row.append(_mae_cell(fs, col))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")

    lines.append("\\end{tabular}")
    lines.append("}")
    # Resample count actually behind the numbers, not the current constant:
    # a table regenerated from evaluation CSVs written at an earlier
    # N_BOOTSTRAP would otherwise claim the new count in its caption.
    n_resamples = N_BOOTSTRAP
    n_boot_seen = pd.concat([df["n_boot"] for df in eval_dfs.values()
                             if "n_boot" in df.columns], ignore_index=True) \
        if any("n_boot" in df.columns for df in eval_dfs.values()) else pd.Series(dtype=float)
    n_boot_seen = n_boot_seen.dropna()
    if not n_boot_seen.empty:
        n_resamples = int(n_boot_seen.mode().iloc[0])
        if n_resamples != N_BOOTSTRAP:
            logger.warning(
                "Evaluation CSVs were bootstrapped at %d resamples, not the "
                "current N_BOOTSTRAP=%d — caption reports the former",
                n_resamples, N_BOOTSTRAP)

    ridge_caption = ""
    if ridge_diffs or ridge_mae_diffs:
        ridge_caption = (
            "The subscript in parentheses is this cell's difference from the "
            "corresponding cell of the "
            f"{_caption_model_phrase(BOOTSTRAP_BASELINE_MODEL)} table "
            "(same feature set, target, dataset and regime), so that this "
            "model being better than the baseline model shows as a positive "
            "value for the correlation and a negative value for MAE; its "
            "stars are the "
            "paired-bootstrap test of that difference, shown as stars in "
            "both directions; it is "
            "coloured \\textcolor{" + DIFF_COLOR_BETTER + "}{blue} when this "
            "model is ahead of the baseline model and \\textcolor{"
            + DIFF_COLOR_WORSE + "}{red} when it is behind. ")
    if show_dispersion:
        metrics_caption = (
            f"Bootstrap mean {metric_display} $\\uparrow$ "
            "$\\pm$ one bootstrap standard deviation "
            f"(over {n_resamples:,} paired resamples; a normal-approximation "
            "95\\% confidence interval is the mean $\\pm 1.96$ standard "
            "deviations) and bootstrap mean Absolute Error (MAE) "
            "$\\downarrow$ $\\pm$ one bootstrap standard deviation for "
            "prediction of ")
    else:
        metrics_caption = (
            f"Bootstrap mean {metric_display} $\\uparrow$ over "
            f"{n_resamples:,} paired resamples and bootstrap mean "
            "Absolute Error (MAE) $\\downarrow$ for prediction of ")
    model_display = MODEL_DISPLAY.get(model_name, model_name.replace("_", " "))
    title_model = f" with {model_display}" if include_model_in_title else ""
    title_scope = (f" on {dataset_display[datasets[0]]}"
                   if len(datasets) == 1 else "")
    # "dataset" only earns a mention when more than one is in the table.
    ds_phrase = ", dataset" if len(datasets) > 1 else ""
    # The row and the prose sentence report the same numbers; when the row is
    # drawn the sentence is dropped rather than repeated.
    if baseline_cells:
        note = " " + BASELINE_ROW_LEGEND
    else:
        note = f" {baseline_mae_note.strip()}" if baseline_mae_note else ""
    if caption_override:
        lines.append("\\caption{" + caption_override + note + "}")
    else:
        lines.append(
        "\\caption{\\textbf{Prediction of standard proficiency test scores"
        f"{title_model}{title_scope}, bootstrap estimates}}. "
        f"{metrics_caption}"
        "external proficiency scores using eye movements with "
        f"{_caption_model_phrase(model_name)}. "
        f"{SIGNIFICANCE_LEGEND} "
        f"{ridge_caption}"
        "``Fixed'' is the Fixed Text "
        "regime in which all the eye movement data is for the same texts "
        "presented to the test participant. ``Any'' is the Any Text regime "
        "in which no eye movement data is available for the texts of the "
        "test participant. The best result for each evaluation measure in "
        f"each regime{ds_phrase} and target proficiency test is marked in "
        f"bold.{note}}}"
    )
    label_suffix = ""
    if corr_metric != "pearson":
        label_suffix += f"_{corr_metric}"
    if not show_dispersion:
        label_suffix += "_noci"
    if len(datasets) == 1:
        ds_only = datasets[0]
        label_suffix += f"_{DATASET_FILE_SLUG.get(ds_only, ds_only.lower())}"
    lines.append(f"\\label{{tab:predictions_bootstrap{label_suffix}}}")
    lines.append(f"\\end{{{_env}}}")
    return "\n".join(lines)


def generate_full_combined_tables(
    eval_csv: Path = None,
    save_dir: Path = None,
    model_name: str = None,
    feature_sets: Optional[List[str]] = None,
    pair_across_preview: bool = False,
):
    """Generate combined OneStop × Meco tables showing both datasets side-by-side.

    If `pair_across_preview` is True, the `preview` column is dropped from the
    grouping so a dataset whose results only exist under one preview (e.g. Meco
    under `all`) can be paired with another dataset whose results only exist
    under a different preview (e.g. OneStop under `gathering`).
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "full"
    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_ORDERED

    df_full = pd.read_csv(eval_csv)
    df = df_full[df_full["feature_set"].isin(feature_sets)]
    if model_name is not None:
        df = df[df["model_name"] == model_name]

    if df.empty:
        logger.warning("No data found for combined tables with model=%s", model_name)
        return

    if pair_across_preview:
        group_cols = ["aug_p", "version_path", "model_name"]
    else:
        group_cols = ["aug_p", "preview", "version_path", "model_name"]

    # Pre-compute Meco × {Transitions, WFC} × Fixed values across the
    # n_paragraphs_geq_1 and n_paragraphs_geq_6 version_paths so those cells
    # can be rendered as "n1/n6" pairs instead of a single value.
    def _build_multi_version_pairs(model_df: pd.DataFrame) -> dict:
        pairs: dict = {}
        for fs in FIXED_TEXT_ONLY_FEATURE_SETS:
            for tc in DATASET_TARGET_COLS.get("Meco", []):
                vals_by_key = {}
                for key, vp in [("n1", "fully_agg/n_paragraphs_geq_1"),
                                ("n6", "fully_agg/n_paragraphs_geq_6")]:
                    rows = model_df[(model_df["dataset"] == "Meco") &
                                    (model_df["feature_set"] == fs) &
                                    (model_df["target_col"] == tc) &
                                    (model_df["version_path"] == vp) &
                                    (model_df["method"] == "seen__pool_all")]
                    if not rows.empty:
                        row = rows.iloc[0]
                        vals_by_key[key] = (row["pearson_r"], row["mae"])
                    else:
                        vals_by_key[key] = (np.nan, np.nan)
                pairs[("Meco", fs, tc, "Fixed", "pearson_r")] = (
                    vals_by_key["n1"][0], vals_by_key["n6"][0])
                pairs[("Meco", fs, tc, "Fixed", "mae")] = (
                    vals_by_key["n1"][1], vals_by_key["n6"][1])
        return pairs

    for group_key, group_df in df.groupby(group_cols):
        if pair_across_preview:
            aug_p, version_path, curr_model_name = group_key
            preview = "combined_preview"
        else:
            aug_p, preview, version_path, curr_model_name = group_key

        # Get data per dataset
        eval_dfs = {}
        target_cols_per_ds = {}
        # In pair_across_preview mode `df` may contain multiple previews per
        # dataset. The lookup in generate_latex_combined_table is not keyed
        # on preview, so a duplicate row would silently overwrite the prior
        # one (last-iterated wins). Pin each dataset to a single canonical
        # preview so the table reflects a specific configuration rather
        # than arbitrary row order.
        canonical_preview = {"Meco": "all", "OneStop": "gathering"}
        for dataset in ["Meco", "OneStop"]:  # Use actual dataset names from data
            ds_df = group_df[group_df["dataset"] == dataset]
            if pair_across_preview and dataset in canonical_preview:
                pinned = canonical_preview[dataset]
                if pinned in set(ds_df["preview"].unique()):
                    ds_df = ds_df[ds_df["preview"] == pinned]
            if not ds_df.empty:
                target_cols = DATASET_TARGET_COLS.get(dataset, [])
                ds_df = ds_df[ds_df["target_col"].isin(target_cols)]
                if not ds_df.empty:
                    eval_dfs[dataset] = ds_df
                    target_cols_per_ds[dataset] = target_cols

        # Skip if we don't have at least 2 datasets
        if len(eval_dfs) < 2:
            continue

        # Determine which scopes are available (use OneStop's scopes or check both)
        scopes = DATASET_SCOPES.get(list(eval_dfs.keys())[0], [])

        dataset_display = {"OneStop": "OneStop", "Meco": "MECO"}
        target_col_display = DATASET_TARGET_COL_DISPLAY

        # Only the primary fully_agg group renders Meco's Transitions/WFC cells
        # as n1/n6 pairs; the n_paragraphs_geq_* groups don't get a combined
        # table of their own (they have no OneStop pair).
        if version_path == "fully_agg":
            same_model_df = df_full[(df_full["aug_p"] == aug_p) &
                                    (df_full["model_name"] == curr_model_name)]
            multi_version_pairs = _build_multi_version_pairs(same_model_df)
        else:
            multi_version_pairs = None

        latex = generate_latex_combined_table(
            eval_dfs,
            target_cols=target_cols_per_ds,
            scopes=scopes,
            method_to_scope={
                "seen__pool_all": "Fixed",
                "unseen__pool_all": "Any",
            },
            target_col_display=target_col_display,
            dataset_display=dataset_display,
            version_path=version_path,
            model_name=curr_model_name,
            multi_version_pairs=multi_version_pairs,
            paper_style=True,
            caption_override=(
                "\\textbf{Prediction of standard proficiency test scores}. "
                "Pearson $r$ $\\uparrow$ and Mean Absolute Error (MAE) "
                "$\\downarrow$ for prediction of external proficiency scores "
                "using eye movements with "
                f"{_caption_model_phrase(curr_model_name)}. ``Fixed'' is the "
                "Fixed Text regime in which all the eye movement data is for "
                "the same texts presented to the test participant. ``Any'' is "
                "the Any Text regime in which no eye movement data is "
                "available for the texts of the test participant. Best result "
                "for each evaluation measure in each regime, dataset and "
                "target proficiency test is marked in bold."
            ),
            label="tab:predictions",
        )

        if not latex:
            continue

        # Save to directory: full/aug_p/preview/version_path/model/table.tex
        table_dir = save_dir / aug_p / preview / version_path / curr_model_name
        table_dir.mkdir(parents=True, exist_ok=True)
        table_path = table_dir / "combined.tex"
        write_table(table_path, latex)
        logger.info("Saved combined table: %s", table_path)

        # Same table with 3 decimal places (combined_3dp.tex)
        latex_3dp = generate_latex_combined_table(
            eval_dfs,
            target_cols=target_cols_per_ds,
            scopes=scopes,
            method_to_scope={
                "seen__pool_all": "Fixed",
                "unseen__pool_all": "Any",
            },
            target_col_display=target_col_display,
            dataset_display=dataset_display,
            version_path=version_path,
            model_name=curr_model_name,
            multi_version_pairs=multi_version_pairs,
            paper_style=True,
            caption_override=(
                "\\textbf{Prediction of standard proficiency test scores}. "
                "Pearson $r$ $\\uparrow$ and Mean Absolute Error (MAE) "
                "$\\downarrow$ for prediction of external proficiency scores "
                "using eye movements with "
                f"{_caption_model_phrase(curr_model_name)}. ``Fixed'' is the "
                "Fixed Text regime in which all the eye movement data is for "
                "the same texts presented to the test participant. ``Any'' is "
                "the Any Text regime in which no eye movement data is "
                "available for the texts of the test participant. Best result "
                "for each evaluation measure in each regime, dataset and "
                "target proficiency test is marked in bold."
            ),
            label="tab:predictions_3dp",
            decimals=3,
        )
        if latex_3dp:
            table_path_3dp = table_dir / "combined_3dp.tex"
            write_table(table_path_3dp, latex_3dp)
            logger.info("Saved 3-decimal combined table: %s", table_path_3dp)

        # Bootstrap variant (mean ± std with bootstrap-vs-WPM stars) — only for
        # the primary fully_agg group, mirroring EyeScore's combined_bootstrap.
        if version_path == "fully_agg":
            # WPM subset file (all any pairwise run refreshes and all these
            # tables need); fall back to the full all-pairs CSV if absent.
            pairwise_csv = EVAL_SAVE_DIR / "pairwise_significance_wpm.csv"
            if not pairwise_csv.exists():
                pairwise_csv = EVAL_SAVE_DIR / "pairwise_significance.csv"
            pairwise_df = None
            if pairwise_csv.exists():
                pairwise_df = pd.read_csv(pairwise_csv)
            else:
                logger.warning(
                    "Pairwise significance CSV not found: %s — stars vs WPM "
                    "omitted from bootstrap table", pairwise_csv)
            # Model-vs-Ridge comparisons, rendered as the difference subscript
            # on every non-Ridge model's cells. The dedicated subset file is
            # the cheap source; the full all-pairs CSV contains those pairs
            # too, so fall back to it when it is what was loaded above.
            ridge_df = None
            if curr_model_name != BOOTSTRAP_BASELINE_MODEL:
                ridge_csv = EVAL_SAVE_DIR / PAIRWISE_RIDGE_CSV
                if ridge_csv.exists():
                    ridge_df = pd.read_csv(ridge_csv)
                elif pairwise_csv.name == "pairwise_significance.csv":
                    ridge_df = pairwise_df
                else:
                    logger.warning(
                        "vs-Ridge comparison CSV not found: %s — difference "
                        "subscripts omitted for model %s (re-run the "
                        "evaluation with --pairwise-vs-ridge)",
                        ridge_csv, curr_model_name)
            # Featureless mean baseline for this exact slice, drawn as the
            # table's first row. It comes from df_full rather than df: the
            # latter is already filtered to this table's model (and to the
            # requested feature sets, which the baseline is not run under).
            baseline_df = df_full[
                (df_full["aug_p"] == aug_p)
                & (df_full["version_path"] == version_path)
                & (df_full["model_name"] == BASELINE_ROW_MODEL)]
            baseline_parts = []
            for ds, ds_df in eval_dfs.items():
                baseline_parts.append(baseline_df[
                    (baseline_df["dataset"] == ds)
                    & (baseline_df["preview"].isin(ds_df["preview"].unique()))])
            baseline_df = (pd.concat(baseline_parts) if baseline_parts
                           else pd.DataFrame())
            if baseline_df.empty:
                logger.info(
                    "No %s (mean-baseline) results for %s/%s — bootstrap "
                    "tables rendered without the baseline row",
                    BASELINE_ROW_MODEL, aug_p, preview)

            # combined_bootstrap[_spearman].tex carry the paired-bootstrap stars
            # (the only significance test behind a reported result).
            # Each table is also emitted with the ± std stripped
            # (combined_bootstrap*_noci.tex) for uses that want stars only.
            for stars_from, corr_metric, output_name in [
                    ("bootstrap", "pearson", BOOTSTRAP_COMBINED_OUTPUT),
                    ("bootstrap", "spearman", BOOTSTRAP_COMBINED_SPEARMAN_OUTPUT)]:
                for show_dispersion in (True, False):
                    base_name = (output_name if show_dispersion
                                 else output_name.replace(".tex", "_noci.tex"))
                    # The combined (both-datasets) table, plus the same table
                    # split per dataset into <model>/<Dataset>/, with the
                    # dataset in the file name and in the \label.
                    variants = [(eval_dfs, target_cols_per_ds,
                                 table_dir, base_name)]
                    for ds in eval_dfs:
                        slug = DATASET_FILE_SLUG.get(ds, ds.lower())
                        variants.append((
                            {ds: eval_dfs[ds]},
                            {ds: target_cols_per_ds[ds]},
                            table_dir / ds,
                            base_name.replace(".tex", f"_{slug}.tex")))
                    for v_eval_dfs, v_target_cols, v_dir, v_name in variants:
                        latex_boot = generate_bootstrap_combined_latex(
                            v_eval_dfs,
                            target_cols=v_target_cols,
                            scopes=scopes,
                            pairwise_df=pairwise_df,
                            model_name=curr_model_name,
                            aug_p=aug_p,
                            stars_from=stars_from,
                            corr_metric=corr_metric,
                            ridge_df=ridge_df,
                            show_dispersion=show_dispersion,
                            baseline_df=baseline_df,
                        )
                        if latex_boot:
                            v_dir.mkdir(parents=True, exist_ok=True)
                            table_path_boot = v_dir / v_name
                            write_table(table_path_boot, latex_boot)
                            logger.info(
                                "Saved bootstrap combined table (%s stars): %s",
                                stars_from, table_path_boot)

    logger.info("Combined tables generated under %s", save_dir)

    # Generate per_item_agg combined tables
    _generate_per_item_agg_combined_tables(eval_csv, save_dir, feature_sets,
                                           model_name=model_name)


def _generate_per_item_agg_single_scope_table_latex(
    onestop_article, onestop_paragraph, meco_df, model_name, scope="Fixed"
):
    """Generate LaTeX for per_item_agg table with only one scope (Fixed or Any)."""
    metrics = ["pearson_r", "mae"]
    metric_labels = {"pearson_r": "$r$", "mae": "MAE"}

    # Get feature sets
    feature_sets = set()
    for df_part in [onestop_article, onestop_paragraph, meco_df]:
        feature_sets.update(df_part["feature_set"].unique())
    feature_sets = [fs for fs in ALL_FEATURE_SETS_ORDERED if fs in feature_sets]

    # Get target columns
    onestop_targets = sorted(onestop_article["target_col"].unique())
    meco_targets = sorted(meco_df["target_col"].unique())

    # Build lookups: (feature_set, target_col, method) -> {pearson_r, mae}
    def _build_lookup(df):
        lookup = {}
        for _, row in df.iterrows():
            method = row.get("method", "seen__pool_all")
            row_scope = "Fixed" if method == "seen__pool_all" else ("Any" if method == "unseen__pool_all" else None)
            if row_scope is None or row_scope != scope:
                continue
            key = (row["feature_set"], row["target_col"])
            lookup[key] = {m: row[m] for m in metrics}
        return lookup

    art_lookup = _build_lookup(onestop_article)
    par_lookup = _build_lookup(onestop_paragraph)
    meco_lookup = _build_lookup(meco_df)

    # Find best values per (dataset, target, metric)
    best = {}
    for target in onestop_targets:
        for m in metrics:
            vals = []
            for fs in feature_sets:
                for lookup in [art_lookup, par_lookup]:
                    v = lookup.get((fs, target), {}).get(m)
                    if v is not None and not pd.isna(v):
                        vals.append(v)
            if vals:
                best[("onestop", target, m)] = max(vals) if m == "pearson_r" else min(vals)

    for target in meco_targets:
        for m in metrics:
            vals = []
            for fs in feature_sets:
                v = meco_lookup.get((fs, target), {}).get(m)
                if v is not None and not pd.isna(v):
                    vals.append(v)
            if vals:
                best[("meco", target, m)] = max(vals) if m == "pearson_r" else min(vals)

    # Build table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")

    # Column spec: features + (OneStop: 2 targets * 2 article|paragraph * 2 metrics)
    #            + (Meco: 2 targets * 2 metrics)
    onestop_cols = len(onestop_targets) * 2 * len(metrics)
    meco_cols = len(meco_targets) * len(metrics)
    col_spec = f"l||{''.join(['cc'] * len(onestop_targets) * 2)}||{''.join(['cc'] * len(meco_targets))}"

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header 1: Dataset names
    header1 = f" & \\multicolumn{{{onestop_cols}}}{{c}}{{\\textbf{{OneStop}}}} & \\multicolumn{{{meco_cols}}}{{c}}{{\\textbf{{MECO}}}}"
    lines.append(header1 + " \\\\")

    # Header 2: Target columns
    header2_parts = [""]
    for target in onestop_targets:
        display = DATASET_TARGET_COL_DISPLAY["OneStop"].get(target, target)
        cols_per_target = 2 * len(metrics)  # article|paragraph * metrics
        header2_parts.append(f"\\multicolumn{{{cols_per_target}}}{{c}}{{{display}}}")
    for target in meco_targets:
        display = DATASET_TARGET_COL_DISPLAY["Meco"].get(target, target)
        header2_parts.append(f"\\multicolumn{{{len(metrics)}}}{{c}}{{{display}}}")
    lines.append(" & ".join(header2_parts) + " \\\\")

    # Header 3: article | paragraph for OneStop
    header3_parts = [""]
    for _ in onestop_targets:
        header3_parts.append("\\multicolumn{2}{c|}{Article}")
        header3_parts.append("\\multicolumn{2}{c}{Paragraph}")
    lines.append(" & ".join(header3_parts) + " \\\\")

    # Header 4: metric labels (r, MAE)
    header4_parts = ["\\textbf{Features}"]
    for _ in onestop_targets:
        for _ in range(2):  # article and paragraph
            for m in metrics:
                header4_parts.append(metric_labels[m])
    for _ in meco_targets:
        for m in metrics:
            header4_parts.append(metric_labels[m])
    lines.append(" & ".join(header4_parts) + " \\\\")
    lines.append("\\hline")

    # Rows: feature sets
    for i, fs in enumerate(feature_sets):
        if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[feature_sets[i - 1]]:
            lines.append("\\hline")

        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]

        # OneStop article | paragraph
        for target in onestop_targets:
            # Article
            art_vals = art_lookup.get((fs, target), {})
            for m in metrics:
                val = art_vals.get(m)
                if val is not None and not pd.isna(val):
                    is_best = np.isclose(val, best.get(("onestop", target, m), np.nan), atol=1e-6)
                    row_parts.append(_fmt_val(val, is_best))
                else:
                    row_parts.append("NA")

            # Paragraph
            par_vals = par_lookup.get((fs, target), {})
            for m in metrics:
                val = par_vals.get(m)
                if val is not None and not pd.isna(val):
                    is_best = np.isclose(val, best.get(("onestop", target, m), np.nan), atol=1e-6)
                    row_parts.append(_fmt_val(val, is_best))
                else:
                    row_parts.append("NA")

        # Meco
        for target in meco_targets:
            meco_vals = meco_lookup.get((fs, target), {})
            for m in metrics:
                val = meco_vals.get(m)
                if val is not None and not pd.isna(val):
                    is_best = np.isclose(val, best.get(("meco", target, m), np.nan), atol=1e-6)
                    row_parts.append(_fmt_val(val, is_best))
                else:
                    row_parts.append("NA")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def _generate_per_item_agg_combined_table_latex(
    onestop_article, onestop_paragraph, meco_df, model_name
):
    """Generate LaTeX for per_item_agg table with OneStop article | paragraph and Fixed | Any for each."""
    metrics = ["pearson_r", "mae"]
    metric_labels = {"pearson_r": "$r$", "mae": "MAE"}
    scopes = ["Fixed", "Any"]

    # Get feature sets
    feature_sets = set()
    for df_part in [onestop_article, onestop_paragraph, meco_df]:
        feature_sets.update(df_part["feature_set"].unique())
    feature_sets = [fs for fs in ALL_FEATURE_SETS_ORDERED if fs in feature_sets]

    # Get target columns
    onestop_targets = sorted(onestop_article["target_col"].unique())
    meco_targets = sorted(meco_df["target_col"].unique())

    # Build lookups: (feature_set, target_col, method) -> {pearson_r, mae}
    def _build_lookup(df):
        lookup = {}
        for _, row in df.iterrows():
            method = row.get("method", "seen__pool_all")
            scope = "Fixed" if method == "seen__pool_all" else ("Any" if method == "unseen__pool_all" else None)
            if scope is None:
                continue
            key = (row["feature_set"], row["target_col"], scope)
            lookup[key] = {m: row[m] for m in metrics}
        return lookup

    art_lookup = _build_lookup(onestop_article)
    par_lookup = _build_lookup(onestop_paragraph)
    meco_lookup = _build_lookup(meco_df)

    # Find best values per (dataset, target, scope, metric)
    best = {}
    for target in onestop_targets:
        for scope in scopes:
            for m in metrics:
                vals = []
                for fs in feature_sets:
                    for lookup in [art_lookup, par_lookup]:
                        v = lookup.get((fs, target, scope), {}).get(m)
                        if v is not None and not pd.isna(v):
                            vals.append(v)
                if vals:
                    best[("onestop", target, scope, m)] = max(vals) if m == "pearson_r" else min(vals)

    for target in meco_targets:
        for scope in scopes:
            for m in metrics:
                vals = []
                for fs in feature_sets:
                    v = meco_lookup.get((fs, target, scope), {}).get(m)
                    if v is not None and not pd.isna(v):
                        vals.append(v)
                if vals:
                    best[("meco", target, scope, m)] = max(vals) if m == "pearson_r" else min(vals)

    # Build table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")

    # Column spec: features + (OneStop: 2 targets * 2 article|paragraph * 2 scopes * 2 metrics)
    #            + (Meco: 2 targets * 2 scopes * 2 metrics)
    onestop_cols = len(onestop_targets) * 2 * len(scopes) * len(metrics)
    meco_cols = len(meco_targets) * len(scopes) * len(metrics)
    col_spec = f"l||{''.join(['cc'] * len(onestop_targets) * 2 * len(scopes))}||{''.join(['cc'] * len(meco_targets) * len(scopes))}"

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header 1: Dataset names
    header1 = f" & \\multicolumn{{{onestop_cols}}}{{c}}{{\\textbf{{OneStop}}}} & \\multicolumn{{{meco_cols}}}{{c}}{{\\textbf{{MECO}}}}"
    lines.append(header1 + " \\\\")

    # Header 2: Target columns with article | paragraph for OneStop
    header2_parts = [""]
    for target in onestop_targets:
        display = DATASET_TARGET_COL_DISPLAY["OneStop"].get(target, target)
        cols_per_target = 2 * len(scopes) * len(metrics)  # article|paragraph * scopes * metrics
        header2_parts.append(f"\\multicolumn{{{cols_per_target}}}{{c}}{{{display}}}")
    for target in meco_targets:
        display = DATASET_TARGET_COL_DISPLAY["Meco"].get(target, target)
        cols_per_target = len(scopes) * len(metrics)
        header2_parts.append(f"\\multicolumn{{{cols_per_target}}}{{c}}{{{display}}}")
    lines.append(" & ".join(header2_parts) + " \\\\")

    # Header 3: article | paragraph subheaders for OneStop
    header3_parts = [""]
    for _ in onestop_targets:
        cols_per_part = len(scopes) * len(metrics)  # Fixed|Any × r,MAE
        header3_parts.append(f"\\multicolumn{{{cols_per_part}}}{{c|}}{{Article}}")
        header3_parts.append(f"\\multicolumn{{{cols_per_part}}}{{c}}{{Paragraph}}")
    lines.append(" & ".join(header3_parts) + " \\\\")

    # Header 4: Fixed | Any for each target
    header4_parts = [""]
    for _ in onestop_targets:
        for _ in range(2):  # article and paragraph
            for scope in scopes:
                header4_parts.append(f"\\multicolumn{{2}}{{c}}{{{scope}}}")
    for _ in meco_targets:
        for scope in scopes:
            header4_parts.append(f"\\multicolumn{{2}}{{c}}{{{scope}}}")
    lines.append(" & ".join(header4_parts) + " \\\\")

    # Header 5: metric labels (r, MAE)
    header5_parts = ["\\textbf{Features}"]
    for _ in onestop_targets:
        for _ in range(2):  # article and paragraph
            for _ in scopes:
                for m in metrics:
                    header5_parts.append(metric_labels[m])
    for _ in meco_targets:
        for _ in scopes:
            for m in metrics:
                header5_parts.append(metric_labels[m])
    lines.append(" & ".join(header5_parts) + " \\\\")
    lines.append("\\hline")

    # Rows: feature sets
    for i, fs in enumerate(feature_sets):
        if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[feature_sets[i - 1]]:
            lines.append("\\hline")

        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]

        # OneStop article | paragraph with Fixed | Any
        for target in onestop_targets:
            # Article
            for scope in scopes:
                art_vals = art_lookup.get((fs, target, scope), {})
                for m in metrics:
                    val = art_vals.get(m)
                    if val is not None and not pd.isna(val):
                        is_best = np.isclose(val, best.get(("onestop", target, scope, m), np.nan), atol=1e-6)
                        row_parts.append(_fmt_val(val, is_best))
                    else:
                        row_parts.append("NA")

            # Paragraph
            for scope in scopes:
                par_vals = par_lookup.get((fs, target, scope), {})
                for m in metrics:
                    val = par_vals.get(m)
                    if val is not None and not pd.isna(val):
                        is_best = np.isclose(val, best.get(("onestop", target, scope, m), np.nan), atol=1e-6)
                        row_parts.append(_fmt_val(val, is_best))
                    else:
                        row_parts.append("NA")

        # Meco with Fixed | Any
        for target in meco_targets:
            for scope in scopes:
                meco_vals = meco_lookup.get((fs, target, scope), {})
                for m in metrics:
                    val = meco_vals.get(m)
                    if val is not None and not pd.isna(val):
                        is_best = np.isclose(val, best.get(("meco", target, scope, m), np.nan), atol=1e-6)
                        row_parts.append(_fmt_val(val, is_best))
                    else:
                        row_parts.append("NA")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def _generate_per_item_agg_combined_tables(
    eval_csv: Path,
    save_dir: Path,
    feature_sets: Optional[List[str]] = None,
    model_name: Optional[str] = None,
):
    """Generate combined per_item_agg tables showing OneStop article and paragraph separately with Meco."""
    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_ORDERED

    df = pd.read_csv(eval_csv)
    df = df[df["feature_set"].isin(feature_sets)]
    df = df[df["version_path"].str.startswith("per_item_agg", na=False)]
    if model_name is not None:
        df = df[df["model_name"] == model_name]

    if df.empty:
        logger.info("No per_item_agg data found for combined tables")
        return

    # Group by (aug_p, model_name)
    group_cols = ["aug_p", "model_name"]
    for aug_p, curr_model_name in df[group_cols].drop_duplicates().values:
        group_df = df[(df["aug_p"] == aug_p) & (df["model_name"] == curr_model_name)]

        # Get OneStop data separated by article/paragraph
        onestop_df = group_df[group_df["dataset"] == "OneStop"]
        onestop_article = onestop_df[onestop_df["version_path"].str.contains("article", na=False)]
        onestop_paragraph = onestop_df[onestop_df["version_path"].str.contains("paragraph", na=False)]

        # Filter to target columns
        onestop_article = onestop_article[onestop_article["target_col"].isin(DATASET_TARGET_COLS["OneStop"])]
        onestop_paragraph = onestop_paragraph[onestop_paragraph["target_col"].isin(DATASET_TARGET_COLS["OneStop"])]

        # Get Meco data
        meco_df = group_df[group_df["dataset"] == "Meco"]
        meco_df = meco_df[meco_df["target_col"].isin(DATASET_TARGET_COLS.get("Meco", []))]

        # Skip if missing data
        if onestop_article.empty or onestop_paragraph.empty or meco_df.empty:
            logger.info(f"Skipping per_item_agg table for {aug_p}/{curr_model_name}: missing data")
            continue

        # Generate LaTeX for all three variants
        latex_combined = _generate_per_item_agg_combined_table_latex(
            onestop_article, onestop_paragraph, meco_df, curr_model_name
        )
        latex_fixed = _generate_per_item_agg_single_scope_table_latex(
            onestop_article, onestop_paragraph, meco_df, curr_model_name, scope="Fixed"
        )
        latex_any = _generate_per_item_agg_single_scope_table_latex(
            onestop_article, onestop_paragraph, meco_df, curr_model_name, scope="Any"
        )

        if not latex_combined and not latex_fixed and not latex_any:
            continue

        # Save to directory: full/per_item_agg/aug_p/model/table.tex
        table_dir = save_dir / "per_item_agg" / aug_p / curr_model_name
        table_dir.mkdir(parents=True, exist_ok=True)

        if latex_combined:
            table_path = table_dir / "combined.tex"
            write_table(table_path, latex_combined)
            logger.info("Saved per_item_agg combined table: %s", table_path)

        if latex_fixed:
            table_path = table_dir / "fixed_only.tex"
            write_table(table_path, latex_fixed)
            logger.info("Saved per_item_agg fixed_only table: %s", table_path)

        if latex_any:
            table_path = table_dir / "any_only.tex"
            write_table(table_path, latex_any)
            logger.info("Saved per_item_agg any_only table: %s", table_path)


# ── Combined LOLO vs Stratified tables ──────────────────────────────────────


def generate_lolo_vs_stratified_combined_tables(
    eval_csv: Path = None,
    save_dir: Path = None,
    model_name: str = None,
    feature_sets: Optional[List[str]] = None,
):
    """Generate combined LOLO vs stratified tables showing OneStop × Meco.

    Combines OneStop (gathering preview) with Meco (all preview) across different previews.
    Generates 3 tables: joint (LOLO + L1-Strat), LOLO-only, and L1-Strat-only.
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "full" / "lolo_vs_stratified"
    if feature_sets is None:
        feature_sets = LOLO_VS_STRAT_FEATURE_SETS

    df = pd.read_csv(eval_csv)
    df = df[
        (df["feature_set"].isin(feature_sets))
        & (df["method"].isin(LOLO_VS_STRAT_METHOD_TO_SCOPE.keys()))
    ]
    if model_name is not None:
        df = df[df["model_name"] == model_name]

    if df.empty:
        logger.warning("No LOLO/stratified data for combined tables")
        return

    # Get data for each dataset (with dataset-specific previews). The dataset
    # column holds "Meco" (display name "MECO" is applied later).
    onestop_df = df[(df["dataset"] == "OneStop") & (df["preview"] == "gathering")]
    meco_df = df[(df["dataset"] == "Meco") & (df["preview"] == "all")]

    if onestop_df.empty or meco_df.empty:
        logger.warning("Missing data for OneStop (gathering) or Meco (all)")
        return

    # Group by (aug_p, version_path, model_name) - NOT by preview since they differ
    for aug_p in onestop_df["aug_p"].unique():
        for version_path in onestop_df[onestop_df["aug_p"] == aug_p]["version_path"].unique():
            for curr_model_name in onestop_df[(onestop_df["aug_p"] == aug_p) & (onestop_df["version_path"] == version_path)]["model_name"].unique():

                # Get data for this combination
                onestop_group = onestop_df[
                    (onestop_df["aug_p"] == aug_p) &
                    (onestop_df["version_path"] == version_path) &
                    (onestop_df["model_name"] == curr_model_name)
                ]

                meco_group = meco_df[
                    (meco_df["aug_p"] == aug_p) &
                    (meco_df["version_path"] == version_path) &
                    (meco_df["model_name"] == curr_model_name)
                ]

                if onestop_group.empty or meco_group.empty:
                    continue

                # Filter to target columns
                target_cols_onestop = DATASET_TARGET_COLS.get("OneStop", [])
                target_cols_meco = DATASET_TARGET_COLS.get("Meco", [])

                onestop_filtered = onestop_group[onestop_group["target_col"].isin(target_cols_onestop)]
                meco_filtered = meco_group[meco_group["target_col"].isin(target_cols_meco)]

                if onestop_filtered.empty or meco_filtered.empty:
                    continue

                target_col_display = DATASET_TARGET_COL_DISPLAY
                dataset_display = {"OneStop": "OneStop", "Meco": "MECO"}
                table_dir = save_dir / aug_p / "gathering_all" / version_path / curr_model_name
                table_dir.mkdir(parents=True, exist_ok=True)

                # Generate joint table (LOLO + L1-Strat)
                eval_dfs = {
                    "OneStop": onestop_filtered,
                    "Meco": meco_filtered
                }
                target_cols_per_ds = {
                    "OneStop": target_cols_onestop,
                    "Meco": target_cols_meco
                }

                latex = generate_latex_combined_table(
                    eval_dfs,
                    target_cols=target_cols_per_ds,
                    scopes=LOLO_VS_STRAT_SCOPES,
                    method_to_scope=LOLO_VS_STRAT_METHOD_TO_SCOPE,
                    target_col_display=target_col_display,
                    dataset_display=dataset_display,
                    version_path=version_path,
                    model_name=curr_model_name,
                )

                if latex:
                    table_path = table_dir / "lolo_vs_stratified_combined.tex"
                    write_table(table_path, latex)
                    logger.info("Saved combined LOLO/stratified table (OneStop/gathering + Meco/all, displayed as MECO): %s", table_path)

                # Generate LOLO-only table
                onestop_lolo = onestop_filtered[onestop_filtered["method"] == "all__lolo"]
                meco_lolo = meco_filtered[meco_filtered["method"] == "all__lolo"]

                if not onestop_lolo.empty and not meco_lolo.empty:
                    eval_dfs_lolo = {
                        "OneStop": onestop_lolo,
                        "Meco": meco_lolo
                    }
                    latex_lolo = generate_latex_combined_table(
                        eval_dfs_lolo,
                        target_cols=target_cols_per_ds,
                        scopes=["LOLO"],
                        method_to_scope={"all__lolo": "LOLO"},
                        target_col_display=target_col_display,
                        dataset_display=dataset_display,
                        version_path=version_path,
                        model_name=curr_model_name,
                        show_scope_row=False,
                        scope_label="Leave-One-Language-Out",
                    )

                    if latex_lolo:
                        table_path_lolo = table_dir / "lolo_combined.tex"
                        write_table(table_path_lolo, latex_lolo)
                        logger.info("Saved LOLO-only table: %s", table_path_lolo)

                # Generate L1-Stratified-only table
                onestop_strat = onestop_filtered[onestop_filtered["method"] == "all__l1_strat_kfold"]
                meco_strat = meco_filtered[meco_filtered["method"] == "all__l1_strat_kfold"]

                if not onestop_strat.empty and not meco_strat.empty:
                    eval_dfs_strat = {
                        "OneStop": onestop_strat,
                        "Meco": meco_strat
                    }
                    latex_strat = generate_latex_combined_table(
                        eval_dfs_strat,
                        target_cols=target_cols_per_ds,
                        scopes=["L1-Strat."],
                        method_to_scope={"all__l1_strat_kfold": "L1-Strat."},
                        target_col_display=target_col_display,
                        dataset_display=dataset_display,
                        version_path=version_path,
                        model_name=curr_model_name,
                        show_scope_row=False,
                        scope_label="L1-Stratified",
                    )

                    if latex_strat:
                        table_path_strat = table_dir / "stratified_combined.tex"
                        write_table(table_path_strat, latex_strat)
                        logger.info("Saved L1-Stratified-only table: %s", table_path_strat)

    logger.info("Combined LOLO/stratified tables generated under %s", save_dir)


def run_all_tables(model_names: Optional[List[str]] = None):
    """Generate every predictions table family, one folder per model.

    Every table family writes into per-model subfolders (e.g.
    ``tables/full/aug_p=1.0/combined_preview/fully_agg/<model_name>/``), so a
    newly evaluated model simply gains its own folder next to the existing
    ones.

    Args:
        model_names: Restrict generation to these models only — other models'
            existing table folders are left untouched. None regenerates for
            every model present in the evaluation CSV.
    """
    for m in (model_names if model_names else [None]):
        tag = m if m else "all models"
        logger.info("Generating combined OneStop × Meco tables (%s)...", tag)
        generate_full_combined_tables(model_name=m)
        # Paper configuration: Meco (all preview) paired with OneStop (gathering
        # preview) — writes under combined_preview/. The default same-preview
        # grouping above pairs OneStop's small `all` slice instead, which does
        # NOT match the paper's Table 2.
        logger.info("Generating combined tables across previews (paper config, %s)...", tag)
        generate_full_combined_tables(model_name=m, pair_across_preview=True)
        logger.info("Generating combined LOLO vs stratified tables (%s)...", tag)
        generate_lolo_vs_stratified_combined_tables(model_name=m)
        logger.info("Generating LOLO vs stratified tables (%s)...", tag)
        generate_lolo_vs_stratified_tables(model_name=m)
        for simplified in (False, True):
            for cv, (_, run_label, _, _) in MICHIGAN_PARTS_CV.items():
                logger.info("Generating %sMichigan test parts tables (%s, %s)...",
                            "simplified " if simplified else "", run_label, tag)
                generate_michigan_parts_tables(cv, simplified=simplified, model_name=m)
        logger.info("Generating OneStop comprehension tables (%s)...", tag)
        generate_comprehension_tables(model_name=m)
        logger.info("Generating comprehension LOLO vs stratified tables (%s)...", tag)
        generate_comprehension_lolo_vs_stratified_tables(model_name=m)
        for simplified in (False, True):
            for cv, (_, run_label, _, _) in MICHIGAN_PARTS_CV.items():
                logger.info("Generating %scomprehension Michigan test parts tables (%s, %s)...",
                            "simplified " if simplified else "", run_label, tag)
                generate_michigan_parts_tables(cv, comprehension=True, simplified=simplified,
                                               model_name=m)

    logger.info("Generating per-model main tables (fully_agg only)...")
    generate_all_tables_for_all_models(model_names=model_names,
                                       version_path_filter="fully_agg")

    logger.info("Done generating tables.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate predictions LaTeX tables")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names to generate tables for. Each model "
             "gets its own folder under every table family (e.g. "
             "tables/full/aug_p=1.0/combined_preview/fully_agg/<model>/); "
             "other models' folders are untouched. Default: all models in "
             "the evaluation CSV.")
    args = parser.parse_args()

    table_model_names = ([m.strip() for m in args.models.split(",")]
                         if args.models else None)
    run_all_tables(table_model_names)
