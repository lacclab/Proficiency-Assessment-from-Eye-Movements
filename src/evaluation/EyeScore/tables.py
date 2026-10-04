"""
Generate LaTeX results tables for the EyeScore pipeline.

Produces one .tex table per (variant, dataset, preview, version_path) combination.
Each table has:
  - Rows: feature sets (single features + combined, separated by \\hline)
  - Columns: content_mode (all/seen/unseen) → target columns → Pearson r
  - Best value per column bolded

Meco has only content_mode='all'; OneStop has all/seen/unseen.

Usage:
    python -m src.evaluation.EyeScore.tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging
import os
import re

from src.evaluation.EyeScore.evaluation import (
    EVAL_SAVE_DIR,
    FEATURE_SET_DISPLAY,
    PER_LANGUAGE_AGGREGATE_LABEL,
    RESULTS_DIR,
    eye_score_csv_path,
    load_metadata,
)
from src.evaluation.EyeScore.language_bias import _merge_eyescore_metadata
from src.evaluation.stats_utils import N_BOOTSTRAP
from src.methods.EyeScore.calculation import (
    RAW_VARIANT,
    SPLIT_SCHEME_TO_VARIANT,
    DEBIAS_METHODS,
    DEFAULT_DEBIAS_METHOD,
    TYPO_CALIB_DISTANCE_TYPE,
    SCORER_NAMES,
)
from src.methods.EyeScore.typology import compute_typological_distances
from src.constants import MICH_TEST_PARTS_EXTENDED, TestCols, SMALL_FEATURE_SETS
from src.latex_captions import rename_datasets_in_tables


def write_table(out_path, latex: str) -> None:
    """Write a finished table, applying the shared dataset-naming rules."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rename_datasets_in_tables(latex))

logger = logging.getLogger(__name__)

TABLES_SAVE_DIR = Path("src/evaluation/EyeScore/tables")

# ── Display name mappings ────────────────────────────────────────────────────


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

# Mapping from raw content_mode in the eval CSV to the scope label shown in
# the table. Only modes listed here are emitted as columns.
DATASET_CONTENT_MODE_TO_SCOPE = {
    "OneStop": {"all": "All", "seen": "Fixed", "unseen": "Any"},
    # Meco's seen/unseen now comes from the half-split LOPO produced by
    # `_run_meco_seen_unseen_eyescore_job`. "all" is the original 12-paragraph
    # eye_score (no split); seen/unseen are the half-averaged variants.
    "Meco": {"all": "All", "seen": "Fixed", "unseen": "Any"},
}

# Per-dataset ordered list of scope labels.
DATASET_SCOPES = {
    "OneStop": ["Fixed", "Any"],
    "Meco": ["Fixed", "Any"],
}

# Feature sets that are inherently per-text — they only have a "Seen" cell;
# the "Unseen" cell is forced to NA in the table even if the eval CSV happens
# to contain a row (the runtime guard should already prevent that).
FIXED_TEXT_ONLY_FEATURE_SETS = {"TRANSITIONS", "WFC"}

# Feature set rows grouped for the table; an \hline is inserted between groups.
# WPM is the baseline and is separated from the other single features.
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

# Target columns per dataset (same as predictions)
DATASET_TARGET_COLS = {
    "OneStop": ["lextale_score", "michtest_score"],
    "Meco": ["lextale_score", "proficiency_agg"],
}


# ── Bias-reduction table (% drop in |r| under typology calibration) ──────────

# language_bias.py's run_simplified_bias_plots writes
# bias_correlation_summary.csv under this tree (alongside the other lang_bias
# evaluation CSVs):
#   {BIAS_SUMMARY_DIR}/{mode}/bias_correlation_summary.csv
# where mode ∈ {"org", "seen", "new"} — see SIMPLIFIED_MODE_SPECS there.
BIAS_SUMMARY_DIR = EVAL_SAVE_DIR / "lang_bias"
BIAS_SUMMARY_FILENAME = "bias_correlation_summary.csv"

# Mode label → (scheme, variant) tuple for the EVAL CSVs (which still live
# under EVAL_SAVE_DIR/lang_bias/{scheme}/{variant}/ — produced by the
# evaluation pipeline, not the simplified bias plot pipeline).
BIAS_MODE_TO_EVAL_PATH = {
    "org":  ("across_langs", "org"),
    "seen": ("inside_langs", "typo_calibrated"),
    "new":  ("across_langs", "typo_calibrated"),
}

# Distance flavor we report in the bias-reduction table.
BIAS_DISTANCE_TYPE = "syntactic+genetic"

BIAS_FEATURE_SET_ORDER = [
    "READING_SPEED",
    "FIXATION_METRICS",
    "S_CLUSTERS_NO_NORM",
    "WP_COEFS_NO_NORM",
    "TRANSITIONS",
    "WFC",
    "READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM",
]

# Per-dataset slice of the bias summary kept in the table.
# OneStop: all three content modes, single difficulty cut.
# Meco: same three content modes — `seen`/`unseen` are the averaged
# same-half/cross-half eye_scores produced by `_run_meco_seen_unseen_eyescore_job`.
# "all" is the original 12-paragraph fully_agg score.
BIAS_DATASET_FILTERS = {
    "Meco": {
        "version_paths": ("fully_agg/all/all",),
        "content_modes": ("all", "seen", "unseen"),
        "previews": ("All",),
    },
    "OneStop": {
        "version_paths": ("fully_agg/ordinary/all",),
        "content_modes": ("all", "seen", "unseen"),
        "previews": ("Gathering",),
    },
}

BIAS_TARGET_DISPLAY = {
    "lextale_score": "LexTALE",
    "michtest_score": "MichTest",
    "proficiency_agg": "Composite",
}



def _display(key: str, mapping: dict) -> str:
    result = mapping.get(key, key)
    # If not in mapping, escape underscores for LaTeX
    if key not in mapping:
        result = result.replace("_", r"\_")
    return result


def _slug(s: str) -> str:
    """LaTeX-friendly slug for use in \\label keys (no slashes/underscores)."""
    return s.replace("/", "-").replace("_", "-").replace(" ", "-").lower()


def _caption_suffix(dataset: str, preview: str, version_path: str) -> str:
    """Paper-ready trailing label for table captions.

    Strips project-internal slugs (preview names, `fully_agg`/`ordinary`)
    and keeps only what a reader needs: the dataset name plus an optional
    reading-difficulty tag for OneStop (advanced / elementary).
    """
    difficulty = None
    if version_path:
        tail = version_path.rstrip("/").split("/")[-1].lower()
        difficulty = {"adv": "advanced", "ele": "elementary"}.get(tail)
    if difficulty:
        return f"{dataset}, {difficulty}"
    return dataset


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
    content_mode_to_scope: dict,
    target_col_display: dict,
    dataset: str = "",
    preview: str = "",
    version_path: str = "",
    subset_label: str = "",
) -> str:
    """Generate a LaTeX table string in the MichTest/LexTALE × Seen/Unseen layout.

    Args:
        eval_df: DataFrame filtered to a single (dataset, preview, version_path).
        target_cols: Ordered list of target column names (outer column group).
        scopes: Ordered list of scope labels (inner column group), e.g. ["Seen","Unseen"].
        content_mode_to_scope: Mapping from raw content_mode -> scope label.
        target_col_display: Mapping from raw target_col -> display label.

    Returns:
        LaTeX table string, or empty string if no data.
    """
    feature_sets = [fs for fs in SMALL_FEATURE_SETS
                    if fs in eval_df["feature_set"].unique()]
    if not feature_sets:
        return ""

    # Build a lookup: (feature_set, target_col, scope) -> pearson_r
    # Drops content_modes that don't map to a scope shown in this table.
    lookup = {}
    for _, row in eval_df.iterrows():
        scope = content_mode_to_scope.get(row["content_mode"])
        if scope is None:
            continue
        key = (row["feature_set"], row["target_col"], scope)
        lookup[key] = row["pearson_r"]

    # Force fixed-text-only feature sets to NA in non-Fixed scopes regardless
    # of what's in the eval CSV.
    for fs in feature_sets:
        if fs in FIXED_TEXT_ONLY_FEATURE_SETS:
            for tc in target_cols:
                for scope in scopes:
                    if scope != "Fixed":
                        lookup[(fs, tc, scope)] = np.nan

    # Find best (max r) per (target_col, scope) across feature sets
    best = {}
    for tc in target_cols:
        for scope in scopes:
            vals = []
            for fs in feature_sets:
                r = lookup.get((fs, tc, scope))
                if r is not None and not pd.isna(r):
                    vals.append(r)
            best[(tc, scope)] = max(vals) if vals else None

    n_scopes = len(scopes)

    lines = []
    lines.append("\\begin{table*}[ht!]")
    lines.append("\\centering")
    lines.append("\\small")

    # Add caption before table content
    if dataset and version_path:
        suffix = _caption_suffix(dataset, preview, version_path)
        lines.append(
            f"\\caption{{EyeScore--proficiency Pearson $r$ ({suffix}).}}"
        )

    # Column spec: first col left-aligned, then per target_col block of scopes
    col_groups = ["|".join(["c"] * n_scopes) for _ in target_cols]
    col_spec = "l||" + "||".join(col_groups)

    # Wrap in resizebox if many columns to prevent overfull hbox
    if len(target_cols) > 6:
        lines.append("\\resizebox{\\textwidth}{!}{")

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header row 1: target column display names spanning their scope cells
    header1_parts = [""]
    for tc in target_cols:
        display = _display(tc, target_col_display)
        header1_parts.append(f"\\multicolumn{{{n_scopes}}}{{c}}{{\\textbf{{{display}}}}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    # Header row 2: scope labels under each target group
    header2_parts = ["\\textbf{Features}"]
    for _tc in target_cols:
        for scope in scopes:
            header2_parts.append(scope)
    lines.append(" & ".join(header2_parts) + " \\\\")
    lines.append("\\hline")

    # Data rows
    for i, fs in enumerate(feature_sets):
        if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[feature_sets[i - 1]]:
            lines.append("\\hline")

        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for tc in target_cols:
            for scope in scopes:
                val = lookup.get((fs, tc, scope))
                if val is not None and not pd.isna(val):
                    best_val = best.get((tc, scope))
                    is_best = (best_val is not None and
                               np.isclose(val, best_val, atol=1e-6))
                    row_parts.append(_fmt_val(val, is_best))
                else:
                    row_parts.append("NA")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")

    # Close resizebox if it was opened
    if len(target_cols) > 6:
        lines.append("}")

    # Add label if we have dataset and version_path
    if dataset and version_path:
        label_parts = ["eyescore"]
        if subset_label:
            label_parts.append(_slug(subset_label))
        label_parts.extend([_slug(dataset), _slug(preview), _slug(version_path)])
        lines.append(f"\\label{{tab:{'-'.join(p for p in label_parts if p)}}}")

    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_per_l1_latex_table(
    per_l1_df: pd.DataFrame,
    target_cols: List[str],
    scopes: List[str],
    content_mode_to_scope: dict,
    target_col_display: dict,
    dataset: str = "",
    preview: str = "",
    version_path: str = "",
    subset_label: str = "",
) -> str:
    """Per-language LaTeX table.

    Layout:
      - Outer columns: target_col, with each split into the same scopes as the
        aggregate table.
      - Rows: (feature_set, L1), with L1='All' first within each feature_set
        block. Per-language pearson_r values; the best per (target, scope, L1)
        is bolded so each language gets its own winner.
    """
    feature_sets = [fs for fs in SMALL_FEATURE_SETS
                    if fs in per_l1_df["feature_set"].unique()]
    if not feature_sets:
        return ""

    other_l1s = sorted(
        l1 for l1 in per_l1_df["L1"].unique()
        if l1 != PER_LANGUAGE_AGGREGATE_LABEL
    )
    l1_order = [PER_LANGUAGE_AGGREGATE_LABEL] + other_l1s
    if len(l1_order) <= 1:
        # Only "All" — no per-language detail to render.
        return ""

    lookup = {}
    for _, row in per_l1_df.iterrows():
        scope = content_mode_to_scope.get(row["content_mode"])
        if scope is None:
            continue
        lookup[(row["feature_set"], row["L1"], row["target_col"], scope)] = row["pearson_r"]

    # Same fixed-text-set guard as the aggregate table — non-Fixed scopes
    # collapse to NA for these feature sets across every L1.
    for fs in feature_sets:
        if fs in FIXED_TEXT_ONLY_FEATURE_SETS:
            for l1 in l1_order:
                for tc in target_cols:
                    for scope in scopes:
                        if scope != "Fixed":
                            lookup[(fs, l1, tc, scope)] = np.nan

    # Best per (target, scope, L1) — bolded winner per row across feature sets.
    best = {}
    for tc in target_cols:
        for scope in scopes:
            for l1 in l1_order:
                vals = []
                for fs in feature_sets:
                    r = lookup.get((fs, l1, tc, scope))
                    if r is not None and not pd.isna(r):
                        vals.append(r)
                best[(tc, scope, l1)] = max(vals) if vals else None

    n_scopes = len(scopes)
    col_groups = ["|".join(["c"] * n_scopes) for _ in target_cols]
    col_spec = "l|l||" + "||".join(col_groups)

    lines = ["\\begin{table*}[ht!]", "\\centering", "\\small"]
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    header1 = ["", ""]
    for tc in target_cols:
        display = _display(tc, target_col_display)
        header1.append(f"\\multicolumn{{{n_scopes}}}{{c}}{{\\textbf{{{display}}}}}")
    lines.append(" & ".join(header1) + " \\\\")

    header2 = ["\\textbf{Features}", "\\textbf{L1}"]
    for _tc in target_cols:
        for scope in scopes:
            header2.append(scope)
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\hline")

    for fi, fs in enumerate(feature_sets):
        if fi > 0:
            lines.append("\\hline")
        fs_display = _display(fs, FEATURE_SET_DISPLAY)
        for li, l1 in enumerate(l1_order):
            row_parts = [fs_display if li == 0 else "", l1]
            for tc in target_cols:
                for scope in scopes:
                    val = lookup.get((fs, l1, tc, scope))
                    if val is not None and not pd.isna(val):
                        best_val = best.get((tc, scope, l1))
                        is_best = (best_val is not None and
                                   np.isclose(val, best_val, atol=1e-6))
                        row_parts.append(_fmt_val(val, is_best))
                    else:
                        row_parts.append("NA")
            lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")

    if dataset and version_path:
        suffix = _caption_suffix(dataset, preview, version_path)
        lines.append(
            f"\\caption{{Per-L1 EyeScore--proficiency Pearson $r$ ({suffix}).}}"
        )
        label_parts = ["eyescore", "per-l1"]
        if subset_label:
            label_parts.append(_slug(subset_label))
        label_parts.extend([_slug(dataset), _slug(preview), _slug(version_path)])
        lines.append(f"\\label{{tab:{'-'.join(p for p in label_parts if p)}}}")

    lines.append("\\end{table*}")
    return "\n".join(lines)


def generate_tables_from_eval(
    eval_csv: Path,
    save_dir: Path,
    feature_sets: Optional[List[str]] = None,
    label: str = "",
    subset_label: str = "",
):
    """Read one evaluation CSV and emit tables under save_dir directly.

    Output layout: save_dir/{dataset}/{preview}/{version_path}/table.tex.
    `label` is only used for log messages.
    `subset_label` is the lang_bias subset (e.g. ``across_langs/typo_calibrated``)
    and is embedded in the LaTeX caption + label of each generated table.
    """
    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_ORDERED

    if not eval_csv.exists():
        logger.warning("Eval CSV not found%s: %s", f" ({label})" if label else "", eval_csv)
        return

    df = pd.read_csv(eval_csv)
    df = df[df["feature_set"].isin(feature_sets)]

    if df.empty:
        logger.warning("No data%s after filtering", f" ({label})" if label else "")
        return

    # Optional per-language sibling CSV — written by evaluate_all_results when
    # the eye_score CSVs carry an L1 column. Used to emit table_per_l1.tex
    # next to the aggregate table.
    per_l1_csv = eval_csv.parent / "per_language_evaluation_results.csv"
    per_l1_df = None
    if per_l1_csv.exists():
        per_l1_df = pd.read_csv(per_l1_csv)
        per_l1_df = per_l1_df[per_l1_df["feature_set"].isin(feature_sets)]

    group_cols = ["dataset", "preview", "version_path"]
    for group_key, group_df in df.groupby(group_cols):
        dataset, preview, version_path = group_key

        target_cols = DATASET_TARGET_COLS.get(dataset)
        if target_cols is None:
            logger.warning("No target columns configured for dataset=%s, skipping", dataset)
            continue

        group_df = group_df[group_df["target_col"].isin(target_cols)]
        if group_df.empty:
            continue

        cm_to_scope = DATASET_CONTENT_MODE_TO_SCOPE.get(dataset)
        scopes = DATASET_SCOPES.get(dataset)
        target_col_display = DATASET_TARGET_COL_DISPLAY.get(dataset)
        if cm_to_scope is None or scopes is None or target_col_display is None:
            logger.warning("No scope mapping configured for dataset=%s, skipping", dataset)
            continue

        # per_item_agg was calculated with content_mode="all" only until R5
        # switched to --content-mode seen,unseen; both vintages exist on disk.
        # Pinning scopes to ["All"] made the current tree — which has no "all"
        # rows at all — produce an empty table, so derive the list from the
        # content modes actually present instead.
        if version_path.startswith("per_item_agg"):
            present = {cm_to_scope.get(cm) for cm in group_df["content_mode"].unique()}
            scopes = ["All"] if present == {"All"} else [s for s in scopes if s in present]
            if not scopes:
                logger.warning("No usable content modes for %s/%s/%s, skipping",
                               dataset, preview, version_path)
                continue

        latex = generate_latex_table(
            group_df,
            target_cols=target_cols,
            scopes=scopes,
            content_mode_to_scope=cm_to_scope,
            target_col_display=target_col_display,
            dataset=dataset,
            preview=preview,
            version_path=version_path,
            subset_label=subset_label,
        )
        if not latex:
            continue

        table_dir = save_dir / dataset / preview / version_path
        table_dir.mkdir(parents=True, exist_ok=True)
        table_path = table_dir / "table.tex"
        write_table(table_path, latex)
        logger.info("Saved table: %s", table_path)

        if per_l1_df is not None:
            per_l1_group = per_l1_df[
                (per_l1_df["dataset"] == dataset)
                & (per_l1_df["preview"] == preview)
                & (per_l1_df["version_path"] == version_path)
                & (per_l1_df["target_col"].isin(target_cols))
            ]
            if not per_l1_group.empty:
                per_l1_latex = generate_per_l1_latex_table(
                    per_l1_group,
                    target_cols=target_cols,
                    scopes=scopes,
                    content_mode_to_scope=cm_to_scope,
                    target_col_display=target_col_display,
                    dataset=dataset,
                    preview=preview,
                    version_path=version_path,
                    subset_label=subset_label,
                )
                if per_l1_latex:
                    per_l1_path = table_dir / "table_per_l1.tex"
                    write_table(per_l1_path, per_l1_latex)
                    logger.info("Saved per-L1 table: %s", per_l1_path)


def _load_bias_summary(mode: str) -> pd.DataFrame:
    """Load the bias_correlation_summary.csv for a simplified-pipeline mode
    label ({"org", "seen", "new"}), filtered to BIAS_DISTANCE_TYPE."""
    path = BIAS_SUMMARY_DIR / mode / BIAS_SUMMARY_FILENAME
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "distance_type" in df.columns:
        df = df[df["distance_type"] == BIAS_DISTANCE_TYPE]
    return df


def _filter_bias_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the (dataset, version_path, content_mode) rows shown in the table."""
    if df.empty:
        return df
    keep = []
    for ds, spec in BIAS_DATASET_FILTERS.items():
        sub = df[(df["dataset"] == ds)
                 & (df["version_path"].isin(spec["version_paths"]))
                 & (df["content_mode"].isin(spec["content_modes"]))
                 & (df["preview"].isin(spec["previews"]))]
        if not sub.empty:
            keep.append(sub)
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]


# ── Combined (r_t, bias_r, bias_p) triplet table ─────────────────────────────

# (display_label, simplified-pipeline mode key) — the first entry is the raw
# EyeScore; the rest are debiased variants grouped under "Debiased EyeScore".
BIAS_TRIPLET_MODES = [
    ("EyeScore", "org"),
    ("Seen Language", "seen"),
    ("New Language", "new"),
]

# Per-debias-target triplet modes. Bias-summary keys map to the simplified-
# pipeline outputs written by run_simplified_bias_plots:
#   seen_michtest / new_michtest → eye_score debiased against michtest_score
#   seen_profagg  / new_profagg  → eye_score debiased against proficiency_agg
# The corresponding eye_score CSVs are produced by the Correction instances
# declared in src/methods/EyeScore/calculation.py.
BIAS_TRIPLET_MODES_MICHTEST = [
    ("EyeScore", "org"),
    ("Seen Language", "seen_michtest"),
    ("New Language", "new_michtest"),
]
BIAS_TRIPLET_MODES_PROFAGG = [
    ("EyeScore", "org"),
    ("Seen Language", "seen_profagg"),
    ("New Language", "new_profagg"),
]

# Inline math label used in the (multi-)target triplet table headers in place
# of the older "Debiased" string. Matches the paper template.
BIAS_TRIPLET_DEBIASED_HEADER = r"$\mathrm{EyeScore}^{\mathrm{db}}$"

# Restrict the triplet table to a single target column.
BIAS_TRIPLET_TARGET = "lextale_score"


def generate_bias_triplet_table(
    save_path: Path,
    modes: Optional[list] = None,
    content_mode: str = "unseen",
    dataset_targets: Optional[dict] = None,
    combine_scopes: bool = False,
) -> str:
    """LaTeX table mixing EyeScore-target correlation with language-bias
    correlation.

    Layout:
        rows = feature_set
        cols = dataset (outer) × target_col (middle) × mode {org, inside, across}
                (inner) × (r_t, r_b, p_b) (innermost)

    `dataset_targets` controls which target_col(s) each dataset shows. When
    None → single-target legacy view (every dataset shows `BIAS_TRIPLET_TARGET`,
    LexTALE by default) with the EyeScore / Debiased header grouping. When any
    dataset declares more than one target → multi-target view mirroring the
    combined EyeScore table: header rows become (dataset, target_col, mode).

    `content_mode` selects which slice of the bias / eval CSVs to surface —
    "all" (default), "seen", or "unseen". For OneStop these correspond to the
    per-content-group cuts; for MECO to the averaged same-half/cross-half
    eye_scores. Rows for a given dataset are dropped if its filter in
    `BIAS_DATASET_FILTERS` doesn't include `content_mode`.

    When `modes` contains a single (label, key) pair, the "Debiased EyeScore"
    group header is omitted and the table collapses to a flat
    feature × (dataset × target_col × (r_t, r_b, sig)) layout.
    """
    if modes is None:
        modes = BIAS_TRIPLET_MODES
    if dataset_targets is None:
        dataset_targets = {ds: [BIAS_TRIPLET_TARGET] for ds in BIAS_CORR_DATASETS}

    # Multi-target = at least one dataset asks for more than one target. The
    # single-target case uses an EyeScore | Debiased grouping; the multi-target
    # case uses a dataset / target / mode header instead so the extra axis has
    # somewhere to live.
    multi_target = any(len(tcs) > 1 for tcs in dataset_targets.values())

    # When combining scopes, each cell holds (Fixed=seen, Any=unseen) values
    # rendered side-by-side. Otherwise the single content_mode is used.
    scope_specs = (
        [("Fixed", "seen"), ("Any", "unseen")]
        if combine_scopes
        else [(None, content_mode)]
    )

    # cells[(mode_label, dataset, target_col, feature_set, scope_label)] -> (bias_r, bias_p)
    # scope_label is None in the single-scope path, "Fixed"/"Any" when combined.
    cells = {}
    for mode_label, mode_key in modes:
        bias_df_all = _load_bias_summary(mode_key)
        if bias_df_all.empty:
            logger.warning(
                "Missing bias summary for mode %s; triplet column will be blank",
                mode_key,
            )
            continue
        bias_df_all = _filter_bias_rows(bias_df_all)

        for scope_label, cm in scope_specs:
            bias_df = bias_df_all[bias_df_all["content_mode"] == cm]
            for ds, tcs in dataset_targets.items():
                for tc in tcs:
                    b_slice = bias_df[(bias_df["dataset"] == ds) & (bias_df["target_col"] == tc)]
                    bias_lookup = {r["feature_set"]: (r["pearson_r"], r["pearson_p"])
                                   for _, r in b_slice.iterrows()}
                    for fs in BIAS_FEATURE_SET_ORDER:
                        bias = bias_lookup.get(fs)
                        if bias is None:
                            continue
                        cells[(mode_label, ds, tc, fs, scope_label)] = (bias[0], bias[1])

    if not cells:
        logger.warning("No data for bias triplet table; skipping")
        return ""

    # The combined "All Combined" feature set is intentionally omitted from
    # the triplet table — it duplicates information already conveyed by the
    # individual feature-set rows and clutters the per-row hline grouping.
    _SKIP_FS = {"READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM"}
    scope_labels_present = [sl for sl, _ in scope_specs]
    fs_present = [
        fs for fs in BIAS_FEATURE_SET_ORDER
        if fs not in _SKIP_FS
        and any((m, d, tc, fs, sl) in cells
                for m, _ in modes
                for d in dataset_targets
                for tc in dataset_targets[d]
                for sl in scope_labels_present)
    ]
    datasets_present = [
        d for d in BIAS_CORR_DATASETS
        if d in dataset_targets
        and any((m, d, tc, fs, sl) in cells
                for m, _ in modes
                for tc in dataset_targets[d]
                for fs in fs_present
                for sl in scope_labels_present)
    ]

    def _n_stars(p):
        """Significance level → number of stars (0–3)."""
        if pd.isna(p):
            return 0
        if p < 0.001:
            return 3
        if p < 0.01:
            return 2
        if p < 0.05:
            return 1
        return 0

    n_modes = len(modes)
    # Flat column order so col_idx and column spec stay in sync regardless of
    # how many targets each dataset declares.
    columns = []  # list of (ds, tc, m_i, mode_label)
    for ds in datasets_present:
        for tc in dataset_targets[ds]:
            for m_i, (mode_label, _) in enumerate(modes):
                columns.append((ds, tc, m_i, mode_label))

    n_data_cols = len(columns)
    n_total_cols = 1 + n_data_cols

    # Column spec: within a dataset block (across targets and modes) use single
    # `|`; between datasets use `||`. Mirrors generate_latex_combined_table.
    # Booktabs column spec: features col `l`, then per-(dataset, target_col)
    # block of `ccc` (one column per mode), with single `|` separating blocks.
    # No `|` after `l`, no `||` between datasets — vertical rules within a row
    # are placed via multicolumn alignment overrides instead so each header row
    # gets only the dividers it needs.
    col_groups = []
    for ds in datasets_present:
        for _tc in dataset_targets[ds]:
            col_groups.append("c" * n_modes)
    col_spec = "l" + "|".join(col_groups)

    # Only go two-column wide when we actually have both datasets — a
    # multi-target single-dataset table (e.g. MECO Composite+LexTALE) still
    # fits in one column once resized. Single-target keeps \columnwidth too.
    span_both_columns = multi_target and len(datasets_present) > 1
    table_env = "table*" if span_both_columns else "table"
    resize_width = "\\textwidth" if span_both_columns else "\\columnwidth"
    lines = [
        f"\\begin{{{table_env}}}[ht!]",
        "\\centering",
        f"\\resizebox{{{resize_width}}}{{!}}{{%",
    ]
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    ds_display = {"Meco": "MECO"}
    # Per-dataset target display map (e.g. {"Meco": {"lextale_score": "LexTALE", ...}}).
    target_display = DATASET_TARGET_COL_DISPLAY
    n_datasets = len(datasets_present)

    # Header row 1 (Datasets): each dataset spans n_targets × n_modes cols.
    # `c|` for all but the last dataset → a single vertical rule at the
    # dataset boundary, no rules between targets within a dataset.
    header1 = [""]
    for ds_idx, ds in enumerate(datasets_present):
        ds_name = ds_display.get(ds, ds)
        span = len(dataset_targets[ds]) * n_modes
        is_last_ds = (ds_idx == n_datasets - 1)
        align = "c" if is_last_ds else "c|"
        header1.append(
            f"\\multicolumn{{{span}}}{{{align}}}{{\\textbf{{{ds_name}}}}}"
        )
    lines.append(" & ".join(header1) + " \\\\")

    # Per-template: the unseen ("Any Text") variant uses the new paper wording
    # with `$\mathrm{EyeScore}^{\mathrm{db}}$` and the explanatory caption;
    # seen/all/combined keep the older "Debiased" label and shorter caption.
    use_new_paper_style = (content_mode == "unseen" and not combine_scopes)
    debiased_label = (
        BIAS_TRIPLET_DEBIASED_HEADER if use_new_paper_style else "Debiased"
    )

    if multi_target:
        n_debiased = n_modes - 1

        # Header row 2 (Targets): per-dataset target_col labels, each spanning
        # n_modes. Vertical rule appears only at the dataset boundary, not at
        # internal target boundaries (so targets within a dataset visually
        # share a single dataset block above).
        header2 = [""]
        for ds_idx, ds in enumerate(datasets_present):
            ds_tdisplay = target_display.get(ds, {})
            n_tcs = len(dataset_targets[ds])
            is_last_ds = (ds_idx == n_datasets - 1)
            for tc_idx, tc in enumerate(dataset_targets[ds]):
                label = _display(tc, ds_tdisplay)
                is_last_tc_in_ds = (tc_idx == n_tcs - 1)
                # | only at dataset boundary (= last target of non-last ds)
                align = "c|" if (is_last_tc_in_ds and not is_last_ds) else "c"
                header2.append(
                    f"\\multicolumn{{{n_modes}}}{{{align}}}{{{label}}}"
                )
        lines.append(" & ".join(header2) + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

        # Header row 3 (EyeScore | Debiased grouping per target): EyeScore is
        # always `c`; Debiased is `c|` everywhere except the very last
        # (dataset, target) block. Net effect: rule between targets within and
        # between datasets, since each target's pair of cells is a sub-group.
        header3 = [""]
        for ds_idx, ds in enumerate(datasets_present):
            n_tcs = len(dataset_targets[ds])
            is_last_ds = (ds_idx == n_datasets - 1)
            for tc_idx, _tc in enumerate(dataset_targets[ds]):
                is_last_tc_in_ds = (tc_idx == n_tcs - 1)
                is_very_last = is_last_ds and is_last_tc_in_ds
                d_align = "c" if is_very_last else "c|"
                header3.append("\\multicolumn{1}{c}{EyeScore}")
                header3.append(
                    f"\\multicolumn{{{n_debiased}}}{{{d_align}}}{{{debiased_label}}}"
                )
        lines.append(" & ".join(header3) + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

        # Header row 4 (mode sub-labels): plain cells. Blank under EyeScore,
        # italic short labels (Seen L1 / New L1) under Debiased.
        header4 = ["\\textbf{Features}"]
        for ds in datasets_present:
            for _tc in dataset_targets[ds]:
                header4.append("")
                for mode_label, _ in modes[1:]:
                    short = mode_label.replace("Language", "L1")
                    header4.append(f"\\textit{{{short}}}")
        lines.append(" & ".join(header4) + " \\\\")
        lines.append(f"\\midrule \\cmidrule{{1-{n_total_cols}}}")
    else:
        # Legacy single-target layout: same booktabs format but with row 2 =
        # EyeScore | Debiased (no target row). Each dataset has exactly one
        # (EyeScore, Debiased) pair.
        n_debiased = n_modes - 1

        header2 = [""]
        for ds_idx, _ in enumerate(datasets_present):
            is_last_ds = (ds_idx == n_datasets - 1)
            d_align = "c" if is_last_ds else "c|"
            header2.append("\\multicolumn{1}{c}{EyeScore}")
            header2.append(
                f"\\multicolumn{{{n_debiased}}}{{{d_align}}}{{{debiased_label}}}"
            )
        lines.append(" & ".join(header2) + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

        header3 = ["\\textbf{Features}"]
        for _ in datasets_present:
            header3.append("")
            for mode_label, _ in modes[1:]:
                short = mode_label.replace("Language", "L1")
                header3.append(f"\\textit{{{short}}}")
        lines.append(" & ".join(header3) + " \\\\")
        lines.append(f"\\midrule \\cmidrule{{1-{n_total_cols}}}")

    # Reuse the bias-correlation feature grouping so hlines match other tables.
    bias_corr_groups = [
        ["READING_SPEED"],
        ["FIXATION_METRICS", "WP_COEFS_NO_NORM", "S_CLUSTERS_NO_NORM"],
        ["TRANSITIONS", "WFC"],
        ["READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM"],
    ]
    fs_group_idx = {fs: i for i, g in enumerate(bias_corr_groups) for fs in g}

    # First pass: build raw (value, n_stars) per (fs, ds, tc, mode_idx) and
    # find the maximum star count per data column so every cell in that column
    # gets padded to the same trailing width via \phantom{^{...}}. When
    # combining scopes the max is taken across both scopes so the Fixed and
    # Any sides of every cell stay vertically aligned.
    raw_cells: dict = {}  # (fs, ds, tc, m_i, scope_label) -> (b_r, ns) | None
    max_stars_per_col: list = [0] * n_data_cols
    for fs in fs_present:
        for col_idx, (ds, tc, m_i, mode_label) in enumerate(columns):
            for scope_label in scope_labels_present:
                cell = cells.get((mode_label, ds, tc, fs, scope_label))
                if cell is None or pd.isna(cell[0]):
                    raw_cells[(fs, ds, tc, m_i, scope_label)] = None
                    continue
                b_r, b_p = cell
                ns = _n_stars(b_p)
                raw_cells[(fs, ds, tc, m_i, scope_label)] = (b_r, ns)
                if ns > max_stars_per_col[col_idx]:
                    max_stars_per_col[col_idx] = ns

    def _fmt_cell(b_r: float, n_stars: int, max_stars: int) -> str:
        """One numeric cell: $sign-pad value stars star-phantom$."""
        rounded = round(b_r, 2)
        # Treat exact-zero as non-negative so we don't print "-0.00".
        if rounded == 0:
            value_str = "0.00"
            sign_pad = "\\phantom{-}"
        elif rounded < 0:
            value_str = f"{rounded:.2f}"
            sign_pad = ""
        else:
            value_str = f"{rounded:.2f}"
            sign_pad = "\\phantom{-}"
        stars = "^{" + ("*" * n_stars) + "}" if n_stars > 0 else ""
        deficit = max_stars - n_stars
        star_pad = "\\phantom{^{" + ("*" * deficit) + "}}" if deficit > 0 else ""
        return f"${sign_pad}{value_str}{stars}{star_pad}$"

    for i, fs in enumerate(fs_present):
        if i > 0 and fs_group_idx[fs] != fs_group_idx[fs_present[i - 1]]:
            lines.append("\\midrule")

        row_cells = [_display(fs, FEATURE_SET_DISPLAY)]
        for col_idx, (ds, tc, m_i, mode_label) in enumerate(columns):
            if combine_scopes:
                parts = []
                for scope_label in scope_labels_present:
                    rc = raw_cells.get((fs, ds, tc, m_i, scope_label))
                    if rc is None:
                        # FIXED_TEXT_ONLY features have no Any value — render NA
                        # on that side rather than dropping the whole row.
                        parts.append(
                            "NA" if (scope_label == "Any"
                                     and fs in FIXED_TEXT_ONLY_FEATURE_SETS)
                            else "--"
                        )
                    else:
                        b_r, ns = rc
                        parts.append(_fmt_cell(b_r, ns, max_stars_per_col[col_idx]))
                row_cells.append(" / ".join(parts))
            else:
                rc = raw_cells.get((fs, ds, tc, m_i, None))
                if rc is None:
                    row_cells.append("--")
                else:
                    b_r, ns = rc
                    row_cells.append(_fmt_cell(b_r, ns, max_stars_per_col[col_idx]))
        # Last data row gets a trailing \bottomrule on the same line, matching
        # booktabs convention.
        is_last_row = (i == len(fs_present) - 1)
        row_suffix = " \\\\ \\bottomrule" if is_last_row else " \\\\"
        lines.append(" & ".join(row_cells) + row_suffix)

    lines.append("\\end{tabular}%")
    lines.append("}")  # close \resizebox
    # Calibration target the debiasing regression was fitted on. Inferred
    # from the mode keys passed in (LexTALE is the default if no calibrated
    # _michtest/_profagg variant is referenced).
    calib_phrase = "LexTALE"
    for _label, key in modes:
        if "_michtest" in key:
            calib_phrase = "MichTest"
            break
        if "_profagg" in key:
            calib_phrase = "Composite"
            break

    if combine_scopes:
        cm_label = "-combined"
        regime_phrase = (
            "across regimes. Each cell shows \\textit{Fixed\\,/\\,Any}: "
            "the Fixed Text regime (eye movements for the test participant's "
            "texts are available from prior participants) and the Any Text "
            "regime (no eye movements for the test participant's texts)."
        )
    elif content_mode == "unseen":
        cm_label = ""
        regime_phrase = "in the Any Text regime."
    else:
        cm_label = f"-{content_mode}"
        regime_phrase = {
            "seen": "in the \\textbf{Fixed Text} regime.",
            "all": "aggregated across all texts.",
        }.get(content_mode, f"({content_mode}).")

    if use_new_paper_style:
        lines.append(
            "\\caption{\\textbf{EyeScore L1 Bias} " + regime_phrase + " "
            "'EyeScore' is the Pearson $r$ coefficient of EyeScore L1 bias "
            "with the linguistic distance of the participant's L1 to English. "
            "$\\mathrm{EyeScore}^{\\mathrm{db}}$ is this coefficient after "
            f"debiasing using a residual regression model fitted on {calib_phrase}. "
            "Seen / New L1: the data for computing the debiased score included / "
            "did not include participants from the same L1 as the test "
            "participant. Statistical significance for Pearson $r$ coefficients "
            "of L1 bias:  $^{*}\\,p<0.05$, $^{**}\\,p<0.01$, $^{***}\\,p<0.001$.}"
        )
    else:
        lines.append(
            "\\caption{\\textbf{EyeScore L1 Bias} " + regime_phrase + " "
            "Pearson $r$ coefficients of EyeScore L1 bias, these coefficients "
            "after debiasing. Seen / New L1: the data for computing the "
            "debiasing score correction included / did not include "
            "participants from the same L1 as the test participant. "
            "Statistical significance for Pearson $r$ coefficients of L1 "
            "bias:  $^{*}\\,p<0.05$, $^{**}\\,p<0.01$, $^{***}\\,p<0.001$.}"
        )
    lines.append(f"\\label{{tab:lang-bias-triplet{cm_label}}}")
    lines.append(f"\\end{{{table_env}}}")

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex)
    logger.info("Saved bias-triplet table: %s", save_path)
    return latex


# ── Raw EyeScore vs L1 distance table ────────────────────────────────────────


def generate_eyescore_distance_table(
    save_path: Path,
    feature_sets: Optional[List[str]] = None,
    target_col_for_filter: str = "lextale_score",
    distance_type: str = "syntactic+genetic",
    results_dir: Optional[Path] = None,
    y_col: str = "eye_score",
    include_lextale_row: bool = True,
) -> str:
    """LaTeX table of Pearson r between a per-participant y variable and L1
    typological distance to English, mirroring the layout of the
    eyescore_vs_distance plot (no proficiency residualization).

    `y_col` picks the y-axis variable:
      - "eye_score" (default): the EyeScore itself; cell value depends on
        (dataset, content_mode, feature_set).
      - "lextale_score": the LexTALE score; cell value depends only on the
        participant subset (no feature_set dependence), but the table still
        breaks it out by (dataset, content_mode, feature_set) so the
        participant pool exactly mirrors the EyeScore table.

    `include_lextale_row`: when y_col="eye_score", prepend a LexTALE row at
    the top showing r(lextale_score, distance) per dataset as a baseline,
    spanning Fixed+Any since LexTALE doesn't depend on the eye_score regime.

    Layout: rows = feature_set, outer cols = dataset (MECO, OneStop), inner
    cols = scope (Fixed=seen, Any=unseen). The participant pool is filtered
    to LexTALE-eligible participants so it matches the plot's sample.
    """
    from scipy.stats import pearsonr

    if feature_sets is None:
        feature_sets = [
            "READING_SPEED",
            "FIXATION_METRICS",
            "S_CLUSTERS_NO_NORM",
            "WP_COEFS_NO_NORM",
        ]
    if results_dir is None:
        results_dir = RESULTS_DIR

    scope_specs = [("Fixed", "seen"), ("Any", "unseen")]
    datasets = ["Meco", "OneStop"]

    cells: dict = {}  # (dataset, feature_set, scope_label) -> (r, p, n) | None
    for dataset in datasets:
        spec = BIAS_DATASET_FILTERS[dataset]
        preview = spec["previews"][0]
        version_path = spec["version_paths"][0]
        meta = load_metadata(dataset, version_path)
        if meta.empty:
            logger.warning("No metadata for %s/%s", dataset, version_path)
            continue
        for scope_label, content_mode in scope_specs:
            for fs in feature_sets:
                csv_path = eye_score_csv_path(
                    results_dir, dataset=dataset, preview=preview,
                    content_mode=content_mode, version_path=version_path,
                    feature_set=fs, variant=RAW_VARIANT,
                )
                if not csv_path.exists():
                    cells[(dataset, fs, scope_label)] = None
                    continue
                merged = _merge_eyescore_metadata(csv_path, meta, target_col_for_filter)
                if merged is None or merged.empty:
                    cells[(dataset, fs, scope_label)] = None
                    continue
                if y_col not in merged.columns:
                    cells[(dataset, fs, scope_label)] = None
                    continue
                distances = compute_typological_distances(
                    merged["L1"].unique().tolist(), distance_type=distance_type,
                )
                merged = merged.copy()
                merged["distance"] = merged["L1"].map(distances)
                merged = merged.dropna(subset=["distance", y_col])
                if len(merged) < 3 or merged["L1"].nunique() < 2:
                    cells[(dataset, fs, scope_label)] = None
                    continue
                r, p = pearsonr(merged[y_col], merged["distance"])
                cells[(dataset, fs, scope_label)] = (float(r), float(p), len(merged))

    def _n_stars(p_val: float) -> int:
        if pd.isna(p_val):
            return 0
        if p_val < 0.001:
            return 3
        if p_val < 0.01:
            return 2
        if p_val < 0.05:
            return 1
        return 0

    columns = [(ds, sl) for ds in datasets for sl, _ in scope_specs]
    max_stars_per_col = [0] * len(columns)
    for col_idx, (ds, sl) in enumerate(columns):
        for fs in feature_sets:
            cell = cells.get((ds, fs, sl))
            if cell is not None:
                max_stars_per_col[col_idx] = max(
                    max_stars_per_col[col_idx], _n_stars(cell[1]),
                )

    def _fmt_cell(r: float, p_val: float, max_stars: int) -> str:
        ns = _n_stars(p_val)
        rounded = round(r, 2)
        if rounded == 0:
            value_str = "0.00"
            sign_pad = "\\phantom{-}"
        elif rounded < 0:
            value_str = f"{rounded:.2f}"
            sign_pad = ""
        else:
            value_str = f"{rounded:.2f}"
            sign_pad = "\\phantom{-}"
        stars = "^{" + ("*" * ns) + "}" if ns > 0 else ""
        deficit = max_stars - ns
        star_pad = "\\phantom{^{" + ("*" * deficit) + "}}" if deficit > 0 else ""
        return f"${sign_pad}{value_str}{stars}{star_pad}$"

    fs_present = [
        fs for fs in feature_sets
        if any(cells.get((ds, fs, sl)) is not None
               for ds in datasets for sl, _ in scope_specs)
    ]
    if not fs_present:
        logger.warning("No data for eyescore-vs-distance table; skipping")
        return ""

    # Optional LexTALE baseline row: r(lextale_score, distance) per dataset,
    # spanning Fixed+Any because LexTALE itself doesn't depend on the eye_score
    # regime. We pick any feature_set's participant pool to compute it (they
    # all share the same LexTALE-eligible participants within a (dataset,
    # scope) cell). Skipped when y_col is already lextale_score.
    lextale_row_cells = None
    if include_lextale_row and y_col == "eye_score":
        lextale_row_cells = []
        for dataset in datasets:
            spec = BIAS_DATASET_FILTERS[dataset]
            preview = spec["previews"][0]
            version_path = spec["version_paths"][0]
            meta = load_metadata(dataset, version_path)
            r_val = None
            p_val = None
            if not meta.empty:
                # Use the first feature_set with a present CSV under either
                # scope — LexTALE values don't depend on the eye_score column,
                # so any (scope, feature_set) participant pool works.
                for content_mode in ("seen", "unseen", "all"):
                    for fs in (feature_sets or []):
                        csv_path = eye_score_csv_path(
                            results_dir, dataset=dataset, preview=preview,
                            content_mode=content_mode, version_path=version_path,
                            feature_set=fs, variant=RAW_VARIANT,
                        )
                        if not csv_path.exists():
                            continue
                        merged = _merge_eyescore_metadata(csv_path, meta, target_col_for_filter)
                        if merged is None or merged.empty:
                            continue
                        distances = compute_typological_distances(
                            merged["L1"].unique().tolist(), distance_type=distance_type,
                        )
                        merged = merged.copy()
                        merged["distance"] = merged["L1"].map(distances)
                        merged = merged.dropna(subset=["distance", "lextale_score"])
                        if len(merged) < 3 or merged["L1"].nunique() < 2:
                            continue
                        r_val, p_val = pearsonr(merged["lextale_score"], merged["distance"])
                        break
                    if r_val is not None:
                        break
            if r_val is None:
                lextale_row_cells.append("--")
            else:
                # Use max stars across both sub-columns of this dataset so the
                # LexTALE cell pads consistently with the feature-set rows below.
                ds_col_indices = [i for i, (d, _) in enumerate(columns) if d == dataset]
                ms = max((max_stars_per_col[i] for i in ds_col_indices), default=0)
                ms = max(ms, _n_stars(p_val))
                lextale_row_cells.append(_fmt_cell(float(r_val), float(p_val), ms))

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcc|cc}",
        "\\toprule",
        " & \\multicolumn{2}{c|}{\\textbf{MECO}} & "
        "\\multicolumn{2}{c}{\\textbf{OneStopL2}} \\\\",
    ]

    if lextale_row_cells is not None:
        lines.append("\\midrule")
        lines.append(
            "LexTALE & "
            f"\\multicolumn{{2}}{{c|}}{{{lextale_row_cells[0]}}} & "
            f"\\multicolumn{{2}}{{c}}{{{lextale_row_cells[1]}}} \\\\"
        )
        lines.append("\\midrule")

    lines += [
        "\\textbf{Features} & \\textit{Fixed} & \\textit{Any} & \\textit{Fixed} & \\textit{Any} \\\\",
        "\\midrule",
    ]

    # Insert a midrule between feature_set groups so WPM is visually separated
    # from the per-AOI rows, matching the existing triplet table layout.
    for i, fs in enumerate(fs_present):
        if i > 0 and FEATURE_SET_GROUP_INDEX.get(fs) != FEATURE_SET_GROUP_INDEX.get(fs_present[i - 1]):
            lines.append("\\midrule")
        row_cells = [_display(fs, FEATURE_SET_DISPLAY)]
        for col_idx, (ds, sl) in enumerate(columns):
            cell = cells.get((ds, fs, sl))
            if cell is None:
                row_cells.append(
                    "NA" if (sl == "Any" and fs in FIXED_TEXT_ONLY_FEATURE_SETS)
                    else "--"
                )
            else:
                r, p_val, _n = cell
                row_cells.append(_fmt_cell(r, p_val, max_stars_per_col[col_idx]))
        is_last = (i == len(fs_present) - 1)
        suffix = " \\\\ \\bottomrule" if is_last else " \\\\"
        lines.append(" & ".join(row_cells) + suffix)

    lines.append("\\end{tabular}%")
    lines.append("}")
    if y_col == "eye_score":
        y_label = "EyeScore"
        caption_title = "Raw EyeScore L1 Bias"
        latex_label = "tab:lang-bias-eyescore-distance"
    elif y_col == "lextale_score":
        y_label = "LexTALE"
        caption_title = "LexTALE L1 Bias"
        latex_label = "tab:lang-bias-lextale-distance"
    else:
        y_label = y_col
        caption_title = f"{y_col} L1 Bias"
        latex_label = f"tab:lang-bias-{_slug(y_col)}-distance"
    lines.append(
        "\\caption{\\textbf{" + caption_title + ".} Pearson $r$ between "
        "per-participant " + y_label + " and the linguistic distance from the "
        "participant's L1 to English. ``Fixed'' is the Fixed Text regime in "
        "which all the eye movement data is for the same texts presented to "
        "the test participant. ``Any'' is the Any Text regime in which no eye "
        "movement data is available for the texts of the test participant. "
        "Statistical significance: $^{*}\\,p<0.05$, $^{**}\\,p<0.01$, "
        "$^{***}\\,p<0.001$.}"
    )
    lines.append(f"\\label{{{latex_label}}}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex)
    logger.info("Saved %s-vs-distance table: %s", y_col, save_path)
    return latex


# ── Per-(feature-set, dataset) bias correlation table ────────────────────────

# Datasets shown as outer column groups in the bias-correlation table.
# Order matters — drives column order.
BIAS_CORR_DATASETS = ["Meco", "OneStop"]

# Per-dataset target columns (inner column groups, in display order).
# Each (dataset, target_col) pair gets its own (r, p) sub-pair.
BIAS_CORR_DATASET_TARGETS = {
    "Meco": ["lextale_score", "proficiency_agg"],
    "OneStop": ["lextale_score", "michtest_score"],
}

# Distance type used by the per-(feature-set, dataset) bias-correlation table.
# When this differs from BIAS_DISTANCE_TYPE the table generator recomputes the
# Pearson r/p on-the-fly from the eye_score CSVs (the cached summary CSV only
# stores BIAS_DISTANCE_TYPE).
BIAS_CORR_DISTANCE_TYPE = "syntactic+genetic"

# Variants of the bias-correlation table, in (filename, mode, label) form,
# where mode is a simplified-pipeline key from BIAS_MODE_TO_EVAL_PATH.
BIAS_CORR_TABLE_VARIANTS = [
    ("bias_correlation_raw.tex", "org", "raw EyeScore"),
    (
        "bias_correlation_inside_langs_corrected.tex",
        "seen",
        "Seen Language (inside-langs typology-calibrated) EyeScore",
    ),
    (
        "bias_correlation_across_langs_corrected.tex",
        "new",
        "New Language (across-langs typology-calibrated) EyeScore",
    ),
]


# (dataset, preview, version_path) triples used by the on-the-fly recomputation
# path. Mirrors BIAS_DATASET_FILTERS but expanded with the preview directory
# needed to locate the eye_score CSV files on disk.
_BIAS_CORR_DATASET_LAYOUT = [
    ("Meco", "All", "fully_agg/all/all"),
    ("OneStop", "Gathering", "fully_agg/ordinary/all"),
]


def _eyescore_variant_name(scheme: str, summary_variant: str) -> str:
    """Map a (scheme, summary_variant) pair from the bias-summary CSV layout
    onto the variant key used to locate the eye_score CSV file on disk.
    """
    if summary_variant == "org":
        return RAW_VARIANT
    if summary_variant == "typo_calibrated":
        return SPLIT_SCHEME_TO_VARIANT[scheme]
    return summary_variant


def _compute_bias_correlation_rows(
    scheme: str,
    summary_variant: str,
    dataset_targets: dict,
    content_mode: str,
    distance_type: str,
) -> pd.DataFrame:
    """Recompute per-(dataset, target_col, feature_set) bias correlation for
    an arbitrary distance_type by walking eye_score CSVs directly. Used when
    the requested distance_type isn't cached in bias_correlation_summary.csv.

    `dataset_targets` maps dataset -> list of target_col names. Only the
    listed (dataset, target_col) pairs are computed.

    Returns rows with the same (dataset, target_col, feature_set, pearson_r,
    pearson_p, spearman_r, spearman_p) schema the cached summary CSV uses.
    """
    # Late import to avoid pulling scipy into module import time.
    from scipy.stats import pearsonr, spearmanr

    eye_variant = _eyescore_variant_name(scheme, summary_variant)
    rows = []
    for dataset, preview, version_path in _BIAS_CORR_DATASET_LAYOUT:
        target_cols = dataset_targets.get(dataset, [])
        if not target_cols:
            continue
        meta = load_metadata(dataset, version_path)
        if meta.empty:
            continue
        for target_col in target_cols:
            for fs in BIAS_FEATURE_SET_ORDER:
                csv_path = eye_score_csv_path(
                    RESULTS_DIR, dataset, preview, content_mode, version_path,
                    fs, eye_variant,
                )
                if not csv_path.exists():
                    continue
                merged = _merge_eyescore_metadata(csv_path, meta, target_col)
                if merged is None:
                    continue

            # OLS fit: eye_score ~ target_col, then per-participant residual.
            # Mirrors plot_residual_vs_distance in language_bias.py.
            slope, intercept = np.polyfit(merged[target_col], merged["eye_score"], 1)
            residual = merged["eye_score"] - (slope * merged[target_col] + intercept)

            distances = compute_typological_distances(
                merged["L1"].unique().tolist(), distance_type=distance_type,
            )
            participant = pd.DataFrame({"L1": merged["L1"], "residual": residual})
            participant["distance"] = participant["L1"].map(distances)
            participant = participant.dropna(subset=["distance"])
            if len(participant) < 3 or participant["L1"].nunique() < 2:
                continue

            pr, pp = pearsonr(participant["distance"], participant["residual"])
            sr, sp = spearmanr(participant["distance"], participant["residual"])
            rows.append({
                "dataset": dataset,
                "feature_set": fs,
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
            })
    return pd.DataFrame(rows)


def _render_per_feature_dataset_table(
    df: pd.DataFrame,
    dataset_targets: dict,
    value_col: str,
    p_col: str,
    value_header: str,
    save_path: Path,
    caption: str,
    log_label: str,
    single_column: bool = False,
) -> str:
    """Render a feature-set × (dataset × target_col) LaTeX table where each
    (dataset, target_col) cell holds a (value, p) pair. Bolds the smallest
    $|value|$ per (dataset, target_col) column (least language-bias)."""
    fs_present = [
        fs for fs in BIAS_FEATURE_SET_ORDER if fs in set(df["feature_set"].unique())
    ]
    if not fs_present:
        return ""

    # Local feature-set groups for hline placement (the predictions-style
    # FEATURE_SET_GROUPS doesn't include the combined set, which we want as
    # its own trailing group here).
    bias_corr_groups = [
        ["READING_SPEED"],
        ["FIXATION_METRICS", "WP_COEFS_NO_NORM", "S_CLUSTERS_NO_NORM"],
        ["TRANSITIONS", "WFC"],
        ["READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM"],
    ]
    fs_group_idx = {fs: i for i, group in enumerate(bias_corr_groups) for fs in group}

    datasets_present = [d for d in BIAS_CORR_DATASETS if d in dataset_targets]

    lookup = {
        (row["dataset"], row["target_col"], row["feature_set"]):
            (row[value_col], row[p_col])
        for _, row in df.iterrows()
    }

    # Smallest |value| per (dataset, target_col) column = least language-bias.
    best_abs_val = {}
    for ds in datasets_present:
        for tc in dataset_targets[ds]:
            candidates = [
                abs(lookup[(ds, tc, fs)][0])
                for fs in fs_present
                if (ds, tc, fs) in lookup and not pd.isna(lookup[(ds, tc, fs)][0])
            ]
            best_abs_val[(ds, tc)] = min(candidates) if candidates else None

    # Column spec: per dataset, an outer ||-separated block; within a dataset,
    # each target_col gets a |-separated "c c" sub-block.
    dataset_blocks = []
    for ds in datasets_present:
        n_targets = len(dataset_targets[ds])
        dataset_blocks.append("|".join(["c c"] * n_targets))
    col_spec = "l||" + "||".join(dataset_blocks)

    table_env = "table" if single_column else "table*"
    lines = [f"\\begin{{{table_env}}}[ht!]", "\\centering", "\\small"]
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header row 1: dataset names spanning all their target sub-columns.
    header1 = [""]
    for ds in datasets_present:
        span = 2 * len(dataset_targets[ds])
        header1.append(f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{ds}}}}}")
    lines.append(" & ".join(header1) + " \\\\")

    # Header row 2: target_col display label spanning its (r, p) pair.
    header2 = [""]
    for ds in datasets_present:
        for tc in dataset_targets[ds]:
            label = BIAS_TARGET_DISPLAY.get(tc, tc)
            header2.append(f"\\multicolumn{{2}}{{c}}{{\\textit{{{label}}}}}")
    lines.append(" & ".join(header2) + " \\\\")

    # Header row 3: r / p sub-headers.
    header3 = ["\\textbf{Features}"]
    for ds in datasets_present:
        for _tc in dataset_targets[ds]:
            header3.extend([value_header, "$p$"])
    lines.append(" & ".join(header3) + " \\\\")
    lines.append("\\hline")

    for i, fs in enumerate(fs_present):
        if i > 0 and fs_group_idx[fs] != fs_group_idx[fs_present[i - 1]]:
            lines.append("\\hline")

        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for ds in datasets_present:
            for tc in dataset_targets[ds]:
                cell = lookup.get((ds, tc, fs))
                if cell is None or pd.isna(cell[0]):
                    row_parts.extend(["--", "--"])
                    continue
                v, p = cell
                v_str = f"{v:.2f}"
                best = best_abs_val.get((ds, tc))
                if best is not None and np.isclose(abs(v), best, atol=1e-6):
                    v_str = f"\\textbf{{{v_str}}}"
                row_parts.extend([v_str, f"{p:.3f}"])
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\end{{{table_env}}}")

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex)
    logger.info("Saved %s table: %s", log_label, save_path)
    return latex


def generate_bias_correlation_table(
    mode: str,
    save_path: Path,
    description: str,
    dataset_targets: Optional[dict] = None,
    content_mode: str = "all",
    distance_type: str = BIAS_CORR_DISTANCE_TYPE,
    correlation_method: str = "pearson",
    single_column: bool = False,
) -> str:
    """LaTeX table of (r, p) for the per-participant residual vs.
    L1$\\to$English typological-distance correlation.

    Rows are feature sets; outer columns are datasets, each split into one
    sub-block per target_col (driven by `dataset_targets`), each of which is
    a (r, p) pair. ``--`` is rendered for (feature_set, dataset, target_col)
    combinations absent from the summary CSV (e.g. TRANSITIONS / WFC under
    OneStop's content_mode=all, which mixes seen/unseen and excludes
    per-text features).

    `distance_type` selects which URIEL+ distance to correlate residuals
    against. The cached bias_correlation_summary.csv only stores
    `BIAS_DISTANCE_TYPE`; for any other distance the table is recomputed
    on-the-fly from the underlying eye_score CSVs.

    `correlation_method` is either ``"pearson"`` or ``"spearman"``; both are
    available from the cached summary CSV and from the recompute path.
    """
    if correlation_method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown correlation_method: {correlation_method!r}")

    if dataset_targets is None:
        dataset_targets = BIAS_CORR_DATASET_TARGETS

    # Flat list of (dataset, target_col) pairs we want to keep.
    wanted_pairs = {(ds, tc) for ds, tcs in dataset_targets.items() for tc in tcs}
    wanted_targets = {tc for _, tc in wanted_pairs}

    if distance_type == BIAS_DISTANCE_TYPE:
        df = _load_bias_summary(mode)
        if df.empty:
            logger.warning(
                "Bias summary missing for mode %s; skipping bias-correlation table",
                mode,
            )
            return ""
        df = _filter_bias_rows(df)
        df = df[df["target_col"].isin(wanted_targets) & (df["content_mode"] == content_mode)]
        df = df[df.apply(
            lambda r: (r["dataset"], r["target_col"]) in wanted_pairs, axis=1
        )]
    else:
        scheme, summary_variant = BIAS_MODE_TO_EVAL_PATH[mode]
        df = _compute_bias_correlation_rows(
            scheme=scheme, summary_variant=summary_variant,
            dataset_targets=dataset_targets, content_mode=content_mode,
            distance_type=distance_type,
        )

    df = df[df["dataset"].isin(dataset_targets.keys())]
    if df.empty:
        logger.warning(
            "No rows for mode %s after target/content filter; skipping",
            mode,
        )
        return ""

    method_label = {"pearson": "Pearson", "spearman": "Spearman"}[correlation_method]
    value_col = {"pearson": "pearson_r", "spearman": "spearman_r"}[correlation_method]
    p_col = {"pearson": "pearson_p", "spearman": "spearman_p"}[correlation_method]
    if value_col not in df.columns or p_col not in df.columns:
        logger.warning(
            "%s columns missing for mode %s; skipping", method_label, mode,
        )
        return ""
    header_symbol = {"pearson": "$r$", "spearman": r"$\rho$"}[correlation_method]

    caption = (
        f"Language bias of the raw EyeScore: {method_label} $r$ (and $p$) "
        "between per-participant residual and L1 typological distance, "
        "per feature set."
    )
    return _render_per_feature_dataset_table(
        df=df, dataset_targets=dataset_targets,
        value_col=value_col, p_col=p_col,
        value_header=header_symbol, save_path=save_path, caption=caption,
        log_label=f"bias-correlation-{correlation_method}",
        single_column=single_column,
    )


# ── Michigan test parts table ──────────────────────────────────────────────────

MICHIGAN_PARTS_FEATURE_GROUPS = [
    ["READING_SPEED"],
    ["FIXATION_METRICS", "WP_COEFS_NO_NORM", "S_CLUSTERS_NO_NORM"],
]
MICHIGAN_PARTS_FEATURE_SETS = [fs for g in MICHIGAN_PARTS_FEATURE_GROUPS for fs in g]
MICHIGAN_PARTS_GROUP_INDEX = {
    fs: i for i, g in enumerate(MICHIGAN_PARTS_FEATURE_GROUPS) for fs in g
}

# Michigan test parts target columns (overall score first, then individual parts + combinations)
MICHIGAN_PARTS_TARGET_COLS = [TestCols.MICHIGEN_TEST_COL] + MICH_TEST_PARTS_EXTENDED

# Michigan test parts simplified (first and second parts only)
MICHIGAN_PARTS_SIMPLIFIED_TARGET_COLS = [TestCols.MICHIGEN_TEST_COL, "MPT_listen_grammar", "MPT_vocab_read"]

# OneStop comprehension target columns (lextale, michtest, comprehension from regular trials)
ONESTOP_COMPREHENSION_TARGET_COLS = ["lextale_score", "michtest_score", "comprehension_score-regular_trials"]

# OneStop comprehension + michigan parts target columns
ONESTOP_COMPREHENSION_MICHIGAN_TARGET_COLS = MICHIGAN_PARTS_TARGET_COLS + ["comprehension_score-regular_trials"]

# OneStop comprehension + simplified michigan parts target columns
ONESTOP_COMPREHENSION_MICHIGAN_SIMPLIFIED_TARGET_COLS = MICHIGAN_PARTS_SIMPLIFIED_TARGET_COLS + ["comprehension_score-regular_trials"]


def generate_michigan_parts_table(
    eval_csv: Path = None,
    save_dir: Path = None,
    feature_sets: Optional[List[str]] = None,
):
    """Generate Michigan test parts table for OneStop fully_agg with Gathering preview.

    Columns: Michigan test parts (overall + individual parts + combinations)
    Rows: Feature sets
    Shows Pearson correlations only.
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "OneStop"

    df = pd.read_csv(eval_csv)
    df = df[
        (df["dataset"] == "OneStop")
        & (df["preview"] == "Gathering")
        & (df["version_path"] == "fully_agg/ordinary/all")
        & (df["target_col"].isin(MICHIGAN_PARTS_TARGET_COLS))
    ]

    if df.empty:
        logger.warning("No Michigan parts data for OneStop/Gathering/fully_agg")
        return

    # Use available feature sets from data
    available_fs = sorted(df["feature_set"].unique())
    if feature_sets is not None:
        available_fs = [fs for fs in available_fs if fs in feature_sets]
    else:
        # Default: use all except per-text features
        available_fs = [fs for fs in available_fs if fs not in {"TRANSITIONS", "WFC"}]

    # Build the table
    lines = ["\\begin{table*}[ht!]", "\\centering", "\\small"]
    lines.append("\\caption{EyeScore correlation with MichTest parts.}")

    # Column spec: features, then per-target column
    n_targets = len(MICHIGAN_PARTS_TARGET_COLS)
    col_spec = "l||" + "|".join(["c"] * n_targets)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header: target column names
    header = ["\\textbf{Features}"]
    for tc in MICHIGAN_PARTS_TARGET_COLS:
        display = MICHIGAN_PARTS_DISPLAY.get(tc, tc)
        header.append(f"\\textbf{{{display}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")

    # Find best correlations per column
    best = {}
    for tc in MICHIGAN_PARTS_TARGET_COLS:
        vals = []
        for _, row in df[(df["target_col"] == tc)].iterrows():
            if not pd.isna(row["pearson_r"]):
                vals.append(row["pearson_r"])
        best[tc] = max(vals) if vals else None

    # Data rows
    for fi, fs in enumerate(available_fs):
        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for tc in MICHIGAN_PARTS_TARGET_COLS:
            subset = df[(df["feature_set"] == fs) & (df["target_col"] == tc)]
            if not subset.empty:
                r = subset.iloc[0]["pearson_r"]
                if not pd.isna(r):
                    best_val = best.get(tc)
                    is_best = best_val is not None and np.isclose(r, best_val, atol=1e-6)
                    formatted = f"{r:.2f}"
                    if is_best:
                        row_parts.append(f"\\textbf{{{formatted}}}")
                    else:
                        row_parts.append(formatted)
                else:
                    row_parts.append("NA")
            else:
                row_parts.append("NA")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    latex = "\n".join(lines)

    save_dir.mkdir(parents=True, exist_ok=True)
    table_path = save_dir / "michigan_parts.tex"
    write_table(table_path, latex)
    logger.info("Saved table: %s", table_path)


# ── OneStop comprehension tables ───────────────────────────────────────────────

def generate_comprehension_table(
    eval_csv: Path = None,
    save_dir: Path = None,
):
    """Generate comprehension table for OneStop (michtest + lextale + comprehension).

    Shows Pearson correlations for OneStop/Gathering/fully_agg.
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "OneStop"

    df = pd.read_csv(eval_csv)
    df = df[
        (df["dataset"] == "OneStop")
        & (df["preview"] == "Gathering")
        & (df["version_path"] == "fully_agg/ordinary/all")
        & (df["target_col"].isin(ONESTOP_COMPREHENSION_TARGET_COLS))
    ]

    if df.empty:
        logger.warning("No comprehension data for OneStop/Gathering/fully_agg")
        return

    # Use available feature sets from data
    available_fs = [fs for fs in SMALL_FEATURE_SETS if fs in df["feature_set"].unique()]

    # Build the table
    lines = ["\\begin{table*}[ht!]", "\\centering", "\\small"]
    lines.append(
        "\\caption{\\textbf{" + BOOTSTRAP_DATASET_DISPLAY["OneStop"] + "}: "
        "EyeScore correlation with reading comprehension scores.}")

    # Column spec
    n_targets = len(ONESTOP_COMPREHENSION_TARGET_COLS)
    col_spec = "l||" + "|".join(["c"] * n_targets)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header
    header = ["\\textbf{Features}"]
    for tc in ONESTOP_COMPREHENSION_TARGET_COLS:
        display = DATASET_TARGET_COL_DISPLAY["OneStop"].get(tc, tc)
        header.append(f"\\textbf{{{display}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")

    # Find best correlations per column
    best = {}
    for tc in ONESTOP_COMPREHENSION_TARGET_COLS:
        vals = []
        for _, row in df[(df["target_col"] == tc)].iterrows():
            if not pd.isna(row["pearson_r"]):
                vals.append(row["pearson_r"])
        best[tc] = max(vals) if vals else None

    # Data rows
    for fs in available_fs:
        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for tc in ONESTOP_COMPREHENSION_TARGET_COLS:
            subset = df[(df["feature_set"] == fs) & (df["target_col"] == tc)]
            if not subset.empty:
                r = subset.iloc[0]["pearson_r"]
                if not pd.isna(r):
                    best_val = best.get(tc)
                    is_best = best_val is not None and np.isclose(r, best_val, atol=1e-6)
                    formatted = f"{r:.2f}"
                    if is_best:
                        row_parts.append(f"\\textbf{{{formatted}}}")
                    else:
                        row_parts.append(formatted)
                else:
                    row_parts.append("NA")
            else:
                row_parts.append("NA")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    latex = "\n".join(lines)

    save_dir.mkdir(parents=True, exist_ok=True)
    table_path = save_dir / "comprehension.tex"
    write_table(table_path, latex)
    logger.info("Saved table: %s", table_path)


def generate_comprehension_michigan_parts_table(
    eval_csv: Path = None,
    save_dir: Path = None,
):
    """Generate Michigan test parts + comprehension table for OneStop EyeScore."""
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "OneStop"

    df = pd.read_csv(eval_csv)
    df = df[
        (df["dataset"] == "OneStop")
        & (df["preview"] == "Gathering")
        & (df["version_path"] == "fully_agg/ordinary/all")
        & (df["target_col"].isin(ONESTOP_COMPREHENSION_MICHIGAN_TARGET_COLS))
    ]

    if df.empty:
        logger.warning("No comprehension+michigan parts data for OneStop/Gathering/fully_agg")
        return

    available_fs = [fs for fs in SMALL_FEATURE_SETS if fs in df["feature_set"].unique()]

    # Build the table
    lines = ["\\begin{table*}[ht!]", "\\centering", "\\small"]
    lines.append("\\caption{EyeScore correlation with comprehension scores and MichTest parts.}")

    n_targets = len(ONESTOP_COMPREHENSION_MICHIGAN_TARGET_COLS)
    col_spec = "l||" + "|".join(["c"] * n_targets)

    # Wrap in resizebox if many columns to prevent overfull hbox
    if n_targets > 6:
        lines.append("\\resizebox{\\textwidth}{!}{")

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header
    header = ["\\textbf{Features}"]
    target_col_display = MICHIGAN_PARTS_DISPLAY.copy()
    target_col_display["comprehension_score-regular_trials"] = "Comprehension"
    for tc in ONESTOP_COMPREHENSION_MICHIGAN_TARGET_COLS:
        display = target_col_display.get(tc, tc)
        header.append(f"\\textbf{{{display}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")

    # Find best correlations
    best = {}
    for tc in ONESTOP_COMPREHENSION_MICHIGAN_TARGET_COLS:
        vals = []
        for _, row in df[(df["target_col"] == tc)].iterrows():
            if not pd.isna(row["pearson_r"]):
                vals.append(row["pearson_r"])
        best[tc] = max(vals) if vals else None

    # Data rows
    for fs in available_fs:
        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for tc in ONESTOP_COMPREHENSION_MICHIGAN_TARGET_COLS:
            subset = df[(df["feature_set"] == fs) & (df["target_col"] == tc)]
            if not subset.empty:
                r = subset.iloc[0]["pearson_r"]
                if not pd.isna(r):
                    best_val = best.get(tc)
                    is_best = best_val is not None and np.isclose(r, best_val, atol=1e-6)
                    formatted = f"{r:.2f}"
                    if is_best:
                        row_parts.append(f"\\textbf{{{formatted}}}")
                    else:
                        row_parts.append(formatted)
                else:
                    row_parts.append("NA")
            else:
                row_parts.append("NA")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")

    # Close resizebox if it was opened
    if n_targets > 6:
        lines.append("}")

    lines.append("\\end{table*}")

    latex = "\n".join(lines)

    save_dir.mkdir(parents=True, exist_ok=True)
    table_path = save_dir / "comprehension_michigan_parts.tex"
    write_table(table_path, latex)
    logger.info("Saved table: %s", table_path)


def generate_comprehension_michigan_parts_simplified_table(
    eval_csv: Path = None,
    save_dir: Path = None,
):
    """Generate simplified Michigan test parts + comprehension table for OneStop EyeScore."""
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "OneStop"

    df = pd.read_csv(eval_csv)
    df = df[
        (df["dataset"] == "OneStop")
        & (df["preview"] == "Gathering")
        & (df["version_path"] == "fully_agg/ordinary/all")
        & (df["target_col"].isin(ONESTOP_COMPREHENSION_MICHIGAN_SIMPLIFIED_TARGET_COLS))
    ]

    if df.empty:
        logger.warning("No comprehension+michigan parts simplified data for OneStop/Gathering/fully_agg")
        return

    available_fs = [fs for fs in SMALL_FEATURE_SETS if fs in df["feature_set"].unique()]

    # Build the table
    lines = ["\\begin{table*}[ht!]", "\\centering", "\\small"]
    lines.append("\\caption{EyeScore correlation with comprehension scores and grouped MichTest parts.}")

    n_targets = len(ONESTOP_COMPREHENSION_MICHIGAN_SIMPLIFIED_TARGET_COLS)
    col_spec = "l||" + "|".join(["c"] * n_targets)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header
    header = ["\\textbf{Features}"]
    target_col_display = MICHIGAN_PARTS_DISPLAY.copy()
    target_col_display["comprehension_score-regular_trials"] = "Comprehension"
    for tc in ONESTOP_COMPREHENSION_MICHIGAN_SIMPLIFIED_TARGET_COLS:
        display = target_col_display.get(tc, tc)
        header.append(f"\\textbf{{{display}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")

    # Find best correlations
    best = {}
    for tc in ONESTOP_COMPREHENSION_MICHIGAN_SIMPLIFIED_TARGET_COLS:
        vals = []
        for _, row in df[(df["target_col"] == tc)].iterrows():
            if not pd.isna(row["pearson_r"]):
                vals.append(row["pearson_r"])
        best[tc] = max(vals) if vals else None

    # Data rows
    for fs in available_fs:
        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for tc in ONESTOP_COMPREHENSION_MICHIGAN_SIMPLIFIED_TARGET_COLS:
            subset = df[(df["feature_set"] == fs) & (df["target_col"] == tc)]
            if not subset.empty:
                r = subset.iloc[0]["pearson_r"]
                if not pd.isna(r):
                    best_val = best.get(tc)
                    is_best = best_val is not None and np.isclose(r, best_val, atol=1e-6)
                    formatted = f"{r:.2f}"
                    if is_best:
                        row_parts.append(f"\\textbf{{{formatted}}}")
                    else:
                        row_parts.append(formatted)
                else:
                    row_parts.append("NA")
            else:
                row_parts.append("NA")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    latex = "\n".join(lines)

    save_dir.mkdir(parents=True, exist_ok=True)
    table_path = save_dir / "comprehension_michigan_parts_simplified.tex"
    write_table(table_path, latex)
    logger.info("Saved table: %s", table_path)


def generate_michigan_parts_simplified_table(
    eval_csv: Path = None,
    save_dir: Path = None,
):
    """Generate simplified Michigan test parts table (first & second parts only) for EyeScore.

    Only shows: overall michtest_score + first part (listen+grammar) + second part (vocab+reading)
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "OneStop"

    df = pd.read_csv(eval_csv)
    df = df[
        (df["dataset"] == "OneStop")
        & (df["preview"] == "Gathering")
        & (df["version_path"] == "fully_agg/ordinary/all")
        & (df["target_col"].isin(MICHIGAN_PARTS_SIMPLIFIED_TARGET_COLS))
    ]

    if df.empty:
        logger.warning("No simplified Michigan parts data for OneStop/Gathering/fully_agg")
        return

    # Use available feature sets from data
    available_fs = [fs for fs in SMALL_FEATURE_SETS if fs in df["feature_set"].unique()]

    # Build the table
    lines = ["\\begin{table*}[ht!]", "\\centering", "\\small"]
    lines.append("\\caption{EyeScore correlation with grouped MichTest parts.}")

    # Column spec: features, then per-target column
    n_targets = len(MICHIGAN_PARTS_SIMPLIFIED_TARGET_COLS)
    col_spec = "l||" + "|".join(["c"] * n_targets)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header: target column names
    header = ["\\textbf{Features}"]
    for tc in MICHIGAN_PARTS_SIMPLIFIED_TARGET_COLS:
        display = MICHIGAN_PARTS_DISPLAY.get(tc, tc)
        header.append(f"\\textbf{{{display}}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")

    # Find best correlations per column
    best = {}
    for tc in MICHIGAN_PARTS_SIMPLIFIED_TARGET_COLS:
        vals = []
        for _, row in df[(df["target_col"] == tc)].iterrows():
            if not pd.isna(row["pearson_r"]):
                vals.append(row["pearson_r"])
        best[tc] = max(vals) if vals else None

    # Data rows
    for fs in available_fs:
        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for tc in MICHIGAN_PARTS_SIMPLIFIED_TARGET_COLS:
            subset = df[(df["feature_set"] == fs) & (df["target_col"] == tc)]
            if not subset.empty:
                r = subset.iloc[0]["pearson_r"]
                if not pd.isna(r):
                    best_val = best.get(tc)
                    is_best = best_val is not None and np.isclose(r, best_val, atol=1e-6)
                    formatted = f"{r:.2f}"
                    if is_best:
                        row_parts.append(f"\\textbf{{{formatted}}}")
                    else:
                        row_parts.append(formatted)
                else:
                    row_parts.append("NA")
            else:
                row_parts.append("NA")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    latex = "\n".join(lines)

    save_dir.mkdir(parents=True, exist_ok=True)
    table_path = save_dir / "michigan_parts_simplified.tex"
    write_table(table_path, latex)
    logger.info("Saved table: %s", table_path)


def generate_all_tables(
    eval_dir: Path = None,
    save_dir: Path = None,
    feature_sets: Optional[List[str]] = None,
):
    """Generate LaTeX tables for the pristine org evaluation plus the
    language-bias subtree (org and typo_calibrated under each split scheme)."""
    if eval_dir is None:
        eval_dir = EVAL_SAVE_DIR
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR

    # Pristine: org-only tables under save_dir/.
    generate_tables_from_eval(
        eval_csv=eval_dir / "all_evaluation_results.csv",
        save_dir=save_dir, feature_sets=feature_sets, label="pristine org",
    )

    # Raw bias-correlation table. Spearman variant and the two
    # typology-corrected variants are intentionally not generated.
    raw_filename, raw_mode, raw_description = BIAS_CORR_TABLE_VARIANTS[0]
    generate_bias_correlation_table(
        mode=raw_mode,
        save_path=save_dir / "lang_bias" / raw_filename,
        description=raw_description,
        correlation_method="pearson",
        # Pass distance_type explicitly so the suffix-rebind in __main__ takes
        # effect — the function default is frozen at def-time and would
        # otherwise force the recompute path on the wrong distance.
        distance_type=BIAS_CORR_DISTANCE_TYPE,
        dataset_targets={"Meco": ["lextale_score"], "OneStop": ["lextale_score"]},
        single_column=True,
    )

    # Combined triplet table: r_to_target / bias_r / bias_p per
    # (feature_set, dataset/target, mode). Full three-mode layout
    # (org | seen language | new language) per dataset. We emit one table per
    # eye_score content_mode — "unseen" (default, MECO cross-half averages and
    # OneStop's held-out-text split) gets the unsuffixed filename; "all" is
    # the full 12-paragraph aggregate.
    #
    # Two flavors:
    #   bias_triplet{cm}.tex       — single-target (LexTALE only)
    #   bias_triplet_multi{cm}.tex — multi-target, MECO Composite+LexTALE and
    #                                OneStop MichTest+LexTALE, mirroring the
    #                                combined EyeScore table.
    for cm in ("unseen", "all", "seen"):
        suffix = "" if cm == "unseen" else f"_{cm}"
        generate_bias_triplet_table(
            save_path=save_dir / "lang_bias" / f"bias_triplet{suffix}.tex",
            content_mode=cm,
        )
        generate_bias_triplet_table(
            save_path=save_dir / "lang_bias" / f"bias_triplet_multi{suffix}.tex",
            content_mode=cm,
            # Use the combined-EyeScore ordering (LexTALE first, then
            # MichTest/Composite) so this table reads in the same direction as
            # the combined-r table.
            dataset_targets=DATASET_TARGET_COLS,
        )
        # Per-debias-target single-dataset tables: eye_score debiased against
        # the dataset's primary proficiency score (MichTest on OneStop,
        # Composite on MECO). Bias evaluated against both that target and
        # LexTALE so the off-target column shows whether the calibration
        # transfers to a different proficiency yardstick.
        generate_bias_triplet_table(
            save_path=save_dir / "lang_bias" / f"bias_triplet_meco_profagg{suffix}.tex",
            content_mode=cm,
            modes=BIAS_TRIPLET_MODES_PROFAGG,
            dataset_targets={"Meco": ["lextale_score", "proficiency_agg"]},
        )
        generate_bias_triplet_table(
            save_path=save_dir / "lang_bias" / f"bias_triplet_onestop_michtest{suffix}.tex",
            content_mode=cm,
            modes=BIAS_TRIPLET_MODES_MICHTEST,
            dataset_targets={"OneStop": ["lextale_score", "michtest_score"]},
        )

    # Combined-scope variants of the per-debias-target single-dataset tables:
    # each cell shows "Fixed / Any" so seen and unseen regimes sit side-by-side
    # in one table, mirroring the combined_debiased_lextale layout.
    generate_bias_triplet_table(
        save_path=save_dir / "lang_bias" / "bias_triplet_meco_profagg_combined.tex",
        modes=BIAS_TRIPLET_MODES_PROFAGG,
        dataset_targets={"Meco": ["lextale_score", "proficiency_agg"]},
        combine_scopes=True,
    )
    generate_bias_triplet_table(
        save_path=save_dir / "lang_bias" / "bias_triplet_onestop_michtest_combined.tex",
        modes=BIAS_TRIPLET_MODES_MICHTEST,
        dataset_targets={"OneStop": ["lextale_score", "michtest_score"]},
        combine_scopes=True,
    )

    # Raw EyeScore vs L1 distance table — same layout as bias_triplet but
    # cells are r(eye_score, distance) directly, matching the
    # eyescore_vs_distance_split plot. A LexTALE baseline row sits at the top
    # spanning Fixed+Any per dataset.
    generate_eyescore_distance_table(
        save_path=save_dir / "lang_bias" / "eyescore_distance.tex",
    )

    logger.info("All tables generated under %s", save_dir)

    # Michigan test parts table
    logger.info("Generating Michigan test parts table...")
    generate_michigan_parts_table()

    # Simplified Michigan test parts table
    logger.info("Generating simplified Michigan test parts table...")
    generate_michigan_parts_simplified_table()

    # Comprehension table
    logger.info("Generating comprehension table...")
    generate_comprehension_table()

    # Comprehension + Michigan parts table
    logger.info("Generating comprehension + Michigan parts table...")
    generate_comprehension_michigan_parts_table()

    # Comprehension + Michigan parts simplified table
    logger.info("Generating comprehension + Michigan parts simplified table...")
    generate_comprehension_michigan_parts_simplified_table()


# ── Combined OneStop × Meco tables ───────────────────────────────────────────


def generate_latex_combined_table(
    eval_dfs: dict,  # {dataset: filtered_eval_df}
    target_cols: dict,  # {dataset: [target_cols]}
    scopes: List[str],
    content_mode_to_scope: dict,
    target_col_display: dict,  # {dataset: {target_col: display_name}}
    dataset_display: dict,  # {dataset: display_name}
    preview: str = "",
    version_path: str = "",
    paper_style: bool = False,
    caption_override: Optional[str] = None,
    label: Optional[str] = None,
    decimals: int = 2,
) -> str:
    """Generate a LaTeX table combining multiple datasets side-by-side.

    Columns structure: Dataset (OneStop | Meco) → Target columns → Content modes → Pearson r
    Rows: Feature sets
    """
    feature_sets = []
    for dataset, df in eval_dfs.items():
        feature_sets.extend(df["feature_set"].unique())
    feature_sets = list(set(feature_sets))
    feature_sets = [fs for fs in SMALL_FEATURE_SETS if fs in feature_sets]

    if not feature_sets:
        return ""

    # Build lookups per dataset: (feature_set, target_col, scope) -> pearson_r
    lookups = {}
    for dataset, df in eval_dfs.items():
        lookup = {}
        for _, row in df.iterrows():
            scope = content_mode_to_scope[dataset].get(row["content_mode"])
            if scope is None:
                continue
            key = (row["feature_set"], row["target_col"], scope)
            lookup[key] = row["pearson_r"]
        lookups[dataset] = lookup

    # Force fixed-text-only feature sets to NA in non-Fixed scopes
    for dataset in eval_dfs.keys():
        for fs in feature_sets:
            if fs in FIXED_TEXT_ONLY_FEATURE_SETS:
                for tc in target_cols.get(dataset, []):
                    for scope in scopes:
                        if scope != "Fixed":
                            lookups[dataset][(fs, tc, scope)] = np.nan

    # Find best (max r) per (dataset, target_col, scope) across feature sets
    best = {}
    for dataset in eval_dfs.keys():
        for tc in target_cols.get(dataset, []):
            for scope in scopes:
                vals = []
                for fs in feature_sets:
                    r = lookups[dataset].get((fs, tc, scope))
                    if r is not None and not pd.isna(r):
                        vals.append(r)
                best[(dataset, tc, scope)] = max(vals) if vals else None

    n_scopes = len(scopes)
    datasets = list(eval_dfs.keys())

    if paper_style:
        # Template format: booktabs rules, no | between cols within a
        # target pair, single | between target pairs, dataset boundary marked
        # via the c|/c alignment of the dataset/target multicolumn headers.
        target_blocks = []
        for ds in datasets:
            for _ in target_cols.get(ds, []):
                target_blocks.append("c" * n_scopes)
        col_spec = "l" + "|".join(target_blocks)

        n_data_cols = sum(len(target_cols.get(d, [])) * n_scopes for d in datasets)
        n_total_cols = 1 + n_data_cols

        lines = [
            "\\begin{table}[ht!]",
            "\\centering",
            "% \\small",
            "\\resizebox{\\columnwidth}{!}{",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
        ]

        hdr_empty = "\\multicolumn{1}{c}{}"

        # Header 1: dataset names. c| on every dataset except the last to
        # draw the dataset-boundary rule.
        h1 = [hdr_empty]
        for i, ds in enumerate(datasets):
            span = len(target_cols.get(ds, [])) * n_scopes
            is_last_ds = (i == len(datasets) - 1)
            align = "c" if is_last_ds else "c|"
            display = dataset_display.get(ds, ds)
            h1.append(f"\\multicolumn{{{span}}}{{{align}}}{{\\textbf{{{display}}}}}")
        lines.append(" & ".join(h1) + " \\\\")

        # Header 2: target labels. | only on the last target of a non-last
        # dataset, so targets within a dataset visually share the dataset block.
        h2 = [hdr_empty]
        for i, ds in enumerate(datasets):
            tcs = target_cols.get(ds, [])
            is_last_ds = (i == len(datasets) - 1)
            for j, tc in enumerate(tcs):
                is_last_tc = (j == len(tcs) - 1)
                align = "c|" if (is_last_tc and not is_last_ds) else "c"
                label_text = _display(tc, target_col_display.get(ds, {}))
                h2.append(f"\\multicolumn{{{n_scopes}}}{{{align}}}{{{label_text}}}")
        lines.append(" & ".join(h2) + f" \\\\ %\\cmidrule{{2-{n_total_cols}}}")

        # Header 3: Features + scope labels (plain cells so the col-spec |s show).
        h3 = ["\\textbf{Features}"]
        for ds in datasets:
            for _tc in target_cols.get(ds, []):
                for scope in scopes:
                    h3.append(scope)
        lines.append(" & ".join(h3) + " \\\\")
        lines.append("\\midrule")

        # Commented-out per-dataset cline for reference; midrule above is the
        # visible separator.
        cline_parts = []
        col = 2
        for i, ds in enumerate(datasets):
            n_cols_in_ds = len(target_cols.get(ds, [])) * n_scopes
            if n_cols_in_ds == 0:
                continue
            start = 1 if i == 0 else col
            end = col + n_cols_in_ds - 1
            cline_parts.append(f"\\cline{{{start}-{end}}}")
            col += n_cols_in_ds
        if cline_parts:
            lines.append("%" + "".join(cline_parts))

        for i, fs in enumerate(feature_sets):
            if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[feature_sets[i - 1]]:
                lines.append("\\midrule")
            row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
            for ds in datasets:
                for tc in target_cols.get(ds, []):
                    for scope in scopes:
                        val = lookups[ds].get((fs, tc, scope))
                        if val is None or pd.isna(val):
                            row_parts.append("-")
                        else:
                            best_val = best.get((ds, tc, scope))
                            is_best = (best_val is not None and
                                       np.isclose(val, best_val, atol=1e-6))
                            row_parts.append(_fmt_val(val, is_best, decimals))
            lines.append(" & ".join(row_parts) + " \\\\")
        lines.append("\\bottomrule")

        lines.append("\\end{tabular}")
        lines.append("}")

        if caption_override is not None:
            lines.append(f"\\caption{{{caption_override}}}")
        elif version_path:
            lines.append("\\caption{Combined EyeScore--proficiency Pearson $r$.}")
        if label is not None:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    # ── Non-paper (standard) style ──────────────────────────────────────────
    hdr_empty = ""
    lines = ["\\begin{table*}[ht!]", "\\centering", "\\small"]

    col_groups = []
    for dataset in datasets:
        dataset_targets = target_cols.get(dataset, [])
        scope_block = "|".join(["c"] * (len(dataset_targets) * n_scopes))
        col_groups.append(scope_block)
    col_spec = "l|" + "|".join(col_groups)

    use_resize = len(eval_dfs) * max(len(target_cols.get(d, [])) for d in datasets) * n_scopes > 4
    if use_resize:
        lines.append("\\resizebox{\\textwidth}{!}{")

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    header1_parts = [hdr_empty]
    for dataset in datasets:
        dataset_targets = target_cols.get(dataset, [])
        span = len(dataset_targets) * n_scopes
        display = dataset_display.get(dataset, dataset)
        header1_parts.append(f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{display}}}}}")
    lines.append(" & ".join(header1_parts) + " \\\\")

    header2_parts = [hdr_empty]
    for dataset in datasets:
        for tc in target_cols.get(dataset, []):
            display = _display(tc, target_col_display.get(dataset, {}))
            header2_parts.append(f"\\multicolumn{{{n_scopes}}}{{c}}{{{display}}}")
    lines.append(" & ".join(header2_parts) + " \\\\")

    header3_parts = ["\\textbf{Features}"]
    for dataset in datasets:
        for _tc in target_cols.get(dataset, []):
            for scope in scopes:
                header3_parts.append(scope)
    lines.append(" & ".join(header3_parts) + " \\\\")
    lines.append("\\hline")

    for i, fs in enumerate(feature_sets):
        if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[feature_sets[i - 1]]:
            lines.append("\\hline")
        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]
        for dataset in datasets:
            for tc in target_cols.get(dataset, []):
                for scope in scopes:
                    val = lookups[dataset].get((fs, tc, scope))
                    if val is not None and not pd.isna(val):
                        best_val = best.get((dataset, tc, scope))
                        is_best = (best_val is not None and
                                   np.isclose(val, best_val, atol=1e-6))
                        row_parts.append(_fmt_val(val, is_best, decimals))
                    else:
                        row_parts.append("NA")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    if use_resize:
        lines.append("}")

    if caption_override is not None:
        lines.append(f"\\caption{{{caption_override}}}")
    elif version_path:
        lines.append("\\caption{Combined EyeScore--proficiency Pearson $r$.}")
    if label is not None:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def generate_full_combined_tables(
    eval_csv: Path = None,
    save_dir: Path = None,
    feature_sets: Optional[List[str]] = None,
):
    """Generate combined OneStop × Meco tables showing both datasets side-by-side.

    Combines Meco (All preview) with OneStop (Gathering preview, ordinary version)
    since they are dataset-specific labels for the same evaluation setup.
    """
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "full"
    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_ORDERED

    if not eval_csv.exists():
        logger.warning("Evaluation CSV not found: %s", eval_csv)
        return

    df = pd.read_csv(eval_csv)
    df = df[df["feature_set"].isin(feature_sets)]

    if df.empty:
        logger.warning("No data found for combined tables")
        return

    # Get fully_agg data for each dataset
    meco_df = df[(df["dataset"] == "Meco") & (df["version_path"] == "fully_agg/all/all")]
    onestop_df = df[(df["dataset"] == "OneStop") & (df["version_path"] == "fully_agg/ordinary/all")]

    if meco_df.empty or onestop_df.empty:
        logger.warning("Missing fully_agg data for one or both datasets")
        return

    # Prepare data per dataset
    eval_dfs = {}
    target_cols_per_ds = {}

    # Filter to target columns for each dataset
    target_cols_meco = DATASET_TARGET_COLS.get("Meco", [])
    target_cols_onestop = DATASET_TARGET_COLS.get("OneStop", [])

    meco_filtered = meco_df[meco_df["target_col"].isin(target_cols_meco)]
    onestop_filtered = onestop_df[onestop_df["target_col"].isin(target_cols_onestop)]

    if not meco_filtered.empty:
        eval_dfs["Meco"] = meco_filtered
        target_cols_per_ds["Meco"] = target_cols_meco

    if not onestop_filtered.empty:
        eval_dfs["OneStop"] = onestop_filtered
        target_cols_per_ds["OneStop"] = target_cols_onestop

    if len(eval_dfs) < 2:
        logger.warning("Missing data for one or both datasets in fully_agg")
        return

    # Determine scopes
    scopes = DATASET_SCOPES.get("OneStop", [])

    content_mode_to_scope = {d: DATASET_CONTENT_MODE_TO_SCOPE[d] for d in eval_dfs.keys()}
    dataset_display = {"OneStop": "OneStop", "Meco": "MECO"}
    target_col_display = DATASET_TARGET_COL_DISPLAY

    latex = generate_latex_combined_table(
        eval_dfs,
        target_cols=target_cols_per_ds,
        scopes=scopes,
        content_mode_to_scope=content_mode_to_scope,
        target_col_display=target_col_display,
        dataset_display=dataset_display,
        preview="Gathering/All",
        version_path="fully_agg",
        paper_style=True,
        caption_override=(
            "\\textbf{EyeScore}. Pearson $r$ correlations between EyeScore "
            "and standard language proficiency tests. ``Fixed'' is the Fixed "
            "Text regime in which all the eye movement data is for the same "
            "texts presented to the test participant. ``Any'' is the Any "
            "Text regime in which no eye movement data is available for the "
            "texts of the test participant. The best result for each "
            "evaluation measure in each regime, dataset, and target "
            "proficiency test is marked in bold."
        ),
        label="tab:eyescore_seen_unseen",
    )

    if not latex:
        logger.warning("Failed to generate combined table")
        return

    # Save to directory: full/combined.tex
    table_dir = save_dir / "fully_agg"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "combined.tex"
    write_table(table_path, latex)
    logger.info("Saved combined EyeScore table (Meco/All + OneStop/Gathering, displayed as MECO): %s", table_path)

    # Same table with 3 decimal places (combined_3dp.tex)
    latex_3dp = generate_latex_combined_table(
        eval_dfs,
        target_cols=target_cols_per_ds,
        scopes=scopes,
        content_mode_to_scope=content_mode_to_scope,
        target_col_display=target_col_display,
        dataset_display=dataset_display,
        preview="Gathering/All",
        version_path="fully_agg",
        paper_style=True,
        caption_override=(
            "\\textbf{EyeScore}. Pearson $r$ correlations between EyeScore "
            "and standard language proficiency tests. ``Fixed'' is the Fixed "
            "Text regime in which all the eye movement data is for the same "
            "texts presented to the test participant. ``Any'' is the Any "
            "Text regime in which no eye movement data is available for the "
            "texts of the test participant. The best result for each "
            "evaluation measure in each regime, dataset, and target "
            "proficiency test is marked in bold."
        ),
        label="tab:eyescore_seen_unseen_3dp",
        decimals=3,
    )
    if latex_3dp:
        table_path_3dp = table_dir / "combined_3dp.tex"
        write_table(table_path_3dp, latex_3dp)
        logger.info("Saved 3-decimal combined EyeScore table: %s", table_path_3dp)

    # Bootstrap variant of the same table: mean ± std with paired-bootstrap
    # stars vs WPM, one call per correlation metric.
    # show_dispersion=False repeats each table with the ± std stripped
    # (combined_bootstrap*_noci.tex) for uses that want stars only.
    for corr_metric in ("pearson", "spearman"):
        for show_dispersion in (True, False):
            # The no-dispersion table drops the "± std" from every cell, so it
            # is narrow enough for a single column -- which is where the paper
            # sets it. The ± variant needs the full text width.
            generate_bootstrap_combined_table(eval_csv=eval_csv, save_dir=save_dir,
                                              feature_sets=feature_sets,
                                              corr_metric=corr_metric,
                                              show_dispersion=show_dispersion,
                                              single_column=not show_dispersion)

    # Generate per_item_agg combined table
    _generate_per_item_agg_combined_table(df, save_dir, feature_sets, target_cols_meco, target_cols_onestop)

    logger.info("Combined EyeScore tables generated under %s", save_dir)


def _generate_per_item_agg_eyescore_latex(
    onestop_article, onestop_paragraph, meco_df
):
    """Generate LaTeX for per_item_agg EyeScore table with OneStop article |
    paragraph side-by-side, each split into Fixed | Any.

    The scope split matters: since R5 runs per-item calculation with
    --content-mode seen,unseen, every (feature_set, target_col) key appears
    twice, once per content mode. Keying the lookups on (feature_set,
    target_col) alone silently kept whichever row iterated last, so the table
    reported an arbitrary mix of seen and unseen cells.
    """
    metrics = ["pearson_r"]

    cm_to_scope = DATASET_CONTENT_MODE_TO_SCOPE["OneStop"]
    # Only the scopes actually present, in the canonical order.
    present_scopes = {
        cm_to_scope.get(cm)
        for df_part in (onestop_article, onestop_paragraph, meco_df)
        if not df_part.empty
        for cm in df_part["content_mode"].unique()
    }
    scopes = [s for s in DATASET_SCOPES["OneStop"] if s in present_scopes]
    if not scopes:
        # Pre-R5 trees carry content_mode="all" only; keep them renderable.
        scopes = ["All"]

    # Get feature sets
    feature_sets = set()
    for df_part in [onestop_article, onestop_paragraph, meco_df]:
        if not df_part.empty:
            feature_sets.update(df_part["feature_set"].unique())
    feature_sets = [fs for fs in ALL_FEATURE_SETS_ORDERED if fs in feature_sets]

    # Get target columns
    onestop_targets = sorted(onestop_article["target_col"].unique()) if not onestop_article.empty else []
    meco_targets = sorted(meco_df["target_col"].unique()) if not meco_df.empty else []

    # Build lookups, keyed on scope so seen and unseen cells stay apart.
    def _build_lookup(df):
        lookup = {}
        if df.empty:
            return lookup
        for _, row in df.iterrows():
            scope = cm_to_scope.get(row["content_mode"])
            if scope is None:
                continue
            key = (row["feature_set"], row["target_col"], scope)
            lookup[key] = {m: row[m] for m in metrics}
        return lookup

    art_lookup = _build_lookup(onestop_article)
    par_lookup = _build_lookup(onestop_paragraph)
    meco_lookup = _build_lookup(meco_df)

    def _cell(lookup, fs, target, scope):
        """Value for one cell, or None when absent/not applicable.

        Per-text feature sets only ever have a seen ("Fixed") cell — their
        column space is tied to the specific texts — so force NA elsewhere,
        matching FIXED_TEXT_ONLY_FEATURE_SETS handling in the other tables.
        """
        if scope != "Fixed" and fs in FIXED_TEXT_ONLY_FEATURE_SETS:
            return None
        val = lookup.get((fs, target, scope), {}).get("pearson_r")
        return None if val is None or pd.isna(val) else val

    # Find best values per (dataset, target, metric, scope). Scope is part of
    # the key: Any is systematically lower than Fixed, so pooling them would
    # bold a Fixed cell in every Any column and never mark the best Any result.
    best = {}
    for target in onestop_targets:
        for m in metrics:
            for scope in scopes:
                vals = [v for fs in feature_sets
                        for lookup in (art_lookup, par_lookup)
                        if (v := _cell(lookup, fs, target, scope)) is not None]
                if vals:
                    best[("onestop", target, m, scope)] = max(vals)

    for target in meco_targets:
        for m in metrics:
            for scope in scopes:
                vals = [v for fs in feature_sets
                        if (v := _cell(meco_lookup, fs, target, scope)) is not None]
                if vals:
                    best[("meco", target, m, scope)] = max(vals)

    # Build table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")

    # Column spec: features + (OneStop: targets * article|paragraph * scopes)
    #            + (MECO: targets * scopes)
    n_scopes = len(scopes)
    onestop_cols = len(onestop_targets) * 2 * n_scopes
    meco_cols = len(meco_targets) * n_scopes
    col_spec = f"l||{'c' * onestop_cols}||{'c' * meco_cols}"

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header 1: Dataset names
    header1 = f" & \\multicolumn{{{onestop_cols}}}{{c}}{{\\textbf{{OneStop}}}} & \\multicolumn{{{meco_cols}}}{{c}}{{\\textbf{{MECO}}}}"
    lines.append(header1 + " \\\\")

    # Header 2: Target columns
    header2_parts = [""]
    for target in onestop_targets:
        display = DATASET_TARGET_COL_DISPLAY["OneStop"].get(target, target)
        header2_parts.append(f"\\multicolumn{{{2 * n_scopes}}}{{c}}{{{display}}}")
    for target in meco_targets:
        display = DATASET_TARGET_COL_DISPLAY["Meco"].get(target, target)
        header2_parts.append(f"\\multicolumn{{{n_scopes}}}{{c}}{{{display}}}")
    lines.append(" & ".join(header2_parts) + " \\\\")

    # Header 3: article | paragraph subheaders (OneStop only; MECO has a
    # single unit of analysis, so its target header spans straight to scopes)
    header3_parts = [""]
    for _ in onestop_targets:
        header3_parts.append(f"\\multicolumn{{{n_scopes}}}{{c|}}{{Article}}")
        header3_parts.append(f"\\multicolumn{{{n_scopes}}}{{c}}{{Paragraph}}")
    for _ in meco_targets:
        header3_parts.append(f"\\multicolumn{{{n_scopes}}}{{c}}{{}}")
    lines.append(" & ".join(header3_parts) + " \\\\")

    # Header 4: Fixed | Any under each unit
    header4_parts = ["\\textbf{Features}"]
    for _ in onestop_targets:
        for _ in range(2):          # article, paragraph
            header4_parts.extend(scopes)
    for _ in meco_targets:
        header4_parts.extend(scopes)
    lines.append(" & ".join(header4_parts) + " \\\\")
    lines.append("\\hline")

    # Rows: feature sets
    for fs in feature_sets:
        row_parts = [_display(fs, FEATURE_SET_DISPLAY)]

        # OneStop article | paragraph, each split Fixed | Any
        for target in onestop_targets:
            for lookup in (art_lookup, par_lookup):
                for scope in scopes:
                    val = _cell(lookup, fs, target, scope)
                    if val is None:
                        row_parts.append("NA")
                        continue
                    is_best = np.isclose(
                        val, best.get(("onestop", target, "pearson_r", scope), np.nan), atol=1e-6)
                    row_parts.append(_fmt_val(val, is_best))

        # Meco
        for target in meco_targets:
            for scope in scopes:
                val = _cell(meco_lookup, fs, target, scope)
                if val is None:
                    row_parts.append("NA")
                    continue
                is_best = np.isclose(
                    val, best.get(("meco", target, "pearson_r", scope), np.nan), atol=1e-6)
                row_parts.append(_fmt_val(val, is_best))

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\end{tabular}")
    lines.append("\\caption{Combined EyeScore--proficiency Pearson $r$ "
                 "(Gathering/All, per\\_item\\_agg). Fixed = seen items, "
                 "Any = unseen items.}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# Sources for the merged Seen / New debiased combined table.
DEBIASED_COMBINED_MERGED_OUTPUT = "combined_debiased.tex"
DEBIASED_COMBINED_MERGED_SOURCES = [
    # (scheme, variant) — first is shown left of "/", second right.
    ("inside_langs", "typo_calibrated"),   # Seen Language
    ("across_langs", "typo_calibrated"),   # New Language
]


def _load_debiased_combined_lookup(scheme: str, variant: str,
                                   feature_sets: List[str]):
    """Return ((feature_set, dataset, target_col, scope) -> pearson_r) for the
    combined-table slice (MECO/All + OneStop/Gathering, fully_agg)."""
    eval_csv = (EVAL_SAVE_DIR / "lang_bias" / scheme / variant
                / "all_evaluation_results.csv")
    if not eval_csv.exists():
        return None
    df = pd.read_csv(eval_csv)
    df = df[df["feature_set"].isin(feature_sets)]
    meco = df[(df["dataset"] == "Meco")
              & (df["preview"] == "All")
              & (df["version_path"] == "fully_agg/all/all")
              & (df["target_col"].isin(DATASET_TARGET_COLS["Meco"]))]
    onestop = df[(df["dataset"] == "OneStop")
                 & (df["preview"] == "Gathering")
                 & (df["version_path"] == "fully_agg/ordinary/all")
                 & (df["target_col"].isin(DATASET_TARGET_COLS["OneStop"]))]
    if meco.empty or onestop.empty:
        return None

    lookup = {}
    for _, row in pd.concat([meco, onestop], ignore_index=True).iterrows():
        scope = DATASET_CONTENT_MODE_TO_SCOPE[row["dataset"]].get(row["content_mode"])
        if scope is None:
            continue
        lookup[(row["feature_set"], row["dataset"], row["target_col"], scope)] = (
            row["pearson_r"]
        )
    return lookup


def generate_debiased_combined_tables(
    save_dir: Path = None,
    feature_sets: Optional[List[str]] = None,
):
    """Emit a single combined OneStop × MECO Pearson-r table whose cells show
    the two debiased EyeScore variants as ``Seen / New`` (inside_langs /
    across_langs typo_calibrated). Saved alongside ``combined.tex`` under
    ``tables/full/fully_agg/`` as ``combined_debiased.tex``.

    Each side of the slash is bolded independently if it ties the column-max
    Pearson r within its own debiasing scheme — preserving the per-column
    winner story for both Seen and New separately.
    """
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "full"
    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_ORDERED

    lookups = []  # [seen_lookup, new_lookup]
    for scheme, variant in DEBIASED_COMBINED_MERGED_SOURCES:
        lk = _load_debiased_combined_lookup(scheme, variant, feature_sets)
        if lk is None:
            logger.warning("Missing debiased eval data for %s/%s; aborting merged table",
                           scheme, variant)
            return
        lookups.append(lk)

    datasets = ["Meco", "OneStop"]
    target_cols = {d: DATASET_TARGET_COLS[d] for d in datasets}
    scopes = DATASET_SCOPES["OneStop"]
    n_scopes = len(scopes)

    # Feature sets actually present in at least one lookup, in canonical order.
    present = set()
    for lk in lookups:
        for (fs, *_rest) in lk.keys():
            present.add(fs)
    feature_sets_ordered = [fs for fs in SMALL_FEATURE_SETS if fs in present]
    if not feature_sets_ordered:
        logger.warning("No feature sets present in debiased eval data; aborting merged table")
        return

    # Force fixed-text-only feature sets to NA in non-Fixed scopes for every
    # variant, mirroring the raw combined table's NA pattern.
    for lk in lookups:
        for fs in feature_sets_ordered:
            if fs in FIXED_TEXT_ONLY_FEATURE_SETS:
                for ds in datasets:
                    for tc in target_cols[ds]:
                        for scope in scopes:
                            if scope != "Fixed":
                                lk[(fs, ds, tc, scope)] = np.nan

    # Best (max r) per (variant, dataset, target_col, scope) across feature sets.
    bests = []
    for lk in lookups:
        best = {}
        for ds in datasets:
            for tc in target_cols[ds]:
                for scope in scopes:
                    vals = []
                    for fs in feature_sets_ordered:
                        v = lk.get((fs, ds, tc, scope))
                        if v is not None and not pd.isna(v):
                            vals.append(v)
                    best[(ds, tc, scope)] = max(vals) if vals else None
        bests.append(best)

    def _fmt_single(v, bold):
        if v is None or pd.isna(v):
            return None
        s = f"{v:.2f}"
        return f"\\textbf{{{s}}}" if bold else s

    def _fmt_cell(fs, ds, tc, scope):
        parts = []
        any_value = False
        for lk, best in zip(lookups, bests):
            v = lk.get((fs, ds, tc, scope))
            best_v = best.get((ds, tc, scope))
            is_best = (v is not None and best_v is not None and not pd.isna(v)
                       and np.isclose(v, best_v, atol=1e-6))
            piece = _fmt_single(v, is_best)
            if piece is None:
                parts.append("NA")
            else:
                parts.append(piece)
                any_value = True
        if not any_value:
            return "NA"
        return " / ".join(parts)

    hdr_empty = "\\multicolumn{1}{c}{}"
    lines = ["\\begin{table*}[ht!]", "\\centering", "\\small"]

    col_groups = []
    for ds in datasets:
        n = len(target_cols[ds]) * n_scopes
        col_groups.append("|".join(["c"] * n))
    col_spec = "l|" + "||".join(col_groups)

    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header row 1: dataset names.
    dataset_display = {"OneStop": "OneStop", "Meco": "MECO"}
    h1 = [hdr_empty]
    for ds in datasets:
        span = len(target_cols[ds]) * n_scopes
        h1.append(f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{dataset_display[ds]}}}}}")
    lines.append(" & ".join(h1) + " \\\\")

    # Header row 2: target column labels.
    h2 = [hdr_empty]
    for ds in datasets:
        for tc in target_cols[ds]:
            label = _display(tc, DATASET_TARGET_COL_DISPLAY.get(ds, {}))
            h2.append(f"\\multicolumn{{{n_scopes}}}{{c}}{{{label}}}")
    n_total_cols = 1 + sum(len(target_cols[d]) * n_scopes for d in datasets)
    lines.append(" & ".join(h2) + f" \\\\ \\cline{{2-{n_total_cols}}}")

    # Header row 3: scope labels.
    h3 = ["\\textbf{Features}"]
    for ds in datasets:
        for _tc in target_cols[ds]:
            for scope in scopes:
                h3.append(scope)
    lines.append(" & ".join(h3) + " \\\\")
    lines.append("\\hline")
    lines.append("%\\cline{1-5}\\cline{6-9}")

    for i, fs in enumerate(feature_sets_ordered):
        if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[feature_sets_ordered[i - 1]]:
            lines.append("\\hline")
        row = [_display(fs, FEATURE_SET_DISPLAY)]
        for ds in datasets:
            for tc in target_cols[ds]:
                for scope in scopes:
                    row.append(_fmt_cell(fs, ds, tc, scope))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\caption{Pearson $r$ correlations between debiased EyeScore and "
        "standard language proficiency tests. Each cell shows "
        "\\textit{Seen\\,/\\,New}: the left value uses the Seen Language "
        "scheme (inside-langs typology-calibrated, where the debiasing data "
        "includes participants from the same L1 as the test participant); "
        "the right uses the New Language scheme (across-langs, no same-L1 "
        "participants). ``Fixed'' is the Fixed Text regime in which all the "
        "eye movement data from prior participants is for the same texts "
        "presented to the test participant. ``Any'' is the Any Text regime "
        "in which no eye movement data is available for the texts of the "
        "test participant.}"
    )
    lines.append("\\label{tab:eyescore_seen_unseen-debiased}")
    lines.append("\\end{table*}")

    table_dir = save_dir / "fully_agg"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / DEBIASED_COMBINED_MERGED_OUTPUT
    write_table(table_path, "\n".join(lines))
    logger.info("Saved merged debiased combined EyeScore table: %s", table_path)


# ── Bootstrap combined table (mean ± std, bootstrap stars vs WPM) ────────────

BOOTSTRAP_COMBINED_OUTPUT = "combined_bootstrap.tex"
# Spearman variants of the two tables above (identical layout, rank correlation).
BOOTSTRAP_COMBINED_SPEARMAN_OUTPUT = "combined_bootstrap_spearman.tex"

# Baseline feature set every other row is tested against (WPM).
BOOTSTRAP_BASELINE_FS = "READING_SPEED"

# stars_from -> (pairwise p-value column, caption fragment describing the test)
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

# Per-dataset (preview, version_path) slice shown in the combined tables —
# same slice generate_full_combined_tables uses.
COMBINED_SLICE_FILTERS = {
    "Meco": ("All", "fully_agg/all/all"),
    "OneStop": ("Gathering", "fully_agg/ordinary/all"),
}


# Dataset names as they appear in the paper tables; see the predictions module.
BOOTSTRAP_DATASET_DISPLAY = {"OneStop": "OneStopL2", "Meco": "MECO"}


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


# See the note on the predictions-side twin: stars mark the level only, in
# either direction, and the correlation itself says whether the cell is above
# or below the baseline.
SIGNIFICANCE_LEGEND = (
    "Stars show the significance of the difference from the performance of "
    "the WPM baseline for the corresponding regime and proficiency test, in "
    "either direction: no marker $p\\geq 0.05$, $^{*}$ $p<0.05$, "
    "$^{**}$ $p<0.01$, $^{***}$ $p<0.001$."
)


SIG_GLYPH_BETTER = "*"
SIG_GLYPH_WORSE = "*"


def _directional_stars(p, r_fs, r_base) -> int:
    """Signed significance level for a two-sided test against the baseline.

    The p-values in pairwise_significance.csv are two-sided, so a feature set
    correlating significantly *worse* than WPM carries the same p as one
    correlating significantly better. The magnitude is the significance level
    (0-3) and the **sign** carries the direction: positive when the feature
    set's correlation is the higher of the two, negative when it is the lower.
    ``_sig_marker`` renders both directions as stars, so a significantly worse
    cell is marked rather than left blank next to a merely non-significant one.
    Zero = not significant, or direction undeterminable -- which includes an
    exact tie: equal correlations have no direction, so marking one as the
    worse of the two would be an artefact of the comparison operator.
    """
    if p is None or pd.isna(p) or pd.isna(r_fs) or pd.isna(r_base):
        return 0
    n = _p_to_n_stars(p)
    if not n or r_fs == r_base:
        return 0
    return n if r_fs > r_base else -n


def _sig_marker(ns: int) -> str:
    """Superscript for a signed significance level from ``_directional_stars``:
    stars in either direction, nothing when non-significant. Only the level is
    marked; the direction is read off the correlation."""
    if not ns:
        return ""
    glyph = SIG_GLYPH_BETTER if ns > 0 else SIG_GLYPH_WORSE
    return "^{" + glyph * abs(ns) + "}"


def generate_bootstrap_combined_table(
    eval_csv: Path = None,
    pairwise_csv: Path = None,
    save_dir: Path = None,
    feature_sets: Optional[List[str]] = None,
    decimals: Optional[int] = None,
    stars_from: str = "bootstrap",
    corr_metric: str = "pearson",
    target_cols_override: Optional[dict] = None,
    slice_filters_override: Optional[dict] = None,
    datasets_override: Optional[List[str]] = None,
    show_dispersion: bool = True,
    single_column: bool = False,
) -> str:
    """Combined OneStop × MECO table with bootstrap point estimates.

    Same layout as ``combined.tex`` (rows = feature sets, cols = dataset ×
    target × Fixed/Any regime), but each cell shows the bootstrap **mean**
    correlation ± one bootstrap resample **std** (``<metric>_r_boot`` /
    ``<metric>_r_boot_std`` from all_evaluation_results.csv; a normal-approx
    95% CI is mean ± 1.96·std), annotated with significance stars for the
    comparison of that feature set's correlation against the WPM baseline
    (from pairwise_significance.csv): * p<.05, ** p<.01, *** p<.001;
    no stars = not significant. The WPM baseline row carries no stars.
    Best bootstrap mean per column is bolded.

    ``target_cols_override`` / ``slice_filters_override`` / ``datasets_override``
    replace the module defaults (DATASET_TARGET_COLS / COMBINED_SLICE_FILTERS /
    ["Meco", "OneStop"]) when a caller needs a different target, a different
    preview slice, or a single dataset -- e.g. the comprehension tables, the
    OneStop/Hunting slice that the standard filters exclude, and the
    OneStop-only Hunting table (MECO columns there would just repeat the
    combined table). All default to None, so existing callers are unaffected.

    ``stars_from`` is retained for signature stability but has one valid
    value, "bootstrap" (paired-bootstrap difference test →
    "bootstrap" (paired-bootstrap difference test →
    combined_bootstrap_bootsig.tex). ``corr_metric`` picks the correlation
    ("pearson" or "spearman") shown in the cells and used for the stars;
    spearman writes the ``*_spearman.tex`` variants.

    ``show_dispersion=False`` drops the ``± std`` from every cell — same
    bootstrap means, same stars, same bolding, just the point estimate — and
    writes the ``*_noci.tex`` variants with their own ``\\label``, so the
    dispersion and no-dispersion tables can coexist in one document.
    """
    if stars_from not in BOOTSTRAP_STARS_SOURCES:
        raise ValueError(f"stars_from must be one of {list(BOOTSTRAP_STARS_SOURCES)}")
    if corr_metric not in BOOTSTRAP_CORR_METRICS:
        raise ValueError(f"corr_metric must be one of {list(BOOTSTRAP_CORR_METRICS)}")
    star_p_col, star_test_desc = BOOTSTRAP_STARS_SOURCES[stars_from]
    star_p_col = star_p_col.format(m=corr_metric)
    metric_col, metric_display = BOOTSTRAP_CORR_METRICS[corr_metric]
    # Every cell is reported to two decimals unless the caller overrides.
    if decimals is None:
        decimals = 2
    if eval_csv is None:
        eval_csv = EVAL_SAVE_DIR / "all_evaluation_results.csv"
    if pairwise_csv is None:
        pairwise_csv = EVAL_SAVE_DIR / "pairwise_significance.csv"
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "full"
    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_ORDERED

    if not eval_csv.exists():
        logger.warning("Evaluation CSV not found: %s", eval_csv)
        return ""
    df = pd.read_csv(eval_csv)
    if f"{metric_col}_r_boot" not in df.columns:
        logger.warning(
            "No bootstrap columns in %s — rerun evaluation.py to add them; "
            "skipping bootstrap combined table", eval_csv)
        return ""
    df = df[df["feature_set"].isin(feature_sets)]

    pw = pd.DataFrame()
    if pairwise_csv.exists():
        pw = pd.read_csv(pairwise_csv)
    else:
        logger.warning(
            "Pairwise significance CSV not found: %s — stars vs WPM omitted",
            pairwise_csv)

    datasets = datasets_override or ["Meco", "OneStop"]
    target_cols = target_cols_override or {d: DATASET_TARGET_COLS[d] for d in datasets}
    slice_filters = slice_filters_override or COMBINED_SLICE_FILTERS
    scopes = DATASET_SCOPES["OneStop"]

    # (fs, ds, tc, scope) -> (boot_mean, boot_std)
    values = {}
    for ds in datasets:
        preview, version_path = slice_filters[ds]
        sub = df[(df["dataset"] == ds)
                 & (df["preview"] == preview)
                 & (df["version_path"] == version_path)
                 & (df["target_col"].isin(target_cols[ds]))]
        for _, row in sub.iterrows():
            scope = DATASET_CONTENT_MODE_TO_SCOPE[ds].get(row["content_mode"])
            if scope is None or scope not in scopes:
                continue
            values[(row["feature_set"], ds, row["target_col"], scope)] = (
                row[f"{metric_col}_r_boot"], row[f"{metric_col}_r_boot_std"])
    if not values:
        logger.warning("No data for bootstrap combined table; skipping")
        return ""

    # (fs, ds, tc, scope) -> n_stars from the bootstrap test of fs vs WPM.
    # pairwise_significance.csv stores each pair once, in either a/b order.
    stars = {}
    if not pw.empty:
        for ds in datasets:
            preview, version_path = slice_filters[ds]
            sub = pw[(pw["dataset"] == ds)
                     & (pw["preview"] == preview)
                     & (pw["version_path"] == version_path)
                     & (pw["target_col"].isin(target_cols[ds]))
                     & ((pw["feature_set_a"] == BOOTSTRAP_BASELINE_FS)
                        | (pw["feature_set_b"] == BOOTSTRAP_BASELINE_FS))]
            for _, row in sub.iterrows():
                scope = DATASET_CONTENT_MODE_TO_SCOPE[ds].get(row["content_mode"])
                if scope is None or scope not in scopes:
                    continue
                # pairwise rows store the pair in either order, so orient the
                # correlations onto (feature set, baseline) before judging
                # which side the significant difference falls on.
                a_is_baseline = row["feature_set_a"] == BOOTSTRAP_BASELINE_FS
                fs = (row["feature_set_b"] if a_is_baseline
                      else row["feature_set_a"])
                r_a, r_b = row.get(f"{metric_col}_r_a"), row.get(f"{metric_col}_r_b")
                r_fs, r_base = (r_b, r_a) if a_is_baseline else (r_a, r_b)
                stars[(fs, ds, row["target_col"], scope)] = _directional_stars(
                    row.get(star_p_col), r_fs, r_base)

    # Force fixed-text-only feature sets to NA in non-Fixed scopes.
    for fs in feature_sets:
        if fs in FIXED_TEXT_ONLY_FEATURE_SETS:
            for ds in datasets:
                for tc in target_cols[ds]:
                    for scope in scopes:
                        if scope != "Fixed":
                            values[(fs, ds, tc, scope)] = (np.nan, np.nan)

    fs_present = [fs for fs in SMALL_FEATURE_SETS
                  if fs in feature_sets
                  and any((fs, ds, tc, scope) in values
                          for ds in datasets
                          for tc in target_cols[ds]
                          for scope in scopes)]
    if not fs_present:
        logger.warning("No feature sets present for bootstrap combined table")
        return ""

    columns = [(ds, tc, scope)
               for ds in datasets
               for tc in target_cols[ds]
               for scope in scopes]

    # Best bootstrap mean and max star count per column (star deficit is
    # phantom-padded so numbers stay vertically aligned).
    best = {}
    max_stars = {}
    for col in columns:
        means = [values[(fs, *col)][0] for fs in fs_present
                 if (fs, *col) in values
                 and not pd.isna(values[(fs, *col)][0])]
        best[col] = max(means) if means else None
        # Magnitude, not the signed value: the sign is not rendered, so a
        # significantly-worse cell needs the same phantom width as a better one.
        max_stars[col] = max(
            (abs(stars.get((fs, *col), 0)) for fs in fs_present), default=0)

    def _cell(fs, col):
        v = values.get((fs, *col))
        if v is None or pd.isna(v[0]):
            return "-"
        mean, std = v
        body = f"{mean:.{decimals}f}"
        if show_dispersion and not pd.isna(std):
            body += f" \\pm {std:.{decimals}f}"
        if best[col] is not None and np.isclose(mean, best[col], atol=1e-6):
            body = f"\\mathbf{{{body}}}"
        ns = 0 if fs == BOOTSTRAP_BASELINE_FS else stars.get((fs, *col), 0)
        star_str = _sig_marker(ns)
        deficit = max_stars[col] - abs(ns)
        pad = "\\phantom{^{" + "*" * deficit + "}}" if deficit > 0 else ""
        return f"${body}{star_str}{pad}$"

    n_scopes = len(scopes)
    target_blocks = []
    for ds in datasets:
        for _ in target_cols[ds]:
            target_blocks.append("c" * n_scopes)
    col_spec = "l" + "|".join(target_blocks)
    n_data_cols = sum(len(target_cols[d]) * n_scopes for d in datasets)
    n_total_cols = 1 + n_data_cols

    hdr_empty = "\\multicolumn{1}{c}{}"
    dataset_display = BOOTSTRAP_DATASET_DISPLAY
    # single_column renders into one column (table + \columnwidth) instead of
    # the default full-width float, for a one-column paper layout.
    _env = "table" if single_column else "table*"
    _width = "\\columnwidth" if single_column else "\\textwidth"
    lines = [
        f"\\begin{{{_env}}}[ht!]",
        "\\centering",
        # Kept commented out: the paper tables scale with \\resizebox, but the
        # hook is left in place for anyone typesetting them at natural size.
        "% \\small",
        f"\\resizebox{{{_width}}}{{!}}{{",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header 1: dataset names (c| on all but the last for the boundary rule).
    # Only earns its row when there is more than one dataset to tell apart --
    # on a single-dataset table the name is already in the caption and the
    # \label, and the banner spanning every column just adds a header level
    # to read past.
    if len(datasets) > 1:
        h1 = [hdr_empty]
        for i, ds in enumerate(datasets):
            span = len(target_cols[ds]) * n_scopes
            align = "c" if i == len(datasets) - 1 else "c|"
            h1.append(
                f"\\multicolumn{{{span}}}{{{align}}}"
                f"{{\\textbf{{{dataset_display[ds]}}}}}")
        lines.append(" & ".join(h1) + " \\\\")

    # Header 2: target labels (| only at the dataset boundary).
    h2 = [hdr_empty]
    for i, ds in enumerate(datasets):
        tcs = target_cols[ds]
        for j, tc in enumerate(tcs):
            is_boundary = (j == len(tcs) - 1 and i != len(datasets) - 1)
            align = "c|" if is_boundary else "c"
            label_text = _display(tc, DATASET_TARGET_COL_DISPLAY.get(ds, {}))
            h2.append(f"\\multicolumn{{{n_scopes}}}{{{align}}}{{{label_text}}}")
    # \cmidrule under the target row, as the prediction tables draw it, so the
    # target labels are visibly separated from the regime labels beneath them.
    lines.append(" & ".join(h2) + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

    # Header 3: Features + scope labels.
    h3 = ["\\textbf{Features}"]
    for ds in datasets:
        for _tc in target_cols[ds]:
            h3.extend(scopes)
    lines.append(" & ".join(h3) + " \\\\")
    lines.append("\\midrule")

    for i, fs in enumerate(fs_present):
        if (i > 0 and FEATURE_SET_GROUP_INDEX.get(fs)
                != FEATURE_SET_GROUP_INDEX.get(fs_present[i - 1])):
            lines.append("\\midrule")
        row = [_display(fs, FEATURE_SET_DISPLAY)]
        row.extend(_cell(fs, col) for col in columns)
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")

    lines.append("\\end{tabular}")
    lines.append("}")
    dispersion_caption = (
        ", $\\pm$ one bootstrap standard deviation "
        f", over {N_BOOTSTRAP:,} paired resamples; a normal-approximation 95\\% "
        "confidence interval is the mean $\\pm 1.96$ standard deviations. "
        if show_dispersion else
        f", over {N_BOOTSTRAP:,} paired resamples. ")
    lines.append(
        "\\caption{\\textbf{EyeScore, bootstrap estimates}. Bootstrap mean "
        f"{metric_display} correlations between EyeScore and standard language "
        f"proficiency tests{dispersion_caption}"
        f"{SIGNIFICANCE_LEGEND} "
        "``Fixed'' is the Fixed Text regime in which all the "
        "eye movement data is for the same texts presented to the test "
        "participant. ``Any'' is the Any Text regime in which no eye "
        "movement data is available for the texts of the test participant. "
        "The best result for each evaluation measure in each regime, "
        "dataset, and target proficiency test is marked in bold.}"
    )
    label_suffix = ""
    if corr_metric != "pearson":
        label_suffix += f"_{corr_metric}"
    if not show_dispersion:
        label_suffix += "_noci"
    lines.append(f"\\label{{tab:eyescore_seen_unseen_bootstrap{label_suffix}}}")
    lines.append(f"\\end{{{_env}}}")

    latex = "\n".join(lines)
    table_dir = save_dir / "fully_agg"
    table_dir.mkdir(parents=True, exist_ok=True)
    # combined_bootstrap[_spearman].tex carry the paired-bootstrap stars (the
    # only significance test behind a reported result).
    output_name = (BOOTSTRAP_COMBINED_OUTPUT if corr_metric == "pearson"
                   else BOOTSTRAP_COMBINED_SPEARMAN_OUTPUT)
    if not show_dispersion:
        output_name = output_name.replace(".tex", "_noci.tex")
    table_path = table_dir / output_name
    write_table(table_path, latex)
    logger.info("Saved bootstrap combined EyeScore table: %s", table_path)
    return latex


def _generate_per_item_agg_combined_table(
    eval_df: pd.DataFrame,
    save_dir: Path,
    feature_sets: List[str],
    target_cols_meco: List[str],
    target_cols_onestop: List[str],
):
    """Generate combined per_item_agg table showing OneStop article and paragraph separately."""
    # Get per_item_agg data for each dataset.
    # MECO's per-item version_path is "per_item_agg/all/all" — no article/
    # paragraph segment, since MECO has a single unit of analysis. The old
    # exact match against ".../paragraph" never hit any row, so this table
    # silently took the "missing data" branch on every run. Both the current
    # and the older paragraph-suffixed layout are accepted here; an exact
    # membership test rather than a prefix keeps superseded *_prerun_* trees
    # out even if one ever reaches the eval CSV.
    meco_pis = eval_df[
        (eval_df["dataset"] == "Meco")
        & (eval_df["version_path"].isin(
            ["per_item_agg/all/all", "per_item_agg/all/all/paragraph"]))
    ]
    onestop_pis = eval_df[(eval_df["dataset"] == "OneStop") & (eval_df["version_path"].str.startswith("per_item_agg"))]

    if meco_pis.empty or onestop_pis.empty:
        logger.info("Skipping per_item_agg combined table (missing data for one or both datasets)")
        return

    # For OneStop, separate article and paragraph
    article_pis = onestop_pis[onestop_pis["version_path"] == "per_item_agg/ordinary/all/article"]
    paragraph_pis = onestop_pis[onestop_pis["version_path"] == "per_item_agg/ordinary/all/paragraph"]

    # Filter to target columns
    article_pis = article_pis[article_pis["target_col"].isin(target_cols_onestop)]
    paragraph_pis = paragraph_pis[paragraph_pis["target_col"].isin(target_cols_onestop)]
    meco_pis = meco_pis[meco_pis["target_col"].isin(target_cols_meco)]

    if article_pis.empty or paragraph_pis.empty or meco_pis.empty:
        logger.info("Missing per_item_agg data for table generation")
        return

    # Generate LaTeX
    latex = _generate_per_item_agg_eyescore_latex(article_pis, paragraph_pis, meco_pis)

    if not latex:
        logger.info("No per_item_agg combined table generated")
        return

    table_dir = save_dir / "per_item_agg"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "combined.tex"
    write_table(table_path, latex)
    logger.info("Saved combined per_item_agg table: %s", table_path)


# ── Distance-coefficient triplet tables (multiple-regression debias models) ──
#
# Mirrors the bias_triplet_multi layout, but each cell is the fitted distance
# COEFFICIENT b_dist (+ significance) instead of the Pearson r. One table per MR
# model, read from that model's own pipeline tree (results_<model>/):
#   distance_mreg : eyescore ~ prof + dist          -> coef on dist
#   interaction   : eyescore ~ prof * dist (centered) -> main-effect coef on dist
# raw  = fit on all raw eyescores; Seen/New = refit on the inside/across-langs
# debiased (out-of-fold) scores.

# (dataset, target label, proficiency col, debias-suffix, meta version_path, unseen results subdir)
_COEF_COLSPEC = [
    ("Meco",    "LexTALE",   "lextale_score",   "",          "fully_agg/all/all",      "Meco/All/unseen/fully_agg/all/all"),
    ("Meco",    "Composite", "proficiency_agg", "_profagg",  "fully_agg/all/all",      "Meco/All/unseen/fully_agg/all/all"),
    ("OneStop", "LexTALE",   "lextale_score",   "",          "fully_agg/ordinary/all", "OneStop/Gathering/unseen/fully_agg/ordinary/all"),
    ("OneStop", "Michigan",  "michtest_score",  "_michtest", "fully_agg/ordinary/all", "OneStop/Gathering/unseen/fully_agg/ordinary/all"),
]
_COEF_FROWS = [("READING_SPEED", "WPM"), ("FIXATION_METRICS", "Avg. Fix."),
               ("S_CLUSTERS_NO_NORM", "S-Clusters"), ("WP_COEFS_NO_NORM", "WP-Coefs")]
_COEF_MODELS = {"two_step": "results_two_step",
                "distance_mreg": "results_distance_mreg",
                "interaction": "results_interaction"}


def _coef_b_dist(path, prof, dist_map, meta, model):
    """Fit the model on `path` and return (coef_on_distance, p), or None."""
    import os
    import statsmodels.formula.api as smf
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)[["participant_id", "eye_score", "L1"]].merge(
        meta[["participant_id", prof]], on="participant_id")
    df = df[df[prof] != -1].dropna(subset=["eye_score", prof, "L1"]).copy()
    df["dist"] = df["L1"].map(dist_map)
    df = df.dropna(subset=["dist"])
    if len(df) < 10 or df["L1"].nunique() < 2:
        return None
    if model == "two_step":
        # Two-step α: residualize eyescore on proficiency, then slope of the
        # residual on distance (the sequential fit the two-step correction uses).
        from scipy.stats import linregress
        s, i = np.polyfit(df[prof].to_numpy(float), df["eye_score"].to_numpy(float), 1)
        resid = df["eye_score"].to_numpy(float) - (s * df[prof].to_numpy(float) + i)
        lr = linregress(df["dist"].to_numpy(float), resid)
        return float(lr.slope), float(lr.pvalue)
    df["p_"] = df[prof] - df[prof].mean()
    df["d_"] = df["dist"] - df["dist"].mean()
    df["y"] = df["eye_score"]
    formula = "y ~ p_ + d_" if model == "distance_mreg" else "y ~ p_ * d_"
    r = smf.ols(formula, df).fit()
    return float(r.params["d_"]), float(r.pvalues["d_"])


def _coef_cell(v):
    if v is None:
        return r"$-$"
    coef, p = v
    st = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    body = f"{coef:.2f}" if coef < 0 else r"\phantom{-}" + f"{coef:.2f}"
    body += ("^{" + st + "}") if st else r"\phantom{^{***}}"
    return f"${body}$"


def _resolve_model_tree(model: str, tree: str, results_suffix: str = "") -> "str | None":
    """Locate a debias model's results tree for the current run.

    Two corrections to the old fixed lookup:
      * the canonical two_step run writes the UNSUFFIXED results/ tree (running
        without --debias-methods uses DEFAULT_DEBIAS_METHOD), so results_two_step/
        never exists and two_step was silently skipped;
      * with --results-suffix the run's trees are results_<model>_<suffix>/, but
        the fixed names pointed at the unsuffixed trees — so a z-scored-distance
        run was reading the ORG-debiased scores and only the evaluation metric
        differed. Prefer the suffixed tree, then the plain one.
    Returns the tree name, or None if nothing usable exists.
    """
    cands = []
    if results_suffix:
        # results_<model>_<suffix> is the fan-out layout. Do NOT fall back to
        # results_<suffix>: when the suffix is itself a method name that would
        # read the OTHER method's tree (fitting two_step on interaction's
        # scores). A suffix that encodes only a distance is handled below.
        cands.append(f"{tree}_{results_suffix}")
        if not any(results_suffix == m or results_suffix.startswith(m + "_")
                   for m in DEBIAS_METHODS):
            cands.append(f"results_{results_suffix}")
    cands.append(tree)
    if model == DEFAULT_DEBIAS_METHOD:
        # The canonical run (no --debias-methods) uses DEFAULT_DEBIAS_METHOD and
        # writes the unsuffixed results/ tree, so results_two_step/ never exists.
        cands.append("results")
    for c in cands:
        if Path(f"src/methods/EyeScore/{c}").is_dir():
            return c
    return None


def _run_debias_method(results_suffix: str) -> str:
    """The debias method this run is about (the suffix's leading method token,
    else the default). Used both for the paper/lang_bias/<method>_<dist>/ name
    and to emit only that method's table there."""
    for m in sorted(DEBIAS_METHODS, key=len, reverse=True):
        if results_suffix == m or results_suffix.startswith(m + "_"):
            return m
    return DEFAULT_DEBIAS_METHOD


def generate_distance_coef_tables(save_dir: Path = None,
                                  distance_type: str = "syntactic+genetic",
                                  results_suffix: str = "") -> list:
    """Write one distance-coefficient triplet .tex per MR model (two_step,
    distance_mreg, interaction) into save_dir (default
    paper/lang_bias/<distance_type>/). Each model's debiased scores come from
    the tree resolved by _resolve_model_tree for this run; a model with no
    usable tree is skipped."""
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "lang_bias" / distance_type
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    meta_by_ds, dist_by_ds = {}, {}
    for ds, _, _, _, vp, _ in _COEF_COLSPEC:
        if ds not in meta_by_ds:
            m = load_metadata(ds, vp)
            meta_by_ds[ds] = m
            dist_by_ds[ds] = compute_typological_distances(
                sorted(m["L1"].dropna().unique()), distance_type=distance_type)

    written = []
    # Only this run's own method: the directory is paper/lang_bias/<method>_<dist>,
    # so emitting every model there would mix trees from other runs (and duplicate
    # them across directories). Each other method has its own run/directory.
    run_method = _run_debias_method(results_suffix or "")
    for model, tree in _COEF_MODELS.items():
        if model != run_method:
            continue
        resolved = _resolve_model_tree(model, tree, results_suffix)
        if resolved is None:
            logger.warning("distance-coef table: no tree for %s (tried %s*); skipping", model, tree)
            continue
        logger.info("distance-coef table: %s -> %s", model, resolved)
        tree = resolved
        rows = []
        for fs, flab in _COEF_FROWS:
            cells = []
            for ds, tl, prof, suf, _, ud in _COEF_COLSPEC:
                base = f"src/methods/EyeScore/{tree}/{ud}/{fs}"
                raw = _coef_b_dist(f"{base}.csv", prof, dist_by_ds[ds], meta_by_ds[ds], model)
                seen = _coef_b_dist(f"{base}_typo_calibrated_inside_langs{suf}.csv", prof, dist_by_ds[ds], meta_by_ds[ds], model)
                new = _coef_b_dist(f"{base}_typo_calibrated_across_langs{suf}.csv", prof, dist_by_ds[ds], meta_by_ds[ds], model)
                cells += [_coef_cell(raw), _coef_cell(seen), _coef_cell(new)]
            rows.append(f"{flab} & " + " & ".join(cells) + r" \\")
        model_name = {
            "two_step": r"two-step residual regression (slope of residual $\sim$ dist, after residualizing on prof.)",
            "distance_mreg": r"multiple regression ($\mathrm{eyescore}\sim\mathrm{prof}+\mathrm{dist}$)",
            "interaction": r"interaction model ($\mathrm{eyescore}\sim\mathrm{prof}\times\mathrm{dist}$, main-effect coef.)",
        }[model]
        lines = [
            r"\begin{table*}[ht!]", r"\centering", r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lccc|ccc|ccc|ccc}", r"\toprule",
            r" & \multicolumn{6}{c|}{\textbf{MECO}} & \multicolumn{6}{c}{\textbf{OneStop}} \\",
            r" & \multicolumn{3}{c}{LexTALE} & \multicolumn{3}{c|}{Composite} & \multicolumn{3}{c}{LexTALE} & \multicolumn{3}{c}{Michigan} \\ \cmidrule{2-13}",
            r" & \multicolumn{1}{c}{$b_{\mathrm{dist}}$} & \multicolumn{2}{c|}{$b_{\mathrm{dist}}^{\mathrm{db}}$} & "
            r"\multicolumn{1}{c}{$b_{\mathrm{dist}}$} & \multicolumn{2}{c|}{$b_{\mathrm{dist}}^{\mathrm{db}}$} & "
            r"\multicolumn{1}{c}{$b_{\mathrm{dist}}$} & \multicolumn{2}{c|}{$b_{\mathrm{dist}}^{\mathrm{db}}$} & "
            r"\multicolumn{1}{c}{$b_{\mathrm{dist}}$} & \multicolumn{2}{c}{$b_{\mathrm{dist}}^{\mathrm{db}}$} \\ \cmidrule{2-13}",
            r"\textbf{Features} &  & \textit{Seen L1} & \textit{New L1} &  & \textit{Seen L1} & \textit{New L1} &  & \textit{Seen L1} & \textit{New L1} &  & \textit{Seen L1} & \textit{New L1} \\",
            r"\midrule \cmidrule{1-13}",
        ]
        lines += rows[:1] + [r"\midrule"] + rows[1:]
        lines[-1] = lines[-1] + r" \bottomrule"
        lines += [
            r"\end{tabular}%", r"}",
            (r"\caption{\textbf{Distance coefficient} $b_{\mathrm{dist}}$ from the " + model_name +
             r" in the Any Text regime. $b_{\mathrm{dist}}$ is the fitted coefficient of the "
             r"participant's L1$\to$English typological distance; the left column of each block is "
             r"fit on all raw eyescores, and $b_{\mathrm{dist}}^{\mathrm{db}}$ (Seen / New L1) refits "
             r"it on the held-out debiased scores (inside- / across-language CV). "
             r"Significance: $^{*}\,p<0.05$, $^{**}\,p<0.01$, $^{***}\,p<0.001$.}"),
            r"\label{tab:lang-bias-triplet-coef-" + model + "}",
            r"\end{table*}",
        ]
        out = save_dir / f"triplets_{model}_coef.tex"
        write_table(out, "\n".join(lines))
        written.append(str(out))
        logger.info("Wrote distance-coefficient table: %s", out)
    return written


def generate_slope_influence_table(
    dataset: str = "Meco",
    feature_set: str = "READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM",
    flabel: str = "COMBINED",
    target: str = "lextale_score",
    save_dir: Path = None,
    distance_type: str = "syntactic+genetic",
    results_tree: str = "results_two_step",
) -> Optional[str]:
    """Per-language influence on the EyeScore residual~distance relationship,
    via statsmodels OLSInfluence on the language-level regression (each L1 = one
    observation: its mean proficiency-residual vs. typological distance). Reports
    Cook's distance, DFBETAS (distance coef.) and leverage per L1. Writes
    <save_dir>/slope_influence_<dataset>_<flabel>.tex."""
    import os
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import OLSInfluence
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "lang_bias" / distance_type
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    DIRS = {"Meco": ("fully_agg/all/all", "Meco/All/unseen/fully_agg/all/all"),
            "OneStop": ("fully_agg/ordinary/all", "OneStop/Gathering/unseen/fully_agg/ordinary/all")}
    mvp, ud = DIRS[dataset]
    path = f"src/methods/EyeScore/{results_tree}/{ud}/{feature_set}.csv"
    if not os.path.exists(path):
        logger.warning("slope-influence: %s missing; skipping", path); return None
    meta = load_metadata(dataset, mvp)
    dist = compute_typological_distances(sorted(meta["L1"].dropna().unique()), distance_type=distance_type)
    df = pd.read_csv(path)[["participant_id", "eye_score", "L1"]].merge(
        meta[["participant_id", target]], on="participant_id")
    df = df[df[target] != -1].dropna(subset=["eye_score", target, "L1"]).copy()
    df["d"] = df["L1"].map(dist); df = df.dropna(subset=["d"])
    if len(df) < 3:
        logger.warning(
            "slope-influence: <3 rows with distance for %s/%s (distance_type=%r); skipping",
            dataset, flabel, distance_type,
        )
        return None
    s, i = np.polyfit(df[target], df["eye_score"], 1)
    df["r"] = df["eye_score"] - (s * df[target] + i)
    g = df.groupby("L1").agg(mr=("r", "mean"), n=("r", "size"), d=("d", "first")).reset_index()
    if len(g) < 4:
        logger.warning("slope-influence: <4 L1 groups for %s/%s; skipping", dataset, flabel); return None
    res = sm.OLS(g["mr"], sm.add_constant(g[["d"]])).fit()
    inf = OLSInfluence(res)
    g["cooks"] = inf.cooks_distance[0]
    g["dfbetas"] = inf.dfbetas[:, 1]
    g["lev"] = inf.hat_matrix_diag
    thr = 4.0 / len(g)
    g = g.sort_values("cooks", ascending=False)
    body = []
    for _, x in g.iterrows():
        dag = r"$^{\dagger}$" if x["cooks"] > thr else ""
        body.append(f"{x['L1']} & {int(x['n'])} & {x['d']:.2f} & {x['cooks']:.3f}{dag} & "
                    f"{x['dfbetas']:+.2f} & {x['lev']:.3f} \\\\")
    lines = [
        r"\begin{table}[ht!]", r"\centering", r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"L1 & $n$ & dist & Cook's $D$ & DFBETAS & leverage \\",
        r"\midrule",
    ] + body + [
        r"\bottomrule", r"\end{tabular}%", r"}",
        (r"\caption{\textbf{Per-language influence on the " + dataset + " (" + flabel +
         r") EyeScore residual$\sim$distance relationship} (Any Text). Diagnostics from "
         r"\texttt{statsmodels}' \texttt{OLSInfluence} on the language-level regression (each L1 is one "
         r"observation: its mean proficiency-residual vs.\ typological distance to English). Cook's $D$, "
         r"DFBETAS for the distance coefficient, and leverage; $^{\dagger}$ marks $\text{Cook's }D > 4/k$.}"),
        r"\label{tab:slope-influence-" + dataset.lower() + "-" + flabel.lower() + "}",
        r"\end{table}",
    ]
    out = save_dir / f"slope_influence_{dataset}_{flabel}.tex"
    write_table(out, "\n".join(lines))
    logger.info("Wrote slope-influence table: %s", out)
    return str(out)


# (dataset, target label, prof col, debias suffix, meta version_path, unseen results subdir)
_L1FX_COLSPEC = [
    ("Meco",    "LexTALE",   "lextale_score",   "",          "fully_agg/all/all",      "Meco/All/unseen/fully_agg/all/all"),
    ("Meco",    "Composite", "proficiency_agg", "_profagg",  "fully_agg/all/all",      "Meco/All/unseen/fully_agg/all/all"),
    ("OneStop", "LexTALE",   "lextale_score",   "",          "fully_agg/ordinary/all", "OneStop/Gathering/unseen/fully_agg/ordinary/all"),
    ("OneStop", "Michigan",  "michtest_score",  "_michtest", "fully_agg/ordinary/all", "OneStop/Gathering/unseen/fully_agg/ordinary/all"),
]
_L1FX_FROWS = [("READING_SPEED", "WPM"), ("FIXATION_METRICS", "Avg. Fix."),
               ("S_CLUSTERS_NO_NORM", "S-Clusters"), ("WP_COEFS_NO_NORM", "WP-Coefs")]


def _l1_ftest_p(path, prof, meta):
    """ANOVA F-test p of L1 on the proficiency-residualized eyescore: does native
    language still predict the eyescore after accounting for proficiency? None if
    missing/too thin."""
    import os
    import statsmodels.formula.api as smf
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)[["participant_id", "eye_score", "L1"]].merge(
        meta[["participant_id", prof]], on="participant_id")
    df = df[df[prof] != -1].dropna(subset=["eye_score", prof, "L1"]).copy()
    if len(df) < 5 or df["L1"].nunique() < 2:
        return None
    s, i = np.polyfit(df[prof], df["eye_score"], 1)
    df["r"] = df["eye_score"] - (s * df[prof] + i)
    return float(smf.ols("r ~ C(L1)", df.rename(columns={"L1": "L1"})).fit().f_pvalue)


def generate_l1_effect_table(save_dir: Path = None, distance_type: str = "syntactic+genetic") -> Optional[str]:
    """L1-effect-removal table for the offset debias methods (RE = re_l1, FE =
    fixed_l1). Each cell is the ANOVA F-test p of native language on the
    proficiency-residualized eyescore — raw vs the RE-/FE-debiased (out-of-fold)
    scores. High p = per-language structure removed. Writes
    <save_dir>/l1_effect_removal.tex."""
    if save_dir is None:
        save_dir = TABLES_SAVE_DIR / "lang_bias" / distance_type
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    if not (Path("src/methods/EyeScore/results_re_l1").exists()
            and Path("src/methods/EyeScore/results_fixed_l1").exists()):
        logger.warning("l1_effect table: results_re_l1/ or results_fixed_l1/ missing; skipping")
        return None
    meta_by_ds = {ds: load_metadata(ds, vp) for ds, _, _, _, vp, _ in _L1FX_COLSPEC}

    def pcell(p):
        if p is None:
            return r"$-$"
        st = "^{***}" if p < .001 else "^{**}" if p < .01 else "^{*}" if p < .05 else ""
        body = "{<}.001" if p < .001 else f"{p:.3f}"
        return f"${body}{st}$"

    rows = []
    for fs, flab in _L1FX_FROWS:
        cells = []
        for ds, tl, prof, suf, vp, ud in _L1FX_COLSPEC:
            meta = meta_by_ds[ds]
            raw = _l1_ftest_p(f"src/methods/EyeScore/results_two_step/{ud}/{fs}.csv", prof, meta)
            re_ = _l1_ftest_p(f"src/methods/EyeScore/results_re_l1/{ud}/{fs}_typo_calibrated_inside_langs{suf}.csv", prof, meta)
            fe = _l1_ftest_p(f"src/methods/EyeScore/results_fixed_l1/{ud}/{fs}_typo_calibrated_inside_langs{suf}.csv", prof, meta)
            cells += [pcell(raw), pcell(re_), pcell(fe)]
        rows.append(f"{flab} & " + " & ".join(cells) + r" \\")
    lines = [
        r"\begin{table*}[ht!]", r"\centering", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccc|ccc|ccc|ccc}", r"\toprule",
        r" & \multicolumn{6}{c|}{\textbf{MECO}} & \multicolumn{6}{c}{\textbf{OneStop}} \\",
        r" & \multicolumn{3}{c}{LexTALE} & \multicolumn{3}{c|}{Composite} & \multicolumn{3}{c}{LexTALE} & \multicolumn{3}{c}{Michigan} \\ \cmidrule{2-13}",
        r"\textbf{Features} & raw & RE & FE & raw & RE & FE & raw & RE & FE & raw & RE & FE \\",
        r"\midrule \cmidrule{1-13}",
    ] + rows[:1] + [r"\midrule"] + rows[1:]
    lines[-1] = lines[-1] + r" \bottomrule"
    lines += [
        r"\end{tabular}%", r"}",
        (r"\caption{\textbf{L1 structure removed by the offset debias models} (Any Text). Each cell is the "
         r"ANOVA $F$-test $p$-value of native language (L1) on the proficiency-residualized eyescore — "
         r"i.e.\ whether L1 still predicts the eyescore after accounting for proficiency. \textbf{raw}: raw "
         r"eyescore; \textbf{RE}: random-intercept (\texttt{linearmodels} RandomEffects, shrunk); \textbf{FE}: "
         r"fixed effect (\texttt{linearmodels} PanelOLS entity effects). High $p$ = L1 structure removed. "
         r"RE/FE are inside-language (Seen L1) only. Significance: $^{*}\,p<0.05$, $^{**}\,p<0.01$, $^{***}\,p<0.001$.}"),
        r"\label{tab:l1-effect-removal}",
        r"\end{table*}",
    ]
    out = save_dir / "l1_effect_removal.tex"
    write_table(out, "\n".join(lines))
    logger.info("Wrote L1-effect-removal table: %s", out)
    return str(out)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-suffix", default="",
        help="Suffix routing inputs/outputs to sibling trees: reads "
             "src/methods/EyeScore/results_<suffix>/ + "
             "src/evaluation/EyeScore/results_<suffix>/, writes to "
             "src/evaluation/EyeScore/tables_<suffix>/. Default: empty "
             "(canonical results/+tables/ trees). Use the same suffix you "
             "passed to src.run.eyescore + evaluation.py + "
             "language_bias.py so tables read those outputs.",
    )
    parser.add_argument(
        "--distance-type", default=None,
        help="URIEL+ distance type these tables were built with. Overrides the "
             "inference from --results-suffix below, which cannot resolve a "
             "composed '<method>_<distance>' suffix. Pass this whenever the "
             "suffix encodes both a debias method and a distance variant.",
    )
    args = parser.parse_args()

    if args.results_suffix:
        suffix = f"_{args.results_suffix}"
        RESULTS_DIR = Path(f"src/methods/EyeScore/results{suffix}")
        EVAL_SAVE_DIR = Path(f"src/evaluation/EyeScore/results{suffix}")
        TABLES_SAVE_DIR = Path(f"src/evaluation/EyeScore/tables{suffix}")
        BIAS_SUMMARY_DIR = EVAL_SAVE_DIR / "lang_bias"
        # The suffix carries three orthogonal meanings depending on the caller:
        #   • a URIEL distance type (the --typo-calib-distance-type sibling run):
        #     the bias_correlation_summary.csv only holds that distance's rows,
        #     so rebind the filter constants to it.
        #   • a debias METHOD name (the --debias-methods fan-out): the distance
        #     type is unchanged (canonical), only the tree location differs — do
        #     NOT rebind, or the typology distance lookups get an invalid metric
        #     and empty out (crashing e.g. generate_slope_influence_table).
        #   • a SCORER name (the --scorers fan-out), possibly composed as
        #     "<scorer>_<method>": likewise the distance type is unchanged — do
        #     NOT rebind. We recognize it by its leading token.
        #   • an explicit --distance-type always wins: a composed
        #     "<method>_<distance>" suffix is not resolvable by the heuristic
        #     below, which would mistake the whole string for a URIEL metric.
        scorer_tok = args.results_suffix.split("_", 1)[0]
        if args.distance_type:
            BIAS_DISTANCE_TYPE = args.distance_type
            BIAS_CORR_DISTANCE_TYPE = args.distance_type
        elif (args.results_suffix not in DEBIAS_METHODS
                and args.results_suffix not in SCORER_NAMES
                and scorer_tok not in SCORER_NAMES):
            BIAS_DISTANCE_TYPE = args.results_suffix
            BIAS_CORR_DISTANCE_TYPE = args.results_suffix
        logger.info(
            "Using suffixed trees: results=%s eval=%s tables=%s bias_summary=%s distance=%s",
            RESULTS_DIR, EVAL_SAVE_DIR, TABLES_SAVE_DIR, BIAS_SUMMARY_DIR,
            BIAS_DISTANCE_TYPE,
        )

    # paper/lang_bias/<method>_<distance> — both axes in the name, so two
    # debias methods never overwrite each other and a results suffix can never
    # be mistaken for a URIEL metric in the path.
    def _lang_bias_dirname(results_suffix: str, distance_type: str) -> str:
        method = _run_debias_method(results_suffix)
        if distance_type == TYPO_CALIB_DISTANCE_TYPE:
            dist = "org"                                   # syntactic+genetic
        elif distance_type == "syntactic+genetic+scriptural":
            dist = "raw3"                                  # 3 components, unweighted raw mean
        elif distance_type.startswith("z:"):
            dist = "z"                                     # z-scored components
        elif distance_type.startswith("mm:"):
            dist = "mm3"                                   # min-max components, bounded [0,1]
        elif distance_type.startswith("max:"):
            dist = "max3"                                  # max-normalised components, [0,1]
        elif distance_type.startswith("z2sd:"):
            dist = "z2sd"                                  # z components, +-2SD anchored to [0,1]
        elif distance_type.startswith("sd:"):
            dist = "sd3"                                   # components / SD, positive, uncentred
        else:
            dist = re.sub(r"[^0-9A-Za-z]+", "_", distance_type).strip("_") or "dist"
        # Centred-correction runs get their own directories so they cannot be
        # confused with, or overwrite, the uncentred ones.
        if os.environ.get("EYESCORE_CENTER_DISTANCE", "").lower() in ("1", "true", "yes"):
            dist = f"{dist}ctr"
        return f"{method}_{dist}"

    # Exploratory variants of the lang_bias tables, one directory per
    # (method, distance) combination. They stay with this module's other
    # output: only src.run.paper writes into the published paper/ tree.
    LANG_BIAS_TABLES_DIR = TABLES_SAVE_DIR / "lang_bias" / _lang_bias_dirname(
        args.results_suffix or "", BIAS_DISTANCE_TYPE
    )
    logger.info("lang_bias tables -> %s  (distance=%s)", LANG_BIAS_TABLES_DIR, BIAS_DISTANCE_TYPE)

    logger.info("Generating combined OneStop × Meco tables...")
    generate_full_combined_tables()
    generate_debiased_combined_tables()
    generate_all_tables()
    # Distance-coefficient triplet tables for the MR debias models. Reads the
    # fixed results_distance_mreg/ + results_interaction/ trees (method-
    # independent), so it's generated once regardless of --results-suffix.
    generate_distance_coef_tables(save_dir=LANG_BIAS_TABLES_DIR, distance_type=BIAS_DISTANCE_TYPE, results_suffix=args.results_suffix or "")
    # Per-language influence (Cook's D / DFBETAS / leverage) on the distance slope.
    generate_slope_influence_table(save_dir=LANG_BIAS_TABLES_DIR, dataset="Meco", distance_type=BIAS_DISTANCE_TYPE)
    generate_slope_influence_table(save_dir=LANG_BIAS_TABLES_DIR, dataset="OneStop", distance_type=BIAS_DISTANCE_TYPE)
    # L1-effect-removal table for the offset methods (RE / FE): F-test p of L1.
    generate_l1_effect_table(save_dir=LANG_BIAS_TABLES_DIR, distance_type=BIAS_DISTANCE_TYPE)
