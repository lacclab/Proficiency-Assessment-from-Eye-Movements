"""EyeScore evaluation plots: correlation bars by feature set and by target,
and EyeScore-vs-target scatters, written under src/evaluation/EyeScore/plots/
by `python -m src.run.eyescore --stage evaluation` (plot_all_results)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm


from src.evaluation.utils import suppress_titles
from src.evaluation.EyeScore.evaluation import (
    EVAL_SAVE_DIR,
    RESULTS_DIR,
    eye_score_csv_path,
    preview_dir_segments,
    BASELINE_FEATURE_SET,
)
from src.methods.EyeScore.calculation import (
    RAW_VARIANT,
)
from src.constants import (
    ALL_TESTS,
    RELEVANT_FEATURE_GROUPS_COMBINATIONS_SMALL,
    DATA_PATH,
    TestCols,
    COMPREHENSION_COLS,
    Fields,
)

logger = logging.getLogger(__name__)

PLOTS_SAVE_DIR = Path("src/evaluation/EyeScore/plots")

CORRELATION_METRICS = ["pearson_r", "spearman_r"]
METRIC_TITLES = {
    "pearson_r": "Pearson r",
    "spearman_r": "Spearman r",
}

# ── Display name mappings ────────────────────────────────────────────────────
FEATURE_SET_DISPLAY_NAMES = {
    "READING_SPEED": "Reading Speed",
    "FIXATION_METRICS": "Fixation Metrics",
    "S_CLUSTERS": "Syntactic Clusters",
    "S_CLUSTERS_NO_NORM": "Syntactic Clusters (unnorm.)",
    "WP_COEFS": "Word-Property Coef.",
    "WP_COEFS_NO_NORM": "Word-Property Coef. (unnorm.)",
    "WP_COEFS_NO_INTERCEPT": "Word-Property Coef. (no intercept)",
    "WP_COEFS_NO_NORM_NO_INTERCEPT": "Word-Property Coef. (unnorm., no intercept)",
    "TRANSITIONS": "Transitions",
    "WFC": "WFC",
    # Combinations
    "READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM": "Speed + Fix. + Syn. + WP (unnorm.)",
    "FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM": "Fix. + Syn. + WP (unnorm.)",
    "S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM": "Syn. + WP (unnorm.)",
}

TARGET_COL_DISPLAY_NAMES = {
    "michtest_score": "Michigan Test (total)",
    "lextale_score": "LexTALE",
    "MPT_listening_score": "MPT Listening",
    "MPT_grammar_score": "MPT Grammar",
    "MPT_vocabulary_score": "MPT Vocabulary",
    "MPT_reading_score": "MPT Reading",
    "MPT_listen_grammar": "MPT Listen + Grammar",
    "MPT_vocab_read": "MPT Vocab + Reading",
    "MPT_grammar_vocab_read": "MPT Grammar + Vocab + Reading",
    "comprehension_score-regular_trials": "Comprehension (regular)",
    "comprehension_score-repeated_reading": "Comprehension (reread)",
    "proficiency_agg": "Composite",
}

PREVIEW_DISPLAY_NAMES = {
    "All": "All Conditions",
    "Hunting": "Hunting (preview)",
    "Gathering": "Gathering (no preview)",
}

CONTENT_MODE_DISPLAY_NAMES = {
    "all": "All L1",
    "seen": "L1 w/ same content",
    "unseen": "L1 w/ different content",
}

# Target columns for bar/scatter plots (readable subset)
PLOT_TARGET_COLS = [TestCols.MICHIGEN_TEST_COL, TestCols.LEXTALE_COL, TestCols.PROFICIENCY_AGG_COL]

# Target columns for the separate "comprehension" bar plots.
PLOT_COMPREHENSION_COLS = list(COMPREHENSION_COLS)

# Canonical x-axis ordering for target columns in bar_by_target plots:
# Michigan on the left, then proficiency_agg, then LexTALE on the right (when present).
# Comprehension targets are also pinned in a stable order in case they appear
# together in any future bar plot — regular trials before repeated reading.
TARGET_COL_ORDER = [
    "michtest_score",
    "proficiency_agg",
    "lextale_score",
    "comprehension_score-regular_trials",
    "comprehension_score-repeated_reading",
]

# Canonical left→right ordering for feature sets in bar_by_target plots:
# most features first, then descending in number of feature groups, with
# Reading Speed (the baseline) pinned on the far right.
FEATURE_SET_ORDER = [

    "READING_SPEED",
    "FIXATION_METRICS",
    # "S_CLUSTERS",
    "S_CLUSTERS_NO_NORM",
    # "WP_COEFS_NO_INTERCEPT",
    # "WP_COEFS",
    "WP_COEFS_NO_NORM",
    # "WP_COEFS_NO_NORM_NO_INTERCEPT",
    "TRANSITIONS",
    "WFC",
    "S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM",
    "FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM",
    "READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM",
]

# Feature sets that are inherently per-text — only valid in the "seen" content
# regime; bar/scatter plotters use this to skip them in non-seen modes.
FIXED_TEXT_ONLY_FEATURE_SETS = {"TRANSITIONS", "WFC"}

# Reading speed is the baseline — drawn in a warm grey color
# that complements the earthy, sophisticated feature-set palette.
BASELINE_COLOR = "#6B6B7A"  # muted slate grey

# Fixed y-axis limits for bar_by_target plots so the scale does not depend
# on the highest value in any individual plot. Correlations live in [-1, 1].
BAR_BY_TARGET_YLIM = (0, 1.0)

# Fixed axis limits for scatter plots (actual vs predicted values)
# Allows consistent comparison across plots
# Note: X-axis (targets) are percentages (0-100); Y-axis (eye_scores) are normalized values
SCATTER_XLIM = (0, 100)           # Target scores range 0-100 (percentages)
SCATTER_YLIM_READING_SPEED = (-3, 6)  # Reading speed ranges approximately -2.2 to +5.0
SCATTER_YLIM_OTHER = (-1, 1)      # Other features are normalized to approximately [-1, 1]

BAR_PALETTE = [ "#2A6E9B","#338B28",  "#E91E63","#FFD54F", "#91BFD9", "#8E24AA"]
SCATTER_PALETTE = "tab20"


def _display_name(value: str, mapping: dict) -> str:
    return mapping.get(value, value)


SINGLE_FEATURE_SETS = {
    "FIXATION_METRICS",
    "S_CLUSTERS",
    "S_CLUSTERS_NO_NORM",
    "WP_COEFS",
    "WP_COEFS_NO_NORM",
    "WP_COEFS_NO_INTERCEPT",
    "WP_COEFS_NO_NORM_NO_INTERCEPT",
    "TRANSITIONS",
    "WFC",
}


def _simplified_plot_tail(
    dataset: str, preview: str, content_mode: str, version_path: str,
) -> Path:
    """Translate the verbose (preview, content_mode, version_path) tuple into
    a paper-friendly tail under plots/{dataset}/.

    Meco:
        fully_agg/all/all           -> all
        seen_unseen/all/all/odd     -> odd     (placeholder until the
        seen_unseen/all/all/even    -> even     calculation pipeline emits
                                                 averaged seen/unseen variants)
    OneStop:
        Gathering / {content_mode} / fully_agg/ordinary/{difficulty}
            -> {content_mode}/{difficulty}
        Hunting  / {content_mode} / fully_agg/ordinary/{difficulty}
            -> Hunting/{content_mode}/{difficulty}   (secondary preview keeps
            its own subtree so it can't overwrite Gathering's figures)

    Falls back to the legacy preview/content_mode/version_path layout for any
    combination not recognised, so unfamiliar configs still produce output.
    """
    vp = version_path.strip("/")
    parts = vp.split("/") if vp else []

    if dataset == "Meco":
        if vp == "fully_agg/all/all":
            return Path("all")
        if vp.startswith("seen_unseen/all/all/"):
            half = parts[-1]  # "odd" or "even"
            if half in ("odd", "even"):
                return Path(half)

    if dataset == "OneStop":
        # Expected shape: fully_agg/ordinary/{adv,all,ele}
        if len(parts) == 3 and parts[0] == "fully_agg" and parts[1] == "ordinary":
            difficulty = parts[2]
            return Path(
                *preview_dir_segments(dataset, preview), content_mode, difficulty,
            )

    # Fallback — preserve full structural path so nothing silently disappears.
    return Path(preview) / content_mode / version_path


def _build_feature_set_color_map(feature_sets) -> dict:
    """Return a stable {feature_set: color} mapping using combo/single color scheme.

    Singles (individual feature groups, including TRANSITIONS/WFC) get one
    palette; combinations (multiple groups joined) get another. The baseline
    keeps its dedicated color. Membership is decided by SINGLE_FEATURE_SETS,
    not by position in the input list.
    """
    feature_sets = list(feature_sets)
    feature_sets_no_base = [fs for fs in feature_sets if fs != BASELINE_FEATURE_SET]

    single_fs = [fs for fs in feature_sets_no_base if fs in SINGLE_FEATURE_SETS]
    combo_fs = [fs for fs in feature_sets_no_base if fs not in SINGLE_FEATURE_SETS]

    # Single-feature palette extended to cover TRANSITIONS / WFC.
    single_colors = ['#FF69B4', '#4169E1', '#9370DB', '#0FA3B1', '#F7B538']
    combo_colors = ['#FFD700', '#FF8C00', '#FF4444', '#A6324B']

    color_map = {}
    if BASELINE_FEATURE_SET in feature_sets:
        color_map[BASELINE_FEATURE_SET] = BASELINE_COLOR

    for i, fs in enumerate(combo_fs):
        color_map[fs] = combo_colors[i] if i < len(combo_colors) else combo_colors[-1]

    for i, fs in enumerate(single_fs):
        color_map[fs] = single_colors[i] if i < len(single_colors) else single_colors[-1]

    return color_map


# Mapping from a main correlation metric column to its baseline-diff p-value column
# produced by evaluation.compute_correlation / _baseline_diff_stats.
_BASELINE_DIFF_P_COLUMN = {
    "pearson_r": "baseline_diff_p_pearson",
    "spearman_r": "baseline_diff_p_spearman",
}

# Mapping from a main correlation metric column to its Fisher-z CI column pair.
_METRIC_CI_COLUMNS = {
    "pearson_r": ("pearson_ci_low", "pearson_ci_high"),
    "spearman_r": ("spearman_ci_low", "spearman_ci_high"),
}


def _ci_pivots(df: pd.DataFrame, metric: str, index: str, columns: str,
               main_pivot: pd.DataFrame):
    """Return (ci_low_pivot, ci_high_pivot) aligned to main_pivot, or (None, None)
    if the CI columns for this metric are not present in df."""
    ci_cols = _METRIC_CI_COLUMNS.get(metric)
    if not ci_cols or not all(c in df.columns for c in ci_cols):
        return None, None
    lo_col, hi_col = ci_cols
    lo = df.pivot_table(index=index, columns=columns, values=lo_col, aggfunc="first")
    hi = df.pivot_table(index=index, columns=columns, values=hi_col, aggfunc="first")
    lo = lo.reindex(index=main_pivot.index, columns=main_pivot.columns)
    hi = hi.reindex(index=main_pivot.index, columns=main_pivot.columns)
    return lo, hi


def _asymmetric_yerr(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Build a shape-(2, N) asymmetric yerr array for ax.errorbar. NaN-safe:
    any bar with non-finite value or missing CI gets 0 magnitudes (so errorbar
    draws nothing at that position)."""
    values = np.asarray(values, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    finite_v = np.isfinite(values)
    lower = np.where(finite_v & np.isfinite(lo), np.maximum(values - lo, 0.0), 0.0)
    upper = np.where(finite_v & np.isfinite(hi), np.maximum(hi - values, 0.0), 0.0)
    return np.array([lower, upper])


def _sig_stars(p: float) -> str:
    """APA-style significance stars for a two-sided p-value."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# Target style convention for the bar_by_feature_set plot — the two targets
# share the feature-set's color and are distinguished only by a fill pattern:
# Michigan / regular comprehension = plain fill, LexTALE / reread = "///" hatch.
_TARGET_HATCH = {
    "michtest_score": "",
    "lextale_score": "///",
    "comprehension_score-regular_trials": "",
    "comprehension_score-repeated_reading": "///",
}
_TARGET_HATCH_DEFAULT = ""

# Line-style convention used for horizontal baseline reference lines when the
# same target appears as a horizontal line rather than a bar (in by_feature_set).
_TARGET_BASELINE_LINESTYLE = {
    "michtest_score": "-",
    "lextale_score": "--",
    "comprehension_score-regular_trials": "-",
    "comprehension_score-repeated_reading": "--",
}
_TARGET_BASELINE_LINESTYLE_DEFAULT = "-"


def _build_title(dataset: str, preview: str, content_mode: str, version_path: str) -> str:
    line1 = (
        f"Dataset: {dataset}  |  Condition: {_display_name(preview, PREVIEW_DISPLAY_NAMES)}"
        f"  |  Compared to: {_display_name(content_mode, CONTENT_MODE_DISPLAY_NAMES)}"
    )
    line2 = f"Version: {version_path}"
    return f"{line1}\n{line2}"


def plot_bar_by_feature_set(eval_df: pd.DataFrame, title: str, save_path: Path,
                            metric: str = "pearson_r"):
    """Vertical grouped bar chart: feature sets on x-axis, one bar per target.

    Ordering is canonical (not data-driven) so plots are comparable across outputs:
    - Feature sets (x-axis): most features on the left, fewer each step, with
      Reading Speed (baseline) pinned on the far right.
    - Targets (within-group bar order / legend): Michigan first, LexTALE last
      (when both present).

    Bars are colored by **feature set** using the same color mapping as
    `plot_bar_by_target`, so the same feature set is the same color in both
    plots. Targets within a group are distinguished only by bar-edge style —
    Michigan gets a solid edge, LexTALE gets a dashed edge (and comprehension
    targets inherit the same regular/reread = solid/dashed convention).
    The Reading Speed x-tick label is rendered in the baseline color.
    """
    if metric not in eval_df.columns:
        logger.warning("Metric %s not in eval_df", metric)
        return

    metric_title = METRIC_TITLES.get(metric, metric)

    df = eval_df.copy()

    # Pivot on raw names so we can apply canonical ordering before relabeling.
    pivot = df.pivot_table(index="feature_set", columns="target_col", values=metric, aggfunc="first")

    # Feature-set (x-axis) ordering: canonical first, then extras preserved.
    fs_order = [fs for fs in FEATURE_SET_ORDER if fs in pivot.index]
    fs_order += [fs for fs in pivot.index if fs not in fs_order]
    pivot = pivot.loc[fs_order]

    # Target (within-group) ordering: Michigan → LexTALE, extras after.
    target_order = [t for t in TARGET_COL_ORDER if t in pivot.columns]
    target_order += [t for t in pivot.columns if t not in target_order]
    pivot = pivot[target_order]

    # Remove baseline from bars (keep only the line)
    baseline_series = pivot.loc[BASELINE_FEATURE_SET] if BASELINE_FEATURE_SET in pivot.index else None
    pivot_no_base = pivot.drop(BASELINE_FEATURE_SET, errors='ignore')

    n_groups = len(pivot_no_base)
    n_targets = len(pivot_no_base.columns)
    bar_width = 0.35 / n_targets  # actual bar width
    bar_spacing = 0.8 / n_targets  # center-to-center spacing (creates gaps between bars)
    x = np.arange(n_groups)

    # Same color mapping as `plot_bar_by_target`, keyed by feature set.
    color_map = _build_feature_set_color_map(pivot_no_base.index)
    bar_colors = [color_map[fs] for fs in pivot_no_base.index]

    # Optional Fisher-z CI pivots — each bar gets an asymmetric error bar if
    # evaluation included the CI columns in the CSV.
    ci_lo_pivot, ci_hi_pivot = _ci_pivots(df, metric, "feature_set", "target_col", pivot)
    if ci_lo_pivot is not None:
        ci_lo_pivot = ci_lo_pivot.drop(BASELINE_FEATURE_SET, errors='ignore')
    if ci_hi_pivot is not None:
        ci_hi_pivot = ci_hi_pivot.drop(BASELINE_FEATURE_SET, errors='ignore')

    # Per-feature-set (x-group) maximum — used to bold the winning value label.
    # NaN-safe: rows with all-NaN produce no max.
    max_target_per_group = {}
    for fs in pivot_no_base.index:
        row = pivot_no_base.loc[fs]
        if row.notna().any():
            max_target_per_group[fs] = row.idxmax()

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_groups), 6.5))
    for i, col in enumerate(pivot_no_base.columns):
        offset = (i - n_targets / 2 + 0.5) * bar_spacing
        hatch = _TARGET_HATCH.get(col, _TARGET_HATCH_DEFAULT)
        # Bar fill color is per-feature-set (x-position); hatch pattern encodes the target.
        bars = ax.bar(x + offset, pivot_no_base[col].values, bar_width,
                      color=bar_colors, edgecolor='#333', linewidth=0.7,
                      hatch=hatch)
        # 95% Fisher-z CIs as asymmetric error bars at bar centers.
        if ci_lo_pivot is not None:
            yerr = _asymmetric_yerr(pivot_no_base[col].values,
                                    ci_lo_pivot[col].values,
                                    ci_hi_pivot[col].values)
            ax.errorbar(x + offset, pivot_no_base[col].values, yerr=yerr,
                        fmt='none', ecolor='#333', elinewidth=0.8,
                        capsize=2, capthick=0.8, zorder=4)
        for bar_idx, bar in enumerate(bars):
            h = bar.get_height()
            if not np.isfinite(h):
                continue
            fs = pivot_no_base.index[bar_idx]
            is_max = max_target_per_group.get(fs) == col
            # Rotate 90° and offset horizontally so the text sits beside, not
            # behind, the vertical CI whisker (which is centered on the bar).
            text_x = bar.get_x() + bar.get_width() * 0.25
            ax.text(text_x, h+0.02, f"{h:.2f}",
                    rotation=90, rotation_mode='anchor',
                    ha='left', va='center', fontsize=8,
                    color='#000' if is_max else '#222',
                    fontweight='bold' if is_max else 'normal')

    # Horizontal baseline reference line per target (analogous to the one in
    # plot_bar_by_target). Each target has a single baseline r value; we draw a
    # horizontal line at that value across the full x-range, using the target's
    # linestyle convention (Michigan = solid, LexTALE = dashed).
    if baseline_series is not None:
        for col, val in baseline_series.items():
            if not np.isfinite(val):
                continue
            ls = _TARGET_BASELINE_LINESTYLE.get(col, _TARGET_BASELINE_LINESTYLE_DEFAULT)
            ax.axhline(val, color=BASELINE_COLOR, linestyle=ls,
                       linewidth=1.1, alpha=0.65, zorder=3)

    # Set x-axis ticks and labels (no baseline coloring since it's not plotted as a bar)
    ax.set_xticks(x)
    xlabels = [_display_name(fs, FEATURE_SET_DISPLAY_NAMES) for fs in pivot_no_base.index]
    ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=9)

    ax.set_ylabel(metric_title, fontsize=11, fontweight='bold')
    ax.set_title(f"{title}\n{metric_title} by Feature Set", fontsize=13, weight="bold")
    ax.set_ylim(*BAR_BY_TARGET_YLIM)

    # Style-only legend: one proxy per target (neutral gray swatch + the target's
    # hatch), plus one entry per target for the horizontal baseline reference line.
    legend_handles = []
    for col in pivot.columns:
        legend_handles.append(
            Patch(facecolor='lightgray', edgecolor='#333', linewidth=0.7,
                  hatch=_TARGET_HATCH.get(col, _TARGET_HATCH_DEFAULT),
                  label=_display_name(col, TARGET_COL_DISPLAY_NAMES))
        )
    for col in pivot.columns:
        ls = _TARGET_BASELINE_LINESTYLE.get(col, _TARGET_BASELINE_LINESTYLE_DEFAULT)
        legend_handles.append(
            plt.Line2D([0], [0], color=BASELINE_COLOR, linestyle=ls, linewidth=1.1,
                       alpha=0.6,
                       label=f"Baseline ({_display_name(col, TARGET_COL_DISPLAY_NAMES)})")
        )
    ax.legend(handles=legend_handles, title="Target", fontsize=8, title_fontsize=9,
              loc="upper right", framealpha=0.9, borderpad=0.5, labelspacing=0.4, edgecolor='#333')

    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=False)

    # Footer: document the bold convention and the meaning of the dashed line(s).
    footer = ("Bold value = best target per feature set. "
              "Dashed/solid horizontal lines = Reading Speed baseline per target.")
    if ci_lo_pivot is not None:
        footer += " Error bars = 95% Fisher-z CI."
    fig.text(0.5, 0.01, footer, ha='center', va='bottom',
             fontsize=8, color='#666', style='italic')

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_bar_by_target(eval_df: pd.DataFrame, title: str, save_path: Path,
                       metric: str = "pearson_r"):
    """Vertical grouped bar chart: targets on x-axis, one bar per feature set.

    Ordering is canonical (not data-driven) so plots are comparable across outputs:
    - Targets: Michigan on the left, LexTALE on the right (when both present).
    - Feature sets (left→right within each target): most features first, fewer
      each step, with Reading Speed (baseline) pinned on the far right.
    Reading Speed is drawn in a distinct color to flag it as the baseline, the
    baseline's value is also drawn as a horizontal dashed line across each
    group for quick visual comparison, and non-baseline bars are annotated
    with APA-style significance stars from the paired-bootstrap test vs.
    baseline (starred only when the bar is above the baseline).
    The y-axis uses fixed limits so scale is consistent across outputs.
    """
    if metric not in eval_df.columns:
        logger.warning("Metric %s not in eval_df", metric)
        return

    metric_title = METRIC_TITLES.get(metric, metric)

    df = eval_df.copy()

    # Pivot on raw names so we can apply canonical ordering before relabeling.
    pivot = df.pivot_table(index="target_col", columns="feature_set", values=metric, aggfunc="first")

    # Target (x-axis) ordering: canonical first, then any extras preserved in their original order.
    target_order = [t for t in TARGET_COL_ORDER if t in pivot.index]
    target_order += [t for t in pivot.index if t not in target_order]
    pivot = pivot.loc[target_order]

    # Feature-set (within-group) ordering: canonical first, then any extras.
    fs_order = [fs for fs in FEATURE_SET_ORDER if fs in pivot.columns]
    fs_order += [fs for fs in pivot.columns if fs not in fs_order]
    pivot = pivot[fs_order]

    # Optional pivot of baseline-difference p-values (for significance stars).
    p_col = _BASELINE_DIFF_P_COLUMN.get(metric)
    p_pivot = None
    if p_col and p_col in df.columns:
        p_pivot = df.pivot_table(index="target_col", columns="feature_set",
                                 values=p_col, aggfunc="first")
        p_pivot = p_pivot.reindex(index=pivot.index, columns=pivot.columns)

    # Apply seaborn style
    sns.set_style("whitegrid")

    # Remove baseline from plotting (keep only line)
    pivot_no_base = pivot[[c for c in pivot.columns if c != BASELINE_FEATURE_SET]]
    baseline_series = pivot[BASELINE_FEATURE_SET] if BASELINE_FEATURE_SET in pivot.columns else None

    n_groups = len(pivot)

    # Divide into two groups by membership in SINGLE_FEATURE_SETS so that
    # adding new singles (e.g. TRANSITIONS, WFC) doesn't push them into the
    # combination palette.
    single_fs = [c for c in pivot_no_base.columns if c in SINGLE_FEATURE_SETS]
    combo_fs = [c for c in pivot_no_base.columns if c not in SINGLE_FEATURE_SETS]
    n_combo = len(combo_fs)
    n_single = len(single_fs)

    bar_width = 0.2 / max(n_combo, n_single)  # smaller bars
    bar_spacing_within = bar_width
    group_gap = 0.25  # gap between singles and combos groups
    x = np.arange(n_groups) * 0.6  # reduced spacing between Michigan and LexTALE groups

    # Single-feature palette extended to cover TRANSITIONS / WFC.
    single_colors = ['#FF69B4', '#4169E1', '#9370DB', '#0FA3B1', '#F7B538']
    combo_colors = ['#FFD700', '#FF8C00', '#FF4444', '#A6324B']
    color_dict = {}

    for i, fs in enumerate(single_fs):
        color_dict[fs] = single_colors[i] if i < len(single_colors) else single_colors[-1]
    for i, fs in enumerate(combo_fs):
        color_dict[fs] = combo_colors[i] if i < len(combo_colors) else combo_colors[-1]


    ci_lo_pivot, ci_hi_pivot = _ci_pivots(df, metric, "target_col", "feature_set", pivot)

    # Per-target (x-group) max feature set
    max_feature_per_group = {}
    for t in pivot_no_base.index:
        row = pivot_no_base.loc[t]
        if row.notna().any():
            max_feature_per_group[t] = row.idxmax()

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * n_groups), 7))
    # Iterate in visual order: combos first (right), then singles (left) - reversed for right-to-left ordering
    feature_order = single_fs + combo_fs
    for i, col in enumerate(feature_order):
        # Calculate offset (combos on left, singles on right - most to least from right)
        if col in single_fs:
            # Single group: centered in right part
            single_idx = single_fs.index(col)
            offset = (single_idx - n_single / 2 + 0.5) * bar_spacing_within - group_gap / 2
        else:
            # Combo group: centered in left part
            combo_idx = combo_fs.index(col)
            offset = (combo_idx - n_combo / 2 + 0.5) * bar_spacing_within + group_gap / 2

        label = _display_name(col, FEATURE_SET_DISPLAY_NAMES)
        bars = ax.bar(x + offset, pivot_no_base[col].values, bar_width, label=label,
                      color=color_dict[col], edgecolor='#333', linewidth=0.7)

        # Error bars
        if ci_lo_pivot is not None:
            yerr = _asymmetric_yerr(pivot_no_base[col].values,
                                    ci_lo_pivot[col].values,
                                    ci_hi_pivot[col].values)
            ax.errorbar(x + offset, pivot_no_base[col].values, yerr=yerr,
                        fmt='none', ecolor='#333', elinewidth=0.8,
                        capsize=2, capthick=0.8, zorder=4)

        # Values above bars and CIs (no bold for max)
        for bar_idx, bar in enumerate(bars):
            h = bar.get_height()
            if not np.isfinite(h):
                continue

            stars = ""
            if p_pivot is not None and col != BASELINE_FEATURE_SET:
                p_val = p_pivot.iat[bar_idx, pivot.columns.get_loc(col)]
                # The baseline-difference p is two-sided, so a feature set that
                # is significantly *worse* than the baseline carries the same p
                # as one that is significantly better. A star on a bar reads as
                # "beats the baseline", so only star bars above the baseline —
                # matching the rule the bootstrap tables apply.
                baseline_h = (baseline_series.iloc[bar_idx]
                              if baseline_series is not None else np.nan)
                if not np.isfinite(baseline_h) or h > baseline_h:
                    stars = _sig_stars(p_val)

            text = f"{h:.2f}{stars}" if stars else f"{h:.2f}"

            # Place text above CI whiskers
            text_y = h + 0.03
            if ci_hi_pivot is not None:
                target = pivot_no_base.index[bar_idx]
                ci_hi = ci_hi_pivot.loc[target, col]
                if np.isfinite(ci_hi):
                    text_y = max(h, ci_hi) + 0.03

            ax.text(bar.get_x() + bar.get_width() / 2, text_y, text,
                   ha='center', va='bottom', fontsize=8, color='#333')

    # Baseline reference lines only (no bars) - shortened with label
    if baseline_series is not None:
        added_label = False
        for i, val in enumerate(baseline_series.values):
            if np.isfinite(val):
                ax.hlines(val, xmin=x[i] - 0.25, xmax=x[i] + 0.25,
                         color=BASELINE_COLOR, linestyle='--', linewidth=1.1,
                         zorder=3, alpha=0.65, label='Reading Speed' if not added_label else '')
                added_label = True

    ax.set_xticks(x)
    xlabels = [_display_name(t, TARGET_COL_DISPLAY_NAMES) for t in pivot.index]
    ax.set_xticklabels(xlabels, ha='center', fontsize=10)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(metric_title, fontsize=11, fontweight='bold')
    ax.set_title(f"{title}\n{metric_title} by Target", fontsize=13, weight="bold")
    ax.set_ylim(*BAR_BY_TARGET_YLIM)

    # Custom legend: Reading Speed alone in first column (centered), singles in second, combos in third
    handles, labels = ax.get_legend_handles_labels()
    handle_map = dict(zip(labels, handles))

    baseline_label = "Reading Speed"
    single_labels = [_display_name(fs, FEATURE_SET_DISPLAY_NAMES) for fs in single_fs]
    combo_labels = [_display_name(fs, FEATURE_SET_DISPLAY_NAMES) for fs in combo_fs]

    # Filter to labels that actually have a registered handle. The baseline can
    # be missing entirely for variants where READING_SPEED has no eval data
    # (e.g. a corrected variant whose CSV hasn't been produced yet); single /
    # combo entries can be missing if a feature_set was filtered out upstream.
    present_single = [l for l in single_labels if l in handle_map]
    present_combo = [l for l in combo_labels if l in handle_map]

    # Put legends in figure coordinates so they sit next to each other
    if baseline_label in handle_map:
        fig.legend(
            [handle_map[baseline_label]],
            [baseline_label],
            fontsize=8,
            loc="lower center",
            bbox_to_anchor=(0.22, 0.01),
            frameon=False,
            ncol=1,
            handletextpad=0.5
        )

    if present_single:
        fig.legend(
            [handle_map[l] for l in present_single],
            present_single,
            fontsize=8,
            loc="lower center",
            bbox_to_anchor=(0.50, 0.01),
            frameon=False,
            ncol=1,
            handletextpad=0.5,
            labelspacing=0.4
        )

    if present_combo:
        fig.legend(
            [handle_map[l] for l in present_combo],
            present_combo,
            fontsize=8,
            loc="lower center",
            bbox_to_anchor=(0.78, 0.01),
            frameon=False,
            ncol=1,
            handletextpad=0.5,
            labelspacing=0.4
    )
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    # Footer
    parts = []
    if p_pivot is not None:
        parts.append("Stars = paired-bootstrap test vs. Reading Speed baseline, shown only where higher: * p<.05, ** p<.01, *** p<.001.")
    if ci_lo_pivot is not None:
        parts.append("Error bars = 95% Fisher-z CI.")
    caption = " ".join(parts)
    fig.text(0.5, -0.01, caption, ha='center', va='bottom', fontsize=8,
             color='#666', style='italic')

    plt.tight_layout(rect=(0, 0.10, 1, 1))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_eyescore_vs_target(eye_score_csv: Path, metadata_df: pd.DataFrame,
                                     feature_set: str, target_col: str,
                                     title: str, save_path: Path,
                                     show_points: bool = True):
    """Scatter plot of eye_score vs target_col, colored by L1 language with trend line.

    Args:
        show_points: If True, draw scatter points and trend lines. If False, draw only trend lines.
    """
    try:
        eye_df = pd.read_csv(eye_score_csv)
    except Exception as e:
        logger.warning("Failed to read %s: %s", eye_score_csv, e)
        return

    if eye_df.empty or "eye_score" not in eye_df.columns:
        return

    merged = eye_df.merge(metadata_df, on=Fields.SUBJECT_ID, how="inner", suffixes=("", "_meta"))
    if target_col not in merged.columns:
        return

    merged = merged.dropna(subset=["eye_score", target_col])
    merged = merged[merged[target_col] != -1]
    if len(merged) < 3:
        return

    l1_col = "L1" if "L1" in merged.columns else ("L1_meta" if "L1_meta" in merged.columns else None)
    fs_display = _display_name(feature_set, FEATURE_SET_DISPLAY_NAMES)
    target_display = _display_name(target_col, TARGET_COL_DISPLAY_NAMES)

    fig, ax = plt.subplots(figsize=(8, 6))

    line_alpha = 0.8 if show_points else 0.95
    line_width = 1.5 if show_points else 2.25

    x_min = merged[target_col].min()
    x_max = merged[target_col].max()
    x_line_full = np.linspace(x_min, x_max, 100)

    if l1_col and merged[l1_col].nunique() > 1:
        languages = sorted(merged[l1_col].unique())
        palette = sns.color_palette(SCATTER_PALETTE, n_colors=len(languages))
        color_map = dict(zip(languages, palette))
        for lang in languages:
            lang_data = merged[merged[l1_col] == lang]
            if show_points:
                ax.scatter(lang_data[target_col], lang_data["eye_score"],
                           color=color_map[lang], label=lang, alpha=0.7,
                           edgecolors='white', linewidth=0.5, s=50)
            # Per-language trend line extended across full x-range
            if len(lang_data) >= 2:
                z = np.polyfit(lang_data[target_col], lang_data["eye_score"], 1)
                p = np.poly1d(z)
                line_kwargs = {} if show_points else {"label": lang}
                ax.plot(x_line_full, p(x_line_full), "-", color=color_map[lang],
                        alpha=line_alpha, linewidth=line_width, **line_kwargs)
        ax.legend(title="L1", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        if show_points:
            ax.scatter(merged[target_col], merged["eye_score"], alpha=0.7,
                       color=sns.color_palette(BAR_PALETTE)[0],
                       edgecolors='white', linewidth=0.5, s=50)
        # Single trend line when no L1 grouping
        z = np.polyfit(merged[target_col], merged["eye_score"], 1)
        p = np.poly1d(z)
        ax.plot(x_line_full, p(x_line_full), "-", color="#555",
                alpha=line_alpha, linewidth=line_width)

    # Use fixed axis limits for consistent comparison across plots
    # Y-axis range depends on feature set: Reading Speed has wider range than other normalized features
    scatter_ylim = SCATTER_YLIM_READING_SPEED if feature_set == BASELINE_FEATURE_SET else SCATTER_YLIM_OTHER
    ax.set_xlim(*SCATTER_XLIM)
    ax.set_ylim(*scatter_ylim)

    ax.set_xlabel(target_display, fontsize=11)
    ax.set_ylabel(f"Eye Score ({fs_display})", fontsize=11)
    ax.set_title(f"{title}\n{fs_display} vs {target_display}", fontsize=12, weight="bold")
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_results(eval_dir: Path = None, plots_dir: Path = None, results_dir: Path = None,
                     target_cols: list = None, feature_sets: list = None,
                     plot_target_cols: list = None,
                     comprehension_target_cols: list = None,
                     variant: str = RAW_VARIANT,
                     no_titles: bool = False):
    """Generate plots for all evaluated EyeScore results.

    Args:
        eval_dir: Directory with evaluation CSVs.
        plots_dir: Directory to save plots.
        results_dir: Directory with raw eye_score CSVs (for scatter plots).
        target_cols: Filter evaluation to these target columns.
        feature_sets: Filter evaluation to these feature sets.
        plot_target_cols: Proficiency target columns used for the main bar/scatter
            plots (defaults to PLOT_TARGET_COLS — Michigan + LexTALE).
        comprehension_target_cols: Target columns used for the separate
            comprehension bar plots (defaults to PLOT_COMPREHENSION_COLS).
        no_titles: If True, suppress axes titles and figure suptitles in
            every produced figure (paper-ready output).
    """
    if no_titles:
        with suppress_titles():
            return plot_all_results(
                eval_dir=eval_dir, plots_dir=plots_dir, results_dir=results_dir,
                target_cols=target_cols, feature_sets=feature_sets,
                plot_target_cols=plot_target_cols,
                comprehension_target_cols=comprehension_target_cols,
                variant=variant, no_titles=False,
            )

    # Reset matplotlib state to avoid stale definitions from previous runs
    plt.close('all')

    if eval_dir is None:
        eval_dir = EVAL_SAVE_DIR
    if plots_dir is None:
        plots_dir = PLOTS_SAVE_DIR
    if results_dir is None:
        results_dir = RESULTS_DIR
    if plot_target_cols is None:
        plot_target_cols = PLOT_TARGET_COLS
    if comprehension_target_cols is None:
        comprehension_target_cols = PLOT_COMPREHENSION_COLS

    combined_csv = eval_dir / "all_evaluation_results.csv"
    if not combined_csv.exists():
        logger.warning("No combined evaluation CSV found at %s", combined_csv)
        return

    all_df = pd.read_csv(combined_csv)
    if target_cols is not None:
        all_df = all_df[all_df["target_col"].isin(target_cols)]
    if feature_sets is not None:
        all_df = all_df[all_df["feature_set"].isin(feature_sets)]

    # Bar plots are produced for two separate target groups, each with its own
    # file-name suffix so they never overwrite each other.
    # (suffix, target list) pairs — suffix "" is the proficiency-test plots.
    bar_plot_target_groups = [
        ("", plot_target_cols),
        ("comprehension_", comprehension_target_cols),
    ]

    group_cols = ["dataset", "preview", "content_mode", "version_path"]
    groups = list(all_df.groupby(group_cols))

    # Cache metadata
    metadata_cache = {}

    for group_key, group_df in tqdm(groups, desc="Generating EyeScore plots"):
        dataset, preview, content_mode, version_path = group_key
        title = _build_title(dataset, preview, content_mode, version_path)
        group_plots_dir = plots_dir / dataset / _simplified_plot_tail(
            dataset, preview, content_mode, version_path,
        )

        # Bar plots — one pass per target group (proficiency, comprehension, ...).
        for suffix, group_targets in bar_plot_target_groups:
            plot_group_df = all_df[
                (all_df["dataset"] == dataset) &
                (all_df["preview"] == preview) &
                (all_df["content_mode"] == content_mode) &
                (all_df["version_path"] == version_path) &
                (all_df["target_col"].isin(group_targets))
            ]
            if plot_group_df.empty:
                continue
            for metric in CORRELATION_METRICS:
                metric_name = metric.replace("_", "")
                plot_bar_by_feature_set(
                    plot_group_df, title=title,
                    save_path=group_plots_dir / f"bar_by_feature_set_{suffix}{metric_name}.png",
                    metric=metric,
                )
                plot_bar_by_target(
                    plot_group_df, title=title,
                    save_path=group_plots_dir / f"bar_by_target_{suffix}{metric_name}.png",
                    metric=metric,
                )

        # Scatter plots: one per (feature_set, target_col) pair
        cache_key = f"{dataset}/{version_path}"
        if cache_key not in metadata_cache:
            data_dataset = f"{dataset}L2"
            meta_path = DATA_PATH / data_dataset / "features_and_targets" / version_path / "features_and_metadata.csv"
            if meta_path.exists():
                meta = pd.read_csv(meta_path).drop_duplicates(Fields.SUBJECT_ID)
                metadata_cache[cache_key] = meta
            else:
                metadata_cache[cache_key] = pd.DataFrame()

        meta = metadata_cache[cache_key]
        if meta.empty:
            continue

        scatter_feature_sets = group_df["feature_set"].unique()
        for fs in scatter_feature_sets:
            eye_csv = eye_score_csv_path(
                results_dir, dataset, preview, content_mode, version_path, fs, variant,
            )
            if not eye_csv.exists():
                continue
            for tc in plot_target_cols:
                scatter_save = group_plots_dir / "scatter" / fs / f"{tc}.png"
                trend_save = group_plots_dir / "scatter_trend_only" / fs / f"{tc}.png"
                plot_scatter_eyescore_vs_target(
                    eye_csv, meta, fs, tc, title, scatter_save, show_points=True,
                )
                plot_scatter_eyescore_vs_target(
                    eye_csv, meta, fs, tc, title, trend_save, show_points=False,
                )

    logger.info("All EyeScore plots saved under %s", plots_dir)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-titles", action="store_true",
        help="Suppress axes titles and figure suptitles in every plot "
             "(paper-ready output).",
    )
    args = parser.parse_args()

    feature_sets = list(RELEVANT_FEATURE_GROUPS_COMBINATIONS_SMALL.keys())
    # Include comprehension targets so the separate comprehension bar plots have data.
    plot_target_cols = list(ALL_TESTS) + list(COMPREHENSION_COLS)

    # Pristine: org-only plots under PLOTS_SAVE_DIR/.
    plot_all_results(
        eval_dir=EVAL_SAVE_DIR, plots_dir=PLOTS_SAVE_DIR,
        target_cols=plot_target_cols, feature_sets=feature_sets, variant=RAW_VARIANT,
        no_titles=args.no_titles,
    )

