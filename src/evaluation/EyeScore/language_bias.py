"""
Language-bias analysis for EyeScore.

Tests whether EyeScore is confounded by typological distance between a
participant's L1 and English, beyond what proficiency alone explains.

Two analyses:
1. Scatter plot of proficiency vs EyeScore with regression line, colored by L1.
   Languages whose points cluster above the line may benefit from typological
   similarity to English.
2. Per-participant regression residual correlated with the typological
   distance from each participant's L1 to English (via URIEL+). A significant
   correlation indicates that EyeScore rewards L1-to-English similarity
   independently of proficiency.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from scipy.stats import linregress, pearsonr, spearmanr
from tqdm import tqdm
import logging

from src.evaluation.utils import suppress_titles
from src.methods.EyeScore.typology import (
    compute_typological_distances,
)

from src.evaluation.EyeScore.label_placement import place_language_labels

from src.evaluation.EyeScore.evaluation import (
    FEATURE_SET_DISPLAY,
    RESULTS_DIR,
    EVAL_SAVE_DIR,
    preview_dir_segments,
    load_metadata,
    collect_eyescore_csvs,
    eye_score_csv_path,
)
from src.methods.EyeScore.calculation import (
    RAW_VARIANT,
    SPLIT_SCHEME_TO_VARIANT,
)
from src.constants import (
    RELEVANT_FEATURE_GROUPS_COMBINATIONS_SMALL,
    TestCols,
    Fields,
    FeatureGroups,
)
from src.evaluation.EyeScore.plot import (
    SCATTER_PALETTE,
    FEATURE_SET_DISPLAY_NAMES,
    TARGET_COL_DISPLAY_NAMES,
    PREVIEW_DISPLAY_NAMES,
    CONTENT_MODE_DISPLAY_NAMES,
    _display_name,
)

logger = logging.getLogger(__name__)

PLOTS_ROOT = Path("src/evaluation/EyeScore/plots")
LANG_BIAS_PLOTS_DIR = PLOTS_ROOT / "lang_bias"




# Fixed axis limits (derived from p2/p98 of observed values across all results
# CSVs, with small padding) so language-bias plots are visually comparable
# across (dataset, preview, content_mode, feature_set, target_col). Rare
# outliers may clip; re-derive and widen if a new feature set or distance
# type pushes the body of the distribution outside these windows.
PROFICIENCY_XLIM = (0, 100)
EYESCORE_YLIM_READING_SPEED = (-2, 3)
EYESCORE_YLIM_OTHER = (-1.05, 1.05)
EYESCORE_YLIM_ADJUSTED = (-1, 1)

# Distance xlim is dataset-specific because OneStop's L1 set sits in a narrower
# (more-distant) window than Meco's (which includes Germanic languages that are
# close to English).
DISTANCE_XLIM_BY_DATASET = {
    "Meco": (0.45, 0.9),
    "OneStop": (0.65, 0.85),
}
DISTANCE_XLIM_DEFAULT = (0.4, 0.9)

# Per-language labels on the paper figures. Positions are solved automatically
# by label_placement; these only fix the type itself.
# How an L1 is NAMED on a figure. The data spells MECO's Portuguese cohort out
# as "Brazilian Portuguese", which is twice the length of any neighbouring name:
# the labels are rotated, so a long one sweeps across the clusters either side of
# it and pushes their labels into far-away lanes. Nothing on these figures
# distinguishes the two Portugueses, so the variety is dropped and the layout
# solves itself. Names not listed here are drawn as they appear in the data.
LANGUAGE_LABEL_DISPLAY = {
    "Brazilian Portuguese": "Portuguese",
}


def _lang_label(name: str) -> str:
    return LANGUAGE_LABEL_DISPLAY.get(name, name)


LANG_LABEL_ROTATION = 30
LANG_LABEL_PAD_Y = 0.04  # data units of clearance above the cluster
# Per-participant residual y-limits, bucketed by feature_set. Numbers are
# padded above the p98 (and below p2) of observed residuals across the full
# results tree (raw + both typo_calibrated variants combined; per-participant
# spread is similar across variants, so we don't split on variant here).
# READING_SPEED has a much wider distribution than the others, mirroring the
# wide-window logic in _eyescore_ylim.
RESIDUAL_YLIM_READING_SPEED = (-2.3, 2.3)
RESIDUAL_YLIM_TRANSITIONS = (-0.2, 0.2)
RESIDUAL_YLIM_DEFAULT = (-1.6, 1.6)
RESIDUAL_YLIM_BY_FEATURE_SET = {
    "READING_SPEED": RESIDUAL_YLIM_READING_SPEED,
    "TRANSITIONS": RESIDUAL_YLIM_TRANSITIONS,
}


def _eyescore_ylim(feature_set: str, variant: str) -> tuple:
    """Pick the eye_score y-range. Raw RS keeps its wide window; everything
    else falls into raw or adjusted buckets based on the variant."""
    if feature_set == FeatureGroups.READING_SPEED and variant == RAW_VARIANT:
        return EYESCORE_YLIM_READING_SPEED
    return EYESCORE_YLIM_OTHER if variant == RAW_VARIANT else EYESCORE_YLIM_ADJUSTED


def _residual_ylim(feature_set: str) -> tuple:
    return RESIDUAL_YLIM_BY_FEATURE_SET.get(feature_set, RESIDUAL_YLIM_DEFAULT)


# Below this, "%.3f" renders as "0.000" and the annotation reads as p == 0.
P_VALUE_DECIMAL_FLOOR = 5e-4
# Fallback exponent for a p-value that arrives as exactly 0.0: scipy's analytic
# tail underflows there, and a literal reading of that zero ("p < 1e-308") would
# claim far more resolution than the test has. Genuine nonzero p-values are
# reported at their own exponent, however small.
P_VALUE_UNDERFLOW_EXPONENT = 16


def _format_beta(b: float) -> str:
    """Format the fitted slope for a figure label.

    Two decimals while they carry information, and a one-significant-figure
    power-of-ten approximation ("$\\approx-3\\times10^{-4}$") once they do
    not, so a slope on a small-scale y variable stays readable instead of
    collapsing to "0.00" or "-0.00". Mirrors `_format_p`'s treatment of
    vanishing p-values.
    """
    if not np.isfinite(b):
        return f"={b}"
    if abs(b) >= BETA_DECIMAL_FLOOR:
        return f"={b:.2f}"
    if b == 0:
        return "=0"
    exponent = int(np.floor(np.log10(abs(b))))
    mantissa = b / 10.0 ** exponent
    # Rounding the mantissa can push it to 10, which is not one significant
    # figure; carry into the exponent instead of printing "10e-4".
    if abs(round(mantissa)) >= 10:
        mantissa /= 10.0
        exponent += 1
    return rf"$\approx{mantissa:.0f}\times10^{{{exponent}}}$"


def _pearson_annotation(r: float, p_val: float,
                        b_dist: Optional[float] = None) -> str:
    """The "beta=..., Pearson r=..., p=..." string shared by every figure.

    `b_dist` is the slope of the same regression the r comes from (y on
    distance), so it shares r's p-value -- it is reported first because it is
    the quantity the bias tables key on, with r following as the standardized
    version of the same fit. Omitted when not supplied, which keeps the
    original two-term string for any caller that has no slope to show.
    """
    rp = f"Pearson $r$={r:.2f}, $p${_format_p(p_val)}"
    if b_dist is None:
        return rp
    return rf"$\hat{{\beta}}_{{\mathrm{{dist}}}}${_format_beta(b_dist)}, " + rp


# Below this the two decimals a slope is shown with stop carrying information
# ("0.00"), and _format_beta switches to a power-of-ten approximation.
BETA_DECIMAL_FLOOR = 0.005


def _format_p(p_val: float, mathtext: bool = True) -> str:
    """Format a p-value as the operator-and-value half of a "p…" annotation.

    Returns "=0.032" while three decimals still carry information, and the
    tightest true power-of-ten bound ("<$10^{-6}$") once they don't, so a
    vanishing p-value stays readable instead of collapsing to "=0.000".
    `mathtext=False` gives a plain-text bound for titles and log lines.
    """
    if not np.isfinite(p_val):
        return f"={p_val}"
    if p_val >= P_VALUE_DECIMAL_FLOOR:
        return f"={p_val:.3f}"

    if p_val <= 0:
        exponent = P_VALUE_UNDERFLOW_EXPONENT
    else:
        exponent = int(np.floor(-np.log10(p_val)))
        # log10 of an exact power of ten can land on the boundary, which would
        # claim p < p. Step down one decade so the bound stays strict.
        if p_val >= 10.0 ** (-exponent):
            exponent -= 1

    return f"<$10^{{-{exponent}}}$" if mathtext else f"<1e-{exponent:02d}"


def _jitter_halfwidth(dataset: str) -> float:
    """Half-width of the horizontal jitter applied to one dataset's points.

    Participants sharing an L1 share an x, so they are spread by this much to
    keep them from stacking. It is also how far the drawn points reach beyond
    the language's own position, which is what an x window has to clear.
    """
    xlim = DISTANCE_XLIM_BY_DATASET.get(dataset, DISTANCE_XLIM_DEFAULT)
    return 0.005 * (xlim[1] - xlim[0])


# What the y axis calls the test that was regressed out. Same names the debias
# tables use ("Composite", "Michigan"), so a figure and the table beside it do
# not name one test two ways; TARGET_COL_DISPLAY_NAMES spells Michigan out in
# full, which is more than an axis label at 22pt can carry.
AXIS_TARGET_DISPLAY = {**TARGET_COL_DISPLAY_NAMES, "michtest_score": "Michigan"}


# One colour per language, used for all three things that carry it: the points,
# the mean diamond and the name. A name set in a darkened version of its colour
# was tried and dropped -- a language should be one colour in the figure, and a
# colour too pale to read as a name is a colour to change in LANGUAGE_COLORS
# rather than to patch at the point it is drawn.

# Participant points, fainter than the 0.55 they were drawn at, so the
# per-language mean diamonds sitting on top of them read as the foreground
# without the clouds losing their shape.
SCATTER_ALPHA = 0.4

# The OLS trend has to be read across a field of scattered points, and at
# #555 / 1.5px / 0.8 alpha it was competing with them rather than sitting on top:
# the dashes are thin enough that the point cloud shows through the gaps.
FIT_LINE_COLOR = "#333333"
FIT_LINE_ALPHA = 0.9
FIT_LINE_WIDTH = 2.0

# ── Language colours ───────────────────────────────────────────────────────
# Written out per language rather than assigned from a palette. The language
# sets are fixed, and so is their order on the x axis (typological distance), so
# the map can be read and corrected by eye: check each neighbouring pair, change
# one colour where two of them look alike, leave the rest alone. A palette
# assigned programmatically pairs neighbours at random and re-pairs them the
# moment a language is added.
#
# The two datasets are kept separate because their x order is separate; a
# language in both may differ between them. Keyed by the L1 as the DATA spells
# it ("Brazilian Portuguese"), not as the figure prints it.
LANGUAGE_COLORS = {
    "Meco": {
        "Danish": "#aec7e8",
        "Serbian": "#7f7f7f",
        "Dutch": "#ff7f0e",
        "Norwegian": "#e377c2",
        "German": "#98df8a",
        "Brazilian Portuguese": "#4d4d4d",
        "Icelandic": "#c5b0d5",
        "Spanish": "#01665e",
        "Italian": "#8c564b",
        "Estonian": "#ffbb78",
        "Greek": "#d62728",
        "Russian": "#f7b6d2",
        "Finnish": "#2ca02c",
        "Hindi": "#9467bd",
        "Basque": "#1f77b4",
        "Hebrew": "#ff9896",
        "Turkish": "#bcbd22",
        "Mandarin": "#4b0082",
    },
    "OneStop": {
        "Portuguese": "#d62728",
        "Spanish": "#9467bd",
        "French": "#ff7f0e",
        "Russian": "#01665e",
        "Vietnamese": "#c5b0d5",
        "Hebrew": "#ffbb78",
        "Korean": "#98df8a",
        "Arabic": "#1f77b4",
        "Japanese": "#2ca02c",
        "Chinese": "#aec7e8",
    },
}


def _language_color_map(languages, dataset=None):
    """Colour per L1: the hand-kept map, with the palette covering anything new.

    `dataset` picks which map; without one (the two-panel figure, which draws
    both corpora) the maps are merged. A language the map has never seen falls
    back to a palette colour, so a new L1 appears in a colour of its own rather
    than crashing -- and lands in LANGUAGE_COLORS the next time someone looks.
    """
    fixed = (LANGUAGE_COLORS.get(dataset) if dataset
             else {k: v for m in LANGUAGE_COLORS.values() for k, v in m.items()})
    fixed = fixed or {}
    missing = [l for l in languages if l not in fixed]
    if missing:
        logger.warning("No colour in LANGUAGE_COLORS for %s; using the palette",
                       ", ".join(missing))
    spare = sns.color_palette(SCATTER_PALETTE, n_colors=max(len(missing), 1))
    fallback = dict(zip(missing, spare))
    return {l: fixed.get(l, fallback.get(l)) for l in languages}


# ── Analysis helpers ─────────────────────────────────────────────────────────


def _merge_eyescore_metadata(
    eye_csv: Path, metadata_df: pd.DataFrame, target_col: str
) -> Optional[pd.DataFrame]:
    """Load an EyeScore CSV, merge with metadata, and clean."""
    try:
        eye_df = pd.read_csv(eye_csv)
    except Exception as e:
        logger.warning("Failed to read %s: %s", eye_csv, e)
        return None

    if eye_df.empty or "eye_score" not in eye_df.columns:
        return None

    merged = eye_df.merge(
        metadata_df, on=Fields.SUBJECT_ID, how="inner", suffixes=("", "_meta")
    )
    if target_col not in merged.columns:
        return None

    l1_col = "L1" if "L1" in merged.columns else ("L1_meta" if "L1_meta" in merged.columns else None)
    if l1_col is None:
        return None

    merged = merged.rename(columns={l1_col: "L1"})
    merged = merged.dropna(subset=["eye_score", target_col, "L1"])
    merged = merged[merged[target_col] != -1]

    if len(merged) < 3:
        return None
    return merged


# ── Plot 1: Proficiency vs EyeScore scatter with L1 coloring ────────────────


def plot_proficiency_vs_eyescore(
    merged: pd.DataFrame,
    target_col: str,
    feature_set: str,
    title_prefix: str,
    save_path: Optional[Path],
    variant: str = RAW_VARIANT,
):
    """Scatter: x = proficiency, y = EyeScore, color = L1, with regression line.

    Points above the line have higher EyeScore than expected from proficiency.
    If ``save_path`` is ``None``, the figure is not written to disk (used by
    the simplified bias pipeline for non-canonical target columns).
    """
    if save_path is None:
        return
    languages = sorted(merged["L1"].unique())
    palette = sns.color_palette(SCATTER_PALETTE, n_colors=len(languages))
    color_map = dict(zip(languages, palette))

    target_display = _display_name(target_col, TARGET_COL_DISPLAY_NAMES)
    fs_display = _display_name(feature_set, FEATURE_SET_DISPLAY_NAMES)

    fig, ax = plt.subplots(figsize=(9, 6))

    for lang in languages:
        lang_data = merged[merged["L1"] == lang]
        ax.scatter(
            lang_data[target_col],
            lang_data["eye_score"],
            color=color_map[lang],
            label=_lang_label(lang),
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            s=50,
        )

    # Regression line
    z = np.polyfit(merged[target_col], merged["eye_score"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged[target_col].min(), merged[target_col].max(), 100)
    ax.plot(x_line, p(x_line), "--", color=FIT_LINE_COLOR, alpha=FIT_LINE_ALPHA,
            linewidth=FIT_LINE_WIDTH,
            label=f"trend (slope={z[0]:.4f})")

    ax.set_xlim(*PROFICIENCY_XLIM)
    ax.set_ylim(*_eyescore_ylim(feature_set, variant))

    ax.set_xlabel(target_display, fontsize=11)
    ax.set_ylabel(f"EyeScore ({fs_display})", fontsize=11)
    ax.set_title(
        f"{title_prefix}\n{target_display} vs EyeScore ({fs_display})",
        fontsize=12,
        weight="bold",
    )
    ax.legend(title="L1", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 150 dpi is plenty for these per-(feature_set, target_col) scatters; the
    # paper split plot below stays at 300.
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Plot 2: Residual vs typological distance ────────────────────────────────


def plot_residual_vs_distance(
    merged: pd.DataFrame,
    target_col: str,
    feature_set: str,
    title_prefix: str,
    save_path: Optional[Path],
    distance_type: str = "syntactic",
    variant: str = RAW_VARIANT,
    dataset: str = None,
):
    """Per-participant residual (from proficiency~EyeScore line) vs typological
    distance to English (assigned via the participant's L1).

    Every participant contributes one point: x = distance(L1 → English), y =
    that participant's residual. Correlation is computed on participant-level
    data, so it reflects within-language spread rather than collapsing each
    language to a single mean.

    If ``save_path`` is ``None``, the correlation is still computed and
    returned but the figure is not rendered or saved — used by the simplified
    bias pipeline to fill the bias_correlation_summary.csv for non-canonical
    (target_col, distance_type) pairs without producing extra plots.
    """
    # Compute residuals from OLS fit
    z = np.polyfit(merged[target_col], merged["eye_score"], 1)
    p = np.poly1d(z)
    merged = merged.copy()
    merged["residual"] = merged["eye_score"] - p(merged[target_col])

    # Typological distance, broadcast to each participant via their L1
    distances = compute_typological_distances(
        merged["L1"].unique().tolist(), distance_type=distance_type
    )
    merged["distance"] = merged["L1"].map(distances)
    merged = merged.dropna(subset=["distance"])

    if len(merged) < 3 or merged["L1"].nunique() < 2:
        logger.warning("Too few participants/languages with distance data for correlation plot")
        return

    target_display = _display_name(target_col, TARGET_COL_DISPLAY_NAMES)
    fs_display = _display_name(feature_set, FEATURE_SET_DISPLAY_NAMES)

    # Correlation (participant-level) — always computed for the summary CSV.
    r, p_val = pearsonr(merged["distance"], merged["residual"])
    sr, sp_val = spearmanr(merged["distance"], merged["residual"])

    if save_path is None:
        logger.debug(
            "Residual vs distance compute-only (%s, %s, %s): r=%.3f p=%.3f",
            feature_set, target_col, distance_type, r, p_val,
        )
        return {"pearson_r": r, "pearson_p": p_val, "spearman_r": sr, "spearman_p": sp_val}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by L1 so within-language clusters are visible at each x.
    languages = sorted(merged["L1"].unique())
    palette = sns.color_palette(SCATTER_PALETTE, n_colors=len(languages))
    color_map = dict(zip(languages, palette))

    # Light horizontal jitter so participants sharing an L1 (and thus an x)
    # don't all stack on top of each other.
    xlim = DISTANCE_XLIM_BY_DATASET.get(dataset, DISTANCE_XLIM_DEFAULT)
    jitter_w = _jitter_halfwidth(dataset)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-jitter_w, jitter_w, size=len(merged))

    for lang in languages:
        mask = (merged["L1"] == lang).to_numpy()
        ax.scatter(
            merged["distance"].to_numpy()[mask] + jitter[mask],
            merged["residual"].to_numpy()[mask],
            color=color_map[lang],
            label=_lang_label(lang),
            s=28,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.4,
            zorder=3,
        )

    # Trend line (participant-level OLS)
    z_fit = np.polyfit(merged["distance"], merged["residual"], 1)
    p_fit = np.poly1d(z_fit)
    x_line = np.linspace(merged["distance"].min(), merged["distance"].max(), 100)
    ax.plot(x_line, p_fit(x_line), "--", color=FIT_LINE_COLOR, alpha=FIT_LINE_ALPHA,
            linewidth=FIT_LINE_WIDTH)

    ax.axhline(0, color="gray", linewidth=1.0, alpha=0.5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*_residual_ylim(feature_set))

    ax.set_xlabel("Typological Distance to English", fontsize=11)
    ax.set_ylabel("EyeScore Residual", fontsize=11)
    ax.set_title(
        f"{title_prefix}\nEyeScore Residual vs Typological Distance"
        f"\n({fs_display}, proficiency = {target_display})"
        f"\nPearson r={r:.3f} (p{_format_p(p_val, mathtext=False)})"
        f"  |  Spearman r={sr:.3f} (p{_format_p(sp_val, mathtext=False)})"
        f"  |  N={len(merged)} participants across {merged['L1'].nunique()} L1s",
        fontsize=11,
        weight="bold",
    )
    ax.legend(title="L1", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # See `plot_proficiency_vs_eyescore` — per-(feature_set, target_col)
    # scatters don't need print-grade dpi. Paper split plot remains 300.
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(
        "Residual vs distance (%s, %s): Pearson r=%.3f p=%.3f, Spearman r=%.3f p=%.3f (N=%d)",
        feature_set, target_col, r, p_val, sr, sp_val, len(merged),
    )
    return {"pearson_r": r, "pearson_p": p_val, "spearman_r": sr, "spearman_p": sp_val}


# ── Paper-figure split plot: Meco (top) + OneStop (bottom) ──────────────────

# Canonical (preview, content_mode, version_path) tuple per dataset for the
# headline language-bias plot. Mirrors BIAS_DATASET_FILTERS in
# evaluation/EyeScore/tables.py.
PAPER_SPLIT_DATASET_LAYOUT = {
    "Meco":    {"preview": "All",       "content_mode": "unseen", "version_path": "fully_agg/all/all"},
    # content_mode="unseen" (Any Text), matching MECO and the lang_bias tables
    # (OneStop has no "all" content mode on disk).
    "OneStop": {"preview": "Gathering", "content_mode": "unseen", "version_path": "fully_agg/ordinary/all"},
}


# A tick sitting exactly on the end of its axis reads as the data being cut off
# there rather than as the scale simply running out. Extending the limits past
# the outermost tick gives the spine a little overshoot, which reads as a scale
# that ends where the axis does.
AXIS_TICK_OVERSHOOT = 0.03       # of the axis range, at each end


def _freeze_yticks(ax):
    """Pin the y ticks to the ones the locator chose for the current window."""
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    if ticks:
        ax.set_yticks(ticks)
    return ticks


def _tick_yaxis_to_data(ax, data_lo, data_hi):
    """Tick the panel against its DATA, independent of the room the labels need.

    Called while the window still is the data window: the locator is given an
    explicit bin budget so the step is the one the data warrants (0.25 here),
    not whatever a window inflated by the label band happens to suggest, and the
    ticks are then frozen so no later widening adds another.
    """
    import matplotlib.ticker as _mticker
    lo, hi = ax.get_ylim()
    ticks = [t for t in _mticker.MaxNLocator(nbins=9, steps=[1, 2.5, 5, 10])
             .tick_values(lo, hi) if lo <= t <= hi]
    if ticks:
        ax.set_yticks(ticks)
    return _trim_yticks_to_data(ax, data_lo, data_hi)


def _trim_yticks_to_data(ax, data_lo, data_hi):
    """Drop the ticks the data never reaches, keeping one past it at each end.

    The window is wider than the data by design -- the label band needs room
    above it, and the axis overshoots its last tick -- but that room is space
    to draw in, not more scale to read. Ticking it anyway walks the scale up
    past the data: a panel whose points stop at 0.70 was ticked to 1.00, which
    tells the reader about 0.30 of range that nothing is plotted in. Trimming
    rather than re-ticking the data window on its own keeps the STEP the
    locator chose (0.25 here, not the 0.20 a narrower window would pick).
    """
    ticks = _freeze_yticks(ax)
    if not ticks:
        return ticks
    above = [t for t in ticks if t >= data_hi]
    below = [t for t in ticks if t <= data_lo]
    top = min(above) if above else max(ticks)
    bottom = max(below) if below else min(ticks)
    kept = [t for t in ticks if bottom <= t <= top]
    if kept:
        ax.set_yticks(kept)
    return kept


def _extend_y_past_ticks(ax, frac=AXIS_TICK_OVERSHOOT):
    """Grow the y limits past the outermost tick, keeping the ticks as chosen."""
    lo, hi = ax.get_ylim()
    _freeze_yticks(ax)
    pad = frac * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def _render_residual_vs_distance_on_ax(
    ax,
    merged: pd.DataFrame,
    target_col: str,
    feature_set: str,
    distance_type: str,
    dataset: str,
    color_map: Optional[dict] = None,
    marker_offsets: Optional[dict] = None,
    show_legend: bool = True,
    show_mean_markers: bool = False,
    lang_ticks: bool = False,
    label_merges: Optional[list] = None,
    flip_axes: bool = False,
    y_mode: str = "residual",
):
    """Draw the per-participant residual-vs-distance scatter onto an Axes.

    If `color_map` is provided, use that mapping from L1 -> color (so the same
    language gets the same color across multiple subplots). If `show_legend`
    is False, the per-axis legend is suppressed.

    Additional paper-figure knobs:
      - `show_mean_markers`: overlay a black-edged diamond at the mean
        residual per language.
      - `lang_ticks`: replace numeric x-tick labels with the language name at
        each language's typological-distance position (rotated for legibility).

    `y_mode` selects what goes on the y-axis:
      - "residual" (default): per-participant residual from the OLS fit of
        eye_score on `target_col` (proficiency).
      - "eyescore": raw per-participant eye_score, no proficiency regression.

    Returns (r, p, n, label_entries) for the participant-level Pearson
    correlation, or None if there isn't enough data to compute one.
    `label_entries` describes the per-language labels but does NOT draw
    them — the caller places them once the figure is laid out.
    """
    merged = merged.copy()
    if y_mode == "residual":
        z = np.polyfit(merged[target_col], merged["eye_score"], 1)
        p_line = np.poly1d(z)
        merged["_y"] = merged["eye_score"] - p_line(merged[target_col])
    elif y_mode == "eyescore":
        merged["_y"] = merged["eye_score"]
    else:
        raise ValueError(f"Unknown y_mode={y_mode!r}; expected 'residual' or 'eyescore'")

    distances = compute_typological_distances(
        merged["L1"].unique().tolist(), distance_type=distance_type,
    )
    merged["distance"] = merged["L1"].map(distances)
    merged = merged.dropna(subset=["distance"])

    if len(merged) < 3 or merged["L1"].nunique() < 2:
        ax.text(0.5, 0.5, f"Insufficient data for {dataset}",
                transform=ax.transAxes, ha="center", va="center")
        return None

    r, p_val = pearsonr(merged["distance"], merged["_y"])
    # Same fit as the r above, reported as an unstandardized slope so the
    # figure and the b_dist tables show the same number.
    b_dist = float(linregress(merged["distance"].to_numpy(float),
                              merged["_y"].to_numpy(float)).slope)

    languages = sorted(merged["L1"].unique())
    if color_map is None:
        color_map = _language_color_map(languages, dataset)

    xlim = DISTANCE_XLIM_BY_DATASET.get(dataset, DISTANCE_XLIM_DEFAULT)
    jitter_w = _jitter_halfwidth(dataset)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-jitter_w, jitter_w, size=len(merged))

    for lang in languages:
        mask = (merged["L1"] == lang).to_numpy()
        dist_vals = merged["distance"].to_numpy()[mask] + jitter[mask]
        y_vals = merged["_y"].to_numpy()[mask]
        sx, sy = (y_vals, dist_vals) if flip_axes else (dist_vals, y_vals)
        ax.scatter(
            sx, sy,
            color=color_map[lang],
            label=_lang_label(lang), s=28, alpha=SCATTER_ALPHA,
            edgecolors="white", linewidth=0.4, zorder=3,
        )

    if show_mean_markers:
        for lang in languages:
            sub = merged[merged["L1"] == lang]
            if sub.empty:
                continue
            # Two languages a few thousandths apart on the axis get mean markers
            # that overlap -- the marker is ~9px across and their centres can be
            # closer than that. An offset here slides one along x, by less than
            # the jitter already applied to that language's own points, purely so
            # the two diamonds read as two.
            dist = sub["distance"].iloc[0] + (marker_offsets or {}).get(lang, 0.0)
            mean_y = sub["_y"].mean()
            mx, my = (mean_y, dist) if flip_axes else (dist, mean_y)
            ax.scatter(
                [mx], [my],
                color=color_map[lang], s=45, marker="D",
                edgecolors="black", linewidth=0.7, zorder=5,
            )

    z_fit = np.polyfit(merged["distance"], merged["_y"], 1)
    p_fit = np.poly1d(z_fit)
    line_dist = np.linspace(merged["distance"].min(), merged["distance"].max(), 100)
    line_y = p_fit(line_dist)
    lx, ly = (line_y, line_dist) if flip_axes else (line_dist, line_y)
    ax.plot(lx, ly, "--", color=FIT_LINE_COLOR, alpha=FIT_LINE_ALPHA,
            linewidth=FIT_LINE_WIDTH)
    y_lim = (
        _residual_ylim(feature_set) if y_mode == "residual"
        else _eyescore_ylim(feature_set, RAW_VARIANT)
    )
    if flip_axes:
        ax.axvline(0, color="gray", linewidth=1.0, alpha=0.5)
        ax.set_ylim(*xlim)
        ax.set_xlim(*y_lim)
    else:
        ax.axhline(0, color="gray", linewidth=1.0, alpha=0.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*y_lim)
    ax.grid(False)
    sns.despine(ax=ax)

    label_entries = []
    if lang_ticks:
        # One tick per language at its exact typological-distance position.
        # `label_merges` entries control how nearly-co-located languages are
        # labeled:
        #   - bare list ["A", "B"]   → split into stacked rows (leader lines
        #                              connect each tick to its own label),
        #   - tuple (["A","B"], "X") → combined into one label "X" placed
        #                              at the mean of their positions.
        split_groups = []          # list[list[str]]
        combine_groups = []        # list[(list[str], override_label)]
        for entry in (label_merges or []):
            if isinstance(entry, tuple) and len(entry) == 2:
                langs, override = entry
                combine_groups.append((list(langs), override))
            else:
                split_groups.append(list(entry))

        stagger_pos = {}
        for group in split_groups:
            for i, lang in enumerate(group):
                stagger_pos[lang] = i

        combine_lookup = {}        # lang -> (override_label, group_langs)
        for langs, override in combine_groups:
            for lang in langs:
                combine_lookup[lang] = (override, langs)

        # Per-language entries (one per L1 in the data).
        per_lang = []
        for lang in languages:
            d = float(merged.loc[merged["L1"] == lang, "distance"].iloc[0])
            per_lang.append((lang, d))
        per_lang.sort(key=lambda x: x[1])

        # Build the tick list: combined groups collapse to a single tick at
        # the group's mean position; everything else gets its own tick.
        seen_combine = set()
        tick_positions = []
        tick_kinds = []  # ("solo", lang, d) | ("combine", label, mean_d)
        for lang, d in per_lang:
            if lang in combine_lookup:
                override, group_langs = combine_lookup[lang]
                key = tuple(group_langs)
                if key in seen_combine:
                    continue
                seen_combine.add(key)
                mean_d = float(np.mean([
                    float(merged.loc[merged["L1"] == g, "distance"].iloc[0])
                    for g in group_langs
                    if g in set(languages)
                ]))
                tick_positions.append(mean_d)
                tick_kinds.append(("combine", override, mean_d))
            else:
                tick_positions.append(d)
                tick_kinds.append(("solo", lang, d))

        if flip_axes:
            ax.set_yticks(tick_positions)
            ax.set_yticklabels([""] * len(tick_positions))
            base_off = -0.018
            step = 0.055
            for kind, label_or_lang, d in tick_kinds:
                if kind == "combine":
                    ax.text(
                        base_off, d, _lang_label(label_or_lang),
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7,
                    )
                else:
                    pi = stagger_pos.get(label_or_lang, 0)
                    x_off = base_off - step * pi
                    ax.text(
                        x_off, d, _lang_label(label_or_lang),
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7,
                    )
        else:
            # No bottom tick labels — language names are drawn above each
            # cluster instead. Only COLLECT them here: the positions are
            # solved in display space by label_placement, which has to run
            # after the figure is laid out (see
            # plot_residual_vs_distance_split), because tight_layout resizes
            # the axes and would invalidate any fit computed now.
            ax.set_xticks([])
            for kind, label_or_lang, d in tick_kinds:
                if kind == "combine":
                    group_langs = next(
                        (g for g, ov in combine_groups if ov == label_or_lang),
                        None,
                    )
                    sub = merged[merged["L1"].isin(group_langs)] if group_langs else None
                    label_color = (
                        color_map.get(group_langs[0], "black")
                        if group_langs else "black"
                    )
                else:
                    sub = merged[merged["L1"] == label_or_lang]
                    label_color = color_map.get(label_or_lang, "black")
                if sub is None or sub.empty:
                    continue
                label_entries.append({
                    "label": _lang_label(label_or_lang),
                    "x": d,
                    "y_top": float(sub["_y"].max()),
                    "points": sub[["distance", "_y"]].to_numpy(),
                    "color": label_color,
                })

    if show_legend:
        n_langs = len(languages)
        ncol = 2 if n_langs > 9 else 1
        ax.legend(title="L1", loc="upper right", fontsize=7, ncol=ncol,
                  framealpha=0.85, borderpad=0.3, handletextpad=0.4,
                  columnspacing=0.8, labelspacing=0.25)
    return r, p_val, len(merged), label_entries, b_dist


# Passes the single-panel figure gets to fit its y window around the label band
# (one is normally enough; a second only if re-solving found a taller layout).
SPLIT_FIT_PASSES = 3
SPLIT_LABEL_HEADROOM_PX = 10     # gap kept between the top label and the top of
                                 # the panel
SPLIT_YLIM_MARGIN = 0.06         # of the data range, at each end, before labels


def _points_yrange(entries):
    """(min, max) of the y values one panel plots, from its label entries."""
    ys = np.concatenate([np.asarray(e["points"], float)[:, 1] for e in entries])
    return float(ys.min()), float(ys.max())


def _fit_ylim_to_points(entries, margin=SPLIT_YLIM_MARGIN):
    """A y window around the data of one panel, from its label entries."""
    lo, hi = _points_yrange(entries)
    pad = margin * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


def _raise_top_for_labels(ax, texts, renderer, headroom_px):
    """Raise the panel's top until its labels clear it by `headroom_px`.

    Returns True when the limit moved, i.e. when the labels have to be solved
    again against the new geometry.
    """
    if not texts:
        return False
    axes_top = ax.transAxes.transform((0.0, 1.0))[1]
    band_top = max(t.get_window_extent(renderer=renderer).y1 for t in texts)
    short_by = band_top + headroom_px - axes_top
    if short_by <= 0:
        return False
    inv = ax.transData.inverted()
    grow = (inv.transform((0.0, axes_top + short_by))[1]
            - inv.transform((0.0, axes_top))[1])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + grow)
    return True


def _replace_language_labels(ax, entries, previous, fontsize):
    """Re-solve one panel's labels, returning the Texts that were drawn."""
    for t in previous:
        t.remove()
    before = list(ax.texts)
    place_language_labels(
        ax, entries, fontsize=fontsize,
        rotation=LANG_LABEL_ROTATION, pad_y=LANG_LABEL_PAD_Y,
        lanes_below=SPLIT_LANG_LANES_BELOW,
        above_weight=SPLIT_LANG_ABOVE_WEIGHT,
    )
    return [t for t in ax.texts if t not in before]


def _band_overhead(ax, texts, renderer, data_hi):
    """How far the label band rises above the data, as a share of the panel."""
    if not texts:
        return 0.0
    y0, y1 = (ax.transAxes.transform((0.0, f))[1] for f in (0.0, 1.0))
    data_px = ax.transData.transform((0.0, data_hi))[1]
    band_px = max(t.get_window_extent(renderer=renderer).y1 for t in texts)
    return (band_px - data_px) / (y1 - y0)


def _fit_label_size(fig, pending_labels, placed, renderer, data_yrange, data_ylim):
    """Largest label size whose settled band stays inside SPLIT_BAND_BUDGET_FRAC.

    Each candidate is measured after a full band fit, not before one: the fit
    grows the window and re-solves, and the layout it settles on is not the one
    a first solve produces -- judging a size on the first solve picked 21pt for
    MECO and landed at 26% of the panel. The window is put back to the data fit
    between candidates, since the fit only ever grows it.

    The smallest size is the floor: a panel too crowded for any of them keeps
    its labels legible and pays the band.
    """
    chosen = SPLIT_LANG_LABEL_SIZES[-1]
    for size in SPLIT_LANG_LABEL_SIZES:
        for ax, _ in pending_labels:
            ax.set_ylim(*data_ylim[ax])
        fig.canvas.draw()
        _fit_label_band(fig, pending_labels, placed, renderer, size)
        chosen = size
        worst = max(_band_overhead(ax, placed[ax], renderer, data_yrange[ax][1])
                    for ax, _ in pending_labels)
        if worst <= SPLIT_BAND_BUDGET_FRAC:
            break
    return chosen


# How far the x axis runs past the outermost drawn POINT, as a fraction of the
# spread of languages -- the axis's own overshoot, the counterpart of
# AXIS_TICK_OVERSHOOT on y. About half the tail the old 2%-of-the-spread
# padding left once the jitter is taken off it.
SPLIT_XLIM_MARGIN = 0.006
# The single-panel figures label their clusters at the largest size the panel can
# carry, tried largest first. One size for the whole family would have to be the
# size MECO's 17 crowded names can take, and OneStop's 10 would wear it for no
# reason: at 15pt MECO's band is 19% of the panel and OneStop's is 9%, so
# OneStop can go several points larger before it costs what MECO already costs.
# The budget is what actually matters -- the band is empty axis, because the y
# window is fitted to it -- so it is the thing held fixed and the type is what
# gives. They also hold the band down harder than the solver's default weights:
# room to drop BELOW a cluster, and a heavy price on climbing over the panel top.
SPLIT_LANG_LABEL_SIZES = (21, 19, 17, 15, 13)
SPLIT_BAND_BUDGET_FRAC = 0.20    # of the panel height, measured from the data
SPLIT_LANG_LANES_BELOW = 3
SPLIT_LANG_ABOVE_WEIGHT = 6.0

SPLIT_RP_PAD_PX = 8              # gap between the label band and the r/p box
SPLIT_RP_MIN_GAP_PX = 14         # gap the r/p box keeps from the top of the
                                 # panel, however low the labels sit (the grid
                                 # header keeps the same two distances)
SPLIT_RP_FONTSIZE = 20


def _annotate_pearson(ax, r, p_val, b_dist, label_texts, renderer):
    """Draw the r/p box above a panel's label band, clear of both it and the panel."""
    axes_top = ax.transAxes.transform((0.0, 1.0))[1]
    band_top = max([t.get_window_extent(renderer=renderer).y1
                    for t in label_texts] or [axes_top])
    base = max(axes_top + SPLIT_RP_MIN_GAP_PX, band_top + SPLIT_RP_PAD_PX)
    text = ax.text(
        0.015, ax.transAxes.inverted().transform((0.0, base))[1],
        _pearson_annotation(r, p_val, b_dist),
        transform=ax.transAxes, ha="left", va="bottom", fontsize=SPLIT_RP_FONTSIZE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#bbb", alpha=0.85),
        clip_on=False,
    )
    # The base positions the glyphs; the box is drawn a few px below them, so
    # measure what was actually painted and lift until both gaps are honoured.
    ax.figure.canvas.draw()
    box_bottom = _drawn_extent(text, renderer).y0
    lift = max(0.0, SPLIT_RP_PAD_PX - (box_bottom - band_top),
               SPLIT_RP_MIN_GAP_PX - (box_bottom - axes_top))
    if lift > 0:
        text.set_position(
            (0.015, ax.transAxes.inverted().transform((0.0, base + lift))[1]))
    return text


def _fit_label_band(fig, pending_labels, placed, renderer, fontsize):
    """Solve every panel's labels, raising tops until each band has its headroom.

    Ends on a SOLVED state that has been measured: the loop only exits when a
    pass placed the labels and needed no further room. Returns False if it ran
    out of passes first, i.e. the geometry that ships was never confirmed.
    """
    for _ in range(SPLIT_FIT_PASSES):
        for ax, entries in pending_labels:
            placed[ax] = _replace_language_labels(ax, entries, placed[ax], fontsize)
        fig.canvas.draw()
        grew = False
        for ax, _ in pending_labels:
            grew |= _raise_top_for_labels(ax, placed[ax], renderer,
                                          SPLIT_LABEL_HEADROOM_PX)
        if not grew:
            return True
        fig.canvas.draw()
    return False


def _swap_label_positions(ax, texts, pairs):
    """Exchange the drawn positions of named pairs of labels.

    A deliberate override of the solver, for the one panel where its answer is
    defensible but reads wrong to someone who knows the languages. The solver
    scores geometry -- overlap, distance from the cluster, room above the panel
    -- and two names it places equally well can still be the wrong way round to
    a reader. Rather than teach it a rule for that, the figure that needs it
    says so by name, in the generator, next to the figure it belongs to.

    Both labels keep their own text and colour and take the other's position, so
    a swap is only safe between labels of similar length -- a much longer name
    landing in a shorter one's slot can foul its neighbour.
    """
    by_name = {t.get_text(): t for t in texts}
    for first, second in pairs or ():
        a, b = by_name.get(first), by_name.get(second)
        if a is None or b is None:
            logger.warning(
                "Label swap %s/%s skipped: %s not drawn on this panel",
                first, second, first if a is None else second)
            continue
        pos_a, pos_b = a.get_position(), b.get_position()
        a.set_position(pos_b)
        b.set_position(pos_a)


def _place_labels_relative(ax, texts, rules):
    """Move a label to a fixed pixel offset from another label.

    The companion to `_swap_label_positions` for the case a swap cannot express:
    a name wanted somewhere specific -- above and a little left of its
    neighbour -- rather than merely traded with one. Offsets are in display px
    from the reference label's own anchor, which is stable against the y window
    moving under the figure the way data coordinates are not: +y is up.

    An offset of ("after", gap) instead of (dx, dy) puts the label a gap past the
    END of the reference along its baseline, measured at draw time -- the form to
    use for "right next to X", since the label size is fitted per figure.

    Rules are applied in order, so one may build on another: nudge a label, then
    hang a second off its new position.
    """
    by_name = {t.get_text(): t for t in texts}
    renderer = ax.figure.canvas.get_renderer()
    for name, reference, offset in rules or ():
        target = by_name.get(name)
        # A reference of None means "from where the solver put it": the same
        # rule shape, used as a nudge rather than a relation.
        anchor = target if reference is None else by_name.get(reference)
        if target is None or anchor is None:
            logger.warning(
                "Label placement %s relative to %s skipped: %s not drawn",
                name, reference or name, name if target is None else reference)
            continue
        if isinstance(offset, str) or (offset and offset[0] == "after"):
            # "after": follow the reference along its own baseline, starting a
            # gap past where it ENDS. Measured rather than given in px, because
            # the label size is fitted per figure -- a fixed offset that cleared
            # a 17pt name lands on top of the same name set at 21pt.
            gap = offset[1] if not isinstance(offset, str) else 8.0
            rot = np.deg2rad(anchor.get_rotation())
            saved = anchor.get_rotation()
            anchor.set_rotation(0)
            width = anchor.get_window_extent(renderer=renderer).width
            anchor.set_rotation(saved)
            step = width + gap
            dx, dy = step * np.cos(rot), step * np.sin(rot)
        else:
            dx, dy = offset
        px = ax.transData.transform(anchor.get_position())
        target.set_position(ax.transData.inverted().transform(px + np.array([dx, dy])))


def plot_residual_vs_distance_split(
    merged_by_dataset: dict,
    target_col: str,
    feature_set: str,
    save_path: Path,
    distance_type: str = "syntactic+genetic",
    flip_axes: bool = False,
    y_mode: str = "residual",
    label_swaps: Optional[list] = None,
    label_placements: Optional[list] = None,
    marker_offsets: Optional[dict] = None,
):
    """Two-panel figure: Meco on top, OneStop on bottom, sharing x-axis.

    Per-participant EyeScore residual (regressed against proficiency) plotted
    against the L1-to-English typological distance. One subplot per dataset.

    `y_mode="eyescore"` swaps the y-axis from the proficiency residual to the
    raw eye_score; everything else (panels, palette, label policy) is shared.
    """
    dataset_order = ["Meco", "OneStop"]
    datasets = [d for d in dataset_order if d in merged_by_dataset]
    if not datasets:
        logger.warning("No datasets supplied for split residual-vs-distance plot; skipping")
        return

    # Shared L1 → color so the same language is the same color in both
    # subplots. `husl` gives perceptually-uniform hues that stay distinct out
    # to ~25 categories — `tab20` repeats hues at this many levels.
    all_languages = sorted({
        l1
        for df in merged_by_dataset.values()
        for l1 in df["L1"].dropna().unique()
    })
    shared_color_map = _language_color_map(
        all_languages, datasets[0] if len(datasets) == 1 else None)

    if flip_axes:
        fig, axes = plt.subplots(
            1, len(datasets), figsize=(5.5 * len(datasets), 6.5), sharey=True,
        )
    else:
        fig, axes = plt.subplots(
            len(datasets), 1, figsize=(11, 12.0 * len(datasets)), sharex=False,
        )
    if len(datasets) == 1:
        axes = [axes]

    # Per-dataset manual merges: pairs of L1s whose typological distances are
    # so close that individual ticks visibly overlap. Each entry becomes one
    # combined "Lang1 / Lang2" tick at the mean position.
    merges_by_dataset = {
        "Meco": [
            ["Dutch", "Norwegian"],
            ["Russian", "Italian"],
            ["Hebrew", "Hindi"],
        ],
        "OneStop": [
            ["Vietnamese", "Chinese"],
        ],
    }

    # Union of distances across datasets — both subplots get the same xlim so
    # they line up visually (each keeps its own per-dataset L1 tick labels).
    all_distances = []
    for dataset in datasets:
        ds_l1s = merged_by_dataset[dataset]["L1"].dropna().unique().tolist()
        all_distances.extend(
            d for d in compute_typological_distances(
                ds_l1s, distance_type=distance_type,
            ).values() if d is not None
        )
    if all_distances:
        x_lo, x_hi = float(min(all_distances)), float(max(all_distances))
        # Measured from the outermost PLOTTED point, jitter included, so the
        # tail of axis past the last cluster is the margin and nothing else.
        # Padding the language position instead left a tail that grew with
        # whatever jitter the dataset happened to use.
        jitter = max(_jitter_halfwidth(d) for d in datasets)
        pad = (jitter + SPLIT_XLIM_MARGIN * (x_hi - x_lo)) if x_hi > x_lo else 0.01
        joint_xlim = (x_lo - pad, x_hi + pad)
    else:
        joint_xlim = None

    if y_mode == "residual":
        # The residual is taken against `target_col`, so the axis has to name
        # THAT test: with the test hardcoded, the Composite and Michigan
        # variants claimed to be LexTALE residuals, which they are not -- on
        # MECO the two tests correlate r=0.21, and the Composite cohort
        # includes ~97 participants with no LexTALE score at all.
        y_label = (f"EyeScore ~ {_display_name(target_col, AXIS_TARGET_DISPLAY)} "
                   f"Residual")
        flipped_y_label = "EyeScore Residual"
    else:
        y_label = "EyeScore"
        flipped_y_label = "EyeScore"

    pending_labels = []
    panel_stats = []
    data_yrange = {}
    for ax, dataset in zip(axes, datasets):
        rendered = _render_residual_vs_distance_on_ax(
            ax, merged_by_dataset[dataset], target_col, feature_set,
            distance_type, dataset,
            color_map=shared_color_map, show_legend=False,
            show_mean_markers=True, lang_ticks=True,
            marker_offsets=marker_offsets,
            label_merges=merges_by_dataset.get(dataset),
            flip_axes=flip_axes,
            y_mode=y_mode,
        )
        if rendered is not None:
            pending_labels.append((ax, rendered[3]))
            data_yrange[ax] = _points_yrange(rendered[3])
            # r, p and the slope, held back until the labels are placed: the
            # annotation goes above them, and how tall they stack is not known
            # until then.
            panel_stats.append((ax, rendered[0], rendered[1], rendered[4]))
        dataset_label = "MECO" if dataset == "Meco" else dataset
        if flip_axes:
            ax.set_xlabel(f"{dataset_label}\n{flipped_y_label}", fontsize=11)
            ax.set_xlim(-1, 1)
            if joint_xlim is not None:
                ax.set_ylim(*joint_xlim)
        else:
            ax.set_ylabel(y_label, fontsize=22)
            ax.tick_params(axis="y", labelsize=21, length=9, width=1.4)
            ax.set_box_aspect(1 / 1.5)
            # Fitted to THIS panel's data rather than to one window shared by
            # every figure in the family: the residual and raw-eye_score views
            # of two corpora do not span the same range, and a window wide
            # enough for all of them leaves the narrow ones as a stripe. The
            # label band is not accounted for here -- how tall it is only
            # becomes known once the labels are placed, so the top is raised
            # again below, once they are.
            if rendered is not None:
                ax.set_ylim(*_fit_ylim_to_points(rendered[3]))
                _tick_yaxis_to_data(ax, *_points_yrange(rendered[3]))
            if joint_xlim is not None:
                ax.set_xlim(*joint_xlim)
            # Numeric typological-distance ticks drawn below the bottom
            # spine — short downward tick marks with labels just below.
            import matplotlib.ticker as _mticker
            xlo, xhi = ax.get_xlim()
            locator = _mticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10])
            for tx in locator.tick_values(xlo, xhi):
                if xlo <= tx <= xhi:
                    ax.plot(
                        [tx, tx], [0, -0.028],
                        transform=ax.get_xaxis_transform(),
                        color="black", linewidth=1.4, clip_on=False, zorder=3,
                    )
                    ax.text(
                        tx, -0.037, f"{tx:.2f}",
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=22, zorder=3,
                    )

    if flip_axes:
        axes[0].set_ylabel("Typological Distance to English", fontsize=11)
    else:
        axes[-1].set_xlabel(
            "Linguistic Distance to English", fontsize=22, labelpad=45,
        )
    plt.tight_layout(pad=0.3, h_pad=0.6)

    # Language labels go on last. The placement solver measures glyphs in
    # display pixels, so it needs the final axes rectangle — running it before
    # tight_layout would fit the layout to an axes size that no longer exists.
    #
    # The panel's top is fitted to the labels here, not to the data: a rotated
    # name rises with its own width, so how much room the band needs is a fact
    # about the glyphs and only measurable once they are placed. Solve, measure,
    # raise the top, re-solve against the taller window -- the labels are solved
    # last either way, so re-solving costs one pass and nothing else.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = {ax: [] for ax, _ in pending_labels}
    # The chosen size leaves the panels already fitted around its band.
    data_ylim = {ax: ax.get_ylim() for ax, _ in pending_labels}
    label_size = _fit_label_size(fig, pending_labels, placed, renderer,
                                 data_yrange, data_ylim)

    # The window has stopped moving for the labels, so the ticks can be settled
    # against it and the axis given its overshoot past the last one.
    if not flip_axes:
        for ax, _ in pending_labels:
            _extend_y_past_ticks(ax)
        fig.canvas.draw()
        # ...and then the band is fitted AGAIN, because that widening is room
        # the solver will spend: given 3% more window it re-solves the labels
        # higher, and on OneStop/residual that put the band back to 1.3px under
        # the panel top when the invariant asks for 10. Checking before the last
        # widening measured a geometry that does not ship; this loop's last act
        # is a measurement of the one that does.
        if not _fit_label_band(fig, pending_labels, placed, renderer, label_size):
            logger.warning(
                "Label band still within %spx of the panel top after %s passes "
                "(%s); the r/p box will sit tight to it.",
                SPLIT_LABEL_HEADROOM_PX, SPLIT_FIT_PASSES, save_path.name)

    # Per-figure overrides land after the solver has had its say, and before the
    # annotation is measured, so the r/p box clears whatever the swap produced.
    for ax, _ in pending_labels:
        _swap_label_positions(ax, placed[ax], label_swaps)
        # After the swaps, so a rule can refer to where a swap left a label.
        _place_labels_relative(ax, placed[ax], label_placements)

    # The r/p annotation goes on last, at a MEASURED height: a pad above the
    # tallest label, and never nearer the panel than SPLIT_RP_MIN_GAP_PX. A
    # fixed axes fraction cannot do this, because the y window is fitted per
    # plot and the labels' height depends on their placement.
    fig.canvas.draw()
    for ax, r, p_val, b_dist in panel_stats:
        _annotate_pearson(ax, r, p_val, b_dist, placed.get(ax, []), renderer)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close()
    logger.info("Saved split %s-vs-distance plot: %s", y_mode, save_path)


# ── Paper-figure grid: one panel per feature set ────────────────────────────

# The four feature sets that have eye_score results for both corpora. The
# fifth SMALL combination (all four concatenated) has never been computed, and
# TRANSITIONS / WFC are empty, so they are deliberately absent.
PAPER_GRID_FEATURE_SETS = [
    "READING_SPEED",
    "FIXATION_METRICS",
    "S_CLUSTERS_NO_NORM",
    "WP_COEFS_NO_NORM",
]

# Feature-set names as they appear in the GRID panel headers. The tables abbreviate
# ("WPM", "Avg. Fix.") because a table column is narrow; a panel header has the
# whole width of the panel, so it spells the feature set out instead.
# Feature sets outside the grid's four keep the table abbreviation.
GRID_FEATURE_SET_DISPLAY = {
    **FEATURE_SET_DISPLAY,
    "READING_SPEED": "Words Per Minute",
    "FIXATION_METRICS": "Average Fixation Metrics",
    "S_CLUSTERS_NO_NORM": "Syntactic Clusters",
    "WP_COEFS_NO_NORM": "Word Property Coefficients",
}

# Labels are placed automatically, but a grid panel is only ~half the width of
# the single-panel figure while type stays a fixed point size, so the same text
# crowds roughly 1.4x harder. Shrinking it buys that room back.
GRID_LABEL_FONTSIZE = 12
GRID_PANEL_ASPECT = 1 / 1.5      # height/width of each panel's box
GRID_TICK_FONTSIZE = 17
GRID_AXIS_FONTSIZE = 15
GRID_RP_FONTSIZE = 13            # Pearson annotation, scaled to the panel
GRID_HEADER_PAD_PX = 8           # gap between the label band and the header row
GRID_HEADER_MIN_GAP_PX = 14      # gap a header keeps from the top of its panel,
                                 # however low the labels below it sit
# Height per grid row. set_box_aspect fits each panel inside its cell and
# centres it, so surplus cell height becomes dead space between the rows that
# no tight_layout padding can reclaim -- the figure height is the real lever.
# 5.8 leaves the panels at full size and the gutter at ~17px; much below 5.7 and
# a lower row's header rides up over the panel above it. It was 5.5 while each
# row's headers found their own level -- one clearance shared by the whole grid
# lifts the lower row's headers, and the gutter has to grow with them.
GRID_ROW_INCHES = 5.8
# Gap left between a row of panels and the header band of the row below it.
# 4px was a bare no-collision floor; at that distance the header box still reads
# as touching the spine above, so it doubles as the fit loop's target.
GRID_MIN_ROW_CLEARANCE_PX = 12.0
GRID_FIT_PASSES = 4              # figure-height growth passes before giving up
# Share of an added inch of figure height that reaches the gutter (the rest goes
# to the margins tight_layout keeps). Only sets the step size -- the loop
# re-measures, so a rough value just costs an extra pass.
GRID_GUTTER_YIELD = 0.8


def _drawn_extent(text, renderer):
    """A Text's extent INCLUDING its bbox patch, when it has one drawn."""
    patch = text.get_bbox_patch()
    if patch is not None:
        try:
            return patch.get_window_extent(renderer=renderer)
        except (AttributeError, RuntimeError):
            pass
    return text.get_window_extent(renderer=renderer)


def _min_row_gutter(axes_in_order, ncols, nrows, renderer):
    """Smallest gap, in px, between a grid row's panels and the row below it.

    The row below is measured by everything it draws, not just its axes: its
    language labels and its header both stand above the panel, and the header's
    box is drawn a few px outside its glyphs.
    """
    gaps = []
    for row in range(1, nrows):
        above = [ax for i, ax in enumerate(axes_in_order) if i // ncols == row - 1]
        below = [ax for i, ax in enumerate(axes_in_order) if i // ncols == row]
        if not above or not below:
            continue
        lowest = min(ax.get_window_extent().y0 for ax in above)
        highest = max(
            [ax.get_window_extent().y1 for ax in below]
            + [_drawn_extent(t, renderer).y1
               for ax in below for t in ax.texts if t.get_text().strip()])
        gaps.append(lowest - highest)
    return min(gaps) if gaps else float("inf")


def plot_distance_grid_by_feature_set(
    merged_by_feature_set: dict,
    target_col: str,
    dataset: str,
    save_path: Path,
    distance_type: str = "syntactic+genetic",
    y_mode: str = "eyescore",
    ncols: int = 2,
    label_swaps: Optional[list] = None,
):
    """2x2 grid of EyeScore-vs-distance panels, one per feature set.

    Same anatomy as the single-panel paper figure — per-participant scatter
    coloured by L1, per-language mean diamonds, an OLS trend and a Pearson
    annotation — but the panels vary the FEATURE SET rather than the dataset,
    so one figure shows whether L1 bias is a property of the scorer or of a
    particular feature family.

    Y limits stay per-feature-set (`_eyescore_ylim` / `_residual_ylim`) rather
    than shared: READING_SPEED runs to about +5 while the other three sit
    inside +/-1, so a common scale would flatten three panels into a stripe.
    That means the panels are comparable in shape, not in absolute height.
    """
    feature_sets = [fs for fs in PAPER_GRID_FEATURE_SETS if fs in merged_by_feature_set]
    if not feature_sets:
        logger.warning("No feature sets supplied for the distance grid; skipping")
        return None

    # One colour per L1 across the whole figure, so a language keeps its
    # identity from panel to panel.
    all_languages = sorted({
        l1 for df in merged_by_feature_set.values() for l1 in df["L1"].dropna().unique()
    })
    shared_color_map = _language_color_map(all_languages, dataset)

    nrows = int(np.ceil(len(feature_sets) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, GRID_ROW_INCHES * nrows), squeeze=False,
    )
    flat = [ax for row in axes for ax in row]

    # Every panel shares the same languages, so one x window keeps the columns
    # aligned and makes the panels directly comparable left to right.
    all_distances = []
    for df in merged_by_feature_set.values():
        all_distances.extend(
            d for d in compute_typological_distances(
                df["L1"].dropna().unique().tolist(), distance_type=distance_type,
            ).values() if d is not None
        )
    if all_distances:
        lo, hi = float(min(all_distances)), float(max(all_distances))
        pad = 0.02 * (hi - lo) if hi > lo else 0.01
        joint_xlim = (lo - pad, hi + pad)
    else:
        joint_xlim = None

    # Named for the test actually regressed out -- see the split figure above.
    y_label = ("EyeScore" if y_mode == "eyescore" else
               f"EyeScore ~ {_display_name(target_col, AXIS_TARGET_DISPLAY)} Residual")

    # One shared y window across the panels. It has to be the UNION of the
    # per-feature-set limits, because READING_SPEED runs to about +5 while the
    # rest sit inside +/-1: anything tighter would clip it. The cost is that
    # the three narrow feature sets use only part of the height.
    per_set_ylim = [
        _residual_ylim(fs) if y_mode == "residual"
        else _eyescore_ylim(fs, RAW_VARIANT)
        for fs in feature_sets
    ]
    shared_ylim = (min(lo for lo, _ in per_set_ylim),
                   max(hi for _, hi in per_set_ylim))

    pending_labels = []
    headers = []
    for idx, (ax, feature_set) in enumerate(zip(flat, feature_sets)):
        rendered = _render_residual_vs_distance_on_ax(
            ax, merged_by_feature_set[feature_set], target_col, feature_set,
            distance_type, dataset,
            color_map=shared_color_map, show_legend=False,
            show_mean_markers=True, lang_ticks=True,
            y_mode=y_mode,
        )
        if rendered is not None:
            pending_labels.append((ax, rendered[3]))
            # Both annotations are deferred: they belong ABOVE the language
            # labels, and how tall that band ends up is only known once the
            # labels have actually been placed.
            headers.append((idx, ax, rendered[0], rendered[1], feature_set,
                            rendered[4]))

        ax.set_box_aspect(GRID_PANEL_ASPECT)
        ax.tick_params(axis="y", labelsize=GRID_TICK_FONTSIZE, length=6,
                       width=1.1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(*shared_ylim)
        # With one shared window the right column's tick labels only repeat the
        # left column's, so they are dropped rather than printed twice.
        if idx % ncols != 0:
            ax.tick_params(axis="y", labelleft=False)
        _extend_y_past_ticks(ax)
        if joint_xlim is not None:
            ax.set_xlim(*joint_xlim)

        # Numeric distance ticks under the bottom row only — repeating them
        # under every panel is noise when the columns share an x window.
        ax.set_xticks([])
        if idx >= len(feature_sets) - ncols:
            import matplotlib.ticker as _mticker
            xlo, xhi = ax.get_xlim()
            locator = _mticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
            for tx in locator.tick_values(xlo, xhi):
                if xlo <= tx <= xhi:
                    ax.plot([tx, tx], [0, -0.028],
                            transform=ax.get_xaxis_transform(),
                            color="black", linewidth=1.1, clip_on=False, zorder=3)
                    ax.text(tx, -0.038, f"{tx:.2f}",
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=GRID_TICK_FONTSIZE,
                            zorder=3)

    for ax in flat[len(feature_sets):]:
        ax.set_visible(False)

    fig.supxlabel("Linguistic Distance to English", fontsize=GRID_AXIS_FONTSIZE + 2)
    sup_y = fig.supylabel(y_label, fontsize=GRID_AXIS_FONTSIZE + 2)
    plt.tight_layout(pad=0.6, h_pad=1.2, w_pad=1.6)

    # Labels last, for the same reason as the single-panel figure: the solver
    # works in display pixels and tight_layout resizes the axes.
    fig.canvas.draw()
    label_texts = {}
    for ax, entries in pending_labels:
        before = list(ax.texts)
        place_language_labels(
            ax, entries, fontsize=GRID_LABEL_FONTSIZE,
            rotation=LANG_LABEL_ROTATION, pad_y=LANG_LABEL_PAD_Y,
        )
        label_texts[ax] = [t for t in ax.texts if t not in before]
        _swap_label_positions(ax, label_texts[ax], label_swaps)

    # Header row per panel: the Pearson result on the left, the feature set on
    # the right, both sitting clear above the tallest language label so they
    # read as the panel's caption rather than as part of the label band.
    #
    # The baseline is shared by every panel in a grid ROW, not by the whole
    # grid: rows may sit at different heights, but two headers side by side at
    # levels 20-30px apart read as a mistake. (Columns need no such treatment --
    # the headers are anchored to the axes edges, and the axes are already a
    # uniform grid.) Whatever the labels below it do, no header comes closer to
    # the top of its panel than GRID_HEADER_MIN_GAP_PX; without that floor a
    # panel whose labels all stay inside the axes gets its header printed
    # against the top of the y axis.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    row_baseline = {}
    for idx, ax, *_ in headers:
        top_px = ax.transAxes.transform((0.0, 1.0))[1] + GRID_HEADER_MIN_GAP_PX
        for t in label_texts.get(ax, []):
            top_px = max(top_px, t.get_window_extent(renderer=renderer).y1)
        row = idx // ncols
        row_baseline[row] = max(row_baseline.get(row, top_px), top_px)

    drawn = []
    for idx, ax, r, p_val, feature_set, b_dist in headers:
        base = row_baseline[idx // ncols] + GRID_HEADER_PAD_PX
        y = ax.transAxes.inverted().transform((0.0, base))[1]
        left = ax.text(
            0.0, y, _pearson_annotation(r, p_val, b_dist), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=GRID_RP_FONTSIZE, zorder=6,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#bbb", alpha=0.85))
        right = ax.text(
            # Spelled out in full: a panel header has room the table columns do
            # not, and the abbreviations only pay off where space is tight.
            1.0, y, _display_name(feature_set, GRID_FEATURE_SET_DISPLAY),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=GRID_AXIS_FONTSIZE, fontweight="bold", zorder=6,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.9))
        drawn.append((idx, ax, left, right, base))

    # The floor above is applied to the text's anchor, but what the reader sees
    # is the box, and it is drawn a few px below the glyphs. Measure it and lift
    # the row until the gap under the lowest box in that row is the one asked
    # for -- by the row, so the headers stay level with each other.
    fig.canvas.draw()
    row_lift = {}
    for idx, ax, left, right, base in drawn:
        axes_top = ax.transAxes.transform((0.0, 1.0))[1]
        gap = min(_drawn_extent(t, renderer).y0 - axes_top for t in (left, right))
        row = idx // ncols
        row_lift[row] = max(row_lift.get(row, 0.0), GRID_HEADER_MIN_GAP_PX - gap)
    lifted = []
    for idx, ax, left, right, base in drawn:
        lift = row_lift.get(idx // ncols, 0.0)
        if lift > 0:
            base += lift
            y = ax.transAxes.inverted().transform((0.0, base))[1]
            for t in (left, right):
                t.set_position((t.get_position()[0], y))
        lifted.append((idx, ax, left, right, base))
    drawn = lifted

    # The two ends of the header fit side by side for the feature sets shipped
    # here, but a longer display name (or a narrower panel) would run them
    # together. Stack rather than collide -- and stack the whole row at once,
    # so fixing one panel cannot re-introduce the stagger just removed.
    fig.canvas.draw()
    crowded_rows = set()
    for idx, ax, left, right, _ in drawn:
        lb_, rb_ = (t.get_window_extent(renderer=renderer) for t in (left, right))
        if rb_.x0 < lb_.x1 + GRID_HEADER_PAD_PX:
            crowded_rows.add(idx // ncols)
    for idx, ax, left, right, base in drawn:
        if idx // ncols not in crowded_rows:
            continue
        lift = left.get_window_extent(renderer=renderer).height + GRID_HEADER_PAD_PX
        right.set_position(
            (1.0, ax.transAxes.inverted().transform((0.0, base + lift))[1]))

    # The gutter between rows has to hold a lower panel's whole header band --
    # the labels that rise above it, plus the header standing clear above those
    # -- and how tall that band is only becomes known once the labels have been
    # placed. GRID_ROW_INCHES is the floor, not the answer: rather than hand-tune
    # it for the worst corpus (89px of band on MECO/Composite against 1px on
    # OneStop, which would leave every other grid with a chasm between its rows),
    # grow the figure until the band fits. The panels are pinned to the column
    # width by set_box_aspect, so the added height all lands in the gutter and
    # nothing already placed moves relative to its own panel.
    fig.canvas.draw()
    for _ in range(GRID_FIT_PASSES):
        gutter = _min_row_gutter(flat[:len(feature_sets)], ncols, nrows, renderer)
        deficit = GRID_MIN_ROW_CLEARANCE_PX - gutter
        if deficit <= 0:
            break
        w_in, h_in = fig.get_size_inches()
        fig.set_size_inches(w_in, h_in + deficit / (fig.dpi * GRID_GUTTER_YIELD))
        plt.tight_layout(pad=0.6, h_pad=1.2, w_pad=1.6)
        fig.canvas.draw()
    gutter = _min_row_gutter(flat[:len(feature_sets)], ncols, nrows, renderer)
    if gutter < GRID_MIN_ROW_CLEARANCE_PX:
        logger.warning(
            "Feature-set grid rows are %.1fpx apart after %s fit passes "
            "(%s, %s); a header may be printed over the panel above.",
            gutter, GRID_FIT_PASSES, dataset, y_mode)

    # supylabel's default x sits inside the tick-label column, so it collides
    # with the numbers at the row boundary. Park it just left of the widest
    # tick label rather than guessing a figure fraction.
    tick_x0 = [t.get_window_extent(renderer=renderer).x0
               for ax in flat[:len(feature_sets)]
               for t in ax.get_yticklabels() if t.get_text() and t.get_visible()]
    if tick_x0:
        # The label is anchored ha="left", so its x is the LEFT edge and the
        # glyphs run rightwards from it: subtract its own width as well, or it
        # is placed straight back over the numbers.
        width = sup_y.get_window_extent(renderer=renderer).width
        target = min(tick_x0) - GRID_HEADER_PAD_PX - width
        sup_y.set_x(fig.transFigure.inverted().transform((target, 0.0))[0])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close()
    logger.info("Saved %s-vs-distance feature-set grid: %s", y_mode, save_path)
    return save_path


def make_paper_feature_set_grid(
    dataset: str = "Meco",
    target_col: str = None,
    distance_type: str = "syntactic+genetic",
    save_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    plots_root: Optional[Path] = None,
    y_mode: str = "eyescore",
    feature_sets: Optional[List[str]] = None,
    label_swaps: Optional[list] = None,
) -> Optional[Path]:
    """Driver: load every feature set for one dataset and render the grid."""
    if target_col is None:
        target_col = TestCols.LEXTALE_COL
    if results_dir is None:
        results_dir = RESULTS_DIR
    if feature_sets is None:
        feature_sets = PAPER_GRID_FEATURE_SETS
    if save_path is None:
        plots_base = plots_root / "lang_bias" if plots_root else LANG_BIAS_PLOTS_DIR
        stem = "residual_vs_distance" if y_mode == "residual" else "eyescore_vs_distance"
        save_path = (plots_base / "split" /
                     f"{stem}_by_feature_set_{dataset.lower()}.pdf")

    layout = PAPER_SPLIT_DATASET_LAYOUT.get(dataset)
    if layout is None:
        logger.warning("No paper layout for dataset %s; skipping grid", dataset)
        return None
    meta = load_metadata(dataset, layout["version_path"])
    if meta.empty:
        logger.warning("Empty metadata for %s/%s", dataset, layout["version_path"])
        return None

    merged_by_feature_set = {}
    for feature_set in feature_sets:
        csv_path = eye_score_csv_path(
            results_dir, dataset=dataset, preview=layout["preview"],
            content_mode=layout["content_mode"], version_path=layout["version_path"],
            feature_set=feature_set, variant=RAW_VARIANT,
        )
        if not csv_path.exists():
            logger.warning("Missing eye_score CSV for %s/%s: %s",
                           dataset, feature_set, csv_path)
            continue
        merged = _merge_eyescore_metadata(csv_path, meta, target_col)
        if merged is None:
            logger.warning("No merged data for %s/%s", dataset, feature_set)
            continue
        merged_by_feature_set[feature_set] = merged

    if not merged_by_feature_set:
        return None

    return plot_distance_grid_by_feature_set(
        merged_by_feature_set, target_col, dataset, save_path,
        distance_type=distance_type, y_mode=y_mode, label_swaps=label_swaps,
    )


def make_paper_split_bias_plot(
    feature_set: str = "WP_COEFS_NO_NORM",
    target_col: str = None,
    distance_type: str = "syntactic+genetic",
    save_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    plots_root: Optional[Path] = None,
    datasets: Optional[List[str]] = None,
    y_mode: str = "residual",
    label_swaps: Optional[list] = None,
    label_placements: Optional[list] = None,
    marker_offsets: Optional[dict] = None,
) -> Optional[Path]:
    """Driver: load the canonical eye_score data for a single feature set and
    render the residual-vs-distance figure. `datasets` selects which datasets
    show up as panels — defaults to ``["Meco"]`` (single-panel MECO figure);
    pass ``["Meco", "OneStop"]`` to recover the two-panel split view.

    `y_mode="eyescore"` switches the y-axis to raw eye_score and changes the
    default filename stem to ``eyescore_vs_distance__…``.
    """
    if target_col is None:
        target_col = TestCols.LEXTALE_COL
    if results_dir is None:
        results_dir = RESULTS_DIR
    if datasets is None:
        datasets = ["Meco"]
    if save_path is None:
        plots_base = plots_root / "lang_bias" if plots_root else LANG_BIAS_PLOTS_DIR
        stem = "residual_vs_distance" if y_mode == "residual" else "eyescore_vs_distance"
        # Single-non-Meco-dataset variants get a dataset tag in the filename so
        # they don't clobber the original MECO-only or two-panel split plot.
        if datasets == ["Meco"] or set(datasets) == {"Meco", "OneStop"}:
            dataset_tag = ""
        else:
            dataset_tag = "_" + "_".join(d.lower() for d in datasets)
        save_path = plots_base / "split" / f"{stem}{dataset_tag}__{feature_set}.pdf"

    merged_by_dataset = {}
    for dataset, layout in PAPER_SPLIT_DATASET_LAYOUT.items():
        if dataset not in datasets:
            continue
        csv_path = eye_score_csv_path(
            results_dir, dataset=dataset, preview=layout["preview"],
            content_mode=layout["content_mode"], version_path=layout["version_path"],
            feature_set=feature_set, variant=RAW_VARIANT,
        )
        if not csv_path.exists():
            logger.warning("Missing eye_score CSV for %s: %s", dataset, csv_path)
            continue
        meta = load_metadata(dataset, layout["version_path"])
        if meta.empty:
            logger.warning("Empty metadata for %s/%s", dataset, layout["version_path"])
            continue
        merged = _merge_eyescore_metadata(csv_path, meta, target_col)
        if merged is None:
            logger.warning("No merged data for %s/%s", dataset, feature_set)
            continue
        merged_by_dataset[dataset] = merged

    if not merged_by_dataset:
        return None

    plot_residual_vs_distance_split(
        merged_by_dataset, target_col, feature_set, save_path,
        distance_type=distance_type,
        y_mode=y_mode,
        label_swaps=label_swaps,
        label_placements=label_placements,
        marker_offsets=marker_offsets,
    )
    return save_path


# ── Orchestration ────────────────────────────────────────────────────────────


DISTANCE_TYPES = ["syntactic+genetic"]


# ── Simplified bias pipeline ─────────────────────────────────────────────────
#
# Output layout (paper-ready, topic-first to mirror tables/lang_bias and
# results/lang_bias):
#   PLOTS_ROOT / lang_bias / {dataset} / {mode_label} / {feature_set} /
#       ├── scatter_proficiency_vs_eyescore.png
#       └── residual_vs_distance.png
#
# Summary CSV (consumed by the table generators in tables.py) lives under
# evaluation results — not plots — since it's tabular data, not an image:
#   EVAL_SAVE_DIR / lang_bias / {mode_label} / bias_correlation_summary.csv

# `which_plots` selector values: emit just the proficiency-vs-eyescore scatter
# ("scatter"), just the residual-vs-distance scatter ("residual"), or both.
WHICH_PLOTS_CHOICES = ("scatter", "residual", "both")

# (mode_label, csv_variant). "org" uses the raw eye_score CSV. The remaining
# entries pair a split scheme (seen = inside_langs, new = across_langs) with a
# debias target (lextale / michtest / profagg) and point at the corresponding
# typology-debiased CSV. "seen" / "new" without a suffix keep the original
# LexTALE-debiased CSVs so existing callers don't break.
SIMPLIFIED_MODE_SPECS = [
    ("org", RAW_VARIANT),
    ("seen", SPLIT_SCHEME_TO_VARIANT["inside_langs"]),
    ("new", SPLIT_SCHEME_TO_VARIANT["across_langs"]),
    ("seen_michtest", "typo_calibrated_inside_langs_michtest"),
    ("new_michtest", "typo_calibrated_across_langs_michtest"),
    ("seen_profagg", "typo_calibrated_inside_langs_profagg"),
    ("new_profagg", "typo_calibrated_across_langs_profagg"),
]

# Canonical distance type for which scatters are emitted. Non-canonical
# distance types still contribute rows to the summary CSV. (Until r3 every
# scatter was also gated on `target_col == LEXTALE_COL`; that gate is now off
# so the prof_agg / michtest views are visible alongside LexTALE.)
SIMPLIFIED_PLOT_DISTANCE_TYPE = "syntactic+genetic"


def run_simplified_bias_plots(
    results_dir: Path = None,
    plots_root: Path = None,
    target_cols: List[str] = None,
    feature_sets: List[str] = None,
    distance_types: List[str] = None,
    plot_distance_type: str = SIMPLIFIED_PLOT_DISTANCE_TYPE,
    no_titles: bool = False,
    which_plots: str = "both",
    bias_subdir: str = "lang_bias",
    eval_save_dir: Path = None,
):
    """Simplified bias plot pipeline.

    For each (dataset, mode in {org, seen[/_michtest/_profagg], new[/_michtest/
    _profagg]}, feature_set, target_col) emit up to two scatters (proficiency
    vs. EyeScore + residual vs. L1 typological distance) at the simplified
    output layout described above. Plot filenames are suffixed with the
    evaluation target_col so all three (LexTALE / MichTest / ProfAgg) sit
    side-by-side in the same per-feature_set folder. The summary CSV is
    written for every (target_col, distance_type) combo so the table
    generators retain full data.

    `no_titles=True` suppresses axes titles and figure suptitles (paper-ready).

    `which_plots` ∈ {"scatter", "residual", "both"} selects which figure(s) to
    render per (dataset, mode, feature_set). Defaults to "both".

    `bias_subdir` is the top-level directory name under both `plots_root` and
    `eval_save_dir` where outputs land. Defaults to "lang_bias".

    `eval_save_dir` is the eval results tree the summary CSV is written under
    (default: EVAL_SAVE_DIR). Set this and `plots_root`/`results_dir` to a
    sibling tree (e.g. results_syntactic) when running with a non-default
    distance type so the canonical outputs aren't overwritten.
    """
    if which_plots not in WHICH_PLOTS_CHOICES:
        raise ValueError(
            f"which_plots must be one of {WHICH_PLOTS_CHOICES!r}, "
            f"got {which_plots!r}"
        )
    if no_titles:
        with suppress_titles():
            return run_simplified_bias_plots(
                results_dir=results_dir, plots_root=plots_root,
                target_cols=target_cols, feature_sets=feature_sets,
                distance_types=distance_types,
                plot_distance_type=plot_distance_type,
                no_titles=False, which_plots=which_plots,
                bias_subdir=bias_subdir, eval_save_dir=eval_save_dir,
            )
    if results_dir is None:
        results_dir = RESULTS_DIR
    if plots_root is None:
        plots_root = PLOTS_ROOT
    if eval_save_dir is None:
        eval_save_dir = EVAL_SAVE_DIR
    if target_cols is None:
        target_cols = [TestCols.MICHIGEN_TEST_COL, TestCols.LEXTALE_COL,
                       TestCols.PROFICIENCY_AGG_COL]
    if distance_types is None:
        distance_types = DISTANCE_TYPES

    for mode_label, csv_variant in SIMPLIFIED_MODE_SPECS:
        entries = collect_eyescore_csvs(
            results_dir, feature_sets=feature_sets, variant=csv_variant,
        )
        if not entries:
            logger.warning(
                "No eye_score CSVs for mode %s (variant=%r); skipping",
                mode_label, csv_variant,
            )
            continue

        metadata_cache = {}
        all_correlations = []

        for entry in tqdm(entries, desc=f"Simplified bias plots ({mode_label})"):
            dataset = entry["dataset"]
            preview = entry["preview"]
            content_mode = entry["content_mode"]
            version_path = entry["version_path"]
            feature_set = entry["feature_set"]
            eye_csv = entry["csv_path"]

            cache_key = f"{dataset}/{version_path}"
            if cache_key not in metadata_cache:
                metadata_cache[cache_key] = load_metadata(dataset, version_path)
            meta = metadata_cache[cache_key]
            if meta.empty:
                continue

            group_dir = plots_root.joinpath(
                bias_subdir, dataset, *preview_dir_segments(dataset, preview),
                mode_label, feature_set,
            )

            want_scatter_kind = which_plots in ("scatter", "both")
            want_residual_kind = which_plots in ("residual", "both")

            for target_col in target_cols:
                merged = _merge_eyescore_metadata(eye_csv, meta, target_col)
                if merged is None:
                    continue

                # Plot 1 — proficiency vs. EyeScore, one file per eval target
                # in the same per-feature_set folder.
                plot_proficiency_vs_eyescore(
                    merged, target_col, feature_set, title_prefix="",
                    save_path=(group_dir / f"scatter_proficiency_vs_eyescore_{target_col}.png"
                               if want_scatter_kind else None),
                    variant=csv_variant,
                )

                # Plot 2 — residual vs. distance, one file per eval target at
                # the canonical distance type. Off-canonical distances still
                # contribute to the summary CSV but don't get a figure.
                for dist_type in distance_types:
                    is_canonical_dist = (dist_type == plot_distance_type)
                    save_path = (
                        group_dir / f"residual_vs_distance_{target_col}.png"
                        if (is_canonical_dist and want_residual_kind)
                        else None
                    )
                    corr = plot_residual_vs_distance(
                        merged, target_col, feature_set, title_prefix="",
                        save_path=save_path, distance_type=dist_type,
                        variant=csv_variant, dataset=dataset,
                    )
                    if corr is not None:
                        all_correlations.append({
                            "dataset": dataset,
                            "preview": preview,
                            "content_mode": content_mode,
                            "version_path": version_path,
                            "feature_set": feature_set,
                            "target_col": target_col,
                            "distance_type": dist_type,
                            "variant": mode_label,
                            **corr,
                        })

        if all_correlations:
            summary_df = pd.DataFrame(all_correlations)
            summary_path = (
                eval_save_dir / bias_subdir / mode_label
                / "bias_correlation_summary.csv"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(summary_path, index=False)
            logger.info(
                "Saved bias correlation summary (%s): %s",
                mode_label, summary_path,
            )


def run_language_bias_analysis(
    results_dir: Path = None,
    plots_dir: Path = None,
    target_cols: List[str] = None,
    feature_sets: List[str] = None,
    distance_types: List[str] = None,
    variant: str = RAW_VARIANT,
):
    """Run the full language-bias analysis for one (variant) of eye_score CSVs.

    For each (dataset, preview, version_path, feature_set, target_col):
      - Plot 1: proficiency vs EyeScore scatter colored by L1
      - Plot 2: per-participant residual vs typological distance
              (one plot per distance type)

    `plots_dir` should already encode any scheme/variant context; this
    function writes directly under it.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    if plots_dir is None:
        plots_dir = LANG_BIAS_PLOTS_DIR
    if target_cols is None:
        target_cols = [TestCols.MICHIGEN_TEST_COL, TestCols.LEXTALE_COL, TestCols.PROFICIENCY_AGG_COL]
    if distance_types is None:
        distance_types = DISTANCE_TYPES

    entries = collect_eyescore_csvs(results_dir, feature_sets=feature_sets, variant=variant)
    if not entries:
        logger.warning("No eye_score CSVs found in %s for variant %r", results_dir, variant)
        return pd.DataFrame()

    logger.info("Running language-bias analysis on %d eye_score files (variant=%r)",
                len(entries), variant)

    metadata_cache = {}
    all_correlations = []

    for entry in tqdm(entries, desc=f"Language-bias analysis ({variant})"):
        dataset = entry["dataset"]
        preview = entry["preview"]
        content_mode = entry["content_mode"]
        version_path = entry["version_path"]
        feature_set = entry["feature_set"]
        eye_csv = entry["csv_path"]

        cache_key = f"{dataset}/{version_path}"
        if cache_key not in metadata_cache:
            metadata_cache[cache_key] = load_metadata(dataset, version_path)
        meta = metadata_cache[cache_key]
        if meta.empty:
            continue

        preview_display = _display_name(preview, PREVIEW_DISPLAY_NAMES)
        content_mode_display = _display_name(content_mode, CONTENT_MODE_DISPLAY_NAMES)
        title_prefix = (
            f"Dataset: {dataset}  |  Condition: {preview_display}  |  Compared to: {content_mode_display}\n"
            f"Version: {version_path}"
        )

        group_dir = (
            plots_dir / dataset / preview / content_mode / version_path
            / "per_feature" / feature_set
        )

        for target_col in target_cols:
            merged = _merge_eyescore_metadata(eye_csv, meta, target_col)
            if merged is None:
                continue

            # Plot 1: proficiency vs EyeScore
            plot_proficiency_vs_eyescore(
                merged,
                target_col,
                feature_set,
                title_prefix,
                save_path=group_dir / f"scatter_proficiency_vs_eyescore_{target_col}.png",
                variant=variant,
            )

            # Plot 2: residual vs typological distance (one per distance type)
            for dist_type in distance_types:
                corr = plot_residual_vs_distance(
                    merged,
                    target_col,
                    feature_set,
                    title_prefix,
                    save_path=group_dir / dist_type / f"residual_vs_distance_{target_col}.png",
                    distance_type=dist_type,
                    variant=variant,
                    dataset=dataset,
                )

                if corr is not None:
                    all_correlations.append(
                        {
                            "dataset": dataset,
                            "preview": preview,
                            "content_mode": content_mode,
                            "version_path": version_path,
                            "feature_set": feature_set,
                            "target_col": target_col,
                            "distance_type": dist_type,
                            "variant": variant,
                            **corr,
                        }
                    )

    if all_correlations:
        summary_df = pd.DataFrame(all_correlations)
        summary_path = plots_dir / "bias_correlation_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved bias correlation summary to %s", summary_path)
        return summary_df

    return pd.DataFrame()


# ── Aggregative bar plot: bias by feature set, one bar per variant ──────────


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-titles", action="store_true",
        help="Suppress axes titles and figure suptitles in every plot "
             "(paper-ready output).",
    )
    parser.add_argument(
        "--plots", choices=WHICH_PLOTS_CHOICES, default="both",
        help="Which lang_bias scatter to emit per (dataset, mode, feature_set): "
             "'scatter' (proficiency vs EyeScore), 'residual' (residual vs L1 "
             "distance), or 'both' (default).",
    )
    parser.add_argument(
        "--distance-type", default=SIMPLIFIED_PLOT_DISTANCE_TYPE,
        help="URIEL+ distance type to use, e.g. 'syntactic' or "
             "'syntactic+genetic' (default).",
    )
    parser.add_argument(
        "--results-suffix", default="",
        help="Suffix routing inputs/outputs to sibling trees: reads from "
             "src/methods/EyeScore/results_<suffix>/ and writes plots/summary "
             "CSV under src/evaluation/EyeScore/plots_<suffix>/lang_bias/ + "
             "src/evaluation/EyeScore/results_<suffix>/lang_bias/. Default: "
             "empty (canonical trees). Use the same suffix you passed to "
             "src.run.eyescore.",
    )
    args = parser.parse_args()

    feature_sets = list(RELEVANT_FEATURE_GROUPS_COMBINATIONS_SMALL.keys())
    target_cols = [TestCols.MICHIGEN_TEST_COL, TestCols.LEXTALE_COL, TestCols.PROFICIENCY_AGG_COL]
    if args.results_suffix:
        suffix = f"_{args.results_suffix}"
        results_dir = Path(f"src/methods/EyeScore/results{suffix}")
        plots_root = Path(f"src/evaluation/EyeScore/plots{suffix}")
        eval_save_dir = Path(f"src/evaluation/EyeScore/results{suffix}")
    else:
        results_dir = None
        plots_root = None
        eval_save_dir = None
    run_simplified_bias_plots(
        results_dir=results_dir,
        plots_root=plots_root,
        eval_save_dir=eval_save_dir,
        target_cols=target_cols,
        feature_sets=feature_sets,
        distance_types=[args.distance_type],
        plot_distance_type=args.distance_type,
        no_titles=args.no_titles,
        which_plots=args.plots,
        bias_subdir="lang_bias",
    )

    # Paper-figure split plot: two-panel Meco + OneStop residual-vs-distance
    # figure, rendered for the canonical WP_COEFS_NO_NORM feature set. This is
    # what paper/lang_bias/.../residual_vs_distance_split.pdf symlinks to.
    def _paper_figures():
        for ym in ("residual", "eyescore"):
            for ds in (["Meco"], ["OneStop"]):
                make_paper_split_bias_plot(
                    distance_type=args.distance_type,
                    results_dir=results_dir, plots_root=plots_root,
                    datasets=ds,
                    y_mode=ym,
                )
        # 2x2 grid per dataset: same figure anatomy, but the panels vary the
        # feature set instead of the corpus.
        for ym in ("residual", "eyescore"):
            for ds in ("Meco", "OneStop"):
                make_paper_feature_set_grid(
                    dataset=ds, distance_type=args.distance_type,
                    results_dir=results_dir, plots_root=plots_root,
                    y_mode=ym,
                )

    if args.no_titles:
        with suppress_titles():
            _paper_figures()
    else:
        _paper_figures()
