"""The language-bias figures: paper/lang_bias/<method>/figures/*.pdf.

Two figure families, both drawn by `evaluation/EyeScore/language_bias.py`:

    <stem>_by_feature_set_<variant>.pdf        2x2 grid, one panel per feature set
    <stem>__WP_COEFS_NO_NORM_<variant>.pdf     the single-panel version

with `<stem>` = eyescore_vs_distance or residual_vs_distance, and `<variant>`
one of the four (dataset, proficiency test) pairs the paper reports. The debias
trees differ only in their tables -- these figures read the RAW eye_score -- but
each tree keeps its own copy so a tree's directory is self-contained.
"""
from __future__ import annotations

from pathlib import Path

from src.constants import TestCols
from src.evaluation.EyeScore.language_bias import (
    make_paper_feature_set_grid,
    make_paper_split_bias_plot,
)
from src.run.paper.lang_bias import PAPER_DISTANCE_TYPE, PAPER_TREES

# Filename tag -> (dataset, proficiency test on the x/regression side).
VARIANTS = [
    ("meco", "", "Meco", TestCols.LEXTALE_COL),
    ("meco_composite", "composite", "Meco", TestCols.PROFICIENCY_AGG_COL),
    ("onestop", "onestop", "OneStop", TestCols.LEXTALE_COL),
    ("onestop_michigan", "onestop_michigan", "OneStop", TestCols.MICHIGEN_TEST_COL),
]

# Per-figure label overrides, keyed by the figure's filename stem. The label
# solver places names by geometry alone, so it has no way to know that two names
# it placed equally well are the wrong way round to a reader who knows which
# language sits where. Rather than encode a special case in the plotting code,
# the figure that wants one says so here, beside the figure it belongs to.
# Each entry is a list of (name, name) pairs whose drawn positions are exchanged;
# swap names of similar length, or the longer one will foul its new neighbour.
# Empty: every published figure reads right as the solver places it.
FIGURE_LABEL_SWAPS = {}

# Where a swap cannot say it: (label, reference label, (dx, dy)) puts one name at
# a fixed offset in display px from another, +y up. A reference of None offsets
# the label from where the solver put it. Applied after the swaps, in order, so a
# later rule can build on an earlier one.
FIGURE_LABEL_PLACEMENTS = {
    "residual_vs_distance__WP_COEFS_NO_NORM_": [
        ("Danish", None, (-12, -10)),            # left, and a little down
        ("Norwegian", None, (-20, -22)),         # left and down
        # German follows Dutch along its own line, starting just past where Dutch
        # ends. ("after", gap) measures Dutch at draw time: a fixed px offset
        # cleared it at 17pt and collided the day the label size fitted to 19.
        ("German", "Dutch", ("after", 9)),
        ("German", None, (0, 14)),               # ...and clear of its line
        ("Spanish", None, (-18, -18)),           # down and left
        ("Italian", None, (-18, -8)),            # left, and a little down
    ],
}
# Per-figure separation of mean markers that sit on top of each other. The
# marker is about 9px across and MECO puts Icelandic and Spanish 0.0028 apart on
# an axis spanning 0.26, so their centres land ~11px apart and the two diamonds
# read as one. Values are in distance units, and are smaller than the jitter the
# figure already applies to each language's own points.
FIGURE_MARKER_OFFSETS = {
    "residual_vs_distance__WP_COEFS_NO_NORM_": {
        "Icelandic": -0.0008,
        "Spanish": +0.0008,
        # Serbian and Dutch are 0.0005 apart -- under 2px -- so they need more.
        "Serbian": -0.0015,
        "Dutch": +0.0015,
    },
}
SINGLE_PANEL_FEATURE_SET = "WP_COEFS_NO_NORM"
Y_MODES = {"eyescore": "eyescore_vs_distance", "residual": "residual_vs_distance"}


def published_name(stem: str) -> str:
    """The file name the paper includes a figure under.

    The single-panel MECO pair disagrees on a trailing underscore --
    eyescore_..._NORM.pdf but residual_..._NORM_.pdf -- and the paper's
    \\includegraphics lines use exactly those names, so they are kept as-is.
    """
    if stem == f"eyescore_vs_distance__{SINGLE_PANEL_FEATURE_SET}_":
        return stem.rstrip("_") + ".pdf"
    return stem + ".pdf"


def render_tree(results_dir: Path, out_dir: Path) -> list[Path]:
    """Draw every figure of one tree into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for y_mode, stem in Y_MODES.items():
        for grid_tag, single_tag, dataset, target_col in VARIANTS:
            grid_stem = f"{stem}_by_feature_set_{grid_tag}"
            single_stem = f"{stem}__{SINGLE_PANEL_FEATURE_SET}_{single_tag}"
            grid_out = out_dir / published_name(grid_stem)
            make_paper_feature_set_grid(
                dataset=dataset, target_col=target_col,
                distance_type=PAPER_DISTANCE_TYPE, results_dir=results_dir,
                save_path=grid_out,
                label_swaps=FIGURE_LABEL_SWAPS.get(grid_stem),
                y_mode=y_mode)
            single_out = out_dir / published_name(single_stem)
            make_paper_split_bias_plot(
                feature_set=SINGLE_PANEL_FEATURE_SET, target_col=target_col,
                distance_type=PAPER_DISTANCE_TYPE, results_dir=results_dir,
                datasets=[dataset], y_mode=y_mode,
                label_swaps=FIGURE_LABEL_SWAPS.get(single_stem),
                label_placements=FIGURE_LABEL_PLACEMENTS.get(single_stem),
                marker_offsets=FIGURE_MARKER_OFFSETS.get(single_stem),
                save_path=single_out)
            written += [grid_out, single_out]
    return written


def render(out_root: Path) -> list[Path]:
    written = []
    for method, tree in PAPER_TREES.items():
        written += render_tree(Path("src/methods/EyeScore") / tree,
                               Path(out_root) / "lang_bias" / method / "figures")
    return written
