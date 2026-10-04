"""LaTeX tables of the reliability results (per_item_reliability.csv,
split_half_reliability.csv) and of the per-item evaluation results, written to
results/tables/[<preview>/]. The paper's versions are rendered into
paper/reliability/ by `python -m src.run.paper --only reliability`.
"""
import logging
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.latex_captions import rename_datasets_in_tables
from src.reliability.utils import DEFAULT_PREVIEW, FEATURE_SET_DISPLAY

logger = logging.getLogger(__name__)

# Datasets a table covers. Only OneStop has a preview axis, so a non-Gathering
# run drops MECO entirely rather than repeating the Gathering MECO numbers
# under an information-seeking heading.
DEFAULT_TABLE_DATASETS = ("Meco", "OneStop")
DATASET_DISPLAY = {"Meco": "MECO", "OneStop": "OneStop"}
# Project rule: the "paragraph" aggregation level is always shown as "Passage".
LEVEL_DISPLAY = {"paragraph": "Passage", "article": "Article"}
ITEM_WORD = {"paragraph": "passages", "article": "articles"}
# Column labels and prediction targets of each dataset's block, paper order
# (MECO left of OneStop; LexTALE left of Composite / Michigan).
PARAGRAPH_SECTION_COL_LABELS = {
    "Meco": ["EyeScore", "LexTALE", "Composite"],
    "OneStop": ["EyeScore", "LexTALE", "Michigan"],
}
PARAGRAPH_SECTION_TARGETS = {
    "Meco": ["lextale_score", "proficiency_agg"],
    "OneStop": ["lextale_score", "michtest_score"],
}


# Blocks of the seen/unseen reliability tables: (dataset, p_agg_level, title).
SEEN_UNSEEN_SECTIONS = (
    ("Meco", "paragraph", "MECO Passage"),
    ("OneStop", "paragraph", "OneStop Passage"),
    ("OneStop", "article", "OneStop Article"),
)
# Prediction targets of each dataset's block, in paper order.
SECTION_TARGETS = {
    "Meco": [("lextale_score", "LexTALE"), ("proficiency_agg", "Composite")],
    "OneStop": [("lextale_score", "LexTALE"), ("michtest_score", "Michigan")],
}


def seen_unseen_sections(datasets: Sequence[str]) -> list:
    """The (dataset, level, title) blocks covered by `datasets`."""
    return [sec for sec in SEEN_UNSEEN_SECTIONS if sec[0] in datasets]


def section_titles(sections: Sequence[tuple]) -> list:
    """Block headings for `sections`, dropping the level word when every block
    sits at the same level.

    Passages are the default notion of a test item, so a passage-only table
    heads its blocks with the corpus alone -- "MECO", not "MECO Passage" --
    and names the item level once, in the caption. A table that mixes levels
    keeps the word, which is the only thing telling its two OneStop blocks
    apart.
    """
    if len({level for _, level, _ in sections}) == 1:
        return [DATASET_DISPLAY[dataset] for dataset, _, _ in sections]
    return [title for _, _, title in sections]


def sections_level(sections: Sequence[tuple]) -> Optional[str]:
    """The one level `sections` covers, or None when it spans several."""
    levels = list(dict.fromkeys(level for _, level, _ in sections))
    return levels[0] if len(levels) == 1 else None


def preview_datasets(preview: str) -> Tuple[str, ...]:
    """Datasets the tables of `preview` cover (non-Gathering: OneStop only)."""
    return DEFAULT_TABLE_DATASETS if preview == DEFAULT_PREVIEW else ("OneStop",)


# Captions/labels of a non-default preview subtree say so explicitly, so a
# Hunting table is not mistaken for (or clashes with) its Gathering twin.
PREVIEW_CAPTION_NOTE = {
    "Hunting": "OneStopL2 results are for the information seeking regime.",
}
PREVIEW_LABEL_SUFFIX = {
    "Hunting": "-information-seeking",
}


def _caption_span(text: str, start: int) -> Optional[Tuple[int, int]]:
    """(open_brace, close_brace) of the ``\\caption`` argument at ``start``."""
    open_idx = text.find("{", start)
    if open_idx == -1:
        return None
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return open_idx, i
    return None


def annotate_preview(output_dir: Path, preview: str) -> None:
    """Name the regime in every caption and label under ``output_dir``.

    Appends ``PREVIEW_CAPTION_NOTE`` to each ``\\caption{...}`` (brace-matched,
    so nested groups survive) and suffixes each ``\\label{...}`` with
    ``PREVIEW_LABEL_SUFFIX``, letting the preview's tables sit in the same
    document as the Gathering ones without duplicate labels. Idempotent: a
    file that already carries the note/suffix is left alone.
    """
    note = PREVIEW_CAPTION_NOTE.get(preview)
    suffix = PREVIEW_LABEL_SUFFIX.get(preview)
    if not note and not suffix:
        logger.warning(
            "No caption note/label suffix configured for preview=%s; tables "
            "are written without a regime marker.", preview)
        return
    for path in sorted(output_dir.glob("*.tex")):
        text = path.read_text()
        original = text
        if note:
            out, pos = [], 0
            while True:
                idx = text.find("\\caption", pos)
                if idx == -1:
                    break
                span = _caption_span(text, idx)
                if span is None:
                    break
                open_idx, close_idx = span
                body = text[open_idx + 1:close_idx]
                out.append(text[pos:open_idx + 1])
                out.append(body if note in body
                           else body.rstrip() + " " + note)
                pos = close_idx
            out.append(text[pos:])
            text = "".join(out)
        if suffix:
            text = re.sub(
                r"\\label\{([^{}]*)\}",
                lambda m: "\\label{" + m.group(1) + (
                    "" if m.group(1).endswith(suffix) else suffix) + "}",
                text)
        if text != original:
            path.write_text(text)
    logger.info("Marked tables in %s as preview=%s", output_dir, preview)


POOL_DISPLAY = {
    "all": "All",
    "seen": "Fixed",
    "unseen": "Any",
}

# The same regimes named in full, for the tables where a regime heads a whole
# block of columns and there is room to say what it is. The bare POOL_DISPLAY
# names stay wherever a regime labels a single narrow column, which is most of
# them -- "Any Text Regime" over one column of correlations would set the
# column width by itself.
POOL_HEADING = {
    "seen": "Fixed Text Regime",
    "unseen": "Any Text Regime",
}


# ─── the rule between blocks in a heading row ─────────────────────────────────
# Drawn on the RIGHT edge of the block before the boundary -- "c|" on the
# preceding \multicolumn -- never on the left edge of the block after it.
#
# The two are not equivalent. A spanned
# heading is a \multicolumn, and \multicolumn replaces the preamble of the
# columns it covers, so the boundary "|" from the column spec is suppressed in
# that row and has to be written back in by hand. Written as "|c" on the
# following block it comes out about a \tabcolsep to the side of the body
# rules below -- not one line reaching higher, two lines that do not meet.
# Written as "c|" on the preceding block it lands on the boundary itself and
# the heading rule and the body rule are one line. This is how the paper's
# main results table draws it, which is the version to copy from.
BLOCK_RULE = "|"


def header_block_cells(entries, rule: str = BLOCK_RULE) -> list:
    r"""``\multicolumn`` cells for one heading row of blocks.

    ``entries`` is (span, text) per block. ``rule`` closes every block but the
    last, which has no boundary to its right -- see BLOCK_RULE above for why
    it goes on that side and not the next block's left.
    """
    last = len(entries) - 1
    return [f"\\multicolumn{{{span}}}{{c{'' if i == last else rule}}}{{{text}}}"
            for i, (span, text) in enumerate(entries)]


# ─── bolding the best cell in every column ────────────────────────────────────
# Applied to the finished LaTeX rather than inside each table builder: they
# assemble their rows in different ways, but the result
# is always the same shape -- some label columns, then one numeric cell per
# metric column -- so one pass over the emitted table covers all of them and
# cannot drift out of sync with a builder that changes.

# A metric column whose header names one of these is better when *smaller*.
LOWER_IS_BETTER_METRICS = ("SEM", "MAE", "RMSE")
# Columns that hold a count rather than a metric. Nothing in them is "best" --
# bolding the largest sample size says the biggest cohort won something.
COUNT_COLUMN_HEADERS = ("$n$", "$k$", "n", "k", "N", "K")
# Rows that are reference lines rather than competitors: never bolded, never
# counted when looking for the column's best.
BOLD_EXCLUDED_ROW_LABELS = ("Avg. Train",)

_MULTICOL_RE = re.compile(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*$", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _split_row(line: str) -> Optional[list]:
    """Cells of a LaTeX row, or None when the line is not a row."""
    body = line.strip()
    if not body.endswith("\\\\"):
        return None
    body = body[:-2]
    if "&" not in body:
        return None
    return body.split("&")


def _expand_header(cells: list) -> list:
    """Header cells with each \\multicolumn{n} repeated over the n columns."""
    out = []
    for cell in cells:
        m = _MULTICOL_RE.match(cell.strip())
        if m:
            out.extend([m.group(2)] * int(m.group(1)))
        else:
            out.append(cell.strip())
    return out


def _cell_number(cell: str) -> Optional[float]:
    """The value a metric cell displays, or None when it holds no number.

    Everything decorative is stripped first -- \\phantom padding (whose star
    count would otherwise contribute digits), significance markers, math
    delimiters -- so "$0.51^{**}\\phantom{^{*}}$" reads as 0.51 and a label
    cell such as "pooled ($n$=151, $k$=54)" is rejected by the caller instead
    of being mistaken for a metric.
    """
    text = re.sub(r"\\phantom\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", cell)
    text = re.sub(r"\^\{[^{}]*\}", "", text)
    text = text.replace("$", "").replace("\\,", "").strip()
    text = re.sub(r"\\(?:text|math)bf\{([^{}]*)\}", r"\1", text)
    m = _NUMBER_RE.fullmatch(text.strip())
    return float(m.group(0)) if m else None


def _bold_cell(cell: str) -> str:
    """Bold the number inside a cell, in math mode when the cell is math."""
    if "\\textbf{" in cell or "\\mathbf{" in cell:
        return cell
    macro = "\\mathbf" if "$" in cell else "\\textbf"
    return _NUMBER_RE.sub(lambda m: macro + "{" + m.group(0) + "}", cell, count=1)


def bold_best_per_column(latex: str) -> str:
    """Bold the best value in each metric column of a finished table.

    The best is the largest value, or the smallest for the columns whose header
    names an error metric (``LOWER_IS_BETTER_METRICS``). Ties are settled on the
    *printed* value, so two cells that both show 0.99 are both bolded even when
    their raw values differ further down -- bolding one of a visually identical
    pair reads as a typo.

    Label columns (the leading bold headers, e.g. Features / Readers) are never
    bolded: their cells can contain numbers -- "pooled ($n$=151, $k$=54)" -- that
    have nothing to do with the metric. Tables split into blocks by a full-width
    ``\\multicolumn`` heading are bolded within each block, since a block is a
    different dataset or target and its numbers are not comparable across the
    heading.

    Returns the LaTeX unchanged when no metric header row is found, so a table
    with a layout this pass does not understand is left alone rather than
    mangled.
    """
    lines = latex.split("\n")
    # The metric header is the row whose first cell is the bold row-label
    # heading (Features / Part / ...) and which the body rule follows straight
    # after. Matching on the rule rather than on the wording keeps this working
    # for every table here without a list of accepted headings.
    header_idx = None
    for i, ln in enumerate(lines[:-1]):
        cells = _split_row(ln)
        if cells is None or not cells[0].strip().startswith("\\textbf{"):
            continue
        if re.match(r"\s*\\(midrule|hline|cline|cmidrule)", lines[i + 1]):
            header_idx = i
            break
    if header_idx is None:
        return latex

    header = _expand_header(_split_row(lines[header_idx]))
    n_label_cols = 0
    for cell in header:
        if cell == "" or cell.startswith("\\textbf{") or cell.startswith("\\multirow"):
            n_label_cols += 1
        else:
            break
    lower = [any(k in cell for k in LOWER_IS_BETTER_METRICS) for cell in header]
    is_count = [cell.strip() in COUNT_COLUMN_HEADERS for cell in header]

    # (line index, cell index) -> value, per block, per column.
    blocks, current = [], {}

    def close_block():
        if current:
            blocks.append(current.copy())
            current.clear()

    for i in range(header_idx + 1, len(lines)):
        line = lines[i]
        if "\\end{tabular}" in line:
            break
        cells = _split_row(line)
        if cells is None:
            # A full-width heading row carries no "&" and starts a new block.
            if "\\multicolumn" in line and line.strip().endswith("\\\\"):
                close_block()
            continue
        if len(cells) == 1:
            close_block()
            continue
        label = re.sub(r"\\[a-zA-Z]+|[{}$]", "", cells[0]).strip()
        if label in BOLD_EXCLUDED_ROW_LABELS:
            continue
        for c in range(n_label_cols, len(cells)):
            if c < len(is_count) and is_count[c]:
                continue
            val = _cell_number(cells[c])
            if val is not None:
                current.setdefault(c, []).append((i, val))
    close_block()

    row_cells = {}
    for block in blocks:
        for c, entries in block.items():
            if not entries:
                continue
            take = min if (c < len(lower) and lower[c]) else max
            best = take(v for _, v in entries)
            for i, v in entries:
                # Compare on the printed string: cells that round to the same
                # number are all best, whatever their raw values.
                if f"{v:.10g}" != f"{best:.10g}":
                    continue
                cells = row_cells.setdefault(i, _split_row(lines[i]))
                cells[c] = _bold_cell(cells[c])

    for i, cells in row_cells.items():
        lines[i] = "&".join(cells) + "\\\\"
    return "\n".join(lines)


def write_table(save_path: Path, latex: str, bold_best: bool = True) -> None:
    """Write a finished table, naming MECO L2 and bolding the best cell.

    ``bold_best=False`` for the reliability tables (alpha, SEM, split-half,
    ICC): they report how stable each measure is, not which measure wins, so
    marking the largest coefficient invites a comparison the table is not
    making. The evaluation tables in this module keep the bolding -- there the
    columns really are a contest between feature sets.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if bold_best:
        latex = bold_best_per_column(latex)
    save_path.write_text(rename_datasets_in_tables(latex))


# Whether a table fits a text column or has to span the page is decided by how
# many metric columns it carries, not by which generator built it: the same
# generator produces a three-block table for Gathering and a two-block one for
# a preview that drops MECO, and only the first needs the full width. Eight
# metric columns is where \resizebox starts shrinking the digits past reading
# size in a two-column layout.
MAX_ONE_COLUMN_METRIC_COLS = 8


def table_env(n_metric_cols: int) -> tuple:
    """(environment, width macro) for a table with this many metric columns."""
    if n_metric_cols <= MAX_ONE_COLUMN_METRIC_COLS:
        return "table", "\\columnwidth"
    return "table*", "\\textwidth"


def should_add_cline_after_feature(feature_code: str, feature_order: list) -> bool:
    """Whether a rule follows `feature_code`: WPM stands alone, then Avg. Fix. /
    S-Clusters / WP-Coefs form one group and Transitions / Word Fix. another.
    Never after the last row.
    """
    current_idx = feature_order.index(feature_code)
    is_last = current_idx == len(feature_order) - 1

    # Group definitions
    wpm_group = ["READING_SPEED"]
    mid_group = ["FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM"]
    end_group = ["TRANSITIONS", "WFC"]

    if is_last:
        return False

    next_feature = feature_order[current_idx + 1]
    current_in_wpm = feature_code in wpm_group
    current_in_mid = feature_code in mid_group
    current_in_end = feature_code in end_group
    next_in_wpm = next_feature in wpm_group
    next_in_mid = next_feature in mid_group
    next_in_end = next_feature in end_group

    # Add cline when transitioning to different group
    if current_in_wpm and not next_in_wpm:
        return True
    if current_in_mid and not next_in_mid:
        return True
    if current_in_end and not next_in_end:
        return True

    return False


def format_value(val: float, missing: str = "??") -> str:
    """Format a correlation or alpha value for display."""
    if pd.isna(val):
        return missing
    return f"{val:.2f}" if -1 <= val <= 1 else missing


def format_sem(val: float, missing: str = "??") -> str:
    """Format a standard error of measurement for display.

    Deliberately not format_value: SEM is on the observed-score scale
    (SD x sqrt(1 - reliability)), not a correlation, so it is not bounded by 1 —
    observed values run to ~2.2 — and format_value's [-1, 1] guard would render
    those as "??".
    """
    if pd.isna(val):
        return missing
    return f"{val:.2f}"


def sem_of(match_df: pd.DataFrame, sem_col: str = "sem_cronbach") -> float:
    """SEM for one matched row, or NaN when the row or the column is absent.

    The column check matters: a per_item_reliability.csv written before the SEM
    columns existed still renders, with "??" in the SEM cells rather than a
    KeyError halfway through table generation.
    """
    if len(match_df) == 0 or sem_col not in match_df.columns:
        return np.nan
    return match_df.iloc[0][sem_col]


def metric_cells(match_df: pd.DataFrame, value_col: str, sem_col: str = None) -> list:
    """Cells for one table position: the metric, plus its SEM when sem_col is set.

    A missing sem_col in the frame yields "??" rather than raising, so a table
    built from a per_item_reliability.csv written before the SEM columns existed
    still renders (with the SEM sub-column flagged as missing).
    """
    val = match_df.iloc[0][value_col] if len(match_df) > 0 else np.nan
    cells = [format_value(val)]
    if sem_col:
        sem = (match_df.iloc[0][sem_col]
               if len(match_df) > 0 and sem_col in match_df.columns else np.nan)
        cells.append(format_sem(sem))
    return cells


def generate_article_onestop_both_pools_table(
    df_results: pd.DataFrame,
    value_col: str,
    caption_metric: str,
    save_path: Path,
    metric_name: str = "metric",
) -> None:
    """Generate OneStop article-level table with both Fixed and Any regimes side by side.

    Table structure: Michigan and LexTALE, each with Fixed and Any subheaders.
    """
    # Filter for article level, OneStop only
    article_onestop = df_results[
        (df_results["p_agg_level"] == "article") &
        (df_results["dataset"] == "OneStop")
    ].copy()

    if article_onestop.empty:
        logger.warning("No data for article level, OneStop")
        return

    # Separate eyescore and predictions
    eyescore_data = article_onestop[article_onestop["source"] == "eyescore"]
    predictions_data = article_onestop[article_onestop["source"] == "predictions"]

    if eyescore_data.empty and predictions_data.empty:
        logger.warning("No eyescore or predictions data for OneStop article")
        return

    # Feature order - same as unseen (no additional features at article level)
    feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM"]

    # Targets for OneStop: LexTALE, Michigan
    targets = ["lextale_score", "michtest_score"]

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{l|c|c|c||c|c|c}",
    ]

    # Header row 1: Regime names (Fixed and Any)
    lines.append(" \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{\\textbf{Fixed}} & \\multicolumn{3}{c}{\\textbf{Any}} \\\\")
    lines.append("\\cline{2-7}")

    # Header row 2: Column labels
    lines.append("\\textbf{Features} & EyeScore & LexTALE & Michigan & EyeScore & LexTALE & Michigan \\\\")
    lines.append("\\hline")

    # Data rows for each feature
    for feature_code in feature_order:
        feature_name = FEATURE_SET_DISPLAY[feature_code]
        row = [feature_name]

        # For each pool (Fixed=seen, Any=unseen)
        for pool in ["seen", "unseen"]:
            # EyeScore
            eye_row = eyescore_data[
                (eyescore_data["feature_set"] == feature_code) &
                (eyescore_data["pool"] == pool)
            ]
            val_eye = eye_row.iloc[0][value_col] if len(eye_row) > 0 else np.nan
            row.append(format_value(val_eye))

            # Predictions for each target (Michigan, LexTALE)
            for target_code in targets:
                pred_row = predictions_data[
                    (predictions_data["feature_set"] == feature_code) &
                    (predictions_data["target"] == target_code) &
                    (predictions_data["pool"] == pool)
                ]
                val_pred = pred_row.iloc[0][value_col] if len(pred_row) > 0 else np.nan
                row.append(format_value(val_pred))

        lines.append(" & ".join(row) + " \\\\")
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    lines.extend([
        "\\end{tabular}",
        "}",
        f"\\caption{{{caption_metric}}}",
        f"\\label{{tab:{metric_name}-article}}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex, bold_best=False)
    logger.info(f"Saved table: {save_path}")


def generate_paragraph_pool_combined_table(
    df_results: pd.DataFrame,
    pool: str,
    value_col: str,
    caption_metric: str,
    save_path: Path,
    metric_name: str = "metric",
    sem_col: str = None,
    value_label: str = "$\\alpha$",
    datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
) -> None:
    """Generate combined EyeScore and Predictions table for paragraph level, specific pool.

    For unseen: features are Reading Speed, Avg.Fix.Metrics, S-Clusters, WP-Coefs.
    For seen: includes additional features Transitions and Word Fixations.

    With `sem_col`, every one of the six value columns splits into a
    `value_label` / SEM pair (12 value columns, three header rows). Without it
    each value is one column, as in the split-half tables, whose CSV carries
    no SEM.

    `datasets` selects the dataset blocks (and therefore the column spec and
    both header rows); a single-dataset list renders that dataset alone.
    """
    # Filter for paragraph level, specific pool
    paragraph_pool = df_results[
        (df_results["p_agg_level"] == "paragraph") &
        (df_results["pool"] == pool)
    ].copy()

    if paragraph_pool.empty:
        logger.warning(f"No data for paragraph level, {pool} pool")
        return

    # Separate eyescore and predictions
    eyescore_data = paragraph_pool[paragraph_pool["source"] == "eyescore"]
    predictions_data = paragraph_pool[paragraph_pool["source"] == "predictions"]

    if eyescore_data.empty and predictions_data.empty:
        logger.warning("No eyescore or predictions data for this pool")
        return

    # The per-text sets (Transitions, Word Fix.) exist in the Fixed regime only.
    if pool == "seen":
        feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]
    else:  # unseen
        feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM"]
    feature_display = FEATURE_SET_DISPLAY

    # Column labels in paper order (MECO left of OneStop; LexTALE left of
    # Composite / Michigan), shared by the plain and the paired-SEM layouts.
    sections = [(ds, DATASET_DISPLAY[ds],
                 PARAGRAPH_SECTION_COL_LABELS[ds],
                 PARAGRAPH_SECTION_TARGETS[ds])
                for ds in datasets]
    col_labels = [label for _, _, labels, _ in sections for label in labels]
    last_col = 1 + len(col_labels) * (2 if sem_col else 1)   # for \cline{2-N}
    cols_per_label = "cc" if sem_col else "c"
    # Single "|" between the columns of one dataset, "||" between datasets.
    col_spec = "l|" + "||".join(
        "|".join([cols_per_label] * len(labels)) for _, _, labels, _ in sections)

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{",
        f"\\begin{{tabular}}{{{col_spec}}}",
    ]

    # Header row 1: Dataset names
    lines.append(
        " \\multicolumn{1}{c}{} & "
        + " & ".join(
            f"\\multicolumn{{{len(labels) * (2 if sem_col else 1)}}}{{c}}"
            f"{{\\textbf{{{display}}}}}"
            for _, display, labels, _ in sections)
        + " \\\\"
    )
    lines.append(f"\\cline{{2-{last_col}}}")

    if sem_col:
        # Row 2: source/target labels, each spanning its value+SEM pair.
        # Row 3: the pair itself. \textbf{Features} goes on the last header row
        # so it sits directly above the feature names.
        # "||" closes each dataset block except the last, which needs no rule.
        seps = [
            "c" + ("" if (i == len(sections) - 1 and j == len(labels) - 1)
                   else "||" if j == len(labels) - 1 else "|")
            for i, (_, _, labels, _) in enumerate(sections)
            for j in range(len(labels))
        ]
        lines.append(
            " \\multicolumn{1}{c}{} & "
            + " & ".join(f"\\multicolumn{{2}}{{{sep}}}{{{label}}}"
                         for label, sep in zip(col_labels, seps))
            + " \\\\"
        )
        lines.append(f"\\cline{{2-{last_col}}}")
        lines.append("\\textbf{Features} & "
                     + " & ".join([value_label, "SEM"] * len(col_labels)) + " \\\\")
    else:
        # Header row 2: Column labels
        lines.append("\\textbf{Features} & " + " & ".join(col_labels) + " \\\\")
    lines.append("\\hline")

    # Data rows for each feature
    for feature_code in feature_order:
        feature_name = feature_display.get(feature_code, feature_code)
        row = [feature_name]

        # Per dataset: EyeScore first, then that dataset's targets.
        for dataset, _, _, target_codes in sections:
            eye_row = eyescore_data[
                (eyescore_data["dataset"] == dataset) &
                (eyescore_data["feature_set"] == feature_code)
            ]
            row.extend(metric_cells(eye_row, value_col, sem_col))

            for target_code in target_codes:
                pred_row = predictions_data[
                    (predictions_data["dataset"] == dataset) &
                    (predictions_data["feature_set"] == feature_code) &
                    (predictions_data["target"] == target_code)
                ]
                row.extend(metric_cells(pred_row, value_col, sem_col))

        lines.append(" & ".join(row) + " \\\\")
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    regime = "any" if pool == "unseen" else "fixed"
    lines.extend([
        "\\end{tabular}",
        "}",
        f"\\caption{{{caption_metric}}}",
        f"\\label{{tab:{metric_name}-{regime}-paragraph}}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex, bold_best=False)
    logger.info(f"Saved table: {save_path}")


def generate_combined_reliability_table(
    cronbach_df: pd.DataFrame,
    split_half_df: pd.DataFrame,
    save_path: Path,
    datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
) -> None:
    """Generate combined table with Cronbach's alpha and split-half reliability.

    Each cell shows: cronbach / split_half (both with .2f format)

    `datasets` selects the dataset blocks, and with them the column spec and
    the two header rows.
    """
    # Filter for paragraph level, unseen pool
    cronbach_para = cronbach_df[
        (cronbach_df["p_agg_level"] == "paragraph") &
        (cronbach_df["pool"] == "unseen")
    ].copy()
    split_half_para = split_half_df[
        (split_half_df["p_agg_level"] == "paragraph") &
        (split_half_df["pool"] == "unseen")
    ].copy()

    if cronbach_para.empty or split_half_para.empty:
        logger.warning("No data for combined reliability paragraph unseen")
        return

    # Separate eyescore and predictions
    cronbach_eye = cronbach_para[cronbach_para["source"] == "eyescore"]
    cronbach_pred = cronbach_para[cronbach_para["source"] == "predictions"]
    split_half_eye = split_half_para[split_half_para["source"] == "eyescore"]
    split_half_pred = split_half_para[split_half_para["source"] == "predictions"]

    feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM"]
    feature_display = FEATURE_SET_DISPLAY

    sections = [(ds, DATASET_DISPLAY[ds],
                 PARAGRAPH_SECTION_COL_LABELS[ds],
                 PARAGRAPH_SECTION_TARGETS[ds])
                for ds in datasets]
    col_labels = [label for _, _, labels, _ in sections for label in labels]
    last_col = 1 + len(col_labels)
    col_spec = "l|" + "||".join(
        "|".join(["c"] * len(labels)) for _, _, labels, _ in sections)

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{",
        f"\\begin{{tabular}}{{{col_spec}}}",
    ]

    # Header row 1: Dataset names
    lines.append(
        " \\multicolumn{1}{c}{} & "
        + " & ".join(f"\\multicolumn{{{len(labels)}}}{{c}}{{\\textbf{{{display}}}}}"
                     for _, display, labels, _ in sections)
        + " \\\\")
    lines.append(f"\\cline{{2-{last_col}}}")

    # Header row 2: Column labels
    lines.append("\\textbf{Features} & " + " & ".join(col_labels) + " \\\\")
    lines.append("\\hline")

    def _pair_cell(cron_rows, split_rows) -> str:
        """``alpha/split-half`` for one cell, ``??/??`` if either is missing."""
        cronbach_val = split_half_val = None
        if len(cron_rows) > 0:
            val = cron_rows.iloc[0]["cronbachs_alpha"]
            if pd.notna(val):
                cronbach_val = val
        if len(split_rows) > 0:
            val = split_rows.iloc[0]["mean_correlation"]
            if pd.notna(val):
                split_half_val = val
        if cronbach_val is None or split_half_val is None:
            return "??/??"
        return f"{cronbach_val:.2f}/{split_half_val:.2f}"

    # Data rows for each feature
    for feature_code in feature_order:
        feature_name = feature_display.get(feature_code, feature_code)
        row = [feature_name]

        # Per dataset: EyeScore first, then that dataset's targets.
        for dataset, _, _, target_codes in sections:
            row.append(_pair_cell(
                cronbach_eye[(cronbach_eye["dataset"] == dataset)
                             & (cronbach_eye["feature_set"] == feature_code)],
                split_half_eye[(split_half_eye["dataset"] == dataset)
                               & (split_half_eye["feature_set"] == feature_code)]))

            for target_code in target_codes:
                row.append(_pair_cell(
                    cronbach_pred[(cronbach_pred["dataset"] == dataset)
                                  & (cronbach_pred["feature_set"] == feature_code)
                                  & (cronbach_pred["target"] == target_code)],
                    split_half_pred[(split_half_pred["dataset"] == dataset)
                                    & (split_half_pred["feature_set"] == feature_code)
                                    & (split_half_pred["target"] == target_code)]))

        lines.append(" & ".join(row) + " \\\\")
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    lines.extend([
        "\\end{tabular}",
        "}",
        "\\caption{Combined internal consistency reliability for EyeScore and prediction models. Each cell shows Cronbach's $\\alpha$ (left) / Split-Half reliability (right) over per-passage aggregation for the Any regime.}",
        "\\label{tab:combined-reliability-any-paragraph}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex, bold_best=False)
    logger.info(f"Saved table: {save_path}")


# --------------------------------------------------------------------------
# Significance stars vs the WPM baseline row.
#
# Mirrors the main prediction tables (evaluation/predictions/tables.py:
# generate_bootstrap_combined_latex): the paired-bootstrap difference test of
# each feature set against WPM, within the same (dataset, preview,
# version_path, target, method) cell. The evaluation_* tables below are the
# per-item counterpart of those fully_agg tables, so they read the same
# pairwise CSV and use the same star thresholds and directional rule.
#
# The rows are produced by any evaluation pass that computes baseline pairs,
# e.g. `python -m src.run.predictions --stage evaluation --agg-types per_item_agg
# --pairwise-baseline-only`. Missing rows simply mean no stars.
# --------------------------------------------------------------------------
PREDICTIONS_PAIRWISE_WPM_PATH = Path(
    "src/evaluation/predictions/results/pairwise_significance_wpm.csv")
PREDICTIONS_PAIRWISE_PATH = Path(
    "src/evaluation/predictions/results/pairwise_significance.csv")
# Baseline row every other feature set is tested against (WPM).
STAR_BASELINE_FEATURE_SET = "READING_SPEED"
# These tables report the Ridge results; pin the pairs to the same model so a
# later model's rows can never be picked up instead.
STAR_MODEL_NAME = "Ridge_Classifier"
# The evaluation tables below report this model's cells. The featureless mean
# baseline runs under READING_SPEED too (it ignores features, but the
# orchestrator still needs a column), so a (target, feature_set, method) lookup
# not pinned to one model would take whichever of the two came first.
EVAL_TABLE_MODEL = STAR_MODEL_NAME
# The baseline is a reference line, not a competitor: no stars, no bolding.
BASELINE_MODEL = "Average"
BASELINE_LABEL = "Avg. Train"
# The rule that fences the reference row off from the feature sets. Emitted
# under our own name, with \addlinespace on both sides: without the padding the
# rule sits hard against the two rows it separates and reads as a smudge rather
# than a divider. \hdashline comes from arydshln, at half its default dash
# pitch (\resizebox shrinks the whole table, and 4pt dashes come out sparse);
# a document without arydshln falls back to a solid \midrule. Same definition
# as the main prediction tables use, so the two families look alike.
BASELINE_RULE_NAME = "\\meanbaselinerule"
BASELINE_RULE_PAD = "2pt"
BASELINE_RULE_DASH = "2pt"
BASELINE_RULE_DEF = (
    "\\makeatletter\n"
    "\\@ifundefined{hdashline}\n"
    "  {\\providecommand{" + BASELINE_RULE_NAME + "}{\\addlinespace["
    + BASELINE_RULE_PAD + "]\\midrule\\addlinespace[" + BASELINE_RULE_PAD + "]}}\n"
    "  {\\@ifundefined{dashlinedash}{}{\\setlength\\dashlinedash{"
    + BASELINE_RULE_DASH + "}\\setlength\\dashlinegap{" + BASELINE_RULE_DASH + "}}%\n"
    "   \\providecommand{" + BASELINE_RULE_NAME + "}{\\addlinespace["
    + BASELINE_RULE_PAD + "]\\hdashline\\addlinespace[" + BASELINE_RULE_PAD + "]}}\n"
    "\\makeatother")


# The evaluation tables report the same estimates the paper's main tables do:
# the bootstrap mean over the resamples, not the single point estimate from the
# one observed sample. The raw column is the fallback for a row the bootstrap
# did not reach (an evaluation run with --no-bootstrap-ci leaves the *_boot
# columns NaN).
BOOTSTRAP_METRIC_COLS = {"pearson_r": "pearson_r_boot", "mae": "mae_boot"}


def eval_metric(row, col: str):
    """``col`` for one evaluation row, preferring its bootstrap mean."""
    boot = BOOTSTRAP_METRIC_COLS.get(col)
    if boot is not None and boot in row.index:
        val = row[boot]
        if pd.notna(val):
            return val
    return row[col]


def baseline_cells(df, target: str, method: str, decimals: int = 2,
                   max_stars: tuple = (0, 0)) -> list:
    """[$R^2$, MAE] for the mean baseline at one table position, or dashes.

    ``df`` is the slice already narrowed to one dataset/preview/level. Missing
    rows give "-" rather than "??": the baseline not having been run for a
    slice is a gap in coverage, not a suspect number.
    """
    rows = df[(df["model_name"] == BASELINE_MODEL)
              & (df["target_col"] == target)
              & (df["method"] == method)]
    if rows.empty:
        return ["-", "-"]
    out = []
    for col, n_max in zip(("r2", "mae"), max_stars):
        val = (eval_metric(rows.iloc[0], col) if col in BOOTSTRAP_METRIC_COLS
               else rows.iloc[0].get(col))
        if val is None or pd.isna(val):
            out.append("-")
            continue
        # An R^2 a hair below zero is this baseline's expected value; printing
        # it as "-0.00" reads as a typo.
        val = 0.0 if round(float(val), decimals) == 0 else float(val)
        # The row carries no stars, but its cells sit in columns whose other
        # rows do: without the same phantom the column's widest marker leaves,
        # these numbers centre differently from every row beneath them.
        pad = "\\phantom{^{" + "*" * n_max + "}}" if n_max else ""
        out.append(f"${val:.{decimals}f}{pad}$")
    return out


# Paired-bootstrap p-value columns — the only test behind a reported result.
STAR_R_P_COL = "boot_diff_pearson_p"
STAR_MAE_P_COL = "boot_diff_mae_p"

_PAIRWISE_CACHE: dict = {}


def _load_predictions_pairwise():
    """Read the WPM-subset pairwise CSV once, falling back to the all-pairs one."""
    if "df" in _PAIRWISE_CACHE:
        return _PAIRWISE_CACHE["df"]
    path = PREDICTIONS_PAIRWISE_WPM_PATH
    if not path.exists():
        path = PREDICTIONS_PAIRWISE_PATH
    df = None
    if path.exists():
        df = pd.read_csv(path)
    else:
        logger.warning(
            "Pairwise significance CSV not found (%s) — stars vs WPM omitted "
            "from the predictions evaluation tables",
            PREDICTIONS_PAIRWISE_WPM_PATH)
    _PAIRWISE_CACHE["df"] = df
    return df


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


def _directional_stars(p, value_fs, value_base, higher_is_better: bool) -> int:
    """Stars for a two-sided test, suppressed unless the feature set wins.

    The pairwise p-values are two-sided, so a feature set that is significantly
    *worse* than WPM carries the same p as one that is significantly better.
    Readers take a star to mean "beats the baseline", so only a favourable
    difference (higher r, lower MAE) is marked.
    """
    if p is None or pd.isna(p) or pd.isna(value_fs) or pd.isna(value_base):
        return 0
    wins = value_fs > value_base if higher_is_better else value_fs < value_base
    return _p_to_n_stars(p) if wins else 0


def wpm_star_map(dataset: str, preview: str, version_path: str,
                 model_name: str = STAR_MODEL_NAME) -> dict:
    """(target_col, method, feature_set) -> (r_stars, mae_stars) vs WPM.

    Empty when the pairwise CSV is missing or holds no rows for this slice.
    """
    df = _load_predictions_pairwise()
    if df is None or df.empty:
        return {}

    sub = df[(df["dataset"] == dataset)
             & (df["preview"] == preview)
             & (df["version_path"] == version_path)
             & (df["model_name_a"] == model_name)
             & (df["model_name_b"] == model_name)
             & ((df["feature_set_a"] == STAR_BASELINE_FEATURE_SET)
                | (df["feature_set_b"] == STAR_BASELINE_FEATURE_SET))]
    if sub.empty:
        logger.warning(
            "No pairwise rows for %s/%s/%s (model=%s) — stars vs WPM omitted",
            dataset, preview, version_path, model_name)
        return {}

    out = {}
    for _, row in sub.iterrows():
        # Each pair is stored once, in either a/b order: orient it onto
        # (feature set, baseline) before judging which way the difference runs.
        a_is_baseline = row["feature_set_a"] == STAR_BASELINE_FEATURE_SET
        feature_set = (row["feature_set_b"] if a_is_baseline
                       else row["feature_set_a"])
        if feature_set == STAR_BASELINE_FEATURE_SET:
            continue  # WPM against itself, if it ever appears

        def _oriented(col_a: str, col_b: str):
            return ((row.get(col_b), row.get(col_a)) if a_is_baseline
                    else (row.get(col_a), row.get(col_b)))

        r_fs, r_base = _oriented("pearson_r_a", "pearson_r_b")
        mae_fs, mae_base = _oriented("mae_a", "mae_b")
        out[(row["target_col"], row["method"], feature_set)] = (
            _directional_stars(row.get(STAR_R_P_COL), r_fs, r_base,
                               higher_is_better=True),
            _directional_stars(row.get(STAR_MAE_P_COL), mae_fs, mae_base,
                               higher_is_better=False),
        )
    return out


def _column_max_stars(star_map: dict, target: str, method: str) -> tuple:
    """Most stars any feature set earns in one (target, method) column.

    Used to \\phantom-pad the shorter cells so the decimal points stay aligned,
    exactly as the main bootstrap tables do.
    """
    r_max = max((v[0] for k, v in star_map.items()
                 if k[0] == target and k[1] == method), default=0)
    mae_max = max((v[1] for k, v in star_map.items()
                   if k[0] == target and k[1] == method), default=0)
    return r_max, mae_max


def _starred_cell(value, n_stars: int, max_stars: int, decimals: int = 2) -> str:
    """Format one numeric cell with its stars and alignment padding."""
    if value is None or pd.isna(value):
        return "??"
    body = f"{value:.{decimals}f}"
    star_str = "^{" + "*" * n_stars + "}" if n_stars else ""
    deficit = max_stars - n_stars
    pad = "\\phantom{^{" + "*" * deficit + "}}" if deficit > 0 else ""
    return f"${body}{star_str}{pad}$"


def generate_evaluation_eyescore_combined(save_path: Path,
                                          preview: str = DEFAULT_PREVIEW,
                                          datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
                                          levels: Optional[Sequence[str]] = None,
                                          delta: bool = False) -> None:
    r"""Generate combined EyeScore evaluation table with MECO paragraph, OneStop paragraph, and OneStop article.

    Structure: MECO Para (Composite, LexTALE) | OneStop Para (Michigan, LexTALE) | OneStop Article (Michigan, LexTALE)
    Each target shows Fixed and Any regimes with Pearson r values.

    ``preview`` selects the OneStop preview rows (the EyeScore CSV spells them
    capitalised, e.g. "Gathering"/"Hunting"); MECO has no preview axis and is
    always read from preview="All". ``datasets`` drops the blocks of any
    dataset it leaves out.

    ``levels`` keeps only the aggregation levels it names. Left at None the
    table spans every level; narrowed to ``("paragraph",)`` it drops the
    OneStop Article block, and the level it does cover goes into the label so
    the narrowed table does not ship the all-levels one's ``\label``.

    ``delta=True`` prints, in the same cells, the per-item $r$ *minus* the
    original fully-aggregated $r$ (the number the paper's main EyeScore table
    reports), i.e. what per-item scoring cost. The original-vs-per-item
    correlation table asks the other question: whether the two scores rank
    participants alike.
    """
    eval_path = Path("src/evaluation/EyeScore/results/all_evaluation_results.csv")
    if not eval_path.exists():
        logger.warning(f"Evaluation file not found: {eval_path}")
        return

    df = pd.read_csv(eval_path)

    # One frame per (dataset, level) block, in paper order.
    block_rows = {
        ("Meco", "paragraph"): df[(df["dataset"] == "Meco") & (df["version_path"] == "per_item_agg/all/all") & (df["preview"] == "All")],
        ("OneStop", "paragraph"): df[(df["dataset"] == "OneStop") & (df["version_path"] == "per_item_agg/ordinary/all/paragraph") & (df["preview"] == preview)],
        ("OneStop", "article"): df[(df["dataset"] == "OneStop") & (df["version_path"] == "per_item_agg/ordinary/all/article") & (df["preview"] == preview)],
    }
    # The original score has no item level: one fully-aggregated frame per
    # dataset backs every level's block. MECO has no preview axis either, so
    # both of its frames are read from preview="All".
    original_rows = {
        "Meco": df[(df["dataset"] == "Meco") & (df["version_path"] == "fully_agg/all/all") & (df["preview"] == "All")],
        "OneStop": df[(df["dataset"] == "OneStop") & (df["version_path"] == "fully_agg/ordinary/all") & (df["preview"] == preview)],
    }
    sections = [sec for sec in seen_unseen_sections(datasets)
                if levels is None or sec[1] in levels]
    if not sections:
        logger.warning("No sections left for levels=%s, datasets=%s",
                       levels, datasets)
        return

    if any(block_rows[(dataset, level)].empty for dataset, level, _ in sections):
        logger.warning(
            "Missing EyeScore evaluation data for combined table "
            "(preview=%s)", preview)
        return
    if delta and any(original_rows[dataset].empty for dataset, _, _ in sections):
        logger.warning(
            "Missing original (fully aggregated) EyeScore rows for the delta "
            "table (preview=%s)", preview)
        return

    feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]
    feature_display = {
        "READING_SPEED": "WPM",
        "FIXATION_METRICS": "Avg. Fix.",
        "S_CLUSTERS_NO_NORM": "S-Clusters",
        "WP_COEFS_NO_NORM": "WP-Coefs",
        "TRANSITIONS": "Transitions",
        "WFC": "Word Fix.",
    }

    # One target block = Fixed/Any.
    target_blocks = [(dataset, level, target, target_display)
                     for dataset, level, _ in sections
                     for target, target_display in SECTION_TARGETS[dataset]]
    n_cols = 1 + 2 * len(target_blocks)

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{l" + "|".join(
            ["cccc" if len(SECTION_TARGETS[dataset]) == 2 else "cc"
             for dataset, _, _ in sections]) + "}",
        "\\toprule",
    ]

    # Header row 1: Dataset (and, only where a table mixes them, the level),
    # with the block rule carried up into the headings.
    titles = section_titles(sections)
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(header_block_cells(
                     [(2 * len(SECTION_TARGETS[dataset]), f"\\textbf{{{title}}}")
                      for (dataset, _, _), title in zip(sections, titles)]))
                 + " \\\\")

    # Header row 2: Targets (LexTALE/Composite for MECO, LexTALE/Michigan for
    # OneStop). The rule belongs at the section boundaries only -- within a
    # section the targets are columns of one block, and the column spec draws
    # no rule between them -- so it closes the last target of every section
    # but the last.
    section_end_idx, idx = set(), 0
    for position, (dataset, _, _) in enumerate(sections):
        idx += len(SECTION_TARGETS[dataset])
        if position < len(sections) - 1:
            section_end_idx.add(idx - 1)
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(
                     f"\\multicolumn{{2}}{{c{BLOCK_RULE if i in section_end_idx else ''}}}"
                     f"{{{target_display}}}"
                     for i, (_, _, _, target_display) in enumerate(target_blocks))
                 + " \\\\")
    lines.append(f"\\cmidrule{{2-{n_cols}}}")

    # Header row 3: Regimes
    lines.append("\\textbf{Features} & "
                 + " & ".join(["Fixed", "Any"] * len(target_blocks)) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for idx, feature_code in enumerate(feature_order):
        feature_name = feature_display.get(feature_code, feature_code)
        row = [feature_name]

        for dataset, level, target, _ in target_blocks:
            block = block_rows[(dataset, level)]
            for regime in ["seen", "unseen"]:
                if regime == "unseen" and feature_code in ["TRANSITIONS", "WFC"]:
                    row.append("-")
                    continue
                eval_data = block[
                    (block["target_col"] == target) &
                    (block["feature_set"] == feature_code) &
                    (block["content_mode"] == regime)
                ]
                if len(eval_data) == 0:
                    row.append("??")
                    continue
                val = eval_metric(eval_data.iloc[0], "pearson_r")
                if delta:
                    # Same feature set, same target, same regime -- the only
                    # thing that differs is per-item vs fully aggregated
                    # scoring, which is what the difference isolates.
                    original = original_rows[dataset]
                    base_row = original[
                        (original["target_col"] == target) &
                        (original["feature_set"] == feature_code) &
                        (original["content_mode"] == regime)
                    ]
                    if len(base_row) == 0:
                        row.append("??")
                        continue
                    val = val - eval_metric(base_row.iloc[0], "pearson_r")
                    # Signed, so a column of near-zero differences still reads
                    # as "which way did it move" rather than as magnitudes --
                    # except where the difference rounds away, which is
                    # printed unsigned: a "-0.00" is a rounding artefact, not
                    # a direction, and "+0.00" claims one just as wrongly.
                    row.append("NA" if pd.isna(val) else
                               f"{val:+.2f}" if round(val, 2) else "0.00")
                else:
                    row.append(f"{val:.2f}" if pd.notna(val) else "NA")

        lines.append(" & ".join(row) + " \\\\")
        # A rule between every feature group, not just after WPM. The
        # per-text group (Transitions, Word Fix.) exists in the Fixed
        # regime alone, so it reads as its own block only when it is
        # fenced off; \midrule rather than \cline/\cmidrule because a
        # partial rule cuts through the Features column.
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    # The levels the table actually prints go into the \label, so the
    # passage-only table and the all-levels one are separate \ref targets.
    # The caption needs no such patch: it lists the block titles it drew.
    single_level = sections_level(sections)
    label_suffix = (f"-{LEVEL_DISPLAY[single_level].lower()}"
                    if single_level else "")
    # The headers name the corpora and the regimes, so the caption's job is to
    # say which item was scored -- the one thing the table does not show. Kept
    # to the single sentence the paper prints; the method belongs to the
    # section around it, not to every table in it.
    covered = list(dict.fromkeys(level for _, level, _ in sections))
    item_note = (", with "
                 + " and with ".join(ITEM_WORD[level] for level in covered)
                 + " as test items")
    if delta:
        caption = (
            "\\caption{\\textbf{EyeScore}, aggregated-per-item variant. Change "
            "in the Pearson $r$ correlations with \\textbf{standard English "
            "proficiency tests} relative to the \\textbf{original EyeScore}"
            + item_note + ". A positive value means the aggregated-per-item "
            "variant correlates more strongly.}")
    else:
        caption = (
            "\\caption{\\textbf{EyeScore}, aggregated-per-item variant. Pearson "
            "$r$ correlations with \\textbf{standard English proficiency "
            "tests}" + item_note + ".}")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        caption,
        "\\label{tab:evaluation-eyescore-"
        + ("delta" if delta else "combined") + label_suffix + "}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # The delta table is not a contest between feature sets: its largest cell
    # is the set that gained most from per-item scoring, which is not the same
    # claim as "best EyeScore", so bolding it would be read as the wrong one.
    write_table(save_path, latex, bold_best=not delta)
    logger.info(f"Saved EyeScore evaluation combined table: {save_path}")


def generate_evaluation_predictions_onestop(save_path: Path,
                                            preview: str = DEFAULT_PREVIEW,
                                            levels: Sequence[str] = ("paragraph", "article"),
                                            ) -> None:
    r"""Generate the OneStop predictions evaluation table.

    Structure per level: OneStop <level> (LexTALE, Michigan), each target with
    Fixed and Any regimes and Pearson r / MAE.

    ``levels`` chooses the aggregation levels the table covers. Both (the
    default) gives the wide two-block table, which needs the full page width
    (``table*``). A single level gives a table half as wide, rendered as a
    one-column ``table`` sized to ``\columnwidth`` -- the same numbers, but
    placeable beside body text instead of spanning the page.

    ``preview`` selects the OneStop preview rows (the predictions CSV spells
    them lowercase, e.g. "gathering"/"hunting"), for both the values and the
    vs-WPM stars.
    """
    eval_path = Path("src/evaluation/predictions/results/all_evaluation_results.csv")
    if not eval_path.exists():
        logger.warning(f"Evaluation file not found: {eval_path}")
        return

    df = pd.read_csv(eval_path)
    preview_key = preview.lower()

    # level -> (rows for that level, its vs-WPM star map)
    blocks = {}
    for level in levels:
        version_path = f"per_item_agg/{level}"
        slice_ = df[(df["dataset"] == "OneStop")
                    & (df["version_path"] == version_path)
                    & (df["preview"] == preview_key)]
        rows = slice_[slice_["model_name"] == EVAL_TABLE_MODEL]
        if rows.empty:
            logger.warning(
                "Missing OneStop %s predictions evaluation data (preview=%s)",
                level, preview_key)
            return
        blocks[level] = (rows, wpm_star_map("OneStop", preview_key, version_path),
                         slice_)

    feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]
    feature_display = {
        "READING_SPEED": "WPM",
        "FIXATION_METRICS": "Avg. Fix.",
        "S_CLUSTERS_NO_NORM": "S-Clusters",
        "WP_COEFS_NO_NORM": "WP-Coefs",
        "TRANSITIONS": "Transitions",
        "WFC": "Word Fix.",
    }
    targets = [("lextale_score", "LexTALE"), ("michtest_score", "Michigan")]

    single = len(levels) == 1
    n_cols = 1 + 8 * len(levels)
    env, width = table_env(n_cols - 1)

    lines = [
        f"\\begin{{{env}}}[ht!]",
        "\\centering",
        f"\\resizebox{{{width}}}{{!}}{{",
        "\\begin{tabular}{l" + "|".join(["cccc"] * (2 * len(levels))) + "}",
        "\\toprule",
    ]

    # Header row 1: level. A single-level table has nothing to separate, so
    # the row would just repeat what its caption already says -- the level is
    # named there instead, and the table starts at the targets.
    if not single:
        lines.append(" \\multicolumn{1}{c}{} & "
                     + " & ".join(
                         f"\\multicolumn{{8}}{{c}}{{\\textbf{{OneStop {LEVEL_DISPLAY[level]}}}}}"
                         for level in levels)
                     + " \\\\")

    # Header row 2: targets
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(f"\\multicolumn{{4}}{{c}}{{{label}}}"
                              for _ in levels for _, label in targets)
                 + " \\\\")
    lines.append(f"\\cmidrule{{2-{n_cols}}}")

    # Header row 3: regimes
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join("\\multicolumn{2}{c}{Fixed} & \\multicolumn{2}{c}{Any}"
                              for _ in levels for _ in targets)
                 + " \\\\")

    # Header row 4: metrics
    lines.append("\\textbf{Features} & "
                 + " & ".join(["$r$", "MAE"] * (2 * 2 * len(levels))) + " \\\\")
    lines.append("\\midrule")

    # Reference row: the featureless mean baseline, fenced off from the feature
    # sets. Same columns, but $R^2$ on the left -- a constant prediction has no
    # correlation -- and no markers of any kind.
    brow = [BASELINE_LABEL]
    for level in levels:
        for target, _ in targets:
            for regime in ["fixed", "any"]:
                method = "seen__pool_all" if regime == "fixed" else "unseen__pool_all"
                brow += baseline_cells(
                    blocks[level][2], target, method,
                    max_stars=_column_max_stars(blocks[level][1], target, method))
    has_baseline = any(cell != "-" for cell in brow[1:])
    if has_baseline:
        lines.append(" & ".join(brow) + " \\\\")
        lines.append(BASELINE_RULE_NAME)
    else:
        logger.info(
            "No %s (mean-baseline) rows at per_item_agg for OneStop/%s — "
            "table rendered without the baseline row; run "
            "python -m src.run.predictions --mode per_item --models %s",
            BASELINE_MODEL, preview_key, BASELINE_MODEL)

    for feature_code in feature_order:
        row = [feature_display.get(feature_code, feature_code)]
        for level in levels:
            level_rows, star_map, _ = blocks[level]
            for target, _ in targets:
                for regime in ["fixed", "any"]:
                    # Per-text features need a shared column space, which the
                    # Any-Text split does not provide.
                    if regime == "any" and feature_code in ["TRANSITIONS", "WFC"]:
                        row.extend(["-", "-"])
                        continue

                    method_filter = "seen__pool_all" if regime == "fixed" else "unseen__pool_all"
                    eval_data = level_rows[
                        (level_rows["target_col"] == target) &
                        (level_rows["feature_set"] == feature_code) &
                        (level_rows["method"] == method_filter)
                    ]

                    r_stars, mae_stars = (
                        (0, 0) if feature_code == STAR_BASELINE_FEATURE_SET
                        else star_map.get((target, method_filter, feature_code), (0, 0)))
                    max_r_stars, max_mae_stars = _column_max_stars(
                        star_map, target, method_filter)
                    if len(eval_data) > 0:
                        row.append(_starred_cell(eval_metric(eval_data.iloc[0], "pearson_r"),
                                                 r_stars, max_r_stars))
                        row.append(_starred_cell(eval_metric(eval_data.iloc[0], "mae"),
                                                 mae_stars, max_mae_stars))
                    else:
                        row.extend(["??", "??"])

        lines.append(" & ".join(row) + " \\\\")
        # A rule between every feature group, not just after WPM. The
        # per-text group (Transitions, Word Fix.) exists in the Fixed
        # regime alone, so it reads as its own block only when it is
        # fenced off; \midrule rather than \cline/\cmidrule because a
        # partial rule cuts through the Features column.
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    if has_baseline:
        lines.insert(0, BASELINE_RULE_DEF)
    # Caption and \label follow the levels the table actually prints. The
    # level name is the paper's ("paragraph" is "Passage" everywhere the paper
    # speaks of it), so a \ref reads as the thing it points at.
    level_phrase = (f"the \\textbf{{{LEVEL_DISPLAY[levels[0]]}-Level}} aggregation"
                    if single else
                    "both \\textbf{Passage-Level and Article-Level} aggregations")
    label_suffix = f"-{LEVEL_DISPLAY[levels[0]].lower()}" if single else ""
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        # The paper prints this caption without the star/baseline legend.
        "\\caption{\\textbf{Prediction of standard proficiency test scores} with "
        "the \\textbf{aggregated-per-item} Ridge regression on "
        "\\textbf{OneStopL2} for " + level_phrase + ".}",
        f"\\label{{tab:evaluation-predictions-onestop{label_suffix}}}",
        f"\\end{{{env}}}",
    ])

    write_table(save_path, "\n".join(lines))
    logger.info(f"Saved OneStop predictions evaluation table: {save_path}")
def generate_reliability_eyescore_cronbach(
    save_path: Path, cronbach_path: Path = None,
    datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
) -> None:
    """Generate EyeScore Cronbach's alpha reliability table.

    Structure: MECO Passage | OneStop Passage | OneStop Article
    Each with Fixed and Any regimes. One value per dataset×level×regime.
    `datasets` drops the blocks of any dataset it leaves out.
    """
    cronbach_path = cronbach_path or Path("src/reliability/results/per_item_reliability/per_item_reliability.csv")
    if not cronbach_path.exists():
        logger.warning(f"Cronbach file not found: {cronbach_path}")
        return

    df = pd.read_csv(cronbach_path)

    # Filter for EyeScore only
    df = df[df["source"] == "eyescore"]

    if df.empty:
        logger.warning("No EyeScore Cronbach data found")
        return

    feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]
    feature_display = {
        "READING_SPEED": "WPM",
        "FIXATION_METRICS": "Avg. Fix.",
        "S_CLUSTERS_NO_NORM": "S-Clusters",
        "WP_COEFS_NO_NORM": "WP-Coefs",
        "TRANSITIONS": "Transitions",
        "WFC": "Word Fix.",
    }

    sections = seen_unseen_sections(datasets)
    n_cols = 1 + 4 * len(sections)   # 1 label + Fixed/Any x (alpha, SEM)
    env, width = table_env(n_cols - 1)

    lines = [
        f"\\begin{{{env}}}[ht!]",
        "\\centering",
        f"\\resizebox{{{width}}}{{!}}{{",
        "\\begin{tabular}{l" + "|".join(["cccc"] * len(sections)) + "}",
        "\\toprule",
    ]

    # Header row 1: Dataset names
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{title}}}}}"
                              for _, _, title in sections)
                 + " \\\\")
    lines.append(f"\\cmidrule{{2-{n_cols}}}")

    # Header row 2: Regimes, each spanning its alpha/SEM pair ("|" closes every
    # block but the last)
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(
                     "\\multicolumn{2}{c}{Fixed} & \\multicolumn{2}{c%s}{Any}"
                     % ("|" if i < len(sections) - 1 else "")
                     for i in range(len(sections)))
                 + " \\\\")

    # Header row 3: the pair itself, with \textbf{Features} on the last header
    # row so it sits directly above the feature names
    lines.append("\\textbf{Features} & "
                 + " & ".join(["$\\alpha$", "SEM"] * (2 * len(sections))) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for feature_code in feature_order:
        feature_name = feature_display.get(feature_code, feature_code)
        row = [feature_name]

        for dataset, level, _ in sections:
            for regime in ["seen", "unseen"]:
                if regime == "unseen" and feature_code in ["TRANSITIONS", "WFC"]:
                    row.extend(["-", "-"])
                    continue

                data = df[
                    (df["dataset"] == dataset) &
                    (df["p_agg_level"] == level) &
                    (df["feature_set"] == feature_code) &
                    (df["pool"] == regime) &
                    (df["target"].isna())
                ]

                if len(data) > 0:
                    val = data.iloc[0]["cronbachs_alpha"]
                    row.append(f"{val:.2f}" if pd.notna(val) else "??")
                else:
                    row.append("??")
                row.append(format_sem(sem_of(data)))

        lines.append(" & ".join(row) + " \\\\")
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Reliability of EyeScore in the Fixed and Any Text regimes. Depicted are Cronbach's $\\alpha$ with passages as test items, and the standard error of measurement $\\mathrm{SEM}=\\mathrm{SD}\\sqrt{1-\\alpha}$ on the observed-score scale.}",
        "\\label{tab:eyescore-cronbach-both-regimes}",
        f"\\end{{{env}}}",
    ])

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex, bold_best=False)
    logger.info(f"Saved EyeScore Cronbach table: {save_path}")


def generate_evaluation_predictions_meco(save_path: Path) -> None:
    """Generate predictions evaluation table for MECO only (passage level).

    Structure: MECO Passage (Composite, LexTALE)
    Each target shows Fixed and Any regimes with Pearson r and MAE values.
    """
    eval_path = Path("src/evaluation/predictions/results/all_evaluation_results.csv")
    if not eval_path.exists():
        logger.warning(f"Evaluation file not found: {eval_path}")
        return

    df = pd.read_csv(eval_path)

    # Filter for MECO data
    meco_slice = df[(df["dataset"] == "Meco") & (df["version_path"] == "per_item_agg/paragraph") & (df["preview"] == "all")]
    meco_para = meco_slice[meco_slice["model_name"] == EVAL_TABLE_MODEL]

    if meco_para.empty:
        logger.warning("Missing MECO predictions evaluation data")
        return

    star_map = wpm_star_map("Meco", "all", "per_item_agg/paragraph")

    feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]
    feature_display = {
        "READING_SPEED": "WPM",
        "FIXATION_METRICS": "Avg. Fix.",
        "S_CLUSTERS_NO_NORM": "S-Clusters",
        "WP_COEFS_NO_NORM": "WP-Coefs",
        "TRANSITIONS": "Transitions",
        "WFC": "Word Fix.",
    }

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{lcccc|cccc}",
        "\\toprule",
    ]

    # Header row 1: Dataset and Level
    lines.append(" \\multicolumn{1}{c}{} & \\multicolumn{8}{c}{\\textbf{MECO Passage}} \\\\")

    # Header row 2: Targets
    lines.append(" \\multicolumn{1}{c}{} & \\multicolumn{4}{c}{LexTALE} & \\multicolumn{4}{c}{Composite} \\\\")
    lines.append("\\cmidrule{2-9}")

    # Header row 3: Regimes
    lines.append(" \\multicolumn{1}{c}{} & \\multicolumn{2}{c}{Fixed} & \\multicolumn{2}{c}{Any} & \\multicolumn{2}{c}{Fixed} & \\multicolumn{2}{c}{Any} \\\\")

    # Header row 4: Metrics
    lines.append("\\textbf{Features} & $r$ & MAE & $r$ & MAE & $r$ & MAE & $r$ & MAE \\\\")
    lines.append("\\midrule")

    # Reference row: the featureless mean baseline, above a dashed rule.
    brow = [BASELINE_LABEL]
    for target in ["lextale_score", "proficiency_agg"]:
        for regime in ["fixed", "any"]:
            method = "seen__pool_all" if regime == "fixed" else "unseen__pool_all"
            brow += baseline_cells(
                meco_slice, target, method,
                max_stars=_column_max_stars(star_map, target, method))
    has_baseline = any(cell != "-" for cell in brow[1:])
    if has_baseline:
        lines.insert(0, BASELINE_RULE_DEF)
        lines.append(" & ".join(brow) + " \\\\")
        lines.append(BASELINE_RULE_NAME)
    else:
        logger.info(
            "No %s (mean-baseline) rows at per_item_agg for MECO — table "
            "rendered without the baseline row; run python -m src.run.predictions "
            "--mode per_item --models %s", BASELINE_MODEL, BASELINE_MODEL)

    # Data rows
    for idx, feature_code in enumerate(feature_order):
        feature_name = feature_display.get(feature_code, feature_code)
        row = [feature_name]

        # MECO Passage: lextale_score and proficiency_agg (Composite)
        for target in ["lextale_score", "proficiency_agg"]:
            for regime in ["fixed", "any"]:
                if regime == "any" and feature_code in ["TRANSITIONS", "WFC"]:
                    row.append("-")
                    row.append("-")
                    continue

                method_filter = "seen__pool_all" if regime == "fixed" else "unseen__pool_all"
                eval_data = meco_para[
                    (meco_para["target_col"] == target) &
                    (meco_para["feature_set"] == feature_code) &
                    (meco_para["method"] == method_filter)
                ]

                r_stars, mae_stars = (
                    (0, 0) if feature_code == STAR_BASELINE_FEATURE_SET
                    else star_map.get((target, method_filter, feature_code), (0, 0)))
                max_r_stars, max_mae_stars = _column_max_stars(
                    star_map, target, method_filter)
                if len(eval_data) > 0:
                    row.append(_starred_cell(eval_metric(eval_data.iloc[0], "pearson_r"),
                                             r_stars, max_r_stars))
                    row.append(_starred_cell(eval_metric(eval_data.iloc[0], "mae"),
                                             mae_stars, max_mae_stars))
                else:
                    row.append("??")
                    row.append("??")

        lines.append(" & ".join(row) + " \\\\")
        # A rule between every feature group, not just after WPM. The
        # per-text group (Transitions, Word Fix.) exists in the Fixed
        # regime alone, so it reads as its own block only when it is
        # fenced off; \midrule rather than \cline/\cmidrule because a
        # partial rule cuts through the Features column.
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        # The paper prints this caption without the star/baseline legend.
        "\\caption{\\textbf{Prediction of standard proficiency test scores} with "
        "the \\textbf{aggregated-per-item} Ridge regression on "
        "\\textbf{MECO L2}.}",
        "\\label{tab:evaluation-predictions-meco}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex)
    logger.info(f"Saved MECO predictions evaluation table: {save_path}")


def generate_reliability_eyescore_splithalf(
    save_path: Path, splithalf_path: Path = None,
    datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
) -> None:
    """Generate EyeScore split-half reliability table.

    Structure: MECO Passage | OneStop Passage | OneStop Article
    Each with Fixed and Any regimes. One value per dataset×level×regime.
    `datasets` drops the blocks of any dataset it leaves out.
    """
    splithalf_path = splithalf_path or Path("src/reliability/results/split_half_reliability/split_half_reliability.csv")
    if not splithalf_path.exists():
        logger.warning(f"Split-half file not found: {splithalf_path}")
        return

    df = pd.read_csv(splithalf_path)

    # Filter for EyeScore only
    df = df[df["source"] == "eyescore"]

    if df.empty:
        logger.warning("No EyeScore split-half data found")
        return

    feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]
    feature_display = {
        "READING_SPEED": "WPM",
        "FIXATION_METRICS": "Avg. Fix.",
        "S_CLUSTERS_NO_NORM": "S-Clusters",
        "WP_COEFS_NO_NORM": "WP-Coefs",
        "TRANSITIONS": "Transitions",
        "WFC": "Word Fix.",
    }

    value_col = "split_half" if "split_half" in df.columns else "mean_correlation"

    sections = seen_unseen_sections(datasets)

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{l" + "|".join(["cc"] * len(sections)) + "}",
        "\\toprule",
    ]

    # Header row 1: Dataset names
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{title}}}}}"
                              for _, _, title in sections)
                 + " \\\\")

    # Header row 2: Features and Regimes
    lines.append("\\textbf{Features} & "
                 + " & ".join(["Fixed", "Any"] * len(sections)) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for feature_code in feature_order:
        feature_name = feature_display.get(feature_code, feature_code)
        row = [feature_name]

        for dataset, level, _ in sections:
            for regime in ["seen", "unseen"]:
                if regime == "unseen" and feature_code in ["TRANSITIONS", "WFC"]:
                    row.append("-")
                    continue

                data = df[
                    (df["dataset"] == dataset) &
                    (df["p_agg_level"] == level) &
                    (df["feature_set"] == feature_code) &
                    (df["pool"] == regime) &
                    (df["target"].isna())
                ]

                if len(data) > 0:
                    val = data.iloc[0][value_col]
                    row.append(f"{val:.2f}" if pd.notna(val) else "??")
                else:
                    row.append("??")

        lines.append(" & ".join(row) + " \\\\")
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Reliability of EyeScore in the Fixed and Any Text regimes. Depicted is the average Pearson $r$ of 20 split-half reliability analyses.}",
        "\\label{tab:eyescore-splithalf-both-regimes}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex, bold_best=False)
    logger.info(f"Saved EyeScore split-half table: {save_path}")


def generate_reliability_predictions_cronbach(
    save_path: Path, cronbach_path: Path = None,
    datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
    levels: Optional[Sequence[str]] = None,
) -> None:
    r"""Generate predictions Cronbach's alpha reliability table.

    Structure: MECO Passage (Composite, LexTALE) | OneStop Passage | OneStop Article
    Targets shown as subtitle (row 2). Fixed and Any regimes in row 3.
    `datasets` drops the blocks of any dataset it leaves out.

    `levels` keeps only the aggregation levels it names. Left at None the table
    spans every level, which needs the full page width (``table*``). Narrowed
    to one level it is half as wide and is rendered as a one-column ``table``
    sized to ``\columnwidth``, with that level named in the caption and
    appended to the label.
    """
    cronbach_path = cronbach_path or Path("src/reliability/results/per_item_reliability/per_item_reliability.csv")
    if not cronbach_path.exists():
        logger.warning(f"Cronbach file not found: {cronbach_path}")
        return

    df = pd.read_csv(cronbach_path)

    # Filter for predictions only
    df = df[df["source"] == "predictions"]

    if df.empty:
        logger.warning("No predictions Cronbach data found")
        return

    feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]
    feature_display = {
        "READING_SPEED": "WPM",
        "FIXATION_METRICS": "Avg. Fix.",
        "S_CLUSTERS_NO_NORM": "S-Clusters",
        "WP_COEFS_NO_NORM": "WP-Coefs",
        "TRANSITIONS": "Transitions",
        "WFC": "Word Fix.",
    }

    sections = [sec for sec in seen_unseen_sections(datasets)
                if levels is None or sec[1] in levels]
    if not sections:
        logger.warning("No sections left for levels=%s, datasets=%s", levels, datasets)
        return
    single = len(sections) == 1
    # One target block = Fixed/Any x (alpha, SEM).
    target_blocks = [(dataset, level, target, target_display)
                     for dataset, level, _ in sections
                     for target, target_display in SECTION_TARGETS[dataset]]
    n_cols = 1 + 4 * len(target_blocks)
    env, width = table_env(n_cols - 1)

    lines = [
        f"\\begin{{{env}}}[ht!]",
        "\\centering",
        f"\\resizebox{{{width}}}{{!}}{{",
        "\\begin{tabular}{l" + "|".join(["cccc"] * len(target_blocks)) + "}",
        "\\toprule",
    ]

    # Header row 1: Dataset names. A one-section table has no blocks to tell
    # apart, so the row would only repeat what the caption says; the corpus
    # and the item word live there instead.
    if not single:
        lines.append(" \\multicolumn{1}{c}{} & "
                     + " & ".join(
                         f"\\multicolumn{{{4 * len(SECTION_TARGETS[dataset])}}}{{c}}"
                         f"{{\\textbf{{{title}}}}}"
                         for dataset, _, title in sections)
                     + " \\\\")

    # Header row 2: Targets ("|" closes each dataset block but the last)
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(
                     "\\multicolumn{4}{c%s}{%s}" % (
                         "|" if (j == len(SECTION_TARGETS[dataset]) - 1
                                 and i < len(sections) - 1) else "",
                         target_display)
                     for i, (dataset, _, _) in enumerate(sections)
                     for j, (_, target_display) in enumerate(SECTION_TARGETS[dataset]))
                 + " \\\\")
    lines.append(f"\\cmidrule{{2-{n_cols}}}")

    # Header row 3: Regimes
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join("\\multicolumn{2}{c}{Fixed} & \\multicolumn{2}{c%s}{Any}"
                              % ("|" if i < len(target_blocks) - 1 else "")
                              for i in range(len(target_blocks)))
                 + " \\\\")
    lines.append("\\textbf{Features} & "
                 + " & ".join(["$\\alpha$", "SEM"] * (2 * len(target_blocks))) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for idx, feature_code in enumerate(feature_order):
        feature_name = feature_display.get(feature_code, feature_code)
        row = [feature_name]

        for dataset, level, target, _ in target_blocks:
            for regime in ["seen", "unseen"]:
                if regime == "unseen" and feature_code in ["TRANSITIONS", "WFC"]:
                    row.extend(["-", "-"])
                    continue

                data = df[
                    (df["dataset"] == dataset) &
                    (df["p_agg_level"] == level) &
                    (df["feature_set"] == feature_code) &
                    (df["pool"] == regime) &
                    (df["target"] == target)
                ]

                if len(data) > 0:
                    val = data.iloc[0]["cronbachs_alpha"]
                    row.append(f"{val:.2f}" if pd.notna(val) else "??")
                    row.append(format_sem(sem_of(data)))
                else:
                    # Both halves of the pair, or the row loses a column.
                    row.extend(["??", "??"])

        lines.append(" & ".join(row) + " \\\\")
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    # The item word is the level's own only when the table shows one level;
    # spanning both, "passages" would mislabel the article half.
    item_word = ITEM_WORD[sections[0][1]] if single else "passages"
    # A one-section table names its corpus in the caption (the item word
    # already says which level it is), and carries both in its \label, so
    # MECO Passage and OneStop Passage do not collide on "-paragraph".
    # The corpus word comes off the block title ("MECO Passage" -> "MECO"),
    # which is the spelling the dataset rename in write_table knows.
    scope = (" on \\textbf{%s}"
             % sections[0][2].replace(f" {LEVEL_DISPLAY[sections[0][1]]}", "")
             if single else "")
    label_suffix = (f"-{sections[0][0].lower()}-{LEVEL_DISPLAY[sections[0][1]].lower()}"
                    if single else "")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        f"\\caption{{Reliability of prediction models{scope} in the Fixed and Any Text regimes. "
        f"Depicted are Cronbach's $\\alpha$ with {item_word} as test items, and the standard error "
        "of measurement $\\mathrm{SEM}=\\mathrm{SD}\\sqrt{1-\\alpha}$, in the units of the "
        f"predicted score. SEM is the consistency of the predicted score across {item_word}, not "
        "its error against the true proficiency score.}",
        f"\\label{{tab:predictions-cronbach-both-regimes{label_suffix}}}",
        f"\\end{{{env}}}",
    ])

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex, bold_best=False)
    logger.info(f"Saved predictions Cronbach table: {save_path}")


def generate_reliability_predictions_splithalf(
    save_path: Path, splithalf_path: Path = None,
    datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
) -> None:
    """Generate predictions split-half reliability table.

    Structure: MECO Passage (LexTALE, Composite) | OneStop Passage | OneStop Article
    Targets shown as subtitle (row 2). Fixed and Any regimes in row 3.
    `datasets` drops the blocks of any dataset it leaves out.
    """
    splithalf_path = splithalf_path or Path("src/reliability/results/split_half_reliability/split_half_reliability.csv")
    if not splithalf_path.exists():
        logger.warning(f"Split-half file not found: {splithalf_path}")
        return

    df = pd.read_csv(splithalf_path)

    # Filter for predictions only
    df = df[df["source"] == "predictions"]

    if df.empty:
        logger.warning("No predictions split-half data found")
        return

    feature_order = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]
    feature_display = {
        "READING_SPEED": "WPM",
        "FIXATION_METRICS": "Avg. Fix.",
        "S_CLUSTERS_NO_NORM": "S-Clusters",
        "WP_COEFS_NO_NORM": "WP-Coefs",
        "TRANSITIONS": "Transitions",
        "WFC": "Word Fix.",
    }

    value_col = "split_half" if "split_half" in df.columns else "mean_correlation"

    sections = seen_unseen_sections(datasets)
    # One target block = Fixed/Any.
    target_blocks = [(dataset, level, target, target_display)
                     for dataset, level, _ in sections
                     for target, target_display in SECTION_TARGETS[dataset]]
    n_cols = 1 + 2 * len(target_blocks)
    env, width = table_env(n_cols - 1)

    lines = [
        f"\\begin{{{env}}}[ht!]",
        "\\centering",
        f"\\resizebox{{{width}}}{{!}}{{",
        "\\begin{tabular}{l" + "|".join(["cc"] * len(target_blocks)) + "}",
        "\\toprule",
    ]

    # Header row 1: Dataset names
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(
                     f"\\multicolumn{{{2 * len(SECTION_TARGETS[dataset])}}}{{c}}"
                     f"{{\\textbf{{{title}}}}}"
                     for dataset, _, title in sections)
                 + " \\\\")

    # Header row 2: Targets ("|" closes each dataset block but the last)
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(
                     "\\multicolumn{2}{c%s}{%s}" % (
                         "|" if (j == len(SECTION_TARGETS[dataset]) - 1
                                 and i < len(sections) - 1) else "",
                         target_display)
                     for i, (dataset, _, _) in enumerate(sections)
                     for j, (_, target_display) in enumerate(SECTION_TARGETS[dataset]))
                 + " \\\\")
    lines.append(f"\\cmidrule{{2-{n_cols}}}")

    # Header row 3: Regimes
    lines.append("\\textbf{Features} & "
                 + " & ".join(["Fixed", "Any"] * len(target_blocks)) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for idx, feature_code in enumerate(feature_order):
        feature_name = feature_display.get(feature_code, feature_code)
        row = [feature_name]

        for dataset, level, target, _ in target_blocks:
            for regime in ["seen", "unseen"]:
                if regime == "unseen" and feature_code in ["TRANSITIONS", "WFC"]:
                    row.append("-")
                    continue

                data = df[
                    (df["dataset"] == dataset) &
                    (df["p_agg_level"] == level) &
                    (df["feature_set"] == feature_code) &
                    (df["pool"] == regime) &
                    (df["target"] == target)
                ]

                if len(data) > 0:
                    val = data.iloc[0][value_col]
                    row.append(f"{val:.2f}" if pd.notna(val) else "??")
                else:
                    row.append("??")

        lines.append(" & ".join(row) + " \\\\")
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Reliability of prediction models in the Fixed and Any Text regimes. Depicted is the average Pearson $r$ of 20 split-half reliability analyses.}",
        "\\label{tab:predictions-splithalf-both-regimes}",
        f"\\end{{{env}}}",
    ])

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex, bold_best=False)
    logger.info(f"Saved predictions split-half table: {save_path}")


ICC_COLUMNS = ["icc1_1", "icc1_k", "icc2_1", "icc2_k", "icc3_1", "icc3_k"]
ICC_COLUMNS_ALL = [f"{c}_all" for c in ICC_COLUMNS]
ICC_COLUMNS_POOLED = [f"{c}_pooled" for c in ICC_COLUMNS]

# LexTALE first in both datasets, per the target-order rule.
ICC_TARGETS = {
    "Meco": [("lextale_score", "LexTALE"), ("proficiency_agg", "Composite")],
    "OneStop": [("lextale_score", "LexTALE"), ("michtest_score", "Michigan")],
}

ICC_FEATURE_ORDER = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM", "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]
ICC_FEATURE_DISPLAY = {
    "READING_SPEED": "WPM",
    "FIXATION_METRICS": "Avg. Fix.",
    "S_CLUSTERS_NO_NORM": "S-Clusters",
    "WP_COEFS_NO_NORM": "WP-Coefs",
    "TRANSITIONS": "Transitions",
    "WFC": "Word Fix.",
}


def _icc_feature_cells(df: pd.DataFrame, level: str, target: str, feature_code: str, columns: list) -> tuple:
    """One feature's cells (Fixed's six variants, then Any's) plus the matching rows.

    Returns (cells, sample_rows) where sample_rows maps regime -> the CSV row, so a
    caller can read n and k from whichever regime is present.
    """
    cells, sample = [], {}
    for regime in ["seen", "unseen"]:
        match = df[
            (df["p_agg_level"] == level)
            & (df["feature_set"] == feature_code)
            & (df["pool"] == regime)
            & (df["target"].isna() if target is None else df["target"] == target)
        ]
        if len(match) == 0:
            cells.extend(["-"] * len(columns))
            continue
        row = match.iloc[0]
        cells.extend(format_value(row[col]) for col in columns)
        sample[regime] = row
    return cells, sample


def _fmt_k(value) -> str:
    """Item count for a heading: 12 rather than 12.0, 8.6 for a mean."""
    return f"{float(value):.1f}".rstrip("0").rstrip(".")


def generate_reliability_icc_table(
    save_path: Path, source: str, dataset: str = None, cronbach_path: Path = None,
    all_readers: bool = False, datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
) -> None:
    """Generate the all-variants ICC table for one source.

    All feature sets are rows; the columns are the six Shrout & Fleiss variants
    under each regime (Fixed then Any). Row blocks are dataset x aggregation level,
    and the target as well for predictions.

    `dataset=None` gives the combined table: every dataset and level in one, one
    row per feature. Naming a dataset splits it out, which is what MECO needs —
    there ~87% of readers are missing at least one passage, so each feature gets
    two rows, the complete-case ANOVA and the all-readers estimate, each with its
    own n and k. OneStop stays one row per feature: its readers x passages matrix
    is complete inside every content group, so the two estimators coincide.

    `all_readers` switches the single-row layouts to the all-readers columns; it is
    ignored for the MECO split table, which shows both families side by side.

    `datasets` bounds the combined table's row blocks; it is redundant when
    `dataset` names one.
    """
    cronbach_path = cronbach_path or Path("src/reliability/results/per_item_reliability/per_item_reliability.csv")
    if not cronbach_path.exists():
        logger.warning(f"ICC table: per-item reliability file not found: {cronbach_path}")
        return

    # Both split tables carry two rows per feature, but they contrast different things:
    # MECO the complete-case vs all-readers sample, OneStop the per-group average vs
    # the pooled mixed-model estimate.
    show_both = dataset in ("Meco", "OneStop")
    needed = list(ICC_COLUMNS)
    if dataset == "Meco" or all_readers:
        needed += ICC_COLUMNS_ALL
    if dataset == "OneStop":
        needed += ICC_COLUMNS_POOLED

    df = pd.read_csv(cronbach_path)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.warning(
            f"ICC table skipped: {cronbach_path} predates the ICC columns "
            f"(missing {', '.join(missing)}). Re-run src.reliability.cronbach_alpha."
        )
        return

    df = df[df["source"] == source]
    if dataset:
        df = df[df["dataset"] == dataset]
    else:
        df = df[df["dataset"].isin(datasets)]
    if df.empty:
        logger.warning(f"No {source} ICC data found{f' for {dataset}' if dataset else ''}")
        return

    # `mean_col`, where present, reports how many passages a reader actually read —
    # distinct from k, which is the length the (*,k) variants are computed at.
    complete_variant = dict(label="complete", columns=ICC_COLUMNS,
                            n_col="icc_n_participants", k_col="icc_n_items")
    all_variant = dict(label="all readers", columns=ICC_COLUMNS_ALL,
                       n_col="icc_n_participants_all", k_col="icc_n_items",
                       mean_col="icc_mean_items_all")
    group_avg_variant = dict(label="group avg.", columns=ICC_COLUMNS,
                             n_col="icc_n_participants", k_col="icc_n_items")
    pooled_variant = dict(label="pooled", columns=ICC_COLUMNS_POOLED,
                          n_col="icc_n_participants_pooled", k_col="icc_n_items_pooled")

    if dataset == "Meco":
        variants = [complete_variant, all_variant]
    elif dataset == "OneStop":
        variants = [group_avg_variant, pooled_variant]
    elif all_readers:
        variants = [{**all_variant, "label": None}]
    else:
        variants = [{**complete_variant, "label": None}]

    blocks = []
    for ds in ([dataset] if dataset else ["Meco", "OneStop"]):
        ds_label = "MECO" if ds == "Meco" else "OneStop"
        levels = [("paragraph", "Passage")] if ds == "Meco" else [("paragraph", "Passage"), ("article", "Article")]
        for level, level_label in levels:
            label = f"{ds_label} {level_label}"
            if source == "predictions":
                # "--" not an em dash: every other generated .tex is pure ASCII
                blocks.extend((ds, level, f"{label} -- {name}", code) for code, name in ICC_TARGETS[ds])
            else:
                blocks.append((ds, level, label, None))

    n_label_cols = 2 if show_both else 1
    n_cols = n_label_cols + 2 * len(ICC_COLUMNS)
    col_spec = ("ll" if show_both else "l") + "|cc|cc|cc||cc|cc|cc"

    lines = [
        "\\begin{table*}[ht!]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f" \\multicolumn{{{n_label_cols}}}{{c}}{{}} & \\multicolumn{{6}}{{c}}{{\\textbf{{Fixed}}}}"
        f" & \\multicolumn{{6}}{{c}}{{\\textbf{{Any}}}} \\\\",
        f"\\cmidrule{{{n_label_cols + 1}-{n_cols}}}",
        f" \\multicolumn{{{n_label_cols}}}{{c}}{{}} "
        + " ".join(["& \\multicolumn{2}{c}{ICC(1)} & \\multicolumn{2}{c}{ICC(2)} & \\multicolumn{2}{c}{ICC(3)}"] * 2)
        + " \\\\",
        f"\\cmidrule{{{n_label_cols + 1}-{n_cols}}}",
        "\\textbf{Features}" + (" & \\textbf{Readers}" if show_both else "")
        + " & " + " & ".join(["Single", "Avg."] * 6) + " \\\\",
        "\\midrule",
    ]

    for block_idx, (ds, level, label, target) in enumerate(blocks):
        body, block_sample, has_data = [], {}, False
        block_df = df[df["dataset"] == ds]

        for feature_code in ICC_FEATURE_ORDER:
            for variant_idx, variant in enumerate(variants):
                columns, n_col, k_col = variant["columns"], variant["n_col"], variant["k_col"]
                cells, sample = _icc_feature_cells(block_df, level, target, feature_code, columns)
                has_data = has_data or any(c != "-" for c in cells)
                row_head = ICC_FEATURE_DISPLAY[feature_code] if variant_idx == 0 else ""

                if show_both:
                    ref = sample.get("seen", sample.get("unseen"))
                    detail = ""
                    if ref is not None and pd.notna(ref.get(n_col)):
                        detail = f" ($n$={int(ref[n_col])}, $k$={_fmt_k(ref[k_col])}"
                        mean_col = variant.get("mean_col")
                        if mean_col and pd.notna(ref.get(mean_col)):
                            # k is the length the ICC is computed at; this is what
                            # readers actually read, and the two differ for MECO.
                            detail += f", read {_fmt_k(ref[mean_col])}"
                        detail += ")"
                    body.append(f"{row_head} & {variant['label']}{detail} & " + " & ".join(cells) + " \\\\")
                else:
                    body.append(f"{row_head} & " + " & ".join(cells) + " \\\\")
                    # n and k are per block, not per feature; keep the first of each regime
                    for regime, row in sample.items():
                        if regime not in block_sample and pd.notna(row.get(n_col)):
                            block_sample[regime] = (int(row[n_col]), _fmt_k(row[k_col]))

            if should_add_cline_after_feature(feature_code, ICC_FEATURE_ORDER):
                body.append("\\midrule")

        if not has_data:
            continue

        # Fixed before Any, per the regime-order rule
        heading_note = "; ".join(
            f"{POOL_DISPLAY[regime]}: $n$={n}, $k$={k}"
            for regime in ["seen", "unseen"] if regime in block_sample
            for n, k in [block_sample[regime]]
        )

        if block_idx > 0:
            lines.append("\\midrule")
        heading = f"\\textbf{{{label}}}" + (f" \\quad {{\\small ({heading_note})}}" if heading_note else "")
        lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{{heading}}} \\\\")
        lines.extend(body)

    source_display = "EyeScore" if source == "eyescore" else "the prediction models"
    where = "" if dataset is None else f" on {'MECO' if dataset == 'Meco' else 'OneStop'}"
    if dataset == "Meco":
        sample_sentence = (
            "Most MECO readers are missing at least one passage, so each feature is given twice: "
            "\\emph{complete} is the ANOVA over readers with a full passage set, and "
            "\\emph{all readers} keeps everyone, taking ICC(1) from the unbalanced one-way ANOVA "
            "and ICC(2)/ICC(3) from REML variance components. Average-measure variants are computed "
            "at $k$ passages; \\emph{read} is how many a reader actually has, so those are the "
            "reliability of a $k$-passage score rather than of the scores as observed."
        )
    elif dataset == "OneStop":
        sample_sentence = (
            "The content groups read disjoint passage sets, so the readers $\\times$ passages matrix "
            "is complete within a group but block diagonal across them. \\emph{group avg.} is the mean "
            "of the six per-group ICCs, the same summary used for Cronbach's $\\alpha$; \\emph{pooled} "
            "is a single estimate over all readers at once, from REML variance components with the "
            "content group as a fixed effect. The average is a summary of six estimates, the pooled "
            "value is one estimate, which is why they need not coincide."
        )
    elif all_readers:
        sample_sentence = (
            "Every reader is used, including those missing passages: ICC(1) from the unbalanced "
            "one-way ANOVA and ICC(2)/ICC(3) from REML variance components, neither of which "
            "requires a complete matrix. Average-measure variants are reported at the full "
            "passage count."
        )
    else:
        sample_sentence = (
            "Each ICC needs a complete readers $\\times$ passages matrix, so $n$ is the "
            "listwise-complete sample."
        )

    label_suffix = f"-{dataset.lower()}" if dataset else ("-all" if all_readers else "")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        f"\\caption{{\\textbf{{Intraclass correlations for {source_display}{where}}}, "
        "with readers as targets and passages as raters. ICC(1) is the one-way model, in which item "
        "effects cannot be separated from error; ICC(2) is two-way random with absolute agreement, "
        "counting systematic item differences as error; ICC(3) is two-way mixed, treating the item set "
        "as fixed (consistency). Single is the reliability of one passage, Avg.\\ that of the mean over "
        f"all $k$ passages; ICC(3,$k$) equals Cronbach's $\\alpha$ by construction. {sample_sentence}}}",
        f"\\label{{tab:icc-variants-{source}{label_suffix}}}",
        "\\end{table*}",
    ])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, "\n".join(lines), bold_best=False)
    logger.info(f"Saved {source} ICC table ({dataset or 'combined'}): {save_path}")


# EyeScore reliability tables: the feature rows, the item word that goes in the
# caption, and the split-half column name are shared by the two layouts below
# (one regime x datasets, and one dataset x both regimes).
# Canonical row order. Transitions and Word Fix. are per-text feature sets:
# they need a shared paragraph column space, which the Any-Text (unseen) split
# does not provide, so they have Fixed-regime rows only. The rows actually
# drawn are derived from the data (_eyescore_feature_rows), so an Any-only
# table keeps the four always-available sets and a table with a Fixed block
# gains the other two.
EYESCORE_RELIABILITY_FEATURES = [
    "READING_SPEED",
    "FIXATION_METRICS",
    "S_CLUSTERS_NO_NORM",
    "WP_COEFS_NO_NORM",
    "TRANSITIONS",
    "WFC",
]
# Cells that cannot exist (a per-text feature set in the Any regime) rather
# than data that failed to compute — the tables' usual "??" would read as the
# latter.
EYESCORE_RELIABILITY_MISSING = "-"


PER_TEXT_MISSING_NOTE = (
    " Per-text feature sets (Transitions, Word Fix.) have no Any-Text column: "
    "they require a shared paragraph column space, which the Any-Text split "
    "does not provide."
)


def _eyescore_reliability_frames(cronbach_df, split_half_df, p_agg_level,
                                 pools):
    """(cronbach, split-half, split_half_col) restricted to EyeScore rows at
    ``p_agg_level`` in ``pools``, or None when either side is empty."""
    cron = cronbach_df[
        (cronbach_df["source"] == "eyescore") &
        (cronbach_df["p_agg_level"] == p_agg_level) &
        (cronbach_df["pool"].isin(pools))
    ].copy()
    split = split_half_df[
        (split_half_df["source"] == "eyescore") &
        (split_half_df["p_agg_level"] == p_agg_level) &
        (split_half_df["pool"].isin(pools))
    ].copy()
    if cron.empty or split.empty:
        return None
    split_half_col = ("split_half" if "split_half" in split.columns
                      else "mean_correlation")
    return cron, split, split_half_col


def _eyescore_reliability_cells(cron, split, split_half_col, dataset, pool,
                                feature_code) -> list:
    """The three cells of one block: alpha, its SEM, split-half. EyeScore rows
    carry a NaN target, which is what separates them from the prediction rows
    sharing the same CSV."""
    ds_cron = cron[(cron["dataset"] == dataset) &
                   (cron["pool"] == pool) &
                   (cron["feature_set"] == feature_code) &
                   (cron["target"].isna())]
    ds_split = split[(split["dataset"] == dataset) &
                     (split["pool"] == pool) &
                     (split["feature_set"] == feature_code) &
                     (split["target"].isna())]
    val_cron = ds_cron.iloc[0]["cronbachs_alpha"] if len(ds_cron) else np.nan
    val_split = ds_split.iloc[0][split_half_col] if len(ds_split) else np.nan
    missing = EYESCORE_RELIABILITY_MISSING
    return [format_value(val_cron, missing=missing),
            format_sem(sem_of(ds_cron), missing=missing),
            format_value(val_split, missing=missing)]


def _datasets_with_rows(cron, split) -> set:
    """Datasets with any EyeScore reliability rows. One without (OneStop, where
    only the MECO results exist) is dashed out wholesale, which is not what the
    caption's missing-cell note explains."""
    return set(cron["dataset"]) | set(split["dataset"])


def _eyescore_feature_rows(cron, split, split_half_col, blocks) -> list:
    """Feature sets with at least one value across ``blocks``, in canonical
    order. Drops a set that no block can report -- Transitions and Word Fix.
    in an Any-Text-only table -- instead of drawing a row of dashes."""
    rows = []
    for feature_code in EYESCORE_RELIABILITY_FEATURES:
        cells = [cell
                 for dataset, pool in blocks
                 for cell in _eyescore_reliability_cells(
                     cron, split, split_half_col, dataset, pool, feature_code)]
        if any(cell != EYESCORE_RELIABILITY_MISSING for cell in cells):
            rows.append(feature_code)
    return rows


def generate_eyescore_reliability_combined_table(
    cronbach_df: pd.DataFrame,
    split_half_df: pd.DataFrame,
    save_path: Path,
    datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
    pool: str = "unseen",
    p_agg_level: str = "paragraph",
) -> None:
    """Generate EyeScore reliability table with both Cronbach's alpha and Split-half metrics.

    Structure: Meco | OneStop, each with Cronbach's α | SEM | Split-Half
    subheaders, for one regime (``pool``: unseen = Any Text, seen = Fixed Text)
    at one item level. `datasets` drops the blocks of any dataset it leaves out.
    """
    frames = _eyescore_reliability_frames(cronbach_df, split_half_df,
                                          p_agg_level, [pool])
    if frames is None:
        logger.warning("Missing eyescore data for %s pool %s level",
                       pool, p_agg_level)
        return
    cron, split, split_half_col = frames

    feature_order = _eyescore_feature_rows(
        cron, split, split_half_col, [(ds, pool) for ds in datasets])
    regime = POOL_DISPLAY[pool]

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{l" + "|".join(["ccc"] * len(datasets)) + "}",
        "\\toprule",
    ]

    # Header row 1: Dataset names, with the block rule reaching up into them.
    lines.append(" \\multicolumn{1}{c}{} & "
                 + " & ".join(header_block_cells(
                     [(3, f"\\textbf{{{DATASET_DISPLAY[ds]}}}") for ds in datasets]))
                 + " \\\\")

    # Header row 2: Metric names. SEM sits directly after the alpha it is
    # derived from (SD*sqrt(1 - alpha)), matching the alpha/SEM pairing in the
    # appendix Cronbach tables.
    lines.append("\\textbf{Features} & "
                 + " & ".join(["Cronbach's $\\alpha$", "SEM", "Split-Half $r$"] * len(datasets))
                 + " \\\\")
    lines.append("\\midrule")

    # Data rows. A dash in a dataset that has results means a cell that cannot
    # exist, which the caption then explains.
    present = _datasets_with_rows(cron, split)
    has_missing = False
    for feature_code in feature_order:
        feature_display = FEATURE_SET_DISPLAY.get(feature_code, feature_code)
        row = [feature_display]
        for dataset in datasets:
            cells = _eyescore_reliability_cells(cron, split, split_half_col,
                                                dataset, pool, feature_code)
            row += cells
            has_missing |= dataset in present and EYESCORE_RELIABILITY_MISSING in cells
        lines.append(" & ".join(row) + " \\\\")
        # Rule between the feature groups: WPM alone, then the per-participant
        # sets, then the per-text ones.
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    # The wording the paper prints, which differs by regime: the Any table is
    # the one in the body, trimmed to a single sentence with the method note
    # left in the file but commented out; the Fixed table is in the appendix
    # and spells the method out. Both are reproduced as the paper prints them.
    method_note = ("Depicted are Cronbach's $\\alpha$ with "
                   + ITEM_WORD[p_agg_level] + " as test items, the average Pearson "
                   "$r$ of 20 split-half reliability analyses, and the standard "
                   "error of measurement $\\mathrm{SEM}=\\mathrm{SD}\\sqrt{1-\\alpha}$ "
                   "on the observed-score scale."
                   + (PER_TEXT_MISSING_NOTE if has_missing else ""))
    if regime == "Any":
        lines.append("\\caption{\\textbf{Reliability of EyeScore} for L2 participants "
                     "in the Any Text regime.}% " + method_note + "}")
    else:
        lines.append("\\caption{\\textbf{Reliability of EyeScore} in the \\textbf{"
                     + regime + " Text} regime, with \\textbf{"
                     + ITEM_WORD[p_agg_level] + "} as test items."
                     + (PER_TEXT_MISSING_NOTE if has_missing else "") + "}")
    lines.append("\\label{tab:eyescore-reliability-" + regime.lower()
                 + "-" + p_agg_level + "}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex, bold_best=False)
    logger.info(f"Saved EyeScore reliability table: {save_path}")


def generate_eyescore_reliability_regimes_table(
    cronbach_df: pd.DataFrame,
    split_half_df: pd.DataFrame,
    save_path: Path,
    datasets: Sequence[str] = DEFAULT_TABLE_DATASETS,
    pools: Sequence[str] = ("seen", "unseen"),
    p_agg_level: str = "article",
) -> None:
    """EyeScore reliability with both regimes side by side, at one item level.

    Same three metrics as ``generate_eyescore_reliability_combined_table``
    (Cronbach's α, its SEM, split-half), but the blocks are the regimes -- Fixed
    (seen) and Any (unseen) -- rather than one regime across datasets. Written
    for articles as test items, where only OneStop has data: MECO's twelve texts
    are unrelated passages with no article structure, so a dataset with no rows
    at this level is dropped and, if none is left, the table is skipped.
    """
    frames = _eyescore_reliability_frames(cronbach_df, split_half_df,
                                          p_agg_level, list(pools))
    if frames is None:
        logger.warning("Missing eyescore data at %s level for pools %s",
                       p_agg_level, ", ".join(pools))
        return
    cron, split, split_half_col = frames

    # Keep only datasets that actually have rows at this level -- MECO has no
    # articles, so asking for one would give a block of "??".
    datasets = [ds for ds in datasets if (cron["dataset"] == ds).any()]
    if not datasets:
        logger.warning("No eyescore %s-level rows for any requested dataset; "
                       "skipping %s", p_agg_level, save_path.name)
        return

    blocks = [(ds, pool) for ds in datasets for pool in pools]
    feature_order = _eyescore_feature_rows(cron, split, split_half_col, blocks)

    lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{l" + "|".join(["ccc"] * len(blocks)) + "}",
        "\\toprule",
    ]

    # A dataset header row only when there is more than one dataset to tell
    # apart; with a single one the caption names it and the row would be noise.
    # The block rule reaches up into whichever of these is the topmost heading.
    # Where both are drawn it is the dataset row, and the regime row below it
    # then carries a full-height rule at the dataset boundaries -- and none at
    # the Fixed/Any boundaries inside a dataset, which the heading above is
    # what separates -- so the line runs unbroken down into the body.
    if len(datasets) > 1:
        lines.append(" \\multicolumn{1}{c}{} & "
                     + " & ".join(header_block_cells(
                         [(3 * len(pools), f"\\textbf{{{DATASET_DISPLAY[ds]}}}")
                          for ds in datasets]))
                     + " \\\\")
        regime_cells = header_block_cells(
            [(3, f"\\textbf{{{POOL_HEADING.get(pool, POOL_DISPLAY[pool])}}}")
             for _, pool in blocks])
    else:
        regime_cells = header_block_cells(
            [(3, f"\\textbf{{{POOL_HEADING.get(pool, POOL_DISPLAY[pool])}}}")
             for _, pool in blocks])

    lines.append(" \\multicolumn{1}{c}{} & " + " & ".join(regime_cells) + " \\\\")
    lines.append("\\textbf{Features} & "
                 + " & ".join(["Cronbach's $\\alpha$", "SEM", "Split-Half $r$"] * len(blocks))
                 + " \\\\")
    lines.append("\\midrule")

    present = _datasets_with_rows(cron, split)
    has_missing = False
    for feature_code in feature_order:
        row = [FEATURE_SET_DISPLAY.get(feature_code, feature_code)]
        for dataset, pool in blocks:
            cells = _eyescore_reliability_cells(cron, split, split_half_col,
                                                dataset, pool, feature_code)
            row += cells
            has_missing |= dataset in present and EYESCORE_RELIABILITY_MISSING in cells
        lines.append(" & ".join(row) + " \\\\")
        if should_add_cline_after_feature(feature_code, feature_order):
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    scope = (" for " + DATASET_DISPLAY[datasets[0]]) if len(datasets) == 1 else ""
    lines.append("\\caption{\\textbf{Reliability of EyeScore} with "
                 + ITEM_WORD[p_agg_level] + " as test items" + scope
                 + ", in the "
                 + " and ".join(POOL_DISPLAY[pool] for pool in pools)
                 + " Text regimes. Depicted are Cronbach's $\\alpha$, the average "
                 "Pearson $r$ of 20 split-half reliability analyses, and the "
                 "standard error of measurement "
                 "$\\mathrm{SEM}=\\mathrm{SD}\\sqrt{1-\\alpha}$ on the "
                 "observed-score scale."
                 + (PER_TEXT_MISSING_NOTE if has_missing else "") + "}")
    lines.append("\\label{tab:eyescore-reliability-regimes-" + p_agg_level + "}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(save_path, latex, bold_best=False)
    logger.info(f"Saved EyeScore reliability table: {save_path}")


def main(preview: str = DEFAULT_PREVIEW):
    """Generate LaTeX tables for reliability metrics.

    `preview` selects which reliability CSVs to read and where tables land.
    Gathering uses the top-level paths; any other preview reads and writes a
    parallel subtree (…/per_item_reliability/hunting/, …/tables/hunting/).

    The evaluation tables read src/evaluation/**/all_evaluation_results.csv,
    which carries its own `preview` column; `preview` is passed down to that
    filter (lowercase in the predictions CSV, capitalised in the EyeScore one),
    so a Hunting run reports Hunting numbers. The MECO-only predictions table
    has no preview axis and is emitted for Gathering alone.

    A non-default preview is OneStop-only (`preview_datasets`): MECO has no
    preview axis, so its blocks are dropped rather than repeating the Gathering
    numbers under an information-seeking heading. Every caption/label is then
    marked by `annotate_preview` so the tables can sit alongside the Gathering
    ones.
    """
    results_dir = Path("src/reliability/results")
    sub = "" if preview == DEFAULT_PREVIEW else preview.lower()
    datasets = preview_datasets(preview)
    cronbach_path = results_dir / "per_item_reliability" / sub / "per_item_reliability.csv"
    split_half_path = results_dir / "split_half_reliability" / sub / "split_half_reliability.csv"
    output_dir = results_dir / "tables" / sub
    logger.info(f"Reliability tables for preview={preview} -> {output_dir}")

    logger.info("Generating reliability tables...")

    # Cronbach's Alpha tables
    if cronbach_path.exists():
        logger.info(f"Loading {cronbach_path}")
        df_cronbach = pd.read_csv(cronbach_path)

        # Unseen pool (Any regime)
        caption_cronbach_unseen = (
            "Cronbach's $\\alpha$ for EyeScore and prediction models over per-passage aggregation "
            "for the Any regime. SEM is the standard error of measurement, "
            "$\\mathrm{SD}\\sqrt{1-\\alpha}$, on the observed-score scale."
        )
        generate_paragraph_pool_combined_table(
            df_cronbach,
            pool="unseen",
            value_col="cronbachs_alpha",
            caption_metric=caption_cronbach_unseen,
            save_path=output_dir / "cronbach_alpha_paragraph_unseen.tex",
            metric_name="cronbach",
            sem_col="sem_cronbach",
            datasets=datasets,
        )

        # Seen pool (Fixed regime)
        caption_cronbach_seen = (
            "Cronbach's $\\alpha$ for EyeScore and prediction models over per-passage aggregation "
            "for the Fixed regime. SEM is the standard error of measurement, "
            "$\\mathrm{SD}\\sqrt{1-\\alpha}$, on the observed-score scale."
        )
        generate_paragraph_pool_combined_table(
            df_cronbach,
            pool="seen",
            value_col="cronbachs_alpha",
            caption_metric=caption_cronbach_seen,
            save_path=output_dir / "cronbach_alpha_paragraph_seen.tex",
            metric_name="cronbach",
            sem_col="sem_cronbach",
            datasets=datasets,
        )
    else:
        logger.warning(f"Cronbach's alpha file not found: {cronbach_path}")

    # Split-Half tables
    if split_half_path.exists():
        logger.info(f"Loading {split_half_path}")
        df_split_half = pd.read_csv(split_half_path)

        # Unseen pool (Any regime)
        caption_split_half_unseen = (
            "Split-Half reliability for EyeScore and prediction models over per-passage aggregation "
            "for the Any regime."
        )
        generate_paragraph_pool_combined_table(
            df_split_half,
            pool="unseen",
            value_col="mean_correlation",
            caption_metric=caption_split_half_unseen,
            save_path=output_dir / "split_half_paragraph_unseen.tex",
            metric_name="split-half",
            datasets=datasets,
        )

        # Seen pool (Fixed regime)
        caption_split_half_seen = (
            "Split-Half reliability for EyeScore and prediction models over per-passage aggregation "
            "for the Fixed regime."
        )
        generate_paragraph_pool_combined_table(
            df_split_half,
            pool="seen",
            value_col="mean_correlation",
            caption_metric=caption_split_half_seen,
            save_path=output_dir / "split_half_paragraph_seen.tex",
            metric_name="split-half",
            datasets=datasets,
        )
    else:
        logger.warning(f"Split-half file not found: {split_half_path}")

    # Article-level tables (OneStop only, both pools)
    if cronbach_path.exists():
        logger.info(f"Loading {cronbach_path} for article tables")
        df_cronbach = pd.read_csv(cronbach_path)

        caption_cronbach_article = (
            "Cronbach's $\\alpha$ for EyeScore and prediction models over per-article aggregation "
            "for OneStop, comparing Fixed and Any regimes."
        )
        generate_article_onestop_both_pools_table(
            df_cronbach,
            value_col="cronbachs_alpha",
            caption_metric=caption_cronbach_article,
            save_path=output_dir / "cronbach_alpha_article_onestop.tex",
            metric_name="cronbach",
        )

    if split_half_path.exists():
        logger.info(f"Loading {split_half_path} for article tables")
        df_split_half = pd.read_csv(split_half_path)

        caption_split_half_article = (
            "Split-Half reliability for EyeScore and prediction models over per-article aggregation "
            "for OneStop, comparing Fixed and Any regimes."
        )
        generate_article_onestop_both_pools_table(
            df_split_half,
            value_col="mean_correlation",
            caption_metric=caption_split_half_article,
            save_path=output_dir / "split_half_article_onestop.tex",
            metric_name="split-half",
        )

    # Per-item split-half tables
    if cronbach_path.exists():
        logger.info(f"Loading {cronbach_path} for per-item split-half tables")
        df_per_item = pd.read_csv(cronbach_path)

        # Paragraph-level tables
        caption_per_item_unseen = (
            "Per-Item Split-Half reliability for EyeScore and prediction models over per-passage aggregation "
            "for the Any regime."
        )
        generate_paragraph_pool_combined_table(
            df_per_item,
            pool="unseen",
            value_col="split_half",
            caption_metric=caption_per_item_unseen,
            save_path=output_dir / "per_item_split_half_paragraph_unseen.tex",
            metric_name="per-item-split-half",
            datasets=datasets,
        )

        caption_per_item_seen = (
            "Per-Item Split-Half reliability for EyeScore and prediction models over per-passage aggregation "
            "for the Fixed regime."
        )
        generate_paragraph_pool_combined_table(
            df_per_item,
            pool="seen",
            value_col="split_half",
            caption_metric=caption_per_item_seen,
            save_path=output_dir / "per_item_split_half_paragraph_seen.tex",
            metric_name="per-item-split-half",
            datasets=datasets,
        )

        # Article-level table (OneStop only)
        caption_per_item_article = (
            "Per-Item Split-Half reliability for EyeScore and prediction models over per-article aggregation "
            "for OneStop, comparing Fixed and Any regimes."
        )
        generate_article_onestop_both_pools_table(
            df_per_item,
            value_col="split_half",
            caption_metric=caption_per_item_article,
            save_path=output_dir / "per_item_split_half_article_onestop.tex",
            metric_name="per-item-split-half",
        )

    # Combined reliability table (Cronbach + Split-Half)
    if cronbach_path.exists() and split_half_path.exists():
        logger.info("Generating combined reliability table...")
        df_cronbach = pd.read_csv(cronbach_path)
        df_split_half = pd.read_csv(split_half_path)
        generate_combined_reliability_table(
            df_cronbach,
            df_split_half,
            save_path=output_dir / "combined_reliability_paragraph_unseen.tex",
            datasets=datasets,
        )

    # EyeScore-only reliability table (Cronbach + Split-Half)
    if cronbach_path.exists() and split_half_path.exists():
        logger.info("Generating EyeScore reliability table...")
        df_cronbach = pd.read_csv(cronbach_path)
        df_split_half = pd.read_csv(split_half_path)
        generate_eyescore_reliability_combined_table(
            df_cronbach,
            df_split_half,
            save_path=output_dir / "eyescore_reliability_combined_unseen.tex",
            datasets=datasets,
        )
        # Same table for the Fixed Text regime -- same layout, same metrics,
        # the seen pool instead of the unseen one.
        generate_eyescore_reliability_combined_table(
            df_cronbach,
            df_split_half,
            save_path=output_dir / "eyescore_reliability_combined_seen.tex",
            datasets=datasets,
            pool="seen",
        )
        # Articles as test items, both regimes side by side. OneStop only:
        # MECO's texts have no article structure, so it has no article-level
        # rows and the generator drops it.
        generate_eyescore_reliability_regimes_table(
            df_cronbach,
            df_split_half,
            save_path=output_dir / "eyescore_reliability_article.tex",
            datasets=datasets,
        )
        # The same layout with passages as test items: one table carrying both
        # regimes and all three metrics, which is what the paper prints for
        # the Any regime in place of a Cronbach table and a split-half table
        # side by side.
        generate_eyescore_reliability_regimes_table(
            df_cronbach,
            df_split_half,
            save_path=output_dir / "eyescore_reliability_regimes_paragraph.tex",
            datasets=datasets,
            p_agg_level="paragraph",
        )

    logger.info("Table generation complete")

    # Generate new combined evaluation tables. These read
    # src/evaluation/**/all_evaluation_results.csv, whose own `preview` column
    # is filtered with `preview` so a Hunting run reports Hunting numbers.
    logger.info("Generating new combined evaluation tables...")
    generate_evaluation_eyescore_combined(
        output_dir / "evaluation_eyescore.tex", preview=preview,
        datasets=datasets)
    # ...and the same numbers with passages as the only test item, for when
    # the paper reports the two corpora side by side and the OneStop Article
    # block is a third unit the comparison does not need.
    generate_evaluation_eyescore_combined(
        output_dir / "evaluation_eyescore_paragraph.tex", preview=preview,
        datasets=datasets, levels=("paragraph",))
    # ...and both again as the difference from the original, fully aggregated
    # EyeScore, so the cost of per-item scoring can be read off directly
    # instead of subtracted cell by cell from the main EyeScore table.
    generate_evaluation_eyescore_combined(
        output_dir / "evaluation_eyescore_delta.tex", preview=preview,
        datasets=datasets, delta=True)
    generate_evaluation_eyescore_combined(
        output_dir / "evaluation_eyescore_delta_paragraph.tex", preview=preview,
        datasets=datasets, levels=("paragraph",), delta=True)
    generate_evaluation_predictions_onestop(
        output_dir / "evaluation_predictions_onestop.tex", preview=preview)
    # ...and the same numbers split one level per table, each sized to a text
    # column, for when the two-level table is too wide to place.
    for level in ("paragraph", "article"):
        generate_evaluation_predictions_onestop(
            output_dir / f"evaluation_predictions_onestop_{level}.tex",
            preview=preview, levels=(level,))
    # MECO-only: no preview axis, so a non-Gathering run would just re-emit the
    # Gathering table under the preview's regime label.
    if preview == DEFAULT_PREVIEW:
        generate_evaluation_predictions_meco(
            output_dir / "evaluation_predictions_meco.tex")
    else:
        logger.info(
            "Skipping the MECO predictions evaluation table for preview=%s: "
            "MECO has no preview axis.", preview)

    # Generate new combined reliability tables
    logger.info("Generating new combined reliability tables...")
    generate_reliability_eyescore_cronbach(
        output_dir / "eyescore_cronbach_seen_unseen.tex", cronbach_path,
        datasets=datasets)
    generate_reliability_eyescore_splithalf(
        output_dir / "eyescore_split_half_seen_unseen.tex", split_half_path,
        datasets=datasets)
    generate_reliability_predictions_cronbach(
        output_dir / "predictions_cronbach_seen_unseen.tex", cronbach_path,
        datasets=datasets)
    # ...and one table per block -- MECO Passage, OneStop Passage, OneStop
    # Article -- each sized to a text column, for when the all-blocks table is
    # too wide to place. Split by corpus as well as by level: a per-level
    # table still carried both corpora at the passage level, so it spanned the
    # page like the combined one.
    for dataset, level, _ in seen_unseen_sections(datasets):
        generate_reliability_predictions_cronbach(
            output_dir / (f"predictions_cronbach_seen_unseen_{dataset.lower()}"
                          f"_{LEVEL_DISPLAY[level].lower()}.tex"),
            cronbach_path, datasets=(dataset,), levels=(level,))
    generate_reliability_predictions_splithalf(
        output_dir / "predictions_split_half_seen_unseen.tex", split_half_path,
        datasets=datasets)

    # All-variants ICC tables (skipped with a warning if the CSV predates the columns)
    logger.info("Generating ICC tables...")
    for icc_source in ["eyescore", "predictions"]:
        # Combined table (every dataset in one)...
        generate_reliability_icc_table(
            output_dir / f"icc_{icc_source}.tex", icc_source, None, cronbach_path,
            datasets=datasets,
        )
        # ...and the per-dataset split, where MECO also carries the all-readers rows.
        for icc_dataset in datasets:
            generate_reliability_icc_table(
                output_dir / f"icc_{icc_source}_{icc_dataset.lower()}.tex",
                icc_source, icc_dataset, cronbach_path,
            )

    # Non-default previews get their regime spelled out in every caption and
    # appended to every label.
    if preview != DEFAULT_PREVIEW:
        annotate_preview(output_dir, preview)

    # This run only writes output_dir; paper/ is written by src.run.paper alone.
    logger.info(
        "Tables written to %s. To render the paper's versions into paper/, run "
        "python -m src.run.paper --only reliability", output_dir)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate reliability LaTeX tables")
    parser.add_argument(
        "--preview",
        default=DEFAULT_PREVIEW,
        help="OneStop preview (Gathering|Hunting). Gathering reads and writes the "
        "top-level paths; any other preview uses a parallel subtree (.../hunting/).",
    )
    args = parser.parse_args()
    main(preview=args.preview)
