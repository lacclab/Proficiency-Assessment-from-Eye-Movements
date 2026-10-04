"""Caption and label rules shared by every LaTeX table generator.

Each generator lays out its own headers and writes its own caption prose, but
the corpora they name are the same everywhere. Keeping the naming rules here
means a rename lands in every table family at once instead of being chased
through fifteen caption strings and a dozen ``dataset_display`` maps.

PAPER_PINS records, for the tables the paper prints, the ``\\label`` and
``\\caption`` they carry there where those differ from the generated ones.
src/run/paper applies them when it renders paper/, so the tables there print
the paper's wording.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Both corpora are read here in their L2 release, and a bare "MECO"/"OneStop"
# names the wrong one. Applied to the whole table -- caption, column headers and
# row labels alike -- so a table never spells the corpus one way in its header
# and another below it. The lookbehinds keep a hypothetical \MECO macro intact;
# the lookaheads make the pass idempotent. "OneStopL2" and "OneStopQA" are not
# matched: \b needs a non-word character after the name.
DATASET_RENAMES = (
    (re.compile(r"(?<!\\)\bMECO\b(?!\s*L2)"), "MECO L2"),
    (re.compile(r"(?<!\\)\bOneStop\b(?!\s*L2)"), "OneStopL2"),
)


def rename_datasets_in_tables(latex: str) -> str:
    r"""Apply DATASET_RENAMES everywhere in a finished table.

    Safe on ``\label``s and file paths, which spell the corpus in lower case.
    """
    for pattern, replacement in DATASET_RENAMES:
        latex = pattern.sub(replacement, latex)
    return latex


def write_table(out_path: Path, latex: str) -> None:
    """Write a finished table, applying the shared dataset-naming rules."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rename_datasets_in_tables(latex))


# The \caption and \label lines of a finished table.
CAPTION_RE = re.compile(r"^\\caption\{.*\}\s*$", re.MULTILINE)
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")


@dataclass(frozen=True)
class PaperPin:
    """What a table carries in the paper where it differs from the generated
    table; None keeps the generated one."""
    # \label the paper's \ref sites use. The generators keep their own names
    # (the _noci suffix separates the two dispersion variants).
    label: str | None = None
    # \caption, for tables whose generated caption is shared by a whole family
    # of outputs (every model's bootstrap table, the Hunting variants, the ± and
    # no-± pair) and so cannot be reworded for the paper alone. Where a caption
    # belongs to one table only, it is edited in the generator instead.
    caption: str | None = None


# Captions as the paper prints them, for the four tables whose generated
# caption is written once and reused by a whole family of outputs. Editing
# the generator would reword every model-comparison and Hunting table with
# it, so the paper wording is pinned here and applied on copy -- the same
# treatment the \label gets, and reviewed the same way.
CAPTION_TABLE1 = (
    "\\caption{\\textbf{EyeScore}. Pearson $r$ correlations between EyeScore "
    "and standard English proficiency tests. Presented are mean $r$ "
    "correlations using bootstrap (100,000 paired resamples). "
    "Statistically significant differences from the WPM baseline using a "
    "bootstrap test are marked with `$^{*}$' $p<0.05$, `$^{**}$' $p<0.01$, "
    "`$^{***}$' $p<0.001$. ``Fixed'' is the Fixed Text regime in which all "
    "the eye movement data is for the same texts presented to the test "
    "participant. ``Any'' is the Any Text regime in which no eye movement "
    "data is available for the texts of the test participant. The best "
    "result for each eye tracking dataset, regime, and target proficiency "
    "test is marked in bold.}"
)

CAPTION_TABLE2 = (
    "\\caption{\\textbf{Prediction of standard English proficiency test "
    "scores}. Pearson $r$ and Mean Absolute Error (MAE) for prediction of "
    "external proficiency scores using eye movements with Ridge "
    "regression. Avg. Train is a baseline which assigns test participants "
    "with the average standard proficiency score in the training set. "
    "Presented are mean values using bootstrap (100,000 paired resamples). "
    "Statistically significant differences from the WPM baseline using a "
    "bootstrap test are marked with `$^{*}$' $p<0.05$, `$^{**}$' $p<0.01$, "
    "`$^{***}$' $p<0.001$. ``Fixed'' is the Fixed Text regime in which all "
    "the eye movement data is for the same texts presented to the test "
    "participant. ``Any'' is the Any Text regime in which no eye movement "
    "data is available for the texts of the test participant. The best "
    "result for each evaluation measure in each eye tracking dataset, "
    "regime and target proficiency test is marked in bold.}"
)

CAPTION_TABPFN_MECO = (
    "\\caption{\\textbf{Prediction of standard English proficiency test "
    "scores} with \\textbf{TabPFN-3} on \\textbf{MECO L2}.}"
)

CAPTION_TABPFN_ONESTOP = (
    "\\caption{\\textbf{Prediction of standard English proficiency test "
    "scores} with \\textbf{TabPFN-3} on \\textbf{OneStopL2}.}"
)


# The paper's tables, by their file in the submission.
PAPER_PINS = {
    "tables/table1_eyescore_proficiency_pearson_r.tex":
        PaperPin(label="tab:eyescore_seen_unseen_bootstrap", caption=CAPTION_TABLE1),
    "tables/table2_predictions.tex":
        PaperPin(label="tab:predictions_bootstrap", caption=CAPTION_TABLE2),
    "tables/appendix/additioinal_prediction_models/meco/table2_tabPFN.tex":
        PaperPin(label="tab:predictions_bootstrap_tabpfn_meco", caption=CAPTION_TABPFN_MECO),
    "tables/appendix/additioinal_prediction_models/onestop/table2_tabPFN.tex":
        PaperPin(label="tab:predictions_bootstrap_tabpfn_onestop", caption=CAPTION_TABPFN_ONESTOP),
    "tables/table4_eyescore_reliability.tex": PaperPin(),
    "tables/appendix/evaluation/evaluation_eyescore_delta.tex":
        PaperPin(label="tab:evaluation-eyescore-delta"),
    "tables/appendix/reliability/correlation/correlations_eyescore.tex":
        PaperPin(label="tab:original_vs_per_item_eyescore"),
    "tables/appendix/reliability/eyescore_reliablity_fixed.tex": PaperPin(),
    "tables/appendix/reliability/hunting/eyescore_reliability.tex": PaperPin(),
}
