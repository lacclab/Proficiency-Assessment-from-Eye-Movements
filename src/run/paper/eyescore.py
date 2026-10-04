"""The EyeScore tables: paper/eyescore/{main,hunting,comprehension}.tex.

    main.tex            MECO + OneStop/Gathering, proficiency tests
    hunting.tex         OneStop/Information Seeking, proficiency tests
    comprehension.tex   MECO + OneStop/Gathering, reading comprehension accuracy

Paper format = bootstrap means over 100,000 paired resamples, two decimals, no
dispersion term, single-column `table` environment. All come from
`generate_bootstrap_combined_table` over the EyeScore evaluation CSVs written by
`python -m src.run.eyescore --stage evaluation`. main.tex is Table 1 of the paper
and carries the paper's caption and label (see published.py).
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from loguru import logger

from src.constants import TestCols
from src.evaluation.EyeScore.tables import (
    BOOTSTRAP_DATASET_DISPLAY,
    generate_bootstrap_combined_table,
)
from src.latex_captions import write_table
from src.run.paper.published import write_published

EVAL_CSV = Path("src/evaluation/EyeScore/results/all_evaluation_results.csv")
PAIRWISE_CSV = Path("src/evaluation/EyeScore/results/pairwise_significance.csv")

FEATURE_SETS = [
    "READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM",
    "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC",
]
ONESTOP_VERSION_PATH = "fully_agg/ordinary/all"

# table -> (path under paper/, slice filters, datasets, label, reading-condition phrase)
TABLES = {
    "main": ("eyescore/main.tex",
             {"Meco": ("All", "fully_agg/all/all"),
              "OneStop": ("Gathering", ONESTOP_VERSION_PATH)},
             ["Meco", "OneStop"], "tab:eyescore_seen_unseen_bootstrap", ""),
    "hunting": ("eyescore/hunting.tex",
                {"OneStop": ("Hunting", ONESTOP_VERSION_PATH)},
                ["OneStop"], "tab:eyescore_seen_unseen_hunting_bootstrap",
                "\\textit{Information Seeking}"),
    "comprehension": ("eyescore/comprehension.tex",
                      {"Meco": ("All", "fully_agg/all/all"),
                       "OneStop": ("Gathering", ONESTOP_VERSION_PATH)},
                      ["Meco", "OneStop"], "tab:eyescore_comprehension_bootstrap", ""),
}

# Tables the submission carries: table -> its file there (its PAPER_PINS key).
PUBLISHED_AS = {"main": "tables/table1_eyescore_proficiency_pearson_r.tex"}

# Tables scored against a target other than the proficiency tests: one target
# per dataset, and the generator's proficiency-test caption is reworded to match.
TARGET_OVERRIDE = {"comprehension": str(TestCols.COMPREHENSION_COL)}
# Longest phrase first: "standard language proficiency tests" would otherwise
# leave a stranded "standard".
TARGET_CAPTION_REWRITES = {
    "comprehension": [
        ("standard language proficiency tests", "reading comprehension accuracy"),
        ("language proficiency tests", "reading comprehension accuracy"),
        ("in each regime, dataset, and target proficiency test",
         "in each regime and dataset"),
        # One target per dataset, so there is no proficiency test to name.
        ("for the corresponding regime and proficiency test",
         "for the corresponding regime and dataset"),
    ],
}


# Per-table paper styling. The Hunting table is half the width of the combined
# one, so it is set at its natural size under \small rather than scaled by
# \resizebox (scaling a narrow table blows the digits up out of proportion to
# the body text), and the header rule under the target names is dropped -- with
# two targets the column groups are already unambiguous. The \resizebox is
# left in the file, commented out, so it can be switched back without
# regenerating.
NATURAL_SIZE = {"hunting"}

# Captions that replace the generated one wholesale. The Hunting table sits
# beside the combined table, which spells out the stars, the two regimes and
# the bolding rule, so repeating all of it here only adds length.
CAPTION_OVERRIDE = {
    "hunting":
        "\\caption{\\textbf{EyeScore}. Pearson $r$ correlations between "
        "EyeScore and standard language proficiency tests. Eye movement data "
        "is from the \\textbf{Information Seeking} portion of "
        "\\textbf{OneStopL2}.}",
}

_CAPTION_RE = re.compile(r"^\\caption\{.*\}$", re.MULTILINE)


def apply_paper_style(latex: str, table: str) -> str:
    """Natural-size layout and short caption, for the tables that want them."""
    if table in NATURAL_SIZE:
        latex = latex.replace("% \\small", " \\small", 1)
        latex = latex.replace("\\resizebox{\\columnwidth}{!}{",
                              "%\\resizebox{\\columnwidth}{!}{", 1)
        # The lone closing brace of the \resizebox, on its own line after the
        # tabular. Commented out with the \resizebox it belongs to.
        latex = latex.replace("\\end{tabular}\n}\n", "\\end{tabular}\n%}\n", 1)
        latex = re.sub(r" \\cmidrule\{2-\d+\}", "", latex)
    caption = CAPTION_OVERRIDE.get(table)
    if caption:
        latex = _CAPTION_RE.sub(lambda _m: caption, latex, count=1)
    return latex


def build(table: str) -> "str | None":
    """The finished LaTeX for one of TABLES, or None if no rows matched."""
    _, slice_filters, datasets, label, condition = TABLES[table]
    target = TARGET_OVERRIDE.get(table)
    scratch = Path(tempfile.mkdtemp(prefix="paper_eyescore_"))
    latex = generate_bootstrap_combined_table(
        eval_csv=EVAL_CSV,
        pairwise_csv=PAIRWISE_CSV,
        save_dir=scratch,
        feature_sets=FEATURE_SETS,
        decimals=2,
        slice_filters_override=slice_filters,
        datasets_override=datasets,
        show_dispersion=False,
        single_column=True,
        target_cols_override={ds: [target] for ds in datasets} if target else None,
    )
    if not latex:
        logger.warning("no rows matched for {}; skipped", table)
        return None
    if condition:
        latex = latex.replace(
            "correlations between EyeScore and standard language proficiency "
            "tests",
            "correlations between EyeScore and standard language proficiency "
            "tests, on the "
            f"{BOOTSTRAP_DATASET_DISPLAY['OneStop']} {condition} trials", 1)
    for old, new in TARGET_CAPTION_REWRITES.get(table, []):
        latex = latex.replace(old, new)
    if len(datasets) == 1:
        # "dataset" only earns a mention when more than one is in the table.
        latex = latex.replace(
            "in each regime, dataset, and target proficiency test",
            "in each regime and target proficiency test", 1)
    latex = re.sub(r"\\label\{tab:eyescore_seen_unseen[^}]*\}",
                   lambda _m: f"\\label{{{label}}}", latex)
    return apply_paper_style(latex, table)


def render(out_root: Path) -> list[Path]:
    written = []
    for table, (relpath, *_rest) in TABLES.items():
        latex = build(table)
        if latex is None:
            continue
        out = Path(out_root) / relpath
        if table in PUBLISHED_AS:
            write_published(out, latex, PUBLISHED_AS[table])
        else:
            write_table(out, latex)
        written.append(out)
    return written
