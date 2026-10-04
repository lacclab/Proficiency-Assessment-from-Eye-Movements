"""The prediction tables: paper/prediction/ and paper/model_comparison/.

    prediction/main.tex                       Ridge, MECO + OneStop/Gathering
    prediction/hunting.tex                    Ridge, OneStop/Information Seeking
    model_comparison/<slice>/<model>.tex      one per published model and slice
    model_comparison/{meco,onestop}/tabpfn.tex   TabPFN, one per dataset

All come from the same renderer (`generate_bootstrap_combined_latex`) over the
prediction evaluation CSVs written by `python -m src.run.predictions --stage
evaluation`, so a caption change lands everywhere at once. They differ only in
slice, in whether the model name goes in the title, and in whether the vs-Ridge
subscripts are drawn.

Paper format = bootstrap means over 100,000 paired resamples, two decimals, no
dispersion term. main.tex is Table 2 of the paper and the TabPFN tables are in
its appendix; those three carry the paper's caption and label (see
published.py), and main.tex keeps the `@{\\hspace{4pt}}` metric padding the
published table has. The Hunting table drops that padding; the
model-comparison tables keep it, because their cells carry a subscript and
need the breathing room.

The TabPFN tables are the evaluation stage's own per-dataset tables for that
model (built here in a scratch directory by the same function), because that is
what the submission prints. Their evaluation rows come from a separate TabPFN
run; without them in the evaluation CSV the two files are skipped.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.predictions.tables import (
    BASELINE_ROW_MODEL,
    BOOTSTRAP_DATASET_DISPLAY,
    COMPARISON_SIGNIFICANCE_LEGEND,
    RIDGE_SUBSCRIPT_LEGEND,
    SIGNIFICANCE_LEGEND,
    caption_model_bold,
    generate_bootstrap_combined_latex,
    generate_full_combined_tables,
)
from src.latex_captions import write_table
from src.run.paper.published import write_published

EVAL_CSV = Path("src/evaluation/predictions/results/all_evaluation_results.csv")
PAIRWISE_WPM_CSV = Path("src/evaluation/predictions/results/pairwise_significance_wpm.csv")
PAIRWISE_RIDGE_CSV = Path("src/evaluation/predictions/results/pairwise_significance_ridge.csv")

AUG_P = "aug_p=1.0"
VERSION_PATH = "fully_agg"
SCOPES = ["Fixed", "Any"]
RIDGE = "Ridge_Classifier"

TARGET_COLS = {"Meco": ["lextale_score", "proficiency_agg"],
               "OneStop": ["lextale_score", "michtest_score"]}
TARGET_LABEL = {"lextale_score": "LexTALE", "proficiency_agg": "Composite",
                "michtest_score": "Michigan"}

# Which (dataset, preview) pair each slice name selects.
SLICES = {
    "combined": {"Meco": "all", "OneStop": "gathering"},
    "meco": {"Meco": "all"},
    "onestop": {"OneStop": "gathering"},
    "hunting": {"OneStop": "hunting"},
}

# (path under paper/, slice, label, reading-condition phrase, single column,
#  file in the submission or None)
MAIN_TABLES = [
    ("prediction/main.tex", "combined", "tab:predictions_bootstrap", "", False,
     "tables/table2_predictions.tex"),
    # OneStop-only, so half the width of the combined one: it sits in a column.
    ("prediction/hunting.tex", "hunting", "tab:predictions_hunting_bootstrap",
     "\\textit{Information Seeking}", True, None),
]

# The TabPFN appendix tables: dataset -> (path under paper/, file in the submission).
TABPFN = "TabPFN"
TABPFN_TABLES = {
    "Meco": ("model_comparison/meco/tabpfn.tex",
             "tables/appendix/additioinal_prediction_models/meco/table2_tabPFN.tex"),
    "OneStop": ("model_comparison/onestop/tabpfn.tex",
                "tables/appendix/additioinal_prediction_models/onestop/table2_tabPFN.tex"),
}

# The published model-comparison tables: file slug -> model_name in the
# evaluation CSV. The renderer handles any model (Ridge, the comparison
# baseline, simply gets no subscripts); the paper reports the tuned LightGBM.
MODEL_COMPARISON = {
    "lgbm_holdout": "LightGBM_holdout",
    "lgbm_kfold": "LightGBM_kfold",
}
MODEL_COMPARISON_SLICES = ("combined", "meco", "onestop")

# pool__fold-method -> the text regime it is reported as.
METHOD_TO_SCOPE = {"seen__pool_all": "Fixed", "unseen__pool_all": "Any"}


def eval_slice(df: pd.DataFrame, previews: dict, model_name: str) -> dict:
    """{dataset: rows} for one (slice, model), or {} if no dataset has rows.

    A dataset without rows (OneStop, where only the MECO results exist) keeps
    its columns in the table, drawn as missing cells.
    """
    out = {}
    for dataset, preview in previews.items():
        rows = df[(df["dataset"] == dataset) & (df["aug_p"] == AUG_P)
                  & (df["preview"] == preview)
                  & (df["version_path"] == VERSION_PATH)
                  & (df["model_name"] == model_name)
                  & (df["target_col"].isin(TARGET_COLS[dataset]))]
        if rows.empty:
            logger.warning("no rows for {}/{}/{}", dataset, preview, model_name)
        out[dataset] = rows
    return out if any(not rows.empty for rows in out.values()) else {}


def baseline_slice(df: pd.DataFrame, previews: dict) -> pd.DataFrame:
    """Evaluation rows for the featureless mean baseline in this table's slice.

    Best-effort per dataset, unlike ``eval_slice``: a slice the baseline has
    not been run for (the Hunting preview, say) loses its half of the row
    rather than the whole table. The feature set is not filtered — the model
    ignores the features, so whatever set it was run under is the same row.
    """
    parts = []
    for dataset, preview in previews.items():
        rows = df[(df["dataset"] == dataset) & (df["aug_p"] == AUG_P)
                  & (df["preview"] == preview)
                  & (df["version_path"] == VERSION_PATH)
                  & (df["model_name"] == BASELINE_ROW_MODEL)
                  & (df["target_col"].isin(TARGET_COLS[dataset]))]
        if rows.empty:
            logger.warning(
                "no {} rows for {}/{} — that dataset's cells in the mean-"
                "baseline row will be blank; run the {} model for this slice",
                BASELINE_ROW_MODEL, dataset, preview, BASELINE_ROW_MODEL)
            continue
        parts.append(rows)
    return pd.concat(parts) if parts else pd.DataFrame()


def baseline_caption_sentence(df: pd.DataFrame, previews: dict,
                              decimals: int = 2) -> str:
    """One caption sentence reporting the mean-baseline MAE for a table's slice.

    ``previews`` is {dataset: preview}, the same mapping the table was built
    from, so the sentence always quotes the cohort the table actually shows
    (Gathering and Information Seeking have different baselines).

    Both regimes are quoted as ``Fixed/Any``: the baseline ignores the text
    entirely, but the regime still changes which participants land in each
    training fold, so the two are not identical. Returns "" when the baseline
    has not been evaluated for that slice, which keeps the caption valid rather
    than quoting a number that does not exist.

    Read from the evaluation CSV like every other number in the table. Its rows
    are pinned to READING_SPEED: the model ignores the features, but the set
    still decides which participants survive filtering (see AvgModel).
    """
    df = df[(df["model_name"] == BASELINE_ROW_MODEL) & (df["aug_p"] == AUG_P)
            & (df["version_path"] == VERSION_PATH)
            & (df["feature_set"] == "READING_SPEED")
            & (df["method"].isin(METHOD_TO_SCOPE))]
    df = df.assign(scope=df["method"].map(METHOD_TO_SCOPE))

    clauses = []
    for dataset, preview in previews.items():
        sub = df[(df["dataset"] == dataset) & (df["preview"] == preview)]
        parts = []
        for target in TARGET_COLS[dataset]:
            cells = {}
            for scope in SCOPES:
                row = sub[(sub["target_col"] == target) & (sub["scope"] == scope)]
                if not row.empty:
                    cells[scope] = float(row["mae"].iloc[0])
            if len(cells) != len(SCOPES):
                continue
            nums = "/".join(f"${cells[s]:.{decimals}f}$" for s in SCOPES)
            parts.append(f"{nums} ({TARGET_LABEL[target]})")
        if parts:
            ds_name = BOOTSTRAP_DATASET_DISPLAY.get(dataset, dataset)
            clauses.append(f"on {ds_name}, " + " and ".join(parts))
    if not clauses:
        return ""

    return (" For reference, a featureless baseline that predicts the mean "
            "target of the training fold, evaluated under the same "
            "cross-validation, obtains the following MAE "
            f"({'/'.join(SCOPES)}): " + "; ".join(clauses) + ".")


def render_table(previews: dict, model_name: str, eval_df: pd.DataFrame,
                 pairwise_df: pd.DataFrame, ridge_df: pd.DataFrame | None,
                 include_model_in_title: bool, strip_metric_padding: bool,
                 label: str, title_condition: str = "",
                 caption_override: str = "",
                 single_column: bool = False,
                 baseline_df: pd.DataFrame | None = None) -> "str | None":
    dfs = eval_slice(eval_df, previews, model_name)
    if not dfs:
        return None
    latex = generate_bootstrap_combined_latex(
        dfs,
        target_cols={ds: TARGET_COLS[ds] for ds in dfs},
        scopes=SCOPES,
        pairwise_df=pairwise_df,
        model_name=model_name,
        aug_p=AUG_P,
        decimals=2,
        show_dispersion=False,
        ridge_df=ridge_df,
        include_model_in_title=include_model_in_title,
        baseline_mae_note=("" if caption_override
                           else baseline_caption_sentence(eval_df, previews)),
        caption_override=caption_override,
        single_column=single_column,
        baseline_df=baseline_df,
    )
    if not latex:
        logger.error("empty LaTeX for {} / {}", label, model_name)
        return None
    if strip_metric_padding:
        latex = latex.replace("@{\\hspace{4pt}}", "")
    if title_condition:
        # The renderer already put " on <Dataset>" in the title of a
        # single-dataset table, so the condition is appended to that rather
        # than naming the dataset a second time.
        latex = latex.replace(
            ", bootstrap estimates}", f" {title_condition}, "
            "bootstrap estimates}", 1)
        latex = latex.replace(
            "using eye movements with ",
            "using eye movements on the "
            f"{BOOTSTRAP_DATASET_DISPLAY['OneStop']} {title_condition} "
            "trials, with ", 1)
    latex = re.sub(r"\\label\{tab:predictions_bootstrap[^}]*\}",
                   lambda _m: f"\\label{{{label}}}", latex)
    return latex


def render(out_root: Path) -> list[Path]:
    out_root = Path(out_root)
    eval_df = pd.read_csv(EVAL_CSV)
    pairwise_df = pd.read_csv(PAIRWISE_WPM_CSV)
    ridge_df = (pd.read_csv(PAIRWISE_RIDGE_CSV)
                if PAIRWISE_RIDGE_CSV.exists() else None)
    if ridge_df is None:
        logger.warning("{} missing — model-comparison tables will carry no "
                       "vs-Ridge subscripts", PAIRWISE_RIDGE_CSV)

    written = []

    # ── the two main paper tables: Ridge, no model name in the title ────────
    for relpath, slice_name, label, condition, single_col, published_as in MAIN_TABLES:
        latex = render_table(SLICES[slice_name], RIDGE, eval_df, pairwise_df,
                             ridge_df=None, include_model_in_title=False,
                             strip_metric_padding=published_as is None, label=label,
                             title_condition=condition, single_column=single_col,
                             baseline_df=baseline_slice(eval_df, SLICES[slice_name]))
        if latex is None:
            continue
        out = out_root / relpath
        if published_as:
            write_published(out, latex, published_as)
        else:
            write_table(out, latex)
        written.append(out)

    # ── model comparison: model name in the title, vs-Ridge subscripts ──────
    # The mean baseline depends on the slice, not on the model, so every table
    # in a slice gets the identical row; resolve it once per slice.
    baselines = {name: baseline_slice(eval_df, SLICES[name])
                 for name in MODEL_COMPARISON_SLICES}
    for slice_name in MODEL_COMPARISON_SLICES:
        for slug, model_name in MODEL_COMPARISON.items():
            label = f"tab:predictions_bootstrap_{slug}"
            if slice_name != "combined":
                label += f"_{slice_name}"
            # Short caption: these are a run of near-identical tables, so the
            # shared machinery is not repeated in every caption.
            scope_phrase = ""
            if len(SLICES[slice_name]) == 1:
                only_ds = next(iter(SLICES[slice_name]))
                scope_phrase = (" on \\textbf{"
                                + BOOTSTRAP_DATASET_DISPLAY[only_ds] + "}")
            # The legend travels with the table: a reader meeting a marked
            # cell in print has nowhere else to look up what it means.
            # Ridge is the comparison baseline, so it has no subscript to
            # explain and takes the plain vs-WPM legend.
            legend = (SIGNIFICANCE_LEGEND if model_name == RIDGE else
                      RIDGE_SUBSCRIPT_LEGEND + " "
                      + COMPARISON_SIGNIFICANCE_LEGEND)
            caption = ("Prediction of standard proficiency test scores with "
                       + caption_model_bold(model_name) + scope_phrase + ". "
                       + legend)
            latex = render_table(SLICES[slice_name], model_name, eval_df, pairwise_df,
                                 ridge_df=None if model_name == RIDGE else ridge_df,
                                 include_model_in_title=True,
                                 strip_metric_padding=False, label=label,
                                 caption_override=caption,
                                 baseline_df=baselines[slice_name])
            if latex is None:
                continue
            out = out_root / "model_comparison" / slice_name / f"{slug}.tex"
            write_table(out, latex)
            written.append(out)

    written += render_tabpfn(out_root, eval_df)
    return written


def render_tabpfn(out_root: Path, eval_df: pd.DataFrame) -> list[Path]:
    """The TabPFN appendix tables: the evaluation stage's per-dataset bootstrap
    tables for that model (no dispersion term), with the paper's caption and label."""
    if not (eval_df["model_name"] == TABPFN).any():
        logger.warning("no {} rows in {} -- the TabPFN appendix tables are skipped; "
                       "render them where the TabPFN evaluation exists", TABPFN, EVAL_CSV)
        return []
    scratch = Path(tempfile.mkdtemp(prefix="paper_tabpfn_"))
    # The evaluation stage's paper configuration: MECO (all) beside
    # OneStop (gathering), written under combined_preview/.
    generate_full_combined_tables(eval_csv=EVAL_CSV, save_dir=scratch,
                                  model_name=TABPFN, pair_across_preview=True)
    written = []
    for dataset, (relpath, published_as) in TABPFN_TABLES.items():
        matches = sorted(scratch.rglob(
            f"{AUG_P}/combined_preview/{VERSION_PATH}/{TABPFN}/{dataset}/"
            f"combined_bootstrap_noci_{dataset.lower()}.tex"))
        if not matches:
            logger.warning("no {} table for {} under {}", TABPFN, dataset, scratch)
            continue
        out = out_root / relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        # Already written with the dataset renames; only the paper's caption and
        # label are left to apply (write_published re-applies the renames, which
        # leaves an already-renamed table as it is).
        write_published(out, matches[0].read_text(), published_as)
        written.append(out)
    return written
