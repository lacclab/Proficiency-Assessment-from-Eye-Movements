"""
Table: agreement between the ORIGINAL score and the per-item-mean score.

For each participant we pair the fully-aggregated score (what the paper
reports) with the mean of their per-item scores, and correlate the two across
participants. A high correlation is the license to read the per-item
reliability analysis as being about the original score.

Emits a CSV (all statistics) and a LaTeX table (Pearson r, one row per feature
set, one column per dataset/level/regime), following the project's table
formatting rules.

Usage (from repo root):
    python -m src.reliability.table_original_vs_per_item
    python -m src.reliability.table_original_vs_per_item --source predictions
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.latex_captions import rename_datasets_in_tables
from src.reliability.utils import (
    FEATURE_SET_DISPLAY,
    FEATURE_SETS,
    load_original,
    load_per_item_mean,
)
from src.reliability.tables import (
    DEFAULT_PREVIEW,
    annotate_preview,
    header_block_cells,
    should_add_cline_after_feature,
)

OUT_DIR = Path("src/reliability/results/per_item_reliability/tables")

# (dataset, level) -> column group; MECO left of OneStop, Passage left of Article
COLUMN_GROUPS = [
    ("Meco", "paragraph"),
    ("OneStop", "paragraph"),
    ("OneStop", "article"),
]
# Fixed left of Any
POOLS = [("seen", "Fixed"), ("unseen", "Any")]
DATASET_DISPLAY = {"Meco": "MECO", "OneStop": "OneStop"}
LEVEL_DISPLAY = {"paragraph": "Passage", "article": "Article"}


def column_groups(preview: str, levels=None) -> list:
    """MECO has no preview axis, so a non-default preview is OneStop-only.

    ``levels`` further keeps only the aggregation levels it names; left at
    None the table spans every level.
    """
    groups = COLUMN_GROUPS if preview == DEFAULT_PREVIEW else [
        (dataset, level) for dataset, level in COLUMN_GROUPS if dataset != "Meco"]
    if levels is not None:
        groups = [(dataset, level) for dataset, level in groups if level in levels]
    return groups


def level_slug(levels) -> str:
    """Filename/label suffix naming the level a narrowed table covers ("" if
    it covers more than one, where no single level names it)."""
    if levels is None:
        return ""
    covered = list(dict.fromkeys(
        level for _, level in COLUMN_GROUPS if level in levels))
    return f"_{LEVEL_DISPLAY[covered[0]].lower()}" if len(covered) == 1 else ""


def collect(source: str, target: str, model: str,
            preview: str = DEFAULT_PREVIEW, levels=None) -> pd.DataFrame:
    """One row per (dataset, level, pool, feature set): the Pearson and Spearman
    agreement between original and per-item-mean scores, over participants."""
    rows = []
    for dataset, level in column_groups(preview, levels):
        for pool, _ in POOLS:
            # MECO has no preview axis: its tree lives under All/ whatever the
            # OneStop preview is, so leave it on the loaders' own default.
            ds_preview = preview if dataset == "OneStop" else None
            for fs in FEATURE_SETS:
                orig = load_original(source, dataset, pool, level, fs, target, model,
                                     preview=ds_preview)
                per_item = load_per_item_mean(source, dataset, pool, level, fs, target,
                                              model, preview=ds_preview)
                if orig is None or per_item is None:
                    continue
                both = pd.concat([orig, per_item], axis=1).dropna()
                if len(both) < 3:
                    continue
                rows.append({
                    "dataset": dataset,
                    "level": level,
                    "pool": pool,
                    "feature_set": fs,
                    "n": len(both),
                    "pearson": both["original"].corr(both["per_item_mean"]),
                    "spearman": both["original"].corr(both["per_item_mean"], method="spearman"),
                })
    return pd.DataFrame(rows)


def format_r(value: float, decimals: int, missing: str = "-") -> str:
    """2 decimals, like every other reliability table; ``--decimals 3`` pulls
    apart the cells that round to the same value up here near 1.0."""
    if pd.isna(value) or not -1 <= value <= 1:
        return missing
    return f"{value:.{decimals}f}"


def generate_latex(df: pd.DataFrame, source: str, save_path: Path, decimals: int,
                   preview: str = DEFAULT_PREVIEW, target: str = "primary",
                   levels=None) -> None:
    """The LaTeX table: Pearson r per feature set (rows) and dataset/level/regime."""
    groups = column_groups(preview, levels)
    # A rule between the dataset blocks, as in every other table here, with
    # normal column padding (a hand-tightened spec runs the "Any" and "Fixed"
    # headings of adjacent blocks together).
    col_spec = "l" + "|".join(["c" * len(POOLS) for _ in groups])
    # The predictions table spans both columns in the paper, the EyeScore one
    # sits in a single column -- same shape, different homes -- so the float
    # and the \resizebox target follow the source rather than a shared default.
    full_width = source == "predictions"
    # No \resizebox on the single-column table: it has five columns and fits a
    # text column as it stands, and \resizebox{\columnwidth} does not only
    # shrink -- it stretches whatever it is given to exactly that width. On a
    # table this narrow that is a ~1.6x enlargement, which prints the digits
    # larger than the body text and, worse, magnifies the ~1pt gaps booktabs
    # leaves where a \midrule crosses a vertical rule into visible breaks.
    # The wide predictions table really is wider than its float and is scaled
    # down, which is the direction \resizebox is safe in.
    lines = [
        r"\begin{table*}[ht!]" if full_width else r"\begin{table}[ht!]",
        r"\centering",
        # \small rather than a \resizebox: it sets the type one step down and
        # leaves it there, so the table reads at a size the page already uses
        # instead of whatever ratio its natural width happens to make.
        r"\small",
    ]
    if full_width:
        lines.append(r"\resizebox{\textwidth}{!}{")
    lines += [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Passages are the default notion of a test item, so a table that covers
    # a single level names the corpus alone in its header -- "MECO L2", not
    # "MECO L2 Passage" -- and the caption says which item it used instead. A
    # table spanning both levels still needs the word to tell its blocks apart.
    covered_levels = list(dict.fromkeys(level for _, level in groups))
    single_level = covered_levels[0] if len(covered_levels) == 1 else None

    # header row 1: dataset / level, with the block rule reaching up into it.
    # No \cmidrule under it: the regimes below belong to the corpus above them,
    # and the main results table heads its corpora the same way, unruled.
    labels = []
    for dataset, level in groups:
        label = rf"\textbf{{{DATASET_DISPLAY[dataset]}}}"
        if single_level is None:
            label += f" {LEVEL_DISPLAY[level]}"
        labels.append((len(POOLS), label))
    header1 = [r"\multirow{2}{*}{\textbf{Features}}"] + header_block_cells(labels)
    lines.append(" & ".join(header1) + r" \\")

    # header row 2: regime
    header2 = [""] + [display for _ in groups for _, display in POOLS]
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for fs in FEATURE_SETS:
        cells = [FEATURE_SET_DISPLAY.get(fs, fs)]
        for dataset, level in groups:
            for pool, _ in POOLS:
                match = df[(df["dataset"] == dataset) & (df["level"] == level)
                           & (df["pool"] == pool) & (df["feature_set"] == fs)]
                value = match["pearson"].iloc[0] if len(match) else float("nan")
                # No bolding here, unlike the other reliability tables: every
                # cell is an agreement between two ways of scoring the same
                # participant, not a contender for a best score, so a "winner"
                # per column would be reading a ranking into numbers that are
                # all meant to be high.
                cells.append(format_r(value, decimals))
        lines.append(" & ".join(cells) + r" \\")
        # A \midrule between the feature groups, as in every other reliability
        # table (a \cline would cut through the Features column).
        if should_add_cline_after_feature(fs, FEATURE_SETS):
            lines.append(r"\midrule")

    # "EyeScore" is the name of the measure, not a description of it, so it
    # keeps its capitalisation in the caption the way it does in the text.
    score_word = "EyeScore" if source == "eyescore" else "predicted score"
    measure = "EyeScore" if source == "eyescore" else "Predictions"
    lines += [
        # The body ends on a \midrule between feature groups; without this the
        # table closes on that thin rule and has no bottom edge.
        r"\bottomrule",
        r"\end{tabular}}" if full_width else r"\end{tabular}",
        rf"\caption{{\textbf{{{measure}}}, aggregated-per-item variant. Pearson "
        rf"$r$ correlations across participants with the "
        rf"\textbf{{original {score_word}}}.}}",
        rf"\label{{tab:original_vs_per_item_{source}{target_slug(target)}"
        rf"{level_slug(levels)}}}",
        r"\end{table*}" if full_width else r"\end{table}",
    ]
    save_path.write_text(
        rename_datasets_in_tables("\n".join(lines) + "\n"))
    print(f"saved {save_path}")


def target_slug(target: str) -> str:
    """Filename/label suffix for a non-default target ("" for the primary one).

    Without it a second target overwrites the primary table's .tex and, worse,
    ships the same \\label, which LaTeX resolves to whichever it read last.
    """
    if target == "primary":
        return ""
    return "_" + target.replace("_score", "").replace("_", "")


def main():
    parser = argparse.ArgumentParser(description="Original vs per-item-mean correlation table")
    parser.add_argument("--source", default="eyescore", choices=["eyescore", "predictions"])
    parser.add_argument("--target", default="primary")
    parser.add_argument("--model", default="Ridge_Classifier")
    parser.add_argument("--preview", default=DEFAULT_PREVIEW,
                        help="OneStop preview to read (Gathering|Hunting). Gathering "
                             "writes the top-level paths; any other preview writes a "
                             "parallel subtree (…/tables/hunting/) and is OneStop-only.")
    parser.add_argument("--levels", nargs="+", default=None,
                        choices=["paragraph", "article"],
                        help="aggregation levels to keep (default: all). "
                             "`--levels paragraph` drops the OneStop Article "
                             "block and writes a _passage-suffixed table.")
    parser.add_argument("--decimals", type=int, default=2,
                        help="decimals in the LaTeX table (3 separates cells that "
                             "round together near 1.0)")
    args = parser.parse_args()

    out_dir = OUT_DIR if args.preview == DEFAULT_PREVIEW else OUT_DIR / args.preview.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect(args.source, args.target, args.model, args.preview, args.levels)
    if df.empty:
        print("no data found")
        return

    slug = target_slug(args.target) + level_slug(args.levels)
    csv_path = out_dir / f"original_vs_per_item_{args.source}{slug}.csv"
    df.round(3).to_csv(csv_path, index=False)
    print(f"saved {csv_path}")
    tex_path = out_dir / f"original_vs_per_item_{args.source}{slug}.tex"
    generate_latex(df, args.source, tex_path, args.decimals, args.preview,
                   args.target, args.levels)
    if args.preview != DEFAULT_PREVIEW:
        annotate_preview(out_dir, args.preview)

    print()
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
