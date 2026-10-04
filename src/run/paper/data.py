"""Dataset statistics: paper/data/.

    data/language_dist_onestop.tex   participants per L1 and reading regime, OneStop L2
    data/language_dist_meco.tex      participants per L1, MECO, with the English L1 group
    data/corpus_stats.csv            word tokens, fixations and passages read, per dataset
    data/feature_counts.csv          Transitions and Word Fixation feature counts

The two tables are in the paper; the CSVs hold the numbers its text quotes.
Everything is read from data/: the participant tables from the fully_agg
feature files (the analysed cohort), the corpus counts from the harmonized IA
(one row per word token) and fixation (one row per fixation) files, and the
feature counts from the per-text feature parquets. The corpus counts stream
several GB of CSV, so this section takes a few minutes. Without the OneStop L2
data, the OneStop parts are skipped.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from src.constants import DATA_PATH

CHUNK = 2_000_000

ONESTOP_AGG = "features_and_targets/fully_agg/ordinary/all"
MECO_AGG = "features_and_targets/fully_agg/all/all"
ONESTOP_FEATURES = DATA_PATH / "OneStopL2" / ONESTOP_AGG

# The per-text feature files, and how many splits each corpus' "average per
# split" divides by: MECO's texts are split in two (odd/even), OneStop's
# participants into its 6 content groups.
PER_TEXT_FILES = {
    "MecoL1": DATA_PATH / "MecoL1" / MECO_AGG / "per_text_features.parquet",
    "MecoL2": DATA_PATH / "MecoL2" / MECO_AGG / "per_text_features.parquet",
    "OneStopL1": DATA_PATH / "OneStopL1" / ONESTOP_AGG / "per_text_features.parquet",
    "OneStopL2": DATA_PATH / "OneStopL2" / ONESTOP_AGG / "per_text_features.parquet",
}
FEATURE_COUNT_DATASETS = {"MECO": (["MecoL1", "MecoL2"], 2),
                          "OneStop": (["OneStopL1", "OneStopL2"], 6)}
FEATURE_KINDS = {
    "transitions": lambda c: "transitions" in c.lower(),
    "wfc": lambda c: "wfc" in c.lower(),
    "wfc_FP": lambda c: "wfc" in c.lower() and "FP" in c,   # one per word: the word count
}


# ── participants per L1 ───────────────────────────────────────────────────────

def onestop_language_counts() -> pd.DataFrame:
    """Participants per L1 x reading regime, as an Ordinary/Seeking/All table."""
    cols = ("participant_id", "L1", "question_preview")
    df = pd.read_csv(ONESTOP_FEATURES / "features_and_metadata.csv", low_memory=False,
                     usecols=lambda c: c in cols).drop_duplicates("participant_id")
    counts = pd.crosstab(df["L1"], df["question_preview"])
    # A regime missing entirely would otherwise drop the column and break the reindex.
    counts = counts.reindex(columns=[False, True], fill_value=0)
    counts.columns = ["Ordinary Reading", "Information Seeking"]
    counts["All"] = counts.sum(axis=1)
    return counts.sort_index()


def onestop_language_table(counts: pd.DataFrame, label: str = "tab:language_dist") -> str:
    rows = "\n".join(
        f"{lang} & " + " & ".join(str(int(v)) for v in row) + r" \\"
        for lang, row in counts.iterrows()
    )
    header = "L1 & " + " & ".join(counts.columns) + r" \\"
    caption = ("Number of participants per L1 language for ordinary reading for "
               "comprehension, information seeking, and the total of both groups.")
    return (
        "\\begin{table}[ht!]\n"
        "\\centering\n"
        "\\resizebox{\\columnwidth}{!}{\n"
        "\\begin{tabular}{l" + "c" * counts.shape[1] + "}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}"
    )


def meco_language_counts(dataset: str = "MecoL2") -> pd.Series:
    """Participants per L1, with an All total."""
    df = pd.read_csv(DATA_PATH / dataset / MECO_AGG / "features_and_metadata.csv",
                     low_memory=False, usecols=lambda c: c in ("participant_id", "L1")
                     ).drop_duplicates("participant_id")
    counts = df["L1"].value_counts().sort_index()
    counts.loc["All"] = len(df)
    return counts


def meco_language_table(counts: pd.Series, l1_count: int,
                        label: str = "tab:meco_language_dist") -> str:
    """L2 languages, then the L1 (English) group below a dashed rule.

    \\hdashline needs \\usepackage{arydshln} in the preamble.
    """
    lines = [f"{lang} & {int(n):,} " + r"\\" for lang, n in counts.items() if lang != "All"]
    lines += ["\\hdashline", "\\addlinespace[4pt]", f"English (L1) & {l1_count:,} " + r"\\"]
    l2_total = int(counts["All"])
    caption = (f"Number of participants per L1 language in MECO. The {l2_total:,} L2 "
               f"participants are listed above the dashed line, the {l1_count} L1 (English) "
               f"participants below it.")
    return (
        "\\begin{table}[ht!]\n\\centering\n\\resizebox{\\columnwidth}{!}{\n"
        "\\begin{tabular}{lr}\n\\toprule\n"
        "L1 & Participants \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n\\end{tabular}}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}"
    )


# ── corpus size ──────────────────────────────────────────────────────────────

def onestop_corpus_stats() -> dict:
    """Word tokens and fixations of the analysed OneStop L2 cohort, first reading
    only: the 54 paragraphs each participant reads once, no practice, no rereads."""
    cohort = set(pd.read_csv(ONESTOP_FEATURES / "features_and_metadata.csv", low_memory=False,
                             usecols=["participant_id"])["participant_id"].unique())
    harmonized = DATA_PATH / "OneStopL2" / "harmonized"
    totals = {}
    for kind, name in (("word_tokens", "ia.csv"), ("fixations", "fixations.csv")):
        per_participant = Counter()
        for chunk in pd.read_csv(harmonized / name, chunksize=CHUNK, low_memory=False,
                                 usecols=["participant_id", "practice_trial",
                                          "repeated_reading_trial"]):
            chunk = chunk[chunk["participant_id"].isin(cohort)]
            first_reading = (~chunk["practice_trial"].astype(bool)
                             & ~chunk["repeated_reading_trial"].astype(bool))
            per_participant.update(chunk.loc[first_reading, "participant_id"].value_counts().to_dict())
        totals[kind] = per_participant
    n = len(totals["fixations"])
    tokens, fixations = sum(totals["word_tokens"].values()), sum(totals["fixations"].values())
    return {"dataset": "OneStopL2", "scope": "first reading of the analysed cohort",
            "participants": n, "word_tokens": tokens, "fixations": fixations,
            "tokens_per_participant": tokens / n, "fixations_per_participant": fixations / n}


def meco_corpus_stats(dataset: str) -> dict:
    """Dataset-level totals: word tokens, fixations, and passage coverage."""
    harmonized = DATA_PATH / dataset / "harmonized"
    tokens, passages = Counter(), {}
    for chunk in pd.read_csv(harmonized / "ia.csv", chunksize=CHUNK, low_memory=False,
                             usecols=["participant_id", "unique_paragraph_id"]):
        tokens.update(chunk["participant_id"].value_counts().to_dict())
        for pid, group in chunk.groupby("participant_id", sort=False):
            passages.setdefault(pid, set()).update(group["unique_paragraph_id"].unique())
    fixations = Counter()
    for chunk in pd.read_csv(harmonized / "fixations.csv", chunksize=CHUNK, low_memory=False,
                             usecols=["participant_id"]):
        fixations.update(chunk["participant_id"].value_counts().to_dict())
    n = len(tokens)
    read = pd.Series({p: len(v) for p, v in passages.items()})
    return {"dataset": dataset, "scope": "whole dataset",
            "participants": n,
            "word_tokens": sum(tokens.values()), "fixations": sum(fixations.values()),
            "tokens_per_participant": sum(tokens.values()) / n,
            "fixations_per_participant": sum(fixations.values()) / n,
            "passages_mean": read.mean(), "passages_median": read.median(),
            "passages_min": int(read.min()), "passages_max": int(read.max()),
            "read_all_12": int((read == 12).sum())}


# ── per-text feature counts ──────────────────────────────────────────────────

def _nonzero_columns(path: Path, pred) -> set:
    """Columns with at least one non-null, non-zero value, read from the row-group
    statistics alone (these files are written as a single row group)."""
    pf = pq.ParquetFile(path, thrift_string_size_limit=2 ** 31 - 1,
                        thrift_container_size_limit=2 ** 31 - 1)
    row_group = pf.metadata.row_group(0)
    keep = set()
    for j, name in enumerate(pf.schema.names):
        if not pred(name):
            continue
        stats = row_group.column(j).statistics
        if stats is None or stats.num_values == 0 or not stats.has_min_max:
            continue
        if stats.min == 0 and stats.max == 0:
            continue
        keep.add(name)
    return keep


def feature_counts() -> pd.DataFrame:
    """Feature columns per corpus (L1 and L2 pooled), and their average per split.
    A corpus whose feature files are absent gets no rows."""
    rows = []
    for corpus, (parts, n_splits) in FEATURE_COUNT_DATASETS.items():
        if not all(PER_TEXT_FILES[p].exists() for p in parts):
            logger.warning("{} per-text feature files not found; left out of "
                           "feature_counts.csv", corpus)
            continue
        for kind, pred in FEATURE_KINDS.items():
            union = set().union(*(_nonzero_columns(PER_TEXT_FILES[p], pred) for p in parts))
            rows.append({"corpus": corpus, "kind": kind, "total": len(union),
                         "splits": n_splits, "avg_per_split": round(len(union) / n_splits, 1)})
    return pd.DataFrame(rows)


def render(out_root: Path) -> list[Path]:
    out_dir = Path(out_root) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    onestop = (ONESTOP_FEATURES / "features_and_metadata.csv").exists()
    if onestop:
        out = out_dir / "language_dist_onestop.tex"
        out.write_text(onestop_language_table(onestop_language_counts()) + "\n")
        written.append(out)
    else:
        logger.warning("{} not found; language_dist_onestop.tex and the OneStop row "
                       "of corpus_stats.csv are skipped", ONESTOP_FEATURES)

    out = out_dir / "language_dist_meco.tex"
    l1_total = int(meco_language_counts("MecoL1")["All"])
    out.write_text(meco_language_table(meco_language_counts("MecoL2"), l1_total) + "\n")
    written.append(out)

    out = out_dir / "feature_counts.csv"
    feature_counts().to_csv(out, index=False)
    written.append(out)

    out = out_dir / "corpus_stats.csv"
    stats = (([onestop_corpus_stats()] if onestop else [])
             + [meco_corpus_stats("MecoL2"), meco_corpus_stats("MecoL1")])
    # convert_dtypes keeps the counts integral though OneStop has no passage columns.
    pd.DataFrame(stats).round(1).convert_dtypes().to_csv(out, index=False)
    written.append(out)
    return written
