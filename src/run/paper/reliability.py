"""The reliability tables: paper/reliability/.

    reliability/main.tex                  EyeScore reliability, Any regime (Table 4)
    reliability/fixed.tex                 the same for the Fixed regime (appendix)
    reliability/hunting.tex               both regimes, OneStop Information Seeking (appendix)
    reliability/original_vs_per_item.tex  original vs per-item-mean EyeScore (appendix)
    reliability/evaluation_delta.tex      per-item minus original EyeScore r (appendix)

Built by the src.reliability generators from the reliability results in
src/reliability/results/, which `python -m src.reliability.split_half` and
`python -m src.reliability.cronbach_alpha` write, with the caption and label
each table has in the submission (see published.py).

evaluation_delta.tex also needs the per-item EyeScore evaluation rows; without
them in the EyeScore evaluation CSV it is skipped. original_vs_per_item.tex is
built from the stored table CSV and printed at its three decimals, so no cell
is rounded twice.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from loguru import logger

from src.reliability.table_original_vs_per_item import generate_latex
from src.reliability.tables import (
    DEFAULT_TABLE_DATASETS,
    annotate_preview,
    generate_evaluation_eyescore_combined,
    generate_eyescore_reliability_combined_table,
    generate_eyescore_reliability_regimes_table,
    preview_datasets,
)
from src.reliability.utils import DEFAULT_PREVIEW
from src.run.paper.published import write_published

RESULTS = Path("src/reliability/results")
HUNTING = "Hunting"
ORIGINAL_VS_PER_ITEM_CSV = (RESULTS / "per_item_reliability" / "tables"
                            / "original_vs_per_item_eyescore_passage.csv")

# path under paper/reliability/ -> file in the submission (its PAPER_PINS key)
PUBLISHED_AS = {
    "main.tex": "tables/table4_eyescore_reliability.tex",
    "fixed.tex": "tables/appendix/reliability/eyescore_reliablity_fixed.tex",
    "hunting.tex": "tables/appendix/reliability/hunting/eyescore_reliability.tex",
    "original_vs_per_item.tex": "tables/appendix/reliability/correlation/correlations_eyescore.tex",
    "evaluation_delta.tex": "tables/appendix/evaluation/evaluation_eyescore_delta.tex",
}


def _results(subdir: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-item reliability, split-half reliability) for one preview's subtree."""
    return (pd.read_csv(RESULTS / "per_item_reliability" / subdir / "per_item_reliability.csv"),
            pd.read_csv(RESULTS / "split_half_reliability" / subdir / "split_half_reliability.csv"))


def render(out_root: Path) -> list[Path]:
    out_dir = Path(out_root) / "reliability"
    # The generators write their own files; each is rendered into a scratch
    # directory and published from there.
    scratch = Path(tempfile.mkdtemp(prefix="paper_reliability_"))
    cronbach, split_half = _results()
    generate_eyescore_reliability_combined_table(
        cronbach, split_half, scratch / "main.tex", datasets=DEFAULT_TABLE_DATASETS)
    generate_eyescore_reliability_combined_table(
        cronbach, split_half, scratch / "fixed.tex", datasets=DEFAULT_TABLE_DATASETS,
        pool="seen")
    generate_evaluation_eyescore_combined(
        scratch / "evaluation_delta.tex", preview=DEFAULT_PREVIEW,
        datasets=DEFAULT_TABLE_DATASETS, levels=("paragraph",), delta=True)
    generate_latex(pd.read_csv(ORIGINAL_VS_PER_ITEM_CSV), "eyescore",
                   scratch / "original_vs_per_item.tex", decimals=3,
                   preview=DEFAULT_PREVIEW, levels=["paragraph"])

    # The Information Seeking table gets its regime marked in caption and label,
    # which annotate_preview does to every table in a directory -- so on its own.
    hunting_dir = scratch / "hunting"
    hunting_dir.mkdir()
    cronbach_h, split_half_h = _results(HUNTING.lower())
    generate_eyescore_reliability_regimes_table(
        cronbach_h, split_half_h, hunting_dir / "hunting.tex",
        datasets=preview_datasets(HUNTING), p_agg_level="paragraph")
    annotate_preview(hunting_dir, HUNTING)

    written = []
    for name, published_as in PUBLISHED_AS.items():
        rendered = hunting_dir / name if name == "hunting.tex" else scratch / name
        if not rendered.exists():
            logger.warning("{} was not rendered (missing input); skipped", name)
            continue
        out = out_dir / name
        write_published(out, rendered.read_text(), published_as)
        written.append(out)
    return written
