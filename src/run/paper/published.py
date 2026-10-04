"""Tables as the paper prints them.

A table the paper carries can print a different \\caption or \\label there than
its generator writes (PAPER_PINS in src/latex_captions.py). The sections that
render such a table pass it through `write_published`, which applies the
paper's caption and label, so the file under paper/ is the one the paper
prints rather than the generator's own wording.
"""
from __future__ import annotations

from pathlib import Path

from src.latex_captions import CAPTION_RE, LABEL_RE, PAPER_PINS, rename_datasets_in_tables


def as_published(latex: str, dest: str) -> str:
    """`latex` with the caption and label the paper prints for its file `dest`."""
    pin = PAPER_PINS[dest]
    if pin.caption is not None:
        latex, n = CAPTION_RE.subn(lambda _m: pin.caption, latex, count=1)
        if not n:
            raise ValueError(f"no \\caption line to replace in the table for {dest}")
    if pin.label is not None:
        latex = LABEL_RE.sub(lambda _m: f"\\label{{{pin.label}}}", latex, count=1)
    return latex


def write_published(out: Path, latex: str, dest: str) -> None:
    """Write a finished table as the paper prints it: the dataset renames every
    table gets, then the paper's caption and label."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(as_published(rename_datasets_in_tables(latex), dest))
