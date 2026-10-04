"""paper/README.md: an index of the rendered files, rewritten on every render."""
from __future__ import annotations

import re
from pathlib import Path

from src.run.paper import eyescore, predictions, reliability

LABEL_RE = re.compile(r"\\label\{([^}]*)\}")

# top-level folder -> (the sections that write it, what it holds)
FOLDERS = {
    "eyescore": ("eyescore", "EyeScore correlations with the proficiency tests and with comprehension"),
    "prediction": ("predictions", "Ridge prediction of the proficiency tests"),
    "model_comparison": ("predictions", "the other prediction models, per dataset slice"),
    "reliability": ("reliability", "reliability of the EyeScore"),
    "lang_bias": ("lang_bias, debiased, figures", "language bias and its correction, per debiasing method"),
    "data": ("data", "participant tables and corpus statistics"),
}

HEADER = """# paper/

Every table and figure the paper reports, rendered by `src.run.paper` from the
outputs of the earlier stages (`src.run.features`, `src.run.predictions`,
`src.run.eyescore`, and the reliability scripts in `src/reliability/`). Nothing
here is edited by hand: change the generator and re-render.

    python -m src.run.paper                         # every section
    python -m src.run.paper --only reliability,data # some sections
    python -m src.run.paper --out /tmp/paper_check  # render elsewhere, e.g. to diff

Tables the submission also carries have the paper's caption and `\\label`
(`PAPER_PINS` in `src/latex_captions.py`); the last column names the file each
one is in the submission.
"""


def published_files() -> dict:
    """Path under paper/ -> file in the submission, for every table both have."""
    out = {eyescore.TABLES[table][0]: dest for table, dest in eyescore.PUBLISHED_AS.items()}
    out.update({relpath: dest for relpath, *_rest, dest in predictions.MAIN_TABLES if dest})
    out.update({relpath: dest for relpath, dest in predictions.TABPFN_TABLES.values()})
    out.update({f"reliability/{name}": dest for name, dest in reliability.PUBLISHED_AS.items()})
    return out


def write_readme(out_root: Path) -> Path:
    out_root = Path(out_root)
    published = published_files()
    lines = [HEADER]
    for folder, (sections, what) in FOLDERS.items():
        files = sorted(p.relative_to(out_root).as_posix()
                       for p in (out_root / folder).rglob("*") if p.is_file())
        if not files:
            continue
        lines += [f"## {folder}/ — {what}", "", f"Section: `{sections}`.", "",
                  "| File | Label | In the submission |", "|---|---|---|"]
        for rel in files:
            label = ""
            if rel.endswith(".tex"):
                label = ", ".join(f"`{l}`" for l in LABEL_RE.findall((out_root / rel).read_text()))
            lines.append(f"| `{rel}` | {label} | {published.get(rel, '')} |")
        lines.append("")
    missing = sorted(rel for rel in published if not (out_root / rel).exists())
    if missing:
        lines += ["## Not rendered here", "",
                  "These need results that were not available when this tree was rendered "
                  "(the OneStop results, the TabPFN evaluation, the per-item EyeScore "
                  "evaluation); rendering where they exist fills them in.", ""]
        lines += [f"- `{rel}` (in the submission: {published[rel]})" for rel in missing]
        lines.append("")
    out = out_root / "README.md"
    out.write_text("\n".join(lines))
    return out
