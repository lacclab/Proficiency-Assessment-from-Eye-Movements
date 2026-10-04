"""Debiased-EyeScore correlation tables: paper/lang_bias/<method>/tables/.

Each table reports Pearson r between debiased EyeScore and the proficiency
tests, where the *debiasing* is calibrated against a specific target (LexTALE,
Michigan or the MECO Composite; see OUTPUTS). Each cell shows ``Seen / New`` --
the left value uses inside_langs typology calibration (debiasing pool includes
same-L1 participants), the right uses across_langs (no same-L1). Three variants
per calibration:

    <name>.tex           the correlations themselves
    <name>_diff.tex      Δr = r(debiased) - r(raw), with a paired-bootstrap star
    <name>_rawcorr.tex   r(EyeScore, EyeScore^db): how far debiasing moved the score

The paper reports the LexTALE-calibrated Δ table for every debias method and the
two-step method's rawcorr table (PUBLISHED); `render_tables` renders any of them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.EyeScore.debias_tests import (
    delta_r_test,
    n_stars,
    stars_tex,
)
from src.evaluation.EyeScore.evaluation import compute_correlation, load_metadata
from src.evaluation.EyeScore.tables import (
    DATASET_TARGET_COL_DISPLAY,
    FEATURE_SET_DISPLAY,
    FEATURE_SET_GROUP_INDEX,
    FEATURE_SET_GROUPS,
    FIXED_TEXT_ONLY_FEATURE_SETS,
)
from src.latex_captions import write_table
from src.methods.EyeScore.typology import compute_typological_distances
from src.run.paper.lang_bias import PAPER_DISTANCE_TYPE, PAPER_TREES

# Dataset config: preview, version_path, and the on-disk layout root for
# eye_score CSVs.
DATASETS = {
    "Meco": {
        "preview": "All",
        "version_path": "fully_agg/all/all",
        "display": "MECO",
        "target_cols": ["lextale_score", "proficiency_agg"],
    },
    "OneStop": {
        "preview": "Gathering",
        "version_path": "fully_agg/ordinary/all",
        # Labelled with the cohort: every target here is an L2 test, matching
        # BOOTSTRAP_DATASET_DISPLAY in the predictions/EyeScore table modules.
        "display": "OneStopL2",
        "target_cols": ["lextale_score", "michtest_score"],
    },
}

# Calibration suffix on the eye_score CSV filename. Empty string = LexTALE
# (the default, no extra suffix beyond the scheme).
CALIB_SUFFIX = {
    "lextale": "",
    "michigan": "_michtest",
    "profagg": "_profagg",
}

# Proficiency column each calibration fitted its debiasing coefficient against.
# The Δ-table significance test refits that coefficient per bootstrap replicate,
# so it needs the column, not just the filename suffix.
CALIB_TARGET_COL = {
    "lextale": "lextale_score",
    "michigan": "michtest_score",
    "profagg": "proficiency_agg",
}


# Bootstrap replicates behind the Δ-table stars. The Monte-Carlo error on a
# p-value near the 0.05 threshold is sqrt(p(1-p)/B) -- +-0.005 at B=2000, which
# is enough to flip a borderline star under a different seed. 10000 halves that
# and lowers the reportable floor to 1e-4.
N_BOOTSTRAP = 10000

# Two debias schemes; left/right inside each cell.
SCHEMES = [("inside_langs", "Seen"), ("across_langs", "New")]

# Scope columns. content_mode maps: seen → Fixed, unseen → Any.
SCOPES = [("seen", "Fixed"), ("unseen", "Any")]

FEATURE_SETS = [fs for group in FEATURE_SET_GROUPS for fs in group]

# Per-output spec: which calibration, which datasets, output filename, label
# suffix, and the human-readable calibration-target name used in the caption.
OUTPUTS = [
    {
        "calib": "lextale",
        "datasets": ["Meco", "OneStop"],
        "filename": "combined_debiased_lextale.tex",
        "label": "tab:eyescore_seen_unseen-debiased-lextale",
        "calib_phrase": "LexTALE",
    },
    {
        "calib": "michigan",
        "datasets": ["OneStop"],
        "filename": "combined_debiased_michigan.tex",
        "label": "tab:eyescore_seen_unseen-debiased-michigan",
        "calib_phrase": "Michigan",
    },
    {
        "calib": "profagg",
        "datasets": ["Meco"],
        "filename": "combined_debiased_profagg.tex",
        "label": "tab:eyescore_seen_unseen-debiased-profagg",
        "calib_phrase": "Composite",
    },
]


def _eye_csv_path(results_dir: Path, dataset: str, content_mode: str, feature_set: str,
                  scheme: str, calib: str) -> Path:
    cfg = DATASETS[dataset]
    suffix = f"_typo_calibrated_{scheme}{CALIB_SUFFIX[calib]}"
    return (
        results_dir / dataset / cfg["preview"] / content_mode / cfg["version_path"]
        / f"{feature_set}{suffix}.csv"
    )


def _compute_lookup(results_dir: Path, calib: str, datasets: list[str]) -> dict:
    """lookup[(dataset, feature_set, target_col, scope_label, scheme)] -> r."""
    lookup: dict = {}
    meta_cache: dict[str, pd.DataFrame] = {}

    for dataset in datasets:
        cfg = DATASETS[dataset]
        meta_key = f"{dataset}/{cfg['version_path']}"
        if meta_key not in meta_cache:
            meta_cache[meta_key] = load_metadata(dataset, cfg["version_path"])
        meta = meta_cache[meta_key]
        if meta.empty:
            logger.warning("Missing metadata for {}", dataset)
            continue

        for fs in FEATURE_SETS:
            for content_mode, scope_label in SCOPES:
                if fs in FIXED_TEXT_ONLY_FEATURE_SETS and scope_label != "Fixed":
                    continue
                for scheme, _ in SCHEMES:
                    csv_path = _eye_csv_path(results_dir, dataset, content_mode, fs, scheme, calib)
                    if not csv_path.exists():
                        logger.warning("Missing {}", csv_path)
                        continue
                    eye_df = pd.read_csv(csv_path)
                    if "eye_score" not in eye_df.columns:
                        continue
                    merged = eye_df.merge(meta, on="participant_id",
                                          how="inner", suffixes=("", "_meta"))
                    if merged.empty:
                        continue
                    for tc in cfg["target_cols"]:
                        if tc not in merged.columns:
                            continue
                        y = merged[tc].values.astype(float)
                        # MECO uses -1 as missing sentinel; symmetric handling.
                        y = np.where(y == -1, np.nan, y)
                        # Only pearson_r is read below, and it comes from pearsonr()
                        # either way — the bootstrap fills the CI columns alone. At
                        # N_BOOTSTRAP=100k that was this table's entire runtime,
                        # spent on numbers it discards.
                        metrics = compute_correlation(merged["eye_score"].values, y,
                                                      include_bootstrap=False)
                        lookup[(dataset, fs, tc, scope_label, scheme)] = metrics["pearson_r"]
    return lookup


def _compute_diff_lookup(results_dir: Path, calib: str, datasets: list[str],
                         distance_type: str, center_distance: bool,
                         n_bootstrap: int) -> dict:
    """diff[(dataset, feature_set, target_col, scope_label, scheme)] -> (Δr, p).

    Δr is computed on the participants present in BOTH the raw and debiased
    score files. That matters for the inside_langs scheme, which drops L1 groups
    smaller than the fold count -- on OneStop it loses Vietnamese (n=5 < k=10),
    so subtracting two independently-computed correlations would compare 146
    participants against 151 and fold the sample change into the reported Δ.

    The p-value is a paired participant bootstrap that refits the debiasing
    coefficient inside every resample; see src.evaluation.EyeScore.debias_tests.
    """
    diff: dict = {}
    calib_col = CALIB_TARGET_COL[calib]
    meta_cache: dict[str, pd.DataFrame] = {}
    dist_cache: dict[str, dict] = {}

    for dataset in datasets:
        cfg = DATASETS[dataset]
        meta_key = f"{dataset}/{cfg['version_path']}"
        if meta_key not in meta_cache:
            meta_cache[meta_key] = load_metadata(dataset, cfg["version_path"])
        meta = meta_cache[meta_key]
        if meta.empty or calib_col not in meta.columns:
            logger.warning("No {} in {} metadata; skipping Δ table",
                           calib_col, dataset)
            continue
        if dataset not in dist_cache:
            dist_cache[dataset] = compute_typological_distances(
                sorted(meta["L1"].dropna().unique()), distance_type=distance_type)
        distances = dist_cache[dataset]

        for fs in FEATURE_SETS:
            for content_mode, scope_label in SCOPES:
                if fs in FIXED_TEXT_ONLY_FEATURE_SETS and scope_label != "Fixed":
                    continue
                raw_path = (results_dir / dataset / cfg["preview"] / content_mode
                            / cfg["version_path"] / f"{fs}.csv")
                if not raw_path.exists():
                    logger.warning("Missing org {}", raw_path)
                    continue
                raw_df = pd.read_csv(raw_path)
                if "eye_score" not in raw_df.columns:
                    continue
                for scheme, _ in SCHEMES:
                    db_path = _eye_csv_path(results_dir, dataset, content_mode, fs, scheme, calib)
                    if not db_path.exists():
                        logger.warning("Missing {}", db_path)
                        continue
                    db_df = pd.read_csv(db_path)
                    if "eye_score" not in db_df.columns:
                        continue
                    for tc in cfg["target_cols"]:
                        if tc not in meta.columns:
                            continue
                        res = delta_r_test(
                            raw_df, db_df, meta, tc, calib_col, distances, scheme,
                            center=center_distance, n_boot=n_bootstrap,
                            cache_key=(str(raw_path), str(db_path), tc, calib_col),
                        )
                        if res is None:
                            continue
                        diff[(dataset, fs, tc, scope_label, scheme)] = res
    return diff


def _compute_rawdb_lookup(results_dir: Path, calib: str, datasets: list[str]) -> dict:
    """rawdb[(dataset, feature_set, scope_label, scheme)] -> r(EyeScore, EyeScore^db).

    This correlation is between the two score vectors themselves, so it does not
    depend on any proficiency test -- one value per (dataset, feature set,
    regime, scheme), no target axis. Computed on the participants present in
    both score files: the inside_langs scheme drops L1 groups smaller than the
    fold count, so the raw file is the larger of the two.
    """
    rawdb: dict = {}

    for dataset in datasets:
        cfg = DATASETS[dataset]
        for fs in FEATURE_SETS:
            for content_mode, scope_label in SCOPES:
                if fs in FIXED_TEXT_ONLY_FEATURE_SETS and scope_label != "Fixed":
                    continue
                raw_path = (results_dir / dataset / cfg["preview"] / content_mode
                            / cfg["version_path"] / f"{fs}.csv")
                if not raw_path.exists():
                    logger.warning("Missing org {}", raw_path)
                    continue
                raw_df = pd.read_csv(raw_path)
                if "eye_score" not in raw_df.columns:
                    continue
                for scheme, _ in SCHEMES:
                    db_path = _eye_csv_path(results_dir, dataset, content_mode, fs, scheme, calib)
                    if not db_path.exists():
                        logger.warning("Missing {}", db_path)
                        continue
                    db_df = pd.read_csv(db_path)
                    if "eye_score" not in db_df.columns:
                        continue
                    merged = raw_df[["participant_id", "eye_score"]].merge(
                        db_df[["participant_id", "eye_score"]],
                        on="participant_id", how="inner", suffixes=("_raw", "_db"))
                    if merged.empty:
                        continue
                    metrics = compute_correlation(
                        merged["eye_score_raw"].values.astype(float),
                        merged["eye_score_db"].values.astype(float),
                        include_bootstrap=False)
                    rawdb[(dataset, fs, scope_label, scheme)] = metrics["pearson_r"]
    return rawdb


def _fmt(val, bold: bool, diff: bool = False) -> str:
    """Format one value. Δ cells arrive as (delta, p) and get significance stars."""
    if diff:
        if val is None:
            return "-"
        delta, p = val
        if pd.isna(delta):
            return "-"
        return f"{delta:+.3f}{stars_tex(n_stars(p))}"
    if val is None or pd.isna(val):
        return "-"
    s = f"{val:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def _missing(val, diff: bool) -> bool:
    """Δ cells hold (delta, p) tuples, which pd.isna would map elementwise."""
    if val is None:
        return True
    return pd.isna(val[0]) if diff else pd.isna(val)


def _fmt_cell(val_in, val_ac, best_in, best_ac, diff: bool = False) -> str:
    if _missing(val_in, diff) and _missing(val_ac, diff):
        return "-"
    is_best_in = (
        not diff
        and not _missing(val_in, diff)
        and best_in is not None and np.isclose(val_in, best_in, atol=1e-6)
    )
    is_best_ac = (
        not diff
        and not _missing(val_ac, diff)
        and best_ac is not None and np.isclose(val_ac, best_ac, atol=1e-6)
    )
    return f"{_fmt(val_in, is_best_in, diff)} / {_fmt(val_ac, is_best_ac, diff)}"


def _render(lookup: dict, datasets: list[str], label: str, calib_phrase: str,
            diff: bool = False) -> str:
    n_scopes = len(SCOPES)
    cols_per_ds = {d: len(DATASETS[d]["target_cols"]) for d in datasets}
    n_data_cols = sum(cols_per_ds[d] * n_scopes for d in datasets)
    n_total_cols = 1 + n_data_cols
    single_dataset = len(datasets) == 1

    # Best (max r) per (dataset, target_col, scope, scheme) across feature sets.
    # Skipped for diff tables — there's no natural "best Δ" to bold.
    best: dict = {}
    if not diff:
        for ds in datasets:
            for tc in DATASETS[ds]["target_cols"]:
                for _, scope_label in SCOPES:
                    for scheme, _ in SCHEMES:
                        vals = [
                            lookup[(ds, fs, tc, scope_label, scheme)]
                            for fs in FEATURE_SETS
                            if (ds, fs, tc, scope_label, scheme) in lookup
                            and not pd.isna(lookup[(ds, fs, tc, scope_label, scheme)])
                        ]
                        best[(ds, tc, scope_label, scheme)] = max(vals) if vals else None

    # Column spec: `l` + every data column joined with `|`. Dataset/target
    # boundaries are marked via the multicolumn `c|`/`c` alignment in the
    # header rows, not by extra `||` in the spec.
    col_spec = "l" + "|".join(["c"] * n_data_cols)

    table_env = "table" if single_dataset else "table*"
    resize_width = "\\columnwidth" if single_dataset else "\\textwidth"

    lines = [
        f"\\begin{{{table_env}}}[ht!]",
        "\\centering",
        "% \\small",
        f"\\resizebox{{{resize_width}}}{{!}}{{",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Row 1: dataset names. `c|` on every dataset but the last, so the boundary
    # rule runs unbroken from \toprule to \bottomrule instead of starting one
    # row down.
    lines.append(
        "\\multicolumn{1}{c}{} & "
        + " & ".join(
            f"\\multicolumn{{{cols_per_ds[ds] * n_scopes}}}"
            f"{{{'c' if i == len(datasets) - 1 else 'c|'}}}"
            f"{{\\textbf{{{DATASETS[ds]['display']}}}}}"
            for i, ds in enumerate(datasets)
        )
        + " \\\\"
    )

    # Row 2: target labels. `c|` only on the last target of a non-last dataset
    # (dataset boundary). Trailing \cmidrule under the targets row.
    target_header_parts = []
    for ds_idx, ds in enumerate(datasets):
        tcs = DATASETS[ds]["target_cols"]
        is_last_ds = (ds_idx == len(datasets) - 1)
        for tc_idx, tc in enumerate(tcs):
            is_last_tc = (tc_idx == len(tcs) - 1)
            align = "c|" if (is_last_tc and not is_last_ds) else "c"
            disp = DATASET_TARGET_COL_DISPLAY[ds].get(tc, tc)
            target_header_parts.append(
                f"\\multicolumn{{{n_scopes}}}{{{align}}}{{{disp}}}"
            )
    lines.append(
        "\\multicolumn{1}{c}{} & "
        + " & ".join(target_header_parts)
        + f" \\\\ \\cmidrule{{2-{n_total_cols}}}"
    )

    # Row 3: features label + scope labels per target.
    scope_labels = []
    for ds in datasets:
        for _ in DATASETS[ds]["target_cols"]:
            for _, scope in SCOPES:
                scope_labels.append(scope)
    lines.append("\\textbf{Features} & " + " & ".join(scope_labels) + " \\\\")
    lines.append("\\midrule")

    # Data rows (booktabs: \midrule between groups, \bottomrule after last row).
    for i, fs in enumerate(FEATURE_SETS):
        if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[FEATURE_SETS[i - 1]]:
            lines.append("\\midrule")
        cells = [FEATURE_SET_DISPLAY.get(fs, fs)]
        for ds in datasets:
            for tc in DATASETS[ds]["target_cols"]:
                for _, scope_label in SCOPES:
                    if fs in FIXED_TEXT_ONLY_FEATURE_SETS and scope_label != "Fixed":
                        cells.append("-")
                        continue
                    v_in = lookup.get((ds, fs, tc, scope_label, "inside_langs"))
                    v_ac = lookup.get((ds, fs, tc, scope_label, "across_langs"))
                    cells.append(_fmt_cell(
                        v_in, v_ac,
                        best.get((ds, tc, scope_label, "inside_langs")),
                        best.get((ds, tc, scope_label, "across_langs")),
                        diff=diff,
                    ))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")

    if diff:
        caption = (
            "Change in Pearson $r$ correlation with standard language "
            "proficiency tests after debiasing (\\textit{debiased $-$ raw}), "
            f"where the debiasing is calibrated against {calib_phrase}. "
            "Each cell shows \\textit{$\\Delta$ Seen\\,/\\,$\\Delta$ New}: "
            "the left value uses the Seen Language scheme (inside-langs "
            "typology-calibrated); the right uses the New Language scheme "
            "(across-langs). ``Fixed'' is the Fixed Text regime; ``Any'' is "
            "the Any Text regime. Negative values indicate that debiasing "
            "reduced the EyeScore--target correlation. Each $\\Delta$ is "
            "computed on the participants scored under both variants; the Seen "
            "Language scheme excludes any L1 with fewer participants than the "
            "number of cross-validation folds (on OneStopL2 this drops "
            "Vietnamese, $n=5$). Stars mark changes significantly different "
            "from zero by a paired participant bootstrap in which the debiasing "
            "coefficient is refitted within every resample: "
            "$^{*}\\,p<0.05$, $^{**}\\,p<0.01$, $^{***}\\,p<0.001$."
        )
    else:
        caption = (
            "Pearson $r$ correlations between debiased EyeScore and "
            "standard language proficiency tests, where the debiasing is "
            f"calibrated against {calib_phrase}. "
            "Each cell shows \\textit{Seen\\,/\\,New}: the left value uses the "
            "Seen Language scheme (inside-langs typology-calibrated, where the "
            "debiasing data includes participants from the same L1 as the test "
            "participant); the right uses the New Language scheme (across-langs, "
            "no same-L1 participants). ``Fixed'' is the Fixed Text regime in "
            "which all the eye movement data from prior participants is for the "
            "same texts presented to the test participant. ``Any'' is the Any "
            "Text regime in which no eye movement data is available for the "
            "texts of the test participant."
        )
    lines.extend([
        "\\end{tabular}",
        "}",
        "\\caption{" + caption + "}",
        f"\\label{{{label}}}",
        f"\\end{{{table_env}}}",
    ])

    return "\n".join(lines)


# Partial vertical rule separating two targets *within* one dataset. Shorter
# than a full `|` so the dataset boundary stays the visually dominant division.
_TARGET_RULE = r"c!{\vrule height 0.7\ht\strutbox depth \dp\strutbox}"


def _render_diff(lookup: dict, datasets: list[str], label: str) -> str:
    """Render the Δ table: one physical column per (target, scope, scheme).

    Distinct from `_render`, which keeps the correlation tables' layout of eight
    combined ``Seen / New`` cells. Here each scheme gets its own column, so the
    grid is 4 columns per target and the header carries a fourth row naming the
    schemes.
    """
    n_targets = sum(len(DATASETS[d]["target_cols"]) for d in datasets)
    n_data_cols = n_targets * len(SCOPES) * len(SCHEMES)
    n_total_cols = 1 + n_data_cols
    single_dataset = len(datasets) == 1

    env = "table" if single_dataset else "table*"
    width = r"\columnwidth" if single_dataset else r"\textwidth"

    lines = [
        f"% \\begin{{{env}}}[ht!]",
        r"\centering",
        r"\setlength{\aboverulesep}{0pt}",
        r"\setlength{\belowrulesep}{0pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        f"\\resizebox{{{width}}}{{!}}{{",
        "\\begin{tabular}{l" + "|".join(["cccc"] * n_targets) + "}",
        r"\toprule",
    ]

    # Row 1: dataset names, spanning every column they own.
    lines.append(
        "\\multicolumn{1}{c}{} & "
        + " & ".join(
            f"\\multicolumn{{{len(DATASETS[ds]['target_cols']) * 4}}}{{c}}"
            f"{{\\textbf{{{DATASETS[ds]['display']}}}}}"
            for ds in datasets
        )
        + r" \\"
    )

    # Rows 2 and 3 share the same boundary logic: a full rule after the last
    # target of a non-final dataset, nothing after the final one, and (row 3
    # only) a partial rule between targets inside a dataset.
    target_cells, regime_cells = [], []
    for ds_idx, ds in enumerate(datasets):
        tcs = DATASETS[ds]["target_cols"]
        last_ds = ds_idx == len(datasets) - 1
        for tc_idx, tc in enumerate(tcs):
            last_tc = tc_idx == len(tcs) - 1
            if last_tc and not last_ds:
                closing = "c|"
            elif last_tc:
                closing = "c"
            else:
                closing = _TARGET_RULE
            disp = DATASET_TARGET_COL_DISPLAY[ds].get(tc, tc)
            target_cells.append(
                f"\\multicolumn{{4}}{{{'c|' if (last_tc and not last_ds) else 'c'}}}{{{disp}}}")
            for scope_idx, (_, scope) in enumerate(SCOPES):
                align = closing if scope_idx == len(SCOPES) - 1 else "c"
                regime_cells.append(f"\\multicolumn{{2}}{{{align}}}{{{scope}}}")

    lines.append("\\multicolumn{1}{c}{} & " + " & ".join(target_cells)
                 + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")
    lines.append("\\multicolumn{1}{c}{} & " + " & ".join(regime_cells)
                 + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

    # Row 4: scheme labels, one per physical column.
    lines.append(
        r"\textbf{Features} & "
        + " & ".join(f"{label_} L1" for _ in range(n_targets * len(SCOPES))
                     for _, label_ in SCHEMES)
        + r" \\"
    )
    lines.append(r"\midrule")

    for i, fs in enumerate(FEATURE_SETS):
        if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[FEATURE_SETS[i - 1]]:
            lines.append(r"\midrule")
        cells = [FEATURE_SET_DISPLAY.get(fs, fs)]
        for ds in datasets:
            for tc in DATASETS[ds]["target_cols"]:
                for _, scope_label in SCOPES:
                    for scheme, _ in SCHEMES:
                        if fs in FIXED_TEXT_ONLY_FEATURE_SETS and scope_label != "Fixed":
                            cells.append("-")
                            continue
                        cells.append(_fmt(
                            lookup.get((ds, fs, tc, scope_label, scheme)),
                            bold=False, diff=True))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "}",
        (r"\captionof{table}{\textbf{The effect of EyeScore L1 debiasing on "
         r"correlations with standard proficiency tests}. Presented is "
         r"$\Delta r_{\mathrm{db}}$, the Pearson $r$ of "
         r"EyeScore\textsuperscript{db} with each standard English proficiency "
         r"test minus the Pearson $r$ of EyeScore with the same test. "
         r"Seen L1 / New L1: did the data for computing the debiased score "
         r"include participants from the same L1 as the test participant.}"),
        f"\\label{{{label}}}",
        f"% \\end{{{env}}}",
    ]
    return "\n".join(lines)


def _render_rawdb(lookup: dict, datasets: list[str], label: str,
                  calib_phrase: str) -> str:
    """Render r(EyeScore, EyeScore^db): the Δ table's layout minus the target axis.

    The correlation is between the two score vectors, so there is nothing to
    split by proficiency test -- each dataset gets one block of
    (Fixed, Any) x (Seen L1, New L1).
    """
    n_data_cols = len(datasets) * len(SCOPES) * len(SCHEMES)
    n_total_cols = 1 + n_data_cols
    single_dataset = len(datasets) == 1

    env = "table" if single_dataset else "table*"
    width = r"\columnwidth" if single_dataset else r"\textwidth"

    lines = [
        f"% \\begin{{{env}}}[ht!]",
        r"\centering",
        r"\setlength{\aboverulesep}{0pt}",
        r"\setlength{\belowrulesep}{0pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        f"\\resizebox{{{width}}}{{!}}{{",
        "\\begin{tabular}{l" + "|".join(["cccc"] * len(datasets)) + "}",
        r"\toprule",
    ]

    # Row 1: dataset names, one block of four columns each.
    lines.append(
        "\\multicolumn{1}{c}{} & "
        + " & ".join(
            f"\\multicolumn{{{len(SCOPES) * len(SCHEMES)}}}{{c}}"
            f"{{\\textbf{{{DATASETS[ds]['display']}}}}}"
            for ds in datasets
        )
        + r" \\"
    )

    # Row 2: regimes. A full rule closes every dataset but the last, so the
    # block boundary is the only vertical division in the header.
    regime_cells = []
    for ds_idx, _ in enumerate(datasets):
        last_ds = ds_idx == len(datasets) - 1
        for scope_idx, (_, scope) in enumerate(SCOPES):
            last_scope = scope_idx == len(SCOPES) - 1
            align = "c|" if (last_scope and not last_ds) else "c"
            regime_cells.append(f"\\multicolumn{{2}}{{{align}}}{{{scope}}}")
    lines.append("\\multicolumn{1}{c}{} & " + " & ".join(regime_cells)
                 + f" \\\\ \\cmidrule{{2-{n_total_cols}}}")

    # Row 3: scheme labels, one per physical column.
    lines.append(
        r"\textbf{Features} & "
        + " & ".join(f"{label_} L1"
                     for _ in range(len(datasets) * len(SCOPES))
                     for _, label_ in SCHEMES)
        + r" \\"
    )
    lines.append(r"\midrule")

    for i, fs in enumerate(FEATURE_SETS):
        if i > 0 and FEATURE_SET_GROUP_INDEX[fs] != FEATURE_SET_GROUP_INDEX[FEATURE_SETS[i - 1]]:
            lines.append(r"\midrule")
        cells = [FEATURE_SET_DISPLAY.get(fs, fs)]
        for ds in datasets:
            for _, scope_label in SCOPES:
                for scheme, _ in SCHEMES:
                    if fs in FIXED_TEXT_ONLY_FEATURE_SETS and scope_label != "Fixed":
                        cells.append("-")
                        continue
                    val = lookup.get((ds, fs, scope_label, scheme))
                    cells.append("-" if val is None or pd.isna(val) else f"{val:.3f}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "}",
        (r"\captionof{table}{\textbf{Agreement between EyeScore and its L1-debiased "
         r"counterpart}. Presented is the Pearson $r$ between EyeScore and "
         r"EyeScore\textsuperscript{db}, where the debiasing is calibrated "
         f"against {calib_phrase}. "
         r"Seen L1 / New L1: did the data for computing the debiased score "
         r"include participants from the same L1 as the test participant. Each "
         r"correlation is computed over the participants scored under both "
         r"variants; the Seen L1 scheme excludes any L1 with fewer participants "
         r"than the number of cross-validation folds (on OneStopL2 this drops "
         r"Vietnamese, $n=5$).}"),
        f"\\label{{{label}}}",
        f"% \\end{{{env}}}",
    ]
    return "\n".join(lines)


VARIANTS = ("table", "diff", "rawcorr")

# method -> the variants of the LexTALE-calibrated table the paper reports.
PUBLISHED = {
    "two_step": ("diff", "rawcorr"),
    "distance_mreg": ("diff",),
    "interaction": ("diff",),
}


def render_tables(results_dir: Path, out_dir: Path, calib: str = "lextale",
                  variants: tuple = VARIANTS,
                  distance_type: str = PAPER_DISTANCE_TYPE,
                  center_distance: bool = True,
                  n_bootstrap: int = N_BOOTSTRAP) -> list[Path]:
    """Render the requested variants of one calibration's table for one tree.

    distance_type MUST be the typological distance the results tree was built
    with, or the Δ test refits a different debiasing coefficient from the one
    the stored debiased scores used. center_distance must likewise match
    whether the tree centred distances on the train-fold mean (the '*ctr'
    trees); it only affects the Δ significance test, not the correlations.
    """
    spec = next(s for s in OUTPUTS if s["calib"] == calib)
    stem = spec["filename"].removesuffix(".tex")
    datasets = spec["datasets"]
    written = []

    if "table" in variants:
        lookup = _compute_lookup(results_dir, calib, datasets)
        if lookup:
            out = Path(out_dir) / f"{stem}.tex"
            write_table(out, _render(lookup, datasets, label=spec["label"],
                                     calib_phrase=spec["calib_phrase"]))
            written.append(out)
        else:
            logger.warning("No data for {}; skipping", spec["filename"])

    if "rawcorr" in variants:
        # r(EyeScore, EyeScore^db): how much the debiasing moved the score at
        # all. Test-independent, so the Δ layout collapses to one block per
        # dataset.
        rawdb_lookup = _compute_rawdb_lookup(results_dir, calib, datasets)
        if rawdb_lookup:
            out = Path(out_dir) / f"{stem}_rawcorr.tex"
            write_table(out, _render_rawdb(rawdb_lookup, datasets,
                                           label=f"{spec['label']}-rawcorr",
                                           calib_phrase=spec["calib_phrase"]))
            written.append(out)
        else:
            logger.warning("No raw/db data for {}; skipping rawcorr", spec["filename"])

    if "diff" in variants:
        # Δ variant: same layout, cells show (debiased − raw) Pearson r with a
        # significance star. Recomputed from the score files rather than
        # subtracting the plain table, so both correlations use the same
        # participants (see _compute_diff_lookup).
        diff_lookup = _compute_diff_lookup(results_dir, calib, datasets,
                                           distance_type, center_distance, n_bootstrap)
        if diff_lookup:
            out = Path(out_dir) / f"{stem}_diff.tex"
            write_table(out, _render_diff(diff_lookup, datasets,
                                          label=f"{spec['label']}-diff"))
            written.append(out)
        else:
            logger.warning("No Δ data for {}; skipping diff", spec["filename"])
    return written


def render(out_root: Path) -> list[Path]:
    written = []
    for method, variants in PUBLISHED.items():
        written += render_tables(
            Path("src/methods/EyeScore") / PAPER_TREES[method],
            Path(out_root) / "lang_bias" / method / "tables",
            calib="lextale", variants=variants)
    return written
