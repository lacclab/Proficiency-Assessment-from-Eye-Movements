"""The EyeScore L1-bias tables: paper/lang_bias/<method>/, one tree per debias method.

    coef/, pearson/          the grid -- {combined, meco, onestop} x {Any, Fixed},
                             plus OneStop Information Seeking -- with every block
                             debiased by the LexTALE-calibrated correction
    composite_debiasing/,    the same grid with every block of a dataset debiased
    michigan_debiasing/      on that dataset's own non-LexTALE test
    tables/eyescore_distance.tex
                             raw EyeScore vs distance; method-independent, so the
                             same table sits in every tree

The statistic throughout is the two-step distance coefficient b_dist:
residualize eye_score on the block's proficiency measure, then regress that
residual on the participant's L1->English typological distance. coef/ reports
the slope and pearson/ the correlation r of the same regression, so p-values and
stars are identical between them. The left column of each block fits the raw
eye_score; the Seen L1 / New L1 columns refit on the held-out debiased scores
(inside- / across-language CV).

``debias_calibration`` (an argument of `build_table`) selects which debiased
score files the Seen/New columns read:

    per_target  each block reads the correction calibrated on its own target
                (MECO Composite -> *_profagg, OneStop Michigan -> *_michtest).
    lextale     every block reads the LexTALE-calibrated correction
                (the unsuffixed *_typo_calibrated_{inside,across}_langs.csv).
                The b_dist fit still residualizes on the block's own target, so
                the columns stay distinct -- only the debiasing changes.
"""
from __future__ import annotations

import os
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import linregress

from src.evaluation.EyeScore.debias_tests import N_BOOTSTRAP, delta_b_test
from src.evaluation.EyeScore.evaluation import load_metadata
from src.evaluation.EyeScore.tables import generate_eyescore_distance_table
from src.latex_captions import write_table
from src.methods.EyeScore.typology import compute_typological_distances


# The three debias trees the paper reports and the distance metric they were
# built with (src.run.eyescore --debias-methods ... --results-suffix cat3ctr).
PAPER_TREES = {
    "two_step": "results_two_step_cat3ctr",
    "distance_mreg": "results_distance_mreg_cat3ctr",
    "interaction": "results_interaction_cat3ctr",
}
PAPER_DISTANCE_TYPE = "cat:syntactic+phonological+scriptural"


# Single-tree defaults for `build_table`.
RESULTS_TREE = PAPER_TREES["two_step"]
DISTANCE_TYPE = PAPER_DISTANCE_TYPE

# (dataset, dataset display, target display, prof col, per-target debias suffix,
#  metadata version_path, results subdir template)
COMBINED_COLSPEC = [
    ("Meco", "MECO", "LexTALE", "lextale_score", "",
     "fully_agg/all/all", "Meco/All/{cm}/fully_agg/all/all"),
    ("Meco", "MECO", "Composite", "proficiency_agg", "_profagg",
     "fully_agg/all/all", "Meco/All/{cm}/fully_agg/all/all"),
    ("OneStop", "OneStopL2", "LexTALE", "lextale_score", "",
     "fully_agg/ordinary/all", "OneStop/Gathering/{cm}/fully_agg/ordinary/all"),
    ("OneStop", "OneStopL2", "Michigan", "michtest_score", "_michtest",
     "fully_agg/ordinary/all", "OneStop/Gathering/{cm}/fully_agg/ordinary/all"),
]
HUNTING_COLSPEC = [
    ("OneStop", "OneStopL2 (Information Seeking)", "LexTALE", "lextale_score", "",
     "fully_agg/ordinary/all", "OneStop/Hunting/{cm}/fully_agg/ordinary/all"),
    ("OneStop", "OneStopL2 (Information Seeking)", "Michigan", "michtest_score", "_michtest",
     "fully_agg/ordinary/all", "OneStop/Hunting/{cm}/fully_agg/ordinary/all"),
]
# Column subsets of the combined colspec, selected by dataset rather than by
# index so a reordering of COMBINED_COLSPEC cannot silently mis-slice them.
MECO_COLSPEC = [c for c in COMBINED_COLSPEC if c[0] == "Meco"]
ONESTOP_COLSPEC = [c for c in COMBINED_COLSPEC if c[0] == "OneStop"]

# Per-text feature sets exist only in the Fixed Text (seen) trees -- an Any Text
# score needs a shared paragraph column space, which crossing content groups
# (OneStop) or odd/even halves (MECO) does not have.
TABULAR_FROWS = [
    ("READING_SPEED", "WPM"),
    ("FIXATION_METRICS", "Avg. Fix."),
    ("S_CLUSTERS_NO_NORM", "S-Clusters"),
    ("WP_COEFS_NO_NORM", "WP-Coefs"),
]
PER_TEXT_FROWS = [("TRANSITIONS", "Transitions"), ("WFC", "Word Fix.")]

# (content_mode dir, label suffix, caption regime phrase, extra feature rows)
REGIMES = {
    "unseen": ("unseen", "", "in the Any Text regime.", []),
    "seen": ("seen", "-seen", r"in the \textbf{Fixed Text} regime.", PER_TEXT_FROWS),
}


# ── the fit ──────────────────────────────────────────────────────────────────

def _fit(path, prof: str, dist_map: dict, meta: pd.DataFrame, stat: str = "coef"):
    """Two-step fit on the eye_score CSV at `path` (or a frame); (slope|r, p) or None.

    Residualize eye_score on `prof`, regress the residual on distance, and
    report linregress's slope (stat="coef") or its rvalue (stat="r").
    """
    if isinstance(path, pd.DataFrame):
        df = path
    elif not os.path.exists(path):
        return None
    else:
        df = pd.read_csv(path)
    df = df[["participant_id", "eye_score", "L1"]].merge(
        meta[["participant_id", prof]], on="participant_id")
    df = df[df[prof] != -1].dropna(subset=["eye_score", prof, "L1"]).copy()
    df["dist"] = df["L1"].map(dist_map)
    df = df.dropna(subset=["dist"])
    if len(df) < 10 or df["L1"].nunique() < 2:
        return None
    slope, intercept = np.polyfit(df[prof].to_numpy(float),
                                  df["eye_score"].to_numpy(float), 1)
    resid = df["eye_score"].to_numpy(float) - (
        slope * df[prof].to_numpy(float) + intercept)
    lr = linregress(df["dist"].to_numpy(float), resid)
    return float(lr.slope if stat == "coef" else lr.rvalue), float(lr.pvalue)


def _b_dist(path, prof: str, dist_map: dict, meta: pd.DataFrame):
    """The distance coefficient b_dist: the slope of the two-step fit."""
    return _fit(path, prof, dist_map, meta, "coef")


# Caption rewrites for the Pearson variant of a coef table.
_CAP_SUBS = [
    (r"'EyeScore' is the fitted coefficient $b_{\mathrm{dist}}$ of "
     r"EyeScore$\sim$target residual on the linguistic distance of the "
     r"participant's L1 to English ($res(p) \sim d(L1_p, Eng)$).",
     r"'EyeScore' is the Pearson correlation $r$ between the "
     r"EyeScore$\sim$target residual and the linguistic distance of the "
     r"participant's L1 to English ($corr(res(p), d(L1_p, Eng))$)."),
    (r"$\mathrm{EyeScore}^{\mathrm{db}}$ is this coefficient refitted on the",
     r"$\mathrm{EyeScore}^{\mathrm{db}}$ is this correlation recomputed on the"),
]


def _delta_b(base: str, scheme_suffix: str, prof: str, calib_prof: str,
             dist_map: dict, meta: pd.DataFrame, scheme: str,
             center: bool = True, n_boot: int = N_BOOTSTRAP):
    """(b_db, p) where p tests the CHANGE from the raw coefficient, not b_db != 0.

    Used for the `EyeScore^db` columns under star_db_diff. The coefficient
    shown is unchanged; only the p-value behind the stars differs. The bootstrap
    refits the debiasing coefficient inside every resample -- reusing the stored
    debiased scores would treat it as a known constant and star every cell.
    """
    raw_path, db_path = f"{base}.csv", f"{base}{scheme_suffix}.csv"
    if not (os.path.exists(raw_path) and os.path.exists(db_path)):
        return None
    raw_df, db_df = pd.read_csv(raw_path), pd.read_csv(db_path)
    b_db = _b_dist(db_path, prof, dist_map, meta)
    if b_db is None:
        return None
    res = delta_b_test(raw_df, db_df, meta, prof, calib_prof, dist_map, scheme,
                       center=center, n_boot=n_boot,
                       cache_key=(raw_path, db_path, prof, calib_prof))
    if res is None:
        return None
    return b_db[0], res[1]


# ── the table ────────────────────────────────────────────────────────────────

def _n_stars(v) -> int:
    if v is None:
        return 0
    p = v[1]
    return 3 if p < .001 else 2 if p < .01 else 1 if p < .05 else 0


def _cell(v, max_stars: int) -> str:
    """Format one coefficient, padding the star slot to the column's widest so
    the decimal points line up (same convention as the bias-triplet tables)."""
    if v is None:
        return r"$-$"
    coef = v[0]
    n = _n_stars(v)
    txt = f"{coef:.2f}"
    if txt.startswith("-") and float(txt) == 0:
        txt = txt[1:]          # a value that rounds to zero is not "-0.00"
    body = txt if txt.startswith("-") else r"\phantom{-}" + txt
    if n:
        body += "^{" + "*" * n + "}"
    if max_stars > n:
        body += r"\phantom{^{" + "*" * (max_stars - n) + "}}"
    return f"${body}$"


def _header_lines(colspec: list, n_cols: int) -> list:
    """Dataset row, target row, EyeScore/debiased row, feature row."""
    ds_groups = []  # (display, n_targets) in order
    for _, ds_disp, *_ in colspec:
        if ds_groups and ds_groups[-1][0] == ds_disp:
            ds_groups[-1][1] += 1
        else:
            ds_groups.append([ds_disp, 1])

    ds_cells = []
    for i, (disp, n_t) in enumerate(ds_groups):
        align = "c" if i == len(ds_groups) - 1 else "c|"
        ds_cells.append(f"\\multicolumn{{{3 * n_t}}}{{{align}}}{{\\textbf{{{disp}}}}}")

    tgt_cells, db_cells = [], []
    seen_in_ds = 0
    ds_idx = 0
    for i, (_, _, tgt_disp, *_rest) in enumerate(colspec):
        last_overall = i == len(colspec) - 1
        seen_in_ds += 1
        last_in_ds = seen_in_ds == ds_groups[ds_idx][1]
        # A rule after the last target of a dataset, except the final dataset.
        tgt_align = "c" if (last_overall or not last_in_ds) else "c|"
        tgt_cells.append(f"\\multicolumn{{3}}{{{tgt_align}}}{{{tgt_disp}}}")
        db_align = "c" if last_overall else "c|"
        db_cells.append(r"\multicolumn{1}{c}{EyeScore} & "
                        f"\\multicolumn{{2}}{{{db_align}}}"
                        r"{$\mathrm{EyeScore}^{\mathrm{db}}$}")
        if last_in_ds:
            ds_idx += 1
            seen_in_ds = 0

    return [
        " & " + " & ".join(ds_cells) + r" \\",
        " & " + " & ".join(tgt_cells) + f" \\\\ \\cmidrule{{2-{n_cols}}}",
        " & " + " & ".join(db_cells) + f" \\\\ \\cmidrule{{2-{n_cols}}}",
        r"\textbf{Features}"
        + r" &  & \textit{Seen L1} & \textit{New L1}" * len(colspec) + r" \\",
        f"\\midrule \\cmidrule{{1-{n_cols}}}",
    ]


def build_table(colspec: list, regime: str, debias_calibration: str,
                results_tree: str = RESULTS_TREE,
                distance_type: str = DISTANCE_TYPE,
                label_suffix_extra: str = "",
                single_column: bool = False,
                star_db_diff: bool = False,
                center_distance: bool = True,
                n_bootstrap: int = N_BOOTSTRAP,
                fit=None) -> "str | None":
    """Build one coefficient table.

    fit: the per-cell statistic, ``fit(csv_path, prof, dist_map, meta) ->
        (value, p) | None``. Defaults to b_dist, the slope; the Pearson tables
        pass the correlation of the same regression instead.

    star_db_diff: stars on the `EyeScore^db` columns test whether the
        coefficient CHANGED from the raw one (Δb_dist, paired participant
        bootstrap with the debiasing coefficient refitted per resample) rather
        than whether the debiased coefficient is itself non-zero. Displayed
        values are identical either way; only the stars move. The Δ test is
        defined on the slope, so it ignores `fit`.

        Note the two tests coincide whenever the debias calibration target
        equals the block's own target -- the debiasing coefficient and b_dist
        are then the same statistic on a training fold versus on everyone. The
        flag adds information only under cross-target calibration (e.g.
        lextale calibration on the Michigan/Composite blocks).
    """
    fit = fit if fit is not None else _b_dist
    content_mode, label_suffix, regime_phrase, extra_rows = REGIMES[regime]
    frows = TABULAR_FROWS + extra_rows

    # A dataset without metadata (OneStop, where only the MECO data exists) has
    # no distances to fit against; its columns are drawn as missing cells.
    meta_by_ds, dist_by_ds = {}, {}
    for ds, _, _, _, _, version_path, _ in colspec:
        if ds not in meta_by_ds:
            meta = load_metadata(ds, version_path)
            meta_by_ds[ds] = meta
            dist_by_ds[ds] = (compute_typological_distances(
                sorted(meta["L1"].dropna().unique()), distance_type=distance_type)
                if "L1" in meta.columns else None)

    # values[row][col] -> (coef, p) | None; three physical columns per colspec entry.
    values = []
    for fs, _ in frows:
        row = []
        for ds, _, _, prof, suf, _, subdir_tpl in colspec:
            if dist_by_ds[ds] is None:
                row += [None, None, None]
                continue
            # LexTALE calibration reads the unsuffixed correction for every
            # block, so the coefficient alpha was fitted on LexTALE regardless
            # of which target this block reports against.
            calib_prof = "lextale_score" if debias_calibration == "lextale" else prof
            if debias_calibration == "lextale":
                suf = ""
            base = (f"src/methods/EyeScore/{results_tree}/"
                    f"{subdir_tpl.format(cm=content_mode)}/{fs}")
            row.append(fit(f"{base}.csv", prof, dist_by_ds[ds], meta_by_ds[ds]))
            for scheme in ("inside_langs", "across_langs"):
                scheme_suffix = f"_typo_calibrated_{scheme}{suf}"
                if star_db_diff:
                    row.append(_delta_b(base, scheme_suffix, prof, calib_prof,
                                        dist_by_ds[ds], meta_by_ds[ds], scheme,
                                        center=center_distance,
                                        n_boot=n_bootstrap))
                else:
                    row.append(fit(f"{base}{scheme_suffix}.csv",
                                   prof, dist_by_ds[ds], meta_by_ds[ds]))
        values.append(row)

    if not any(v is not None for row in values for v in row):
        logger.warning("No data for {} / {}; skipping", regime, debias_calibration)
        return None

    n_phys = len(colspec) * 3
    max_stars = [max(_n_stars(values[r][c]) for r in range(len(values)))
                 for c in range(n_phys)]

    rows = []
    for (_, flab), row in zip(frows, values):
        rows.append(f"{flab} & " + " & ".join(
            _cell(v, max_stars[c]) for c, v in enumerate(row)) + r" \\")

    n_cols = n_phys + 1
    spec = "l" + "|".join(["ccc"] * len(colspec))

    # The Hunting table covers OneStop alone, so it is half the width of the
    # combined ones and sits in a single column.
    env = "table" if single_column else "table*"
    width = r"\columnwidth" if single_column else r"\textwidth"
    lines = [
        f"\\begin{{{env}}}[ht!]", r"\centering",
        f"\\resizebox{{{width}}}{{!}}{{%",
        f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
    ]
    lines += _header_lines(colspec, n_cols)
    # WPM is the baseline; the tabular sets and the per-text sets each get a rule.
    lines += rows[:1] + [r"\midrule"] + rows[1:4]
    if len(rows) > 4:
        lines += [r"\midrule"] + rows[4:]
    lines[-1] = lines[-1] + r" \bottomrule"

    if debias_calibration == "lextale":
        calib_phrase = (
            "debiasing uses a residual regression model fitted on the LexTALE"
        )
    else:
        calib_phrase = (
            "debiasing uses a residual regression model fitted on the "
            "proficiency measure of that block"
        )

    if star_db_diff:
        sig_phrase = (
            r"Stars on the \textbf{EyeScore} column test $b_{\mathrm{dist}} \neq 0$; "
            r"stars on the $\mathrm{EyeScore}^{\mathrm{db}}$ columns test whether "
            r"the coefficient \emph{changed} from the raw one "
            r"($\Delta b_{\mathrm{dist}}$), via a paired participant bootstrap in "
            r"which the debiasing coefficient is refitted within every resample. "
        )
    else:
        sig_phrase = r"Statistical significance for the L1-bias coefficients:  "

    lines += [
        r"\end{tabular}%", r"}",
        (r"\caption{\textbf{EyeScore L1 Bias} " + regime_phrase +
         r" 'EyeScore' is the fitted coefficient $b_{\mathrm{dist}}$ of "
         r"EyeScore$\sim$target residual on the linguistic distance of the "
         r"participant's L1 to English ($res(p) \sim d(L1_p, Eng)$). "
         r"$\mathrm{EyeScore}^{\mathrm{db}}$ is this coefficient refitted on the "
         r"held-out debiased scores, where " + calib_phrase +
         r". Seen / New L1: the data for computing the debiased score included / "
         r"did not include participants from the same L1 as the test participant. "
         + sig_phrase +
         r"$^{*}\,p<0.05$, $^{**}\,p<0.01$, $^{***}\,p<0.001$.}"),
        r"\label{tab:lang-bias-triplet-coef-two_step" + label_suffix
        + label_suffix_extra + "}",
        f"\\end{{{env}}}",
    ]
    return "\n".join(lines)


# ── coef/ and pearson/: the grid ─────────────────────────────────────────────

# (scope dir, output stem, colspec, regime key, single-column layout)
# Only the 4-block combined table needs the full text width; every scoped table
# is half of it and sits in one column, as the Hunting table already did.
CELLS = [
    ("combined", "any",           COMBINED_COLSPEC, "unseen", False),
    ("combined", "fixed",         COMBINED_COLSPEC, "seen",   False),
    ("meco",     "any",           MECO_COLSPEC,     "unseen", True),
    ("meco",     "fixed",         MECO_COLSPEC,     "seen",   True),
    ("onestop",  "any",           ONESTOP_COLSPEC,  "unseen", True),
    ("onestop",  "fixed",         ONESTOP_COLSPEC,  "seen",   True),
    ("onestop",  "hunting_any",   HUNTING_COLSPEC,  "unseen", True),
    ("onestop",  "hunting_fixed", HUNTING_COLSPEC,  "seen",   True),
]

STATS = {"coef": "coef", "pearson": "r"}   # output dir -> statistic passed to _fit


def build_cell(colspec: list, regime: str, stat: str, calibration: str,
               results_tree: str, distance_type: str,
               label_extra: str, single_column: bool,
               star_db_diff: bool = False,
               center_distance: bool = True,
               n_bootstrap: "int | None" = None) -> "str | None":
    """One table. `stat` is _fit's statistic name ('coef' or 'r')."""
    tex = build_table(colspec, regime, calibration,
                      results_tree=results_tree,
                      distance_type=distance_type,
                      label_suffix_extra=label_extra,
                      single_column=single_column,
                      # The Δ test is defined on the slope, so it only
                      # matches what the 'coef' pass displays. The 'r'
                      # pass shows the correlation of the same regression
                      # and keeps its own stars.
                      star_db_diff=star_db_diff and stat == "coef",
                      center_distance=center_distance,
                      fit=partial(_fit, stat=stat),
                      **({} if n_bootstrap is None
                         else {"n_bootstrap": n_bootstrap}))
    if tex is None:
        return None
    if stat == "r":
        # The generator's caption is written for the slope; a table of
        # correlations that calls itself a fitted coefficient is worse than no
        # caption at all.
        for old, new in _CAP_SUBS:
            tex = tex.replace(old, new)
    return tex


def render_grid(results_tree: str, out_dir: Path, tag: str,
                debias_calibration: str = "lextale",
                distance_type: str = PAPER_DISTANCE_TYPE,
                star_db_diff: bool = False, center_distance: bool = True,
                n_bootstrap: "int | None" = None) -> list[Path]:
    """The full {coef,pearson} x scope x regime grid for one tree, into out_dir."""
    written = []
    for stat_dir, stat in STATS.items():
        for scope, stem, colspec, regime, single in CELLS:
            # Labels must be unique across the whole grid, or LaTeX silently
            # resolves every \ref to the last one defined.
            tex = build_cell(colspec, regime, stat, debias_calibration,
                             results_tree, distance_type,
                             f"-{tag}-{scope}-{stem}-{stat_dir}", single,
                             star_db_diff=star_db_diff,
                             center_distance=center_distance,
                             n_bootstrap=n_bootstrap)
            out = Path(out_dir) / stat_dir / scope / f"{stem}.tex"
            if tex is None:
                logger.warning("skipped (no data): {}", out)
                continue
            write_table(out, tex + "\n")
            written.append(out)
    return written


# ── composite_debiasing/ and michigan_debiasing/ ─────────────────────────────
#
# The grid again, but with every block of a dataset reading the correction
# fitted on that dataset's OWN non-LexTALE test. This is NOT the `per_target`
# calibration: under per_target the LexTALE block keeps its LexTALE-fitted
# correction and only the Composite/Michigan block moves; here the LexTALE block
# is corrected with Composite/Michigan too, so the pair is comparable --
# "LexTALE and Michigan, both debiased by Michigan" against "Michigan alone,
# debiased by Michigan".
#
# Mechanically that is a forced `suf`: COMBINED_COLSPEC carries '' on the
# LexTALE blocks and _profagg / _michtest on the others, and build_table appends
# it to the debiased filename. Overriding every entry's `suf` to one value, with
# calibration="per_target" so build_table leaves it alone, reads the one
# correction for every block. The per-target corrections are computed in the
# same pass as the LexTALE ones, so no extra pipeline run is involved.
#
#     <target>_debiasing/{coef,pearson}/{full,<target>}_{any,fixed}.tex
#
# `full` = both of the dataset's tests; `<target>` = that test's block alone.

# target key -> (forced suffix, dataset, block label, single-table stem, caption phrase)
TARGETS = {
    "composite": ("_profagg",  "Meco",    "Composite", "composit",
                  "the MECO L2 Composite score"),
    "michigan":  ("_michtest", "OneStop", "Michigan",  "michigan",
                  "the Michigan test score"),
}
REGIME_OF = {"any": "unseen", "fixed": "seen"}

# build_table's per_target wording describes each block reading its own target,
# which is precisely what a forced suffix stops being true.
_PER_TARGET_PHRASE = ("debiasing uses a residual regression model fitted on the "
                      "proficiency measure of that block")


def _forced(colspec: list, suffix: str) -> list:
    """colspec with every entry's `suf` (index 4) replaced by `suffix`."""
    return [e[:4] + (suffix,) + e[5:] for e in colspec]


def render_target_debiasing(results_tree: str, out_dir: Path, tag: str,
                            distance_type: str = PAPER_DISTANCE_TYPE) -> list[Path]:
    """The composite_/michigan_debiasing tables for one tree, into out_dir."""
    written = []
    for target, (suffix, dataset, block, stem, phrase) in TARGETS.items():
        ds_blocks = [e for e in COMBINED_COLSPEC if e[0] == dataset]
        cells = [
            ("full", _forced(ds_blocks, suffix), False),
            (stem,   _forced([e for e in ds_blocks if e[2] == block], suffix), True),
        ]
        for stat_dir, stat in STATS.items():
            for name, colspec, single in cells:
                for regime_name, regime in REGIME_OF.items():
                    tex = build_cell(
                        colspec, regime, stat, "per_target",
                        results_tree, distance_type,
                        f"-{tag}-{target}db-{name}-{regime_name}-{stat_dir}", single)
                    out = (Path(out_dir) / f"{target}_debiasing" / stat_dir
                           / f"{name}_{regime_name}.tex")
                    if tex is None:
                        logger.warning("skipped (no data): {}", out)
                        continue
                    tex = tex.replace(
                        _PER_TARGET_PHRASE,
                        f"debiasing uses a residual regression model fitted on "
                        f"{phrase} for every block")
                    write_table(out, tex + "\n")
                    written.append(out)
    return written


# ── tables/eyescore_distance.tex ─────────────────────────────────────────────

def render_eyescore_distance(out_dirs: list[Path]) -> list[Path]:
    """Raw eye_score vs typological distance. It reads the raw scores, so it is
    identical for every debias method: drawn once, then copied into each tree."""
    first, *rest = [Path(d) / "tables" / "eyescore_distance.tex" for d in out_dirs]
    generate_eyescore_distance_table(save_path=first, distance_type=PAPER_DISTANCE_TYPE)
    for out in rest:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(first, out)
    return [first, *rest]


def render(out_root: Path) -> list[Path]:
    trees = {method: Path(out_root) / "lang_bias" / method for method in PAPER_TREES}
    written = []
    for method, results_tree in PAPER_TREES.items():
        written += render_grid(results_tree, trees[method], method)
        written += render_target_debiasing(results_tree, trees[method], method)
    written += render_eyescore_distance(list(trees.values()))
    return written
