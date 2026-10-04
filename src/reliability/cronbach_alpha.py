"""Per-item reliability of EyeScore and predictions: Cronbach's alpha, the
per-item split-half (Spearman-Brown corrected), the six Shrout & Fleiss ICCs
and the SEM, per content group and averaged for OneStop. Writes
results/per_item_reliability/[<preview>/]per_item_reliability.csv, which
tables.py renders.
"""
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import Fields, SMALL_FEATURE_SETS
from src.reliability.utils import (
    DATASET_TARGET_COLS,
    DEFAULT_PREVIEW,
    add_content_group_from_metadata,
    load_eyescore_raw_data_with_pool,
    load_predictions_raw_data,
    onestop_preview,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("src/reliability")
PER_ITEM_RELIABILITY_DIR = RESULTS_DIR / "results" / "per_item_reliability"


def per_item_reliability_dir(preview: str) -> Path:
    """Output dir for a preview: the top-level path for Gathering, a subdir otherwise."""
    if preview == DEFAULT_PREVIEW:
        return PER_ITEM_RELIABILITY_DIR
    return PER_ITEM_RELIABILITY_DIR / preview.lower()


def split_half_reliability_items(
    df: pd.DataFrame,
    value_col: str = "eye_score",
    participant_col: str = "participant_id",
    item_col: str = None,
    n_iterations: int = 100,
    random_state: int = 42,
    strata: dict = None,
) -> float:
    """Split-half reliability over `n_iterations` random item splits.

    Each split is the same for every participant: their mean score on one half
    of the items is correlated with their mean on the other, across
    participants, and the mean correlation is Spearman-Brown corrected. NaN
    when no split yields a correlation. Without `item_col`, rows stand in for
    items.

    `strata` ({item: stratum}) draws each half as half of every stratum's items,
    so the halves match in composition instead of being a uniform draw over the
    whole item pool.
    """
    if len(df) < 2 or participant_col not in df.columns or value_col not in df.columns:
        return np.nan

    # Get unique items
    if item_col and item_col in df.columns:
        unique_items = df[item_col].unique()
    else:
        # Use row indices as items
        unique_items = np.arange(len(df))

    n_items = len(unique_items)
    n_half = n_items // 2
    if n_half < 1:
        return np.nan

    rng = np.random.RandomState(random_state)
    correlations = []

    # Index positions of each stratum's items, so a split can be drawn within stratum.
    if strata is not None:
        stratum_of = np.array([strata[item] for item in unique_items])
        strata_indices = [np.flatnonzero(stratum_of == s) for s in pd.unique(stratum_of)]

    for _ in tqdm(range(n_iterations), desc="Computing split-half reliability", leave=False):
        # Randomly split items (same split for all participants)
        if strata is not None:
            # Half of each stratum, with an odd stratum's leftover item going to
            # either half at random so neither half is systematically longer.
            picked = [rng.choice(idx, size=len(idx) // 2 + int(len(idx) % 2 and rng.rand() < 0.5),
                                 replace=False)
                      for idx in strata_indices]
            indices_half1 = set(int(i) for i in np.concatenate(picked)) if picked else set()
        else:
            indices_half1 = set(rng.choice(n_items, size=n_half, replace=False))
        indices_half2 = set(range(n_items)) - indices_half1

        # Get actual item values for the two halves
        half1_items = set(unique_items[list(indices_half1)])
        half2_items = set(unique_items[list(indices_half2)])

        half1_means = []
        half2_means = []

        # For each participant, calculate mean scores for each half
        for _, group in df.groupby(participant_col):
            values = group[value_col].values

            # Filter by item membership
            if item_col and item_col in group.columns:
                half1_mask = group[item_col].isin(half1_items)
                half2_mask = group[item_col].isin(half2_items)
            else:
                # Use row indices
                half1_mask = np.arange(len(group)) < len(half1_items)
                half2_mask = np.arange(len(group)) >= len(half1_items)

            half1_vals = values[half1_mask]
            half2_vals = values[half2_mask]

            # Skip if either half has no valid data
            half1_vals_clean = half1_vals[~np.isnan(half1_vals)]
            half2_vals_clean = half2_vals[~np.isnan(half2_vals)]

            if len(half1_vals_clean) < 1 or len(half2_vals_clean) < 1:
                continue

            half1_means.append(np.mean(half1_vals_clean))
            half2_means.append(np.mean(half2_vals_clean))

        # Need at least 3 participants for meaningful correlation
        if len(half1_means) < 3:
            continue

        # Correlate half1 means with half2 means across participants
        corr = np.corrcoef(half1_means, half2_means)[0, 1]
        if not np.isnan(corr) and -1 < corr < 1:
            correlations.append(corr)

    if not correlations:
        return np.nan

    # Average correlation across all iterations
    avg_corr = np.mean(correlations)

    # Apply Spearman-Brown prophecy formula
    if avg_corr >= 1 or avg_corr <= -1:
        return np.nan
    alpha = 2 * avg_corr / (1 + avg_corr)
    return float(alpha)


def cronbachs_alpha_single(df: pd.DataFrame, value_col: str, item_col: str) -> float:
    """Cronbach's alpha of one participants x items matrix:
    α = (k/(k-1)) × (1 - Σ Vᵢ / Vₜ), Vᵢ the variance of item i across
    participants and Vₜ that of the total score. NaN with fewer than two
    participants or items.
    """
    if len(df) < 2:
        return np.nan

    # Create pivot: subjects × items
    try:
        pivot = df.pivot_table(
            index='participant_id',
            columns=item_col,
            values=value_col,
            aggfunc='first'
        )
    except Exception:
        return np.nan

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return np.nan

    # Remove all-NaN rows/columns
    pivot = pivot.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return np.nan

    # Variance per item (column) - across subjects
    item_vars = pivot.var(ddof=1)
    sum_item_vars = item_vars.sum()

    # Total score variance - sum of each subject's scores across items
    # Handle NaN by computing sum only on non-NaN values per subject
    total_scores = pivot.sum(axis=1, skipna=True)
    if len(total_scores) < 2:
        return np.nan

    var_total = total_scores.var(ddof=1)
    if var_total == 0:
        return np.nan

    k = pivot.shape[1]
    alpha = (k / (k - 1)) * (1 - sum_item_vars / var_total)
    return float(alpha)


# The six Shrout & Fleiss ICCs, with participants as targets and items as raters.
# (1,*) one-way: each participant read their own set of items, so item effects are
#       inseparable from error. (2,*) two-way random: items are a sample from a
#       population of texts and item differences count as error (absolute agreement).
# (3,*) two-way mixed: these items are the whole universe, so item differences are
#       not error (consistency). (*,1) is one item, (*,k) is the mean over k items.
# icc3_k is Cronbach's alpha by construction and is kept as a self-check.
ICC_VARIANTS = ["icc1_1", "icc1_k", "icc2_1", "icc2_k", "icc3_1", "icc3_k"]
ICC_EMPTY = {**{v: np.nan for v in ICC_VARIANTS}, "n_participants": np.nan, "n_items": np.nan}

# Optimizers tried for every REML fit, best log-likelihood wins.
#
# MixedLM's default optimizer stops short of the optimum on these crossed designs.
# On the worst cell (OneStop / Fixed / WPM pooled) it reports |grad|=210 and lands
# 3 log-likelihood units below powell, which moves ICC(2,1) from 0.689 to 0.707 —
# so the spread across optimizers is under-convergence, not genuine uncertainty.
# Selecting on the likelihood makes the result deterministic and throws out
# blow-ups for free: cg converges to ICC(2,1)=0.333 there, 540 log-likelihood
# units adrift, and simply loses the comparison.
REML_METHODS = ("powell", "lbfgs", "bfgs")


def _best_reml_fit(model):
    """Fit with each optimizer and keep the one that actually maximises the likelihood.

    Convergence warnings are silenced here deliberately: they fire on nearly every
    fit, and the likelihood comparison is a stricter check than the warning is.
    """
    best = None
    for method in REML_METHODS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(reml=True, method=method)
        except Exception:
            continue
        if res is None or not np.isfinite(getattr(res, "llf", np.nan)):
            continue
        if best is None or res.llf > best.llf:
            best = res

    if best is None:
        raise RuntimeError(f"all REML optimizers failed ({', '.join(REML_METHODS)})")
    return best


def icc_variants_single(
    df: pd.DataFrame, value_col: str, item_col: str, participant_col: str = "participant_id"
) -> dict:
    """
    All six ICC variants for one participants x items matrix.

    Uses listwise deletion: the ANOVA needs a complete matrix, so participants
    missing any item are dropped and the surviving count is returned as
    n_participants. Every caller therefore gets the sample the numbers describe,
    rather than having to know the filtering happened.
    """
    if participant_col not in df.columns or value_col not in df.columns or item_col not in df.columns:
        return dict(ICC_EMPTY)

    try:
        pivot = df.pivot_table(
            index=participant_col, columns=item_col, values=value_col, aggfunc="first"
        )
    except Exception:
        return dict(ICC_EMPTY)

    pivot = pivot.dropna(axis=1, how="all").dropna(axis=0, how="any")
    n, k = pivot.shape
    if n < 2 or k < 2:
        return dict(ICC_EMPTY)

    x = pivot.values
    grand = x.mean()
    row_means = x.mean(axis=1, keepdims=True)
    col_means = x.mean(axis=0, keepdims=True)

    # Mean squares: rows (participants), columns (items), residual, and within-row
    msr = (k * ((row_means.ravel() - grand) ** 2).sum()) / (n - 1)
    msc = (n * ((col_means.ravel() - grand) ** 2).sum()) / (k - 1)
    mse = ((x - row_means - col_means + grand) ** 2).sum() / ((n - 1) * (k - 1))
    msw = ((x - row_means) ** 2).sum() / (n * (k - 1))

    if msr <= 0:
        return dict(ICC_EMPTY)

    def _safe(numer, denom):
        return float(numer / denom) if denom > 0 else np.nan

    return {
        "icc1_1": _safe(msr - msw, msr + (k - 1) * msw),
        "icc1_k": _safe(msr - msw, msr),
        "icc2_1": _safe(msr - mse, msr + (k - 1) * mse + k * (msc - mse) / n),
        "icc2_k": _safe(msr - mse, msr + (msc - mse) / n),
        "icc3_1": _safe(msr - mse, msr + (k - 1) * mse),
        "icc3_k": _safe(msr - mse, msr),
        "n_participants": int(n),
        "n_items": int(k),
    }


def icc_variants_all_readers(
    df: pd.DataFrame, value_col: str, item_col: str, participant_col: str = "participant_id"
) -> dict:
    """
    The same six ICCs, but using every reader instead of only the complete ones.

    The ANOVA in icc_variants_single needs a rectangular matrix, so it deletes any
    reader missing an item — for MECO that is ~87% of them. Here the variances are
    estimated from every observed cell instead:

      ICC(1,*) from the unbalanced one-way ANOVA, whose k0 correction handles
               readers having different numbers of items (closed form, no fit).
      ICC(2,*) and ICC(3,*) from REML variance components of
               score = reader + item + noise, which never forms the matrix at all.

    The (*,k) variants are reported at the full item count k, i.e. "reliability of
    a score averaged over all k items" — with unbalanced data k is a reporting
    choice, not something the data fixes, so it is stated rather than inferred.

    Returns the same keys as icc_variants_single, plus mean_items. Falls back to
    the complete-case values when the matrix is already complete (OneStop), where
    the two estimators coincide and the fit would be wasted work.
    """
    cols = [participant_col, item_col, value_col]
    if any(c not in df.columns for c in cols):
        return {**ICC_EMPTY, "mean_items": np.nan}

    d = df[cols].dropna()
    n = d[participant_col].nunique()
    k = d[item_col].nunique()
    if n < 2 or k < 2:
        return {**ICC_EMPTY, "mean_items": np.nan}

    # Already complete: the ANOVA is exact and much cheaper than a REML fit.
    if len(d) == n * k:
        return {**icc_variants_single(d, value_col, item_col, participant_col), "mean_items": float(k)}

    # --- ICC(1,*): unbalanced one-way ANOVA, closed form
    grp = d.groupby(participant_col)[value_col]
    n_i = grp.size().values
    m_i = grp.mean().values
    N = n_i.sum()
    grand = d[value_col].mean()
    msb = (n_i * (m_i - grand) ** 2).sum() / (n - 1)
    msw = ((d[value_col] - grp.transform("mean")) ** 2).sum() / (N - n)
    k0 = (N - (n_i ** 2).sum() / N) / (n - 1)

    icc1_1 = (msb - msw) / (msb + (k0 - 1) * msw) if (msb + (k0 - 1) * msw) > 0 else np.nan

    def _spearman_brown(icc_single, n_items):
        if np.isnan(icc_single) or icc_single <= -1 / (n_items - 1):
            return np.nan
        return float(n_items * icc_single / (1 + (n_items - 1) * icc_single))

    out = {
        "icc1_1": float(icc1_1) if not np.isnan(icc1_1) else np.nan,
        "icc1_k": _spearman_brown(icc1_1, k),
        "icc2_1": np.nan, "icc2_k": np.nan, "icc3_1": np.nan, "icc3_k": np.nan,
        "n_participants": int(n),
        "n_items": int(k),
        "mean_items": float(N / n),
    }

    import_error = None
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:  # statsmodels missing
        import_error, smf = exc, None

    if smf is not None:
        fit_df = d.rename(columns={participant_col: "pid", item_col: "item", value_col: "y"})
        fit_df["item"] = fit_df["item"].astype(str)
        fit_df["_all"] = 1

        # --- ICC(2,*): items RANDOM, crossed with readers (absolute agreement)
        try:
            model = smf.mixedlm(
                "y ~ 1", fit_df, groups=fit_df["_all"],
                vc_formula={"pid": "0 + C(pid)", "item": "0 + C(item)"},
            )
            res = _best_reml_fit(model)

            # vcomp is ordered by component NAME (alphabetical), not by the order
            # the vc_formula dict was written: reading it positionally silently
            # swaps the reader and item variances and yields a plausible-looking
            # wrong ICC.
            vc = dict(zip(list(model.exog_vc.names), res.vcomp))
            var_reader, var_item, var_resid = vc["pid"], vc["item"], float(res.scale)

            if var_reader > 0 and (var_reader + var_item + var_resid) > 0:
                out["icc2_1"] = float(var_reader / (var_reader + var_item + var_resid))
                out["icc2_k"] = float(var_reader / (var_reader + (var_item + var_resid) / k))
        except Exception as exc:
            logger.warning(f"All-readers ICC(2) fit failed ({exc})")

        # --- ICC(3,*): items FIXED, readers random — its own fit, deliberately.
        # Dropping var_item from the crossed fit above is a different model: random
        # items are shrunk toward the grand mean, which moves var_reader and
        # var_resid. The gap measures ~1e-6 to 1e-4 here, invisible at three
        # decimals, but this fit is also ~15x cheaper, so there is nothing to trade.
        try:
            res3 = _best_reml_fit(smf.mixedlm("y ~ C(item)", fit_df, groups=fit_df["pid"]))
            var_reader3, var_resid3 = float(res3.cov_re.iloc[0, 0]), float(res3.scale)

            if var_reader3 > 0 and (var_reader3 + var_resid3) > 0:
                out["icc3_1"] = float(var_reader3 / (var_reader3 + var_resid3))
                out["icc3_k"] = float(var_reader3 / (var_reader3 + var_resid3 / k))
        except Exception as exc:
            logger.warning(f"All-readers ICC(3) fit failed ({exc})")
    else:
        logger.warning(f"All-readers ICC(2)/ICC(3) unavailable ({import_error}); ICC(1) still reported")

    return out


def icc_variants_pooled(
    df: pd.DataFrame, value_col: str, item_col: str, group_col: str = "content_group",
    participant_col: str = "participant_id",
) -> dict:
    """
    One ICC estimated over all content groups at once, instead of averaging per-group ICCs.

    OneStop's content groups read disjoint passage sets (6 x 54, no overlap), so the
    readers x passages matrix across groups is block diagonal and an ANOVA cannot be
    pooled — which is why icc_variants_by_group averages six separate estimates. That
    average is a summary rather than an estimator: the mean of six ratios is not the
    ratio for the combined readers, it weights a 21-reader group like a 28-reader one,
    and it has no standard error.

    A mixed model does pool. The content group enters as a fixed effect so that
    set-level differences between groups land there rather than in the reader
    variance, matching what the per-group ANOVAs implicitly remove.

    The (*,k) variants use the passages one reader sees (their group's k), not the
    324 distinct passages in the study.
    """
    cols = [participant_col, item_col, group_col, value_col]
    if any(c not in df.columns for c in cols):
        return dict(ICC_EMPTY)

    d = df[cols].dropna()
    n = d[participant_col].nunique()
    if n < 2 or d[item_col].nunique() < 2:
        return dict(ICC_EMPTY)

    # k is per reader: each reader only ever sees their own group's passages.
    k = int(round(d.groupby(participant_col)[item_col].nunique().mean()))
    if k < 2:
        return dict(ICC_EMPTY)

    out = dict(ICC_EMPTY)
    out["n_participants"] = int(n)
    out["n_items"] = k

    # --- ICC(1,*): one-way over readers, on group-centred scores so that between-group
    # differences are not counted as reader variance (the per-group ANOVAs drop them).
    centred = d.assign(_y=d[value_col] - d.groupby(group_col)[value_col].transform("mean"))
    grp = centred.groupby(participant_col)["_y"]
    n_i, m_i = grp.size().values, grp.mean().values
    N, grand = n_i.sum(), centred["_y"].mean()
    msb = (n_i * (m_i - grand) ** 2).sum() / (n - 1)
    msw = ((centred["_y"] - grp.transform("mean")) ** 2).sum() / (N - n)
    k0 = (N - (n_i ** 2).sum() / N) / (n - 1)
    if (msb + (k0 - 1) * msw) > 0:
        icc1_1 = (msb - msw) / (msb + (k0 - 1) * msw)
        out["icc1_1"] = float(icc1_1)
        if icc1_1 > -1 / (k - 1):
            out["icc1_k"] = float(k * icc1_1 / (1 + (k - 1) * icc1_1))

    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        logger.warning(f"Pooled ICC(2)/ICC(3) unavailable ({exc}); ICC(1) still reported")
        return out

    fit_df = d.rename(
        columns={participant_col: "pid", item_col: "item", group_col: "grp", value_col: "y"}
    )
    fit_df["item"] = fit_df["item"].astype(str)
    fit_df["_all"] = 1

    # --- ICC(2,*): passages random and crossed with readers, group fixed
    try:
        model = smf.mixedlm(
            "y ~ C(grp)", fit_df, groups=fit_df["_all"],
            vc_formula={"pid": "0 + C(pid)", "item": "0 + C(item)"},
        )
        res = _best_reml_fit(model)
        vc = dict(zip(list(model.exog_vc.names), res.vcomp))   # ordered by name, not insertion
        var_reader, var_item, var_resid = vc["pid"], vc["item"], float(res.scale)
        if var_reader > 0 and (var_reader + var_item + var_resid) > 0:
            out["icc2_1"] = float(var_reader / (var_reader + var_item + var_resid))
            out["icc2_k"] = float(var_reader / (var_reader + (var_item + var_resid) / k))
    except Exception as exc:
        logger.warning(f"Pooled ICC(2) fit failed ({exc})")

    # --- ICC(3,*): passages fixed (which absorbs the group, passages being nested in it)
    try:
        res3 = _best_reml_fit(smf.mixedlm("y ~ C(item)", fit_df, groups=fit_df["pid"]))
        var_reader3, var_resid3 = float(res3.cov_re.iloc[0, 0]), float(res3.scale)
        if var_reader3 > 0 and (var_reader3 + var_resid3) > 0:
            out["icc3_1"] = float(var_reader3 / (var_reader3 + var_resid3))
            out["icc3_k"] = float(var_reader3 / (var_reader3 + var_resid3 / k))
    except Exception as exc:
        logger.warning(f"Pooled ICC(3) fit failed ({exc})")

    return out


def icc_variants_by_group(
    df: pd.DataFrame, value_col: str, item_col: str, group_col: str = "content_group",
    estimator=icc_variants_single,
) -> tuple:
    """
    ICC variants per group, and their average across groups (as
    cronbachs_alpha_by_group does): (per_group, averaged).

    n_participants sums across groups because the groups partition the readers;
    n_items is the per-group item count, which is what each ICC was computed on.
    `estimator` selects complete-case ANOVA or the all-readers version.
    """
    per_group = {
        str(g): estimator(gdf, value_col, item_col)
        for g, gdf in df.groupby(group_col)
    }

    avg = {}
    for variant in ICC_VARIANTS:
        vals = [r[variant] for r in per_group.values() if not np.isnan(r[variant])]
        avg[variant] = float(np.mean(vals)) if vals else np.nan

    n_vals = [r["n_participants"] for r in per_group.values() if not np.isnan(r["n_participants"])]
    k_vals = [r["n_items"] for r in per_group.values() if not np.isnan(r["n_items"])]
    avg["n_participants"] = int(np.sum(n_vals)) if n_vals else np.nan
    avg["n_items"] = int(np.mean(k_vals)) if k_vals else np.nan

    mi_vals = [r["mean_items"] for r in per_group.values()
               if "mean_items" in r and not np.isnan(r["mean_items"])]
    if mi_vals:
        avg["mean_items"] = float(np.mean(mi_vals))

    return per_group, avg


def icc_for(df: pd.DataFrame, value_col: str, item_col: str, group_col: str = None) -> dict:
    """
    ICC columns for one cell, in two families.

    The bare names are the complete-case ANOVA (readers missing any item dropped);
    the `_all` names keep every reader via icc_variants_all_readers. Both are
    emitted because for MECO they answer different questions — the first describes
    the 13% with a full text set, the second the whole cohort — and their agreement
    is itself the evidence that the complete-case sample is not distorting things.

    Averaged over groups when group_col is present (OneStop), else pooled.
    """
    empty = {
        **{v: np.nan for v in ICC_VARIANTS},
        **{f"{v}_all": np.nan for v in ICC_VARIANTS},
        **{f"{v}_pooled": np.nan for v in ICC_VARIANTS},
        "icc_n_participants": np.nan, "icc_n_items": np.nan,
        "icc_n_participants_all": np.nan, "icc_mean_items_all": np.nan,
        "icc_n_participants_pooled": np.nan, "icc_n_items_pooled": np.nan,
        "icc_per_group": None,
    }
    if item_col is None:
        return empty

    grouped = bool(group_col) and group_col in df.columns

    if grouped:
        per_group, avg = icc_variants_by_group(df, value_col, item_col, group_col)
        _, avg_all = icc_variants_by_group(
            df, value_col, item_col, group_col, estimator=icc_variants_all_readers
        )
        pooled = icc_variants_pooled(df, value_col, item_col, group_col)
        per_group_str = str(per_group) if per_group else None
    else:
        avg = icc_variants_single(df, value_col, item_col)
        avg_all = icc_variants_all_readers(df, value_col, item_col)
        pooled = dict(ICC_EMPTY)   # nothing to pool: MECO has no content groups
        per_group_str = None

    return {
        **{v: avg[v] for v in ICC_VARIANTS},
        **{f"{v}_all": avg_all[v] for v in ICC_VARIANTS},
        **{f"{v}_pooled": pooled[v] for v in ICC_VARIANTS},
        "icc_n_participants": avg["n_participants"],
        "icc_n_items": avg["n_items"],
        "icc_n_participants_all": avg_all["n_participants"],
        "icc_mean_items_all": avg_all.get("mean_items", np.nan),
        "icc_n_participants_pooled": pooled["n_participants"],
        "icc_n_items_pooled": pooled["n_items"],
        "icc_per_group": per_group_str,
    }


def participant_score_sd(df: pd.DataFrame, value_col: str, participant_col: str = "participant_id") -> float:
    """
    SD across participants of their mean per-item score.

    This is the observed-score scale the SEM is reported in: each participant's
    score is the mean over the items they read, matching how the per-item scores
    are aggregated downstream.
    """
    if participant_col not in df.columns or value_col not in df.columns:
        return np.nan

    means = df.groupby(participant_col)[value_col].mean().dropna()
    if len(means) < 2:
        return np.nan

    return float(means.std(ddof=1))


def sem_from_reliability(sd: float, reliability: float) -> float:
    """
    Standard error of measurement: SEM = SD × sqrt(1 - reliability).

    Expresses the reliability coefficient on the score scale, i.e. the expected
    SD of a participant's observed score around their true score.
    """
    if sd is None or reliability is None:
        return np.nan
    if np.isnan(sd) or np.isnan(reliability) or reliability > 1:
        return np.nan

    # Reliability can come out slightly negative on noisy cells; clamp so the sqrt is defined.
    return float(sd * np.sqrt(max(0.0, 1.0 - reliability)))


def sem_by_group(df: pd.DataFrame, value_col: str, per_group_reliability: dict, group_col: str = "content_group") -> tuple:
    """SEM per group, from that group's reliability (as the *_by_group
    functions return it) and score SD, and their average: (per_group, average)."""
    per_group = {}
    for group_val, reliability in per_group_reliability.items():
        group_df = df[df[group_col].astype(str) == str(group_val)]
        per_group[str(group_val)] = sem_from_reliability(
            participant_score_sd(group_df, value_col), reliability
        )

    valid = [v for v in per_group.values() if not np.isnan(v)]
    avg = np.mean(valid) if valid else np.nan

    return per_group, avg


def sem_for(df: pd.DataFrame, value_col: str, reliability: float, per_group_reliability: dict = None, group_col: str = None) -> tuple:
    """SEM for one cell, as (sem, per-group string or None): averaged over groups
    when per-group reliabilities are given, else from the pooled SD."""
    if per_group_reliability and group_col and group_col in df.columns:
        per_group, avg = sem_by_group(df, value_col, per_group_reliability, group_col)
        return avg, (str(per_group) if per_group else None)

    return sem_from_reliability(participant_score_sd(df, value_col), reliability), None


def cronbachs_alpha_by_group(df: pd.DataFrame, value_col: str, item_col: str, group_col: str = "content_group") -> tuple:
    """Cronbach's alpha per group (OneStop's content groups) and the mean of the
    non-NaN ones: (per_group, average). One alpha overall without the group
    column."""
    if group_col not in df.columns:
        return {}, cronbachs_alpha_single(df, value_col, item_col)

    per_group = {}
    for group_val in df[group_col].unique():
        group_df = df[df[group_col] == group_val]
        alpha = cronbachs_alpha_single(group_df, value_col, item_col)
        per_group[str(group_val)] = alpha

    # Average of non-NaN alphas
    valid_alphas = [a for a in per_group.values() if not np.isnan(a)]
    avg = np.mean(valid_alphas) if valid_alphas else np.nan

    return per_group, avg


def split_half_reliability_by_group(df: pd.DataFrame, value_col: str, item_col: str, group_col: str = "content_group", n_iterations: int = 100) -> tuple:
    """Split-half reliability per group and the mean of the non-NaN ones:
    (per_group, average). One split-half overall without the group column."""
    if group_col not in df.columns:
        return {}, split_half_reliability_items(df, value_col=value_col, item_col=item_col, n_iterations=n_iterations)

    per_group = {}
    for group_val in df[group_col].unique():
        group_df = df[df[group_col] == group_val]
        split_half = split_half_reliability_items(group_df, value_col=value_col, item_col=item_col, n_iterations=n_iterations)
        per_group[str(group_val)] = split_half

    # Average of non-NaN split-halves
    valid_values = [v for v in per_group.values() if not np.isnan(v)]
    avg = np.mean(valid_values) if valid_values else np.nan

    return per_group, avg


def calculate_eyescore_reliability_items(calc_split_half=True, preview: str = DEFAULT_PREVIEW) -> pd.DataFrame:
    """One reliability row per EyeScore per-item result found (dataset, pool,
    level, feature set)."""
    results = []

    for dataset in ["OneStop", "Meco"]:
        ds_preview = onestop_preview(dataset, preview)

        # OneStop has separate article and paragraph levels; Meco only has paragraph
        p_agg_levels = ["article", "paragraph"] if dataset == "OneStop" else ["paragraph"]

        # Only "seen" and "unseen" pools exist in the data (no "all" pool)
        for pool in ["seen", "unseen"]:
            for p_agg_level in p_agg_levels:
                for feature_set in SMALL_FEATURE_SETS:
                    # For Meco, don't filter by p_agg_level since the path structure is different
                    level_param = None if dataset == "Meco" else p_agg_level
                    result = load_eyescore_raw_data_with_pool(dataset, ds_preview, pool, feature_set, level_param)
                    if result is None:
                        continue

                    df, _ = result

                    # Add content_group from metadata for per-group calculations
                    df = add_content_group_from_metadata(df, dataset)

                    # For all targets combined (they share the same eye_score)
                    if "eye_score" in df.columns and len(df) > 1:
                        # Item column: paragraph first, since paragraph-level
                        # data also carries article_id (not the reverse).
                        item_col = None
                        for col in [Fields.UNIQUE_PARAGRAPH_ID, Fields.ARTICLE_ID]:
                            if col in df.columns:
                                item_col = col
                                break

                        # Calculate both reliability metrics
                        split_half = None
                        split_half_per_group = None
                        split_half_groups = {}
                        if calc_split_half:
                            if dataset == "OneStop" and item_col:
                                # Calculate per content group and average
                                split_half_groups, split_half = split_half_reliability_by_group(df, "eye_score", item_col, Fields.CONTENT_GROUP)
                                split_half_per_group = str(split_half_groups) if split_half_groups else None
                            elif item_col:
                                split_half = split_half_reliability_items(df, value_col="eye_score", item_col=item_col)

                        cronbach_avg = np.nan
                        cronbach_per_group = None
                        cronbach_groups = {}
                        # Frame the alpha was computed on; the SEM must use the same participants.
                        alpha_df = df

                        # Compute Cronbach's alpha if we have item IDs
                        if item_col:
                            if dataset == "OneStop":
                                # Calculate per content group and average
                                cronbach_groups, cronbach_avg = cronbachs_alpha_by_group(df, "eye_score", item_col, Fields.CONTENT_GROUP)
                                cronbach_per_group = str(cronbach_groups) if cronbach_groups else None
                            else:
                                # For Meco, only participants who read every item
                                participant_items = df.groupby("participant_id")[item_col].nunique()
                                n_items = df[item_col].nunique()
                                complete_participants = participant_items[participant_items == n_items].index
                                df_complete = df[df["participant_id"].isin(complete_participants)]
                                alpha_df = df_complete

                                if len(df_complete) > 1:
                                    cronbach_avg = cronbachs_alpha_single(df_complete, "eye_score", item_col)
                                else:
                                    cronbach_avg = np.nan
                                cronbach_per_group = None

                        # Standard error of measurement: the reliabilities above on the score scale
                        sem_cronbach, sem_cronbach_per_group = sem_for(
                            alpha_df, "eye_score", cronbach_avg, cronbach_groups, Fields.CONTENT_GROUP
                        )
                        sem_split_half, sem_split_half_per_group = sem_for(
                            df, "eye_score", split_half, split_half_groups, Fields.CONTENT_GROUP
                        )

                        # ICC variants, on the same grouping as the alpha above
                        icc_cols = icc_for(
                            df, "eye_score", item_col,
                            Fields.CONTENT_GROUP if dataset == "OneStop" else None,
                        )

                        results.append({
                            "dataset": dataset,
                            "p_agg_level": p_agg_level,
                            "feature_set": feature_set,
                            "pool": pool,
                            "split_half": split_half,
                            "split_half_per_group": split_half_per_group,
                            "cronbachs_alpha": cronbach_avg,
                            "cronbachs_alpha_per_group": cronbach_per_group,
                            **icc_cols,
                            "sem_cronbach": sem_cronbach,
                            "sem_cronbach_per_group": sem_cronbach_per_group,
                            "sem_split_half": sem_split_half,
                            "sem_split_half_per_group": sem_split_half_per_group,
                            "n_observations": len(df),
                        })

    return pd.DataFrame(results)


def calculate_predictions_reliability_items(model: str = "Ridge_Classifier", calc_split_half=True, preview: str = DEFAULT_PREVIEW) -> pd.DataFrame:
    """One reliability row per per-item prediction result found (dataset,
    pool, level, target, feature set)."""
    results = []

    for dataset in ["OneStop", "Meco"]:
        ds_preview = onestop_preview(dataset, preview)
        targets = DATASET_TARGET_COLS[dataset]

        # OneStop has separate article and paragraph levels; Meco only has paragraph
        p_agg_levels = ["article", "paragraph"] if dataset == "OneStop" else ["paragraph"]

        # Only "seen" and "unseen" pools exist in the data (no "all" pool)
        for pool in ["seen", "unseen"]:
            for p_agg_level in p_agg_levels:
                for target in targets:
                    for feature_set in SMALL_FEATURE_SETS:
                        result = load_predictions_raw_data(dataset, ds_preview, target, feature_set, model, p_agg_level, pool)
                        if result is None:
                            continue

                        df, _ = result

                        # Add content_group from metadata for per-group calculations
                        df = add_content_group_from_metadata(df, dataset)

                        # Calculate reliability for this target's predictions
                        if "pred" in df.columns and len(df) > 1:
                            # Determine item column (check for all possible item ID columns)
                            item_col = None
                            for col in [Fields.UNIQUE_PARAGRAPH_ID, Fields.ARTICLE_ID]:
                                if col in df.columns:
                                    item_col = col
                                    break

                            if item_col is None:
                                continue

                            # Calculate both reliability metrics
                            split_half = None
                            split_half_per_group = None
                            split_half_groups = {}
                            if calc_split_half:
                                if dataset == "OneStop":
                                    # Calculate per content group and average
                                    split_half_groups, split_half = split_half_reliability_by_group(df, "pred", item_col, "content_group")
                                    split_half_per_group = str(split_half_groups) if split_half_groups else None
                                else:
                                    split_half = split_half_reliability_items(df, value_col="pred", item_col=item_col)

                            cronbach_groups = {}
                            # Frame the alpha was computed on; the SEM must use the same participants.
                            alpha_df = df

                            if dataset == "OneStop":
                                # Calculate per content group and average
                                cronbach_groups, cronbach_avg = cronbachs_alpha_by_group(df, "pred", item_col, "content_group")
                                cronbach_per_group = str(cronbach_groups) if cronbach_groups else None
                            else:
                                # For Meco, only participants who read every item
                                participant_items = df.groupby("participant_id")[item_col].nunique()
                                n_items = df[item_col].nunique()
                                complete_participants = participant_items[participant_items == n_items].index
                                df_complete = df[df["participant_id"].isin(complete_participants)]
                                alpha_df = df_complete

                                if len(df_complete) > 1:
                                    cronbach_avg = cronbachs_alpha_single(df_complete, "pred", item_col)
                                else:
                                    cronbach_avg = np.nan
                                cronbach_per_group = None

                            # Standard error of measurement: the reliabilities above on the score scale
                            sem_cronbach, sem_cronbach_per_group = sem_for(
                                alpha_df, "pred", cronbach_avg, cronbach_groups, "content_group"
                            )
                            sem_split_half, sem_split_half_per_group = sem_for(
                                df, "pred", split_half, split_half_groups, "content_group"
                            )

                            # ICC variants, on the same grouping as the alpha above
                            icc_cols = icc_for(
                                df, "pred", item_col,
                                "content_group" if dataset == "OneStop" else None,
                            )

                            results.append({
                                "dataset": dataset,
                                "p_agg_level": p_agg_level,
                                "target": target,
                                "feature_set": feature_set,
                                "pool": pool,
                                "split_half": split_half,
                                "split_half_per_group": split_half_per_group,
                                "cronbachs_alpha": cronbach_avg,
                                "cronbachs_alpha_per_group": cronbach_per_group,
                                **icc_cols,
                                "sem_cronbach": sem_cronbach,
                                "sem_cronbach_per_group": sem_cronbach_per_group,
                                "sem_split_half": sem_split_half,
                                "sem_split_half_per_group": sem_split_half_per_group,
                                "n_observations": len(df),
                            })

    return pd.DataFrame(results)


def main(run_per_item_split_half: bool = True, preview: str = DEFAULT_PREVIEW):
    """Per-item reliability for one OneStop preview; `run_per_item_split_half=False`
    skips the (slow) per-item split-half."""
    logger.info("=" * 60)
    logger.info("Starting per-item reliability analysis...")
    logger.info(f"  Per-item split-half: {'ENABLED' if run_per_item_split_half else 'DISABLED'}")
    logger.info("=" * 60)

    out_dir = per_item_reliability_dir(preview)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Calculate EyeScore reliability
    logger.info("Processing EyeScore data...")
    eyescore_df = calculate_eyescore_reliability_items(calc_split_half=run_per_item_split_half, preview=preview)
    logger.info("  ✓ Cronbach's alpha and split-half (per-item) calculated")

    if not eyescore_df.empty:
        eyescore_df["source"] = "eyescore"
        all_results.append(eyescore_df)
    else:
        logger.warning("  No EyeScore data found")

    # Calculate Predictions reliability
    logger.info("Processing Predictions data...")
    predictions_df = calculate_predictions_reliability_items(calc_split_half=run_per_item_split_half, preview=preview)
    logger.info("  ✓ Cronbach's alpha and split-half (per-item) calculated")

    if not predictions_df.empty:
        predictions_df["source"] = "predictions"
        all_results.append(predictions_df)
    else:
        logger.warning("  No Predictions data found")

    # Combine and save
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        results_path = out_dir / "per_item_reliability.csv"
        combined_df.to_csv(results_path, index=False)
        logger.info(f"✓ Saved {len(combined_df)} rows to {results_path}")
    else:
        logger.warning("No results to save")

    logger.info("=" * 60)
    logger.info("✓ Per-item reliability analysis complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Calculate per-item reliability metrics (Cronbach's alpha and optional split-half)"
    )
    parser.add_argument(
        "--skip-per-item-split-half",
        action="store_true",
        help="Skip per-item split-half calculations (faster, only computes Cronbach's alpha)",
    )
    parser.add_argument(
        "--preview",
        default=DEFAULT_PREVIEW,
        help="OneStop preview to read (Gathering|Hunting). Gathering writes the "
        "top-level paths; any other preview writes to a parallel subtree, e.g. "
        "src/reliability/results/per_item_reliability/hunting/.",
    )

    args = parser.parse_args()
    run_split_half = not args.skip_per_item_split_half

    main(run_per_item_split_half=run_split_half, preview=args.preview)
