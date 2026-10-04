"""Shared statistical machinery for the evaluation step.

Two goals, used by both the EyeScore and the predictions evaluators:

1. Bootstrap confidence intervals — the reported correlation is the *mean* of
   the bootstrap resample correlations, with the resample *std* giving the
   error bar and a 2.5/97.5 percentile 95% CI alongside it.
   See `bootstrap_correlation_ci`.

2. Pairwise significance testing between two models that share the same target
   (two dependent, overlapping correlations). `bootstrap_difference_test` is
   the test behind every reported result: a paired bootstrap over the shared
   participants with a null-centred p.

`steiger_dependent_overlap` (Steiger 1980, the test cocor exposes as
``steiger1980``) and `meng_dependent_overlap` (Meng-Rosenthal-Rubin 1992) are
retained and still emitted by `pairwise_dependent_tests`, but **nothing
downstream reads them** — no table, plot or caption is derived from their
columns. They are kept as an independent cross-check on the bootstrap: both
are closed-form and essentially free to compute, and Steiger in particular is
the better-calibrated test (simulated at a nominal 5% it rejects ~5.0% under a
true null, against ~5.6-5.8% for the bootstrap at n=50-100), so a large
disagreement between the two is worth investigating.

Terminology for the dependent-correlation tests: ``j`` is the common variable
(the proficiency target Y), and ``k`` / ``h`` are the two models being
compared. So we test corr(model_k, Y) vs corr(model_h, Y), where the two
models are themselves correlated (r_kh).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm, rankdata, pearsonr, spearmanr

# Number of bootstrap resamples. Sized by the *p-value* tail, not the point
# estimate: means and stds converge long before 10k, but a two-sided p near
# 0.05 carries a Monte Carlo SE of 2*sqrt(0.025*0.975/B), which at B=10k is
# ~0.003 — enough that a star could flip with the seed. Measured on the real
# Ridge comparisons, 8 of 92 changed star count across five seeds at 10k, three
# of them crossing 0.05 outright. At 100k the SE drops to ~0.001 and those
# settle. Costs ~1.7s per pairwise comparison; override per-call if needed.
N_BOOTSTRAP = 100000

# Fixed seed so the seeded resample is reproducible across runs.
BOOTSTRAP_SEED = 42

# Two-sided 95% percentile interval. Percentile rather than the normal
# approximation (mean +- 1.96*std) so every interval in this module is built
# the same way: the difference test was already percentile-based, and the
# bootstrap distribution of a correlation is bounded and skewed, which the
# normal form ignores. On this project's data the two agree to within 0.006 at
# the participant-level sample sizes, so the switch is about consistency and
# principle rather than moving any reported number.
CI_PERCENTILES_95 = (2.5, 97.5)


# Peak memory target for one block of resamples. The resample index matrix and
# the arrays gathered from it are (block, n) each, so materialising all
# `n_bootstrap` rows at once costs ~2.4 MB per observation and scales with n:
# at n=16k that is tens of GB, which is what OOM-killed a full evaluation pass.
# Drawing the same resamples in blocks and reducing each block as it is built
# holds peak memory at this constant instead, for identical results (the
# Generator fills row-major, so blocked draws consume the stream exactly as one
# big draw does — asserted in src/tests/test_stats_utils_chunking.py).
BOOTSTRAP_BLOCK_BYTES = 64 << 20  # 64 MB


def _block_rows(n_obs: int, n_arrays: int) -> int:
    """How many resamples to build at once, given `n_arrays` (block, n) arrays."""
    per_row = max(1, n_obs * 8 * n_arrays)
    return max(1, min(N_BOOTSTRAP, BOOTSTRAP_BLOCK_BYTES // per_row))


def _blocks(n_bootstrap: int, block_rows: int):
    """Yield (start, stop) row ranges covering `n_bootstrap` resamples."""
    for start in range(0, n_bootstrap, block_rows):
        yield start, min(start + block_rows, n_bootstrap)


def _rowwise_pearson(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r for each row of X against the matching row of Y.

    X, Y are (n_boot, n) arrays (one resample per row). Returns (n_boot,)
    correlations; rows with zero variance in either series yield NaN.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    num = (Xc * Yc).sum(axis=1)
    den = np.sqrt((Xc ** 2).sum(axis=1) * (Yc ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r


def _summ(name: str, boot_r: np.ndarray, ci_pct: Tuple[float, float]) -> Dict[str, float]:
    """Reduce a vector of bootstrap correlations to point/std/CI columns."""
    boot_r = boot_r[np.isfinite(boot_r)]
    if boot_r.size < 2:
        return {
            f"{name}_r_boot": np.nan,
            f"{name}_r_boot_std": np.nan,
            f"{name}_ci_low_boot": np.nan,
            f"{name}_ci_high_boot": np.nan,
        }
    lo, hi = np.percentile(boot_r, ci_pct)
    return {
        f"{name}_r_boot": float(np.mean(boot_r)),
        f"{name}_r_boot_std": float(np.std(boot_r, ddof=1)),
        f"{name}_ci_low_boot": float(lo),
        f"{name}_ci_high_boot": float(hi),
    }


def _summ_mae(boot_mae: np.ndarray, ci_pct: Tuple[float, float]) -> Dict[str, float]:
    """Reduce a vector of bootstrap MAEs to point/std/CI columns."""
    boot_mae = boot_mae[np.isfinite(boot_mae)]
    if boot_mae.size < 2:
        return {
            "mae_boot": np.nan,
            "mae_boot_std": np.nan,
            "mae_ci_low_boot": np.nan,
            "mae_ci_high_boot": np.nan,
        }
    lo, hi = np.percentile(boot_mae, ci_pct)
    return {
        "mae_boot": float(np.mean(boot_mae)),
        "mae_boot_std": float(np.std(boot_mae, ddof=1)),
        "mae_ci_low_boot": float(lo),
        "mae_ci_high_boot": float(hi),
    }


def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    ci_pct: tuple = CI_PERCENTILES_95,
    include_mae: bool = False,
) -> Dict[str, float]:
    """Bootstrap the Pearson and Spearman correlation between x and y.

    Draws `n_bootstrap` full-size resamples with replacement (paired: the same
    row indices are used for x and y), recomputes the correlation on each, and
    reports the mean as the point estimate and the std as the error bar.

    Spearman is bootstrapped by ranking x and y once (over the full sample) and
    then resampling those ranks — a fast, standard approximation to resampling
    the Spearman statistic directly. Pearson is exact.

    Returns a dict with, for each of ``pearson`` / ``spearman``:
        <m>_r_boot        mean of the resample correlations (the reported value)
        <m>_r_boot_std    std of the resample correlations
        <m>_ci_low_boot   2.5th percentile of the resample correlations
        <m>_ci_high_boot  97.5th percentile (a percentile 95% CI by default)
    plus ``n_boot`` (number of usable resamples) and ``n_boot_obs`` (sample size
    that was resampled). All-NaN if fewer than 3 finite paired observations.

    With `include_mae` (for prediction evaluations, where x is the true target
    and y the prediction), the mean absolute error is bootstrapped on the same
    resamples: ``mae_boot`` / ``mae_boot_std`` / ``mae_ci_low_boot`` /
    ``mae_ci_high_boot``. Off by default — MAE is meaningless when x and y are
    on different scales (e.g. EyeScore vs. a proficiency test).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size

    nan_out = {
        **_summ("pearson", np.array([]), ci_pct),
        **_summ("spearman", np.array([]), ci_pct),
        "n_boot": 0,
        "n_boot_obs": n,
    }
    if include_mae:
        nan_out.update(_summ_mae(np.array([]), ci_pct))
    if n < 3:
        return nan_out

    rng = np.random.default_rng(seed)

    # Spearman (approx) on resampled ranks of the full sample — ranked once,
    # outside the resample loop.
    rx = rankdata(x)
    ry = rankdata(y)

    boot_pearson = np.empty(n_bootstrap, dtype=float)
    boot_spearman = np.empty(n_bootstrap, dtype=float)
    boot_mae = np.empty(n_bootstrap, dtype=float) if include_mae else None

    # Resamples are built and reduced a block at a time: px/py/rx[idx]/ry[idx]
    # plus idx is five (block, n) arrays live at once.
    for start, stop in _blocks(n_bootstrap, _block_rows(n, 5)):
        idx = rng.integers(0, n, size=(stop - start, n))
        # Pearson (exact) on the resampled raw values.
        px = x[idx]
        py = y[idx]
        boot_pearson[start:stop] = _rowwise_pearson(px, py)
        boot_spearman[start:stop] = _rowwise_pearson(rx[idx], ry[idx])
        if include_mae:
            boot_mae[start:stop] = np.abs(px - py).mean(axis=1)

    out = {
        **_summ("pearson", boot_pearson, ci_pct),
        **_summ("spearman", boot_spearman, ci_pct),
        "n_boot": int(np.isfinite(boot_pearson).sum()),
        "n_boot_obs": n,
    }
    if include_mae:
        out.update(_summ_mae(boot_mae, ci_pct))
    return out


def steiger_dependent_overlap(
    r_jk: float, r_jh: float, r_kh: float, n: int
) -> Tuple[float, float]:
    """Steiger's (1980) Z-test for two dependent, overlapping correlations.

    Compares corr(model_k, Y)=r_jk against corr(model_h, Y)=r_jh, where the two
    models share the common variable Y (index j) and are themselves correlated
    r_kh. This is the pure-Python equivalent of R cocor's ``steiger1980`` test.

    Returns (z_statistic, two_sided_p_value), or (nan, nan) for degenerate
    input (n < 4, non-finite r, |r| >= 1, or a non-positive variance term).
    """
    if n < 4 or not all(np.isfinite([r_jk, r_jh, r_kh])):
        return (np.nan, np.nan)
    if abs(r_jk) >= 1 or abs(r_jh) >= 1 or abs(r_kh) >= 1:
        return (np.nan, np.nan)

    z_jk = np.arctanh(r_jk)
    z_jh = np.arctanh(r_jh)

    # Asymptotic covariance term (correlation between the two sample
    # correlations), Steiger (1980).
    psi = (
        r_kh * (1.0 - r_jk ** 2 - r_jh ** 2)
        - 0.5 * r_jk * r_jh * (1.0 - r_jk ** 2 - r_jh ** 2 - r_kh ** 2)
    )
    denom_c = (1.0 - r_jk ** 2) * (1.0 - r_jh ** 2)
    if denom_c <= 0:
        return (np.nan, np.nan)
    c = psi / denom_c

    var_term = 2.0 - 2.0 * c
    if var_term <= 0:
        return (np.nan, np.nan)

    z_stat = (z_jk - z_jh) * np.sqrt(n - 3) / np.sqrt(var_term)
    p = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    return float(z_stat), float(p)


def meng_dependent_overlap(
    r_jk: float, r_jh: float, r_kh: float, n: int
) -> Tuple[float, float]:
    """Meng-Rosenthal-Rubin (1992) Z-test for the same hypothesis as
    `steiger_dependent_overlap` — a widely used sibling test, emitted alongside
    Steiger for cross-checking. Same argument convention (j = common target).

    Returns (z, two-sided p), or (nan, nan) for degenerate input.
    """
    if n < 4 or not all(np.isfinite([r_jk, r_jh, r_kh])):
        return (np.nan, np.nan)
    if abs(r_jk) >= 1 or abs(r_jh) >= 1 or abs(r_kh) >= 1:
        return (np.nan, np.nan)
    z1 = np.arctanh(r_jk)
    z2 = np.arctanh(r_jh)
    rm_sq = (r_jk ** 2 + r_jh ** 2) / 2.0
    denom = 1.0 - rm_sq
    if denom <= 0 or (1.0 - r_kh) <= 0:
        return (np.nan, np.nan)
    f = min((1.0 - r_kh) / (2.0 * denom), 1.0)
    h = (1.0 - f * rm_sq) / denom
    if h <= 0:
        return (np.nan, np.nan)
    z_stat = (z1 - z2) * np.sqrt((n - 3) / (2.0 * (1.0 - r_kh) * h))
    p = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    return float(z_stat), float(p)


def bootstrap_difference_test(
    y: np.ndarray, x_a: np.ndarray, x_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    include_mae: bool = False,
) -> Dict[str, float]:
    """Paired-bootstrap test of whether model_a correlates with y more strongly
    than model_b (the bootstrap analogue of the Steiger/Meng test).

    For each of `n_bootstrap` resamples, the *same* row indices are used for
    both models (they are paired, since they share the participant set), and we
    record d_i = corr(x_a, y)_i - corr(x_b, y)_i. This is the same idea as the
    Readability repo's permutation test over the bootstrap correlation
    distributions, computed directly on the paired differences:

        boot_diff_<m>        mean of d_i (the bootstrap estimate of r_a - r_b)
        boot_diff_<m>_std    std of d_i
        boot_diff_<m>_ci_low/high   2.5 / 97.5 percentile CI of d_i
        boot_diff_<m>_p      two-sided null-centred achieved significance level,
                             P(|d_i - d_obs| >= |d_obs|), floored at 1/n_bootstrap,
                             where d_obs is the difference on the original sample
        boot_diff_<m>_sig    significance stars for that p

    The p-value is deliberately *not* the percentile form 2*min(P(d<=0),
    P(d>=0)). The resample distribution is centred on the observed difference,
    so that form inverts a confidence interval rather than testing a null, and
    it over-rejects: simulated under a true null at n=50-100 it fires 6.2% of
    the time at a nominal 5%, against 5.5% for the null-centred version used
    here (the remaining excess is the bootstrap's own small-sample behaviour on
    a difference of dependent correlations, which no choice of p can remove).
    The CI columns are unaffected and stay percentile-based.

    for m in {pearson, spearman}. NaN/None if fewer than 4 shared observations.

    With `include_mae` (prediction evaluations, where y is the true target and
    x_a/x_b are two models' predictions), the same paired resamples also test
    the MAE difference d_i = MAE(x_a, y)_i - MAE(x_b, y)_i, emitted as the
    ``boot_diff_mae*`` columns. Note the sign convention: lower MAE is better,
    so a *negative* boot_diff_mae favors model_a (opposite of the correlation
    diffs, where positive favors model_a).
    """
    y = np.asarray(y, dtype=float)
    x_a = np.asarray(x_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x_a) & np.isfinite(x_b)
    y, x_a, x_b = y[mask], x_a[mask], x_b[mask]
    n = y.size

    def _empty(name: str) -> Dict[str, float]:
        return {
            f"boot_diff_{name}": np.nan, f"boot_diff_{name}_std": np.nan,
            f"boot_diff_{name}_ci_low": np.nan, f"boot_diff_{name}_ci_high": np.nan,
            f"boot_diff_{name}_p": np.nan, f"boot_diff_{name}_sig": None,
        }

    if n < 4:
        out = {**_empty("pearson"), **_empty("spearman"), "n_boot_diff": 0}
        if include_mae:
            out.update(_empty("mae"))
        return out

    rng = np.random.default_rng(seed)

    def _summ_diff(name: str, d: np.ndarray, d_obs: float) -> Dict[str, float]:
        d = d[np.isfinite(d)]
        if d.size < 2 or not np.isfinite(d_obs):
            return _empty(name)
        n_eff = d.size
        # Null-centred achieved significance level. The resample distribution
        # of d is centred on the *observed* difference, not on the null, so
        # asking "how much of it sits past zero" answers a confidence-interval
        # question rather than a hypothesis-test one. Recentre it on the null
        # and ask how often it lands at least as far from zero as d_obs.
        p = float(np.mean(np.abs(d - d_obs) >= abs(d_obs)))
        p = max(p, 1.0 / n_eff)  # can't be more extreme than the resolution allows
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {
            f"boot_diff_{name}": float(np.mean(d)),
            f"boot_diff_{name}_std": float(np.std(d, ddof=1)),
            f"boot_diff_{name}_ci_low": float(lo),
            f"boot_diff_{name}_ci_high": float(hi),
            f"boot_diff_{name}_p": p,
            f"boot_diff_{name}_sig": p_to_stars(p),
        }

    obs_pearson = pearsonr(x_a, y)[0] - pearsonr(x_b, y)[0]
    # Spearman (rank once, resample ranks). The observed statistic uses the
    # same full-sample ranks, so it matches what the resamples are built from.
    ry, rxa, rxb = rankdata(y), rankdata(x_a), rankdata(x_b)
    obs_spearman = spearmanr(x_a, y)[0] - spearmanr(x_b, y)[0]

    d_pearson = np.empty(n_bootstrap, dtype=float)
    d_spearman = np.empty(n_bootstrap, dtype=float)
    d_mae = np.empty(n_bootstrap, dtype=float) if include_mae else None

    # As in bootstrap_correlation_ci: draw and reduce a block at a time so peak
    # memory stays flat in n_bootstrap. Up to seven (block, n) arrays are live
    # per block (idx, the two models' values, y, and the three rank gathers).
    for start, stop in _blocks(n_bootstrap, _block_rows(n, 7)):
        idx = rng.integers(0, n, size=(stop - start, n))
        yb = y[idx]
        # Pearson (paired: same idx for both models).
        d_pearson[start:stop] = (_rowwise_pearson(x_a[idx], yb)
                                 - _rowwise_pearson(x_b[idx], yb))
        ryb = ry[idx]
        d_spearman[start:stop] = (_rowwise_pearson(rxa[idx], ryb)
                                  - _rowwise_pearson(rxb[idx], ryb))
        if include_mae:
            d_mae[start:stop] = (np.abs(x_a[idx] - yb).mean(axis=1)
                                 - np.abs(x_b[idx] - yb).mean(axis=1))

    out = {
        **_summ_diff("pearson", d_pearson, obs_pearson),
        **_summ_diff("spearman", d_spearman, obs_spearman),
        "n_boot_diff": int(np.isfinite(d_pearson).sum()),
    }
    if include_mae:
        obs_mae = float(np.mean(np.abs(x_a - y)) - np.mean(np.abs(x_b - y)))
        out.update(_summ_diff("mae", d_mae, obs_mae))
    return out


def pairwise_dependent_tests(y: np.ndarray, x_a: np.ndarray, x_b: np.ndarray,
                             include_bootstrap_diff: bool = True,
                             n_bootstrap: int = N_BOOTSTRAP,
                             include_mae: bool = False) -> Dict[str, float]:
    """Full test bundle comparing two models (x_a, x_b) against a shared target y.

    Computes, for both Pearson and Spearman:
        r_a  = corr(x_a, y)   r_b = corr(x_b, y)   r_ab = corr(x_a, x_b)
    then the Steiger (1980) and Meng (1992) Z-tests for r_a vs r_b, and (when
    `include_bootstrap_diff`) the paired-bootstrap difference test. NaN rows in
    any of the three vectors are dropped pairwise first.

    With `include_mae` (prediction evaluations only), each model's MAE point
    estimate is added (``mae_a`` / ``mae_b``) and the bootstrap-difference test
    also covers the MAE difference (``boot_diff_mae*``; negative favors
    model_a since lower MAE is better).

    Returns a flat dict of correlations, z/p statistics, sample size `n`,
    significance stars, and (optionally) the bootstrap-difference columns from
    `bootstrap_difference_test`.
    """
    y = np.asarray(y, dtype=float)
    x_a = np.asarray(x_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x_a) & np.isfinite(x_b)
    y, x_a, x_b = y[mask], x_a[mask], x_b[mask]
    n = y.size

    nan_bundle = {
        "n": n,
        "pearson_r_a": np.nan, "pearson_r_b": np.nan, "pearson_r_ab": np.nan,
        "steiger_z_pearson": np.nan, "steiger_p_pearson": np.nan, "steiger_sig_pearson": None,
        "meng_z_pearson": np.nan, "meng_p_pearson": np.nan,
        "spearman_r_a": np.nan, "spearman_r_b": np.nan, "spearman_r_ab": np.nan,
        "steiger_z_spearman": np.nan, "steiger_p_spearman": np.nan, "steiger_sig_spearman": None,
        "meng_z_spearman": np.nan, "meng_p_spearman": np.nan,
    }
    if include_mae:
        nan_bundle.update({"mae_a": np.nan, "mae_b": np.nan})
    if n < 4:
        if include_bootstrap_diff:
            nan_bundle.update(bootstrap_difference_test(
                y, x_a, x_b, n_bootstrap=n_bootstrap, include_mae=include_mae))
        return nan_bundle

    rp_a = pearsonr(x_a, y)[0]
    rp_b = pearsonr(x_b, y)[0]
    rp_ab = pearsonr(x_a, x_b)[0]
    rs_a = spearmanr(x_a, y)[0]
    rs_b = spearmanr(x_b, y)[0]
    rs_ab = spearmanr(x_a, x_b)[0]

    st_z_p, st_p_p = steiger_dependent_overlap(rp_a, rp_b, rp_ab, n)
    mn_z_p, mn_p_p = meng_dependent_overlap(rp_a, rp_b, rp_ab, n)
    st_z_s, st_p_s = steiger_dependent_overlap(rs_a, rs_b, rs_ab, n)
    mn_z_s, mn_p_s = meng_dependent_overlap(rs_a, rs_b, rs_ab, n)
    out = {
        "n": n,
        "pearson_r_a": rp_a, "pearson_r_b": rp_b, "pearson_r_ab": rp_ab,
        "steiger_z_pearson": st_z_p, "steiger_p_pearson": st_p_p, "steiger_sig_pearson": p_to_stars(st_p_p),
        "meng_z_pearson": mn_z_p, "meng_p_pearson": mn_p_p,
        "spearman_r_a": rs_a, "spearman_r_b": rs_b, "spearman_r_ab": rs_ab,
        "steiger_z_spearman": st_z_s, "steiger_p_spearman": st_p_s, "steiger_sig_spearman": p_to_stars(st_p_s),
        "meng_z_spearman": mn_z_s, "meng_p_spearman": mn_p_s,
    }
    if include_mae:
        out.update({
            "mae_a": float(np.mean(np.abs(x_a - y))),
            "mae_b": float(np.mean(np.abs(x_b - y))),
        })
    if include_bootstrap_diff:
        out.update(bootstrap_difference_test(
            y, x_a, x_b, n_bootstrap=n_bootstrap, include_mae=include_mae))
    return out


def p_to_stars(p: Optional[float]) -> Optional[str]:
    """Significance symbol for a p-value: ***<0.001, **<0.01, *<0.05, else 'ns'."""
    if p is None or not np.isfinite(p):
        return None
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p <= 1:
        return "ns"
    return None
