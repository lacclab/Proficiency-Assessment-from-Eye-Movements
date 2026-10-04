"""Significance tests for the effect of debiasing on EyeScore.

Two paired bootstrap tests, both answering "did debiasing change X?" rather
than "is X non-zero":

    delta_r  -- change in Pearson r between EyeScore and a proficiency target
                (feeds the stars in combined_debiased_*_diff.tex)
    delta_b  -- change in the two-step L1-distance coefficient b_dist
                (feeds the optional db-column stars in the coef tables)

Both resample PARTICIPANTS with replacement and recompute the statistic on the
raw and debiased scores within the same resample, so the paired difference
carries the covariance between them.

Two design points matter, and getting either wrong makes the test useless:

1. ``alpha`` is REFITTED inside every replicate. Reusing the pipeline's stored
   debiased column treats the debiasing coefficient as a known constant with
   zero uncertainty. Because ``eye_score_db = eye_score - alpha * (d - c)`` and
   b_dist is linear in the score, that makes ``delta_b`` a near-deterministic
   read-back of alpha: its standard error collapses by 10-40x and every cell
   comes out significant, including cells with no measurable bias to remove.
   Refitting restores the fact that alpha was estimated from the same noisy
   data, and delta_b then inherits the uncertainty of the coefficient it is
   correcting.

2. Both statistics are computed on the participants present in BOTH the raw and
   debiased score files. The inside_langs scheme drops L1 groups smaller than
   the fold count (OneStop loses Vietnamese, n=5 < k=10), so reading each file
   independently would subtract correlations measured on different samples --
   which on OneStop shifts delta_r by up to 0.027 and flips one sign.

The fold structure is fixed once on the original data and participants carry
their fold label into the resample, so a duplicated participant never lands in
both the train and test side of the same fold.

LIMITATION -- the refit is an approximation of the pipeline, and how good an
approximation depends on the scheme. The pipeline's leak-free path also
RE-SCORES eye_score per fold, with the held-out participants removed from the
L2 z-score fit pool; this module refits only the debiasing coefficient, starting
from the stored raw scores. For inside_langs the excluded fold is 1/k of the
participants drawn from every language, and the approximation is tight
(measured on MECO/WPM: delta_r -0.01517 refit vs -0.01507 stored, 0.7%). For
across_langs it excludes an entire language -- up to 13% of the sample, and a
non-random 13% -- so the refit debiasing is visibly not the pipeline's
(delta_r -0.00974 vs -0.00226; delta_b +1.144 vs +0.945).

The reported delta is therefore always taken from the pipeline's stored debiased
scores, and the bootstrap distribution is re-centred on it (see `_pvalue`): the
resamples supply the spread, not the location. For across_langs, read that
spread as an estimate rather than an exact reproduction of the pipeline's
sampling variability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.constants import Fields
from src.methods.EyeScore.calculation import TYPO_CALIB_INSIDE_RANDOM_STATE

# Bootstrap replicates. The Monte-Carlo error on a p-value is sqrt(p(1-p)/B),
# so B=2000 leaves +-0.005 at the 0.05 threshold and, worse, a floor of 1/2000
# = 0.0005 that sits right on the 0.001 star boundary -- cells whose true p is
# near 0.002 land on either side depending on B. 10000 halves the error and
# drops the floor to 1e-4. Callers doing exploratory work can lower it.
N_BOOTSTRAP = 10000

# Missing-value sentinel used by the proficiency target columns.
MISSING = -1


def two_step_slope(y: np.ndarray, p: np.ndarray, d: np.ndarray) -> "float | None":
    """b_dist: residualize `y` on proficiency `p`, then slope of residual on `d`.

    Closed form rather than two np.polyfit calls -- identical result, but this
    runs inside the bootstrap loop several hundred thousand times.
    """
    if len(y) < 3:
        return None
    pc = p - p.mean()
    dc = d - d.mean()
    vp = pc @ pc
    vd = dc @ dc
    if vp == 0 or vd == 0:
        return None
    yc = y - y.mean()
    resid = yc - (pc @ yc) / vp * pc
    return float((dc @ resid) / vd)


def _pearson(a: np.ndarray, b: np.ndarray) -> "float | None":
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def assign_folds(df: pd.DataFrame, scheme: str) -> "tuple[pd.DataFrame, np.ndarray] | None":
    """Reproduce the pipeline's fold split, once, on the original data.

    Returns (possibly filtered df, fold-key array) or None if the scheme cannot
    be built. Mirrors _typology_{inside,across}_langs_correction: inside_langs is
    a StratifiedKFold over participants with k = number of L1s, which requires
    every L1 to have at least k participants; across_langs is leave-one-language
    -out, keyed by L1 itself.
    """
    if scheme == "across_langs":
        return df.reset_index(drop=True), df[Fields.L1].to_numpy()

    n_splits = df[Fields.L1].nunique()
    if n_splits < 2:
        return None
    counts = df.groupby(Fields.L1).size()
    df = df[~df[Fields.L1].isin(counts[counts < n_splits].index)].reset_index(drop=True)
    n_splits = df[Fields.L1].nunique()
    if n_splits < 2 or len(df) < n_splits:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=TYPO_CALIB_INSIDE_RANDOM_STATE)
    key = np.full(len(df), -1)
    for i, (_, test_idx) in enumerate(skf.split(df, df[Fields.L1])):
        key[test_idx] = i
    return df, key


def _debias_once(y, calib, dist, fold_key, fit_mask, center: bool):
    """Refit alpha per fold and apply it; returns the debiased scores or None.

    `fit_mask` restricts the alpha fit to participants who have the calibration
    target, matching the pipeline (which fits on LexTALE-available participants
    but applies the correction to everyone with a known distance).
    """
    out = y.copy()
    for key in pd.unique(fold_key):
        test = fold_key == key
        train = (~test) & fit_mask
        if train.sum() < 3 or len(np.unique(dist[train])) < 2:
            return None
        alpha = two_step_slope(y[train], calib[train], dist[train])
        if alpha is None:
            return None
        shift = dist[test] - (dist[train].mean() if center else 0.0)
        out[test] = y[test] - alpha * shift
    return out


def _pvalue(draws: np.ndarray, delta: float) -> float:
    """Two-sided bootstrap p for H0: delta = 0, anchored at the observed `delta`.

    The resamples are re-centred on the reported delta before the tails are
    counted. A plain percentile p would ask where zero sits in the *resample*
    distribution, which is only equivalent when that distribution is centred on
    the reported estimate -- and it is not, because the reported delta comes
    from the pipeline's stored debiased scores while the resamples refit the
    debiasing coefficient (see the module docstring's LIMITATION note). On
    across_langs cells the two centres differ by more than the standard error,
    which without this correction turns a p of 0.62 into 0.007.

    Floored at 1/B, so B controls the smallest reportable p.
    """
    centred = draws - draws.mean() + delta
    p = 2 * min((centred >= 0).mean(), (centred <= 0).mean())
    return float(max(p, 1 / len(draws)))


def _bootstrap(stat_fn, y_raw, y_db, target, calib, dist, fold_key,
               eval_mask, fit_mask, center, n_boot, seed):
    """Shared paired-bootstrap driver. `stat_fn(scores, target)` -> float|None."""
    point_db = stat_fn(y_db[eval_mask], target[eval_mask])
    point_raw = stat_fn(y_raw[eval_mask], target[eval_mask])
    if point_db is None or point_raw is None:
        return None
    delta = point_db - point_raw

    rng = np.random.default_rng(seed)
    n = len(y_raw)
    draws = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        ev = eval_mask[idx]
        if ev.sum() < 10:
            continue
        adjusted = _debias_once(y_raw[idx], calib[idx], dist[idx],
                                fold_key[idx], fit_mask[idx], center)
        if adjusted is None:
            continue
        a = stat_fn(adjusted[ev], target[idx][ev])
        b = stat_fn(y_raw[idx][ev], target[idx][ev])
        if a is None or b is None:
            continue
        draws.append(a - b)

    if len(draws) < n_boot // 10:
        return None
    return delta, _pvalue(np.asarray(draws), delta)


def _prepare(raw_df, db_df, meta, target_col, calib_col, distances, scheme):
    """Merge raw + debiased + metadata onto the participants common to both."""
    keep = list(dict.fromkeys([Fields.SUBJECT_ID, target_col, calib_col]))
    missing = [c for c in keep if c not in meta.columns]
    if missing:
        return None
    merged = (
        raw_df[[Fields.SUBJECT_ID, "eye_score", Fields.L1]]
        .merge(db_df[[Fields.SUBJECT_ID, "eye_score"]], on=Fields.SUBJECT_ID,
               suffixes=("", "_db"))
        .merge(meta[keep].drop_duplicates(Fields.SUBJECT_ID), on=Fields.SUBJECT_ID)
    )
    merged["dist"] = merged[Fields.L1].map(distances)
    merged = merged.dropna(subset=["eye_score", "eye_score_db", Fields.L1, "dist"])
    if merged.empty:
        return None

    split = assign_folds(merged, scheme)
    if split is None:
        return None
    merged, fold_key = split

    target = merged[target_col].to_numpy(dtype=float)
    calib = merged[calib_col].to_numpy(dtype=float)
    eval_mask = (target != MISSING) & np.isfinite(target)
    fit_mask = (calib != MISSING) & np.isfinite(calib)
    if eval_mask.sum() < 30 or fit_mask.sum() < 10:
        return None
    return dict(
        y_raw=merged["eye_score"].to_numpy(dtype=float),
        y_db=merged["eye_score_db"].to_numpy(dtype=float),
        target=target, calib=calib,
        dist=merged["dist"].to_numpy(dtype=float),
        fold_key=fold_key, eval_mask=eval_mask, fit_mask=fit_mask,
        n=int(eval_mask.sum()),
    )


_CACHE: dict = {}


def delta_r_test(raw_df, db_df, meta, target_col, calib_col, distances, scheme,
                 center: bool = True, n_boot: int = N_BOOTSTRAP, seed: int = 42,
                 cache_key=None) -> "tuple[float, float] | None":
    """Paired bootstrap for delta_r -- the change debiasing makes to the Pearson
    correlation between EyeScore and `target_col`.

    Args:
        raw_df / db_df: eye_score frames for the raw and debiased variants.
        meta: metadata carrying `target_col` and `calib_col`.
        target_col: proficiency measure the correlation is computed against.
        calib_col: proficiency measure alpha is refitted on (the debias
            calibration target -- LexTALE for the canonical tables).
        distances: L1 -> typological distance.
        scheme: 'inside_langs' or 'across_langs'.
        center: subtract the train fold's mean distance before applying alpha,
            matching the `*ctr` result trees (EYESCORE_CENTER_DISTANCE=1).

    Returns (delta_r, p) or None when the cell cannot be computed.
    """
    if cache_key is not None and cache_key in _CACHE:
        return _CACHE[cache_key]

    prepared = _prepare(raw_df, db_df, meta, target_col, calib_col,
                        distances, scheme)
    result = None
    if prepared is not None:
        result = _bootstrap(_pearson, prepared["y_raw"], prepared["y_db"],
                            prepared["target"], prepared["calib"],
                            prepared["dist"], prepared["fold_key"],
                            eval_mask=prepared["eval_mask"],
                            fit_mask=prepared["fit_mask"],
                            center=center, n_boot=n_boot, seed=seed)

    if cache_key is not None:
        _CACHE[cache_key] = result
    return result


def delta_b_test(raw_df, db_df, meta, target_col, calib_col, distances, scheme,
                 center: bool = True, n_boot: int = N_BOOTSTRAP, seed: int = 42,
                 cache_key=None) -> "tuple[float, float] | None":
    """Paired bootstrap for delta_b -- the change in the two-step L1-distance
    coefficient. Same design as `delta_r_test`, but the statistic needs the
    resampled distances, so the resampling loop lives here.

    Note this is the same test as `b_raw != 0` whenever the calibration target
    equals the evaluation target: alpha and b_dist are then the same statistic
    fitted on a training fold versus on everyone, so the two agree to two
    decimal places. It carries independent information only under cross-target
    calibration (e.g. LexTALE-fitted debiasing reported against Michigan).
    """
    if cache_key is not None and cache_key in _CACHE:
        return _CACHE[cache_key]

    prepared = _prepare(raw_df, db_df, meta, target_col, calib_col,
                        distances, scheme)
    result = None
    if prepared is not None:
        y_raw, y_db = prepared["y_raw"], prepared["y_db"]
        target, calib = prepared["target"], prepared["calib"]
        dist, fold_key = prepared["dist"], prepared["fold_key"]
        ev, fit = prepared["eval_mask"], prepared["fit_mask"]

        b_db = two_step_slope(y_db[ev], target[ev], dist[ev])
        b_raw = two_step_slope(y_raw[ev], target[ev], dist[ev])
        if b_db is not None and b_raw is not None:
            rng = np.random.default_rng(seed)
            n = len(y_raw)
            draws = []
            for _ in range(n_boot):
                idx = rng.choice(n, n, replace=True)
                m = ev[idx]
                dd = dist[idx]
                if m.sum() < 10 or len(np.unique(dd[m])) < 3:
                    continue
                adjusted = _debias_once(y_raw[idx], calib[idx], dd,
                                        fold_key[idx], fit[idx], center)
                if adjusted is None:
                    continue
                a = two_step_slope(adjusted[m], target[idx][m], dd[m])
                b = two_step_slope(y_raw[idx][m], target[idx][m], dd[m])
                if a is None or b is None:
                    continue
                draws.append(a - b)
            if len(draws) >= n_boot // 10:
                delta = b_db - b_raw
                result = (delta, _pvalue(np.asarray(draws), delta))

    if cache_key is not None:
        _CACHE[cache_key] = result
    return result


def n_stars(p: "float | None") -> int:
    """Star count for a p-value; 0 when absent or non-significant."""
    if p is None:
        return 0
    return 3 if p < 0.001 else 2 if p < 0.01 else 1 if p < 0.05 else 0


def stars_tex(count: int) -> str:
    """LaTeX superscript for `count` stars ('' when zero)."""
    return "$^{" + "*" * count + "}$" if count else ""
