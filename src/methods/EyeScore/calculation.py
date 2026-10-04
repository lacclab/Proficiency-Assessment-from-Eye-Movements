"""EyeScore: each L2 reader's similarity to the L1 (native-reader) prototype.

Sections:
  scorers           — how a standardized feature vector becomes an eye_score
                      (prototype similarity or a discriminative classifier)
  eye_score         — the scoring itself, per content_mode, with leave-one-out
  corrections       — the typology debias corrections and the debias-method
                      registry (two_step, distance_mreg, interaction, ...)
  fold scorers      — leak-free per-fold rescoring used by the corrections
  runners           — run_one_feature_set / run_one_dataset_version, driven by
                      src/run/eyescore.py
"""
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.constants import (
    ALL_FEATURE_SETS_DICT,
    ALL_TARGET_COLS_SMALL,
    DATA_PATH,
    DataSets,
    FIXED_TEXT_GROUP_PREFIXES,
    Fields,
    TestCols,
)
from src.methods.utils import load_per_text_features_pair, zscore_preserve_nan_general
from src.utils.filter import filter_by_preview

logger.add(DATA_PATH / 'logs' / 'calculate_eyescore.log', rotation='10 MB', retention='7 days')

# Fast leave-one-out path (see _loo_fit_stats and the leave-one-out branch in
# eye_score): an O(n·d) Welford downdate of the z-score statistics instead of
# recomputing them for every participant, which on the wide per-text sets is
# the difference between hours and weeks. Matches the reference path to
# floating-point precision on real data (max |Δ| ≈ 2.8e-16 on WFC). On by
# default; set EYESCORE_FAST_LOO=0 to force the exact reference path.
_ENABLE_FAST_LOO = True


# ── Scorer registry ──────────────────────────────────────────────────────────
#
# A "scorer" selects HOW an L2 reader's standardized feature vector is turned
# into an eye_score. Two families:
#   • prototype     — score against the L1 prototype (mean of native readers)
#                     via a similarity/distance metric. "cosine" is the historical
#                     default; "euclidean" keeps magnitude (drop-in, no L2 training).
#   • discriminative — a classifier trained on L1-vs-L2 identity; the score is the
#                      out-of-fold log-odds of being native (needs L2 training, so
#                      it is produced by cross-fitting — see _eye_score_discriminative).
#
# Each scorer writes to its own sibling results tree (results_<scorer>/), exactly
# like the debias methods, so downstream evaluation/plots/tables are unchanged.
# The scorer is threaded explicitly (not a module global) because the pipeline
# fans out multiple trees inside one ProcessPoolExecutor run.

# Above this width, shrinkage LDA is not fittable: solver="lsqr" with
# shrinkage="auto" forms a p x p covariance and solves it (O(p^2) memory,
# O(p^3) time), which on the per-text TRANSITIONS set (~78k columns) means a
# 48 GB matrix and ~5e14 flops. Wider matrices fall back to the SVD solver.
LDA_SHRINKAGE_MAX_FEATURES = 10_000


def _make_logistic(n_features: Optional[int] = None):
    """Balanced logistic regression with a dimension-scaled regularization default.

    C = 1 / n_features. Rationale: features are standardized before fitting, so
    ||x||^2 grows with p and the logit scale grows with it; the L2 penalty has to
    scale with p to keep per-feature shrinkage constant. This is the same family
    as sklearn's own ``gamma="scale"`` default for RBF SVMs (1/(n_features*var)),
    and the constant (C0 = 1) is sklearn's default C at p = 1.

    Chosen a priori: it depends only on the feature-matrix width, never on the
    target, so it needs neither an inner CV nor any look at proficiency labels —
    preserving the label-free contract. The previous flat C=1.0 was badly
    under-regularized where p >> n_positives (MECO has 95 natives): on the
    133-dim set it scored 0.372 vs 0.511 for cosine, where C=1/p scores 0.511.

    Caveat: a heuristic, not an optimum. It recovers ~70% of an oracle C sweep's
    headroom and scales with p only, ignoring n and n_positives.
    """
    from sklearn.linear_model import LogisticRegression
    C = 1.0 / n_features if n_features else 1.0
    return LogisticRegression(class_weight="balanced", C=C, max_iter=1000)


def _make_lda(n_features: Optional[int] = None):
    """Shrinkage LDA, falling back to the SVD solver on very wide matrices.

    NOTE the fallback is a DIFFERENT estimator: sklearn's svd solver handles
    p > n without ever forming the p x p covariance, but it does not support
    shrinkage, so those cells are unregularized LDA. Footnote them when
    reporting alongside the shrinkage cells.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    if n_features is not None and n_features > LDA_SHRINKAGE_MAX_FEATURES:
        logger.warning(
            f"LDA: {n_features} features exceeds {LDA_SHRINKAGE_MAX_FEATURES}; "
            "using solver='svd' WITHOUT shrinkage for this fit."
        )
        return LinearDiscriminantAnalysis(solver="svd")
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


@dataclass(frozen=True)
class Scorer:
    name: str
    family: str                                  # "prototype" | "discriminative"
    metric: Optional[str] = None                 # prototype: "cosine" | "euclidean"
    # discriminative: (n_features) -> sklearn estimator. The width is passed so a
    # factory can adapt its solver to very wide matrices (see _make_lda).
    estimator_factory: Optional[Callable] = None


SCORERS = {
    "cosine":    Scorer("cosine",    "prototype", metric="cosine"),
    "euclidean": Scorer("euclidean", "prototype", metric="euclidean"),
    "logistic":  Scorer("logistic",  "discriminative", estimator_factory=_make_logistic),
    "lda":       Scorer("lda",       "discriminative", estimator_factory=_make_lda),
}
DEFAULT_SCORER = "cosine"
SCORER_NAMES = frozenset(SCORERS)


def get_prototype_L1(L1_X_scaled: pd.DataFrame) -> pd.Series:
    """The L1 prototype: the column-wise mean of the scaled L1 feature rows."""
    return L1_X_scaled.mean(axis=0)


def eye_score_one_participant(features_participant: pd.Series, prototype_L1: pd.Series,
                              metric: str = "cosine") -> float:
    """Calculate the eye score for a single participant based on their features and the prototype L1 features.

    Args:
        features_participant: Feature values for a single participant.
        prototype_L1: Prototype L1 feature values.
        metric: "cosine" (default) or "euclidean". Higher = more native for both.

    Returns:
        Eye score between participant and prototype: cosine similarity, or the
        negated Euclidean distance (so larger is still "more native").

    NaN handling differs by metric because the two need different comparability:
      • cosine    — cells NaN in either the participant OR the prototype are
        dropped; cosine is scale-normalized so a per-row-masked subset is fine.
        Natural for per-text features where unread-article columns are NaN.
      • euclidean — an unnormalized distance over a per-row-masked subset is NOT
        comparable across participants (fewer valid columns → artificially
        smaller distance). Instead we drop only columns where the *prototype* is
        NaN (a global column mask, same for everyone) and impute the
        participant's remaining NaNs to 0 — the fit-mean in z-space, i.e. the
        prototype's own value — so every participant is scored in the same full
        dimensionality.
    """
    p = np.asarray(features_participant, dtype=float)
    q = np.asarray(prototype_L1, dtype=float)
    if metric == "euclidean":
        colmask = ~np.isnan(q)
        if not colmask.any():
            return float('nan')
        p, q = p[colmask], q[colmask]
        p = np.where(np.isnan(p), 0.0, p)          # impute to fit-mean (0 in z-space)
        return float(-np.linalg.norm(p - q))
    mask = ~(np.isnan(p) | np.isnan(q))
    if not mask.any():
        return float('nan')
    p, q = p[mask], q[mask]
    pn = np.linalg.norm(p)
    qn = np.linalg.norm(q)
    if pn == 0 or qn == 0:
        return float('nan')
    return float(np.dot(p, q) / (pn * qn))


def eye_score_batch(features_matrix, prototype_L1, metric: str = "cosine") -> np.ndarray:
    """Vectorized eye score of many participants against one prototype.

    Numerically identical to calling ``eye_score_one_participant`` row-by-row for
    the same ``metric`` (same NaN handling — see that function), but computed as
    array ops so scoring a whole block is one pass instead of a Python loop.

    Args:
        features_matrix: (n_participants, n_features) array-like of scaled features.
        prototype_L1: (n_features,) prototype vector.
        metric: "cosine" (default) or "euclidean".

    Returns:
        (n_participants,) array of eye_scores; NaN where a cosine row shares no
        non-NaN feature with the prototype (or has zero norm on the overlap), or
        where the prototype is all-NaN for euclidean.
    """
    P = np.asarray(features_matrix, dtype=float)
    q = np.asarray(prototype_L1, dtype=float)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    if metric == "euclidean":
        colmask = ~np.isnan(q)
        if not colmask.any():
            return np.full(P.shape[0], np.nan)
        Pv = P[:, colmask]
        qv = q[colmask]
        Pv = np.where(np.isnan(Pv), 0.0, Pv)       # impute to fit-mean (0 in z-space)
        D = Pv - qv[None, :]
        return -np.sqrt(np.einsum("ij,ij->i", D, D))
    # Per-(row, col) validity: finite in both the row and the prototype.
    valid = ~np.isnan(P) & ~np.isnan(q)[None, :]
    Pv = np.where(valid, P, 0.0)
    Qv = np.where(valid, q[None, :], 0.0)          # prototype re-masked per row
    dot = np.einsum("ij,ij->i", Pv, Qv)
    pn = np.sqrt(np.einsum("ij,ij->i", Pv, Pv))
    qn = np.sqrt(np.einsum("ij,ij->i", Qv, Qv))
    denom = pn * qn
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = dot / denom
    scores[denom == 0] = np.nan
    return scores


def _loo_fit_stats(fit_df: pd.DataFrame, id_series: pd.Series):
    """Precompute per-participant leave-one-out column mean/std (ddof=1) for a
    z-score fit frame, in O(n·d) total instead of O(participants·n·d).

    The naive leave-one-out path recomputes ``fit_subset.mean()`` / ``.std()``
    over the whole fit set once per participant — the dominant cost of the
    seen/unseen and split-half runs. Here the full-set (count, sum, M2) are
    computed once, then each participant's rows are *removed* with a
    numerically stable Welford downdate:

        M2_A = M2_AB − M2_B − (mean_B − mean_A)² · n_A·n_B / n_AB

    Because M2 is deviations-from-mean (not Σx² − (Σx)²/n), a column that is
    constant within the leave-one-out subset yields M2_A == 0 exactly, so it is
    dropped by the ``std > 0`` test just as pandas' two-pass ``.std()`` would —
    keeping the kept-column set identical. std values match pandas to floating
    point; the ``EYESCORE_FAST_LOO`` gate + regression test guard the rest.

    Returns (columns, {participant_id: (mean_array, std_array)}) with arrays
    aligned to ``columns``. NaN std (a column left with < 2 finite values) maps
    to "drop" downstream, matching pandas.
    """
    cols = list(fit_df.columns)
    F = fit_df.to_numpy(dtype=float)
    ids = id_series.to_numpy()
    finite = ~np.isnan(F)
    N = finite.sum(0).astype(float)
    S = np.nansum(F, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = S / N
    dev = F - mean
    M2 = np.nansum(dev * dev, 0)

    out = {}
    for pid in pd.unique(ids):
        rows = ids == pid
        Fb = F[rows]
        n_B = (~np.isnan(Fb)).sum(0).astype(float)
        S_B = np.nansum(Fb, 0)
        n_A = N - n_B
        S_A = S - S_B
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_A = S_A / n_A
            mean_B = S_B / n_B
        devB = Fb - mean_B
        M2_B = np.nansum(devB * devB, 0)
        delta = mean_B - mean_A
        term = np.where(n_B > 0, (delta * delta) * (n_A * n_B / N), 0.0)
        M2_A = M2 - M2_B - term
        with np.errstate(invalid="ignore", divide="ignore"):
            var_A = np.where(n_A >= 2, M2_A / (n_A - 1.0), np.nan)
        std_A = np.sqrt(np.clip(var_A, 0.0, None))
        out[pid] = (mean_A, std_A)

    # Full-set stats, for participants that are NOT in the fit frame at all.
    # The leak-free fold scorer's "apply" call scores held-out participants
    # against a train-only fit pool, so every target misses `out` — and for
    # them the reference path's `fit_df[ids != pid]` is the whole fit set, i.e.
    # no downdate is needed. Same ddof=1 / n>=2 -> NaN rule as above so the
    # kept-column set matches pandas exactly.
    with np.errstate(invalid="ignore", divide="ignore"):
        var_full = np.where(N >= 2, M2 / (N - 1.0), np.nan)
    full = (mean, np.sqrt(np.clip(var_full, 0.0, None)))
    return cols, out, full


def eye_score_per_item_independent(
    features_df_L1: pd.DataFrame,
    features_df_L2: pd.DataFrame,
    save_path: Path,
    replace_existing: bool = False,
    per_text_parquet_dir: Path = None,
    content_mode: Literal["all", "seen", "unseen"] = "all",
    fit_on_df: Optional[pd.DataFrame] = None,
    item_col: str = None,
    random_state: int = 42,
    meco_seen_unseen: str = None,
    l1_parquet_dir: Path | None = None,
    l2_parquet_dir: Path | None = None,
    feature_set: str = None,
    metric: str = "cosine",
) -> pd.DataFrame:
    """Per-item eye scores (one row per participant × item), averaged per participant.

    Two paths:
      content_mode="all" with fit_on_df — the MECO half-split: each participant
          is scored against a z-score fit and L1 pool with that participant left
          out; `meco_seen_unseen` selects whether the fit/pool are restricted to
          the same item ("seen") or span every item ("unseen").
      content_mode="seen" / "unseen" — OneStop, per content group, with the same
          pool definitions as `eye_score` (same cg for seen, other article
          batches for unseen), leaving the participant out in seen mode.

    Per-text feature sets (TRANSITIONS / WFC) are merged in per item from the
    per_text_features_item_<id>.parquet files in `l1_parquet_dir` /
    `l2_parquet_dir` (or `per_text_parquet_dir` for both).

    Writes the per-item scores to <save_path stem>_items.csv and the
    per-participant means to `save_path`. `replace_existing` and `random_state`
    are accepted for a uniform call signature and are not used.

    Returns:
        Aggregated eye scores DataFrame (one row per participant)
    """
    if content_mode != "all" and Fields.CONTENT_GROUP not in features_df_L1.columns:
        raise ValueError(
            f"content_mode={content_mode!r} requires '{Fields.CONTENT_GROUP}' column; "
            "re-run feature extraction to add it."
        )
    if fit_on_df is not None and content_mode != "all":
        raise ValueError(
            f"fit_on_df only applies to content_mode='all'; got {content_mode!r}."
        )

    if item_col is None:
        raise ValueError("item_col must be specified for per-item eye score calculation")
    features_df_L2 = features_df_L2.reset_index(drop=True)
    features_df_L1 = features_df_L1.reset_index(drop=True)
    features_df_L1_x = _drop_metadata_cols(features_df_L1)
    features_df_L2_x = _drop_metadata_cols(features_df_L2)

    single_feature = len(features_df_L2_x.columns.tolist()) == 1
    if single_feature:
        logger.warning(
            "Only one feature available for eye score calculation. Eye scores will be directly that feature."
            f"{features_df_L2_x.columns.tolist()[0]}"
        )

    if content_mode == "all" and fit_on_df is None:
        raise NotImplementedError("content_mode='all' without fit_on_df is not implemented ")
    # Leave-one-out path: train on fit_on_df minus each participant, score that participant
    if content_mode == "all" and fit_on_df is not None:
        per_item_scores = []
        eyescores_dict = {}

        fit_on_df_clean = _drop_metadata_cols(fit_on_df)
        l1_x = _drop_metadata_cols(features_df_L1)
        l2_x = _drop_metadata_cols(features_df_L2)

        for participant_id in tqdm(features_df_L2[Fields.SUBJECT_ID].unique(), desc="MECO participants", leave=False):

            if meco_seen_unseen == "unseen":
                # Remove participant from fit and L1 (global, not per item)
                fit_mask = fit_on_df[Fields.SUBJECT_ID] != participant_id
                l1_mask = features_df_L1[Fields.SUBJECT_ID] != participant_id

                fit_subset = fit_on_df_clean[fit_mask]
                l1_subset = l1_x[l1_mask]

                if len(fit_subset) == 0 or len(l1_subset) == 0:
                    continue

                l1_scaled = zscore_preserve_nan_general(fit_subset, l1_subset)
                prototype = get_prototype_L1(l1_scaled)

            # Loop through items THIS participant read
            participant_items = features_df_L2[features_df_L2[Fields.SUBJECT_ID] == participant_id][item_col].unique()
            for item_id in tqdm(participant_items, desc=f"Items for {participant_id}", leave=False):
                if meco_seen_unseen == "seen":
                    # For SEEN: filter to only this item, exclude current participant
                    fit_mask = (fit_on_df[item_col] == item_id) & (fit_on_df[Fields.SUBJECT_ID] != participant_id)
                    l1_mask = (features_df_L1[item_col] == item_id) & (features_df_L1[Fields.SUBJECT_ID] != participant_id)

                    fit_subset = fit_on_df_clean[fit_mask]
                    l1_subset = l1_x[l1_mask]

                    # Load per-item parquets if dirs provided (TRANSITIONS/WFC feature sets only)
                    # Support both per-item SEEN/UNSEEN path (l1_parquet_dir/l2_parquet_dir)
                    # and regular per-item path (per_text_parquet_dir)
                    pt_l1_dir = l1_parquet_dir or per_text_parquet_dir
                    pt_l2_dir = l2_parquet_dir or per_text_parquet_dir
                    if pt_l1_dir is not None and pt_l2_dir is not None:
                        # Parquet filenames:
                        # - article level: per_text_features_item_{numeric_id}.parquet
                        # - paragraph level: per_text_features_item_{unique_paragraph_id}.parquet
                        if isinstance(item_id, str):
                            parquet_id = item_id  # Use unique_paragraph_id directly
                        else:
                            parquet_id = str(item_id)  # Use article_id

                        l1_pt_path = pt_l1_dir / f"per_text_features_item_{parquet_id}.parquet"
                        l2_pt_path = pt_l2_dir / f"per_text_features_item_{parquet_id}.parquet"
                        if l1_pt_path.exists() and l2_pt_path.exists():
                            try:
                                l1_pt = pd.read_parquet(l1_pt_path, thrift_string_size_limit=2**31-1, thrift_container_size_limit=2**31-1)
                                l2_pt = pd.read_parquet(l2_pt_path, thrift_string_size_limit=2**31-1, thrift_container_size_limit=2**31-1)

                                # Extract per-text columns - filter by feature_set if provided
                                feature_prefix = None
                                if feature_set and feature_set in FIXED_TEXT_GROUP_PREFIXES:
                                    feature_prefix = FIXED_TEXT_GROUP_PREFIXES[feature_set]

                                # Filter to just the relevant prefix
                                if feature_prefix:
                                    is_pt_col = lambda c: c.startswith(feature_prefix)
                                else:
                                    raise ValueError(f"Invalid feature_set {feature_set!r}, expected one of {list(FIXED_TEXT_GROUP_PREFIXES.keys())}")

                                l1_pt_cols = [c for c in l1_pt.columns if is_pt_col(c)]
                                l2_pt_cols = [c for c in l2_pt.columns if is_pt_col(c)]
                                union_cols = sorted(set(l1_pt_cols) | set(l2_pt_cols))

                                # Keep only this item's columns. Paragraph ids are strings
                                # (e.g. '1_5_Ele_1' -> 'transitions_1_5_Ele_1_0_0'); article
                                # ids are numbers matched as '_<id>_'.
                                if isinstance(item_id, str):
                                    item_feature_cols = [c for c in union_cols if f"{feature_prefix}{item_id}_" in c]
                                else:
                                    item_feature_cols = [c for c in union_cols if f"_{item_id}_" in c]
                                if item_feature_cols:
                                    merge_cols = [Fields.SUBJECT_ID, item_col]
                                    l1_pt_for_merge = l1_pt.reindex(columns=merge_cols + item_feature_cols, fill_value=0)
                                    l2_pt_for_merge = l2_pt.reindex(columns=merge_cols + item_feature_cols, fill_value=0)

                                    # Merge per-text columns into the feature subsets
                                    fit_subset = fit_subset.reset_index(drop=True).merge(
                                        fit_on_df[fit_mask][merge_cols].reset_index(drop=True)
                                            .merge(l2_pt_for_merge, on=merge_cols, how='left')[item_feature_cols],
                                        left_index=True, right_index=True
                                    )
                                    l1_subset = l1_subset.reset_index(drop=True).merge(
                                        features_df_L1[l1_mask][merge_cols].reset_index(drop=True)
                                            .merge(l1_pt_for_merge, on=merge_cols, how='left')[item_feature_cols],
                                        left_index=True, right_index=True
                                    )
                            except Exception as e:
                                logger.warning(f"Could not load per-text features for item {item_id}: {e}")

                    if len(fit_subset) == 0 or len(l1_subset) == 0:
                        continue

                    l1_scaled = zscore_preserve_nan_general(fit_subset, l1_subset)
                    prototype = get_prototype_L1(l1_scaled)

                # Filter test to just this item
                p_mask = (features_df_L2[Fields.SUBJECT_ID] == participant_id) & (features_df_L2[item_col] == item_id)
                p_features = l2_x[p_mask]

                # Add per-text features if available (SEEN mode only)
                pt_l2_dir_test = l2_parquet_dir or per_text_parquet_dir
                if meco_seen_unseen == "seen" and pt_l2_dir_test is not None:
                    # Parquet filenames:
                    # - article level: per_text_features_item_{numeric_id}.parquet
                    # - paragraph level: per_text_features_item_{unique_paragraph_id}.parquet
                    if isinstance(item_id, str):
                        parquet_id = item_id  # Use unique_paragraph_id directly
                    else:
                        parquet_id = str(item_id)  # Use article_id

                    l2_pt_path = pt_l2_dir_test / f"per_text_features_item_{parquet_id}.parquet"
                    if l2_pt_path.exists():
                        try:
                            l2_pt = pd.read_parquet(l2_pt_path, thrift_string_size_limit=2**31-1, thrift_container_size_limit=2**31-1)
                            # Filter by feature_set if provided
                            feature_prefix = None
                            if feature_set and feature_set in FIXED_TEXT_GROUP_PREFIXES:
                                feature_prefix = FIXED_TEXT_GROUP_PREFIXES[feature_set]

                            if feature_prefix:
                                is_pt_col = lambda c: c.startswith(feature_prefix)
                            else:
                                raise ValueError(f"Invalid feature_set {feature_set!r}, expected one of {list(FIXED_TEXT_GROUP_PREFIXES.keys())}")
                            l2_pt_cols = [c for c in l2_pt.columns if is_pt_col(c)]
                            union_cols = sorted(l2_pt_cols)
                            # Keep only this item's columns (same rule as above)
                            if isinstance(item_id, str):
                                item_feature_cols = [c for c in union_cols if f"{feature_prefix}{item_id}_" in c]
                            else:
                                item_feature_cols = [c for c in union_cols if f"_{item_id}_" in c]

                            if item_feature_cols:
                                merge_cols = [Fields.SUBJECT_ID, item_col]
                                l2_pt_for_merge = l2_pt.reindex(columns=merge_cols + item_feature_cols, fill_value=0)
                                p_features = p_features.reset_index(drop=True).merge(
                                    features_df_L2[p_mask][merge_cols].reset_index(drop=True)
                                        .merge(l2_pt_for_merge, on=merge_cols, how='left')[item_feature_cols],
                                    left_index=True, right_index=True
                                )
                        except Exception as e:
                            logger.warning(f"Could not add per-text features for test participant: {e}")

                if len(p_features) == 0:
                    continue

                p_features_scaled = zscore_preserve_nan_general(fit_subset, p_features)

                if single_feature:
                    score = p_features_scaled.iloc[0, 0]
                else:
                    score = eye_score_one_participant(p_features_scaled.iloc[0], prototype, metric=metric)

                per_item_scores.append({
                    Fields.SUBJECT_ID: participant_id,
                    item_col: item_id,
                    'eye_score': score,
                    Fields.L1: features_df_L2[p_mask][Fields.L1].iloc[0],
                })

                if participant_id not in eyescores_dict:
                    eyescores_dict[participant_id] = []
                eyescores_dict[participant_id].append(score)

        # Save per-item and aggregate
        if per_item_scores:
            pd.DataFrame(per_item_scores).to_csv(save_path.parent / f"{save_path.stem}_items.csv", index=False)

        eyescores = [{
            Fields.SUBJECT_ID: pid,
            'eye_score': np.nanmean(scores),
            Fields.L1: next(s[Fields.L1] for s in per_item_scores if s[Fields.SUBJECT_ID] == pid)
        } for pid, scores in eyescores_dict.items()]

        eye_scores_df = pd.DataFrame(eyescores).sort_values(by=Fields.SUBJECT_ID).reset_index(drop=True)
        eye_scores_df.to_csv(save_path, index=False)
        return eye_scores_df

    else:
        # Per-cg fit + per-cg L1 pool. The fit and pool definitions:
        #   seen   — both = same cg as the participant.
        #   unseen — both = participants whose article_batch differs from
        #            the participant's batch (parsed from content_group =
        #            "<batch>_<level>"). This excludes the same-batch
        #            sibling cg, which shares source text at a different
        #            difficulty and would otherwise leak text similarity.
        # The L2 row being scored is always sliced from features_df_L2_x
        # by cg; in unseen mode it is standardized against the
        # out-of-batch L2 distribution (intentional cross-population view).

        l1_groups = features_df_L1[Fields.CONTENT_GROUP].reset_index(drop=True)
        l2_groups = features_df_L2[Fields.CONTENT_GROUP].reset_index(drop=True)
        features_df_L1_x = features_df_L1_x.reset_index(drop=True)
        features_df_L2_x = features_df_L2_x.reset_index(drop=True)

        features_df_L2 = features_df_L2.reset_index(drop=True)
        features_df_L1 = features_df_L1.reset_index(drop=True)
        if content_mode == "unseen":
            l1_batch = l1_groups.astype(str).str.split("_").str[0]
            l2_batch = l2_groups.astype(str).str.split("_").str[0]

        per_item_scores = []
        eyescores = []
        content_groups = features_df_L2[Fields.CONTENT_GROUP].unique()
        for cg in tqdm(content_groups, desc="Processing content groups", leave=False):
            l2_in_cg = np.flatnonzero((l2_groups == cg).to_numpy())

            if content_mode == "seen" and fit_on_df is None:
                cg_fit_on_df = features_df_L2.iloc[l2_in_cg].copy()
            else:
                cg_fit_on_df = fit_on_df
            # Leave-one-out in seen mode: score each participant with fit/L1 excluding them
            if content_mode == "seen":
                participants_in_cg = features_df_L2.iloc[l2_in_cg][Fields.SUBJECT_ID].unique()

                for participant_id in tqdm(participants_in_cg, desc=f"CG {cg}: participants", leave=False):
                    p_mask = (features_df_L2[Fields.SUBJECT_ID] == participant_id) & (l2_groups == cg)
                    p_row_indices = np.flatnonzero(p_mask.to_numpy())

                    item_scores = []
                    for row_idx in tqdm(p_row_indices, desc=f"Items for {participant_id}", leave=False):
                        # Remove participant from fit/L1, but KEEP this item
                        item_value = features_df_L2.iloc[row_idx][item_col]

                        # Load per-item parquets if dirs provided (TRANSITIONS/WFC feature sets only, SEEN mode only)
                        pt_l1_dir = l1_parquet_dir or per_text_parquet_dir
                        pt_l2_dir = l2_parquet_dir or per_text_parquet_dir
                        pt_item_feature_cols = []
                        if pt_l1_dir is not None and pt_l2_dir is not None:
                            # Parquet filenames:
                            # - article level: per_text_features_item_{numeric_id}.parquet
                            # - paragraph level: per_text_features_item_{unique_paragraph_id}.parquet
                            if isinstance(item_value, str):
                                parquet_id = item_value  # Use unique_paragraph_id directly
                            else:
                                parquet_id = str(item_value)  # Use article_id

                            l1_pt_path = pt_l1_dir / f"per_text_features_item_{parquet_id}.parquet"
                            l2_pt_path = pt_l2_dir / f"per_text_features_item_{parquet_id}.parquet"
                            if l1_pt_path.exists() and l2_pt_path.exists():
                                try:
                                    l1_pt = pd.read_parquet(l1_pt_path, thrift_string_size_limit=2**31-1, thrift_container_size_limit=2**31-1)
                                    l2_pt = pd.read_parquet(l2_pt_path, thrift_string_size_limit=2**31-1, thrift_container_size_limit=2**31-1)

                                    prefix = FIXED_TEXT_GROUP_PREFIXES.get(feature_set, None)
                                    if prefix is None:
                                        raise ValueError(f"Invalid feature_set {feature_set!r}, expected one of {list(FIXED_TEXT_GROUP_PREFIXES.keys())}")
                                    is_pt_col = lambda c: c.startswith(prefix)
                                    l1_pt_cols = [c for c in l1_pt.columns if is_pt_col(c)]
                                    l2_pt_cols = [c for c in l2_pt.columns if is_pt_col(c)]
                                    union_cols = sorted(set(l1_pt_cols) | set(l2_pt_cols))
                                    # Keep only this item's columns (paragraph ids are
                                    # strings, article ids numbers matched as '_<id>_')
                                    if isinstance(item_value, str):
                                        # Paragraph level: filter columns that start with the feature prefix and item_value
                                        pt_item_feature_cols = [c for c in union_cols if f"{prefix}{item_value}_" in c]
                                    else:
                                        # Article level: filter columns with correct prefix containing the article_id (e.g. _5_)
                                        pt_item_feature_cols = [c for c in union_cols if f"_{item_value}_" in c]
                                except Exception as e:
                                    logger.warning(f"Could not load per-text features for item {item_value}: {e}")

                        fit_mask = (cg_fit_on_df[Fields.SUBJECT_ID] != participant_id) & (cg_fit_on_df[item_col] == item_value)
                        fit_indices = np.flatnonzero(fit_mask.to_numpy())
                        fit_full = cg_fit_on_df.iloc[fit_indices].copy()

                        l1_mask = (features_df_L1[Fields.SUBJECT_ID] != participant_id) & (l1_groups == cg) & (features_df_L1[item_col] == item_value)
                        l1_indices = np.flatnonzero(l1_mask.to_numpy())
                        l1_full = features_df_L1.iloc[l1_indices].copy()
                        p_full = features_df_L2.iloc[row_idx:row_idx+1].copy()

                        # Merge per-text features if loaded
                        if pt_item_feature_cols:
                            merge_cols = [Fields.SUBJECT_ID, item_col]
                            l1_pt_for_merge = l1_pt.reindex(columns=merge_cols + pt_item_feature_cols, fill_value=0)
                            l2_pt_for_merge = l2_pt.reindex(columns=merge_cols + pt_item_feature_cols, fill_value=0)
                            fit_full = fit_full.reset_index(drop=True).merge(
                                l2_pt_for_merge, on=merge_cols, how='left'
                            )
                            l1_full = l1_full.reset_index(drop=True).merge(
                                l1_pt_for_merge, on=merge_cols, how='left'
                            )
                            p_full = p_full.reset_index(drop=True).merge(
                                l2_pt_for_merge, on=merge_cols, how='left'
                            )

                        # Now drop metadata columns after merging parquets
                        fit_subset = _drop_metadata_cols(fit_full)
                        l1_subset = _drop_metadata_cols(l1_full)
                        p_features = _drop_metadata_cols(p_full)
                        if len(p_features) == 0:
                            logger.warning(f"No features for participant {participant_id} on item {item_value} in {cg}, skipping.")
                            continue

                        if len(fit_subset) == 0 or len(l1_subset) == 0:
                            logger.warning(f"No fit or L1 data for participant {participant_id} on item {item_value} in {cg}, skipping.")
                            continue

                        # Z-score and score using SAME item fit
                        l1_scaled = zscore_preserve_nan_general(fit_subset, l1_subset)
                        p_scaled = zscore_preserve_nan_general(fit_subset, p_features)
                        prototype = get_prototype_L1(l1_scaled)

                        if p_scaled.shape[0] == 0:
                            score = np.nan
                        elif single_feature:
                            score = p_scaled.iloc[0, 0]
                        elif p_scaled.shape[1] == 0:
                            score = np.nan
                        else:
                            score = eye_score_one_participant(p_scaled.iloc[0], prototype, metric=metric)

                        item_scores.append({
                            Fields.SUBJECT_ID: participant_id,
                            item_col: item_value,
                            'eye_score': score,
                            Fields.L1: features_df_L2.iloc[row_idx][Fields.L1],
                        })

                    per_item_scores.extend(item_scores)

                    # Aggregate across items
                    if len(item_scores) > 0:
                        final_score = np.nanmean([item['eye_score'] for item in item_scores])
                        participant_l1 = item_scores[0][Fields.L1]  # Get L1 from first valid item
                    else:
                        final_score = np.nan
                        # Get L1 from features_df_L2 if available
                        p_indices = np.flatnonzero(p_mask.to_numpy())
                        if len(p_indices) > 0:
                            participant_l1 = features_df_L2.iloc[p_indices[0]][Fields.L1]
                        else:
                            participant_l1 = np.nan
                        logger.warning(f"No valid scores for {participant_id} in {cg}")

                    eyescores.append({
                        Fields.SUBJECT_ID: participant_id,
                        'eye_score': final_score,  # Average across items for this participant
                        Fields.L1: participant_l1,
                    })

            else:  # content_mode == "unseen"
                own_batch = str(cg).split("_")[0]
                fit_idx = np.flatnonzero((l2_batch != own_batch).to_numpy())

                # Z-score all at once, but compute prototype per participant
                fit = features_df_L2_x.iloc[fit_idx]
                l2_block = zscore_preserve_nan_general(fit, features_df_L2_x.iloc[l2_in_cg])

                participants_in_cg = features_df_L2.iloc[l2_in_cg][Fields.SUBJECT_ID].unique()

                for participant_id in participants_in_cg:
                    p_mask = (features_df_L2[Fields.SUBJECT_ID] == participant_id) & (l2_groups == cg)
                    p_row_indices = np.flatnonzero(p_mask.to_numpy())

                    # Compute prototype per participant (excluding them from L1 pool)
                    l1_pool_mask = (features_df_L1[Fields.SUBJECT_ID] != participant_id) & (l1_batch != own_batch)
                    l1_pool_indices = np.flatnonzero(l1_pool_mask.to_numpy())
                    if len(l1_pool_indices) == 0:
                        logger.warning(f"No L1 pool available for {participant_id} in {cg}")
                        continue
                    l1_pool = features_df_L1_x.iloc[l1_pool_indices]
                    l1_pool_scaled = zscore_preserve_nan_general(fit, l1_pool)
                    prototype = get_prototype_L1(l1_pool_scaled)

                    item_scores = []
                    for row_idx in p_row_indices:
                        # Map row_idx to position in l2_block
                        idx_matches = np.where(l2_in_cg == row_idx)[0]
                        if len(idx_matches) == 0:
                            continue
                        idx_in_block = idx_matches[0]
                        item_value = features_df_L2.iloc[row_idx][item_col]

                        if single_feature and l2_block.shape[1] == 1:
                            score = l2_block.iloc[idx_in_block].iloc[0]
                        elif l2_block.shape[1] == 0:
                            score = np.nan
                        else:
                            score = eye_score_one_participant(l2_block.iloc[idx_in_block], prototype, metric=metric)

                        item_scores.append({
                            Fields.SUBJECT_ID: participant_id,
                            item_col: item_value,
                            'eye_score': score,
                            Fields.L1: features_df_L2.iloc[row_idx][Fields.L1],
                        })
                    per_item_scores.extend(item_scores)

                    # Aggregate across items
                    if len(item_scores) > 0:
                        final_score = np.nanmean([item['eye_score'] for item in item_scores])
                        participant_l1 = item_scores[0][Fields.L1]  # Get L1 from first item
                    else:
                        final_score = np.nan
                        # Get L1 from features_df_L2 if available
                        p_indices = np.flatnonzero(p_mask.to_numpy())
                        if len(p_indices) > 0:
                            participant_l1 = features_df_L2.iloc[p_indices[0]][Fields.L1]
                        else:
                            participant_l1 = np.nan
                        logger.warning(f"No valid scores for {participant_id} in {cg}")

                    eyescores.append({
                        Fields.SUBJECT_ID: participant_id,
                        'eye_score': final_score,  # Average across items for this participant
                        Fields.L1: participant_l1,
                    })

    eye_scores_df_items = pd.DataFrame(per_item_scores)
    save_path_items = save_path.parent / f"{save_path.stem}_items.csv"
    save_path_items.parent.mkdir(parents=True, exist_ok=True)
    eye_scores_df_items.to_csv(save_path_items, index=False)
    logger.info(f"Eye scores per item calculated (leave-one-out) for all L2 participants and saved to {save_path}")

    eye_scores_df = pd.DataFrame(eyescores)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    eye_scores_df.to_csv(save_path, index=False)
    logger.info(f"Eye scores saved to {save_path}")
    return eye_scores_df


# ── Bias-correction framework ──────────────────────────────────────────────
#
# Each correction is a Correction record: a name (used as the output filename
# suffix), a callable that does the math given a context dict, and an
# optional set of feature_set names to skip plus a set of context keys it
# requires. Adding a new adjustment method is:
#   1. Write a function _<name>_correction(ctx) that returns an adjusted
#      DataFrame, or None if it can't be applied.
#   2. Wrap it in a Correction instance, declaring requires={...} for any
#      context keys beyond the always-present ones.
#   3. Append it to DEFAULT_CORRECTIONS (or pass corrections=... to
#      run_one_feature_set).
#
# Context dict keys:
#   Always present (built by run_one_feature_set):
#     "feature_set"          — feature_set name being corrected
#     "eye_scores_df"        — raw eye_scores for this feature_set
#     "all_features_df_L2"   — full L2 features+metadata DataFrame
#   Optional (provided by run_one_dataset_version when it can):
#     "typological_distances"— {L1: distance_to_English}
#     "target_col"           — proficiency column for typology calibration


@dataclass(frozen=True)
class Correction:
    name: str
    apply: Callable[[dict], Optional[pd.DataFrame]]
    skip_for: frozenset = field(default_factory=frozenset)
    requires: frozenset = field(default_factory=frozenset)
    target_col: Optional[str] = None

    @property
    def suffix(self) -> str:
        return f"_{self.name}"


def apply_correction_to_eyescore(
    correction: Correction, context: dict,
) -> Optional[pd.DataFrame]:
    """Apply a Correction given a context dict.

    Returns None if the correction is opted out of for this feature_set
    (skip_for), if any required context key is missing, or if the correction
    function itself decides it cannot run.

    If the correction declares its own `target_col`, it overrides whatever the
    caller put in the context — lets us fit several debias variants (LexTALE,
    MichTest, ProfAgg) against the same eye_score CSV without re-plumbing the
    pipeline.
    """
    feature_set = context["feature_set"]
    if feature_set in correction.skip_for:
        return None
    missing = [k for k in correction.requires if context.get(k) is None]
    if missing:
        logger.warning(
            f"Skipping correction {correction.name!r} for feature_set "
            f"{feature_set!r}: missing context keys {missing}."
        )
        return None
    if correction.target_col is not None:
        context = {**context, "target_col": correction.target_col}
    return correction.apply(context)


# Typology-calibrated correction defaults. Change here if you want a different
# canonical proficiency target or distance metric for the calibration.
TYPO_CALIB_TARGET_COL = "lextale_score"
TYPO_CALIB_DISTANCE_TYPE = "syntactic+genetic"

# Stratified k-fold split for the inside_langs scheme. k is set per dataset
# to the number of L1 languages present (see _typology_inside_langs_correction);
# the seed is held fixed so re-runs stay reproducible.
TYPO_CALIB_INSIDE_RANDOM_STATE = 42


def _prepare_typology_frame(
    eye_scores_df: pd.DataFrame,
    target_df: pd.DataFrame,
    distances: dict,
    target_col: str,
) -> pd.DataFrame:
    """Attach target_col + typological distance to each participant, once.

    This is the fold-independent part of α fitting: the merge with the target,
    the `!= -1` filter, the dropna, and the distance `.map` depend only on each
    participant's own attributes, not on which fold it lands in. Computing it
    once (per feature set) and slicing per fold is numerically identical to the
    old per-fold recomputation, but avoids re-merging the whole frame k times.

    The distance column is kept even where NaN — matching the original, which
    fits the residual slope/intercept over all rows that have a target before
    dropping distance-less rows for the α fit.
    """
    merged = eye_scores_df.merge(target_df, on=Fields.SUBJECT_ID, how="inner")
    merged = merged[merged[target_col] != -1]
    merged = merged.dropna(subset=["eye_score", target_col, Fields.L1])
    merged = merged.copy()
    merged["distance"] = merged[Fields.L1].map(distances)
    return merged


def _fit_typology_alpha_prepared(
    prepared: pd.DataFrame,
    target_col: str,
) -> Optional[float]:
    """Fit α from an already-prepared TRAIN slice (see _prepare_typology_frame).

    `prepared` must carry eye_score, target_col, L1 and distance columns.
    """
    if len(prepared) < 3:
        return None
    slope, intercept = np.polyfit(prepared[target_col], prepared["eye_score"], 1)
    merged = prepared.copy()
    merged["residual"] = merged["eye_score"] - (slope * merged[target_col] + intercept)
    merged = merged.dropna(subset=["distance", "residual"])
    if len(merged) < 3 or merged[Fields.L1].nunique() < 2:
        return None
    alpha, _ = np.polyfit(merged["distance"], merged["residual"], 1)
    return float(alpha)


def _fit_typology_alpha(
    eye_scores_df: pd.DataFrame,
    target_df: pd.DataFrame,
    distances: dict,
    target_col: str,
) -> Optional[float]:
    """Fit α from a TRAIN slice of eye_scores: residualize against target_col,
    then fit slope α of per-participant residual vs typological distance.

    Per-participant (vs per-language-mean) weights each participant equally,
    so α is the same slope that's reported in the lang_bias evaluation plots.
    Returns None if there isn't enough data.

    Thin wrapper over prepare + fit; hot callers should call
    `_prepare_typology_frame` once and `_fit_typology_alpha_prepared` per fold.
    """
    prepared = _prepare_typology_frame(eye_scores_df, target_df, distances, target_col)
    return _fit_typology_alpha_prepared(prepared, target_col)


def _apply_typology_correction(
    test_df: pd.DataFrame, alpha: float, distances: dict,
    center: "float | None" = None,
) -> pd.DataFrame:
    """Subtract α · (distance(L1) − center) from each test participant's eye_score.
    Rows whose L1 has no distance map to NaN (correction can't be applied).

    `center` is the train fold's mean distance. It matters because α is refit per
    fold: the fit is `residual ≈ α·d + c` but only α is kept, and c ≈ −α·mean(d_train)
    is exactly this centring term. Dropping it leaves a per-fold constant α·mean(d_train)
    that differs between folds, so a distance scale sitting far from zero
    reintroduces between-language structure — measured: Spearman ρ=0.88 between
    |mean|/sd of the distance and surviving bias. With centring the correction is
    invariant to any affine rescaling of the distance, so a positive, bounded
    metric performs identically to a z-scored one.

    center=None reproduces the original (uncentred) behaviour; opt in with
    EYESCORE_CENTER_DISTANCE=1.
    """
    out = test_df.copy()
    distance_per_p = out[Fields.L1].map(distances)
    if center is not None:
        distance_per_p = distance_per_p - center
    out["eye_score"] = out["eye_score"] - alpha * distance_per_p
    return out


def _center_distances_enabled() -> bool:
    return os.environ.get("EYESCORE_CENTER_DISTANCE", "").lower() in ("1", "true", "yes")


def _train_distance_mean(fit_df, distances) -> "float | None":
    """Mean distance over the fit fold — the centring constant."""
    if fit_df is None or Fields.L1 not in fit_df.columns:
        return None
    d = fit_df[Fields.L1].map(distances).astype(float)
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else None


def _fit_and_apply_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_df: pd.DataFrame,
    distances: dict,
    target_col: str,
    score_fn: Optional[Callable[[list, list], Optional[pd.DataFrame]]] = None,
    fit_fn: Optional[Callable] = None,
    prepared_full: Optional[pd.DataFrame] = None,
) -> Optional[tuple]:
    """Fit the typology slope α on the train fold and apply it to the test fold.

    `fit_fn(fit_df, target_df, distances, target_col) -> float | None` estimates
    the per-unit-distance coefficient subtracted in the apply step. Defaults to
    the two-step `_fit_typology_alpha`; the single-regression distance variant
    passes `_fit_distance_mreg_coef` instead. Both share the same apply
    (subtract coef · distance), so only the coefficient estimation differs.

    Shared by both correction schemes — they differ only in how the fold is
    drawn (leave-one-language-out vs stratified k-fold), not in what happens
    once the split exists.

    Legacy path (`score_fn is None`): α is fit on the precomputed eye_scores in
    `train_df` and subtracted from the precomputed `test_df`. This is the
    behavior the standard (OneStop / fully_agg "all") pipeline keeps.

    Leak-free path (`score_fn` provided): eye_scores for BOTH folds are
    RECOMPUTED with the L1 prototype AND the z-score fit pool restricted to the
    train participants, so no test participant contaminates the scaling of any
    score used to fit α. `score_fn(pool_ids, target_ids)` returns a
    [participant_id, eye_score, L1] frame (or None); it is injected by the
    caller so dataset-specific scoring (MECO half-split vs OneStop
    content-group) lives outside this helper.

      train scores: score_fn(train_ids, train_ids) — leave-one-out within train,
                    feeds _fit_typology_alpha so α never sees test data.
      test  scores: score_fn(train_ids, test_ids)  — held-out participants scored
                    against the train pool, then α·distance is subtracted.

    Optimization (`prepared_full` on the legacy default-α path): the caller may
    pass a once-prepared frame (`_prepare_typology_frame` over all participants)
    so the fit slice is obtained by sub-selecting the train participants instead
    of re-merging the whole frame every fold. This is numerically identical to
    the per-fold prepare — every op in `_prepare_typology_frame` is row-wise, so
    preparing the full frame then slicing equals preparing the train slice — but
    skips k redundant merges. It applies ONLY when `score_fn is None` (scores are
    precomputed and fold-independent) and `fit_fn is None` (the two-step α fit);
    the leak-free and single-regression paths re-prepare from their own frames.

    Returns (alpha, adjusted_test_df) or None if the fold can't be scored/fit.
    The caller annotates the returned frame with scheme-specific columns
    (holdout_l1 / fold).
    """
    fit_prepared = None
    if score_fn is None:
        fit_df, apply_df = train_df, test_df
        if prepared_full is not None and fit_fn is None:
            train_ids = train_df[Fields.SUBJECT_ID]
            fit_prepared = prepared_full[
                prepared_full[Fields.SUBJECT_ID].isin(train_ids)
            ]
    else:
        train_ids = train_df[Fields.SUBJECT_ID].tolist()
        test_ids = test_df[Fields.SUBJECT_ID].tolist()
        fit_df = score_fn(train_ids, train_ids)
        apply_df = score_fn(train_ids, test_ids)
        if (
            fit_df is None or apply_df is None
            or fit_df.empty or apply_df.empty
        ):
            return None

    if fit_prepared is not None:
        alpha = _fit_typology_alpha_prepared(fit_prepared, target_col)
    else:
        if fit_fn is None:
            fit_fn = _fit_typology_alpha
        alpha = fit_fn(fit_df, target_df, distances, target_col)
    if alpha is None:
        return None
    center = None
    if _center_distances_enabled():
        center = _train_distance_mean(
            fit_prepared if fit_prepared is not None else fit_df, distances)
    return alpha, _apply_typology_correction(apply_df, alpha, distances, center=center)


def _typology_inside_langs_correction(ctx: dict, fit_fn: Optional[Callable] = None) -> Optional[pd.DataFrame]:
    """Stratified k-fold over participants, stratified by L1.

    For each fold:
      train = participants in the other k-1 folds → fit α via _fit_typology_alpha.
      test  = held-out fold → apply α · distance(L1).

    Returns one row per participant (each appears in exactly one test fold),
    annotated with the fold index and α used.
    """
    from sklearn.model_selection import StratifiedKFold

    eye_scores_df = ctx["eye_scores_df"]
    all_features_df_L2 = ctx["all_features_df_L2"]
    distances = ctx["typological_distances"]
    target_col = ctx.get("target_col") or TYPO_CALIB_TARGET_COL
    random_state = ctx.get("random_state")
    if random_state is None:
        random_state = TYPO_CALIB_INSIDE_RANDOM_STATE

    if target_col not in all_features_df_L2.columns:
        logger.warning(
            f"target_col {target_col!r} not in L2 features; "
            f"skipping inside_langs typology calibration."
        )
        return None

    target_df = all_features_df_L2[[Fields.SUBJECT_ID, target_col]].drop_duplicates(Fields.SUBJECT_ID)

    df = eye_scores_df.dropna(subset=[Fields.L1]).reset_index(drop=True)
    # k = number of L1 languages in this dataset (override via ctx['n_splits']
    # if a caller really needs to). StratifiedKFold then requires each L1 to
    # have ≥ k participants, so smaller groups are dropped below.
    n_splits = ctx.get("n_splits") or df[Fields.L1].nunique()
    if n_splits < 2:
        logger.warning(
            f"inside_langs: need ≥ 2 L1 languages, got {n_splits}; skipping."
        )
        return None
    counts = df.groupby(Fields.L1).size()
    too_small = counts[counts < n_splits]
    if not too_small.empty:
        logger.warning(
            f"inside_langs: dropping L1 groups smaller than n_splits={n_splits}: "
            f"{too_small.to_dict()}"
        )
        df = df[~df[Fields.L1].isin(too_small.index)].reset_index(drop=True)
    if len(df) < n_splits:
        logger.warning(
            f"inside_langs: only {len(df)} eligible participants; need ≥ {n_splits}."
        )
        return None

    score_fn = ctx.get("fold_scorer")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Prepare the target/distance-enriched frame once; the legacy two-step fold
    # just slices it (see _fit_and_apply_fold's prepared_full optimization).
    prepared_full = _prepare_typology_frame(df, target_df, distances, target_col)
    test_outputs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[Fields.L1])):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        result = _fit_and_apply_fold(
            train_df, test_df, target_df, distances, target_col, score_fn,
            fit_fn=fit_fn, prepared_full=prepared_full,
        )
        if result is None:
            logger.warning(f"inside_langs fold {fold_idx}: could not score/fit α; skipping.")
            continue
        alpha, adjusted = result
        adjusted["fold"] = fold_idx
        adjusted["alpha"] = alpha
        test_outputs.append(adjusted)

    if not test_outputs:
        return None
    return (
        pd.concat(test_outputs, ignore_index=True)
        .sort_values(Fields.SUBJECT_ID)
        .reset_index(drop=True)
    )


def _typology_across_langs_correction(ctx: dict, fit_fn: Optional[Callable] = None) -> Optional[pd.DataFrame]:
    """Leave-one-language-out over L1.

    For each L1 with a known typological distance:
      train = participants from the other languages → fit α.
      test  = held-out language's participants → apply α · distance(L1).

    Returns one row per participant in a held-out language, annotated with
    the held-out L1 and the α used in that fold.
    """
    eye_scores_df = ctx["eye_scores_df"]
    all_features_df_L2 = ctx["all_features_df_L2"]
    distances = ctx["typological_distances"]
    target_col = ctx.get("target_col") or TYPO_CALIB_TARGET_COL

    if target_col not in all_features_df_L2.columns:
        logger.warning(
            f"target_col {target_col!r} not in L2 features; "
            f"skipping across_langs typology calibration."
        )
        return None

    target_df = all_features_df_L2[[Fields.SUBJECT_ID, target_col]].drop_duplicates(Fields.SUBJECT_ID)

    df = eye_scores_df.dropna(subset=[Fields.L1])
    df = df[df[Fields.L1].map(distances).notna()]
    languages = sorted(df[Fields.L1].unique())
    # Need ≥ 3 languages so train (≥ 2) can support the per-language slope fit.
    if len(languages) < 3:
        logger.warning(
            f"across_langs: only {len(languages)} languages with distance; need ≥ 3."
        )
        return None

    score_fn = ctx.get("fold_scorer")
    # Prepare the target/distance-enriched frame once; the legacy two-step fold
    # just slices it (train = all other languages).
    prepared_full = _prepare_typology_frame(df, target_df, distances, target_col)
    test_outputs = []
    for lang in languages:
        train_df = df[df[Fields.L1] != lang]
        test_df = df[df[Fields.L1] == lang]
        if test_df.empty:
            continue
        result = _fit_and_apply_fold(
            train_df, test_df, target_df, distances, target_col, score_fn,
            fit_fn=fit_fn, prepared_full=prepared_full,
        )
        if result is None:
            logger.warning(f"across_langs holding out {lang!r}: could not score/fit α; skipping.")
            continue
        alpha, adjusted = result
        adjusted["holdout_l1"] = lang
        adjusted["alpha"] = alpha
        test_outputs.append(adjusted)

    if not test_outputs:
        return None
    return (
        pd.concat(test_outputs, ignore_index=True)
        .sort_values(Fields.SUBJECT_ID)
        .reset_index(drop=True)
    )


# ── Single-regression debias variants ───────────────────────────────────────
#
# Two alternatives to the two-step (residualize-on-proficiency, THEN slope-on-
# distance) typology correction. Each fits ONE regression:
#
#   distance_mreg — multiple regression  eye_score ~ target_col + distance(L1);
#                   subtract (distance coef)·distance(L1). Proficiency and
#                   distance are entered jointly, so the distance coefficient is
#                   the partial effect holding proficiency fixed (Frisch-Waugh-
#                   Lovell) — NOT equal to the two-step α unless proficiency and
#                   distance are uncorrelated. Supports inside_langs AND
#                   across_langs (distance extrapolates to a held-out language).
#
#   re_l1        — mixed model  eye_score ~ target_col + (1 | L1); subtract each
#                   language's fitted random intercept (BLUP). Uses NO typological
#                   distance — removes each L1's idiosyncratic (partial-pooled)
#                   offset after accounting for proficiency. inside_langs ONLY:
#                   a held-out language has no fitted intercept, so across_langs
#                   would apply a zero correction (degenerate) and is undefined.


def _fit_distance_mreg_coef(
    eye_scores_df: pd.DataFrame,
    target_df: pd.DataFrame,
    distances: dict,
    target_col: str,
) -> Optional[float]:
    """Single multiple regression: eye_score ~ target_col + distance(L1).

    Returns the fitted distance coefficient (subtracted per unit distance in the
    apply step), or None if there isn't enough data. Same call signature as
    `_fit_typology_alpha` so it drops straight into `_fit_and_apply_fold` via
    `fit_fn`, reusing the identical apply (subtract coef · distance).
    """
    merged = eye_scores_df.merge(target_df, on=Fields.SUBJECT_ID, how="inner")
    merged = merged[merged[target_col] != -1]
    merged = merged.dropna(subset=["eye_score", target_col, Fields.L1]).copy()
    merged["distance"] = merged[Fields.L1].map(distances)
    merged = merged.dropna(subset=["distance"])
    if len(merged) < 3 or merged[Fields.L1].nunique() < 2:
        return None
    X = np.column_stack([
        np.ones(len(merged)),
        merged[target_col].to_numpy(dtype=float),
        merged["distance"].to_numpy(dtype=float),
    ])
    y = merged["eye_score"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[2])  # coefficient on distance, holding target_col fixed


def _distance_mreg_inside_langs_correction(ctx: dict) -> Optional[pd.DataFrame]:
    """inside_langs CV; distance coefficient from the single multiple regression."""
    return _typology_inside_langs_correction(ctx, fit_fn=_fit_distance_mreg_coef)


def _distance_mreg_across_langs_correction(ctx: dict) -> Optional[pd.DataFrame]:
    """across_langs CV; distance coefficient from the single multiple regression."""
    return _typology_across_langs_correction(ctx, fit_fn=_fit_distance_mreg_coef)


def _fit_l1_effects(
    eye_scores_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str,
    model: str,
) -> Optional[dict]:
    """Per-L1 offsets from the panel  eye_score ~ target_col  grouped by L1,
    estimated with the `linearmodels` package. Returns {L1: mean-centered offset}
    or None if the data is too thin / the fit fails.

    model="random" → linearmodels RandomEffects: shrunk (partial-pooled) random
        intercepts via Swamy–Arora GLS variance components. Robust where
        statsmodels MixedLM goes singular; matches a mixed-effects fit.
    model="fixed"  → linearmodels PanelOLS with entity (L1) fixed effects: the
        full per-language offset, no shrinkage.

    Offsets are centered to sum to zero so the correction shifts no overall mean.
    """
    merged = eye_scores_df.merge(target_df, on=Fields.SUBJECT_ID, how="inner")
    merged = merged[merged[target_col] != -1]
    merged = merged.dropna(subset=["eye_score", target_col, Fields.L1]).copy()
    if len(merged) < 5 or merged[Fields.L1].nunique() < 2:
        return None
    try:
        from linearmodels.panel import RandomEffects, PanelOLS
        from statsmodels.tools import add_constant

        d = merged[[Fields.L1, "eye_score", target_col]].copy()
        d["_t"] = d.groupby(Fields.L1).cumcount()          # within-L1 index → panel "time"
        d = d.set_index([Fields.L1, "_t"])                 # (entity=L1, time)
        exog = add_constant(d[[target_col]])
        if model == "random":
            res = RandomEffects(d["eye_score"], exog).fit()
        else:
            res = PanelOLS(d["eye_score"], exog, entity_effects=True).fit()
        eff = res.estimated_effects
        eff.index = d.index
        per = {L1v: float(eff.xs(L1v, level=0).iloc[0, 0])
               for L1v in merged[Fields.L1].unique()}
    except Exception as e:
        logger.warning(f"linearmodels {model} L1-effects fit failed for {target_col!r}: {e}")
        return None
    if not per:
        return None
    mean = float(np.mean(list(per.values())))
    return {L1v: v - mean for L1v, v in per.items()}


def _fit_re_l1_intercepts(
    eye_scores_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str,
) -> Optional[dict]:
    """Random per-L1 intercepts of  eye_score ~ target_col + (1 | L1)  via
    linearmodels RandomEffects (shrunk / partial-pooled). {L1: intercept} or None."""
    return _fit_l1_effects(eye_scores_df, target_df, target_col, "random")


def _fit_fixed_l1_intercepts(
    eye_scores_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str,
) -> Optional[dict]:
    """Fixed per-L1 offsets of  eye_score ~ target_col  with L1 entity effects via
    linearmodels PanelOLS (entity_effects; no shrinkage). {L1: offset} or None."""
    return _fit_l1_effects(eye_scores_df, target_df, target_col, "fixed")


def _fixed_l1_inside_langs_correction(ctx: dict) -> Optional[pd.DataFrame]:
    """inside_langs CV with fixed-effect (unshrunk) per-L1 offsets."""
    return _re_l1_inside_langs_correction(ctx, fit_intercepts_fn=_fit_fixed_l1_intercepts)


def _apply_re_correction(test_df: pd.DataFrame, intercepts: dict) -> pd.DataFrame:
    """Subtract each participant's L1 random intercept from their eye_score.
    Rows whose L1 has no fitted intercept map to NaN (correction inapplicable)."""
    out = test_df.copy()
    out["eye_score"] = out["eye_score"] - out[Fields.L1].map(intercepts)
    return out


def _fit_and_apply_re_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str,
    score_fn: Optional[Callable[[list, list], Optional[pd.DataFrame]]] = None,
    fit_intercepts_fn: Optional[Callable] = None,
) -> Optional[tuple]:
    """RE analogue of `_fit_and_apply_fold`: fit per-L1 intercepts on the train
    fold, subtract the matching intercept from each test participant.
    `fit_intercepts_fn(fit_df, target_df, target_col) -> {L1: offset}` defaults to
    the shrunk random-intercept `_fit_re_l1_intercepts`; the fixed-effect variant
    passes `_fit_fixed_l1_intercepts` (no shrinkage). Returns
    (intercepts_dict, adjusted_test_df) or None."""
    if fit_intercepts_fn is None:
        fit_intercepts_fn = _fit_re_l1_intercepts
    if score_fn is None:
        fit_df, apply_df = train_df, test_df
    else:
        train_ids = train_df[Fields.SUBJECT_ID].tolist()
        test_ids = test_df[Fields.SUBJECT_ID].tolist()
        fit_df = score_fn(train_ids, train_ids)
        apply_df = score_fn(train_ids, test_ids)
        if (
            fit_df is None or apply_df is None
            or fit_df.empty or apply_df.empty
        ):
            return None
    intercepts = fit_intercepts_fn(fit_df, target_df, target_col)
    if not intercepts:
        return None
    return intercepts, _apply_re_correction(apply_df, intercepts)


def _re_l1_inside_langs_correction(ctx: dict, fit_intercepts_fn: Optional[Callable] = None) -> Optional[pd.DataFrame]:
    """Stratified k-fold over participants (stratified by L1). For each fold, fit
    the per-L1 intercepts on the train folds and subtract the matching intercept
    from each held-out participant. `fit_intercepts_fn` selects the estimator
    (shrunk random-intercept by default; fixed-effect for the fixed_l1 variant).

    inside_langs ONLY — every L1 must appear in training for its intercept to
    exist, so there is no across_langs counterpart.
    """
    from sklearn.model_selection import StratifiedKFold

    eye_scores_df = ctx["eye_scores_df"]
    all_features_df_L2 = ctx["all_features_df_L2"]
    target_col = ctx.get("target_col") or TYPO_CALIB_TARGET_COL
    random_state = ctx.get("random_state")
    if random_state is None:
        random_state = TYPO_CALIB_INSIDE_RANDOM_STATE

    if target_col not in all_features_df_L2.columns:
        logger.warning(
            f"target_col {target_col!r} not in L2 features; "
            f"skipping re_l1 inside_langs correction."
        )
        return None

    target_df = all_features_df_L2[[Fields.SUBJECT_ID, target_col]].drop_duplicates(Fields.SUBJECT_ID)

    df = eye_scores_df.dropna(subset=[Fields.L1]).reset_index(drop=True)
    n_splits = ctx.get("n_splits") or df[Fields.L1].nunique()
    if n_splits < 2:
        logger.warning(f"re_l1 inside_langs: need ≥ 2 L1 languages, got {n_splits}; skipping.")
        return None
    counts = df.groupby(Fields.L1).size()
    too_small = counts[counts < n_splits]
    if not too_small.empty:
        logger.warning(
            f"re_l1 inside_langs: dropping L1 groups smaller than n_splits={n_splits}: "
            f"{too_small.to_dict()}"
        )
        df = df[~df[Fields.L1].isin(too_small.index)].reset_index(drop=True)
    if len(df) < n_splits:
        logger.warning(f"re_l1 inside_langs: only {len(df)} eligible participants; need ≥ {n_splits}.")
        return None

    score_fn = ctx.get("fold_scorer")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_outputs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[Fields.L1])):
        result = _fit_and_apply_re_fold(
            df.iloc[train_idx], df.iloc[test_idx], target_df, target_col, score_fn,
            fit_intercepts_fn=fit_intercepts_fn,
        )
        if result is None:
            logger.warning(f"re_l1 inside_langs fold {fold_idx}: could not fit; skipping.")
            continue
        intercepts, adjusted = result
        adjusted = adjusted.copy()
        adjusted["fold"] = fold_idx
        adjusted["re_intercept"] = adjusted[Fields.L1].map(intercepts)
        test_outputs.append(adjusted)

    if not test_outputs:
        return None
    return (
        pd.concat(test_outputs, ignore_index=True)
        .sort_values(Fields.SUBJECT_ID)
        .reset_index(drop=True)
    )


# LexTALE-debiased corrections. Name kept un-suffixed since LexTALE was the
# original (and remains the canonical) debias target — renaming would break
# every downstream consumer that reads `typo_calibrated_inside_langs.csv` on
# disk. The {michtest,profagg} variants below carry an explicit target suffix.
TYPOLOGY_CALIBRATED_INSIDE_LANGS = Correction(
    name="typo_calibrated_inside_langs",
    apply=_typology_inside_langs_correction,
    requires=frozenset({"typological_distances"}),
)


TYPOLOGY_CALIBRATED_ACROSS_LANGS = Correction(
    name="typo_calibrated_across_langs",
    apply=_typology_across_langs_correction,
    requires=frozenset({"typological_distances"}),
)


# MichTest-debiased — α is fit against michtest_score instead of LexTALE.
# Applicable to OneStop only (Meco doesn't have a Michigan score column);
# _fit_typology_alpha logs a warning and skips when target_col is missing.
TYPOLOGY_CALIBRATED_INSIDE_LANGS_MICHTEST = Correction(
    name="typo_calibrated_inside_langs_michtest",
    apply=_typology_inside_langs_correction,
    requires=frozenset({"typological_distances"}),
    target_col=TestCols.MICHIGEN_TEST_COL,
)


TYPOLOGY_CALIBRATED_ACROSS_LANGS_MICHTEST = Correction(
    name="typo_calibrated_across_langs_michtest",
    apply=_typology_across_langs_correction,
    requires=frozenset({"typological_distances"}),
    target_col=TestCols.MICHIGEN_TEST_COL,
)


# ProfAgg-debiased — α is fit against the composite proficiency score.
# Applicable to Meco only (OneStop doesn't have proficiency_agg).
TYPOLOGY_CALIBRATED_INSIDE_LANGS_PROFAGG = Correction(
    name="typo_calibrated_inside_langs_profagg",
    apply=_typology_inside_langs_correction,
    requires=frozenset({"typological_distances"}),
    target_col=TestCols.PROFICIENCY_AGG_COL,
)


TYPOLOGY_CALIBRATED_ACROSS_LANGS_PROFAGG = Correction(
    name="typo_calibrated_across_langs_profagg",
    apply=_typology_across_langs_correction,
    requires=frozenset({"typological_distances"}),
    target_col=TestCols.PROFICIENCY_AGG_COL,
)


# Set of corrections applied by default during eye_score calculation. Append
# new Correction instances here to have them produced for every feature set
# and to extend ALL_VARIANTS automatically.
DEFAULT_CORRECTIONS = (
    TYPOLOGY_CALIBRATED_INSIDE_LANGS,
    TYPOLOGY_CALIBRATED_ACROSS_LANGS,
    TYPOLOGY_CALIBRATED_INSIDE_LANGS_MICHTEST,
    TYPOLOGY_CALIBRATED_ACROSS_LANGS_MICHTEST,
    TYPOLOGY_CALIBRATED_INSIDE_LANGS_PROFAGG,
    TYPOLOGY_CALIBRATED_ACROSS_LANGS_PROFAGG,
)


# ── Variant naming used by downstream consumers ──────────────────────────
#
# A "variant" is a string identifying which version of an eye_score CSV is
# being produced/read: "org" for raw eye_scores, or a Correction.name for any
# corrected variant. Filenames follow `{feature_set}{variant_suffix(variant)}.csv`.

RAW_VARIANT = "org"
ALL_VARIANTS = (RAW_VARIANT,) + tuple(c.name for c in DEFAULT_CORRECTIONS)


# Results-tree dataset name → the proficiency targets that dataset actually
# scores. Keyed by the short name the tree uses, not the DataSets member.
_DATASET_TARGET_COLS = {
    "OneStop": tuple(ALL_TARGET_COLS_SMALL[DataSets.ONESTOPL2]),
    "Meco": tuple(ALL_TARGET_COLS_SMALL[DataSets.MECOL2]),
}


def expected_correction_suffixes(
    dataset: str, corrections=DEFAULT_CORRECTIONS,
) -> tuple[str, ...]:
    """Filename suffixes of the corrections that should produce a CSV for `dataset`.

    A correction declaring a `target_col` the dataset has no scores for
    (MichTest on MECO, ProfAgg on OneStop) self-skips and writes nothing. It
    must therefore be left out of any "are all variants present?" test, or that
    test would be permanently false and recompute the cell on every run.

    An unknown dataset expects every correction: better to recompute a cell
    than to treat a genuinely missing variant as complete.
    """
    targets = _DATASET_TARGET_COLS.get(dataset)
    return tuple(
        c.suffix for c in corrections
        if c.target_col is None or targets is None or c.target_col in targets
    )


def corrections_complete(
    raw_path: Path, dataset: str, corrections=DEFAULT_CORRECTIONS,
) -> bool:
    """Is every correction this dataset expects already written beside `raw_path`?

    The runner skips a cell only when this holds, so a correction registered
    after a cell was computed re-runs exactly the cells that lack it.
    """
    return all(
        raw_path.with_name(raw_path.stem + suffix + raw_path.suffix).exists()
        for suffix in expected_correction_suffixes(dataset, corrections)
    )


def variant_suffix(variant: str) -> str:
    """Filename suffix for a variant. 'org' -> '', else '_<variant>'."""
    return "" if variant == RAW_VARIANT else f"_{variant}"


# Names of the two split schemes — referenced by downstream evaluation/plot
# code to organize the lang_bias subtree.
SPLIT_SCHEME_INSIDE = "inside_langs"
SPLIT_SCHEME_ACROSS = "across_langs"
SPLIT_SCHEMES = (SPLIT_SCHEME_INSIDE, SPLIT_SCHEME_ACROSS)
SPLIT_SCHEME_TO_VARIANT = {
    SPLIT_SCHEME_INSIDE: TYPOLOGY_CALIBRATED_INSIDE_LANGS.name,
    SPLIT_SCHEME_ACROSS: TYPOLOGY_CALIBRATED_ACROSS_LANGS.name,
}


# ── Debias-method registry ───────────────────────────────────────────────────
#
# The eyescore pipeline can run any of several debias METHODS. Each method is a
# tuple of Correction instances that REUSE the canonical typo_calibrated_* names
# (so every downstream consumer — evaluation, tables, plots, paper generators —
# that hardcodes those names works unchanged) but swaps the underlying math.
# The method's identity is carried by the results TREE it writes to
# (results_<method>/ via --results-suffix), not by the variant name.
#
# Methods:
#   two_step      — the canonical residualize-then-slope typology correction.
#   distance_mreg — single multiple regression eye ~ prof + distance.
#   interaction   — eye ~ prof * distance (centered); debias with main-effect b_dist.
#   re_l1         — random-intercept eye ~ prof + (1|L1). inside_langs ONLY (a new
#                   language has no fitted intercept), so it emits no across variant.


def _fit_interaction_coef(
    eye_scores_df: pd.DataFrame,
    target_df: pd.DataFrame,
    distances: dict,
    target_col: str,
) -> Optional[float]:
    """Centered interaction fit eye ~ target + distance + target:distance; return
    the main-effect distance coefficient (the debias slope). Same signature as
    _fit_typology_alpha so it drops into _fit_and_apply_fold via fit_fn."""
    merged = eye_scores_df.merge(target_df, on=Fields.SUBJECT_ID, how="inner")
    merged = merged[merged[target_col] != -1]
    merged = merged.dropna(subset=["eye_score", target_col, Fields.L1]).copy()
    merged["distance"] = merged[Fields.L1].map(distances)
    merged = merged.dropna(subset=["distance"])
    if len(merged) < 4 or merged[Fields.L1].nunique() < 2:
        return None
    p = merged[target_col].to_numpy(dtype=float); p = p - p.mean()
    d = merged["distance"].to_numpy(dtype=float); dc = d - d.mean()
    X = np.column_stack([np.ones(len(merged)), p, dc, p * dc])
    y = merged["eye_score"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[2])  # main-effect distance slope (at mean proficiency)


def _interaction_inside_langs_correction(ctx: dict) -> Optional[pd.DataFrame]:
    """inside_langs CV; distance coefficient from the centered interaction fit."""
    return _typology_inside_langs_correction(ctx, fit_fn=_fit_interaction_coef)


def _interaction_across_langs_correction(ctx: dict) -> Optional[pd.DataFrame]:
    """across_langs CV; distance coefficient from the centered interaction fit."""
    return _typology_across_langs_correction(ctx, fit_fn=_fit_interaction_coef)


_DIST_REQUIRES = frozenset({"typological_distances"})


def _canonical_debias_set(inside_apply, across_apply, requires) -> tuple:
    """Build a correction set that reuses the canonical typo_calibrated_* variant
    names (LexTALE default + michtest + profagg fan-out, inside/across) but with
    the given apply callables. across_apply=None omits the across variants
    (used by re_l1, which has no new-language correction)."""
    out = [Correction("typo_calibrated_inside_langs", inside_apply, requires=requires)]
    if across_apply is not None:
        out.append(Correction("typo_calibrated_across_langs", across_apply, requires=requires))
    out.append(Correction("typo_calibrated_inside_langs_michtest", inside_apply,
                          requires=requires, target_col=TestCols.MICHIGEN_TEST_COL))
    if across_apply is not None:
        out.append(Correction("typo_calibrated_across_langs_michtest", across_apply,
                              requires=requires, target_col=TestCols.MICHIGEN_TEST_COL))
    out.append(Correction("typo_calibrated_inside_langs_profagg", inside_apply,
                          requires=requires, target_col=TestCols.PROFICIENCY_AGG_COL))
    if across_apply is not None:
        out.append(Correction("typo_calibrated_across_langs_profagg", across_apply,
                              requires=requires, target_col=TestCols.PROFICIENCY_AGG_COL))
    return tuple(out)


# method name → correction set (canonical variant names, swapped math).
DEBIAS_METHODS = {
    "two_step": DEFAULT_CORRECTIONS,
    "distance_mreg": _canonical_debias_set(
        _distance_mreg_inside_langs_correction, _distance_mreg_across_langs_correction, _DIST_REQUIRES),
    "interaction": _canonical_debias_set(
        _interaction_inside_langs_correction, _interaction_across_langs_correction, _DIST_REQUIRES),
    "re_l1": _canonical_debias_set(
        _re_l1_inside_langs_correction, None, frozenset()),
    "fixed_l1": _canonical_debias_set(
        _fixed_l1_inside_langs_correction, None, frozenset()),
}
DEFAULT_DEBIAS_METHOD = "two_step"


def _drop_metadata_cols(features_df: pd.DataFrame) -> pd.DataFrame:
    """The feature columns only: drops the id, L1, content_group and item columns."""
    cols = [Fields.SUBJECT_ID, Fields.L1]
    if Fields.CONTENT_GROUP in features_df.columns:
        cols.append(Fields.CONTENT_GROUP)
    # Also drop item columns (only used for per_item_agg detection/filtering)
    if Fields.UNIQUE_PARAGRAPH_ID in features_df.columns:
        cols.append(Fields.UNIQUE_PARAGRAPH_ID)
    if Fields.ARTICLE_ID in features_df.columns:
        cols.append(Fields.ARTICLE_ID)
    return features_df.drop(columns=cols)


def _discriminative_fit_predict(X1: np.ndarray, X2_train: np.ndarray,
                                X_eval: np.ndarray, scorer: "Scorer") -> np.ndarray:
    """Fit the discriminative classifier once and score ``X_eval``.

    Design matrix = L1 rows (label 1) over the given L2 training rows (label 0);
    the standardizer (mean/std, NaN→0 impute) is fit on that design matrix only,
    then applied to ``X_eval``. Returns decision_function (log-odds of native).
    """
    Xtr = np.vstack([X1, X2_train])
    ytr = np.concatenate([np.ones(X1.shape[0]), np.zeros(X2_train.shape[0])])
    # Standardize on the training design matrix only (leakage guard); constant
    # columns get std=1 (→ 0 after centering); NaN → fit-mean (0).
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    ztr = np.nan_to_num((Xtr - mu) / sd, nan=0.0)
    zev = np.nan_to_num((X_eval - mu) / sd, nan=0.0)
    clf = scorer.estimator_factory(ztr.shape[1])
    clf.fit(ztr, ytr)
    return clf.decision_function(zev)


def _discriminative_oof(X1: np.ndarray, X2: np.ndarray, l2_lang: np.ndarray,
                        scorer: "Scorer", random_state: int = 42,
                        n_splits: int = 5) -> np.ndarray:
    """Out-of-fold log-odds for every row of ``X2``.

    Cross-fits over L2 folds (StratifiedKFold by L1 where possible, else KFold);
    each held-out row is scored by a classifier trained on L1 ∪ the other L2
    folds, so no row is in its own training set. L1 is in every training fold.
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    n2 = X2.shape[0]
    scores = np.full(n2, np.nan)
    counts = pd.Series(l2_lang).value_counts()
    k = int(min(n_splits, counts.min())) if len(counts) else 0
    if k >= 2:
        splits = StratifiedKFold(
            n_splits=k, shuffle=True, random_state=random_state
        ).split(X2, l2_lang)
    else:
        kk = max(2, min(n_splits, n2))
        splits = KFold(
            n_splits=kk, shuffle=True, random_state=random_state
        ).split(X2)

    for tr_idx, te_idx in splits:
        scores[te_idx] = _discriminative_fit_predict(X1, X2[tr_idx], X2[te_idx], scorer)
    return scores


def _participant_folds(ids: np.ndarray, langs: np.ndarray, random_state: int,
                       n_splits: int):
    """Yield (held-out participant id array) per fold, stratified by L1 where the
    group sizes allow it, else unstratified. Folds are over PARTICIPANTS (not
    rows) so a reader with several rows is held out as a unit."""
    from sklearn.model_selection import StratifiedKFold, KFold

    uniq = pd.unique(ids)
    lang_by_p = (
        pd.Series(langs, index=ids).groupby(level=0).first().reindex(uniq).to_numpy()
    )
    counts = pd.Series(lang_by_p).value_counts()
    k = int(min(n_splits, counts.min())) if len(counts) else 0
    dummy = np.zeros((len(uniq), 1))
    if k >= 2:
        splits = StratifiedKFold(
            n_splits=k, shuffle=True, random_state=random_state
        ).split(dummy, lang_by_p)
    else:
        kk = max(2, min(n_splits, len(uniq)))
        if len(uniq) < 2:
            return
        splits = KFold(
            n_splits=kk, shuffle=True, random_state=random_state
        ).split(dummy)
    for _, te in splits:
        yield uniq[te]


def _discriminative_oof_paired(X1: np.ndarray,
                               X2_fit: np.ndarray, fit_ids: np.ndarray,
                               X2_tgt: np.ndarray, tgt_ids: np.ndarray,
                               tgt_langs: np.ndarray, scorer: "Scorer",
                               random_state: int = 42, n_splits: int = 5) -> np.ndarray:
    """Score ``X2_tgt`` with the classifier's L2 negatives drawn from ``X2_fit``,
    holding every participant out of their own training set.

    Covers the three regimes the pipeline needs, uniformly:
      • fit rows ARE the target rows (content_mode "seen", or "all") →
        ordinary out-of-fold cross-fitting;
      • fit rows are DIFFERENT rows of the SAME participants (MECO odd/even
        half-split, ``fit_on_df``) → leave-participant-out: a reader's
        target-half row is scored by a model that never saw their fit-half row;
      • fit participants are DISJOINT from the targets (OneStop "unseen", where
        the fit pool is the other article batches) → no overlap, so a single fit
        on the whole pool is already leak-free.

    L1 (``X1``) is the positive class in every fit. Returns an array aligned to
    ``X2_tgt`` rows (NaN where a fold could not be scored).
    """
    n_t = X2_tgt.shape[0]
    scores = np.full(n_t, np.nan)
    if X2_fit.shape[0] == 0 or n_t == 0 or X1.shape[0] == 0:
        return scores

    fit_ids = np.asarray(fit_ids)
    tgt_ids = np.asarray(tgt_ids)
    if np.intersect1d(fit_ids, tgt_ids).size == 0:
        # Disjoint populations — the targets are already out of the training set.
        return _discriminative_fit_predict(X1, X2_fit, X2_tgt, scorer)

    for held in _participant_folds(tgt_ids, tgt_langs, random_state, n_splits):
        train_mask = ~np.isin(fit_ids, held)
        tgt_mask = np.isin(tgt_ids, held)
        if train_mask.sum() < 1 or tgt_mask.sum() < 1:
            continue
        scores[tgt_mask] = _discriminative_fit_predict(
            X1, X2_fit[train_mask], X2_tgt[tgt_mask], scorer
        )
    return scores


def _eye_score_discriminative_per_cg(
    features_df_L1_x: pd.DataFrame,
    features_df_L2_x: pd.DataFrame,
    features_df_L1: pd.DataFrame,
    features_df_L2: pd.DataFrame,
    scorer: "Scorer",
    content_mode: str,
    random_state: int = 42,
) -> pd.Series:
    """Discriminative scoring for content_mode "seen"/"unseen".

    Mirrors the prototype per-content-group pools exactly:
      seen   — positives = L1 in the same cg; L2 negatives = L2 in the same cg
               (so the targets overlap the pool → out-of-fold within the cg).
      unseen — positives = L1 whose article_batch differs from the cg's batch;
               L2 negatives = L2 in those other batches (disjoint from the
               targets → a single leak-free fit). This excludes the same-batch
               sibling cg, which shares source text at another difficulty.
    """
    l1_groups = features_df_L1[Fields.CONTENT_GROUP].reset_index(drop=True)
    l2_groups = features_df_L2[Fields.CONTENT_GROUP].reset_index(drop=True)
    L1x = features_df_L1_x.reset_index(drop=True)
    L2x = features_df_L2_x.reset_index(drop=True)

    cols = list(L2x.columns)
    X1_all = L1x.reindex(columns=cols).to_numpy(dtype=float)
    X2_all = L2x.to_numpy(dtype=float)
    l2_ids = features_df_L2[Fields.SUBJECT_ID].to_numpy()
    l2_langs = features_df_L2[Fields.L1].to_numpy()
    scores = np.full(len(L2x), np.nan)

    if content_mode == "unseen":
        l1_batch = l1_groups.astype(str).str.split("_").str[0]
        l2_batch = l2_groups.astype(str).str.split("_").str[0]

    for cg in features_df_L2[Fields.CONTENT_GROUP].unique():
        l2_in_cg = np.flatnonzero((l2_groups == cg).to_numpy())
        if content_mode == "seen":
            pool_idx = np.flatnonzero((l1_groups == cg).to_numpy())
            fit_idx = l2_in_cg
        else:
            own_batch = str(cg).split("_")[0]
            pool_idx = np.flatnonzero((l1_batch != own_batch).to_numpy())
            fit_idx = np.flatnonzero((l2_batch != own_batch).to_numpy())

        if len(pool_idx) == 0 or len(l2_in_cg) == 0 or len(fit_idx) == 0:
            logger.warning(
                f"No L1/L2/fit rows available for {content_mode!r} discriminative "
                f"scoring against content_group={cg!r}; eye_score will be NaN."
            )
            continue

        scores[l2_in_cg] = _discriminative_oof_paired(
            X1_all[pool_idx], X2_all[fit_idx], l2_ids[fit_idx],
            X2_all[l2_in_cg], l2_ids[l2_in_cg], l2_langs[l2_in_cg],
            scorer, random_state,
        )

    return pd.Series(scores, index=features_df_L2_x.index)


def _eye_score_discriminative(
    features_df_L1_x: pd.DataFrame,
    features_df_L2_x: pd.DataFrame,
    l1_meta: pd.DataFrame,
    l2_meta: pd.DataFrame,
    scorer: "Scorer",
    random_state: int = 42,
    n_splits: int = 5,
) -> pd.Series:
    """Out-of-fold discriminative eye_score: the log-odds of being native.

    Fits ``scorer.estimator_factory()`` to separate L1 (native, label 1) from L2
    (label 0) on the standardized features, and scores each L2 reader by the
    ``decision_function`` of a model that did NOT train on them (cross-fit over
    L2 folds; L1 is in every training fold as the positive class). Higher = more
    native-like — the same orientation as cosine/euclidean.

    Leakage discipline: each held-out L2 fold is excluded from its own training
    set; the standardizer is fit on the training design matrix only; L1 (the
    reference population, disjoint from L2) stays in every train fold. Folds are
    StratifiedKFold by L1 (k=min(n_splits, smallest L1 group)), falling back to
    unstratified KFold if a singleton L1 group makes stratification impossible,
    so every L2 still receives an out-of-fold score.

    Returns a Series positionally aligned to ``features_df_L2_x``.
    """
    cols = list(features_df_L2_x.columns)
    X1 = features_df_L1_x.reindex(columns=cols).to_numpy(dtype=float)
    X2 = features_df_L2_x.to_numpy(dtype=float)
    scores = _discriminative_oof(
        X1, X2, l2_meta[Fields.L1].to_numpy(), scorer, random_state, n_splits
    )
    return pd.Series(scores, index=features_df_L2_x.index)


def _make_discriminative_fold_scorer(
    all_features_df_L1: pd.DataFrame,
    all_features_df_L2: pd.DataFrame,
    feature_set: str,
    feature_cols: list,
    scorer: "Scorer",
    content_mode: str = "all",
    random_state: int = 42,
):
    """Build ``score_fn(pool_ids, target_ids)`` for the leak-free discriminative
    debias recompute (nested CV).

    The typology correction fits its slope α per outer fold, calling this twice:
      • ``score_fn(train_ids, train_ids)`` — scores the α-train participants, and
        must be OUT-OF-FOLD *within the train pool* (so α is not fit on
        in-sample-overfit scores);
      • ``score_fn(train_ids, test_ids)`` — scores the held-out α-test
        participants against a classifier trained on the FULL train pool.

    In BOTH cases the classifier's L2 negative class is drawn ONLY from
    ``pool_ids`` — the α-test participants never enter any training set, closing
    the second-order leak the precomputed-scores path would otherwise carry. L1
    is the fixed native reference (never fold-filtered), as in the prototype
    fold scorer. ``_discriminative_oof_paired`` picks the right regime per
    direction (cross-fit when pool and targets overlap, single fit when disjoint).

    ``content_mode`` selects the same direction structure the prototype fold
    scorer uses: one global direction for "all", one per content_group for
    "seen"/"unseen" (averaged per participant). Returns a
    [participant_id, eye_score, L1] frame, or None when nothing could be scored.
    """
    l2 = all_features_df_L2
    has_cg = Fields.CONTENT_GROUP in l2.columns

    def _feat(df):
        return _drop_metadata_cols(
            get_feature_set_df(df.copy(), feature_set, list(feature_cols))
        )

    def score_fn(pool_ids, target_ids):
        pool_set, target_set = set(pool_ids), set(target_ids)
        if not target_set:
            return None

        if content_mode == "all" or not has_cg:
            # (L1 prototype pool, L2 fit candidates, target candidates)
            directions = [(all_features_df_L1, l2, l2)]
        else:
            l1_cg = all_features_df_L1[Fields.CONTENT_GROUP].astype(str)
            l2_cg = l2[Fields.CONTENT_GROUP].astype(str)
            directions = []
            for cg in l2[Fields.CONTENT_GROUP].dropna().unique():
                cg = str(cg)
                if content_mode == "seen":
                    l1_pool = all_features_df_L1[l1_cg == cg]
                    fit_cand = l2[l2_cg == cg]
                else:  # unseen: pool/fit from the other article batches
                    own = cg.split("_")[0]
                    l1_pool = all_features_df_L1[l1_cg.str.split("_").str[0] != own]
                    fit_cand = l2[l2_cg.str.split("_").str[0] != own]
                directions.append((l1_pool, fit_cand, l2[l2_cg == cg]))

        per_direction = []
        for l1_pool_df, fit_cand_df, tgt_cand_df in directions:
            fit_df = fit_cand_df[fit_cand_df[Fields.SUBJECT_ID].isin(pool_set)]
            tgt_df = tgt_cand_df[tgt_cand_df[Fields.SUBJECT_ID].isin(target_set)]
            if l1_pool_df.empty or fit_df.empty or tgt_df.empty:
                continue
            fit_feat = _feat(fit_df)
            cols = list(fit_feat.columns)
            scores = _discriminative_oof_paired(
                _feat(l1_pool_df).reindex(columns=cols).to_numpy(dtype=float),
                fit_feat.to_numpy(dtype=float),
                fit_df[Fields.SUBJECT_ID].to_numpy(),
                _feat(tgt_df).reindex(columns=cols).to_numpy(dtype=float),
                tgt_df[Fields.SUBJECT_ID].to_numpy(),
                tgt_df[Fields.L1].to_numpy(),
                scorer, random_state,
            )
            if not np.isfinite(scores).any():
                # Direction too thin to cross-fit (e.g. a single participant in
                # the pool) — drop it rather than propagate an all-NaN frame.
                continue
            per_direction.append(pd.DataFrame({
                Fields.SUBJECT_ID: tgt_df[Fields.SUBJECT_ID].to_numpy(),
                "eye_score": scores,
                Fields.L1: tgt_df[Fields.L1].to_numpy(),
            }))

        return _average_eye_scores(per_direction)

    return score_fn


def eye_score(features_df_L1, features_df_L2, save_path: Path,
              content_mode: Literal["all", "seen", "unseen"] = "all",
              fit_on_df: pd.DataFrame | None = None,
              replace_existing: bool = False,
              per_text_parquet_dir: Path = None,
              l1_parquet_dir: Path = None,
              l2_parquet_dir: Path = None,
              feature_set: str = None,
              scorer: str = DEFAULT_SCORER) -> pd.DataFrame:
    """Calculate eye scores for each participant in the L2 dataset based on their features and the prototype L1 features.

    Args:
        features_df_L1: DataFrame containing features of L1 participants.
        features_df_L2: DataFrame containing features of L2 participants.
        save_path: Path where the eye scores DataFrame will be saved.
        scorer: Which scorer from SCORERS to use. Prototype scorers ("cosine"
            default, "euclidean") swap the metric used against the L1 prototype;
            discriminative scorers ("logistic", "lda") produce out-of-fold
            classifier log-odds instead (see _eye_score_discriminative).
        content_mode: Which L1 pool / z-score fit data to use.
            "all"    — every L1 participant. Z-score fit on `fit_on_df` if
                       provided, else on `features_df_L2`.
            "seen"   — per L2 participant in content_group X: fit z-score on
                       L2-in-cg-X, compare against the L1-in-cg-X prototype.
            "unseen" — per L2 participant in content_group X (format
                       "<batch>_<level>"): fit z-score on L2 whose
                       article_batch ≠ X's batch, compare against the
                       L1-in-other-batches prototype. Drops the participant's
                       own cg AND the same-batch sibling cg (which shares
                       the same source text at a different difficulty), so
                       the comparison is across articles, not difficulty.
        fit_on_df: Optional explicit z-score fit data. Only valid in "all"
            mode — lets the MECO pipeline pass the L1-matching half as the
            fit source in unseen directions. When provided, enables
            leave-one-out scoring: each participant is scored using fit data
            with that participant removed.
        replace_existing: Passed through to the per-item path (unused there).
        per_text_parquet_dir, l1_parquet_dir, l2_parquet_dir, feature_set:
            Per-item data only — where the per-item per-text parquets live and
            which per-text feature set to merge (see
            eye_score_per_item_independent).

    Returns:
        DataFrame containing participant IDs, eye scores, and L1 information.
    """
    # Resolve the scorer once. Prototype scorers carry a metric threaded into the
    # cosine/euclidean helpers; discriminative scorers have metric=None and are
    # dispatched to their own out-of-fold path.
    metric = SCORERS[scorer].metric

    # Detect per-item data and dispatch to per-item independent calculation
    item_col = Fields.UNIQUE_PARAGRAPH_ID if Fields.UNIQUE_PARAGRAPH_ID in features_df_L2.columns else None
    if item_col is None:
        item_col = Fields.ARTICLE_ID if Fields.ARTICLE_ID in features_df_L2.columns else None

    if item_col is not None:
        rows_per_participant = features_df_L2.groupby(Fields.SUBJECT_ID).size()
        is_per_item = (rows_per_participant > 1).any()
        if is_per_item:
            if SCORERS[scorer].family == "discriminative":
                raise NotImplementedError(
                    f"Discriminative scorer {scorer!r} is not supported on per-item data."
                )
            logger.info("Per-item data detected. Using per-item independent scoring.")
            return eye_score_per_item_independent(features_df_L1, features_df_L2, save_path, replace_existing, per_text_parquet_dir, content_mode, fit_on_df, item_col, l1_parquet_dir=l1_parquet_dir, l2_parquet_dir=l2_parquet_dir, feature_set=feature_set, metric=metric)

    if content_mode != "all" and Fields.CONTENT_GROUP not in features_df_L1.columns:
        raise ValueError(
            f"content_mode={content_mode!r} requires '{Fields.CONTENT_GROUP}' column; "
            "re-run feature extraction to add it."
        )
    if fit_on_df is not None and content_mode != "all":
        raise ValueError(
            f"fit_on_df only applies to content_mode='all'; got {content_mode!r}."
        )

    features_df_L1_x = _drop_metadata_cols(features_df_L1)
    features_df_L2_x = _drop_metadata_cols(features_df_L2)

    single_feature = len(features_df_L2_x.columns.tolist()) == 1
    if single_feature:
        logger.warning(
            "Only one feature available for eye score calculation. Eye scores will be directly that feature."
            f"{features_df_L2_x.columns.tolist()[0]}"
        )

    # Discriminative scorers produce their base score out-of-fold (a classifier
    # trained on L1-vs-L2 identity), then reuse the shared frame build/save.
    # Per-item data was rejected above.
    if SCORERS[scorer].family == "discriminative":
        sc = SCORERS[scorer]
        if content_mode != "all":
            # OneStop content-group directions (seen / unseen).
            eye_scores_L2 = _eye_score_discriminative_per_cg(
                features_df_L1_x, features_df_L2_x, features_df_L1, features_df_L2,
                sc, content_mode,
            )
        elif fit_on_df is not None:
            # MECO half-split direction emulation: the L2 negatives come from the
            # fit half (the SAME participants, other items), so hold each reader
            # out of their own training set.
            fit_clean = _drop_metadata_cols(fit_on_df)
            cols = list(features_df_L2_x.columns)
            eye_scores_L2 = pd.Series(
                _discriminative_oof_paired(
                    features_df_L1_x.reindex(columns=cols).to_numpy(dtype=float),
                    fit_clean.reindex(columns=cols).to_numpy(dtype=float),
                    fit_on_df[Fields.SUBJECT_ID].to_numpy(),
                    features_df_L2_x.to_numpy(dtype=float),
                    features_df_L2[Fields.SUBJECT_ID].to_numpy(),
                    features_df_L2[Fields.L1].to_numpy(),
                    sc,
                ),
                index=features_df_L2_x.index,
            )
        else:
            eye_scores_L2 = _eye_score_discriminative(
                features_df_L1_x, features_df_L2_x, features_df_L1, features_df_L2, sc,
            )
        eye_scores_df = pd.DataFrame({
            Fields.SUBJECT_ID: features_df_L2[Fields.SUBJECT_ID].values,
            "eye_score": eye_scores_L2.values,
            Fields.L1: features_df_L2[Fields.L1].values,
        })
        if 'window_index' in features_df_L2.columns:
            eye_scores_df['window_index'] = features_df_L2['window_index'].values
        eye_scores_df = eye_scores_df.sort_values(by=Fields.SUBJECT_ID).reset_index(drop=True)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        eye_scores_df.to_csv(save_path, index=False)
        logger.info(f"Discriminative ({scorer}) eye scores saved to {save_path}")
        return eye_scores_df

    # Leave-one-out path: train on fit_on_df minus each participant, score that participant
    if content_mode == "all" and fit_on_df is not None:
        fit_on_df_clean = _drop_metadata_cols(fit_on_df)
        features_df_L1_x_orig = features_df_L1_x.copy()
        features_df_L2_x_orig = features_df_L2_x.copy()

        # Fast leave-one-out: precompute every participant's LOO z-score
        # mean/std once (O(n·d)) instead of recomputing over the whole fit set
        # per participant (O(participants·n·d) — the seen/unseen hotspot).
        # On by default; EYESCORE_FAST_LOO=0 forces the reference path.
        _fast_loo = _ENABLE_FAST_LOO and os.environ.get("EYESCORE_FAST_LOO", "1").lower() in ("1", "true", "yes")
        _loo_stats = None
        if _fast_loo:
            # Precompute LOO stats once, plus numpy views of the L1 pool and L2
            # test frames aligned to the fit columns, so the per-participant
            # transform / prototype / cosine are all pure numpy (no per-row
            # pandas Series construction, which negated the win otherwise).
            _loo_cols, _loo_stats, _loo_full = _loo_fit_stats(
                fit_on_df_clean, fit_on_df[Fields.SUBJECT_ID]
            )
            _L1_mat = features_df_L1_x_orig.reindex(columns=_loo_cols).to_numpy(dtype=float)
            _L1_ids = features_df_L1[Fields.SUBJECT_ID].to_numpy()
            _L2_mat = features_df_L2_x_orig.reindex(columns=_loo_cols).to_numpy(dtype=float)
            _L2_ids = features_df_L2[Fields.SUBJECT_ID].to_numpy()

        eyescores = []
        for participant_id in features_df_L2[Fields.SUBJECT_ID].unique():
            p_mask = features_df_L2[Fields.SUBJECT_ID] == participant_id

            # A participant absent from the fit frame needs no downdate — the
            # reference path would z-score against the entire fit set — so the
            # full-set stats are exactly equivalent. This keeps every held-out
            # target of the leak-free fold scorer (fit pool and targets are
            # disjoint by construction) on the fast path.
            if _fast_loo and (participant_id in _loo_stats or _loo_full is not None):
                mean_arr, std_arr = _loo_stats.get(participant_id, _loo_full)
                keep = std_arr > 0
                if not keep.any():
                    score = float("nan")
                else:
                    m, s = mean_arr[keep], std_arr[keep]
                    l1_scaled = (_L1_mat[_L1_ids != participant_id][:, keep] - m) / s
                    with np.errstate(invalid="ignore"):
                        prototype = np.nanmean(l1_scaled, axis=0)
                    p_scaled = (_L2_mat[_L2_ids == participant_id][0, keep] - m) / s
                    if single_feature:
                        score = float(p_scaled[0]) if p_scaled.size else float("nan")
                    else:
                        score = eye_score_one_participant(p_scaled, prototype, metric=metric)
            else:
                # Reference path: recompute z-score stats over fit minus p.
                l1_subset = features_df_L1_x_orig[features_df_L1[Fields.SUBJECT_ID] != participant_id]
                p_features = features_df_L2_x_orig[p_mask]
                fit_subset = fit_on_df_clean[fit_on_df[Fields.SUBJECT_ID] != participant_id]
                l1_scaled = zscore_preserve_nan_general(fit_subset, l1_subset)
                p_features_scaled = zscore_preserve_nan_general(fit_subset, p_features)
                prototype = get_prototype_L1(l1_scaled)
                if single_feature:
                    score = p_features_scaled.iloc[0, 0]
                else:
                    score = eye_score_one_participant(p_features_scaled.iloc[0], prototype, metric=metric)

            eyescores.append({
                Fields.SUBJECT_ID: participant_id,
                'eye_score': score,
                Fields.L1: features_df_L2[p_mask][Fields.L1].iloc[0],
            })

        eye_scores_df = pd.DataFrame(eyescores)
        eye_scores_df = eye_scores_df.sort_values(by=Fields.SUBJECT_ID).reset_index(drop=True)

        # Handle window_index if present
        if 'window_index' in features_df_L2.columns:
            window_idx_df = features_df_L2[[Fields.SUBJECT_ID, 'window_index']].drop_duplicates(subset=[Fields.SUBJECT_ID])
            eye_scores_df = eye_scores_df.merge(window_idx_df, on=Fields.SUBJECT_ID, how='left')

        save_path.parent.mkdir(parents=True, exist_ok=True)
        eye_scores_df.to_csv(save_path, index=False)
        logger.info(f"Eye scores calculated (leave-one-out) for all L2 participants and saved to {save_path}")
        return eye_scores_df

    # Global mode: z-score all data together
    if content_mode == "all":
        fit_src = _drop_metadata_cols(fit_on_df) if fit_on_df is not None else features_df_L2_x
        logger.info("Z-scoring features (global)...")
        features_df_L1_x = zscore_preserve_nan_general(fit_src, features_df_L1_x)
        features_df_L2_x = zscore_preserve_nan_general(fit_src, features_df_L2_x)

    eye_scores_L2 = pd.Series(np.nan, index=range(len(features_df_L2_x)), dtype=float)

    if content_mode == "all":
        prototype = get_prototype_L1(features_df_L1_x)
        if single_feature:
            eye_scores_L2.iloc[:] = features_df_L2_x.iloc[:, 0].to_numpy()
        else:
            eye_scores_L2.iloc[:] = eye_score_batch(
                features_df_L2_x.to_numpy(), prototype.to_numpy(), metric=metric
            )
    else:
        # Per-cg fit + per-cg L1 pool. The fit and pool definitions:
        #   seen   — both = same cg as the participant.
        #   unseen — both = participants whose article_batch differs from
        #            the participant's batch (parsed from content_group =
        #            "<batch>_<level>"). This excludes the same-batch
        #            sibling cg, which shares source text at a different
        #            difficulty and would otherwise leak text similarity.
        # The L2 row being scored is always sliced from features_df_L2_x
        # by cg; in unseen mode it is standardized against the
        # out-of-batch L2 distribution (intentional cross-population view).
        l1_groups = features_df_L1[Fields.CONTENT_GROUP].reset_index(drop=True)
        l2_groups = features_df_L2[Fields.CONTENT_GROUP].reset_index(drop=True)
        features_df_L1_x = features_df_L1_x.reset_index(drop=True)
        features_df_L2_x = features_df_L2_x.reset_index(drop=True)

        if content_mode == "unseen":
            l1_batch = l1_groups.astype(str).str.split("_").str[0]
            l2_batch = l2_groups.astype(str).str.split("_").str[0]

        for cg in features_df_L2[Fields.CONTENT_GROUP].unique():
            l2_in_cg = np.flatnonzero((l2_groups == cg).to_numpy())

            if content_mode == "seen":
                pool_idx = np.flatnonzero((l1_groups == cg).to_numpy())
                fit_idx = l2_in_cg
            else:  # "unseen"
                own_batch = str(cg).split("_")[0]
                pool_idx = np.flatnonzero((l1_batch != own_batch).to_numpy())
                fit_idx = np.flatnonzero((l2_batch != own_batch).to_numpy())

            if len(pool_idx) == 0 or len(l2_in_cg) == 0 or len(fit_idx) == 0:
                logger.warning(
                    f"No L1/L2/fit rows available for {content_mode!r} comparison "
                    f"against content_group={cg!r}; eye_score will be NaN."
                )
                eye_scores_L2.iloc[l2_in_cg] = np.nan
                continue

            # Leave-one-out in seen mode: score each participant with fit/L1 excluding them
            if content_mode == "seen" and fit_on_df is not None:
                for j, i in enumerate(l2_in_cg):
                    participant_id = features_df_L2.iloc[i][Fields.SUBJECT_ID]

                    # Remove participant from fit and L1 pool for this CG
                    fit_mask = (fit_on_df[Fields.SUBJECT_ID] != participant_id)
                    l1_mask = (features_df_L1[Fields.SUBJECT_ID] != participant_id) & (l1_groups == cg)

                    fit_subset = _drop_metadata_cols(fit_on_df[fit_mask])
                    l1_subset = features_df_L1_x[l1_mask]
                    p_features = features_df_L2_x.iloc[i:i+1]

                    if len(fit_subset) == 0 or len(l1_subset) == 0:
                        eye_scores_L2.iloc[i] = np.nan
                        continue

                    # Z-score and score
                    l1_scaled = zscore_preserve_nan_general(fit_subset, l1_subset)
                    p_scaled = zscore_preserve_nan_general(fit_subset, p_features)
                    prototype = get_prototype_L1(l1_scaled)

                    if single_feature:
                        eye_scores_L2.iloc[i] = p_scaled.iloc[0, 0]
                    elif p_scaled.shape[1] == 0:
                        eye_scores_L2.iloc[i] = np.nan
                    else:
                        eye_scores_L2.iloc[i] = eye_score_one_participant(
                            p_scaled.iloc[0], prototype, metric=metric
                        )
            else:
                # Standard mode: z-score all at once, score each participant
                fit = features_df_L2_x.iloc[fit_idx]
                l1_block = zscore_preserve_nan_general(fit, features_df_L1_x.iloc[pool_idx])
                l2_block = zscore_preserve_nan_general(fit, features_df_L2_x.iloc[l2_in_cg])
                prototype = get_prototype_L1(l1_block)

                if single_feature and l2_block.shape[1] == 1:
                    eye_scores_L2.iloc[l2_in_cg] = l2_block.iloc[:, 0].to_numpy()
                elif l2_block.shape[1] == 0:
                    eye_scores_L2.iloc[l2_in_cg] = np.nan
                else:
                    eye_scores_L2.iloc[l2_in_cg] = eye_score_batch(
                        l2_block.to_numpy(), prototype.to_numpy(), metric=metric
                    )

    eye_scores_df = pd.DataFrame({
        Fields.SUBJECT_ID: features_df_L2[Fields.SUBJECT_ID].values,
        "eye_score": eye_scores_L2.values,
        Fields.L1: features_df_L2[Fields.L1].values,
    })
    if 'window_index' in features_df_L2.columns:
        eye_scores_df['window_index'] = features_df_L2['window_index'].values
    logger.info(f"Eye scores calculated for all L2 participants (mode={content_mode}).")
    eye_scores_df = eye_scores_df.sort_values(by=Fields.SUBJECT_ID).reset_index(drop=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    eye_scores_df.to_csv(save_path, index=False)
    logger.info(f"Eye scores saved to {save_path}")
    return eye_scores_df


def get_feature_set_df(all_features_df: pd.DataFrame, feature_set: str, feature_cols:list) -> pd.DataFrame:
    """Extract the specified feature set from the full features DataFrame.

    Args:
        all_features_df: DataFrame containing all features and participant IDs.
        feature_set: Name of the feature set to extract.
        feature_cols: The feature set's columns. Ignored for the per-text sets
            (TRANSITIONS / WFC), whose columns are found by prefix.

    Keeps the id, L1, content_group and item columns alongside the features;
    features missing from the frame are dropped with a warning.
    """
    logger.info(f"Extracting feature set: {feature_set}")
    logger.debug(f"DataFrame shape: {all_features_df.shape}, columns: {list(all_features_df.columns)[:10]}...")
    logger.debug(f"Requested feature_cols: {feature_cols}")
    fixed_text_prefix = FIXED_TEXT_GROUP_PREFIXES.get(feature_set)
    if fixed_text_prefix is not None:
        feature_cols = sorted(c for c in all_features_df.columns if c.startswith(fixed_text_prefix))
    missing = [f for f in feature_cols if f not in all_features_df.columns]
    if missing:
        logger.warning(
            f"Feature set {feature_set}: {len(missing)} features missing from DataFrame. "
            f"Missing features: {missing}. Available columns: {list(all_features_df.columns)[:30]}... "
            f"Proceeding with {len(feature_cols) - len(missing)}/{len(feature_cols)} features."
        )
        feature_cols = [f for f in feature_cols if f in all_features_df.columns]
    keep_cols = [Fields.SUBJECT_ID, Fields.L1]
    if Fields.CONTENT_GROUP in all_features_df.columns:
        keep_cols.append(Fields.CONTENT_GROUP)
    # Keep item columns for per_item_agg detection in eye_score()
    if Fields.UNIQUE_PARAGRAPH_ID in all_features_df.columns:
        keep_cols.append(Fields.UNIQUE_PARAGRAPH_ID)
    if Fields.ARTICLE_ID in all_features_df.columns:
        keep_cols.append(Fields.ARTICLE_ID)
    # Remove any duplicates while preserving order
    cols_to_select = list(dict.fromkeys(feature_cols + keep_cols))
    return all_features_df[cols_to_select]


# ── Leak-free per-fold rescoring helpers ────────────────────────────────────
#
# Shared by the bias-correction `fold_scorer`s. A "direction" is one atomic
# (prototype pool, scored targets, z-score fit) scoring event; datasets differ
# only in how they enumerate directions (MECO half-splits vs OneStop content
# groups) — see _make_meco_fold_scorer and _make_content_group_fold_scorer below.


def _average_eye_scores(per_dir_dfs: list) -> "pd.DataFrame | None":
    """Concatenate per-direction eye_scores frames and average per participant.

    Each input has columns [participant_id, eye_score, L1]. Output keeps one row
    per participant with eye_score = mean across directions, L1 = first.
    """
    valid = [d for d in per_dir_dfs if d is not None and not d.empty]
    if not valid:
        return None
    combined = pd.concat(valid, ignore_index=True)
    return (
        combined.groupby(Fields.SUBJECT_ID, as_index=False)
        .agg({"eye_score": "mean", Fields.L1: "first"})
    )


def _eye_score_one_direction(
    df_l1_pool: pd.DataFrame,
    df_l2_target: pd.DataFrame,
    df_l2_fit: "pd.DataFrame | None",
    feature_set_name: str,
    feature_cols: list,
    save_dir: Path,
    scorer: str = DEFAULT_SCORER,
) -> pd.DataFrame:
    """Run eye_score for a single scoring direction.

    The atomic scoring step shared by the reported pass and the leak-free
    bias-correction recompute. Callers control the three roles directly:

      df_l1_pool   : L1 rows that form the prototype (native readers; the
                     caller does NOT pool-filter these — see the scorer docs).
      df_l2_target : L2 rows to actually score (restricted to the targets).
      df_l2_fit    : L2 rows whose distribution defines the z-score fit, or
                     None to fit on df_l2_target itself. Passing a frame here
                     activates eye_score's leave-one-out path (a target that is
                     also in the fit is dropped from it).

    eye_score needs a save_path, so we write to a throwaway temp and discard it.
    """
    fs_l1 = get_feature_set_df(df_l1_pool.copy(), feature_set_name, list(feature_cols))
    fs_l2 = get_feature_set_df(df_l2_target.copy(), feature_set_name, list(feature_cols))
    fit_on_df = (
        get_feature_set_df(df_l2_fit.copy(), feature_set_name, list(feature_cols))
        if df_l2_fit is not None else None
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", dir=str(save_dir), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        return eye_score(
            fs_l1, fs_l2, tmp_path, content_mode="all", fit_on_df=fit_on_df,
            scorer=scorer,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _fold_cache_fingerprint(*frames_and_tags) -> str:
    """Content fingerprint for the fold-rescoring disk cache.

    MUST cover everything the rescored eye_scores depend on, because a stale
    hit would silently substitute scores computed from different data. Hashes
    the scalar tags plus, for each frame, its shape, column names, and a full
    row-wise content hash (pd.util.hash_pandas_object is vectorized and avoids
    materializing the frame as bytes). Any data rebuild changes the digest and
    misses the cache.
    """
    h = hashlib.sha256()
    for item in frames_and_tags:
        if isinstance(item, pd.DataFrame):
            h.update(str(item.shape).encode())
            h.update("\x00".join(map(str, item.columns)).encode())
            row_hashes = pd.util.hash_pandas_object(item, index=False).to_numpy()
            h.update(row_hashes.tobytes())
        else:
            h.update(f"{item}".encode())
        h.update(b"|")
    return h.hexdigest()


def _seen_unseen_feature_paths_eyescore(dataset: str, half: str) -> tuple:
    """Resolve (l1_csv, l2_csv) paths for a half-split feature set."""
    suffix = f"seen_unseen/all/all/{half}/features_and_metadata.csv"
    return (
        Path(DATA_PATH / f"{dataset}L1/features_and_targets/{suffix}"),
        Path(DATA_PATH / f"{dataset}L2/features_and_targets/{suffix}"),
    )


def _make_meco_fold_scorer(
    halves_data: dict,
    feature_set_name: str,
    feature_cols: list,
    is_per_text: bool,
    directions: list,
    save_dir: Path,
    scorer: str = DEFAULT_SCORER,
):
    """Build the ``score_fn(pool_ids, target_ids)`` that _fit_and_apply_fold needs.

    Returns the two-direction-averaged eye_score for ``target_ids``, with the L2
    z-score fit pool restricted to ``pool_ids`` (the bias correction's train
    fold). This is the leak-free recompute: the scaling of every score a fold
    consumes is estimated only from that fold's training participants.

    NOTE the L1 prototype is NOT pool-filtered. In MECO (as in OneStop) the L1
    pool is native-English readers — a separate population from the L2 learners
    the folds are drawn over (disjoint participant ids), so no L1 reader is ever
    a fold participant and the prototype is a fixed reference. The scaling leak
    lives entirely in the L2 fit, which is what we restrict.

    The fit pool is supplied for EVERY direction (seen included), so eye_score
    always takes its leave-one-out path: a target that is also in the train fit
    pool is dropped from it (LOO within train), while a held-out target stays
    out and is scored against the full train fit. Returns None when the
    pool/target is too thin to score.
    """
    l1_key = "per_text_l1" if is_per_text else "l1"
    l2_key = "per_text_l2" if is_per_text else "l2"

    # Same memoization as _make_content_group_fold_scorer: score_fn depends only
    # on (pool_ids, target_ids) — never on the correction's target, the debias
    # method, or the typological distance — so every correction in a job, and
    # every combo in a debias fan-out, asks for identical rescored folds.
    # In-memory dedupes within a job; EYESCORE_FOLD_CACHE=<dir> persists across
    # combos. The fingerprint covers the actual half frames, so rebuilt features
    # miss the cache rather than silently returning stale scores.
    _score_cache: dict = {}
    _disk_dir = os.environ.get("EYESCORE_FOLD_CACHE")
    _fp = None
    if _disk_dir:
        os.makedirs(_disk_dir, exist_ok=True)
        _frames = []
        for _h in sorted(halves_data):
            for _k in (l1_key, l2_key):
                _f = halves_data[_h].get(_k)
                if _f is not None:
                    _frames.append(_f)
        _fp = _fold_cache_fingerprint(
            *_frames, "meco_halves", feature_set_name,
            str(sorted(feature_cols or [])), is_per_text, str(directions), scorer,
        )

    def _disk_path(pool_set, target_set):
        h = hashlib.sha256()
        h.update(_fp.encode())
        h.update(b"|P|"); h.update(",".join(sorted(map(str, pool_set))).encode())
        h.update(b"|T|"); h.update(",".join(sorted(map(str, target_set))).encode())
        return Path(_disk_dir) / f"meco_{h.hexdigest()}.parquet"

    def score_fn(pool_ids, target_ids):
        pool_set, target_set = set(pool_ids), set(target_ids)
        cache_key = (frozenset(pool_set), frozenset(target_set))
        if cache_key in _score_cache:
            hit = _score_cache[cache_key]
            return None if hit is None else hit.copy()
        if _disk_dir:
            p = _disk_path(pool_set, target_set)
            if p.exists():
                try:
                    hit = pd.read_parquet(p)
                    _score_cache[cache_key] = hit
                    return hit.copy()
                except Exception as e:
                    logger.warning(f"meco fold cache read failed ({p.name}): {e}; recomputing")
        per_direction = []
        for l1_half, l2_half in directions:
            df_l1 = halves_data[l1_half].get(l1_key)          # native-English prototype pool (unfiltered)
            df_l2 = halves_data[l2_half].get(l2_key)
            df_l2_fit = halves_data[l1_half].get(l2_key)      # z-score fit = L1-half's L2
            if df_l1 is None or df_l2 is None or df_l2_fit is None:
                continue

            df_l2_fit_pool = df_l2_fit[df_l2_fit[Fields.SUBJECT_ID].isin(pool_set)]
            df_l2_tgt = df_l2[df_l2[Fields.SUBJECT_ID].isin(target_set)]
            if df_l1.empty or df_l2_fit_pool.empty or df_l2_tgt.empty:
                continue

            per_direction.append(
                _eye_score_one_direction(
                    df_l1, df_l2_tgt, df_l2_fit_pool,
                    feature_set_name, feature_cols, save_dir,
                    scorer=scorer,
                )
            )
        result = _average_eye_scores(per_direction)
        _score_cache[cache_key] = result
        if _disk_dir and result is not None:
            p = _disk_path(pool_set, target_set)
            try:
                tmp = p.with_suffix(f".tmp{os.getpid()}")
                result.to_parquet(tmp, index=False)
                os.replace(tmp, p)
            except Exception as e:
                logger.warning(f"meco fold cache write failed ({p.name}): {e}")
        return None if result is None else result.copy()

    return score_fn


def _make_content_group_fold_scorer(
    all_features_df_L1: pd.DataFrame,
    all_features_df_L2: pd.DataFrame,
    feature_set: str,
    feature_cols: list,
    content_mode: str,
    save_dir: Path,
    scorer: str = DEFAULT_SCORER,
):
    """Build the ``score_fn(pool_ids, target_ids)`` for the OneStop / global path.

    Emulates the content-group pooling that `eye_score` does internally for each
    `content_mode`, but with the L2 z-score fit restricted to ``pool_ids`` (the
    bias correction's train fold) — the leak-free recompute for the
    `run_one_feature_set` path (OneStop all/seen/unseen and MECO fully_agg "all").

    Directions (one atomic scoring event each), mirroring eye_score's blocks:
      all    — one global direction; prototype = all L1.
      seen   — one direction per content_group cg; prototype = L1 in cg.
      unseen — one direction per cg; prototype = L1 whose article_batch differs
               from cg's batch (batch = cg.split("_")[0]).

    In every mode the L1 prototype pool is NOT pool-filtered: L1 are native
    readers, a population disjoint from the L2 folds (verified: zero id overlap
    in both OneStop and MECO), so no L1 reader is ever a held-out participant.
    Only the L2 fit (and the scored targets) are filtered. eye_score's LOO then
    drops any train target from its own fit, and scores held-out targets against
    the full train fit. Returns None when a direction is too thin to score.
    """
    has_cg = Fields.CONTENT_GROUP in all_features_df_L2.columns

    # Per-fold rescoring is the dominant cost of the leak-free corrections, and
    # it is recomputed identically several times per job: score_fn depends ONLY
    # on (pool_ids, target_ids) — not on target_col, not on the debias method,
    # not on the typological distance, none of which it receives. The folds
    # themselves are target-independent (built from eye_scores_df, fixed
    # random_state), so every correction in `corrections` asks for the same
    # splits. Memoize on the id sets; the closure's frames/feature_set/scorer
    # are fixed for the life of the scorer, and eye_score is deterministic.
    _score_cache: dict = {}

    # Opt-in via EYESCORE_FOLD_CACHE=<dir>: persist rescored folds to
    # disk so sibling runs that differ only in debias method / distance / target
    # — i.e. every combo in a debias fan-out — reuse them instead of recomputing.
    # The fingerprint covers the actual frame contents, so a data rebuild misses.
    _disk_dir = os.environ.get("EYESCORE_FOLD_CACHE")
    _fp = None
    if _disk_dir:
        os.makedirs(_disk_dir, exist_ok=True)
        _fp = _fold_cache_fingerprint(
            all_features_df_L1, all_features_df_L2,
            feature_set, content_mode, scorer, str(sorted(feature_cols or [])),
        )

    def _disk_path(pool_set, target_set):
        h = hashlib.sha256()
        h.update(_fp.encode())
        h.update(b"|P|"); h.update(",".join(sorted(map(str, pool_set))).encode())
        h.update(b"|T|"); h.update(",".join(sorted(map(str, target_set))).encode())
        return Path(_disk_dir) / f"fold_{h.hexdigest()}.parquet"

    def score_fn(pool_ids, target_ids):
        pool_set, target_set = set(pool_ids), set(target_ids)
        cache_key = (frozenset(pool_set), frozenset(target_set))
        if cache_key in _score_cache:
            hit = _score_cache[cache_key]
            # Hand back a copy: callers pass the frame into fit/apply helpers,
            # and a shared object could be mutated across corrections.
            return None if hit is None else hit.copy()
        if _disk_dir:
            p = _disk_path(pool_set, target_set)
            if p.exists():
                try:
                    hit = pd.read_parquet(p)
                    _score_cache[cache_key] = hit
                    return hit.copy()
                except Exception as e:  # corrupt/partial file: recompute
                    logger.warning(f"fold cache read failed ({p.name}): {e}; recomputing")
        l1 = all_features_df_L1
        l2 = all_features_df_L2

        if content_mode == "all" or not has_cg:
            directions = [(l1, l2)]  # (prototype-pool rows, fit-candidate rows)
        else:
            l1_cg = l1[Fields.CONTENT_GROUP].astype(str)
            l2_cg = l2[Fields.CONTENT_GROUP].astype(str)
            directions = []
            for cg in l2[Fields.CONTENT_GROUP].dropna().unique():
                cg = str(cg)
                if content_mode == "seen":
                    l1_pool = l1[l1_cg == cg]
                    fit_cand = l2[l2_cg == cg]
                else:  # unseen: prototype/fit from other article batches
                    own = cg.split("_")[0]
                    l1_pool = l1[l1_cg.str.split("_").str[0] != own]
                    fit_cand = l2[l2_cg.str.split("_").str[0] != own]
                tgt = l2[(l2_cg == cg)]
                directions.append((l1_pool, fit_cand, tgt))

        per_direction = []
        for direction in directions:
            if content_mode == "all" or not has_cg:
                l1_pool, fit_cand = direction
                tgt = l2
            else:
                l1_pool, fit_cand, tgt = direction

            fit_pool = fit_cand[fit_cand[Fields.SUBJECT_ID].isin(pool_set)]
            tgt_sel = tgt[tgt[Fields.SUBJECT_ID].isin(target_set)]
            if l1_pool.empty or fit_pool.empty or tgt_sel.empty:
                continue
            per_direction.append(
                _eye_score_one_direction(
                    l1_pool, tgt_sel, fit_pool, feature_set, feature_cols, save_dir,
                    scorer=scorer,
                )
            )
        result = _average_eye_scores(per_direction)
        _score_cache[cache_key] = result
        if _disk_dir and result is not None:
            # Atomic publish: parallel workers can race on the same key, and a
            # half-written parquet would be read as a corrupt hit.
            p = _disk_path(pool_set, target_set)
            try:
                tmp = p.with_suffix(f".tmp{os.getpid()}")
                result.to_parquet(tmp, index=False)
                os.replace(tmp, p)
            except Exception as e:
                logger.warning(f"fold cache write failed ({p.name}): {e}")
        return None if result is None else result.copy()

    return score_fn


def run_one_feature_set(all_features_df_L1: pd.DataFrame, all_features_df_L2: pd.DataFrame,
                        feature_set: str, feature_cols: list, save_path: Path,
                        content_mode: Literal["all", "seen", "unseen"] = "all",
                        corrections=DEFAULT_CORRECTIONS,
                        calibration: dict = None,
                        replace_existing: bool = False,
                        per_text_parquet_dir: Path = None,
                        l1_parquet_dir: Path = None,
                        l2_parquet_dir: Path = None,
                        scorer: str = DEFAULT_SCORER) -> pd.DataFrame:
    """Compute and save eye_scores, then save one CSV per configured Correction.

    Args:
        calibration: Optional dict merged into each correction's context. Use
            it to supply keys that some corrections require (e.g.
            typological_distances, target_col). Corrections whose required
            keys are missing are skipped with a warning.
        replace_existing: Passed through to eye_score.

    Returns:
        The raw (uncorrected) eye_scores_df, so callers (e.g.
        run_one_dataset_version) can cache it for later corrections that
        depend on it.
    """
    # Per-text feature groups (TRANSITIONS, WFC) are only comparable when
    # train and test participants read the same fixed text. In eye_score
    # terms that's content_mode="seen" on OneStop; MECO has no
    # content_group column so any mode trivially satisfies this.
    if (
        feature_set in FIXED_TEXT_GROUP_PREFIXES
        and Fields.CONTENT_GROUP in all_features_df_L2.columns
        and content_mode != "seen"
    ):
        raise ValueError(
            f"Feature set {feature_set!r} is per-text and requires "
            f"content_mode='seen' on OneStop; got content_mode={content_mode!r}."
        )

    # No width-based skipping: every scorer runs on every feature set. Very wide
    # matrices are handled by the estimator factory adapting its solver (see
    # _make_lda), not by dropping the cell from the results.
    features_df_L1 = get_feature_set_df(all_features_df_L1, feature_set, feature_cols)
    features_df_L2 = get_feature_set_df(all_features_df_L2, feature_set, feature_cols)

    eye_scores_df = eye_score(
        features_df_L1, features_df_L2, save_path,
        content_mode=content_mode,
        replace_existing=replace_existing,
        per_text_parquet_dir=per_text_parquet_dir,
        l1_parquet_dir=l1_parquet_dir,
        l2_parquet_dir=l2_parquet_dir,
        feature_set=feature_set,
        scorer=scorer,
    )
    logger.info(f"Eye scores calculated and saved to {save_path}")

    base_ctx = {
        "feature_set": feature_set,
        "eye_scores_df": eye_scores_df,
        "all_features_df_L2": all_features_df_L2,
    }
    if calibration:
        base_ctx.update(calibration)

    # Leak-free per-fold rescoring for the typology corrections: recompute
    # eye_scores with the L2 z-score fit restricted to each fold's train pool
    # (content-group emulation of the current content_mode). Gated only to
    # one-row-per-participant (fully_agg) data — multi-row aggregations
    # (moving/first_p, per-item) don't fit the direction model. Per-text sets
    # (TRANSITIONS/WFC) ARE included: their ~10^5-10^6 text-specific columns make
    # this expensive, but the leak-free recompute is applied to them too rather
    # than silently falling back to the legacy (precomputed-scores) path.
    # Prototype and discriminative scorers each get a leak-free fold scorer that
    # recomputes scores per α-fold from a train-only pool. For discriminative
    # this is nested CV (the classifier's negative class is restricted to the
    # α-train pool), which closes the second-order α leak the precomputed-scores
    # path would otherwise carry. The raw eye_scores_df is untouched.
    if (
        not eye_scores_df.empty
        and all_features_df_L2.groupby(Fields.SUBJECT_ID).size().max() == 1
    ):
        if SCORERS[scorer].family == "discriminative":
            base_ctx["fold_scorer"] = _make_discriminative_fold_scorer(
                all_features_df_L1, all_features_df_L2, feature_set, feature_cols,
                SCORERS[scorer], content_mode=content_mode,
            )
        else:
            # Hand the fold scorer the ALREADY-EXTRACTED frames. It calls
            # get_feature_set_df on every fold (3x per direction), and that
            # function is pure column selection — idempotent, and independent of
            # which rows are present. Passing the full frames made each of the
            # ~700 fold calls copy and re-select a 600k-column matrix; passing
            # the extracted ones is numerically identical and drops the width to
            # the feature set's own columns.
            base_ctx["fold_scorer"] = _make_content_group_fold_scorer(
                features_df_L1, features_df_L2, feature_set, feature_cols,
                content_mode, save_path.parent, scorer=scorer,
            )

    for correction in corrections:
        adjusted = apply_correction_to_eyescore(correction, base_ctx)
        if adjusted is None:
            continue
        adj_save_path = save_path.with_name(save_path.stem + correction.suffix + save_path.suffix)
        adj_save_path.parent.mkdir(parents=True, exist_ok=True)
        adjusted.to_csv(adj_save_path, index=False)
        logger.info(f"{correction.name} eye scores saved to {adj_save_path}")

    return eye_scores_df


def get_per_item_parquet_dir(features_csv_path: Path, path_from_folder: str) -> Path | None:
    """Get the directory containing per-item parquets for per_item_agg.

    Returns None for other agg types (they use combined parquet files).
    """
    if "per_item_agg" not in path_from_folder:
        return None
    # Per-item parquets are in the same directory as the features CSV
    return Path(features_csv_path).parent


def run_one_dataset_version(src_path, path_from_folder, dataset_name:str,
                            preview:Literal["Hunting", "Gathering", "All"],
                            replace_existing:bool, feature_sets:dict=None,
                            content_mode: Literal["all", "seen", "unseen"] = "all",
                            results_root: Path = None,
                            typo_calib_distance_type: str = TYPO_CALIB_DISTANCE_TYPE,
                            corrections=DEFAULT_CORRECTIONS,
                            scorer: str = DEFAULT_SCORER) -> None:
    """Compute every feature set's eye_scores (and corrections) for one dataset
    version (`path_from_folder`, e.g. fully_agg/all/all), preview and content_mode.

    `results_root` overrides the output tree (default: src/methods/EyeScore/results).
    `typo_calib_distance_type` overrides the URIEL+ distance used for the
    typo_calibrated_* corrections (default: syntactic+genetic). Use a sibling
    `results_root` when switching distance types so the existing outputs aren't
    overwritten.
    """
    if results_root is None:
        results_root = src_path / "methods/EyeScore/results"

    dataset_version_path_L1 = Path(DATA_PATH/f"{dataset_name}L1/features_and_targets/{path_from_folder}/features_and_metadata.csv")
    dataset_version_path_L2 = Path(DATA_PATH/f"{dataset_name}L2/features_and_targets/{path_from_folder}/features_and_metadata.csv")

    all_features_l1_df = pd.read_csv(dataset_version_path_L1)
    all_features_l2_df = pd.read_csv(dataset_version_path_L2)
    all_features_l1_df, all_features_l2_df = filter_by_preview(all_features_l1_df, all_features_l2_df, preview)

    if feature_sets is None:
        feature_sets = ALL_FEATURE_SETS_DICT

    # Pre-compute typological distances once per dataset version. URIEL+ is a
    # heavy import, so failures are non-fatal — corrections that need it will
    # be skipped with a warning.
    typological_distances = None
    try:
        from src.methods.EyeScore.typology import compute_typological_distances
        l1_set = all_features_l2_df[Fields.L1].dropna().unique().tolist()
        typological_distances = compute_typological_distances(
            l1_set, distance_type=typo_calib_distance_type,
        )
    except Exception as e:
        logger.warning(
            f"Could not compute typological distances ({e}); "
            f"typology-calibrated correction will be skipped."
        )

    per_text_loaded = False
    per_text_l1_df = None
    per_text_l2_df = None
    for featureset_name, feature_cols in feature_sets.items():
        logger.info(f"Running eye score calculation with feature set: {featureset_name} (content_mode={content_mode})")
        save_path = Path(results_root / f"{dataset_name}/{preview}/{content_mode}/{path_from_folder}/{featureset_name}.csv")
        if save_path.exists() and not replace_existing:
            logger.info(f"Eye scores for {dataset_name} with feature set {featureset_name} already exist at {save_path}. Skipping calculation.")
            continue

        # Check if this is per_item_agg (which uses individual item parquets)
        is_per_item_agg = "per_item_agg" in path_from_folder

        # Per-text feature sets read from a self-contained sibling parquet.
        if featureset_name in FIXED_TEXT_GROUP_PREFIXES:
            # Datasets with a content_group axis only support per-text in
            # seen mode (train/test share text). Datasets without it
            # trivially satisfy this for any mode.
            needs_seen = Fields.CONTENT_GROUP in all_features_l2_df.columns
            if needs_seen and content_mode != "seen":
                logger.warning(
                    f"Skipping per-text feature set '{featureset_name}' on {dataset_name}: "
                    f"requires content_mode='seen', got '{content_mode}'."
                )
                continue

            # For per_item_agg, individual item parquets are loaded per-item during calculation
            # For other aggregations, use consolidated per-text parquets
            if not is_per_item_agg:
                if not per_text_loaded:
                    per_text_l1_df, per_text_l2_df = load_per_text_features_pair(
                        dataset_version_path_L1, dataset_version_path_L2,
                    )
                    per_text_loaded = True
                    if per_text_l1_df is not None:
                        per_text_l1_df, per_text_l2_df = filter_by_preview(per_text_l1_df, per_text_l2_df, preview)
                if per_text_l1_df is None:
                    continue
                df_l1, df_l2 = per_text_l1_df, per_text_l2_df
            else:
                # For per_item_agg, use regular features and pass per-item parquets separately
                df_l1, df_l2 = all_features_l1_df, all_features_l2_df
        else:
            df_l1, df_l2 = all_features_l1_df, all_features_l2_df

        calibration = {
            "typological_distances": typological_distances,
            "target_col": TYPO_CALIB_TARGET_COL,
        }
        per_item_parquet_dir = None
        l1_item_parquet_dir = None
        l2_item_parquet_dir = None
        if featureset_name in FIXED_TEXT_GROUP_PREFIXES:
            per_item_parquet_dir = get_per_item_parquet_dir(dataset_version_path_L2, path_from_folder)
            l1_item_parquet_dir = get_per_item_parquet_dir(dataset_version_path_L1, path_from_folder)
            l2_item_parquet_dir = get_per_item_parquet_dir(dataset_version_path_L2, path_from_folder)

        run_one_feature_set(
            df_l1, df_l2,
            featureset_name, feature_cols, save_path,
            content_mode=content_mode,
            corrections=corrections,
            calibration=calibration,
            replace_existing=replace_existing,
            per_text_parquet_dir=per_item_parquet_dir,
            l1_parquet_dir=l1_item_parquet_dir,
            l2_parquet_dir=l2_item_parquet_dir,
            scorer=scorer,
        )
