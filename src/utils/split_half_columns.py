"""Per-split common feature columns for split-half reliability.

Split-half reliability correlates half_1 against half_2, so both halves must be
modelled on the same features. Otherwise the correlation mixes the quantity of
interest (model stability) with feature-set churn.

Churn happens because rare PTB/universal tags drop out of whichever paragraphs a
half happens to contain: the extractor emits no column for a tag that has no
rows in that subset. Measured on MECO, S_CLUSTERS_NO_NORM spans 222-252 of its
258 columns across a split's four (half, parity) frames, and differs across
halves in all 20 splits. OneStop never drifts — its cohort and texts are large
enough that every tag appears — so the filter is inert there.

Used by both the predictions runner (src.run.predictions) and the EyeScore
runner (src.run.eyescore), which read the same underlying CSVs and
would otherwise drift independently.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.constants import DATA_PATH


def _meco_half_paths(dataset: str, split_idx: int) -> list[Path]:
    """L1+L2 CSVs for every (half, parity) frame of one MECO split."""
    paths = []
    for half_num in (1, 2):
        for odd_even in ("odd", "even"):
            for cohort in ("L1", "L2"):
                paths.append(
                    Path(DATA_PATH) / f"{dataset}{cohort}/features_and_targets/split_half"
                    / f"fully_agg/all/all/split_{split_idx}/half_{half_num}"
                    / f"seen_unseen/{odd_even}/features_and_metadata.csv"
                )
    return paths


def _onestop_half_paths(dataset: str, split_idx: int, item_level: str) -> list[Path]:
    """L1+L2 CSVs for both halves of one OneStop split at a given item level."""
    return [
        Path(DATA_PATH) / f"{dataset}{cohort}/features_and_targets/split_half"
        / f"fully_agg/ordinary/all/{item_level}/split_{split_idx}"
        / f"half_{half_num}/features_and_metadata.csv"
        for half_num in (1, 2)
        for cohort in ("L1", "L2")
    ]


def split_half_common_feature_cols(
    dataset: str, split_idx: int, item_level: str = "paragraph"
) -> set | None:
    """Columns present in EVERY frame of one split-half split.

    Reads headers only (nrows=1), so it costs a few KB per split. Returns None
    when any frame is missing, in which case callers should keep their existing
    per-frame behaviour rather than silently filtering against a partial set.
    """
    paths = (
        _meco_half_paths(dataset, split_idx)
        if dataset == "Meco"
        else _onestop_half_paths(dataset, split_idx, item_level)
    )
    common: set | None = None
    for p in paths:
        if not Path(p).exists():
            logger.debug(f"split-common: {p} missing, skipping filter for split_{split_idx}")
            return None
        cols = set(pd.read_csv(p, low_memory=False, nrows=1).columns)
        common = cols if common is None else (common & cols)
    return common


def apply_split_common_filter(feature_cols, common_cols, context: str):
    """Restrict feature_cols to common_cols, logging when anything is dropped.

    No-op when common_cols is None (frames unavailable) or when nothing is
    dropped — which is the case for every OneStop split and for the fixed-name
    feature sets (READING_SPEED, FIXATION_METRICS, WP_COEFS_NO_NORM).
    """
    if common_cols is None or not feature_cols:
        return list(feature_cols)
    kept = [c for c in feature_cols if c in common_cols]
    if len(kept) != len(feature_cols):
        logger.info(
            f"{context}: split-common filter kept {len(kept)}/{len(feature_cols)} feature cols"
        )
    return kept
