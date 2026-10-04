

"""Shared pieces of the reliability analysis: display names, prediction
targets, the OneStop preview convention, and the loaders for per-item and
original (fully aggregated) EyeScore scores and predictions."""
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.constants import DATA_PATH, Fields

logger = logging.getLogger(__name__)

PREDICTIONS_RESULTS_DIR = Path("src/methods/predictions/results")
EYESCORE_RESULTS_DIR = Path("src/methods/EyeScore/results")

# The OneStop preview to read. Gathering writes to the top-level result paths;
# any other preview writes to a parallel subtree (e.g. .../hunting/) instead of
# adding a `preview` column, so the CSVs and every table built from them keep
# one schema.
DEFAULT_PREVIEW = "Gathering"


def onestop_preview(dataset: str, preview: str) -> str:
    """The preview directory to read for `dataset`: only OneStop has previews,
    MECO is always "All"."""
    return preview if dataset == "OneStop" else "All"


# Prediction targets per dataset
DATASET_TARGET_COLS = {
    "OneStop": ["michtest_score", "lextale_score"],
    "Meco": ["proficiency_agg", "lextale_score"],
}

FEATURE_SET_DISPLAY = {
    "READING_SPEED": "WPM",
    "FIXATION_METRICS": "Avg. Fix.",
    "S_CLUSTERS_NO_NORM": "S-Clusters",
    "WP_COEFS_NO_NORM": "WP-Coefs",
    "TRANSITIONS": "Transitions",
    "WFC": "Word Fix."
}


def _item_level(path: Path) -> str:
    """The per-item aggregation level a results path belongs to."""
    s = str(path)
    return "article" if "/article/" in s else "paragraph" if "/paragraph/" in s else "unknown"


def load_eyescore_raw_data_with_pool(
    dataset: str, preview: str, pool: str, feature_set: str, p_agg_level: str = None
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Per-item EyeScore scores (`{feature_set}_items.csv`) of one dataset,
    preview and pool, as (frame, item level); None when nothing matches.

    `p_agg_level` restricts the match to one level; otherwise the first match
    is taken.
    """
    paths = list(EYESCORE_RESULTS_DIR.glob(
        f"{dataset}/{preview}/{pool}/per_item_agg/**/{feature_set}_items.csv"
    ))

    if not paths:
        return None

    paths = [p for p in paths if "typo" not in p.name and "_no_intercept" not in p.name]
    if not paths:
        return None

    if p_agg_level:
        paths = [p for p in paths if f"/{p_agg_level}/" in str(p)]

    if not paths:
        return None

    try:
        df = pd.read_csv(paths[0])
        if len(df) == 0 or "eye_score" not in df.columns:
            return None

        item_cols = [col for col in [Fields.UNIQUE_PARAGRAPH_ID, Fields.ARTICLE_ID] if col in df.columns]
        if not item_cols:
            logger.warning(f"No item ID column found in {paths[0]}")
            return None

        return df.copy(), _item_level(paths[0])
    except Exception as e:
        logger.debug(f"Could not load {paths[0]}: {e}")
        return None


def load_predictions_raw_data(
    dataset: str, preview: str, target: str, feature_set: str, model: str = "Ridge_Classifier", p_agg_level: str = None, pool: str = "all"
) -> Optional[Tuple[pd.DataFrame, str]]:
    """Per-item predictions (`{pool}__pool_all_items.csv`) of one dataset,
    preview, target, feature set and model, as (frame with a `pred` column,
    item level); None when nothing matches.
    """
    # The predictions tree spells previews in lowercase.
    preview_lower = preview.lower()
    glob_pattern = f"{dataset}/*/{preview_lower}/per_item_agg/**/{target}/{feature_set}/{model}/{pool}__pool_all_items.csv"
    paths = list(PREDICTIONS_RESULTS_DIR.glob(glob_pattern))

    if not paths:
        logger.debug(f"No files found for pattern: {PREDICTIONS_RESULTS_DIR}/{glob_pattern}")
        return None

    # If p_agg_level is specified, filter to that level only
    if p_agg_level:
        paths = [p for p in paths if f"/{p_agg_level}/" in str(p)]

    if not paths:
        return None

    try:
        df = pd.read_csv(paths[0])
        if len(df) == 0:
            return None
        # Older runs name the column 'prediction'.
        pred_col = "prediction" if "prediction" in df.columns else "pred" if "pred" in df.columns else None
        if pred_col is None:
            return None
        if pred_col == "prediction":
            df = df.rename(columns={"prediction": "pred"})

        return df.copy(), _item_level(paths[0])
    except Exception as e:
        logger.debug(f"Could not load {paths[0]}: {e}")
        return None


def add_content_group_from_metadata(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add each participant's content_group from the OneStop L2 metadata.

    A copy of `df` unchanged for MECO (no content groups), when the column is
    already there, or when the metadata cannot be read.
    """
    if dataset != "OneStop":
        return df.copy()

    if "content_group" in df.columns:
        return df.copy()

    metadata_path = DATA_PATH / "OneStopL2/metadata/metadata.csv"

    if not metadata_path.exists():
        logger.debug(f"Features metadata not found at {metadata_path}")
        return df.copy()

    try:
        # participant -> content_group
        metadata_df_cg = pd.read_csv(metadata_path, usecols=[Fields.SUBJECT_ID, Fields.CONTENT_GROUP])
        metadata_df_cg = metadata_df_cg.dropna(subset=[Fields.SUBJECT_ID, Fields.CONTENT_GROUP])
        participant_ids = set(df[Fields.SUBJECT_ID].unique())
        metadata_participants = set(metadata_df_cg[Fields.SUBJECT_ID].unique())
        missing_participants = participant_ids - metadata_participants
        if missing_participants:
            logger.warning(f"Participants in per-item data not found in metadata: {missing_participants}")
        df_merged = df.merge(metadata_df_cg, on=Fields.SUBJECT_ID, how='left')
        return df_merged
    except Exception as e:
        logger.debug(f"Could not merge content_group from metadata: {e}")
        return df.copy()


# Original (fully aggregated) score vs the mean of the per-item scores, for
# table_original_vs_per_item.
PRIMARY_TARGET = {"OneStop": "michtest_score", "Meco": "proficiency_agg"}
FEATURE_SETS = ["READING_SPEED", "FIXATION_METRICS", "S_CLUSTERS_NO_NORM",
                "WP_COEFS_NO_NORM", "TRANSITIONS", "WFC"]


def load_original(source, dataset, pool, level, fs, target, model, preview=None):
    """Participant -> original score: the fully_agg eye_score, or the pool-fold
    prediction on fully_agg features; None when the results are missing."""
    preview = preview or ("Gathering" if dataset == "OneStop" else "All")
    if source == "eyescore":
        paths = list(EYESCORE_RESULTS_DIR.glob(f"{dataset}/{preview}/{pool}/fully_agg/**/{fs}.csv"))
        paths = [p for p in paths if not any(
            s in p.name for s in ["distance_mreg", "typo_calibrated", "_no_intercept", "michtest"])]
        if not paths:
            return None
        df = pd.read_csv(paths[0])
        if "eye_score" not in df.columns:
            return None
        return df.groupby(Fields.SUBJECT_ID)["eye_score"].mean().rename("original")
    else:
        tgt = PRIMARY_TARGET[dataset] if target == "primary" else target
        preview_lower = preview.lower()
        paths = list(PREDICTIONS_RESULTS_DIR.glob(
            f"{dataset}/*/{preview_lower}/fully_agg/{tgt}/{fs}/{model}/{pool}__pool_all.csv"))
        if not paths:
            return None
        df = pd.read_csv(paths[0])
        if "pred" not in df.columns:
            return None
        return df.groupby(Fields.SUBJECT_ID)["pred"].mean().rename("original")


def load_per_item_mean(source, dataset, pool, level, fs, target, model, preview=None):
    """Participant -> mean of their per-item scores (eye_score, or prediction);
    None when the per-item results are missing."""
    preview = preview or ("Gathering" if dataset == "OneStop" else "All")
    level_param = None if dataset == "Meco" else level
    if source == "eyescore":
        result = load_eyescore_raw_data_with_pool(dataset, preview, pool, fs, level_param)
        col = "eye_score"
    else:
        tgt = PRIMARY_TARGET[dataset] if target == "primary" else target
        result = load_predictions_raw_data(dataset, preview, tgt, fs, model, level_param, pool)
        col = "pred"
    if result is None:
        return None
    df, _ = result
    return df.dropna(subset=[col]).groupby(Fields.SUBJECT_ID)[col].mean().rename("per_item_mean")
