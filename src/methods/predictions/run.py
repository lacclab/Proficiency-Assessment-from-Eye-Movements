"""Cross-validated prediction of proficiency scores: model construction, the
per-item SEEN/UNSEEN runners, and data loading. Driven by src/run/predictions.py.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.configs import MODELS, ModelNames
from src.constants import (
    AggTypes,
    DATA_PATH,
    FIXED_TEXT_GROUP_PREFIXES,
    Fields,
    FoldsMethods,
)
from src.utils.filter import filter_by_preview


def run_cross_val_per_item_independent(features_df_L1, features_df_L2, feature_cols,
                                       target_col, model_name: str, method, n_splits,
                                       aug_precentage=1, save_path: Path=None,
                                       batch_framework=None, n_seeds: int=1,
                                       per_text_parquet_dir: Path=None, feature_set: str=None) -> pd.DataFrame:
    """Run cross-validation independently for each item (paragraph/article).

    Each item is treated as a separate dataset:
    - Train model ONLY on L1 data for that item
    - Test on L2 data for that item
    - Use leave-one-participant-out CV within that item
    - No data leakage between items
    - Final predictions are aggregated across items at participant level

    Args:
        features_df_L1: L1 features (per-item format with item column, behavioral features only)
        features_df_L2: L2 features (per-item format with item column, behavioral features only)
        feature_cols: Feature columns to use (behavioral only)
        target_col: Target column name
        model_name: Model name
        method: CV fold method
        n_splits: Number of CV splits
        save_path: Path to save predictions
        batch_framework: Batch framework (unused for per-item)
        n_seeds: Number of random seeds
        per_text_parquet_dir: Directory containing per_text_features_item_*.parquet files (optional)

    Returns:
        Per-item predictions DataFrame
    """
    item_col = Fields.UNIQUE_PARAGRAPH_ID if Fields.UNIQUE_PARAGRAPH_ID in features_df_L2.columns else Fields.ARTICLE_ID

    if item_col not in features_df_L1.columns or item_col not in features_df_L2.columns:
        raise ValueError(f"Item column '{item_col}' not found in DataFrames")

    # For article-level items with content_group, treat (article_id, content_group) as separate items
    if item_col == Fields.ARTICLE_ID and Fields.CONTENT_GROUP in features_df_L2.columns:
        unique_items = features_df_L2[[item_col, Fields.CONTENT_GROUP]].drop_duplicates().values.tolist()
        logger.info(f"Running per-item independent CV for {len(unique_items)} (article, content_group) combinations")
    else:
        unique_items = [(item_id,) for item_id in features_df_L2[item_col].unique()]
        logger.info(f"Running per-item independent cross-validation for {len(unique_items)} items")

    logger.info(f"Total L2 participants: {features_df_L2[Fields.SUBJECT_ID.value].nunique()}")
    logger.info(f"Total L1 participants: {features_df_L1[Fields.SUBJECT_ID.value].nunique()}")
    logger.info(f"Per-text parquet dir: {per_text_parquet_dir}")

    all_predictions = []
    skipped_items = []

    for item_tuple in unique_items:
        item_id = item_tuple[0]
        content_group = item_tuple[1] if len(item_tuple) > 1 else None

        # Filter to only this item (and content_group if applicable)
        L1_for_item = features_df_L1[features_df_L1[item_col] == item_id]
        L2_for_item = features_df_L2[features_df_L2[item_col] == item_id]

        if content_group is not None:
            L1_for_item = L1_for_item[L1_for_item[Fields.CONTENT_GROUP.value] == content_group]
            L2_for_item = L2_for_item[L2_for_item[Fields.CONTENT_GROUP.value] == content_group]

        if len(L1_for_item) == 0 or len(L2_for_item) == 0:
            item_str = f"{item_col}={item_id}" if content_group is None else f"{item_col}={item_id}, {Fields.CONTENT_GROUP}={content_group}"
            logger.debug(f"Skipping {item_str}: no data (L1={len(L1_for_item)}, L2={len(L2_for_item)})")
            skipped_items.append((item_id, content_group, "no_data"))
            continue

        unique_participants_l2 = L2_for_item[Fields.SUBJECT_ID.value].nunique()
        unique_participants_l1 = L1_for_item[Fields.SUBJECT_ID.value].nunique()

        # Check if item has enough data for CV
        if unique_participants_l2 < 2:
            item_str = f"{item_col}={item_id}" if content_group is None else f"{item_col}={item_id}, {Fields.CONTENT_GROUP}={content_group}"
            logger.warning(f"Skipping {item_str}: only {unique_participants_l2} L2 participant(s) (need ≥2 for CV)")
            skipped_items.append((item_id, content_group, f"only_{unique_participants_l2}_l2_participants"))
            continue

        item_str = f"{item_col}={item_id}" if content_group is None else f"{item_col}={item_id}, {Fields.CONTENT_GROUP}={content_group}"
        logger.info(f"Processing {item_str} (L1={unique_participants_l1} participants, L2={unique_participants_l2} participants)")

        # Load per-item fixed features if parquet directory is provided
        item_feature_cols = list(feature_cols)  # Copy to avoid modifying original
        L1_to_use = L1_for_item.copy()
        L2_to_use = L2_for_item.copy()

        if per_text_parquet_dir is not None:
            per_item_parquet = per_text_parquet_dir / f"per_text_features_item_{item_id}.parquet"
            logger.info(f"Looking for per-item parquet at: {per_item_parquet} (exists: {per_item_parquet.exists()})")
            if per_item_parquet.exists():
                try:
                    pt_df = pd.read_parquet(
                        per_item_parquet,
                        thrift_string_size_limit=2**31 - 1,
                        thrift_container_size_limit=2**31 - 1,
                    )
                    logger.info(f"Loaded per-item fixed features for {item_str}: {pt_df.shape}")

                    # Merge with behavioral features on participant and item columns
                    merge_cols = [Fields.SUBJECT_ID, item_col]
                    if Fields.CONTENT_GROUP in pt_df.columns and content_group is not None:
                        merge_cols.append(Fields.CONTENT_GROUP)

                    L1_to_use = L1_for_item.merge(pt_df, on=merge_cols, how='left')
                    L2_to_use = L2_for_item.merge(pt_df, on=merge_cols, how='left')

                    # Add the per-text columns of this feature set (TRANSITIONS / WFC)
                    if feature_set in FIXED_TEXT_GROUP_PREFIXES:
                        feature_prefix = FIXED_TEXT_GROUP_PREFIXES[feature_set]
                        per_text_cols = [c for c in pt_df.columns if c.startswith(feature_prefix)]
                        logger.debug(f"Loaded {len(per_text_cols)} {feature_set} columns for {item_str}")
                    else:
                        raise ValueError(f"Invalid feature_set {feature_set!r} for per-item fixed features, expected one of {list(FIXED_TEXT_GROUP_PREFIXES.keys())}")
                    item_feature_cols.extend(per_text_cols)
                except Exception as e:
                    logger.warning(f"Could not load per-item fixed features for {item_str}: {str(e)}")

        # Create model instance for this item
        model = _create_model(model_name)

        # Run cross-validation on this item's data
        try:
            item_pred_df = model.run_cross_validation(
                L1_to_use,
                L2_to_use,
                aug_precentage,
                feature_cols=item_feature_cols,
                target_col=target_col,
                fold_method=method,
                n_splits=n_splits,
                save_path=None,  # Don't save intermediate item results
                agg_type=AggTypes.FULL,  # Treat each item as full agg
                df_l2_full=L2_for_item,
                train_on_full=False,
            )
        except Exception as e:
            logger.error(f"Error processing {item_col}={item_id}: {str(e)}")
            raise

        if item_pred_df is None or len(item_pred_df) == 0:
            logger.warning(f"No predictions returned for {item_col}={item_id}, skipping")
            continue

        # Add item column to predictions
        item_pred_df[item_col] = item_id
        all_predictions.append(item_pred_df)

    if skipped_items:
        logger.warning(f"Skipped {len(skipped_items)} items: {skipped_items}")

    if not all_predictions:
        raise ValueError(f"No valid item-level data for predictions (skipped all {len(skipped_items)} items)")

    pred_df = pd.concat(all_predictions, ignore_index=True)
    logger.info(f"Generated predictions for {len(pred_df)} (participant, item) pairs across {len(all_predictions)} items")

    return pred_df


def aggregate_per_item_predictions(per_item_df: pd.DataFrame) -> pd.DataFrame:
    """Average item-level predictions to participant level.

    Args:
        per_item_df: DataFrame with multiple rows per participant (one per item)
                     Expected columns: pred, true, Fields.SUBJECT_ID, Fields.L1, fold, etc.

    Returns:
        DataFrame with one row per participant.
        Uses columns: pred (averaged), true (first value), Fields.SUBJECT_ID, Fields.L1, etc.
    """
    # Group by participant and average pred, keep true (same per participant)
    agg_df = per_item_df.groupby(Fields.SUBJECT_ID.value).agg({
        'pred': 'mean',
        'true': 'first',
    }).reset_index()

    # Merge back other columns that are constant per participant (L1, etc.)
    # Exclude item columns (unique_paragraph_id, article_id) since they vary per item
    item_cols = {Fields.UNIQUE_PARAGRAPH_ID, Fields.ARTICLE_ID}
    other_cols = [c for c in per_item_df.columns if c not in ['pred', 'true', Fields.SUBJECT_ID] and c not in item_cols]
    if other_cols:
        const_per_participant = per_item_df.drop_duplicates(subset=[Fields.SUBJECT_ID.value])[
            [Fields.SUBJECT_ID.value] + other_cols
        ]
        agg_df = agg_df.merge(const_per_participant, on=Fields.SUBJECT_ID.value, how='left')

    return agg_df


def _prune_empty_features(train_df: pd.DataFrame, feature_cols: list) -> list:
    """Drop feature columns with no data at all in the training frame.

    A POS-cluster feature is NaN when the tag has no words in the text a row
    covers, so for a single item some columns are empty for everyone. Those
    carry no signal for this segment and would standardize to a constant, so
    they go.

    Columns that survive may still have scattered NaN (tag present in the text
    but not in that participant's read subset, or a metric undefined for a whole
    site — see the ch_s case in compute_grouped_features). Those are left in
    place deliberately: the models impute them from training-fold statistics.
    Filling them here would use PRE-FOLD means, mixing held-out rows into the
    values used for training rows and hiding the missingness from the model's
    own fold-scoped imputation, which would then count filled cells as real
    observations.
    """
    kept = [c for c in feature_cols if c in train_df.columns and train_df[c].notna().any()]
    dropped = len(feature_cols) - len(kept)
    if dropped:
        logger.info(f"per-item: dropped {dropped}/{len(feature_cols)} feature cols with no data in training frame")
    return kept


def run_per_item_seen(
    lopo_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    item_col: str,
    model_name: str,
    aug_percentage: float = 1.0,
    l1_parquet_dir: Path = None,
    l2_parquet_dir: Path = None,
    pool=None,
    feature_set: str = None,
) -> pd.DataFrame:
    """Run LOPO per-item predictions for SEEN pool.

    For each item, performs leave-one-participant-out cross-validation within
    that item's data. Used for OneStop SEEN (within content_group) and MECO SEEN.

    Args:
        lopo_df: L2 data for this segment (filtered by content_group for OneStop SEEN, or one item)
        l1_df: L1 augmentation data
        feature_cols: Feature columns to use
        target_col: Target column name
        item_col: Item column name (UNIQUE_PARAGRAPH_ID or ARTICLE_ID)
        model_name: Model name
        aug_percentage: L1 augmentation fraction
        l1_parquet_dir: Directory containing L1 per_text_features_item_*.parquet files
        l2_parquet_dir: Directory containing L2 per_text_features_item_*.parquet files
        pool: Pool enum (SEEN/UNSEEN/ALL) for per-text feature validation
        feature_set: Feature set name; selects the per-text column prefix

    Returns:
        Per-item predictions (one row per participant × item pair)
    """
    # This is called from per-item pipeline with segment data that may already be
    # filtered to specific items. Check if per-text parquets should be loaded.
    item_feature_cols = list(feature_cols)
    L1_to_use = l1_df.copy()
    L2_to_use = lopo_df.copy()

    # Ensure paths are Path objects
    if l1_parquet_dir is not None and not isinstance(l1_parquet_dir, Path):
        l1_parquet_dir = Path(l1_parquet_dir)
    if l2_parquet_dir is not None and not isinstance(l2_parquet_dir, Path):
        l2_parquet_dir = Path(l2_parquet_dir)

    logger.info(f"run_per_item_seen: lopo_df shape={lopo_df.shape}, unique items={lopo_df[item_col].nunique()}, l1_parquet_dir exists={l1_parquet_dir is not None}, l2_parquet_dir exists={l2_parquet_dir is not None}")

    # Extract item_id (function receives exactly 1 item from resolver)
    segment_item_id = lopo_df[item_col].unique()[0] if len(lopo_df) > 0 else None

    # Load per-text features if parquet directories provided
    if l1_parquet_dir is not None and l2_parquet_dir is not None:
        unique_items = lopo_df[item_col].unique()
        logger.info(f"  Unique items in segment: {unique_items}")
        if len(unique_items) == 1:
            item_id = unique_items[0]
            l1_pt_parquet = l1_parquet_dir / f"per_text_features_item_{item_id}.parquet"
            l2_pt_parquet = l2_parquet_dir / f"per_text_features_item_{item_id}.parquet"

            try:
                # Load per-text parquets directly (item-specific files)
                l1_pt = None
                l2_pt = None

                if l1_pt_parquet.exists():
                    l1_pt = pd.read_parquet(
                        l1_pt_parquet,
                        thrift_string_size_limit=2**31 - 1,
                        thrift_container_size_limit=2**31 - 1,
                    )
                if l2_pt_parquet.exists():
                    l2_pt = pd.read_parquet(
                        l2_pt_parquet,
                        thrift_string_size_limit=2**31 - 1,
                        thrift_container_size_limit=2**31 - 1,
                    )

                if l1_pt is not None and l2_pt is not None:
                    # Align L1 and L2 per-text columns to union
                    feature_prefix = FIXED_TEXT_GROUP_PREFIXES.get(feature_set, None)
                    if feature_prefix is None:
                        raise ValueError(f"Invalid feature_set {feature_set!r} for per-item fixed features, expected one of {list(FIXED_TEXT_GROUP_PREFIXES.keys())}")
                    def is_per_text(col: str) -> bool:
                        return col.startswith(feature_prefix)

                    pt1_cols = {c for c in l1_pt.columns if is_per_text(c)}
                    pt2_cols = {c for c in l2_pt.columns if is_per_text(c)}
                    union_cols = sorted(pt1_cols | pt2_cols)
                    meta1_cols = [c for c in l1_pt.columns if not is_per_text(c)]
                    meta2_cols = [c for c in l2_pt.columns if not is_per_text(c)]

                    l1_pt = l1_pt.reindex(columns=meta1_cols + union_cols, fill_value=0)
                    l2_pt = l2_pt.reindex(columns=meta2_cols + union_cols, fill_value=0)
                    logger.info(f"Loaded aligned per-text features for item {item_id}: L1={l1_pt.shape}, L2={l2_pt.shape}")

                    # Merge aligned per-text features (keep only per-text cols to avoid suffix conflicts)
                    merge_cols = [Fields.SUBJECT_ID.value, item_col]
                    if Fields.CONTENT_GROUP.value in l1_pt.columns and Fields.CONTENT_GROUP.value in L1_to_use.columns:
                        merge_cols.append(Fields.CONTENT_GROUP.value)

                    # Keep only merge keys and per-text features (drop metadata to avoid _x/_y suffixes)
                    per_text_cols = [c for c in l1_pt.columns if c.startswith((feature_prefix))]
                    l1_pt_for_merge = l1_pt[merge_cols + per_text_cols] if per_text_cols else l1_pt[merge_cols]
                    l2_pt_for_merge = l2_pt[merge_cols + per_text_cols] if per_text_cols else l2_pt[merge_cols]

                    L1_to_use = L1_to_use.merge(l1_pt_for_merge, on=merge_cols, how='left')
                    L2_to_use = L2_to_use.merge(l2_pt_for_merge, on=merge_cols, how='left')

                    # Filter per-text columns to only those for the current item
                    # Columns are named transitions_{item_id}_... or wfc_{item_id}_...
                    all_per_text_cols = [c for c in l2_pt.columns if c.startswith(feature_prefix)]
                    per_text_cols = [c for c in all_per_text_cols if f"_{item_id}_" in c]
                    item_feature_cols.extend(per_text_cols)
                    logger.info(f"Added {len(per_text_cols)} per-text columns for item {item_id} (filtered from {len(all_per_text_cols)} total columns)")
                else:
                    logger.warning(f"Could not load aligned per-text features for item {item_id}")
            except Exception as e:
                logger.exception(f"Could not load per-text features for item {item_id}: {e}")
        else:
            logger.warning(f"Expected 1 item in segment but found {len(unique_items)}: {unique_items}")

    # Prune columns with no data for this item — they carry no signal and would
    # otherwise standardize to a constant. Surviving NaN is left in place for the
    # models to impute per fold; filling it here would use pre-fold statistics.
    item_feature_cols = _prune_empty_features(L2_to_use, item_feature_cols)

    # Run LOPO cross-validation using the model's existing method
    model = _create_model(model_name)
    result = model.run_cross_validation(
        L1_to_use,
        L2_to_use,
        aug_percentage,
        feature_cols=item_feature_cols,
        target_col=target_col,
        fold_method=FoldsMethods.LEAVE_ONE_PARTICIPANT_OUT,
        n_splits=None,
        save_path=None,
        agg_type=AggTypes.FULL,
        df_l2_full=L2_to_use,
        train_on_full=False,
        pool=pool,
    )

    # Add item column to segment predictions
    if result is not None and not result.empty and segment_item_id is not None:
        result[item_col] = segment_item_id

    return result


def run_per_item_unseen(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    model_name: str,
    aug_percentage: float = 1.0,
    segment_id: str = None) -> pd.DataFrame:
    """Run per-item predictions for UNSEEN pool with fixed train/test split.

    For MECO (segment_id contains "odd" or "even"): does LOIO per participant.
    For OneStop: trains single model on all training data.

    Args:
        train_df: L2 training data (all items from train batches/parities)
        test_df: L2 test data (all items from test batch/parity)
        l1_df: L1 augmentation data (for training batches only)
        feature_cols: Feature columns to use
        target_col: Target column name
        model_name: Model name
        aug_percentage: L1 augmentation fraction
        segment_id: Segment identifier to detect if LOIO needed (MECO has "odd"/"even")

    Returns:
        Per-item predictions (one row per test participant × item pair)
    """
    # Detect if this is MECO UNSEEN (odd/even parity split) which needs LOIO
    is_meco_unseen = segment_id and ("odd" in segment_id or "even" in segment_id)
    logger.info(f"run_per_item_unseen: segment_id={segment_id}, is_meco_unseen={is_meco_unseen}")

    model = _create_model(model_name)

    # Extract feature and target columns
    train_feature_cols = [c for c in feature_cols if c in train_df.columns]
    test_feature_cols = [c for c in feature_cols if c in test_df.columns]

    # Use intersection of columns present in both datasets
    shared_cols = set(train_feature_cols) & set(test_feature_cols)
    if not shared_cols:
        raise ValueError("No shared feature columns between train and test for UNSEEN per-item")
    shared_cols = list(shared_cols)

    # Prune columns with no data in the training frame; leave surviving NaN for
    # the models to impute per fold. See _prune_empty_features.
    shared_cols = _prune_empty_features(train_df, shared_cols)

    # Clean test data: remove sentinel -1.0 values from target before evaluation
    logger.info(f"UNSEEN: test_df shape={test_df.shape}, target mean={test_df[target_col].mean():.2f}")
    test_df_clean = test_df[test_df[target_col] != -1].copy()
    logger.info(f"UNSEEN: test_df_clean shape={test_df_clean.shape}, target mean={test_df_clean[target_col].mean():.2f}")

    # For MECO UNSEEN (odd/even splits): do LOIO per participant since same participants in train and test
    if is_meco_unseen:
        all_results = []
        for test_participant in sorted(test_df_clean[Fields.SUBJECT_ID.value].unique()):
            # Train without this participant
            train_for_participant = train_df[train_df[Fields.SUBJECT_ID.value] != test_participant]
            # Drop on the TARGET, not on the features: per item a POS tag is
            # often absent from the words a participant read, so listwise
            # deletion on the features would drop almost every row of the wide
            # sets. Same contract as BaseModel.drop_missing_and_negative_ones:
            # a partially-undefined row is imputed from training-fold means;
            # only a row with NO features at all is dropped.
            train_clean = train_for_participant[shared_cols + [target_col]].dropna(subset=[target_col])
            train_clean = train_clean[~train_clean[shared_cols].isna().all(axis=1)]
            train_clean = train_clean[train_clean[target_col] != -1]

            if len(train_clean) == 0:
                continue

            model_fold = _create_model(model_name)
            model_fold.fit_with_augmentation(
                df_l1=l1_df,
                df_l2=train_clean,
                aug_precentage=aug_percentage,
                feature_cols=shared_cols,
                target_col=target_col,
            )

            # Predict only for this participant
            test_for_participant = test_df_clean[test_df_clean[Fields.SUBJECT_ID.value] == test_participant]
            y_test = test_for_participant[target_col].values
            # Predict unless the row has no features at all; predict_df ->
            # prepare_test_data imputes the gaps from the means stored at fit
            # time, so this introduces no test leakage.
            test_feature_mask = ~pd.isna(test_for_participant[shared_cols]).all(axis=1)

            if test_feature_mask.any():
                test_valid_df = test_for_participant[test_feature_mask][shared_cols]
                predictions_valid = model_fold.predict_df(test_valid_df, feature_cols=shared_cols)
                predictions = np.full(len(test_for_participant), np.nan)
                predictions[test_feature_mask] = predictions_valid
            else:
                predictions = np.full(len(test_for_participant), np.nan)

            # Build result for this participant
            item_cols = {Fields.UNIQUE_PARAGRAPH_ID, Fields.ARTICLE_ID}

            result_data = {
                Fields.SUBJECT_ID: test_for_participant[Fields.SUBJECT_ID].values,
                'true': y_test,
                'pred': predictions,
            }
            for col in item_cols:
                if col in test_for_participant.columns:
                    result_data[col] = test_for_participant[col].values

            all_results.append(pd.DataFrame(result_data))

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            raise ValueError("No valid LOIO predictions generated")
    else:
        # OneStop UNSEEN: train a single model on all training data. Drop on
        # the target and the sentinel, keep rows whose features are only partly
        # defined — same contract as the MECO branch above.
        train_df_clean = train_df[shared_cols + [target_col]].dropna(subset=[target_col])
        train_df_clean = train_df_clean[~train_df_clean[shared_cols].isna().all(axis=1)]
        train_df_clean = train_df_clean[train_df_clean[target_col] != -1]
        if len(train_df_clean) == 0:
            raise ValueError("No valid training samples after dropping NaN values and sentinel -1.0")

        # Fit model on cleaned training data (with L1 augmentation)
        model.fit_with_augmentation(
            df_l1=l1_df,
            df_l2=train_df_clean,
            aug_precentage=aug_percentage,
            feature_cols=shared_cols,
            target_col=target_col,
        )

    # Predict every test row that has any features (same contract as the MECO
    # branch above); predict_df -> prepare_test_data imputes the gaps from the
    # means stored at fit time, so there is no test leakage.
    y_test = test_df_clean[target_col].values
    test_feature_mask = ~pd.isna(test_df_clean[shared_cols]).all(axis=1)

    if not test_feature_mask.any():
        raise ValueError("No valid test samples (all feature rows have NaN)")

    # Get predictions for valid feature rows using predict_df (applies scaler)
    test_valid_df = test_df_clean[test_feature_mask][shared_cols]
    predictions_valid = model.predict_df(test_valid_df, feature_cols=shared_cols)

    # Reconstruct predictions array with NaN for invalid rows
    predictions = np.full(len(test_df_clean), np.nan)
    predictions[test_feature_mask] = predictions_valid

    # Build results DataFrame (use test_df_clean since we filtered -1.0 values)
    item_cols = {Fields.UNIQUE_PARAGRAPH_ID, Fields.ARTICLE_ID}

    result_data = {
        Fields.SUBJECT_ID: test_df_clean[Fields.SUBJECT_ID].values,
        'true': y_test,
        'pred': predictions,
    }

    # Add item columns only (no metadata)
    for col in item_cols:
        if col in test_df_clean.columns:
            result_data[col] = test_df_clean[col].values

    results = pd.DataFrame(result_data)
    return results


def _create_model(model_name: str):
    """Instantiate a fresh model. Separate per-call to keep parallel jobs thread-safe.

    Ridge / LogRidge select α per fit via RidgeCV's closed-form LOOCV over
    _DEFAULT_ALPHA_GRID (leak-free: scored on that fold's training data only).
    On by default so a cell can't silently fall back to an untuned α=1.0 — the
    POOL_ALL fast paths (cross-fit LOOCV, _unseen_ridgecv_fast) always tuned,
    so leaving the per-fold loop untuned made Fixed and Any Text cells on the
    same dataset differ in procedure rather than in regime. Set
    PROF_CHOOSE_ALPHA=0 to restore the fixed α=1.0 behaviour.
    """
    choose_alpha = os.environ.get("PROF_CHOOSE_ALPHA", "1") == "1"
    if model_name == ModelNames.RIDGE_CLASSIFIER:
        from src.methods.predictions.models.RidgeRegression import RidgeRegression
        return RidgeRegression(choose_alpha=choose_alpha)
    elif model_name == ModelNames.LOG_RIDGE_REGRESSION:
        from src.methods.predictions.models.RidgeRegression import LogRidgeRegression
        return LogRidgeRegression(choose_alpha=choose_alpha)
    elif model_name == ModelNames.LINEAR_REGRESSION:
        from src.methods.predictions.models.LinearRegression import LinearRegression
        return LinearRegression()
    elif model_name == ModelNames.RANDOM_FOREST:
        from src.methods.predictions.models.RandomForest import RandomForest
        return RandomForest()
    elif model_name == ModelNames.DECISION_TREE:
        from src.methods.predictions.models.DecisionTree import DecisionTree
        return DecisionTree()
    elif model_name == ModelNames.LIGHTGBM:
        from src.methods.predictions.models.GradientBoosting import LightGBM
        return LightGBM()
    elif model_name == ModelNames.XGBOOST:
        from src.methods.predictions.models.GradientBoosting import XGBoost
        return XGBoost()
    elif model_name == ModelNames.TABSTAR:
        from src.methods.predictions.models.TabStar import TabStar
        return TabStar()
    elif model_name == ModelNames.TABPFN:
        from src.methods.predictions.models.TabPFN import TabPFN
        return TabPFN()
    elif model_name == ModelNames.AVERAGE:
        from src.methods.predictions.models.AvgModel import AvgModel
        return AvgModel()

    return MODELS[model_name]


def run_cross_val(features_df_L1, features_df_L2, feature_cols,
                  target_col, model_name:str, method, n_splits,
                  aug_precentage=1, choose_alpha=False, save_path: Path=None,
                  agg_type=AggTypes.FULL,
                  df_l2_full=None, train_on_full:bool=False) -> pd.DataFrame:
    """Cross-validated predictions for every L2 participant (BaseModel.run_cross_validation).
    `choose_alpha` is accepted for a uniform call signature and not used."""
    model = _create_model(model_name)

    pred_df = model.run_cross_validation(
        features_df_L1,
        features_df_L2,
        aug_precentage,
        feature_cols=feature_cols,
        target_col=target_col,
        fold_method=method,
        n_splits=n_splits,
        save_path=save_path,
        agg_type=agg_type,
        df_l2_full=df_l2_full,
        train_on_full=train_on_full,
    )
    return pred_df


def run_one_feature_set(
    features_df_L1: pd.DataFrame,
    features_df_L2: pd.DataFrame,
    feature_set: str,
    feature_cols: list,
    target_col: str = None,
    model_name: str = None,
    method=FoldsMethods.RANDOM,
    n_splits: int = 5,
    aug_precentage: float = 1,
    save_path: Path = None,
    agg_type=AggTypes.FULL,
    df_l2_full=None,
    train_on_full=False,
    n_seeds: int = 1,
    replace_existing: bool = False,
    per_text_parquet_dir: Path = None,
) -> pd.DataFrame:
    """Predict `target_col` from one feature set; per-item data is routed to
    run_cross_val_per_item_independent and aggregated to participants."""
    if target_col is None:
        raise ValueError("target_col must be provided")
    if model_name is None:
        raise ValueError("model_name must be provided")

    # Per-text feature groups (TRANSITIONS, WFC) carry an empty static
    # feature_cols list — resolve their columns from the data at runtime
    # by prefix match.
    fixed_text_prefix = FIXED_TEXT_GROUP_PREFIXES.get(feature_set)
    if fixed_text_prefix is not None:
        available_cols = set(features_df_L2.columns) & set(features_df_L1.columns)
        feature_cols = sorted(c for c in available_cols if c.startswith(fixed_text_prefix))

    # Filter out feature columns not present in the data
    available_cols = set(features_df_L2.columns) & set(features_df_L1.columns)
    missing_cols = [c for c in feature_cols if c not in available_cols]
    if missing_cols:
        logger.warning(f"Feature set '{feature_set}': dropping {len(missing_cols)} columns not found in data: {missing_cols}")
        feature_cols = [c for c in feature_cols if c in available_cols]
    if not feature_cols:
        logger.warning(f"Feature set '{feature_set}': no valid columns remain, skipping.")
        return pd.DataFrame()

    # Detect per-item data and use per-item independent CV
    item_col = Fields.UNIQUE_PARAGRAPH_ID if Fields.UNIQUE_PARAGRAPH_ID in features_df_L2.columns else None
    if item_col is None:
        item_col = Fields.ARTICLE_ID if Fields.ARTICLE_ID in features_df_L2.columns else None

    is_per_item = False
    if item_col is not None and agg_type == AggTypes.PER_ITEM:
        rows_per_participant = features_df_L2.groupby(Fields.SUBJECT_ID.value).size()
        is_per_item = (rows_per_participant > 1).any()

    if is_per_item:
        logger.info("Per-item data detected. Using per-item independent cross-validation.")
        pred_df = run_cross_val_per_item_independent(
            features_df_L1=features_df_L1,
            features_df_L2=features_df_L2,
            feature_cols=feature_cols,
            target_col=target_col,
            model_name=model_name,
            method=method,
            n_splits=n_splits,
            aug_precentage=aug_precentage,
            save_path=save_path,
            batch_framework=None,
            n_seeds=n_seeds,
            per_text_parquet_dir=per_text_parquet_dir,
            feature_set=feature_set,
        )
    else:
        pred_df = run_cross_val(
            features_df_L1=features_df_L1,
            features_df_L2=features_df_L2,
            feature_cols=feature_cols,
            target_col=target_col,
            model_name=model_name,
            method=method,
            n_splits=n_splits,
            aug_precentage=aug_precentage,
            save_path=save_path,
            agg_type=agg_type,
            df_l2_full=df_l2_full,
            train_on_full=train_on_full,
        )

    # For per_item_agg: handle per-item predictions and aggregation
    if is_per_item and save_path is not None:
        items_save_path = save_path.parent / f"{save_path.stem}_items.csv"

        # Check if per-item results already exist
        if items_save_path.exists() and not replace_existing:
            # Load existing per-item predictions instead of recomputing
            logger.info(f"Per-item predictions already exist at {items_save_path}, loading and aggregating...")
            pred_df = pd.read_csv(items_save_path)
        else:
            # Save newly computed per-item predictions
            pred_df.to_csv(items_save_path, index=False)
            logger.info(f"Saved per-item predictions to {items_save_path}")

        # Aggregate to participant level
        agg_pred_df = aggregate_per_item_predictions(pred_df)
        agg_pred_df.to_csv(save_path, index=False)
        logger.info(f"Saved aggregated predictions to {save_path}")

        return agg_pred_df

    return pred_df


def load_and_filter_data(dataset, agg_type, p_agg_level, p_agg, preview):
    """Load L1/L2 feature DataFrames for a dataset+agg+preview combo and apply preview filtering.

    Returns: (l1_df, l2_df, agg_label) or (None, None, agg_label) if data not found
    """
    is_meco = dataset == "Meco"
    reread_level = "all" if is_meco else "ordinary"

    agg_type_str = agg_type.value if hasattr(agg_type, 'value') else str(agg_type)
    if agg_type == AggTypes.FULL:
        agg_path_suffix = f"{agg_type_str}/{reread_level}/all"
        agg_label = agg_type_str
    elif agg_type == AggTypes.PER_ITEM:
        agg_path_suffix = f"{agg_type_str}/{reread_level}/all/{p_agg_level}"
        agg_label = f"{agg_type_str}_{p_agg_level}"
    else:
        agg_path_suffix = f"{agg_type_str}/{reread_level}/all/{p_agg_level}_{p_agg}"
        agg_label = f"{agg_type_str}_{p_agg_level}_{p_agg}"

    l1_path = Path(DATA_PATH / f"{dataset}L1/features_and_targets/{agg_path_suffix}/features_and_metadata.csv")
    l2_path = Path(DATA_PATH / f"{dataset}L2/features_and_targets/{agg_path_suffix}/features_and_metadata.csv")

    if not l1_path.exists() or not l2_path.exists():
        return None, None, agg_label

    l1_df = pd.read_csv(l1_path)
    l2_df = pd.read_csv(l2_path)

    if not is_meco:
        l1_df, l2_df = filter_by_preview(l1_df, l2_df, preview)

    return l1_df, l2_df, agg_label


DATASET_PREVIEWS = {
    "OneStop": ("Gathering",),
    "Meco": ("All",),
}
DEFAULT_METHODS = [
    FoldsMethods.LEAVE_ONE_LANGUAGE_OUT,
    FoldsMethods.LEAVE_ONE_PARTICIPANT_OUT,
    FoldsMethods.RANDOM,  # L1-stratified k-fold (legacy enum name)
]
