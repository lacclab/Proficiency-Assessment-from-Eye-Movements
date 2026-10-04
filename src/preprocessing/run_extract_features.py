"""Feature extraction for one (dataset, condition, aggregation) configuration.

run_one_df filters the harmonized data to the configuration and writes its
features_and_metadata.csv; get_half_split_jobs builds the split-half jobs.
Both are driven by src/run/features.py.
"""
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from src.constants import (
    AggTypes,
    DATA_PATH,
    DataSets,
    FeatureGroups,
    Fields,
    FIXED_TEXT_GROUP_PREFIXES,
)
from src.preprocessing.extract_features import (
    add_moving_agg_features,
    add_participant_level_features,
    add_per_item_features,
)
from src.preprocessing.utils import _load_harmonized

logger.add(DATA_PATH / 'logs' / 'extract_features.log', rotation='10 MB', retention='7 days')


# ---------------------------------------------------------------- row filters


def filter_rows_full_reading(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Filter rows to only include those from participants who completed the full reading."""
    if dataset_name in [DataSets.ONESTOPL1, DataSets.ONESTOPL2]:
        # OneStop: keep participants with all 54 first-reading trials
        no_practice = df[df[Fields.PRACTICE] == False]
        participant_trial_counts = no_practice[no_practice[Fields.REREAD] == False].groupby(Fields.SUBJECT_ID)[Fields.UNIQUE_PARAGRAPH_ID].nunique()
        participants_full_reading = participant_trial_counts[participant_trial_counts == 54].index
        logger.info(f"Filtering to participants with full reading in {dataset_name}: {len(participants_full_reading)} participants with 54 trials.")
        df = df[df[Fields.SUBJECT_ID].isin(participants_full_reading)]
        logger.info(f"After filtering to full reading participants in {dataset_name}, {len(df)} rows remain.")
    elif dataset_name in [DataSets.MECOL1, DataSets.MECOL2]:
        # MECO: keep participants who read all 12 texts
        participant_text_counts = df.groupby(Fields.SUBJECT_ID)[Fields.UNIQUE_PARAGRAPH_ID].nunique()
        participants_full_reading = participant_text_counts[participant_text_counts == 12].index
        logger.info(f"Filtering to participants with full reading in {dataset_name}: {len(participants_full_reading)} participants with 12 texts.")
        df = df[df[Fields.SUBJECT_ID].isin(participants_full_reading)]
        logger.info(f"After filtering to full reading participants in {dataset_name}, {len(df)} rows remain.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return df


def filter_rows_by_conditions(df: pd.DataFrame,
                              dataset_name: str,
                              level: Literal["adv", "ele", "all"],
                              reread: Literal["ordinary", "reread", "all"],
                              keep_practice: bool = False
                              ) -> pd.DataFrame:
    """Filter a OneStop participant's trials by difficulty level, reading
    (first reading / rereading) and practice. Filters within participants only;
    participant-level splits (batch, preview) happen elsewhere. MECO has no such
    conditions and is returned unchanged."""

    if dataset_name in [DataSets.ONESTOPL1, DataSets.ONESTOPL2]:
        if level == "adv":
            df = df[df[Fields.LEVEL] == 1]
        elif level == "ele":
            df = df[df[Fields.LEVEL] == 0]
        elif level != "all":
            raise ValueError(f"Invalid value for level: {level}")

        if reread == "ordinary":
            df = df[df[Fields.REREAD] == False]
        elif reread == "reread":
            df = df[df[Fields.REREAD] == True]
        elif reread != "all":
            raise ValueError(f"Invalid value for reread: {reread}")

        if not keep_practice:
            df = df[df[Fields.PRACTICE] == False]

    return df


def filter_rows_first_p_agg(df: pd.DataFrame, p_agg_level: Literal["paragraph", "article"], num_first_agg: int, rearange_list:list=None) -> pd.DataFrame:
    """Keep each participant's first `num_first_agg` paragraphs or articles.
    `rearange_list`, if given, reorders the articles before taking the first ones."""
    if rearange_list is not None:
        df[f"old_{Fields.ARTICLE_ID}"] = df[Fields.ARTICLE_ID]
        df[Fields.ARTICLE_ID] = df[Fields.ARTICLE_ID].apply(lambda x: rearange_list.index(x) if x in rearange_list else x)

    # Validate inputs
    if not isinstance(num_first_agg, int):
        raise TypeError(f"num_first_agg must be int, got {type(num_first_agg).__name__}: {num_first_agg}")
    if p_agg_level not in ("paragraph", "article"):
        raise ValueError(f"p_agg_level must be 'paragraph' or 'article', got {p_agg_level!r}")

    if p_agg_level == "paragraph":
        if Fields.ARTICLE_ID in df.columns:
            # OneStop: keep first N unique (article_id, paragraph_id) combinations per participant
            df = df.sort_values(by=[Fields.SUBJECT_ID, Fields.ARTICLE_ID, Fields.PARAGRAPH_ID])
            first_para_pairs = df.groupby(Fields.SUBJECT_ID).apply(
                lambda group: set(map(tuple, group[[Fields.ARTICLE_ID, Fields.PARAGRAPH_ID]].drop_duplicates().values[:num_first_agg]))
            )
            df = df[df.apply(lambda row: (row[Fields.ARTICLE_ID], row[Fields.PARAGRAPH_ID]) in first_para_pairs[row[Fields.SUBJECT_ID]], axis=1)]
        else:
            # MECO: keep paragraphs with indices 1 to num_first_agg that each participant has, exclude if none
            df = df.sort_values(by=[Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID])
            first_paragraphs = df.groupby(Fields.SUBJECT_ID)[Fields.UNIQUE_PARAGRAPH_ID].apply(
                lambda x: set(range(1, num_first_agg + 1)) & set(x.unique())
            )
            df = df[df.apply(lambda row: row[Fields.UNIQUE_PARAGRAPH_ID] in first_paragraphs[row[Fields.SUBJECT_ID]], axis=1)]
    elif p_agg_level == "article":
        df = df[df[Fields.ARTICLE_ID] <= num_first_agg]
    else:
        raise ValueError(f"Invalid value for p_agg_level: {p_agg_level}")

    if rearange_list is not None:
        df[Fields.ARTICLE_ID] = df[f"old_{Fields.ARTICLE_ID}"]
        df = df.drop(columns=[f"old_{Fields.ARTICLE_ID}"])
    return df


def filter_rows_by_trials(df: pd.DataFrame,
                          agg_type: Literal[AggTypes.FIRST_P, AggTypes.MOVING_P, AggTypes.FULL, AggTypes.SEEN_UNSEEN],
                          p_agg_level: Literal["paragraph", "article"] = "paragraph",
                          p_agg: int = None,
                          half: Literal["odd", "even"] | None = None) -> pd.DataFrame:
    """Restrict the trials to those one aggregation type uses."""
    if agg_type == AggTypes.FULL:
        return df
    elif agg_type == AggTypes.FIRST_P:
        return filter_rows_first_p_agg(df, p_agg_level, num_first_agg=p_agg)
    elif agg_type == AggTypes.SEEN_UNSEEN:
        return filter_rows_seen_unseen_agg(df, half=half)
    raise ValueError(
        f"filter_rows_by_trials does not handle agg_type={agg_type!r}; "
        "moving_p_agg is processed via add_moving_agg_features upstream."
    )


SEEN_UNSEEN_PARAGRAPHS = {
    "odd": frozenset({1, 3, 5, 7, 9, 11}),
    "even": frozenset({2, 4, 6, 8, 10, 12}),
}


def filter_rows_seen_unseen_agg(df: pd.DataFrame, half: Literal["odd", "even"]) -> pd.DataFrame:
    """Keep only rows whose unique_paragraph_id falls in the chosen half.

    MECO has 12 paragraphs in fixed order; the odd/even split balances within-session
    position effects across the two halves. No coverage threshold is enforced here —
    callers can filter on the n_paragraphs column written downstream.
    """
    if half not in SEEN_UNSEEN_PARAGRAPHS:
        raise ValueError(f"half must be 'odd' or 'even', got {half!r}")
    return df[df[Fields.UNIQUE_PARAGRAPH_ID].isin(SEEN_UNSEEN_PARAGRAPHS[half])]


# ---------------------------------------------------------------- extraction


def run_one_df(harmonized_fixation_data: pd.DataFrame| None,
               harmonized_ia_data: pd.DataFrame,
               metadata_df: pd.DataFrame,
               participant_groupby_columns: list[str],
               dataset_name: str,
               level: Literal["adv", "ele", "all"],
               reread: Literal["ordinary", "reread", "all"],
               keep_practice: bool = False,
               replace_files: bool = False,
               feature_groups_to_add: list[FeatureGroups] = list(FeatureGroups),
               agg_type: Literal[AggTypes.FIRST_P, AggTypes.MOVING_P, AggTypes.FULL, AggTypes.SEEN_UNSEEN] = AggTypes.FULL,
               p_agg_level: Literal["paragraph", "article"] = None,
               p_agg: int = None,
               split_idx: int | None = None,
               half: int | None = None,
               odd_even: str | None = None,
               paragraphs_in_half: set | None = None,
               ) -> pd.DataFrame:
    """Extract and save the features of one configuration. Returns the saved
    features and metadata, or None if the files already exist."""
    if dataset_name in [DataSets.ONESTOPL1, DataSets.ONESTOPL2] or agg_type == AggTypes.MOVING_P:
        harmonized_ia_data = filter_rows_full_reading(harmonized_ia_data, dataset_name)
        harmonized_fixation_data = filter_rows_full_reading(harmonized_fixation_data, dataset_name) if harmonized_fixation_data is not None else None

    if agg_type in (AggTypes.FIRST_P, AggTypes.MOVING_P):
        if p_agg_level is None or p_agg is None:
            raise ValueError(f"{agg_type} requires p_agg_level and p_agg, got p_agg_level={p_agg_level}, p_agg={p_agg}")
        sub_dir = f"{p_agg_level}_{p_agg}/"
    elif agg_type == AggTypes.PER_ITEM:
        if p_agg_level is None:
            raise ValueError(f"per_item_agg requires p_agg_level, got p_agg_level={p_agg_level}")
        sub_dir = f"{p_agg_level}/"
    elif agg_type == AggTypes.SEEN_UNSEEN:
        if half is None:
            raise ValueError(f"seen_unseen requires half, got half={half}")
        if dataset_name not in [DataSets.MECOL1, DataSets.MECOL2]:
            raise ValueError(f"seen_unseen agg is MECO-only; got dataset_name={dataset_name}")
        sub_dir = f"{half}/"
    else:
        sub_dir = ""

    # For split-half analysis all files live under a dedicated split_half/ root so
    # they are clearly separated from regular feature files.
    # Structure:
    #   split_half/{agg_type}/{reread}/{level}/{p_agg_level}/split_{n}/half_{m}/  (OneStop)
    #   split_half/{agg_type}/{reread}/{level}/split_{n}/half_{m}/seen_unseen/{odd_even}/  (MECO odd/even)
    if split_idx is not None and half is not None:
        if odd_even is not None:
            # MECO odd/even split path
            feature_and_md_path = DATA_PATH / f'{dataset_name}/features_and_targets/split_half/{agg_type}/{reread}/{level}/split_{split_idx}/half_{half}/seen_unseen/{odd_even}/features_and_metadata.csv'
        elif p_agg_level is not None:
            # FULL (or other) split-half with explicit paragraph/article split level
            feature_and_md_path = DATA_PATH / f'{dataset_name}/features_and_targets/split_half/{agg_type}/{reread}/{level}/{p_agg_level}/split_{split_idx}/half_{half}/features_and_metadata.csv'
        else:
            # Fallback: no p_agg_level specified
            feature_and_md_path = DATA_PATH / f'{dataset_name}/features_and_targets/split_half/{agg_type}/{reread}/{level}/split_{split_idx}/half_{half}/features_and_metadata.csv'
    else:
        feature_and_md_path = DATA_PATH / f'{dataset_name}/features_and_targets/{agg_type}/{reread}/{level}/{sub_dir}features_and_metadata.csv'
    per_text_path = feature_and_md_path.parent / 'per_text_features.parquet'

    # moving_p_agg has no per-text writer (add_moving_agg_features never
    # produces a parquet), so never gate its skip/repair logic on one —
    # otherwise every no-replace run re-extracts all moving_p configs.
    regular_exists = feature_and_md_path.exists()
    per_text_exists = per_text_path.exists() or agg_type == AggTypes.MOVING_P

    if regular_exists and per_text_exists and not replace_files:
        logger.info(f"Features already exist at {feature_and_md_path}, skipping.")
        return None

    # Regular features exist but the per-text ones are missing: extract only those.
    if regular_exists and not per_text_exists and not replace_files:
        logger.info(f"Regular features exist at {feature_and_md_path}, but per-text features missing. Extracting only fixed-text features.")
        per_text_groups = [fg for fg in feature_groups_to_add if fg in FIXED_TEXT_GROUP_PREFIXES]
        if not per_text_groups:
            logger.info("No fixed-text feature groups to extract, skipping.")
            return None
        feature_groups_to_add = per_text_groups

    fixations_df = filter_rows_by_conditions(harmonized_fixation_data, dataset_name, level, reread, keep_practice) if harmonized_fixation_data is not None else None
    logger.info(f"After filtering by conditions, for dataset {dataset_name} with level {level} and reread {reread}, {len(fixations_df) if fixations_df is not None else 'None'} rows remain in fixation data")
    ia_df = filter_rows_by_conditions(harmonized_ia_data, dataset_name, level, reread, keep_practice)
    logger.info(f"After filtering by conditions, for dataset {dataset_name} with level {level} and reread {reread}, {len(ia_df)} rows remain in IA data")

    # Split-half paragraph filter: applied AFTER full_reading + conditions so the
    # 54-trial check sees the complete participant data before we restrict to one half.
    if paragraphs_in_half is not None:
        # paragraphs_in_half always contains UNIQUE_PARAGRAPH_ID values
        ia_df = ia_df[ia_df[Fields.UNIQUE_PARAGRAPH_ID].isin(paragraphs_in_half)].reset_index(drop=True)
        if fixations_df is not None:
            fixations_df = fixations_df[fixations_df[Fields.UNIQUE_PARAGRAPH_ID].isin(paragraphs_in_half)].reset_index(drop=True)

    if agg_type == AggTypes.MOVING_P:
        logger.info(f"Extracting moving aggregated features for dataset {dataset_name} with level {level} and reread {reread} and agg type {agg_type} (window_size={p_agg}, p_agg_level={p_agg_level}) ...")
        feature_and_md_df = add_moving_agg_features(
            fixations_df,
            ia_df,
            metadata_df,
            participant_groupby_columns,
            dataset_name,
            feature_and_md_path,
            p_agg_level=p_agg_level,
            window_size=p_agg,
            feature_groups_to_add=feature_groups_to_add,
            replace_file=replace_files
        )
    elif agg_type == AggTypes.PER_ITEM:
        logger.info(f"Extracting per-item features for dataset {dataset_name} with level {level} and reread {reread} and p_agg_level {p_agg_level} ...")
        feature_and_md_df = add_per_item_features(
            fixations_df,
            ia_df,
            metadata_df,
            participant_groupby_columns,
            feature_and_md_path,
            feature_groups_to_add=feature_groups_to_add,
            p_agg_level=p_agg_level,
            replace_file=replace_files,
            dataset_name=dataset_name,
        )
    else:
        fixations_df = filter_rows_by_trials(fixations_df, agg_type, p_agg_level, p_agg, half=half) if fixations_df is not None else None
        ia_df = filter_rows_by_trials(ia_df, agg_type, p_agg_level, p_agg, half=half)
        logger.info(f"After filtering by trials, for dataset {dataset_name} with level {level} and reread {reread} and agg type {agg_type} ({p_agg_level=}, {p_agg=}, {half=}), {len(fixations_df) if fixations_df is not None else 'None'} rows remain in fixation data.")
        logger.info(f"After filtering by trials, for dataset {dataset_name} with level {level} and reread {reread} and agg type {agg_type} ({p_agg_level=}, {p_agg=}, {half=}), {len(ia_df)} rows remain in IA data.")

        logger.info(f"Extracting features for dataset {dataset_name} with level {level} and reread {reread} and agg type {agg_type} ...")
        feature_and_md_df = add_participant_level_features(
            fixations_df,
            ia_df,
            metadata_df,
            participant_groupby_columns,
            feature_and_md_path,
            feature_groups_to_add=feature_groups_to_add,
            replace_file=replace_files,
            dataset_name=dataset_name,
        )

        if agg_type == AggTypes.SEEN_UNSEEN:
            n_paragraphs = (
                ia_df.groupby(Fields.SUBJECT_ID)[Fields.UNIQUE_PARAGRAPH_ID]
                .nunique()
                .rename("n_paragraphs")
                .reset_index()
            )
            feature_and_md_df = feature_and_md_df.drop(columns=["n_paragraphs"], errors="ignore").merge(
                n_paragraphs, on=Fields.SUBJECT_ID, how="left",
            )
            feature_and_md_df["n_paragraphs"] = feature_and_md_df["n_paragraphs"].fillna(0).astype(int)
            feature_and_md_df.to_csv(feature_and_md_path, index=False)

    logger.info(f"Finished extracting features for dataset {dataset_name} with level {level} and reread {reread} and agg type {agg_type}. The features and metadata are saved to {feature_and_md_path}.")

    return feature_and_md_df


def prepare_agg_files_for_dataset(dataset_name: str, dataset_path: Path):
    """Load a dataset's harmonized fixation, IA and metadata frames (with
    IA_RR), plus the participant grouping columns. Does not apply the OneStop
    L1/L2 cascade alignment."""
    logger.info(f"Preparing agg files for dataset {dataset_path} ...")
    harmonized_fixation_data, harmonized_ia_data, metadata_df = _load_harmonized(dataset_name)
    participant_groupby_columns = [Fields.SUBJECT_ID]
    return harmonized_fixation_data, harmonized_ia_data, metadata_df, participant_groupby_columns


def get_half_split_jobs(
    dataset_name: str,
    level: Literal["adv", "ele", "all"],
    agg_config: dict,
    df_fixations: pd.DataFrame | None,
    df_ia: pd.DataFrame,
    metadata_df: pd.DataFrame,
    n_splits: int,
    split_definitions: dict,
    reread: Literal["ordinary", "reread", "all"] = "ordinary",
    split_definitions_odd: dict = None,
    split_definitions_even: dict = None,
    skip_fixed_features: bool = False,
) -> list[tuple]:
    """Generate half-split jobs for parallel feature extraction.

    Jobs carry the full (unfiltered) dataframes plus a paragraphs_in_half set
    in agg_config. Workers apply the paragraph filter on-demand, avoiding the
    cost of creating 160 filtered DataFrame copies in the parent process.

    Args:
        dataset_name: Dataset name (e.g., 'OneStopL1')
        level: Reading level ('adv', 'ele', 'all')
        agg_config: Aggregation config dict (will be extended with split info)
        df_fixations: Full fixation dataframe (passed as-is to each job)
        df_ia: Full IA dataframe (passed as-is to each job)
        metadata_df: Metadata dataframe
        n_splits: Total number of splits
        split_definitions: Dict from generate_split_definitions: {split_idx: {half_num: set(paragraph_ids)}}
        reread: Reread type ('ordinary', 'reread', 'all')
        split_definitions_odd: Split definitions for odd paragraphs (MECO only)
        split_definitions_even: Split definitions for even paragraphs (MECO only)
        skip_fixed_features: Skip TRANSITIONS/WFC extraction, reusing the
            per-text parquets already on disk (default: False)

    Returns:
        List of job tuples: (dataset_name, level, agg_config_with_split,
                            df_fixations, df_ia, metadata_df, reread, skip_fixed_features)
    """
    jobs = []
    is_meco = dataset_name in [DataSets.MECOL1, DataSets.MECOL2]

    # For MECO: create jobs for odd and even separately
    if is_meco and split_definitions_odd is not None and split_definitions_even is not None:
        for odd_even, split_defs in [("odd", split_definitions_odd), ("even", split_definitions_even)]:
            for split_idx in range(1, n_splits + 1):
                for half_num in [1, 2]:
                    agg_config_with_split = agg_config.copy()
                    agg_config_with_split['split_idx'] = split_idx
                    agg_config_with_split['half'] = half_num
                    agg_config_with_split['odd_even'] = odd_even
                    agg_config_with_split['paragraphs_in_half'] = split_defs[split_idx][half_num]

                    jobs.append((
                        dataset_name,
                        level,
                        agg_config_with_split,
                        df_fixations,
                        df_ia,
                        metadata_df,
                        reread,
                        skip_fixed_features,
                    ))

                    logger.debug(f"Created job: {dataset_name} split {split_idx} half {half_num} ({odd_even})")
    else:
        # OneStop or MECO without odd/even: regular split-half
        for split_idx in range(1, n_splits + 1):
            for half_num in [1, 2]:
                agg_config_with_split = agg_config.copy()
                agg_config_with_split['split_idx'] = split_idx
                agg_config_with_split['half'] = half_num
                agg_config_with_split['paragraphs_in_half'] = split_definitions[split_idx][half_num]

                jobs.append((
                    dataset_name,
                    level,
                    agg_config_with_split,
                    df_fixations,
                    df_ia,
                    metadata_df,
                    reread,
                    skip_fixed_features,
                ))

                logger.debug(f"Created job: {dataset_name} split {split_idx} half {half_num}")

    return jobs
