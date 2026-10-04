"""Stage 1: extract features from the harmonized eye-tracking data.

Reads data/<dataset>/harmonized/ and writes data/<dataset>/features_and_targets/,
which every later stage (predictions, eyescore) reads. Parallelised over
(dataset, level, aggregation config, reread) jobs; each dataset is loaded once
in the parent process and shared read-only with the workers.

    python -m src.run.features                                  # all four datasets
    python -m src.run.features --datasets OneStopL1,OneStopL2 --agg-types fully_agg
    python -m src.run.features --half-split --half-split-n-splits 20   # split-half reliability
"""
import argparse
import multiprocessing
import traceback
import warnings
from datetime import datetime

import pandas as pd
from loguru import logger

from src.agg_configs import get_agg_configs
from src.constants import (
    DATA_PATH, FIXED_TEXT_GROUP_PREFIXES, AggTypes, DataSets, FeatureGroups, Fields,
)
from src.preprocessing.extract_features import generate_split_definitions
from src.preprocessing.run_extract_features import (
    _load_harmonized,
    get_half_split_jobs,
    prepare_agg_files_for_dataset,
    run_one_df,
)
from src.utils.cli import parse_csv

# Suppress verbose pandas SettingWithCopyWarning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

logger.add(DATA_PATH / 'logs' / 'features.log', rotation='10 MB', retention='7 days')

ALL_DATASETS = [DataSets.ONESTOPL1, DataSets.ONESTOPL2, DataSets.MECOL1, DataSets.MECOL2]


def extract_worker(
    dataset_name: str,
    level: str,
    agg_config: dict,
    df_fixations: pd.DataFrame,
    df_ia: pd.DataFrame,
    metadata_df: pd.DataFrame,
    reread: str = "ordinary",
    skip_fixed_features: bool = False,
    replace_files: bool = False
) -> None:
    """Worker function for multiprocessing - extracts features for a single (dataset, level, agg_config).

    Args:
        dataset_name: Dataset name (e.g., 'OneStopL1')
        level: Single reading level (e.g., 'adv')
        agg_config: Single aggregation config dict
        df_fixations: Loaded fixations DataFrame (read-only, shared across workers)
        df_ia: Loaded IA DataFrame (read-only, shared across workers)
        metadata_df: Loaded metadata DataFrame (read-only, shared across workers)
        reread: Reread type (default: 'ordinary')
        skip_fixed_features: Skip extraction of fixed feature groups (default: False)
        replace_files: Whether to replace existing files
    """
    try:
        agg_type = agg_config.get("agg_type")
        p_agg_level = agg_config.get("p_agg_level")
        p_agg = agg_config.get("p_agg")
        split_idx = agg_config.get("split_idx")
        half = agg_config.get("half")
        odd_even = agg_config.get("odd_even")
        paragraphs_in_half = agg_config.get("paragraphs_in_half")

        feature_groups = list(FeatureGroups)
        if skip_fixed_features:
            feature_groups = [fg for fg in feature_groups if fg not in FIXED_TEXT_GROUP_PREFIXES]

        run_one_df(
            df_fixations,
            df_ia,
            metadata_df,
            [Fields.SUBJECT_ID],
            dataset_name,
            level,
            reread,
            keep_practice=False,
            replace_files=replace_files,
            feature_groups_to_add=feature_groups,
            agg_type=agg_type,
            p_agg_level=p_agg_level,
            p_agg=p_agg,
            split_idx=split_idx,
            half=half,
            odd_even=odd_even,
            paragraphs_in_half=paragraphs_in_half,
        )
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in extract_worker for {dataset_name}/{level}/{agg_type}/{reread}")
        print(f"{'='*60}")
        print(f"agg_config: {agg_config}")
        print(f"agg_type={agg_type}, p_agg_level={p_agg_level}, p_agg={p_agg}, half={half}")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise


def run_extract_features_parallel(
    datasets: list[str],
    n_jobs: int = -1,
    agg_types: set[str] = None,
    levels: list[str] = None,
    rereads: list[str] = None,
    skip_fixed_features: bool = False,
    n_splits: int = 20,
    half_split: bool = False,
    replace_files: bool = False,
) -> None:
    """Run feature extraction for multiple datasets in parallel using multiprocessing.Pool.

    Data is loaded once per dataset in the parent process and shared (read-only) across workers.
    Parallelization happens at the (dataset, level, agg_config) level.

    Args:
        datasets: List of dataset names
        n_jobs: Number of parallel jobs (-1 = use all CPUs)
        agg_types: Aggregation types to include (e.g., {AggTypes.FULL, AggTypes.FIRST_P}). Default: all
        levels: Reading levels to extract (e.g., ['adv', 'ele', 'all']; only for OneStop)
        rereads: Reread types to process (default: ['ordinary'])
        skip_fixed_features: Skip extraction of fixed feature sets (default: False)
        half_split: Extract split-half features for reliability analysis (default: False)
        replace_files: Replace existing feature files (default: False)
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting parallel feature extraction for {len(datasets)} datasets with {n_jobs} workers")
    logger.info(f"  Agg types: {agg_types if agg_types else 'all'}")
    if levels:
        logger.info(f"  Levels: {levels}")
    logger.info(f"{'='*60}\n")

    errors = []

    # Prepare jobs: load data per dataset, create jobs for each (level, agg_config) combo
    jobs = []

    # Pre-generate split definitions for OneStop paired alignment (L1 splits apply to L2)
    split_definitions = {}
    if half_split and any(d in [DataSets.ONESTOPL1, DataSets.ONESTOPL2] for d in datasets):
        logger.info("Generating split definitions from OneStop L1 for L1+L2 alignment...")
        _, ia_l1, _ = _load_harmonized(DataSets.ONESTOPL1)

        # Generate for all possible p_agg_levels (paragraph and article)
        split_definitions[DataSets.ONESTOPL1] = {}
        split_definitions[DataSets.ONESTOPL2] = {}
        for p_agg_level in ["paragraph", "article"]:
            logger.debug(f"  Generating {p_agg_level}-level splits...")
            defs = generate_split_definitions(ia_l1, DataSets.ONESTOPL1, n_splits, p_agg_level=p_agg_level)
            split_definitions[DataSets.ONESTOPL1][p_agg_level] = defs
            split_definitions[DataSets.ONESTOPL2][p_agg_level] = defs  # Use same splits for L2

    # Pre-generate split definitions for MECO paired alignment (L1 splits apply to L2)
    if half_split and any(d in [DataSets.MECOL1, DataSets.MECOL2] for d in datasets):
        logger.info("Generating split definitions from MECO L1 for L1+L2 alignment...")
        _, ia_l1, _ = _load_harmonized(DataSets.MECOL1)

        # MECO uses paragraph level, plus odd/even for split-half
        split_definitions[DataSets.MECOL1] = {}
        split_definitions[DataSets.MECOL2] = {}
        defs = generate_split_definitions(ia_l1, DataSets.MECOL1, n_splits, p_agg_level="paragraph")
        split_definitions[DataSets.MECOL1]["paragraph"] = defs
        split_definitions[DataSets.MECOL2]["paragraph"] = defs  # Use same splits for L2

        # For MECO split-half: also generate odd/even splits
        logger.debug("  Generating odd/even splits for MECO...")
        defs_odd = generate_split_definitions(ia_l1, DataSets.MECOL1, n_splits, p_agg_level="paragraph", odd_even="odd")
        defs_even = generate_split_definitions(ia_l1, DataSets.MECOL1, n_splits, p_agg_level="paragraph", odd_even="even")
        split_definitions[DataSets.MECOL1]["odd"] = defs_odd
        split_definitions[DataSets.MECOL1]["even"] = defs_even
        split_definitions[DataSets.MECOL2]["odd"] = defs_odd
        split_definitions[DataSets.MECOL2]["even"] = defs_even

    for dataset in datasets:
        # Get agg configs for this dataset, filtered by agg_types
        agg_configs = get_agg_configs(dataset, agg_types)

        # Determine levels and rereads: OneStop uses provided values, Meco always uses "all"
        is_onestop = dataset in [DataSets.ONESTOPL1, DataSets.ONESTOPL2]
        dataset_levels = levels if is_onestop else ["all"]
        dataset_rereads = rereads if is_onestop else ["all"]

        # Load data ONCE per dataset in parent process
        logger.info(f"Loading data for {dataset}...")

        dataset_path = DATA_PATH / dataset
        df_fixations, df_ia, metadata_df, _ = prepare_agg_files_for_dataset(dataset, dataset_path)

        # Create a job for each (level, agg_config, reread) combo with the loaded data
        for level in dataset_levels:
            for agg_config in agg_configs:
                for reread in dataset_rereads:
                    if half_split:
                        agg_type = agg_config.get("agg_type")
                        is_meco = dataset in [DataSets.MECOL1, DataSets.MECOL2]
                        split_defs_odd = split_definitions[dataset].get("odd") if is_meco else None
                        split_defs_even = split_definitions[dataset].get("even") if is_meco else None

                        # For FULL agg (no inherent p_agg_level), generate jobs for both
                        # paragraph and article split levels so both strategies are covered.
                        # Other agg types use their own p_agg_level from the config.
                        if agg_type == AggTypes.FULL and is_onestop:
                            split_levels = ["paragraph", "article"]
                        else:
                            split_levels = [agg_config.get("p_agg_level", "paragraph")]

                        for split_level in split_levels:
                            split_defs_for_config = split_definitions[dataset].get(split_level)
                            if split_defs_for_config is None:
                                logger.warning(f"No split definitions for {dataset} p_agg_level={split_level}, skipping")
                                continue

                            # Inject p_agg_level into the config so run_one_df uses it in the path
                            agg_config_for_split = {**agg_config, "p_agg_level": split_level} if agg_type == AggTypes.FULL else agg_config

                            half_jobs = get_half_split_jobs(
                                dataset, level, agg_config_for_split, df_fixations, df_ia, metadata_df, n_splits,
                                split_defs_for_config, reread=reread,
                                split_definitions_odd=split_defs_odd, split_definitions_even=split_defs_even,
                                skip_fixed_features=skip_fixed_features,
                            )
                            # Add replace_files to each half-split job
                            half_jobs_with_replace = [(job[0], job[1], job[2], job[3], job[4], job[5], job[6], job[7], replace_files) for job in half_jobs]
                            jobs.extend(half_jobs_with_replace)
                    else:
                        # For regular extraction: create jobs for each (level, agg_config, reread) combo
                        jobs.append((dataset, level, agg_config, df_fixations, df_ia, metadata_df, reread, skip_fixed_features, replace_files))

    total_jobs = len(jobs)
    logger.info(f"Total jobs: {total_jobs} (dataset × level × agg_config × reread combinations)")

    # Run jobs in parallel using multiprocessing.Pool
    # Workers will share read-only copies of the loaded DataFrames via copy-on-write
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = []
        for job_args in jobs:
            result = pool.apply_async(extract_worker, job_args)
            results.append(result)

        # Collect results
        for i, result in enumerate(results):
            dataset, level, agg_config, _, _, _, reread, _, _ = jobs[i]
            agg_type = agg_config.get("agg_type")
            split_idx = agg_config.get("split_idx")
            half = agg_config.get("half")

            # Build job descriptor for logging
            if split_idx is not None and half is not None:
                job_desc = f"{dataset} | level={level} | agg_type={agg_type} | split={split_idx} | half={half}"
                job_key = f"{dataset}/{level}/{agg_type}/split{split_idx}_half{half}"
            else:
                job_desc = f"{dataset} | level={level} | agg_type={agg_type} | reread={reread}"
                job_key = f"{dataset}/{level}/{agg_type}/{reread}"

            try:
                result.get(timeout=None)
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ {job_desc}")
            except Exception as e:
                errors.append((job_key, e))
                logger.error(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ {job_desc} failed: {e}")

    if errors:
        logger.error(f"\n{len(errors)} errors occurred:")
        for job, err in errors:
            logger.error(f"  {job}: {err}")
        raise RuntimeError(f"Feature extraction failed for {len(errors)} jobs")

    logger.info(f"\n{'='*60}")
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Feature extraction completed for all {total_jobs} jobs")
    logger.info(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', type=str, default=",".join(ALL_DATASETS),
                        help='Datasets to extract, by their L1/L2 name '
                             f'(default: {",".join(ALL_DATASETS)})')
    parser.add_argument('--agg-types', type=str,
                        help='Aggregation types: fully_agg, first_p_agg, moving_p_agg, '
                             'per_item_agg, seen_unseen (MECO only) (default: all)')
    parser.add_argument('--levels', type=str, default='all',
                        help='OneStop reading levels: ele, adv, all (default: all)')
    parser.add_argument('--rereads', type=str, default='ordinary',
                        help='OneStop reread trials: ordinary, reread, all (default: ordinary)')
    parser.add_argument('--half-split', action='store_true',
                        help='Extract split-half features for reliability analysis')
    parser.add_argument('--half-split-n-splits', type=int, default=20,
                        help='Random splits for --half-split (default: 20)')
    parser.add_argument('--skip-fixed-features', action='store_true',
                        help='Skip the per-text feature sets (TRANSITIONS, WFC)')
    parser.add_argument('--replace-existing', action='store_true',
                        help='Recompute feature files that already exist')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel workers: -1 = all CPUs (default: -1)')
    args = parser.parse_args()

    run_extract_features_parallel(
        parse_csv(args.datasets),
        n_jobs=args.n_jobs,
        agg_types=set(parse_csv(args.agg_types)) if args.agg_types else None,
        levels=parse_csv(args.levels),
        rereads=parse_csv(args.rereads),
        skip_fixed_features=args.skip_fixed_features,
        n_splits=args.half_split_n_splits,
        half_split=args.half_split,
        replace_files=args.replace_existing,
    )


if __name__ == '__main__':
    main()
