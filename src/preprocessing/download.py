import argparse
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from src.constants import DataSets
from src.preprocessing.dataset_preprocessing.base import DatasetProcessor
from src.preprocessing.dataset_preprocessing.onestop import (
    ONESTOP_L2_UNAVAILABLE,
    OneStopL1Processor,
    OneStopL2Processor,
)
from src.preprocessing.dataset_preprocessing.meco import MecoUnifiedProcessor
from src.preprocessing.utils import _load_harmonized, add_content_group_to_metadata, realign_ia_cascade_onestop_paired

MECO_COHORTS_TOGETHER = (
    f"MECO's L1 and L2 readers are prepared together, into data/{DataSets.MECOL1}/ and "
    f"data/{DataSets.MECOL2}/: use --dataset {DataSets.MECO}."
)


def prepare_dataset_object(dataset_name: str) -> DatasetProcessor:
    """The processor for one dataset name."""
    if dataset_name == DataSets.ONESTOPL1:
        processor = OneStopL1Processor(DataSets.ONESTOPL1)
    elif dataset_name == DataSets.ONESTOPL2:
        processor = OneStopL2Processor(DataSets.ONESTOPL2)
    elif dataset_name == DataSets.MECO:
        processor = MecoUnifiedProcessor()
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return processor


def run_paired_cascade_fix(processorL1: DatasetProcessor, processorL2: DatasetProcessor) -> None:
    """Align the OneStop L1 and L2 IA layouts together (identical renumbering in
    both cohorts; see realign_ia_cascade_onestop_paired) and overwrite their
    harmonized and metadata files."""
    fix_l1, ia_l1, md_l1 = _load_harmonized(DataSets.ONESTOPL1)
    fix_l2, ia_l2, md_l2 = _load_harmonized(DataSets.ONESTOPL2)

    md_l1 = add_content_group_to_metadata(ia_l1, md_l1)
    md_l2 = add_content_group_to_metadata(ia_l2, md_l2)

    before_ia = len(ia_l1) + len(ia_l2)
    before_fix = (
        (len(fix_l1) if fix_l1 is not None else 0)
        + (len(fix_l2) if fix_l2 is not None else 0)
    )
    ia_l1, fix_l1, ia_l2, fix_l2 = realign_ia_cascade_onestop_paired(
        ia_l1, fix_l1, md_l1, ia_l2, fix_l2, md_l2,
    )
    after_ia = len(ia_l1) + len(ia_l2)
    after_fix = (
        (len(fix_l1) if fix_l1 is not None else 0)
        + (len(fix_l2) if fix_l2 is not None else 0)
    )
    logger.info(
        f"Paired cascade fix dropped IA rows: {before_ia - after_ia}, "
        f"fixation rows: {before_fix - after_fix}"
    )

    if fix_l1 is not None:
        fix_l1.to_csv(f"{processorL1.harmonized_data_path}/fixations.csv", index=False)
    if ia_l1 is not None:
        ia_l1.to_csv(f"{processorL1.harmonized_data_path}/ia.csv", index=False)
    if fix_l2 is not None:
        fix_l2.to_csv(f"{processorL2.harmonized_data_path}/fixations.csv", index=False)
    if ia_l2 is not None:
        ia_l2.to_csv(f"{processorL2.harmonized_data_path}/ia.csv", index=False)
    if md_l1 is not None:
        md_l1.to_csv(f"{processorL1.processed_metadata_path}/metadata.csv", index=False)
    if md_l2 is not None:
        md_l2.to_csv(f"{processorL2.processed_metadata_path}/metadata.csv", index=False)


def main() -> int:
    """Download, harmonize and process the metadata of each dataset in
    --dataset (default: OneStop L1 and MECO); with both OneStop cohorts, align
    their IA layouts."""
    data_path = Path('data')
    data_path.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()

    datasets = args.dataset

    if datasets:
        datasets_names = datasets.split(',')
    else:
        datasets_names = [
            DataSets.ONESTOPL1,
            DataSets.MECO,
        ]
    if DataSets.ONESTOPL2 in datasets_names:
        parser.error(ONESTOP_L2_UNAVAILABLE)
    if DataSets.MECOL1 in datasets_names or DataSets.MECOL2 in datasets_names:
        parser.error(MECO_COHORTS_TOGETHER)

    for dataset_name in tqdm(
        datasets_names,
        desc='Downloading datasets',
        unit='dataset',
        total=len(datasets_names),
    ):
        dataset = prepare_dataset_object(dataset_name)
        logger.info(f'Downloading {dataset_name} ...')
        dataset.download()

        logger.info(f'Downloading metadata for {dataset_name} ...')
        dataset.download_metadata()

        logger.info(f'Processing data for {dataset_name} ...')
        dataset.process_data()

        logger.info(f'Processing metadata for {dataset_name} ...')
        dataset.process_metadata()

    # Align the L1 and L2 IA layouts together, once both are processed
    if DataSets.ONESTOPL1 in datasets_names and DataSets.ONESTOPL2 in datasets_names:
        processorL1 = prepare_dataset_object(DataSets.ONESTOPL1)
        processorL2 = prepare_dataset_object(DataSets.ONESTOPL2)
        run_paired_cascade_fix(processorL1, processorL2)
    elif DataSets.ONESTOPL1 in datasets_names:
        logger.warning(
            'OneStop L1 was processed without L2, so its IA layout was not aligned '
            'with the L2 one (run_paired_cascade_fix) and its metadata has no '
            'content_group: these are not the OneStop L1 files the paper used.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
