from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import (
    DataType,
    Fields,
    SURPRISAL_MODELS,
    OSF_RESOURCE_IDS,
    IA_FEATURES_TO_ADD_TO_FIXATION_REPORT,
    MERGE_IA_FIXATION_COLS,
    TRIAL_IDENTIFIER_COLS,
    MICH_TEST_PARTS_EXTENDED,
    TestCols,
)
from src.preprocessing.dataset_preprocessing.base import DatasetProcessor
from src.preprocessing.utils import (
    ArgsParser,
    add_missing_features,
    download_from_osf,
    process_onestop_report,
)

# Columns the L2 metadata fills from the reported proficiency exams.
REPORTED_EXAM_COLUMNS = [
    'converted_toefl_score', 'test', 'listening', 'reading', 'speaking', 'writing', 'year',
    'toefl_lr',
]


def l1_proficiency_metadata(session_summary: pd.DataFrame) -> pd.DataFrame:
    """The OneStop L1 proficiency metadata, one row per session: its LexTALE and
    comprehension scores, with the Michigan-test and reported-exam columns of
    the L2 metadata left empty. Empty cells are written as -1.0, as in the L2
    metadata."""
    sessions = session_summary[[
        Fields.SUBJECT_ID, TestCols.PREVIEW_COL, TestCols.BATCH_COL, TestCols.LEXTALE_COL,
        TestCols.COMPREHENSION_COL, TestCols.COMPREHENSION_COL_REREAD,
    ]].dropna(subset=[Fields.SUBJECT_ID])
    michigan = {col: np.nan for part in MICH_TEST_PARTS_EXTENDED for col in (part, f'log_{part}')}
    metadata = pd.concat([
        pd.DataFrame(michigan, index=sessions.index),
        sessions[Fields.SUBJECT_ID].map(lambda pid: int(pid.split('_')[-1]))
        .rename('participant_id_clean'),
        sessions,
    ], axis=1)
    metadata[TestCols.LANGUAGEC_COL] = 'English'
    metadata[[TestCols.MICHIGEN_TEST_COL, f'log_{TestCols.MICHIGEN_TEST_COL}',
              *REPORTED_EXAM_COLUMNS]] = np.nan
    metadata = metadata.drop_duplicates()
    metadata[TestCols.LEXTALE_COL] = metadata[TestCols.LEXTALE_COL].astype(float)
    floats = metadata.select_dtypes('float').columns
    metadata[floats] = metadata[floats].round(5).fillna(-1.0)
    return metadata


class OneStopL1Processor(DatasetProcessor):
    """Processor for the OneStop L1 dataset."""

    def __init__(self, dataset_name: str):
        dataset_path = Path.cwd() / "data" / dataset_name
        raw_paths = {
            DataType.IA: dataset_path / 'downloads/ia_Paragraph.csv',
            DataType.FIXATIONS: dataset_path / 'downloads/fixations_Paragraph.csv',
        }
        self.onestopqa_path = Path(f'{dataset_path}/additional_raw/onestop_qa.json')

        super().__init__(raw_paths, dataset_name)

    def get_column_map(self, data_type: DataType) -> dict:
        """OneStop's reports already use the standard column names."""
        return {}

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """OneStop-specific processing steps."""
        surprisal_models = SURPRISAL_MODELS

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            if data_type not in data_dict or data_dict[data_type] is None:
                continue

            df = data_dict[data_type]

            args = [
                '--mode',
                data_type,
                '--SURPRISAL_MODELS',
                *surprisal_models,
                '--onestopqa_path',
                str(self.onestopqa_path),
            ]
            cfg = ArgsParser().parse_args(args)

            # Keep the release's real Penn Treebank tags in ptb_pos; the
            # reduced 4-category POS (Reduced_POS, IA report only) is not used downstream.
            if data_type == DataType.IA:
                df = df.drop(columns=['Reduced_POS'])
            df = process_onestop_report(df=df, args=cfg)

            df['unique_trial_id'] = (
                df['participant_id'].astype(str)
                + '_'
                + df['unique_paragraph_id'].astype(str)
                + '_'
                + df['repeated_reading_trial'].astype(str)
                + '_'
                + df['practice_trial'].astype(str)
            )

            # OneStop lists the correct answer as option A.
            df['is_correct'] = (df.selected_answer == 'A').astype(int)

            df[Fields.LEVEL] = df[Fields.LEVEL].map({'Adv': 1, 'Ele': 0}).astype(int)
            if data_type == DataType.IA:
                df['head_direction'] = df['distance_to_head'] > 0
                df['head_direction'] = df['head_direction'].astype(int)

            data_dict[data_type] = df

        data_dict['fixations'] = self.add_ia_report_features_to_fixation_data(
            data_dict['ia'], data_dict['fixations']
        )

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            data_dict[data_type] = add_missing_features(
                et_data=data_dict[data_type],
                mode=data_type,
            )

        return data_dict

    def download(self) -> None:
        """Download the OneStop L1 dataset (IA and Fixations CSVs) from OSF."""
        resource_ids = OSF_RESOURCE_IDS.get(self.dataset_name, {})
        urls = {
            f"{DataType.IA}_Paragraph.csv.zip": resource_ids.get(DataType.IA),
            f"{DataType.FIXATIONS}_Paragraph.csv.zip": resource_ids.get(DataType.FIXATIONS)}

        download_from_osf(self.dataset_path, urls)

    def load_raw_data(self) -> dict[str, pd.DataFrame]:
        """Load raw IA and Fixations data from CSV files."""
        data_dict = {}
        for data_type in [DataType.IA, DataType.FIXATIONS]:
            raw_path = self.raw_paths.get(data_type)
            if raw_path is not None and Path(raw_path).exists():
                logger.info(f'Loading {data_type} data from {raw_path}')
                data_dict[data_type] = pd.read_csv(raw_path)
            else:
                logger.warning(f'Raw path for {data_type} not found: {raw_path}')
                data_dict[data_type] = None
        return data_dict

    def download_metadata(self) -> None:
        """Download the session summary from OSF."""
        resource_ids = OSF_RESOURCE_IDS.get(self.dataset_name, {})
        urls = {"session_summary.csv": resource_ids.get(DataType.METADATA)}

        download_from_osf(self.dataset_path, urls, is_metadata=True)

    def process_metadata(self) -> pd.DataFrame:
        """Write the proficiency metadata (metadata/metadata.csv) from the session summary."""
        session_summary = pd.read_csv(self.dataset_path / 'downloads/metadata/session_summary.csv')
        metadata = l1_proficiency_metadata(session_summary)
        out = self.processed_metadata_path / 'metadata.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(out, index=False)
        logger.info(f'Wrote {len(metadata)} rows to {out}')
        return metadata

    def add_ia_report_features_to_fixation_data(
        self, ia_df: pd.DataFrame, fix_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge per-IA (interest-area) features into the fixation-level data.

        Result: one row per fixation, enriched with IA-level attributes.
        """
        # 1. Unify the IA-ID column name
        ia_df = ia_df.rename(
            columns={
                Fields.IA_DATA_IA_ID_COL_NAME: Fields.FIXATION_REPORT_IA_ID_COL_NAME
            }
        )

        # 2. Keep the merge keys and the IA features to add
        required_cols = (
            MERGE_IA_FIXATION_COLS + IA_FEATURES_TO_ADD_TO_FIXATION_REPORT
        )
        ia_df = ia_df[list(set(required_cols))]

        # 3. Drop columns that also exist in the fixation table
        merge_keys = set(MERGE_IA_FIXATION_COLS)
        dup_cols = (set(fix_df.columns) & set(ia_df.columns)) - merge_keys
        ia_df = ia_df.drop(columns=list(dup_cols))

        # 4. Drop a nuisance column
        if 'normalized_part_ID' in fix_df.columns:
            if fix_df['normalized_part_ID'].isna().any():
                logger.warning('normalized_part_ID contains NaNs; dropping it.')
            fix_df = fix_df.drop(columns='normalized_part_ID')

        # 5. Merge
        enriched_fix_df = fix_df.merge(
            ia_df,
            on=list(merge_keys),
            how='left',
            validate='many_to_one',
        )

        # num_of_words_in_trial = number of IA rows per trial
        num_of_words_in_trials_series = ia_df.groupby(
            TRIAL_IDENTIFIER_COLS,
        ).size()
        num_of_words_in_trials_series.name = 'num_of_words_in_trial'
        enriched_fix_df = enriched_fix_df.merge(
            num_of_words_in_trials_series,
            on=TRIAL_IDENTIFIER_COLS,
            how='left',
        )

        return enriched_fix_df


ONESTOP_L2_UNAVAILABLE = (
    "OneStop L2 is not publicly available, so it can't be downloaded or processed "
    "here (OneStop L1 and MECO can)."
)


class OneStopL2Processor(OneStopL1Processor):
    """Processor for the OneStop L2 dataset. The dataset is not publicly
    available, so downloading it and processing its metadata raise."""

    def download(self) -> None:
        raise RuntimeError(ONESTOP_L2_UNAVAILABLE)

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """OneStop L2-specific processing steps."""
        data_dict = super().dataset_specific_processing(data_dict)

        # Each fixation's IA label comes from IA_LABEL, with DataViewer's "."
        # for fixations outside any interest area.
        fixations_df = data_dict[DataType.FIXATIONS].copy()
        fixations_df['CURRENT_FIX_INTEREST_AREA_LABEL'] = fixations_df['IA_LABEL'].fillna(".")
        data_dict[DataType.FIXATIONS] = fixations_df
        return data_dict

    def download_metadata(self) -> None:
        raise RuntimeError(ONESTOP_L2_UNAVAILABLE)

    def process_metadata(self) -> pd.DataFrame:
        raise RuntimeError(ONESTOP_L2_UNAVAILABLE)
