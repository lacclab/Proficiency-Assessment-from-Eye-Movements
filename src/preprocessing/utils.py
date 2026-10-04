import gzip
import os
import tempfile
import zipfile
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import requests
from loguru import logger
from tap import Tap
from tqdm import tqdm

from src.constants import DATA_PATH, DataType, Fields

IA_ID_COL = 'IA_ID'
FIXATION_ID_COL = 'CURRENT_FIX_INTEREST_AREA_INDEX'
NEXT_FIXATION_ID_COL = 'NEXT_FIX_INTEREST_AREA_INDEX'


class Mode(Enum):
    """Whether to process interest-area (IA) or fixation data."""

    IA = 'ia'
    FIXATION = 'fixations'


class ArgsParser(Tap):
    """Options for processing a OneStop report (see process_onestop_report)."""

    SURPRISAL_MODELS: list[str] = ['gpt2']  # models to take surprisal from
    onestopqa_path: Path = Path('metadata/onestop_qa.json')
    mode: Mode = Mode.IA  # whether to process interest-area or fixation data


# ---------------------------------------------------------------- general



def add_missing_features(
    et_data: pd.DataFrame,
    mode: DataType,
) -> pd.DataFrame:
    """Cast is_content_word to integer and, for fixation data, add is_reg and
    is_progressive (whether the next fixation lands on an earlier / later IA).

    Args:
        et_data: IA or fixation data. Needs is_content_word and, for fixation
            data, CURRENT_FIX_INTEREST_AREA_INDEX and NEXT_FIX_INTEREST_AREA_INDEX.
        mode: DataType.IA or DataType.FIXATIONS.
    """
    et_data['is_content_word'] = et_data['is_content_word'].astype('Int64')

    if mode == DataType.FIXATIONS:
        et_data['is_reg'] = (
            et_data['NEXT_FIX_INTEREST_AREA_INDEX']
            < et_data['CURRENT_FIX_INTEREST_AREA_INDEX']
        )
        et_data['is_progressive'] = (
            et_data['NEXT_FIX_INTEREST_AREA_INDEX']
            > et_data['CURRENT_FIX_INTEREST_AREA_INDEX']
        )

    return et_data


# ---------------------------------------------------------------- OneStop


def process_onestop_report(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """Clean a OneStop IA or fixation report and add the derived columns.

    Steps: integer and float conversions, 0-based indexing, dropping fixations
    without an IA, unique_paragraph_id, per-trial span metrics, the normalized
    word position, and (IA reports only) line position and the additional
    metrics of add_additional_metrics.
    """

    duration_field, ia_field = get_constants_by_mode(args.mode)

    df = convert_to_int_features(df, args)
    df = convert_to_float_features(df, args)
    df = adjust_indexing(df, args)
    df = drop_missing_fixation_data(df, args)
    df = add_unique_paragraph_id(df)
    df = compute_span_level_metrics(df, ia_field, args.mode, duration_field)
    df = compute_normalized_features(df, duration_field, ia_field)
    if args.mode == Mode.IA:
        df = compute_start_end_line(df)
        df = add_additional_metrics(df)

    return df


def convert_to_int_features(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Convert specified columns to integer type.

    Handles missing values and dots by replacing them with 0 before conversion.
    Different columns are processed based on whether in IA or FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with converted integer columns
    """
    # Only columns that pandas does not convert on its own (they contain '.' or NaN).

    to_int_features = [
        'article_batch',
        'article_id',
        'paragraph_id',
        'repeated_reading_trial',
        'practice_trial',
        # "question_preview",
    ]
    if args.mode == Mode.IA:
        to_int_features += [
            'IA_DWELL_TIME',
            'IA_FIRST_FIXATION_DURATION',
            'IA_REGRESSION_PATH_DURATION',
            'IA_FIRST_RUN_DWELL_TIME',
            'IA_FIXATION_COUNT',
            'IA_REGRESSION_IN_COUNT',
            'IA_REGRESSION_OUT_FULL_COUNT',
            'IA_RUN_COUNT',
            'IA_FIRST_FIXATION_VISITED_IA_COUNT',
            'IA_FIRST_RUN_FIXATION_COUNT',
            'IA_SKIP',
            'IA_REGRESSION_OUT_COUNT',
            'IA_SELECTIVE_REGRESSION_PATH_DURATION',
            'IA_SPILLOVER',
            'IA_LAST_FIXATION_DURATION',
            'IA_LAST_RUN_DWELL_TIME',
            'IA_LAST_RUN_FIXATION_COUNT',
            'IA_LEFT',
            'IA_TOP',
            'TRIAL_DWELL_TIME',
            'TRIAL_FIXATION_COUNT',
            'TRIAL_IA_COUNT',
            'TRIAL_INDEX',
            'TRIAL_TOTAL_VISITED_IA_COUNT',
            'IA_FIRST_FIX_PROGRESSIVE',
        ]
    elif args.mode == Mode.FIXATION:
        to_int_features += [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
            'CURRENT_FIX_DURATION',
            'CURRENT_FIX_PUPIL',
            'CURRENT_FIX_X',
            'CURRENT_FIX_Y',
            'CURRENT_FIX_INDEX',
            'NEXT_SAC_DURATION',
        ]
    df[to_int_features] = df[to_int_features].replace({'.': 0, np.nan: 0}).astype(int)
    logger.info(
        "{} fields converted to int, nan ('.') values replaced with 0.",
        to_int_features,
    )
    return df


def convert_to_float_features(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Convert specified columns to float type.

    Handles missing values and dots by replacing them with None before conversion.
    Different columns are processed based on whether in IA or FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with converted float columns
    """
    if args.mode == Mode.IA:
        to_float_features = [
            'IA_AVERAGE_FIX_PUPIL_SIZE',
            'IA_DWELL_TIME_%',
            'IA_FIXATION_%',
            'IA_FIRST_RUN_FIXATION_%',
            'IA_FIRST_SACCADE_AMPLITUDE',
            'IA_FIRST_SACCADE_ANGLE',
            'IA_LAST_RUN_FIXATION_%',
            'IA_LAST_SACCADE_AMPLITUDE',
            'IA_LAST_SACCADE_ANGLE',
            'IA_FIRST_RUN_LANDING_POSITION',
            'IA_LAST_RUN_LANDING_POSITION',
        ]
    elif args.mode == Mode.FIXATION:
        to_float_features = [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
            'NEXT_FIX_ANGLE',
            'PREVIOUS_FIX_ANGLE',
            'NEXT_FIX_DISTANCE',
            'PREVIOUS_FIX_DISTANCE',
            'NEXT_SAC_AMPLITUDE',
            'NEXT_SAC_ANGLE',
            'NEXT_SAC_AVG_VELOCITY',
            'NEXT_SAC_PEAK_VELOCITY',
            'NEXT_SAC_END_X',
            'NEXT_SAC_START_X',
            'NEXT_SAC_END_Y',
            'NEXT_SAC_START_Y',
        ]
    else:
        raise ValueError(f'Unknown mode: {args.mode}')
    df[to_float_features] = (
        df[to_float_features].replace(to_replace={'.': None}).astype(float)
    )
    logger.info(
        "{} fields converted to float, nan ('.') values replaced with None.",
        to_float_features,
    )
    return df


def adjust_indexing(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Adjust indexing to be 0-indexed.

    Subtracts 1 from specified columns based on whether in IA or FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with adjusted indexing
    """
    if args.mode == Mode.IA:
        subtract_one_fields = [IA_ID_COL]
    elif args.mode == Mode.FIXATION:
        subtract_one_fields = [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
        ]
    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    df[subtract_one_fields] -= 1
    logger.info('{} values adjusted to be 0-indexed.', subtract_one_fields)
    return df


def drop_missing_fixation_data(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Drop rows with missing fixation data.

    Drops rows with missing values in specified columns for FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with dropped rows
    """
    if args.mode == Mode.FIXATION:
        dropna_fields = [FIXATION_ID_COL, NEXT_FIXATION_ID_COL]
        df = df.dropna(subset=dropna_fields)
        logger.info(
            'After dropping rows with missing data in {}: {} records left in total.',
            dropna_fields,
            len(df),
        )
    return df


def add_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regression_rate, total_skip, is_content_word and IA_RR.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with added metrics
    """

    logger.info('Adding additional metrics...')
    df['regression_rate'] = df['IA_REGRESSION_OUT_FULL_COUNT'] / df['IA_RUN_COUNT']
    df['total_skip'] = df['IA_DWELL_TIME'] == 0
    df['is_content_word'] = df['universal_pos'].apply(is_content_word)
    df = add_ia_rr(df)

    return df


def add_ia_rr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the per-word first-pass regression-out indicator IA_RR. 1 where the
    word was reached progressively (first fixation entered from the left) AND
    had at least one first-pass regression-out; 0 where reached progressively
    without a first-pass regression; NaN otherwise. Downstream means /
    logistic fits naturally skip the NaN rows.

    Idempotent — if IA_RR is already present, returns df unchanged. This lets
    callers add IA_RR to harmonized CSVs that were written before the column
    was introduced, without re-running harmonization.
    """
    if 'IA_RR' in df.columns:
        return df
    progressive_mask = df['IA_FIRST_FIX_PROGRESSIVE'] == 1
    rr_indicator = (df['IA_REGRESSION_OUT_COUNT'] >= 1).astype(float)
    df['IA_RR'] = rr_indicator.where(progressive_mask, np.nan)
    return df


def get_constants_by_mode(mode: Mode) -> tuple[str, str]:
    """
    Get constants based on processing mode.

    Returns duration and IA field names based on whether in IA or FIXATION mode.

    Args:
        mode (Mode): Processing mode (IA or FIXATION)

    Returns:
        tuple[str, str]: Duration and IA field names
    """
    duration_field = 'IA_DWELL_TIME' if mode == Mode.IA else 'CURRENT_FIX_DURATION'
    ia_field = IA_ID_COL if mode == Mode.IA else FIXATION_ID_COL

    return duration_field, ia_field


def add_unique_paragraph_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add unique paragraph ID to the DataFrame.

    Creates a new column 'unique_paragraph_id' by combining article_batch,
    article_id, difficulty_level, and paragraph_id.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with added unique paragraph ID
    """
    logger.info('Adding unique paragraph id...')
    df['unique_paragraph_id'] = (
        df[['article_batch', 'article_id', 'difficulty_level', 'paragraph_id']]
        .astype(str)
        .apply('_'.join, axis=1)
    )
    return df


def compute_start_end_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute for each word whether it is the first/last word on its line (not sentence).

    This function adds two new columns to the input DataFrame: 'start_of_line' and 'end_of_line'.
    A word is considered to be at the start of a line if its
        'IA_LEFT' value is smaller than the previous word's.
    A word is considered to be at the end of a line if its
        'IA_LEFT' value is larger than the next word's.

    Parameters:
    df (pd.DataFrame): Input DataFrame. Must contain the columns 'participant_id',
        'unique_paragraph_id', and 'IA_LEFT'.

    Returns:
    pd.DataFrame: The input DataFrame with two new columns: 'start_of_line' and 'end_of_line'.
    """

    logger.info('Adding start_of_line and end_of_line columns...')
    grouped_df = df.groupby(
        ['participant_id', 'unique_paragraph_id', 'repeated_reading_trial']
    )
    df['start_of_line'] = (
        grouped_df['IA_LEFT'].shift(periods=1, fill_value=1000000) > df['IA_LEFT']
    )
    df['end_of_line'] = (
        grouped_df['IA_LEFT'].shift(periods=-1, fill_value=-1) < df['IA_LEFT']
    )
    return df


def compute_span_level_metrics(
    df: pd.DataFrame, ia_field: str, mode: Mode, duration_col: str
) -> pd.DataFrame:
    """
    Calculate aggregated metrics for different text spans.

    Computes:
    - Total dwell time per trial/span
    - Min/max word indices per trial/span
    - For fixations: count per span
    - Normalizes indices to start at 0

    Args:
        df (pd.DataFrame): Input DataFrame
        ia_field (str): Column name for word/fixation index
        mode (Mode): IA or FIXATION processing mode
        duration_col (str): Column name for duration values

    Returns:
        pd.DataFrame: DataFrame with added span-level metrics
    """
    logger.info('Computing span-level metrics...')

    group_by_fields = [
        'participant_id',
        'unique_paragraph_id',
        'repeated_reading_trial',
    ]

    # Fix trials where ID does not start at 0
    if mode == Mode.IA:
        temp_max_per_trial = df.groupby(group_by_fields).agg(
            min_IA_ID=pd.NamedAgg(column=ia_field, aggfunc='min'),
            max_IA_ID=pd.NamedAgg(column=ia_field, aggfunc='max'),
        )
        non_zero_min_ia_id_trials = temp_max_per_trial[
            temp_max_per_trial['min_IA_ID'] != 0
        ]
        logger.info(
            'Number of trials where min_IA_ID is not zero: {} out of {} trials.',
            len(non_zero_min_ia_id_trials),
            len(temp_max_per_trial),
        )
        df = df.merge(
            temp_max_per_trial,
            on=group_by_fields,
            validate='m:1',
            suffixes=(None, '_y'),
        )
        logger.info('Shifting IA_ID to start at 0...')
        df[ia_field] -= df['min_IA_ID']
        df.drop(columns=['min_IA_ID', 'max_IA_ID'], inplace=True)

    max_per_trial = df.groupby(group_by_fields).agg(
        total_IA_DWELL_TIME=pd.NamedAgg(column=duration_col, aggfunc='sum'),
        min_IA_ID=pd.NamedAgg(column=ia_field, aggfunc='min'),
        max_IA_ID=pd.NamedAgg(column=ia_field, aggfunc='max'),
    )
    df = df.merge(
        max_per_trial, on=group_by_fields, validate='m:1', suffixes=(None, '_y')
    )
    return df


def compute_normalized_features(
    df: pd.DataFrame, duration_col: str, ia_field: str
) -> pd.DataFrame:
    """
    Add normalized_ID, the word/fixation index scaled to [0, 1] within its trial.

    Args:
        df (pd.DataFrame): Input DataFrame, with min_IA_ID and max_IA_ID per trial
        duration_col (str): Unused; kept for a uniform call signature
        ia_field (str): Column name for word/fixation index

    Returns:
        pd.DataFrame: DataFrame with normalized_ID
    """
    logger.info('Computing normalized word indices...')
    df = df.assign(
        normalized_ID=(df[ia_field] - df.min_IA_ID) / (df.max_IA_ID - df.min_IA_ID),
    ).copy()
    return df

# ---------------------------------------------------------------- word properties (from text_metrics)

CONTENT_WORDS = {
    "PUNCT": False,
    "PROPN": True,
    "NOUN": True,
    "PRON": False,
    "VERB": True,
    "SCONJ": False,
    "NUM": False,
    "DET": False,
    "CCONJ": False,
    "ADP": False,
    "AUX": False,
    "ADV": True,
    "ADJ": True,
    "INTJ": False,
    "X": False,
    "PART": False,
}


def is_content_word(pos: str) -> bool:
    """Whether a universal POS tag is a content-word tag."""
    return CONTENT_WORDS.get(pos, False)


# ---------------------------------------------------------------- downloads and loading


def download_from_osf(
    directory: str | Path,
    urls: dict[str, str],
    is_metadata: bool = False,
) -> None:
    """Download files from OSF into directory/downloads (or downloads/metadata).

    Args:
        directory: The base dataset directory.
        urls: Maps each file name to an OSF resource ID or a full https:// URL.
            Zip files are extracted in place and then deleted.
        is_metadata: Download into downloads/metadata instead.
    """
    base_dir = Path(directory)
    download_dir = base_dir / 'downloads'
    if is_metadata:
        download_dir = base_dir / 'downloads/metadata'

    download_dir.mkdir(parents=True, exist_ok=True)

    for filename, resource in (pbar := tqdm(urls.items())):
        pbar.set_description(f'Downloading {filename}')
        if resource.startswith('https://'):
            url = resource
        else:
            url = f'https://osf.io/download/{resource}'

        path = download_dir / filename

        if path.exists():
            logger.warning(f'{path} already exists, skipping download.')
        else:
            req = requests.get(url, stream=True)
            with open(path, 'wb') as output_file:
                for chunk in req.iter_content(chunk_size=8192):
                    output_file.write(chunk)

        if filename.endswith('.zip'):
            logger.info(f'Extracting {filename} to {download_dir}...')
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            path.unlink()


def load_cp1255_rda(rda_path: str) -> dict:
    """Reads a gzipped RDA file, patching a CP1255 encoding marker to UTF-8 if present.

    This is needed because pyreadr cannot handle CP1255-marked RDA files. The
    patch replaces only the encoding marker byte sequence while leaving the
    actual string data untouched.

    Parameters
    ----------
    rda_path : str
        Path to the gzipped RDA file.

    Returns
    -------
    dict
        Dictionary of dataframes as returned by pyreadr.read_r.

    Notes
    -----
    !approx: Assumes actual string data is UTF-8 compatible. If this fails,
    re-export the original RDA from R with UTF-8 encoding.
    """
    with gzip.open(rda_path, 'rb') as f:
        data = f.read()

    if b'CP1255' in data:
        cp1255_pos = data.find(b'CP1255')
        # Replace CP1255 (7 bytes) with UTF-8\x00\x00 (7 bytes) to preserve binary structure
        patched_data = data[:cp1255_pos] + b'UTF-8\x00\x00' + data[cp1255_pos + 7:]
        logger.info(f'Patching RDA encoding marker from CP1255 to UTF-8 at position {cp1255_pos}')
    else:
        patched_data = data

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.rda')
    with gzip.open(temp_file.name, 'wb') as f:
        f.write(patched_data)
    temp_file.close()

    contents = pyreadr.read_r(temp_file.name)
    logger.info(f'Contents of patched RDA file: {list(contents.keys())}')
    os.remove(temp_file.name)

    return contents


def _load_harmonized(dataset_name: str):
    """Load fixations (optional), IA, and metadata for one dataset."""
    dataset_path = DATA_PATH / dataset_name
    fix_path = dataset_path / 'harmonized' / 'fixations.csv'
    fix_data = pd.read_csv(fix_path) if fix_path.exists() else None
    ia_data = pd.read_csv(dataset_path / 'harmonized' / 'ia.csv')
    ia_data = add_ia_rr(ia_data)
    md = pd.read_csv(dataset_path / 'metadata' / 'metadata.csv')
    return fix_data, ia_data, md


# ---------------------------------------------------------------- OneStop IA alignment


def add_content_group_to_metadata(ia_data: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Derive a content_group column from the IA data and add it to metadata.

    Within each batch, participants read one of two complementary version
    assignments.  The difficulty level of a reference article (the 2nd by
    article_id, which differs between the two assignments in all batches)
    is used as the group discriminator, giving labels like "1_0", "1_1".
    This is tied to actual content, so L1 and L2 participants who read the
    same thing share the same label.
    """
    REFERENCE_ARTICLE_POSITION = 1  # 2nd article by article_id within each batch

    per_article = (
        ia_data.groupby([Fields.SUBJECT_ID, Fields.BATCH, Fields.ARTICLE_ID])[Fields.LEVEL]
        .first()
        .reset_index()
        .sort_values([Fields.SUBJECT_ID, Fields.BATCH, Fields.ARTICLE_ID])
    )
    reference = (
        per_article.groupby([Fields.SUBJECT_ID, Fields.BATCH])
        .nth(REFERENCE_ARTICLE_POSITION)
        .reset_index(drop=True)
    )
    reference[Fields.CONTENT_GROUP] = (
        reference[Fields.BATCH].astype(str) + '_' + reference[Fields.LEVEL].astype(int).astype(str)
    )

    groups_per_batch = reference.groupby(Fields.BATCH)[Fields.CONTENT_GROUP].nunique()
    assert (groups_per_batch == 2).all(), (
        f"Expected exactly 2 content groups per batch; got {groups_per_batch.to_dict()}. "
        "The reference article may not discriminate in all batches."
    )

    # A stale content_group (e.g. metadata saved by a previous cascade-fix run)
    # must be replaced, not merged alongside as content_group_x/_y.
    metadata_df = metadata_df.drop(columns=[Fields.CONTENT_GROUP], errors='ignore')
    return metadata_df.merge(
        reference[[Fields.SUBJECT_ID, Fields.CONTENT_GROUP]],
        on=Fields.SUBJECT_ID,
        how='left',
    )


def _realign_ia_cascade_onepass(
    ia_df: pd.DataFrame,
    fix_df: pd.DataFrame | None,
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None, int]:
    """Single pass of the OneStop cascade fix.

    For each (content_group, unique_paragraph_id) where the boundary IA's
    label disagrees across participants, drops the offending row(s) on each
    side and shifts subsequent IA_IDs so the layouts line up. Returns the
    updated frames plus the number of paragraphs that were patched (0 means
    nothing left to do).
    """
    cg_map = (metadata_df.set_index(Fields.SUBJECT_ID)[Fields.CONTENT_GROUP]
              .to_dict())
    ia_with_cg = ia_df.assign(**{
        Fields.CONTENT_GROUP: ia_df[Fields.SUBJECT_ID].map(cg_map),
    })

    gk = [
        Fields.CONTENT_GROUP,
        Fields.UNIQUE_PARAGRAPH_ID,
        Fields.IA_DATA_IA_ID_COL_NAME,
    ]
    label_counts = ia_with_cg.groupby(gk)['IA_LABEL'].nunique()
    bad = label_counts[label_counts > 1]
    if bad.empty:
        return ia_df, fix_df, 0

    # Per (cg, paragraph) take the smallest bad IA_ID as the cascade boundary.
    boundary_per_para = (
        bad.reset_index()
        .groupby([Fields.CONTENT_GROUP, Fields.UNIQUE_PARAGRAPH_ID])
        [Fields.IA_DATA_IA_ID_COL_NAME].min()
    )

    # Look up boundary-row labels in one merge instead of per-paragraph filtering.
    boundaries_df = boundary_per_para.reset_index().rename(
        columns={Fields.IA_DATA_IA_ID_COL_NAME: '_boundary'}
    )
    boundary_rows = ia_with_cg.merge(
        boundaries_df,
        on=[Fields.CONTENT_GROUP, Fields.UNIQUE_PARAGRAPH_ID],
        how='inner',
    )
    boundary_rows = boundary_rows[
        boundary_rows[Fields.IA_DATA_IA_ID_COL_NAME] == boundary_rows['_boundary']
    ][[Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, Fields.CONTENT_GROUP, '_boundary']]
    # When a participant has multiple trials on the same paragraph (e.g.
    # rereading), the boundary row appears once per trial. The IA layout for
    # the same content_group is identical across trials, so collapse to one
    # row per (subject_id, paragraph) before emitting drop/shift records.
    boundary_rows = boundary_rows.drop_duplicates(
        subset=[Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID]
    )

    # Classify each participant at the boundary as split-side or joined-side
    # by their total IA count for the paragraph. The split side has exactly
    # one more IA (because one word was rendered as two interest areas
    # instead of one). This is robust against non-hyphen splits like
    # "6.30am;" -> "6.30" + "am;" where label characters give no signal.
    ia_count = (
        ia_with_cg.groupby(
            [Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID]
        )[Fields.IA_DATA_IA_ID_COL_NAME]
        .nunique()
        .reset_index(name='_ia_count')
    )
    boundary_rows = boundary_rows.merge(
        ia_count,
        on=[Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID],
        how='left',
    )
    grp = boundary_rows.groupby(
        [Fields.CONTENT_GROUP, Fields.UNIQUE_PARAGRAPH_ID]
    )['_ia_count']
    boundary_rows['_high_count'] = grp.transform('max')
    boundary_rows['_low_count'] = grp.transform('min')
    # Cascade-style mismatch: pools differ in IA count by 1; the higher-
    # count pool is the split side. Cosmetic-only mismatch (counts equal):
    # both pools drop the boundary IA and shift +1+ down by 1.
    boundary_rows['_is_split'] = (
        (boundary_rows['_high_count'] != boundary_rows['_low_count'])
        & (boundary_rows['_ia_count'] == boundary_rows['_high_count'])
    )

    split_side = boundary_rows[boundary_rows['_is_split']]
    other_side = boundary_rows[~boundary_rows['_is_split']]

    if len(boundary_rows) == 0:
        return ia_df, fix_df, 0

    # Build drop/shift records vectorized rather than per-row.
    # Other side (joined or cosmetic-only): drop boundary, shift boundary+1+
    # down by 1.
    other_drops = other_side[[
        Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, '_boundary',
    ]].rename(columns={'_boundary': '_drop_ia'})
    other_shifts = other_side[[
        Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, '_boundary',
    ]].copy()
    other_shifts['_from_ia'] = other_shifts['_boundary'] + 1
    other_shifts['_shift'] = 1
    other_shifts = other_shifts.drop(columns='_boundary')

    # Split side: drop boundary AND boundary+1; shift boundary+2+ down by 2.
    split_drops_a = split_side[[
        Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, '_boundary',
    ]].rename(columns={'_boundary': '_drop_ia'})
    split_drops_b = split_side[[
        Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, '_boundary',
    ]].copy()
    split_drops_b['_drop_ia'] = split_drops_b['_boundary'] + 1
    split_drops_b = split_drops_b.drop(columns='_boundary')
    split_shifts = split_side[[
        Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, '_boundary',
    ]].copy()
    split_shifts['_from_ia'] = split_shifts['_boundary'] + 2
    split_shifts['_shift'] = 2
    split_shifts = split_shifts.drop(columns='_boundary')

    drops_df = pd.concat(
        [other_drops, split_drops_a, split_drops_b], ignore_index=True,
    )
    shifts_df = pd.concat(
        [other_shifts, split_shifts], ignore_index=True,
    )
    if drops_df.empty:
        return ia_df, fix_df, 0

    # Pre-compute lookup structures so we never create a wide merged copy of
    # the fixation frame (which is many GB on OneStop).
    drop_keys = pd.MultiIndex.from_frame(
        drops_df[[Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID, '_drop_ia']]
    )
    shifts_indexed = shifts_df.set_index(
        [Fields.SUBJECT_ID, Fields.UNIQUE_PARAGRAPH_ID]
    )

    def _apply(df, ia_cols):
        pid_arr = df[Fields.SUBJECT_ID].to_numpy()
        para_arr = df[Fields.UNIQUE_PARAGRAPH_ID].to_numpy()

        # 1) Drop rows: a row is dropped if (pid, para, val) is in drop_keys for
        #    ANY of the ia_cols (saccade source or target on a removed IA).
        keep = np.ones(len(df), dtype=bool)
        for col in ia_cols:
            triple_idx = pd.MultiIndex.from_arrays(
                [pid_arr, para_arr, df[col].to_numpy()]
            )
            keep &= ~triple_idx.isin(drop_keys)
        df_new = df.loc[keep].copy()

        # 2) Shifts: per (pid, para), the post-boundary IAs shift down by N.
        pair_idx = pd.MultiIndex.from_arrays([
            df_new[Fields.SUBJECT_ID].to_numpy(),
            df_new[Fields.UNIQUE_PARAGRAPH_ID].to_numpy(),
        ])
        from_ia_arr = shifts_indexed['_from_ia'].reindex(pair_idx).to_numpy()
        shift_arr = shifts_indexed['_shift'].reindex(pair_idx).to_numpy()
        has_shift = ~pd.isna(from_ia_arr)
        if has_shift.any():
            for col in ia_cols:
                col_vals = df_new[col].to_numpy()
                # Skip NaN col_vals (off-IA fixations) — float comparisons return False for NaN
                with np.errstate(invalid='ignore'):
                    mask = has_shift & (col_vals >= from_ia_arr)
                if mask.any():
                    new_vals = col_vals.copy()
                    new_vals[mask] = col_vals[mask] - shift_arr[mask]
                    df_new[col] = new_vals
        return df_new

    ia_new = _apply(ia_df, [Fields.IA_DATA_IA_ID_COL_NAME])

    fix_new = fix_df
    if fix_df is not None:
        fix_new = _apply(
            fix_df,
            [Fields.FIXATION_REPORT_IA_ID_COL_NAME, NEXT_FIXATION_ID_COL],
        )

    return ia_new, fix_new, len(boundary_per_para)


def realign_ia_cascade_onestop(
    ia_df: pd.DataFrame,
    fix_df: pd.DataFrame | None,
    metadata_df: pd.DataFrame,
    max_iter: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Resolve IA layout cascades caused by inconsistent line-wrap
    hyphenation across OneStop participants in the same content_group.

    Iterates the single-pass fix until no boundary mismatches remain
    (handles paragraphs with multiple split points). Loses one IA position
    per split point per affected paragraph; preserves all other IAs.
    """
    if metadata_df is None or Fields.CONTENT_GROUP not in metadata_df.columns:
        return ia_df, fix_df
    # Participants without a content_group are silently excluded from the
    # alignment (NaN groups drop out of the groupby) — surface that loudly.
    cg_map_check = (metadata_df.dropna(subset=[Fields.CONTENT_GROUP])
                    .set_index(Fields.SUBJECT_ID)[Fields.CONTENT_GROUP].to_dict())
    unaligned = set(ia_df[Fields.SUBJECT_ID].unique()) - set(cg_map_check)
    if unaligned:
        logger.warning(
            f'{len(unaligned)} participant(s) have no content_group and are '
            f'EXCLUDED from cascade alignment: {sorted(unaligned)[:10]}'
        )
    for i in range(max_iter):
        ia_df, fix_df, n_patched = _realign_ia_cascade_onepass(
            ia_df, fix_df, metadata_df,
        )
        logger.info(
            f'IA cascade fix pass {i + 1}: patched {n_patched} (cg, paragraph) pairs.'
        )
        if n_patched == 0:
            if i > 0:
                logger.info(f'IA cascade fix converged after {i + 1} pass(es).')
            return ia_df, fix_df
    # Surface what's still broken so we can see which paragraphs the
    # algorithm couldn't resolve.
    cg_map = (metadata_df.set_index(Fields.SUBJECT_ID)[Fields.CONTENT_GROUP]
              .to_dict())
    ia_with_cg = ia_df.assign(**{
        Fields.CONTENT_GROUP: ia_df[Fields.SUBJECT_ID].map(cg_map),
    })
    gk = [Fields.CONTENT_GROUP, Fields.UNIQUE_PARAGRAPH_ID, Fields.IA_DATA_IA_ID_COL_NAME]
    bad = ia_with_cg.groupby(gk)['IA_LABEL'].nunique()
    bad = bad[bad > 1]
    bad_per_para = (bad.reset_index()
                    .groupby([Fields.CONTENT_GROUP, Fields.UNIQUE_PARAGRAPH_ID])
                    .size())
    raise RuntimeError(
        f'realign_ia_cascade_onestop did not converge in {max_iter} passes. '
        f'Remaining bad keys: {len(bad)}. Per (cg, paragraph):\n{bad_per_para}'
    )


def realign_ia_cascade_onestop_paired(
    ia_l1: pd.DataFrame,
    fix_l1: pd.DataFrame | None,
    md_l1: pd.DataFrame,
    ia_l2: pd.DataFrame,
    fix_l2: pd.DataFrame | None,
    md_l2: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.DataFrame | None,
    pd.DataFrame, pd.DataFrame | None,
]:
    """Run the cascade fix on the *combined* L1+L2 IA layout, then split
    back by cohort. Guarantees identical IA renumbering across cohorts —
    a precondition for comparing per-text features (TRANSITIONS / WFC)
    between L1 and L2, since the fix's drop/shift decisions depend on
    which participants are in the pool.
    """
    cohort_col = '_cascade_cohort'
    L1, L2 = 'l1', 'l2'

    md_combined = pd.concat([md_l1, md_l2], ignore_index=True)
    if md_combined[Fields.SUBJECT_ID].duplicated().any():
        # The fix uses subject_id to count IAs per participant; collisions
        # between cohorts would corrupt the split-side detection.
        raise ValueError(
            'L1 and L2 share subject_ids; cannot pair-align without '
            'disambiguating them first.'
        )

    ia_combined = pd.concat([
        ia_l1.assign(**{cohort_col: L1}),
        ia_l2.assign(**{cohort_col: L2}),
    ], ignore_index=True)

    fix_combined = None
    if fix_l1 is not None or fix_l2 is not None:
        parts = []
        if fix_l1 is not None:
            parts.append(fix_l1.assign(**{cohort_col: L1}))
        if fix_l2 is not None:
            parts.append(fix_l2.assign(**{cohort_col: L2}))
        fix_combined = pd.concat(parts, ignore_index=True)

    ia_fixed, fix_fixed = realign_ia_cascade_onestop(
        ia_combined, fix_combined, md_combined,
    )

    ia_l1_out = ia_fixed[ia_fixed[cohort_col] == L1].drop(columns=[cohort_col])
    ia_l2_out = ia_fixed[ia_fixed[cohort_col] == L2].drop(columns=[cohort_col])
    fix_l1_out = fix_l2_out = None
    if fix_fixed is not None:
        if fix_l1 is not None:
            fix_l1_out = fix_fixed[fix_fixed[cohort_col] == L1].drop(columns=[cohort_col])
        if fix_l2 is not None:
            fix_l2_out = fix_fixed[fix_fixed[cohort_col] == L2].drop(columns=[cohort_col])

    return ia_l1_out, fix_l1_out, ia_l2_out, fix_l2_out
