"""Shared helpers for the EyeScore and prediction pipelines: z-scoring and
loading the per-text (TRANSITIONS / WFC) feature parquets."""
from pathlib import Path

import pandas as pd
from loguru import logger

from src.constants import (
    PER_TEXT_FEATURES_FILENAME,
    TRANSITIONS_FEATURE_PREFIX,
    WFC_FEATURE_PREFIX,
    Fields,
)


def _is_per_text(col: str) -> bool:
    return col.startswith(TRANSITIONS_FEATURE_PREFIX) or col.startswith(WFC_FEATURE_PREFIX)


def load_per_text_features(features_csv_path: Path) -> pd.DataFrame | None:
    """Read the per-text features parquet that sits next to
    `features_csv_path`. The file is self-contained (metadata + per-text
    feature columns), so it can be used as a drop-in replacement for the
    main features DataFrame for per-text feature sets.

    Cells for (participant × paragraph) pairs the participant did not read
    are NaN in the parquet (extraction-time fix in
    `compute_per_text_features`); downstream NaN-aware operations skip them
    so each participant's eye_score lives in the column space of articles
    they actually read.

    Returns None if the parquet does not exist.
    """
    per_text_path = Path(features_csv_path).with_name(PER_TEXT_FEATURES_FILENAME)
    if not per_text_path.exists():
        logger.warning(
            f"Per-text features file not found at {per_text_path}; "
            "TRANSITIONS / WFC feature sets cannot be evaluated."
        )
        return None

    logger.info(f"Loading per-text features from {per_text_path}")
    # Per-text parquets carry hundreds of thousands of columns; pyarrow's
    # default thrift decode limits (string=100MB, container=1M) reject the
    # encoded schema footer. Bump both to int32 max.
    df = pd.read_parquet(
        per_text_path,
        thrift_string_size_limit=2**31 - 1,
        thrift_container_size_limit=2**31 - 1,
    )
    return _reconcile_metadata_with_csv(df, Path(features_csv_path))


def _reconcile_metadata_with_csv(
    df: pd.DataFrame, features_csv_path: Path
) -> pd.DataFrame:
    """Override the per-text parquet's metadata columns from the sibling
    features_and_metadata.csv (the corrected source of truth), matched by
    participant id.

    The self-contained per-text parquet carries its own metadata snapshot,
    which can drift from the CSV when a label is fixed only in the CSV (e.g. 43
    MECO participants relabeled Serbian in the CSV but still 'Swedish' in the
    parquet — which then had no typological distance and broke the bias
    correction). For every non-per-text column shared with the CSV, take the
    CSV's value where it has one and keep the parquet's otherwise (so parquet
    rows absent from the CSV are left untouched rather than nulled). Per-text
    feature columns are never touched.
    """
    if not features_csv_path.exists() or Fields.SUBJECT_ID not in df.columns:
        return df

    meta_csv = pd.read_csv(features_csv_path)
    if Fields.SUBJECT_ID not in meta_csv.columns:
        return df
    override_cols = [
        c for c in df.columns
        if not _is_per_text(c) and c != Fields.SUBJECT_ID and c in meta_csv.columns
    ]
    if not override_cols:
        return df

    lut = (
        meta_csv[[Fields.SUBJECT_ID] + override_cols]
        .drop_duplicates(Fields.SUBJECT_ID)
        .set_index(Fields.SUBJECT_ID)
    )
    ids = df[Fields.SUBJECT_ID]
    n_changed = 0
    for col in override_cols:
        csv_val = ids.map(lut[col])
        new_col = csv_val.where(csv_val.notna(), df[col].to_numpy())
        n_changed += int((new_col.to_numpy() != df[col].to_numpy()).sum())
        df[col] = new_col.to_numpy()
    if n_changed:
        logger.info(
            f"Reconciled per-text metadata against {features_csv_path.name}: "
            f"{n_changed} cell(s) overridden across {len(override_cols)} column(s)."
        )
    return df


def load_per_text_features_pair(
    l1_features_csv_path: Path,
    l2_features_csv_path: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load L1 and L2 per-text parquets and align their per-text columns
    to the **union**, with missing entries filled with 0.

    A per-text column is missing for a cohort iff no participant in that
    cohort observed the event (saccade for `transitions_*`, fixation for
    `wfc_*`). Semantically that's a 0 count / 0 dwell — so 0-fill is
    value-preserving, not lossy. The column order of the per-text block
    is the same on both sides (sorted union), guaranteeing positional
    alignment for any downstream operation.

    Metadata columns (subject id, L1, batch, target columns, ...) are
    cohort-specific and pass through untouched.

    Returns (None, None) if either parquet is missing. Requires the
    paired cascade fix at extraction time so that IA indices in the
    column names refer to the same words across cohorts.
    """
    df1 = load_per_text_features(l1_features_csv_path)
    df2 = load_per_text_features(l2_features_csv_path)
    if df1 is None or df2 is None:
        return None, None

    pt1 = {c for c in df1.columns if _is_per_text(c)}
    pt2 = {c for c in df2.columns if _is_per_text(c)}
    union = sorted(pt1 | pt2)
    meta1 = [c for c in df1.columns if not _is_per_text(c)]
    meta2 = [c for c in df2.columns if not _is_per_text(c)]

    df1 = df1.reindex(columns=meta1 + union, fill_value=0)
    df2 = df2.reindex(columns=meta2 + union, fill_value=0)

    logger.info(
        f"Aligned per-text columns to union: {len(union)} cols "
        f"(L1 added {len(pt2 - pt1)}, L2 added {len(pt1 - pt2)})."
    )
    return df1, df2


def zscore_preserve_nan_general(fit_on_df, df_to_scale):
    """Z-score `df_to_scale` with the column means / sds of `fit_on_df` (NaN
    stays NaN). Callers pass the L2 fit pool as `fit_on_df`."""
    means = fit_on_df.mean()
    stds = fit_on_df.std()
    # Drop zero-variance columns: their z-score would be NaN (divide-by-zero),
    # contaminating downstream cosine/dot products. Per-text feature loaders
    # 0-fill cohort-missing columns to align L1 and L2; those all-zero columns
    # carry no signal in the fit data, so excluding them is value-preserving.
    keep = stds.index[stds > 0]
    return (df_to_scale[keep] - means[keep]) / stds[keep]
