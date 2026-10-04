from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

from src.constants import (
    DataSets,
    FeatureGroups,
    Fields,
    EYE_METRICS_NICKNAMES,
    EYE_METRICS_NICKNAMES_INVERTED,
    WORD_PROPERTY_COLUMNS,
    FIXATIONS_METRICS_CLUSTERS,
    FIXATIONS_METRICS_COEFS,
    PosCols,
    TRANSITIONS_FEATURE_PREFIX,
    WFC_FEATURE_PREFIX,
    PER_TEXT_FEATURES_FILENAME,
    POS_POSSIBLE_VALUES,
)

NEXT_FIX_IA_INDEX_COL = 'NEXT_FIX_INTEREST_AREA_INDEX'


# ---------------------------------------------------------------- participant-level features


def compute_grouped_features(
    df: pd.DataFrame,
    grouped_partial: partial,
    label: str,
    participant_groupby_columns: list[str],
) -> pd.DataFrame:
    logger.info(
        f'Number of participant groups in {label}: {len(df.groupby(participant_groupby_columns).groups)}'
    )
    grouped_features = df.groupby(participant_groupby_columns).apply(grouped_partial)
    # No fillna(0) on purpose: a NaN from the feature functions means the value
    # is undefined, not zero. E.g. RR is 0/0 for the MECO ch_s site (no word can
    # be identified as first-pass progressive), and a rare-POS cluster has no
    # mean for a participant who never read that tag. Imputation happens
    # downstream, per training fold (Ridge's _transform maps non-finite values
    # to 0 after z-scoring, i.e. to the training-fold mean).
    return pd.DataFrame(list(grouped_features), index=grouped_features.index)


def compute_participant_level_feature_group(
    feature_groups: list[FeatureGroups],
    harmonized_fixations_data: pd.DataFrame | None,
    harmonized_ia_data: pd.DataFrame,
    participant_groupby_columns: list[str],
):
    """
    Compute participant-level features for a list of feature groups.

    Per-text feature groups (TRANSITIONS, WFC) are excluded here and
    computed separately by `compute_per_text_features` so they can be
    written to a sibling file rather than bloating the main features CSV.
    """

    logger.info(f'Computing participant level features for {harmonized_ia_data.shape[0]} rows. This might take a couple of minutes...')
    ia_partial = partial(compute_ia_participant_level_features, feature_groups=feature_groups)
    participant_level_features = compute_grouped_features(
        harmonized_ia_data, ia_partial, label='ia', participant_groupby_columns=participant_groupby_columns,
    )

    if harmonized_fixations_data is not None:
        logger.info(f'Computing fixation participant level features for {harmonized_fixations_data.shape[0]} rows...')
        fixation_partial = partial(compute_fixation_trial_level_features, feature_groups=feature_groups)
        fixation_participant_features = compute_grouped_features(
            harmonized_fixations_data, fixation_partial, label='fix', participant_groupby_columns=participant_groupby_columns,
        )
        participant_level_features = pd.concat([fixation_participant_features, participant_level_features], axis=1)

    return participant_level_features


def compute_per_text_features(
    feature_groups: list[FeatureGroups],
    harmonized_fixations_data: pd.DataFrame | None,
    harmonized_ia_data: pd.DataFrame,
    participant_groupby_columns: list[str],
    metadata_df: pd.DataFrame | None,
    dataset_name: str | None,
) -> pd.DataFrame | None:
    """Compute per-text features (TRANSITIONS / WFC).

    Returns None when none of the requested groups are per-text. The
    returned frame is indexed by `participant_groupby_columns`.
    """
    feature_groups_set = set(feature_groups)
    needs_per_text = feature_groups_set & {FeatureGroups.TRANSITIONS, FeatureGroups.WFC}
    if not needs_per_text:
        return None

    if dataset_name is None:
        raise ValueError(
            "TRANSITIONS / WFC feature groups require dataset_name to be passed "
            "to compute_per_text_features."
        )
    validate_ia_id_alignment(harmonized_ia_data, metadata_df, dataset_name)

    per_text_frames = []
    if FeatureGroups.TRANSITIONS in feature_groups_set:
        if harmonized_fixations_data is None:
            raise ValueError(
                "TRANSITIONS feature group requires fixation data, got None."
            )
        # Restrict the fixation source for TRANSITIONS to the same
        # (participant × paragraph) coverage as the IA dataframe. Without
        # this, the fixation file's broader coverage (fly-over fixations on
        # paragraphs the IA pipeline filtered out) leaks into TRANSITIONS
        # and breaks the "this participant read this paragraph" semantic
        # the rest of the per-text features rely on.
        ia_pairs = harmonized_ia_data[
            list(participant_groupby_columns) + [Fields.UNIQUE_PARAGRAPH_ID]
        ].drop_duplicates()
        before = len(harmonized_fixations_data)
        fixations_for_transitions = harmonized_fixations_data.merge(
            ia_pairs,
            on=list(participant_groupby_columns) + [Fields.UNIQUE_PARAGRAPH_ID],
            how="inner",
        )
        logger.info(
            f'Computing TRANSITIONS features (fixations restricted to IA coverage: '
            f'{len(fixations_for_transitions)}/{before} rows kept)...'
        )
        per_text_frames.append(
            compute_transitions_features(fixations_for_transitions, participant_groupby_columns)
        )
    if FeatureGroups.WFC in feature_groups_set:
        logger.info('Computing WFC features...')
        per_text_frames.append(
            compute_wfc_features(harmonized_ia_data, participant_groupby_columns)
        )
    # No 0-fill across the concat: per-text frames carry NaN for
    # (participant × paragraph) pairs the participant skipped, and that NaN
    # must survive to the parquet so the EyeScore pipeline can distinguish
    # "didn't read this paragraph" (NaN, ignored by NaN-aware cosine) from
    # "read but had no event for this specific cell" (real 0).
    return pd.concat(per_text_frames, axis=1)


# ---------------------------------------------------------------- per-text features (TRANSITIONS, WFC)


def _fill_zero_for_read_paragraphs(
    wide: pd.DataFrame,
    source_df: pd.DataFrame,
    participant_groupby_columns: list[str],
) -> pd.DataFrame:
    """Fill NaN→0 in `wide` cells for (participant, paragraph) pairs that
    appear in `source_df`. Cells for (participant, paragraph) pairs absent
    from `source_df` stay NaN — those represent unread paragraphs.

    The pivot/unstack callers of this helper use no fill_value (NaN-fill by
    default), so absent (participant × paragraph × …) combinations are NaN.
    For paragraphs the participant *did* read, the absence of an event for
    a specific (IA, metric) or (src, tgt) is a real 0 (read but no
    fixation / no saccade), not missing data — this helper restores those
    to 0 while leaving truly-unread blocks as NaN.

    Args:
        wide: pivot_table / unstack output. Index spans
            `participant_groupby_columns`; columns are a MultiIndex with
            `Fields.UNIQUE_PARAGRAPH_ID` at level 0.
        source_df: long-format event source. Must contain
            `participant_groupby_columns + [UNIQUE_PARAGRAPH_ID]`.
        participant_groupby_columns: columns identifying a participant.
    """
    pgc = list(participant_groupby_columns)
    pid_col = Fields.UNIQUE_PARAGRAPH_ID
    read = source_df[pgc + [pid_col]].drop_duplicates()

    col_paragraphs = wide.columns.get_level_values(0).to_numpy()
    paragraphs_in_wide = pd.unique(col_paragraphs)

    if len(pgc) == 1:
        row_keys = wide.index.to_numpy()
    else:
        row_keys = np.array([tuple(k) for k in wide.index], dtype=object)

    vals = wide.to_numpy(dtype=float, copy=True)

    for paragraph in paragraphs_in_wide:
        par_col_idx = np.flatnonzero(col_paragraphs == paragraph)
        if par_col_idx.size == 0:
            continue

        readers = read[read[pid_col] == paragraph][pgc].drop_duplicates()
        if readers.empty:
            continue

        if len(pgc) == 1:
            reader_set = set(readers[pgc[0]].tolist())
        else:
            reader_set = set(map(tuple, readers.itertuples(index=False, name=None)))
        row_is_reader = np.fromiter((k in reader_set for k in row_keys),
                                    dtype=bool, count=len(row_keys))
        if not row_is_reader.any():
            continue

        reader_row_idx = np.flatnonzero(row_is_reader)
        sub_idx = np.ix_(reader_row_idx, par_col_idx)
        sub = vals[sub_idx]
        sub[np.isnan(sub)] = 0.0
        vals[sub_idx] = sub

    return pd.DataFrame(vals, index=wide.index, columns=wide.columns)


def compute_transitions_features(
    fix_df: pd.DataFrame,
    participant_groupby_columns: list[str],
) -> pd.DataFrame:
    """Per-participant n x n saccade transition counts, flattened.

    Returns a DataFrame indexed by participant_groupby_columns with columns
    named transitions_{unique_paragraph_id}_{i}_{j} where t_{i,j} is the
    number of saccades launched from word i and landing on word j within
    that paragraph. Computed as a single vectorized groupby/unstack — no
    inner per-trial loop.

    (participant × paragraph) pairs the participant did not read (no rows
    in `fix_df` for that pair) come out as NaN across that paragraph's
    columns, distinguishing them from read paragraphs where the participant
    happened to make no saccades for a particular (i, j) pair (real 0).
    """
    needed_cols = list(participant_groupby_columns) + [
        Fields.UNIQUE_PARAGRAPH_ID,
        Fields.FIXATION_REPORT_IA_ID_COL_NAME,
        NEXT_FIX_IA_INDEX_COL,
    ]
    df = fix_df[needed_cols].dropna(subset=[
        Fields.FIXATION_REPORT_IA_ID_COL_NAME, NEXT_FIX_IA_INDEX_COL,
    ])
    df = df[
        (df[Fields.FIXATION_REPORT_IA_ID_COL_NAME] >= 0)
        & (df[NEXT_FIX_IA_INDEX_COL] >= 0)
    ]
    df = df.assign(**{
        Fields.FIXATION_REPORT_IA_ID_COL_NAME: df[Fields.FIXATION_REPORT_IA_ID_COL_NAME].astype(int),
        NEXT_FIX_IA_INDEX_COL: df[NEXT_FIX_IA_INDEX_COL].astype(int),
    })

    group_cols = list(participant_groupby_columns) + [
        Fields.UNIQUE_PARAGRAPH_ID,
        Fields.FIXATION_REPORT_IA_ID_COL_NAME,
        NEXT_FIX_IA_INDEX_COL,
    ]
    counts = df.groupby(group_cols).size()
    n_keep = len(participant_groupby_columns)
    # No fill_value here: missing combinations come out as NaN. The helper
    # below restores 0 within paragraphs the participant actually read.
    wide = counts.unstack(level=list(range(n_keep, n_keep + 3)))
    wide = _fill_zero_for_read_paragraphs(wide, df, participant_groupby_columns)
    wide.columns = [
        f'{TRANSITIONS_FEATURE_PREFIX}{p}_{i}_{j}' for p, i, j in wide.columns
    ]
    return wide


def compute_wfc_features(
    ia_df: pd.DataFrame,
    participant_groupby_columns: list[str],
) -> pd.DataFrame:
    """Per-participant first-pass and total dwell times for each fixed-text word.

    Returns a DataFrame indexed by participant_groupby_columns with columns
    named wfc_{unique_paragraph_id}_{ia_id}_{FP|TF}.

    (participant × paragraph) pairs the participant did not read (no rows
    in `ia_df` for that pair) come out as NaN across that paragraph's
    columns, distinguishing them from read paragraphs where the participant
    happened to not fixate a particular IA (real 0 dwell time).
    """
    fp_col = EYE_METRICS_NICKNAMES['FP']  # IA_FIRST_RUN_DWELL_TIME
    tf_col = EYE_METRICS_NICKNAMES['TF']  # IA_DWELL_TIME

    long_df = ia_df.melt(
        id_vars=list(participant_groupby_columns) + [
            Fields.UNIQUE_PARAGRAPH_ID,
            Fields.IA_DATA_IA_ID_COL_NAME,
        ],
        value_vars=[fp_col, tf_col],
        var_name='metric',
        value_name='value',
    )
    long_df['metric'] = long_df['metric'].map({fp_col: 'FP', tf_col: 'TF'})
    # No fill_value here: missing combinations come out as NaN. The helper
    # below restores 0 within paragraphs the participant actually read.
    wide = long_df.pivot_table(
        index=list(participant_groupby_columns),
        columns=[
            Fields.UNIQUE_PARAGRAPH_ID,
            Fields.IA_DATA_IA_ID_COL_NAME,
            'metric',
        ],
        values='value',
    )
    wide = _fill_zero_for_read_paragraphs(wide, ia_df, participant_groupby_columns)
    # int(ia_id): MECO's harmonized IA_ID is float64 where OneStop's is int64,
    # so without the cast MECO columns would be named "..._1.0_..." and could
    # not be parsed back to a word index. The cast only affects the names; the
    # pivot already groups 1.0 and 1 as the same key.
    wide.columns = [
        f'{WFC_FEATURE_PREFIX}{p}_{int(ia_id)}_{m}' for p, ia_id, m in wide.columns
    ]
    return wide


def validate_ia_id_alignment(
    ia_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None,
    dataset_name: str,
) -> None:
    """Sanity check: within a content_group (OneStop) or globally (MECO),
    the same (unique_paragraph_id, IA_ID) must map to a single IA_LABEL
    across all participants. If not, our per-text feature columns would
    silently collapse different words into the same column.
    """
    if dataset_name in (DataSets.ONESTOPL1, DataSets.ONESTOPL2):
        if metadata_df is None or Fields.CONTENT_GROUP not in metadata_df.columns:
            raise ValueError(
                "validate_ia_id_alignment requires metadata with "
                f"{Fields.CONTENT_GROUP} column for OneStop datasets."
            )
        df = ia_df.merge(
            metadata_df[[Fields.SUBJECT_ID, Fields.CONTENT_GROUP]],
            on=Fields.SUBJECT_ID, how='left',
        )
        group_keys = [
            Fields.CONTENT_GROUP,
            Fields.UNIQUE_PARAGRAPH_ID,
            Fields.IA_DATA_IA_ID_COL_NAME,
        ]
    else:
        df = ia_df
        group_keys = [
            Fields.UNIQUE_PARAGRAPH_ID,
            Fields.IA_DATA_IA_ID_COL_NAME,
        ]

    label_counts = df.groupby(group_keys)['IA_LABEL'].nunique()
    bad = label_counts[label_counts > 1]
    if not bad.empty:
        sample_keys = bad.index[:5]
        # For each bad key, list the distinct IA_LABEL values that collide.
        sample_rows = []
        for key in sample_keys:
            labels = df.set_index(group_keys).loc[[key], 'IA_LABEL']
            unique_labels = labels.dropna().unique().tolist()
            sample_rows.append({
                **dict(zip(group_keys, key if isinstance(key, tuple) else (key,))),
                'unique_labels': unique_labels,
                'n_unique': len(unique_labels),
                'n_rows': len(labels),
            })
        sample_df = pd.DataFrame(sample_rows)
        raise ValueError(
            f"IA_ID alignment broken: {len(bad)} (group, paragraph, IA_ID) "
            f"keys map to multiple IA_LABEL values. Sample of conflicting keys:\n"
            f"{sample_df.to_string(index=False)}"
        )


def compute_ia_participant_level_features(
    participant_df: pd.DataFrame,
    feature_groups: list[FeatureGroups],
) -> dict:
    """
    Compute IA participant-level features for the given feature groups.

    Args:
        participant_df (pd.DataFrame): The participant data.
        feature_groups (list[FeatureGroups]): The feature groups to compute and return.

    Returns:
        dict: Flat dict of feature name -> value for all selected feature groups.
    """
    result = {}
    feature_groups_set = set(feature_groups)
    if FeatureGroups.READING_SPEED in feature_groups_set:
        result.update({'reading_speed': calc_reading_speed(participant_df)})

    # Means are over all words. Never-fixated words contribute 0 to the duration
    # measures in both corpora: OneStop via convert_to_int_features in utils.py
    # (which maps DataViewer's "." missing marker to 0), MECO via the explicit
    # fill in meco.py. mean_SK is a first-pass skip rate in both. mean_RR
    # averages IA_RR, which is NaN outside first-pass-read words.
    if FeatureGroups.FIXATION_METRICS in feature_groups_set:
        result.update({
            'mean_FF': participant_df['IA_FIRST_FIXATION_DURATION'].mean(),
            'mean_FP': participant_df['IA_FIRST_RUN_DWELL_TIME'].mean(),
            'mean_TF': participant_df['IA_DWELL_TIME'].mean(),
            'mean_RP': participant_df['IA_REGRESSION_PATH_DURATION'].mean(),
            'mean_SK': participant_df['IA_SKIP'].mean(),
            'mean_RR': participant_df['IA_RR'].mean(),
        })

    if FeatureGroups.S_CLUSTERS in feature_groups_set:
        result.update(create_s_clusters_dict(participant_df, FIXATIONS_METRICS_CLUSTERS, PosCols.PTB_POS, normalize_RS=True))
        result.update(create_s_clusters_dict(participant_df, FIXATIONS_METRICS_CLUSTERS, PosCols.UNIVERSAL_POS, normalize_RS=True))

    if FeatureGroups.S_CLUSTERS_NO_NORM in feature_groups_set:
        result.update(create_s_clusters_dict(participant_df, FIXATIONS_METRICS_CLUSTERS, PosCols.PTB_POS, normalize_RS=False))
        result.update(create_s_clusters_dict(participant_df, FIXATIONS_METRICS_CLUSTERS, PosCols.UNIVERSAL_POS, normalize_RS=False))

    wp_coefs_needed = feature_groups_set & {FeatureGroups.WP_COEFS, FeatureGroups.WP_COEFS_NO_INTERCEPT}
    wp_coefs_no_norm_needed = feature_groups_set & {FeatureGroups.WP_COEFS_NO_NORM, FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT}

    if wp_coefs_needed:
        wp_coefs = find_wp_coefs(participant_df, FIXATIONS_METRICS_COEFS, normalize_RS=True)
        if FeatureGroups.WP_COEFS in feature_groups_set:
            result.update(wp_coefs)
        if FeatureGroups.WP_COEFS_NO_INTERCEPT in feature_groups_set:
            result.update({k: v for k, v in wp_coefs.items() if 'intercept' not in k})

    if wp_coefs_no_norm_needed:
        wp_coefs_no_norm = find_wp_coefs(participant_df, FIXATIONS_METRICS_COEFS, normalize_RS=False)
        if FeatureGroups.WP_COEFS_NO_NORM in feature_groups_set:
            result.update(wp_coefs_no_norm)
        if FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT in feature_groups_set:
            result.update({k: v for k, v in wp_coefs_no_norm.items() if 'intercept' not in k})

    return result


def compute_fixation_trial_level_features(
    trial: pd.DataFrame,
    feature_groups: list[FeatureGroups],
) -> dict:
    """
    Compute fixation-report features for the given feature groups.

    No feature group currently takes features from the fixation report, so this
    returns an empty dict; it is the hook for adding such features.
    """
    return {}


def create_s_clusters_dict_inner(fix_met_trial_df:pd.DataFrame, fixation_metrics:list[str], criterion:str, normalize_RS:bool=True) -> dict:
    """S-Clusters for one participant: the mean of each eye-movement metric per
    POS tag of `criterion`, optionally divided by the participant's overall mean.
    Expects the metric columns under their nicknames (see create_s_clusters_dict).
    """

    # Filter out rows with POS tags not in the allowed list (e.g., X, SYM for UNIVERSAL_POS)
    allowed_pos_values = POS_POSSIBLE_VALUES.get(criterion, [])
    if allowed_pos_values:
        fix_met_trial_df = fix_met_trial_df[fix_met_trial_df[criterion].isin(allowed_pos_values)]

    # Group by the criterion and calculate the average fixation times
    cluster_means = fix_met_trial_df.groupby(criterion)[fixation_metrics].mean().reset_index()
    total_means = fix_met_trial_df[fixation_metrics].mean().to_frame().T

    if normalize_RS:
        cluster_means[fixation_metrics] = cluster_means[fixation_metrics].div(total_means.iloc[0])

    # Rename the columns to include the criterion name
    cluster_means = cluster_means.rename(columns={col: f"{criterion}_{col}" for col in fixation_metrics})
    cluster_means = cluster_means[[criterion] + [f"{criterion}_{col}" for col in fixation_metrics]]

    cluster_means = flip_group_to_features(cluster_means, criterion, fixation_metrics)

    cluster_means_dict = cluster_means.to_dict(orient='records')[0]
    if not normalize_RS:
        cluster_means_dict = {f"{k}_no_norm": v for k, v in cluster_means_dict.items()}

    return cluster_means_dict


def create_s_clusters_dict(ia_trial_df:pd.DataFrame, fixation_metrics:list[str], criterion:str, normalize_RS:bool=True) -> dict:
    """S-Clusters for one participant's IA data (renames the metric columns to
    their nicknames, then calls create_s_clusters_dict_inner)."""
    fix_met_trial_df = ia_trial_df.rename(columns=EYE_METRICS_NICKNAMES_INVERTED)
    return create_s_clusters_dict_inner(fix_met_trial_df, fixation_metrics, criterion, normalize_RS)


def flip_group_to_features(df: pd.DataFrame, criterion:str, fixation_metrics:list) -> pd.DataFrame:
    """Turn one row per POS tag into a single row with one column per
    (tag, metric), named "<tag>_<criterion>_<metric>"."""
    df_long = df.melt(id_vars=[criterion],
                      value_vars=[f"{criterion}_{col}" for col in fixation_metrics],
                      var_name="measure", value_name="value")

    df_pivot = df_long.pivot_table(
        columns=[criterion, "measure"],
        values="value"
    )

    df_pivot.columns = [f"{c}_{measure}" for c, measure in df_pivot.columns]
    df_pivot = df_pivot.reset_index(drop=True)
    return df_pivot


def calc_reading_speed(participant_df: pd.DataFrame) -> float:
    """Words per minute over all of the participant's trials."""
    num_of_words = len(participant_df)
    # PARAGRAPH_RT is constant within a trial, so take it from the first row.
    paragraph_reading_times = participant_df.groupby(Fields.UNIQUE_TRIAL_ID)['PARAGRAPH_RT'].first()/1000/60 # convert from ms to minutes
    return (
        num_of_words / paragraph_reading_times.sum()
    )


# Backstop bound on the logistic WP_COEFS *slopes*. The intercept is
# deliberately excluded: the word properties are uncentered (wordfreq_frequency
# runs 4-36), so the intercept extrapolates to a zero-length, zero-frequency
# word far outside the data and is legitimately large whenever the event rate is
# low -- e.g. an item fit at intercept -27.5 with slopes of 0.88/1.17/-0.63 and
# a likelihood that is a genuine maximum. Bounding max|coef| across all four
# params instead flagged 7 such healthy fits per 1500. The slopes themselves
# never exceed 7.1 on any fit whose MLE exists.
MAX_ABS_LOGIT_SLOPE = 10.0


def find_wp_coefs(ia_trial_df: pd.DataFrame, fixation_metrics:list[str], normalize_RS:bool=True) -> dict:
    """WP-Coefs for one participant's IA data (renames the metric columns to
    their nicknames, then calls find_wp_coefs_inner)."""
    fix_met_trial_df = ia_trial_df.rename(columns=EYE_METRICS_NICKNAMES_INVERTED)
    return find_wp_coefs_inner(fix_met_trial_df, fixation_metrics, normalize_RS)


def _reject_separated_logit(model, eye_metric: str) -> None:
    """Raise if a logistic WP_COEFS fit is (quasi-)separated rather than fitted.

    Separation means the maximum-likelihood estimate does not exist: the
    likelihood keeps climbing as the coefficients are scaled outward, and what
    gets returned is wherever the optimizer happened to stop. statsmodels
    signals none of this -- it raises nothing, and for roughly a third of these
    fits it reports `converged=True` with a finite log-likelihood and no
    warning at all, because the likelihood surface is flat enough out there
    that the gradient falls under tolerance. So the condition has to be tested
    here.

    The test for the silent cases is definitional rather than a threshold: walk
    the coefficient vector outward and see whether the likelihood follows. A
    real maximum drops off; a separated fit does not. On 1500 per-item fits
    this agreed exactly with a ground-truth labelling of which fits have an
    MLE -- no false positives, nothing missed.

    Args:
        model: A fitted statsmodels Logit result.
        eye_metric: Metric name, for the message.

    Raises:
        RuntimeError: when the fit is unusable. The caller's except clause
            turns it into NaN coefs for this metric.
    """
    if not model.mle_retvals.get("converged", False):
        raise RuntimeError(f"{eye_metric}: logistic fit did not converge (separation)")
    if not np.isfinite(model.llf):
        raise RuntimeError(f"{eye_metric}: logistic fit has non-finite log-likelihood "
                           "(complete separation)")

    # No MLE exists if scaling the estimate outward does not lower the likelihood.
    params = model.params.to_numpy()
    if model.model.loglike(2.0 * params) >= model.model.loglike(params) - 1e-9:
        raise RuntimeError(f"{eye_metric}: likelihood does not peak at the estimate "
                           "-- no MLE exists (separation)")

    # Backstop, on the slopes only -- see MAX_ABS_LOGIT_SLOPE.
    max_slope = float(np.max(np.abs(params[1:]))) if params.size > 1 else 0.0
    if max_slope > MAX_ABS_LOGIT_SLOPE:
        raise RuntimeError(f"{eye_metric}: logistic slope {max_slope:.4g} exceeds "
                           f"{MAX_ABS_LOGIT_SLOPE}")


def find_wp_coefs_inner(subject_df: pd.DataFrame, fixation_metrics:list[str], normalize_RS:bool=True) -> dict:
    """WP-Coefs for one participant: per metric, the coefficients of a regression
    of the metric on the word properties (WORD_PROPERTY_COLUMNS); logistic for
    the binary SK and RR, OLS otherwise."""
    # Metrics with binary per-word outcomes (SK = first-pass skip, RR =
    # first-pass regression-out indicator). They are fit with logistic
    # regression and never y-normalized — dividing a 0/1 outcome by its mean
    # would break the binary assumption. Their coefficients are numerically
    # identical regardless of normalize_RS and are emitted under both the
    # no-suffix and _no_norm name variants.
    logistic_metrics = {"SK", "RR"}

    coefs = {}
    linear_metrics = [m for m in fixation_metrics if m not in logistic_metrics]

    subject_df = subject_df.copy()
    if normalize_RS and linear_metrics:
        total_means = subject_df[linear_metrics].mean().to_frame().T
        subject_df[linear_metrics] = subject_df[linear_metrics].div(total_means.iloc[0])

    for eye_metric in fixation_metrics:
        X = sm.add_constant(subject_df[WORD_PROPERTY_COLUMNS])
        y = subject_df[eye_metric]

        try:
            if eye_metric in logistic_metrics:
                model = sm.Logit(y, X, missing='drop').fit(disp=0)
                _reject_separated_logit(model, eye_metric)
            else:
                model = sm.OLS(y, X, missing='drop').fit()
        except Exception as exc:
            # Common cause: y is all-NaN after dropna (e.g. ch_s participants
            # in MecoL2 have IA_FIRST_FIX_PROGRESSIVE==0 for every row, so
            # IA_RR is uniformly NaN). Other causes: perfect separation,
            # singular design matrix. Emit NaN coefs so the participant gets
            # written out with the other features and the pipeline continues.
            pid = (
                subject_df['participant_id'].iloc[0]
                if 'participant_id' in subject_df.columns
                else 'unknown'
            )
            logger.warning(
                f"WP_COEFS fit failed for participant={pid} metric={eye_metric}: "
                f"{type(exc).__name__}: {exc}"
            )
            coefs[f"{eye_metric}_intercept"] = np.nan
            for wp in WORD_PROPERTY_COLUMNS:
                coefs[f"{eye_metric}_{wp}_coef"] = np.nan
            continue

        coefs[f"{eye_metric}_intercept"] = model.params['const']
        for wp in WORD_PROPERTY_COLUMNS:
            coefs[f"{eye_metric}_{wp}_coef"] = model.params[wp]

    if not normalize_RS:
        coefs = {f"{k}_no_norm": v for k, v in coefs.items()}

    return coefs


# ---------------------------------------------------------------- aggregation variants


def create_moving_agg_features(
    harmonized_fixation_data: pd.DataFrame | None,
    harmonized_ia_data: pd.DataFrame,
    dataset_name: str,
    p_agg_level: Literal["paragraph", "article"] = "paragraph",
    window_size: int = 1,
    feature_groups_to_add: list[FeatureGroups] = list(FeatureGroups),
) -> pd.DataFrame:
    """
    Create moving window aggregations for features.

    For each participant and each window position:
    - Filter data to only that window's trials/paragraphs
    - Aggregate features for the window
    - Add window_index column

    Args:
        harmonized_fixation_data: Raw fixation data or None
        harmonized_ia_data: Raw IA data
        dataset_name: Dataset identifier (OneStop or Meco)
        p_agg_level: "paragraph" or "article"
        window_size: Number of consecutive paragraphs/articles per window
        feature_groups_to_add: Which feature groups to compute

    Returns:
        DataFrame with columns: [subject_id, window_index, feature1, feature2, ...]
    """

    all_results = []
    participants = sorted(harmonized_ia_data[Fields.SUBJECT_ID].unique())

    for participant_id in participants:
        participant_ia = harmonized_ia_data[
            harmonized_ia_data[Fields.SUBJECT_ID] == participant_id
        ]

        if harmonized_fixation_data is not None:
            participant_fix = harmonized_fixation_data[
                harmonized_fixation_data[Fields.SUBJECT_ID] == participant_id
            ]
        else:
            participant_fix = None

        # Determine ordering and get unique trials
        if dataset_name in [DataSets.ONESTOPL1, DataSets.ONESTOPL2]:
            if p_agg_level == "paragraph":
                # Order by TRIAL_INDEX for OneStop paragraphs
                unique_trials = sorted(participant_ia[Fields.TRIAL_INDEX].unique())
                order_col = Fields.TRIAL_INDEX
            elif p_agg_level == "article":
                # For articles, get unique articles in order of first appearance (by TRIAL_INDEX)
                unique_trials = participant_ia.drop_duplicates(
                    subset=[Fields.ARTICLE_ID], keep='first'
                ).sort_values(Fields.TRIAL_INDEX)[Fields.ARTICLE_ID].values
                order_col = Fields.ARTICLE_ID
        else:
            # For Meco, windows are defined by paragraph indices, include participants with any in that range
            participant_paragraphs = sorted(participant_ia[Fields.UNIQUE_PARAGRAPH_ID].unique())
            max_paragraph = max(participant_paragraphs) if participant_paragraphs else 0
            order_col = Fields.UNIQUE_PARAGRAPH_ID

        # Create moving windows
        if dataset_name in [DataSets.ONESTOPL1, DataSets.ONESTOPL2]:
            num_trials = len(unique_trials)
            # Always create at least one window, even if partial (smaller than window_size)
            num_windows = max(1, num_trials - window_size + 1)
            window_indices = range(num_windows)
        else:
            # For Meco, windows are paragraph-index ranges: [1, window_size],
            # [2, window_size + 1], ... Same count rule as OneStop above.
            num_windows = max(1, max_paragraph - window_size + 1)
            window_indices = range(num_windows)

        # window_index is a per-participant counter ("the n-th window this
        # participant contributed"), not the loop variable: windows that match
        # no paragraph are skipped below, so for MECO the loop variable can have
        # holes. Consequently a MECO window index does not identify an absolute
        # paragraph position.
        emitted = 0
        for window_start_idx in window_indices:
            if dataset_name in [DataSets.ONESTOPL1, DataSets.ONESTOPL2]:
                window_end_idx = window_start_idx + window_size
                window_trials = unique_trials[window_start_idx:window_end_idx]
            else:
                # For Meco, define window by paragraph indices (1-indexed: 1, 2, 3, ...)
                window_trial_indices = set(range(window_start_idx + 1, window_start_idx + window_size + 1))
                window_trials = [p for p in participant_paragraphs if p in window_trial_indices]

            if not window_trials:
                continue

            window_ia = participant_ia[participant_ia[order_col].isin(window_trials)]
            window_fix = participant_fix[participant_fix[order_col].isin(window_trials)] if participant_fix is not None else None

            if len(window_ia) == 0:
                continue

            # Compute features for this window
            window_features = compute_ia_participant_level_features(window_ia, feature_groups_to_add)
            if window_fix is not None and len(window_fix) > 0:
                window_fix_features = compute_fixation_trial_level_features(window_fix, feature_groups_to_add)
                window_features.update(window_fix_features)

            # Add participant and window index
            window_features[Fields.SUBJECT_ID] = participant_id
            window_features['window_index'] = emitted
            emitted += 1

            all_results.append(window_features)

    # Create DataFrame from results
    result_df = pd.DataFrame(all_results)

    # Identifiers first. They are added after the features above, so plain dict
    # insertion order buries them among the metric columns.
    if not result_df.empty:
        lead = [c for c in (Fields.SUBJECT_ID, 'window_index') if c in result_df.columns]
        result_df = result_df[lead + [c for c in result_df.columns if c not in lead]]

    # Undefined stays undefined, as in compute_grouped_features. A window spans
    # only a few paragraphs, so an undefined cell is commoner here than at
    # fully-agg — a rare POS tag need only be absent from those paragraphs, and
    # RR is undefined for a whole site. The models impute per fold.
    return result_df


def add_moving_agg_features(
    harmonized_fixation_data: pd.DataFrame | None,
    harmonized_ia_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    participant_groupby_columns: list[str],
    dataset_name: str,
    features_and_md_path: Path,
    p_agg_level: Literal["paragraph", "article"] = "paragraph",
    window_size: int = 1,
    feature_groups_to_add: list[FeatureGroups] = list(FeatureGroups),
    replace_file: bool = False,
) -> pd.DataFrame:
    """
    Create and save moving window aggregated features.

    Similar to add_participant_level_features but creates multiple rows per participant (one per window).

    Args:
        harmonized_fixation_data: Raw fixation data or None
        harmonized_ia_data: Raw IA data
        metadata_df: Metadata dataframe
        dataset_name: Dataset identifier
        features_and_md_path: Path to save features CSV
        p_agg_level: "paragraph" or "article"
        window_size: Number of consecutive items per window
        feature_groups_to_add: Which feature groups to compute
        replace_file: Whether to recompute even if file exists

    Returns:
        DataFrame with moving aggregated features
    """

    if replace_file or not features_and_md_path.exists():
        if metadata_df is None:
            raise ValueError('`metadata_df` is required when creating/replacing feature file.')
        features_and_md_path.parent.mkdir(parents=True, exist_ok=True)
        features_and_md_df = metadata_df.copy()
    else:
        features_and_md_df = pd.read_csv(features_and_md_path)

    logger.info(f'Creating moving aggregated features with window_size={window_size}, p_agg_level={p_agg_level}')

    df_to_add = create_moving_agg_features(
        harmonized_fixation_data,
        harmonized_ia_data,
        dataset_name,
        p_agg_level=p_agg_level,
        window_size=window_size,
        feature_groups_to_add=feature_groups_to_add,
    )

    # A row is identified by (participant, window). On a fresh build
    # features_and_md_df is the metadata (one row per participant), so the merge
    # fans out to one row per window. On a top-up build it is this function's
    # own earlier output, already one row per (participant, window), so the
    # window column must be part of the key or the merge becomes a cartesian
    # product. `validate` below asserts the multiplicity.
    merge_keys = list(participant_groupby_columns)
    if 'window_index' in features_and_md_df.columns and 'window_index' in df_to_add.columns:
        merge_keys.append('window_index')

    # Drop any columns that already exist so they get replaced — never the merge
    # keys, which both sides need to keep.
    existing_cols = [c for c in df_to_add.columns if c not in merge_keys and c in features_and_md_df.columns]
    features_and_md_df = features_and_md_df.drop(columns=existing_cols)

    # Participants with metadata but no eye-tracking data (or vice versa) are dropped.
    if set(features_and_md_df[Fields.SUBJECT_ID]) != set(df_to_add[Fields.SUBJECT_ID].unique()):
        missing_participants = set(features_and_md_df[Fields.SUBJECT_ID]) - set(df_to_add[Fields.SUBJECT_ID].unique())
        missing_participants_2 = set(df_to_add[Fields.SUBJECT_ID].unique()) - set(features_and_md_df[Fields.SUBJECT_ID])
        logger.warning(
            f"Mismatch between participants in metadata and computed features, \n missing participants:"\
            f"{list(missing_participants)} "\
            f"{list(missing_participants_2)} "\
            "missing participants will be dropped from features"
        )

    features_and_md_df = features_and_md_df.merge(
        df_to_add,
        on=merge_keys,
        how='inner',
        # Fresh build: metadata is unique per participant and fans out over
        # windows. Top-up build: the key is complete, so it must be row-for-row.
        validate='1:1' if len(merge_keys) > len(participant_groupby_columns) else '1:m',
    )

    features_and_md_df.to_csv(features_and_md_path, index=False)
    return features_and_md_df


def add_participant_level_features(
    harmonized_fixation_data: pd.DataFrame | None,
    harmonized_ia_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    participant_groupby_columns: list[str],
    features_and_md_path: Path,
    feature_groups_to_add: list[FeatureGroups],
    replace_file: bool = False,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """
    Compute participant-level features and save them, merged with the metadata,
    to features_and_md_path. Per-text features (TRANSITIONS, WFC) go to a
    sibling parquet.

    Args:
        harmonized_fixation_data: Harmonized fixation data, or None.
        harmonized_ia_data: Harmonized IA data.
        metadata_df: Participant metadata (targets and grouping columns).
        participant_groupby_columns: The columns that identify a participant.
        features_and_md_path: Where to save the features and metadata CSV.
        feature_groups_to_add: The feature groups to compute.
        replace_file: Start from the metadata even if the CSV already exists
            (otherwise the new features are added to the existing file).
        dataset_name: Dataset name; required for the per-text feature groups.

    Returns:
        pd.DataFrame: The saved features and metadata.
    """
    if replace_file or not features_and_md_path.exists():
        if metadata_df is None:
            raise ValueError('`metadata_df` is required when creating/replacing feature file.')
        features_and_md_path.parent.mkdir(parents=True, exist_ok=True)
        features_and_md_df = metadata_df.copy()
    else:
        features_and_md_df = pd.read_csv(features_and_md_path)

    df_to_add = compute_participant_level_feature_group(
        feature_groups=feature_groups_to_add,
        harmonized_fixations_data=harmonized_fixation_data,
        harmonized_ia_data=harmonized_ia_data,
        participant_groupby_columns=participant_groupby_columns,
    )

    # Drop any columns that already exist (excluding groupby keys) so they get replaced
    existing_cols = [c for c in df_to_add.columns if c not in participant_groupby_columns and c in features_and_md_df.columns]
    features_and_md_df = features_and_md_df.drop(columns=existing_cols)

    # Participants with metadata but no eye-tracking data (or vice versa) are dropped.
    if set(features_and_md_df[Fields.SUBJECT_ID]) != set(df_to_add.index.get_level_values(Fields.SUBJECT_ID)):
        missing_participants = set(features_and_md_df[Fields.SUBJECT_ID]) - set(df_to_add.index.get_level_values(Fields.SUBJECT_ID))
        missing_participants_2 = set(df_to_add.index.get_level_values(Fields.SUBJECT_ID)) - set(features_and_md_df[Fields.SUBJECT_ID])
        logger.warning(
            f"Mismatch between participants in metadata and computed features, \n missing participants:"\
            f"{list(missing_participants)} "\
            f"{list(missing_participants_2)} "\
            "missing participants will be dropped from features"
        )

    features_and_md_df = features_and_md_df.merge(
        df_to_add,
        on=participant_groupby_columns,
        how='inner',
    )

    features_and_md_df.to_csv(features_and_md_path, index=False)

    # Per-text features are written to a sibling file so the main CSV stays
    # small. Downstream code that doesn't use TRANSITIONS / WFC never reads
    # this file.
    per_text_df = compute_per_text_features(
        feature_groups=feature_groups_to_add,
        harmonized_fixations_data=harmonized_fixation_data,
        harmonized_ia_data=harmonized_ia_data,
        participant_groupby_columns=participant_groupby_columns,
        metadata_df=metadata_df,
        dataset_name=dataset_name,
    )
    per_text_path = features_and_md_path.with_name(PER_TEXT_FEATURES_FILENAME)
    if per_text_df is not None:
        # Bundle metadata (subject id, L1, batch, content_group, target
        # columns, …) into the parquet so consumers can use it as a
        # drop-in replacement for the main CSV — no merge required at
        # read time.
        metadata_only_df = features_and_md_df.drop(columns=list(df_to_add.columns))
        per_text_full_df = metadata_only_df.merge(
            per_text_df,
            on=participant_groupby_columns,
            how='inner',
        )
        per_text_full_df.to_parquet(per_text_path, index=False)
        logger.info(
            f"Saved {per_text_df.shape[1]} per-text feature columns "
            f"(+ {metadata_only_df.shape[1]} metadata columns) to {per_text_path}"
        )
    elif (replace_file and per_text_path.exists()
          and set(feature_groups_to_add) & {FeatureGroups.TRANSITIONS, FeatureGroups.WFC}):
        # Per-text groups were requested but produced nothing — the existing
        # parquet is stale. When they were skipped entirely, leave it alone:
        # replace only replaces what this run extracts.
        per_text_path.unlink()

    return features_and_md_df


def add_per_item_features(
    harmonized_fixation_data: pd.DataFrame | None,
    harmonized_ia_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    participant_groupby_columns: list[str],
    features_and_md_path: Path,
    feature_groups_to_add: list[FeatureGroups],
    p_agg_level: str,
    replace_file: bool = False,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """
    Compute per-item features (one row per participant × item).

    For each paragraph/article, computes features aggregated over only that item's readings.
    Outputs multiple rows per participant (one per item).

    Args:
        harmonized_fixation_data: Raw fixation data or None
        harmonized_ia_data: Raw IA data
        metadata_df: Metadata dataframe
        participant_groupby_columns: Columns to group by for participants (typically [Fields.SUBJECT_ID])
        features_and_md_path: Path to save features CSV
        feature_groups_to_add: Which feature groups to compute
        p_agg_level: "paragraph" or "article"
        replace_file: Whether to recompute even if file exists
        dataset_name: Dataset identifier (required for per-text features)

    Returns:
        DataFrame with per-item features (multiple rows per participant)
    """
    # Determine item column first (needed for checks)
    item_col = Fields.UNIQUE_PARAGRAPH_ID if p_agg_level == "paragraph" else Fields.ARTICLE_ID

    # Check if there are any non-per-text feature groups (regular features to compute)
    regular_feature_groups = [fg for fg in feature_groups_to_add if fg not in {FeatureGroups.TRANSITIONS, FeatureGroups.WFC}]

    # Safety: if regular features file exists and not replacing, don't overwrite
    skip_regular_computation = False
    if features_and_md_path.exists() and not replace_file and regular_feature_groups:
        logger.info(f"Regular features already exist at {features_and_md_path} and replace_file=False. Will skip regular feature computation.")
        skip_regular_computation = True

    if (replace_file or not features_and_md_path.exists()) and regular_feature_groups:
        if metadata_df is None:
            raise ValueError('`metadata_df` is required when creating/replacing feature file.')
        features_and_md_path.parent.mkdir(parents=True, exist_ok=True)

    # Get unique items
    unique_items = harmonized_ia_data[item_col].unique()
    logger.info(f'Computing per-item features for {len(unique_items)} {p_agg_level}(s), {len(harmonized_ia_data)} total rows')

    if not regular_feature_groups and features_and_md_path.exists():
        logger.info("No regular features to compute (only per-text groups), and regular features already exist. Skipping regular feature computation.")
        # If we're only computing per-text features and regular features already exist, load them
        features_and_md_df = pd.read_csv(features_and_md_path)
        # Create a dummy df_to_add with the per-item structure from the loaded file
        df_to_add = features_and_md_df[[Fields.SUBJECT_ID, item_col]].drop_duplicates().reset_index(drop=True)
        skip_regular_computation = True
    elif skip_regular_computation and features_and_md_path.exists():
        logger.info(f"Loading existing regular features from {features_and_md_path}")
        features_and_md_df = pd.read_csv(features_and_md_path)
        # Create a dummy df_to_add with the per-item structure from the loaded file
        df_to_add = features_and_md_df[[Fields.SUBJECT_ID, item_col]].drop_duplicates().reset_index(drop=True)
    else:
        # Compute features for each item
        all_item_features = []

        # Pre-group data by item for efficiency (avoid O(n) filtering in the loop)
        ia_grouped = {item_id: group for item_id, group in harmonized_ia_data.groupby(item_col)}
        fix_grouped = None
        if harmonized_fixation_data is not None:
            fix_grouped = {item_id: group for item_id, group in harmonized_fixation_data.groupby(item_col)}

        for item_id in unique_items:
            # Get pre-grouped data for this item
            ia_for_item = ia_grouped.get(item_id)
            fix_for_item = fix_grouped.get(item_id) if fix_grouped else None

            if ia_for_item is None or ia_for_item.empty:
                continue

            # Filter fixations to only participants in ia_for_item to avoid phantom rows
            if fix_for_item is not None and not fix_for_item.empty:
                ia_participants = ia_for_item[participant_groupby_columns[0]].unique()
                fix_for_item = fix_for_item[fix_for_item[participant_groupby_columns[0]].isin(ia_participants)]

            # Compute features for this item (groups by participant within this item)
            item_features = compute_participant_level_feature_group(
                feature_groups=feature_groups_to_add,
                harmonized_fixations_data=fix_for_item,
                harmonized_ia_data=ia_for_item,
                participant_groupby_columns=participant_groupby_columns,
            )

            # Add item_id column and reset index
            item_features_reset = item_features.reset_index()
            item_col_str = item_col.value if isinstance(item_col, Fields) else item_col
            item_features_reset[item_col_str] = item_id
            all_item_features.append(item_features_reset)

        # Concatenate all items
        df_to_add = pd.concat(all_item_features, ignore_index=True)

        # Merge with metadata
        features_and_md_df = metadata_df.copy()

        # Check for participant mismatch
        if set(features_and_md_df[Fields.SUBJECT_ID]) != set(df_to_add[Fields.SUBJECT_ID].unique()):
            missing_in_features = set(features_and_md_df[Fields.SUBJECT_ID]) - set(df_to_add[Fields.SUBJECT_ID].unique())
            missing_in_metadata = set(df_to_add[Fields.SUBJECT_ID].unique()) - set(features_and_md_df[Fields.SUBJECT_ID])
            logger.warning(
                f"Mismatch between participants in metadata and computed features. "
                f"Missing in features: {list(missing_in_features)}. "
                f"Missing in metadata: {list(missing_in_metadata)}. "
                "Participants will be kept only if in both metadata and features."
            )

        # Merge on participant only (metadata has 1 row/participant, expanded to match per-item rows)
        features_and_md_df = features_and_md_df.merge(
            df_to_add,
            on=participant_groupby_columns,
            how='inner',
        )

        features_and_md_df.to_csv(features_and_md_path, index=False)

    # Per-text features (WFC/TRANSITIONS) - compute per-item to avoid massive dataframe
    # Check if we need per-text features at all
    per_text_groups = [fg for fg in feature_groups_to_add if fg in {FeatureGroups.TRANSITIONS, FeatureGroups.WFC}]

    per_text_path = features_and_md_path.with_name(PER_TEXT_FEATURES_FILENAME)
    if per_text_groups:
        per_text_path.parent.mkdir(parents=True, exist_ok=True)

        # Pre-group data by item for efficiency (avoid O(n) filtering in the loop)
        ia_grouped_for_pertext = {item_id: group for item_id, group in harmonized_ia_data.groupby(item_col)}
        fix_grouped_for_pertext = None
        if harmonized_fixation_data is not None:
            fix_grouped_for_pertext = {item_id: group for item_id, group in harmonized_fixation_data.groupby(item_col)}

        num_items_saved = 0
        for item_id in unique_items:
            # Check if this item's parquet already exists
            item_parquet_path = per_text_path.parent / f"per_text_features_item_{item_id}.parquet"
            if item_parquet_path.exists() and not replace_file:
                logger.debug(f"Skipping {item_col}={item_id}: parquet already exists")
                continue

            # Get data for this item only
            ia_for_item = ia_grouped_for_pertext.get(item_id)
            fix_for_item = fix_grouped_for_pertext.get(item_id) if fix_grouped_for_pertext else None

            if ia_for_item is None or ia_for_item.empty:
                continue

            # Filter fixations to only participants in ia_for_item to avoid phantom rows
            if fix_for_item is not None and not fix_for_item.empty:
                ia_participants = ia_for_item[participant_groupby_columns[0]].unique()
                fix_for_item = fix_for_item[fix_for_item[participant_groupby_columns[0]].isin(ia_participants)]

            # Compute per-text features for just this item
            per_text_item_df = compute_per_text_features(
                feature_groups=feature_groups_to_add,
                harmonized_fixations_data=fix_for_item,
                harmonized_ia_data=ia_for_item,
                participant_groupby_columns=participant_groupby_columns,
                metadata_df=metadata_df,
                dataset_name=dataset_name,
            )

            if per_text_item_df is not None:
                # Expand to per-item structure and add metadata
                per_item_index_for_item = df_to_add[df_to_add[item_col] == item_id][[Fields.SUBJECT_ID, item_col]].drop_duplicates()

                per_text_reset = per_text_item_df.reset_index()
                per_text_expanded = per_item_index_for_item.merge(
                    per_text_reset,
                    on=participant_groupby_columns,
                    how='left',
                )

                # Add metadata for this item
                metadata_cols = [c for c in metadata_df.columns if c not in df_to_add.columns]
                if metadata_cols:
                    metadata_unique_for_item = features_and_md_df[features_and_md_df[item_col] == item_id][[Fields.SUBJECT_ID, item_col] + metadata_cols].drop_duplicates()
                    per_text_full_df_item = per_text_expanded.merge(
                        metadata_unique_for_item,
                        on=[Fields.SUBJECT_ID, item_col],
                        how='left',
                    )
                else:
                    per_text_full_df_item = per_text_expanded

                # Save immediately
                per_text_full_df_item.to_parquet(item_parquet_path, index=False)
                num_items_saved += 1

                if num_items_saved % 50 == 0:
                    logger.info(f"Saved {num_items_saved} per-item per-text parquets...")

        logger.info(f"Completed: Saved {num_items_saved} per-item per-text parquets to {per_text_path.parent}")

        # Delete old combined parquet if it exists
        if per_text_path.exists():
            per_text_path.unlink()

    elif replace_file and per_text_path.exists() and per_text_groups:
        # Same rule as above: only treat the combined parquet as stale when
        # per-text groups were actually requested by this run.
        per_text_path.unlink()

    return features_and_md_df


# ---------------------------------------------------------------- split-half reliability


def generate_split_definitions(
    df_ia: pd.DataFrame,
    dataset: str,
    n_splits: int = 20,
    p_agg_level: str = "paragraph",
    odd_even: str = None,
) -> dict:
    """Generate split definitions for reliability analysis splits.

    For paragraph level: randomly splits paragraphs in half.
    For article level (OneStop only): randomly splits articles in half (5 vs 5).
    For MECO odd_even: splits odd (1,3,5,7,9,11) or even (2,4,6,8,10,12) separately.

    This is computed once from paired-aligned data and reused for both L1 and L2
    to ensure they have identical splits.

    Args:
        df_ia: IA data (should be paired-aligned for OneStop L1+L2)
        dataset: Dataset name
        n_splits: Number of random splits
        p_agg_level: "paragraph" or "article" (article only for OneStop)
        odd_even: "odd" or "even" (MECO only, for split-half analysis)

    Returns:
        Dict mapping {split_idx: {half_num: set(paragraph_ids or article_ids)}}
    """
    from src.preprocessing.run_extract_features import filter_rows_full_reading, filter_rows_by_conditions

    HALF_SPLIT_SEED = 42
    is_onestop = dataset in [DataSets.ONESTOPL1, DataSets.ONESTOPL2]

    if p_agg_level == "article" and not is_onestop:
        raise ValueError(f"Article-level splits only supported for OneStop, got {dataset}")

    if odd_even is not None and is_onestop:
        raise ValueError(f"odd_even parameter only supported for MECO, got {dataset}")

    # Filter data
    df_ia_f = filter_rows_full_reading(df_ia, dataset)
    df_ia_f = filter_rows_by_conditions(df_ia_f, dataset_name=dataset, level='all', reread='ordinary', keep_practice=False)

    split_defs = {}

    for split_idx in range(1, n_splits + 1):
        split_defs[split_idx] = {}
        # One permutation per split; the two halves are complementary slices of
        # the same shuffle, so they are disjoint and jointly exhaustive.
        np.random.seed(HALF_SPLIT_SEED + split_idx)

        if is_onestop and p_agg_level == "article":
            # For OneStop article level: split articles (5 vs 5)
            all_articles = sorted(df_ia_f[Fields.ARTICLE_ID].unique())
            shuffled_articles = np.random.permutation(all_articles)
            split_pt = len(shuffled_articles) // 2
            halves = {1: set(shuffled_articles[:split_pt]),
                      2: set(shuffled_articles[split_pt:])}
            for half_num, articles_in_half in halves.items():
                # Get all paragraphs (PARAGRAPH_ID) belonging to these articles
                split_defs[split_idx][half_num] = set(
                    df_ia_f[df_ia_f[Fields.ARTICLE_ID].isin(articles_in_half)][Fields.UNIQUE_PARAGRAPH_ID].unique()
                )

        elif is_onestop:
            # For OneStop paragraph level: split (article_id, paragraph_id) pairs evenly
            # to ensure each article contributes balanced paragraphs to each half
            all_paragraphs = sorted(df_ia_f[[Fields.ARTICLE_ID, Fields.PARAGRAPH_ID]].drop_duplicates().values.tolist())
            shuffled = np.random.permutation(all_paragraphs)
            split_pt = len(shuffled) // 2
            halves = {1: set(map(tuple, shuffled[:split_pt])),
                      2: set(map(tuple, shuffled[split_pt:]))}
            for half_num, paragraphs_in_half_base in halves.items():
                # All UNIQUE_PARAGRAPH_IDs (one per difficulty level) of each (article, paragraph) pair
                unique_paragraphs_in_half = set()
                for article_id, paragraph_id in paragraphs_in_half_base:
                    unique_paragraphs = df_ia_f[
                        (df_ia_f[Fields.ARTICLE_ID] == article_id) &
                        (df_ia_f[Fields.PARAGRAPH_ID] == paragraph_id)
                    ][Fields.UNIQUE_PARAGRAPH_ID].unique()
                    unique_paragraphs_in_half.update(unique_paragraphs)
                split_defs[split_idx][half_num] = unique_paragraphs_in_half

        else:
            # For MECO: partition paragraphs 1-12, or only the odd / even ones
            if odd_even == "odd":
                paragraphs_to_split = [1, 3, 5, 7, 9, 11]
            elif odd_even == "even":
                paragraphs_to_split = [2, 4, 6, 8, 10, 12]
            else:
                paragraphs_to_split = list(range(1, 13))

            shuffled = np.random.permutation(paragraphs_to_split)
            split_pt = len(shuffled) // 2
            split_defs[split_idx][1] = set(shuffled[:split_pt])
            split_defs[split_idx][2] = set(shuffled[split_pt:])

    return split_defs
