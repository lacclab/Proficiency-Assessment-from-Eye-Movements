import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import spacy
from loguru import logger
from tqdm import tqdm
from text_metrics.ling_metrics_funcs import get_metrics
from text_metrics.surprisal_extractors.extractor_switch import get_surp_extractor
from text_metrics.surprisal_extractors.extractors_constants import SurpExtractorType

from src.constants import (
    MERGE_IA_FIXATION_COLS,
    OSF_RESOURCE_IDS,
    TRIAL_IDENTIFIER_COLS,
    DataSets,
    DataType,
    Fields,
    TestCols,
)
from src.preprocessing.dataset_preprocessing.base import DatasetProcessor
from src.preprocessing.dataset_preprocessing.meco_leapq_maps import (
    CANONICAL_YEAR_MONTH_MAP,
    LEAP_Q_VARIANT_MAPS,
    MECO_PARTICIPANTS_COLUMN_MAP,
    MECO_W1_COMMON_RENAME_MAP,
)
from src.preprocessing.utils import add_missing_features, download_from_osf, load_cp1255_rda

METADATA_RENAME_MAP = {
    'uniform_id': Fields.SUBJECT_ID,
    'lextale': Fields.LEXTALE_SCORE,
    'lang': Fields.L1,
    'subject': Fields.SUBJECT_ID,
    'CFT20Total': 'cft20',
    'TOWRE_RealWords_FinalScore': 'TOWRE_word',
    'TOWRE_NonWords_FinalScore': 'TOWRE_nonword',
    'vocab.t2.5': 'vocab_t2_5',
}

# The tests combined into proficiency_agg (and proficiency_agg_mean).
PROFICIENCY_AGG_TESTS = ['spelling', 'TOWRE_word', 'TOWRE_nonword', 'vocab_t2_5']

# Columns replace_missing_values adds as 0 when a report lacks them.
IA_FEATURES: list[str] = []

# Fixation-report columns replace_missing_values adds as 0 when missing.
FIXATION_FEATURES: list[str] = [
    'CURRENT_FIX_INDEX',
    'CURRENT_FIX_DURATION',
    'CURRENT_FIX_PUPIL',
    'CURRENT_FIX_X',
    'CURRENT_FIX_Y',
    'NEXT_FIX_ANGLE',
    'CURRENT_FIX_INTEREST_AREA_INDEX',
    'NEXT_FIX_INTEREST_AREA_INDEX',
    'PREVIOUS_FIX_ANGLE',
    'NEXT_FIX_DISTANCE',
    'PREVIOUS_FIX_DISTANCE',
    'NEXT_SAC_AMPLITUDE',
    'NEXT_SAC_ANGLE',
    'NEXT_SAC_AVG_VELOCITY',
    'NEXT_SAC_DURATION',
    'NEXT_SAC_PEAK_VELOCITY',
]

MECO_LANG_CODE_TO_NAME = {
    "du": "Dutch",
    "ee": "Estonian",
    "fi": "Finnish",
    "ge": "German",
    "ge_po": "German",
    "ge_zu": "German",
    "gr": "Greek",
    "he": "Hebrew",
    "it": "Italian",
    "no": "Norwegian",
    "ru": "Russian",
    "ru_mo": "Russian",
    "sp": "Spanish",
    "sp_ch": "Spanish",
    "tr": "Turkish",
    "bp": "Brazilian Portuguese",
    "ch_s": "Mandarin",
    "ch_t": "Mandarin",
    "da": "Danish",
    "hi_iiith": "Hindi",
    "hi_iitk": "Hindi",
    "ic": "Icelandic",
    "se": "Serbian",
    "ba": "Basque",
    "en": "English",
    "en_uk": "English",
}


def _ppca_em_scores(
    z: pd.DataFrame,
    n_iter: int = 500,
    tol: float = 1e-8,
    seed: int = 0,
) -> pd.Series:
    """One-component PPCA (Tipping & Bishop 1999) fitted by EM on observed entries.

    Model: x = W z + mu + eps, z ~ N(0, 1), eps ~ N(0, sigma^2 I). Missing entries
    are treated as latent and never imputed into the covariance, so the loadings
    are estimated from real data only. Classical PCA is the sigma^2 -> 0 limit.

    Returns the posterior mean E[z | x_observed] per row, NaN where a row has no
    observed component (its score would otherwise be pure model prior).

    Input `z` is expected already z-scored per column (skipna), so mu starts at 0.
    """
    X = z.to_numpy(dtype=float)
    n, p = X.shape
    obs = ~np.isnan(X)
    n_obs_per_row = obs.sum(axis=1)

    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.1, size=p)          # q = 1 -> W is a p-vector
    mu = np.where(obs.any(axis=0), np.nanmean(np.where(obs, X, np.nan), axis=0), 0.0)
    mu = np.nan_to_num(mu)
    sigma2 = 1.0

    prev_ll = -np.inf
    for _ in range(n_iter):
        # --- E step: per-row posterior of the latent scalar, using its own observed set
        Ez = np.zeros(n)
        Ezz = np.ones(n)                        # rows with no data keep the prior
        for i in range(n):
            o = obs[i]
            if not o.any():
                continue
            Wo = W[o]
            prec = float(Wo @ Wo) + sigma2       # M_i (1x1)
            Ez[i] = float(Wo @ (X[i, o] - mu[o])) / prec
            Ezz[i] = sigma2 / prec + Ez[i] ** 2

        # --- M step: update mu, W, sigma^2 from observed entries only
        for j in range(p):
            r = obs[:, j]
            if not r.any():
                continue
            mu[j] = np.mean(X[r, j] - W[j] * Ez[r])
            denom = Ezz[r].sum()
            W[j] = float(((X[r, j] - mu[j]) * Ez[r]).sum() / denom) if denom > 0 else 0.0

        resid_sq = 0.0
        for j in range(p):
            r = obs[:, j]
            if not r.any():
                continue
            c = X[r, j] - mu[j]
            resid_sq += float((c ** 2 - 2 * c * W[j] * Ez[r] + (W[j] ** 2) * Ezz[r]).sum())
        sigma2 = max(resid_sq / max(obs.sum(), 1), 1e-12)

        # --- convergence on the observed-data log-likelihood
        ll = 0.0
        for i in range(n):
            o = obs[i]
            if not o.any():
                continue
            Wo = W[o][:, None]
            C = Wo @ Wo.T + sigma2 * np.eye(o.sum())
            d = (X[i, o] - mu[o])[:, None]
            sign, logdet = np.linalg.slogdet(C)
            ll += -0.5 * (logdet + (d.T @ np.linalg.solve(C, d)).item())
        if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            break
        prev_ll = ll

    # Orient so that higher score = higher proficiency (mean loading positive)
    if W.mean() < 0:
        W, Ez = -W, -Ez

    scores = pd.Series(Ez, index=z.index)
    scores[n_obs_per_row == 0] = np.nan
    logger.info(
        f'proficiency_agg PPCA-EM: loadings={np.round(W, 3).tolist()} sigma2={sigma2:.4f}; '
        f'{int((n_obs_per_row == 0).sum())} participant(s) with no observed component -> NaN'
    )
    return scores


class MecoL2Processor(DatasetProcessor):
    """Processing of the MECO L2 release (both data-collection waves). Not run on
    its own: MecoUnifiedProcessor runs it and splits the readers into MecoL1
    and MecoL2."""

    def __init__(self, dataset_name: str):
        dataset_path = Path.cwd() / "data" / dataset_name
        raw_paths = {file_name: dataset_path / "downloads" / file_name for file_name in OSF_RESOURCE_IDS.get(dataset_name, {}).keys()}
        self.surp_extractor = get_surp_extractor(extractor_type=SurpExtractorType.CAT_CTX_LEFT, model_name='gpt2')
        self.nlp = spacy.load('en_core_web_sm')
        self.type = 'L2'

        super().__init__(raw_paths, dataset_name)

    def download(self) -> None:
        resource_ids = OSF_RESOURCE_IDS.get(self.dataset_name, {})
        data_ids = {
            id: path for id, path in resource_ids.items() if 'ia' in id or 'fixations' in id or 'stimuli' in id
        }
        download_from_osf(self.dataset_path, data_ids)

    def download_metadata(self) -> None:
        """Download the individual-differences, LEAP-Q and comprehension files from OSF."""
        resource_ids = OSF_RESOURCE_IDS.get(self.dataset_name, {})
        urls = {
            id: path for id, path in resource_ids.items()
            if 'diff' in id or 'leap_q' in id or 'comp' in id
        }

        download_from_osf(self.dataset_path, urls, is_metadata=True)

    def load_raw_data(self) -> dict[str, pd.DataFrame]:
        """Load and concatenate both waves of the IA and fixation data."""
        data_dict = {}
        for data_type in [DataType.IA, DataType.FIXATIONS]:
            data_type_dfs = []
            for wave in ['w1', 'w2']:
                raw_path = self.raw_paths.get(f'{data_type}_{wave}.rda')
                if raw_path is not None and Path(raw_path).exists():
                    logger.info(f'Loading {data_type} data from {raw_path}')
                    data_type_dfs.append(pyreadr.read_r(raw_path)[f'joint.fix{'.l2' if self.type == 'L2' else ''}{'_w2' if wave == 'w2' and self.type == 'L2' else ''}' if data_type == DataType.FIXATIONS else 'joint.data'])
                else:
                    logger.warning(f'Raw path for {data_type} not found: {raw_path}')
            data_dict[data_type] = pd.concat(data_type_dfs, ignore_index=True) if data_type_dfs else None
        return data_dict

    def _add_comprehension_scores(self, diff_df: pd.DataFrame) -> pd.DataFrame:
        """Merge subject-level comprehension accuracy from the MECO L2 release.

        Both waves score the same 24 questions over the same 12 English texts, so
        the column is comparable across the en/en_uk (MecoL1) and non-English
        (MecoL2) halves that MecoUnifiedProcessor later splits out. Rescaled to
        0-100 to match the OneStop convention for TestCols.COMPREHENSION_COL;
        participants absent from the release stay NaN here and are turned into
        the -1 sentinel by the caller's float-conversion loop.
        """
        comp_col = str(TestCols.COMPREHENSION_COL)
        comp_waves = []
        for wave in ['w1', 'w2']:
            comp_path = self.dataset_path / "downloads" / "metadata" / f"comp_{wave}.rda"
            if not comp_path.exists():
                logger.warning(f'Comprehension data not found: {comp_path}')
                continue
            comp_waves.append(next(iter(pyreadr.read_r(comp_path).values()))[['uniform_id', 'accuracy']])

        if not comp_waves:
            logger.warning(f'No comprehension data loaded; {comp_col} will be absent.')
            return diff_df

        comp_df = pd.concat(comp_waves, ignore_index=True).drop_duplicates('uniform_id')
        comp_df = comp_df.rename(columns={'uniform_id': Fields.SUBJECT_ID, 'accuracy': comp_col})
        comp_df[comp_col] = pd.to_numeric(comp_df[comp_col], errors='coerce') * 100

        merged = diff_df.merge(comp_df, on=Fields.SUBJECT_ID, how='left')
        n_missing = int(merged[comp_col].isna().sum())
        if n_missing:
            logger.warning(
                f'{n_missing}/{len(merged)} participants have no comprehension score '
                f'in the OSF release; setting {comp_col} to the -1 sentinel.'
            )
        return merged

    def process_metadata(self) -> pd.DataFrame:
        diff_waves = []
        leap_q_waves = []
        for wave in ['w1', 'w2']:
            normalize_meco_leap_q_files(str(self.dataset_path / "downloads" / "metadata"), variant=f'{self.type}{wave.upper()}')
            if self.type == 'L2':
                diff_path = self.dataset_path / "downloads" / "metadata" / f"diff_{wave}.rda"
                diff_df = pyreadr.read_r(diff_path)[f'joint_id{'_w2' if wave == 'w2' else ''}']
            else:
                diff_path = self.dataset_path / "downloads" / "metadata" / f"diff_en_{'uk_' if wave == 'w2' else ''}{wave}.xlsx"
                diff_df = pd.read_excel(diff_path)

            leap_q_files = [path for path in self.raw_paths.keys() if 'leap_q' in str(path) and wave in str(path)]
            leap_q_dfs = []
            for leap_q_path in leap_q_files:
                leap_q_path = self.dataset_path / "downloads" / "metadata" / leap_q_path
                leap_q_df = pd.read_excel(leap_q_path)
                leap_q_dfs.append(leap_q_df)

            leap_q_df = pd.concat(leap_q_dfs, ignore_index=True)

            diff_waves.append(diff_df)
            leap_q_waves.append(leap_q_df)

        diff_df = pd.concat(diff_waves, ignore_index=True)
        if self.type == 'L2':
            diff_df = diff_df.drop(columns = ['subid'])
        else:
            diff_df = diff_df.drop(columns = ['old-id', 'old_id', 'id-num', 'included', ])
            diff_df[Fields.L1] = 'en'

        diff_df = diff_df.rename(columns=METADATA_RENAME_MAP)

        test_cols = [c for c in PROFICIENCY_AGG_TESTS if c in diff_df.columns]
        if len(test_cols) < len(PROFICIENCY_AGG_TESTS):
            missing = set(PROFICIENCY_AGG_TESTS) - set(test_cols)
            logger.warning(f'Missing columns for proficiency_agg: {missing}. Using {test_cols}.')
        test_values = diff_df[test_cols].apply(pd.to_numeric, errors='coerce')
        z_scored = (test_values - test_values.mean(skipna=True)) / test_values.std(skipna=True)
        mean_agg = z_scored.mean(axis=1, skipna=True)
        valid = mean_agg.dropna()
        if len(valid) > 0 and valid.max() > valid.min():
            lo, hi = valid.min(), valid.max()
            mean_agg = ((mean_agg - lo) / (hi - lo) * 99 + 1).round()
        diff_df['proficiency_agg_mean'] = mean_agg

        # PPCA fitted by EM on the observed entries only (see _ppca_em_scores).
        # Mean-filling instead would pin the 105 TOWRE-missing participants to 0
        # on both TOWRE columns, inflating that pair's correlation and distorting
        # the loadings. Participants with no observed component get NaN (-> the
        # -1 sentinel below).
        pca_agg = _ppca_em_scores(z_scored)
        valid_pca = pca_agg.dropna()
        if len(valid_pca) > 0 and valid_pca.max() > valid_pca.min():
            lo, hi = valid_pca.min(), valid_pca.max()
            pca_agg = ((pca_agg - lo) / (hi - lo) * 99 + 1).round()
        diff_df['proficiency_agg'] = pca_agg

        # After proficiency_agg: comprehension is an outcome measure, not one of
        # the PROFICIENCY_AGG_TESTS components, and must not enter the composite.
        if self.type == 'L2':
            diff_df = self._add_comprehension_scores(diff_df)

        for col in diff_df.columns:
            if col not in ['participant_id', 'L1']:
                diff_df[col] = diff_df[col].apply(lambda x: float(x) if pd.notna(x) else -1.0)
                diff_df[col] = diff_df[col].astype('float32')
        leap_q_df = pd.concat(leap_q_waves, ignore_index=True)

        os.makedirs(self.processed_metadata_path, exist_ok=True)
        diff_df.to_csv(self.processed_metadata_path / "metadata.csv", index=False)
        leap_q_df.to_csv(self.processed_metadata_path / "leap_q_metadata.csv", index=False)
        logger.info(f'Saved processed metadata to {self.processed_metadata_path}')

    def get_column_map(self, data_type: DataType) -> dict:
        column_maps = {
            DataType.IA: {
                'uniform_id': Fields.SUBJECT_ID,
                'trialid': Fields.UNIQUE_PARAGRAPH_ID,
                'skip': 'total_skip',
                'wordnum': 'IA_ID',
                'word': 'IA_LABEL',
                'nfix': 'IA_FIXATION_COUNT',
                'reg.in': 'IA_REGRESSION_IN',
                'reg.out': 'IA_REGRESSION_OUT',
                'dur': 'IA_DWELL_TIME',
                'firstrun.nfix': 'IA_FIRST_RUN_FIXATION_COUNT',
                'firstrun.dur': 'IA_FIRST_RUN_DWELL_TIME',
                'firstfix.launch': 'IA_FIRST_RUN_LAUNCH_SITE',
                'firstfix.land': 'IA_FIRST_RUN_LANDING_POSITION',
                'firstfix.dur': 'IA_FIRST_FIXATION_DURATION',
            },
            DataType.FIXATIONS: {
                'xn': 'CURRENT_FIX_X',
                'yn': 'CURRENT_FIX_Y',
                'dur': 'CURRENT_FIX_DURATION',
                'uniform_id': Fields.SUBJECT_ID,
                'trialid': Fields.UNIQUE_PARAGRAPH_ID,
                'fixid': 'CURRENT_FIX_INDEX',
                'start': 'CURENT_FIX_START',
                'stop': 'CURRENT_FIX_END',
                'ps': 'CURRENT_FIX_PUPIL_SIZE',
                'blink': 'CURRENT_FIX_BLINK_AROUND',
                'word': 'CURRENT_FIX_INTEREST_AREA_LABEL',
                'ianum': 'CURRENT_FIX_INTEREST_AREA_INDEX',
                'ia': 'CURRENT_FIX_LABEL',
                'ia.fix': 'CURRENT_FIX_INTEREST_AREA_FIX_COUNT',
                'ia.runid': 'CURRENT_FIX_INTEREST_AREA_RUN_ID',
            },
        }

        return column_maps.get(data_type, {})

    def dataset_specific_processing(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """MECO-specific processing steps."""

        # Normalize trailing hyphen-like characters in IA labels. The MECO raw
        # data inconsistently strips line-wrap hyphens for some participants
        # (e.g. "Grief-" vs "Grief" at the same IA_ID with the same next-IA),
        # which has no effect on the underlying IA layout but trips per-text
        # feature alignment checks. Stripping here keeps all downstream code
        # uniform.
        _trailing_hyphen_chars = '-‐‑‒–—'
        label_cols = {
            DataType.IA: 'IA_LABEL',
            DataType.FIXATIONS: 'CURRENT_FIX_INTEREST_AREA_LABEL',
        }
        for data_type, label_col in label_cols.items():
            df = data_dict[data_type]
            if label_col in df.columns:
                df[label_col] = (
                    df[label_col].astype('string').str.rstrip(_trailing_hyphen_chars)
                )
                data_dict[data_type] = df

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            df = data_dict[data_type]

            df[Fields.UNIQUE_TRIAL_ID] = (
                df[Fields.SUBJECT_ID].astype(str)
                + '_'
                + df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
            )

            data_dict[data_type] = df

        data_dict['fixations'], data_dict['ia'] = (
            self.add_ia_report_features_to_fixation_data(
                data_dict['ia'],
                data_dict['fixations'],
            )
        )

        for data_type in [DataType.IA, DataType.FIXATIONS]:
            data_dict[data_type] = add_missing_features(
                et_data=data_dict[data_type],
                mode=data_type,
            )
            data_dict[data_type] = data_dict[data_type].assign(
                normalized_ID=(
                    data_dict[data_type]['IA_ID'] - data_dict[data_type]['IA_ID'].min()
                )
                / (
                    data_dict[data_type]['IA_ID'].max()
                    - data_dict[data_type]['IA_ID'].min()
                ),
            )

        data_dict = replace_missing_values(data_dict)

        return data_dict

    def add_ia_report_features_to_fixation_data(
        self,
        ia_df: pd.DataFrame,
        fix_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Add text metrics, go-past and trial counts to the IA data, and merge
        the IA-level columns into the fixation data. Returns (fixations, ia)."""
        ia_df = self._prepare_ia_dataframe(ia_df)

        # These participants' paragraph ids are off by one from some text onward
        # (e.g. sp_36's texts 6-12 are labeled 5-11); shift them back.
        key_value_map = {
            'sp_36': {11: 12, 10: 11, 9: 10, 8: 9, 7: 8, 6: 7, 5: 6},
            'gr_45': {11: 12, 10: 11, 9: 10, 8: 9, 7: 8},
            'it_25': {10: 11, 9: 10, 7: 8, 6: 7},
            'se_38': {11: 12, 10: 11, 9: 10, 8: 9, 7: 8, 6: 7, 5: 6, 4: 5},
            'bp_23': {6: 7, 5: 6, 4: 5, 3: 4},
        }
        for participant_id, mapping in key_value_map.items():
            mask = ia_df.participant_id == participant_id
            fix_mask = fix_df.participant_id == participant_id
            for old_id, new_id in sorted(mapping.items(), reverse=True):
                ia_df.loc[
                    mask & (ia_df.unique_paragraph_id == old_id), 'unique_paragraph_id'
                ] = new_id
                fix_df.loc[
                    fix_mask & (fix_df.unique_paragraph_id == old_id),
                    'unique_paragraph_id',
                ] = new_id

        ia_df['unique_trial_id'] = (
            ia_df[Fields.SUBJECT_ID].astype(str)
            + '_'
            + ia_df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
        )
        fix_df['unique_trial_id'] = (
            fix_df[Fields.SUBJECT_ID].astype(str)
            + '_'
            + fix_df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
        )

        ia_df.rename(columns={'lang_x': Fields.L1}, inplace=True)
        fix_df.rename(columns={'lang_x': Fields.L1}, inplace=True)

        stimuli_df = self._load_stimuli_df()

        ia_df = ia_df.merge(
            stimuli_df,
            on='unique_paragraph_id',
            validate='many_to_one',
        )

        fix_df = fix_df.merge(
            stimuli_df,
            on='unique_paragraph_id',
            validate='many_to_one',
        )

        groups = [group for _, group in ia_df.groupby(['unique_trial_id'])]
        metrics_df = self._process_metrics_batch(groups)

        ia_df = self._merge_metrics_to_ia(ia_df, metrics_df)

        # GP uses DataViewer's first-entry anchor so it matches OneStop. MECO's shipped
        # `firstrun.gopast` (and the popEye/EMTeC port that reproduces it exactly) uses
        # a first-PASS anchor instead, scoring 0 for any word first reached by
        # regressing -- ~17% of words, where OneStop reports a real duration.
        fixation_metrics = compute_gopast_first_entry(fix_df)
        fixation_metrics['IA_ID'] = fixation_metrics['IA_ID'].apply(lambda x: float(x) if pd.notna(x) else x).astype('Float64')
        ia_df = ia_df.merge(
            fixation_metrics,
            on=['participant_id', 'unique_paragraph_id', 'IA_ID'],
            how='left',
        )

        # Match OneStop: never-fixated words contribute 0 to the duration measures
        # instead of being dropped from the average (mean_FF 229 -> 171 ms).
        # Mask is total_skip, NOT IA_SKIP -- IA_SKIP is a first-pass skip and 17% of
        # those words were fixated later and hold real dwell times.
        never_fixated = ia_df['total_skip'] == 1
        ia_df.loc[never_fixated, [
            'IA_FIRST_FIXATION_DURATION',
            'IA_FIRST_RUN_DWELL_TIME',
            'IA_DWELL_TIME',
            'IA_REGRESSION_PATH_DURATION',
        ]] = 0

        fix_df['NEXT_FIX_INTEREST_AREA_INDEX'] = fix_df[
            'CURRENT_FIX_INTEREST_AREA_INDEX'
        ].shift(-1)
        fix_df['CURRENT_FIX_INTEREST_AREA_INDEX'] = fix_df[
            'CURRENT_FIX_INTEREST_AREA_INDEX'
        ].fillna(-1)

        if 'normalized_part_ID' in fix_df.columns:
            if fix_df['normalized_part_ID'].isna().any():
                logger.info('normalized_part_ID contains NaNs; dropping it.')
                fix_df = fix_df.drop(columns='normalized_part_ID')

        merge_keys = set(MERGE_IA_FIXATION_COLS)

        dup_cols = (set(fix_df.columns) & set(ia_df.columns)) - set(merge_keys)
        _ia_df = ia_df.drop(columns=list(dup_cols))
        enriched_fix_df = fix_df.merge(
            _ia_df.drop_duplicates(subset=merge_keys, keep='first'),
            on=list(merge_keys),
            how='left',
            validate='many_to_one',
        )

        num_of_words_in_trials_series = ia_df.groupby(TRIAL_IDENTIFIER_COLS,).size()
        num_of_words_in_trials_series.name = 'num_of_words_in_trial'
        enriched_fix_df = enriched_fix_df.merge(
            num_of_words_in_trials_series,
            on=TRIAL_IDENTIFIER_COLS,
            how='left',
        )

        enriched_fix_df['TRIAL_IA_COUNT'] = enriched_fix_df['TRIAL_IA_COUNT'].fillna(0)

        return enriched_fix_df, ia_df

    def _load_stimuli_df(self) -> pd.DataFrame:
        """Load stimuli paragraphs as a DataFrame with columns
        ['unique_paragraph_id', 'paragraph']. L2 reads the CP1255 RDA file."""
        stimuli_path = self.raw_paths.get('stimuli.rda')
        if stimuli_path is not None and Path(stimuli_path).exists():
            logger.info(f'Loading stimuli data from {stimuli_path}')
            stimuli_df = load_cp1255_rda(str(stimuli_path))['d']
            stimuli_df = stimuli_df[['trialid', 'text']].rename(
                columns={'trialid': 'unique_paragraph_id', 'text': 'paragraph'}
            )
            stimuli_df['unique_paragraph_id'] = stimuli_df['unique_paragraph_id'].apply(
                lambda x: int(x) if pd.notna(x) else x
            ).astype('Int64')
            return stimuli_df

        logger.warning(f'Stimuli path not found: {stimuli_path}')
        return pd.DataFrame(columns=['unique_paragraph_id', 'paragraph'])

    def _prepare_ia_dataframe(self, ia_df: pd.DataFrame) -> pd.DataFrame:
        ia_df = ia_df.rename(
            columns={
                Fields.IA_DATA_IA_ID_COL_NAME: Fields.FIXATION_REPORT_IA_ID_COL_NAME
            }
        )

        ia_df = ia_df.sort_values(
            ['unique_trial_id', 'CURRENT_FIX_INTEREST_AREA_INDEX']
        )

        grouped = ia_df.groupby('unique_trial_id')
        ia_df['TRIAL_IA_COUNT'] = grouped['unique_trial_id'].transform('count')
        ia_df['IA_DWELL_TIME_%'] = grouped['IA_DWELL_TIME'].transform(
            lambda x: x / np.sum(x)
        )
        ia_df['PARAGRAPH_RT'] = ia_df.groupby(Fields.UNIQUE_TRIAL_ID)[
            'IA_DWELL_TIME'
        ].transform('sum')

        ia_df['IA_FIXATION_%'] = ia_df.groupby('unique_trial_id')[
            'IA_FIXATION_COUNT'
        ].transform(lambda x: x / x.sum())

        # DataViewer's flag means first pass (== IA_SKIP 0). The old firstfix.sac.in
        # proxy means "entered from the left" and agrees on only 87% of words.
        ia_df['IA_FIRST_FIX_PROGRESSIVE'] = (ia_df['firstrun.skip'] == 0).astype(int)
        ia_df['IA_RUN_COUNT'] = ia_df['nrun']
        ia_df['IA_SELECTIVE_REGRESSION_PATH_DURATION'] = ia_df['firstrun.gopast.sel']
        # IA_SKIP is a FIRST-PASS skip in DataViewer; MECO's `skip` (total_skip) means
        # never fixated -- 0.405 vs 0.230, r=0.52 across participants. total_skip is
        # left alone and now matches OneStop's IA_DWELL_TIME == 0.
        ia_df['IA_SKIP'] = ia_df['firstrun.skip']
        ia_df['word_length'] = ia_df['IA_LABEL'].str.len()

        # MECO's reg.in / reg.out / firstrun.reg.out are binary indicators, not
        # counts. That is enough here: downstream only tests these columns for > 0.
        ia_df['IA_REGRESSION_IN_COUNT'] = ia_df['IA_REGRESSION_IN']
        # _FULL_COUNT is all-time (reg.out); _COUNT is DataViewer's first-pass
        # measure, so it takes firstrun.reg.out. Feeds IA_RR.
        ia_df['IA_REGRESSION_OUT_FULL_COUNT'] = ia_df['IA_REGRESSION_OUT']
        ia_df['IA_REGRESSION_OUT_COUNT'] = ia_df['firstrun.reg.out']
        ia_df['IA_ID'] = ia_df[Fields.FIXATION_REPORT_IA_ID_COL_NAME]

        # Columns OneStop has and MECO does not; set to 0.
        zero_cols = [
            'NEXT_FIX_INTEREST_AREA_INDEX',
            'CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE',
            'start_of_line',
            'end_of_line',
            'IA_LAST_FIXATION_DURATION',
            'IA_LAST_RUN_DWELL_TIME',
            'IA_LAST_RUN_FIXATION_COUNT',
            'IA_FIRST_FIXATION_VISITED_IA_COUNT',
            'IA_LEFT',
            'IA_RIGHT',
            'IA_TOP',
            'IA_BOTTOM',
            'NEXT_SAC_DURATION',
            'NEXT_SAC_AVG_VELOCITY',
            'NEXT_SAC_AMPLITUDE',
            'NEXT_SAC_END_X',
            'NEXT_SAC_START_X',
            'NEXT_SAC_START_Y',
            'NEXT_SAC_END_Y',
        ]
        ia_df[zero_cols] = 0

        return ia_df

    def _process_metrics_batch(self, groups: list[pd.DataFrame]) -> pd.DataFrame:
        metrics_list = []

        for group in tqdm(groups, desc='Sequential metric extraction'):
            try:
                sentence = str(group['paragraph'].iloc[0])
                # The L1 stimuli xlsx contains literal backslash-escape
                # sequences ("\n", "\r") that aren't real whitespace. Replace
                # every backslash with a space so tokens on either side don't
                # get fused into one.
                sentence = sentence.replace('\\', ' ')
                # Collapse any real whitespace too.
                sentence = ' '.join(sentence.split())
                # The MECO IA report splits hyphenated compounds at the hyphen
                # ("performance-" + "enhancing"); insert a space after each
                # hyphen so text_metrics' tokenizer matches that split.
                sentence = sentence.replace('-', '- ')
                sentence = ' '.join(sentence.split())
                metrics = get_metrics(
                    target_text=sentence,
                    surp_extractor=self.surp_extractor,
                    parsing_model=self.nlp,
                    parsing_mode='re-tokenize',
                    add_parsing_features=True,
                    language='en',
                )
                metrics['unique_paragraph_id'] = group['unique_paragraph_id'].iloc[0]
                metrics['participant_id'] = group['participant_id'].iloc[0]
                metrics[Fields.FIXATION_REPORT_IA_ID_COL_NAME] = (
                    metrics['Token_idx'] + 1
                )
                metrics_list.append(metrics)
            except Exception as e:
                logger.error(
                    f'Error processing group {group["unique_paragraph_id"].iloc[0]}: {e}'
                )
                raise

        return (
            pd.concat(metrics_list, ignore_index=True)
            if metrics_list
            else pd.DataFrame()
        )

    def _merge_metrics_to_ia(
        self, ia_df: pd.DataFrame, metrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        merge_keys = ['unique_trial_id', Fields.FIXATION_REPORT_IA_ID_COL_NAME]

        metrics_df[Fields.UNIQUE_TRIAL_ID] = (
            metrics_df[Fields.SUBJECT_ID].astype(str)
            + '_'
            + metrics_df[Fields.UNIQUE_PARAGRAPH_ID].astype(str)
        )

        drop_keys = (set(metrics_df.columns) & set(ia_df.columns)) - set(merge_keys)
        cols_to_drop = list(drop_keys) + ['Morph', 'Reduced_POS']
        cols_to_drop = [c for c in cols_to_drop if c in metrics_df.columns]

        metrics_clean = metrics_df.drop(columns=cols_to_drop).drop_duplicates()

        ia_df = ia_df.merge(
            metrics_clean,
            on=merge_keys,
            how='left',
            validate='many_to_one',
        )

        rename_map = {
            'POS': 'universal_pos',
            'Length': 'word_length_no_punctuation',
            'Wordfreq_Frequency': 'wordfreq_frequency',
            'subtlex_Frequency': 'subtlex_frequency',
            # TAG is spaCy's token.tag_, i.e. real Penn Treebank tags
            'TAG': 'ptb_pos',
            'Head_word_idx': 'head_word_index',
            'Dependency_Relation': 'dependency_relation',
            'Entity': 'entity_type',
            'gpt2_Surprisal': 'gpt2_surprisal',
            'gpt2': 'gpt2_surprisal',
            'Head_Direction': 'head_direction',
            'Is_Content_Word': 'is_content_word',
            'n_Lefts': 'left_dependents_count',
            'n_Rights': 'right_dependents_count',
            'Distance2Head': 'distance_to_head',
        }
        ia_df = ia_df.rename(columns=rename_map)

        return ia_df


class MecoUnifiedProcessor(MecoL2Processor):
    """Processes all MECO L2 data (all languages reading the same English texts),
    then splits output into MecoL1 (English native speakers) and MecoL2
    (non-English speakers).

    This ensures L1 and L2 share the same texts, stimuli, and preprocessing
    pipeline, making EyeScore comparisons valid.
    """

    def __init__(self):
        # Initialize using MecoL2 resources (all participants read the same texts)
        super().__init__(DataSets.MECOL2)
        self.l1_path = Path.cwd() / "data" / DataSets.MECOL1
        self.l2_path = Path.cwd() / "data" / DataSets.MECOL2

    def _get_lang_col(self, df: pd.DataFrame) -> str:
        """Find the language column name in the DataFrame."""
        if Fields.L1 in df.columns:
            return Fields.L1
        if 'lang' in df.columns:
            return 'lang'
        raise ValueError(f"No language column found. Columns: {df.columns.tolist()}")

    def _map_lang_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace short language codes with full language names."""
        lang_col = self._get_lang_col(df)
        unmapped = set(df[lang_col].unique()) - set(MECO_LANG_CODE_TO_NAME.keys())
        if unmapped:
            logger.warning(f"No language name mapping for codes: {unmapped}")
        df[lang_col] = df[lang_col].map(MECO_LANG_CODE_TO_NAME).fillna(df[lang_col])
        return df

    def _split_by_language(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Map language codes to names, then split into L1 (English) and L2 (non-English)."""
        df = self._map_lang_codes(df)
        lang_col = self._get_lang_col(df)
        is_english = df[lang_col] == "English"
        return df[is_english].copy(), df[~is_english].copy()

    def process_data(self) -> dict[str, pd.DataFrame]:
        """Process all L2 data, then split and save into MecoL1/MecoL2."""
        # Run the standard L2 pipeline on all participants
        processed_data = super().process_data()

        # Split harmonized data by language and save to separate directories
        for data_type in [DataType.IA, DataType.FIXATIONS]:
            if data_type not in processed_data or processed_data[data_type] is None:
                continue
            l1_df, l2_df = self._split_by_language(processed_data[data_type])

            l1_dir = self.l1_path / "harmonized"
            l2_dir = self.l2_path / "harmonized"
            l1_dir.mkdir(parents=True, exist_ok=True)
            l2_dir.mkdir(parents=True, exist_ok=True)

            l1_df.to_csv(l1_dir / f"{data_type}.csv")
            l2_df.to_csv(l2_dir / f"{data_type}.csv")

            logger.info(
                f"Split {data_type}: {len(l1_df)} L1 rows (en/en_uk) -> {l1_dir}, "
                f"{len(l2_df)} L2 rows -> {l2_dir}"
            )

        return processed_data

    def process_metadata(self) -> pd.DataFrame:
        """Process metadata, then split and save into MecoL1/MecoL2."""
        # Run the standard L2 metadata pipeline
        super().process_metadata()

        # Load the saved metadata and split by language
        metadata_path = self.processed_metadata_path / "metadata.csv"
        if not metadata_path.exists():
            logger.warning(f"Metadata not found at {metadata_path}")
            return

        metadata_df = pd.read_csv(metadata_path)
        l1_meta, l2_meta = self._split_by_language(metadata_df)

        l1_meta_dir = self.l1_path / "metadata"
        l2_meta_dir = self.l2_path / "metadata"
        l1_meta_dir.mkdir(parents=True, exist_ok=True)
        l2_meta_dir.mkdir(parents=True, exist_ok=True)

        l1_meta.to_csv(l1_meta_dir / "metadata.csv", index=False)
        l2_meta.to_csv(l2_meta_dir / "metadata.csv", index=False)

        logger.info(
            f"Split metadata: {len(l1_meta)} L1 participants -> {l1_meta_dir}, "
            f"{len(l2_meta)} L2 participants -> {l2_meta_dir}"
        )

        # Also split leap_q if it exists
        leap_q_path = self.processed_metadata_path / "leap_q_metadata.csv"
        if leap_q_path.exists():
            leap_q_df = pd.read_csv(leap_q_path)
            if 'uniform_id' in leap_q_df.columns:
                # Match participants by prefix (e.g. "en_1", "du_2")
                en_ids = set(l1_meta[Fields.SUBJECT_ID])
                l1_leap = leap_q_df[leap_q_df['uniform_id'].isin(en_ids)]
                l2_leap = leap_q_df[~leap_q_df['uniform_id'].isin(en_ids)]
                l1_leap.to_csv(l1_meta_dir / "leap_q_metadata.csv", index=False)
                l2_leap.to_csv(l2_meta_dir / "leap_q_metadata.csv", index=False)


def replace_missing_values(
    data_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Add any FIXATION_FEATURES / IA_FEATURES column that is missing, as 0."""
    for col in FIXATION_FEATURES:
        if col not in data_dict[DataType.FIXATIONS].columns:
            logger.warning(f'Adding missing column {col} to fixation data')
            data_dict[DataType.FIXATIONS][col] = 0

    for col in IA_FEATURES:
        if col not in data_dict[DataType.IA].columns:
            logger.warning(f'Adding missing column {col} to IA data')
            data_dict[DataType.IA][col] = 0
    return data_dict


def normalize_meco_leap_q_files(interim_dir: str, *, variant: str) -> None:
    """Normalizes all LEAP-Q Excel files in interim_dir to a consistent column layout.

    Parameters
    ----------
    interim_dir : str
        Directory containing the leap_q_*.xlsx files to normalize (in-place).
    variant : str
        Dataset variant — one of 'L2W1', 'L1W1', 'L1W2'.
    """
    if variant == 'L2W2':
        logger.warning('No known LEAP-Q files for L2W2 variant; skipping normalization.')
        return
    split_header_columns, file_rename_maps = LEAP_Q_VARIANT_MAPS[variant]
    for filename in sorted(os.listdir(interim_dir)):
        if not filename.startswith('leap_q_') or not filename.endswith('.xlsx') or not variant[2:].lower() in filename:
            continue
        leap_q_path = os.path.join(interim_dir, filename)
        normalized_df = _normalize_meco_leap_q_file(leap_q_path, split_header_columns, file_rename_maps)
        normalized_df.to_excel(leap_q_path, index=False)
        logger.info(f'Normalized LEAP-Q file and saved: {filename}')


def _normalize_meco_leap_q_file(
    leap_q_path: str,
    split_header_columns: dict[str, list[str]],
    file_rename_maps: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Normalizes a single LEAP-Q file to a consistent column layout."""
    filename = os.path.basename(leap_q_path)
    raw_df = pd.read_excel(leap_q_path)
    file_rename_key = filename.replace('_w1', '').replace('_w2', '')

    # Idempotency guard: skip if already normalized
    if all(col in MECO_PARTICIPANTS_COLUMN_MAP for col in raw_df.columns):
        if file_rename_key == 'leap_q_se.xlsx' and 'uniform_id' in raw_df.columns:
            raw_df = raw_df.drop_duplicates(subset=['uniform_id'], keep='first')
        logger.info(f'File {filename} already has expected columns. Skipping normalization.')
        return raw_df

    # KO files store English translation labels in the first data row — promote to headers
    if file_rename_key == 'leap_q_ko.xlsx':
        first_row_tokens = {str(v).strip() for v in raw_df.iloc[0].tolist() if pd.notna(v)}
        if not {'subject', 'old-id'}.issubset(first_row_tokens):
            raise ValueError(
                'Unexpected KO LEAP-Q format: could not find translation-row headers '
                "('subject', 'old-id'). Re-copy data/MECOL1W1/downloads/leap_q_ko.xlsx "
                'to data/MECOL1W1/interim/leap_q_ko.xlsx and rerun normalization.'
            )
        raw_df.columns = raw_df.iloc[0].astype(str)
        raw_df = raw_df.iloc[1:].copy()

    if file_rename_key in split_header_columns:
        normalized_df = raw_df.iloc[2:].copy()
        expected_columns = split_header_columns[file_rename_key]
        if normalized_df.shape[1] != len(expected_columns):
            logger.warning(
                f'Unexpected number of columns in {file_rename_key}: '
                f'expected {len(expected_columns)}, got {normalized_df.shape[1]}'
            )
        normalized_df.columns = expected_columns[: normalized_df.shape[1]]
    else:
        normalized_df = raw_df.copy()

    rename_map = {**MECO_W1_COMMON_RENAME_MAP, **file_rename_maps.get(file_rename_key, {})}
    normalized_df = normalized_df.rename(columns=rename_map)

    # Rename any year/month columns to canonical names, then combine to total-years
    for col in list(normalized_df.columns):
        if col in CANONICAL_YEAR_MONTH_MAP:
            year_name, month_name = CANONICAL_YEAR_MONTH_MAP[col]
            if year_name and col != year_name:
                normalized_df = normalized_df.rename(columns={col: year_name})
            if month_name and col != month_name:
                normalized_df = normalized_df.rename(columns={col: month_name})
    for year_col, month_col, target_col in [
        ('english_country_years',    'english_country_months',    'years_in_english_country'),
        ('english_family_years',     'english_family_months',     'years_lived_with_english_speaking_family'),
        ('english_school_work_years', 'english_school_work_months', 'years_school_or_work_english'),
    ]:
        normalized_df = _combine_year_month_columns(normalized_df, year_col, month_col, target_col)

    normalized_df = normalized_df[[col for col in normalized_df.columns if col in MECO_PARTICIPANTS_COLUMN_MAP]]

    if 'ee' in file_rename_key:
        normalized_df = normalized_df.iloc[1:53].copy()
    normalized_df['uniform_id'] = normalized_df['uniform_id'].apply(
        lambda x: x.replace('_0', '_') if isinstance(x, str) and '_0' in x else x
    )
    return normalized_df


def _combine_year_month_columns(
    df: pd.DataFrame, year_col: str, month_col: str, target_col: str
) -> pd.DataFrame:
    """Combines separate year and month columns into total years, preserving missing values."""
    if year_col not in df.columns and month_col not in df.columns:
        return df

    years  = pd.to_numeric(df[year_col],  errors='coerce') if year_col  in df.columns else pd.Series(pd.NA, index=df.index, dtype='Float64')
    months = pd.to_numeric(df[month_col], errors='coerce') if month_col in df.columns else pd.Series(pd.NA, index=df.index, dtype='Float64')

    has_value = years.notna() | months.notna()
    df[target_col] = (years.fillna(0) + months.fillna(0) / 12).where(has_value, float('nan'))

    drop_cols = [col for col in [year_col, month_col] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def compute_gopast_first_entry(
    fixations: pd.DataFrame,
    paragraph_col: str = 'unique_paragraph_id',
    subject_col: str = 'participant_id',
) -> pd.DataFrame:
    """Go-past (regression path duration) on DataViewer's convention.

    For each word, sum fixation durations from the word's FIRST fixation --
    whenever it occurs -- until the eye first fixates a word to its right.
    This differs from popEye/EMTeC's RPD_inc, which only accumulates while the
    word is the rightmost one reached and therefore scores 0 for a word first
    arrived at by regressing. Validated against OneStop's DataViewer column on
    single-trial pairs: 97.7% exact overall, 98.8% on the words the popEye
    anchor zeroes. Out-of-IA fixations are excluded (including them drops the
    match to 96.5%).

    Words with no fixation produce no row, so a left merge leaves them NaN.

    NOTE: groups by (paragraph, subject). Valid for MECO, where a participant
    reads each text once. OneStop has repeated-reading trials and would need
    unique_trial_id here.
    """
    rows: list[tuple] = []
    for (para_id, subj_id), g in fixations.groupby([paragraph_col, subject_col], sort=False):
        g = g.sort_values('CURRENT_FIX_INDEX')
        aoi = g['CURRENT_FIX_INTEREST_AREA_INDEX'].fillna(-1).astype(int).to_numpy()
        dur = g['CURRENT_FIX_DURATION'].fillna(0).astype(int).to_numpy()
        n = len(aoi)
        first: dict[int, int] = {}
        for i in range(n):
            w = int(aoi[i])
            if w > 0 and w not in first:
                first[w] = i
        for w, start in first.items():
            total = 0
            for i in range(start, n):
                a = int(aoi[i])
                if a > w:
                    break
                if a > 0:
                    total += int(dur[i])
            rows.append((subj_id, para_id, float(w), total))
    return pd.DataFrame(rows, columns=[subject_col, paragraph_col, 'IA_ID',
                                       'IA_REGRESSION_PATH_DURATION'])
