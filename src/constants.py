"""Names shared across the pipeline: datasets, column names, feature groups and
feature sets, pools and fold methods, prediction targets and aggregation types,
and the OSF resource ids the raw data is downloaded from."""
from enum import StrEnum
from pathlib import Path

DATA_PATH = Path('data')


class DataSets(StrEnum):
    """Dataset names, as used under data/ and in the results trees."""
    ONESTOPL2 = 'OneStopL2'
    ONESTOPL1 = 'OneStopL1'
    MECO = 'Meco'
    MECOL1 = 'MecoL1'
    MECOL2 = 'MecoL2'


class DataType(StrEnum):
    """Kinds of raw data file."""

    IA = 'ia'
    FIXATIONS = 'fixations'
    METADATA = 'metadata'


class Fields(StrEnum):
    """Column names shared by the harmonized data, the metadata and the feature files."""

    UNIQUE_TRIAL_ID = 'unique_trial_id'
    BATCH = 'article_batch'
    PARAGRAPH_ID = 'paragraph_id'
    UNIQUE_PARAGRAPH_ID = 'unique_paragraph_id'
    ARTICLE_ID = 'article_id'
    LEVEL = 'difficulty_level'
    LIST = 'list_number'
    PARAGRAPH = 'paragraph'
    HAS_PREVIEW = 'question_preview'
    SUBJECT_ID = 'participant_id'
    REREAD = 'repeated_reading_trial'
    IA_DATA_IA_ID_COL_NAME = 'IA_ID'
    FIXATION_REPORT_IA_ID_COL_NAME = 'CURRENT_FIX_INTEREST_AREA_INDEX'
    IS_CORRECT = 'is_correct'
    PRACTICE = 'practice_trial'
    QUESTION = 'question'
    L1 = 'L1'
    L1_GROUP = 'L1_group'
    LEXTALE_SCORE = 'lextale_score'
    CONTENT_GROUP = 'content_group'
    TRIAL_INDEX = 'TRIAL_INDEX'
    READING_SPEED = 'reading_speed'
    HALF = 'half'


MERGE_IA_FIXATION_COLS = [
    Fields.SUBJECT_ID,
    Fields.UNIQUE_PARAGRAPH_ID,
    Fields.UNIQUE_TRIAL_ID,
    Fields.FIXATION_REPORT_IA_ID_COL_NAME,
]

TRIAL_IDENTIFIER_COLS = [
    Fields.SUBJECT_ID,
    Fields.UNIQUE_PARAGRAPH_ID,
    Fields.UNIQUE_TRIAL_ID,
]

# IA columns copied onto each fixation of the OneStop fixation report
# (preprocessing/dataset_preprocessing/onestop.py).
EYE_FEATURES_COLUMNS = [
    'IA_DWELL_TIME',
    'IA_DWELL_TIME_%',
    'IA_FIXATION_%',
    'IA_FIXATION_COUNT',
    'IA_REGRESSION_IN_COUNT',
    'IA_REGRESSION_OUT_FULL_COUNT',
    'IA_RUN_COUNT',
    'IA_FIRST_FIXATION_DURATION',
    'IA_FIRST_FIXATION_VISITED_IA_COUNT',
    'IA_FIRST_RUN_DWELL_TIME',
    'IA_FIRST_RUN_FIXATION_COUNT',
    'IA_SKIP',
    'IA_REGRESSION_PATH_DURATION',
    'IA_REGRESSION_OUT_COUNT',
    'IA_SELECTIVE_REGRESSION_PATH_DURATION',
    'IA_LAST_FIXATION_DURATION',
    'IA_LAST_RUN_DWELL_TIME',
    'IA_LAST_RUN_FIXATION_COUNT',
    'IA_TOP',
    'IA_LEFT',
    'IA_FIRST_FIX_PROGRESSIVE',
    'normalized_ID',
    'PARAGRAPH_RT',
    'total_skip',
]

WORD_FEATURES_COLUMNS = [
    'is_content_word',
    'ptb_pos',
    'left_dependents_count',
    'right_dependents_count',
    'distance_to_head',
    'head_direction',
    'gpt2_surprisal',
    'wordfreq_frequency',
    'word_length',
    'universal_pos',
    'start_of_line',
    'end_of_line',
    'entity_type',
    'IA_LABEL'
]

IA_FEATURES_TO_ADD_TO_FIXATION_REPORT = EYE_FEATURES_COLUMNS + WORD_FEATURES_COLUMNS


SURPRISAL_MODELS = ['gpt2']


# Raw-data downloads (OSF)
BASE_OSF_URL = 'https://osf.io/download/'

OSF_RESOURCE_IDS: dict[str, dict[str, str]] = {
    DataSets.ONESTOPL1: {'ia': "zhywq",
                         'fixations': "tbxdc",
                         'metadata': "yvu5w"},

    DataSets.MECOL2: {'ia_w1.rda': '5eskj',
                     'fixations_w1.rda': '5sm8v',
                     'diff_w1.rda': '4zu8d',
                     'leap_q_du_w1.xlsx': 'xnfd5',
                     'leap_q_ee_w1.xlsx': 'z47yd',
                     'leap_q_en_w1.xlsx': 'nygz4',
                     'leap_q_fi_w1.xlsx': 'cb4np',
                     'leap_q_ge_w1.xlsx': 'xwskb',
                     'leap_q_gr_w1.xlsx': 'f7a9h',
                     'leap_q_he_w1.xlsx': 'xph86',
                     'leap_q_it_w1.xlsx': 'mxjqg',
                     'leap_q_no_w1.xlsx': '6t2qd',
                     'leap_q_ru_w1.xlsx': 't2xhw',
                     'leap_q_sp_w1.xlsx': 'frts6',
                     'leap_q_tr_w1.xlsx': 'enhzu',

                     'ia_w2.rda': 'hqaxg',
                     'fixations_w2.rda': 'tjxz3',
                     'diff_w2.rda': 'keuvm',
                     'leap_q_bp_w2.xlsx': '2an7j',
                     'leap_q_ch_s_w2.xlsx': '3vtxj',
                     'leap_q_ch_t_w2.xlsx': 'u4kd8',
                     'leap_q_da_w2.xlsx': 'f5hbg',
                     'leap_q_en_uk_w2.xlsx': 'g6u4j',
                     'leap_q_ge_po_w2.xlsx': 'ra84d',
                     'leap_q_ge_zu_w2.xlsx': 'qdpfy',
                     'leap_q_hi_iii_w2.xlsx': 'fgvb3',
                     'leap_q_hi_iitk_w2.xlsx': 'fzwbc',
                     'leap_q_ic_w2.xlsx': 'bax3n',
                     'leap_q_no_w2.xlsx': 'zpt69',
                     'leap_q_ru_mo_w2.xlsx': '9a3xv',
                     'leap_q_se_w2.xlsx': 'rs8eh',
                     'leap_q_sp_ch_w2.xlsx': 'bdp9f',
                     'leap_q_tr_w2.xlsx': 'pumrq',

                     'stimuli.rda': '6te23',

                     # Subject-level comprehension accuracy (24 questions over the
                     # same 12 English texts in both waves). Long OSF file ids from
                     # node q9h43, release 2.0/version 2.1 -- the same release the
                     # ia/fixations/diff files above come from.
                     'comp_w1.rda': '687f2cf299698b13c60eb0a0',
                     'comp_w2.rda': '687f2e1965adb44c7302c0f9'},
}


class FeatureGroups(StrEnum):
    """Feature groups; a feature set is one group or several joined by '_'."""
    READING_SPEED = 'READING_SPEED'
    FIXATION_METRICS = 'FIXATION_METRICS'
    S_CLUSTERS = 'S_CLUSTERS'
    S_CLUSTERS_NO_NORM = 'S_CLUSTERS_NO_NORM'
    WP_COEFS = 'WP_COEFS'
    WP_COEFS_NO_NORM = 'WP_COEFS_NO_NORM'
    WP_COEFS_NO_INTERCEPT = 'WP_COEFS_NO_INTERCEPT'
    WP_COEFS_NO_NORM_NO_INTERCEPT = 'WP_COEFS_NO_NORM_NO_INTERCEPT'
    TRANSITIONS = 'TRANSITIONS'
    WFC = 'WFC'


# Feature column name prefixes for the per-text feature groups. Column names
# are dataset-dependent (depend on which paragraphs/words are in the data),
# so they cannot be enumerated up-front like the other feature groups.
TRANSITIONS_FEATURE_PREFIX = 'transitions_'
WFC_FEATURE_PREFIX = 'wfc_'

# Prefixes of the feature groups that are only valid where training and test
# participants read the same texts (same content_group / fixed text).
SEEN_CV_ONLY_PREFIXES = (TRANSITIONS_FEATURE_PREFIX, WFC_FEATURE_PREFIX)

# For these feature groups, the actual list of column names depends on the
# loaded data (which paragraphs / words are present), so the dict in
# FEATURE_GROUPS_TO_FEATURES is empty — runtime callers must expand the
# matching prefix against the columns of the loaded DataFrame instead.
FIXED_TEXT_GROUP_PREFIXES: dict[str, str] = {
    str(FeatureGroups.TRANSITIONS): TRANSITIONS_FEATURE_PREFIX,
    str(FeatureGroups.WFC): WFC_FEATURE_PREFIX,
}

# Per-text features (TRANSITIONS / WFC) blow up to hundreds of thousands of
# columns and are used by only a handful of feature sets. They are written
# to a sibling CSV next to features_and_metadata.csv and lazy-loaded by
# downstream callers (predictions, eye_score) only when iterating a
# per-text feature set.
PER_TEXT_FEATURES_FILENAME = 'per_text_features.parquet'


class PosCols(StrEnum):
    """Part-of-speech tag columns."""
    PTB_POS = 'ptb_pos'
    UNIVERSAL_POS = 'universal_pos'


POS_POSSIBLE_VALUES = {
    # Real Penn Treebank tags. Restricted to tags present in ALL datasets
    # (OneStopL1/L2, MecoL1/L2) — excludes punctuation and rare tags that
    # only some datasets produce (',', ':', 'HYPH', 'NNPS', 'UH', ...),
    # which would otherwise yield NaN feature columns for some batches.
    PosCols.PTB_POS: [
        'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD',
        'NN', 'NNP', 'NNS', 'PDT', 'PRP', 'PRP$',
        'RB', 'RBR', 'RBS', 'RP', 'TO',
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'WDT', 'WP', 'WRB',
    ],
    # Universal tags, without X, SYM and INTJ (NaN columns for some batches).
    PosCols.UNIVERSAL_POS: ['VERB', 'NOUN', 'AUX', 'NUM', 'ADP', 'DET', 'ADJ', 'ADV', 'SCONJ', 'PART', 'PRON', 'CCONJ', 'PROPN', 'PUNCT']
}

FIXATIONS_METRICS_CLUSTERS = ["FF", "FP", "TF", "RP", "SK", "RR"]
FIXATIONS_METRICS_COEFS = ["FF", "FP", "TF", "RP", "SK", "RR"]


WORD_PROPERTY_COLUMNS = [
    "word_length",
    "wordfreq_frequency",
    "gpt2_surprisal",
]

s_clusters_temp = [f'{pos_catag}_{pos_col}_{measure}' for pos_col in list(PosCols) for pos_catag in POS_POSSIBLE_VALUES[pos_col] for measure in FIXATIONS_METRICS_CLUSTERS]

FEATURE_GROUPS_TO_FEATURES = {
    FeatureGroups.READING_SPEED : ['reading_speed'],
    FeatureGroups.FIXATION_METRICS : ['mean_FF', 'mean_FP', 'mean_TF', 'mean_RP', 'mean_SK', 'mean_RR'],
    FeatureGroups.S_CLUSTERS : s_clusters_temp,
    FeatureGroups.S_CLUSTERS_NO_NORM : [feature + '_no_norm' for feature in s_clusters_temp],

    FeatureGroups.WP_COEFS : [f"{eye_metric}_{wp}_coef" for eye_metric in FIXATIONS_METRICS_COEFS for wp in WORD_PROPERTY_COLUMNS] + [f"{eye_metric}_intercept" for eye_metric in FIXATIONS_METRICS_COEFS],
    FeatureGroups.WP_COEFS_NO_INTERCEPT : [f"{eye_metric}_{wp}_coef" for eye_metric in FIXATIONS_METRICS_COEFS for wp in WORD_PROPERTY_COLUMNS],
    FeatureGroups.WP_COEFS_NO_NORM : [f"{eye_metric}_{wp}_coef_no_norm" for eye_metric in FIXATIONS_METRICS_COEFS for wp in WORD_PROPERTY_COLUMNS] + [f"{eye_metric}_intercept_no_norm" for eye_metric in FIXATIONS_METRICS_COEFS],
    FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT : [f"{eye_metric}_{wp}_coef_no_norm" for eye_metric in FIXATIONS_METRICS_COEFS for wp in WORD_PROPERTY_COLUMNS],
    # Per-text features. Column names depend on the data and are resolved
    # at feature-extraction time via the prefixes above.
    FeatureGroups.TRANSITIONS : [],
    FeatureGroups.WFC : [],
}

RELEVANT_FEATURE_GROUPS_COMBINATIONS = {
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.S_CLUSTERS}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.WP_COEFS}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.S_CLUSTERS}_{FeatureGroups.WP_COEFS}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.S_CLUSTERS}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.WP_COEFS}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.S_CLUSTERS}_{FeatureGroups.WP_COEFS}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS],

    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.S_CLUSTERS_NO_NORM}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS_NO_NORM],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.WP_COEFS_NO_NORM}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_NORM],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.S_CLUSTERS_NO_NORM}_{FeatureGroups.WP_COEFS_NO_NORM}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS_NO_NORM] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_NORM],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.S_CLUSTERS_NO_NORM}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS_NO_NORM],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.WP_COEFS_NO_NORM}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_NORM],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.S_CLUSTERS_NO_NORM}_{FeatureGroups.WP_COEFS_NO_NORM}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS_NO_NORM] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_NORM],

    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.WP_COEFS_NO_INTERCEPT}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_INTERCEPT],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.S_CLUSTERS}_{FeatureGroups.WP_COEFS_NO_INTERCEPT}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_INTERCEPT],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.WP_COEFS_NO_INTERCEPT}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_INTERCEPT],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.S_CLUSTERS}_{FeatureGroups.WP_COEFS_NO_INTERCEPT}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_INTERCEPT],

    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.S_CLUSTERS_NO_NORM}_{FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS_NO_NORM] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.S_CLUSTERS_NO_NORM}_{FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS_NO_NORM] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_NORM_NO_INTERCEPT],
}

RELEVANT_FEATURE_GROUPS_COMBINATIONS_SMALL = {
    FeatureGroups.READING_SPEED : ['reading_speed'],
    FeatureGroups.FIXATION_METRICS : ['mean_FF', 'mean_FP', 'mean_TF', 'mean_RP', 'mean_SK', 'mean_RR'],
    FeatureGroups.S_CLUSTERS_NO_NORM : [feature + '_no_norm' for feature in s_clusters_temp],
    FeatureGroups.WP_COEFS_NO_NORM : [f"{eye_metric}_{wp}_coef_no_norm" for eye_metric in FIXATIONS_METRICS_COEFS for wp in WORD_PROPERTY_COLUMNS] + [f"{eye_metric}_intercept_no_norm" for eye_metric in FIXATIONS_METRICS_COEFS],
    f"{FeatureGroups.READING_SPEED}_{FeatureGroups.FIXATION_METRICS}_{FeatureGroups.S_CLUSTERS_NO_NORM}_{FeatureGroups.WP_COEFS_NO_NORM}": FEATURE_GROUPS_TO_FEATURES[FeatureGroups.READING_SPEED] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.FIXATION_METRICS] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.S_CLUSTERS_NO_NORM] + FEATURE_GROUPS_TO_FEATURES[FeatureGroups.WP_COEFS_NO_NORM],
    # Per-text feature groups. Columns are dataset-dependent and resolved
    # at runtime by prefix; the empty list here is a placeholder. They run
    # only where training and test participants read the same texts (the
    # Seen pool on OneStop, any pool on MECO); other combinations are
    # skipped with a warning.
    FeatureGroups.TRANSITIONS: [],
    FeatureGroups.WFC: [],
}

RELEVANT_FEATURE_GROUPS_COMBINATIONS_MINIMAL = {
    FeatureGroups.READING_SPEED : ['reading_speed'],
    FeatureGroups.FIXATION_METRICS : ['mean_FF', 'mean_FP', 'mean_TF', 'mean_RP', 'mean_SK', 'mean_RR'],
    FeatureGroups.S_CLUSTERS_NO_NORM : [feature + '_no_norm' for feature in s_clusters_temp],
    FeatureGroups.WP_COEFS_NO_NORM : [f"{eye_metric}_{wp}_coef_no_norm" for eye_metric in FIXATIONS_METRICS_COEFS for wp in WORD_PROPERTY_COLUMNS] + [f"{eye_metric}_intercept_no_norm" for eye_metric in FIXATIONS_METRICS_COEFS],
}

ALL_FEATURE_SETS_DICT = FEATURE_GROUPS_TO_FEATURES | RELEVANT_FEATURE_GROUPS_COMBINATIONS


EYE_METRICS_NICKNAMES = {
    "GD": "IA_FIRST_RUN_DWELL_TIME",
    "FirstFixProg": "IA_FIRST_FIX_PROGRESSIVE",
    "RegPD": "IA_REGRESSION_PATH_DURATION",
    'NF': "IA_FIXATION_COUNT",
    "FirstPassFF": "FirstPassFFD",
    # the fixation metrics of the feature groups
    "FF": "IA_FIRST_FIXATION_DURATION",
    "FP":"IA_FIRST_RUN_DWELL_TIME",
    "TF": "IA_DWELL_TIME",
    "RP":"IA_REGRESSION_PATH_DURATION",
    "SK": "IA_SKIP",
    "RR": "IA_RR",
}

EYE_METRICS_NICKNAMES_INVERTED = {value: key for key, value in EYE_METRICS_NICKNAMES.items()}


class FoldsMethods(StrEnum):
    """Fold methods of the first_p_agg / moving_p_agg path (BaseModel.create_folds);
    fully_agg uses Pool x FoldMethod below. BATCH_CROSS_VALIDATION is no longer
    supported: src/run/predictions.py rejects it with an explicit error.
    """
    RANDOM = "random"
    LEAVE_ONE_LANGUAGE_OUT = "leave_one_language_out"
    LEAVE_ONE_PARTICIPANT_OUT = "leave_one_participant_out"
    BATCH_CROSS_VALIDATION = "batch_cross_validation"  # rejected by src/run/predictions.py


# Two-axis split surface (Pool × FoldMethod). The canonical naming for the
# fully_agg orchestration. first_p_agg / moving_p_agg still flow through the
# legacy FoldsMethods enum above but write CSVs using these names.
class Pool(StrEnum):
    """Which training rows are eligible relative to the test rows.

    ALL    — no text-context filter; train on everything else
    SEEN   — train rows share the same text context as test
             (OneStop: same content_group; MECO: same paragraph-half agg)
    UNSEEN — train rows have a different text context from test
             (OneStop: different content_group; MECO: opposite paragraph-half agg)
    """
    ALL = "all"
    SEEN = "seen"
    UNSEEN = "unseen"


class FoldMethod(StrEnum):
    """How to partition the pool into folds.

    POOL_ALL        — per-participant LOPO within the pool (FoldsMethods: LEAVE_ONE_PARTICIPANT_OUT)
    LOLO            — leave-one-L1-out within the pool (FoldsMethods: LEAVE_ONE_LANGUAGE_OUT)
    L1_STRAT_KFOLD  — L1-stratified k-fold within the pool, k = #L1 (FoldsMethods: RANDOM)
    """
    POOL_ALL = "pool_all"
    LOLO = "lolo"
    L1_STRAT_KFOLD = "l1_strat_kfold"


class InnerValidation(StrEnum):
    """How the tree models select a hyperparameter config inside one outer fold.

    The outer loop (leave-one-participant-out) is unchanged and identical for
    every model. This picks what happens *inside* it, using only that fold's
    training rows:

    NONE     — no search; the model runs at its constructor defaults (the
               default).
    HOLDOUT  — one stratified train/validation split of the outer training set.
               ~1/k the cost of KFOLD; the fallback when the full loop is too
               slow or hits trouble.
    KFOLD    — inner k-fold over the outer training set, scored on pooled
               out-of-fold predictions.

    Ridge/LogRidge/Linear ignore this entirely: they select alpha through
    RidgeCV's closed-form LOO (see RidgeRegression.try_loocv_segment), which is
    free and strictly better than an explicit split.
    """
    NONE = "none"
    HOLDOUT = "holdout"
    KFOLD = "kfold"


INNER_KFOLD_K = 5
INNER_HOLDOUT_FRAC = 0.2

# Hyperparameter grids for the tree family, keyed by the ModelNames *value*
# (plain strings here to avoid a circular import with src.configs, which imports
# the model classes). Consumed via BaseModel.hyperparameter_grid().
#
# Every entry must be a valid kwarg for the underlying estimator's set_params:
# sklearn's DecisionTreeRegressor / RandomForestRegressor, lgb.LGBMRegressor,
# xgb.XGBRegressor.
#
# max_features is deliberately NOT tuned for RandomForest. It is pinned to
# 'sqrt' in the constructor: that is the canonical definition of a random forest
# (1.0 makes it bagged trees), and measured on this data 1.0 costs 135x on
# per-text / 14.4x on scalar for no accuracy gain.
TREE_PARAM_GRIDS: dict[str, list[dict]] = {
    "Decision_Tree": [
        {"max_depth": d, "min_samples_leaf": leaf}
        for d in (3, 6, 12, None)
        for leaf in (1, 5)
    ][:10],
    "Random_Forest": [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
        for n in (200, 500)
        for d in (6, 12, None)
        for leaf in (1, 5)
    ][:10],
    "LightGBM": [
        {"num_leaves": nl, "learning_rate": lr, "colsample_bytree": cs}
        for nl in (15, 31, 63)
        for lr in (0.03, 0.1)
        for cs in (0.3, 1.0)
    ][:10],
    "XGBoost": [
        {"max_depth": d, "learning_rate": lr, "colsample_bytree": cs}
        for d in (3, 6, 10)
        for lr in (0.03, 0.1)
        for cs in (0.3, 1.0)
    ][:10],
}

class TestCols(StrEnum):
    """Proficiency-test columns, and the metadata columns read alongside them."""
    MICHIGEN_TEST_COL = "michtest_score"
    LEXTALE_COL = "lextale_score"
    PROFICIENCY_AGG_COL = "proficiency_agg"
    COMPREHENSION_COL = "comprehension_score-regular_trials"
    COMPREHENSION_COL_REREAD = "comprehension_score-repeated_reading"
    LANGUAGEC_COL = "L1"
    PREVIEW_COL = "question_preview"
    BATCH_COL = "article_batch"
    DATE_COL = "date"


MICH_TEST_PARTS = ["MPT_listening_score","MPT_grammar_score","MPT_vocabulary_score","MPT_reading_score"]
MICH_TEST_SUM_PARTS = [ "MPT_listen_grammar", "MPT_vocab_read", "MPT_grammar_vocab_read"]
MICH_TEST_PARTS_EXTENDED = MICH_TEST_PARTS + MICH_TEST_SUM_PARTS
COMPREHENSION_COLS = [TestCols.COMPREHENSION_COL, TestCols.COMPREHENSION_COL_REREAD]
ALL_TESTS = [TestCols.MICHIGEN_TEST_COL,TestCols.LEXTALE_COL, TestCols.PROFICIENCY_AGG_COL, TestCols.COMPREHENSION_COL]+ MICH_TEST_PARTS_EXTENDED

ALL_TESTS_MAX_SCORES = {
    "MPT_listening_score":20,
    "MPT_grammar_score":30,
    "MPT_vocabulary_score":30,
    "MPT_reading_score":20,
    "MPT_first_part":50,
    "MPT_second_part":50,
    "MPT_listen_grammar":50,
    "MPT_vocab_read":50,
    "MPT_grammar_vocab_read":80,
    TestCols.LEXTALE_COL:100,
    "converted_toefl_score":120,
    "toefl_lr":120,
    TestCols.MICHIGEN_TEST_COL:100,
    TestCols.COMPREHENSION_COL: 100,
    TestCols.COMPREHENSION_COL_REREAD: 100,
}


TEST_COLS_IN_L1 = [TestCols.LEXTALE_COL, TestCols.PROFICIENCY_AGG_COL]

ALL_TARGET_COLS = {
    DataSets.ONESTOPL2: ALL_TESTS,
    DataSets.ONESTOPL1: [TestCols.LEXTALE_COL],
    DataSets.MECOL1: [TestCols.LEXTALE_COL, TestCols.PROFICIENCY_AGG_COL],
    DataSets.MECOL2: [TestCols.LEXTALE_COL, TestCols.PROFICIENCY_AGG_COL],
}

ALL_TARGET_COLS_SMALL = {
    DataSets.ONESTOPL2: [TestCols.LEXTALE_COL, TestCols.MICHIGEN_TEST_COL],
    DataSets.ONESTOPL1: [TestCols.LEXTALE_COL],
    DataSets.MECOL1: [TestCols.LEXTALE_COL, TestCols.PROFICIENCY_AGG_COL],
    DataSets.MECOL2: [TestCols.LEXTALE_COL, TestCols.PROFICIENCY_AGG_COL],
}

# Comprehension-only sweeps: reading comprehension accuracy on its own, without
# the proficiency tests riding along. Both corpora store it on a 0-100 scale
# under the same column name, so --datasets picks whether a run covers Meco,
# OneStop, or both. The repeated-reading variant exists only in OneStop and is
# deliberately left out, so the setting means the same thing in either corpus --
# pass --target-cols comprehension_score-repeated_reading to run that instead.
ALL_TARGET_COLS_COMPREHENSION = {
    DataSets.ONESTOPL2: [TestCols.COMPREHENSION_COL],
    DataSets.ONESTOPL1: [TestCols.COMPREHENSION_COL],
    DataSets.MECOL1: [TestCols.COMPREHENSION_COL],
    DataSets.MECOL2: [TestCols.COMPREHENSION_COL],
}

TARGET_COL_SETS = {
    'small': ALL_TARGET_COLS_SMALL,
    'all': ALL_TARGET_COLS,
    'comprehension': ALL_TARGET_COLS_COMPREHENSION,
}

# argparse choices for --target-set-size, kept in sync with the table above.
TARGET_SET_SIZES = list(TARGET_COL_SETS)


def get_target_cols(
    dataset_key: str,
    target_set_size: str = 'small',
    target_cols: list[str] | None = None,
) -> list[str]:
    """Resolve the target columns to sweep for one dataset.

    An explicit --target-cols override wins when given; otherwise the named set
    decides. An unknown set name raises instead of silently falling back to the
    full sweep, which would quietly run far more jobs than asked for.
    """
    if target_cols:
        return list(target_cols)
    if target_set_size not in TARGET_COL_SETS:
        raise ValueError(
            f'Unknown target_set_size {target_set_size!r}; '
            f'expected one of {sorted(TARGET_COL_SETS)}'
        )
    return list(TARGET_COL_SETS[target_set_size].get(dataset_key, []))


# Small feature set: the 4 single-family scalar sets and the 2 per-text sets.
# The all-scalar union (READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_
# WP_COEFS_NO_NORM) is in ALL_FEATURE_SETS_DICT; request it with --feature-sets.
SMALL_FEATURE_SETS = [
    'READING_SPEED',
    'FIXATION_METRICS',
    'S_CLUSTERS_NO_NORM',
    'WP_COEFS_NO_NORM',
    'TRANSITIONS',
    'WFC',
]


class AggTypes(StrEnum):
    """Feature aggregation types (the directory level under features_and_targets/)."""
    FULL = 'fully_agg'
    FIRST_P = 'first_p_agg'
    MOVING_P = 'moving_p_agg'
    PER_ITEM = 'per_item_agg'
    # MECO-only per-half pseudo-agg. Evaluation-side the contrast is expressed
    # as Pool ∈ {SEEN, UNSEEN} layered on top of fully_agg; this agg type only
    # (re)generates the per-half feature exports those pools (and the EyeScore
    # seen/unseen jobs) read from features_and_targets/seen_unseen/all/all/.
    SEEN_UNSEEN = 'seen_unseen'


class TrainTypes(StrEnum):
    """What a model is trained on when evaluated on partially aggregated data
    (first_p / moving_p): the partial data itself, or the full data."""
    PARTIAL = 'partial_train'
    FULL = 'train_on_full'
