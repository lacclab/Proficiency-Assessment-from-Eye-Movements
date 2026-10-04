# Language Proficiency Assessment from Eye-Tracking Data

A pipeline for assessing second-language proficiency from eye movements in
reading. It extracts eye-tracking features from the MECO and OneStop corpora,
trains models that predict proficiency test scores, computes EyeScore, how
similar a reader's eye movements are to those of native (L1) readers, and
renders the paper's tables and figures.

## Data availability

| Corpus | Readers | Source |
|---|---|---|
| MECO | English L1 and L2 readers of the same English texts | OSF, MECO L2 release 2.0 (version 2.1) |
| OneStop | English L1 readers | OSF, OneStop v1.0.3 (files unchanged since v1.0.2) |
| OneStop | English L2 readers | not publicly available |

The repository holds the code only: every result, `paper/` included, comes
from running the pipeline. The MECO results can be rerun from the raw data. The
OneStop results cannot: they need the OneStop L2 readers, and the OneStop L1
files they were computed from were aligned together with L2 (see stage 0).

## Setup

The environment pins Python 3.12.

```bash
conda env create -f prof_env.yml
conda activate prof_env
# URIEL+ (typological distances, used by EyeScore) has to be installed on its
# own with --no-deps: it declares tensorflow-addons, which has no Python 3.12
# wheels, although nothing in it imports tensorflow.
pip install urielplus==1.2 --no-deps
pip install tabpfn
```

## Running the pipeline

Each stage is a module under `src/run/`, run from the repository root, and
reads what the stage before it wrote. `--help` on any stage lists its options.

| Stage | Command | Writes |
|---|---|---|
| 0. Data | `python -m src.preprocessing.download` | `data/<dataset>/{downloads,harmonized,metadata}/` |
| 1. Features | `python -m src.run.features` | `data/<dataset>/features_and_targets/` |
| 2a. Predictions | `python -m src.run.predictions` | `src/methods/predictions/results/`, `src/evaluation/predictions/` |
| 2b. EyeScore | `python -m src.run.eyescore` | `src/methods/EyeScore/results*/`, `src/evaluation/EyeScore/` |
| 3. Paper | `python -m src.run.paper` | `paper/` |

Stages 2a and 2b are independent of each other; both need stage 1. The
commands below run on MECO: without `--datasets`, stages 1, 2a and 2b take
OneStop as well, which needs the L2 readers.

### 0. Download and harmonize the data

```bash
python -m src.preprocessing.download                  # OneStop L1 and MECO
python -m src.preprocessing.download --dataset Meco   # MECO only
```

Downloads the corpora from their pinned OSF releases, harmonizes column names
across datasets, and writes each dataset's metadata. `Meco` processes MECO's
L1 and L2 readers together, into `data/MecoL1/` and `data/MecoL2/`, so both
cohorts get the same preprocessing. Asking for `OneStopL2` stops with an
error. OneStop L1 is processed on its own, without
the interest-area alignment with L2 that the paper's OneStop files went
through, and a warning says so.

### 1. Extract features

```bash
python -m src.run.features --datasets MecoL1,MecoL2   # every aggregation
python -m src.run.features --datasets MecoL1,MecoL2 --agg-types fully_agg,seen_unseen
python -m src.run.features --datasets MecoL1,MecoL2 --agg-types fully_agg --half-split   # split-half features, for reliability
```

Aggregations: `fully_agg` (the whole session), `seen_unseen` (MECO's odd/even
paragraph halves, which feed the Fixed/Any text regimes), `first_p_agg` (the
first N paragraphs/articles), `moving_p_agg` (a sliding window) and
`per_item_agg` (one row per participant x item, for reliability).

### 2a. Predictions

```bash
python -m src.run.predictions --datasets Meco --models Ridge_Classifier   # train
python -m src.run.predictions --stage evaluation --agg-types fully_agg
```

`--stage train` (default) fits every (dataset, preview, model, target, feature
set) cell; `--stage evaluation` scores the prediction CSVs on disk -- metrics,
bootstrap CIs, pairwise significance -- and renders the per-model tables;
`--stage both` does one after the other. `--mode` selects the cross-validation
design: `pool_fold` (default, the paper's), `per_item`, `split_half`
(reliability) or `standard` (the first_p/moving_p sweeps). The tree models
select hyperparameters with `--inner-validation holdout|kfold`.

### 2b. EyeScore

```bash
python -m src.run.eyescore --datasets Meco --agg-types fully_agg,seen_unseen   # calculation, then evaluation
python -m src.run.eyescore --datasets Meco --stage calculation --agg-types seen_unseen
```

Calculation scores each L2 reader against the L1 prototype and applies every
registered debiasing correction; evaluation correlates the scores with the
proficiency tests and runs the significance tests, the language-bias analysis
(`src/evaluation/EyeScore/language_bias.py`), plots and tables.
MECO's Fixed and Any regimes come from `seen_unseen`, which the default
`--agg-types` leaves out. `--debias-methods` builds one results tree per
debiasing method.

### 3. Paper tables and figures

```bash
python -m src.run.paper                             # every published table and figure
python -m src.run.paper --only eyescore,predictions
python -m src.run.paper --out /tmp/paper_check      # render elsewhere, e.g. to diff against paper/
```

`src/run/paper/` is the only code that writes into `paper/`; its `__main__`
lists which file each section produces. With the MECO results alone, the
tables render their MECO parts: OneStop cells show `-`, and the tables on
OneStop alone are skipped with a warning.

### Reproducing the paper

The paper's tables read, in addition to the runs above:

- the `Average` baseline model (the mean-baseline row) and the tuned LightGBM
  (the model-comparison tables), evaluated against Ridge:

  ```bash
  python -m src.run.predictions --datasets Meco --models Ridge_Classifier,Average
  python -m src.run.predictions --datasets Meco --models TabPFN
  python -m src.run.predictions --datasets Meco --models LightGBM --inner-validation kfold
  python -m src.run.predictions --stage evaluation --agg-types fully_agg \
      --models Ridge_Classifier,LightGBM,Average --pairwise-baseline-only --pairwise-vs-ridge
  ```

- debiasing result tree for the L1-bias tables and figures:

  ```bash
  python -m src.run.eyescore --datasets Meco --agg-types seen_unseen \
      --debias-methods two_step \
      --results-suffix cat3ctr --center-distance \
      --typo-calib-distance-type 'cat:syntactic+phonological+scriptural'
  ```

### Reliability analysis

Cronbach's alpha and the split-half reliability of EyeScore and of the
predictions. They read the split-half and per-item runs: features with
`--half-split` and `--agg-types per_item_agg`, predictions with
`--mode split_half` and `--mode per_item`, and EyeScore with
`--split-half-eyescore` and `--agg-types per_item_agg`. Then:

```bash
python -m src.reliability.split_half
python -m src.reliability.cronbach_alpha
python -m src.reliability.tables
python -m src.run.paper --only reliability          # the paper's reliability tables
```

The Michigan test's reliability needs the OneStop L2 readers' item responses,
so `src.reliability.calculations_for_external_tests` cannot be run on the public
data.

## Tests

```bash
python -m pytest -q -p no:cacheprovider src/tests
python -m pytest -q -p no:cacheprovider src/tests --runslow   # adds the tests on the real data
```

The `--runslow` tests run on the data under `data/` (the harmonized files and
the `moving_p_agg` feature trees) and fail where it is absent.

## Known Issues

### `WP_COEFS fit failed for participant=ch_s_* metric=RR` warnings during feature extraction

Expected and benign. The `ch_s_*` (Chinese-script) MECO participants have
`IA_FIRST_FIX_PROGRESSIVE == 0` for every row, so their regression-rate (RR)
values are uniformly NaN and the per-participant WP-coefficient regression has
nothing to fit. The handler in `src/preprocessing/extract_features.py`
(`find_wp_coefs_inner`) logs the warning, writes NaN for that participant's RR
WP-coefficients, and continues — all other features for the participant are
extracted normally. These warnings appear in every extraction run and do not
indicate data corruption.

### `RankWarning: Polyfit may be poorly conditioned` during EyeScore plotting

Expected and cosmetic. `src/evaluation/EyeScore/plot.py` draws a per-L1
trend line in the EyeScore-vs-test scatter plots with `np.polyfit(..., 1)`;
the warning fires for language groups that are degenerate for a linear fit
(as few as 2 participants, or near-identical test scores within the group).
At worst that single language's trend line is unreliable in the plot — the
evaluation results and tables are computed before plotting and are unaffected.

## AI tool acknowledgment

Parts of this project were developed with the assistance of AI coding tools, including Anthropic's Claude Code VS Code extension and GitHub Copilot, and carefully reviewed and edited by the authors.
