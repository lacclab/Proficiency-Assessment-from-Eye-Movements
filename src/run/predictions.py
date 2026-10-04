"""Stage 2a: train the prediction models, then evaluate them.

    --stage train       (default) fit every (dataset, preview, model, target,
                        feature set) cell in parallel and write its predictions
                        to src/methods/predictions/results/
    --stage evaluation  score the result CSVs on disk -- metrics, bootstrap CIs,
                        pairwise significance -- into src/evaluation/predictions/
                        results/, and render the per-model tables beside them
    --stage both        train, then evaluate

--mode picks the cross-validation design. pool_fold (default) is the paper's:
each dataset's participants split into pools (all / seen / unseen text) crossed
with a fold method (pool_all / lolo / l1_strat_kfold). per_item predicts one row
per participant x item, split_half reruns pool_fold on each half of the random
split-half partitions (reliability), and standard is the older
methods-based loop kept for the first_p_agg / moving_p_agg sweeps.

    python -m src.run.predictions --models Ridge_Classifier
    python -m src.run.predictions --models LightGBM --inner-validation kfold
    python -m src.run.predictions --stage evaluation --agg-types fully_agg
"""
import os

# BLAS threads must be pinned BEFORE numpy/sklearn load — OpenBLAS reads these
# at import time. This pipeline fits many small matrices, where multi-threaded
# BLAS thrashes: RidgeCV on ~1000×78 measures 423ms with default threads vs
# 5.7ms pinned (one MECO segment: 506s vs 11s). Parallelism here is at the job
# level anyway, so one BLAS thread per job is the right split. Export the vars
# yourself to override — e.g. per-text sweeps with --n-jobs 1, where the
# matrices are wide enough for threading to pay off.
for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import itertools
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm
import multiprocessing
import argparse
import traceback
import sys

from src.constants import (
    FoldsMethods, AggTypes, get_target_cols, TARGET_SET_SIZES,
    ALL_FEATURE_SETS_DICT, DATA_PATH, Fields, TestCols,
    SMALL_FEATURE_SETS,
    FIXED_TEXT_GROUP_PREFIXES, Pool, FoldMethod, InnerValidation)
from src.methods.predictions.pool_resolver import (
    OneStopPoolResolver, MecoPoolResolver, PoolSegment,
)
from src.configs import MODELS
from src.agg_configs import get_agg_configs
from src.methods.predictions.run import (
    run_one_feature_set,
    load_and_filter_data,
    _create_model,
    DATASET_PREVIEWS,
    DEFAULT_METHODS,
    run_per_item_seen,
    run_per_item_unseen,
    aggregate_per_item_predictions,
)
from src.methods.predictions.per_item_resolver import (
    PerItemOneStopResolver,
    PerItemMecoResolver,
)
from src.utils.cli import parse_csv
from src.methods.utils import load_per_text_features_pair

def get_per_item_parquet_dir(l2_features_path: Path, agg_type) -> Path | None:
    """Get the directory containing per-item parquets for per_item_agg.

    Returns None for other agg types (they use combined parquet files).
    """
    if agg_type != AggTypes.PER_ITEM:
        return None
    # Per-item parquets are in the same directory as the features CSV
    return Path(l2_features_path).parent
from src.utils.filter import filter_by_preview
from src.utils.split_half_columns import split_half_common_feature_cols
import pandas as pd

logger.add(DATA_PATH / 'logs' / 'predictions.log', rotation='10 MB', retention='7 days')


SEEN_UNSEEN_HALVES = ("odd", "even")

# How prediction jobs are executed; see _JobRunner for why processes win.
EXECUTOR_KINDS = ("processes", "threads")

RESULTS_ROOT = Path("src/methods/predictions/results")


def _results_root(require_l1_lextale: bool) -> Path:
    """Results root. --require-l1-lextale runs write to a separate subtree so
    they never collide with default runs and the two can be compared later."""
    return RESULTS_ROOT / "require_l1_lextale" if require_l1_lextale else RESULTS_ROOT


def _filter_l1_lextale(l1_df: pd.DataFrame | None, context: str = "") -> pd.DataFrame | None:
    """Restrict an L1 frame to participants with a measured LexTALE score
    (drops NaN and the -1 missing sentinel).

    Applied to every L1 frame that can reach training as an augmentation
    source — including runs targeting Michigan/aggregate, which then augment
    (with imputation) over the smaller LexTALE-having pool. L2 is untouched.
    """
    if l1_df is None:
        return None
    col = TestCols.LEXTALE_COL
    if col not in l1_df.columns:
        raise ValueError(f"--require-l1-lextale: L1 frame has no '{col}' column ({context})")
    scores = pd.to_numeric(l1_df[col], errors="coerce")
    kept = l1_df[(scores.notna() & (scores != -1)).to_numpy()]
    logger.info(
        f"require_l1_lextale{f' [{context}]' if context else ''}: kept "
        f"{kept[Fields.SUBJECT_ID].nunique()}/{l1_df[Fields.SUBJECT_ID].nunique()} L1 participants "
        f"({len(kept)}/{len(l1_df)} rows)"
    )
    return kept


def _filter_halves_l1_lextale(halves_data: dict | None, context: str = "") -> dict | None:
    """Apply the LexTALE filter to the l1 frame of each half in a halves dict."""
    if halves_data is None:
        return None
    return {
        half: {"l1": _filter_l1_lextale(d["l1"], f"{context}/{half}"), "l2": d["l2"]}
        for half, d in halves_data.items()
    }


def _seen_unseen_feature_paths(dataset: str, half: str) -> tuple:
    """Resolve (l1_csv, l2_csv) paths for a half-split feature set."""
    suffix = f"seen_unseen/all/all/{half}/features_and_metadata.csv"
    return (
        Path(DATA_PATH / f"{dataset}L1/features_and_targets/{suffix}"),
        Path(DATA_PATH / f"{dataset}L2/features_and_targets/{suffix}"),
    )


# _run_seen_unseen_prediction_job (the MECO odd/even per-half orchestration)
# was removed in Phase 3. The same contrast is now produced by the
# Pool × FoldMethod orchestration below with MecoPoolResolver; the half
# averaging happens in _combine_segment_predictions (strategy="average").


# ============================================================
# Pool × FoldMethod orchestration. Writes results to
# .../fully_agg/.../{model}/{pool}__{fold_method}.csv.
# ============================================================
def _warn_l1_dropped_cols(feature_cols, available, l2_frame, context: str) -> None:
    """Log feature columns dropped because the L1 augmentation frame lacks them.

    A dropped column that is CONSTANT in L2 carries no signal, so losing it is
    free (this covers the known MECO cases, where the tag has no fixations in
    that text subset and L2 records a flat 0). A dropped column that VARIES in
    L2 is a real feature disappearing from one segment but not another, which
    would silently make two segments incomparable — that one is worth shouting
    about.
    """
    dropped = [c for c in feature_cols if c not in available and c in l2_frame.columns]
    if not dropped:
        return
    varying, constant = [], []
    for c in dropped:
        (varying if l2_frame[c].nunique(dropna=True) > 1 else constant).append(c)
    if constant:
        logger.info(
            f"{context}: dropped {len(constant)} feature col(s) absent from L1 and "
            f"constant in L2 (no signal lost): {constant[:5]}"
        )
    if varying:
        logger.warning(
            f"{context}: dropped {len(varying)} feature col(s) absent from L1 that VARY "
            f"in L2 — this segment is now fitted on a different feature set than its "
            f"peers: {varying[:5]}"
        )


# Shared with the EyeScore runner, which reads the same split-half CSVs and would
# otherwise drift independently. See src/utils/split_half_columns.py.
_split_half_common_feature_cols = split_half_common_feature_cols


def _resolve_meco_halves_paths(dataset: str):
    """Return {half: (l1_path, l2_path)} for MECO paragraph-half feature CSVs."""
    return {
        half: _seen_unseen_feature_paths(dataset, half)
        for half in SEEN_UNSEEN_HALVES
    }


# ============================================================
# Shared helper functions for pool-fold data loading
# ============================================================
def _load_pool_fold_data(
    dataset: str, preview: str, split_idx: int | None = None, half_num: int | None = None,
    item_level: str | None = None, require_l1_lextale: bool = False
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load L1/L2 features for pool-fold (regular or split-half).

    If split_idx and half_num are provided, loads split-half data; otherwise loads regular data.
    For OneStop split-half, item_level determines paragraph vs article; for Meco, defaults to paragraph.
    """
    if split_idx is not None and half_num is not None:
        # For OneStop split-half: use item_level; for Meco split-half: always use "paragraph"
        p_agg_level = item_level if dataset == "OneStop" else "paragraph"
        l1_df, l2_df = load_split_half_data(dataset, split_idx, half_num, AggTypes.FULL, p_agg_level, None, preview)[:2]
    else:
        l1_df, l2_df, _ = load_and_filter_data(dataset, AggTypes.FULL, None, None, preview)
    if require_l1_lextale:
        l1_df = _filter_l1_lextale(l1_df, f"{dataset}/{preview}")
    return l1_df, l2_df


def _load_pool_fold_per_text(
    dataset: str, preview: str, needs_per_text: bool,
    split_idx: int | None = None, half_num: int | None = None,
    item_level: str | None = None, require_l1_lextale: bool = False
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load per-text features for pool-fold (regular or split-half).

    Returns (pt_l1_df, pt_l2_df) or (None, None) if not needed or missing.
    For split-half, loads from split-half-specific parquets; for regular, loads from aggregated CSV.
    """
    if not needs_per_text:
        return None, None

    is_meco_ds = (dataset == "Meco")
    reread_level = "all" if is_meco_ds else "ordinary"

    if split_idx is not None and half_num is not None:
        # Split-half per-text features are in split-half-specific parquets
        if is_meco_ds:
            # For Meco, per-text parquets are at split/half/seen_unseen/{odd|even}/
            # Load aggregated (not split by odd/even) - return None since per-text
            # per-text features are stored separately per odd/even
            return None, None
        else:
            # For OneStop, per-text parquets are directly in split/half/
            p_level = item_level or "paragraph"
            pt_l1_parquet = DATA_PATH / f"{dataset}L1/features_and_targets/split_half/fully_agg/{reread_level}/all/{p_level}/split_{split_idx}/half_{half_num}/per_text_features.parquet"
            pt_l2_parquet = DATA_PATH / f"{dataset}L2/features_and_targets/split_half/fully_agg/{reread_level}/all/{p_level}/split_{split_idx}/half_{half_num}/per_text_features.parquet"

            if not pt_l1_parquet.exists() or not pt_l2_parquet.exists():
                return None, None

            # Same thrift bump as load_per_text_features: these parquets carry
            # hundreds of thousands of columns (OneStop fully_agg/all reaches
            # ~500k), and pyarrow's defaults (string=100MB, container=1M) reject
            # the encoded schema footer.
            pt_l1_raw = pd.read_parquet(
                pt_l1_parquet,
                thrift_string_size_limit=2**31 - 1,
                thrift_container_size_limit=2**31 - 1,
            )
            pt_l2_raw = pd.read_parquet(
                pt_l2_parquet,
                thrift_string_size_limit=2**31 - 1,
                thrift_container_size_limit=2**31 - 1,
            )
    else:
        # Regular per-text features from CSV
        pt_l1_path = DATA_PATH / f"{dataset}L1/features_and_targets/fully_agg/{reread_level}/all/features_and_metadata.csv"
        pt_l2_path = DATA_PATH / f"{dataset}L2/features_and_targets/fully_agg/{reread_level}/all/features_and_metadata.csv"
        pt_l1_raw, pt_l2_raw = load_per_text_features_pair(pt_l1_path, pt_l2_path)
        if pt_l1_raw is None or pt_l2_raw is None:
            return None, None

    if not is_meco_ds:
        pt_l1_raw, pt_l2_raw = filter_by_preview(pt_l1_raw, pt_l2_raw, preview)
    if require_l1_lextale:
        pt_l1_raw = _filter_l1_lextale(pt_l1_raw, f"{dataset}/{preview}/per_text")
    return pt_l1_raw, pt_l2_raw


def _load_pool_fold_halves(
    dataset: str, pools: list[str],
    split_idx: int | None = None, needs_per_text: bool = False,
    item_level: str | None = None, half_num: int | None = None,
    require_l1_lextale: bool = False
) -> tuple[dict | None, dict | None]:
    """Load MECO halves for pool-fold (regular or split-half).

    Returns (halves_data, pt_halves_data) where each is {half: {l1, l2}} or None.
    Only loads if dataset is MECO and pools contain SEEN/UNSEEN.
    item_level is ignored for halves (always paragraph for MECO).
    For split-half: half_num specifies which half (1 or 2) to load.
    """
    if dataset != "Meco" or not ({Pool.SEEN.value, Pool.UNSEEN.value} & set(pools)):
        return None, None

    if split_idx is not None:
        # Split-half halves: load only the specified half_num
        paths_all = _resolve_split_half_meco_halves_paths(dataset, split_idx)
        # Filter to only the requested half_num (e.g., if half_num=1, keep odd_half1, even_half1)
        if half_num is not None:
            paths = {k: v for k, v in paths_all.items() if f"_half{half_num}" in k}
            # Rename keys from "odd_half1" → "odd", "even_half1" → "even" etc.
            paths = {k.replace(f"_half{half_num}", ""): v for k, v in paths.items()}
        else:
            paths = paths_all
    else:
        # Regular halves
        paths = _resolve_meco_halves_paths(dataset)

    halves_data = {}
    for key, (p1, p2) in paths.items():
        if p1.exists() and p2.exists():
            halves_data[key] = {"l1": pd.read_csv(p1), "l2": pd.read_csv(p2)}

    if not halves_data:
        return None, None

    pt_halves_data = None
    if needs_per_text:
        pt_halves_data = {}
        for key, (l1p, l2p) in paths.items():
            pt1, pt2 = load_per_text_features_pair(l1p, l2p)
            if pt1 is None or pt2 is None:
                pt_halves_data = None
                break
            pt_halves_data[key] = {"l1": pt1, "l2": pt2}

    if require_l1_lextale:
        halves_data = _filter_halves_l1_lextale(halves_data, f"{dataset}/halves")
        pt_halves_data = _filter_halves_l1_lextale(pt_halves_data, f"{dataset}/pt_halves")

    return halves_data, pt_halves_data


# ============================================================
# Split-half prediction orchestration
# ============================================================
def load_split_half_data(
    dataset: str,
    split_idx: int,
    half_num: int,
    agg_type,
    p_agg_level: str | None,
    p_agg: int | None,
    preview: str | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Load L1/L2 feature DataFrames for a specific split-half combo.

    Args:
        dataset: Dataset name (e.g., "OneStop", "Meco")
        split_idx: Split index (1-based)
        half_num: Half number (1 or 2)
        agg_type: Aggregation type (FULL, FIRST_P, MOVING_P, PER_ITEM)
        p_agg_level: Paragraph or article level (for FIRST_P, MOVING_P, PER_ITEM)
        p_agg: Window size for FIRST_P/MOVING_P
        preview: Preview filter (OneStop only)

    Returns:
        (l1_df, l2_df, agg_label) or (None, None, agg_label) if data not found
    """
    is_meco = dataset == "Meco"
    agg_type_str = agg_type.value if hasattr(agg_type, "value") else str(agg_type)

    # Build agg_label (same as load_and_filter_data)
    if agg_type_str == AggTypes.FULL:
        agg_label = agg_type_str
    elif agg_type_str == AggTypes.PER_ITEM:
        agg_label = f"{agg_type_str}_{p_agg_level}"
    else:
        agg_label = f"{agg_type_str}_{p_agg_level}_{p_agg}"

    # Build path suffix based on agg_type
    if agg_type_str == AggTypes.FULL:
        if is_meco:
            path_suffix = f"{agg_type_str}/all/all"
        else:
            # OneStop: needs p_agg_level (paragraph or article)
            p_level = p_agg_level or "paragraph"
            path_suffix = f"{agg_type_str}/ordinary/all/{p_level}"
    elif agg_type_str == AggTypes.PER_ITEM:
        path_suffix = f"{agg_type_str}/all/all/{p_agg_level}" if is_meco else f"{agg_type_str}/ordinary/all/{p_agg_level}"
    else:
        path_suffix = f"{agg_type_str}/all/all/{p_agg_level}_{p_agg}" if is_meco else f"{agg_type_str}/ordinary/all/{p_agg_level}_{p_agg}"

    if is_meco:
        l1_dfs, l2_dfs = [], []
        for odd_even in SEEN_UNSEEN_HALVES:
            l1_path = Path(DATA_PATH / f"{dataset}L1/features_and_targets/split_half/{path_suffix}/split_{split_idx}/half_{half_num}/seen_unseen/{odd_even}/features_and_metadata.csv")
            l2_path = Path(DATA_PATH / f"{dataset}L2/features_and_targets/split_half/{path_suffix}/split_{split_idx}/half_{half_num}/seen_unseen/{odd_even}/features_and_metadata.csv")

            if not l1_path.exists() or not l2_path.exists():
                continue

            l1_dfs.append(pd.read_csv(l1_path))
            l2_dfs.append(pd.read_csv(l2_path))

        if not l1_dfs or not l2_dfs:
            return None, None, agg_label

        l1_df = pd.concat(l1_dfs, ignore_index=True)
        l2_df = pd.concat(l2_dfs, ignore_index=True)
    else:
        l1_path = Path(DATA_PATH / f"{dataset}L1/features_and_targets/split_half/{path_suffix}/split_{split_idx}/half_{half_num}/features_and_metadata.csv")
        l2_path = Path(DATA_PATH / f"{dataset}L2/features_and_targets/split_half/{path_suffix}/split_{split_idx}/half_{half_num}/features_and_metadata.csv")

        if not l1_path.exists() or not l2_path.exists():
            return None, None, agg_label

        l1_df = pd.read_csv(l1_path)
        l2_df = pd.read_csv(l2_path)

        if  not is_meco and preview is not None:
            l1_df, l2_df = filter_by_preview(l1_df, l2_df, preview)

    return l1_df, l2_df, agg_label


def _resolve_split_half_meco_halves_paths(dataset: str, split_idx: int) -> dict:
    """Return {odd/even: (l1_path, l2_path)} for MECO split-half odd/even feature CSVs.

    For split-half, each split has its own odd/even halves at:
    .../split_{split_idx}/half_{1|2}/seen_unseen/{odd|even}/...
    """
    paths = {}
    for odd_even in SEEN_UNSEEN_HALVES:
        for split_half in [1, 2]:
            key = f"{odd_even}_half{split_half}"
            l1_path = (
                Path(DATA_PATH) /
                f"MecoL1/features_and_targets/split_half/fully_agg/all/all/split_{split_idx}/half_{split_half}/seen_unseen/{odd_even}/features_and_metadata.csv"
            )
            l2_path = (
                Path(DATA_PATH) /
                f"MecoL2/features_and_targets/split_half/fully_agg/all/all/split_{split_idx}/half_{split_half}/seen_unseen/{odd_even}/features_and_metadata.csv"
            )
            paths[key] = (l1_path, l2_path)
    return paths


def _combine_segment_predictions(per_segment, strategy: str) -> pd.DataFrame | None:
    """Concat (OneStop, disjoint participants) or average (MECO, overlapping)."""
    valid = [d for d in per_segment if d is not None and not d.empty]
    if not valid:
        return None
    combined = pd.concat(valid, ignore_index=True)
    if strategy == "average":
        return (combined.groupby(Fields.SUBJECT_ID, as_index=False)
                .agg({Fields.L1: 'first', 'true': 'first', 'pred': 'mean', 'fold': 'first'}))
    return combined


def _run_pool_fold_prediction_job(
    dataset: str,
    preview: str,
    model: str,
    pool: str,
    fold_method: str,
    target_col: str,
    feature_set_name: str,
    feature_cols: list,
    l1_df: pd.DataFrame | None,
    l2_df: pd.DataFrame | None,
    halves_data: dict | None,
    n_splits: int,
    aug_percentage: float,
    replace_existing: bool,
    pt_l1_df: pd.DataFrame | None = None,
    pt_l2_df: pd.DataFrame | None = None,
    pt_halves_data: dict | None = None,
    version_suffix: str = "",
    require_l1_lextale: bool = False,
    inner_validation: str = InnerValidation.NONE,
    common_feature_cols: set | None = None,
) -> tuple[str, str]:
    """Run one (pool, fold_method, target, feature_set, model) cell.

    Output path:
        src/methods/predictions/results/{dataset}/aug_p={aug_p}/{preview}/
        fully_agg/{target_col}/{feature_set}/{model}/{pool}__{fold_method}.csv

    Per-text feature sets (TRANSITIONS / WFC) work only under pool=SEEN —
    UNSEEN's train and test rows live in disjoint text-column spaces so the
    feature columns wouldn't align. For per-text runs the resolver consumes
    the per-text parquets (`pt_*` args) instead of the scalar CSVs.
    """
    pool_enum = Pool(pool)
    fold_method_enum = FoldMethod(fold_method)
    job_id = f"{dataset}/{preview}/{model}/{target_col}/{feature_set_name}/{pool}__{fold_method}"

    try:
        fixed_text_prefix = FIXED_TEXT_GROUP_PREFIXES.get(feature_set_name)
        is_per_text = fixed_text_prefix is not None
        # Per-text feature columns are paragraph-indexed. They only align
        # across train and test when both come from the same text universe:
        #   OneStop: SEEN (within one content_group) only — different CGs
        #            cover different paragraphs.
        #   MECO:    ALL or SEEN — every participant reads every paragraph,
        #            so the column space is shared across all participants;
        #            within a half it's also shared. UNSEEN crosses halves
        #            (odd-paragraph cols vs even-paragraph cols) so it's out.
        if is_per_text:
            if dataset == "Meco":
                if pool_enum == Pool.UNSEEN:
                    return (job_id, "SKIPPED PER-TEXT (MECO UNSEEN cross-half columns differ)")
            else:
                if pool_enum != Pool.SEEN:
                    return (job_id, "SKIPPED PER-TEXT (OneStop needs SEEN)")

        # A tuned run is a different protocol from an untuned one, so it must not
        # collide with it on disk: without this the existence-skip below would
        # treat a kfold result as satisfying a holdout request (and vice versa),
        # and the two would silently overwrite each other. NONE adds nothing, so
        # every pre-existing path is unchanged.
        inner_validation = InnerValidation(inner_validation)
        if inner_validation != InnerValidation.NONE:
            version_suffix = f"{version_suffix}/inner_{inner_validation.value}"

        # For split-half predictions, version_suffix replaces "fully_agg"; otherwise append to it
        if "split_half" in version_suffix:
            save_path = (
                _results_root(require_l1_lextale)
                / f"{dataset}/aug_p={aug_percentage}/{preview.lower()}{version_suffix}"
                / f"{target_col}/{feature_set_name}/{model}/{pool}__{fold_method}.csv"
            )
        else:
            save_path = (
                _results_root(require_l1_lextale)
                / f"{dataset}/aug_p={aug_percentage}/{preview.lower()}/fully_agg{version_suffix}"
                / f"{target_col}/{feature_set_name}/{model}/{pool}__{fold_method}.csv"
            )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not replace_existing:
            return (job_id, "SKIPPED")

        # Build dataset-appropriate resolver. For per-text runs swap to the
        # per-text parquets — the scalar CSVs don't carry the transitions_*
        # / wfc_* columns.
        if dataset == "Meco":
            if is_per_text:
                # Per-text + MECO. Pool decides whether we need fully_agg
                # per-text (for ALL) or halves per-text (for SEEN); UNSEEN
                # is already short-circuited above.
                if pool_enum == Pool.ALL:
                    if pt_l1_df is None or pt_l2_df is None:
                        return (job_id, "MISSING PER-TEXT (MECO fully_agg)")
                    halves_l1, halves_l2 = {}, {}
                    all_for_resolver_l1, all_for_resolver_l2 = pt_l1_df, pt_l2_df
                else:  # SEEN
                    if pt_halves_data is None:
                        return (job_id, "MISSING PER-TEXT HALVES")
                    halves_l1 = {h: pt_halves_data[h]["l1"] for h in pt_halves_data}
                    halves_l2 = {h: pt_halves_data[h]["l2"] for h in pt_halves_data}
                    all_for_resolver_l1, all_for_resolver_l2 = None, None
            else:
                if halves_data is None and pool_enum != Pool.ALL:
                    return (job_id, "NO HALVES DATA")
                halves_l1 = {h: halves_data[h]["l1"] for h in halves_data} if halves_data else {}
                halves_l2 = {h: halves_data[h]["l2"] for h in halves_data} if halves_data else {}
                all_for_resolver_l1, all_for_resolver_l2 = l1_df, l2_df
            resolver = MecoPoolResolver(
                halves_l1=halves_l1, halves_l2=halves_l2,
                all_l1=all_for_resolver_l1, all_l2=all_for_resolver_l2,
            )
            narrow_l1 = False
        else:
            if is_per_text:
                if pt_l1_df is None or pt_l2_df is None:
                    return (job_id, "MISSING PER-TEXT")
                resolver = OneStopPoolResolver(l1_df=pt_l1_df, l2_df=pt_l2_df)
            else:
                if l1_df is None or l2_df is None:
                    return (job_id, "NO DATA")
                resolver = OneStopPoolResolver(l1_df=l1_df, l2_df=l2_df)
            narrow_l1 = True

        model_obj = _create_model(model)
        feature_cols = list(feature_cols)
        # Split-half only: restrict to the columns present in EVERY frame of this
        # split, so half_1 and half_2 are fitted on identical features and the
        # reliability correlation isn't confounded by feature-set churn (see
        # _split_half_common_feature_cols). Per-text sets are excluded — their
        # columns are paragraph-indexed and resolved by the branch below, where
        # the halves deliberately span different column spaces.
        if common_feature_cols is not None and not is_per_text:
            pruned = [c for c in feature_cols if c in common_feature_cols]
            if len(pruned) != len(feature_cols):
                logger.info(
                    f"{job_id}: split-common filter kept {len(pruned)}/{len(feature_cols)} "
                    f"feature cols"
                )
            feature_cols = pruned
            if not feature_cols:
                return (job_id, "NO COMMON FEATURE COLUMNS ACROSS SPLIT")
        # For per-text feature sets the static feature_cols is empty by
        # design — resolve columns by prefix match against the per-text data
        # source actually being used for this (dataset, pool) combo.
        if is_per_text:
            if dataset == "Meco":
                if pool_enum == Pool.ALL:
                    available_cols = set(pt_l1_df.columns) & set(pt_l2_df.columns)
                else:  # SEEN
                    # Union across halves, not intersection — each half has
                    # its own paragraph-indexed column space (odd: para_0/2/4..,
                    # even: para_1/3/5..) so the intersection is essentially
                    # empty. The per-segment filter inside the iteration loop
                    # picks the cols actually present in that segment's half.
                    per_half = [
                        set(halves_l1[h].columns) & set(halves_l2[h].columns)
                        for h in halves_l1 if h in halves_l2
                    ]
                    available_cols = set.union(*per_half) if per_half else set()
            else:
                available_cols = set(resolver.l1_df.columns) & set(resolver.l2_df.columns)
            feature_cols = sorted(c for c in available_cols if c.startswith(fixed_text_prefix))
            if not feature_cols:
                return (job_id, "NO PER-TEXT COLUMNS")

        per_segment = []
        for segment in resolver.iter_segments(pool_enum):
            if (target_col not in segment.train_pool.columns
                    or target_col not in segment.test_pool.columns
                    or target_col not in segment.train_l1.columns):
                return (job_id, "NO TARGET COL")
            # train_l1 belongs in the intersection: fit_with_augmentation vstacks
            # the L1 rows into the training matrix, so a column present in L2 but
            # absent from L1 is a KeyError, not a usable feature. This bites only
            # when a cohort is small enough for a tag x measure to empty out —
            # MECO L1 is 95 readers and split-half narrows the texts further, so
            # e.g. WRB_ptb_pos_RR_no_norm is missing from four L1 half-files.
            available = (set(segment.train_pool.columns)
                         & set(segment.test_pool.columns)
                         & set(segment.train_l1.columns))
            seg_feature_cols = [c for c in feature_cols if c in available]
            _warn_l1_dropped_cols(
                feature_cols, available, segment.train_pool,
                f"{job_id} seg={segment.segment_id}",
            )
            if not seg_feature_cols:
                continue
            if is_per_text:
                # Narrow each frame ONCE to this job's per-text columns plus all
                # non-per-text (scalar/metadata) columns. The frames carry the
                # full per-text union (~670k cols for OneStop, ~40k for MECO
                # halves) and every row operation downstream — the dropna, the
                # per-fold iloc in the LOPO loop, the L1 masking — copies the
                # full width (multi-GB per refit × hundreds of refits per job).
                # Dropping the other segments'/feature-sets' per-text columns,
                # which this job can never read, makes those copies tens of MB.
                # The fold-level l2_live_mask feature selection still runs
                # unchanged on the surviving columns.
                keep = set(seg_feature_cols)
                pertext_prefixes = tuple(FIXED_TEXT_GROUP_PREFIXES.values())
                def _narrow_to_job_cols(frame):
                    cols = [c for c in frame.columns
                            if c in keep
                            or not any(c.startswith(p) for p in pertext_prefixes)]
                    return frame[cols]
                segment = PoolSegment(
                    segment_id=segment.segment_id,
                    train_pool=_narrow_to_job_cols(segment.train_pool),
                    test_pool=_narrow_to_job_cols(segment.test_pool),
                    train_l1=_narrow_to_job_cols(segment.train_l1),
                )
            train_clean = model_obj.drop_missing_and_negative_ones(segment.train_pool, seg_feature_cols, target_col)
            test_clean = model_obj.drop_missing_and_negative_ones(segment.test_pool, seg_feature_cols, target_col)
            if train_clean.empty or test_clean.empty:
                continue
            # Hand each segment its own L1 copy: fit_with_augmentation runs
            # l1_imputation which mutates the target column in place. Shared
            # halves_data across threads + segments would race otherwise.
            seg_l1 = segment.train_l1.copy()

            # α is chosen per fold inside the model (RidgeCV's closed-form LOO
            # on that fold's augmented training data), so every participant
            # receives a leave-one-out prediction.

            clean_segment = PoolSegment(
                segment_id=segment.segment_id,
                train_pool=train_clean,
                test_pool=test_clean,
                train_l1=seg_l1,
            )
            pred = model_obj.run_pool_fold_segment(
                clean_segment,
                df_l1=seg_l1,
                pool=pool_enum,
                fold_method=fold_method_enum,
                feature_cols=seg_feature_cols,
                target_col=target_col,
                aug_precentage=aug_percentage,
                narrow_l1=narrow_l1,
                inner_validation=inner_validation,
            )
            if pred is not None and not pred.empty:
                per_segment.append(pred)

        if not per_segment:
            return (job_id, "NO PREDICTIONS")

        combined = _combine_segment_predictions(per_segment, resolver.combine_strategy)
        if combined is None or combined.empty:
            return (job_id, "EMPTY COMBINED")
        # Stamped after the combine, not inside the per-fold row dicts: the MECO
        # branch of _combine_segment_predictions aggregates with a hardcoded agg
        # dict that would silently drop any column not listed there.
        combined["inner_validation"] = inner_validation.value
        combined.to_csv(save_path, index=False)
        return (job_id, "DONE")
    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        origin = tb[-1]
        msg = str(e)[:120].replace("\n", " ")
        logger.error(f"Error in {job_id}: {origin.filename}:{origin.lineno} in {origin.name} - {msg}")
        return (job_id, f"ERROR: {origin.filename.split('/')[-1]}:{origin.lineno} {msg}")


def _is_valid_pool_fold_combo(dataset: str, pool: Pool, fold_method: FoldMethod) -> bool:
    """Gate cells that don't make sense.

    - UNSEEN × L1_STRAT_KFOLD is skipped: the K-fold exclusion is at the
      participant level, and under UNSEEN the test participants are disjoint
      from train, so the exclusion is a no-op — predictions would be identical
      to UNSEEN × POOL_ALL.
    - UNSEEN × LOLO IS kept: it excludes by L1, which still removes same-L1
      participants from the other-context train pool.
    - SEEN × L1_STRAT_KFOLD on OneStop runs despite tiny within-CG strata
      (user: "run it and we'll most likely just not use it").
    """
    if pool == Pool.UNSEEN and fold_method == FoldMethod.L1_STRAT_KFOLD:
        return False
    return True


def _filter_halves_by_n_paragraphs(halves_data, pt_halves_data, threshold):
    """Per-half independent threshold filter.

    For each half (odd/even) keep participants whose `n_paragraphs >= threshold`
    on THAT half. A participant who clears the threshold on only one half
    contributes predictions from that half only; the downstream
    `_combine_segment_predictions(strategy="average")` handles the asymmetry.
    Applies to halves_data (scalar) and pt_halves_data (per-text) together,
    so both stay in sync by participant.
    """
    if threshold is None or halves_data is None:
        return halves_data, pt_halves_data
    new_halves: dict = {}
    new_pt: dict | None = {} if pt_halves_data else None
    for half, hd in halves_data.items():
        l2 = hd["l2"]
        keep_ids = set(l2[l2["n_paragraphs"] >= threshold][Fields.SUBJECT_ID])
        new_halves[half] = {
            "l1": hd["l1"],
            "l2": l2[l2[Fields.SUBJECT_ID].isin(keep_ids)],
        }
        if new_pt is not None and half in pt_halves_data:
            pt = pt_halves_data[half]
            new_pt[half] = {
                "l1": pt["l1"],
                "l2": pt["l2"][pt["l2"][Fields.SUBJECT_ID].isin(keep_ids)],
            }
    return new_halves, new_pt


def run_pool_fold_predictions_parallel(
    datasets: list[str] | None = None,
    previews: list[str] | None = None,
    models: list[str] | None = None,
    pools: list[str] | None = None,
    fold_methods: list[str] | None = None,
    n_splits: int = 5,
    aug_percentage: float = 1.0,
    feature_sets: list[str] | None = None,
    target_cols: list[str] | None = None,
    target_set_size: str = "small",
    feature_set_size: str = "small",
    replace_existing: bool = False,
    n_jobs: int = -1,
    executor_kind: str = 'processes',
    skip_fixed_features: bool = False,
    n_paragraphs_thresholds: list[int] | None = None,
    require_l1_lextale: bool = False,
    inner_validation: str = InnerValidation.NONE,
) -> None:
    """Drive the Pool × FoldMethod orchestration.

    Iterates (dataset, preview, model, pool, fold_method, target, feature_set)
    over fully_agg only. Per-text features are auto-gated to pool=SEEN.
    """
    if datasets is None:
        datasets = list(DATASET_PREVIEWS.keys())
    if models is None:
        models = list(MODELS.keys())
    if pools is None:
        pools = [Pool.ALL.value, Pool.SEEN.value, Pool.UNSEEN.value]
    if fold_methods is None:
        fold_methods = [FoldMethod.POOL_ALL.value, FoldMethod.LOLO.value, FoldMethod.L1_STRAT_KFOLD.value]

    for p in pools:
        Pool(p)  # validate
    for fm in fold_methods:
        FoldMethod(fm)

    if feature_sets is None:
        if feature_set_size == "small":
            feature_sets_dict = {k: v for k, v in ALL_FEATURE_SETS_DICT.items() if k in SMALL_FEATURE_SETS}
        else:
            feature_sets_dict = ALL_FEATURE_SETS_DICT
    else:
        feature_sets_dict = {k: v for k, v in ALL_FEATURE_SETS_DICT.items() if k in feature_sets}
    if skip_fixed_features:
        feature_sets_dict = {k: v for k, v in feature_sets_dict.items() if k not in FIXED_TEXT_GROUP_PREFIXES}

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    logger.info(f"\n{'='*60}")
    logger.info(f"Pool × FoldMethod predictions: {n_jobs} workers")
    logger.info(f"  Datasets: {datasets}  Pools: {pools}  Fold methods: {fold_methods}")
    logger.info(f"  Models: {models}  Feature sets: {len(feature_sets_dict)}")
    logger.info(f"{'='*60}\n")

    needs_per_text = any(fs in FIXED_TEXT_GROUP_PREFIXES for fs in feature_sets_dict)

    results = []
    total_jobs = 0
    with _JobRunner(n_jobs, executor_kind) as executor:
        for dataset in datasets:
            dataset_previews = ["all"] if dataset == "Meco" else (previews or DATASET_PREVIEWS.get(dataset, ("All",)))
            for preview in dataset_previews:
                # Load fully_agg L1/L2 features
                l1_df, l2_df = _load_pool_fold_data(dataset, preview, require_l1_lextale=require_l1_lextale)
                if l1_df is None or l2_df is None:
                    logger.warning(f"Skipping {dataset}/{preview}: fully_agg data missing")
                    continue

                # Load per-text features (if needed)
                pt_l1_df, pt_l2_df = _load_pool_fold_per_text(dataset, preview, needs_per_text, require_l1_lextale=require_l1_lextale)

                # Load MECO halves (if needed)
                halves_data, pt_halves_data = _load_pool_fold_halves(dataset, pools, needs_per_text=needs_per_text, require_l1_lextale=require_l1_lextale)

                dataset_key = dataset + "L2"
                ds_target_cols = get_target_cols(dataset_key, target_set_size, target_cols)

                # Per-half n_paragraphs threshold sweep. Only applies to MECO
                # SEEN/UNSEEN cells (n_paragraphs column lives on the halves
                # frames). Baseline threshold None gives version_suffix="" =
                # plain fully_agg path; explicit thresholds add
                # /n_paragraphs_geq_{N} suffix and filter halves data per N.
                meco_thresholds = (list(n_paragraphs_thresholds)
                                   if (dataset == "Meco" and n_paragraphs_thresholds)
                                   else [])
                threshold_variants = [(None, "", halves_data, pt_halves_data)]
                for thr in meco_thresholds:
                    th_h, th_pt = _filter_halves_by_n_paragraphs(halves_data, pt_halves_data, thr)
                    threshold_variants.append((thr, f"/n_paragraphs_geq_{thr}", th_h, th_pt))

                combo_jobs = []
                for model, pool_val, fm_val in itertools.product(models, pools, fold_methods):
                    if not _is_valid_pool_fold_combo(dataset, Pool(pool_val), FoldMethod(fm_val)):
                        continue
                    if dataset == "Meco" and pool_val in {Pool.SEEN.value, Pool.UNSEEN.value} and halves_data is None:
                        continue
                    is_meco_halves_cell = (
                        dataset == "Meco" and pool_val in {Pool.SEEN.value, Pool.UNSEEN.value}
                    )
                    # Only threshold-replicate MECO seen/unseen cells; everything
                    # else uses the baseline (None) variant once.
                    cell_variants = threshold_variants if is_meco_halves_cell else threshold_variants[:1]
                    for thr, version_suffix, halves_for_job, pt_halves_for_job in cell_variants:
                        for target_col in ds_target_cols:
                            for feature_set_name, feature_cols in feature_sets_dict.items():
                                combo_jobs.append((
                                    dataset, preview, model, pool_val, fm_val,
                                    target_col, feature_set_name, feature_cols,
                                    l1_df, l2_df, halves_for_job,
                                    n_splits, aug_percentage, replace_existing,
                                    pt_l1_df, pt_l2_df, pt_halves_for_job,
                                    version_suffix, require_l1_lextale,
                                    inner_validation,
                                ))

                if not combo_jobs:
                    continue
                batch = executor.run_batch(
                    _run_pool_fold_prediction_job, combo_jobs, desc=f"{dataset}/{preview}",
                )
                results.extend(batch)
                total_jobs += len(batch)

    done = sum(1 for _, s in results if s == "DONE")
    skipped = sum(1 for _, s in results if "SKIP" in s)
    errors = sum(1 for _, s in results if "ERROR" in s)
    logger.info(f"\nPool × FoldMethod summary: done={done} skipped={skipped} errors={errors} (total {total_jobs})")
    if errors:
        for job_id, status in results:
            if "ERROR" in status:
                logger.error(f"  {job_id}: {status}")


# Legacy FoldsMethods → canonical Pool×FoldMethod filename stem. The legacy
# path is ALL pool only (BCV gone), so the pool prefix is always "all".
_LEGACY_METHOD_TO_FILENAME = {
    FoldsMethods.LEAVE_ONE_PARTICIPANT_OUT: "all__pool_all",
    FoldsMethods.LEAVE_ONE_LANGUAGE_OUT:    "all__lolo",
    FoldsMethods.RANDOM:                    "all__l1_strat_kfold",
}


def _log_predictions_start(mode_name, n_jobs, datasets, previews, models, methods, agg_types, feature_set_count, extra_info=""):
    """Log prediction pipeline startup info."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{mode_name}: {n_jobs} workers{extra_info}")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Previews: {previews}")
    logger.info(f"  Models: {models}")
    logger.info(f"  Methods: {methods}")
    logger.info(f"  Agg types: {agg_types if agg_types else 'all'}")
    logger.info(f"  Feature sets: {feature_set_count}")
    logger.info(f"{'='*60}\n")

    print(f"\n>>> [{datetime.now().strftime('%H:%M:%S')}] Starting {mode_name.lower()} with {n_jobs} workers")
    print(f">>> Monitor with: ps aux | grep run_one_feature_set | grep -v grep | wc -l\n")


# --------------------------------------------------------------------------
# Job execution: threads (historical) or processes.
#
# Ridge fitting only partly releases the GIL -- enough of each fold is Python
# (sklearn construction/validation, numpy slicing, scaler prep) that the thread
# pool plateaus around 2.8x no matter how many workers it is given, and slowly
# degrades past 8. Measured on MecoL2 fully_agg x the 289-feature set, with
# BLAS pinned to one thread as this module already does at import:
#
#   workers      threads     processes
#         4        2.42x         2.36x
#         8        2.91x         4.32x
#        16        2.84x         9.18x
#        32        2.68x         9.94x
#
# Processes only stay ahead if the per-combo frames are not pickled once per
# job. The per-text feature frames run to hundreds of thousands of columns, so
# shipping them through the submit queue would cost more than the fits. Instead
# they go into _SHARED before the pool is forked and reach workers by
# copy-on-write; jobs carry a _SharedRef in their place.
#
# That is why process mode forks a fresh pool per batch instead of reusing one:
# a batch's data is loaded inside the orchestration loop, after any earlier
# pool would already have forked. This costs nothing in scheduling terms --
# every call site already drains its futures before loading the next combo.
# --------------------------------------------------------------------------

_SHARED: dict[int, object] = {}


class _SharedRef:
    """Stands in for a large job argument passed by fork COW, not by pickle."""

    __slots__ = ("key",)

    def __init__(self, key: int):
        self.key = key


def _share(obj):
    """Register `obj` for COW sharing and return its placeholder."""
    _SHARED[id(obj)] = obj  # the dict keeps it alive, so id() stays valid
    return _SharedRef(id(obj))


def _prepare_job(job: tuple) -> tuple:
    """Swap the bulky frames in a job tuple for placeholders."""
    return tuple(
        _share(a) if isinstance(a, (pd.DataFrame, dict)) else a
        for a in job
    )


def _worker_init() -> None:
    """Per-worker setup. Workers inherit the module-level BLAS pinning at the
    top of this file through fork, so there is nothing to do about threads here
    — and nothing that inspects loaded libraries belongs in this function.
    threadpoolctl.threadpool_limits() looks like the obvious way to pin BLAS
    per worker, but it walks /proc/self/maps and dlopens what it finds, and
    several children doing that at once right after fork aborts the process
    (SIGABRT inside _load_libraries, with 232 extension modules loaded).

    What does need doing: drop the inherited loguru file sink, so workers do
    not all rotate the same log concurrently. Warnings still reach stderr, and
    job outcomes come back through the returned status either way.
    """
    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def _worker_entry(job_func, job: tuple):
    """Resolve placeholders back to the shared frames, then run the job."""
    return job_func(*(_SHARED[a.key] if isinstance(a, _SharedRef) else a for a in job))


class _JobRunner:
    """Runs batches of prediction jobs on threads or processes.

    Thread mode keeps one pool for the whole run, exactly as before. Process
    mode forks a fresh pool per batch so each batch's frames reach workers by
    copy-on-write. Job functions and their signatures are identical either way.
    """

    def __init__(self, n_jobs: int, kind: str = "threads"):
        if kind not in EXECUTOR_KINDS:
            raise ValueError(f"Unknown executor kind {kind!r}; expected one of {EXECUTOR_KINDS}")
        self.n_jobs = n_jobs
        self.kind = kind
        self._threads = None

    def __enter__(self):
        if self.kind == "threads":
            self._threads = ThreadPoolExecutor(max_workers=self.n_jobs)
            self._threads.__enter__()
        return self

    def __exit__(self, *exc_info):
        if self._threads is not None:
            return self._threads.__exit__(*exc_info)
        return False

    def run_batch(self, job_func, combo_jobs, desc: str, leave: bool = True) -> list:
        """Run one batch to completion; returns [(job_id, status), ...]."""
        if not combo_jobs:
            return []

        if self.kind == "threads":
            pool = self._threads
            jobs = combo_jobs
            submit = lambda job: pool.submit(job_func, *job)  # noqa: E731
            closing = None
        else:
            jobs = [_prepare_job(job) for job in combo_jobs]
            # fork (not spawn): workers must inherit _SHARED, and re-importing
            # this module in a spawned child would reload every dependency.
            closing = pool = ProcessPoolExecutor(
                max_workers=self.n_jobs,
                mp_context=multiprocessing.get_context("fork"),
                initializer=_worker_init,
            )
            submit = lambda job: pool.submit(_worker_entry, job_func, job)  # noqa: E731

        out = []
        try:
            futures = [submit(job) for job in jobs]
            with tqdm(total=len(jobs), desc=desc, unit="job", leave=leave) as pbar:
                for future in as_completed(futures):
                    job_id, status = future.result()
                    out.append((job_id, status))
                    icon = "✓" if status == "DONE" else "⊘" if "SKIP" in status else "✗" if "ERROR" in status else "•"
                    pbar.update(1)
                    pbar.set_postfix_str(f"{icon} {job_id.split('/')[-1]}")
        finally:
            if closing is not None:
                closing.shutdown(wait=True)
            _SHARED.clear()
        return out


def _submit_pool_fold_batch(
    executor, dataset, preview, models, pools, fold_methods,
    target_cols, target_set_size, feature_sets_dict,
    l1_df, l2_df, halves_data, pt_l1_df, pt_l2_df, pt_halves_data,
    aug_percentage, replace_existing, version_suffix, results
):
    """Submit a batch of pool-fold prediction jobs to executor.

    Shared helper used by both regular and split-half pool-fold orchestrations.
    """
    dataset_key = dataset + "L2"
    ds_target_cols = get_target_cols(dataset_key, target_set_size, target_cols)

    combo_jobs = []
    for model, pool_val, fm_val in itertools.product(models, pools, fold_methods):
        if not _is_valid_pool_fold_combo(dataset, Pool(pool_val), FoldMethod(fm_val)):
            continue
        if dataset == "Meco" and pool_val in {Pool.SEEN.value, Pool.UNSEEN.value} and halves_data is None:
            continue
        for target_col in ds_target_cols:
            for feature_set_name, feature_cols in feature_sets_dict.items():
                combo_jobs.append((
                    dataset, preview, model, pool_val, fm_val,
                    target_col, feature_set_name, feature_cols,
                    l1_df, l2_df, halves_data,
                    5, aug_percentage, replace_existing,
                    pt_l1_df, pt_l2_df, pt_halves_data,
                    version_suffix,
                ))

    if not combo_jobs:
        return 0

    results.extend(executor.run_batch(
        _run_pool_fold_prediction_job, combo_jobs,
        desc=f"{dataset}/{preview}{version_suffix}", leave=False,
    ))

    return len(combo_jobs)


def _log_predictions_summary(mode_name, total_jobs, done, skipped, errors, extra_info=""):
    """Log prediction pipeline completion summary."""
    other = total_jobs - done - skipped - errors

    logger.info(f"\n{'='*60}")
    logger.info(f"{mode_name} Summary:{extra_info}")
    logger.info(f"  ✓ Done: {done}")
    logger.info(f"  ⊘ Skipped (exist): {skipped}")
    logger.info(f"  ✗ Errors: {errors}")
    logger.info(f"  - Other: {other}")
    logger.info(f"{'='*60}\n")

    return errors > 0


def _setup_predictions_common(
    datasets,
    models,
    methods,
    previews,
    feature_sets,
    feature_set_size,
    n_jobs,
    skip_fixed_features,
):
    """Common setup for all prediction orchestration functions.

    Returns:
        (datasets, models, methods, previews, feature_sets_dict, n_jobs)
    """
    # Set defaults
    if datasets is None:
        datasets = list(DATASET_PREVIEWS.keys())
    if models is None:
        models = list(MODELS.keys())
    if methods is None:
        methods = list(DEFAULT_METHODS)

    # Validate inputs
    for model in models:
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}")

    for method in methods:
        try:
            m_enum = FoldsMethods(method)
        except ValueError:
            available = [m.value for m in FoldsMethods if m != FoldsMethods.BATCH_CROSS_VALIDATION]
            raise ValueError(f"Unknown method: {method}. Available: {available}")
        if m_enum == FoldsMethods.BATCH_CROSS_VALIDATION:
            raise ValueError(
                "BATCH_CROSS_VALIDATION is no longer supported on the partial-agg path. "
                "Use --mode pool_fold for SEEN / UNSEEN cells."
            )

    # Prepare feature sets dict
    if feature_sets is None:
        if feature_set_size == 'small':
            feature_sets_dict = {k: v for k, v in ALL_FEATURE_SETS_DICT.items() if k in SMALL_FEATURE_SETS}
        else:
            feature_sets_dict = ALL_FEATURE_SETS_DICT
    else:
        feature_sets_dict = {k: v for k, v in ALL_FEATURE_SETS_DICT.items() if k in feature_sets}

    feature_sets_dict = {k: v for k, v in feature_sets_dict.items() if k not in FIXED_TEXT_GROUP_PREFIXES}
    if skip_fixed_features:
        pass

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    return datasets, models, methods, previews, feature_sets_dict, n_jobs


def _run_single_prediction_job(l1_df, l2_df, agg_label,
                               pt_l1_df, pt_l2_df, dataset,
                               preview, model, method, target_col,
                               feature_set_name, feature_cols,
                               agg_type, p_agg_level, p_agg,
                               n_splits,
                               aug_percentage, replace_existing,
                               df_l2_full=None, train_on_full=False, version_suffix="",
                               require_l1_lextale=False) -> tuple[str, str]:
    """Run predictions for a single (dataset, preview, model, method, target,
    feature_set, agg_type) combo on the partial-agg path (first_p / moving_p).

    The legacy ALL-pool path only — BCV and MECO seen/unseen are handled by
    the new Pool × FoldMethod orchestration (run_pool_fold_predictions_parallel)
    which owns the fully_agg cells. Output CSVs use the canonical
    {pool}__{fold_method}.csv naming (pool=all always for this path).
    """
    try:
        method_enum = FoldsMethods(method)
        if method_enum == FoldsMethods.BATCH_CROSS_VALIDATION:
            # BCV is dead in the legacy path; the new orchestration handles
            # any seen/unseen contrast. Reject early to make the failure
            # obvious if someone tries to revive an old --methods flag value.
            return (f"{dataset}/{preview}/{model}/{method}/{target_col}/{feature_set_name}/{agg_label}",
                    "SKIPPED (BCV removed; use --mode pool_fold)")
        method_filename = _LEGACY_METHOD_TO_FILENAME[method_enum]

        # Organize into subfolders by agg_type and p_agg level
        if agg_type == AggTypes.FULL:
            agg_folder = AggTypes.FULL
        elif agg_type == AggTypes.FIRST_P:
            agg_folder = f"first_p_agg/{p_agg_level}_{p_agg}"
        elif agg_type == AggTypes.MOVING_P:
            agg_folder = f"moving_p_agg/{p_agg_level}_{p_agg}"
        elif agg_type == AggTypes.PER_ITEM:
            agg_folder = f"per_item_agg/{p_agg_level}"
        else:
            agg_folder = agg_label

        # For FULL agg_type, use flat structure; otherwise separate by train_on_full mode
        if agg_type == AggTypes.FULL:
            save_path = _results_root(require_l1_lextale) / f"{dataset}/aug_p={aug_percentage}/{preview.lower()}{version_suffix}/{agg_folder}/{target_col}/{feature_set_name}/{model}/{method_filename}.csv"
        else:
            mode_folder = "train_on_full" if train_on_full else "partial_train"
            save_path = _results_root(require_l1_lextale) / f"{dataset}/aug_p={aug_percentage}/{preview.lower()}{version_suffix}/{agg_folder}/{target_col}/{feature_set_name}/{model}/{mode_folder}/{method_filename}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists and not replacing
        if save_path.exists() and not replace_existing:
            return (f"{dataset}/{preview}/{model}/{method}/{target_col}/{feature_set_name}/{agg_label}", "SKIPPED")

        # Per-text feature sets need Pool.SEEN, which the legacy path can't
        # provide (it's ALL only). They're handled by the new orchestration.
        if feature_set_name in FIXED_TEXT_GROUP_PREFIXES:
            return (f"{dataset}/{preview}/{model}/{method}/{target_col}/{feature_set_name}/{agg_label}",
                    "SKIPPED PER-TEXT (needs Pool.SEEN; use --mode pool_fold)")

        # Check if target column exists
        if target_col not in l1_df.columns or target_col not in l2_df.columns:
            return (f"{dataset}/{preview}/{model}/{method}/{target_col}/{feature_set_name}/{agg_label}", "NO TARGET COL")

        pred_df = run_one_feature_set(
            l1_df,
            l2_df,
            feature_set_name,
            feature_cols,
            target_col=target_col,
            model_name=model,
            method=method_enum,
            n_splits=n_splits,
            aug_precentage=aug_percentage,
            save_path=save_path,
            agg_type=agg_type,
            df_l2_full=df_l2_full,
            train_on_full=train_on_full,
            replace_existing=replace_existing,
        )

        return (f"{dataset}/{preview}/{model}/{method}/{target_col}/{feature_set_name}/{agg_label}", "DONE")

    except Exception as e:

        error_msg = str(e)[:80].replace('\n', ' ')
        tb_lines = traceback.extract_tb(sys.exc_info()[2])
        origin = tb_lines[-1]  # Get the innermost traceback
        logger.error(f"Error in {dataset}/{preview}/{model}/{method}/{target_col}/{feature_set_name}/{agg_label}: {origin.filename}:{origin.lineno} in {origin.name} - {error_msg}")
        return (f"{dataset}/{preview}/{model}/{method}/{target_col}/{feature_set_name}/{agg_label}", f"ERROR: {origin.filename.split('/')[-1]}:{origin.lineno} {error_msg}")


def run_predictions_parallel(
    datasets: list[str] = None,
    previews: list[str] = None,
    models: list[str] = None,
    methods: list[str] = None,
    n_splits: int = 5,
    aug_percentage: float = 1.0,
    feature_sets: list[str] = None,
    target_cols: list[str] = None,
    target_set_size: str = 'small',
    feature_set_size: str = 'small',
    agg_types: set[str] = None,
    replace_existing: bool = False,
    n_jobs: int = -1,
    executor_kind: str = 'processes',
    train_on_full_mode: str = "no",
    skip_fixed_features: bool = False,
    require_l1_lextale: bool = False,
) -> None:
    """Run partial-agg predictions in parallel (first_p_agg / moving_p_agg).

    Phase 3 narrowed this path to the ALL pool only — BCV and MECO
    seen/unseen are handled by run_pool_fold_predictions_parallel (which
    owns fully_agg). Methods here are {LOPO, LOLO, RANDOM}; CSVs are
    written under the canonical {pool}__{fold_method}.csv naming.
    """
    # Common setup for all prediction paths
    datasets, models, methods, previews, feature_sets_dict, n_jobs = _setup_predictions_common(
        datasets, models, methods, previews, feature_sets, feature_set_size, n_jobs, skip_fixed_features
    )

    # Validate inputs specific to this path
    if train_on_full_mode not in ("no", "yes", "both"):
        raise ValueError(f"train_on_full_mode must be 'no', 'yes', or 'both', got '{train_on_full_mode}'")

    # Determine which modes to run
    train_on_full_modes = []
    if train_on_full_mode in ("no", "both"):
        train_on_full_modes.append(False)
    if train_on_full_mode in ("yes", "both"):
        train_on_full_modes.append(True)

    _log_predictions_start(
        "Partial-agg predictions pipeline",
        n_jobs,
        datasets,
        previews,
        models,
        methods,
        agg_types,
        len(feature_sets_dict),
        extra_info="\n  Parallelization: agg_type × dataset × preview × model × method × target × feature_set"
    )

    # Hierarchical job creation: load data per (dataset, agg_type, preview), then batch jobs
    total_jobs = 0
    results = []

    for train_on_full in train_on_full_modes:
        mode_label = "train_on_full" if train_on_full else "standard"
        logger.info(f"\n{'='*60}")
        logger.info(f"Running predictions with mode: {mode_label}")
        logger.info(f"{'='*60}\n")

        with _JobRunner(n_jobs, executor_kind) as executor:
            for dataset in datasets:
                # Get agg configs for this dataset, filtered by agg_types if specified
                dataset_agg_configs = get_agg_configs(dataset, agg_types)

                # Per-dataset previews: MECO always uses "all", OneStop uses input or defaults
                if dataset == "Meco":
                    dataset_previews = ["all"]
                else:
                    dataset_previews = previews if previews is not None else DATASET_PREVIEWS.get(dataset, ("All",))

                for preview in dataset_previews:
                    # Load full dfs ONCE per (dataset, preview) if train_on_full
                    df_l1_full, df_l2_full = None, None
                    if train_on_full:
                        df_l1_full, df_l2_full, _ = load_and_filter_data(dataset, AggTypes.FULL, None, None, preview)
                        if require_l1_lextale:
                            df_l1_full = _filter_l1_lextale(df_l1_full, f"{dataset}/{preview}/full")

                    for agg_config in dataset_agg_configs:
                        agg_type = agg_config["agg_type"]
                        p_agg_level = agg_config.get("p_agg_level")
                        p_agg = agg_config.get("p_agg")

                        # Load/use L1 based on train_on_full flag
                        if train_on_full and agg_type != AggTypes.FULL:
                            l1_df = df_l1_full
                            # Still need to load L2 for testing
                            _, l2_df, agg_label = load_and_filter_data(dataset, agg_type, p_agg_level, p_agg, preview)
                        else:
                            # Load both L1 and L2 in a single call
                            l1_df, l2_df, agg_label = load_and_filter_data(dataset, agg_type, p_agg_level, p_agg, preview)
                            if require_l1_lextale:
                                l1_df = _filter_l1_lextale(l1_df, f"{dataset}/{preview}/{agg_label}")

                        if l1_df is None or l2_df is None:
                            logger.warning(f"Skipping {dataset}/{agg_label}/{preview}: data not found")
                            continue

                        # Create jobs for this data (model × method × target × feature_set ± batch_framework)
                        combo_jobs = []
                        dataset_key = dataset + "L2"
                        # Determine target_cols for this dataset (use parameter if specified, otherwise use dataset defaults)
                        dataset_target_cols = get_target_cols(dataset_key, target_set_size, target_cols)

                        for model, method in itertools.product(models, methods):
                            method_str = method.value if hasattr(method, 'value') else method
                            for target_col in dataset_target_cols:
                                for feature_set_name, feature_cols in feature_sets_dict.items():
                                    combo_jobs.append((
                                        l1_df, l2_df, agg_label, None, None, dataset,
                                        preview, model, method_str, target_col, feature_set_name, feature_cols,
                                        agg_type, p_agg_level, p_agg, n_splits, aug_percentage, replace_existing,
                                        df_l2_full, train_on_full, "", require_l1_lextale,
                                    ))

                        # Submit batch for this (dataset, agg_type, preview) to executor
                        if combo_jobs:
                            batch = executor.run_batch(
                                _run_single_prediction_job, combo_jobs,
                                desc=f"{dataset}/{agg_label}/{preview}",
                            )
                            results.extend(batch)
                            total_jobs += len(batch)

    # Summary
    done = sum(1 for _, s in results if s == "DONE")
    skipped = sum(1 for _, s in results if s == "SKIPPED")
    errors = sum(1 for _, s in results if "ERROR" in s)

    has_errors = _log_predictions_summary("Predictions", total_jobs, done, skipped, errors)

    if has_errors:
        logger.error("Some predictions failed:")
        for job_id, status in results:
            if "ERROR" in status:
                logger.error(f"  {job_id}: {status}")


# ============================================================
# Per-item predictions orchestration
# ============================================================

def _run_per_item_prediction_job(
    dataset: str,
    preview: str,
    model: str,
    pool: str,
    target_col: str,
    feature_set_name: str,
    feature_cols: list,
    l1_df: pd.DataFrame | None,
    l2_df: pd.DataFrame | None,
    halves_data: dict | None,
    aug_percentage: float,
    replace_existing: bool,
    item_level: str = "paragraph",
    require_l1_lextale: bool = False,
) -> tuple[str, str]:
    """Run one (pool, target, feature_set, model) per-item prediction cell.

    Output path:
        src/methods/predictions/results/{dataset}/aug_p={aug_p}/{preview}/
        per_item_agg/{p_agg_level}/{target_col}/{feature_set}/{model}/
        {pool}__pool_all_items.csv   (per-item predictions, skipped by eval)
        {pool}__pool_all.csv         (aggregated to participant-level)
    """
    pool_enum = Pool(pool)
    job_id = f"{dataset}/{preview}/{model}/{target_col}/{feature_set_name}/{pool}/{item_level}"

    try:
        logger.info(f"Job: {job_id} | feature_set={feature_set_name} | pool={pool_enum} | item_level={item_level}")

        # Per-text features are only valid for SEEN
        if feature_set_name in FIXED_TEXT_GROUP_PREFIXES and pool_enum != Pool.SEEN:
            return (job_id, "SKIPPED PER-TEXT (UNSEEN/ALL not valid for per-text)")

        # Construct save paths
        save_path_items = (
            _results_root(require_l1_lextale)
            / f"{dataset}/aug_p={aug_percentage}/{preview.lower()}/per_item_agg"
            / item_level
            / f"{target_col}/{feature_set_name}/{model}/{pool}__pool_all_items.csv"
        )
        save_path_agg = save_path_items.with_name(f"{pool}__pool_all.csv")
        save_path_items.parent.mkdir(parents=True, exist_ok=True)

        # Skip if aggregated file exists and not replacing
        if save_path_agg.exists() and not replace_existing:
            return (job_id, "SKIPPED")

        if l1_df is None or l2_df is None:
            return (job_id, "NO DATA")

        if target_col not in l1_df.columns or target_col not in l2_df.columns:
            return (job_id, "NO TARGET COL")

        # For fixed text features, use minimal resolver with only segment structure columns
        is_fixed_text = feature_set_name in FIXED_TEXT_GROUP_PREFIXES
        if is_fixed_text:
            # Keep only columns needed for segment iteration and cross-validation
            minimal_cols_l2 = [col for col in [Fields.UNIQUE_PARAGRAPH_ID, Fields.ARTICLE_ID, Fields.CONTENT_GROUP, Fields.SUBJECT_ID, Fields.BATCH, Fields.L1, target_col] if col in l2_df.columns]
            l2_minimal = l2_df[minimal_cols_l2].copy()
            minimal_cols_l1 = [col for col in [Fields.UNIQUE_PARAGRAPH_ID, Fields.ARTICLE_ID, Fields.CONTENT_GROUP, Fields.SUBJECT_ID, Fields.BATCH, Fields.L1, target_col] if col in l1_df.columns]
            l1_minimal = l1_df[minimal_cols_l1].copy()
            resolver_l1, resolver_l2 = l1_minimal, l2_minimal
        else:
            resolver_l1, resolver_l2 = l1_df, l2_df

        # Select resolver
        if dataset == "Meco":
            resolver = PerItemMecoResolver(
                l1_df=resolver_l1,
                l2_df=resolver_l2,
                halves_l1={h: halves_data[h]["l1"] for h in halves_data} if halves_data else {},
                halves_l2={h: halves_data[h]["l2"] for h in halves_data} if halves_data else {},
            )
        else:
            resolver = PerItemOneStopResolver(l1_df=resolver_l1, l2_df=resolver_l2)

        feature_cols = list(feature_cols)
        per_segment = []

        # Determine item_col based on item_level
        if item_level == "article":
            item_col = Fields.ARTICLE_ID
        else:
            item_col = Fields.UNIQUE_PARAGRAPH_ID

        # Set up per-text parquet directories for fixed text features (once, before loop)
        l1_parquet_dir = None
        l2_parquet_dir = None
        if is_fixed_text and pool_enum == Pool.SEEN:
            reread_level = "all" if dataset == "Meco" else "ordinary"
            l1_parquet_dir = (
                Path(DATA_PATH) / f"{dataset}L1/features_and_targets"
                / f"per_item_agg/{reread_level}/all/{item_level}"
            )
            l2_parquet_dir = (
                Path(DATA_PATH) / f"{dataset}L2/features_and_targets"
                / f"per_item_agg/{reread_level}/all/{item_level}"
            )
            logger.info(f"Using per-text parquets for {feature_set_name}: L1={l1_parquet_dir.exists()}, L2={l2_parquet_dir.exists()}")

        # Iterate segments from resolver
        for segment in resolver.iter_segments(pool_enum, item_col=item_col):
            # Same reasoning as the pool-fold path: the L1 augmentation frame is
            # concatenated into the training matrix, so a column it lacks is a
            # KeyError rather than a feature. No current per-item frame trips
            # this (checked for MECO paragraph and OneStop paragraph/article) —
            # it is here so a future narrower cohort fails cleanly instead.
            available = set(l2_df.columns) & set(feature_cols) & set(segment.train_l1.columns)
            seg_feature_cols = [c for c in feature_cols if c in available]
            # For fixed text features, feature_cols is empty—they're resolved from parquets at runtime
            if not seg_feature_cols and not is_fixed_text:
                logger.debug(f"Skipping segment {segment.segment_id}: no feature columns available")
                continue

            # Skip per-text features (TRANSITIONS/WFC) unless Pool.SEEN
            # They're only valid for Pool.SEEN because participants must read the same texts
            if is_fixed_text and segment.pool != Pool.SEEN:
                logger.debug(f"Skipping segment {segment.segment_id}: per-text features not allowed for {segment.pool} regime")
                continue

            logger.debug(f"Processing segment {segment.segment_id}: is_fixed_text={is_fixed_text}, seg_feature_cols={len(seg_feature_cols)}")

            try:
                logger.info(f"segment.pool={segment.pool} (type={type(segment.pool)}), pool_enum={pool_enum} (type={type(pool_enum)}), match={segment.pool == pool_enum}")
                if segment.pool == Pool.SEEN or segment.pool == Pool.ALL:
                    # LOPO within each item
                    seg_pred = run_per_item_seen(
                        lopo_df=segment.lopo_df,
                        l1_df=segment.train_l1,
                        feature_cols=seg_feature_cols,
                        target_col=target_col,
                        item_col=segment.item_col,
                        model_name=model,
                        aug_percentage=aug_percentage,
                        l1_parquet_dir=l1_parquet_dir,
                        l2_parquet_dir=l2_parquet_dir,
                        pool=segment.pool,
                        feature_set=feature_set_name,
                    )
                else:  # Pool.UNSEEN
                    # Train one model on all train data, predict per-item (or LOIO for MECO)
                    seg_pred = run_per_item_unseen(
                        train_df=segment.train_df,
                        test_df=segment.test_df,
                        l1_df=segment.train_l1,
                        feature_cols=seg_feature_cols,
                        target_col=target_col,
                        model_name=model,
                        aug_percentage=aug_percentage,
                        segment_id=segment.segment_id,
                    )

                if seg_pred is not None and not seg_pred.empty:
                    per_segment.append(seg_pred)
            except Exception as e:
                logger.warning(f"Segment {segment.segment_id} failed: {e}")
                continue

        if not per_segment:
            return (job_id, "NO PREDICTIONS")

        # Combine segment results
        combined = _combine_segment_predictions(per_segment, resolver.combine_strategy)
        if combined is None or combined.empty:
            return (job_id, "EMPTY COMBINED")

        # Add item column if not already present (for per-item tracking)
        if item_col not in combined.columns and item_col in per_segment[0].columns:
            combined[item_col] = per_segment[0][item_col] if len(per_segment) == 1 else None
            # If combined from multiple segments, try to reconstruct item col from index or other means
            if item_col not in combined.columns or combined[item_col].isna().all():
                # Fallback: reconstruct if possible from segment_id if available
                if 'segment_id' in per_segment[0].columns:
                    segment_map = {}
                    for seg_df in per_segment:
                        if not seg_df.empty and 'segment_id' in seg_df.columns:
                            seg_id = seg_df['segment_id'].iloc[0]
                            segment_map[seg_id] = seg_id
                    if segment_map and len(per_segment) == 1:
                        combined[item_col] = per_segment[0].get(item_col, per_segment[0].get('segment_id', None))

        # Save per-item predictions
        combined.to_csv(save_path_items, index=False)

        # Aggregate to participant level
        aggregated = aggregate_per_item_predictions(combined)
        aggregated.to_csv(save_path_agg, index=False)

        return (job_id, "DONE")

    except Exception as e:
        tb = traceback.extract_tb(sys.exc_info()[2])
        origin = tb[-1]
        msg = str(e)[:120].replace("\n", " ")
        logger.error(f"Error in {job_id}: {origin.filename}:{origin.lineno} in {origin.name} - {msg}")
        return (job_id, f"ERROR: {origin.filename.split('/')[-1]}:{origin.lineno} {msg}")


def run_per_item_predictions_parallel(
    datasets: list[str] | None = None,
    previews: list[str] | None = None,
    models: list[str] | None = None,
    pools: list[str] | None = None,
    n_splits: int = 5,
    aug_percentage: float = 1.0,
    feature_sets: list[str] | None = None,
    target_cols: list[str] | None = None,
    target_set_size: str = "small",
    feature_set_size: str = "small",
    replace_existing: bool = False,
    n_jobs: int = -1,
    executor_kind: str = 'processes',
    skip_fixed_features: bool = False,
    require_l1_lextale: bool = False,
) -> None:
    """Run per-item predictions in parallel.

    Parallelizes over (dataset, preview, pool, model, target, feature_set).
    Per-item: one row per (participant, paragraph), aggregated to participant level for evaluation.
    """
    if datasets is None:
        datasets = list(DATASET_PREVIEWS.keys())
    if models is None:
        models = list(MODELS.keys())
    if pools is None:
        pools = [Pool.ALL.value, Pool.SEEN.value, Pool.UNSEEN.value]

    for p in pools:
        Pool(p)  # validate

    if feature_sets is None:
        if feature_set_size == "small":
            feature_sets_dict = {k: v for k, v in ALL_FEATURE_SETS_DICT.items() if k in SMALL_FEATURE_SETS}
        else:
            feature_sets_dict = ALL_FEATURE_SETS_DICT
    else:
        feature_sets_dict = {k: v for k, v in ALL_FEATURE_SETS_DICT.items() if k in feature_sets}

    if skip_fixed_features:
        feature_sets_dict = {k: v for k, v in feature_sets_dict.items() if k not in FIXED_TEXT_GROUP_PREFIXES}

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    logger.info(f"\n{'='*60}")
    logger.info(f"Per-item predictions: {n_jobs} workers")
    logger.info(f"  Datasets: {datasets}  Pools: {pools}")
    logger.info(f"  Models: {models}  Feature sets: {len(feature_sets_dict)}")
    logger.info(f"{'='*60}\n")

    results = []
    total_jobs = 0

    with _JobRunner(n_jobs, executor_kind) as executor:
        for dataset in datasets:
            dataset_previews = ["all"] if dataset == "Meco" else (previews or DATASET_PREVIEWS.get(dataset, ("All",)))
            for preview in dataset_previews:
                preview_titlecase = preview.title() if preview.lower() != "all" else preview

                dataset_key = dataset + "L2"
                ds_target_cols = get_target_cols(dataset_key, target_set_size, target_cols)

                # For OneStop, run both paragraph and article level; for Meco, only paragraph
                item_levels = ["paragraph", "article"] if dataset == "OneStop" else ["paragraph"]

                for item_level in item_levels:
                    # Load per-item L1/L2 data for this item_level
                    l1_df, l2_df, agg_label = load_and_filter_data(
                        dataset, AggTypes.PER_ITEM, item_level, None, preview_titlecase
                    )
                    if l1_df is None or l2_df is None:
                        logger.warning(f"Skipping {dataset}/{preview}/{item_level}: per-item data missing")
                        continue
                    if require_l1_lextale:
                        l1_df = _filter_l1_lextale(l1_df, f"{dataset}/{preview}/{item_level}/per_item")

                    # Per-item halves data for MECO UNSEEN (only for paragraph level)
                    halves_data = None
                    if dataset == "Meco" and item_level == "paragraph" and Pool.UNSEEN.value in pools:
                        halves_data = {}
                        for half in ("odd", "even"):
                            path_l1 = (
                                Path(DATA_PATH) / f"MecoL1/features_and_targets"
                                / f"per_item_agg/all/all/paragraph/features_and_metadata.csv"
                            )
                            path_l2 = (
                                Path(DATA_PATH) / f"MecoL2/features_and_targets"
                                / f"per_item_agg/all/all/paragraph/features_and_metadata.csv"
                            )
                            if path_l1.exists() and path_l2.exists():
                                df_l1 = pd.read_csv(path_l1)
                                df_l2 = pd.read_csv(path_l2)
                                halves_data[half] = {"l1": df_l1, "l2": df_l2}
                        if require_l1_lextale:
                            halves_data = _filter_halves_l1_lextale(halves_data, f"{dataset}/per_item_halves")

                    combo_jobs = []
                    for model, pool_val, target_col in itertools.product(models, pools, ds_target_cols):
                        for feature_set_name, feature_cols in feature_sets_dict.items():
                            combo_jobs.append((
                                dataset, preview, model, pool_val, target_col,
                                feature_set_name, feature_cols,
                                l1_df, l2_df, halves_data,
                                aug_percentage, replace_existing,
                                item_level, require_l1_lextale,
                            ))

                    if not combo_jobs:
                        continue

                    batch = executor.run_batch(
                        _run_per_item_prediction_job, combo_jobs,
                        desc=f"{dataset}/{preview}/{item_level}/per-item",
                    )
                    results.extend(batch)
                    total_jobs += len(batch)

    done = sum(1 for _, s in results if s == "DONE")
    skipped = sum(1 for _, s in results if "SKIP" in s)
    errors = sum(1 for _, s in results if "ERROR" in s)
    logger.info(f"\nPer-item summary: done={done} skipped={skipped} errors={errors} (total {total_jobs})")
    if errors:
        for job_id, status in results:
            if "ERROR" in status:
                logger.error(f"  {job_id}: {status}")


# ============================================================
# Split-half pool-fold predictions orchestration
# ============================================================

def run_split_half_pool_fold_predictions_parallel(
    datasets: list[str] | None = None,
    previews: list[str] | None = None,
    models: list[str] | None = None,
    pools: list[str] | None = None,
    fold_methods: list[str] | None = None,
    n_splits: int = 20,
    aug_percentage: float = 1.0,
    feature_sets: list[str] | None = None,
    target_cols: list[str] | None = None,
    target_set_size: str = "small",
    feature_set_size: str = "small",
    replace_existing: bool = False,
    n_jobs: int = -1,
    executor_kind: str = 'processes',
    skip_fixed_features: bool = False,
    n_paragraphs_thresholds: list[int] | None = None,
    require_l1_lextale: bool = False,
    inner_validation: str = InnerValidation.NONE,
) -> None:
    """Run split-half pool-fold predictions.

    Same as run_pool_fold_predictions_parallel() but iterates over all (split_idx, half)
    combos, loading split-half features and organizing output under split_half/ subfolder.
    Reuses shared pool-fold machinery and helpers.
    """
    if datasets is None:
        datasets = list(DATASET_PREVIEWS.keys())
    if models is None:
        models = list(MODELS.keys())
    if pools is None:
        pools = [Pool.ALL.value, Pool.SEEN.value, Pool.UNSEEN.value]
    if fold_methods is None:
        fold_methods = [FoldMethod.POOL_ALL.value, FoldMethod.LOLO.value, FoldMethod.L1_STRAT_KFOLD.value]

    for p in pools:
        Pool(p)
    for fm in fold_methods:
        FoldMethod(fm)

    if feature_sets is None:
        if feature_set_size == "small":
            feature_sets_dict = {k: v for k, v in ALL_FEATURE_SETS_DICT.items() if k in SMALL_FEATURE_SETS}
        else:
            feature_sets_dict = ALL_FEATURE_SETS_DICT
    else:
        feature_sets_dict = {k: v for k, v in ALL_FEATURE_SETS_DICT.items() if k in feature_sets}
    if skip_fixed_features:
        feature_sets_dict = {k: v for k, v in feature_sets_dict.items() if k not in FIXED_TEXT_GROUP_PREFIXES}

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    _log_predictions_start(
        f"Split-half pool-fold ({n_splits} splits × 2 halves)",
        n_jobs,
        datasets,
        previews,
        models,
        fold_methods,
        None,
        len(feature_sets_dict),
        extra_info=f"\n  Pools: {pools}\n  Parallelization: split × half × pool × fold × model × target × feature_set"
    )

    needs_per_text = any(fs in FIXED_TEXT_GROUP_PREFIXES for fs in feature_sets_dict)
    results = []

    with _JobRunner(n_jobs, executor_kind) as executor:
        for dataset in datasets:
            dataset_previews = ["all"] if dataset == "Meco" else (previews or DATASET_PREVIEWS.get(dataset, ("All",)))
            # For OneStop, run both paragraph and article level; for Meco, only paragraph
            item_levels = ["paragraph", "article"] if dataset == "OneStop" else ["paragraph"]

            split_common_cols: dict = {}
            for split_idx in range(1, n_splits + 1):
                for half_num in [1, 2]:
                    for item_level in item_levels:
                        # Columns common to BOTH halves of this split, so the two
                        # halves being correlated are fitted on identical features.
                        # Computed once per (split, item_level) and shared by both
                        # half_num iterations; headers only, so it is nearly free.
                        ck = (split_idx, item_level)
                        if ck not in split_common_cols:
                            split_common_cols[ck] = _split_half_common_feature_cols(
                                dataset, split_idx, item_level
                            )
                        common_cols = split_common_cols[ck]
                        for preview in dataset_previews:
                            # Load split-half fully_agg features
                            l1_df, l2_df = _load_pool_fold_data(dataset, preview, split_idx, half_num, item_level, require_l1_lextale=require_l1_lextale)
                            if l1_df is None or l2_df is None:
                                continue

                            # Load per-text split-half features
                            pt_l1_df, pt_l2_df = _load_pool_fold_per_text(dataset, preview, needs_per_text, split_idx, half_num, item_level, require_l1_lextale=require_l1_lextale)

                            # Load MECO halves for this split and half
                            halves_data, pt_halves_data = _load_pool_fold_halves(dataset, pools, split_idx, needs_per_text, item_level, half_num, require_l1_lextale=require_l1_lextale)
                            if halves_data is None and dataset == "Meco" and {Pool.SEEN.value, Pool.UNSEEN.value} & set(pools):
                                logger.warning(f"  {dataset}/{preview}/split_{split_idx}/half_{half_num}: halves_data is None for pools {pools}")

                            # Build version suffix for split-half output organization
                            # Include item_level for OneStop (paragraph vs article) before split/half
                            item_level_path = f"/{item_level}" if dataset == "OneStop" else ""
                            version_suffix_base = f"/split_half{item_level_path}/split_{split_idx}/half_{half_num}"

                            # Job creation: reuse pool-fold job logic with version_suffix
                            dataset_key = dataset + "L2"
                            ds_target_cols = get_target_cols(dataset_key, target_set_size, target_cols)

                            combo_jobs = []
                            for model, pool_val, fm_val in itertools.product(models, pools, fold_methods):
                                if not _is_valid_pool_fold_combo(dataset, Pool(pool_val), FoldMethod(fm_val)):
                                    continue
                                if dataset == "Meco" and pool_val in {Pool.SEEN.value, Pool.UNSEEN.value} and halves_data is None:
                                    continue

                                # With/without n_paragraphs thresholds
                                threshold_variants = [(None, "", halves_data, pt_halves_data)]
                                if n_paragraphs_thresholds and dataset == "Meco" and pool_val in {Pool.SEEN.value, Pool.UNSEEN.value} and halves_data:
                                    for thr in n_paragraphs_thresholds:
                                        th_h, th_pt = _filter_halves_by_n_paragraphs(halves_data, pt_halves_data, thr)
                                        threshold_variants.append((thr, f"/n_paragraphs_geq_{thr}", th_h, th_pt))

                                for thr, thr_suffix, halves_for_job, pt_halves_for_job in threshold_variants:
                                    version_suffix = version_suffix_base + thr_suffix
                                    for target_col in ds_target_cols:
                                        for feature_set_name, feature_cols in feature_sets_dict.items():
                                            combo_jobs.append((
                                                dataset, preview, model, pool_val, fm_val,
                                                target_col, feature_set_name, feature_cols,
                                                l1_df, l2_df, halves_for_job,
                                                5, aug_percentage, replace_existing,
                                                pt_l1_df, pt_l2_df, pt_halves_for_job,
                                                version_suffix, require_l1_lextale,
                                                inner_validation,
                                                common_cols,
                                            ))

                            # Submit batch
                            if combo_jobs:
                                batch = executor.run_batch(
                                    _run_pool_fold_prediction_job, combo_jobs,
                                    desc=f"{dataset}/{preview}/split_{split_idx}/half_{half_num}",
                                    leave=False,
                                )
                                results.extend(batch)
                                for job_id, status in batch:
                                    if status != "DONE" and "ERROR" not in status and "SKIP" not in status:
                                        logger.debug(f"Job {job_id}: {status}")

    done = sum(1 for _, s in results if s == "DONE")
    skipped = sum(1 for _, s in results if "SKIP" in s)
    errors = sum(1 for _, s in results if "ERROR" in s)

    _log_predictions_summary(
        "Split-half pool-fold",
        len(results),
        done,
        skipped,
        errors,
        extra_info=f"\n  Total jobs: {len(results)} across {n_splits} × 2 splits"
    )


MODES = ("pool_fold", "per_item", "split_half", "standard")
STAGES = ("train", "evaluation", "both")


def _thresholds(arg: "str | None") -> "list[int] | None":
    return [int(t) for t in parse_csv(arg)] if arg else None


def _train(args) -> None:
    # Exported before any worker is forked; RidgeRegression reads PROF_LOO_MODE
    # per segment via _loo_mode().
    os.environ['PROF_LOO_MODE'] = args.loo_mode
    logger.info(f"LOO mode: {args.loo_mode}")

    common = dict(
        datasets=parse_csv(args.datasets),
        previews=parse_csv(args.previews),
        models=parse_csv(args.models),
        target_cols=parse_csv(args.target_cols),
        target_set_size=args.target_set_size,
        feature_sets=parse_csv(args.feature_sets),
        feature_set_size=args.feature_set_size,
        n_jobs=args.n_jobs,
        executor_kind=args.executor,
        replace_existing=args.replace_existing,
        skip_fixed_features=args.skip_fixed_features,
        aug_percentage=args.aug_percentage,
        require_l1_lextale=args.require_l1_lextale,
    )
    if args.mode == "pool_fold":
        run_pool_fold_predictions_parallel(
            **common,
            pools=parse_csv(args.pools),
            fold_methods=parse_csv(args.fold_methods),
            n_paragraphs_thresholds=_thresholds(args.n_paragraphs_thresholds),
            inner_validation=args.inner_validation,
        )
    elif args.mode == "split_half":
        run_split_half_pool_fold_predictions_parallel(
            **common,
            pools=parse_csv(args.pools),
            fold_methods=parse_csv(args.fold_methods),
            n_splits=args.half_split_n_splits,
            n_paragraphs_thresholds=_thresholds(args.n_paragraphs_thresholds),
            inner_validation=args.inner_validation,
        )
    elif args.mode == "per_item":
        run_per_item_predictions_parallel(**common, pools=parse_csv(args.pools))
    else:
        run_predictions_parallel(
            **common,
            methods=parse_csv(args.methods),
            n_splits=args.n_splits,
            agg_types=set(parse_csv(args.agg_types)) if args.agg_types else None,
            train_on_full_mode=args.train_on_full_mode,
        )


def _evaluate(args) -> None:
    import logging
    # REQUIRE_L1_LEXTALE is read by these modules at import time, which is why
    # main() sets it before either stage runs and they are imported only here.
    from src.evaluation.predictions.evaluation import run_evaluation_pipeline
    from src.evaluation.predictions.tables import run_all_tables

    logging.basicConfig(level=logging.INFO)
    agg_types = parse_csv(args.agg_types)
    if agg_types and len(agg_types) > 1:
        raise ValueError("--stage evaluation takes a single --agg-types value "
                         f"(a version_path prefix); got {agg_types}")
    model_names = parse_csv(args.models)
    if not args.tables_only:
        run_evaluation_pipeline(model_names=model_names,
                                skip_pairwise=args.skip_pairwise,
                                pairwise_baseline_only=args.pairwise_baseline_only,
                                pairwise_vs_ridge=args.pairwise_vs_ridge,
                                comparisons_only=args.comparisons_only,
                                include_bootstrap=not args.no_bootstrap_ci,
                                agg_type=agg_types[0] if agg_types else None)
    if not args.skip_tables:
        run_all_tables(model_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=STAGES, default='train',
                        help='train, evaluation, or both (default: train)')

    data = parser.add_argument_group('data (both stages)')
    data.add_argument('--datasets', type=str, help='Comma-separated datasets: OneStop, Meco (default: both)')
    data.add_argument('--models', type=str,
                      help='Comma-separated models. Training takes registry names (Ridge_Classifier, '
                           'LightGBM, ...); evaluation takes result-directory names, which are the '
                           'same -- not the suffixed row labels such as LightGBM_kfold.')
    data.add_argument('--agg-types', type=str,
                      help='Training (standard mode): aggregation types to fit, comma-separated. '
                           'Evaluation: a single version_path prefix to restrict to, e.g. fully_agg '
                           '(the paper tables read only fully_agg). Default: all.')
    data.add_argument('--require-l1-lextale', action='store_true',
                      help='Use only L1 participants with a measured LexTALE score (drops NaN/-1) '
                           'as the augmentation source, for ALL targets. L2 is untouched. Results '
                           'and their evaluation live under require_l1_lextale/ subtrees, so default '
                           'runs are never overwritten and the two can be compared.')

    train = parser.add_argument_group('training')
    train.add_argument('--mode', choices=MODES, default='pool_fold',
                       help='Cross-validation design (default: pool_fold, the paper\'s)')
    train.add_argument('--previews', type=str, help='Comma-separated previews')
    train.add_argument('--target-cols', type=str, help='Comma-separated target columns')
    train.add_argument('--target-set-size', type=str, choices=TARGET_SET_SIZES, default='small',
                       help='"small" (the two main targets), "all", or "comprehension" '
                            '(comprehension accuracy alone). Default: small')
    train.add_argument('--feature-sets', type=str, help='Comma-separated feature sets')
    train.add_argument('--feature-set-size', type=str, choices=['small', 'all'], default='small',
                       help='"small" (the curated sets) or "all" (default: small)')
    train.add_argument('--pools', type=str,
                       help='pool_fold / per_item / split_half: pools, comma-separated (all, seen, unseen)')
    train.add_argument('--fold-methods', type=str,
                       help='pool_fold / split_half: fold methods, comma-separated '
                            '(pool_all, lolo, l1_strat_kfold)')
    train.add_argument('--inner-validation', type=str, choices=['none', 'holdout', 'kfold'], default='none',
                       help='How the TREE models (Decision_Tree, Random_Forest, LightGBM, XGBoost) '
                            'select a hyperparameter config inside each outer LOPO fold. "none" '
                            '(default): constructor defaults. "kfold": inner 5-fold over the outer '
                            'training rows, scored on pooled out-of-fold predictions. "holdout": a '
                            'single stratified train/validation split -- roughly 1/5 the cost. The '
                            'ridge family ignores this (it selects alpha via RidgeCV\'s closed-form '
                            'LOO). Non-default values add an inner_{strategy} path segment so tuned '
                            'and untuned runs cannot overwrite each other.')
    train.add_argument('--n-paragraphs-thresholds', type=str,
                       help='pool_fold / split_half, MECO only: comma-separated integers N; each adds a '
                            'variant dropping participants with fewer than N paragraphs per half. '
                            'Outputs land under fully_agg/n_paragraphs_geq_N/.')
    train.add_argument('--half-split-n-splits', type=int, default=20,
                       help='split_half: number of random split-half partitions (default: 20)')
    train.add_argument('--methods', type=str, help='standard: comma-separated fold methods')
    train.add_argument('--n-splits', type=int, default=5, help='standard: folds (default: 5)')
    train.add_argument('--train-on-full-mode', type=str, choices=['no', 'yes', 'both'], default='no',
                       help='standard, partial aggregations: train on the full aggregation (default: no)')
    train.add_argument('--aug-percentage', type=float, default=1.0,
                       help='L1 augmentation fraction: 0 disables it, 1 uses all of L1 (default: 1.0)')
    train.add_argument('--skip-fixed-features', action='store_true',
                       help='Skip the per-text feature sets (TRANSITIONS, WFC)')
    train.add_argument('--replace-existing', action='store_true',
                       help='Refit cells whose prediction CSV already exists')
    train.add_argument('--loo-mode', type=str, choices=['nested', 'crossfit', 'auto'], default='nested',
                       help='Leave-one-out implementation for the POOL_ALL seen/all fast path. '
                            '"nested" (default): one RidgeCV per held-out participant, alpha and '
                            'scaler fit on that fold\'s own n-1 rows. "crossfit": the legacy '
                            'stratified A/B half-split, for reproducing previously published '
                            'numbers; degrades badly below ~100 participants. "auto": nested on '
                            'small segments, crossfit on large ones.')
    train.add_argument('--n-jobs', type=int, default=-1, help='Parallel workers: -1 = all CPUs (default: -1)')
    train.add_argument('--executor', type=str, choices=EXECUTOR_KINDS, default='processes',
                       help='Run jobs on worker "processes" (default) or "threads". Ridge fitting only '
                            'partly releases the GIL, so threads plateau near 2.8x regardless of '
                            '--n-jobs; processes scale ~3-4x further.')

    evaluation = parser.add_argument_group('evaluation')
    evaluation.add_argument('--skip-pairwise', action='store_true',
                            help='Skip the pairwise significance stage')
    evaluation.add_argument('--pairwise-baseline-only', action='store_true',
                            help='Only compute pairwise tests involving the WPM baseline feature set')
    evaluation.add_argument('--pairwise-vs-ridge', action='store_true',
                            help='Also compute the model-vs-Ridge comparison behind the model-comparison '
                                 'tables\' subscripts')
    evaluation.add_argument('--comparisons-only', action='store_true',
                            help='Refresh only the WPM-baseline and vs-Ridge comparison CSVs the tables '
                                 'read, without re-running the per-model metrics')
    evaluation.add_argument('--no-bootstrap-ci', action='store_true',
                            help='Skip the per-row bootstrap in the metrics stage (point estimates only)')
    evaluation.add_argument('--tables-only', action='store_true',
                            help='Only re-render the per-model tables from the existing evaluation CSVs')
    evaluation.add_argument('--skip-tables', action='store_true',
                            help='Only evaluate; do not re-render the per-model tables')

    args = parser.parse_args()
    # Route console log lines through tqdm so they don't break the progress bars.
    logger.remove(0)  # loguru's default stderr sink
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    if args.require_l1_lextale:
        os.environ["REQUIRE_L1_LEXTALE"] = "1"
    if args.stage in ("train", "both"):
        _train(args)
    if args.stage in ("evaluation", "both"):
        _evaluate(args)


if __name__ == '__main__':
    main()
