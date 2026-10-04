"""EyeScore evaluation: correlate every eye_score CSV with the proficiency tests.

evaluate_all_results writes all_evaluation_results.csv (and a per-L1 version):
Pearson/Spearman r with bootstrap CIs, plus a paired-bootstrap test against the
WPM (READING_SPEED) baseline. compute_pairwise_significance tests every pair of
feature sets. Run by `python -m src.run.eyescore --stage evaluation`, or
directly (`python -m src.evaluation.EyeScore.evaluation`).
"""
import pandas as pd
import numpy as np
import re
import itertools
from pathlib import Path
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, norm

from src.constants import (
    DATA_PATH,
    ALL_TARGET_COLS,
    DataSets,
    ALL_TESTS,
    RELEVANT_FEATURE_GROUPS_COMBINATIONS_SMALL,
    COMPREHENSION_COLS,
    Fields,
    FeatureGroups,
)
from src.methods.EyeScore.calculation import (
    RAW_VARIANT,
    variant_suffix,
    DEFAULT_CORRECTIONS,
    SPLIT_SCHEMES,
    SPLIT_SCHEME_TO_VARIANT,
)
from src.evaluation.stats_utils import (
    N_BOOTSTRAP,
    bootstrap_correlation_ci,
    bootstrap_difference_test,
    pairwise_dependent_tests,
)
from src.evaluation.utils import save_preserving_unevaluated

# Suffixes of all known corrections, used by collect_eyescore_csvs to identify
# which CSV files belong to which variant.
_ALL_CORRECTION_SUFFIXES = tuple(c.suffix for c in DEFAULT_CORRECTIONS)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("src/methods/EyeScore/results")
EVAL_SAVE_DIR = Path("src/evaluation/EyeScore/results")

# Single source of truth for which previews flow into evaluation, plots,
# language-bias analysis, and tables. Calculation still produces eye_score
# CSVs for every preview on disk (so they remain inspectable directly), but
# downstream consumers iterate this filter. Datasets absent from the map are
# unrestricted.
DATASET_ALLOWED_PREVIEWS = {
    "Meco": ("All",),
    "OneStop": ("Gathering", "Hunting"),
}

# Agg types skipped by every evaluation entry point unless explicitly asked for.
# Single source of truth: the CLI default below and the src.run.eyescore
# stage-2 calls both read it, so `--stage evaluation` and `python -m ...evaluation`
# sweep the same tree.
#   split_half   — scored by src/reliability/split_half.py; its metadata doesn't
#                  resolve here (causes "Metadata not found" spam).
#   first_p_agg  — feed only src/evaluation/agg_plots/, which nothing in the paper
#   moving_p_agg   reads, and they are the memory bound at N_BOOTSTRAP=100k:
#                  moving_p reaches n=27,700, i.e. ~103 GB for one bootstrap call
#                  (five (n_bootstrap, n) arrays), versus ~0.6 GB at fully_agg's
#                  n=152. Pass --exclude-agg-types none to evaluate everything.
DEFAULT_EXCLUDE_AGG_TYPES = ("split_half", "first_p_agg", "moving_p_agg")

# Markers of superseded result trees kept beside the live ones for comparison
# (e.g. split_half_prerun_20260806-1404/, per_item_agg_backup/). Always skipped:
# they are duplicates of the live tree at an older code revision, and the agg-type
# deny-list can't catch them because it matches the leading segment exactly.
_STALE_TREE_MARKERS = ("_prerun_", "_backup", "_prefix_")


def version_path_matches(version_path: str, names) -> bool:
    """True when `version_path` is one of `names` or nested under one of them."""
    return any(version_path == a or version_path.startswith(a + "/") for a in names)


def _evaluation_scope(target_cols=None, feature_sets=None, agg_types=None,
                      exclude_agg_types=None, previews=None):
    """Map this run's filters onto the columns of the saved evaluation CSVs.

    Returns (scope, preview_covered) for save_preserving_unevaluated. Mirrors
    collect_eyescore_csvs: membership for the list filters, prefix matching for
    the agg types, and — because allowed previews depend on the dataset when no
    explicit list is given — the preview rule as a row predicate.

    `variant` is deliberately absent: it selects which CSVs are read, not a
    column of the output, and different variants are kept apart by save_dir.
    Two variants sharing one save_dir would still overwrite each other.
    """
    def version_covered(vp: str) -> bool:
        vp = str(vp)
        if agg_types is not None and not version_path_matches(vp, agg_types):
            return False
        if exclude_agg_types and version_path_matches(vp, exclude_agg_types):
            return False
        return True

    def preview_covered(row) -> bool:
        allowed = (tuple(previews) if previews is not None
                   else DATASET_ALLOWED_PREVIEWS.get(str(row.get("dataset"))))
        return allowed is None or str(row.get("preview")) in allowed

    scope = {
        "target_col": target_cols,
        "feature_set": feature_sets,
        "version_path": (version_covered
                         if agg_types is not None or exclude_agg_types else None),
    }
    return scope, preview_covered


def preview_dir_segments(dataset: str, preview: str) -> Tuple[str, ...]:
    """Extra output-path segments that keep two previews of one dataset apart.

    Several plot layouts drop the preview from the path — harmless while each
    dataset evaluated exactly one preview, but OneStop now evaluates Gathering
    *and* Hunting, which would silently overwrite each other's figures. The
    dataset's first allowed preview keeps its historical location (so existing
    figure references stay valid); every additional preview gets its own
    subdirectory.
    """
    allowed = DATASET_ALLOWED_PREVIEWS.get(dataset)
    if not allowed or preview == allowed[0]:
        return ()
    return (preview,)


# Feature set treated as the baseline for significance-vs-baseline tests.
BASELINE_FEATURE_SET = FeatureGroups.READING_SPEED


# Map results dataset name -> DataSets enum for looking up target cols and data path
DATASET_NAME_TO_ENUM = {
    "OneStop": DataSets.ONESTOPL2,
    "Meco": DataSets.MECOL2,
}

# Feature-set names as they appear in the paper tables — and figures, which is
# why they live here: tables.py imports from language_bias.py, so
# language_bias.py cannot import them from tables.py.
FEATURE_SET_DISPLAY = {
    "READING_SPEED": "WPM",
    "FIXATION_METRICS": "Avg. Fix.",
    "S_CLUSTERS_NO_NORM": "S-Clusters",
    "WP_COEFS_NO_NORM": "WP-Coefs",
    "READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM": "All Combined",
    "TRANSITIONS": "Transitions",
    "WFC": "Word Fix.",
}

# Bootstrap columns: the reported correlation is the bootstrap mean
# (<m>_r_boot) with the resample std as the error bar and a 2.5/97.5 percentile
# 95% CI. The exact Pearson/Spearman r is kept alongside; the plain
# <m>_ci_low/high columns now carry these same bootstrap percentiles, so a
# plotted interval matches the one the bootstrap tables report.
BOOTSTRAP_COLUMNS = [
    "pearson_r_boot", "pearson_r_boot_std", "pearson_ci_low_boot", "pearson_ci_high_boot",
    "spearman_r_boot", "spearman_r_boot_std", "spearman_ci_low_boot", "spearman_ci_high_boot",
    "n_boot",
]

EVALUATION_COLUMNS = [
    "dataset", "preview", "content_mode", "version_path", "feature_set",
    "target_col",
    "pearson_r", "pearson_p", "pearson_ci_low", "pearson_ci_high",
    "spearman_r", "spearman_p", "spearman_ci_low", "spearman_ci_high",
] + BOOTSTRAP_COLUMNS + [
    "baseline_diff_z_pearson", "baseline_diff_p_pearson",
    "baseline_diff_z_spearman", "baseline_diff_p_spearman",
    "n",
]

# Per-language CSV adds L1 right after the grouping keys. Baseline-diff stats
# are kept in the schema but stay NaN for per-L1 rows (sample sizes are too
# small for a baseline-difference test to be meaningful).
PER_LANGUAGE_AGGREGATE_LABEL = "All"
PER_LANGUAGE_EVALUATION_COLUMNS = (
    EVALUATION_COLUMNS[:6] + ["L1"] + EVALUATION_COLUMNS[6:]
)


def _meng_z_test(r1: float, r2: float, r12: float, n: int) -> Tuple[float, float]:
    """Meng-Rosenthal-Rubin (1992) Z-test for the difference between two
    dependent correlations that share a common variable (here: the target).

    Not used by the pipeline (baseline-difference significance is the paired
    bootstrap, see `_baseline_diff_stats`); kept as an analytic cross-check,
    which supplementary/eyescore_experiments uses.

    r1 = corr(X1, Y), r2 = corr(X2, Y), r12 = corr(X1, X2), same sample size n.
    Returns (z, two-sided p-value), or (nan, nan) if inputs are invalid.
    """
    if n < 4:
        return (np.nan, np.nan)
    if not all(np.isfinite([r1, r2, r12])):
        return (np.nan, np.nan)
    if abs(r1) >= 1 or abs(r2) >= 1 or abs(r12) >= 1:
        return (np.nan, np.nan)
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    rm_sq = (r1 ** 2 + r2 ** 2) / 2.0
    denom = 1.0 - rm_sq
    if denom <= 0:
        return (np.nan, np.nan)
    f = (1.0 - r12) / (2.0 * denom)
    f = min(f, 1.0)
    h = (1.0 - f * rm_sq) / denom
    # Guard against h <= 0 (can happen with pathological inputs).
    if h <= 0 or (1.0 - r12) <= 0:
        return (np.nan, np.nan)
    z_stat = (z1 - z2) * np.sqrt((n - 3) / (2.0 * (1.0 - r12) * h))
    p = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    return float(z_stat), float(p)


def compute_correlation(eye_scores: np.ndarray, test_scores: np.ndarray,
                        include_bootstrap: bool = True,
                        n_bootstrap: int = N_BOOTSTRAP) -> Dict[str, float]:
    """Pearson and Spearman correlations and, when `include_bootstrap` is set,
    their bootstrap estimates: mean of the resamples with the resample std as
    the error bar and a 2.5/97.5 percentile CI (see stats_utils). The plain
    <m>_ci_low/high columns carry those same percentiles."""
    mask = ~(np.isnan(eye_scores) | np.isnan(test_scores))
    eye_scores, test_scores = eye_scores[mask], test_scores[mask]
    n = len(eye_scores)

    # n<3, or a constant array on either side (correlation undefined) -> NaN,
    # the same NaN SciPy would return, without its ConstantInputWarning.
    # Checked before the bootstrap: every resample of such an array is itself
    # degenerate, so resampling it would only cost time (the per-language grid
    # has hundreds of such slices). n_boot is 0 for these rows.
    if n < 3 or np.ptp(eye_scores) == 0 or np.ptp(test_scores) == 0:
        boot_cols = {c: np.nan for c in BOOTSTRAP_COLUMNS}
        if include_bootstrap and n >= 3:
            boot_cols["n_boot"] = 0
        return {
            "pearson_r": np.nan, "pearson_p": np.nan,
            "pearson_ci_low": np.nan, "pearson_ci_high": np.nan,
            "spearman_r": np.nan, "spearman_p": np.nan,
            "spearman_ci_low": np.nan, "spearman_ci_high": np.nan,
            **boot_cols,
            "n": n,
        }

    boot = (
        bootstrap_correlation_ci(eye_scores, test_scores, n_bootstrap=n_bootstrap)
        if include_bootstrap
        else {c: np.nan for c in BOOTSTRAP_COLUMNS}
    )
    boot_cols = {c: boot.get(c, np.nan) for c in BOOTSTRAP_COLUMNS}

    pr, pp = pearsonr(eye_scores, test_scores)
    sr, sp = spearmanr(eye_scores, test_scores)
    # CIs come from the bootstrap, so the interval plotted for a feature set is
    # the interval the bootstrap tables report for it. With include_bootstrap
    # off there are no resamples to take a percentile of, so the CI columns
    # are NaN rather than silently falling back to a different estimator.
    p_lo, p_hi = boot_cols.get("pearson_ci_low_boot", np.nan), boot_cols.get("pearson_ci_high_boot", np.nan)
    s_lo, s_hi = boot_cols.get("spearman_ci_low_boot", np.nan), boot_cols.get("spearman_ci_high_boot", np.nan)
    return {
        "pearson_r": pr, "pearson_p": pp,
        "pearson_ci_low": p_lo, "pearson_ci_high": p_hi,
        "spearman_r": sr, "spearman_p": sp,
        "spearman_ci_low": s_lo, "spearman_ci_high": s_hi,
        **boot_cols,
        "n": n,
    }


def _baseline_diff_stats(eye_df_fs: pd.DataFrame, eye_df_base: pd.DataFrame,
                         meta: pd.DataFrame, target_col: str) -> Dict[str, float]:
    """Paired-bootstrap test of feature-set correlation vs baseline correlation
    on the shared participant subset (joined on participant_id, target
    non-null).

    Fills ``baseline_diff_p_pearson`` / ``baseline_diff_p_spearman``, which are
    what the EyeScore bar plots star against the baseline. The matching
    ``baseline_diff_z_*`` columns are kept in the schema but left NaN: the
    bootstrap has no z statistic.

    Returns NaNs if the subset is too small or any correlation is degenerate.
    """
    empty = {
        "baseline_diff_z_pearson": np.nan, "baseline_diff_p_pearson": np.nan,
        "baseline_diff_z_spearman": np.nan, "baseline_diff_p_spearman": np.nan,
    }
    if eye_df_fs is None or eye_df_base is None or target_col not in meta.columns:
        return empty

    joined = (
        eye_df_fs[[Fields.SUBJECT_ID, "eye_score"]]
        .rename(columns={"eye_score": "x1"})
        .merge(
            eye_df_base[[Fields.SUBJECT_ID, "eye_score"]].rename(columns={"eye_score": "x2"}),
            on=Fields.SUBJECT_ID, how="inner",
        )
        .merge(meta[[Fields.SUBJECT_ID, target_col]], on=Fields.SUBJECT_ID, how="inner")
    )
    joined = joined.dropna(subset=["x1", "x2", target_col])
    joined = joined[joined[target_col] != -1]
    n = len(joined)
    if n < 4:
        return empty

    y = joined[target_col].values.astype(float)
    x1 = joined["x1"].values
    x2 = joined["x2"].values

    # Paired bootstrap over the shared participants — the same test behind the
    # stars in the bootstrap tables, so a plot and a table never disagree about
    # whether a feature set beats the baseline.
    boot = bootstrap_difference_test(y, x1, x2)

    return {
        "baseline_diff_z_pearson": np.nan,
        "baseline_diff_p_pearson": boot["boot_diff_pearson_p"],
        "baseline_diff_z_spearman": np.nan,
        "baseline_diff_p_spearman": boot["boot_diff_spearman_p"],
    }


def load_metadata(dataset_name: str, version_path: str) -> pd.DataFrame:
    """Load metadata with test scores for joining with eye_score results.

    Args:
        dataset_name: Results-level dataset name (e.g. 'OneStop').
        version_path: Sub-path under features_and_targets (e.g. 'fully_agg/all/all').
    """
    data_dataset = f"{dataset_name}L2"
    meta_path = DATA_PATH / data_dataset / "features_and_targets" / version_path / "features_and_metadata.csv"
    if not meta_path.exists():
        # Threshold-derived version paths (e.g. 'fully_agg/n_paragraphs_geq_4/all/all')
        # don't have their own feature file — they reuse the unfiltered baseline's
        # metadata (target scores, L1) and just drop participants. Strip the
        # threshold segment and try again before giving up.
        stripped = re.sub(r'/n_paragraphs_geq_\d+', '', version_path)
        if stripped != version_path:
            fallback = DATA_PATH / data_dataset / "features_and_targets" / stripped / "features_and_metadata.csv"
            if fallback.exists():
                meta = pd.read_csv(fallback)
                return meta.drop_duplicates(Fields.SUBJECT_ID)
        # For per_item_agg, metadata may be stored in a paragraph subdirectory
        if "per_item_agg" in version_path:
            para_fallback = meta_path.parent / "paragraph" / "features_and_metadata.csv"
            if para_fallback.exists():
                meta = pd.read_csv(para_fallback)
                return meta.drop_duplicates(Fields.SUBJECT_ID)
        logger.warning("Metadata not found: %s", meta_path)
        return pd.DataFrame()
    meta = pd.read_csv(meta_path)
    return meta.drop_duplicates(Fields.SUBJECT_ID)


def collect_eyescore_csvs(
    results_dir: Path,
    feature_sets: List[str] = None,
    variant: str = RAW_VARIANT,
    agg_types: List[str] = None,
    exclude_agg_types: List[str] = None,
    previews: List[str] = None,
) -> List[dict]:
    """Walk the results directory and collect eye_score CSV paths for one variant.

    Expected structure: results_dir/{dataset}/{preview}/{content_mode}/{version_path...}/{feature_set}{variant_suffix}.csv

    Args:
        results_dir: Root directory containing eye_score results.
        feature_sets: Optional filter of feature set names (base names, without
            any correction suffix).
        variant: Which variant to collect — RAW_VARIANT for raw eye_scores, or
            a Correction.name for the corresponding corrected CSVs.
        agg_types: Optional allow-list on the leading version_path segment
            (e.g. ["fully_agg"]). "fully_agg" captures both the main fully_agg
            results and seen/unseen (whose version_path is "fully_agg/all/all").
            None = no allow-list (every agg type on disk).
        exclude_agg_types: Optional deny-list on the leading version_path
            segment (e.g. ["split_half"] to skip the reliability tree, which is
            evaluated separately by src/reliability/split_half.py and whose
            metadata doesn't resolve here). Applied after `agg_types`.
        previews: Optional explicit allow-list of preview directory names that
            REPLACES DATASET_ALLOWED_PREVIEWS for every dataset (e.g.
            ["Hunting"] to evaluate that slice alone). None = the standard
            per-dataset map.
    """
    expected_suffix = variant_suffix(variant)
    _matches = version_path_matches

    entries = []
    for csv_path in results_dir.rglob("*.csv"):
        rel = csv_path.relative_to(results_dir)
        parts = rel.parts
        # minimum: dataset/preview/content_mode/version.../feature_set.csv
        if len(parts) < 5:
            continue
        stem = csv_path.stem

        # Skip per-item intermediate files (only evaluate aggregated)
        if stem.endswith("_items"):
            continue

        if expected_suffix:
            if not stem.endswith(expected_suffix):
                continue
            feature_set = stem[: -len(expected_suffix)]
        else:
            # Raw variant: skip any file that ends in a known correction suffix.
            if any(stem.endswith(s) for s in _ALL_CORRECTION_SUFFIXES):
                continue
            feature_set = stem

        if feature_sets is not None and feature_set not in feature_sets:
            continue

        version_path = "/".join(parts[3:-1])
        # Filter agg types by leading version_path segment. "fully_agg" also
        # matches seen/unseen (version_path "fully_agg/all/all"); "split_half"
        # matches the reliability tree.
        if agg_types is not None and not _matches(version_path, agg_types):
            continue
        if exclude_agg_types and _matches(version_path, exclude_agg_types):
            continue
        # Superseded trees kept on disk for comparison (…_prerun_<stamp>,
        # …_backup, …_sclusters_prefix_<stamp>). The deny-list above matches
        # the leading segment exactly and would not catch them, so match on
        # the marker instead.
        if any(m in parts[3] for m in _STALE_TREE_MARKERS):
            continue

        allowed_previews = (
            tuple(previews) if previews is not None
            else DATASET_ALLOWED_PREVIEWS.get(parts[0])
        )
        if allowed_previews is not None and parts[1] not in allowed_previews:
            continue

        entries.append({
            "csv_path": csv_path,
            "dataset": parts[0],
            "preview": parts[1],
            "content_mode": parts[2],
            "version_path": version_path,
            "feature_set": feature_set,
        })
    return entries


def eye_score_csv_path(results_dir: Path, dataset: str, preview: str,
                       content_mode: str, version_path: str,
                       feature_set: str, variant: str) -> Path:
    """Return the eye_score CSV path for a given (feature_set, variant) pair."""
    return (
        results_dir / dataset / preview / content_mode / version_path
        / f"{feature_set}{variant_suffix(variant)}.csv"
    )


def _safe_read_eye_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Read an eye_score CSV defensively — return None on any problem."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("Failed to read %s: %s", csv_path, e)
        return None
    if df.empty or "eye_score" not in df.columns or Fields.SUBJECT_ID not in df.columns:
        return None
    return df


def evaluate_all_results(results_dir: Path = None, save_dir: Path = None,
                         target_cols: List[str] = None, feature_sets: List[str] = None,
                         variant: str = RAW_VARIANT, agg_types: List[str] = None,
                         exclude_agg_types: List[str] = None,
                         previews: List[str] = None) -> pd.DataFrame:
    """Evaluate all eye_score results by correlating with test scores.

    For each (feature_set, target_col) row, we also compute:
      - bootstrap 95% percentile confidence intervals around Pearson and
        Spearman r (the same intervals the bootstrap tables report), and
      - a paired-bootstrap test against the READING_SPEED baseline feature
        set's correlation on the shared participant subset (NaN for the
        baseline's own rows).

    `agg_types` / `exclude_agg_types` filter which aggregation types are
    evaluated by version_path prefix. The direct-module invocation defaults to
    excluding "split_half" (the reliability tree, handled separately by
    src/reliability/split_half.py). None/None = everything on disk.

    Saves:
        - all_evaluation_results.csv: one row per (dataset, preview, version_path, feature_set, target_col)
        - Grouped CSVs per (dataset, preview, version_path) with all feature_sets x target_cols
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    if save_dir is None:
        save_dir = EVAL_SAVE_DIR

    entries = collect_eyescore_csvs(results_dir, feature_sets=feature_sets, variant=variant,
                                    agg_types=agg_types, exclude_agg_types=exclude_agg_types,
                                    previews=previews)
    if not entries:
        logger.warning("No eye_score CSVs found in %s", results_dir)
        return pd.DataFrame()

    logger.info("Found %d eye_score files to evaluate", len(entries))

    # Group entries by (dataset, preview, content_mode, version_path) so we can
    # share the baseline eye_score CSV across all non-baseline feature sets in
    # the group.  The baseline must come from the same content_mode — a "seen"
    # feature set is compared against the "seen" READING_SPEED baseline.
    grouped_entries: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for entry in entries:
        grouped_entries[(entry["dataset"], entry["preview"], entry["content_mode"], entry["version_path"])].append(entry)

    metadata_cache: Dict[str, pd.DataFrame] = {}
    all_rows = []
    per_l1_rows = []

    empty_diff = {
        "baseline_diff_z_pearson": np.nan, "baseline_diff_p_pearson": np.nan,
        "baseline_diff_z_spearman": np.nan, "baseline_diff_p_spearman": np.nan,
    }

    for group_key, group_entries in tqdm(grouped_entries.items(),
                                         desc="Evaluating eye_score correlations"):
        dataset, preview, content_mode, version_path = group_key
        cache_key = f"{dataset}/{version_path}"
        if cache_key not in metadata_cache:
            metadata_cache[cache_key] = load_metadata(dataset, version_path)
        meta = metadata_cache[cache_key]
        if meta.empty:
            continue

        # Pre-load the baseline eye_score CSV for this group (if present).
        baseline_entry = next(
            (e for e in group_entries if e["feature_set"] == BASELINE_FEATURE_SET), None,
        )
        baseline_df = _safe_read_eye_csv(baseline_entry["csv_path"]) if baseline_entry else None

        # Target columns to evaluate
        if target_cols is not None:
            ds_target_cols = target_cols
        else:
            ds_enum = DATASET_NAME_TO_ENUM.get(dataset)
            if ds_enum is None:
                logger.warning("Unknown dataset %s, skipping", dataset)
                continue
            ds_target_cols = ALL_TARGET_COLS.get(ds_enum, [])

        for entry in group_entries:
            eye_df = _safe_read_eye_csv(entry["csv_path"])
            if eye_df is None:
                continue

            merged = eye_df.merge(meta, on=Fields.SUBJECT_ID, how="inner", suffixes=("", "_meta"))
            if merged.empty:
                logger.warning("No matching participants for %s", entry["csv_path"])
                continue

            is_baseline = entry["feature_set"] == BASELINE_FEATURE_SET

            for target_col in ds_target_cols:
                if target_col not in merged.columns:
                    continue
                target_vals = merged[target_col].values.astype(float)
                # -1 is the missing-value sentinel in Meco metadata; treat as NaN
                # so compute_correlation drops those rows (consistent with
                # _baseline_diff_stats).
                target_vals = np.where(target_vals == -1, np.nan, target_vals)
                metrics = compute_correlation(
                    merged["eye_score"].values,
                    target_vals,
                )
                # Baseline-difference stats: NaN for the baseline row itself.
                if is_baseline or baseline_df is None:
                    diff_stats = dict(empty_diff)
                else:
                    diff_stats = _baseline_diff_stats(eye_df, baseline_df, meta, target_col)

                base_row = {
                    "dataset": dataset,
                    "preview": entry["preview"],
                    "content_mode": content_mode,
                    "version_path": version_path,
                    "feature_set": entry["feature_set"],
                    "target_col": target_col,
                }
                all_rows.append({**base_row, **metrics, **diff_stats})

                # Per-language slice: same correlation but stratified by L1,
                # plus an "All" row mirroring the aggregate so a single CSV is
                # self-contained for downstream consumers (tables / plots).
                if Fields.L1 in merged.columns:
                    per_l1_rows.append({
                        **base_row, "L1": PER_LANGUAGE_AGGREGATE_LABEL,
                        **metrics, **diff_stats,
                    })
                    for l1, subset in merged.groupby(Fields.L1):
                        sub_target = subset[target_col].values.astype(float)
                        sub_target = np.where(sub_target == -1, np.nan, sub_target)
                        sub_metrics = compute_correlation(
                            subset["eye_score"].values, sub_target,
                        )
                        per_l1_rows.append({
                            **base_row, "L1": l1, **sub_metrics, **empty_diff,
                        })

    all_results_df = pd.DataFrame(all_rows)
    if all_results_df.empty:
        logger.warning("No evaluation results produced")
        return all_results_df

    # Enforce a stable column order (keeps CSV schema predictable).
    ordered_cols = [c for c in EVALUATION_COLUMNS if c in all_results_df.columns]
    extra_cols = [c for c in all_results_df.columns if c not in ordered_cols]
    all_results_df = all_results_df[ordered_cols + extra_cols]

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save combined, keeping any rows this run's filters did not cover (see
    # save_preserving_unevaluated).
    scope, preview_covered = _evaluation_scope(
        target_cols, feature_sets, agg_types, exclude_agg_types, previews)
    combined_path = save_dir / "all_evaluation_results.csv"
    save_preserving_unevaluated(all_results_df, combined_path, scope,
                                row_predicate=preview_covered)

    # Per-language: same scoring stratified by L1, written next to the
    # aggregate. Useful for the across_langs LOLO view in particular
    # (per-language is the whole transferability question), but emitted
    # whenever an L1 column is present so consumers don't have to special-case.
    if per_l1_rows:
        per_l1_df = pd.DataFrame(per_l1_rows)
        ordered_cols = [c for c in PER_LANGUAGE_EVALUATION_COLUMNS if c in per_l1_df.columns]
        extra_cols = [c for c in per_l1_df.columns if c not in ordered_cols]
        per_l1_df = per_l1_df[ordered_cols + extra_cols]

        per_l1_path = save_dir / "per_language_evaluation_results.csv"
        save_preserving_unevaluated(per_l1_df, per_l1_path, scope,
                                    row_predicate=preview_covered)

    return all_results_df


def compute_pairwise_significance(results_dir: Path = None, save_dir: Path = None,
                                  target_cols: List[str] = None, feature_sets: List[str] = None,
                                  variant: str = RAW_VARIANT, agg_types: List[str] = None,
                                  exclude_agg_types: List[str] = None,
                                  previews: List[str] = None) -> pd.DataFrame:
    """Pairwise paired-bootstrap test between every two feature sets (models)
    sharing the same target.

    Each feature set's eye_score correlates with the shared target; the two
    feature sets' eye_scores are themselves correlated — i.e. two dependent,
    overlapping correlations. Feature sets are grouped by
    (dataset, preview, content_mode, version_path); within a group every pair
    is compared, per target_col, on the participants they share. Writes
    `pairwise_significance.csv`.

    `agg_types` / `exclude_agg_types` filter aggregation types by version_path
    prefix (default invocation excludes "split_half").
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    if save_dir is None:
        save_dir = EVAL_SAVE_DIR

    entries = collect_eyescore_csvs(results_dir, feature_sets=feature_sets, variant=variant,
                                    agg_types=agg_types, exclude_agg_types=exclude_agg_types,
                                    previews=previews)
    if not entries:
        logger.warning("No eye_score CSVs found in %s for pairwise significance", results_dir)
        return pd.DataFrame()

    grouped_entries: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for entry in entries:
        grouped_entries[(entry["dataset"], entry["preview"], entry["content_mode"], entry["version_path"])].append(entry)

    metadata_cache: Dict[str, pd.DataFrame] = {}
    rows = []
    for group_key, group_entries in tqdm(grouped_entries.items(),
                                         desc="Pairwise eye_score significance"):
        dataset, preview, content_mode, version_path = group_key
        cache_key = f"{dataset}/{version_path}"
        if cache_key not in metadata_cache:
            metadata_cache[cache_key] = load_metadata(dataset, version_path)
        meta = metadata_cache[cache_key]
        if meta.empty:
            continue

        # Load each feature set's per-participant eye_score once. Skip any file
        # with multiple rows per participant (e.g. moving_p_agg window rows):
        # those can't be aligned one-to-one across feature sets, and merging on
        # participant_id alone would cartesian-explode the pair.
        loaded = []
        for entry in group_entries:
            eye_df = _safe_read_eye_csv(entry["csv_path"])
            if eye_df is None:
                continue
            eye_df = eye_df[[Fields.SUBJECT_ID, "eye_score"]].dropna()
            if eye_df[Fields.SUBJECT_ID].duplicated().any():
                continue
            loaded.append((entry, eye_df))
        if len(loaded) < 2:
            continue

        if target_cols is not None:
            ds_target_cols = target_cols
        else:
            ds_enum = DATASET_NAME_TO_ENUM.get(dataset)
            ds_target_cols = ALL_TARGET_COLS.get(ds_enum, []) if ds_enum else []

        for target_col in ds_target_cols:
            if target_col not in meta.columns:
                continue
            tgt = meta[[Fields.SUBJECT_ID, target_col]].copy()
            # -1 is the Meco missing-value sentinel; treat as NaN.
            tgt[target_col] = tgt[target_col].replace(-1, np.nan)

            for (ea, da), (eb, db) in itertools.combinations(loaded, 2):
                merged = (
                    da.rename(columns={"eye_score": "x_a"})
                    .merge(db.rename(columns={"eye_score": "x_b"}), on=Fields.SUBJECT_ID)
                    .merge(tgt, on=Fields.SUBJECT_ID)
                    .dropna(subset=["x_a", "x_b", target_col])
                )
                if len(merged) < 4:
                    continue
                stats_row = pairwise_dependent_tests(
                    merged[target_col].values, merged["x_a"].values, merged["x_b"].values,
                )
                rows.append({
                    "dataset": dataset, "preview": preview, "content_mode": content_mode,
                    "version_path": version_path, "target_col": target_col,
                    "feature_set_a": ea["feature_set"], "feature_set_b": eb["feature_set"],
                    **stats_row,
                })

    result_df = pd.DataFrame(rows)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "pairwise_significance.csv"

    # A comparison is regenerated only when BOTH of its endpoints were in
    # scope, so the feature-set filter applies to the _a and _b columns alike.
    base, preview_covered = _evaluation_scope(
        target_cols, feature_sets, agg_types, exclude_agg_types, previews)
    scope = {
        "target_col": base["target_col"],
        "version_path": base["version_path"],
        "feature_set_a": feature_sets, "feature_set_b": feature_sets,
    }
    save_preserving_unevaluated(result_df, out_path, scope,
                                row_predicate=preview_covered)
    return result_df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-suffix", default="",
        help="Suffix routing inputs/outputs to sibling trees: reads from "
             "src/methods/EyeScore/results_<suffix>/ and writes to "
             "src/evaluation/EyeScore/results_<suffix>/. Default: empty "
             "(canonical results/ trees). Use the same suffix you passed to "
             "src.run.eyescore so eval consumes that stage 1 output.",
    )
    parser.add_argument(
        "--agg-types", default="all",
        help="Comma-separated aggregation types to KEEP, matched by version_path "
             "prefix. Default 'all' = no allow-list (every agg type on disk, "
             "then minus --exclude-agg-types).",
    )
    parser.add_argument(
        "--exclude-agg-types", default=",".join(DEFAULT_EXCLUDE_AGG_TYPES),
        help="Comma-separated aggregation types to SKIP, matched by version_path "
             f"prefix. Default '{','.join(DEFAULT_EXCLUDE_AGG_TYPES)}': the "
             "reliability tree is scored by src/reliability/split_half.py (its "
             "metadata doesn't resolve here), and the window regimes feed only "
             "agg_plots while dominating runtime and memory. Pass 'none' to "
             "disable the deny-list.",
    )
    parser.add_argument(
        "--include-split-half", action="store_true",
        help="Also evaluate the split-half tree (2640 extra files, n~1107 per "
             "file instead of ~152, so ~27x the work and ~7x the peak bootstrap "
             "memory). Off by default: split-half reliability is computed from "
             "the results tree by src/reliability/split_half.py and does not "
             "read these rows, so this is only for inspecting split-half "
             "correlations in the evaluation CSV.",
    )
    args = parser.parse_args()

    if args.results_suffix:
        RESULTS_DIR = Path(f"src/methods/EyeScore/results_{args.results_suffix}")
        EVAL_SAVE_DIR = Path(f"src/evaluation/EyeScore/results_{args.results_suffix}")
        logger.info("Using suffixed trees: %s → %s", RESULTS_DIR, EVAL_SAVE_DIR)

    agg_types = (
        None if args.agg_types.strip().lower() == "all"
        else [a.strip() for a in args.agg_types.split(",") if a.strip()]
    )
    exclude_agg_types = (
        None if args.exclude_agg_types.strip().lower() == "none"
        else [a.strip() for a in args.exclude_agg_types.split(",") if a.strip()]
    )
    # Opt-in: lift split_half out of the deny-list rather than making the caller
    # retype the rest of it, so forgetting the flag can only ever be the cheap,
    # correct default.
    if args.include_split_half and exclude_agg_types:
        exclude_agg_types = [a for a in exclude_agg_types if a != "split_half"]
    logger.info("Evaluating agg_types: keep=%s, exclude=%s%s",
                agg_types if agg_types else "ALL", exclude_agg_types or "none",
                "  (--include-split-half)" if args.include_split_half else "")

    feature_sets = list(RELEVANT_FEATURE_GROUPS_COMBINATIONS_SMALL.keys())
    # Include comprehension targets so the separate comprehension bar plots have
    # data. ALL_TESTS already carries COMPREHENSION_COL, so deduplicate
    # (dict.fromkeys keeps first-seen order) rather than evaluate it twice.
    target_cols = list(dict.fromkeys(list(ALL_TESTS) + list(COMPREHENSION_COLS)))

    # Pristine: org evaluated on all participants — the canonical EyeScore
    # table, with no train/test split or bias correction.
    evaluate_all_results(
        results_dir=RESULTS_DIR,
        save_dir=EVAL_SAVE_DIR, target_cols=target_cols,
        feature_sets=feature_sets, variant=RAW_VARIANT,
        agg_types=agg_types, exclude_agg_types=exclude_agg_types,
    )

    # Pairwise significance between every two feature sets per target.
    compute_pairwise_significance(
        results_dir=RESULTS_DIR,
        save_dir=EVAL_SAVE_DIR, target_cols=target_cols,
        feature_sets=feature_sets, variant=RAW_VARIANT,
        agg_types=agg_types, exclude_agg_types=exclude_agg_types,
    )

    # Lang-bias subtree: per split scheme, evaluate the corrected variant
    # plus an org baseline next to it for direct comparison. The "test slice"
    # under both schemes is every participant (each lands in exactly one test
    # fold), so the org baseline numbers match the pristine ones — they live
    # here so org vs typo_calibrated_<scheme> sits in the same directory.
    for scheme in SPLIT_SCHEMES:
        bias_dir = EVAL_SAVE_DIR / "lang_bias" / scheme
        evaluate_all_results(
            results_dir=RESULTS_DIR,
            save_dir=bias_dir / "org", target_cols=target_cols,
            feature_sets=feature_sets, variant=RAW_VARIANT,
            agg_types=agg_types, exclude_agg_types=exclude_agg_types,
        )
        evaluate_all_results(
            results_dir=RESULTS_DIR,
            save_dir=bias_dir / "typo_calibrated", target_cols=target_cols,
            feature_sets=feature_sets, variant=SPLIT_SCHEME_TO_VARIANT[scheme],
            agg_types=agg_types, exclude_agg_types=exclude_agg_types,
        )
