"""Evaluation plots for first_p_agg and moving_p_agg EyeScore regimes."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging
from scipy import stats

from src.constants import AggTypes
from src.agg_configs import AGG_CONFIGS_BY_DATASET
from src.evaluation.EyeScore.plot import TARGET_COL_DISPLAY_NAMES
from src.evaluation.agg_plots.plot_predictions import (
    PEARSON_Y_RANGE, BIAS_Y_RANGE, FIRST_P_X_RANGE_PARAGRAPH_MECO, FIRST_P_X_RANGE_PARAGRAPH_ONESTOP,
    FIRST_P_X_RANGE_ARTICLE_ONESTOP, _plot_line_chart, _get_content_group_map,
    PARTIAL_TARGET_COLS
)

logger = logging.getLogger(__name__)

PLOTS_SAVE_DIR = Path("src/evaluation/agg_plots/plots/eyescore")


def _get_eyescore_path(results_base: Path, dataset: str, preview: str, agg_type: str, p_level: str, p_n: int, feature_set: str) -> Path:
    """Construct path to EyeScore result CSV."""
    # EyeScore results structure: results/{dataset}/{Preview}/{agg_type}/{filter}/{cg}/{p_level}_{p_n}/{fs}.csv
    # filter and cg are 'ordinary' and 'all' for OneStop, and 'all' for Meco
    if dataset == "OneStop":
        return results_base / dataset / preview / agg_type / "ordinary" / "all" / f"{p_level}_{p_n}" / f"{feature_set}.csv"
    else:  # Meco
        return results_base / dataset / preview / agg_type / "all" / "all" / f"{p_level}_{p_n}" / f"{feature_set}.csv"


def _load_eyescore_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load EyeScore CSV; return None if file doesn't exist or is invalid."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or 'eye_score' not in df.columns or 'participant_id' not in df.columns:
            return None
        return df
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def _load_metadata_for_target(data_base: Path, dataset: str, agg_type: str, p_level: str, p_n: int, target_col: str) -> Optional[pd.DataFrame]:
    """Load metadata for target column correlation."""
    if agg_type == AggTypes.FULL:
        path = data_base / f"{dataset}L2" / "features_and_targets" / AggTypes.FULL / "ordinary" / "all" / "features_and_metadata.csv"
    else:
        path = data_base / f"{dataset}L2" / "features_and_targets" / agg_type / "ordinary" / "all" / f"{p_level}_{p_n}" / "features_and_metadata.csv"

    if not path.exists():
        logger.warning(f"Metadata not found: {path}")
        return None

    try:
        df = pd.read_csv(path, usecols=['participant_id', target_col])
        if target_col not in df.columns:
            return None
        # Handle -1 as missing value (Meco convention)
        df[target_col] = df[target_col].replace(-1, np.nan)
        return df
    except Exception as e:
        logger.warning(f"Failed to load metadata {path}: {e}")
        return None


def _get_window_index(df: pd.DataFrame) -> np.ndarray:
    """Get window_index from df; fall back to cumcount if not present."""
    if 'window_index' in df.columns:
        return df['window_index'].values
    return df.groupby('participant_id').cumcount().values


def _pearson_r(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute Pearson r; return None if unable to compute."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return None
    try:
        r, _ = stats.pearsonr(x[mask], y[mask])
        return r
    except (ValueError, RuntimeError):
        return None


def _get_first_p_values(dataset: str, p_level: str) -> List[int]:
    """Get all p values for first_p_agg from agg_configs."""
    dataset_key = f"{dataset}L2" if not dataset.endswith(('L1', 'L2')) else dataset
    configs = AGG_CONFIGS_BY_DATASET.get(dataset_key, [])
    p_values = []
    for config in configs:
        if config.get('agg_type') == 'first_p_agg' and config.get('p_agg_level') == p_level:
            p_values.append(config['p_agg'])
    return sorted(set(p_values))


def _get_moving_p_values(dataset: str, p_level: str) -> List[int]:
    """Get all p values for moving_p_agg from agg_configs."""
    dataset_key = f"{dataset}L2" if not dataset.endswith(('L1', 'L2')) else dataset
    configs = AGG_CONFIGS_BY_DATASET.get(dataset_key, [])
    p_values = []
    for config in configs:
        if config.get('agg_type') == 'moving_p_agg' and config.get('p_agg_level') == p_level:
            p_values.append(config['p_agg'])
    return sorted(set(p_values))


def _get_first_p_x_range(dataset: str, p_level: str) -> Tuple[float, float]:
    """Get x-axis range for first_p_agg plots."""
    if dataset == "Meco":
        if p_level == "paragraph":
            return FIRST_P_X_RANGE_PARAGRAPH_MECO
        else:
            return (0, 10)
    elif dataset == "OneStop":
        if p_level == "paragraph":
            return FIRST_P_X_RANGE_PARAGRAPH_ONESTOP
        else:
            return FIRST_P_X_RANGE_ARTICLE_ONESTOP
    else:
        return (0, 10)


def plot_first_p_agg_pearson(
    results_base: Path, data_base: Path,
    dataset: str, preview: str,
    target_col: str, feature_sets: List[str],
    p_level: str, save_dir: Path,
    content_group: Optional[str] = None
) -> None:
    """Plot Pearson r vs p for first_p_agg EyeScore."""
    if target_col in PARTIAL_TARGET_COLS:
        logger.info(f"Skipping partial target column {target_col}")
        return

    p_values = _get_first_p_values(dataset, p_level)
    if not p_values:
        logger.warning(f"No first_p_agg configs found for {dataset}/{p_level}")
        return

    data = {fs: [] for fs in feature_sets}
    fully_agg_baselines = {fs: None for fs in feature_sets}
    cg_map = _get_content_group_map(data_base, dataset) if dataset == "OneStop" else {}

    # Load fully_agg baseline for each feature set
    meta_fa = _load_metadata_for_target(data_base, dataset, AggTypes.FULL, "paragraph", 1, target_col)
    if meta_fa is None:
        logger.warning(f"Could not load metadata for fully_agg {dataset}/{target_col}")
        return

    for fs in feature_sets:
        path = _get_eyescore_path(results_base, dataset, preview, AggTypes.FULL, "paragraph", 1, fs)
        df_fa = _load_eyescore_csv(path)
        if df_fa is not None:
            df_fa = df_fa.merge(meta_fa, on='participant_id', how='inner')
            if content_group and cg_map:
                df_fa = df_fa[df_fa['participant_id'].isin(
                    [pid for pid, cg in cg_map.items() if cg == content_group]
                )]
            if not df_fa.empty:
                r = _pearson_r(df_fa['eye_score'].values, df_fa[target_col].values)
                if r is not None:
                    fully_agg_baselines[fs] = r

    # Load first_p_agg results for each p
    # (EyeScore doesn't have multiple methods, just one file per feature set)
    for p in p_values:
        meta_p = _load_metadata_for_target(data_base, dataset, AggTypes.FIRST_P, p_level, p, target_col)
        if meta_p is None:
            continue

        for fs in feature_sets:
            path = _get_eyescore_path(results_base, dataset, preview, AggTypes.FIRST_P, p_level, p, fs)
            df = _load_eyescore_csv(path)
            if df is not None:
                df = df.merge(meta_p, on='participant_id', how='inner')
                if content_group and cg_map:
                    df = df[df['participant_id'].isin(
                        [pid for pid, cg in cg_map.items() if cg == content_group]
                    )]
                if not df.empty:
                    r = _pearson_r(df['eye_score'].values, df[target_col].values)
                    if r is not None:
                        data[fs].append((p, r))

    title = f"{dataset} | {preview} | {p_level} | {TARGET_COL_DISPLAY_NAMES.get(target_col, target_col)}"
    if content_group:
        title += f" | Content Group {content_group}"

    cg_folder = f"content_group_{content_group}" if content_group else "all"
    save_path = save_dir / dataset / preview / target_col / AggTypes.FIRST_P / p_level / cg_folder / "pearson.png"

    x_range = _get_first_p_x_range(dataset, p_level)
    _plot_line_chart(data, "Number of Trials (p)", "Pearson r", title,
                    x_range, PEARSON_Y_RANGE, save_path,
                    fully_agg_baselines)


def plot_moving_p_agg_accuracy_pearson(
    results_base: Path, data_base: Path,
    dataset: str, preview: str,
    target_col: str, feature_sets: List[str],
    p_level: str, p_n: int, save_dir: Path,
    content_group: Optional[str] = None
) -> None:
    """Plot Pearson r per window_index for moving_p_agg EyeScore."""
    if target_col in PARTIAL_TARGET_COLS:
        return

    data = {fs: [] for fs in feature_sets}
    fully_agg_baselines = {fs: None for fs in feature_sets}
    cg_map = _get_content_group_map(data_base, dataset) if dataset == "OneStop" else {}

    # Load fully_agg baseline for each feature set
    meta_fa = _load_metadata_for_target(data_base, dataset, AggTypes.FULL, "paragraph", 1, target_col)
    if meta_fa is None:
        logger.warning(f"Could not load metadata for fully_agg {dataset}/{target_col}")
        return

    for fs in feature_sets:
        path = _get_eyescore_path(results_base, dataset, preview, AggTypes.FULL, "paragraph", 1, fs)
        df_fa = _load_eyescore_csv(path)
        if df_fa is not None:
            df_fa = df_fa.merge(meta_fa, on='participant_id', how='inner')
            if content_group and cg_map:
                df_fa = df_fa[df_fa['participant_id'].isin(
                    [pid for pid, cg in cg_map.items() if cg == content_group]
                )]
            if not df_fa.empty:
                r = _pearson_r(df_fa['eye_score'].values, df_fa[target_col].values)
                if r is not None:
                    fully_agg_baselines[fs] = r

    # Load moving_p_agg results
    meta_p = _load_metadata_for_target(data_base, dataset, AggTypes.MOVING_P, p_level, p_n, target_col)
    if meta_p is None:
        logger.warning(f"Could not load metadata for moving_p_agg {dataset}/{target_col}/{p_level}_{p_n}")
        return

    for fs in feature_sets:
        path = _get_eyescore_path(results_base, dataset, preview, AggTypes.MOVING_P, p_level, p_n, fs)
        df = _load_eyescore_csv(path)
        if df is not None:
            df = df.merge(meta_p, on='participant_id', how='inner')
            if content_group and cg_map:
                df = df[df['participant_id'].isin(
                    [pid for pid, cg in cg_map.items() if cg == content_group]
                )]
            if not df.empty:
                df['window_index'] = _get_window_index(df)
                for win_idx in sorted(df['window_index'].unique()):
                    df_win = df[df['window_index'] == win_idx]
                    r = _pearson_r(df_win['eye_score'].values, df_win[target_col].values)
                    if r is not None:
                        data[fs].append((win_idx, r))

    title = f"{dataset} | {preview} | {p_level}_{p_n} | {TARGET_COL_DISPLAY_NAMES.get(target_col, target_col)}"
    if content_group:
        title += f" | Content Group {content_group}"

    cg_folder = f"content_group_{content_group}" if content_group else "all"
    save_path = save_dir / dataset / preview / target_col / AggTypes.MOVING_P / f"{p_level}_{p_n}" / cg_folder / "pearson.png"

    max_windows = max([x[0] for xs in data.values() for x in xs], default=8)
    x_range = (-0.5, max_windows + 0.5)
    _plot_line_chart(data, "Window Index", "Pearson r", title,
                    x_range, PEARSON_Y_RANGE, save_path,
                    fully_agg_baselines)


def plot_moving_p_agg_bias(
    results_base: Path, data_base: Path,
    dataset: str, preview: str,
    target_col: str, feature_sets: List[str],
    p_level: str, p_n: int, save_dir: Path,
    content_group: Optional[str] = None
) -> None:
    """Plot EyeScore bias (mean_eye_score - fully_agg_mean) per window_index."""
    if target_col in PARTIAL_TARGET_COLS:
        return

    data = {fs: [] for fs in feature_sets}
    fully_agg_mean_eye_scores = {fs: None for fs in feature_sets}
    cg_map = _get_content_group_map(data_base, dataset) if dataset == "OneStop" else {}

    # Load fully_agg baseline mean eye_score for each feature set
    meta_fa = _load_metadata_for_target(data_base, dataset, AggTypes.FULL, "paragraph", 1, target_col)
    if meta_fa is None:
        logger.warning(f"Could not load metadata for fully_agg {dataset}/{target_col}")
        return

    for fs in feature_sets:
        path = _get_eyescore_path(results_base, dataset, preview, AggTypes.FULL, "paragraph", 1, fs)
        df_fa = _load_eyescore_csv(path)
        if df_fa is not None:
            df_fa = df_fa.merge(meta_fa, on='participant_id', how='inner')
            if content_group and cg_map:
                df_fa = df_fa[df_fa['participant_id'].isin(
                    [pid for pid, cg in cg_map.items() if cg == content_group]
                )]
            if not df_fa.empty:
                fully_agg_mean_eye_scores[fs] = df_fa['eye_score'].mean()

    if all(v is None for v in fully_agg_mean_eye_scores.values()):
        logger.warning(f"Could not compute fully_agg baseline for {dataset}/{target_col}")
        return

    # Load moving_p_agg results
    meta_p = _load_metadata_for_target(data_base, dataset, AggTypes.MOVING_P, p_level, p_n, target_col)
    if meta_p is None:
        return

    for fs in feature_sets:
        path = _get_eyescore_path(results_base, dataset, preview, AggTypes.MOVING_P, p_level, p_n, fs)
        df = _load_eyescore_csv(path)
        if df is not None and fully_agg_mean_eye_scores[fs] is not None:
            df = df.merge(meta_p, on='participant_id', how='inner')
            if content_group and cg_map:
                df = df[df['participant_id'].isin(
                    [pid for pid, cg in cg_map.items() if cg == content_group]
                )]
            if not df.empty:
                df['window_index'] = _get_window_index(df)
                for win_idx in sorted(df['window_index'].unique()):
                    df_win = df[df['window_index'] == win_idx]
                    mean_eye_score = df_win['eye_score'].mean()
                    bias = mean_eye_score - fully_agg_mean_eye_scores[fs]
                    data[fs].append((win_idx, bias))

    title = f"{dataset} | {preview} | {p_level}_{p_n} | {TARGET_COL_DISPLAY_NAMES.get(target_col, target_col)} | Bias"
    if content_group:
        title += f" | Content Group {content_group}"

    cg_folder = f"content_group_{content_group}" if content_group else "all"
    save_path = save_dir / dataset / preview / target_col / AggTypes.MOVING_P / f"{p_level}_{p_n}" / cg_folder / "bias.png"

    max_windows = max([x[0] for xs in data.values() for x in xs], default=8)
    x_range = (-0.5, max_windows + 0.5)
    _plot_line_chart(data, "Window Index", "EyeScore Bias", title,
                    x_range, BIAS_Y_RANGE, save_path,
                    {fs: 0 for fs in feature_sets})


def run_for_dataset(
    results_base: Path, data_base: Path,
    dataset: str, preview: str,
    target_cols: List[str], feature_sets: List[str],
    plots_dir: Path,
    pbar=None
) -> None:
    """Run all EyeScore plots for a dataset."""
    logger.info(f"Running EyeScore plots for {dataset}/{preview}")

    for target_col in target_cols:
        if target_col in PARTIAL_TARGET_COLS:
            logger.info(f"Skipping partial target {target_col}")
            continue

        for p_level in ["paragraph", "article"]:
            p_values = _get_first_p_values(dataset, p_level)
            if not p_values:
                continue

            # first_p_agg plots
            plot_first_p_agg_pearson(results_base, data_base, dataset, preview,
                                    target_col, feature_sets, p_level, plots_dir)
            if pbar:
                pbar.update(1)

            # moving_p_agg plots
            moving_p_values = _get_moving_p_values(dataset, p_level)
            for p_n in moving_p_values:
                plot_moving_p_agg_accuracy_pearson(results_base, data_base, dataset, preview,
                                                  target_col, feature_sets, p_level, p_n, plots_dir)
                if pbar:
                    pbar.update(1)
                plot_moving_p_agg_bias(results_base, data_base, dataset, preview,
                                     target_col, feature_sets, p_level, p_n, plots_dir)
                if pbar:
                    pbar.update(1)

            # OneStop content_group breakdowns
            if dataset == "OneStop":
                cg_map = _get_content_group_map(data_base, dataset)
                for cg in sorted(set(cg_map.values())):
                    plot_first_p_agg_pearson(results_base, data_base, dataset, preview,
                                            target_col, feature_sets, p_level, plots_dir, cg)
                    if pbar:
                        pbar.update(1)

                    for p_n in moving_p_values:
                        plot_moving_p_agg_accuracy_pearson(results_base, data_base, dataset, preview,
                                                          target_col, feature_sets, p_level, p_n, plots_dir, cg)
                        if pbar:
                            pbar.update(1)
                        plot_moving_p_agg_bias(results_base, data_base, dataset, preview,
                                             target_col, feature_sets, p_level, p_n, plots_dir, cg)
                        if pbar:
                            pbar.update(1)
