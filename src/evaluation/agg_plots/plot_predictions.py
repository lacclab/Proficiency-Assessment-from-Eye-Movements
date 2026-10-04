"""Evaluation plots for first_p_agg and moving_p_agg prediction regimes."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from scipy import stats

from src.constants import AggTypes, TrainTypes
from src.agg_configs import AGG_CONFIGS_BY_DATASET
from src.evaluation.predictions.plot import FEATURE_SET_DISPLAY_NAMES, TARGET_COL_DISPLAY_NAMES

logger = logging.getLogger(__name__)

PLOTS_SAVE_DIR = Path("src/evaluation/agg_plots/plots/predictions")

# Fixed axis ranges (same across all plots for comparability)
PEARSON_Y_RANGE = (0.0, 0.8)
MAE_Y_RANGE_LEXTALE = (0, 25)  # 0-100 scale
MAE_Y_RANGE_PROFICIENCY = (0, 25)  # 0-100 scale
BIAS_Y_RANGE = (-20, 20)

# X-axis ranges for first_p_agg
FIRST_P_X_RANGE_PARAGRAPH_MECO = (0, 13)
FIRST_P_X_RANGE_PARAGRAPH_ONESTOP = (0, 55)
FIRST_P_X_RANGE_ARTICLE_ONESTOP = (0, 11)

# Color palette (fixed mapping)
FEATURE_SET_COLORS = {
    "READING_SPEED": sns.color_palette("tab10")[0],
    "FIXATION_METRICS": sns.color_palette("tab10")[1],
    "S_CLUSTERS_NO_NORM": sns.color_palette("tab10")[2],
    "WP_COEFS_NO_NORM": sns.color_palette("tab10")[3],
    "READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM": sns.color_palette("tab10")[4],
    "TRANSITIONS": sns.color_palette("tab10")[5],
    "WFC": sns.color_palette("tab10")[6],
}

# Partial target columns to skip (sub-scores)
PARTIAL_TARGET_COLS = {
    "MPT_listening_score", "MPT_grammar_score", "MPT_vocabulary_score", "MPT_reading_score",
    "MPT_listen_grammar", "MPT_vocab_read", "MPT_grammar_vocab_read"
}


def _get_window_index(df: pd.DataFrame) -> np.ndarray:
    """Get window_index from df; fall back to cumcount if not present."""
    if 'window_index' in df.columns:
        return df['window_index'].values
    return df.groupby('participant_id').cumcount().values


def _pearson_r(true_vals: np.ndarray, pred_vals: np.ndarray) -> Optional[float]:
    """Compute Pearson r; return None if unable to compute."""
    mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    if mask.sum() < 3:
        return None
    try:
        r, _ = stats.pearsonr(true_vals[mask], pred_vals[mask])
        return r
    except (ValueError, RuntimeError):
        return None


def _mae(true_vals: np.ndarray, pred_vals: np.ndarray) -> Optional[float]:
    """Compute MAE; return None if unable to compute."""
    mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    if mask.sum() < 1:
        return None
    return float(np.mean(np.abs(true_vals[mask] - pred_vals[mask])))


def _load_pred_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load prediction CSV; return None if file doesn't exist or is invalid."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or 'true' not in df.columns or 'pred' not in df.columns:
            return None
        return df
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def _get_first_p_values(dataset: str, p_level: str) -> List[int]:
    """Get all p values for first_p_agg from agg_configs."""
    dataset_key = f"{dataset}L2" if not dataset.endswith(('L1', 'L2')) else dataset
    configs = AGG_CONFIGS_BY_DATASET.get(dataset_key, [])
    p_values = []
    for config in configs:
        if config.get('agg_type') == AggTypes.FIRST_P and config.get('p_agg_level') == p_level:
            p_values.append(config['p_agg'])
    return sorted(set(p_values))


def _get_moving_p_values(dataset: str, p_level: str) -> List[int]:
    """Get all p values for moving_p_agg from agg_configs."""
    dataset_key = f"{dataset}L2" if not dataset.endswith(('L1', 'L2')) else dataset
    configs = AGG_CONFIGS_BY_DATASET.get(dataset_key, [])
    p_values = []
    for config in configs:
        if config.get('agg_type') == AggTypes.MOVING_P and config.get('p_agg_level') == p_level:
            p_values.append(config['p_agg'])
    return sorted(set(p_values))


def _get_batch_cv_methods(results_base: Path, dataset: str, aug_p: float, preview: str) -> List[str]:
    """Scan for SEEN / UNSEEN result files and return canonical method names.

    Covers both the new Pool × FoldMethod convention ({seen,unseen}__*.csv)
    and the legacy batch_cross_validation_*.csv files; legacy names are
    normalized to their canonical equivalent via evaluation.normalize_method.
    """
    from src.evaluation.predictions.evaluation import normalize_method

    methods = []
    base_path = results_base / dataset / f"aug_p={aug_p}" / preview
    if base_path.exists():
        patterns = ("seen__*.csv", "unseen__*.csv", "batch_cross_validation*.csv")
        for pattern in patterns:
            for csv_file in base_path.rglob(pattern):
                method_name = normalize_method(csv_file.stem)
                if method_name not in methods:
                    methods.append(method_name)
    return sorted(methods)


def _get_content_group_map(data_base: Path, dataset: str) -> Dict[str, str]:
    """Load content_group mapping from fully_agg metadata for OneStop."""
    if dataset != "OneStop":
        return {}

    dataset_full = f"{dataset}L2"
    meta_path = data_base / dataset_full / "features_and_targets" / AggTypes.FULL / "ordinary" / "all" / "features_and_metadata.csv"

    if not meta_path.exists():
        logger.warning(f"Content group metadata not found: {meta_path}")
        return {}

    try:
        meta = pd.read_csv(meta_path, usecols=['participant_id', 'content_group'])
        return dict(zip(meta['participant_id'], meta['content_group']))
    except Exception as e:
        logger.warning(f"Failed to load content group metadata: {e}")
        return {}


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


def _get_mae_y_range(target_col: str) -> Tuple[float, float]:
    """Get y-axis range for MAE plots based on target column."""
    if target_col == "proficiency_agg":
        return MAE_Y_RANGE_PROFICIENCY
    else:
        return MAE_Y_RANGE_LEXTALE


def _plot_line_chart(
    data: Dict[str, List[Tuple[float, float]]],
    x_label: str, y_label: str, title: str,
    x_range: Tuple[float, float], y_range: Tuple[float, float],
    save_path: Path,
    fully_agg_baselines: Optional[Dict[str, Optional[float]]] = None,
    x_ticks: Optional[List[float]] = None
) -> None:
    """Generic line chart for feature_sets across x values."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for feature_set, points in data.items():
        if not points:
            continue
        points = sorted(points, key=lambda p: p[0])
        xs, ys = zip(*points)
        color = FEATURE_SET_COLORS.get(feature_set, 'gray')
        display_name = FEATURE_SET_DISPLAY_NAMES.get(feature_set, feature_set)
        ax.plot(xs, ys, marker='o', label=display_name, color=color, linewidth=2, markersize=6)

    # Baseline fully aggregated reference line
    if fully_agg_baselines:
        # Find first non-None baseline value for the legend line
        first_baseline = next((v for v in fully_agg_baselines.values() if v is not None), None)
        if first_baseline is not None:
            ax.axhline(y=first_baseline, color='black', linestyle='--', linewidth=2, label="Baseline Fully Aggregated")

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='best', framealpha=0.9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {save_path}")


def plot_first_p_agg_pearson(
    results_base: Path, data_base: Path,
    dataset: str, aug_p: float, preview: str,
    target_col: str, feature_sets: List[str], model_name: str, methods: List[str],
    p_level: str, save_dir: Path,
    content_group: Optional[str] = None,
    train_type: str = TrainTypes.PARTIAL,
    pbar=None
) -> None:
    """Plot Pearson r vs p for first_p_agg predictions."""
    if target_col in PARTIAL_TARGET_COLS:
        logger.info(f"Skipping partial target column {target_col}")
        return

    p_values = _get_first_p_values(dataset, p_level)
    if not p_values:
        logger.warning(f"No first_p_agg configs found for {dataset}/{p_level}")
        return

    cg_map = _get_content_group_map(data_base, dataset) if content_group else {}

    # Generate separate plots for each method
    cg_folder = f"content_group_{content_group}" if content_group else "all"
    train_type_name = "partial" if train_type == TrainTypes.PARTIAL else "full"
    train_type_display = "Train on Partial" if train_type == TrainTypes.PARTIAL else "Train on Full"

    for method in methods:
        data = {fs: [] for fs in feature_sets}
        fully_agg_baselines = {fs: None for fs in feature_sets}

        # Load fully_agg baseline for this specific method
        fa_path = (results_base / dataset / f"aug_p={aug_p}" / preview / AggTypes.FULL / target_col)
        for fs in feature_sets:
            fully_agg_csv = fa_path / fs / model_name / f"{method}.csv"
            df_fa = _load_pred_csv(fully_agg_csv)
            if df_fa is not None:
                if content_group and cg_map:
                    df_fa = df_fa[df_fa['participant_id'].isin(
                        [pid for pid, cg in cg_map.items() if cg == content_group]
                    )]
                if not df_fa.empty:
                    fully_agg_baselines[fs] = _pearson_r(df_fa['true'].values, df_fa['pred'].values)

        # Load first_p_agg results for each p for this specific method
        for p in p_values:
            for fs in feature_sets:
                path = (results_base / dataset / f"aug_p={aug_p}" / preview /
                       AggTypes.FIRST_P / f"{p_level}_{p}" / target_col / fs / model_name / train_type / f"{method}.csv")
                df = _load_pred_csv(path)
                if df is not None:
                    if content_group and cg_map:
                        df = df[df['participant_id'].isin(
                            [pid for pid, cg in cg_map.items() if cg == content_group]
                        )]
                    if not df.empty:
                        r = _pearson_r(df['true'].values, df['pred'].values)
                        if r is not None:
                            data[fs].append((p, r))

        # Generate plot for this method if we have data
        if any(data[fs] for fs in feature_sets):
            method_display = method.replace('_', ' ').title()
            title = f"{dataset} | {preview} | {p_level} | {TARGET_COL_DISPLAY_NAMES.get(target_col, target_col)} | {train_type_display} | {method_display}"
            if content_group:
                title += f" | Content Group {content_group}"

            save_path = save_dir / dataset / preview / target_col / AggTypes.FIRST_P / p_level / cg_folder / f"{method}_pearson_{train_type_name}.png"
            x_range = _get_first_p_x_range(dataset, p_level)
            # Extract unique p values for discrete x ticks
            all_p_values = sorted(set(x for fs in data.values() for x, y in fs if isinstance(x, (int, float))))
            _plot_line_chart(data, "Number of Trials (p)", "Pearson r", title,
                            x_range, PEARSON_Y_RANGE, save_path,
                            fully_agg_baselines, x_ticks=all_p_values)
            if pbar:
                pbar.update(1)


def plot_first_p_agg_mae(
    results_base: Path, data_base: Path,
    dataset: str, aug_p: float, preview: str,
    target_col: str, feature_sets: List[str], model_name: str, methods: List[str],
    p_level: str, save_dir: Path,
    content_group: Optional[str] = None,
    train_type: str = TrainTypes.PARTIAL,
    pbar=None
) -> None:
    """Plot MAE vs p for first_p_agg predictions."""
    if target_col in PARTIAL_TARGET_COLS:
        return

    p_values = _get_first_p_values(dataset, p_level)
    if not p_values:
        return

    cg_map = _get_content_group_map(data_base, dataset) if content_group else {}

    # Generate separate plots for each method
    cg_folder = f"content_group_{content_group}" if content_group else "all"
    train_type_name = "partial" if train_type == TrainTypes.PARTIAL else "full"
    train_type_display = "Train on Partial" if train_type == TrainTypes.PARTIAL else "Train on Full"

    for method in methods:
        data = {fs: [] for fs in feature_sets}
        fully_agg_baselines = {fs: None for fs in feature_sets}

        # Load fully_agg baseline for this specific method
        fa_path = (results_base / dataset / f"aug_p={aug_p}" / preview / AggTypes.FULL / target_col)
        for fs in feature_sets:
            fully_agg_csv = fa_path / fs / model_name / f"{method}.csv"
            df_fa = _load_pred_csv(fully_agg_csv)
            if df_fa is not None:
                if content_group and cg_map:
                    df_fa = df_fa[df_fa['participant_id'].isin(
                        [pid for pid, cg in cg_map.items() if cg == content_group]
                    )]
                if not df_fa.empty:
                    fully_agg_baselines[fs] = _mae(df_fa['true'].values, df_fa['pred'].values)

        # Load first_p_agg results for each p for this specific method
        for p in p_values:
            for fs in feature_sets:
                path = (results_base / dataset / f"aug_p={aug_p}" / preview /
                       AggTypes.FIRST_P / f"{p_level}_{p}" / target_col / fs / model_name / train_type / f"{method}.csv")
                df = _load_pred_csv(path)
                if df is not None:
                    if content_group and cg_map:
                        df = df[df['participant_id'].isin(
                            [pid for pid, cg in cg_map.items() if cg == content_group]
                        )]
                    if not df.empty:
                        mae = _mae(df['true'].values, df['pred'].values)
                        if mae is not None:
                            data[fs].append((p, mae))

        # Generate plot for this method if we have data
        if any(data[fs] for fs in feature_sets):
            method_display = method.replace('_', ' ').title()
            title = f"{dataset} | {preview} | {p_level} | {TARGET_COL_DISPLAY_NAMES.get(target_col, target_col)} | {train_type_display} | {method_display}"
            if content_group:
                title += f" | Content Group {content_group}"

            save_path = save_dir / dataset / preview / target_col / AggTypes.FIRST_P / p_level / cg_folder / f"{method}_mae_{train_type_name}.png"
            x_range = _get_first_p_x_range(dataset, p_level)
            y_range = _get_mae_y_range(target_col)
            # Extract unique p values for discrete x ticks
            all_p_values = sorted(set(x for fs in data.values() for x, y in fs if isinstance(x, (int, float))))
            _plot_line_chart(data, "Number of Trials (p)", "MAE", title,
                            x_range, y_range, save_path,
                            fully_agg_baselines, x_ticks=all_p_values)
            if pbar:
                pbar.update(1)


def plot_moving_p_agg_accuracy_pearson(
    results_base: Path, data_base: Path,
    dataset: str, aug_p: float, preview: str,
    target_col: str, feature_sets: List[str], model_name: str, methods: List[str],
    p_level: str, p_n: int, save_dir: Path,
    content_group: Optional[str] = None,
    train_type: str = TrainTypes.PARTIAL,
    pbar=None
) -> None:
    """Plot Pearson r per window_index for moving_p_agg predictions."""
    if target_col in PARTIAL_TARGET_COLS:
        return

    cg_map = _get_content_group_map(data_base, dataset) if content_group else {}

    # Generate separate plots for each method
    cg_folder = f"content_group_{content_group}" if content_group else "all"
    train_type_name = "partial" if train_type == TrainTypes.PARTIAL else "full"
    train_type_display = "Train on Partial" if train_type == TrainTypes.PARTIAL else "Train on Full"

    for method in methods:
        data = {fs: [] for fs in feature_sets}
        fully_agg_baselines = {fs: None for fs in feature_sets}

        # Load fully_agg baseline for this specific method
        fa_path = (results_base / dataset / f"aug_p={aug_p}" / preview / AggTypes.FULL / target_col)
        for fs in feature_sets:
            fully_agg_csv = fa_path / fs / model_name / f"{method}.csv"
            df_fa = _load_pred_csv(fully_agg_csv)
            if df_fa is not None:
                if content_group and cg_map:
                    df_fa = df_fa[df_fa['participant_id'].isin(
                        [pid for pid, cg in cg_map.items() if cg == content_group]
                    )]
                if not df_fa.empty:
                    fully_agg_baselines[fs] = _pearson_r(df_fa['true'].values, df_fa['pred'].values)

        # Load moving_p_agg results for this specific method
        for fs in feature_sets:
            path = (results_base / dataset / f"aug_p={aug_p}" / preview /
                   AggTypes.MOVING_P / f"{p_level}_{p_n}" / target_col / fs / model_name / train_type / f"{method}.csv")
            df = _load_pred_csv(path)
            if df is not None:
                if content_group and cg_map:
                    df = df[df['participant_id'].isin(
                        [pid for pid, cg in cg_map.items() if cg == content_group]
                    )]
                if not df.empty:
                    df['window_index'] = _get_window_index(df)
                    for win_idx in sorted(df['window_index'].unique()):
                        df_win = df[df['window_index'] == win_idx]
                        r = _pearson_r(df_win['true'].values, df_win['pred'].values)
                        if r is not None:
                            data[fs].append((win_idx, r))

        # Generate plot for this method if we have data
        if any(data[fs] for fs in feature_sets):
            method_display = method.replace('_', ' ').title()
            title = f"{dataset} | {preview} | {p_level}_{p_n} | {TARGET_COL_DISPLAY_NAMES.get(target_col, target_col)} | {train_type_display} | {method_display}"
            if content_group:
                title += f" | Content Group {content_group}"

            save_path = save_dir / dataset / preview / target_col / AggTypes.MOVING_P / f"{p_level}_{p_n}" / cg_folder / f"{method}_pearson_{train_type_name}.png"
            max_windows = max([x[0] for xs in data.values() for x in xs], default=8)
            x_range = (-0.5, max_windows + 0.5)
            _plot_line_chart(data, "Window Index", "Pearson r", title,
                            x_range, PEARSON_Y_RANGE, save_path,
                            fully_agg_baselines)
            if pbar:
                pbar.update(1)


def plot_moving_p_agg_accuracy_mae(
    results_base: Path, data_base: Path,
    dataset: str, aug_p: float, preview: str,
    target_col: str, feature_sets: List[str], model_name: str, methods: List[str],
    p_level: str, p_n: int, save_dir: Path,
    content_group: Optional[str] = None,
    train_type: str = TrainTypes.PARTIAL,
    pbar=None
) -> None:
    """Plot MAE per window_index for moving_p_agg predictions."""
    if target_col in PARTIAL_TARGET_COLS:
        return

    cg_map = _get_content_group_map(data_base, dataset) if content_group else {}

    # Generate separate plots for each method
    cg_folder = f"content_group_{content_group}" if content_group else "all"
    train_type_name = "partial" if train_type == TrainTypes.PARTIAL else "full"
    train_type_display = "Train on Partial" if train_type == TrainTypes.PARTIAL else "Train on Full"

    for method in methods:
        data = {fs: [] for fs in feature_sets}
        fully_agg_baselines = {fs: None for fs in feature_sets}

        # Load fully_agg baseline for this specific method
        fa_path = (results_base / dataset / f"aug_p={aug_p}" / preview / AggTypes.FULL / target_col)
        for fs in feature_sets:
            fully_agg_csv = fa_path / fs / model_name / f"{method}.csv"
            df_fa = _load_pred_csv(fully_agg_csv)
            if df_fa is not None:
                if content_group and cg_map:
                    df_fa = df_fa[df_fa['participant_id'].isin(
                        [pid for pid, cg in cg_map.items() if cg == content_group]
                    )]
                if not df_fa.empty:
                    fully_agg_baselines[fs] = _mae(df_fa['true'].values, df_fa['pred'].values)

        # Load moving_p_agg results for this specific method
        for fs in feature_sets:
            path = (results_base / dataset / f"aug_p={aug_p}" / preview /
                   AggTypes.MOVING_P / f"{p_level}_{p_n}" / target_col / fs / model_name / train_type / f"{method}.csv")
            df = _load_pred_csv(path)
            if df is not None:
                if content_group and cg_map:
                    df = df[df['participant_id'].isin(
                        [pid for pid, cg in cg_map.items() if cg == content_group]
                    )]
                if not df.empty:
                    df['window_index'] = _get_window_index(df)
                    for win_idx in sorted(df['window_index'].unique()):
                        df_win = df[df['window_index'] == win_idx]
                        mae = _mae(df_win['true'].values, df_win['pred'].values)
                        if mae is not None:
                            data[fs].append((win_idx, mae))

        # Generate plot for this method if we have data
        if any(data[fs] for fs in feature_sets):
            method_display = method.replace('_', ' ').title()
            title = f"{dataset} | {preview} | {p_level}_{p_n} | {TARGET_COL_DISPLAY_NAMES.get(target_col, target_col)} | {train_type_display} | {method_display}"
            if content_group:
                title += f" | Content Group {content_group}"

            save_path = save_dir / dataset / preview / target_col / AggTypes.MOVING_P / f"{p_level}_{p_n}" / cg_folder / f"{method}_mae_{train_type_name}.png"
            max_windows = max([x[0] for xs in data.values() for x in xs], default=8)
            x_range = (-0.5, max_windows + 0.5)
            y_range = _get_mae_y_range(target_col)
            _plot_line_chart(data, "Window Index", "MAE", title,
                            x_range, y_range, save_path,
                            fully_agg_baselines)
            if pbar:
                pbar.update(1)


def plot_moving_p_agg_bias(
    results_base: Path, data_base: Path,
    dataset: str, aug_p: float, preview: str,
    target_col: str, feature_sets: List[str], model_name: str, methods: List[str],
    p_level: str, p_n: int, save_dir: Path,
    content_group: Optional[str] = None,
    train_type: str = TrainTypes.PARTIAL,
    skip_existing: bool = False,
    pbar=None
) -> None:
    """Plot prediction bias (mean_pred - fully_agg_mean) per window_index."""
    if target_col in PARTIAL_TARGET_COLS:
        return

    cg_map = _get_content_group_map(data_base, dataset) if content_group else {}

    # Generate separate plots for each method
    cg_folder = f"content_group_{content_group}" if content_group else "all"
    train_type_name = "partial" if train_type == TrainTypes.PARTIAL else "full"
    train_type_display = "Train on Partial" if train_type == TrainTypes.PARTIAL else "Train on Full"

    for method in methods:
        data = {fs: [] for fs in feature_sets}

        # Load fully_agg baseline for this specific method (keep full df for filtering later)
        fa_path = (results_base / dataset / f"aug_p={aug_p}" / preview / AggTypes.FULL / target_col)
        fully_agg_dfs = {fs: None for fs in feature_sets}
        for fs in feature_sets:
            fully_agg_csv = fa_path / fs / model_name / f"{method}.csv"
            df_fa = _load_pred_csv(fully_agg_csv)
            if df_fa is not None:
                if content_group and cg_map:
                    df_fa = df_fa[df_fa['participant_id'].isin(
                        [pid for pid, cg in cg_map.items() if cg == content_group]
                    )]
                if not df_fa.empty:
                    fully_agg_dfs[fs] = df_fa

        # Load moving_p_agg results for this specific method
        for fs in feature_sets:
            if fully_agg_dfs[fs] is None:
                continue
            path = (results_base / dataset / f"aug_p={aug_p}" / preview /
                   AggTypes.MOVING_P / f"{p_level}_{p_n}" / target_col / fs / model_name / train_type / f"{method}.csv")
            df = _load_pred_csv(path)
            if df is not None:
                if content_group and cg_map:
                    df = df[df['participant_id'].isin(
                        [pid for pid, cg in cg_map.items() if cg == content_group]
                    )]
                if not df.empty:
                    df['window_index'] = _get_window_index(df)

                    # Filter fully_agg to only include participants in moving_p_agg
                    moving_participants = set(df['participant_id'].unique())
                    df_fa_filtered = fully_agg_dfs[fs][fully_agg_dfs[fs]['participant_id'].isin(moving_participants)]
                    fully_agg_mean_pred = df_fa_filtered['pred'].mean() if not df_fa_filtered.empty else None

                    if fully_agg_mean_pred is not None:
                        for win_idx in sorted(df['window_index'].unique()):
                            df_win = df[df['window_index'] == win_idx]
                            mean_pred = df_win['pred'].mean()
                            bias = mean_pred - fully_agg_mean_pred
                            data[fs].append((win_idx, bias))

        # Generate plot for this method if we have data
        if any(data[fs] for fs in feature_sets):
            method_display = method.replace('_', ' ').title()
            title = f"{dataset} | {preview} | {p_level}_{p_n} | {TARGET_COL_DISPLAY_NAMES.get(target_col, target_col)} | {train_type_display} | {method_display} | Bias"
            if content_group:
                title += f" | Content Group {content_group}"

            save_path = save_dir / dataset / preview / target_col / AggTypes.MOVING_P / f"{p_level}_{p_n}" / cg_folder / f"{method}_bias_{train_type_name}.png"

            # Skip if exists and skip_existing is True
            if skip_existing and save_path.exists():
                if pbar:
                    pbar.update(1)
                continue

            max_windows = max([x[0] for xs in data.values() for x in xs], default=8)
            x_range = (-0.5, max_windows + 0.5)
            _plot_line_chart(data, "Window Index", "Prediction Bias", title,
                            x_range, BIAS_Y_RANGE, save_path,
                            {fs: 0 for fs in feature_sets})
            if pbar:
                pbar.update(1)


def run_for_dataset(
    results_base: Path, data_base: Path,
    dataset: str, aug_p: float, preview: str,
    target_cols: List[str], feature_sets: List[str],
    model_name: str, methods: List[str],
    plots_dir: Path,
    skip_existing: bool = False,
    pbar=None
) -> None:
    """Run all prediction plots for a dataset."""
    logger.info(f"Running prediction plots for {dataset}/{preview}")

    for target_col in target_cols:
        if target_col in PARTIAL_TARGET_COLS:
            logger.info(f"Skipping partial target {target_col}")
            continue

        for p_level in ["paragraph", "article"]:
            p_values = _get_first_p_values(dataset, p_level)
            if not p_values:
                continue

            # Generate plots for both train types
            for train_type in [TrainTypes.PARTIAL, TrainTypes.FULL]:
                # first_p_agg plots
                plot_first_p_agg_pearson(results_base, data_base, dataset, aug_p, preview,
                                         target_col, feature_sets, model_name, methods, p_level, plots_dir,
                                         train_type=train_type, pbar=pbar)
                plot_first_p_agg_mae(results_base, data_base, dataset, aug_p, preview,
                                   target_col, feature_sets, model_name, methods, p_level, plots_dir,
                                   train_type=train_type, pbar=pbar)

                # moving_p_agg plots
                moving_p_values = _get_moving_p_values(dataset, p_level)
                for p_n in moving_p_values:
                    plot_moving_p_agg_accuracy_pearson(results_base, data_base, dataset, aug_p, preview,
                                                      target_col, feature_sets, model_name, methods, p_level, p_n, plots_dir,
                                                      train_type=train_type, pbar=pbar)
                    plot_moving_p_agg_accuracy_mae(results_base, data_base, dataset, aug_p, preview,
                                                 target_col, feature_sets, model_name, methods, p_level, p_n, plots_dir,
                                                 train_type=train_type, pbar=pbar)
                    plot_moving_p_agg_bias(results_base, data_base, dataset, aug_p, preview,
                                         target_col, feature_sets, model_name, methods, p_level, p_n, plots_dir,
                                         train_type=train_type, skip_existing=skip_existing, pbar=pbar)

                # OneStop content_group breakdowns
                if dataset == "OneStop":
                    cg_map = _get_content_group_map(data_base, dataset)
                    for cg in sorted(set(cg_map.values())):
                        plot_first_p_agg_pearson(results_base, data_base, dataset, aug_p, preview,
                                                 target_col, feature_sets, model_name, methods, p_level, plots_dir, cg,
                                                 train_type=train_type, pbar=pbar)
                        plot_first_p_agg_mae(results_base, data_base, dataset, aug_p, preview,
                                           target_col, feature_sets, model_name, methods, p_level, plots_dir, cg,
                                           train_type=train_type, pbar=pbar)

                        for p_n in moving_p_values:
                            plot_moving_p_agg_accuracy_pearson(results_base, data_base, dataset, aug_p, preview,
                                                              target_col, feature_sets, model_name, methods, p_level, p_n, plots_dir, cg,
                                                              train_type=train_type, pbar=pbar)
                            plot_moving_p_agg_accuracy_mae(results_base, data_base, dataset, aug_p, preview,
                                                         target_col, feature_sets, model_name, methods, p_level, p_n, plots_dir, cg,
                                                         train_type=train_type, pbar=pbar)
                            plot_moving_p_agg_bias(results_base, data_base, dataset, aug_p, preview,
                                                 target_col, feature_sets, model_name, methods, p_level, p_n, plots_dir, cg,
                                                 train_type=train_type, pbar=pbar)
