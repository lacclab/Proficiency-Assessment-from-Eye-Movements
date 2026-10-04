"""Correlations between the EyeScores of different feature groups.

For each (dataset, preview, version path) under src/methods/EyeScore/results,
merges the per-feature-group EyeScore CSVs by participant and saves the Pearson
and Spearman correlation matrices, plus heatmaps of those and of the mean
absolute differences between feature groups.

    python -m src.correlations.between_measures.eyescore
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from src.constants import DATA_PATH

logger.add(DATA_PATH / 'logs' / 'eyescore_correlations.log', rotation='10 MB', retention='7 days')

EYESCORE_RESULTS_DIR = Path("src/methods/EyeScore/results")
CORRELATION_RESULTS_DIR = Path("src/correlations/between_measures/results")
CORRELATION_PLOTS_DIR = Path("src/correlations/between_measures/plots")


def discover_version_paths(
    dataset: str,
    preview: str = "All",
) -> List[str]:
    """Discover all available version paths for a dataset.

    Args:
        dataset: Dataset name (e.g., 'OneStop', 'Meco')
        preview: Preview type ('All', 'Hunting', 'Gathering')

    Returns:
        List of version paths (e.g., ['fully_agg/all/all', 'fully_agg/ordinary/all']).
    """
    base_dir = EYESCORE_RESULTS_DIR / dataset / preview

    if not base_dir.exists():
        logger.warning(f"Directory {base_dir} does not exist.")
        return []

    csv_files = list(base_dir.rglob("*.csv"))

    if not csv_files:
        logger.warning(f"No eyescore files found in {base_dir}")
        return []

    # A version path is the directory of a feature-group CSV, relative to base_dir.
    version_paths = {str(csv_file.relative_to(base_dir).parent) for csv_file in csv_files}
    return sorted(version_paths)


def load_eyescore_files(
    dataset: str,
    preview: str = "All",
    version_path: Optional[str] = None,
    feature_groups: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load eyescore files for different feature groups.

    Args:
        dataset: Dataset name (e.g., 'OneStop', 'Meco')
        preview: Preview type ('All', 'Hunting', 'Gathering')
        version_path: Optional specific version path within the dataset.
                     If None, uses the first available version.
        feature_groups: List of feature group names to load. If None, loads all available.

    Returns:
        Dictionary mapping feature group names to their eyescore DataFrames.
    """
    eyescores = {}

    if version_path is None:
        available_paths = discover_version_paths(dataset, preview)
        if not available_paths:
            logger.error(f"No version paths found for {dataset}/{preview}")
            return {}
        version_path = available_paths[0]
        logger.info(f"No version_path specified. Using first available: {version_path}")

    base_dir = EYESCORE_RESULTS_DIR / dataset / preview / version_path

    if not base_dir.exists():
        logger.error(f"Version directory {base_dir} does not exist.")
        logger.info(f"Available versions: {discover_version_paths(dataset, preview)}")
        return {}

    csv_files = list(base_dir.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No eyescore CSV files found in {base_dir}")
        return {}

    for csv_file in csv_files:
        feature_group = csv_file.stem
        if feature_groups is not None and feature_group not in feature_groups:
            continue

        try:
            df = pd.read_csv(csv_file)
            eyescores[feature_group] = df
            logger.info(f"Loaded eyescore for feature group '{feature_group}': {len(df)} participants")
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")

    return eyescores


def merge_eyescores(eyescores: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge eyescore dataframes by participant ID.

    Args:
        eyescores: Dictionary mapping feature group names to eyescore DataFrames.

    Returns:
        Merged DataFrame with columns like: participant_id, L1, READING_SPEED, FIXATION_METRICS, etc.
    """
    if not eyescores:
        logger.error("No eyescore dataframes to merge")
        return pd.DataFrame()

    feature_groups = list(eyescores.keys())
    merged_df = eyescores[feature_groups[0]][["participant_id", "L1"]].copy()
    merged_df = merged_df.set_index("participant_id")

    # One eye_score column per feature group
    for feature_group, df in eyescores.items():
        eyescore_series = df.set_index("participant_id")["eye_score"]
        eyescore_series.name = feature_group
        merged_df = merged_df.join(eyescore_series, how='inner')

    merged_df = merged_df.reset_index()

    # Drop participants missing any EyeScore (a missing L1 is fine)
    eyescore_cols = [col for col in merged_df.columns if col not in ["participant_id", "L1"]]
    merged_df = merged_df.dropna(subset=eyescore_cols)

    logger.info(f"Merged eyescores for {len(merged_df)} participants across {len(feature_groups)} feature groups")
    return merged_df


def compute_correlations(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlations between eyescores.

    Args:
        merged_df: DataFrame with participant_id, L1, and eyescore columns for each feature group.

    Returns:
        Tuple of (pearson_corr_matrix, spearman_corr_matrix)
    """
    eyescore_cols = [col for col in merged_df.columns if col not in ["participant_id", "L1"]]

    if len(eyescore_cols) < 2:
        logger.warning("Need at least 2 feature groups to compute correlations")
        return pd.DataFrame(), pd.DataFrame()

    pearson_corr = merged_df[eyescore_cols].corr(method='pearson')
    spearman_corr = merged_df[eyescore_cols].corr(method='spearman')

    logger.info(f"Computed correlations between {len(eyescore_cols)} feature groups")
    return pearson_corr, spearman_corr


def compute_correlations_per_language(merged_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Compute correlations separately for each language.

    Args:
        merged_df: DataFrame with participant_id, L1, and eyescore columns.

    Returns:
        Tuple of (pearson_corr_by_lang, spearman_corr_by_lang) dictionaries.
    """
    eyescore_cols = [col for col in merged_df.columns if col not in ["participant_id", "L1"]]

    pearson_by_lang = {}
    spearman_by_lang = {}

    for lang, group_df in merged_df.groupby("L1"):
        if len(group_df) < 3:  # Need at least 3 points for correlation
            logger.warning(f"Skipping language {lang}: only {len(group_df)} participants")
            continue

        pearson_by_lang[lang] = group_df[eyescore_cols].corr(method='pearson')
        spearman_by_lang[lang] = group_df[eyescore_cols].corr(method='spearman')
        logger.info(f"Computed correlations for language {lang} ({len(group_df)} participants)")

    return pearson_by_lang, spearman_by_lang


def _plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    save_path: Optional[Path],
    figsize: Optional[Tuple[int, int]],
    **heatmap_kwargs,
) -> None:
    """Annotated square heatmap of a feature-group x feature-group matrix,
    saved to save_path (or shown if None). figsize defaults to a size that
    scales with the number of feature groups."""
    if figsize is None:
        base_size = max(6, len(matrix) * 0.8)
        figsize = (base_size, base_size)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    labels = [_shorten_feature_name(col) for col in matrix.columns]

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 8},
        **heatmap_kwargs,
    )

    plt.title(title, fontsize=12, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_mean_differences_heatmap(
    diff_matrix: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """Heatmap of the mean absolute differences between EyeScores."""
    _plot_heatmap(diff_matrix, title, save_path, figsize,
                  cmap="YlOrRd", cbar_kws={"label": "Mean Absolute Difference"})


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """Heatmap of a correlation matrix, on a fixed [-1, 1] color scale."""
    _plot_heatmap(corr_matrix, title, save_path, figsize,
                  cmap="coolwarm", center=0, vmin=-1, vmax=1,
                  cbar_kws={"label": "Correlation"})


def compute_mean_differences(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean absolute differences between eyescores for all pairs.

    Args:
        merged_df: DataFrame with participant_id, L1, and eyescore columns.

    Returns:
        Matrix of mean absolute differences between feature groups.
    """
    eyescore_cols = [col for col in merged_df.columns if col not in ["participant_id", "L1"]]

    diff_matrix = pd.DataFrame(
        index=eyescore_cols,
        columns=eyescore_cols,
        dtype=float
    )

    for i, col1 in enumerate(eyescore_cols):
        for j, col2 in enumerate(eyescore_cols):
            if i == j:
                diff_matrix.loc[col1, col2] = 0.0
            else:
                mean_abs_diff = (merged_df[col1] - merged_df[col2]).abs().mean()
                diff_matrix.loc[col1, col2] = mean_abs_diff

    return diff_matrix.astype(float)


def _shorten_feature_name(name: str) -> str:
    """Abbreviate a feature group name for plot labels (e.g. READING_SPEED -> RS)."""
    abbreviations = {
        'READING_SPEED': 'RS',
        'FIXATION_METRICS': 'FM',
        'S_CLUSTERS_NO_NORM': 'SC',
        'WP_COEFS_NO_NORM': 'WP',
    }

    result = name
    for full, abbr in abbreviations.items():
        result = result.replace(full, abbr)

    return result.replace('_', '')


def save_correlations_to_csv(
    pearson_corr: pd.DataFrame,
    spearman_corr: pd.DataFrame,
    dataset: str,
    preview: str,
    version_path: Optional[str] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """Save correlation matrices to CSV files.

    Args:
        pearson_corr: Pearson correlation matrix.
        spearman_corr: Spearman correlation matrix.
        dataset: Dataset name.
        preview: Preview type.
        version_path: Version path (optional).
        save_dir: Directory to save files. If None, uses default.
    """
    if save_dir is None:
        save_dir = CORRELATION_RESULTS_DIR

    output_dir = save_dir / dataset / preview
    if version_path:
        output_dir = output_dir / version_path

    output_dir.mkdir(parents=True, exist_ok=True)

    pearson_path = output_dir / "eyescore_correlations_pearson.csv"
    spearman_path = output_dir / "eyescore_correlations_spearman.csv"

    pearson_corr.to_csv(pearson_path)
    spearman_corr.to_csv(spearman_path)

    logger.info(f"Saved correlations to {output_dir}")


def analyze_eyescore_correlations(
    dataset: str,
    preview: str = "All",
    version_path: Optional[str] = None,
    feature_groups: Optional[List[str]] = None,
    save_plots: bool = True,
    save_csv: bool = True,
    run_per_language: bool = False,
) -> Dict:
    """Complete analysis of eyescore correlations.

    Args:
        dataset: Dataset name (e.g., 'OneStop', 'Meco').
        preview: Preview type ('All', 'Hunting', 'Gathering').
        version_path: Optional specific version path.
        feature_groups: List of feature groups to include.
        save_plots: Whether to save correlation heatmaps.
        save_csv: Whether to save correlation matrices to CSV.
        run_per_language: Also compute (and plot) correlations within each L1.

    Returns:
        Dictionary with merged_df, pearson_corr, spearman_corr and
        mean_diff_matrix, plus pearson_by_lang / spearman_by_lang when
        run_per_language is set.
    """
    logger.info(f"Analyzing eyescore correlations for {dataset}/{preview}")

    eyescores = load_eyescore_files(dataset, preview, version_path, feature_groups)
    if not eyescores:
        logger.error("No eyescore data loaded")
        return {}

    merged_df = merge_eyescores(eyescores)
    if merged_df.empty:
        logger.error("Failed to merge eyescore dataframes")
        return {}

    pearson_corr, spearman_corr = compute_correlations(merged_df)
    mean_diff_matrix = compute_mean_differences(merged_df)

    if run_per_language:
        pearson_by_lang, spearman_by_lang = compute_correlations_per_language(merged_df)

    if save_csv:
        save_correlations_to_csv(
            pearson_corr, spearman_corr, dataset, preview, version_path
        )

    if save_plots:
        plot_dir = CORRELATION_PLOTS_DIR / dataset / preview
        if version_path:
            plot_dir = plot_dir / version_path

        if not pearson_corr.empty:
            plot_correlation_heatmap(
                pearson_corr,
                f"{dataset}/{preview} - Eyescore Pearson Correlations",
                save_path=plot_dir / "eyescore_correlations_pearson.png",
            )

        if not spearman_corr.empty:
            plot_correlation_heatmap(
                spearman_corr,
                f"{dataset}/{preview} - Eyescore Spearman Correlations",
                save_path=plot_dir / "eyescore_correlations_spearman.png",
            )

        if not mean_diff_matrix.empty:
            plot_mean_differences_heatmap(
                mean_diff_matrix,
                f"{dataset}/{preview} - Mean Absolute Differences Between Eyescores",
                save_path=plot_dir / "eyescore_mean_differences.png",
            )

        if run_per_language:
            for lang, corr_matrix in pearson_by_lang.items():
                plot_correlation_heatmap(
                    corr_matrix,
                    f"{dataset}/{preview} - {lang} - Eyescore Pearson Correlations",
                    save_path=plot_dir / f"eyescore_correlations_pearson_{lang}.png",
                )

    results = {
        "merged_df": merged_df,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "mean_diff_matrix": mean_diff_matrix,
    }
    if run_per_language:
        results["pearson_by_lang"] = pearson_by_lang
        results["spearman_by_lang"] = spearman_by_lang
    return results


def analyze_all_eyescore_correlations(
    save_plots: bool = True,
    save_csv: bool = True,
) -> Dict:
    """Analyze eyescore correlations for all available datasets, previews, and difficulty levels.

    Returns:
        Dictionary mapping (dataset, preview, version_path) tuples to their results.
    """
    all_results = {}
    results_base_dir = EYESCORE_RESULTS_DIR

    if not results_base_dir.exists():
        logger.error(f"Results directory {results_base_dir} does not exist.")
        return {}

    dataset_dirs = [d for d in results_base_dir.iterdir() if d.is_dir()]

    for dataset_dir in sorted(dataset_dirs):
        dataset = dataset_dir.name
        preview_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

        for preview_dir in sorted(preview_dirs):
            preview = preview_dir.name
            version_paths = discover_version_paths(dataset, preview)

            for version_path in version_paths:
                try:
                    logger.info(f"Analyzing {dataset}/{preview}/{version_path}...")
                    results = analyze_eyescore_correlations(
                        dataset=dataset,
                        preview=preview,
                        version_path=version_path,
                        save_plots=save_plots,
                        save_csv=save_csv,
                    )
                    if results:
                        all_results[(dataset, preview, version_path)] = results
                        logger.info(f"✓ Completed {dataset}/{preview}/{version_path}")
                except Exception as e:
                    logger.warning(f"Failed to analyze {dataset}/{preview}/{version_path}: {e}")

    logger.info(f"Analysis complete. Processed {len(all_results)} dataset/preview/version combinations")
    return all_results


if __name__ == "__main__":
    all_results = analyze_all_eyescore_correlations(
        save_plots=True,
        save_csv=True,
    )

    print(f"\n\nProcessed {len(all_results)} dataset/preview/difficulty combinations:\n")
    for (dataset, preview, version_path), results in sorted(all_results.items()):
        print(f"\n{dataset}/{preview}/{version_path} - {len(results['merged_df'])} participants")
        print("Pearson Correlations (sample):")
        print(results["pearson_corr"].iloc[:3, :3])  # Print top-left corner
