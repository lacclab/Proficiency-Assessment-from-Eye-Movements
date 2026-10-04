"""Prediction evaluation plots (per-method bar charts and per-language
heatmaps). Used by the first_p_agg / moving_p_agg plots in agg_plots/."""
import contextlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from pathlib import Path
import logging
from tqdm import tqdm
from joblib import Parallel, delayed


from src.evaluation.utils import suppress_titles
from src.evaluation.predictions.evaluation import EVAL_SAVE_DIR
from src.constants import ALL_TESTS, RELEVANT_FEATURE_GROUPS_COMBINATIONS_MINIMAL
from src.configs import MODELS

logger = logging.getLogger(__name__)

PLOTS_SAVE_DIR = Path("src/evaluation/predictions/plots")

BAR_METRIC_TITLES = {
    "pearson_r": "Pearson r",
    "spearman_r": "Spearman r",
    "r2": "R²",
    "adjusted_r2": "Adjusted R²",
    "rmse": "RMSE",
    "mae": "MAE",
}
# Metrics where higher is better (for color normalization)
HIGHER_IS_BETTER = {"pearson_r", "spearman_r", "r2", "adjusted_r2"}

SIMPLIFIED_METRIC_TITLES = {
    "pearson_r": "Pearson r",
    "rmse": "RMSE",
}
# Fixed color ranges for simplified plots
FIXED_COLOR_RANGES = {
    "pearson_r": (0.4, 0.7),
    "rmse": (7.0, 13.0),
}

# ── Display name mappings (fill in your official names) ──────────────────────
FEATURE_SET_DISPLAY_NAMES = {
    "READING_SPEED": "Reading Speed",
    "FIXATION_METRICS": "Fixation Metrics",
    "S_CLUSTERS": "Syntactic Clusters",
    "S_CLUSTERS_NO_NORM": "Syntactic Clusters (unnorm.)",
    "WP_COEFS": "Word-Property Coefficients",
    "WP_COEFS_NO_NORM": "Word-Property Coef. (unnorm.)",
    "WP_COEFS_NO_INTERCEPT": "Word-Property Coef. (no intercept)",
    "WP_COEFS_NO_NORM_NO_INTERCEPT": "Word-Property Coef. (unnorm., no intercept)",
    "TRANSITIONS": "Transitions",
    "WFC": "WFC",
    # Combinations
    "READING_SPEED_FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM": "Speed + Fixation + Syntactic + WP Coef. (all unnorm.)",
    "FIXATION_METRICS_S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM": "Fixation + Syntactic + WP Coef. (all unnorm.)",
    "S_CLUSTERS_NO_NORM_WP_COEFS_NO_NORM": "Syntactic + WP Coef. (unnorm.)",
}

TARGET_COL_DISPLAY_NAMES = {
    "michtest_score": "Michigan Test (total)",
    "lextale_score": "LexTALE",
    "MPT_listening_score": "MPT Listening",
    "MPT_grammar_score": "MPT Grammar",
    "MPT_vocabulary_score": "MPT Vocabulary",
    "MPT_reading_score": "MPT Reading",
    "MPT_listen_grammar": "MPT Listen + Grammar",
    "MPT_vocab_read": "MPT Vocab + Reading",
    "MPT_grammar_vocab_read": "MPT Grammar + Vocab + Reading",
    "comprehension_score-regular_trials": "Comprehension (regular)",
    "comprehension_score-repeated_reading": "Comprehension (reread)",
    "proficiency_agg": "Composite",
}

MODEL_DISPLAY_NAMES = {
    "Ridge_Classifier": "Ridge Regression",
    "Log_Ridge_Regression": "Log Ridge Regression",
}

PREVIEW_DISPLAY_NAMES = {
    "All": "All Conditions",
    "Hunting": "Hunting (preview)",
    "Gathering": "Gathering (no preview)",
}

METHOD_DISPLAY_NAMES = {
    # Canonical Pool × FoldMethod names (post evaluation.normalize_method).
    "all__pool_all": "LOPO",
    "all__lolo": "LOLO",
    "all__l1_strat_kfold": "L1-Strat",
    "seen__pool_all": "SEEN",
    "seen__lolo": "SEEN (LOLO)",
    "seen__l1_strat_kfold": "SEEN (L1-Strat)",
    "unseen__pool_all": "UNSEEN",
    "unseen__lolo": "UNSEEN (LOLO)",
}

# Backward-compat alias: name was historically about batch frameworks; kept
# pointing at the same dict so other modules that imported it still work.
BATCH_FRAMEWORK_DISPLAY_NAMES = METHOD_DISPLAY_NAMES

def _display_name(value: str, mapping: dict) -> str:
    return mapping.get(value, value)


def _build_title_oneline(dataset: str, preview: str,
                         version_path: str, target_col: str, model_name: str) -> str:
    parts = [
        f"Dataset: {dataset}",
        f"Cond: {_display_name(preview, PREVIEW_DISPLAY_NAMES)}",
        f"Ver: {version_path}",
        f"Target: {_display_name(target_col, TARGET_COL_DISPLAY_NAMES)}",
        f"Model: {_display_name(model_name, MODEL_DISPLAY_NAMES)}",
    ]
    return "  |  ".join(parts)


def _build_title_multiline(dataset: str, preview: str,
                           version_path: str, target_col: str, model_name: str) -> str:
    line1 = f"Dataset: {dataset}  |  Condition: {_display_name(preview, PREVIEW_DISPLAY_NAMES)}"
    line2 = f"Version: {version_path}"
    line3 = f"Target: {_display_name(target_col, TARGET_COL_DISPLAY_NAMES)}  |  Model: {_display_name(model_name, MODEL_DISPLAY_NAMES)}"
    return f"{line1}\n{line2}\n{line3}"


def plot_simplified_results_by_method(eval_df: pd.DataFrame, title: str, save_dir: Path):
    """Create simplified plots showing only Pearson r and RMSE, grouped by method.

    Each method gets its own plot with Pearson r and RMSE side by side.
    Uses fixed color ranges and adds colorbars.
    Feature sets displayed in a dedicated left column.

    Args:
        eval_df: DataFrame with columns 'feature_set', 'method', and metric columns.
        title: Base title for plots.
        save_dir: Directory to save plots into.
    """
    if "method" not in eval_df.columns:
        logger.warning("'method' column not found in eval_df")
        return

    eval_df = eval_df.copy()

    # Group by method
    methods = eval_df["method"].unique()

    for method in methods:
        method_df = eval_df[eval_df["method"] == method].reset_index(drop=True)

        # Get display name for method
        display_method = _display_name(method, BATCH_FRAMEWORK_DISPLAY_NAMES)
        method_title = f"{title}\nMethod: {display_method}"

        # Build labels
        display_fs = method_df["feature_set"].map(lambda x: _display_name(x, FEATURE_SET_DISPLAY_NAMES))
        method_df["label"] = display_fs

        # Sort by feature_set display name
        method_df = method_df.sort_values("label").reset_index(drop=True)

        y_labels = method_df["label"]
        y_pos = np.arange(len(y_labels))

        # Create figure with gridspec: left column for feature sets, then metric plots
        metrics = ["pearson_r", "rmse"]
        figsize_width = 18
        figsize_height = max(6, 0.4 * len(y_labels))
        fig = plt.figure(figsize=(figsize_width, figsize_height))

        gs = fig.add_gridspec(1, 3, width_ratios=[0.15, 1, 1], wspace=0.3)
        ax_fs = fig.add_subplot(gs[0])  # Feature set names column
        ax_pearson = fig.add_subplot(gs[1])  # Pearson r
        ax_rmse = fig.add_subplot(gs[2])  # RMSE
        axes = [ax_pearson, ax_rmse]

        # Draw feature set names on the left column
        ax_fs.set_xlim(0, 1)
        ax_fs.set_ylim(-0.5, len(y_labels) - 0.5)
        ax_fs.invert_yaxis()
        ax_fs.axis('off')

        # Add feature set names as text in the left column
        for i, label in enumerate(y_labels):
            ax_fs.text(0.05, i, label, fontsize=9, fontweight="bold",
                      va='center', ha='left', wrap=True)

        # Add header
        ax_fs.text(0.05, -0.8, "Feature Sets", fontsize=10, fontweight="bold",
                  va='center', ha='left')

        for ax_idx, (ax, metric) in enumerate(zip(axes, metrics)):
            if metric not in method_df.columns:
                logger.warning("Metric %s not in method_df", metric)
                continue

            values = method_df[metric].values.astype(float)

            # Get fixed color range for this metric
            v_min, v_max = FIXED_COLOR_RANGES[metric]

            # Normalize values to [0, 1] using fixed range
            if metric in HIGHER_IS_BETTER:
                # For higher-is-better metrics (Pearson r), clamp and normalize
                norm_values = np.clip((values - v_min) / (v_max - v_min), 0, 1)
                cmap = cm.viridis
            else:
                # For lower-is-better metrics (RMSE), invert normalization
                norm_values = np.clip((v_max - values) / (v_max - v_min), 0, 1)
                cmap = cm.plasma

            colors = cmap(norm_values)

            bars = ax.barh(y_pos, values, color=colors)

            # Set x-axis limits to fixed range
            ax.set_xlim(v_min - 0.01, v_max + 0.01)

            # Value labels
            for bar, val in zip(bars, values):
                x = bar.get_width()
                y = bar.get_y() + bar.get_height() / 2
                offset = 0.01 * (v_max - v_min)
                ha = 'left'
                ax.text(x + offset, y, f"{val:.3f}", va='center', ha=ha, fontsize=8)

            # Remove y-axis labels (feature sets are on the left column now)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([])

            ax.tick_params(axis='y', which='both', left=False)
            metric_title = SIMPLIFIED_METRIC_TITLES.get(metric, metric)
            ax.set_title(metric_title, fontsize=13, weight="bold")
            ax.invert_yaxis()
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=v_min, vmax=v_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label(metric_title, fontsize=10)

        plt.suptitle(method_title, weight='bold', fontsize=12, y=0.98)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Safe filename for method
        safe_method_name = method.replace("/", "_").replace(" ", "_").lower()
        save_path = save_dir / f"results_{safe_method_name}.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        # logger.info("Saved simplified results plot to %s", save_path)


    # logger.info("Saved bar plot to %s", save_path)


def plot_language_heatmap(lang_eval_df: pd.DataFrame, title: str, save_path: Path,
                          metric: str = "pearson_r"):
    """Heatmap of a metric per language x feature_set.

    Args:
        lang_eval_df: DataFrame with columns 'language', 'feature_set', 'method', and metric columns.
        title: Plot title.
        save_path: Where to save the figure.
        metric: Which metric to visualize.
    """
    if metric not in lang_eval_df.columns:
        logger.warning("Metric %s not in lang_eval_df", metric)
        return

    df = lang_eval_df.copy()
    display_fs = df["feature_set"].map(lambda x: _display_name(x, FEATURE_SET_DISPLAY_NAMES))
    if "method" in df.columns and df["method"].nunique() > 1:
        display_method = df["method"].map(lambda x: _display_name(x, BATCH_FRAMEWORK_DISPLAY_NAMES))
        df["label"] = display_fs + " | " + display_method
    else:
        df["label"] = display_fs

    heatmap_df = df.pivot_table(index="language", columns="label", values=metric, aggfunc="first")

    fig_width = max(10, 0.8 * heatmap_df.shape[1])
    fig_height = max(4, 0.6 * heatmap_df.shape[0])
    plt.figure(figsize=(fig_width, fig_height))

    metric_title = BAR_METRIC_TITLES.get(metric, metric)
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis", center=0)
    plt.title(f"{title}\n{metric_title} per Language x Feature Set", fontsize=14, weight="bold")
    plt.ylabel("Language")
    plt.xlabel("Feature Set")
    plt.xticks(rotation=45, ha='right')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    # logger.info("Saved language heatmap to %s", save_path)


def plot_all_results(eval_dir: Path = None, plots_dir: Path = None, replace_existing: bool = True,
                     target_cols: list = None, feature_sets: list = None, model_names: list = None,
                     no_titles: bool = False):
    """Generate plots for all evaluated results.

    Reads grouped evaluation CSVs from eval_dir and saves plots under plots_dir,
    mirroring the same directory structure.

    Args:
        no_titles: If True, suppress axes titles and figure suptitles in
            every produced figure (paper-ready output).
    """
    if eval_dir is None:
        eval_dir = EVAL_SAVE_DIR
    if plots_dir is None:
        plots_dir = PLOTS_SAVE_DIR

    # Used by parallel workers; loky spawns fresh processes so the patch needs
    # to be re-applied inside each worker, not just once in this caller.
    def _maybe_suppress():
        return suppress_titles() if no_titles else contextlib.nullcontext()

    # Plot combined overview
    combined_csv = eval_dir / "all_evaluation_results.csv"
    combined_lang_csv = eval_dir / "all_evaluation_results_per_language.csv"

    if combined_csv.exists():
        all_df = pd.read_csv(combined_csv)
        if target_cols is not None:
            all_df = all_df[all_df["target_col"].isin(target_cols)]
        if feature_sets is not None:
            all_df = all_df[all_df["feature_set"].isin(feature_sets)]
        if model_names is not None:
            all_df = all_df[all_df["model_name"].isin(model_names)]

        # Build group_cols dynamically to include window_index if it exists
        base_group_cols = ["dataset", "aug_p", "preview", "version_path", "target_col", "model_name"]
        group_cols = base_group_cols + (["window_index"] if "window_index" in all_df.columns else [])
        groups = list(all_df.groupby(group_cols))

        def _plot_simplified(group_key, group_df):
            with _maybe_suppress():
                if "window_index" in all_df.columns:
                    dataset, aug_p, preview, version_path, target_col, model_name, window_index = group_key
                    title = _build_title_oneline(dataset, preview, version_path, target_col, model_name)
                    group_plots_dir = plots_dir / dataset / aug_p / preview / version_path / target_col / model_name / f"window_{int(window_index)}"
                else:
                    dataset, aug_p, preview, version_path, target_col, model_name = group_key
                    title = _build_title_oneline(dataset, preview, version_path, target_col, model_name)
                    group_plots_dir = plots_dir / dataset / aug_p / preview / version_path / target_col / model_name
                plot_simplified_results_by_method(group_df, title=title, save_dir=group_plots_dir / "by_method")

        Parallel(n_jobs=-1)(delayed(_plot_simplified)(gk, gdf) for gk, gdf in tqdm(groups, desc="Simplified plots"))

    # Existence is not enough: evaluation.py only emits per-language rows for
    # pool=all methods ({all__lolo, all__l1_strat_kfold, all__pool_all}), so a
    # seen/unseen-only study writes this file with zero rows. pd.read_csv then
    # raises EmptyDataError on the header-less file. Treat empty as "nothing to
    # plot" and carry on with the rest of the plots.
    if combined_lang_csv.exists() and combined_lang_csv.stat().st_size > 0:
        try:
            all_lang_df = pd.read_csv(combined_lang_csv)
        except pd.errors.EmptyDataError:
            logger.warning(
                f"{combined_lang_csv} has no rows (per-language evaluation only "
                "covers pool=all methods); skipping per-language plots."
            )
            all_lang_df = None
    else:
        all_lang_df = None

    if all_lang_df is not None:
        if target_cols is not None:
            all_lang_df = all_lang_df[all_lang_df["target_col"].isin(target_cols)]
        if feature_sets is not None:
            all_lang_df = all_lang_df[all_lang_df["feature_set"].isin(feature_sets)]
        if model_names is not None:
            all_lang_df = all_lang_df[all_lang_df["model_name"].isin(model_names)]

        # Build group_cols dynamically to include window_index if it exists
        base_group_cols = ["dataset", "aug_p", "preview", "version_path", "target_col", "model_name"]
        group_cols = base_group_cols + (["window_index"] if "window_index" in all_lang_df.columns else [])
        lang_groups = list(all_lang_df.groupby(group_cols))

        def _plot_heatmaps(group_key, group_df):
            with _maybe_suppress():
                if "window_index" in all_lang_df.columns:
                    dataset, aug_p, preview, version_path, target_col, model_name, window_index = group_key
                    title = _build_title_multiline(dataset, preview, version_path, target_col, model_name)
                    group_plots_dir = plots_dir / dataset / aug_p / preview / version_path / target_col / model_name / f"window_{int(window_index)}"
                else:
                    dataset, aug_p, preview, version_path, target_col, model_name = group_key
                    title = _build_title_multiline(dataset, preview, version_path, target_col, model_name)
                    group_plots_dir = plots_dir / dataset / aug_p / preview / version_path / target_col / model_name
                if not (group_plots_dir / "language_heatmap_pearson_r.png").exists() or replace_existing:
                    for metric in ["pearson_r", "spearman_r", "rmse"]:
                        plot_language_heatmap(group_df, title=title, save_path=group_plots_dir / f"language_heatmap_{metric}.png", metric=metric)

        Parallel(n_jobs=-1)(delayed(_plot_heatmaps)(gk, gdf) for gk, gdf in tqdm(lang_groups, desc="Language heatmaps"))

    logger.info("All plots saved under %s", plots_dir)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-titles", action="store_true",
        help="Suppress axes titles and figure suptitles in every plot "
             "(paper-ready output).",
    )
    args = parser.parse_args()

    feature_sets = list(RELEVANT_FEATURE_GROUPS_COMBINATIONS_MINIMAL.keys())
    plot_all_results(
        target_cols=ALL_TESTS, model_names=list(MODELS.keys()),
        feature_sets=feature_sets, no_titles=args.no_titles,
    )
