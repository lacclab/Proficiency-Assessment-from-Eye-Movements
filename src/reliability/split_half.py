"""Split-half reliability of EyeScore and predictions, from the pre-computed
split-half result trees.

Each of up to 20 splits divides the items into two halves that were scored
separately upstream. Per split, participants' mean score on one half is
correlated with their mean on the other (per content group for OneStop), and
the mean over splits is reported, uncorrected. Writes
results/split_half_reliability/[<preview>/]split_half_reliability.csv.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import Fields
from src.reliability.utils import (
    DATASET_TARGET_COLS,
    DEFAULT_PREVIEW,
    FEATURE_SET_DISPLAY,
    add_content_group_from_metadata,
    onestop_preview,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("src/reliability")
PREDICTIONS_DIR = Path("src/methods/predictions/results")
EYESCORE_DIR = Path("src/methods/EyeScore/results")
SPLIT_HALF_DATA_DIR = RESULTS_DIR / "results" / "split_half_reliability"


def split_half_data_dir(preview: str) -> Path:
    """Output dir for a preview: the top-level path for Gathering, a subdir otherwise."""
    if preview == DEFAULT_PREVIEW:
        return SPLIT_HALF_DATA_DIR
    return SPLIT_HALF_DATA_DIR / preview.lower()


def calculate_eyescore_split_half_correlations(
    dataset: str, p_agg_level: str, feature_set: str, preview: str = DEFAULT_PREVIEW
) -> dict:
    """Calculate split-half correlations from pre-computed EyeScore splits.

    For each split 1-20 and each pool (all, seen, unseen):
    - Load eye_score from half_1 and half_2
    - Average eye_score per participant
    - Calculate correlation between halves
    Returns dict with 'all', 'seen', 'unseen' pools.
    """
    preview = onestop_preview(dataset, preview)
    results = {}

    for pool in ["all", "seen", "unseen"]:
        correlations = []
        pvalues = []

        for split_idx in range(1, 21):
            # Construct paths for half_1 and half_2
            # Path structure differs between datasets:
            # OneStop: dataset/preview/pool/split_half/fully_agg/ordinary/all/p_agg_level/split_idx
            # Meco: dataset/preview/pool/split_half/fully_agg/all/all/split_idx (no p_agg_level)
            if dataset == "Meco":
                base_path = (
                    EYESCORE_DIR
                    / dataset
                    / preview
                    / pool
                    / "split_half"
                    / "fully_agg"
                    / "all"
                    / "all"
                    / f"split_{split_idx}"
                )
            else:
                base_path = (
                    EYESCORE_DIR
                    / dataset
                    / preview
                    / pool
                    / "split_half"
                    / "fully_agg"
                    / "ordinary"
                    / "all"
                    / p_agg_level
                    / f"split_{split_idx}"
                )
            half1_path = base_path / "half_1" / f"{feature_set}.csv"
            half2_path = base_path / "half_2" / f"{feature_set}.csv"

            if not half1_path.exists() or not half2_path.exists():
                continue

            try:
                df_half1 = pd.read_csv(half1_path)
                df_half2 = pd.read_csv(half2_path)

                # Average eye_score per participant
                mean_half1 = df_half1.groupby(Fields.SUBJECT_ID)["eye_score"].mean()
                mean_half2 = df_half2.groupby(Fields.SUBJECT_ID)["eye_score"].mean()

                # Align on common participants and drop NaN scores; a NaN in
                # either half would otherwise make pearsonr return NaN and
                # silently discard the whole split.
                both = pd.concat(
                    [mean_half1, mean_half2], axis=1, keys=["h1", "h2"]
                ).dropna()
                if len(both) < 3:
                    logger.debug(f"Not enough common participants in split {split_idx} for {pool}")
                    continue

                # Calculate correlation
                corr, p_val = stats.pearsonr(both["h1"], both["h2"])
                if not np.isnan(corr) and -1 < corr < 1:
                    correlations.append(corr)
                    pvalues.append(p_val)

            except Exception as e:
                logger.debug(f"Error loading split {split_idx} for {pool}: {e}")
                continue

        if correlations:
            results[pool] = {
                "mean_correlation": np.mean(correlations),
                "std_correlation": np.std(correlations),
                "mean_pvalue": np.mean(pvalues),
                "n_splits": len(correlations),
            }
        else:
            results[pool] = {"mean_correlation": np.nan, "std_correlation": np.nan, "n_splits": 0}

    return results


def calculate_predictions_split_half_correlations(
    dataset: str, preview: str, target: str, feature_set: str, p_agg_level: str = "article", model: str = "Ridge_Classifier"
) -> dict:
    """Calculate split-half correlations from pre-computed splits.

    For each split 1-20:
    - Load predictions from half_1 and half_2
    - Average predictions per participant
    - Calculate correlation between halves
    Returns dict with 'all', 'seen', 'unseen' pools.
    """
    results = {}

    for pool in ["all", "seen", "unseen"]:
        correlations = []
        pvalues = []

        for split_idx in range(1, 21):
            # Construct paths for half_1 and half_2
            preview_lower = preview.lower()
            base_path = (
                PREDICTIONS_DIR
                / dataset
                / "aug_p=1.0"
                / preview_lower
                / "split_half"
            )
            # Meco doesn't have p_agg_level in split_half structure
            if dataset == "Meco":
                half1_path = (
                    base_path
                    / f"split_{split_idx}"
                    / "half_1"
                    / target
                    / feature_set
                    / model
                    / f"{pool}__pool_all.csv"
                )
                half2_path = (
                    base_path
                    / f"split_{split_idx}"
                    / "half_2"
                    / target
                    / feature_set
                    / model
                    / f"{pool}__pool_all.csv"
                )
            else:
                # OneStop has p_agg_level in the path
                half1_path = (
                    base_path
                    / p_agg_level
                    / f"split_{split_idx}"
                    / "half_1"
                    / target
                    / feature_set
                    / model
                    / f"{pool}__pool_all.csv"
                )
                half2_path = (
                    base_path
                    / p_agg_level
                    / f"split_{split_idx}"
                    / "half_2"
                    / target
                    / feature_set
                    / model
                    / f"{pool}__pool_all.csv"
                )

            if not half1_path.exists() or not half2_path.exists():
                continue

            try:
                df_half1 = pd.read_csv(half1_path)
                df_half2 = pd.read_csv(half2_path)

                # Average predictions per participant
                mean_half1 = df_half1.groupby(Fields.SUBJECT_ID)["pred"].mean()
                mean_half2 = df_half2.groupby(Fields.SUBJECT_ID)["pred"].mean()

                # Align on common participants and drop NaN predictions; a NaN
                # in either half would otherwise make pearsonr return NaN and
                # silently discard the whole split.
                both = pd.concat(
                    [mean_half1, mean_half2], axis=1, keys=["h1", "h2"]
                ).dropna()
                if len(both) < 3:
                    logger.debug(f"Not enough common participants in split {split_idx} for {pool}")
                    continue

                # Calculate correlation
                corr, p_val = stats.pearsonr(both["h1"], both["h2"])
                if not np.isnan(corr) and -1 < corr < 1:
                    correlations.append(corr)
                    pvalues.append(p_val)

            except Exception as e:
                logger.debug(f"Error loading split {split_idx} for {pool}: {e}")
                continue

        if correlations:
            results[pool] = {
                "mean_correlation": np.mean(correlations),
                "std_correlation": np.std(correlations),
                "mean_pvalue": np.mean(pvalues),
                "n_splits": len(correlations),
            }
        else:
            results[pool] = {"mean_correlation": np.nan, "std_correlation": np.nan, "n_splits": 0}

    return results


def calculate_eyescore_split_half_correlations_by_group(
    dataset: str, p_agg_level: str, feature_set: str, preview: str = DEFAULT_PREVIEW
) -> dict:
    """Calculate split-half correlations per content group for EyeScore.

    Returns dict with pools as keys, each containing per_group_correlations and mean_correlation.
    """
    preview = onestop_preview(dataset, preview)
    results = {}

    if dataset != "OneStop":
        # Non-OneStop datasets don't have content_group, just return regular results
        return calculate_eyescore_split_half_correlations(dataset, p_agg_level, feature_set, preview)

    for pool in ["all", "seen", "unseen"]:
        per_group_correlations = {}

        for split_idx in range(1, 21):
            # Path structure: dataset/preview/pool/split_half/fully_agg/ordinary/all/p_agg_level/split_idx/half_1
            base_path = (
                EYESCORE_DIR / dataset / preview / pool / "split_half"
                / "fully_agg" / "ordinary" / "all" / p_agg_level / f"split_{split_idx}"
            )
            half1_path = base_path / "half_1" / f"{feature_set}.csv"
            half2_path = base_path / "half_2" / f"{feature_set}.csv"

            if not half1_path.exists() or not half2_path.exists():
                continue

            try:
                df_half1 = pd.read_csv(half1_path)
                df_half2 = pd.read_csv(half2_path)

                # Add content_group
                df_half1 = add_content_group_from_metadata(df_half1, dataset)
                df_half2 = add_content_group_from_metadata(df_half2, dataset)

                # Group by content_group and calculate correlations
                for group_val in df_half1['content_group'].unique():
                    if pd.isna(group_val):
                        continue

                    group_str = str(group_val)
                    df_h1_group = df_half1[df_half1['content_group'] == group_val]
                    df_h2_group = df_half2[df_half2['content_group'] == group_val]

                    mean_half1 = df_h1_group.groupby(Fields.SUBJECT_ID)["eye_score"].mean()
                    mean_half2 = df_h2_group.groupby(Fields.SUBJECT_ID)["eye_score"].mean()

                    # Align and drop NaN scores so one NaN participant doesn't
                    # NaN out the whole group's correlation for this split
                    both = pd.concat(
                        [mean_half1, mean_half2], axis=1, keys=["h1", "h2"]
                    ).dropna()
                    if len(both) < 3:
                        continue

                    corr, _ = stats.pearsonr(both["h1"], both["h2"])
                    if not np.isnan(corr) and -1 < corr < 1:
                        if group_str not in per_group_correlations:
                            per_group_correlations[group_str] = []
                        per_group_correlations[group_str].append(corr)

            except Exception as e:
                logger.debug(f"Error loading split {split_idx} for {pool}: {e}")
                continue

        # Calculate averages per group
        per_group_means = {}
        all_correlations = []
        for group_str, corrs in per_group_correlations.items():
            if corrs:
                per_group_means[group_str] = np.mean(corrs)
                all_correlations.extend(corrs)

        results[pool] = {
            "mean_correlation": np.mean(all_correlations) if all_correlations else np.nan,
            "per_group_correlations": per_group_means if per_group_means else {},
            "n_splits": len(all_correlations),
        }

    return results


def calculate_predictions_split_half_correlations_by_group(
    dataset: str, preview: str, target: str, feature_set: str, p_agg_level: str = "article", model: str = "Ridge_Classifier"
) -> dict:
    """Calculate split-half correlations per content group for Predictions.

    Returns dict with pools as keys, each containing per_group_correlations and mean_correlation.
    """
    results = {}

    if dataset != "OneStop":
        # Non-OneStop datasets don't have content_group, just return regular results
        return calculate_predictions_split_half_correlations(dataset, preview, target, feature_set, p_agg_level, model)

    for pool in ["all", "seen", "unseen"]:
        per_group_correlations = {}
        preview_lower = preview.lower()

        for split_idx in range(1, 21):
            half1_path = (
                PREDICTIONS_DIR / dataset / "aug_p=1.0" / preview_lower / "split_half"
                / p_agg_level / f"split_{split_idx}" / "half_1" / target / feature_set / model / f"{pool}__pool_all.csv"
            )
            half2_path = (
                PREDICTIONS_DIR / dataset / "aug_p=1.0" / preview_lower / "split_half"
                / p_agg_level / f"split_{split_idx}" / "half_2" / target / feature_set / model / f"{pool}__pool_all.csv"
            )

            if not half1_path.exists() or not half2_path.exists():
                continue

            try:
                df_half1 = pd.read_csv(half1_path)
                df_half2 = pd.read_csv(half2_path)

                # Add content_group
                df_half1 = add_content_group_from_metadata(df_half1, dataset)
                df_half2 = add_content_group_from_metadata(df_half2, dataset)

                # Group by content_group and calculate correlations
                for group_val in df_half1['content_group'].unique():
                    if pd.isna(group_val):
                        continue

                    group_str = str(group_val)
                    df_h1_group = df_half1[df_half1['content_group'] == group_val]
                    df_h2_group = df_half2[df_half2['content_group'] == group_val]

                    mean_half1 = df_h1_group.groupby(Fields.SUBJECT_ID)["pred"].mean()
                    mean_half2 = df_h2_group.groupby(Fields.SUBJECT_ID)["pred"].mean()

                    # Align and drop NaN predictions so one NaN participant
                    # doesn't NaN out the whole group's correlation for this split
                    both = pd.concat(
                        [mean_half1, mean_half2], axis=1, keys=["h1", "h2"]
                    ).dropna()
                    if len(both) < 3:
                        continue

                    corr, _ = stats.pearsonr(both["h1"], both["h2"])
                    if not np.isnan(corr) and -1 < corr < 1:
                        if group_str not in per_group_correlations:
                            per_group_correlations[group_str] = []
                        per_group_correlations[group_str].append(corr)

            except Exception as e:
                logger.debug(f"Error loading split {split_idx} for {pool}: {e}")
                continue

        # Calculate averages per group
        per_group_means = {}
        all_correlations = []
        for group_str, corrs in per_group_correlations.items():
            if corrs:
                per_group_means[group_str] = np.mean(corrs)
                all_correlations.extend(corrs)

        results[pool] = {
            "mean_correlation": np.mean(all_correlations) if all_correlations else np.nan,
            "per_group_correlations": per_group_means if per_group_means else {},
            "n_splits": len(all_correlations),
        }

    return results


def generate_split_half_results(preview: str = DEFAULT_PREVIEW):
    """Calculate and save split-half correlations for all datasets/targets/features.

    `preview` selects the OneStop tree to read. Meco is unaffected (always "All")
    and its rows are emitted into every preview's output, so each tree is a
    self-contained OneStop-vs-Meco table rather than a fragment.
    """

    logger.info(f"Calculating split-half correlations (OneStop preview: {preview})...")

    all_results = []

    # Predictions
    logger.info("Processing Predictions...")
    for dataset in ["OneStop", "Meco"]:
        ds_preview = onestop_preview(dataset, preview)
        targets = DATASET_TARGET_COLS[dataset]
        p_agg_levels = ["article", "paragraph"] if dataset == "OneStop" else ["paragraph"]

        for p_agg_level in p_agg_levels:
            for target in targets:
                for feature_set in FEATURE_SET_DISPLAY.keys():
                    # Use per-group version for OneStop, regular version for others
                    if dataset == "OneStop":
                        corr_results = calculate_predictions_split_half_correlations_by_group(
                            dataset, ds_preview, target, feature_set, p_agg_level
                        )
                    else:
                        corr_results = calculate_predictions_split_half_correlations(
                            dataset, ds_preview, target, feature_set, p_agg_level
                        )

                    for pool, corr_data in corr_results.items():
                        all_results.append(
                            {
                                "source": "predictions",
                                "dataset": dataset,
                                "p_agg_level": p_agg_level,
                                "target": target,
                                "feature_set": feature_set,
                                "pool": pool,
                                "mean_correlation": corr_data["mean_correlation"],
                                "std_correlation": corr_data.get("std_correlation", np.nan),
                                "per_group_correlations": corr_data.get("per_group_correlations"),
                                "n_splits": corr_data["n_splits"],
                            }
                        )

    # EyeScore
    logger.info("Processing EyeScore...")
    for dataset in ["OneStop", "Meco"]:
        p_agg_levels = ["article", "paragraph"] if dataset == "OneStop" else ["paragraph"]

        for p_agg_level in p_agg_levels:
            for feature_set in FEATURE_SET_DISPLAY.keys():
                # Use per-group version for OneStop, regular version for others
                if dataset == "OneStop":
                    corr_results = calculate_eyescore_split_half_correlations_by_group(
                        dataset, p_agg_level, feature_set, preview
                    )
                else:
                    corr_results = calculate_eyescore_split_half_correlations(
                        dataset, p_agg_level, feature_set, preview
                    )

                for pool, corr_data in corr_results.items():
                    all_results.append(
                        {
                            "source": "eyescore",
                            "dataset": dataset,
                            "p_agg_level": p_agg_level,
                            "feature_set": feature_set,
                            "pool": pool,
                            "mean_correlation": corr_data["mean_correlation"],
                            "std_correlation": corr_data.get("std_correlation", np.nan),
                            "per_group_correlations": corr_data.get("per_group_correlations"),
                            "n_splits": corr_data["n_splits"],
                        }
                    )

    df_results = pd.DataFrame(all_results)

    # EyeScore rows have no target
    df_results.loc[df_results["source"] == "eyescore", "target"] = None

    out_dir = split_half_data_dir(preview)
    results_path = out_dir / "split_half_reliability.csv"
    # Not a single split anywhere means the split-half trees are missing, not
    # that every coefficient is: keep the existing CSV instead of blanking it.
    if not (df_results["n_splits"] > 0).any():
        logger.warning(
            "No split-half correlation could be computed (split-half results "
            "under %s and %s?); leaving %s unchanged.",
            PREDICTIONS_DIR, EYESCORE_DIR, results_path)
        return df_results

    out_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")

    return df_results


def main(preview: str = DEFAULT_PREVIEW):
    """Run the split-half analysis for one OneStop preview."""
    logger.info("Starting split-half reliability analysis...")
    generate_split_half_results(preview=preview)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Calculate split-half reliability")
    parser.add_argument(
        "--preview",
        default=DEFAULT_PREVIEW,
        help="OneStop preview to read (Gathering|Hunting). Gathering writes the "
        "top-level paths; any other preview writes to a parallel subtree, e.g. "
        "src/reliability/results/split_half_reliability/hunting/. Meco is "
        "unaffected and appears in both.",
    )
    args = parser.parse_args()
    main(preview=args.preview)
