"""Stage 2b: compute EyeScore, then evaluate it.

    --stage calculation  score every L2 participant's similarity to the L1
                         prototype, per (dataset, preview, content mode,
                         aggregation, feature set), plus each registered
                         debiasing correction, into src/methods/EyeScore/results*/
    --stage evaluation   correlate the scores with the proficiency tests,
                         significance tests, language-bias analysis, plots and
                         tables, into src/evaluation/EyeScore/
    --stage both         (default) calculation, then evaluation

The paper reads two kinds of tree. Its main EyeScore tables come from the
canonical results/ tree (a plain run). Its L1-bias tables and figures come from
three debiasing trees, results_<method>_cat3ctr/, built by:

    python -m src.run.eyescore \\
        --debias-methods two_step,distance_mreg,interaction \\
        --results-suffix cat3ctr --center-distance \\
        --typo-calib-distance-type 'cat:syntactic+phonological+scriptural'

    python -m src.run.eyescore                                  # canonical tree
    python -m src.run.eyescore --stage calculation --agg-types fully_agg
"""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import itertools

from src.constants import (
    DATA_PATH, SMALL_FEATURE_SETS, AggTypes, Fields, FIXED_TEXT_GROUP_PREFIXES,
)
from src.agg_configs import ONESTOP_AGG_CONFIGS, MECO_AGG_CONFIGS
from src.methods.EyeScore.calculation import (
    run_one_dataset_version, ALL_FEATURE_SETS_DICT,
    get_feature_set_df, eye_score_per_item_independent,
    apply_correction_to_eyescore, DEFAULT_CORRECTIONS,
    TYPO_CALIB_TARGET_COL, TYPO_CALIB_DISTANCE_TYPE,
    RAW_VARIANT, SPLIT_SCHEMES, SPLIT_SCHEME_TO_VARIANT, corrections_complete,
    DEBIAS_METHODS, DEFAULT_DEBIAS_METHOD,
    _average_eye_scores, _eye_score_one_direction, _make_meco_fold_scorer, _seen_unseen_feature_paths_eyescore,
    SCORERS, SCORER_NAMES, DEFAULT_SCORER,
)
import os
from src.methods.utils import load_per_text_features_pair
from src.evaluation.EyeScore.evaluation import evaluate_all_results
from src.evaluation.EyeScore.plot import plot_all_results
from src.evaluation.EyeScore.language_bias import run_language_bias_analysis, LANG_BIAS_PLOTS_DIR
from src.utils.cli import parse_csv
from src.utils.split_half_columns import (
    split_half_common_feature_cols,
    apply_split_common_filter,
)
import pandas as pd

logger.add(DATA_PATH / 'logs' / 'eyescore.log', rotation='10 MB', retention='7 days')


SEEN_UNSEEN_HALVES = ("odd", "even")


def _require_prototype_metric(scorer: str) -> str:
    """Return the prototype metric for ``scorer``, or raise if it is a
    discriminative scorer reaching a per-item path (out of scope — discriminative
    scorers only run on scalar fully_agg data)."""
    metric = SCORERS[scorer].metric
    if metric is None:
        raise NotImplementedError(
            f"Discriminative scorer {scorer!r} is not supported on per-item data."
        )
    return metric


def _run_meco_seen_unseen_eyescore_job(
    halves_data: dict,
    dataset: str,
    preview: str,
    feature_set_name: str,
    feature_cols: list,
    replace_existing: bool,
    calibration: dict | None,
    corrections=DEFAULT_CORRECTIONS,
    min_n_paragraphs: int | None = None,
    results_dirname: str = "results",
    scorer: str = DEFAULT_SCORER,
) -> tuple[str, str]:
    """Compute SEEN and UNSEEN eye_scores for one feature set on the MECO half-split.

    SEEN: average two same-half eye_scores (odd-prototype↔odd-test, even↔even).
    UNSEEN: average two cross-half eye_scores (odd-prototype↔even-test, even↔odd).

    Corrections are applied AFTER averaging (one fit per content_mode rather than
    per direction) so the regression sees the lower-variance per-participant
    estimate. Per-text feature sets get SEEN only — different paragraph IDs across
    halves give them no shared cosine column space for UNSEEN.

    `min_n_paragraphs` (optional): drop participants whose `n_paragraphs` < threshold
    on each half independently — odd cohort and even cohort can differ. A reader
    who clears the threshold on only one half contributes a score from that half
    only; the averaging step (groupby participant_id) handles the asymmetry. When
    set, outputs land under `fully_agg/n_paragraphs_geq_{N}/...`.
    """
    threshold_label = (
        f"/n_paragraphs_geq_{min_n_paragraphs}" if min_n_paragraphs is not None else ""
    )
    job_id = (
        f"{dataset}/{preview}/{feature_set_name}/seen_unseen{threshold_label}"
    )
    try:
        is_per_text = feature_set_name in FIXED_TEXT_GROUP_PREFIXES
        src_path = Path.cwd() / Path("src")

        # Per-half independent filter — same semantic as predictions.
        if min_n_paragraphs is not None:
            halves_data = {
                half: {
                    **halves_data[half],
                    "l2": halves_data[half]["l2"][
                        halves_data[half]["l2"]["n_paragraphs"] >= min_n_paragraphs
                    ],
                }
                for half in halves_data
            }

        # Mirror OneStop's results layout: {results_dirname}/{dataset}/{preview}/{content_mode}/{path_from_folder}
        results_root = src_path / f"methods/EyeScore/{results_dirname}/{dataset}/{preview}"
        seen_dir = results_root / f"seen/fully_agg{threshold_label}/all/all"
        unseen_dir = results_root / f"unseen/fully_agg{threshold_label}/all/all"
        seen_dir.mkdir(parents=True, exist_ok=True)
        unseen_dir.mkdir(parents=True, exist_ok=True)
        seen_raw_path = seen_dir / f"{feature_set_name}.csv"
        unseen_raw_path = unseen_dir / f"{feature_set_name}.csv"

        # A cell is done only when its raw scores AND every correction this
        # dataset expects are on disk: testing the raw file alone meant a
        # correction added to the registry after the cell was last computed
        # never appeared without a full --replace-existing rerun.
        skip_seen = (seen_raw_path.exists()
                     and corrections_complete(seen_raw_path, dataset, corrections)
                     and not replace_existing)
        skip_unseen = (unseen_raw_path.exists()
                       and corrections_complete(unseen_raw_path, dataset, corrections)
                       and not replace_existing)
        if skip_seen and (skip_unseen or is_per_text):
            return (job_id, "SKIPPED")

        directions_by_mode = {
            "seen": [("odd", "odd"), ("even", "even")],
            "unseen": [("odd", "even"), ("even", "odd")],
        }

        for content_mode, save_dir, raw_path, skip in (
            ("seen", seen_dir, seen_raw_path, skip_seen),
            ("unseen", unseen_dir, unseen_raw_path, skip_unseen),
        ):
            if skip:
                continue
            if content_mode == "unseen" and is_per_text:
                continue  # different paragraph IDs across halves → no shared col space

            per_direction_raw = []
            l2_for_corrections = None
            for l1_half, l2_half in directions_by_mode[content_mode]:
                df_l1 = halves_data[l1_half]["per_text_l1" if is_per_text else "l1"]
                df_l2 = halves_data[l2_half]["per_text_l2" if is_per_text else "l2"]
                if df_l1 is None or df_l2 is None:
                    continue

                # Unseen directions cross halves (e.g. L1-odd ↔ L2-even): the L1
                # prototype from one half is standardized against the L2 from the
                # SAME half so prototype and L2 row share coordinates. Seen
                # directions use the same half on both sides, so eye_score's
                # default (fit on the scored L2) already matches — pass no fit.
                df_l2_fit = None
                if content_mode == "unseen" and l1_half != l2_half:
                    df_l2_fit = halves_data[l1_half]["per_text_l2" if is_per_text else "l2"]

                # Same atomic scoring the leak-free fold_scorer uses (proven
                # bit-identical to the former inline loop).
                per_direction_raw.append(
                    _eye_score_one_direction(
                        df_l1, df_l2, df_l2_fit,
                        feature_set_name, feature_cols, save_dir,
                        scorer=scorer,
                    )
                )
                # Cache one half's L2 for the correction step's context — only
                # participant-level metadata (L1, target_col) is read, which is
                # invariant across halves.
                if l2_for_corrections is None:
                    l2_for_corrections = df_l2

            avg_raw = _average_eye_scores(per_direction_raw)
            if avg_raw is None:
                continue
            avg_raw.to_csv(raw_path, index=False)
            logger.info(f"Wrote {content_mode} raw eye_scores → {raw_path}")

            # Apply each correction to the averaged eye_scores. The fold_scorer
            # lets the typology corrections RECOMPUTE eye_scores per CV fold from
            # a train-only prototype + z-score pool (scaling-leak fix); the raw
            # avg_raw written above is untouched. Applied to per-text sets too
            # (is_per_text is threaded into the fold scorer, which uses the
            # per_text halves) rather than falling back to the legacy path.
            base_ctx = {
                "feature_set": feature_set_name,
                "eye_scores_df": avg_raw,
                "all_features_df_L2": l2_for_corrections,
            }
            base_ctx["fold_scorer"] = _make_meco_fold_scorer(
                halves_data, feature_set_name, feature_cols, is_per_text,
                directions_by_mode[content_mode], save_dir, scorer=scorer,
            )
            if calibration:
                base_ctx.update(calibration)
            for correction in corrections:
                adjusted = apply_correction_to_eyescore(correction, base_ctx)
                if adjusted is None:
                    continue
                adj_path = raw_path.with_name(raw_path.stem + correction.suffix + raw_path.suffix)
                adjusted.to_csv(adj_path, index=False)

        return (job_id, "DONE")

    except Exception as e:
        import traceback, sys
        tb = traceback.extract_tb(sys.exc_info()[2])
        origin = tb[-1]
        msg = str(e)[:80].replace("\n", " ")
        logger.error(f"Error in {job_id}: {origin.filename}:{origin.lineno} in {origin.name} - {msg}")
        return (job_id, f"ERROR: {origin.filename.split('/')[-1]}:{origin.lineno} {msg}")
    


def _run_meco_split_half_seen_unseen_eyescore_job(
    halves_data: dict,
    dataset: str,
    preview: str,
    feature_set_name: str,
    feature_cols: list,
    replace_existing: bool,
    calibration: dict | None,
    corrections=DEFAULT_CORRECTIONS,
    min_n_paragraphs: int | None = None,
    split_idx: int = None,
    half_num: int = None,
    reread: str = "all",
    results_dirname: str = "results",
    scorer: str = DEFAULT_SCORER,
) -> tuple[str, str]:
    """Wrapper for MECO split-half eyescore calculation.

    Reuses _run_meco_seen_unseen_eyescore_job logic but saves to split-half paths.
    For split-half, preview is always "All" but we accept it for consistency.
    """
    threshold_label = (
        f"/n_paragraphs_geq_{min_n_paragraphs}" if min_n_paragraphs is not None else ""
    )
    job_id = (
        f"{dataset}/split_half/split_{split_idx}/half_{half_num}/{feature_set_name}{threshold_label}"
    )
    try:
        is_per_text = feature_set_name in FIXED_TEXT_GROUP_PREFIXES
        src_path = Path.cwd() / Path("src")

        # Per-half independent filter
        if min_n_paragraphs is not None:
            halves_data = {
                half: {
                    **halves_data[half],
                    "l2": halves_data[half]["l2"][
                        halves_data[half]["l2"]["n_paragraphs"] >= min_n_paragraphs
                    ],
                }
                for half in halves_data
            }

        # Results to split-half path: seen/unseen before split_half, then agg params, then split/half
        seen_root = src_path / f"methods/EyeScore/{results_dirname}/{dataset}/All/seen/split_half/fully_agg/{reread}/all/split_{split_idx}/half_{half_num}"
        unseen_root = src_path / f"methods/EyeScore/{results_dirname}/{dataset}/All/unseen/split_half/fully_agg/{reread}/all/split_{split_idx}/half_{half_num}"
        seen_dir = seen_root / f"{threshold_label}" if threshold_label else seen_root
        unseen_dir = unseen_root / f"{threshold_label}" if threshold_label else unseen_root
        seen_dir.mkdir(parents=True, exist_ok=True)
        unseen_dir.mkdir(parents=True, exist_ok=True)
        seen_raw_path = seen_dir / f"{feature_set_name}.csv"
        unseen_raw_path = unseen_dir / f"{feature_set_name}.csv"

        # A cell is done only when its raw scores AND every correction this
        # dataset expects are on disk: testing the raw file alone meant a
        # correction added to the registry after the cell was last computed
        # never appeared without a full --replace-existing rerun.
        skip_seen = (seen_raw_path.exists()
                     and corrections_complete(seen_raw_path, dataset, corrections)
                     and not replace_existing)
        skip_unseen = (unseen_raw_path.exists()
                       and corrections_complete(unseen_raw_path, dataset, corrections)
                       and not replace_existing)
        if skip_seen and (skip_unseen or is_per_text):
            return (job_id, "SKIPPED")

        directions_by_mode = {
            "seen": [("odd", "odd"), ("even", "even")],
            "unseen": [("odd", "even"), ("even", "odd")],
        }

        for content_mode, save_dir, raw_path, skip in (
            ("seen", seen_dir, seen_raw_path, skip_seen),
            ("unseen", unseen_dir, unseen_raw_path, skip_unseen),
        ):
            if skip:
                continue
            if content_mode == "unseen" and is_per_text:
                continue

            per_direction_raw = []
            l2_for_corrections = None
            for l1_half, l2_half in directions_by_mode[content_mode]:
                df_l1 = halves_data[l1_half]["per_text_l1" if is_per_text else "l1"]
                df_l2 = halves_data[l2_half]["per_text_l2" if is_per_text else "l2"]
                if df_l1 is None or df_l2 is None:
                    continue

                df_l2_fit = None
                if content_mode == "unseen" and l1_half != l2_half:
                    df_l2_fit = halves_data[l1_half]["per_text_l2" if is_per_text else "l2"]

                per_direction_raw.append(
                    _eye_score_one_direction(
                        df_l1, df_l2, df_l2_fit,
                        feature_set_name, feature_cols, save_dir,
                        scorer=scorer,
                    )
                )
                if l2_for_corrections is None:
                    l2_for_corrections = df_l2

            avg_raw = _average_eye_scores(per_direction_raw)
            if avg_raw is None:
                continue
            avg_raw.to_csv(raw_path, index=False)
            logger.info(f"Wrote split-half {content_mode} eyescores → {raw_path}")

            # Apply corrections. fold_scorer recomputes eye_scores per CV fold
            # from a train-only prototype + z-score pool (scaling-leak fix);
            # avg_raw is untouched. Applied to per-text sets too (the fold scorer
            # uses the per_text halves via is_per_text).
            base_ctx = {
                "feature_set": feature_set_name,
                "eye_scores_df": avg_raw,
                "all_features_df_L2": l2_for_corrections,
            }
            base_ctx["fold_scorer"] = _make_meco_fold_scorer(
                halves_data, feature_set_name, feature_cols, is_per_text,
                directions_by_mode[content_mode], save_dir, scorer=scorer,
            )
            if calibration:
                base_ctx.update(calibration)
            for correction in corrections:
                adjusted = apply_correction_to_eyescore(correction, base_ctx)
                if adjusted is None:
                    continue
                adj_path = raw_path.with_name(raw_path.stem + correction.suffix + raw_path.suffix)
                adjusted.to_csv(adj_path, index=False)

        return (job_id, "DONE")

    except Exception as e:
        import traceback, sys
        tb = traceback.extract_tb(sys.exc_info()[2])
        origin = tb[-1]
        msg = str(e)[:80].replace("\n", " ")
        logger.error(f"Error in {job_id}: {origin.filename}:{origin.lineno} in {origin.name} - {msg}")
        return (job_id, f"ERROR: {origin.filename.split('/')[-1]}:{origin.lineno} {msg}")


def _run_meco_per_item_seen_unseen_eyescore_job(
    full_data: dict,
    dataset: str,
    preview: str,
    feature_set_name: str,
    feature_cols: list,
    replace_existing: bool,
    calibration: dict | None,
    corrections=DEFAULT_CORRECTIONS,
    min_n_paragraphs: int | None = None,
    l1_parquet_dir: Path | None = None,
    l2_parquet_dir: Path | None = None,
    results_dirname: str = "results",
    scorer: str = DEFAULT_SCORER,
) -> tuple[str, str]:
    """Compute SEEN and UNSEEN eye_scores for per-item aggregation on the MECO half-split.

    Computes odd/even halves internally based on item parity, then applies
    parity splits within each half for SEEN/UNSEEN validation.

    SEEN: average two same-half per-item eye_scores (odd-prototype↔odd-test, even↔even).
    UNSEEN: average two cross-half per-item eye_scores (odd-prototype↔even-test, even↔odd).

    This works with per-item aggregated data (multiple rows per participant) and applies
    the seen/unseen split logic via half-splits of the data.
    """
    threshold_label = (
        f"/n_paragraphs_geq_{min_n_paragraphs}" if min_n_paragraphs is not None else ""
    )
    job_id = (
        f"{dataset}/{preview}/{feature_set_name}/per_item_seen_unseen{threshold_label}"
    )
    try:
        is_per_text = feature_set_name in FIXED_TEXT_GROUP_PREFIXES
        src_path = Path.cwd() / Path("src")

        # Compute halves internally: split by item parity (odd/even item IDs)
        l1_df = full_data["l1"].copy()
        l2_df = full_data["l2"].copy()

        item_col = Fields.UNIQUE_PARAGRAPH_ID if Fields.UNIQUE_PARAGRAPH_ID in l1_df.columns else Fields.ARTICLE_ID

        # Determine which items are odd vs even based on item ID
        l1_df['_item_parity'] = l1_df[item_col] % 2  # 1 if odd item ID, 0 if even item ID
        l2_df['_item_parity'] = l2_df[item_col] % 2

        # Create halves_data from full data by item parity
        # Note: Per-item SEEN/UNSEEN loads per-item parquets in the loop, not combined parquets
        halves_data = {
            "odd": {
                "l1": l1_df[l1_df['_item_parity'] == 1].drop(columns=['_item_parity']),
                "l2": l2_df[l2_df['_item_parity'] == 1].drop(columns=['_item_parity']),
            },
            "even": {
                "l1": l1_df[l1_df['_item_parity'] == 0].drop(columns=['_item_parity']),
                "l2": l2_df[l2_df['_item_parity'] == 0].drop(columns=['_item_parity']),
            },
        }

        # Per-half independent filter
        if min_n_paragraphs is not None:
            halves_data = {
                half: {
                    **halves_data[half],
                    "l2": halves_data[half]["l2"][
                        halves_data[half]["l2"]["n_paragraphs"] >= min_n_paragraphs
                    ],
                }
                for half in halves_data
            }

        results_root = src_path / f"methods/EyeScore/{results_dirname}/{dataset}/{preview}"
        seen_dir = results_root / f"seen/per_item_agg{threshold_label}/all/all"
        unseen_dir = results_root / f"unseen/per_item_agg{threshold_label}/all/all"
        seen_dir.mkdir(parents=True, exist_ok=True)
        unseen_dir.mkdir(parents=True, exist_ok=True)
        seen_raw_path = seen_dir / f"{feature_set_name}.csv"
        unseen_raw_path = unseen_dir / f"{feature_set_name}.csv"

        # A cell is done only when its raw scores AND every correction this
        # dataset expects are on disk: testing the raw file alone meant a
        # correction added to the registry after the cell was last computed
        # never appeared without a full --replace-existing rerun.
        skip_seen = (seen_raw_path.exists()
                     and corrections_complete(seen_raw_path, dataset, corrections)
                     and not replace_existing)
        skip_unseen = (unseen_raw_path.exists()
                       and corrections_complete(unseen_raw_path, dataset, corrections)
                       and not replace_existing)
        if skip_seen and (skip_unseen or is_per_text):
            return (job_id, "SKIPPED")

        directions_by_mode = {
            "seen": [("odd", "odd"), ("even", "even")],
            "unseen": [("odd", "even"), ("even", "odd")],
        }

        for content_mode, save_dir, raw_path, skip in (
            ("seen", seen_dir, seen_raw_path, skip_seen),
            ("unseen", unseen_dir, unseen_raw_path, skip_unseen),
        ):
            if skip:
                continue
            if content_mode == "unseen" and is_per_text:
                continue  # different paragraph IDs across halves → no shared col space

            per_direction_items = []  # Collect per-item results from all directions
            per_direction_agg = []    # Collect aggregated results from all directions
            l2_for_corrections = None

            for direction_idx, (l1_half, l2_half) in enumerate(directions_by_mode[content_mode]):
                # Per-item SEEN/UNSEEN always uses regular features; per-text features loaded per-item in the loop
                df_l1 = halves_data[l1_half]["l1"]
                df_l2 = halves_data[l2_half]["l2"]
                if df_l1 is None or df_l2 is None:
                    continue
                df_l1 = df_l1.copy()
                df_l2 = df_l2.copy()

                item_col = Fields.UNIQUE_PARAGRAPH_ID if Fields.UNIQUE_PARAGRAPH_ID in df_l2.columns else Fields.ARTICLE_ID

                # Halves already split by item parity:
                # halves_data["odd"] = items where item_id % 2 == 1
                # halves_data["even"] = items where item_id % 2 == 0
                # SEEN uses matching halves: (odd, odd) and (even, even)
                # UNSEEN uses opposite halves: (odd, even) and (even, odd)
                fs_l1 = get_feature_set_df(
                    df_l1,
                    feature_set_name, list(feature_cols)
                )
                fs_l2 = get_feature_set_df(
                    df_l2,
                    feature_set_name, list(feature_cols)
                )
                # For UNSEEN, l2_half != l1_half, so fitting on opposite parity
                # For SEEN, l2_half == l1_half, so fitting on same parity
                fit_on_df = get_feature_set_df(
                    df_l2,
                    feature_set_name, list(feature_cols)
                )

                # Save each direction to a temp path so they don't overwrite
                temp_direction_path = raw_path.parent / f"{raw_path.stem}_direction_{direction_idx}{raw_path.suffix}"
                agg_df = eye_score_per_item_independent(
                    fs_l1, fs_l2, temp_direction_path,
                    content_mode="all",
                    fit_on_df=fit_on_df,
                    item_col=item_col,
                    replace_existing=replace_existing,
                    meco_seen_unseen=content_mode,
                    l1_parquet_dir=l1_parquet_dir if content_mode == "seen" and is_per_text else None,
                    l2_parquet_dir=l2_parquet_dir if content_mode == "seen" and is_per_text else None,
                    feature_set=feature_set_name,
                    metric=_require_prototype_metric(scorer),
                )
                per_direction_agg.append(agg_df)

                # Load the per-item results that were saved
                items_path = temp_direction_path.parent / f"{temp_direction_path.stem}_items.csv"
                if items_path.exists():
                    items_df = pd.read_csv(items_path)
                    per_direction_items.append(items_df)

                if l2_for_corrections is None:
                    l2_for_corrections = df_l2

            # Concatenate per-item results from all directions
            if per_direction_items:
                joint_items_df = pd.concat(per_direction_items, ignore_index=True)
                joint_items_path = raw_path.parent / f"{raw_path.stem}_items.csv"
                joint_items_df.to_csv(joint_items_path, index=False)
                logger.info(f"Saved joint per-item results ({len(joint_items_df)} rows) → {joint_items_path}")

            # Aggregate all per-item results by participant (one aggregation, all items together)
            if per_direction_items:
                # Average across ALL per-item rows from all directions, not direction-by-direction
                avg_raw = (
                    joint_items_df.groupby(Fields.SUBJECT_ID, as_index=False)
                    .agg({"eye_score": "mean", Fields.L1: "first"})
                )
                avg_raw = avg_raw.sort_values(by=Fields.SUBJECT_ID).reset_index(drop=True)
            else:
                avg_raw = None

            if avg_raw is None:
                continue

            avg_raw.to_csv(raw_path, index=False)
            logger.info(f"Wrote {content_mode} aggregated eye_scores → {raw_path}")

            # # Apply each correction to the averaged eye_scores
            # base_ctx = {
            #     "feature_set": feature_set_name,
            #     "eye_scores_df": avg_raw,
            #     "all_features_df_L2": l2_for_corrections,
            # }
            # if calibration:
            #     base_ctx.update(calibration)
            # for correction in corrections:
            #     adjusted = apply_correction_to_eyescore(correction, base_ctx)
            #     if adjusted is None:
            #         continue
            #     adj_path = raw_path.with_name(raw_path.stem + correction.suffix + raw_path.suffix)
            #     adjusted.to_csv(adj_path, index=False)

        return (job_id, "DONE")

    except Exception as e:
        import traceback, sys
        tb = traceback.extract_tb(sys.exc_info()[2])
        origin = tb[-1]
        msg = str(e)[:80].replace("\n", " ")
        logger.error(f"Error in {job_id}: {origin.filename}:{origin.lineno} in {origin.name} - {msg}")
        return (job_id, f"ERROR: {origin.filename.split('/')[-1]}:{origin.lineno} {msg}")


def _run_single_eyescore_job(
    dataset: str,
    preview: str,
    version_path: str,
    feature_sets_dict: dict,
    replace_existing: bool,
    content_mode: str = "all",
    results_dirname: str = "results",
    typo_calib_distance_type: str = TYPO_CALIB_DISTANCE_TYPE,
    corrections=DEFAULT_CORRECTIONS,
    scorer: str = DEFAULT_SCORER,
) -> tuple:
    """Run eyescore calculation for a single (dataset, preview, version_path) combo."""
    try:
        src_path = Path.cwd() / Path("src")
        results_root = src_path / f"methods/EyeScore/{results_dirname}"

        job_id = f"{dataset}/{preview}/{version_path}"

        # Check if result already exists
        if not replace_existing:
            # Check if at least one feature set result exists
            results_dir = results_root / f"{dataset}/{preview}/{content_mode}/{version_path}"
            if results_dir.exists():
                existing_files = list(results_dir.glob("*.csv"))
                if existing_files and len(existing_files) >= len(feature_sets_dict):
                    return (job_id, "SKIPPED")

        # Run eyescore calculation for this version
        run_one_dataset_version(
            src_path=src_path,
            path_from_folder=version_path,
            dataset_name=dataset,
            preview=preview,
            replace_existing=replace_existing,
            feature_sets=feature_sets_dict,
            content_mode=content_mode,
            results_root=results_root,
            typo_calib_distance_type=typo_calib_distance_type,
            corrections=corrections,
            scorer=scorer,
        )

        return (job_id, "DONE")

    except Exception as e:
        error_msg = str(e)[:80].replace('\n', ' ')
        return (f"{dataset}/{preview}/{version_path}", f"ERROR: {error_msg}")


def run_eyescore_calculation_parallel(
    datasets: list[str] = None,
    previews: list[str] = None,
    include_fully_agg: bool = True,
    include_first_p_agg: bool = True,
    include_moving_agg: bool = True,
    include_seen_unseen: bool = True,
    include_per_item: bool = True,
    include_split_half: bool = False,
    feature_sets: list[str] = None,
    replace_existing: bool = False,
    skip_fixed_features: bool = False,
    n_jobs: int = -1,
    n_paragraphs_thresholds: list[int] | None = None,
    content_mode: str = "all",
    results_dirname: str = "results",
    typo_calib_distance_type: str = TYPO_CALIB_DISTANCE_TYPE,
    corrections=DEFAULT_CORRECTIONS,
    scorer: str = DEFAULT_SCORER,
) -> dict:
    """Run eyescore calculation in parallel for all combinations.

    Args:
        datasets: Dataset names (default: OneStop, Meco)
        previews: Preview types (default: All, Hunting, Gathering)
        include_fully_agg: Include fully aggregated versions
        include_first_p_agg: Include first N paragraph/article aggregation
        include_moving_agg: Include moving window aggregation
        include_per_item: Include per-item aggregation
        include_split_half: Include split-half versions for reliability analysis
        feature_sets: Feature set names (default: all)
        replace_existing: Replace existing eyescores
        skip_fixed_features: Skip fixed text features (WFC/TRANSITIONS)
        n_jobs: Number of parallel jobs (-1 = use all CPUs)
        n_paragraphs_thresholds: Thresholds for paragraph-level aggregation
        content_mode: Content mode for calculation ("all", "seen", "unseen") - default "all"
    """
    import multiprocessing

    # Set defaults
    if datasets is None:
        datasets = ["OneStop", "Meco"]
    if previews is None:
        previews = ["All", "Hunting", "Gathering"]

    # Prepare feature sets
    if feature_sets is None:
        feature_sets_dict = ALL_FEATURE_SETS_DICT
    else:
        feature_sets_dict = {k: v for k, v in ALL_FEATURE_SETS_DICT.items() if k in feature_sets}

    # Skip fixed text features if requested
    if skip_fixed_features:
        feature_sets_dict = {k: v for k, v in feature_sets_dict.items() if k not in FIXED_TEXT_GROUP_PREFIXES}

    # The huge per-text parquets are only ever read by TRANSITIONS/WFC jobs
    # (is_per_text branches). When no per-text feature set is in the run,
    # skip loading them during job discovery — for MECO seen/unseen and
    # split-half this is the difference between holding 40+ parquet pairs in
    # memory and holding none.
    needs_per_text = any(fs in FIXED_TEXT_GROUP_PREFIXES for fs in feature_sets_dict)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting parallel eyescore calculation with {n_jobs} workers")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Previews: {previews}")
    logger.info(f"  Agg types: fully_agg={include_fully_agg}, first_p_agg={include_first_p_agg}, moving_agg={include_moving_agg}, per_item={include_per_item}, seen_unseen={include_seen_unseen}    ")
    logger.info(f"  Content mode: {content_mode}")
    logger.info(f"  Feature sets: {len(feature_sets_dict)}")
    logger.info(f"{'='*60}\n")

    print(f"\n>>> [{datetime.now().strftime('%H:%M:%S')}] Starting eyescore calculation with {n_jobs} workers")
    print(f">>> Monitor with: ps aux | grep run_one_dataset_version | grep -v grep | wc -l\n")

    # Build all jobs: for each dataset, discover available agg versions
    all_jobs = []

    for dataset in datasets:
        dataset_key = dataset + "L2"
        dataset_agg_configs = MECO_AGG_CONFIGS if dataset == "Meco" else ONESTOP_AGG_CONFIGS

        # Filter agg configs based on what's enabled
        filtered_configs = []
        for cfg in dataset_agg_configs:
            agg_type = cfg.get("agg_type")
            if (agg_type == AggTypes.FULL and include_fully_agg) or \
               (agg_type == AggTypes.FIRST_P and include_first_p_agg) or \
               (agg_type == AggTypes.MOVING_P and include_moving_agg) or \
               (agg_type == AggTypes.PER_ITEM and include_per_item and not include_seen_unseen):
                filtered_configs.append(cfg)

        # Map configs to version paths
        for preview, config in itertools.product(previews, filtered_configs):
            # Skip preview filters for Meco (no preview data)
            if dataset == "Meco" and preview != "All":
                continue

            agg_type = config["agg_type"]
            p_agg_level = config.get("p_agg_level")
            p_agg = config.get("p_agg")

            # Build version path (same format as feature extraction)
            if agg_type == AggTypes.FULL:
                version_path = f"fully_agg/{('all' if dataset == 'Meco' else 'ordinary')}/all"
            elif agg_type == AggTypes.PER_ITEM:
                version_path = f"{agg_type}/{('all' if dataset == 'Meco' else 'ordinary')}/all/{p_agg_level}"
            else:
                version_path = f"{agg_type}/{('all' if dataset == 'Meco' else 'ordinary')}/all/{p_agg_level}_{p_agg}"

            all_jobs.append((dataset, preview, version_path, feature_sets_dict, replace_existing, content_mode))

    # ── MECO seen_unseen jobs (one per feature_set, halves loaded once per dataset×preview) ──
    # Only run normal fully_agg seen/unseen if other agg types are enabled
    # (If only per_item_agg is enabled with seen_unseen, use per_item_seen_unseen instead)
    seen_unseen_jobs = []
    seen_unseen_calibration = {}
    # `not include_per_item` lets --agg-types seen_unseen stand alone: the block
    # below builds its jobs from the half-split feature files directly and never
    # reads filtered_configs, so the only thing that required a base agg type was
    # this guard — whose actual purpose is to hand "per_item + seen_unseen" to the
    # per_item_seen_unseen path below instead. Existing invocations are unaffected:
    # fully_agg+seen_unseen still enters here, per_item+seen_unseen still does not.
    if include_seen_unseen and (include_fully_agg or include_first_p_agg
                                or include_moving_agg or not include_per_item):
        for dataset in datasets:
            if dataset != "Meco":
                # OneStop's seen/unseen is its own content_group split, not the half-split.
                continue
            for preview in previews:
                if preview != "All":
                    continue
                # Load both halves' L1 + L2 (scalar + per-text) once for this (dataset, preview)
                halves_data = {}
                missing = False
                for half in SEEN_UNSEEN_HALVES:
                    l1_p, l2_p = _seen_unseen_feature_paths_eyescore(dataset, half)
                    if not l1_p.exists() or not l2_p.exists():
                        logger.warning(f"Skipping seen_unseen/{dataset}/{half}: features missing")
                        missing = True
                        break
                    # Use the pair loader so per-text columns are aligned
                    # (union, 0-filled) between L1 and L2 within this half.
                    # The singular loader leaves cohort-specific columns
                    # missing on the other side, which breaks the cosine.
                    pt_l1, pt_l2 = (
                        load_per_text_features_pair(l1_p, l2_p)
                        if needs_per_text else (None, None)
                    )
                    halves_data[half] = {
                        "l1": pd.read_csv(l1_p),
                        "l2": pd.read_csv(l2_p),
                        "per_text_l1": pt_l1,
                        "per_text_l2": pt_l2,
                    }
                if missing:
                    continue

                # Pre-compute typological distances once per (dataset, preview).
                calibration = {"target_col": TYPO_CALIB_TARGET_COL}
                try:
                    from src.methods.EyeScore.typology import compute_typological_distances
                    l1_set = halves_data["odd"]["l2"][Fields.L1].dropna().unique().tolist()
                    calibration["typological_distances"] = compute_typological_distances(
                        l1_set, distance_type=typo_calib_distance_type,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not compute typological distances ({e}); "
                        f"typology corrections will be skipped for {dataset}/{preview}."
                    )

                seen_unseen_calibration[(dataset, preview)] = calibration
                # Iterate thresholds (None = no filter, the unfiltered baseline).
                # Each threshold becomes a distinct version_path subdir under
                # fully_agg/.
                thresholds_to_run = list(n_paragraphs_thresholds) if n_paragraphs_thresholds else [None]
                for fset_name, fset_cols in feature_sets_dict.items():
                    for min_n in thresholds_to_run:
                        seen_unseen_jobs.append((
                            halves_data, dataset, preview, fset_name, fset_cols,
                            replace_existing, calibration, corrections, min_n,
                        ))

    # ── MECO per-item seen_unseen jobs (per-item agg with half-splits) ──
    per_item_seen_unseen_jobs = []
    if include_per_item and include_seen_unseen:
        for dataset in datasets:
            if dataset != "Meco":
                # Only MECO supports per-item seen/unseen (via half-splits)
                continue
            for preview in previews:
                if preview != "All":
                    continue
                # Load full per-item L1 + L2 data (function will compute halves internally)
                l1_path = Path(DATA_PATH / f"{dataset}L1/features_and_targets/per_item_agg/all/all/paragraph/features_and_metadata.csv")
                l2_path = Path(DATA_PATH / f"{dataset}L2/features_and_targets/per_item_agg/all/all/paragraph/features_and_metadata.csv")

                if not l1_path.exists() or not l2_path.exists():
                    logger.warning(f"Skipping per-item seen_unseen/{dataset}: per-item data missing")
                    continue

                l1_df = pd.read_csv(l1_path)
                l2_df = pd.read_csv(l2_path)

                full_data = {
                    "l1": l1_df,
                    "l2": l2_df,
                }

                # Parquet dirs for per-item fixed text features (TRANSITIONS, WFC)
                l1_parquet_dir = l1_path.parent
                l2_parquet_dir = l2_path.parent

                # Pre-compute typological distances once per (dataset, preview)
                calibration = {"target_col": TYPO_CALIB_TARGET_COL}
                try:
                    from src.methods.EyeScore.typology import compute_typological_distances
                    l1_set = l2_df[Fields.L1].dropna().unique().tolist()
                    calibration["typological_distances"] = compute_typological_distances(
                        l1_set, distance_type=TYPO_CALIB_DISTANCE_TYPE,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not compute typological distances for per-item ({e}); "
                        f"typology corrections will be skipped for {dataset}/{preview}."
                    )

                thresholds_to_run = list(n_paragraphs_thresholds) if n_paragraphs_thresholds else [None]
                for fset_name, fset_cols in feature_sets_dict.items():
                    for min_n in thresholds_to_run:
                        per_item_seen_unseen_jobs.append((
                            full_data, dataset, preview, fset_name, fset_cols,
                            replace_existing, calibration, corrections, min_n,
                            l1_parquet_dir, l2_parquet_dir,
                        ))

    # ── Split-half eyescore jobs ──
    split_half_jobs = []
    split_half_calibration = {}
    split_half_common_cols: dict = {}  # split_idx -> columns common to all its frames
    if include_split_half:
        for dataset in datasets:
            # Discover split-half directories
            if dataset == "OneStop":
                # OneStop: split_half/fully_agg/ordinary/all/{paragraph|article}/split_N/half_M/
                reread = "ordinary"
                base_path = DATA_PATH / f"{dataset}L1/features_and_targets/split_half/fully_agg/{reread}/all"
                item_levels = ["paragraph", "article"]
            else:  # Meco
                # MECO: split_half/fully_agg/all/all/split_N/half_M/
                reread = "all"
                base_path = DATA_PATH / f"{dataset}L1/features_and_targets/split_half/fully_agg/{reread}/all"
                item_levels = ["all"]

            # For OneStop, iterate over previews; for MECO use "All"
            split_half_previews = previews if dataset == "OneStop" else ["All"]

            for preview in split_half_previews:
                # Skip non-All previews for Meco
                if dataset == "Meco" and preview != "All":
                    continue

                for item_level in item_levels:
                    level_path = base_path / item_level if dataset == "OneStop" else base_path
                    if not level_path.exists():
                        continue

                    # Find all split_N/half_M combos
                    for split_dir in sorted(level_path.glob("split_*")):
                        if not split_dir.is_dir():
                            continue
                        split_idx = int(split_dir.name.split("_")[1])

                        for half_dir in sorted(split_dir.glob("half_*")):
                            if not half_dir.is_dir():
                                continue
                            half_num = int(half_dir.name.split("_")[1])

                            # Check if features exist (location varies by dataset)
                            if dataset == "Meco":
                                # MECO: check if odd/even subdirs exist
                                l1_odd = half_dir / "seen_unseen/odd/features_and_metadata.csv"
                                if not l1_odd.exists():
                                    continue
                                # Pre-load MECO odd/even halves for this split/half
                                halves_data = {}
                                missing_halves = False
                                for odd_even in SEEN_UNSEEN_HALVES:
                                    l1_p = half_dir / f"seen_unseen/{odd_even}/features_and_metadata.csv"
                                    l2_p = (DATA_PATH / f"{dataset}L2/features_and_targets/split_half/fully_agg/{reread}/all/split_{split_idx}/half_{half_num}/seen_unseen/{odd_even}/features_and_metadata.csv")

                                    if not l1_p.exists() or not l2_p.exists():
                                        missing_halves = True
                                        break

                                    pt_l1, pt_l2 = (
                                        load_per_text_features_pair(l1_p, l2_p)
                                        if needs_per_text else (None, None)
                                    )
                                    halves_data[odd_even] = {
                                        "l1": pd.read_csv(l1_p),
                                        "l2": pd.read_csv(l2_p),
                                        "per_text_l1": pt_l1,
                                        "per_text_l2": pt_l2,
                                    }

                                if missing_halves:
                                    continue

                                # Pre-compute typological distances once per split/half
                                cal_key = (dataset, split_idx, half_num)
                                if cal_key not in split_half_calibration:
                                    calibration = {"target_col": TYPO_CALIB_TARGET_COL}
                                    try:
                                        from src.methods.EyeScore.typology import compute_typological_distances
                                        l1_set = halves_data["odd"]["l2"][Fields.L1].dropna().unique().tolist()
                                        calibration["typological_distances"] = compute_typological_distances(
                                            l1_set, distance_type=TYPO_CALIB_DISTANCE_TYPE,
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Could not compute typological distances for split-half {split_idx}/{half_num} ({e})"
                                        )
                                    split_half_calibration[cal_key] = calibration

                                # Restrict to columns present in EVERY frame of this
                                # split, so half_1 and half_2 are scored on identical
                                # features. Without it, rare PTB tags that are absent
                                # from one half's paragraphs silently shrink that
                                # half's feature set (MECO S_CLUSTERS_NO_NORM spans
                                # 222-252 of 258 columns and differs across halves in
                                # all 20 splits), and the split-half correlation then
                                # mixes score stability with feature-set churn.
                                # get_feature_set_df drops missing columns with a
                                # warning rather than raising, so this never surfaced
                                # as an error the way it did in the predictions path.
                                if split_idx not in split_half_common_cols:
                                    split_half_common_cols[split_idx] = (
                                        split_half_common_feature_cols(dataset, split_idx)
                                    )
                                common_cols = split_half_common_cols[split_idx]
                                for fset_name, fset_cols in feature_sets_dict.items():
                                    # Per-text sets carry an empty static list and are
                                    # resolved by prefix inside get_feature_set_df.
                                    job_cols = (
                                        fset_cols
                                        if fset_name in FIXED_TEXT_GROUP_PREFIXES
                                        else apply_split_common_filter(
                                            fset_cols, common_cols,
                                            f"{dataset}/split_{split_idx}/half_{half_num}/{fset_name}",
                                        )
                                    )
                                    split_half_jobs.append((
                                        halves_data, dataset, preview, fset_name, job_cols,
                                        replace_existing, split_half_calibration[cal_key], corrections, None,
                                        split_idx, half_num, reread,
                                    ))
                            else:
                                # OneStop: check if features exist at half_N level, then add jobs
                                l1_features = half_dir / "features_and_metadata.csv"
                                if l1_features.exists():
                                    version_path = f"split_half/fully_agg/{reread}/all/{item_level}/split_{split_idx}/half_{half_num}"
                                    all_jobs.append((dataset, preview, version_path, feature_sets_dict, replace_existing, content_mode))

    total_jobs = len(all_jobs) + len(seen_unseen_jobs) + len(per_item_seen_unseen_jobs) + len(split_half_jobs)
    logger.info(f"Total jobs to run: {total_jobs} ({len(all_jobs)} regular + {len(seen_unseen_jobs)} seen_unseen + {len(per_item_seen_unseen_jobs)} per_item_seen_unseen + {len(split_half_jobs)} split_half)")

    # Run eyescores in parallel.
    #
    # The eyescore calculation is CPU-bound pure-Python pandas/numpy (per-row
    # z-scoring + cosine), so a ThreadPoolExecutor barely scales — every hot
    # frame gets serialized on the GIL. A ProcessPoolExecutor runs jobs in
    # separate interpreters and scales with cores; on Linux it forks, so
    # workers inherit already-imported modules cheaply. All job callables are
    # module-level and all args (DataFrames/dicts/strings) are picklable.
    # Override with EYESCORE_EXECUTOR=thread to fall back (e.g. for debugging
    # or if a future job arg becomes unpicklable).
    import os
    executor_kind = os.environ.get("EYESCORE_EXECUTOR", "process").lower()
    Executor = ThreadPoolExecutor if executor_kind == "thread" else ProcessPoolExecutor
    logger.info(f"Using {Executor.__name__} with {n_jobs} workers for calculation.")
    results = []
    with Executor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _run_single_eyescore_job, dataset, preview, version_path,
                feature_sets_dict, replace_existing, content_mode,
                results_dirname, typo_calib_distance_type,
                corrections=corrections, scorer=scorer,
            )
            for dataset, preview, version_path, feature_sets_dict, replace_existing, content_mode in all_jobs
        ]
        futures += [
            executor.submit(_run_meco_seen_unseen_eyescore_job, *job, results_dirname=results_dirname, scorer=scorer)
            for job in seen_unseen_jobs
        ]
        futures += [
            executor.submit(_run_meco_per_item_seen_unseen_eyescore_job, *job, results_dirname=results_dirname, scorer=scorer)
            for job in per_item_seen_unseen_jobs
        ]
        futures += [
            executor.submit(_run_meco_split_half_seen_unseen_eyescore_job, *job, results_dirname=results_dirname, scorer=scorer)
            for job in split_half_jobs
        ]

        completed = 0
        for future in as_completed(futures):
            job_id, status = future.result()
            results.append((job_id, status))
            completed += 1

            if completed % max(1, total_jobs // 10) == 0:
                pct = int(100 * completed / total_jobs)
                logger.info(f"Progress: {completed}/{total_jobs} ({pct}%) - {job_id}: {status}")

    # Summary
    done = sum(1 for _, s in results if s == "DONE")
    skipped = sum(1 for _, s in results if s == "SKIPPED")
    errors = sum(1 for _, s in results if "ERROR" in s)
    other = total_jobs - done - skipped - errors

    logger.info(f"\n{'='*60}")
    logger.info("Eyescore Calculation Summary:")
    logger.info(f"  ✓ Done: {done}")
    logger.info(f"  ⊘ Skipped (exist): {skipped}")
    logger.info(f"  ✗ Errors: {errors}")
    logger.info(f"  - Other: {other}")
    logger.info(f"{'='*60}\n")

    if errors > 0:
        logger.error("Some eyescore calculations failed:")
        for job_id, status in results:
            if "ERROR" in status:
                logger.error(f"  {job_id}: {status}")

    return {"done": done, "skipped": skipped, "errors": errors, "total": total_jobs}


def run_eyescore_evaluation_and_plotting(
    datasets: list[str] = None,
    previews: list[str] = None,
    include_fully_agg: bool = True,
    include_first_p_agg: bool = True,
    include_moving_agg: bool = True,
    include_seen_unseen: bool = True,
    include_per_item: bool = True,
    feature_sets: list[str] = None,
    results_dir: Path = None,
    eval_save_dir: Path = None,
    plots_save_dir: Path = None,
    lang_bias_plots_dir: Path = None,
    distance_types: list[str] = None,
    results_suffix: str = "",
    include_split_half: bool = False,
    skip_plots: bool = False,
    skip_lang_bias: bool = False,
) -> None:
    """Run eyescore evaluation and plotting for all versions.

    Evaluates eyescores against reading comprehension test scores.

    Args:
        datasets: Dataset names (default: OneStop, Meco)
        previews: Preview types (default: All, Hunting, Gathering)
        include_fully_agg: Include fully aggregated versions
        include_first_p_agg: Include first N paragraph/article aggregation
        include_moving_agg: Include moving window aggregation
        feature_sets: Feature sets to evaluate (default: all)
        results_dir: Eye_score CSV tree to read (default: canonical
            src/methods/EyeScore/results/).
        eval_save_dir: Where to write per-feature-set evaluation CSVs
            (default: canonical src/evaluation/EyeScore/results/).
        plots_save_dir: Where to write the evaluation plots
            (default: canonical src/evaluation/EyeScore/plots/).
        lang_bias_plots_dir: Where to write the language-bias plots/CSV
            (default: canonical src/evaluation/EyeScore/plots/lang_bias/).
        distance_types: URIEL+ distance types passed to the lang_bias step
            (default: ["syntactic+genetic"]).
    """
    # Set defaults
    if datasets is None:
        datasets = ["OneStop", "Meco"]
    if previews is None:
        previews = ["All", "Hunting", "Gathering"]

    logger.info(f"\n{'='*60}")
    logger.info("Running eyescore evaluation and plotting")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Previews: {previews}")
    if feature_sets:
        logger.info(f"  Feature sets: {len(feature_sets)} selected")
    if results_dir is not None:
        logger.info(f"  Reading eye_scores from: {results_dir}")
    if eval_save_dir is not None:
        logger.info(f"  Writing eval CSVs to: {eval_save_dir}")
    if plots_save_dir is not None:
        logger.info(f"  Writing plots to: {plots_save_dir}")
    if lang_bias_plots_dir is not None:
        logger.info(f"  Writing lang_bias to: {lang_bias_plots_dir}")
    if distance_types is not None:
        logger.info(f"  Lang_bias distance types: {distance_types}")
    logger.info(f"{'='*60}\n")

    try:
        logger.info("Stage 2a: Evaluating eyescores against test scores...")
        # Same deny-list as evaluation.py's CLI default, so this stage and a
        # direct `python -m src.evaluation.EyeScore.evaluation` sweep the same
        # tree. Without it this call evaluated every agg type on disk, including
        # the window regimes whose n reaches 27,700 (~103 GB per bootstrap call
        # at N_BOOTSTRAP=100k) and the split-half tree scored elsewhere.
        from src.evaluation.EyeScore.evaluation import DEFAULT_EXCLUDE_AGG_TYPES
        _exclude_agg_types = [
            a for a in DEFAULT_EXCLUDE_AGG_TYPES
            if not (include_split_half and a == "split_half")
        ]
        if include_split_half:
            logger.info("--include-split-half: evaluating the split-half tree too "
                        "(~27x the files, ~7x the peak bootstrap memory)")
        eval_df = evaluate_all_results(
            results_dir=results_dir, save_dir=eval_save_dir,
            feature_sets=feature_sets,
            exclude_agg_types=_exclude_agg_types,
        )
        logger.info(f"✓ Evaluation complete: {len(eval_df)} rows")

        # Per-scheme lang_bias eval subtree — written under
        # {eval_save_dir}/lang_bias/{scheme}/{org,typo_calibrated}/.
        # The triplet table in tables.py (Stage 2d) reads from here; without
        # this loop those tables come out empty. Mirrors the same loop in
        # evaluation.py's __main__.
        from src.evaluation.EyeScore.evaluation import (
            EVAL_SAVE_DIR as _CANON_EVAL_SAVE_DIR,
        )
        bias_eval_root = (eval_save_dir or _CANON_EVAL_SAVE_DIR) / "lang_bias"
        for scheme in (() if skip_lang_bias else SPLIT_SCHEMES):
            logger.info(f"Stage 2a (lang_bias): scheme={scheme} org + typo_calibrated")
            evaluate_all_results(
                results_dir=results_dir,
                save_dir=bias_eval_root / scheme / "org",
                feature_sets=feature_sets, variant=RAW_VARIANT,
                exclude_agg_types=_exclude_agg_types,
            )
            evaluate_all_results(
                results_dir=results_dir,
                save_dir=bias_eval_root / scheme / "typo_calibrated",
                feature_sets=feature_sets,
                variant=SPLIT_SCHEME_TO_VARIANT[scheme],
                exclude_agg_types=_exclude_agg_types,
            )
        logger.info("✓ Lang_bias per-scheme eval CSVs written")

        if skip_plots:
            logger.info("Stage 2b: SKIPPED (--skip-plots)")
        else:
            logger.info("Stage 2b: Generating evaluation plots...")
            plot_all_results(
                eval_dir=eval_save_dir, plots_dir=plots_save_dir,
                results_dir=results_dir, feature_sets=feature_sets,
            )
            logger.info("✓ Evaluation plots complete")

        if skip_lang_bias:
            logger.info("Stage 2c: SKIPPED (--skip-lang-bias) -- lang_bias plots "
                        "and the bias/triplet tables keep their previous numbers")
        else:
            logger.info("Stage 2c: Running language bias analysis...")
            # org (raw) residual-vs-distance — the pre-debiasing picture (same for
            # every debias method).
            run_language_bias_analysis(
                results_dir=results_dir, plots_dir=lang_bias_plots_dir,
                feature_sets=feature_sets, distance_types=distance_types,
            )
            # Debiased residual-vs-distance, one subtree per split scheme, using the
            # corrected variant — so the plots actually reflect the correction and
            # differ across methods. (re_l1 has no across variant → that scheme just
            # finds no CSVs and is skipped.)
            _lb_base = lang_bias_plots_dir if lang_bias_plots_dir is not None else LANG_BIAS_PLOTS_DIR
            for scheme in SPLIT_SCHEMES:
                run_language_bias_analysis(
                    results_dir=results_dir,
                    plots_dir=Path(_lb_base) / scheme / "typo_calibrated",
                    feature_sets=feature_sets, distance_types=distance_types,
                    variant=SPLIT_SCHEME_TO_VARIANT[scheme],
                )
            logger.info("✓ Language bias analysis complete")

        # Stage 2d is delegated to tables.py as a subprocess — its __main__
        # handles the --results-suffix global rebinding (RESULTS_DIR /
        # EVAL_SAVE_DIR / TABLES_SAVE_DIR / BIAS_SUMMARY_DIR / distance type),
        # which would be brittle to replicate via in-process imports.
        import sys
        logger.info("Stage 2d: Generating LaTeX tables...")
        tables_cmd = [sys.executable, "-m", "src.evaluation.EyeScore.tables"]
        if results_suffix:
            tables_cmd.extend(["--results-suffix", results_suffix])
            # Pass the distance EXPLICITLY. Without it tables.py falls back to
            # inferring the distance from the suffix, and a composed suffix like
            # "two_step_zsyngenscr" is not a URIEL metric — every typological
            # distance comes back NaN and the lang_bias tables render as all
            # dashes (looking like a null result rather than a broken lookup).
            if distance_types:
                tables_cmd.extend(["--distance-type", str(distance_types[0])])
        try:
            subprocess.run(tables_cmd, check=True)
            logger.info("✓ Tables complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Tables generation failed (exit {e.returncode}); continuing")

        tables_save_dir = (
            f"src/evaluation/EyeScore/tables_{results_suffix}/"
            if results_suffix else "src/evaluation/EyeScore/tables/"
        )

        logger.info(f"\n{'='*60}")
        logger.info("Eyescore Evaluation Summary:")
        logger.info(f"  ✓ Evaluation results: {eval_save_dir or 'src/evaluation/EyeScore/results/'}")
        logger.info(f"  ✓ Plots: {plots_save_dir or 'src/evaluation/EyeScore/plots/'}")
        logger.info(f"  ✓ Language bias analysis: {lang_bias_plots_dir or 'src/evaluation/EyeScore/plots/lang_bias/'}")
        logger.info(f"  ✓ Tables: {tables_save_dir}")
        logger.info(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"✗ Evaluation and plotting failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Stage selection
    parser.add_argument(
        '--stage',
        type=str,
        choices=['calculation', 'evaluation', 'both'],
        default='both',
        help='Which pipeline stage to run (default: both)'
    )

    # Eyescore calculation options
    parser.add_argument(
        '--datasets',
        type=str,
        default='OneStop,Meco',
        help='Comma-separated dataset names (default: OneStop,Meco)'
    )
    parser.add_argument(
        '--previews',
        type=str,
        help='Comma-separated preview types (e.g. all,hunting,gathering for OneStop; all for Meco)'
    )
    parser.add_argument(
        '--agg-types',
        type=str,
        default='fully_agg,first_p_agg,moving_p_agg',
        help='Comma-separated aggregation types (default: fully_agg,first_p_agg,moving_p_agg)'
    )
    parser.add_argument(
        '--no-seen-unseen',
        action='store_true',
        help='Skip MECO seen/unseen (half-split) calculation'
    )
    parser.add_argument(
        '--split-half-eyescore',
        action='store_true',
        help='Calculate eyescore for split-half versions (reliability analysis)'
    )
    parser.add_argument(
        '--n-paragraphs-thresholds',
        type=str,
        help='Comma-separated n_paragraphs minimums for MECO seen/unseen '
             '(e.g. "1,2,3,4,5,6"). Filters odd and even cohorts independently. '
             'Default: no filter.'
    )
    parser.add_argument(
        '--feature-sets',
        type=str,
        help='Comma-separated feature set names (default: all)'
    )
    parser.add_argument(
        '--feature-set-size',
        type=str,
        choices=['small', 'all'],
        default='small',
        help='Feature set size: "small" (7 curated sets — the only ones any '
             'table/plot consumes) or "all" (31 sets; the extra 24 are computed '
             'but read by no downstream artifact). Default: small.'
    )
    parser.add_argument(
        '--replace-existing',
        action='store_true',
        help='Replace existing eyescore calculations'
    )
    parser.add_argument(
        '--skip-fixed-features',
        action='store_true',
        help='Skip eyescore calculation for fixed text features (TRANSITIONS, WFC)'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Stage 2: skip evaluation plot rendering (Stage 2b). Measured at '
             '~29 min of a ~3h49m evaluation stage. Eval CSVs are unaffected.'
    )
    parser.add_argument(
        '--skip-lang-bias',
        action='store_true',
        help='Stage 2: skip the four per-scheme lang_bias eval passes and the '
             'language bias analysis (Stage 2c). Together ~2.5 h of a ~3h49m '
             'stage, the single biggest saving. Leaves the bias/triplet tables '
             'reading their previous numbers, so only use it when those are '
             'not being refreshed.'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Parallel workers: -1=all CPUs, 1=sequential (default: -1)'
    )
    parser.add_argument(
        '--content-mode',
        type=str,
        default='all',
        help='Content mode(s) for calculation: comma-separated list from ["all", "seen", "unseen"] (default: all). Per-text features require "seen" mode. Example: --content-mode all,seen,unseen'
    )
    parser.add_argument(
        '--typo-calib-distance-type',
        type=str,
        default=TYPO_CALIB_DISTANCE_TYPE,
        help='URIEL+ distance type used to fit the typology calibration α. '
             'Default: %(default)s. Use e.g. "syntactic" to swap the calibration '
             'metric; combine with --results-suffix to route the new CSVs to a '
             'sibling results tree so existing outputs are not overwritten.'
    )
    parser.add_argument(
        '--results-suffix',
        type=str,
        default='',
        help='Suffix appended to the result/output directory names when '
             'set, so a non-default --typo-calib-distance-type run can '
             'coexist with the canonical one. Stage 1 writes eye_score CSVs '
             'to methods/EyeScore/results_<suffix>/; Stage 2 reads from that '
             'tree and routes its outputs to evaluation/EyeScore/results_<suffix>/, '
             'evaluation/EyeScore/plots_<suffix>/, and the lang_bias subdir '
             'under it. Default: empty (canonical results/ + plots/ trees).'
    )
    parser.add_argument(
        '--center-distance',
        action='store_true',
        help='Centre the typological distance on each training fold\'s mean '
             'before subtracting the debiasing term, which makes the correction '
             'invariant to affine rescalings of the distance (see '
             '_apply_typology_correction). The paper\'s *_cat3ctr trees were '
             'built with it. Sets EYESCORE_CENTER_DISTANCE for this run and its '
             'workers.'
    )

    parser.add_argument(
        '--debias-methods',
        type=str,
        default=None,
        help='Comma-separated debias methods to run, each into its own tree '
             f'(results_<method>/, plots_<method>/, tables_<method>/). Choices: '
             f'{",".join(DEBIAS_METHODS)}. Default: unset → the original single '
             'run into the canonical results/ tree with the two-step correction. '
             'Pass e.g. "two_step,distance_mreg,interaction,re_l1" to fan out.'
    )
    parser.add_argument(
        '--raw-only',
        action='store_true',
        help='Compute and write only the raw eye_score CSVs; skip all bias '
             'corrections (passes an empty correction set). Use to populate the '
             'raw trees fast before running the (expensive) per-fold debiasing '
             'pass separately. Ignored if --debias-methods is also set.'
    )
    parser.add_argument(
        '--scorers',
        type=str,
        default=None,
        help='Comma-separated scorers, each fanned out into its own results tree. '
             f'Choices: {",".join(sorted(SCORER_NAMES))}. Default: {DEFAULT_SCORER} '
             '(canonical results/ tree). Non-default scorers write to '
             'results_<scorer>/ (and _<scorer>_<method>/ when combined with '
             '--debias-methods); cosine stays unmarked so existing outputs are '
             'byte-identical.'
    )
    parser.add_argument(
        '--include-split-half',
        action='store_true',
        help='Stage 2 only: also evaluate the split-half tree (2640 extra files, '
             'n~1107 per file instead of ~152, so ~27x the work and ~7x the peak '
             'bootstrap memory). Off by default — split-half reliability is '
             'computed from the results tree by src/reliability/split_half.py and '
             'never reads these rows, so this is only for inspecting split-half '
             'correlations in the evaluation CSV. Does NOT affect stage 1: '
             'calculating split-half eye_scores is --split-half-eyescore.'
    )

    args = parser.parse_args()
    if args.center_distance:
        # Read at call time by calculation._center_distances_enabled(), and
        # inherited by the worker processes and the Stage-2d tables subprocess.
        os.environ["EYESCORE_CENTER_DISTANCE"] = "1"

    datasets = parse_csv(args.datasets)
    previews = parse_csv(args.previews)
    feature_sets = parse_csv(args.feature_sets)
    agg_types = parse_csv(args.agg_types)
    content_modes = parse_csv(args.content_mode)
    n_paragraphs_thresholds = (
        [int(x) for x in parse_csv(args.n_paragraphs_thresholds)]
        if args.n_paragraphs_thresholds else None
    )

    # Validate content modes
    valid_content_modes = {'all', 'seen', 'unseen'}
    for cm in content_modes:
        if cm not in valid_content_modes:
            raise ValueError(f"Invalid content mode: {cm}. Must be one of {valid_content_modes}")

    # Apply feature set size filter if not explicitly specified
    if feature_sets is None and args.feature_set_size == 'small':
        feature_sets = SMALL_FEATURE_SETS

    # Convert agg_types to boolean flags
    # If agg_types specified, only run those; otherwise run defaults (excluding per_item)
    if agg_types:
        include_fully_agg = "fully_agg" in agg_types
        include_first_p_agg = "first_p_agg" in agg_types
        include_moving_agg = "moving_p_agg" in agg_types
        include_per_item = "per_item_agg" in agg_types
        include_seen_unseen = "seen_unseen" in agg_types
    else:
        # Default: run fully_agg and moving window modes (per_item must be explicitly requested)
        include_fully_agg = True
        include_first_p_agg = True
        include_moving_agg = True
        include_per_item = False
        include_seen_unseen = False

    # Override seen_unseen if --no-seen-unseen flag is set
    if args.no_seen_unseen:
        include_seen_unseen = False

    # Log configuration
    logger.info(f"\n{'='*60}")
    logger.info("Eyescore Pipeline Configuration")
    logger.info(f"  Stage: {args.stage}")
    logger.info(f"  Datasets: {datasets if datasets else 'all'}")
    logger.info(f"  Previews: {previews if previews else 'all'}")
    logger.info(f"  Agg types: {agg_types}")
    logger.info(f"  Feature sets: {len(feature_sets) if feature_sets else 'all'}")
    logger.info(f"  Parallel workers: {args.n_jobs if args.n_jobs > 1 else 'sequential'}")
    logger.info(f"{'='*60}\n")

    # Build the runs. Default (no --debias-methods, no --scorers) preserves the
    # original single run into the canonical results/ tree with cosine + two-step.
    # --debias-methods fans out one tree per method; --scorers fans out one tree
    # per scorer; together they take the product. cosine is the unmarked scorer
    # (keeps results/ byte-identical); non-cosine scorers get a results_<scorer>/
    # (or results_<scorer>_<method>/) sibling tree.
    debias_methods = parse_csv(args.debias_methods)
    if debias_methods:
        unknown = [m for m in debias_methods if m not in DEBIAS_METHODS]
        if unknown:
            raise ValueError(
                f"Unknown --debias-methods {unknown}; choices: {list(DEBIAS_METHODS)}"
            )
        # --results-suffix composes with the method name so the same debias
        # method can be run under two different --typo-calib-distance-type
        # values without the second silently overwriting the first's tree
        # (the distance type is not otherwise encoded in the path).
        method_specs = [
            (m, DEBIAS_METHODS[m], f"{m}_{args.results_suffix}" if args.results_suffix else m)
            for m in debias_methods
        ]
    elif args.raw_only:
        # Raw pass: write only the uncorrected eye_score CSVs (empty correction
        # set), into the canonical results/ tree (or --results-suffix).
        method_specs = [("raw_only", (), args.results_suffix)]
    else:
        method_specs = [(DEFAULT_DEBIAS_METHOD, DEFAULT_CORRECTIONS, args.results_suffix)]

    scorers = parse_csv(args.scorers) or [DEFAULT_SCORER]
    unknown_s = [s for s in scorers if s not in SCORER_NAMES]
    if unknown_s:
        raise ValueError(
            f"Unknown --scorers {unknown_s}; choices: {sorted(SCORER_NAMES)}"
        )

    # (method_name, corrections, run_suffix, scorer). run_suffix combines the
    # scorer token (empty for the default cosine) and the method suffix.
    run_specs = []
    for scorer in scorers:
        scorer_part = "" if scorer == DEFAULT_SCORER else scorer
        for method_name, corrections, method_suffix in method_specs:
            run_suffix = "_".join(p for p in (scorer_part, method_suffix) if p)
            run_specs.append((method_name, corrections, run_suffix, scorer))

    if len(run_specs) > 6:
        logger.warning(
            f"Fanning out {len(run_specs)} result trees (scorers × debias methods)."
        )

    for method_name, corrections, run_suffix, scorer in run_specs:
        results_dirname = f"results_{run_suffix}" if run_suffix else "results"
        if len(run_specs) > 1:
            logger.info(f"\n{'#'*60}")
            logger.info(f"# SCORER: {scorer}  DEBIAS: {method_name}  →  methods/EyeScore/{results_dirname}/")
            logger.info(f"{'#'*60}")

        # Run pipeline stages
        if args.stage in ['calculation', 'both']:
            logger.info(f"\n{'='*60}")
            logger.info("STAGE 1: Eyescore Calculation")
            logger.info(f"  results tree: methods/EyeScore/{results_dirname}/")
            logger.info(f"  typo calib distance: {args.typo_calib_distance_type}")
            logger.info(f"{'='*60}")
            for content_mode in content_modes:
                logger.info(f"\n--- Running calculation for content_mode: {content_mode} ---\n")
                run_eyescore_calculation_parallel(
                    datasets=datasets,
                    previews=previews,
                    include_fully_agg=include_fully_agg,
                    include_first_p_agg=include_first_p_agg,
                    include_moving_agg=include_moving_agg,
                    include_seen_unseen=include_seen_unseen,
                    include_per_item=include_per_item,
                    include_split_half=args.split_half_eyescore,
                    feature_sets=feature_sets,
                    replace_existing=args.replace_existing,
                    skip_fixed_features=args.skip_fixed_features,
                    n_jobs=args.n_jobs,
                    n_paragraphs_thresholds=n_paragraphs_thresholds,
                    content_mode=content_mode,
                    results_dirname=results_dirname,
                    typo_calib_distance_type=args.typo_calib_distance_type,
                    corrections=corrections,
                    scorer=scorer,
                )

        if args.stage in ['evaluation', 'both']:
            logger.info(f"\n{'='*60}")
            logger.info("STAGE 2: Eyescore Evaluation & Analysis")
            logger.info(f"{'='*60}")
            suffix = f"_{run_suffix}" if run_suffix else ""
            if suffix:
                stage2_results_dir = Path(f"src/methods/EyeScore/{results_dirname}")
                stage2_eval_save_dir = Path(f"src/evaluation/EyeScore/results{suffix}")
                stage2_plots_save_dir = Path(f"src/evaluation/EyeScore/plots{suffix}")
                stage2_lang_bias_dir = stage2_plots_save_dir / "lang_bias"
                stage2_distance_types = [args.typo_calib_distance_type]
            else:
                stage2_results_dir = None
                stage2_eval_save_dir = None
                stage2_plots_save_dir = None
                stage2_lang_bias_dir = None
                stage2_distance_types = None
            run_eyescore_evaluation_and_plotting(
                datasets=datasets,
                previews=previews,
                include_fully_agg=include_fully_agg,
                include_first_p_agg=include_first_p_agg,
                include_moving_agg=include_moving_agg,
                include_seen_unseen=include_seen_unseen,
                include_per_item=include_per_item,
                feature_sets=feature_sets,
                results_dir=stage2_results_dir,
                eval_save_dir=stage2_eval_save_dir,
                plots_save_dir=stage2_plots_save_dir,
                lang_bias_plots_dir=stage2_lang_bias_dir,
                distance_types=stage2_distance_types,
                results_suffix=run_suffix,
                include_split_half=args.include_split_half,
                skip_plots=args.skip_plots,
                skip_lang_bias=args.skip_lang_bias,
            )

    logger.info("\n✓ Eyescore pipeline completed successfully!")


if __name__ == '__main__':
    raise SystemExit(main())
