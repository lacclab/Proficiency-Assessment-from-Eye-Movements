"""Tests for the --inner-validation selector on the tree models.

The outer leave-one-participant-out loop is unchanged and identical for every
model; this covers what happens *inside* one outer fold. The contract that
matters:

  * an empty grid is a no-op, so the ridge family and every non-tree model keep
    their exact previous behaviour;
  * the search never sees data it must not see — no L1 augmentation row is ever
    scored against (they carry imputed targets), and the outer held-out
    participant is not in the training frame the inner split partitions;
  * both strategies still produce a prediction for every test participant.

Synthetic data throughout so these run without the data/ tree.
"""
import unittest

import numpy as np
import pandas as pd

from src.constants import (
    Fields, FoldMethod, Pool, InnerValidation, INNER_KFOLD_K, TREE_PARAM_GRIDS,
)
from src.methods.predictions.pool_resolver import PoolSegment
from src.methods.predictions.models.BaseModel import BaseModel
from src.methods.predictions.models.DecisionTree import DecisionTree
from src.methods.predictions.models.RandomForest import RandomForest
from src.methods.predictions.models.RidgeRegression import RidgeRegression

TARGET = "lextale_score"
N_FEATURES = 8
FEATS = [f"f{i}" for i in range(N_FEATURES)]


def _frame(n, seed, native=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_FEATURES))
    beta = rng.standard_normal(N_FEATURES)
    y = np.clip(50 + 10 * (X @ beta) / np.sqrt(N_FEATURES), 0, 100)
    df = pd.DataFrame(X, columns=FEATS)
    df[Fields.SUBJECT_ID] = [f"{'l1' if native else 'l2'}_{seed}_{i}" for i in range(n)]
    df[Fields.L1] = "English" if native else np.repeat(["Spanish", "German"], n)[:n]
    df[TARGET] = y
    return df


class TestEmptyGridIsANoOp(unittest.TestCase):
    """Models that opt out must be bit-identical to the pre-patch path."""

    def test_ridge_family_has_no_grid(self):
        self.assertEqual(RidgeRegression().hyperparameter_grid(), [])

    def test_tuned_wrapper_matches_untuned_for_empty_grid(self):
        l2, l1 = _frame(40, 1), _frame(12, 2, native=True)
        test = _frame(5, 3)

        a = RidgeRegression()
        a.fit_with_augmentation(l1.copy(), l2, 1, FEATS, TARGET)
        want = a.predict_df(test, FEATS)

        b = RidgeRegression()
        b.fit_with_augmentation_tuned(l1.copy(), l2, 1, FEATS, TARGET,
                                      inner_validation=InnerValidation.KFOLD)
        got = b.predict_df(test, FEATS)

        np.testing.assert_allclose(got, want)

    def test_none_strategy_matches_untuned_for_a_tree(self):
        """A tree with a real grid must still be untouched when NONE is asked for."""
        l2, l1 = _frame(40, 4), _frame(12, 5, native=True)
        test = _frame(5, 6)

        a = DecisionTree()
        a.fit_with_augmentation(l1.copy(), l2, 1, FEATS, TARGET)
        want = a.predict_df(test, FEATS)

        b = DecisionTree()
        b.fit_with_augmentation_tuned(l1.copy(), l2, 1, FEATS, TARGET,
                                      inner_validation=InnerValidation.NONE)
        got = b.predict_df(test, FEATS)

        np.testing.assert_allclose(got, want)


class TestSelection(unittest.TestCase):
    def test_trees_expose_a_grid(self):
        for model in (DecisionTree(), RandomForest()):
            self.assertGreater(len(model.hyperparameter_grid()), 1, model.model_name)

    def test_single_config_grid_matches_untuned(self):
        """With nothing to choose between, tuning must not change the answer."""
        l2, l1 = _frame(40, 7), _frame(12, 8, native=True)
        test = _frame(5, 9)
        only = {"max_depth": 6, "min_samples_leaf": 1}

        a = DecisionTree()
        a.apply_config(only)
        a.fit_with_augmentation(l1.copy(), l2, 1, FEATS, TARGET)
        want = a.predict_df(test, FEATS)

        b = DecisionTree()
        b.hyperparameter_grid = lambda: [only]
        b.fit_with_augmentation_tuned(l1.copy(), l2, 1, FEATS, TARGET,
                                      inner_validation=InnerValidation.KFOLD)
        got = b.predict_df(test, FEATS)

        np.testing.assert_allclose(got, want)

    def test_chosen_config_comes_from_the_grid(self):
        l2, l1 = _frame(40, 10), _frame(12, 11, native=True)
        m = RandomForest()
        m.fit_with_augmentation_tuned(l1.copy(), l2, 1, FEATS, TARGET,
                                      inner_validation=InnerValidation.KFOLD)
        self.assertIn(m._chosen_config, TREE_PARAM_GRIDS["Random_Forest"])

    def test_both_strategies_predict_every_participant(self):
        for strategy in (InnerValidation.HOLDOUT, InnerValidation.KFOLD):
            with self.subTest(strategy=strategy):
                l2, l1 = _frame(30, 12), _frame(10, 13, native=True)
                seg = PoolSegment(segment_id="seg", train_pool=l2, test_pool=l2,
                                  train_l1=l1)
                out = DecisionTree().run_pool_fold_segment(
                    seg, l1.copy(), pool=Pool.SEEN, fold_method=FoldMethod.POOL_ALL,
                    feature_cols=FEATS, target_col=TARGET, aug_precentage=1,
                    narrow_l1=False, inner_validation=strategy,
                )
                self.assertIsNotNone(out)
                self.assertEqual(len(out), len(l2))
                self.assertEqual(set(out[Fields.SUBJECT_ID]), set(l2[Fields.SUBJECT_ID]))

    def test_tiny_training_frame_falls_back_to_defaults(self):
        """Below 4 rows there is nothing to hold out; must not raise."""
        l2, l1 = _frame(3, 14), _frame(6, 15, native=True)
        m = DecisionTree()
        m.fit_with_augmentation_tuned(l1.copy(), l2, 1, FEATS, TARGET,
                                      inner_validation=InnerValidation.KFOLD)
        self.assertIsNone(getattr(m, "_chosen_config", None))


class TestSplitHygiene(unittest.TestCase):
    """The inner split must partition exactly the L2 training rows, nothing else."""

    def test_splits_cover_and_partition_the_training_rows(self):
        l2 = _frame(30, 16)
        for strategy in (InnerValidation.HOLDOUT, InnerValidation.KFOLD):
            with self.subTest(strategy=strategy):
                splits = list(BaseModel._inner_splits(DecisionTree(), l2, strategy))
                for tr, va in splits:
                    self.assertEqual(set(tr) & set(va), set(), "train/val overlap")
                    self.assertLessEqual(set(tr) | set(va), set(range(len(l2))))
                    self.assertGreater(len(va), 0)
                if strategy == InnerValidation.KFOLD:
                    self.assertEqual(len(splits), min(INNER_KFOLD_K, len(l2)))
                    covered = np.concatenate([va for _, va in splits])
                    self.assertEqual(sorted(covered), list(range(len(l2))))
                else:
                    self.assertEqual(len(splits), 1)

    def test_no_l1_row_can_reach_a_validation_slice(self):
        """L1 targets are imputed, so scoring against them would be scoring noise.

        _inner_splits is handed only the L2 frame, so its indices can never
        address an L1 row. Pin that: every index must map to an l2_* subject.
        """
        l2 = _frame(25, 17)
        ids = l2[Fields.SUBJECT_ID].to_numpy()
        for _, va in BaseModel._inner_splits(DecisionTree(), l2, InnerValidation.KFOLD):
            self.assertTrue(all(str(s).startswith("l2_") for s in ids[va]))

    def test_held_out_participant_is_absent_from_inner_splits(self):
        """The outer fold removes the test participant before the inner split runs."""
        l2 = _frame(20, 18)
        held = l2[Fields.SUBJECT_ID].iloc[0]
        outer_train = l2[l2[Fields.SUBJECT_ID] != held].reset_index(drop=True)
        ids = outer_train[Fields.SUBJECT_ID].to_numpy()
        for tr, va in BaseModel._inner_splits(DecisionTree(), outer_train,
                                              InnerValidation.KFOLD):
            self.assertNotIn(held, ids[tr])
            self.assertNotIn(held, ids[va])


class TestGridShape(unittest.TestCase):
    def test_configs_in_a_grid_share_the_same_keys(self):
        """apply_config mutates a persistent estimator via set_params.

        Nothing resets it between configs, so a config that omitted a key would
        silently inherit the previous config's value for it — the search would
        then be scoring something that is not in the grid. Uniform key sets make
        each apply_config a complete overwrite.
        """
        for name, grid in TREE_PARAM_GRIDS.items():
            with self.subTest(model=name):
                self.assertEqual(len({frozenset(c) for c in grid}), 1)

    def test_no_duplicate_configs(self):
        for name, grid in TREE_PARAM_GRIDS.items():
            with self.subTest(model=name):
                seen = {tuple(sorted(c.items(), key=lambda kv: kv[0])) for c in grid}
                self.assertEqual(len(seen), len(grid))


class TestContract(unittest.TestCase):
    def test_strategy_values_are_the_contract(self):
        self.assertEqual({e.value for e in InnerValidation},
                         {"none", "holdout", "kfold"})

    def test_bad_strategy_rejected(self):
        l2, l1 = _frame(20, 19), _frame(8, 20, native=True)
        with self.assertRaises(ValueError):
            DecisionTree().fit_with_augmentation_tuned(
                l1.copy(), l2, 1, FEATS, TARGET, inner_validation="five-fold")

    def test_degenerate_scores_do_not_propagate_nan(self):
        self.assertEqual(BaseModel._inner_score(np.array([1.0]), np.array([2.0])), -1.0)
        self.assertEqual(
            BaseModel._inner_score(np.array([5.0, 5.0]), np.array([1.0, 2.0])), -1.0)


if __name__ == "__main__":
    unittest.main()
