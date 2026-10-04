"""Tests for the --loo-mode / PROF_LOO_MODE selector on the POOL_ALL fast path.

Covers the contract that matters for reproducibility: `crossfit` keeps the
legacy stratified A/B behaviour (and therefore stays sensitive to
RIDGE_CF_SEED), `nested` is deterministic, `auto` routes by segment size, and
the default is `nested`.

Synthetic data throughout so these run without the data/ tree.
"""
import os
import unittest

import numpy as np
import pandas as pd

from src.constants import Fields, FoldMethod, Pool
from src.methods.predictions.pool_resolver import PoolSegment
from src.methods.predictions.models.RidgeRegression import (
    RidgeRegression, LogRidgeRegression, LOO_MODES, _AUTO_NESTED_BELOW_N,
)

TARGET = "lextale_score"
N_FEATURES = 12


def _frame(n, seed, native=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_FEATURES))
    beta = rng.standard_normal(N_FEATURES)
    y = np.clip(50 + 10 * (X @ beta) / np.sqrt(N_FEATURES), 0, 100)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    df[Fields.SUBJECT_ID] = [f"{'l1' if native else 'l2'}_{seed}_{i}" for i in range(n)]
    df[Fields.L1] = "English" if native else np.repeat(["Spanish", "German"], n)[:n]
    df[TARGET] = y
    return df


def _segment(n_l2=30, seed=0):
    l2 = _frame(n_l2, seed)
    l1 = _frame(max(8, n_l2 // 3), seed + 100, native=True)
    return PoolSegment(segment_id="seg", train_pool=l2, test_pool=l2, train_l1=l1), l1


def _run(model, segment, l1):
    return model.try_loocv_segment(
        segment, l1, fold_method=FoldMethod.POOL_ALL, pool=Pool.SEEN,
        feature_cols=[f"f{i}" for i in range(N_FEATURES)], target_col=TARGET,
        aug_precentage=1, narrow_l1=False,
    )


def _preds(results):
    return np.array([r["pred"] for r in sorted(results, key=lambda r: r[Fields.SUBJECT_ID])])


class TestLooModeSelection(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("PROF_LOO_MODE", "RIDGE_CF_SEED")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_default_is_nested(self):
        """No env var, no explicit arg -> nested."""
        segment, l1 = _segment()
        default = _preds(_run(RidgeRegression(choose_alpha=True), segment, l1))
        nested = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="nested"), segment, l1))
        np.testing.assert_allclose(default, nested)

    def test_env_var_selects_mode(self):
        segment, l1 = _segment()
        os.environ["PROF_LOO_MODE"] = "crossfit"
        env = _preds(_run(RidgeRegression(choose_alpha=True), segment, l1))
        explicit = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="crossfit"), segment, l1))
        np.testing.assert_allclose(env, explicit)

    def test_explicit_arg_beats_env_var(self):
        segment, l1 = _segment()
        os.environ["PROF_LOO_MODE"] = "crossfit"
        got = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="nested"), segment, l1))
        want = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="nested"), segment, l1))
        np.testing.assert_allclose(got, want)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            RidgeRegression(loo_mode="bogus")
        os.environ["PROF_LOO_MODE"] = "bogus"
        segment, l1 = _segment()
        with self.assertRaises(ValueError):
            _run(RidgeRegression(choose_alpha=True), segment, l1)

    def test_modes_tuple_is_the_contract(self):
        self.assertEqual(set(LOO_MODES), {"nested", "crossfit", "auto"})


class TestLooModeBehaviour(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("PROF_LOO_MODE", "RIDGE_CF_SEED")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_nested_is_seed_invariant(self):
        """Nested has no A/B split, so RIDGE_CF_SEED must not move it."""
        segment, l1 = _segment()
        out = []
        for seed in ("7", "42", "123"):
            os.environ["RIDGE_CF_SEED"] = seed
            out.append(_preds(_run(RidgeRegression(choose_alpha=True, loo_mode="nested"), segment, l1)))
        np.testing.assert_allclose(out[0], out[1])
        np.testing.assert_allclose(out[0], out[2])

    def test_crossfit_remains_seed_sensitive(self):
        """Regression guard: the legacy path must keep its documented
        split-dependence, so that reproducing old numbers stays possible."""
        segment, l1 = _segment(n_l2=40, seed=3)
        os.environ["RIDGE_CF_SEED"] = "7"
        a = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="crossfit"), segment, l1))
        os.environ["RIDGE_CF_SEED"] = "123"
        b = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="crossfit"), segment, l1))
        self.assertFalse(np.allclose(a, b),
                         "crossfit became seed-invariant — the A/B split may have been lost")

    def test_auto_routes_by_segment_size(self):
        small, l1_small = _segment(n_l2=_AUTO_NESTED_BELOW_N // 4, seed=5)
        auto_small = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="auto"), small, l1_small))
        nested_small = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="nested"), small, l1_small))
        np.testing.assert_allclose(auto_small, nested_small)

        big, l1_big = _segment(n_l2=_AUTO_NESTED_BELOW_N * 2, seed=6)
        auto_big = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="auto"), big, l1_big))
        cf_big = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode="crossfit"), big, l1_big))
        np.testing.assert_allclose(auto_big, cf_big)

    def test_every_test_participant_gets_a_prediction(self):
        segment, l1 = _segment(n_l2=25, seed=9)
        for mode in ("nested", "crossfit"):
            res = _run(RidgeRegression(choose_alpha=True, loo_mode=mode), segment, l1)
            self.assertEqual(len(res), 25, f"{mode} dropped participants")
            self.assertEqual(len({r[Fields.SUBJECT_ID] for r in res}), 25)

    def test_predictions_within_score_bounds(self):
        segment, l1 = _segment(n_l2=25, seed=11)
        for mode in ("nested", "crossfit"):
            preds = _preds(_run(RidgeRegression(choose_alpha=True, loo_mode=mode), segment, l1))
            self.assertTrue(np.all(preds >= 0) and np.all(preds <= 100), mode)

    def test_log_ridge_inverts_target_in_nested(self):
        """LogRidge fits in log space; nested must invert before clipping, or
        predictions collapse to the low end of the score range."""
        segment, l1 = _segment(n_l2=25, seed=13)
        preds = _preds(_run(LogRidgeRegression(choose_alpha=True, loo_mode="nested"), segment, l1))
        self.assertGreater(preds.mean(), 10.0,
                           "LogRidge nested predictions look like un-inverted log values")

    def test_unseen_still_bypasses_both_modes(self):
        """UNSEEN must not take either LOO path — its prediction has to come
        from the other text context, not the participant's own row."""
        l2 = _frame(20, 21)
        l1 = _frame(8, 121, native=True)
        seg = PoolSegment(segment_id="s", train_pool=l2, test_pool=l2, train_l1=l1)
        for mode in ("nested", "crossfit", "auto"):
            m = RidgeRegression(choose_alpha=True, loo_mode=mode)
            self.assertIsNone(
                m.try_loocv_segment(seg, l1, fold_method=FoldMethod.POOL_ALL,
                                    pool=Pool.UNSEEN,
                                    feature_cols=[f"f{i}" for i in range(N_FEATURES)],
                                    target_col=TARGET, aug_precentage=1, narrow_l1=False),
                f"{mode} took the LOO path under UNSEEN")


if __name__ == "__main__":
    unittest.main()
