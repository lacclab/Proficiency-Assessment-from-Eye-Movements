"""Regression tests guarding the eyescore performance refactors.

These lock in numerical equivalence for the optimizations in
src/methods/EyeScore/calculation.py:

  #3  eye_score_batch          == row-by-row eye_score_one_participant
  #4  prepare-once + slice      == per-fold _fit_typology_alpha
  #2  _loo_fit_stats            == brute-force leave-one-out pandas mean/std
  #2  EYESCORE_FAST_LOO=1 path  == the exact reference LOO path (end to end)

The #2 fast path is opt-in (EYESCORE_FAST_LOO); this test is what lets a run
turn it on with confidence. The edge cases below are the ones that could break
the *kept-column set* rather than just perturb a value at float level:
constant columns and columns that become constant only after one participant
is removed.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.constants import Fields
from src.methods.EyeScore.calculation import (
    eye_score,
    eye_score_batch,
    eye_score_one_participant,
    _loo_fit_stats,
    _prepare_typology_frame,
    _fit_typology_alpha,
    _fit_typology_alpha_prepared,
)

SID = Fields.SUBJECT_ID
L1 = Fields.L1
RTOL = 1e-9


def _make_feature_frame(n_participants, n_feats, seed, langs):
    """One row per participant, continuous features with scattered NaN cells —
    the realistic regime (z-scored eye-movement features are continuous). We
    deliberately do NOT inject literally-constant columns here: pandas .std()
    of a constant column is ~1e-15 (not 0), so the reference `std > 0` guard
    keeps it and z-scores 0/1e-15 into garbage, whereas the stable Welford
    downdate in _loo_fit_stats yields exactly 0 and drops it. That (benign,
    more-stable) divergence is covered on its own in
    TestConstantColumnStability rather than mixed into the equivalence checks.
    """
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_feats)]
    X = rng.normal(size=(n_participants, n_feats))
    mask = rng.random((n_participants, n_feats)) < 0.1
    X[mask] = np.nan
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, SID, [f"p{seed}_{i}" for i in range(n_participants)])
    df.insert(1, L1, [langs[i % len(langs)] for i in range(n_participants)])
    return df


class TestCosineBatch(unittest.TestCase):
    def test_batch_matches_rowwise(self):
        rng = np.random.default_rng(0)
        M = rng.normal(size=(50, 12))
        M[rng.random((50, 12)) < 0.15] = np.nan
        proto = rng.normal(size=12)
        proto[3] = np.nan
        expected = np.array([
            eye_score_one_participant(pd.Series(M[i]), pd.Series(proto))
            for i in range(M.shape[0])
        ])
        got = eye_score_batch(M, proto)
        # NaN pattern identical
        np.testing.assert_array_equal(np.isnan(expected), np.isnan(got))
        m = ~np.isnan(expected)
        np.testing.assert_allclose(got[m], expected[m], rtol=RTOL, atol=1e-12)

    def test_row_with_no_overlap_is_nan(self):
        proto = np.array([np.nan, 1.0, 2.0])
        row = np.array([5.0, np.nan, np.nan])  # no shared non-NaN cell
        self.assertTrue(np.isnan(eye_score_batch(row, proto)[0]))


class TestLooFitStats(unittest.TestCase):
    def test_matches_bruteforce_including_keep_set(self):
        df = _make_feature_frame(40, 6, seed=1, langs=["a", "b", "c"])
        fit_x = df.drop(columns=[SID, L1])
        cols, stats, _full = _loo_fit_stats(fit_x, df[SID])
        self.assertEqual(cols, list(fit_x.columns))
        for pid in df[SID].unique():
            subset = fit_x[df[SID].values != pid]
            exp_mean = subset.mean()          # pandas skipna
            exp_std = subset.std()            # ddof=1, skipna
            exp_keep = set(exp_std.index[exp_std > 0])
            mean_arr, std_arr = stats[pid]
            got_keep = {c for c, s in zip(cols, std_arr) if s > 0}
            # The kept-column set must match EXACTLY (this is what would change
            # results rather than just perturb them).
            self.assertEqual(got_keep, exp_keep, f"keep-set mismatch for {pid}")
            for c in exp_keep:
                j = cols.index(c)
                self.assertAlmostEqual(mean_arr[j], exp_mean[c], places=9)
                np.testing.assert_allclose(std_arr[j], exp_std[c], rtol=1e-9)


class TestConstantColumnStability(unittest.TestCase):
    """Document the ONE place the fast LOO intentionally differs from the
    reference: a truly-constant column. pandas .std() returns ~1e-15 there
    (float noise), so the reference keeps it and produces a garbage z-score;
    the Welford downdate yields exactly 0 and drops it. Real feature data has
    no such columns (the full-pipeline equivalence check passes at ~3.6e-10),
    but this pins the behavior so a future change can't regress it silently.
    """

    def test_constant_column_gets_exact_zero_std(self):
        df = _make_feature_frame(20, 3, seed=5, langs=["a", "b"])
        df["const"] = 2.71828
        fit_x = df.drop(columns=[SID, L1])
        cols, stats, _full = _loo_fit_stats(fit_x, df[SID])
        j = cols.index("const")
        for pid in df[SID].unique():
            _, std_arr = stats[pid]
            self.assertEqual(std_arr[j], 0.0)          # exactly 0 -> dropped


class TestTypologyPrepareSlice(unittest.TestCase):
    def test_prepare_once_slice_equals_perfold(self):
        rng = np.random.default_rng(2)
        langs = ["eng", "ita", "ger", "spa"]
        n = 60
        eye = pd.DataFrame({
            SID: [f"s{i}" for i in range(n)],
            "eye_score": rng.normal(size=n),
            L1: [langs[i % len(langs)] for i in range(n)],
        })
        target_df = pd.DataFrame({
            SID: eye[SID],
            "lextale_score": rng.normal(size=n),
        })
        distances = {"eng": 0.0, "ita": 0.3, "ger": 0.5, "spa": 0.8}
        prepared_full = _prepare_typology_frame(eye, target_df, distances, "lextale_score")
        # Leave-one-language-out: prepared-then-sliced must equal prepared-on-subset.
        for lang in langs:
            sub = eye[eye[L1] != lang]
            a_ref = _fit_typology_alpha(sub, target_df, distances, "lextale_score")
            a_fast = _fit_typology_alpha_prepared(
                prepared_full[prepared_full[L1] != lang], "lextale_score"
            )
            if a_ref is None:
                self.assertIsNone(a_fast)
            else:
                np.testing.assert_allclose(a_fast, a_ref, rtol=1e-12)


class TestFastLooEndToEnd(unittest.TestCase):
    def _run(self, fast: bool):
        langs = ["eng", "ita", "ger", "spa"]
        fit_l2 = _make_feature_frame(48, 5, seed=7, langs=langs)      # z-score fit + test pool
        pool_l1 = _make_feature_frame(30, 5, seed=99, langs=["eng"])   # prototype pool
        # align L1 feature columns to the fit's
        pool_l1 = pool_l1[fit_l2.columns]
        with tempfile.TemporaryDirectory() as d:
            save = Path(d) / "es.csv"
            env = {"EYESCORE_FAST_LOO": "1" if fast else "0"}
            # The fast path is guarded by BOTH the module master switch
            # (_ENABLE_FAST_LOO, parked False by default) and the env var, so
            # force the switch on here to actually exercise it when fast=True.
            with patch.dict(os.environ, env), \
                 patch("src.methods.EyeScore.calculation._ENABLE_FAST_LOO", fast):
                out = eye_score(
                    pool_l1, fit_l2, save,
                    content_mode="all", fit_on_df=fit_l2,
                )
        return out.sort_values(SID).reset_index(drop=True)

    def test_fast_loo_matches_reference(self):
        ref = self._run(fast=False)
        fast = self._run(fast=True)
        self.assertEqual(list(ref[SID]), list(fast[SID]))
        a = ref["eye_score"].to_numpy(float)
        b = fast["eye_score"].to_numpy(float)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        m = ~np.isnan(a)
        np.testing.assert_allclose(b[m], a[m], rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
