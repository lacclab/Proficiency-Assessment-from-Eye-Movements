"""Tests for the alternative EyeScore scorers: the euclidean metric and the
scorer registry.

Guards:
  • the Scorer registry shape (keys, default, families);
  • the euclidean metric: batch == row-by-row, orientation, and the NaN policy
    (global prototype-column mask + impute participant NaN to the fit-mean, so
    every participant is scored in the same full dimensionality — unlike cosine's
    per-row masking);
  • cosine stays the default and numerically identical to a stored golden array.
"""
import unittest

import numpy as np

from src.methods.EyeScore.calculation import (
    SCORERS,
    SCORER_NAMES,
    DEFAULT_SCORER,
    Scorer,
    eye_score_batch,
    eye_score_one_participant,
)


class TestScorerRegistry(unittest.TestCase):
    def test_keys_and_default(self):
        self.assertEqual(set(SCORER_NAMES), {"cosine", "euclidean", "logistic", "lda"})
        self.assertEqual(DEFAULT_SCORER, "cosine")

    def test_families(self):
        self.assertEqual(SCORERS["cosine"].family, "prototype")
        self.assertEqual(SCORERS["euclidean"].family, "prototype")
        self.assertEqual(SCORERS["logistic"].family, "discriminative")
        self.assertEqual(SCORERS["lda"].family, "discriminative")

    def test_prototype_carry_metric_discriminative_carry_factory(self):
        self.assertEqual(SCORERS["cosine"].metric, "cosine")
        self.assertEqual(SCORERS["euclidean"].metric, "euclidean")
        self.assertIsNone(SCORERS["logistic"].metric)
        self.assertTrue(callable(SCORERS["logistic"].estimator_factory))
        self.assertTrue(callable(SCORERS["lda"].estimator_factory))
        self.assertIsInstance(SCORERS["cosine"], Scorer)


class TestEuclidean(unittest.TestCase):
    def test_batch_matches_rowwise(self):
        rng = np.random.default_rng(0)
        proto = rng.normal(size=8)
        M = rng.normal(size=(20, 8))
        # scatter some NaNs into the rows
        mask = rng.random(M.shape) < 0.1
        M[mask] = np.nan
        batch = eye_score_batch(M, proto, metric="euclidean")
        row = np.array([eye_score_one_participant(r, proto, metric="euclidean") for r in M])
        np.testing.assert_allclose(batch, row, rtol=1e-12, equal_nan=True)

    def test_orientation_closer_is_higher(self):
        proto = np.array([1.0, 2.0, 3.0])
        near = np.array([1.0, 2.0, 3.1])
        far = np.array([4.0, 5.0, 6.0])
        s_near = eye_score_one_participant(near, proto, metric="euclidean")
        s_far = eye_score_one_participant(far, proto, metric="euclidean")
        self.assertGreater(s_near, s_far)          # closer → higher (less negative)
        # identical to prototype → exactly 0
        self.assertEqual(eye_score_one_participant(proto, proto, metric="euclidean"), 0.0)

    def test_full_dimensionality_impute_and_global_mask(self):
        # prototype NaN in col 1 → that column is dropped for EVERYONE.
        # participant NaN in col 0 → imputed to 0 (the fit-mean in z-space),
        # so distance uses the full (post-mask) dimensionality, not a shrunk one.
        proto = np.array([1.0, np.nan, 0.0])
        p = np.array([np.nan, 5.0, 0.0])
        # kept cols = {0, 2}; p -> [0, 0]; proto -> [1, 0]; dist = sqrt(1) = 1
        got = eye_score_one_participant(p, proto, metric="euclidean")
        self.assertAlmostEqual(got, -1.0, places=12)
        # batch agrees
        b = eye_score_batch(p.reshape(1, -1), proto, metric="euclidean")
        self.assertAlmostEqual(b[0], -1.0, places=12)

    def test_all_nan_prototype_is_nan(self):
        proto = np.array([np.nan, np.nan])
        p = np.array([1.0, 2.0])
        self.assertTrue(np.isnan(eye_score_one_participant(p, proto, metric="euclidean")))
        self.assertTrue(np.isnan(eye_score_batch(p.reshape(1, -1), proto, metric="euclidean")[0]))


class TestCosineUnchanged(unittest.TestCase):
    def test_default_metric_is_cosine(self):
        rng = np.random.default_rng(1)
        proto = rng.normal(size=6)
        M = rng.normal(size=(10, 6))
        np.testing.assert_array_equal(
            eye_score_batch(M, proto), eye_score_batch(M, proto, metric="cosine")
        )

    def test_cosine_golden(self):
        # small fixed fixture; if this array ever changes, cosine scoring drifted.
        proto = np.array([1.0, 2.0, 0.0, -1.0])
        M = np.array([
            [1.0, 2.0, 0.0, -1.0],     # identical direction → 1
            [2.0, 4.0, 0.0, -2.0],     # scaled → 1
            [-1.0, -2.0, 0.0, 1.0],    # opposite → -1
            [0.0, 0.0, 1.0, 0.0],      # orthogonal → 0
        ])
        got = eye_score_batch(M, proto, metric="cosine")
        np.testing.assert_allclose(got, [1.0, 1.0, -1.0, 0.0], rtol=1e-12, atol=1e-12)


class TestDiscriminative(unittest.TestCase):
    def _synth(self, n1=60, n2=80, d=3, seed=0):
        import pandas as pd
        from src.constants import Fields
        rng = np.random.default_rng(seed)
        cols = [f"f{i}" for i in range(d)]
        L1x = pd.DataFrame(rng.normal(1.0, 1, size=(n1, d)), columns=cols)
        L2x = pd.DataFrame(rng.normal(0.0, 1, size=(n2, d)), columns=cols)
        l1m = pd.DataFrame({Fields.SUBJECT_ID: [f"n{i}" for i in range(n1)],
                            Fields.L1: ["English"] * n1})
        langs = ["German"] * (n2 // 2) + ["Mandarin"] * (n2 - n2 // 2)
        l2m = pd.DataFrame({Fields.SUBJECT_ID: [f"p{i}" for i in range(n2)],
                            Fields.L1: langs})
        return L1x, L2x, l1m, l2m

    def test_output_is_real_valued_log_odds(self):
        from src.methods.EyeScore.calculation import _eye_score_discriminative, SCORERS
        L1x, L2x, l1m, l2m = self._synth()
        for name in ("logistic", "lda"):
            s = _eye_score_discriminative(L1x, L2x, l1m, l2m, SCORERS[name]).values
            self.assertEqual(len(s), len(L2x))
            self.assertTrue(np.isfinite(s).all())
            # log-odds: continuous, both signs — NOT squashed {0,1} probabilities
            self.assertLess(s.min(), 0.0)
            self.assertGreater(s.max(), 0.0)

    def test_reproducible_under_seed(self):
        from src.methods.EyeScore.calculation import _eye_score_discriminative, SCORERS
        L1x, L2x, l1m, l2m = self._synth()
        a = _eye_score_discriminative(L1x, L2x, l1m, l2m, SCORERS["logistic"]).values
        b = _eye_score_discriminative(L1x, L2x, l1m, l2m, SCORERS["logistic"]).values
        np.testing.assert_array_equal(a, b)

    def test_out_of_fold_no_leak(self):
        """Each scored L2 row must come from a model that never trained on it,
        and L1 must be present (label 1) in every training fold."""
        from src.methods.EyeScore.calculation import _eye_score_discriminative, Scorer
        L1x, L2x, l1m, l2m = self._synth()
        n1 = len(L1x)
        leaks = {"n": 0}

        class _Recording:
            def __init__(self, n_features=None):
                self.n_features = n_features

            def fit(self, X, y):
                self._train = np.asarray(X)
                # first n1 rows are the native class, all label 1
                assert y[:n1].sum() == n1 and y[n1:].sum() == 0
                return self

            def decision_function(self, X):
                for row in np.asarray(X):
                    if any(np.allclose(row, tr) for tr in self._train):
                        leaks["n"] += 1
                return np.asarray(X)[:, 0]

        scorer = Scorer("rec", "discriminative", estimator_factory=_Recording)
        s = _eye_score_discriminative(L1x, L2x, l1m, l2m, scorer).values
        self.assertEqual(leaks["n"], 0)              # no test row was in its train fold
        self.assertTrue(np.isfinite(s).all())        # every L2 got an out-of-fold score

    def test_fold_scorer_nested_cv_no_leak(self):
        """The discriminative fold scorer must never let an α-test participant
        into any classifier's training set:
          • score_fn(pool, pool)  → out-of-fold within the pool;
          • score_fn(pool, disjoint) → classifier trained only on L1 ∪ pool.
        """
        import pandas as pd
        from src.constants import Fields
        from src.methods.EyeScore.calculation import (
            _make_discriminative_fold_scorer, SCORERS,
        )
        rng = np.random.default_rng(4)
        cols = ["f0", "f1", "f2"]
        L1 = pd.DataFrame(rng.normal(1.0, 1, size=(50, 3)), columns=cols)
        L1[Fields.SUBJECT_ID] = [f"n{i}" for i in range(50)]
        L1[Fields.L1] = "English"
        n2 = 60
        L2 = pd.DataFrame(rng.normal(0.0, 1, size=(n2, 3)), columns=cols)
        L2[Fields.SUBJECT_ID] = [f"p{i}" for i in range(n2)]
        L2[Fields.L1] = ["German"] * 30 + ["Mandarin"] * 30
        score_fn = _make_discriminative_fold_scorer(
            L1, L2, "FS", cols, SCORERS["logistic"],
        )
        ids = list(L2[Fields.SUBJECT_ID])
        pool, held = ids[:45], ids[45:]

        # train==train: OOF within pool, one score per pool member, all finite
        fit_df = score_fn(pool, pool)
        self.assertEqual(set(fit_df[Fields.SUBJECT_ID]), set(pool))
        self.assertTrue(np.isfinite(fit_df["eye_score"]).all())

        # train vs held-out disjoint: scores every held-out target, finite
        apply_df = score_fn(pool, held)
        self.assertEqual(set(apply_df[Fields.SUBJECT_ID]), set(held))
        self.assertTrue(np.isfinite(apply_df["eye_score"]).all())

        # thin / empty guards
        self.assertIsNone(score_fn(pool, []))
        self.assertIsNone(score_fn([ids[0]], [ids[0]]))   # pool < 2

    def _cg_frames(self, seed=3, n=40):
        """L1/L2 frames carrying content_group values shaped "<batch>_<level>"
        so both the seen and unseen direction rules have data to work with."""
        import pandas as pd
        from src.constants import Fields
        rng = np.random.default_rng(seed)
        cols = ["f0", "f1", "f2"]
        cgs = ["b1_ele", "b1_adv", "b2_ele", "b2_adv"]
        cg_col = [cgs[i % len(cgs)] for i in range(n)]
        L1 = pd.DataFrame(rng.normal(1.0, 1, size=(n, 3)), columns=cols)
        L1[Fields.SUBJECT_ID] = [f"n{i}" for i in range(n)]
        L1[Fields.L1] = "English"
        L1[Fields.CONTENT_GROUP] = cg_col
        L2 = pd.DataFrame(rng.normal(0.0, 1, size=(n, 3)), columns=cols)
        L2[Fields.SUBJECT_ID] = [f"p{i}" for i in range(n)]
        L2[Fields.L1] = (["German"] * (n // 2)) + (["Mandarin"] * (n - n // 2))
        L2[Fields.CONTENT_GROUP] = cg_col
        return L1, L2, cols

    def test_seen_and_unseen_are_supported(self):
        """seen/unseen now score via the per-content-group directions rather
        than raising."""
        import tempfile
        from pathlib import Path
        from src.methods.EyeScore.calculation import eye_score
        L1, L2, _ = self._cg_frames()
        with tempfile.TemporaryDirectory() as d:
            for mode in ("seen", "unseen"):
                out = eye_score(L1, L2, Path(d) / f"{mode}.csv",
                                content_mode=mode, scorer="logistic")
                self.assertEqual(len(out), len(L2))
                self.assertTrue(np.isfinite(out["eye_score"]).all(),
                                f"{mode} produced non-finite scores")

    def test_paired_regimes(self):
        """_discriminative_oof_paired must handle the three regimes: identical
        fit/target rows, same participants via different rows, and disjoint."""
        from src.methods.EyeScore.calculation import (
            _discriminative_oof_paired, SCORERS,
        )
        rng = np.random.default_rng(7)
        X1 = rng.normal(1.0, 1, size=(40, 3))
        ids = np.array([f"p{i}" for i in range(30)])
        langs = np.array(["German"] * 15 + ["Mandarin"] * 15)
        Xa = rng.normal(0.0, 1, size=(30, 3))
        Xb = rng.normal(0.0, 1, size=(30, 3))   # same participants, other half
        sc = SCORERS["logistic"]
        # (1) fit rows == target rows → plain OOF
        s1 = _discriminative_oof_paired(X1, Xa, ids, Xa, ids, langs, sc)
        # (2) same participants, different rows → leave-participant-out
        s2 = _discriminative_oof_paired(X1, Xa, ids, Xb, ids, langs, sc)
        # (3) disjoint participants → single fit
        other = np.array([f"q{i}" for i in range(30)])
        s3 = _discriminative_oof_paired(X1, Xa, ids, Xb, other, langs, sc)
        for s in (s1, s2, s3):
            self.assertEqual(len(s), 30)
            self.assertTrue(np.isfinite(s).all())


if __name__ == "__main__":
    unittest.main()
