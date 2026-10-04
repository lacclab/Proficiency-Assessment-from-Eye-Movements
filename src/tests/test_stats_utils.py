"""Tests for the shared statistical machinery in src/evaluation/stats_utils.py.

Every significance star in the paper's bootstrap tables comes from formulas
hand-transcribed in that module (Steiger 1980 and Meng-Rosenthal-Rubin 1992),
so a silent transcription error in a variance term would move published claims
without breaking anything visibly. This file guards them.

On the absence of cocor reference values: the ideal test pins our Steiger z
against R's ``cocor::cocor.dep.groups.overlap(..., test="steiger1980")``. No R
interpreter is installed in this environment, so that pinning is deferred. In
its place, `TestNullCalibration` validates the formulas
end-to-end by Monte Carlo: it simulates trivariate-normal data in which the two
correlations are equal *in the population*, runs the test thousands of times,
and checks that the false-positive rate matches the nominal alpha. A wrong
variance term or a dropped (n-3) factor changes the rejection rate sharply, so
this catches the failure mode the cocor pin would catch. `TestCharacterization`
separately freezes current outputs so a refactor cannot shift them unnoticed —
that is regression protection, not external validation.

Covered:
  • Steiger/Meng: null calibration, power, antisymmetry, monotonicity in n and
    in the correlation gap, and every degenerate-input guard branch;
  • bootstrap CI + paired-difference test: seed determinism, pairwise NaN
    dropping, the 1/n_bootstrap p floor, and the MAE sign convention
    (negative favors model_a) that the table layer depends on;
  • p_to_stars threshold boundaries.
"""
import unittest

import numpy as np
from scipy.stats import pearsonr

from src.evaluation.stats_utils import (
    BOOTSTRAP_SEED,
    bootstrap_correlation_ci,
    bootstrap_difference_test,
    meng_dependent_overlap,
    p_to_stars,
    pairwise_dependent_tests,
    steiger_dependent_overlap,
)


def _simulate_p_values(rho_a, rho_b, r_ab, n, n_sims, seed=7):
    """Sample p-values from both tests on data with a known correlation structure.

    Draws `n_sims` datasets of size `n` from a trivariate normal whose
    population correlations are corr(x_a, y)=rho_a, corr(x_b, y)=rho_b and
    corr(x_a, x_b)=r_ab, then runs Steiger and Meng on each sample. With
    rho_a == rho_b the null hypothesis is true, so the returned p-values should
    be uniform; with rho_a != rho_b they measure power.
    """
    corr = np.array([
        [1.0, rho_a, rho_b],
        [rho_a, 1.0, r_ab],
        [rho_b, r_ab, 1.0],
    ])
    chol = np.linalg.cholesky(corr)  # raises if the structure isn't valid
    rng = np.random.default_rng(seed)

    steiger_p, meng_p = [], []
    for _ in range(n_sims):
        y, x_a, x_b = chol @ rng.standard_normal((3, n))
        r_a = pearsonr(x_a, y)[0]
        r_b = pearsonr(x_b, y)[0]
        r_obs_ab = pearsonr(x_a, x_b)[0]
        steiger_p.append(steiger_dependent_overlap(r_a, r_b, r_obs_ab, n)[1])
        meng_p.append(meng_dependent_overlap(r_a, r_b, r_obs_ab, n)[1])
    return np.array(steiger_p), np.array(meng_p)


class TestNullCalibration(unittest.TestCase):
    """Under a true null, the rejection rate must match the nominal alpha.

    This is the stand-in for a cocor pin: it exercises the whole formula
    (Fisher z transform, the psi covariance term, the sqrt(n-3) scaling) against
    a ground truth that does not depend on our implementation.
    """

    # 3000 sims -> SE of a 0.05 rate is ~0.004, so a +-0.015 band is ~4 SE:
    # wide enough not to flake, far tighter than any real formula error.
    N_SIMS = 3000
    TOLERANCE = 0.015

    def test_type_one_error_at_nominal_alpha(self):
        # Three correlation structures, including the r_ab=0.8 case where the
        # dependence between the two models is strong and the covariance term
        # matters most.
        for rho, r_ab in [(0.5, 0.5), (0.3, 0.2), (0.6, 0.8)]:
            steiger_p, meng_p = _simulate_p_values(
                rho, rho, r_ab, n=100, n_sims=self.N_SIMS)
            for name, p_vals in (("steiger", steiger_p), ("meng", meng_p)):
                with self.subTest(rho=rho, r_ab=r_ab, test=name):
                    rate = float(np.mean(p_vals < 0.05))
                    self.assertAlmostEqual(rate, 0.05, delta=self.TOLERANCE,
                                           msg=f"{name} rejects at {rate:.4f}, "
                                               f"expected ~0.05")

    def test_type_one_error_at_one_percent(self):
        # A variance-term error can leave alpha=0.05 roughly intact while
        # distorting the tail, so check a second level.
        steiger_p, meng_p = _simulate_p_values(
            0.5, 0.5, 0.5, n=100, n_sims=self.N_SIMS)
        for name, p_vals in (("steiger", steiger_p), ("meng", meng_p)):
            with self.subTest(test=name):
                rate = float(np.mean(p_vals < 0.01))
                self.assertAlmostEqual(rate, 0.01, delta=0.008,
                                       msg=f"{name} rejects at {rate:.4f}, "
                                           f"expected ~0.01")

    def test_type_one_error_at_small_n(self):
        # The sqrt(n-3) correction is nearly invisible at n=100 (it shifts z by
        # ~1.5%), so a mutant using sqrt(n) passes the tests above. At n=10 the
        # same mutant inflates the false-positive rate to ~0.10, which this
        # catches. Keep a small-n case here for that reason.
        steiger_p, meng_p = _simulate_p_values(
            0.5, 0.5, 0.5, n=10, n_sims=4000)
        for name, p_vals in (("steiger", steiger_p), ("meng", meng_p)):
            with self.subTest(test=name):
                rate = float(np.mean(p_vals < 0.05))
                self.assertAlmostEqual(rate, 0.05, delta=0.02,
                                       msg=f"{name} rejects at {rate:.4f} for "
                                           f"n=10, expected ~0.05")

    def test_has_power_against_a_real_difference(self):
        # Calibration alone is satisfied by a test that never rejects, so
        # confirm a genuine gap is detected most of the time.
        steiger_p, meng_p = _simulate_p_values(
            0.7, 0.3, 0.5, n=100, n_sims=500)
        for name, p_vals in (("steiger", steiger_p), ("meng", meng_p)):
            with self.subTest(test=name):
                self.assertGreater(float(np.mean(p_vals < 0.05)), 0.9)


class TestDependentTestProperties(unittest.TestCase):
    """Algebraic properties both tests must satisfy by construction."""

    def test_equal_correlations_give_zero_z(self):
        for test in (steiger_dependent_overlap, meng_dependent_overlap):
            with self.subTest(test=test.__name__):
                z, p = test(0.5, 0.5, 0.4, 100)
                self.assertEqual(z, 0.0)
                self.assertEqual(p, 1.0)

    def test_swapping_models_flips_z_and_keeps_p(self):
        for test in (steiger_dependent_overlap, meng_dependent_overlap):
            with self.subTest(test=test.__name__):
                z_fwd, p_fwd = test(0.6, 0.4, 0.5, 100)
                z_rev, p_rev = test(0.4, 0.6, 0.5, 100)
                self.assertAlmostEqual(z_fwd, -z_rev, places=12)
                self.assertAlmostEqual(p_fwd, p_rev, places=12)

    def test_z_grows_with_sample_size(self):
        for test in (steiger_dependent_overlap, meng_dependent_overlap):
            with self.subTest(test=test.__name__):
                z_by_n = [test(0.6, 0.4, 0.5, n)[0] for n in (50, 100, 200, 400)]
                self.assertEqual(z_by_n, sorted(z_by_n))

    def test_z_grows_with_the_correlation_gap(self):
        for test in (steiger_dependent_overlap, meng_dependent_overlap):
            with self.subTest(test=test.__name__):
                z_by_gap = [test(0.5 + g, 0.5 - g, 0.5, 100)[0]
                            for g in (0.02, 0.05, 0.10, 0.20)]
                self.assertEqual(z_by_gap, sorted(z_by_gap))

    def test_steiger_and_meng_broadly_agree(self):
        # They are different estimators of the same quantity; they should not
        # land on opposite sides of a decision for ordinary inputs.
        for r_a, r_b, r_ab, n in [(0.6, 0.4, 0.5, 100), (0.55, 0.45, 0.3, 64),
                                  (0.7, 0.2, 0.8, 80), (0.3, 0.25, 0.1, 200)]:
            with self.subTest(r_a=r_a, r_b=r_b, r_ab=r_ab, n=n):
                _, p_steiger = steiger_dependent_overlap(r_a, r_b, r_ab, n)
                _, p_meng = meng_dependent_overlap(r_a, r_b, r_ab, n)
                self.assertAlmostEqual(p_steiger, p_meng, delta=0.05)


class TestDegenerateInputs(unittest.TestCase):
    """Every guard branch must return (nan, nan) rather than raise or emit a
    number — these feed straight into table cells."""

    BOTH = (steiger_dependent_overlap, meng_dependent_overlap)

    def _assert_nan_pair(self, result):
        z, p = result
        self.assertTrue(np.isnan(z))
        self.assertTrue(np.isnan(p))

    def test_sample_too_small(self):
        for test in self.BOTH:
            for n in (0, 1, 3):
                with self.subTest(test=test.__name__, n=n):
                    self._assert_nan_pair(test(0.5, 0.4, 0.3, n))

    def test_non_finite_correlations(self):
        for test in self.BOTH:
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(test=test.__name__, bad=bad):
                    self._assert_nan_pair(test(bad, 0.4, 0.3, 50))
                    self._assert_nan_pair(test(0.5, bad, 0.3, 50))
                    self._assert_nan_pair(test(0.5, 0.4, bad, 50))

    def test_correlations_at_or_beyond_unity(self):
        for test in self.BOTH:
            for bad in (1.0, -1.0, 1.5):
                with self.subTest(test=test.__name__, bad=bad):
                    self._assert_nan_pair(test(bad, 0.4, 0.3, 50))
                    self._assert_nan_pair(test(0.5, bad, 0.3, 50))
                    self._assert_nan_pair(test(0.5, 0.4, bad, 50))

    def test_meng_rejects_perfectly_correlated_models(self):
        # Meng divides by (1 - r_kh); r_kh just under 1 must stay finite and
        # r_kh == 1 must be caught by the |r| >= 1 guard.
        self._assert_nan_pair(meng_dependent_overlap(0.5, 0.4, 1.0, 50))
        z, _ = meng_dependent_overlap(0.5, 0.4, 0.999, 50)
        self.assertTrue(np.isfinite(z))

    def test_n_is_exactly_four(self):
        # The n < 4 boundary: n == 4 is allowed through and must be finite.
        for test in self.BOTH:
            with self.subTest(test=test.__name__):
                z, p = test(0.6, 0.4, 0.5, 4)
                self.assertTrue(np.isfinite(z))
                self.assertTrue(0.0 <= p <= 1.0)


class TestBootstrapCorrelationCI(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 60
        self.y = rng.standard_normal(self.n)
        self.x = self.y * 0.8 + rng.standard_normal(self.n) * 0.6

    def test_is_deterministic_under_the_fixed_seed(self):
        first = bootstrap_correlation_ci(self.y, self.x, n_bootstrap=500)
        second = bootstrap_correlation_ci(self.y, self.x, n_bootstrap=500)
        self.assertEqual(first, second)

    def test_recovers_the_sample_correlation(self):
        out = bootstrap_correlation_ci(self.y, self.x, n_bootstrap=2000)
        sample_r = pearsonr(self.y, self.x)[0]
        self.assertAlmostEqual(out["pearson_r_boot"], sample_r, delta=0.02)
        self.assertLess(out["pearson_ci_low_boot"], sample_r)
        self.assertGreater(out["pearson_ci_high_boot"], sample_r)

    def test_ci_is_the_percentile_interval_not_the_normal_approximation(self):
        # The whole module reports percentile intervals; the normal form
        # (mean +- 1.96*std) ignores that a correlation is bounded and its
        # bootstrap distribution skewed. Recompute the percentiles from the
        # same seeded resamples and pin to those.
        n_boot = 500
        out = bootstrap_correlation_ci(self.y, self.x, n_bootstrap=n_boot)

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = self.y.size
        idx = rng.integers(0, n, size=(n_boot, n))
        a, b = self.y[idx], self.x[idx]
        ac = a - a.mean(1, keepdims=True)
        bc = b - b.mean(1, keepdims=True)
        r = (ac * bc).sum(1) / np.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1))
        lo, hi = np.percentile(r[np.isfinite(r)], [2.5, 97.5])

        self.assertAlmostEqual(out["pearson_ci_low_boot"], float(lo), places=12)
        self.assertAlmostEqual(out["pearson_ci_high_boot"], float(hi), places=12)

        # And explicitly not the normal approximation.
        half_width = 1.959963984540054 * out["pearson_r_boot_std"]
        self.assertNotAlmostEqual(out["pearson_ci_low_boot"],
                                  out["pearson_r_boot"] - half_width, places=6)

    def test_ci_brackets_the_point_estimate(self):
        out = bootstrap_correlation_ci(self.y, self.x, n_bootstrap=2000)
        self.assertLess(out["pearson_ci_low_boot"], out["pearson_r_boot"])
        self.assertGreater(out["pearson_ci_high_boot"], out["pearson_r_boot"])

    def test_ci_stays_inside_the_valid_correlation_range(self):
        # A percentile interval cannot leave [-1, 1]; the normal form can.
        rng = np.random.default_rng(3)
        n = 25
        x = rng.standard_normal(n)
        y = x * 0.99 + rng.standard_normal(n) * 0.02  # r very close to 1
        out = bootstrap_correlation_ci(x, y, n_bootstrap=2000)
        self.assertLessEqual(out["pearson_ci_high_boot"], 1.0)
        self.assertGreaterEqual(out["pearson_ci_low_boot"], -1.0)

    def test_drops_non_finite_pairs_before_resampling(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        out = bootstrap_correlation_ci(y, x, n_bootstrap=100)
        self.assertEqual(out["n_boot_obs"], 3)  # rows 2 and 3 dropped pairwise

    def test_too_few_observations_yield_nan(self):
        out = bootstrap_correlation_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertEqual(out["n_boot"], 0)
        for key in ("pearson_r_boot", "spearman_r_boot",
                    "pearson_ci_low_boot", "pearson_ci_high_boot"):
            self.assertTrue(np.isnan(out[key]), msg=key)

    def test_mae_columns_appear_only_when_requested(self):
        without = bootstrap_correlation_ci(self.y, self.x, n_bootstrap=200)
        self.assertNotIn("mae_boot", without)
        with_mae = bootstrap_correlation_ci(self.y, self.x, n_bootstrap=200,
                                            include_mae=True)
        self.assertAlmostEqual(with_mae["mae_boot"],
                               float(np.mean(np.abs(self.y - self.x))),
                               delta=0.05)


class TestBootstrapDifferenceTest(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        n = 60
        self.y = rng.standard_normal(n)
        # x_a is the stronger predictor of y, on the same scale as y.
        self.x_a = self.y * 0.8 + rng.standard_normal(n) * 0.6
        self.x_b = self.y * 0.3 + rng.standard_normal(n) * 0.95

    def test_favors_the_stronger_model(self):
        out = bootstrap_difference_test(self.y, self.x_a, self.x_b,
                                        n_bootstrap=2000)
        self.assertGreater(out["boot_diff_pearson"], 0)  # positive favors a
        self.assertLess(out["boot_diff_pearson_p"], 0.05)

    def test_mae_difference_is_negative_for_the_better_model(self):
        # The table layer relies on this inverted sign: lower MAE is better, so
        # a negative boot_diff_mae favors model_a. Flipping it would silently
        # star the wrong rows.
        out = bootstrap_difference_test(self.y, self.x_a, self.x_b,
                                        n_bootstrap=2000, include_mae=True)
        self.assertLess(out["boot_diff_mae"], 0)
        self.assertGreater(out["boot_diff_pearson"], 0)

    def test_p_is_null_centred_not_the_percentile_form(self):
        # The percentile form 2*min(P(d<=0), P(d>=0)) inverts a CI instead of
        # testing a null, and over-rejects. Assert we are not computing it: on
        # a clearly separated pair the two disagree, and the implementation
        # must match the null-centred ASL.
        out = bootstrap_difference_test(self.y, self.x_a, self.x_b,
                                        n_bootstrap=4000)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = self.y.size
        idx = rng.integers(0, n, size=(4000, n))

        def _r(a, b):
            ac = a - a.mean(1, keepdims=True)
            bc = b - b.mean(1, keepdims=True)
            return (ac * bc).sum(1) / np.sqrt((ac ** 2).sum(1) * (bc ** 2).sum(1))

        d = _r(self.x_a[idx], self.y[idx]) - _r(self.x_b[idx], self.y[idx])
        d_obs = pearsonr(self.x_a, self.y)[0] - pearsonr(self.x_b, self.y)[0]
        expected = max(float(np.mean(np.abs(d - d_obs) >= abs(d_obs))), 1.0 / d.size)
        self.assertAlmostEqual(out["boot_diff_pearson_p"], expected, places=12)

    def test_p_is_floored_at_the_resample_resolution(self):
        # The two-sided p cannot be more extreme than 1/n_bootstrap; the
        # markdown caption in tex_tables_to_markdown quotes this floor.
        for n_boot in (500, 2000):
            with self.subTest(n_bootstrap=n_boot):
                out = bootstrap_difference_test(self.y, self.x_a, self.x_b,
                                                n_bootstrap=n_boot)
                self.assertGreaterEqual(out["boot_diff_pearson_p"], 1.0 / n_boot)

    def test_identical_models_are_not_significant(self):
        out = bootstrap_difference_test(self.y, self.x_a, self.x_a.copy(),
                                        n_bootstrap=1000)
        self.assertAlmostEqual(out["boot_diff_pearson"], 0.0, places=12)
        self.assertGreater(out["boot_diff_pearson_p"], 0.05)

    def test_ci_brackets_the_mean_difference(self):
        out = bootstrap_difference_test(self.y, self.x_a, self.x_b,
                                        n_bootstrap=2000)
        self.assertLess(out["boot_diff_pearson_ci_low"], out["boot_diff_pearson"])
        self.assertGreater(out["boot_diff_pearson_ci_high"], out["boot_diff_pearson"])

    def test_too_few_observations_yield_nan(self):
        out = bootstrap_difference_test(np.array([1.0, 2.0, 3.0]),
                                        np.array([1.0, 2.0, 3.0]),
                                        np.array([3.0, 2.0, 1.0]))
        self.assertEqual(out["n_boot_diff"], 0)
        self.assertTrue(np.isnan(out["boot_diff_pearson"]))
        self.assertIsNone(out["boot_diff_pearson_sig"])


class TestPairwiseBundle(unittest.TestCase):
    """The dict `pairwise_dependent_tests` returns is written straight to
    pairwise_significance.csv, so its keys are a contract with the table layer."""

    def setUp(self):
        rng = np.random.default_rng(1)
        n = 80
        self.y = rng.standard_normal(n)
        self.x_a = self.y * 0.8 + rng.standard_normal(n) * 0.6
        self.x_b = self.y * 0.3 + rng.standard_normal(n) * 0.95

    def test_emits_every_column_the_tables_read(self):
        out = pairwise_dependent_tests(self.y, self.x_a, self.x_b,
                                       n_bootstrap=500, include_mae=True)
        # tables.py BOOTSTRAP_STARS_SOURCES formats these two templates with
        # {m} in {pearson, spearman}, plus the MAE difference column.
        for metric in ("pearson", "spearman"):
            self.assertIn(f"steiger_p_{metric}", out)
            self.assertIn(f"boot_diff_{metric}_p", out)
        self.assertIn("boot_diff_mae_p", out)
        self.assertIn("n", out)

    def test_reported_correlations_match_scipy(self):
        out = pairwise_dependent_tests(self.y, self.x_a, self.x_b,
                                       include_bootstrap_diff=False)
        self.assertAlmostEqual(out["pearson_r_a"], pearsonr(self.x_a, self.y)[0],
                               places=12)
        self.assertAlmostEqual(out["pearson_r_b"], pearsonr(self.x_b, self.y)[0],
                               places=12)

    def test_drops_rows_missing_in_any_of_the_three_vectors(self):
        y = np.append(self.y, np.nan)
        x_a = np.append(self.x_a, 1.0)
        x_b = np.append(self.x_b, 1.0)
        out = pairwise_dependent_tests(y, x_a, x_b, include_bootstrap_diff=False)
        self.assertEqual(out["n"], len(self.y))

    def test_mae_columns_gated_on_include_mae(self):
        without = pairwise_dependent_tests(self.y, self.x_a, self.x_b,
                                           include_bootstrap_diff=False)
        self.assertNotIn("mae_a", without)
        with_mae = pairwise_dependent_tests(self.y, self.x_a, self.x_b,
                                            include_bootstrap_diff=False,
                                            include_mae=True)
        self.assertAlmostEqual(with_mae["mae_a"],
                               float(np.mean(np.abs(self.x_a - self.y))),
                               places=12)


class TestPToStars(unittest.TestCase):

    def test_thresholds(self):
        self.assertEqual(p_to_stars(0.0005), "***")
        self.assertEqual(p_to_stars(0.005), "**")
        self.assertEqual(p_to_stars(0.03), "*")
        self.assertEqual(p_to_stars(0.5), "ns")

    def test_boundaries_are_strict_inequalities(self):
        # Exactly at a threshold falls into the weaker bucket.
        self.assertEqual(p_to_stars(0.001), "**")
        self.assertEqual(p_to_stars(0.01), "*")
        self.assertEqual(p_to_stars(0.05), "ns")

    def test_missing_p_gives_no_star(self):
        self.assertIsNone(p_to_stars(None))
        self.assertIsNone(p_to_stars(np.nan))


class TestCharacterization(unittest.TestCase):
    """Frozen outputs of the current implementation.

    NOT external validation — these values were produced by the code under test.
    They exist so a refactor that changes a published number fails loudly. If a
    cocor pin is added later it should replace these, not sit alongside them.
    """

    def test_steiger_reference_points(self):
        for (r_a, r_b, r_ab, n), expected_z in [
            ((0.6, 0.4, 0.5, 100), 2.413132567334629),
            ((0.55, 0.45, 0.3, 64), 0.8137363282206055),
        ]:
            with self.subTest(r_a=r_a, r_b=r_b, r_ab=r_ab, n=n):
                z, _ = steiger_dependent_overlap(r_a, r_b, r_ab, n)
                self.assertAlmostEqual(z, expected_z, places=12)

    def test_meng_reference_points(self):
        for (r_a, r_b, r_ab, n), expected_z in [
            ((0.6, 0.4, 0.5, 100), 2.390681112663852),
            ((0.55, 0.45, 0.3, 64), 0.8124525456179834),
        ]:
            with self.subTest(r_a=r_a, r_b=r_b, r_ab=r_ab, n=n):
                z, _ = meng_dependent_overlap(r_a, r_b, r_ab, n)
                self.assertAlmostEqual(z, expected_z, places=12)


if __name__ == "__main__":
    unittest.main()
