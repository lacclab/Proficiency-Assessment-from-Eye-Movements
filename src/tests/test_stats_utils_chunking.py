"""The blocked bootstrap must be bit-identical to the single-slab version.

`bootstrap_correlation_ci` and `bootstrap_difference_test` used to materialise
all `n_bootstrap` resamples at once, which costs ~2.4 MB per observation and
OOM-killed a full evaluation pass at large n. They now draw and reduce the same
resamples a block at a time. Every published CI, std and p-value comes out of
these two functions, so the change is only safe if it moves no number at all.

Two invariants pin that down:

  * the RNG stream — `Generator.integers` fills row-major, so blocked draws of
    (rows, n) must reconstruct exactly the single (n_bootstrap, n) draw;
  * block size must not matter — forcing a one-row block has to reproduce the
    result of a block large enough to hold every resample at once, which is the
    old code path.
"""
import unittest

import numpy as np

from src.evaluation import stats_utils
from src.evaluation.stats_utils import (
    bootstrap_correlation_ci,
    bootstrap_difference_test,
)


def _paired_data(n: int, seed: int = 0):
    """Target plus two correlated predictors, the shape these tests see."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    x_a = 0.6 * y + rng.normal(size=n)
    x_b = 0.3 * y + rng.normal(size=n)
    return y, x_a, x_b


def _assert_identical(case, expected: dict, actual: dict, label: str):
    case.assertEqual(set(expected), set(actual), f"{label}: different keys")
    for key, want in expected.items():
        got = actual[key]
        if isinstance(want, float) and np.isnan(want):
            case.assertTrue(np.isnan(got), f"{label}: {key} expected NaN, got {got!r}")
        else:
            # Exact equality on purpose: these are published numbers, so
            # "close enough" is not the bar.
            case.assertEqual(want, got, f"{label}: {key} moved")


class TestBlockedDraw(unittest.TestCase):
    """The RNG invariant the whole rewrite rests on."""

    def test_blocked_draw_matches_single_draw(self):
        for n in (7, 103, 1005):
            for block in (1, 13, 2000):
                one = np.random.default_rng(42).integers(0, n, size=(5000, n))
                rng = np.random.default_rng(42)
                blocked = np.vstack([
                    rng.integers(0, n, size=(stop - start, n))
                    for start, stop in stats_utils._blocks(5000, block)
                ])
                self.assertTrue(np.array_equal(one, blocked),
                                f"n={n} block={block}: resample indices differ")


class TestBlockSizeInvariance(unittest.TestCase):
    """Results must not depend on how the resamples are blocked."""

    def setUp(self):
        self._saved = stats_utils.BOOTSTRAP_BLOCK_BYTES

    def tearDown(self):
        stats_utils.BOOTSTRAP_BLOCK_BYTES = self._saved

    def _run(self, fn, args, kwargs, block_bytes):
        stats_utils.BOOTSTRAP_BLOCK_BYTES = block_bytes
        return fn(*args, **kwargs)

    def test_correlation_ci_invariant_to_block_size(self):
        for n in (17, 140, 1005):
            y, x_a, _ = _paired_data(n)
            for include_mae in (False, True):
                args, kwargs = (y, x_a), dict(n_bootstrap=2000,
                                              include_mae=include_mae)
                # 1 GB holds every resample at once — the pre-change path.
                whole = self._run(bootstrap_correlation_ci, args, kwargs, 1 << 30)
                # 1 byte forces the minimum block (one resample per pass).
                tiny = self._run(bootstrap_correlation_ci, args, kwargs, 1)
                _assert_identical(self, whole, tiny,
                                  f"corr_ci n={n} mae={include_mae}")

    def test_difference_test_invariant_to_block_size(self):
        for n in (17, 140, 1005):
            y, x_a, x_b = _paired_data(n)
            for include_mae in (False, True):
                args, kwargs = (y, x_a, x_b), dict(n_bootstrap=2000,
                                                   include_mae=include_mae)
                whole = self._run(bootstrap_difference_test, args, kwargs, 1 << 30)
                tiny = self._run(bootstrap_difference_test, args, kwargs, 1)
                _assert_identical(self, whole, tiny,
                                  f"diff n={n} mae={include_mae}")


class TestBlockSizing(unittest.TestCase):
    """Peak memory is what the blocking exists to bound."""

    def test_block_rows_shrink_as_n_grows(self):
        small = stats_utils._block_rows(100, 5)
        large = stats_utils._block_rows(16000, 5)
        self.assertGreater(small, large)
        self.assertGreaterEqual(large, 1)
        # A block never exceeds the budget it is sized against.
        for n in (10, 1005, 16000):
            rows = stats_utils._block_rows(n, 5)
            if rows > 1:
                self.assertLessEqual(rows * n * 8 * 5,
                                     stats_utils.BOOTSTRAP_BLOCK_BYTES)

    def test_block_rows_never_exceeds_total_resamples(self):
        self.assertLessEqual(stats_utils._block_rows(1, 1),
                             stats_utils.N_BOOTSTRAP)


if __name__ == "__main__":
    unittest.main()
