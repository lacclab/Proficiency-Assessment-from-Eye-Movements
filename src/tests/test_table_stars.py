"""Tests for the significance stars the bootstrap tables print.

Every p-value in pairwise_significance.csv is two-sided, so a feature set that
is significantly *worse* than WPM carries the same p as one that is
significantly better. `_directional_stars` therefore returns a SIGNED level:
the magnitude is the significance level (0-3) and the sign is the direction --
positive when the difference runs the favourable way (higher r, lower MAE),
negative when it runs against, zero when non-significant or directionless.

The renderer marks both directions with stars and lets the printed value say
which way the cell runs; the captions state this explicitly. An earlier design
suppressed the wrong-direction cells entirely, which is why these tests once
asserted 0 where they now assert a negative level.

Both table generators duplicate the rule (they already duplicate
`_p_to_n_stars`), so both are tested here, including the direction flip: for
correlations higher is better, for MAE lower is better.
"""
import unittest

import numpy as np
import pandas as pd

from src.evaluation.EyeScore.tables import (
    _directional_stars as eyescore_stars,
    _p_to_n_stars as eyescore_p_to_n,
)
from src.evaluation.predictions.tables import (
    _directional_stars as prediction_stars,
    _p_to_n_stars as prediction_p_to_n,
)

SIGNIFICANT = 0.0001  # would be *** on its own


class TestCorrelationStars(unittest.TestCase):
    """Higher correlation is better, so a higher r takes the positive sign."""

    def test_better_feature_set_keeps_its_stars(self):
        self.assertEqual(
            prediction_stars(SIGNIFICANT, 0.60, 0.45, higher_is_better=True), 3)
        self.assertEqual(eyescore_stars(SIGNIFICANT, 0.60, 0.45), 3)

    def test_worse_feature_set_takes_the_negative_sign(self):
        # Significant, but in the wrong direction: same level, opposite sign.
        self.assertEqual(
            prediction_stars(SIGNIFICANT, 0.30, 0.45, higher_is_better=True), -3)
        self.assertEqual(eyescore_stars(SIGNIFICANT, 0.30, 0.45), -3)

    def test_star_count_still_tracks_the_p_value_when_direction_is_right(self):
        for p, expected in [(0.0005, 3), (0.005, 2), (0.03, 1), (0.20, 0)]:
            with self.subTest(p=p):
                self.assertEqual(
                    prediction_stars(p, 0.60, 0.45, higher_is_better=True),
                    expected)
                self.assertEqual(eyescore_stars(p, 0.60, 0.45), expected)

    def test_a_tie_is_not_a_win(self):
        self.assertEqual(
            prediction_stars(SIGNIFICANT, 0.45, 0.45, higher_is_better=True), 0)
        self.assertEqual(eyescore_stars(SIGNIFICANT, 0.45, 0.45), 0)

    def test_negative_correlations_compare_by_value_not_magnitude(self):
        # -0.6 is worse than -0.2 for a proficiency predictor, despite the
        # larger magnitude, so it takes the negative sign.
        self.assertEqual(
            prediction_stars(SIGNIFICANT, -0.60, -0.20, higher_is_better=True), -3)
        self.assertEqual(
            prediction_stars(SIGNIFICANT, -0.20, -0.60, higher_is_better=True), 3)


class TestMaeStars(unittest.TestCase):
    """Lower MAE is better, so the sign convention inverts: the smaller error
    takes the positive sign."""

    def test_lower_error_keeps_its_stars(self):
        self.assertEqual(
            prediction_stars(SIGNIFICANT, 6.0, 8.0, higher_is_better=False), 3)

    def test_higher_error_takes_the_negative_sign(self):
        self.assertEqual(
            prediction_stars(SIGNIFICANT, 9.0, 8.0, higher_is_better=False), -3)

    def test_worse_than_baseline_cells_carry_the_negative_sign(self):
        # Two real require_l1_lextale Ridge cells whose error exceeds the
        # baseline's: significant at p<0.01, so level 2 with the sign against.
        for mae_fs, mae_base in [(8.157, 6.646), (11.559, 9.448)]:
            with self.subTest(mae_fs=mae_fs):
                self.assertEqual(
                    prediction_stars(0.005, mae_fs, mae_base,
                                     higher_is_better=False), -2)


class TestMissingValues(unittest.TestCase):
    """A star must never be emitted on incomplete information."""

    def test_missing_p_value(self):
        for p in (None, np.nan, pd.NA):
            with self.subTest(p=p):
                self.assertEqual(
                    prediction_stars(p, 0.60, 0.45, higher_is_better=True), 0)
                self.assertEqual(eyescore_stars(p, 0.60, 0.45), 0)

    def test_undeterminable_direction(self):
        # If either correlation is missing the direction is unknown, so the
        # conservative choice is no star rather than a possibly-wrong one.
        self.assertEqual(
            prediction_stars(SIGNIFICANT, np.nan, 0.45, higher_is_better=True), 0)
        self.assertEqual(
            prediction_stars(SIGNIFICANT, 0.60, np.nan, higher_is_better=True), 0)
        self.assertEqual(eyescore_stars(SIGNIFICANT, np.nan, 0.45), 0)
        self.assertEqual(eyescore_stars(SIGNIFICANT, 0.60, np.nan), 0)


class TestUnderlyingThresholds(unittest.TestCase):
    """`_p_to_n_stars` itself is unchanged; both copies must stay in step."""

    def test_both_generators_agree(self):
        for p in (0.0005, 0.005, 0.03, 0.20, 0.001, 0.01, 0.05):
            with self.subTest(p=p):
                self.assertEqual(prediction_p_to_n(p), eyescore_p_to_n(p))

    def test_boundaries_fall_into_the_weaker_bucket(self):
        self.assertEqual(prediction_p_to_n(0.001), 2)
        self.assertEqual(prediction_p_to_n(0.01), 1)
        self.assertEqual(prediction_p_to_n(0.05), 0)


if __name__ == "__main__":
    unittest.main()
