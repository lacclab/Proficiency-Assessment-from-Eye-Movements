"""Tests for the calculation-stage "is this cell finished?" check.

The runner's skip test used to look at the raw eye_score CSV alone, so a cell
whose raw scores existed was skipped whole. A Correction appended to
DEFAULT_CORRECTIONS after that cell was last computed therefore never appeared
unless the whole tree was rebuilt with --replace-existing, which also throws
away and recomputes the expensive raw scores. That gap was covered by hand, by
a script hardcoded to the four corrections that were new at the time.

`corrections_complete` closes it generically: a cell counts as done only when
every correction the dataset expects is on disk beside the raw file, so a newly
registered correction re-runs exactly the cells that lack it.

The subtlety these tests pin down is which corrections a dataset *expects*.
MichTest exists only for OneStop and ProfAgg only for MECO; those corrections
self-skip and write nothing on the other corpus, so counting them as expected
would leave every cell permanently "incomplete" and recompute the tree forever.
"""
import unittest

from src.methods.EyeScore.calculation import (
    DEBIAS_METHODS,
    DEFAULT_CORRECTIONS,
    corrections_complete,
    expected_correction_suffixes,
)

MICHTEST = "_typo_calibrated_inside_langs_michtest"
PROFAGG = "_typo_calibrated_inside_langs_profagg"


def _write_cell(tmp_path, dataset, suffixes, stem="READING_SPEED"):
    """A raw eye_score CSV plus one file per given correction suffix."""
    raw = tmp_path / f"{stem}.csv"
    raw.write_text("participant_id,eye_score\n1,0.5\n")
    for suffix in suffixes:
        raw.with_name(f"{stem}{suffix}.csv").write_text("participant_id,eye_score\n1,0.4\n")
    return raw


class TestExpectedSuffixes(unittest.TestCase):
    def test_michtest_is_expected_only_for_onestop(self):
        self.assertIn(MICHTEST, expected_correction_suffixes("OneStop"))
        self.assertNotIn(MICHTEST, expected_correction_suffixes("Meco"))

    def test_profagg_is_expected_only_for_meco(self):
        self.assertIn(PROFAGG, expected_correction_suffixes("Meco"))
        self.assertNotIn(PROFAGG, expected_correction_suffixes("OneStop"))

    def test_an_unknown_dataset_expects_everything(self):
        """Recomputing a cell is cheaper than silently accepting a missing variant."""
        self.assertEqual(len(expected_correction_suffixes("Wat")), len(DEFAULT_CORRECTIONS))

    def test_a_debias_method_expects_only_its_own_corrections(self):
        # re_l1 registers the inside_langs half only, so it expects fewer files
        # than two_step; checking against the wrong set would rerun every cell.
        two_step = expected_correction_suffixes("OneStop", DEBIAS_METHODS["two_step"])
        re_l1 = expected_correction_suffixes("OneStop", DEBIAS_METHODS["re_l1"])
        self.assertLess(len(re_l1), len(two_step))


class TestCorrectionsComplete(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_cell_with_every_expected_variant_is_complete(self):
        raw = _write_cell(self.tmp, "OneStop", expected_correction_suffixes("OneStop"))
        self.assertTrue(corrections_complete(raw, "OneStop"))

    def test_one_missing_variant_makes_the_cell_incomplete(self):
        """The regression: this is the cell a newly added correction must rerun."""
        expected = expected_correction_suffixes("OneStop")
        raw = _write_cell(self.tmp, "OneStop", expected[:-1])
        self.assertFalse(corrections_complete(raw, "OneStop"))

    def test_an_inapplicable_correction_is_not_required(self):
        """ProfAgg never lands on OneStop, so its absence must not block."""
        raw = _write_cell(self.tmp, "OneStop", expected_correction_suffixes("OneStop"))
        self.assertFalse(raw.with_name(raw.stem + PROFAGG + ".csv").exists())
        self.assertTrue(corrections_complete(raw, "OneStop"))

    def test_a_raw_only_cell_is_incomplete(self):
        raw = _write_cell(self.tmp, "Meco", [])
        self.assertFalse(corrections_complete(raw, "Meco"))


if __name__ == "__main__":
    unittest.main()
