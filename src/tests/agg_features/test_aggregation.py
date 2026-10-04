"""Unit tests for first_p_agg aggregation logic, on the real harmonized IA data."""
import unittest
from functools import lru_cache

import pandas as pd
import pytest

from src.constants import Fields, DATA_PATH
from src.preprocessing.run_extract_features import filter_rows_first_p_agg, filter_rows_by_conditions

pytestmark = pytest.mark.slow


@lru_cache(maxsize=None)
def _read_harmonized_ia(dataset: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / dataset / 'harmonized' / 'ia.csv')


def harmonized_ia(dataset: str) -> pd.DataFrame:
    """A fresh copy of `dataset`'s harmonized IA file. The file is read from disk
    once per session (OneStop's is 5.8 GB) rather than once per test."""
    return _read_harmonized_ia(dataset).copy()


class TestOneStopArticleAgg(unittest.TestCase):
    """Test OneStop article aggregation."""

    def setUp(self):
        """Load OneStop sample data for a single participant (with condition filtering)."""
        dataset = 'OneStopL1'
        ia_df = harmonized_ia(dataset)
        # Apply condition filtering to match real feature extraction (removes article 0 / practice trials)
        ia_df = filter_rows_by_conditions(ia_df, 'OneStopL1', 'all', 'all', keep_practice=False)
        sample_subj = ia_df[Fields.SUBJECT_ID].unique()[0]
        self.sample_data = ia_df[ia_df[Fields.SUBJECT_ID] == sample_subj].copy()

    def test_article_filtering_keeps_correct_ids(self):
        """Test that article filtering keeps articles with ID <= num_first_agg."""
        filtered = filter_rows_first_p_agg(self.sample_data.copy(), "article", 3)
        article_ids = sorted(filtered[Fields.ARTICLE_ID].unique())

        # Should keep articles 1, 2, 3 (articles are 1-indexed after condition filtering removes article 0)
        for aid in article_ids:
            self.assertLessEqual(aid, 3, f"Found article with ID {aid} > 3")
        self.assertEqual(len(article_ids), 3, f"Kept {len(article_ids)} articles, expected 3")

    def test_article_filtering_removes_higher_ids(self):
        """Test that articles with ID > num_first_agg are removed."""
        max_article = self.sample_data[Fields.ARTICLE_ID].max()
        if max_article > 5:
            filtered = filter_rows_first_p_agg(self.sample_data.copy(), "article", 5)
            filtered_max = filtered[Fields.ARTICLE_ID].max()
            self.assertLessEqual(filtered_max, 5, f"Article {filtered_max} > 5 was not filtered")


class TestOneStopParagraphAgg(unittest.TestCase):
    """Test OneStop paragraph aggregation."""

    def setUp(self):
        """Load OneStop sample data for a single participant."""
        dataset = 'OneStopL1'
        ia_df = harmonized_ia(dataset)
        sample_subj = ia_df[Fields.SUBJECT_ID].unique()[0]
        self.sample_data = ia_df[ia_df[Fields.SUBJECT_ID] == sample_subj].copy()

    def test_paragraph_filtering_keeps_n_combinations(self):
        """Test that paragraph filtering keeps first N (article_id, paragraph_id) combinations."""
        n = 5
        filtered = filter_rows_first_p_agg(self.sample_data.copy(), "paragraph", n)

        unique_pairs = filtered[[Fields.ARTICLE_ID, Fields.PARAGRAPH_ID]].drop_duplicates()
        self.assertEqual(len(unique_pairs), n, f"Expected {n} unique pairs, got {len(unique_pairs)}")

    def test_paragraph_filtering_maintains_article_paragraph_pairs(self):
        """Test that filtered data maintains valid (article_id, paragraph_id) pairs."""
        filtered = filter_rows_first_p_agg(self.sample_data.copy(), "paragraph", 3)

        # Check that all kept rows have valid article and paragraph IDs
        self.assertTrue(filtered[Fields.ARTICLE_ID].notna().all(), "Found NaN in ARTICLE_ID")
        self.assertTrue(filtered[Fields.PARAGRAPH_ID].notna().all(), "Found NaN in PARAGRAPH_ID")

    def test_paragraph_filtering_reduces_data_size(self):
        """Test that paragraph filtering reduces the dataset size."""
        original_size = len(self.sample_data)
        filtered = filter_rows_first_p_agg(self.sample_data.copy(), "paragraph", 3)
        filtered_size = len(filtered)

        self.assertLess(filtered_size, original_size, f"Filtering didn't reduce size: {filtered_size} >= {original_size}")


class TestMECOParagraphAgg(unittest.TestCase):
    """Test MECO paragraph aggregation."""

    def setUp(self):
        """Load MECO sample data for a single participant."""
        dataset = 'MecoL1'
        ia_df = harmonized_ia(dataset)
        sample_subj = ia_df[Fields.SUBJECT_ID].unique()[0]
        self.sample_data = ia_df[ia_df[Fields.SUBJECT_ID] == sample_subj].copy()

    def test_meco_paragraph_filtering_keeps_n_paragraphs(self):
        """Test that MECO paragraph filtering keeps first N unique_paragraph_id values."""
        n = 5
        filtered = filter_rows_first_p_agg(self.sample_data.copy(), "paragraph", n)

        unique_paras = filtered[Fields.UNIQUE_PARAGRAPH_ID].unique()
        self.assertEqual(len(unique_paras), n, f"Expected {n} unique paragraphs, got {len(unique_paras)}")

    def test_meco_paragraph_filtering_sequential(self):
        """Test that MECO keeps sequential paragraph IDs (1, 2, 3...)."""
        filtered = filter_rows_first_p_agg(self.sample_data.copy(), "paragraph", 6)

        unique_paras = sorted(filtered[Fields.UNIQUE_PARAGRAPH_ID].unique())
        expected = list(range(1, 7))  # MECO paragraphs are 1-indexed

        self.assertEqual(unique_paras, expected, f"Expected {expected}, got {unique_paras}")

    def test_meco_paragraph_filtering_reduces_data(self):
        """Test that MECO paragraph filtering reduces dataset size."""
        original_size = len(self.sample_data)
        filtered = filter_rows_first_p_agg(self.sample_data.copy(), "paragraph", 4)
        filtered_size = len(filtered)

        self.assertLess(filtered_size, original_size, f"Filtering didn't reduce size: {filtered_size} >= {original_size}")

    def test_meco_all_12_paragraphs(self):
        """Test that requesting all 12 paragraphs works correctly."""
        filtered = filter_rows_first_p_agg(self.sample_data.copy(), "paragraph", 12)

        unique_paras = len(filtered[Fields.UNIQUE_PARAGRAPH_ID].unique())
        self.assertLessEqual(unique_paras, 12, f"Expected <= 12 paragraphs, got {unique_paras}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Load MECO sample data for edge case tests."""
        dataset = 'MecoL1'
        ia_df = harmonized_ia(dataset)
        sample_subj = ia_df[Fields.SUBJECT_ID].unique()[0]
        self.sample_data = ia_df[ia_df[Fields.SUBJECT_ID] == sample_subj].copy()

    def test_first_paragraph_only(self):
        """Test filtering for just the first paragraph."""
        filtered = filter_rows_first_p_agg(self.sample_data.copy(), "paragraph", 1)

        unique_paras = filtered[Fields.UNIQUE_PARAGRAPH_ID].unique()
        self.assertEqual(len(unique_paras), 1, f"Expected 1 paragraph, got {len(unique_paras)}")
        self.assertEqual(unique_paras[0], 1, f"Expected paragraph 1, got {unique_paras[0]}")

    def test_invalid_agg_level_raises_error(self):
        """Test that invalid aggregation level raises ValueError."""
        with self.assertRaises(ValueError):
            filter_rows_first_p_agg(self.sample_data.copy(), "invalid_level", 5)


if __name__ == '__main__':
    unittest.main()
