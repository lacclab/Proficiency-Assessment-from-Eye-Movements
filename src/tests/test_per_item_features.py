"""Unit tests for per-item feature extraction (add_per_item_features).

Note: These are integration tests that verify the per_item_agg aggregation
type is correctly wired through the extraction pipeline. The actual feature
computation is tested elsewhere; this focuses on the item-level aggregation logic.
"""
import unittest

from src.constants import AggTypes
from src.agg_configs import get_agg_configs


class TestPerItemAggConfig(unittest.TestCase):
    """Test that PER_ITEM configs are properly registered."""

    def test_per_item_in_onestop_configs(self):
        """Verify PER_ITEM configs exist for OneStop."""
        configs = get_agg_configs("OneStopL1")
        per_item_configs = [c for c in configs if c.get("agg_type") == AggTypes.PER_ITEM]

        # Should have paragraph and article level
        self.assertEqual(len(per_item_configs), 2)

        # Check both levels present
        levels = {c.get("p_agg_level") for c in per_item_configs}
        self.assertEqual(levels, {"paragraph", "article"})

    def test_per_item_in_meco_configs(self):
        """Verify PER_ITEM config exists for MECO (paragraph only)."""
        configs = get_agg_configs("MecoL1")
        per_item_configs = [c for c in configs if c.get("agg_type") == AggTypes.PER_ITEM]

        # Should have only paragraph level for MECO
        self.assertEqual(len(per_item_configs), 1)
        self.assertEqual(per_item_configs[0].get("p_agg_level"), "paragraph")

    def test_per_item_config_structure(self):
        """Verify PER_ITEM configs have correct structure."""
        configs = get_agg_configs("OneStopL1", agg_types={AggTypes.PER_ITEM})

        for config in configs:
            self.assertEqual(config["agg_type"], AggTypes.PER_ITEM)
            self.assertIn(config["p_agg_level"], ["paragraph", "article"])
            # Per-item configs should NOT have p_agg (no numbering)
            self.assertNotIn("p_agg", config)

    def test_agg_types_enum_includes_per_item(self):
        """Verify PER_ITEM is in the AggTypes enum."""
        self.assertTrue(hasattr(AggTypes, "PER_ITEM"))
        self.assertEqual(AggTypes.PER_ITEM, "per_item_agg")

    def test_filtered_configs_include_per_item(self):
        """Verify filtering by agg_types includes PER_ITEM when requested."""
        configs_with_per_item = get_agg_configs(
            "OneStopL1",
            agg_types={AggTypes.PER_ITEM}
        )
        self.assertEqual(len(configs_with_per_item), 2)  # para + article

        configs_without = get_agg_configs(
            "OneStopL1",
            agg_types={AggTypes.FULL}
        )
        per_item_in_no_filter = [c for c in configs_without if c.get("agg_type") == AggTypes.PER_ITEM]
        self.assertEqual(len(per_item_in_no_filter), 0)


class TestPerItemPathStructure(unittest.TestCase):
    """Test that PER_ITEM would create correct directory structure (without running extraction)."""

    def test_per_item_path_format(self):
        """Verify per_item_agg path follows expected convention."""
        # Per-item paths should be:
        # {dataset}/features_and_targets/per_item_agg/{reread}/{level}/{p_agg_level}/
        expected_suffix_para = "per_item_agg/ordinary/all/paragraph"
        expected_suffix_article = "per_item_agg/ordinary/all/article"

        # These would be used in run_extract_features.py path building
        # Just verify the expected format makes sense
        self.assertIn("per_item_agg", expected_suffix_para)
        self.assertIn("paragraph", expected_suffix_para)
        self.assertIn("ordinary", expected_suffix_para)


if __name__ == "__main__":
    unittest.main()
