"""Aggregation configurations for different datasets."""
from src.constants import AggTypes

# Aggregation configs for OneStop datasets
ONESTOP_AGG_CONFIGS = (
    [{"agg_type": AggTypes.FULL}]
    + [{"agg_type": AggTypes.FIRST_P, "p_agg_level": "article", "p_agg": n} for n in range(1, 11)]
    + [{"agg_type": AggTypes.FIRST_P, "p_agg_level": "paragraph", "p_agg": n} for n in [1, 2, 5, 10, 15, 20, 30, 40, 54]]
    + [{"agg_type": AggTypes.MOVING_P, "p_agg_level": "paragraph", "p_agg": n} for n in [1, 2, 5, 10, 15, 20, 30, 40, 54]]
    + [{"agg_type": AggTypes.MOVING_P, "p_agg_level": "article", "p_agg": n} for n in [1, 2, 5, 10]]
    + [{"agg_type": AggTypes.PER_ITEM, "p_agg_level": "paragraph"}]
    + [{"agg_type": AggTypes.PER_ITEM, "p_agg_level": "article"}]
)

# Aggregation configs for MECO datasets (max 12 paragraphs). The seen/unseen
# paragraph-half contrast is evaluated through the Pool axis on fully_agg, but
# the per-half feature files those pools read
# (features_and_targets/seen_unseen/all/all/{odd,even}/) come from the
# SEEN_UNSEEN configs below, so they must stay.
MECO_AGG_CONFIGS = (
    [{"agg_type": AggTypes.FULL}]
    + [{"agg_type": AggTypes.FIRST_P, "p_agg_level": "paragraph", "p_agg": n} for n in range(1, 13)]
    + [{"agg_type": AggTypes.MOVING_P, "p_agg_level": "paragraph", "p_agg": n} for n in range(1, 13)]
    + [{"agg_type": AggTypes.PER_ITEM, "p_agg_level": "paragraph"}]
    + [{"agg_type": AggTypes.SEEN_UNSEEN, "half": h} for h in ("odd", "even")]
)

AGG_CONFIGS_BY_DATASET = {
    "OneStopL1": ONESTOP_AGG_CONFIGS,
    "OneStopL2": ONESTOP_AGG_CONFIGS,
    "MecoL1": MECO_AGG_CONFIGS,
    "MecoL2": MECO_AGG_CONFIGS,
}


def get_agg_configs(dataset: str = "OneStop", agg_types: set[str] = None) -> tuple:
    """Aggregation configs of `dataset` (a MECO or OneStop name), optionally
    only those whose agg_type is in `agg_types`."""
    if dataset.startswith("Meco"):
        base_configs = MECO_AGG_CONFIGS
    else:
        base_configs = ONESTOP_AGG_CONFIGS

    if agg_types is None:
        return base_configs

    return tuple(cfg for cfg in base_configs if cfg.get("agg_type") in agg_types)
