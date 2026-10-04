"""Pool resolvers for the Pool × FoldMethod evaluation matrix.

A resolver yields PoolSegments — one (train_pool, test_pool, train_l1) tuple per
text-context partition. The orchestrator iterates segments, runs the chosen
FoldMethod inside each, and combines per-segment predictions according to
`combine_strategy`:

  - "concat":   segments are disjoint in participants (OneStop content_groups).
                Per-participant predictions from all segments are concatenated.
  - "average":  segments overlap in participants (MECO paragraph-halves).
                Predictions for the same participant across segments are averaged.

Pool semantics per dataset:

  OneStop (has content_group):
    ALL    — one segment over the full df
    SEEN   — one segment per content_group; train_pool == test_pool == cg rows
    UNSEEN — one segment per content_group; test = cg rows, train = other cgs

  MECO (no content_group; pre-aggregated per-half dfs):
    ALL    — one segment using the fully_agg dfs (all 12 paragraphs)
    SEEN   — segments (odd, odd) and (even, even); train and test on same half
    UNSEEN — segments (odd→even) and (even→odd); train and test on opposite halves
"""
from dataclasses import dataclass
from typing import Iterator, Literal

import pandas as pd

from src.constants import Fields, Pool


@dataclass
class PoolSegment:
    """One (train_pool, test_pool) pair within a Pool-axis evaluation."""
    segment_id: str
    train_pool: pd.DataFrame
    test_pool: pd.DataFrame
    train_l1: pd.DataFrame


class OneStopPoolResolver:
    """Per-content-group pool resolution for OneStop.

    Content_group has the form `f"{batch}_{level}"` (e.g. "1_0", "1_1", "2_0").
    Within a batch, the two levels (`{batch}_0` and `{batch}_1`) share the
    same source articles at different difficulty levels — so participants in
    different levels of the same batch overlap in text content. UNSEEN
    therefore drops ALL same-batch participants from train, not just same-CG
    participants, so test text is truly novel to the training set.
    """
    combine_strategy: Literal["concat", "average"] = "concat"

    def __init__(self, l1_df: pd.DataFrame, l2_df: pd.DataFrame):
        self.l1_df = l1_df
        self.l2_df = l2_df

    @staticmethod
    def _batch_of(cg) -> str:
        return str(cg).split("_", 1)[0]

    def iter_segments(self, pool: Pool) -> Iterator[PoolSegment]:
        if pool == Pool.ALL:
            yield PoolSegment(
                segment_id="all",
                train_pool=self.l2_df,
                test_pool=self.l2_df,
                train_l1=self.l1_df,
            )
            return
        if Fields.CONTENT_GROUP not in self.l2_df.columns:
            raise ValueError(
                f"OneStopPoolResolver requires {Fields.CONTENT_GROUP} for pool={pool}"
            )
        # Sorted for determinism — downstream rng draws inside the fold method
        # don't depend on cg order, but the result-CSV row order does.
        cgs = sorted(self.l2_df[Fields.CONTENT_GROUP].unique())
        cg_series = self.l2_df[Fields.CONTENT_GROUP].astype(str)
        batch_series = cg_series.str.split("_", n=1).str[0]
        for cg in cgs:
            in_cg = (self.l2_df[Fields.CONTENT_GROUP] == cg).to_numpy()
            if pool == Pool.SEEN:
                yield PoolSegment(
                    segment_id=str(cg),
                    train_pool=self.l2_df.iloc[in_cg],
                    test_pool=self.l2_df.iloc[in_cg],
                    train_l1=self.l1_df,
                )
            elif pool == Pool.UNSEEN:
                # Strict: exclude entire test batch from train. Same-batch
                # participants read the same articles at a different
                # difficulty level, so they overlap with test text content.
                test_batch = self._batch_of(cg)
                in_test_batch = (batch_series == test_batch).to_numpy()
                yield PoolSegment(
                    segment_id=str(cg),
                    train_pool=self.l2_df.iloc[~in_test_batch],
                    test_pool=self.l2_df.iloc[in_cg],
                    train_l1=self.l1_df,
                )
            else:
                raise ValueError(f"Unknown pool: {pool}")


class MecoPoolResolver:
    """Per paragraph-half pool resolution for MECO.

    halves_l1/l2: dict {"odd": df, "even": df} of per-half aggregated frames
    (loaded from features_and_targets/seen_unseen/all/all/{half}/).
    all_l1/l2: full-agg frames (fully_agg/all/all/). Required only for pool=ALL.
    """
    combine_strategy: Literal["concat", "average"] = "average"

    def __init__(
        self,
        halves_l1: dict,
        halves_l2: dict,
        all_l1: pd.DataFrame | None = None,
        all_l2: pd.DataFrame | None = None,
    ):
        self.halves_l1 = halves_l1
        self.halves_l2 = halves_l2
        self.all_l1 = all_l1
        self.all_l2 = all_l2

    def iter_segments(self, pool: Pool) -> Iterator[PoolSegment]:
        if pool == Pool.ALL:
            if self.all_l1 is None or self.all_l2 is None:
                raise ValueError("MecoPoolResolver requires all_l1/all_l2 for pool=ALL")
            yield PoolSegment(
                segment_id="all",
                train_pool=self.all_l2,
                test_pool=self.all_l2,
                train_l1=self.all_l1,
            )
            return
        for half in ("odd", "even"):
            if half not in self.halves_l2:
                continue
            if pool == Pool.SEEN:
                yield PoolSegment(
                    segment_id=half,
                    train_pool=self.halves_l2[half],
                    test_pool=self.halves_l2[half],
                    train_l1=self.halves_l1[half],
                )
            elif pool == Pool.UNSEEN:
                other = "even" if half == "odd" else "odd"
                if other not in self.halves_l2:
                    continue
                yield PoolSegment(
                    segment_id=f"{other}_to_{half}",
                    train_pool=self.halves_l2[other],
                    test_pool=self.halves_l2[half],
                    train_l1=self.halves_l1[other],
                )
            else:
                raise ValueError(f"Unknown pool: {pool}")
