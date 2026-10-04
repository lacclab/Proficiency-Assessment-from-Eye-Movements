"""Per-item prediction resolvers for SEEN/UNSEEN splits.

Handles (participant, item) level training/testing where items are paragraphs or articles.
"""
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from src.constants import Pool, Fields


@dataclass
class PerItemSegment:
    """Data for one per-item prediction segment."""
    segment_id: str
    pool: Pool
    item_col: str  # Fields.UNIQUE_PARAGRAPH_ID or Fields.ARTICLE_ID
    train_l1: pd.DataFrame
    lopo_df: pd.DataFrame | None = None  # SEEN: data to LOPO within
    train_df: pd.DataFrame | None = None  # UNSEEN: training rows
    test_df: pd.DataFrame | None = None   # UNSEEN: test rows


class PerItemOneStopResolver:
    """OneStop per-item resolver: splits by content_group (SEEN) or batch (UNSEEN)."""

    combine_strategy = "concat"  # segments are disjoint in participants

    def __init__(self, l1_df: pd.DataFrame, l2_df: pd.DataFrame):
        self.l1_df = l1_df
        self.l2_df = l2_df

    def iter_segments(self, pool: Pool, item_col: str = Fields.UNIQUE_PARAGRAPH_ID) -> Iterator[PerItemSegment]:
        """Iterate per-item segments.

        Args:
            pool: Pool.SEEN, Pool.UNSEEN, or Pool.ALL
            item_col: column identifying items (UNIQUE_PARAGRAPH_ID or ARTICLE_ID)
        """
        if pool == Pool.SEEN:
            # SEEN: one segment per item (LOPO within that item across all participants)
            # For paragraph: unique_paragraph_id is already globally unique
            # For article: article_id repeats per content_group, so use (article_id, content_group) pairs
            if item_col == Fields.UNIQUE_PARAGRAPH_ID:
                # Paragraph level: iterate over unique_paragraph_id
                unique_items = sorted(self.l2_df[item_col].unique())
                for item_id in unique_items:
                    item_df = self.l2_df[self.l2_df[item_col] == item_id]
                    # Filter L1 to same paragraph_id and content_group(s) for augmentation
                    if item_col in self.l1_df.columns:
                        cgs_in_item = item_df[Fields.CONTENT_GROUP].unique() if Fields.CONTENT_GROUP in item_df.columns else None
                        if cgs_in_item is not None and Fields.CONTENT_GROUP in self.l1_df.columns:
                            item_l1 = self.l1_df[(self.l1_df[item_col] == item_id) & (self.l1_df[Fields.CONTENT_GROUP].isin(cgs_in_item))]
                        else:
                            item_l1 = self.l1_df[self.l1_df[item_col] == item_id]
                    else:
                        item_l1 = self.l1_df
                    yield PerItemSegment(
                        segment_id=str(item_id),
                        pool=Pool.SEEN,
                        item_col=item_col,
                        train_l1=item_l1,
                        lopo_df=item_df,
                    )
            else:
                # Article level: article_id repeats per content_group, use (article_id, content_group) pairs
                if Fields.ARTICLE_ID in self.l2_df.columns and Fields.CONTENT_GROUP in self.l2_df.columns:
                    article_cg_pairs = sorted(self.l2_df[[Fields.ARTICLE_ID, Fields.CONTENT_GROUP]].drop_duplicates().values.tolist())
                    for article_id, cg in article_cg_pairs:
                        item_df = self.l2_df[(self.l2_df[Fields.ARTICLE_ID] == article_id) & (self.l2_df[Fields.CONTENT_GROUP] == cg)]
                        # Filter L1 to same article and content_group
                        if Fields.ARTICLE_ID in self.l1_df.columns and Fields.CONTENT_GROUP in self.l1_df.columns:
                            item_l1 = self.l1_df[(self.l1_df[Fields.ARTICLE_ID] == article_id) & (self.l1_df[Fields.CONTENT_GROUP] == cg)]
                        else:
                            item_l1 = self.l1_df
                        segment_id = f"{article_id}_{cg}"
                        yield PerItemSegment(
                            segment_id=segment_id,
                            pool=Pool.SEEN,
                            item_col=item_col,
                            train_l1=item_l1,
                            lopo_df=item_df,
                        )
                else:
                    raise KeyError(f"Article-level per-item requires both {Fields.ARTICLE_ID} and {Fields.CONTENT_GROUP} in dataframes for proper splitting.")
        elif pool == Pool.UNSEEN:
            # UNSEEN: one segment per batch (train on 2, test on 1)
            batches = sorted(self.l2_df[Fields.BATCH].unique())
            for test_batch in batches:
                test_df = self.l2_df[self.l2_df[Fields.BATCH] == test_batch]
                train_df = self.l2_df[self.l2_df[Fields.BATCH] != test_batch]
                # Filter L1 to the same training batches
                train_batches = [b for b in batches if b != test_batch]
                if Fields.BATCH in self.l1_df.columns:
                    train_l1 = self.l1_df[self.l1_df[Fields.BATCH].isin(train_batches)]
                else:
                    train_l1 = self.l1_df
                yield PerItemSegment(
                    segment_id=str(test_batch),
                    pool=Pool.UNSEEN,
                    item_col=item_col,
                    train_l1=train_l1,
                    train_df=train_df,
                    test_df=test_df,
                )
        elif pool == Pool.ALL:
            # ALL: single segment, full data, LOPO per item
            yield PerItemSegment(
                segment_id="all",
                pool=Pool.ALL,
                item_col=item_col,
                train_l1=self.l1_df,
                lopo_df=self.l2_df,
            )


class PerItemMecoResolver:
    """MECO per-item resolver: per-paragraph LOPO (SEEN) or odd/even halves (UNSEEN)."""

    combine_strategy = "concat"  # odd and even predict on disjoint paragraphs

    def __init__(
        self,
        l1_df: pd.DataFrame,
        l2_df: pd.DataFrame,
        halves_l1: dict | None = None,
        halves_l2: dict | None = None,
    ):
        """Initialize resolver.

        Args:
            l1_df: L1 fully_agg data
            l2_df: L2 fully_agg data
            halves_l1: dict of {half: l1_df} for odd/even per-item splits (UNSEEN only)
            halves_l2: dict of {half: l2_df} for odd/even per-item splits (UNSEEN only)
        """
        self.l1_df = l1_df
        self.l2_df = l2_df
        self.halves_l1 = halves_l1 or {}
        self.halves_l2 = halves_l2 or {}

    def iter_segments(self, pool: Pool, item_col: str = Fields.UNIQUE_PARAGRAPH_ID) -> Iterator[PerItemSegment]:
        """Iterate per-item segments.

        Args:
            pool: Pool.SEEN, Pool.UNSEEN, or Pool.ALL
            item_col: always UNIQUE_PARAGRAPH_ID for MECO (no article-level per-item)
        """
        if pool == Pool.SEEN:
            # SEEN: one segment per paragraph, LOPO within that paragraph
            unique_items = sorted(self.l2_df[item_col].unique())
            for item_id in unique_items:
                item_df = self.l2_df[self.l2_df[item_col] == item_id]
                # Filter L1 to same paragraph_id for augmentation
                if item_col in self.l1_df.columns:
                    item_l1 = self.l1_df[self.l1_df[item_col] == item_id]
                else:
                    item_l1 = self.l1_df
                yield PerItemSegment(
                    segment_id=str(item_id),
                    pool=Pool.SEEN,
                    item_col=item_col,
                    train_l1=item_l1,
                    lopo_df=item_df,
                )
        elif pool == Pool.UNSEEN:
            # UNSEEN: paragraph-parity splits (odd→even, even→odd)
            # Split by paragraph index within each participant (0=even, 1=odd, 2=even, etc.)
            for train_parity, test_parity in [(0, 1), (1, 0)]:
                # Create train set: all train_parity paragraphs from all participants
                train_rows = []
                for participant in sorted(self.l2_df[Fields.SUBJECT_ID].unique()):
                    p_rows = self.l2_df[self.l2_df[Fields.SUBJECT_ID] == participant].reset_index(drop=True)
                    # Select paragraphs with matching parity
                    train_rows.append(p_rows[p_rows.index % 2 == train_parity])
                train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()

                # Create test set: all test_parity paragraphs from all participants
                test_rows = []
                for participant in sorted(self.l2_df[Fields.SUBJECT_ID].unique()):
                    p_rows = self.l2_df[self.l2_df[Fields.SUBJECT_ID] == participant].reset_index(drop=True)
                    # Select paragraphs with matching parity
                    test_rows.append(p_rows[p_rows.index % 2 == test_parity])
                test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()

                # Create corresponding L1 data
                train_l1_rows = []
                for participant in sorted(self.l1_df[Fields.SUBJECT_ID].unique()):
                    p_rows = self.l1_df[self.l1_df[Fields.SUBJECT_ID] == participant].reset_index(drop=True)
                    train_l1_rows.append(p_rows[p_rows.index % 2 == train_parity])
                train_l1 = pd.concat(train_l1_rows, ignore_index=True) if train_l1_rows else pd.DataFrame()

                parity_names = {0: "even", 1: "odd"}
                segment_id = f"{parity_names[train_parity]}→{parity_names[test_parity]}"

                yield PerItemSegment(
                    segment_id=segment_id,
                    pool=Pool.UNSEEN,
                    item_col=item_col,
                    train_l1=train_l1,
                    train_df=train_df,
                    test_df=test_df,
                )
        elif pool == Pool.ALL:
            # ALL: single segment, full data, LOPO per paragraph
            yield PerItemSegment(
                segment_id="all",
                pool=Pool.ALL,
                item_col=item_col,
                train_l1=self.l1_df,
                lopo_df=self.l2_df,
            )
