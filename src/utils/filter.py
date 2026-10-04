"""Utility functions for filtering DataFrames."""
import pandas as pd
from src.constants import Fields


def filter_by_preview(l1_df: pd.DataFrame, l2_df: pd.DataFrame, preview: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to one OneStop preview: 'Hunting' (question shown
    before reading), 'Gathering' (not shown) or 'All'."""
    if preview == "All":
        return l1_df, l2_df
    elif preview == "Hunting":
        return (
            l1_df[l1_df[Fields.HAS_PREVIEW] == 1],
            l2_df[l2_df[Fields.HAS_PREVIEW] == 1],
        )
    elif preview == "Gathering":
        return (
            l1_df[l1_df[Fields.HAS_PREVIEW] == 0],
            l2_df[l2_df[Fields.HAS_PREVIEW] == 0],
        )
    else:
        raise ValueError(f"Invalid preview type: {preview}. Must be 'All', 'Hunting', or 'Gathering'")
