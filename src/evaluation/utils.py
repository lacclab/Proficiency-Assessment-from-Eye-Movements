"""Utilities for evaluation scripts."""
import contextlib
import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)




def save_preserving_unevaluated(new_df: pd.DataFrame, path: Path,
                                scope: dict, label: str = "",
                                row_predicate: Optional[Callable] = None) -> pd.DataFrame:
    """Write `new_df` to `path`, keeping existing rows this run did not cover.

    Evaluation entrypoints accept filters (target_cols, feature_sets,
    model_names, agg_type, previews...). A plain `to_csv` would let a filtered
    run replace the whole file with its own narrow slice, silently deleting
    every row outside the filter.

    `scope` maps a column name to the filter the caller applied to it. An
    existing row is "covered" — and so replaced — only when EVERY active filter
    matches it. Uncovered rows are carried over unchanged. `row_predicate` adds
    one more condition that needs the whole row, for filters spanning columns
    (EyeScore's allowed previews depend on the dataset).

    When neither a filter nor a row_predicate is active the run regenerates
    everything, so the file is overwritten wholesale. A filter naming a column
    the existing CSV does not have is skipped with a warning: that widens what
    counts as covered, which risks dropping rows but cannot produce duplicates
    — and duplicate rows would corrupt every downstream table silently, where
    dropped rows are recoverable by re-running.
    """
    path = Path(path)
    tag = label or path.name

    def _overwrite(reason: str = "") -> pd.DataFrame:
        if reason:
            logger.info("%s: writing %d rows (%s)", tag, len(new_df), reason)
        new_df.to_csv(path, index=False)
        return new_df

    active = {c: m for c, m in scope.items() if m is not None}
    if not active and row_predicate is None:
        return _overwrite("unfiltered run — regenerates the whole file")
    if not path.exists():
        return _overwrite("no existing file to merge into")

    try:
        old = pd.read_csv(path)
    except Exception as e:  # unreadable/corrupt — better to rewrite than to fail
        logger.warning("%s: could not read existing file for merge (%s); overwriting", tag, e)
        return _overwrite()
    if old.empty:
        return _overwrite("existing file is empty")

    covered = pd.Series(True, index=old.index)
    for col, matcher in active.items():
        if col not in old.columns:
            logger.warning(
                "%s: existing file has no %r column, so the filter on it cannot be "
                "applied; rows matching the other filters will be replaced.", tag, col)
            continue
        values = old[col].astype(str)
        if callable(matcher):
            covered &= values.map(matcher)
        else:
            allowed = {str(v) for v in matcher}
            covered &= values.isin(allowed)

    if row_predicate is not None:
        covered &= old.apply(row_predicate, axis=1).astype(bool)

    kept = old[~covered]
    if kept.empty:
        return _overwrite("filters cover every existing row")

    missing = [c for c in kept.columns if c not in new_df.columns]
    if missing:
        logger.warning("%s: kept rows carry columns absent from this run (%s); "
                       "they will be NaN for the new rows.", tag, missing)

    merged = pd.concat([kept, new_df], ignore_index=True)
    merged.to_csv(path, index=False)
    logger.info("%s: kept %d rows outside this run's scope, wrote %d new rows (%d total)",
                tag, len(kept), len(new_df), len(merged))
    return merged


@contextlib.contextmanager
def suppress_titles():
    """Make matplotlib's title APIs (Axes.set_title, Figure.suptitle, plt.title)
    no-ops for the duration of the block, so paper-ready figures are produced
    without titles and no call site has to change. Restores the originals on
    exit, even if the block raises. Worker processes that draw figures must
    enter it themselves (a fresh process imports matplotlib unpatched)."""
    import matplotlib.axes as mpl_axes
    import matplotlib.figure as mpl_figure
    import matplotlib.pyplot as plt

    orig_set_title = mpl_axes.Axes.set_title
    orig_suptitle = mpl_figure.Figure.suptitle
    orig_plt_title = plt.title
    mpl_axes.Axes.set_title = lambda self, *a, **k: None
    mpl_figure.Figure.suptitle = lambda self, *a, **k: None
    plt.title = lambda *a, **k: None
    try:
        yield
    finally:
        mpl_axes.Axes.set_title = orig_set_title
        mpl_figure.Figure.suptitle = orig_suptitle
        plt.title = orig_plt_title
