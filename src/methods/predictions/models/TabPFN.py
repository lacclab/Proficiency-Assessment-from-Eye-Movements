import re

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import TRANSITIONS_FEATURE_PREFIX, WFC_FEATURE_PREFIX
from src.methods.predictions.models.BaseModel import BaseModel


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _parse_per_text_col(col: str) -> tuple[str, int, int] | None:
    """``col`` -> (paragraph, lowest word, highest word), or None if not per-text.

    ``wfc_{p}_{w}_{FP|TF}`` spans the single word w, so FP and TF always share a
    span and are kept or dropped together. ``transitions_{p}_{i}_{j}`` spans
    [min(i, j), max(i, j)], so a column survives only when BOTH of its endpoints
    are inside a kept window — i.e. every kept paragraph contributes a coherent
    sub-block of its transition matrix rather than a ragged edge.

    The int(float(...)) on the WFC branch is not symmetry with transitions: it
    is needed only there. MECO's harmonized IA_ID is float64, and
    compute_wfc_features interpolates it into the name uncast, so MECO words
    read "1.0" where OneStop's read "1" and a plain int() raises.
    compute_transitions_features DOES cast its two IA columns, so transitions
    names are integer by construction on every dataset. (The WFC name is fixed
    at source now too, but parquets extracted before that keep the float form.)
    """
    if col.startswith(WFC_FEATURE_PREFIX):
        pid, w, _metric = col[len(WFC_FEATURE_PREFIX):].rsplit("_", 2)
        w = int(float(w))
        return pid, w, w
    if col.startswith(TRANSITIONS_FEATURE_PREFIX):
        pid, i, j = col[len(TRANSITIONS_FEATURE_PREFIX):].rsplit("_", 2)
        i, j = int(i), int(j)
        return pid, min(i, j), max(i, j)
    return None


def _window_mask(lo, hi, words: list[int], cap: int, anchor: str):
    """Largest word window anchored at start/middle/end whose column count fits
    `cap`. Returns a boolean mask over the paragraph's columns."""
    if cap <= 0 or not words:
        return np.zeros(len(lo), dtype=bool)
    if anchor == "start":
        spans = [(words[0], w) for w in words]
    elif anchor == "end":
        spans = [(w, words[-1]) for w in reversed(words)]
    else:  # middle — expand symmetrically around the median word
        idx = len(words) // 2
        spans = [(words[max(0, idx - k)], words[min(len(words) - 1, idx + k)])
                 for k in range(len(words))]
    best = np.zeros(len(lo), dtype=bool)
    for a, b in spans:
        m = (lo >= a) & (hi <= b)
        if m.sum() > cap:
            break
        best = m
    return best


def _truncate_reading_order(cols: list[str], budget: int) -> list[str]:
    """Whole word units in reading order until the budget runs out — the
    original rule, kept for WFC.

    Units are (paragraph, word): a WFC word's FP and TF always survive together,
    and a transitions column joins the unit of its later endpoint so a kept word
    contributes exactly its transitions to/from already-kept words. Paragraphs
    are natural-sorted and words ascend, so the text is cut where the budget
    runs out — never mid-word, never one measure of a word.
    """
    units: dict[tuple, list[str]] = {}
    always_keep = []
    for c in cols:
        parsed = _parse_per_text_col(c)
        if parsed is None:
            always_keep.append(c)
            continue
        pid, _lo, hi = parsed
        units.setdefault((pid, hi), []).append(c)

    remaining = budget - len(always_keep)
    if remaining < 0:
        raise ValueError(
            f"{len(always_keep)} non-per-text columns already exceed the "
            f"feature budget of {budget}."
        )

    selected: set[str] = set()
    cut_at = None
    for key in sorted(units, key=lambda k: (_natural_key(k[0]), k[1])):
        unit_cols = units[key]
        if len(unit_cols) > remaining:
            cut_at = key
            break
        selected.update(unit_cols)
        remaining -= len(unit_cols)

    if not selected:
        raise ValueError(
            f"Feature budget {budget} too small to fit even the first "
            f"per-text word unit ({cut_at})."
        )
    kept = always_keep + [c for c in cols if c in selected]
    logger.info(
        f"Per-text truncation (reading order): kept {len(kept)}/{len(cols)} "
        f"columns (budget {budget}); text cut at (paragraph, word) = {cut_at}."
    )
    return kept


def truncate_per_text_columns(cols: list[str], budget: int) -> list[str] | None:
    """Cut per-text (TRANSITIONS / WFC) columns down to `budget`.

    The two feature families are truncated differently, because the budget buys
    very different amounts of text in each:

    WFC — reading order, `_truncate_reading_order`. A word costs 2 columns, so
    the budget already spans 9-11 OneStop paragraphs (and MECO's halves fit
    under it entirely, so nothing is cut). Left as it was.

    TRANSITIONS — the budget is split evenly across paragraphs and each
    paragraph's share is spent on ONE word window centred on its middle. A
    paragraph's transitions cost ~2k columns, so reading order reached only 1-2
    of MECO's 12 texts and 1-2 of OneStop's 54 paragraphs: TabPFN judged a
    participant from the opening of one text while every other model saw all of
    them, and any participant missing that one text fell back to pure mean
    imputation. The middle anchor was chosen empirically over start, end, and a
    three-window start+middle+end split — best mean and best worst case across
    MECO and OneStop on both targets, using fewer columns than reading order.
    The three-window split has a higher ceiling but needs a generous
    per-paragraph budget to reach it (on OneStop's 54 paragraphs, thirds of a
    ~37-column share shrink to ~3-word windows, too fragmented to carry signal).
    End-of-paragraph is the weakest anchor everywhere.

    Non-per-text columns are always kept and count against the budget. Returns
    the selected columns (original relative order), or None when no truncation
    is needed. Selection depends only on column names, so it is identical across
    folds (leak-free).
    """
    if len(cols) <= budget:
        return None

    # Feature sets are single-prefix (FIXED_TEXT_GROUP_PREFIXES maps one prefix
    # per set), so this picks the family, it does not mix them.
    if not any(c.startswith(TRANSITIONS_FEATURE_PREFIX) for c in cols):
        return _truncate_reading_order(cols, budget)

    by_par: dict[str, list[tuple[str, int, int]]] = {}
    always_keep = []
    for c in cols:
        parsed = _parse_per_text_col(c)
        if parsed is None:
            always_keep.append(c)
            continue
        pid, lo, hi = parsed
        by_par.setdefault(pid, []).append((c, lo, hi))

    remaining = budget - len(always_keep)
    if remaining < 0:
        raise ValueError(
            f"{len(always_keep)} non-per-text columns already exceed the "
            f"feature budget of {budget}."
        )
    if not by_par:
        return None

    pars = sorted(by_par, key=_natural_key)
    per_par = max(1, remaining // len(pars))

    selected: set[str] = set()
    for pid in pars:
        entries = by_par[pid]
        cols_p = np.array([e[0] for e in entries], dtype=object)
        lo = np.array([e[1] for e in entries])
        hi = np.array([e[2] for e in entries])
        words = sorted({*lo.tolist(), *hi.tolist()})
        mask = _window_mask(lo, hi, words, per_par, "middle")
        selected.update(cols_p[mask].tolist())

    if not selected:
        raise ValueError(
            f"Feature budget {budget} too small to fit a single word window "
            f"in any of the {len(pars)} paragraphs."
        )
    kept = always_keep + [c for c in cols if c in selected]
    # Only reachable when per_par was floored to 1 by a budget smaller than the
    # paragraph count; the even split cannot otherwise overshoot.
    if len(kept) > budget:
        kept = kept[:budget]
    logger.info(
        f"Per-text truncation (mid-paragraph windows): kept {len(kept)}/"
        f"{len(cols)} columns (budget {budget}) spread over {len(pars)} "
        f"paragraphs (~{per_par} per paragraph)."
    )
    return kept


class TabPFN(BaseModel):
    """Wrapper around the TabPFN tabular foundation model.

    TabPFN (https://github.com/PriorLabs/TabPFN) is a pretrained transformer
    for tabular prediction. Like TabStar it does its OWN preprocessing
    (encoding, normalization), so we skip the z-scaling pipeline and inherit
    BaseModel's ``prepare_training_data`` / ``prepare_test_data`` (fold-mean
    imputation only).

    Per-text feature sets wider than ``max_features`` (the pretraining limit,
    MAX_FEATURES) are cut down by ``truncate_per_text_columns``;
    ``ignore_pretraining_limits`` is passed through to TabPFN.

    The ``tabpfn`` package is an optional heavyweight dependency (torch +
    pretrained weights downloaded from HuggingFace on first fit), so it is
    imported lazily inside ``_build_model`` — this keeps ``import src.configs``
    (which instantiates every model) working even when tabpfn isn't installed.
    The ImportError only fires if you actually train a TabPFN model.
    """

    # TabPFN v3's official pretraining limit (v2 was 500). Checked against
    # inference_config_.MAX_NUMBER_OF_FEATURES of the installed weights.
    MAX_FEATURES = 2000

    def __init__(
        self,
        device: str | None = None,
        random_state: int = 42,
        ignore_pretraining_limits: bool = False,
        max_features: int = MAX_FEATURES,
        **tabpfn_kwargs,
    ):
        super().__init__(model_name="TabPFN")
        self.device = device
        self.random_state = random_state
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.max_features = max_features
        self.tabpfn_kwargs = tabpfn_kwargs
        self.model = None  # built lazily at fit time
        self._fit_columns = None  # columns actually fitted (post-truncation)
        self._truncation_cache: dict = {}

    def _build_model(self):
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as e:
            raise ImportError(
                "TabPFN requires the 'tabpfn' package, which is not installed. "
                "Install it with `pip install tabpfn` (pulls in torch + pretrained "
                "weights) to use this model."
            ) from e

        kwargs = dict(self.tabpfn_kwargs)
        if self.device is not None:
            kwargs.setdefault("device", self.device)
        kwargs.setdefault("random_state", self.random_state)
        kwargs.setdefault(
            "ignore_pretraining_limits", self.ignore_pretraining_limits
        )
        return TabPFNRegressor(**kwargs)

    def _truncated_columns(self, cols: list[str]) -> list[str] | None:
        """Memoized truncate_per_text_columns — the same fold column set
        recurs thousands of times per LOPO cell."""
        key = (hash(tuple(cols)), self.max_features)
        if key not in self._truncation_cache:
            self._truncation_cache[key] = truncate_per_text_columns(
                cols, self.max_features
            )
        return self._truncation_cache[key]

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = y.squeeze() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        sel = self._truncated_columns(list(X.columns))
        if sel is not None:
            X = X[sel]
        self._fit_columns = list(X.columns)
        self.model = self._build_model()
        logger.info(
            f"Fitting TabPFN on {X.shape[0]} rows × {X.shape[1]} features"
        )
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self._fit_columns is not None and list(X.columns) != self._fit_columns:
            X = X[self._fit_columns]
        preds = self.model.predict(X)
        return np.asarray(preds).ravel()
