"""Typological distance utilities (URIEL+).

Shared between EyeScore bias-calibration corrections (calculation.py) and
the language-bias plotting pipeline (evaluation/EyeScore/language_bias.py).
"""

import logging
import os
from functools import lru_cache
from typing import List

import numpy as np

# urielplus runs logging.basicConfig(level=INFO) at import, then logs timing and
# missing-feature notes at INFO straight to the root logger on every distance
# call. Undo the first and drop the second, leaving logging as it was for
# everything else.
_root = logging.getLogger()
_root_level, _root_handlers = _root.level, list(_root.handlers)
from urielplus import urielplus  # noqa: E402
_root.setLevel(_root_level)
for _h in [h for h in _root.handlers if h not in _root_handlers]:
    _root.removeHandler(_h)
_URIELPLUS_DIR = os.path.dirname(urielplus.__file__)
_root.addFilter(lambda r: r.levelno >= logging.WARNING or not r.pathname.startswith(_URIELPLUS_DIR))

logger = logging.getLogger(__name__)


ONESTOP_L1_TO_ISO = {
    "Chinese": "cmn",
    "Spanish": "spa",
    "Russian": "rus",
    "Portuguese": "por",
    "Hebrew": "heb",
    "Arabic": "ara",
    "Korean": "kor",
    "French": "fra",
    "Japanese": "jpn",
    "Vietnamese": "vie",
}

MECO_L1_TO_ISO = {
    "Dutch": "nld",
    "Estonian": "est",
    "Finnish": "fin",
    "German": "deu",
    "Greek": "ell",
    "Hebrew": "heb",
    "Italian": "ita",
    "Norwegian": "nor",
    "Russian": "rus",
    "Spanish": "spa",
    "Turkish": "tur",
    "Brazilian Portuguese": "por",
    "Mandarin": "cmn",
    "Danish": "dan",
    "Hindi": "hin",
    "Icelandic": "isl",
    "Serbian": "srp",
    "Basque": "eus",
}

L1_TO_ISO = {**ONESTOP_L1_TO_ISO, **MECO_L1_TO_ISO}


_uriel_instance = None


def _get_uriel() -> urielplus.URIELPlus:
    """Lazily initialize a single URIEL+ instance."""
    global _uriel_instance
    if _uriel_instance is None:
        _uriel_instance = urielplus.URIELPlus()
    return _uriel_instance


# Prefix marking a z-scored combination, e.g. "z:syntactic+genetic+scriptural".
# Without it the components are averaged raw, which is scale-dependent: URIEL+
# genetic and syntactic distances do not share a range, so a raw mean silently
# weights whichever component has the wider spread.
ZSCORE_PREFIX = "z:"

# Prefix marking a min-max-normalised combination, e.g.
# "mm:syntactic+genetic+scriptural". Like "z:" it stops the widest-spread
# component from dominating an unweighted mean (scriptural's sd is ~2.4x
# syntactic's, so a raw mean is mostly a script distance), but unlike z-scores
# the result stays bounded and directly readable: each component is mapped to
# [0, 1] over the SHARED reference language set, so 0 = the closest language to
# English in that set on that component, 1 = the farthest, and the average of
# the components is itself in [0, 1].
#
# Trade-off vs z: min-max is set by the two extreme languages, so one unusual
# language rescales the whole axis, whereas z uses mean/sd and is less hostage
# to a single point. Prefer "z:" for inference, "mm:" when a bounded,
# explainable number matters.
MINMAX_PREFIX = "mm:"

# Prefix marking a z-scored combination RESCALED to [0, 1], e.g.
# "z01:syntactic+genetic+scriptural".
#
# The component weighting is z's — mean/sd over the shared reference set, so all
# 25 languages set the scale, not just the two extremes as under "mm:". The
# [0, 1] mapping is applied to the COMPOSITE afterwards, and because that is a
# single positive linear transform it cannot change any regression inference:
# alpha scales inversely, fitted values, residuals and p-values are identical to
# "z:". So this is z's statistics wearing an interpretable scale, where 0 is the
# closest language to English in the reference set and 1 the farthest.
#
# Contrast with "mm:", which normalises each component BEFORE averaging: that
# reweights the components and is therefore a genuinely different regressor.
ZSCORE01_PREFIX = "z01:"

# Prefix marking a max-normalised combination, e.g.
# "max:syntactic+genetic+scriptural". Each component is divided by its own
# maximum over the shared reference set — positive and bounded in [0, 1] by
# construction (URIEL+ distances are non-negative), with NO centring and no
# post-hoc rescale of the composite.
#
# Note this equalises each component's CEILING, not its spread, so it is a
# genuinely different regressor from "z:"/"z01:" (r ~ 0.986), not a rescaling of
# them: a component whose values sit in a narrow band just below its max still
# contributes mostly a constant. Whether that matters for debiasing is an
# empirical question, not a derivation.
MAXNORM_PREFIX = "max:"

# Prefix marking a CONCATENATED-VECTOR distance, e.g. "cat:syntactic+scriptural".
#
# Every other distance type here asks URIEL+ for one distance PER component and
# then averages the numbers. This one instead concatenates the components' raw
# FEATURE VECTORS into a single vector per language and takes one angular
# distance over the whole thing — so the components are weighted by how many
# features each contributes (syntactic 103, scriptural 42), not equally, and a
# language's script and syntax are compared jointly rather than scored apart
# and blended.
#
# URIEL+ cannot do this itself: new_custom_distance builds its feature space as
# concatenate(feats[0], feats[2], feats[1]) — phylogeny, geography, typological
# — and never includes the scriptural array (feats[3]), so any "SC_*" feature
# is rejected as unknown. The construction below mirrors what URIEL+ does per
# language (source aggregation under self.aggregation, features restricted to
# those BOTH languages actually have, then _angular_distance) while spanning
# the typological and scriptural arrays.
CONCAT_PREFIX = "cat:"

# (URIEL+ data-array index, feature-name prefix) per component. The typological
# array holds syntactic/phonological/inventory side by side, distinguished only
# by the feature-name prefix; scriptural is its own array.
_CONCAT_COMPONENT_SPEC = {
    "syntactic": (1, "S_"),
    "phonological": (1, "P_"),
    "inventory": (1, "INV_"),
    "morphological": (1, "M_"),
    "scriptural": (3, "SC_"),
}

# Prefix marking a z-scored combination mapped to [0,1] with +-2 SD ANCHORS,
# e.g. "z2sd:syntactic+genetic+scriptural". Same component weighting as "z:";
# the composite is mapped by (z - (mu - 2*sd)) / (4*sd) where mu/sd are the
# composite's own moments over the shared reference set. Like "z01:" this is a
# positive affine map, so inference is identical to "z:" — but the endpoints are
# defined by the DISTRIBUTION rather than by whichever two languages happen to
# be extreme, so no language is pinned to exactly 0 or 1.
#
# Deliberately NOT clipped: clipping would break the affine property (and hence
# the inference-preservation) for any language beyond +-2 SD. A warning is logged
# instead if that happens.
ZSCORE2SD_PREFIX = "z2sd:"

# Prefix marking a SD-scaled combination, e.g. "sd:syntactic+genetic+scriptural".
# Each component is divided by its own SD over the shared reference set, with NO
# centring: positive by construction (URIEL+ distances are non-negative) and no
# post-hoc rescale. Algebraically this is "z:" plus a constant — centring only
# ever contributed a language-independent shift — so inference is identical to
# "z:". Unbounded above; use when a positive, untransformed scale matters more
# than a [0,1] range.
SDSCALE_PREFIX = "sd:"


# ---------------------------------------------------------------------------
# Scriptural proxy for Mandarin.
#
# DISCLAIMER — this is a substitution we chose, not a URIEL+ measurement.
#
# URIEL+ has no scriptural vector for `cmn` whatsoever: the scriptural table is
# keyed by Glottocode and Mandarin's (mand1415) is simply absent from it, while
# every sibling Sinitic variety is present. Asking for it makes urielplus log
# "No shared scriptural features between eng and cmn" and call sys.exit(1),
# which surfaced here as NaN. The practical effect was that Mandarin's
# "z:syntactic+genetic+scriptural" value was a mean over 2 of 3 components
# while every other language averaged 3 — an inconsistency that was invisible
# at the call site.
#
# We stand in Gan Chinese (`gan`), whose 40/42 script vector is byte-identical
# to Jin (`cjy`), Huizhou (`czh`), Literary (`lzh`) and Late Middle Chinese
# (`ltc`): Han characters, no romanization, no inter-word spacing, no case, no
# diacritics. That is an accurate description of Mandarin reading experience —
# Pinyin is an input method and a pedagogical annotation, not a reading medium.
#
# Known weaknesses, so nobody has to rediscover them:
#   * URIEL+ credits Cantonese and Wu with SC_ALPHABET (for Jyutping, which is
#     less used than Pinyin), which halves their distance to English: 0.39 vs
#     0.85. Applying our reasoning consistently would move them too, and we
#     have NOT done that — only Mandarin is patched here.
#   * The wider scriptural axis is weakly discriminating regardless: Hindi,
#     Japanese, Cantonese, Vietnamese and Russian all land at 0.39, and
#     Korean's SC_FEATURAL bit (Hangul) is unset while it is credited with
#     hanja.
# Prefer "syntactic+genetic" when a complete, judgment-free distance matters.
SCRIPTURAL_PROXY_ISO = {"cmn": "gan"}

_logged_scriptural_proxy = set()


def _effective_iso(component: str, iso: str) -> str:
    """ISO code to actually query URIEL+ with, applying SCRIPTURAL_PROXY_ISO.

    Scoped to the scriptural component on purpose: Mandarin's syntactic and
    genetic distances are present and correct, and must not be redirected.
    """
    if component != "scriptural":
        return iso
    proxy = SCRIPTURAL_PROXY_ISO.get(iso)
    if proxy is None:
        return iso
    if iso not in _logged_scriptural_proxy:
        _logged_scriptural_proxy.add(iso)
        logger.warning(
            "Scriptural distance for %r is a PROXY: using %r (URIEL+ has no "
            "scriptural vector for %r). See SCRIPTURAL_PROXY_ISO.", iso, proxy, iso,
        )
    return proxy


def _raw_component_distance(u, component: str, iso: str, use_proxy: bool = True) -> float:
    """Single URIEL+ component distance from English, NaN on failure.

    `use_proxy=False` reports URIEL+ verbatim, bypassing SCRIPTURAL_PROXY_ISO —
    used to show what the unpatched data actually contains.
    """
    if use_proxy:
        iso = _effective_iso(component, iso)
    try:
        return float(u.new_distance(component, ["eng", iso]))
    except (Exception, SystemExit) as e:
        logger.warning("Could not compute %s distance for %s: %s", component, iso, e)
        return np.nan


@lru_cache(maxsize=None)
def _zscore_reference_stats(component: str) -> tuple:
    """(mean, sd) of a component's distance over the SHARED reference language
    set — every language in L1_TO_ISO, i.e. OneStop ∪ MECO.

    Deliberately not the caller's language subset: z-scoring within each
    dataset's own languages would put OneStop and MECO on different scales and
    make their debiased numbers incomparable. Fixing the reference set means a
    given language has the same distance in both datasets.
    """
    u = _get_uriel()
    vals = [_raw_component_distance(u, component, iso) for iso in sorted(set(L1_TO_ISO.values()))]
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        logger.warning("No finite %s distances over the reference set; z-scoring disabled.", component)
        return (0.0, 1.0)
    mean = float(finite.mean())
    sd = float(finite.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        logger.warning("Zero/non-finite sd for %s over the reference set; using sd=1.", component)
        sd = 1.0
    logger.info("z-score reference for %s: mean=%.4f sd=%.4f over %d languages",
                component, mean, sd, finite.size)
    return (mean, sd)


@lru_cache(maxsize=None)
def _minmax_reference_stats(component: str) -> tuple:
    """(min, max) of a component's distance over the SHARED reference language
    set (every language in L1_TO_ISO, i.e. OneStop u MECO).

    Same fixed-reference rule as _zscore_reference_stats: normalising within
    each dataset's own languages would put OneStop and MECO on different scales
    and make their debiased numbers incomparable.
    """
    u = _get_uriel()
    vals = [_raw_component_distance(u, component, iso)
            for iso in sorted(set(L1_TO_ISO.values()))]
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        logger.warning("No finite %s distances over the reference set; min-max disabled.", component)
        return (0.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    if not np.isfinite(hi - lo) or hi <= lo:
        logger.warning("Degenerate %s range over the reference set; min-max disabled.", component)
        return (lo, lo + 1.0)
    logger.info("min-max reference for %s: [%.4f, %.4f] over %d languages",
                component, lo, hi, finite.size)
    return (lo, hi)


@lru_cache(maxsize=None)
def _maxnorm_reference_max(component: str) -> float:
    """Max of a component's distance over the SHARED reference language set."""
    u = _get_uriel()
    vals = [_raw_component_distance(u, component, iso)
            for iso in sorted(set(L1_TO_ISO.values()))]
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or finite.max() <= 0:
        logger.warning("No usable %s max over the reference set; max-norm disabled.", component)
        return 1.0
    return float(finite.max())


@lru_cache(maxsize=None)
def _zcomposite_reference_values(components: tuple) -> np.ndarray:
    """The z-scored COMPOSITE of every reference-set language that has at least
    one resolvable component (the "z:" mean over those components)."""
    u = _get_uriel()
    stats = {c: _zscore_reference_stats(c) for c in components}
    vals = []
    for iso in sorted(set(L1_TO_ISO.values())):
        zs = []
        for c in components:
            raw = _raw_component_distance(u, c, iso)
            mean, sd = stats[c]
            zs.append((raw - mean) / sd)
        arr = np.asarray(zs, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(float(finite.mean()))
    return np.asarray(vals, dtype=float)


@lru_cache(maxsize=None)
def _zcomposite_moments(components: tuple) -> tuple:
    """(mean, sd) of the z-scored COMPOSITE over the shared reference set."""
    arr = _zcomposite_reference_values(components)
    if arr.size == 0:
        return (0.0, 1.0)
    mu, sd = float(arr.mean()), float(arr.std(ddof=0))
    return (mu, sd if np.isfinite(sd) and sd > 0 else 1.0)


@lru_cache(maxsize=None)
def _z01_reference_span(components: tuple) -> tuple:
    """(min, max) of the z-scored COMPOSITE over the shared reference set.

    Applied after averaging, so it is one positive linear transform of the "z:"
    composite — inference-preserving by construction (see ZSCORE01_PREFIX).
    """
    arr = _zcomposite_reference_values(components)
    if arr.size == 0:
        return (0.0, 1.0)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return (lo, lo + 1.0)
    logger.info("z01 composite span over the reference set: [%.4f, %.4f]", lo, hi)
    return (lo, hi)


def _concat_component_block(u, comp: str, iso: str):
    """(values, known_mask) for one component's feature block for `iso`.

    Mirrors URIEL+'s _process_custom_language aggregation: a feature is KNOWN
    when any source reports something other than -1, and its value is the union
    ('U': 1.0 if any source says 1.0) or the mean of the known sources ('A').
    """
    arr_idx, prefix = _CONCAT_COMPONENT_SPEC[comp]
    data, langs, feats = u.data[arr_idx], u.langs[arr_idx], u.feats[arr_idx]
    # Same SCRIPTURAL_PROXY_ISO redirection the per-component path uses, so
    # Mandarin's script block is not silently all-missing here either.
    iso = _effective_iso(comp, iso)
    where = np.where(langs == iso)[0]
    sel = [i for i, f in enumerate(feats) if str(f).startswith(prefix)]
    if where.size == 0:
        return np.zeros(len(sel)), np.zeros(len(sel), dtype=bool)
    block = np.asarray(data[where[0]], dtype=float)[sel]      # (n_feats, n_sources)
    known = np.any(block != -1.0, axis=1)
    if u.aggregation == "U":
        vals = np.where(np.any(block == 1.0, axis=1), 1.0, 0.0)
    else:
        masked = np.where(block == -1.0, np.nan, block)
        with np.errstate(invalid="ignore"):
            vals = np.nanmean(masked, axis=1)
        vals = np.nan_to_num(vals, nan=0.0)
    return vals, known


def _concat_vector_distance(u, components: tuple, iso: str,
                            base_iso: str = "eng") -> float:
    """Angular distance over the components' concatenated feature vectors."""
    vecs_a, vecs_b = [], []
    for comp in components:
        if comp not in _CONCAT_COMPONENT_SPEC:
            logger.warning("cat: unsupported component %r; skipping", comp)
            continue
        a_vals, a_known = _concat_component_block(u, comp, base_iso)
        b_vals, b_known = _concat_component_block(u, comp, iso)
        # Only features BOTH languages have — URIEL+'s shared-index rule. A
        # feature one side is missing would otherwise read as a real 0.0 and
        # count as agreement.
        shared = a_known & b_known
        vecs_a.append(a_vals[shared])
        vecs_b.append(b_vals[shared])
    if not vecs_a:
        return np.nan
    a, b = np.concatenate(vecs_a), np.concatenate(vecs_b)
    if a.size == 0:
        return np.nan
    return float(u._angular_distance(a, b))


@lru_cache(maxsize=None)
def _cached_distances(
    languages_key: tuple,
    distance_type: str,
) -> dict:
    """URIEL+ lookups are stable for a fixed (language set, distance_type), so
    cache by a tuple key. Caller normalizes the language list to a sorted tuple
    to maximize cache hits across feature_sets within a dataset.

    `distance_type` is "+"-joined component names, optionally behind one of the
    prefixes defined above (cat:, z:, z01:, z2sd:, sd:, mm:, max:); without a
    prefix the components are averaged raw.
    """
    u = _get_uriel()
    concat = distance_type.startswith(CONCAT_PREFIX)
    if concat:
        # One vector, one distance — no per-component numbers to combine, so
        # this returns before any of the averaging/normalising paths below.
        components = tuple(distance_type[len(CONCAT_PREFIX):].split("+"))
        out = {}
        for lang in languages_key:
            iso = L1_TO_ISO.get(lang)
            out[lang] = np.nan if iso is None else _concat_vector_distance(u, components, iso)
        finite = [v for v in out.values() if np.isfinite(v)]
        if finite:
            logger.info("cat:%s over %d languages: range %.4f..%.4f",
                        "+".join(components), len(finite), min(finite), max(finite))
        return out

    maxnorm = distance_type.startswith(MAXNORM_PREFIX)
    z01 = distance_type.startswith(ZSCORE01_PREFIX)
    z2sd = distance_type.startswith(ZSCORE2SD_PREFIX)
    sdscale = distance_type.startswith(SDSCALE_PREFIX)
    zscored = distance_type.startswith(ZSCORE_PREFIX) and not (z01 or z2sd)
    minmaxed = distance_type.startswith(MINMAX_PREFIX)
    if maxnorm:
        spec = distance_type[len(MAXNORM_PREFIX):]
    elif z2sd:
        spec = distance_type[len(ZSCORE2SD_PREFIX):]
    elif sdscale:
        spec = distance_type[len(SDSCALE_PREFIX):]
    elif z01:
        spec = distance_type[len(ZSCORE01_PREFIX):]
    elif zscored:
        spec = distance_type[len(ZSCORE_PREFIX):]
    elif minmaxed:
        spec = distance_type[len(MINMAX_PREFIX):]
    else:
        spec = distance_type
    components = spec.split("+")

    iso_codes = {}
    for lang in languages_key:
        iso = L1_TO_ISO.get(lang)
        if iso is not None:
            iso_codes[lang] = iso

    if not iso_codes:
        return {}

    if z2sd or sdscale:
        stats = {c: _zscore_reference_stats(c) for c in components}
        if z2sd:
            mu, sd_c = _zcomposite_moments(tuple(components))
            lo, span = mu - 2.0 * sd_c, 4.0 * sd_c
        distances = {}
        for lang, iso in iso_codes.items():
            vals = []
            for c in components:
                raw = _raw_component_distance(u, c, iso)
                mean, sd = stats[c]
                vals.append(raw / sd if sdscale else (raw - mean) / sd)
            arr = np.asarray(vals, dtype=float)
            finite = arr[np.isfinite(arr)]
            if not finite.size:
                distances[lang] = np.nan
                continue
            v = float(finite.mean())
            distances[lang] = v if sdscale else (v - lo) / span
        if z2sd:
            out = [v for v in distances.values() if np.isfinite(v)]
            if out and (min(out) < 0 or max(out) > 1):
                logger.warning("z2sd: %d language(s) fall outside [0,1] (range %.3f..%.3f); "
                               "left unclipped to keep the map affine.",
                               sum(1 for v in out if v < 0 or v > 1), min(out), max(out))
        return distances

    if maxnorm:
        mx = {c: _maxnorm_reference_max(c) for c in components}
        distances = {}
        for lang, iso in iso_codes.items():
            vs = [_raw_component_distance(u, c, iso) / mx[c] for c in components]
            arr = np.asarray(vs, dtype=float)
            finite = arr[np.isfinite(arr)]
            distances[lang] = float(np.clip(finite.mean(), 0.0, 1.0)) if finite.size else np.nan
        return distances

    if z01:
        stats = {c: _zscore_reference_stats(c) for c in components}
        lo, hi = _z01_reference_span(tuple(components))
        distances = {}
        for lang, iso in iso_codes.items():
            zs = []
            for c in components:
                raw = _raw_component_distance(u, c, iso)
                mean, sd = stats[c]
                zs.append((raw - mean) / sd)
            arr = np.asarray(zs, dtype=float)
            finite = arr[np.isfinite(arr)]
            if not finite.size:
                distances[lang] = np.nan
                continue
            z = float(finite.mean())
            distances[lang] = float(np.clip((z - lo) / (hi - lo), 0.0, 1.0))
        return distances

    if zscored:
        stats = {c: _zscore_reference_stats(c) for c in components}
        distances = {}
        for lang, iso in iso_codes.items():
            zs = []
            for c in components:
                raw = _raw_component_distance(u, c, iso)
                mean, sd = stats[c]
                zs.append((raw - mean) / sd)
            arr = np.asarray(zs, dtype=float)
            finite = arr[np.isfinite(arr)]
            # Mean over the components that resolved, matching the raw path's
            # tolerance for partial URIEL+ coverage.
            distances[lang] = float(finite.mean()) if finite.size else np.nan
        return distances

    if minmaxed:
        stats = {c: _minmax_reference_stats(c) for c in components}
        distances = {}
        for lang, iso in iso_codes.items():
            ms = []
            for c in components:
                raw = _raw_component_distance(u, c, iso)
                lo, hi = stats[c]
                ms.append((raw - lo) / (hi - lo))
            arr = np.asarray(ms, dtype=float)
            finite = arr[np.isfinite(arr)]
            # Mean over the components that resolved, matching the raw path.
            # Clipped because a language outside the reference set's range would
            # otherwise escape [0, 1] and break the interpretation.
            distances[lang] = float(np.clip(finite.mean(), 0.0, 1.0)) if finite.size else np.nan
        return distances

    distances = {}
    needs_proxy = "scriptural" in components
    for lang, iso in iso_codes.items():
        try:
            if len(components) == 1:
                distances[lang] = u.new_distance(
                    components[0], ["eng", _effective_iso(components[0], iso)],
                )
            elif needs_proxy and iso in SCRIPTURAL_PROXY_ISO:
                # A single URIEL+ call takes one ISO code for every component,
                # so a per-component proxy has to be resolved component-wise.
                # Only reachable for a proxied language in a scriptural combo,
                # which URIEL+ would otherwise fail outright — so this cannot
                # perturb "syntactic+genetic" or any unproxied language.
                vals = [_raw_component_distance(u, c, iso) for c in components]
                distances[lang] = float(np.mean(vals))
            else:
                vals = u.new_distance(components, ["eng", iso])
                distances[lang] = float(np.mean(vals))
        except (Exception, SystemExit) as e:
            logger.warning("Could not compute %s distance for %s (%s): %s",
                           distance_type, lang, iso, e)
            distances[lang] = np.nan
    return distances


def compute_typological_distances(
    languages: List[str],
    distance_type: str = "syntactic",
) -> dict:
    """Return {L1_label: distance_to_English} for each language.

    Uses URIEL+ precomputed distances. Languages without a known ISO mapping
    or missing URIEL+ coverage are assigned NaN.

    distance_type may be a single URIEL+ distance name (e.g. "syntactic") or
    a "+"-joined combination (e.g. "syntactic+genetic"), in which case the
    component distances are averaged — raw, or normalised first under one of
    the prefixes defined above (e.g. "z:syntactic+genetic"). The "cat:" prefix
    instead concatenates the components' feature vectors and takes one angular
    distance (e.g. "cat:syntactic+phonological+scriptural", the paper's).

    Memoized by `(sorted languages, distance_type)` — the language_bias
    pipeline calls this once per (entry × target_col × distance_type), so
    caching at this level cuts URIEL+ traffic by a factor of ~thousands.
    Caller should not mutate the returned dict (it's shared across hits).
    """
    languages_key = tuple(sorted(set(languages)))
    return _cached_distances(languages_key, distance_type)
