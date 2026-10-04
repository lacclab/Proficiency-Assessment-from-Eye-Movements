"""Collision-free placement of the per-language labels on the paper figures.

`_render_residual_vs_distance_on_ax` labels each L1 cluster in place rather
than with a legend, which reads well until two languages land on top of each
other -- and in MECO eight clusters share about 96px of x, so at the figure's
label size they always do. Hand-tuned per-language nudges would need
re-tuning whenever the data or the distance metric moved, so the layout is
solved instead. Every label shares one rotation, so rotating
display space by -rotation turns each into an AXIS-ALIGNED rectangle, and
overlap becomes plain rectangle intersection. Each label then gets a finite
menu of placements built from two geometric knobs:

  lane   integer steps perpendicular to the text, one text-height each. Labels
         in different lanes provably cannot touch, however much their x
         positions coincide.
  align  how far the text slides along its own baseline: 0 starts it at the
         cluster (the historical look), 1 ends it there so it grows up-and-
         LEFT, negatives push it further right. This lets a crowded label
         escape sideways instead of climbing into the whitespace above.

A deterministic local search over those menus minimises a penalty that trades
overlap against how far a label strays from its cluster -- W_GAP keeps every
label sitting on the cluster it names, so the figure needs no leader lines to
explain itself. Nothing here reads a language
name, a dataset or a y variable -- the only inputs are measured pixel geometry,
so it re-solves itself when the data changes.
"""
import numpy as np

# ── tunables ───────────────────────────────────────────────────────────────
# All of these are geometry-derived and dataset-independent.  Distances are in
# display pixels so they stay meaningful whatever the data range happens to be.
LANE_PAD_PX = 1.5          # breathing room between two stacked lanes
# Fraction of its own width the label is slid along its baseline: 0 keeps the
# text starting at the cluster (the historical look), 1 makes it *end* there,
# and negative values push it forward, up-and-right, which is the only escape
# route left for a label pinned against the left edge of the axes.
ALIGN_STEPS = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
LANES_BELOW = 1            # how far a label may drop below its cluster top
MIN_LANES_ABOVE = 3        # menu depth even when the labels are far apart
MAX_LANES_ABOVE = 8        # floor on menu depth; raised to len(entries)
                           # when that many lanes are needed (see below)
OBSTACLE_PAD_PX = 7.0      # halo around pre-existing text (the r/p box)
FURNITURE_PAD_PX = 12.0    # margin held between a label and the panel edges:
                           # the y-axis spine and tick labels on the left, the
                           # neighbouring panel on the right. At 3px a label
                           # pinned to the leftmost cluster came out printed
                           # against the spine, reading as part of the axis.
SEPARATION_PX = 1.0        # inflate rects so "solved" means visibly apart

W_LABEL = 200.0            # px^-2 -- label/label overlap dominates everything
W_OBSTACLE = 200.0         # px^-2 -- ditto for the r/p annotation box
W_POINT = 40.0             # per scatter point hidden under a label
W_FURNITURE = 20.0         # px^-2 -- label area straying into the axis furniture
W_GAP = 0.55               # per px between the cluster anchor and the label
W_ALIGN = 0.28             # per px the text is slid back along its baseline
W_ABOVE = 0.60             # per px the tallest label rises above the axes

MAX_SWEEPS = 16
MAX_REPAIRS = 4


# ── small geometry helpers ─────────────────────────────────────────────────
def _text_size_px(txt, renderer):
    """Unrotated width/height of a Text in display px."""
    rot = txt.get_rotation()
    txt.set_rotation(0)
    bb = txt.get_window_extent(renderer=renderer)
    txt.set_rotation(rot)
    return bb.width, bb.height


def _corners(pos_px, w, h, cos_t, sin_t):
    """The four display-space corners of a rotated, anchor-mode Text box."""
    local = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return local @ rot.T + np.asarray(pos_px)


def _clip_convex(subject, clipper):
    """Sutherland-Hodgman intersection of two convex CCW polygons."""
    out = [np.asarray(p, dtype=float) for p in subject]
    for i in range(len(clipper)):
        if not out:
            return []
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = b - a
        prev, out = out, []
        for j in range(len(prev)):
            cur, pre = prev[j], prev[j - 1]
            cur_in = edge[0] * (cur[1] - a[1]) - edge[1] * (cur[0] - a[0]) >= 0
            pre_in = edge[0] * (pre[1] - a[1]) - edge[1] * (pre[0] - a[0]) >= 0
            if cur_in != pre_in:
                d = cur - pre
                den = edge[0] * d[1] - edge[1] * d[0]
                if abs(den) > 1e-12:
                    t = (edge[0] * (a[1] - pre[1]) - edge[1] * (a[0] - pre[0])) / den
                    out.append(pre + t * d)
            if cur_in:
                out.append(cur)
    return out


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1))
                     - np.dot(p[:, 1], np.roll(p[:, 0], 1)))


def _ccw(poly):
    p = np.asarray(poly, dtype=float)
    s = np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))
    return p if s > 0 else p[::-1]


def _overlap_area(a, b):
    return _poly_area(_clip_convex(list(_ccw(a)), list(_ccw(b))))


def _area_outside(quad, x_min, y_min, x_max):
    """Label area that leaves the panel: left of / below the axes, i.e. onto the
    tick labels and axis titles, or past its right edge.  Neither region is a
    Text child of the axes, so the obstacle list cannot see them; the axes
    rectangle stands in for them.  The right edge matters because a grid panel
    is not the last thing drawn -- the neighbouring axes paints its own
    background over anything that spills across, truncating the name mid-word
    ("Mandarin" -> "Mand"), and the search has an escape route for it: slide the
    text back along its baseline so it grows up-and-LEFT instead."""
    big = 1e6
    keep = np.array([[x_min, y_min], [x_max, y_min], [x_max, big], [x_min, big]])
    return max(0.0, _poly_area(_ccw(quad)) - _overlap_area(quad, keep))


def _pair_overlaps(u0, v0, w, h):
    """Upper-triangular pairwise rect-intersection areas, vectorised."""
    du = np.minimum(u0[:, None] + w[:, None], u0[None, :] + w[None, :]) \
        - np.maximum(u0[:, None], u0[None, :])
    dv = np.minimum(v0[:, None] + h, v0[None, :] + h) \
        - np.maximum(v0[:, None], v0[None, :])
    area = np.clip(du, 0, None) * np.clip(dv, 0, None)
    return np.triu(area, 1)


# ── candidate menu ─────────────────────────────────────────────────────────
def _build_menu(anchors_px, widths, height, lanes_above, dir_v, nrm_v,
                lanes_below=LANES_BELOW):
    """Menu of (lane, align) placements for every label.

    Returns arrays indexed [label, candidate]: the rotated-frame rect origin,
    the display-space position to hand to Text.set_position, plus the two
    "how far did we move it" costs.
    """
    lanes = list(range(-lanes_below, lanes_above + 1))
    pitch = height + LANE_PAD_PX
    n, c = len(widths), len(lanes) * len(ALIGN_STEPS)
    pos = np.empty((n, c, 2))
    gap = np.empty((n, c))
    slide = np.empty((n, c))
    order = sorted(((abs(lane), abs(align), lane, align)
                    for lane in lanes for align in ALIGN_STEPS))
    for i, a_px in enumerate(anchors_px):
        for k, (_, _, lane, align) in enumerate(order):
            back = align * widths[i]
            pos[i, k] = a_px - back * dir_v + lane * pitch * nrm_v
            # Distance from the cluster anchor to the label box, measured in
            # the rotated frame where the box is axis aligned.  For 0<=align<=1
            # the anchor sits on the baseline, so only the lane offset counts.
            gap[i, k] = float(np.hypot(max(0.0, -back), abs(lane) * pitch))
            slide[i, k] = abs(back)
    return pos, gap, slide


# ── search ─────────────────────────────────────────────────────────────────
def _static_cost(rp_area, outside, pts, gap, slide):
    return (W_OBSTACLE * rp_area + W_FURNITURE * outside + W_POINT * pts
            + W_GAP * gap + W_ALIGN * slide)


def _total(choice, u0, v0, w_pad, h_pad, static, above):
    idx = (np.arange(len(choice)), choice)
    pair = _pair_overlaps(u0[idx], v0[idx], w_pad, h_pad).sum()
    return (W_LABEL * pair + float(static[idx].sum())
            + W_ABOVE * float(np.max(above[idx])))


def _solve(u0, v0, w_pad, h_pad, static, above, seed):
    """Ordered sweeps to a fixed point, then joint repair of bad pairs."""
    n, c = static.shape
    choice = seed.copy()
    if n == 1:
        return np.array([int(np.argmin(static[0] + W_ABOVE * above[0]))])

    for sweep in range(MAX_SWEEPS):
        changed = False
        order = range(n) if sweep % 2 == 0 else range(n - 1, -1, -1)
        for i in order:
            others = np.array([j for j in range(n) if j != i])
            oj = choice[others]
            ou, ov, ow = u0[others, oj], v0[others, oj], w_pad[others]
            # Overlap of each of label i's candidates against the fixed rest.
            du = np.minimum(u0[i][:, None] + w_pad[i], ou[None, :] + ow[None, :]) \
                - np.maximum(u0[i][:, None], ou[None, :])
            dv = np.minimum(v0[i][:, None] + h_pad, ov[None, :] + h_pad) \
                - np.maximum(v0[i][:, None], ov[None, :])
            ov_area = (np.clip(du, 0, None) * np.clip(dv, 0, None)).sum(axis=1)
            rest_above = float(np.max(above[others, oj]))
            cost = (W_LABEL * ov_area + static[i]
                    + W_ABOVE * np.maximum(above[i], rest_above))
            best = int(np.argmin(cost))
            if cost[best] < cost[choice[i]] - 1e-9:
                choice[i] = best
                changed = True
        if not changed:
            break

    # Sweeps move one label at a time, so two labels that can only be separated
    # by moving *both* stay stuck.  Re-optimise such pairs jointly.
    for _ in range(MAX_REPAIRS):
        idx = (np.arange(n), choice)
        pair = _pair_overlaps(u0[idx], v0[idx], w_pad, h_pad)
        bad = np.argwhere(pair > 0.0)
        if not len(bad):
            break
        improved = False
        for i, j in bad:
            base = _total(choice, u0, v0, w_pad, h_pad, static, above)
            trial = choice.copy()
            best, best_cost = None, base
            for ci in range(c):
                trial[i] = ci
                for cj in range(c):
                    trial[j] = cj
                    t = _total(trial, u0, v0, w_pad, h_pad, static, above)
                    if t < best_cost - 1e-9:
                        best, best_cost = (ci, cj), t
            trial[i], trial[j] = choice[i], choice[j]
            if best is not None:
                choice[i], choice[j] = best
                improved = True
        if not improved:
            break
    return choice


def _visible_top(entry, y_lo, y_hi):
    """Highest y of the entry's points that falls inside the axes window."""
    pts = np.asarray(entry.get("points"), dtype=float)
    if pts.size:
        ys = pts[:, 1]
        inside = ys[(ys >= y_lo) & (ys <= y_hi)]
        if inside.size:
            return float(inside.max())
    return float(entry["y_top"])


# ── entry point ────────────────────────────────────────────────────────────
def place_language_labels(ax, entries, fontsize, rotation, pad_y,
                          lanes_below=LANES_BELOW, above_weight=W_ABOVE):
    """Draw one label per entry, solved so that none of them overlap.

    Call this AFTER the figure has been laid out (tight_layout + a draw):
    the solver works in display pixels, and a later layout pass would
    resize the axes out from under the solution.

    entries: dicts with `label`, `x`, `y_top` (data coords of the cluster's
    highest point), `points` (Nx2 data coords, used only to prefer
    placements that hide fewer of them) and `color`.

    `lanes_below` and `above_weight` let a caller trade band height against how
    tightly labels track their clusters: more lanes below, and a heavier price
    on climbing over the panel top, keep the band low. A figure that then fits
    its y window to that band ends up with far less empty axis above the data.
    """
    if not entries:
        return
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    rot = float(rotation)
    theta = np.deg2rad(rot)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dir_v = np.array([cos_t, sin_t])          # along the text baseline
    nrm_v = np.array([-sin_t, cos_t])         # perpendicular: one lane step

    y_lo, y_hi = ax.get_ylim()
    # Anything else already written on the axes (the Pearson r/p box, the
    # hand-drawn x tick labels) is an obstacle we must not land on.
    obstacles = []
    for t in ax.texts:
        bb = t.get_window_extent(renderer=renderer)
        p = OBSTACLE_PAD_PX
        obstacles.append(np.array([[bb.x0 - p, bb.y0 - p], [bb.x1 + p, bb.y0 - p],
                                   [bb.x1 + p, bb.y1 + p], [bb.x0 - p, bb.y1 + p]]))

    # 1. Create one Text per entry at its natural anchor and measure it.
    #    The axes y limits are clamped tighter than the data, so a cluster top
    #    can sit off-screen. Anchoring to the highest point that is actually
    #    DRAWN, not to the cluster's raw maximum, is what keeps the label
    #    sitting on its own circles: a single clipped outlier would otherwise
    #    pin the label to the top of the axes, far above anything visible.
    #    (A cluster hidden entirely falls back to its raw top, then clamps.)
    texts, anchors_px, widths = [], [], []
    for e in entries:
        y_anchor = min(max(_visible_top(e, y_lo, y_hi) + pad_y, y_lo), y_hi)
        t = ax.text(e["x"], y_anchor, e["label"], ha="left", va="bottom",
                    fontsize=fontsize, color=e["color"], rotation=rot,
                    rotation_mode="anchor", clip_on=False, zorder=6)
        texts.append(t)
        anchors_px.append(ax.transData.transform((e["x"], y_anchor)))
        widths.append(_text_size_px(t, renderer)[0])
    anchors_px = np.asarray(anchors_px, dtype=float)
    widths = np.asarray(widths, dtype=float)
    height = _text_size_px(texts[0], renderer)[1]

    # 2. Menu depth follows crowding: total label length over the span the
    #    labels have to share tells us how many lanes are unavoidable.
    u_anchor = anchors_px @ dir_v
    span = float(u_anchor.max() - u_anchor.min())
    # Never offer fewer lanes than labels: one label per lane is always a
    # feasible layout (lanes are a full text-height apart, so same-lane is the
    # only way two labels can touch), which keeps the search solvable even when
    # every cluster sits at the same x and no amount of sliding can help.
    lane_cap = max(MAX_LANES_ABOVE, len(widths))
    need = int(np.ceil(widths.sum() / span)) if span > 1.0 else lane_cap
    lanes_above = int(np.clip(need + 1, MIN_LANES_ABOVE, lane_cap))
    pos, gap, slide = _build_menu(anchors_px, widths, height,
                                  lanes_above, dir_v, nrm_v, lanes_below)
    n, c = gap.shape

    # 3. Score every candidate against the things that do not move.
    all_pts = np.vstack([e["points"] for e in entries])
    pts_px = ax.transData.transform(all_pts)
    pts_u, pts_v = pts_px @ dir_v, pts_px @ nrm_v
    axes_top = ax.transAxes.transform((0.0, 1.0))[1]

    u0 = pos.reshape(-1, 2) @ dir_v
    v0 = pos.reshape(-1, 2) @ nrm_v
    u0 = u0.reshape(n, c)
    v0 = v0.reshape(n, c)

    axes_x0, axes_y0 = ax.transAxes.transform((0.0, 0.0))
    axes_x0 += FURNITURE_PAD_PX
    axes_y0 += FURNITURE_PAD_PX
    axes_x1 = ax.transAxes.transform((1.0, 0.0))[0] - FURNITURE_PAD_PX

    rp_area = np.zeros((n, c))
    outside = np.zeros((n, c))
    n_pts = np.zeros((n, c))
    above = np.zeros((n, c))
    for i in range(n):
        for j in range(c):
            quad = _corners(pos[i, j], widths[i], height, cos_t, sin_t)
            # Scaled, so the solver's fixed `W_ABOVE * above` term prices this
            # caller's weight. `above` is only ever a cost input, never a
            # distance that is read back.
            above[i, j] = (max(0.0, quad[:, 1].max() - axes_top)
                           * (above_weight / W_ABOVE))
            rp_area[i, j] = sum(_overlap_area(quad, ob) for ob in obstacles)
            outside[i, j] = _area_outside(quad, axes_x0, axes_y0, axes_x1)
            inside = ((pts_u >= u0[i, j]) & (pts_u <= u0[i, j] + widths[i])
                      & (pts_v >= v0[i, j]) & (pts_v <= v0[i, j] + height))
            n_pts[i, j] = inside.sum()
    static = _static_cost(rp_area, outside, n_pts, gap, slide)

    # Solve on slightly inflated rects so a "zero overlap" answer really is
    # zero once matplotlib rounds the glyphs to pixels.
    half = 0.5 * SEPARATION_PX
    w_pad = widths + SEPARATION_PX
    h_pad = height + SEPARATION_PX
    u_pad, v_pad = u0 - half, v0 - half

    # 4. Two deterministic starting points: the natural layout, and a greedy
    #    left-to-right pass (which is what an interval-colouring heuristic
    #    would produce).  Keep whichever local optimum scores better.
    natural = np.zeros(n, dtype=int)          # candidate 0 == lane 0, align 0
    greedy = _greedy_seed(u_pad, v_pad, w_pad, h_pad, static, above, u_anchor)
    choice, best_cost = None, np.inf
    for seed in (natural, greedy):
        sol = _solve(u_pad, v_pad, w_pad, h_pad, static, above, seed)
        cost = _total(sol, u_pad, v_pad, w_pad, h_pad, static, above)
        if cost < best_cost - 1e-9:
            choice, best_cost = sol, cost

    # 5. Commit, and connect anything that had to travel back to its cluster.
    inv = ax.transData.inverted()
    for i, t in enumerate(texts):
        t.set_position(inv.transform(pos[i, choice[i]]))


def _greedy_seed(u0, v0, w_pad, h_pad, static, above, u_anchor):
    """Place labels left-to-right, each into its cheapest free slot."""
    n, c = static.shape
    seed = np.zeros(n, dtype=int)
    done = []
    for i in np.argsort(u_anchor, kind="stable"):
        if not done:
            seed[i] = int(np.argmin(static[i] + W_ABOVE * above[i]))
            done.append(i)
            continue
        d = np.array(done)
        ou, ov, ow = u0[d, seed[d]], v0[d, seed[d]], w_pad[d]
        du = np.minimum(u0[i][:, None] + w_pad[i], ou[None, :] + ow[None, :]) \
            - np.maximum(u0[i][:, None], ou[None, :])
        dv = np.minimum(v0[i][:, None] + h_pad, ov[None, :] + h_pad) \
            - np.maximum(v0[i][:, None], ov[None, :])
        ov_area = (np.clip(du, 0, None) * np.clip(dv, 0, None)).sum(axis=1)
        seed[i] = int(np.argmin(W_LABEL * ov_area + static[i] + W_ABOVE * above[i]))
        done.append(i)
    return seed
