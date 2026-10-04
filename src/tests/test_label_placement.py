"""Regression tests for automatic language-label placement.

The figures these labels sit on used to be kept legible by hand-tuned per
-language (dx, dy) offsets, which went stale whenever the data or the distance
metric moved. These tests pin the property that replaced them: whatever the
inputs, no two labels overlap.
"""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.EyeScore.label_placement import place_language_labels

FONTSIZE, ROTATION, PAD_Y = 17, 30, 0.04


def _rotated_quad(txt, renderer):
    """True rotated box of a Text in display px (its window extent is the
    axis-aligned box of the rotated glyphs, which is far too loose at 30
    degrees to tell a real overlap from a near miss)."""
    rot = txt.get_rotation()
    txt.set_rotation(0)
    bb = txt.get_window_extent(renderer=renderer)
    txt.set_rotation(rot)
    w, h = bb.width, bb.height
    origin = txt.get_transform().transform(txt.get_position())
    th = np.deg2rad(rot)
    c, s = np.cos(th), np.sin(th)
    local = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    return local @ np.array([[c, s], [-s, c]]) + origin


def _sat_overlap(a, b):
    """Do two convex quads intersect? Separating-axis test."""
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            n = np.linalg.norm(axis)
            if n < 1e-12:
                continue
            axis = axis / n
            pa, pb = a @ axis, b @ axis
            if pa.max() <= pb.min() + 1e-6 or pb.max() <= pa.min() + 1e-6:
                return False
    return True


def _place(labels, xs, y_tops, xlim=(0.35, 0.85)):
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set_xlim(*xlim)
    ax.set_ylim(-1, 1)
    ax.set_box_aspect(1 / 1.5)
    fig.tight_layout()
    fig.canvas.draw()
    entries = [
        {"label": l, "x": x, "y_top": y,
         "points": np.array([[x, y]]), "color": (0.2, 0.3, 0.7)}
        for l, x, y in zip(labels, xs, y_tops)
    ]
    place_language_labels(ax, entries, fontsize=FONTSIZE, rotation=ROTATION,
                          pad_y=PAD_Y)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in ax.texts if t.get_text() in set(labels)]
    quads = [_rotated_quad(t, renderer) for t in texts]
    return fig, texts, quads


def _assert_disjoint(test, labels, quads, texts):
    for i in range(len(quads)):
        for j in range(i + 1, len(quads)):
            test.assertFalse(
                _sat_overlap(quads[i], quads[j]),
                f"{texts[i].get_text()!r} overlaps {texts[j].get_text()!r}",
            )


class TestNoOverlap(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_meco_like_spacing(self):
        """17 languages at MECO's real distances: the tightest pairs sit
        0.0024 apart, well inside one label's width."""
        labels = ["Norwegian", "Danish", "Icelandic", "German", "Dutch",
                  "Serbian", "Spanish", "Italian", "Estonian", "Finnish",
                  "Greek", "Russian", "Basque", "Hindi", "Turkish", "Hebrew",
                  "Mandarin"]
        xs = [0.3600, 0.3676, 0.3800, 0.3927, 0.4190, 0.4959, 0.5062, 0.5109,
              0.5400, 0.5639, 0.5698, 0.5763, 0.5792, 0.6207, 0.6237, 0.6935,
              0.7920]
        fig, texts, quads = _place(labels, xs, [0.6] * len(labels))
        self.assertEqual(len(texts), len(labels))
        _assert_disjoint(self, labels, quads, texts)

    def test_all_clusters_at_one_x(self):
        """Degenerate input: nothing can be solved by sliding sideways, so the
        solver has to fall back on stacking one label per lane."""
        labels = [f"Lang{i:02d}" for i in range(12)]
        fig, texts, quads = _place(labels, [0.6] * 12, [0.5] * 12)
        self.assertEqual(len(texts), 12)
        _assert_disjoint(self, labels, quads, texts)

    def test_cluster_top_above_axes(self):
        """y_top may sit outside the clamped y limits; the label must stay on
        the figure rather than following it off the top."""
        fig, texts, quads = _place(["Alpha", "Beta"], [0.45, 0.46], [5.0, 6.0])
        top = fig.canvas.get_width_height()[1]
        for q, t in zip(quads, texts):
            self.assertLess(q[:, 1].max(), top * 2,
                            f"{t.get_text()!r} flew off the figure")
        _assert_disjoint(self, ["Alpha", "Beta"], quads, texts)

    def test_single_and_empty(self):
        fig, texts, quads = _place(["Solo"], [0.5], [0.5])
        self.assertEqual(len(texts), 1)
        fig2, ax = plt.subplots()
        place_language_labels(ax, [], fontsize=FONTSIZE, rotation=ROTATION,
                              pad_y=PAD_Y)  # must not raise

    def test_deterministic(self):
        """Paper figures must be reproducible byte for byte."""
        labels = ["Greek", "Russian", "Basque", "Finnish", "Estonian"]
        xs = [0.5639, 0.5698, 0.5763, 0.5792, 0.5800]
        runs = []
        for _ in range(2):
            fig, texts, _ = _place(labels, xs, [0.5] * len(labels))
            runs.append({t.get_text(): t.get_position() for t in texts})
            plt.close(fig)
        self.assertEqual(runs[0], runs[1])


if __name__ == "__main__":
    unittest.main()
