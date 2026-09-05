"""Pipeline schematic (Figure 1a) and the decoding figure (supplementary S1).

    python3 sim/make_paper_fig_pipeline.py [--refresh]

    docs/paper/figs/paper2/fig0_pipeline.{png,pdf}   schematic, no data
    figs/readme/fig0_pipeline.png                    copy for the README
    docs/paper/figs/paper2/figS1_decoding.{png,pdf}  real held-out decoding (supplementary)
    docs/paper/figs/paper2/figS1_decoding_scores.csv cache for the ROC panel

fig0 is a schematic that uses no data. It prints no result numbers, so re-running it
changes nothing.

The decoding figure recomputes the nested-LOIO procedure of
`analysis/rq3/analyze_rq3_context_decoding_robustness.py` from the canonical activations
(`results/rq3_pcr_locked_20260825/activations`). All 12 cells were checked to match the
`conjunction_h3` rows of `analysis/rq3a_context_decoder_robustness.csv` to four decimals.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from sklearn.metrics import roc_auc_score, roc_curve

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "docs" / "paper" / "figs" / "paper2"
README_FIGS = PROJ / "figs" / "readme"   # fig0 is also used by the README
LOCK = PROJ / "results" / "rq3_pcr_locked_20260825"
ACT = LOCK / "activations"
ANA = LOCK / "analysis"
CACHE = OUT / "figS1_decoding_scores.csv"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "axes.unicode_minus": False,
    "figure.dpi": 200, "savefig.dpi": 400,
    "axes.grid": True, "grid.alpha": .20, "grid.linewidth": .5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
})

REL = "#3B6EA5"     # relevant context, and the target direction
IRR = "#9E9E9E"     # irrelevant context, and the controls
EVI = "#B07C2E"     # the evidential plane, rho and C
EDIT = "#8E5572"    # the two edits
INK = "#333333"
FAINT = "#F1F3F7"

MODELS = {
    "Qwen__Qwen2.5-Math-1.5B": "Qwen2.5-Math",
    "Qwen__Qwen3-4B": "Qwen3-4B",
    "Qwen__Qwen3-4B-Instruct-2507": "Qwen3-Instruct",
    "google__gemma-3-4b-it": "Gemma-3-4B",
    "microsoft__Phi-4-mini-instruct": "Phi-Instruct",
    "microsoft__Phi-4-mini-reasoning": "Phi-Reasoning",
}
LONG_TO_SHORT = {
    "Qwen2.5-Math-1.5B": "Qwen2.5-Math",
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-4B-Instruct-2507": "Qwen3-Instruct",
    "gemma-3-4b-it": "Gemma-3-4B",
    "Phi-4-mini-instruct": "Phi-Instruct",
    "Phi-4-mini-reasoning": "Phi-Reasoning",
}
ORDER = list(MODELS.values())
CMAP = dict(zip(ORDER, ["#2F5D8C", "#6D9DC5", "#A8C0DA",
                        "#B07C2E", "#8E5572", "#4F8A6B"]))
I, R = 1, 2
OPTIONS = (2,)      # conj_h3, the canonical R,h3 versus I,h3 contrast


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    tight = getattr(fig, "_paper_tight", True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}",
                    bbox_inches="tight" if tight else None, pad_inches=.02)
    plt.close(fig)
    print(f"  {(OUT / (name + '.png')).relative_to(PROJ)}")


# ══════════════════════════════════════════ Figure 1a, the schematic
# the coordinate system is x 0..152, y 0..76; one unit is 5.5in/152 ≈ 2.61pt. DejaVu Sans
# averages about 0.55em per glyph, so one glyph at s pt takes about 0.21*s units.
# a stage is 35 units wide, so about 33 characters fit on a line at 5.0pt.
UNIT_PT = 5.5 * 72 / 152


def width_of(s: str, size: float) -> float:
    """Approximate string width in coordinate units."""
    return len(s) * 0.55 * size / UNIT_PT


def box(ax, x, y, w, h, *, fc="white", ec=INK, lw=.7, r=1.2, z=2, alpha=1.):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))


def txt(ax, x, y, s, *, size=5.0, color=INK, ha="left", va="center", weight=None,
        z=5):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=z,
            fontweight=weight)


def arrow(ax, a, b, *, color=INK, lw=.9, rad=0., ms=6, z=4, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, zorder=z, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def runs(ax, x, y, parts, *, size=5.0, va="center", z=6):
    """Concatenate differently coloured pieces on one line; widths are measured by the renderer."""
    renderer = ax.figure.canvas.get_renderer()
    inverse = ax.transData.inverted()
    cursor = x
    for text, color, weight in parts:
        stripped = text.rstrip()
        item = ax.text(cursor, y, stripped, fontsize=size, color=color,
                       fontweight=weight, ha="left", va=va, zorder=z)
        bbox = item.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inverse.transform((bbox.x0, bbox.y0)), \
            inverse.transform((bbox.x1, bbox.y1))
        cursor += (x1 - x0) + (len(text) - len(stripped)) * .30 * size / UNIT_PT
    return cursor


def strip(ax, x, y, w, h, cells=14, *, seed=3, z=6):
    """Activation-vector glyph: one component per cell, shown as lightness."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, cells)
    ch = h / cells
    for k, value in enumerate(values):
        t = float(np.clip(.5 + value / 4.2, .04, .96))
        ax.add_patch(plt.Rectangle((x, y + k * ch), w, ch,
                                   fc=plt.get_cmap("Blues")(.16 + .62 * t),
                                   ec="white", lw=.35, zorder=z))
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="none", ec="#B9C1CA", lw=.6,
                               zorder=z + 1))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(5.5, 3.18))
    ax.set_xlim(0, 152); ax.set_ylim(0, 88); ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.canvas.draw()

    grey, mid = "#6E7378", "#8A9098"

    def title(x, y, tag, label):
        txt(ax, x, y, tag, size=7.6, weight="bold", va="baseline")
        txt(ax, x + 3.4, y, label, size=6.4, weight="bold", va="baseline")

    # ═════════════════ a. the item, the one place where text has to be text
    box(ax, 0, 61.0, 152, 21.0, fc=FAINT, ec="none", r=1.6, z=0)
    title(0, 84.4, "a", "One conjunction item under three context prefixes")

    for tag, col, parts, y in [
        ("N", IRR, [("Dan has a degree in violin performance.", grey, None)], 77.0),
        ("I", IRR, [("… research for ", grey, None),
                    ("a dairy company", mid, "bold"), (" …", grey, None)], 71.4),
        ("R", REL, [("… research for ", INK, None),
                    ("an umbrella company", REL, "bold"), (" …", INK, None)], 65.8)]:
        box(ax, 2.6, y - 2.2, 4.4, 4.4, fc=col, ec="none", r=.9, z=3)
        txt(ax, 4.8, y, tag, size=5.4, color="white", ha="center", weight="bold", z=4)
        end = runs(ax, 10.4, y, parts, size=5.0)
        if tag != "N":
            txt(ax, end + 1.6, y, "+ same evidence", size=4.4, color="#A6ACB3")
    ax.plot([8.2, 9.2, 9.2, 8.2], [71.4, 71.4, 65.8, 65.8], color=REL, lw=.7,
            zorder=4)

    ax.plot([68.0, 68.0], [63.8, 79.0], color="#D5DAE0", lw=.7, zorder=2)

    txt(ax, 71.4, 76.4, "$h_1$", size=5.0, color="#A6ACB3")
    txt(ax, 75.8, 76.4, "Dan is an expert mountaineer", size=5.0, color=grey)
    for tag, col, weight, text, y in [
            ("$h_3$", REL, "bold", "and owns an umbrella.", 70.8),
            ("$h_2$", mid, None, "and gives music lessons.", 65.8)]:
        txt(ax, 71.4, y, tag, size=5.0, color=col)
        txt(ax, 79.4, y, text, size=5.0, color=col, weight=weight)

    bx = 130.0
    ax.plot([bx - 2.0, bx + 15.0], [63.4, 63.4], color="#C9CFD6", lw=.6, zorder=3)
    for k, (value, col) in enumerate([(4.4, mid), (9.0, REL)]):
        ax.add_patch(plt.Rectangle((bx + 1.4 + k * 7.0, 63.4), 4.6, value, fc=col,
                                   ec="none", zorder=4))
    ax.plot([bx + 3.7, bx + 3.7, bx + 10.7, bx + 10.7],
            [68.4, 74.4, 74.4, 73.0], color=INK, lw=.6, zorder=4)
    txt(ax, bx + 7.2, 77.0, "DoD", size=5.4, ha="center", weight="bold")

    # ═════════════════ b, c, d. the mechanism
    W, GAP = 47.0, 5.5
    X = [0.0, 52.5, 105.0]
    TOP, BOT = 50.5, 12.5
    for x in X:
        box(ax, x, BOT, W, TOP - BOT, fc=FAINT, ec="none", r=1.6, z=0)
    for x in X[:-1]:
        arrow(ax, (x + W + 1.0, 31.5), (x + W + GAP - 1.0, 31.5), lw=1.0)
    arrow(ax, (12.5, 60.4), (12.5, 57.2), lw=1.0)

    # ───────────────────────── b. what the residual stream is, and where we read it
    x0 = X[0]
    title(x0, 54.0, "b", "Read the residual stream")

    tx = x0 + 8.0
    arrow(ax, (tx, 15.0), (tx, 47.6), color="#B6BEC7", lw=2.6, ms=9, z=1)
    for k in range(7):
        ax.add_patch(plt.Rectangle((tx - 6.2 + k * 2.3, 12.9), 1.8, 2.2,
                                   fc=REL if k == 6 else "#DDE2E8", ec="none",
                                   zorder=4))
    txt(ax, tx + 2.0, 16.4, "residual stream", size=4.6, color="#8A9098")

    def sublayer(label, y_out, y_in, y_add):
        """One layer: branch off the stream, pass through the block, add back."""
        cx = x0 + 23.0
        box(ax, x0 + 15.0, y_in, 16.0, 5.2, fc="white", ec="#C4CBD3", lw=.75, z=3)
        txt(ax, cx, y_in + 2.6, label, size=5.2, ha="center", color=INK, z=4)
        ax.plot([tx, cx, cx], [y_out, y_out, y_in], color="#C4CBD3", lw=.75, zorder=2)
        ax.plot([cx, cx, tx + 1.5], [y_in + 5.2, y_add, y_add], color="#C4CBD3",
                lw=.75, zorder=2)
        ax.add_patch(plt.Circle((tx, y_add), 1.5, fc="white", ec="#8A9098", lw=.75,
                                zorder=5))
        ax.plot([tx - .8, tx + .8], [y_add, y_add], color="#8A9098", lw=.7, zorder=6)
        ax.plot([tx, tx], [y_add - .8, y_add + .8], color="#8A9098", lw=.7, zorder=6)

    sublayer("attention", 19.4, 20.6, 28.4)
    sublayer("MLP", 30.4, 31.6, 39.4)
    txt(ax, tx, 43.0, "$\\vdots$", size=7.0, ha="center", color="#9AA1A9")

    hook_y = 44.2
    ax.plot([tx], [hook_y], "o", ms=2.8, color=REL, zorder=6)
    ax.plot([tx, x0 + 35.5], [hook_y, hook_y], color=REL, lw=.8, ls=(0, (2.2, 1.6)),
            zorder=5)
    arrow(ax, (x0 + 35.5, hook_y), (x0 + 35.5, 43.6), color=REL, lw=.8, ms=5, z=5)
    txt(ax, tx + 1.6, hook_y + 2.2, r"$\ell$", size=6.0, color=REL)

    strip(ax, x0 + 33.0, 20.4, 5.0, 22.6, seed=5)
    txt(ax, x0 + 35.5, 17.6, "$a$", size=6.2, ha="center")


    # ───────────────────────── c. the direction, minus the evidential plane
    x0 = X[1]
    title(x0, 54.0, "c", "Two separated directions")

    ox, oy = x0 + 16.0, 19.0
    ax.add_patch(Polygon([(ox - 10.0, oy - 1.8), (ox + 13.0, oy + 2.4),
                          (ox + 18.8, oy + 7.6), (ox - 4.2, oy + 3.4)],
                         closed=True, fc=EVI, ec=EVI, lw=.7, alpha=.16, zorder=2))
    arrow(ax, (ox, oy + 1.0), (ox + 10.0, oy + 2.8), color=EVI, lw=.9, ms=5, z=6)
    arrow(ax, (ox, oy + 1.0), (ox + 4.0, oy + 5.0), color=EVI, lw=.9, ms=5, z=6)
    txt(ax, ox + 10.6, oy + 2.7, r"$\rho$", size=5.8, color=EVI)
    txt(ax, ox + 3.2, oy + 5.8, "$C$", size=5.8, color=EVI)

    tip, foot = (ox + 7.6, oy + 24.4), (ox + 7.6, oy + 3.2)
    arrow(ax, (ox, oy + 1.0), tip, color=REL, lw=1.6, ms=8, z=7)
    txt(ax, tip[0] + 1.4, tip[1] - 1.2, "$v_R$", size=6.4, color=REL, weight="bold")
    ax.plot([tip[0], foot[0]], [tip[1], foot[1]], color="#9AA1A9", lw=.7,
            ls=(0, (1.8, 1.6)), zorder=6)
    arrow(ax, foot, tip, color=EDIT, lw=1.2, ms=6, z=8)
    txt(ax, foot[0] + 1.6, foot[1] + 9.4, r"$R_{|PC}$", size=5.8, color=EDIT)
    arrow(ax, (ox, oy + 1.0), (ox - 6.6, oy + 0.2), color="#8C6222", lw=1.5, ms=6,
          z=7)
    txt(ax, ox - 7.4, oy + 0.2, r"$PC_{|R}$", size=5.4, color="#8C6222", ha="right")
    ax.plot([foot[0] - 1.5, foot[0] - 1.5, foot[0]],
            [foot[1] + 1.5, foot[1], foot[1]], color="#9AA1A9", lw=.55, zorder=6)

    rng = np.random.default_rng(7)
    for lab, col, cx, cy in [("$I,h_3$", IRR, ox - 1.4, oy + 12.0),
                             ("$R,h_3$", REL, ox + 2.6, oy + 20.4)]:
        p = rng.normal(0, 1.0, (12, 2)) * np.array([1.7, 1.0])
        ax.plot(cx + p[:, 0], cy + p[:, 1], "o", ms=2.0, color=col, alpha=.8,
                mec="none", zorder=9)
        txt(ax, cx - 5.4, cy, lab, size=4.8, color=col, ha="right")

    # ───────────── d. edit that same stream, one item-assigned stage at a time
    x0 = X[2]
    title(x0, 54.0, "d", "Rescue or knock out the fallacy")

    tx = x0 + 4.8
    arrow(ax, (tx, 15.0), (tx, 47.6), color="#B6BEC7", lw=2.6, ms=9, z=1)
    ax.add_patch(plt.Circle((tx, 31.4), 2.0, fc="white", ec=EDIT, lw=1.0, zorder=5))
    ax.plot([tx - 1.0, tx + 1.0], [31.4, 31.4], color=EDIT, lw=.9, zorder=6)
    ax.plot([tx, tx], [30.4, 32.4], color=EDIT, lw=.9, zorder=6)
    txt(ax, tx - 2.6, 31.4, r"$\ell$", size=6.0, color=EDIT, ha="right")

    for y_out, y_in in ((32.6, 36.6), (30.2, 17.6)):
        ax.plot([tx + 2.3, x0 + 9.8, x0 + 9.8, x0 + 12.4],
                [y_out, y_out, y_in, y_in], color=EDIT, lw=.7, ls=(0, (2.0, 1.6)),
                zorder=3)

    def ray(oxx, oyy, length=13.6):
        ax.plot([oxx, oxx + length], [oyy, oyy], color=REL, lw=.9,
                ls=(0, (2.0, 1.8)), zorder=4)
        txt(ax, oxx + length + .9, oyy, r"$\hat{v}$", size=5.2, color=REL)

    def outcome(cxx, base_y, sign):
        """What the edit is scored on; shows the predicted sign only."""
        ax.plot([cxx - 2.6, cxx + 2.6], [base_y, base_y], color="#C4CBD3", lw=.7,
                zorder=4)
        arrow(ax, (cxx, base_y), (cxx, base_y + sign * 4.2), color=EDIT, lw=1.4,
              ms=6, z=5)
        txt(ax, cxx, base_y + sign * 6.4, r"$\Delta$CF", size=4.8, color=EDIT,
            ha="center")

    # knock out: the fallacy is there, take the direction out
    oxx, oyy = x0 + 13.6, 36.6
    ray(oxx, oyy)
    a_tip = (oxx + 8.8, oyy + 7.0)
    arrow(ax, (oxx, oyy), a_tip, color="#9AA1A9", lw=1.3, ms=6, z=6)
    txt(ax, a_tip[0] + 1.0, a_tip[1] + .2, "$a$", size=5.2, color="#7C838A")
    ax.plot([a_tip[0], a_tip[0]], [a_tip[1], oyy], color="#9AA1A9", lw=.6,
            ls=(0, (1.6, 1.4)), zorder=5)
    ax.plot([oxx, a_tip[0]], [oyy, oyy], color=REL, lw=2.4, zorder=6,
            solid_capstyle="butt")
    for dx, dy in ((1.1, 1.1), (1.1, -1.1)):
        ax.plot([oxx + 4.4 - dx, oxx + 4.4 + dx], [oyy - dy, oyy + dy], color=EDIT,
                lw=1.0, zorder=8)
    arrow(ax, (oxx, oyy), (oxx, oyy + 7.0), color=EDIT, lw=1.6, ms=7, z=7)
    txt(ax, oxx - 1.1, oyy + 7.0, "$a'$", size=5.2, color=EDIT, ha="right",
        weight="bold")
    txt(ax, x0 + 10.4, 48.2, "$-$", size=8.5, color=EDIT, weight="bold")
    txt(ax, x0 + 13.6, 48.2, r"knock out,  CF $>0$", size=4.8, color=EDIT)
    outcome(x0 + 37.4, 41.4, -1)

    # rescue: the fallacy is absent, put the direction in
    oxx, oyy = x0 + 13.6, 17.6
    ray(oxx, oyy)
    a_tip = (oxx + 4.2, oyy + 7.0)
    arrow(ax, (oxx, oyy), a_tip, color="#9AA1A9", lw=1.3, ms=6, z=6)
    txt(ax, a_tip[0] - 1.0, a_tip[1] + .2, "$a$", size=5.2, color="#7C838A",
        ha="right")
    new = (a_tip[0] + 8.0, a_tip[1])
    arrow(ax, a_tip, new, color=REL, lw=1.3, ms=6, z=7)
    txt(ax, (a_tip[0] + new[0]) / 2, new[1] + 1.6, r"$\alpha\hat{v}$", size=5.0,
        color=REL, ha="center")
    arrow(ax, (oxx, oyy), new, color=EDIT, lw=1.6, ms=7, z=8)
    txt(ax, new[0] + 1.0, new[1] - 1.2, "$a'$", size=5.2, color=EDIT, weight="bold")
    txt(ax, x0 + 10.4, 29.2, "$+$", size=8.5, color=EDIT, weight="bold")
    txt(ax, x0 + 13.6, 29.2, r"rescue,  CF $\leq 0$", size=4.8, color=EDIT)
    outcome(x0 + 37.4, 19.6, +1)


    # ───────────────────────── the edited stream goes back through the model
    ay = 6.6
    ax.plot([X[2] + 4.8, X[2] + 4.8, X[0] + 8.0], [BOT - .8, ay, ay], color=EDIT,
            lw=.8, ls=(0, (2.4, 1.8)), zorder=3, solid_capstyle="butt")
    arrow(ax, (X[0] + 8.0, ay), (X[0] + 8.0, 11.8), color=EDIT, lw=.8, ms=6,
          ls=(0, (2.4, 1.8)), z=3)

    save(fig, "fig0_pipeline")
    README_FIGS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT / "fig0_pipeline.png", README_FIGS / "fig0_pipeline.png")
    print(f"  {(README_FIGS / 'fig0_pipeline.png').relative_to(PROJ)}")


# ═══════════════════════════════ Supplementary figure S1, the decoding result
def unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / n if n > 0 else x


def nested_loio(a: np.ndarray) -> tuple[list[int], np.ndarray, np.ndarray]:
    """analyze_rq3_context_decoding_robustness.nested_loio, options fixed to conj_h3."""
    n, _, _, layers, _ = a.shape
    items = np.arange(n)
    selected, labels, scores = [], [], []
    for outer in range(n):
        train = items[items != outer]
        layer_auc = []
        for layer in range(layers):
            cell = a[train, :, :, layer][:, :, list(OPTIONS), :].mean(axis=2)
            contrast = cell[:, R] - cell[:, I]
            total = contrast.sum(axis=0)
            inner_labels, inner_scores = [], []
            for position, inner in enumerate(train):
                d = unit((total - contrast[position]) / (n - 2))
                state = a[inner, :, :, layer]
                inner_scores += list(state[R, list(OPTIONS)] @ d)
                inner_scores += list(state[I, list(OPTIONS)] @ d)
                inner_labels += [1] * len(OPTIONS) + [0] * len(OPTIONS)
            layer_auc.append(roc_auc_score(inner_labels, inner_scores))
        chosen = int(np.argmax(layer_auc))
        selected.append(chosen)
        cell = a[train, :, :, chosen][:, :, list(OPTIONS), :].mean(axis=2)
        d = unit((cell[:, R] - cell[:, I]).mean(axis=0))
        state = a[outer, :, :, chosen]
        scores += list(state[R, list(OPTIONS)] @ d) + list(state[I, list(OPTIONS)] @ d)
        labels += [1] * len(OPTIONS) + [0] * len(OPTIONS)
    return selected, np.asarray(labels), np.asarray(scores, dtype=float)


def decoding_scores(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE)
    rows = []
    for slug, name in MODELS.items():
        for task in (2, 3):
            path = ACT / slug / f"task{task}_cellmean.npy"
            if not path.exists():
                print(f"  missing {path.relative_to(PROJ)}, skipped")
                continue
            a = np.load(path)
            selected, labels, scores = nested_loio(a)
            auc = roc_auc_score(labels, scores)
            print(f"  {name} task{task}  AUC={auc:.4f}", flush=True)
            for k, (y, s) in enumerate(zip(labels, scores)):
                rows.append({"model": name, "task": task, "item": k // 2,
                             "label": int(y), "score": float(s),
                             "layer": selected[k // 2], "auc_cell": auc,
                             "n_layers": int(a.shape[3])})
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE, index=False)
    return frame


def fig_s1_decoding(refresh: bool = False):
    d = decoding_scores(refresh)
    gate = pd.read_csv(ANA / "rq3a_context_decoder_robustness.csv")
    gate = gate[gate.family == "conjunction_h3"].copy()
    gate["short"] = gate.model.map(LONG_TO_SHORT)
    gate["rank"] = gate["short"].map({m: k for k, m in enumerate(ORDER)})
    gate = gate.sort_values(["rank", "task"]).reset_index(drop=True)
    curves = pd.read_csv(ANA / "rq3a_layer_curves_h3_locked_20260825.csv")
    curves["short"] = curves.model.map(LONG_TO_SHORT)

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.30),
                             gridspec_kw={"width_ratios": [1.00, 1.02, 1.28],
                                          "wspace": .34})
    fig.subplots_adjust(left=.085, right=.988, top=.90, bottom=.28)

    # a. held-out ROC, one curve per cell
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="0.55", lw=.7, ls=(0, (3, 2)), zorder=1)
    for (model, task), g in d.groupby(["model", "task"], sort=False):
        fpr, tpr, _ = roc_curve(g.label, g.score)
        ax.plot(fpr, tpr, color=CMAP[model], lw=.85, alpha=.9, zorder=3)
    mean_auc = d.groupby(["model", "task"]).auc_cell.first().mean()
    ax.text(.95, .07, f"{mean_auc:.3f}", ha="right", va="bottom", fontsize=8.0,
            color=REL, weight="bold", transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("a  $R,h_3$ against $I,h_3$", loc="left")

    # b. every cell clears both null distributions
    ax = axes[1]
    for k, row in gate.iterrows():
        ax.plot([row.random_95, row.permutation_95], [k, k], color="0.82", lw=2.0,
                solid_capstyle="butt", zorder=2)
        ax.plot([row.permutation_95], [k], "|", ms=3.6, color="0.45", zorder=3)
        ax.plot([row.auc_nested_loio], [k], "o", ms=3.6, color=CMAP[row["short"]],
                mec="none", zorder=4)
    ax.axvline(.5, color="k", lw=.7, ls=(0, (3, 2)))
    ax.set_yticks([])
    ax.set_ylim(-.8, len(gate) - .2); ax.invert_yaxis()
    ax.set_xlim(.46, .89); ax.set_xticks([.5, .6, .7, .8])
    ax.set_xlabel("nested-LOIO AUC")
    ax.set_ylabel("12 cells")
    ax.set_title("b  Above both nulls", loc="left")
    ax.grid(axis="y", visible=False)

    # c. where in depth relevance becomes decodable
    ax = axes[2]
    for name in ORDER:
        g = curves[curves["short"] == name]
        if g.empty:
            continue
        m = g.groupby("layer").auc_loio.mean().reset_index()
        ax.plot(m.layer / m.layer.max(), m.auc_loio, color=CMAP[name], lw=1.0,
                label=name, zorder=3)
    ax.axhline(.5, color="k", lw=.7, ls=(0, (3, 2)))
    ax.set_xlim(0, 1); ax.set_ylim(.44, .90)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel(r"relative depth  $\ell / L$")
    ax.set_ylabel("LOIO AUC")
    ax.set_title("c  Decodable by mid-depth", loc="left")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=6, loc="lower center",
               bbox_to_anchor=(.5, .004), handlelength=1.0, handletextpad=.35,
               columnspacing=1.1, fontsize=5.6)

    fig._paper_tight = False
    save(fig, "figS1_decoding")


if __name__ == "__main__":
    print("pipeline schematic and decoding figure")
    fig_pipeline()
    fig_s1_decoding(refresh="--refresh" in sys.argv)
