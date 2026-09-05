r"""Paper figures for the two-run design (15-item and 165-item banks).

    python3 sim/make_paper_figs_2runs.py

English labels, NeurIPS single-column width (5.5in). Reads only analysis artefacts
that the three runs already produced; recomputes nothing except cross-cell means and
bootstraps, which is what the notebooks do on screen.

    docs/paper/figs/paper2/fig1_behavior.png        H1 and H1b, both runs
    docs/paper/figs/paper2/fig2_representation.png  decoding, specificity, R-PC independence
    docs/paper/figs/legacy/fig3_intervention.png    target vs controls, both runs
    docs/paper/figs/legacy/fig_combined.png         behaviour + representation combined

The last two are retired. `fig3_intervention` comes from the abandoned intervention pipeline; the paper
uses `fig3_cf_intervention`. `fig_combined` lost its use when the alternative layout was dropped,
although its panels e and f (post-hoc projection controls) appear in no other figure.

main_A.tex lives in docs/paper/, so the \includegraphics paths resolve to figs/paper2/...
Outputs go in a sub-folder so they do not collide with other scripts' docs/paper/figs/*.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "docs" / "paper" / "figs" / "paper2"   # figs/paper2/ relative to main_A.tex
LEGACY = PROJ / "docs" / "paper" / "figs" / "legacy"  # retired figures
K = PROJ / "results" / "rq1_spec_cached_20260821" / "analysis"
KR = PROJ / "results" / "rq3_pcr_locked_20260825" / "analysis"
KI = PROJ / "results" / "rq3_intervention_final_20260827" / "analysis"
N = PROJ / "additional_Qs" / "rq12_on_165"
NR = PROJ / "additional_Qs" / "rq3_run_server" / "analysis"
NI = PROJ / "additional_Qs" / "rq3_intervention_server" / "analysis"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "axes.unicode_minus": False,
    "figure.dpi": 200, "savefig.dpi": 200,
    "axes.grid": True, "grid.alpha": .20, "grid.linewidth": .5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
})
C15, C165 = "#9DB4D4", "#3B6EA5"      # 15-item, 165-item
TGT, CTL = "#3B6EA5", "#9E9E9E"       # target, control
POS, NEG = "#2E8B57", "#C44E52"

ORDER = ["Qwen2.5-Math", "Qwen3-4B", "Qwen3-Instruct",
         "Gemma-3-4B", "Phi-Instruct", "Phi-Reasoning"]
ABBR = {"Qwen2.5-Math": "Q2.5-M", "Qwen3-4B": "Q3-4B", "Qwen3-Instruct": "Q3-In",
        "Gemma-3-4B": "Gemma", "Phi-Instruct": "Phi-In", "Phi-Reasoning": "Phi-Rs"}
SEED = 20260824


def short(x: str) -> str:
    return (str(x).split("/")[-1]
            .replace("Qwen2.5-Math-1.5B", "Qwen2.5-Math")
            .replace("Qwen3-4B-Instruct-2507", "Qwen3-Instruct")
            .replace("gemma-3-4b-it", "Gemma-3-4B")
            .replace("Phi-4-mini-instruct", "Phi-Instruct")
            .replace("Phi-4-mini-reasoning", "Phi-Reasoning"))


def truth(s):
    return s if pd.api.types.is_bool_dtype(s) else \
        s.astype(str).str.lower().isin(["true", "1", "yes"])


def bci(v, B=4000, seed=SEED):
    r = np.random.default_rng(seed); v = np.asarray(v, float)
    return np.percentile([v[r.integers(0, len(v), len(v))].mean() for _ in range(B)],
                         [2.5, 97.5])


def save(fig, name, out=None):
    out = out or OUT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    print(f"  {(out / name).relative_to(PROJ)}")


# ────────────────────────────────────────────────── Figure 1
def fig1():
    k = pd.read_csv(K / "logprob_cell_tests.csv"); k["m"] = k.model_id.map(short)
    n = pd.read_csv(N / "h1_cell_tests_165.csv"); n["m"] = n.model.map(short)
    six = set(n.m); k = k[k.m.isin(six)]
    idx = [m for m in ORDER if m in six]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05),
                             gridspec_kw={"width_ratios": [2.15, 1], "wspace": .34})

    ax = axes[0]
    x = np.arange(len(idx)); w = .36
    for i, (d, lab, c) in enumerate([(k, "15-item", C15), (n, "165-item", C165)]):
        s = d[d.arm == "conj"]
        m = s.groupby("m")["mean"].mean().reindex(idx)
        lo = s.groupby("m")["ci_lo"].mean().reindex(idx)
        hi = s.groupby("m")["ci_hi"].mean().reindex(idx)
        ax.bar(x + (i - .5) * w, m, w, color=c, label=lab, linewidth=0)
        ax.vlines(x + (i - .5) * w, lo, hi, color="0.25", lw=.7)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks(x, [ABBR[m] for m in idx], rotation=18, ha="right")
    ax.set_ylabel("DoD, conjunction arm")
    ax.set_title("a  Context favours the relevant conjunct", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, borderaxespad=.2)
    ax.grid(axis="x", visible=False)
    ax.margins(x=.04)

    ax.margins(y=.10)

    ax = axes[1]
    xa = np.arange(2)
    for i, (d, lab, c) in enumerate([(k, "15-item", C15), (n, "165-item", C165)]):
        mu, lo, hi = [], [], []
        for a in ("conj", "bare"):
            v = d[d.arm == a]["mean"].to_numpy()
            mu.append(v.mean()); l, h = bci(v); lo.append(l); hi.append(h)
        ax.bar(xa + (i - .5) * w, mu, w, color=c, linewidth=0)
        ax.vlines(xa + (i - .5) * w, lo, hi, color="0.25", lw=.7)
    ax.set_xticks(xa, ["conj", "bare"])
    ax.set_ylabel("mean DoD")
    ax.set_title("b  Not specific to conjunction", loc="left")
    ax.grid(axis="x", visible=False)
    ax.margins(x=.18)
    save(fig, "fig1_behavior.png")


# ────────────────────────────────────────────────── Figure 2
def fig2():
    kr = pd.read_csv(KR / "rq3a_context_decoder_robustness.csv")
    nr = pd.read_csv(NR / "rq3a_context_decoder_robustness.csv")
    kr = kr[kr.model_id.isin(set(nr.model_id))]
    fams = ["conjunction_h3", "conjunction", "base", "conjunction_h2"]
    flab = ["$h_3$\n(target)", "pooled", "base", "$h_2$"]

    kg = pd.read_csv(KR / "rq3a_specificity_gap_summary.csv")
    ng = pd.read_csv(NR / "rq3a_specificity_gap_summary.csv")
    kc = pd.read_csv(KR / "joint_r_pc_iut_no_holm" / "component_tests.csv")
    kc = kc[kc.model_id.isin(set(nr.model_id))]
    nc = pd.read_csv(NR / "joint_r_pc_iut_no_holm" / "component_tests.csv")

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95),
                             gridspec_kw={"width_ratios": [1.5, 1.25, .85], "wspace": .42})

    ax = axes[0]
    x = np.arange(len(fams)); w = .36
    for i, (d, lab, c) in enumerate([(kr, "15-item", C15), (nr, "165-item", C165)]):
        v = [d[d.family == f].auc_nested_loio.mean() for f in fams]
        ax.bar(x + (i - .5) * w, v, w, color=c, label=lab, linewidth=0)
    ax.axhline(.5, color="k", lw=.7, ls=(0, (3, 2)))
    ax.set_xticks(x, flab)
    ax.set_ylim(.45, .88); ax.set_ylabel("nested-LOIO AUC")
    ax.set_title("a  Relevance is decodable", loc="left")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2, borderaxespad=.2)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ctrls = ["conjunction_h2", "base", "conjunction"]
    clab = ["$h_2$", "base", "pooled"]
    for i, (d, c) in enumerate([(kg, C15), (ng, C165)]):
        for y, ctrl in enumerate(ctrls):
            s = d[d.control == ctrl]
            if s.empty:
                continue
            s = s.iloc[0]; yy = y + (i - .5) * .24
            ax.plot([s.model_lo, s.model_hi], [yy, yy], color=c, lw=2.2,
                    solid_capstyle="butt")
            ax.plot([s.gap_mean], [yy], "o", color=c, ms=3.2, mec="none")
    ax.axvline(0, color="k", lw=.7)
    ax.set_yticks(range(len(ctrls)), clab); ax.invert_yaxis()
    ax.set_xlabel("AUC gap vs control")
    ax.set_title("b  Specificity, 95% CI", loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(y=.28)

    ax = axes[2]
    xr = np.arange(2)
    for i, (d, c) in enumerate([(kc, C15), (nc, C165)]):
        v = [int(d[d.target == t].component_pass.sum()) for t in ("R", "PC")]
        ax.bar(xr + (i - .5) * w, v, w, color=c, linewidth=0)
    ax.set_xticks(xr, ["R|PC", "PC|R"])
    ax.set_yticks([0, 4, 8, 12])
    ax.set_ylim(0, 13.4); ax.set_ylabel("cells passing (of 12)")
    ax.set_title("c  R–PC independence", loc="left")
    ax.grid(axis="x", visible=False)
    ax.margins(x=.18)
    save(fig, "fig2_representation.png")


# ─────────────────────────────────── Combined figure (retired 4-page layout only)
def fig_combined():
    """Behaviour and representation in one float; the 4-page layout could not afford two."""
    k = pd.read_csv(K / "logprob_cell_tests.csv"); k["m"] = k.model_id.map(short)
    n = pd.read_csv(N / "h1_cell_tests_165.csv"); n["m"] = n.model.map(short)
    six = set(n.m); k = k[k.m.isin(six)]
    idx = [m for m in ORDER if m in six]

    kr = pd.read_csv(KR / "rq3a_context_decoder_robustness.csv")
    nr = pd.read_csv(NR / "rq3a_context_decoder_robustness.csv")
    kr = kr[kr.model_id.isin(set(nr.model_id))]
    fams = ["conjunction_h3", "conjunction", "base", "conjunction_h2"]
    flab = ["$h_3$", "pooled", "base", "$h_2$"]
    kg = pd.read_csv(KR / "rq3a_specificity_gap_summary.csv")
    ng = pd.read_csv(NR / "rq3a_specificity_gap_summary.csv")

    pc = pd.read_csv(PROJ / "additional_Qs" / "pc_projection_control" / "cells.csv")
    pc["run"] = pc.run.astype(str)
    vc = pd.read_csv(PROJ / "additional_Qs" / "rpc_vector_control" / "cells.csv")
    vc["run"] = vc.run.astype(str)
    vc = vc[(vc.contrast == "did") & (vc.model.isin(set(pc.model)))]

    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.55),
                             gridspec_kw={"width_ratios": [1.60, 0.86, 1.24],
                                          "wspace": .52, "hspace": .52})
    w = .36

    ax = axes[0, 0]
    x = np.arange(len(idx))
    for i, (d, lab, c) in enumerate([(k, "15-item", C15), (n, "165-item", C165)]):
        s_ = d[d.arm == "conj"]
        m = s_.groupby("m")["mean"].mean().reindex(idx)
        lo = s_.groupby("m")["ci_lo"].mean().reindex(idx)
        hi = s_.groupby("m")["ci_hi"].mean().reindex(idx)
        ax.bar(x + (i - .5) * w, m, w, color=c, label=lab, linewidth=0)
        ax.vlines(x + (i - .5) * w, lo, hi, color="0.25", lw=.7)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks(x, [ABBR[m] for m in idx], rotation=45, ha="right",
                  fontsize=5.9, rotation_mode="anchor")
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylabel("DoD, conj arm")
    ax.set_title("a  DoD by model", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.1, borderaxespad=.2)
    ax.grid(axis="x", visible=False); ax.margins(x=.04, y=.12)

    ax = axes[0, 1]
    xa = np.arange(2)
    for i, (d, c) in enumerate([(k, C15), (n, C165)]):
        mu, lo, hi = [], [], []
        for a in ("conj", "bare"):
            v = d[d.arm == a]["mean"].to_numpy()
            mu.append(v.mean()); l, h = bci(v); lo.append(l); hi.append(h)
        ax.bar(xa + (i - .5) * w, mu, w, color=c, linewidth=0)
        ax.vlines(xa + (i - .5) * w, lo, hi, color="0.25", lw=.7)
    ax.set_xticks(xa, ["conj", "bare"]); ax.set_ylabel("mean DoD")
    ax.set_title("b  DoD by arm", loc="left")
    ax.grid(axis="x", visible=False); ax.margins(x=.18)

    ax = axes[0, 2]
    xf = np.arange(len(fams))
    for i, (d, c) in enumerate([(kr, C15), (nr, C165)]):
        v = [d[d.family == f].auc_nested_loio.mean() for f in fams]
        ax.bar(xf + (i - .5) * w, v, w, color=c, linewidth=0)
    ax.axhline(.5, color="k", lw=.7, ls=(0, (3, 2)))
    ax.set_xticks(xf, flab, fontsize=5.8)
    ax.set_ylim(.45, .88); ax.set_ylabel("LOIO AUC")
    ax.set_title("c  Decoding AUC", loc="left")
    ax.grid(axis="x", visible=False)

    ax = axes[1, 0]
    ctrls = ["conjunction_h2", "base", "conjunction"]
    for i, (d, c) in enumerate([(kg, C15), (ng, C165)]):
        for y, ctrl in enumerate(ctrls):
            s_ = d[d.control == ctrl]
            if s_.empty:
                continue
            s_ = s_.iloc[0]; yy = y + (i - .5) * .24
            ax.plot([s_.model_lo, s_.model_hi], [yy, yy], color=c, lw=2.2,
                    solid_capstyle="butt")
            ax.plot([s_.gap_mean], [yy], "o", color=c, ms=3.2, mec="none")
    ax.axvline(0, color="k", lw=.7)
    ax.set_yticks(range(len(ctrls)), ["$h_2$", "base", "pooled"]); ax.invert_yaxis()
    ax.set_xlabel("AUC gap vs control")
    ax.set_title("d  Specificity gap", loc="left")
    ax.grid(axis="y", visible=False); ax.margins(y=.28)

    ax = axes[1, 1]
    combos = [("P", "none"), ("P", "project"), ("C", "none"), ("C", "project")]
    xp = np.arange(len(combos))
    for i, (run, c) in enumerate([("15", C15), ("165", C165)]):
        # cells.csv holds the did contrast only
        v = [int(pc[(pc.run == run) & (pc.target == t)
                    & (pc["mode"] == md)].gate.sum()) for t, md in combos]
        ax.bar(xp + (i - .5) * w, v, w, color=c, linewidth=0)
    ax.set_xticks(xp, ["$P$", "$P^{\\perp}$", "$C$", "$C^{\\perp}$"], fontsize=6)
    ax.set_yticks([0, 4, 8, 12]); ax.set_ylim(0, 13.4)
    ax.set_ylabel("cells of 12")
    ax.set_title("e  Norm decoding", loc="left")
    ax.grid(axis="x", visible=False); ax.margins(x=.10)

    ax = axes[1, 2]
    combos = [("R", "scalar"), ("R", "vector"), ("PC", "scalar"), ("PC", "vector")]
    xr = np.arange(len(combos))
    for i, (run, c) in enumerate([("15", C15), ("165", C165)]):
        v = [int(vc[(vc.run == run) & (vc.target == t) & (vc["mode"] == md)]["pass"].sum())
             for t, md in combos]
        ax.bar(xr + (i - .5) * w, v, w, color=c, linewidth=0)
    ax.set_xticks(xr, ["R|PC", "R|PC$^{\\perp}$", "PC|R", "PC|R$^{\\perp}$"],
                  fontsize=5.4, rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=1.0)
    ax.set_yticks([0, 4, 8, 12]); ax.set_ylim(0, 13.4)
    ax.set_ylabel("cells of 12")
    ax.set_title("f  Separation", loc="left")
    ax.grid(axis="x", visible=False); ax.margins(x=.10)

    save(fig, "fig_combined.png", LEGACY)


# ────────────────────────────────────────────────── Figure 3
def fig3():
    RAND = ["random", "random_k2_00", "random_k2_01", "shuffle"]

    def load(p, sel=None, scope=None):
        d = pd.read_csv(p)
        if scope and "ablate_scope" in d:
            d = d[d.ablate_scope == scope]
        d = d[truth(d.guard_pass)]
        if sel and sel in d:
            d = d[truth(d[sel])]
        return d

    sets = [("15-item", load(KI / "rq3b_abl_tests.csv", scope="selected"),
             load(KI / "rq3b_add_tests.csv", "selected_safe")),
            ("165-item", load(NI / "rq3b_abl_tests.csv"),
             load(NI / "rq3b_add_tests.csv", "selected_safe"))]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0),
                             gridspec_kw={"wspace": .34})
    rng = np.random.default_rng(1)
    for ax, mode, pred in [(axes[0], "ABL", "negative"), (axes[1], "ADD", "positive")]:
        pos, ticks, labs = 0., [], []
        for j, (lab, abl, add) in enumerate(sets):
            d = abl if mode == "ABL" else add
            tgt = d[d.direction == "R_h3"]["mean"].to_numpy()
            ctl = d[d.direction.isin(RAND)]["mean"].to_numpy()
            for kk, (v, c) in enumerate([(tgt, TGT), (ctl, CTL)]):
                xx = pos + kk * .42
                ax.plot(np.full(len(v), xx) + rng.uniform(-.07, .07, len(v)), v,
                        "o", ms=2.6, color=c, alpha=.75, mec="none", zorder=3,
                        label=("$v_R$ target" if kk == 0 else "random controls")
                        if j == 0 else None)
                if len(v):
                    ax.hlines(v.mean(), xx - .15, xx + .15, color="k", lw=1.0, zorder=4)
            ticks.append(pos + .21); labs.append(lab)
            pos += 1.25
        ax.axhline(0, color="k", lw=.7)
        ax.set_xticks(ticks, labs)
        ax.set_ylabel(r"$\Delta$DoD" if mode == "ABL" else "induced DoD")
        ax.set_title(f"{'a' if mode == 'ABL' else 'b'}  {mode}", loc="left")
        ax.grid(axis="x", visible=False)
        ax.margins(x=.16, y=.14)
        # shade the half-plane the hypothesis predicts
        lo, hi = ax.get_ylim()
        band = (lo, 0) if mode == "ABL" else (0, hi)
        ax.axhspan(*band, color=POS, alpha=.055, zorder=0, lw=0)
        ax.set_ylim(lo, hi)
        ax.text(.985, .04 if mode == "ABL" else .96, "predicted", transform=ax.transAxes,
                ha="right", va="bottom" if mode == "ABL" else "top",
                fontsize=5.8, color=POS)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(.5, -.10), handlelength=.8, handletextpad=.3,
               columnspacing=1.4)
    save(fig, "fig3_intervention.png", LEGACY)


if __name__ == "__main__":
    print("paper figures (two runs)")
    fig1(); fig2(); fig3(); fig_combined()
