"""Figure 3 and supplementary S2 — the conditional rescue/knockout intervention.

    python3 sim/make_paper_fig3_0830.py

    docs/paper/figs/paper2/fig3_cf_intervention.{png,pdf}       main text, model × direction grid
    docs/paper/figs/paper2/figS2_intervention_detail.{png,pdf}  appendix, dose–response and manipulation check

Follows the panel specification of the revision plan §6.2 but **splits it in two for length.** The main
text keeps the grid; dose–response and the two manipulation-check axes go to the appendix, which the main text cites.
**Nothing is recomputed; only analysis CSVs are read.**

    fig3       additional_Qs/cf_rescue_fluency_posthoc/cells_annotated.csv
    figS2 a    additional_Qs/cf_rescue_fluency_posthoc/dose_curve.csv
    figS2 b·c  additional_Qs/rq3_norm_manipulation_20260902/analysis/norm_manipulation_tests.csv
               additional_Qs/rq3_norm_manipulation_20260902/analysis/norm_manipulation_joint_verdicts.csv

The plan's §6.2 table placed the first two under `rq3_cf_rescue_20260830/analysis/`, but they live
directly under `additional_Qs/`. The joint (IUT) verdict of the manipulation check cannot be built from
`tests.csv` alone, so `joint_verdicts.csv` is read too and the two are checked for agreement at run time.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "docs" / "paper" / "figs" / "paper2"
POSTHOC = PROJ / "additional_Qs" / "cf_rescue_fluency_posthoc"
NORM = PROJ / "additional_Qs" / "rq3_norm_manipulation_20260902" / "analysis"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "axes.unicode_minus": False,
    "figure.dpi": 200, "savefig.dpi": 400,
    "axes.grid": True, "grid.alpha": .20, "grid.linewidth": .5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
    "font.size": 7.5, "axes.titlesize": 6.6, "axes.labelsize": 6.6,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 5.6,
})

POS = "#2E8B57"
R_RAMP = ["#A8C0DA", "#5E88B4", "#2F5D8C"]      # R|PC, by strength
P_RAMP = ["#E0C08A", "#C89B4A", "#8C6222"]      # PC|R, by strength
STAGE_C = {"ABL": "#8E5572", "ADD": "#3B6EA5"}

SHORT = {
    "Qwen2.5-Math-1.5B": "Q2.5-M", "Qwen3-4B": "Q3-4B",
    "Qwen3-4B-Instruct-2507": "Q3-In", "gemma-3-4b-it": "Gemma",
    "Phi-4-mini-instruct": "Phi-In", "Phi-4-mini-reasoning": "Phi-Rs",
}
ORDER = ["Q2.5-M", "Q3-4B", "Q3-In", "Gemma", "Phi-In", "Phi-Rs"]
DIR_LABEL = {"R_given_PC": "$R_{|PC}$", "PC_given_R": "$PC_{|R}$"}
STRENGTH = {"ADD": (0.5, 1.0, 2.0), "ABL": (0.5, 1.0)}
COMBOS = [("ADD", "R_given_PC"), ("ADD", "PC_given_R"),
          ("ABL", "R_given_PC"), ("ABL", "PC_given_R")]


def short_id(value: str) -> str:
    return SHORT[str(value).split("/")[-1]]


def truth(series: pd.Series) -> pd.Series:
    return (series if pd.api.types.is_bool_dtype(series)
            else series.astype(str).str.lower().isin(["true", "1", "yes"]))


# ──────────────────────────────── Figure 3, the model by direction grid
def panel_grid(axes, cells):
    cells = cells[cells.direction.isin(DIR_LABEL)].copy()
    cells["m"] = cells.model.map(SHORT)
    for ax, (stage, direction) in zip(axes, COMBOS):
        sub = cells[(cells.stage == stage) & (cells.direction == direction)]
        ramp = R_RAMP if direction == "R_given_PC" else P_RAMP
        levels = STRENGTH[stage]
        width = .80 / len(levels)
        x = np.arange(len(ORDER))
        for k, level in enumerate(levels):
            s = sub[sub.strength == level]
            offset = (k - (len(levels) - 1) / 2) * width
            mean = s.groupby("m").mean_delta_cf.mean().reindex(ORDER)
            ax.bar(x + offset, mean, width * .92, color=ramp[k], linewidth=0,
                   label=f"{level:g}" if ax is axes[0] else None)
            for _, row in s[truth(s["pass"])].iterrows():
                ax.plot([ORDER.index(row.m) + offset], [row.mean_delta_cf], "o",
                        ms=2.2, color="#1B1B1B", mec="white", mew=.4, zorder=5)
        ax.axhline(0, color="k", lw=.7)
        lo, hi = -.50, .30
        ax.axhspan(*((0, hi) if stage == "ADD" else (lo, 0)), color=POS, alpha=.055,
                   zorder=0, lw=0)
        ax.set_ylim(lo, hi)
        ax.set_xticks(x, ORDER, rotation=45, ha="right", fontsize=4.6,
                      rotation_mode="anchor")
        ax.tick_params(axis="x", pad=1.0)
        letter = "abcd"[COMBOS.index((stage, direction))]
        ax.set_title(f"{letter}  {stage} · {DIR_LABEL[direction]}", loc="left")
        ax.grid(axis="x", visible=False)
        ax.margins(x=.04)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel(r"mean $\Delta$CF")
    axes[0].legend(frameon=False, title="strength", title_fontsize=5.2,
                   loc="upper left", handlelength=.9, handletextpad=.3,
                   labelspacing=.18, borderaxespad=.15)


# ──────────────────────────────── S2 a, dose-response of the injections
def panel_dose(ax, dose):
    add = dose[dose.stage == "ADD"].copy()
    add["m"] = add.model.map(SHORT)
    for (model, task, direction), g in add.groupby(["m", "task", "direction"]):
        g = g.sort_values("strength")
        colour = R_RAMP[2] if direction == "R_given_PC" else P_RAMP[2]
        ax.plot(g.strength, g.delta_cf, "-", color=colour, lw=.9, zorder=3)
        keep = truth(g.fluency_pass)
        ax.plot(g.strength[keep], g.delta_cf[keep], "o", ms=3.2, color=colour,
                mec="none", zorder=4)
        ax.plot(g.strength[~keep], g.delta_cf[~keep], "o", ms=3.2, mfc="white",
                mec=colour, mew=.8, zorder=4)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks([0.5, 1.0, 2.0], ["0.5", "1", "2"])
    ax.set_xlim(.32, 2.18)
    ax.set_xlabel(r"injection strength  $\alpha$")
    ax.set_ylabel(r"$\Delta$CF")
    ax.set_title("a  Dose–response, passing injections", loc="left")
    ax.grid(axis="x", visible=False)
    ax.text(.97, .55, "open circles fail\nthe fluency guard", transform=ax.transAxes,
            fontsize=5.0, color="0.35", va="center", ha="right")
    for label, colour in (("$R_{|PC}$", R_RAMP[2]), ("$PC_{|R}$", P_RAMP[2])):
        ax.plot([], [], "-o", color=colour, lw=.9, ms=3.2, mec="none", label=label)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(.26, .44),
              handlelength=1.0, handletextpad=.35, labelspacing=.18)


# ──────────────────────────────── S2 b·c, the manipulation check
def panel_check(ax_left, ax_right, tests, verdicts):
    target = tests[tests.control_role == "target"].copy()
    target["ratio"] = target["mean"].abs() / target["margin"]
    key = ["model_id", "task", "stage", "strength"]

    # left: for R|PC the binding norm must stay inside the equivalence margin
    r = target[target.target_direction == "R_given_PC"]
    binding = r.groupby(key).ratio.max().reset_index()
    joint = verdicts[verdicts.target_direction == "R_given_PC"].set_index(key)
    check = binding.set_index(key).ratio.lt(1)
    if not check.eq(joint.joint_pass.reindex(check.index)).all():
        raise RuntimeError("binding ratio disagrees with the recorded joint IUT verdict")

    rng = np.random.default_rng(3)
    for stage, g in binding.groupby("stage"):
        ax_left.plot(g.strength + rng.uniform(-.055, .055, len(g)), g.ratio, "o",
                     ms=3.4, color=STAGE_C[stage], mec="none", alpha=.9, zorder=4,
                     label=stage)
    ax_left.axhline(1, color="k", lw=.8, ls=(0, (3, 2)), zorder=3)
    ax_left.set_yscale("log")
    ax_left.set_xticks([0.5, 1.0, 2.0], ["0.5", "1", "2"])
    ax_left.set_xlim(.32, 2.18)
    ax_left.set_xlabel("strength")
    ax_left.set_ylabel(r"$|\Delta\mathrm{DID}| \,/\, \delta$")
    ax_left.set_title(r"b  $R_{|PC}$ stays in margin", loc="left")
    ax_left.grid(axis="x", visible=False)
    ax_left.legend(frameon=False, loc="lower right", handlelength=.8,
                   handletextpad=.3, labelspacing=.18, borderaxespad=.15)
    ax_left.text(.36, 1.14, "margin", fontsize=5.0, color="0.35", ha="left",
                 va="bottom")

    # right: for PC|R both norms must move in the predicted direction; the predicted quadrant is upper right
    p = target[target.target_direction == "PC_given_R"]
    wide = p.pivot_table(index=key, columns="metric",
                         values="expected_oriented_mean").reset_index()
    lim = 0.25
    ax_right.axhspan(0, lim, xmin=.5, xmax=1., color=POS, alpha=.07, lw=0, zorder=0)
    for stage, g in wide.groupby("stage"):
        ax_right.plot(g.P, g.C, "o", ms=3.4, color=STAGE_C[stage], mec="none",
                      alpha=.9, zorder=4, label=stage)
    ax_right.axhline(0, color="k", lw=.7)
    ax_right.axvline(0, color="k", lw=.7)
    ax_right.set_xlim(-lim, lim); ax_right.set_ylim(-.10, lim)
    ax_right.set_xticks([-.2, 0, .2]); ax_right.set_yticks([0, .1, .2])
    ax_right.set_xlabel(r"oriented $\Delta\mathrm{DID}_\rho$")
    ax_right.set_ylabel(r"oriented $\Delta\mathrm{DID}_C$")
    ax_right.set_title(r"c  $PC_{|R}$ misses the quadrant", loc="left")
    ax_right.text(.955, .04, "predicted", transform=ax_right.transAxes, ha="right",
                  va="bottom", fontsize=5.0, color=POS)


def main():
    cells = pd.read_csv(POSTHOC / "cells_annotated.csv")
    dose = pd.read_csv(POSTHOC / "dose_curve.csv")
    tests = pd.read_csv(NORM / "norm_manipulation_tests.csv")
    verdicts = pd.read_csv(NORM / "norm_manipulation_joint_verdicts.csv")
    OUT.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(5.5, 1.58))
    grid = gridspec.GridSpec(1, 4, figure=fig, wspace=.14, left=.082, right=.995,
                             top=.845, bottom=.235)
    panel_grid([fig.add_subplot(grid[k]) for k in range(4)], cells)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_cf_intervention.{ext}", pad_inches=.02)
    plt.close(fig)
    print(f"  {(OUT / 'fig3_cf_intervention.png').relative_to(PROJ)}")

    fig = plt.figure(figsize=(5.5, 1.92))
    grid = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.12, .94, 1.0],
                             wspace=.46, left=.082, right=.992, top=.865,
                             bottom=.185)
    panel_dose(fig.add_subplot(grid[0]), dose)
    panel_check(fig.add_subplot(grid[1]), fig.add_subplot(grid[2]), tests, verdicts)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figS2_intervention_detail.{ext}", pad_inches=.02)
    plt.close(fig)
    print(f"  {(OUT / 'figS2_intervention_detail.png').relative_to(PROJ)}")


if __name__ == "__main__":
    print("figure 3 and supplementary S2")
    main()
