"""Re-aggregate with fluency as a post-hoc gate instead of a pre-gate.

    python3 additional_Qs/analysis/18_cf_rescue_fluency_posthoc.py

The original run applied fluency (WikiText PPL increase <10%) as a **pre-gate**, discarding cells above
the threshold before the effect test. Hence `alpha=2` loses all 24 cells and shows only as `0/12`.
**An absent effect cannot be told from a damaged model.**

Here the order is reversed: decide first on direction gate, base collapse, specificity, sign-flip and predicted sign,
**then attach fluency to the passing cells post hoc and report how much the model degraded.**

Neither the canonical nor the original outputs are modified; read only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
SRC = PROJ / "additional_Qs" / "rq3_cf_rescue_20260830" / "analysis"
OUT = PROJ / "additional_Qs" / "cf_rescue_fluency_posthoc"
TARGETS = ("R_given_PC", "PC_given_R")
FLUENCY_CEILING = 0.10


def load() -> pd.DataFrame:
    t = pd.read_csv(SRC / "cf_rescue_knockout_tests.csv")
    t["model"] = t.model_id.str.split("/").str[-1]
    for column in ("direction_gate", "fluency_pass", "base_guard_all", "pass"):
        t[column] = t[column].astype(str).str.lower().isin(("true", "1"))
    t["spec"] = (t.specificity_random_p < 0.05) & (t.specificity_shuffle_p < 0.05)
    t["sig"] = t.p_flip_one < 0.05
    t["strength"] = t.alpha.fillna(t.dose)
    t["expected_sign"] = np.where(t.stage == "ADD",
                                  t.mean_delta_cf > 0, t.mean_delta_cf < 0)
    # verdict without fluency
    t["effect_pass"] = (t.direction_gate & t.base_guard_all & t.spec
                        & t.sig & t.expected_sign)
    t["fluency_band"] = pd.cut(
        t.fluency_increase_max, [-np.inf, 0.10, 0.25, 0.50, np.inf],
        labels=["<10%", "10-25%", "25-50%", ">50%"])
    return t


def ladder(t: pd.DataFrame) -> pd.DataFrame:
    s = t[t.direction.isin(TARGETS)]
    g = s.groupby(["stage", "strength"]).agg(
        cells=("pass", "size"),
        with_fluency_guard=("pass", "sum"),
        effect_only=("effect_pass", "sum"),
        fluency_ok=("fluency_pass", "sum"),
        ppl_median=("fluency_increase_max", "median"),
        ppl_max=("fluency_increase_max", "max"),
    ).reset_index()
    g["gained"] = g.effect_only - g.with_fluency_guard
    return g


def by_model(t: pd.DataFrame) -> pd.DataFrame:
    """Per model: at which strengths the effect holds and how much fluency degrades."""
    s = t[t.direction.isin(TARGETS)]
    rows = []
    for (model, stage, direction), g in s.groupby(["model", "stage", "direction"]):
        g = g.sort_values("strength")
        eff = g[g.effect_pass]
        if eff.empty:
            continue
        rows.append({
            "model": model, "stage": stage, "direction": direction,
            "strengths_passing": list(eff.strength.unique()),
            "tasks_passing": sorted(set(eff.task)),
            "effect_min": eff.mean_delta_cf.abs().min(),
            "effect_max": eff.mean_delta_cf.abs().max(),
            "ppl_min": eff.fluency_increase_max.min(),
            "ppl_max": eff.fluency_increase_max.max(),
            "clears_fluency": int(eff.fluency_pass.sum()),
            "cells": len(eff),
        })
    return pd.DataFrame(rows).sort_values(["stage", "direction", "model"])


def dose_curve(t: pd.DataFrame) -> pd.DataFrame:
    """Dose curves for models where the effect holds; fluency-failing cells are kept."""
    s = t[t.direction.isin(TARGETS)]
    keep = s[s.effect_pass].groupby(["model", "task", "stage", "direction"]).size().index
    rows = []
    for key in keep:
        model, task, stage, direction = key
        g = s[(s.model == model) & (s.task == task) & (s.stage == stage)
              & (s.direction == direction)].sort_values("strength")
        for r in g.itertuples():
            rows.append({
                "model": model, "task": task, "stage": stage, "direction": direction,
                "strength": r.strength, "delta_cf": r.mean_delta_cf,
                "p_flip_one": r.p_flip_one, "ppl_increase": r.fluency_increase_max,
                "effect_pass": r.effect_pass, "fluency_pass": r.fluency_pass,
            })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = load()
    t.to_csv(OUT / "cells_annotated.csv", index=False)

    a = ladder(t); a.to_csv(OUT / "ladder.csv", index=False)
    b = by_model(t); b.to_csv(OUT / "by_model.csv", index=False)
    c = dose_curve(t); c.to_csv(OUT / "dose_curve.csv", index=False)

    print("── moving fluency from pre-gate to post-hoc gate")
    print(a.round(3).to_string(index=False))

    lost = t[t.effect_pass & ~t["pass"]]
    print(f"\n── {len(lost)} cells that failed on fluency alone")
    print(lost[["model", "task", "stage", "direction", "strength", "mean_delta_cf",
                "p_flip_one", "fluency_increase_max"]]
          .sort_values(["stage", "strength"]).round(4).to_string(index=False))

    print("\n── per model")
    print(b.round(3).to_string(index=False))

    print("\n── dose curves (passing combinations, fluency failures included)")
    for (model, task, stage, direction), g in c.groupby(
            ["model", "task", "stage", "direction"]):
        marks = " ".join(
            f"{r.strength:g}:{r.delta_cf:+.3f}"
            f"{'*' if r.effect_pass else ' '}{'!' if not r.fluency_pass else ' '}"
            for r in g.itertuples())
        print(f"   {model:24s} T{task} {stage} {direction:12s} {marks}")
    print("\n   * effect verdict passed · ! fluency above 10%")
    print(f"\n[saved] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
