"""Re-aggregate the CF rescue/knockout results over passing cells.

    python3 additional_Qs/analysis/17_cf_rescue_reanalysis.py

The original `cf_rescue_knockout_summary.csv` reports 12-cell means, which the failing cells dominate,
**so some entries carry the opposite sign to the passing cells.** The `0/12` at $\\alpha=2$ is a
fluency-guard failure, not an absent effect, and the two tasks correlate at $\\rho\\approx.96$, so 12 cells
are six in effect. These three points are separated here.

Neither the canonical outputs nor the original scripts are modified; read only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
SRC = PROJ / "additional_Qs" / "rq3_cf_rescue_20260830" / "analysis"
OUT = PROJ / "additional_Qs" / "cf_rescue_reanalysis"
TARGETS = ("R_given_PC", "PC_given_R")


def load() -> pd.DataFrame:
    t = pd.read_csv(SRC / "cf_rescue_knockout_tests.csv")
    t["model"] = t.model_id.str.split("/").str[-1]
    for column in ("direction_gate", "fluency_pass", "base_guard_all", "pass"):
        t[column] = t[column].astype(str).str.lower().isin(("true", "1"))
    t["spec_ok"] = (t.specificity_random_p < 0.05) & (t.specificity_shuffle_p < 0.05)
    t["sig"] = t.p_flip_one < 0.05
    t["strength"] = t.alpha.fillna(t.dose)
    return t


def by_pass(t: pd.DataFrame) -> pd.DataFrame:
    """Mean over passing cells only, placed beside the 12-cell mean."""
    rows = []
    for (stage, direction, strength), g in t[t.direction.isin(TARGETS)].groupby(
            ["stage", "direction", "strength"]):
        p = g[g["pass"]]
        rows.append({
            "stage": stage, "direction": direction, "strength": strength,
            "cells": len(g), "passed": len(p),
            "models_passed": p.model.nunique(),
            "mean_all_cells": g.mean_delta_cf.mean(),
            "mean_passing": p.mean_delta_cf.mean() if len(p) else np.nan,
            "sign_agrees": bool(len(p)) and bool(
                np.sign(g.mean_delta_cf.mean()) == np.sign(p.mean_delta_cf.mean())),
        })
    return pd.DataFrame(rows).sort_values(["stage", "direction", "strength"])


def gate_ladder(t: pd.DataFrame) -> pd.DataFrame:
    """How many cells each guard removes, by strength."""
    s = t[t.direction.isin(TARGETS)]
    g = s.groupby(["stage", "strength"]).agg(
        cells=("pass", "size"),
        direction_gate=("direction_gate", "sum"),
        fluency=("fluency_pass", "sum"),
        base_guard=("base_guard_all", "sum"),
        specificity=("spec_ok", "sum"),
        signflip=("sig", "sum"),
        passed=("pass", "sum"),
        ppl_max=("fluency_increase_max", "max"),
    ).reset_index()
    return g


def monotone(t: pd.DataFrame) -> pd.DataFrame:
    """Whether each passing model × task × direction is monotone in strength."""
    p = t[t["pass"] & t.direction.isin(TARGETS)]
    rows = []
    for (model, task, stage, direction), g in p.groupby(
            ["model", "task", "stage", "direction"]):
        g = g.sort_values("strength")
        values = g.mean_delta_cf.to_numpy()
        expect = 1.0 if stage == "ADD" else -1.0
        rows.append({
            "model": model, "task": task, "stage": stage, "direction": direction,
            "strengths": list(g.strength), "effects": [round(v, 4) for v in values],
            "sign_ok": bool(np.all(np.sign(values) == expect)),
            "monotone": bool(len(values) < 2 or np.all(
                np.diff(values * expect) > 0)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = load()
    t.to_csv(OUT / "cells_annotated.csv", index=False)

    a = by_pass(t); a.to_csv(OUT / "means_by_pass.csv", index=False)
    b = gate_ladder(t); b.to_csv(OUT / "gate_ladder.csv", index=False)
    c = monotone(t); c.to_csv(OUT / "monotonicity.csv", index=False)

    print("── passing-cell mean vs 12-cell mean")
    print(a.round(4).to_string(index=False))
    print("\n   entries with disagreeing sign:",
          ", ".join(f"{r.stage} {r.direction} {r.strength}"
                    for r in a.itertuples() if r.passed and not r.sign_agrees) or "none")

    print("\n── guard ladder by strength (24 target-direction cells)")
    print(b.round(3).to_string(index=False))

    print("\n── monotonicity and sign of the passing cells")
    print(c.to_string(index=False))
    print(f"\n   sign agrees {int(c.sign_ok.sum())}/{len(c)} · monotone {int(c.monotone.sum())}/{len(c)}")

    print("\n── passing models by arm")
    for (stage, direction), g in t[t["pass"]].groupby(["stage", "direction"]):
        print(f"   {stage} {direction:12s} {sorted(set(g.model))}")

    print(f"\n[saved] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
