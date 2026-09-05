"""Normalise the outputs of `run_rq3_behavior.py` to the canonical run-directory layout.

    python3 additional_Qs/normalize_rq3_outputs.py --run-dir additional_Qs/rq3_run

The analysis scripts in `analysis/rq3/` assume the canonical file names and columns. This runner
wrote norms in long format (one row per metric) under different names, so **the outputs are moved
instead of the analysis being changed.** Changing the analysis would break the claim that the same code ran.

    behavior/behavior_<tag>.csv  ->  behavior/rq3_behavior_<model_id>.csv
    norming/norm_<tag>.csv       ->  norming/norm_rq3_<model_id>.csv   (long -> wide format)

The wide format is defined by `analysis/rq3/run_rq3_representation_gpu.py`.

    rho = rho_yes_lp - rho_no_lp
    C   = C_strengthen_lp - C_weaken_lp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

KEYS = ["model", "model_id", "scenario", "label", "kind", "prefix",
        "hypothesis", "frame", "variant", "prompt_contract"]


def widen(long: pd.DataFrame) -> pd.DataFrame:
    p = long[long.metric == "P"].set_index(KEYS)
    c = long[long.metric == "C"].set_index(KEYS)
    if len(p) != len(c):
        sys.exit(f"P has {len(p)} rows but C has {len(c)}")
    out = pd.DataFrame(index=p.index)
    out["rho"] = p.lp_pos - p.lp_neg
    out["C"] = c.lp_pos - c.lp_neg
    out["rho_yes_lp"], out["rho_no_lp"] = p.lp_pos, p.lp_neg
    out["rho_yes_nt"], out["rho_no_nt"] = p.nt_pos, p.nt_neg
    out["C_strengthen_lp"], out["C_weaken_lp"] = c.lp_pos, c.lp_neg
    out["C_strengthen_nt"], out["C_weaken_nt"] = c.nt_pos, c.nt_neg
    return out.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("additional_Qs/rq3_run"))
    a = ap.parse_args()

    nb = 0
    for f in sorted((a.run_dir / "behavior").glob("behavior_*.csv")):
        d = pd.read_csv(f)
        mid = d.model_id.iloc[0].replace("/", "__")
        out = f.parent / f"rq3_behavior_{mid}.csv"
        d.to_csv(out, index=False)
        print(f"  behavior {len(d):5d} rows -> {out.name}")
        nb += 1

    nn = 0
    for f in sorted((a.run_dir / "norming").glob("norm_*.csv")):
        if f.name.startswith("norm_rq3_"):
            continue
        d = pd.read_csv(f)
        w = widen(d)
        mid = d.model_id.iloc[0].replace("/", "__")
        out = f.parent / f"norm_rq3_{mid}.csv"
        w.to_csv(out, index=False)
        print(f"  norming  {len(d):5d} rows (long) -> {len(w)} rows (wide) -> {out.name}")
        print(f"    mean rho {w.rho.mean():+.3f} · mean C {w.C.mean():+.3f}")
        nn += 1

    if not nb or not nn:
        sys.exit("nothing to normalise")
    print(f"\nnormalised {nb} behavior and {nn} norming files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
