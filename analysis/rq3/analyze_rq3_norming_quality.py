#!/usr/bin/env python3
"""Degeneracy and reliability check of the norm values (CPU).

$P$ and $C$ are measured by asking the model directly, but **the question may fail to register**:
if the model intends neither candidate answer, the log-probability difference measures noise in the
tail of the distribution. A corr(P, C) from such a model is a measurement failure, not a property
of the representation. This script separates the two.

Three quantities are measured.

1. **Answer-space saturation**: mean log-probability of the **preferred** candidate.
   If it is low the model never intended " yes"/" no" or " strengthen"/" weaken"
   in the first place, and the measure is undefined.
2. **Split-half reliability**: each cell has 6 replicates (2 frames × 3 wordings).
   Split odd/even, correlate over the 15 items, and apply the Spearman-Brown correction.
   Computed for $\\Delta P$ and $\\Delta C$ separately.
3. **Attenuation correction**: the classical correction
   $r_{\\text{corrected}} = r_{\\text{obs}} / \\sqrt{\\rho_{P}\\rho_{C}}$.
   If the correlation does not rise after correction, **the low correlation is not due to noise.**

**Corrected values are not reported as results.** This is a diagnostic of whether the observed low
correlation can be explained by measurement failure, not an estimate of the true value. Low-reliability
cells are **excluded from the verdict**, not corrected.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPLICATES = 6          # frame 2 × variant 3


def spearman_brown(half_a: np.ndarray, half_b: np.ndarray) -> float:
    if np.std(half_a) < 1e-12 or np.std(half_b) < 1e-12:
        return float("nan")
    r = float(np.corrcoef(half_a, half_b)[0, 1])
    return 2 * r / (1 + r) if r > -1 else float("nan")


def analyze(run: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((run / "norming").glob("norm_rq3_*.csv")):
        frame = pd.read_csv(path)
        model = str(frame.model.iloc[0]).split("/")[-1]
        best_rho = np.maximum(frame.rho_yes_lp, frame.rho_no_lp)
        best_c = np.maximum(frame.C_strengthen_lp, frame.C_weaken_lp)

        h3 = frame[frame.hypothesis == "h3"].copy()
        h3["replicate"] = h3.frame * 3 + h3.variant
        table = h3.pivot_table(index=["label", "replicate"], columns="prefix", values=["rho", "C"])
        delta_p = (table[("rho", "R")] - table[("rho", "I")]).unstack("replicate")
        delta_c = (table[("C", "R")] - table[("C", "I")]).unstack("replicate")
        if delta_p.shape[1] != REPLICATES:
            raise ValueError(f"{model}: {delta_p.shape[1]} replicates, expected {REPLICATES}")

        odd, even = [0, 2, 4], [1, 3, 5]
        reliability_p = spearman_brown(delta_p.iloc[:, odd].mean(1).to_numpy(),
                                       delta_p.iloc[:, even].mean(1).to_numpy())
        reliability_c = spearman_brown(delta_c.iloc[:, odd].mean(1).to_numpy(),
                                       delta_c.iloc[:, even].mean(1).to_numpy())
        observed = float(np.corrcoef(delta_p.mean(1), delta_c.mean(1))[0, 1])
        denominator = np.sqrt(max(reliability_p, 0.0) * max(reliability_c, 0.0))
        rows.append({
            "model": model,
            "preferred_answer_logp": float(best_rho.mean()),
            "preferred_confirmation_logp": float(best_c.mean()),
            "delta_P_sd": float(delta_p.mean(1).std(ddof=1)),
            "delta_C_sd": float(delta_c.mean(1).std(ddof=1)),
            "reliability_delta_P": reliability_p,
            "reliability_delta_C": reliability_c,
            "observed_corr": observed,
            "attenuation_corrected_corr": observed / denominator if denominator > 0.05 else float("nan"),
            "usable_P": bool(reliability_p >= 0.6),
            "usable_C": bool(reliability_c >= 0.6),
        })
    return pd.DataFrame(rows).sort_values("observed_corr").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--reliability-floor", type=float, default=0.6,
                        help="below this the target is excluded from the verdict (preregistered)")
    args = parser.parse_args()
    frame = analyze(args.input_dir)
    frame["usable_P"] = frame.reliability_delta_P >= args.reliability_floor
    frame["usable_C"] = frame.reliability_delta_C >= args.reliability_floor
    out = args.input_dir / "analysis" / f"rq3x_norming_quality{args.suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    pd.set_option("display.width", 210); pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(frame.to_string(index=False))
    print(f"\nbelow the reliability floor {args.reliability_floor}: "
          f"P in {int((~frame.usable_P).sum())} models, C in {int((~frame.usable_C).sum())} models")
    print(out)


if __name__ == "__main__":
    main()
