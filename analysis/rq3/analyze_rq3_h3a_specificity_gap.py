#!/usr/bin/env python3
"""Restate H3a specificity as an **AUC gap** and test that gap (CPU only).

    python3 analysis/rq3/analyze_rq3_h3a_specificity_gap.py --input-dir <run>

Why this is needed
-----------
The canonical 15-item analysis stated H3a specificity as **gate pass counts**: `conjunction_h3` 12/12 against
`conjunction_h2` 1/12 and `base` 3/12, so only the target passed.

**That statement breaks at 165 items.** The gate is a 200-draw permutation test whose smallest attainable p is
1/200 = .005, and with 11 times as many held-out samples even a weak AUC 0.52 signal clears that null.
All 48 cells sit at the floor and pass. **Pass counts no longer discriminate.**

Yet **the size gap grew** (h3−h2 +0.194 → +0.233). Specificity is therefore restated as
follows.

    specificity = AUC(conjunction_h3) − AUC(control)      control ∈ {conjunction_h2, base, conjunction}

What is tested
-----------------
One gap per cell, so two levels are asked, as for H2b in `10_rq12_on_165.py`.

  (1) **Within cell**: both families' AUCs come from the same items, so the bootstrap is **paired**.
      Items are resampled with replacement, both AUCs recomputed, and the 2.5/97.5 percentiles of the difference taken.
      Drawing items jointly is the point: item difficulty is shared by both families and cancels.

  (2) **Across units**: the per-cell gaps are resampled. **The unit is the model.**
      The two tasks have highly correlated item profiles and are pseudo-replicates; resampling cells would
      narrow the CI artificially (the same judgment as `sim/make_30_spec_results_nb.py` §7.2).

What is recomputed
-------------------
The canonical `analyze_rq3_context_decoding_robustness.py` stores only the AUC and **discards the
per-item scores.** The paired bootstrap needs them, so they are rebuilt here. However,
**layers are not reselected**: the stored `selected_layers` are used per item as saved.
Only direction estimation and projection are repeated, so the resulting AUCs must equal the stored ones
(the script compares and warns on mismatch).

Canonical outputs are untouched; the output is two new files inside `--input-dir`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_rq3_context_decoding_robustness import (  # noqa: E402
    FAMILIES, I, R, means, score_item, unit_rows,
)

SEED = 20260824
BOOT = 2000
TARGET = "conjunction_h3"
CONTROLS = ["conjunction_h2", "base", "conjunction"]


def short(x: str) -> str:
    return (str(x).split("/")[-1]
            .replace("Qwen2.5-Math-1.5B", "Qwen2.5-Math")
            .replace("Qwen3-4B-Instruct-2507", "Qwen3-Instruct")
            .replace("DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-Distill")
            .replace("gemma-3-4b-it", "Gemma-3-4B")
            .replace("Phi-4-mini-instruct", "Phi-Instruct")
            .replace("Phi-4-mini-reasoning", "Phi-Reasoning"))


def loio_scores(array: np.ndarray, options: tuple[int, ...],
                selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Repeat only the outer fold of the canonical `nested_loio`, with the stored layers.

    Returns labels and scores shaped (items, 2*len(options)), so items can be resampled.
    """
    n = array.shape[0]
    items = np.arange(n)
    lab, sco = [], []
    for outer in range(n):
        train = items[items != outer]
        layer = int(selected[outer])
        cell = means(array, train, layer, options)
        direction = unit_rows((cell[:, R] - cell[:, I]).mean(axis=0)[None])[0]
        y, s = score_item(array[outer, :, :, layer], direction, options)
        lab.append(y); sco.append(s)
    return np.asarray(lab), np.asarray(sco, dtype=float)


def auc_from(lab: np.ndarray, sco: np.ndarray) -> float:
    return float(roc_auc_score(lab.ravel(), sco.ravel()))


def paired_gap_ci(labT, scoT, labC, scoC, rng) -> tuple[float, float, float]:
    """95% CI of the AUC gap with items resampled **jointly**."""
    n = labT.shape[0]
    gaps = np.empty(BOOT)
    for b in range(BOOT):
        idx = rng.integers(0, n, n)
        try:
            gaps[b] = (auc_from(labT[idx], scoT[idx]) - auc_from(labC[idx], scoC[idx]))
        except ValueError:      # degenerate draw with a single label
            gaps[b] = np.nan
    gaps = gaps[np.isfinite(gaps)]
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    # share of gaps at or below zero → bootstrap approximation of a two-sided p
    p = 2 * min((gaps <= 0).mean(), (gaps >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args()
    root = a.input_dir.resolve()
    ana = root / "analysis"
    rng = np.random.default_rng(SEED)

    ref = pd.read_csv(ana / "rq3a_context_decoder_robustness.csv")
    ref["m"] = ref.model_id.map(short)
    print(f"run {root.name} · cells {ref.groupby(['model_id','task']).ngroups} · "
          f"family {sorted(ref.family.unique())}")

    rows, mism = [], 0
    for (model_id, task), g in ref.groupby(["model_id", "task"]):
        slug = model_id.replace("/", "__")
        array = np.load(root / "activations" / slug / f"task{task}_cellmean.npy",
                        mmap_mode="r")
        cache = {}
        for fam in [TARGET] + CONTROLS:
            row = g[g.family == fam]
            if row.empty:
                continue
            sel = np.asarray(json.loads(row.selected_layers.iloc[0]), dtype=int)
            lab, sco = loio_scores(array, FAMILIES[fam], sel)
            auc = auc_from(lab, sco)
            if abs(auc - float(row.auc_nested_loio.iloc[0])) > 1e-9:
                mism += 1
                print(f"   ⚠ {short(model_id)} task{task} {fam}: recomputed {auc:.6f} vs "
                      f"stored {row.auc_nested_loio.iloc[0]:.6f}")
            cache[fam] = (lab, sco, auc)

        for ctrl in CONTROLS:
            if TARGET not in cache or ctrl not in cache:
                continue
            lo, hi, p = paired_gap_ci(*cache[TARGET][:2], *cache[ctrl][:2], rng)
            rows.append({"model_id": model_id, "model": short(model_id), "task": task,
                         "control": ctrl,
                         "auc_target": cache[TARGET][2], "auc_control": cache[ctrl][2],
                         "gap": cache[TARGET][2] - cache[ctrl][2],
                         "ci_lo": lo, "ci_hi": hi, "boot_p": p,
                         "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    cells = pd.DataFrame(rows)
    cells.to_csv(ana / "rq3a_specificity_gap_cells.csv", index=False)
    if mism == 0:
        print("   all recomputed AUCs match the stored values (expected, since the layer selection is reused)")

    # ── aggregation across units; the model level is the verdict, as in the canonical analysis.
    agg = []
    for ctrl, s in cells.groupby("control"):
        cell = s.gap.to_numpy()
        per = s.groupby("model_id").gap.mean().to_numpy()
        def ci(v):
            r = np.random.default_rng(SEED)
            return np.percentile([v[r.integers(0, len(v), len(v))].mean()
                                  for _ in range(4000)], [2.5, 97.5])
        lc, hc = ci(cell); lm, hm = ci(per)
        agg.append({"control": ctrl, "n_cells": len(cell), "n_models": len(per),
                    "gap_mean": cell.mean(),
                    "cell_lo": lc, "cell_hi": hc, "cell_excludes_zero": bool(lc > 0 or hc < 0),
                    "model_lo": lm, "model_hi": hm, "model_excludes_zero": bool(lm > 0 or hm < 0),
                    "cells_ci_excludes_zero": int(s.ci_excludes_zero.sum())})
    AGG = pd.DataFrame(agg)
    AGG.to_csv(ana / "rq3a_specificity_gap_summary.csv", index=False)

    print("\n" + "=" * 92)
    print("H3a specificity: AUC gap and its test")
    print("\n[1] Paired bootstrap within cells: 95% CI of the gap with items resampled jointly")
    print(cells[["model", "task", "control", "auc_target", "auc_control", "gap",
                 "ci_lo", "ci_hi", "boot_p", "ci_excludes_zero"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    for ctrl, s in cells.groupby("control"):
        print(f"   {ctrl:<16} cells whose CI excludes zero {int(s.ci_excludes_zero.sum())}/{len(s)}")

    print("\n[2] Aggregation across units: the model level is the verdict")
    print(f"{'control':<16}{'mean gap':>10}{'cell-level CI':>22}{'0':>5}"
          f"{'model-level CI':>22}{'0':>5}{'cells excl 0':>11}")
    for _, r in AGG.iterrows():
        print(f"{r.control:<16}{r.gap_mean:>+10.4f}"
              f"{f'[{r.cell_lo:+.3f},{r.cell_hi:+.3f}]':>22}"
              f"{'excl' if r.cell_excludes_zero else 'incl':>4}"
              f"{f'[{r.model_lo:+.3f},{r.model_hi:+.3f}]':>22}"
              f"{'excl' if r.model_excludes_zero else 'incl':>4}"
              f"{int(r.cells_ci_excludes_zero):>7}/{int(r.n_cells)}")
    print(f"\noutput -> {(ana / 'rq3a_specificity_gap_cells.csv').relative_to(root.parent.parent)}")
    print(f"       {(ana / 'rq3a_specificity_gap_summary.csv').relative_to(root.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
