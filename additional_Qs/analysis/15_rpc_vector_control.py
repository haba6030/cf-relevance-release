r"""Upgrade the R–PC independence control from scalar residualisation to vector projection removal.

    python3 additional_Qs/analysis/15_rpc_vector_control.py

The canonical `analyze_rq3_joint_rpc_iut.py` builds a **scalar** PC1 from the norm values and
residualises the activations on it (`shared_pc_train_held`, line 67).

    PC1_i = (z(Δρ_i) + sgn(r)·z(ΔC_i)) / √2

So `R | PC` does not remove the $P$/$C$ **axes** inside the activations. `13_pc_projection_control.py`
showed that $v_P$ and $v_C$ can be used as vectors, so the same control is applied on the R side.

| mode | `R \| PC` | `PC \| R` |
|---|---|---|
| `scalar` (canonical) | residualise activations on the PC1 scalar | residualise on the intercept (centre) |
| `vector` | project span{v_P, v_C} out of the activations | project the v_R axis out of the activations |

Inputs are the canonical ones: the `delta_h3` activation contrast, the `delta_h3` norm contrast,
and the layer selections in `joint_pcr_iut_no_holm/canonical_r_layer_selections.csv`.
The script checks that `scalar` reproduces the canonical verdicts.

Neither the canonical scripts nor their outputs are modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
for p in (str(PROJ / "analysis" / "rq3"), str(PROJ)):
    if p not in sys.path:
        sys.path.insert(0, p)

from analyze_rq3_joint_pcr_iut import H3, I, R, h30_no_holm  # noqa: E402
from analyze_rq3_representation import norm_arrays, unit  # noqa: E402
import analyze_rq3_pc_hierarchy as PH  # noqa: E402

SEED, N_MC, MAX_EXACT = 20260824, 200_000, 20
RUNS = {"15": PROJ / "results" / "rq3_pcr_locked_20260825",
        "165": PROJ / "additional_Qs" / "rq3_run_server"}
OUT = PROJ / "additional_Qs" / "rpc_vector_control"
FAMILY = {"Qwen2.5-Math-1.5B": "Qwen", "Qwen3-4B": "Qwen",
          "Qwen3-4B-Instruct-2507": "Qwen", "Phi-4-mini-instruct": "Phi",
          "Phi-4-mini-reasoning": "Phi", "gemma-3-4b-it": "Gemma"}


def sign_flip(values: np.ndarray, alternative: str = "greater") -> float:
    v = np.asarray(values, dtype=float)
    if len(v) <= MAX_EXACT:
        return PH.exact_sign_flip(v, alternative)
    rng = np.random.default_rng(SEED)
    obs, hits = float(v.mean()), 0
    for lo in range(0, N_MC, 20_000):
        s = rng.choice((-1.0, 1.0), size=(min(20_000, N_MC - lo), len(v)))
        null = (s @ v) / len(v)
        hits += int((null >= obs - 1e-15).sum() if alternative == "greater"
                    else (np.abs(null) >= abs(obs) - 1e-15).sum())
    return (hits + 1) / (N_MC + 1)


def shared_pc(p_train, c_train, p_held, c_held):
    """Same construction as the canonical `shared_pc_train_held`."""
    p_mean, c_mean = float(p_train.mean()), float(c_train.mean())
    p_sd, c_sd = float(p_train.std(ddof=1)), float(c_train.std(ddof=1))
    pz, cz = (p_train - p_mean) / p_sd, (c_train - c_mean) / c_sd
    orientation = 1.0 if float(np.corrcoef(pz, cz)[0, 1]) >= 0 else -1.0
    return ((pz + orientation * cz) / np.sqrt(2.0),
            (((p_held - p_mean) / p_sd) + orientation * ((c_held - c_mean) / c_sd)) / np.sqrt(2.0))


def project_out(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """basis is (k, d) orthonormal; subtract the projection onto its span."""
    return x - (x @ basis.T) @ basis


def run_cell(activation, delta_p, delta_c, selected, mode):
    """Score both targets by LOIO. activation is [item, layer, dim]."""
    n = len(activation)
    contributions = {"R": [], "PC": []}
    removed, pc_on_r = [], []
    for held in range(n):
        train = np.flatnonzero(np.arange(n) != held)
        layer = int(selected[held])
        x_train = np.asarray(activation[train, layer], dtype=np.float64)
        x_held = np.asarray(activation[[held], layer], dtype=np.float64)
        pc_train, pc_held = shared_pc(delta_p[train], delta_c[train],
                                      delta_p[[held]], delta_c[[held]])

        # ── target R, control PC. The mean contrast is the target, so no centring.
        if mode == "scalar":
            denominator = float(pc_train @ pc_train)
            beta_a = (pc_train @ x_train) / denominator
            xr_train = x_train - np.outer(pc_train, beta_a)
            xr_held = x_held - np.outer(pc_held, beta_a)
            y_train = np.ones(len(train)) - pc_train * float(pc_train.sum() / denominator)
            y_held = np.ones(1) - pc_held * float(pc_train.sum() / denominator)
        else:
            centered = x_train - x_train.mean(axis=0, keepdims=True)
            v_p = unit(centered.T @ (delta_p[train] - delta_p[train].mean()))
            v_c = unit(centered.T @ (delta_c[train] - delta_c[train].mean()))
            basis = np.linalg.qr(np.stack([v_p, v_c]).T)[0].T
            xr_train, xr_held = project_out(x_train, basis), project_out(x_held, basis)
            y_train, y_held = np.ones(len(train)), np.ones(1)
            # v_R is the **uncentred** mean contrast; centered.mean is zero by definition.
            v_r = unit(x_train.mean(axis=0))
            removed.append(float(np.linalg.norm(basis @ v_r)))
        contributions["R"].append(score(xr_train, xr_held, y_train, y_held))

        # ── target PC, control R. Both modes centre; the vector mode also removes the v_R axis.
        centered = x_train - x_train.mean(axis=0, keepdims=True)
        held_centered = x_held - x_train.mean(axis=0, keepdims=True)
        if mode == "vector":
            v_r = unit(x_train.mean(axis=0))[None]
            centered, held_centered = project_out(centered, v_r), project_out(held_centered, v_r)
        y_train = pc_train - pc_train.mean()
        y_held = pc_held - pc_train.mean()
        contributions["PC"].append(score(centered, held_centered, y_train, y_held))
        if mode == "vector":
            # symmetric diagnostic: the share of the PC-predicting direction lying on the v_R axis
            base = x_train - x_train.mean(axis=0, keepdims=True)
            v_pc = unit(base.T @ (pc_train - pc_train.mean()))
            pc_on_r.append(float(abs(v_pc @ unit(x_train.mean(axis=0)))))

    return (contributions,
            float(np.mean(removed)) if removed else float("nan"),
            float(np.mean(pc_on_r)) if pc_on_r else float("nan"))


def score(x_train, x_held, y_train, y_held) -> float:
    direction = unit(x_train.T @ y_train)
    train_score = x_train @ direction
    denominator = float(train_score @ train_score)
    slope = float(train_score @ y_train / denominator) if denominator > 1e-12 else 0.0
    return float(y_held[0] * (x_held @ direction * slope)[0])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, root in RUNS.items():
        eligible = sorted(h30_no_holm(root).model_id.unique())
        selections = pd.read_csv(
            root / "analysis" / "joint_pcr_iut_no_holm" / "canonical_r_layer_selections.csv"
        ).set_index(["model_id", "task"])
        print(f"[{tag}] {len(eligible)} models pass H3-0 · {len(selections)} layer-selection rows")
        for model_id in eligible:
            slug = model_id.replace("/", "__")
            model = model_id.split("/")[-1]
            norming = pd.read_csv(root / "norming" / f"norm_rq3_{slug}.csv")
            metadata = json.loads(
                (root / "activations" / slug / "task2_metadata.json").read_text())
            p_values, c_values = norm_arrays(norming, metadata["labels"])
            delta_p = p_values[:, R, 1] - p_values[:, I, 1]
            delta_c = c_values[:, R, 1] - c_values[:, I, 1]
            delta_p2 = p_values[:, R, 0] - p_values[:, I, 0]
            delta_c2 = c_values[:, R, 0] - c_values[:, I, 0]
            for task in (2, 3):
                if (model_id, task) not in selections.index:
                    continue
                selected = np.asarray(
                    json.loads(selections.loc[(model_id, task)].selected_layers), dtype=int)
                array = np.load(root / "activations" / slug / f"task{task}_cellmean.npy",
                                mmap_mode="r")
                h3 = np.asarray(array[:, R, H3] - array[:, I, H3], dtype=np.float32)
                h2 = np.asarray(array[:, R, H3 - 1] - array[:, I, H3 - 1], dtype=np.float32)
                grids = {"delta_h3": (h3, delta_p, delta_c),
                         "did": (h3 - h2, delta_p - delta_p2, delta_c - delta_c2)}
                for contrast, (delta, dp, dc) in grids.items():
                    for mode in ("scalar", "vector"):
                        contributions, overlap, pc_overlap = run_cell(
                            delta, dp, dc, selected, mode)
                        for target in ("R", "PC"):
                            v = np.asarray(contributions[target])
                            rows.append({
                                "run": tag, "family": FAMILY.get(model, "?"),
                                "model": model, "task": task, "target": target,
                                "contrast": contrast, "mode": mode,
                                "n_items": len(delta),
                                "mean_covariance": float(v.mean()),
                                "p_flip_one": sign_flip(v, "greater"),
                                "v_R_in_PC_plane": overlap,
                                "v_PC_on_v_R": pc_overlap,
                            })
                print(f"  [{tag:>3}] {model:24s} task{task} done", flush=True)

    cells = pd.DataFrame(rows)
    cells["pass"] = (cells.mean_covariance > 0) & (cells.p_flip_one < 0.05)
    cells.to_csv(OUT / "cells.csv", index=False)

    print("\n" + "=" * 70)
    for tag in ("15", "165"):
        print(f"\n── {tag} items")
        for contrast in ("delta_h3", "did"):
            print(f"   [{contrast}]")
            for target, label in (("R", "R | PC"), ("PC", "PC | R")):
                for mode in ("scalar", "vector"):
                    s = cells[(cells.run == tag) & (cells.target == target)
                              & (cells["mode"] == mode) & (cells.contrast == contrast)]
                    print(f"      {label:8s} {mode:7s}: {int(s['pass'].sum())}/{len(s)} · "
                          f"mean covariance {s.mean_covariance.mean():+.4g}")
            s = cells[(cells.run == tag) & (cells["mode"] == "vector")
                      & (cells.contrast == contrast)]
            print(f"      removal strength: share of v_R in the PC plane {s.v_R_in_PC_plane.mean():.1%} · "
                  f"share of v_PC on v_R {s.v_PC_on_v_R.mean():.1%}")

    print("\n── per model (vector · delta_h3)")
    for tag in ("15", "165"):
        for target in ("R", "PC"):
            s = cells[(cells.run == tag) & (cells.target == target)
                      & (cells["mode"] == "vector") & (cells.contrast == "delta_h3")]
            detail = " ".join(
                f"{m.split('-')[0][:8]}:{int(g['pass'].sum())}/2" for m, g in s.groupby("model"))
            print(f"   {tag:>3} items {target:2s} | {detail}")
    print(f"\n[saved] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
