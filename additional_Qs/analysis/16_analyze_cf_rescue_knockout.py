#!/usr/bin/env python3
"""Analyze cross-frame conditional CF rescue and knockout shards."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260830
N_MC = 200_000
TARGETS = ("R_given_PC", "PC_given_R")
CONTROLS = {
    "R_given_PC": ("random_R_given_PC", "shuffle_R_given_PC"),
    "PC_given_R": ("random_PC_given_R", "shuffle_PC_given_R"),
}


def sign_flip(values: np.ndarray, alternative: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = float(values.mean())
    if len(values) <= 20:
        null = np.asarray([
            np.mean(values * np.asarray(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ])
        if alternative == "greater":
            return float(np.mean(null >= observed - 1e-15))
        if alternative == "less":
            return float(np.mean(null <= observed + 1e-15))
        return float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    rng = np.random.default_rng(SEED)
    hits = 0
    for start in range(0, N_MC, 20_000):
        size = min(20_000, N_MC - start)
        null = (rng.choice((-1.0, 1.0), size=(size, len(values))) @ values) / len(values)
        if alternative == "greater":
            hits += int((null >= observed - 1e-15).sum())
        elif alternative == "less":
            hits += int((null <= observed + 1e-15).sum())
        else:
            hits += int((np.abs(null) >= abs(observed) - 1e-15).sum())
    return (hits + 1) / (N_MC + 1)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def summarize(values: np.ndarray, alternative: str, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(values, seed)
    return {
        "n": len(values), "mean_delta_cf": float(values.mean()),
        "median_delta_cf": float(np.median(values)), "ci_lo": lo, "ci_hi": hi,
        "p_flip_one": sign_flip(values, alternative),
        "positive": int((values > 0).sum()), "negative": int((values < 0).sum()),
    }


def load_cells(root: Path) -> pd.DataFrame:
    paths = sorted((root / "shards").glob("*/task*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"No rescue shards below {root}")
    data = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if data.duplicated([
        "model_id", "task", "label", "select_frame", "eval_frame",
        "stage", "direction", "alpha", "dose",
    ]).any():
        raise RuntimeError("Duplicate intervention cells")
    data["base_post"] = data.lp_base / data.nt_base
    data["h2_post"] = data.lp_conj_h2 / data.nt_conj_h2
    data["h3_post"] = data.lp_conj_h3 / data.nt_conj_h3
    data["cf_post"] = data.h3_post - data.base_post
    data["gap_post"] = data.h3_post - data.h2_post
    data["delta_cf"] = data.cf_post - data.baseline_cf_eval
    data["baseline_dod"] = data.baseline_gap_R_eval - data.baseline_gap_I_eval
    data["dod_post"] = data.gap_post - data.baseline_gap_I_eval
    data["delta_dod"] = data.dod_post - data.baseline_dod
    data["base_shift"] = (data.base_post - data.baseline_base_eval).abs()
    # Evaluation-frame sign agreement is not used to select the intervention,
    # but threshold-crossing claims require it. Continuous effects use all rows.
    data["eval_state_agrees"] = np.where(
        data.stage == "ADD", data.baseline_cf_eval <= 0, data.baseline_cf_eval > 0
    )
    data["crossed"] = np.where(
        data.stage == "ADD", data.cf_post > 0, data.cf_post <= 0
    ) & data.eval_state_agrees
    return data


def add_direction_gates(data: pd.DataFrame, project: Path) -> pd.DataFrame:
    gate_path = project / "additional_Qs" / "rpc_vector_control" / "cells.csv"
    gates = pd.read_csv(gate_path)
    gates = gates[(gates.run == 165) & (gates.contrast == "did")
                  & (gates["mode"] == "vector")]
    short_to_id = data[["model_id"]].drop_duplicates().assign(
        model=lambda x: x.model_id.str.split("/").str[-1]
    )
    gates = gates.merge(short_to_id, on="model", validate="many_to_one")
    gates["direction"] = gates.target.map({"R": "R_given_PC", "PC": "PC_given_R"})
    gates = gates[["model_id", "task", "direction", "pass", "p_flip_one"]].rename(
        columns={"pass": "direction_gate", "p_flip_one": "direction_gate_p"}
    )
    return data.merge(gates, on=["model_id", "task", "direction"], how="left")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()
    out = args.input_dir / "analysis"; out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(args.input_dir)

    guard_paths = sorted((args.input_dir / "guard_shards").glob("*/task*/*.csv"))
    if not guard_paths:
        raise FileNotFoundError("No conditional-CF fluency guard shards")
    guard_folds = pd.concat((pd.read_csv(path) for path in guard_paths), ignore_index=True)
    guard_folds.to_csv(out / "cf_rescue_fluency_folds.csv", index=False)
    guard_keys = ["model_id", "task", "mode", "direction", "alpha", "dose"]
    fluency = (guard_folds.groupby(guard_keys, dropna=False, as_index=False)
               .agg(fluency_n=("held", "nunique"),
                    fluency_increase_max=("increase", "max"),
                    fluency_pass=("passes", "all")))
    fluency.to_csv(out / "cf_rescue_fluency.csv", index=False)

    # Two-SD base-collapse guard, computed without intervention outcomes.
    baseline_units = cells.drop_duplicates(
        ["model_id", "task", "label", "eval_frame"]
    )
    thresholds = (baseline_units.groupby(["model_id", "task"])
                  .baseline_base_eval.std(ddof=1).mul(2).rename("base_guard_threshold")
                  .reset_index())
    cells = cells.merge(thresholds, on=["model_id", "task"], validate="many_to_one")
    cells["base_guard"] = cells.base_shift <= cells.base_guard_threshold
    target_cells = add_direction_gates(cells[cells.direction.isin(TARGETS)].copy(), args.project)
    other = cells[~cells.direction.isin(TARGETS)].copy()
    other["direction_gate"] = np.nan; other["direction_gate_p"] = np.nan
    cells = pd.concat([target_cells, other], ignore_index=True)
    cells.to_csv(out / "cf_rescue_knockout_cells.csv", index=False)

    rows = []
    group_columns = ["model_id", "task", "stage", "direction", "alpha", "dose"]
    for keys, group in cells.groupby(group_columns, dropna=False):
        alternative = "greater" if keys[2] == "ADD" else "less"
        # The two cross-frame folds are repeated measurements of the same item,
        # not independent observations.  Average them before every inferential
        # test so n never exceeds the 165 locked items.
        item_units = (group.groupby("label", as_index=False)
                      .agg(delta_cf=("delta_cf", "mean"),
                           delta_dod=("delta_dod", "mean")))
        result = summarize(item_units.delta_cf.to_numpy(), alternative, SEED + len(rows))
        stable = group[group.eval_state_agrees]
        result.update({
            **dict(zip(group_columns, keys)),
            "n_sign_stable": len(stable),
            "crossing_count": int(stable.crossed.sum()),
            "crossing_rate": float(stable.crossed.mean()) if len(stable) else np.nan,
            "mean_delta_dod": float(item_units.delta_dod.mean()),
            "base_guard_all": bool(group.base_guard.all()),
            "base_shift_max": float(group.base_shift.max()),
            "direction_gate": (bool(group.direction_gate.iloc[0])
                               if group.direction.isin(TARGETS).all() else np.nan),
            "direction_gate_p": (float(group.direction_gate_p.iloc[0])
                                 if group.direction.isin(TARGETS).all() else np.nan),
        })
        rows.append(result)
    tests = pd.DataFrame(rows)
    tests = tests.merge(
        fluency, left_on=["model_id", "task", "stage", "direction", "alpha", "dose"],
        right_on=["model_id", "task", "mode", "direction", "alpha", "dose"],
        how="left", validate="one_to_one",
    ).drop(columns="mode")

    # Paired target-vs-control specificity at the same item, split and dose.
    specificity = []
    index = ["model_id", "task", "label", "select_frame", "eval_frame", "stage", "alpha", "dose"]
    for target, controls in CONTROLS.items():
        t = cells[cells.direction == target][index + ["delta_cf"]].rename(
            columns={"delta_cf": "target_delta"})
        for control in controls:
            c = cells[cells.direction == control][index + ["delta_cf"]].rename(
                columns={"delta_cf": "control_delta"})
            paired = t.merge(c, on=index, validate="one_to_one")
            # Positive means the target is more hypothesis-consistent than control.
            paired["advantage"] = np.where(
                paired.stage == "ADD",
                paired.target_delta - paired.control_delta,
                paired.control_delta - paired.target_delta,
            )
            for keys, group in paired.groupby(["model_id", "task", "stage", "alpha", "dose"], dropna=False):
                item_units = group.groupby("label", as_index=False).advantage.mean()
                specificity.append({
                    **dict(zip(["model_id", "task", "stage", "alpha", "dose"], keys)),
                    "target": target, "control": control, "n": len(item_units),
                    "mean_advantage": float(item_units.advantage.mean()),
                    "p_flip_one": sign_flip(item_units.advantage.to_numpy(), "greater"),
                })
    specificity = pd.DataFrame(specificity)
    specificity.to_csv(out / "cf_rescue_knockout_specificity.csv", index=False)

    # A target passes only with the representation gate, primary effect,
    # both matched controls, and the item-level base-collapse guard.
    tests["specificity_random_p"] = np.nan; tests["specificity_shuffle_p"] = np.nan
    for idx, row in tests[tests.direction.isin(TARGETS)].iterrows():
        s = specificity[(specificity.model_id == row.model_id)
                        & (specificity.task == row.task)
                        & (specificity.stage == row.stage)
                        & (specificity.target == row.direction)]
        if pd.notna(row.alpha): s = s[s.alpha == row.alpha]
        else: s = s[s.dose == row.dose]
        for control, name in (("random", "specificity_random_p"),
                              ("shuffle", "specificity_shuffle_p")):
            q = s[s.control.str.startswith(control)]
            if len(q) == 1: tests.loc[idx, name] = float(q.iloc[0].p_flip_one)
    tests["pass"] = (
        tests.direction.isin(TARGETS)
        & tests.direction_gate.fillna(False).astype(bool)
        & (tests.p_flip_one < .05)
        & (tests.specificity_random_p < .05)
        & (tests.specificity_shuffle_p < .05)
        & tests.base_guard_all
        & tests.fluency_pass.fillna(False).astype(bool)
    )
    tests.to_csv(out / "cf_rescue_knockout_tests.csv", index=False)

    summary = (tests[tests.direction.isin(TARGETS)]
               .groupby(["stage", "direction", "alpha", "dose"], dropna=False)
               .agg(cells=("pass", "size"), passed=("pass", "sum"),
                    mean_delta_cf=("mean_delta_cf", "mean"),
                    mean_crossing=("crossing_rate", "mean"))
               .reset_index())
    summary.to_csv(out / "cf_rescue_knockout_summary.csv", index=False)

    lines = [
        "# Conditional CF rescue and knockout", "",
        "Selection uses one baseline frame and evaluation uses the other; the split is cross-swapped. ",
        "The two cross-frame estimates are averaged within item before inference (maximum n=165). ",
        "Primary outcome is normalized `conj_h3 - base`; DoD is secondary. Raw p-values are reported without Holm.", "",
        "| stage | direction | alpha | dose | cells passed | mean ΔCF | mean crossing |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.stage} | {row.direction} | "
            f"{'' if pd.isna(row.alpha) else f'{row.alpha:g}'} | "
            f"{'' if pd.isna(row.dose) else f'{row.dose:g}'} | "
            f"{int(row.passed)}/{int(row.cells)} | {row.mean_delta_cf:+.4f} | "
            f"{row.mean_crossing:.1%} |"
        )
    lines += ["", "A cell passes only if the direction gate, one-sided item sign-flip, ",
              "angle/constraint-matched random and shuffle comparisons, full-165-fold WikiText fluency guard, ",
              "and the base-collapse guard all pass.", ""]
    (out / "CF_RESCUE_KNOCKOUT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    manifest = {
        "input_rows": len(cells), "shards": len(list((args.input_dir / "shards").glob("*/task*/*.csv"))),
        "raw_p_no_holm": True, "hypothesis_count": int(len(tests[tests.direction.isin(TARGETS)])),
        "signflip_mc": N_MC, "seed": SEED,
        "inferential_unit": "item; two cross-frame estimates averaged within label",
        "fluency_threshold": "all 165 target folds have PPL increase < 10%",
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
