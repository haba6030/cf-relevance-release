#!/usr/bin/env python3
"""RQ3 H3-0 gate: decompose the old R|N effect into R|I and I|N."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260824
PAIRS = {"R-I": ("R", "I"), "I-N": ("I", "N"), "R-N": ("R", "N")}


def exact_signflip(values: np.ndarray, alternative: str = "two-sided") -> float:
    values = np.asarray(values, dtype=float)
    observed = values.mean()
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = (signs * values).mean(axis=1)
    if alternative == "greater":
        return float((null >= observed - 1e-15).mean())
    return float((np.abs(null) >= abs(observed) - 1e-15).mean())


def holm(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy())
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, values.iloc[index] * (len(values) - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n != 15 or not np.isfinite(values).all():
        raise ValueError(f"Expected 15 finite item values, got {n}")
    t = stats.ttest_1samp(values, 0.0)
    ci = stats.t.interval(0.95, n - 1, loc=values.mean(), scale=stats.sem(values))
    rng = np.random.default_rng(SEED)
    boot = rng.choice(values, size=(2000, n), replace=True).mean(axis=1)
    loo = [exact_signflip(np.delete(values, i)) for i in range(n)]
    nonzero = values[values != 0]
    sign_two = stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5, alternative="two-sided").pvalue if len(nonzero) else 1.0
    sign_positive = stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5, alternative="greater").pvalue if len(nonzero) else 1.0
    return {
        "n": n,
        "mean": values.mean(),
        "t": t.statistic,
        "p_t_two": t.pvalue,
        "ci_lo": ci[0],
        "ci_hi": ci[1],
        "boot_lo": np.quantile(boot, 0.025),
        "boot_hi": np.quantile(boot, 0.975),
        "p_flip_two": exact_signflip(values),
        "p_flip_one_positive": exact_signflip(values, "greater"),
        "positive": int((values > 0).sum()),
        "negative": int((values < 0).sum()),
        "zero": int((values == 0).sum()),
        "p_exact_sign_two": sign_two,
        "p_exact_sign_positive": sign_positive,
        "loo_worst_p_flip_two": max(loo),
        "loo_worst_index": int(np.argmax(loo)),
    }


def item_effects(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for arm in ("conj", "bare"):
        for metric in ("normalized", "raw"):
            d = data.copy()
            for hypothesis in ("h2", "h3"):
                if metric == "normalized":
                    d[hypothesis] = d[f"lp_{arm}_{hypothesis}"] / d[f"nt_{arm}_{hypothesis}"] - d.lp_base / d.nt_base
                else:
                    d[hypothesis] = d[f"lp_{arm}_{hypothesis}"] - d.lp_base
            grouped = (
                d.groupby(["model_id", "model", "task", "label", "scenario", "prefix"])[["h2", "h3"]]
                .mean().reset_index()
            )
            wide = grouped.pivot(index=["model_id", "model", "task", "label", "scenario"], columns="prefix", values=["h2", "h3"])
            wide.columns = [f"{hyp}_{prefix_}" for hyp, prefix_ in wide.columns]; wide = wide.reset_index()
            for contrast, (hi, lo) in PAIRS.items():
                wide[f"dod_{contrast}"] = (wide[f"h3_{hi}"] - wide[f"h3_{lo}"]) - (wide[f"h2_{hi}"] - wide[f"h2_{lo}"])
            wide["additivity_error"] = wide["dod_R-N"] - (wide["dod_R-I"] + wide["dod_I-N"])
            wide["arm"] = arm; wide["metric"] = metric; frames.append(wide)
    return pd.concat(frames, ignore_index=True)


def tests(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in items.groupby(["model_id", "model", "task", "arm", "metric"]):
        for contrast in PAIRS:
            result = summarize(group[f"dod_{contrast}"].to_numpy())
            result.update(
                model_id=keys[0], model=keys[1], task=keys[2], arm=keys[3], metric=keys[4],
                contrast=contrast,
                loo_worst_dropped=group.iloc[result["loo_worst_index"]].label,
            )
            rows.append(result)
    out = pd.DataFrame(rows)
    out["holm_task_p_flip_two"] = np.nan
    for _, index in out.groupby(["model_id", "arm", "metric", "contrast"]).groups.items():
        out.loc[index, "holm_task_p_flip_two"] = holm(out.loc[index, "p_flip_two"])
    return out


def gate_table(result: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    primary = result[(result.arm == "conj") & (result.metric == "normalized")].copy()
    rows = []
    eligible = []
    for (model_id, model), group in primary.groupby(["model_id", "model"]):
        r = group[group.contrast == "R-I"].sort_values("task")
        i = group[group.contrast == "I-N"].sort_values("task")
        rn = group[group.contrast == "R-N"].sort_values("task")
        r_sig = (r["mean"].to_numpy() > 0) & (r.holm_task_p_flip_two.to_numpy() < 0.05)
        i_sig = (i.holm_task_p_flip_two.to_numpy() < 0.05)
        # The registered table permits continuation when both components exist.
        if r_sig.all() and not i_sig.any():
            state = "pass_relevance"
            eligible.append(model_id)
        elif r_sig.all() and i_sig.any():
            state = "pass_split_effect"
            eligible.append(model_id)
        elif (not r_sig.any()) and i_sig.any():
            state = "stop_paragraph_only"
        else:
            state = "indeterminate"
        rows.append(
            {
                "model_id": model_id,
                "model": model,
                "state": state,
                "eligible_rq3b": model_id in eligible,
                "r_i_means": r["mean"].tolist(),
                "r_i_holm_p": r.holm_task_p_flip_two.tolist(),
                "i_n_means": i["mean"].tolist(),
                "i_n_holm_p": i.holm_task_p_flip_two.tolist(),
                "r_i_fraction_of_r_n": (r["mean"].to_numpy() / rn["mean"].replace(0, np.nan).to_numpy()).tolist(),
                "i_n_fraction_of_r_n": (i["mean"].to_numpy() / rn["mean"].replace(0, np.nan).to_numpy()).tolist(),
                "warning": "I-N non-significance is not equivalence evidence",
            }
        )
    return pd.DataFrame(rows), {"eligible_models": eligible, "rule": "docs/EXPERIMENT_RQ3.md 5.0; strict across L2C and L3C"}


def write_report(result: pd.DataFrame, gates: pd.DataFrame, path: Path) -> None:
    lines = [
        "# RQ3 H3-0 gate",
        "",
        "Unit: item (n=15). Twelve frame/question variants are averaged within prefix.",
        "Primary test: exact two-sided sign-flip; Holm only across L2C/L3C within contrast.",
        "`I-N` non-significance is descriptive, not an equivalence result.",
        "",
        "## Gate",
        "",
        "| Model | State | RQ3-B eligible | R-I/R-N (L2,L3) | I-N/R-N (L2,L3) |",
        "|---|---|---|---|---|",
    ]
    for row in gates.itertuples(index=False):
        r_fraction = ", ".join("nan" if not np.isfinite(value) else f"{value:.2f}" for value in row.r_i_fraction_of_r_n)
        i_fraction = ", ".join("nan" if not np.isfinite(value) else f"{value:.2f}" for value in row.i_n_fraction_of_r_n)
        lines.append(f"| {row.model} | {row.state} | {row.eligible_rq3b} | {r_fraction} | {i_fraction} |")
    lines += [
        "",
        "## Conjunction-arm tests",
        "",
        "| Model | Task | Contrast | Mean | 95% t CI | p sign-flip | Holm p | sign +/− | LOO worst p |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result[(result.arm == "conj") & (result.metric == "normalized")].itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.task} | {row.contrast} | {row.mean:.4f} | "
            f"[{row.ci_lo:.4f}, {row.ci_hi:.4f}] | {row.p_flip_two:.4f} | "
            f"{row.holm_task_p_flip_two:.4f} | {row.positive}/{row.negative} | "
            f"{row.loo_worst_p_flip_two:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.input_dir / "behavior").glob("rq3_behavior_*.csv"))
    files = [path for path in files if "__chung-" not in path.name]
    if not files:
        raise FileNotFoundError("No complete RQ3 behavior files")
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    items = item_effects(data)
    max_additivity = float(items.additivity_error.abs().max())
    if max_additivity > 1e-10:
        raise RuntimeError(f"DoD additivity check failed: {max_additivity}")
    result = tests(items)
    gates, gate_json = gate_table(result)
    out = args.input_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "h30_item_effects.csv", index=False)
    result.to_csv(out / "h30_tests.csv", index=False)
    gates.to_csv(out / "h30_gate.csv", index=False)
    gate_json["max_additivity_error"] = max_additivity
    (out / "h30_gate.json").write_text(json.dumps(gate_json, ensure_ascii=False, indent=2) + "\n")
    write_report(result, gates, out / "h30_report.md")
    print(gates[["model", "state", "eligible_rq3b"]].to_string(index=False))
    print(f"max additivity error={max_additivity:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
