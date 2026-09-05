#!/usr/bin/env python3
"""RQ3 H3d/H3e analysis for item-sharded intervention outputs."""

from __future__ import annotations

import argparse
import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@lru_cache(maxsize=None)
def sign_matrix(n: int) -> np.ndarray:
    """Cache the exact sign grid reused by every model/task/direction cell."""
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=np.float64)


def signflip(values: np.ndarray, alternative: str = "two-sided") -> float:
    values = np.asarray(values, dtype=float)
    observed = values.mean()
    signs = sign_matrix(len(values))
    null = (signs * values).mean(axis=1)
    if alternative == "less":
        return float((null <= observed + 1e-15).mean())
    if alternative == "greater":
        return float((null >= observed - 1e-15).mean())
    return float((np.abs(null) >= abs(observed) - 1e-15).mean())


def summarize(values: np.ndarray, alternative: str) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) != 15 or not np.isfinite(values).all():
        raise ValueError(f"Expected 15 finite effects, got {len(values)}")
    ci = stats.t.interval(0.95, 14, loc=values.mean(), scale=stats.sem(values))
    rng = np.random.default_rng(20260824)
    boot = rng.choice(values, size=(2000, 15), replace=True).mean(axis=1)
    t = stats.ttest_1samp(values, 0.0)
    nonzero = values[values != 0]
    sign_two = stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5, alternative="two-sided").pvalue if len(nonzero) else 1.0
    sign_direction = stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5, alternative="greater" if alternative == "greater" else "less").pvalue if len(nonzero) else 1.0
    loo = [signflip(np.delete(values, index)) for index in range(len(values))]
    return {
        "n": 15, "mean": values.mean(), "ci_lo": ci[0], "ci_hi": ci[1],
        "t": t.statistic, "p_t_two": t.pvalue,
        "boot_lo": np.quantile(boot, .025), "boot_hi": np.quantile(boot, .975),
        "p_flip_two": signflip(values), "p_flip_directional": signflip(values, alternative),
        "p_exact_sign_two": sign_two, "p_exact_sign_directional": sign_direction,
        "loo_worst_p_flip_two": max(loo), "loo_worst_index": int(np.argmax(loo)),
        "positive": int((values > 0).sum()), "negative": int((values < 0).sum()),
        "zero": int((values == 0).sum()),
    }


def holm(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy()); adjusted = np.empty(len(values)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values.iloc[index] * (len(values) - rank)))
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def margins(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for hypothesis in ("h2", "h3"):
        out[hypothesis] = (
            out[f"lp_conj_{hypothesis}"] / out[f"nt_conj_{hypothesis}"]
            - out.lp_base / out.nt_base
        )
    out["gap"] = out.h3 - out.h2
    out["gap_raw"] = out.lp_conj_h3 - out.lp_conj_h2
    out["base_norm"] = out.lp_base / out.nt_base
    out["base_raw"] = out.lp_base
    return out


def baseline_cells(behavior: pd.DataFrame) -> pd.DataFrame:
    d = margins(behavior)
    return d.groupby(["model_id", "task", "label", "prefix"])[["h2", "h3", "gap", "gap_raw", "base_norm", "base_raw"]].mean().reset_index()


def guard_tables(
    abl_items: pd.DataFrame,
    add_items: pd.DataFrame,
    baseline: pd.DataFrame,
    guards_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine fluency and base-collapse guards without inspecting DoD effects."""
    threshold = baseline.groupby(["model_id", "task", "prefix"]).base_raw.std(ddof=1).mul(2).rename("base_limit").reset_index()

    fluency_files = sorted(guards_dir.glob("*/fluency.csv"))
    if not fluency_files:
        raise FileNotFoundError("No fluency guard files")
    fluency = pd.concat((pd.read_csv(path) for path in fluency_files), ignore_index=True)

    # abl_items stores the maximum shift across I/R.  Use the stricter of the
    # two prefix-specific thresholds for that task.
    strict_limit = threshold.groupby(["model_id", "task"]).base_limit.min().rename("base_limit").reset_index()
    abl_guard = abl_items.merge(strict_limit, on=["model_id", "task"], validate="many_to_one")
    abl_guard["base_pass"] = abl_guard.base_shift_max <= abl_guard.base_limit
    abl_f = fluency[fluency["mode"] == "ABL"][["model_id", "task", "direction", "dose", "ablate_scope", "passes", "increase"]].rename(columns={"passes": "fluency_pass", "increase": "fluency_increase"})
    abl_guard = abl_guard.merge(abl_f, on=["model_id", "task", "direction", "dose", "ablate_scope"], how="left", validate="many_to_one")
    abl_guard["guard_pass"] = abl_guard.base_pass & abl_guard.fluency_pass.fillna(False)

    add_limit = threshold[threshold.prefix == "I"][["model_id", "task", "base_limit"]]
    add_guard = add_items.merge(add_limit, on=["model_id", "task"], validate="many_to_one")
    add_guard["base_pass"] = add_guard.base_shift <= add_guard.base_limit
    add_f = fluency[fluency["mode"] == "ADD"][["model_id", "task", "direction", "alpha", "passes", "increase"]].rename(columns={"passes": "fluency_pass", "increase": "fluency_increase"})
    add_guard = add_guard.merge(add_f, on=["model_id", "task", "direction", "alpha"], how="left", validate="many_to_one")
    add_guard["fluency_pass"] = add_guard.fluency_pass.fillna(False)
    add_guard["guard_pass"] = add_guard.base_pass & add_guard.fluency_pass.fillna(False)
    safe = (
        add_guard.groupby(["model_id", "task", "direction", "alpha"], as_index=False)
        .agg(base_pass=("base_pass", "all"), fluency_pass=("fluency_pass", "all"), guard_pass=("guard_pass", "all"), max_base_shift=("base_shift", "max"), base_limit=("base_limit", "first"), fluency_increase=("fluency_increase", "max"))
    )
    safe["selected_safe"] = False
    passing = safe[safe.guard_pass]
    if not passing.empty:
        selected = passing.groupby(["model_id", "task", "direction"]).alpha.idxmax()
        safe.loc[selected, "selected_safe"] = True
    return abl_guard, add_guard, safe


def item_effects(intervention: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = margins(intervention)
    abl = d[d.stage == "ABL"].groupby(
        ["model_id", "task", "label", "direction", "dose", "ablate_scope", "prefix"]
    )[["gap", "gap_raw", "base_norm", "base_raw"]].mean().reset_index()
    abl_wide = abl.pivot(
        index=["model_id", "task", "label", "direction", "dose", "ablate_scope"], columns="prefix", values=["gap", "gap_raw", "base_norm", "base_raw"]
    )
    abl_wide.columns = [f"{metric}_{prefix_}" for metric, prefix_ in abl_wide.columns]
    abl_wide = abl_wide.reset_index()
    base_wide = baseline[baseline.prefix.isin(["I", "R"])].pivot(
        index=["model_id", "task", "label"], columns="prefix", values=["gap", "gap_raw", "base_norm", "base_raw"]
    )
    base_wide.columns = [f"baseline_{metric}_{prefix_}" for metric, prefix_ in base_wide.columns]
    base_wide = base_wide.reset_index()
    abl_wide = abl_wide.merge(base_wide, on=["model_id", "task", "label"], validate="many_to_one")
    abl_wide["dod_ablate"] = abl_wide.gap_R - abl_wide.gap_I
    abl_wide["dod_baseline"] = abl_wide.baseline_gap_R - abl_wide.baseline_gap_I
    abl_wide["delta_dod"] = abl_wide.dod_ablate - abl_wide.dod_baseline
    abl_wide["dod_ablate_raw"] = abl_wide.gap_raw_R - abl_wide.gap_raw_I
    abl_wide["dod_baseline_raw"] = abl_wide.baseline_gap_raw_R - abl_wide.baseline_gap_raw_I
    abl_wide["delta_dod_raw"] = abl_wide.dod_ablate_raw - abl_wide.dod_baseline_raw
    abl_wide["base_shift_max"] = np.maximum(
        (abl_wide.base_raw_R - abl_wide.baseline_base_raw_R).abs(),
        (abl_wide.base_raw_I - abl_wide.baseline_base_raw_I).abs(),
    )

    add = d[d.stage == "ADD"].groupby(
        ["model_id", "task", "label", "direction", "alpha"]
    )[["gap", "gap_raw", "base_norm", "base_raw"]].mean().reset_index()
    baseline_i = baseline[baseline.prefix == "I"][
        ["model_id", "task", "label", "gap", "gap_raw", "base_norm", "base_raw"]
    ].rename(columns={"gap": "baseline_gap_I", "gap_raw": "baseline_gap_raw_I", "base_norm": "baseline_base_I", "base_raw": "baseline_base_raw_I"})
    add = add.merge(baseline_i, on=["model_id", "task", "label"], validate="many_to_one")
    add["induced_dod"] = add.gap - add.baseline_gap_I
    add["induced_dod_raw"] = add.gap_raw - add.baseline_gap_raw_I
    add["base_shift"] = (add.base_raw - add.baseline_base_raw_I).abs()
    return abl_wide, add


def prompt_sensitivity(intervention: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    """Count the 12 frame/question signs within every held item."""
    base = margins(behavior)
    base = base.groupby(["model_id", "task", "label", "prefix", "frame", "qvar"])["gap"].mean().reset_index()
    changed = margins(intervention)
    rows = []
    abl = changed[changed.stage == "ABL"].groupby(["model_id", "task", "label", "direction", "dose", "ablate_scope", "prefix", "frame", "qvar"])["gap"].mean().reset_index()
    # Keep the intervention-cell identifiers through the wide reshape.  They
    # are required by the per-cell prompt-sensitivity aggregation below.
    abl = abl.pivot(
        index=[
            "model_id", "task", "label", "direction", "dose",
            "ablate_scope", "frame", "qvar",
        ],
        columns="prefix",
        values="gap",
    ).reset_index()
    base_abl = base[base.prefix.isin(["I", "R"])].pivot(index=["model_id", "task", "label", "frame", "qvar"], columns="prefix", values="gap").reset_index()
    abl = abl.merge(base_abl, on=["model_id", "task", "label", "frame", "qvar"], suffixes=("_changed", "_base"), validate="many_to_one")
    abl["effect"] = (abl.R_changed - abl.I_changed) - (abl.R_base - abl.I_base)
    for keys, group in abl.groupby(["model_id", "task", "label", "direction", "dose", "ablate_scope"]):
        rows.append({"model_id": keys[0], "task": keys[1], "label": keys[2], "stage": "ABL", "direction": keys[3], "dose": keys[4], "ablate_scope": keys[5], "alpha": np.nan, "hypothesized_sign": "negative", "negative_variants": int((group.effect < 0).sum()), "positive_variants": int((group.effect > 0).sum()), "zero_variants": int((group.effect == 0).sum()), "n_variants": len(group)})

    add = changed[changed.stage == "ADD"].groupby(["model_id", "task", "label", "direction", "alpha", "frame", "qvar"])["gap"].mean().reset_index()
    base_i = base[base.prefix == "I"].drop(columns="prefix").rename(columns={"gap": "baseline_gap"})
    add = add.merge(base_i, on=["model_id", "task", "label", "frame", "qvar"], validate="many_to_one")
    add["effect"] = add.gap - add.baseline_gap
    for keys, group in add.groupby(["model_id", "task", "label", "direction", "alpha"]):
        rows.append({"model_id": keys[0], "task": keys[1], "label": keys[2], "stage": "ADD", "direction": keys[3], "alpha": keys[4], "hypothesized_sign": "positive", "negative_variants": int((group.effect < 0).sum()), "positive_variants": int((group.effect > 0).sum()), "zero_variants": int((group.effect == 0).sum()), "n_variants": len(group)})
    result = pd.DataFrame(rows)
    # Full: frame 2 x qvar 6 = 12.  Registered server-lite runs may retain fewer
    # qvars, but every cell must still have the same non-degenerate frame grid.
    counts = sorted(result.n_variants.unique().tolist())
    if len(counts) != 1 or counts[0] < 2:
        raise RuntimeError(
            f"Prompt-sensitivity cells need one uniform >=2-variant grid, got {counts}"
        )
    return result


def test_tables(abl: pd.DataFrame, add: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    abl_rows = []
    for keys, group in abl.groupby(["model_id", "task", "direction", "dose", "ablate_scope"]):
        result = summarize(group.delta_dod.to_numpy(), "less")
        rng = np.random.default_rng(20260824); ratio_boot = []
        for _ in range(5000):
            index = rng.integers(0, len(group), len(group)); denominator = group.dod_baseline.to_numpy()[index].mean()
            ratio_boot.append(group.dod_ablate.to_numpy()[index].mean() / denominator if abs(denominator) > 1e-12 else np.nan)
        baseline_mean = group.dod_baseline.mean()
        result.update(model_id=keys[0], model=keys[0].split("/")[-1], task=keys[1], direction=keys[2], dose=keys[3], ablate_scope=keys[4], residual_ratio=group.dod_ablate.mean() / baseline_mean if abs(baseline_mean) > 1e-12 else np.nan, residual_ratio_boot_lo=np.nanquantile(ratio_boot, .025), residual_ratio_boot_hi=np.nanquantile(ratio_boot, .975), max_base_shift=group.base_shift_max.max(), loo_worst_dropped=group.iloc[result["loo_worst_index"]].label)
        abl_rows.append(result)
    abl_tests = pd.DataFrame(abl_rows)
    abl_tests["holm_task_p"] = np.nan
    for _, index in abl_tests.groupby(["model_id", "direction", "dose", "ablate_scope"]).groups.items():
        abl_tests.loc[index, "holm_task_p"] = holm(abl_tests.loc[index, "p_flip_two"])

    add_rows = []
    for keys, group in add.groupby(["model_id", "task", "direction", "alpha"]):
        result = summarize(group.induced_dod.to_numpy(), "greater")
        result.update(model_id=keys[0], model=keys[0].split("/")[-1], task=keys[1], direction=keys[2], alpha=keys[3], max_base_shift=group.base_shift.max(), loo_worst_dropped=group.iloc[result["loo_worst_index"]].label)
        add_rows.append(result)
    add_tests = pd.DataFrame(add_rows)
    add_tests["holm_task_p"] = np.nan
    for _, index in add_tests.groupby(["model_id", "direction", "alpha"]).groups.items():
        add_tests.loc[index, "holm_task_p"] = holm(add_tests.loc[index, "p_flip_two"])
    monotonic = []
    for keys, group in add.groupby(["model_id", "task", "direction", "label"]):
        rho = stats.spearmanr(np.r_[0.0, group.alpha.to_numpy()], np.r_[0.0, group.induced_dod.to_numpy()]).statistic
        monotonic.append(dict(model_id=keys[0], task=keys[1], direction=keys[2], label=keys[3], alpha_spearman=rho))
    monotonic = pd.DataFrame(monotonic)
    return abl_tests, add_tests, monotonic


def raw_test_tables(abl: pd.DataFrame, add: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    abl_rows, add_rows = [], []
    for keys, group in abl.groupby(["model_id", "task", "direction", "dose", "ablate_scope"]):
        result = summarize(group.delta_dod_raw.to_numpy(), "less")
        result.update(model_id=keys[0], model=keys[0].split("/")[-1], task=keys[1], direction=keys[2], loo_worst_dropped=group.iloc[result["loo_worst_index"]].label)
        abl_rows.append(result)
    for keys, group in add.groupby(["model_id", "task", "direction", "alpha"]):
        result = summarize(group.induced_dod_raw.to_numpy(), "greater")
        result.update(model_id=keys[0], model=keys[0].split("/")[-1], task=keys[1], direction=keys[2], alpha=keys[3], loo_worst_dropped=group.iloc[result["loo_worst_index"]].label)
        add_rows.append(result)
    abl_out, add_out = pd.DataFrame(abl_rows), pd.DataFrame(add_rows)
    abl_out["holm_task_p"] = np.nan; add_out["holm_task_p"] = np.nan
    for _, index in abl_out.groupby(["model_id", "direction"]).groups.items(): abl_out.loc[index, "holm_task_p"] = holm(abl_out.loc[index, "p_flip_two"])
    for _, index in add_out.groupby(["model_id", "direction", "alpha"]).groups.items(): add_out.loc[index, "holm_task_p"] = holm(add_out.loc[index, "p_flip_two"])
    return abl_out, add_out


def monotonicity_tests(monotonic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in monotonic.groupby(["model_id", "task", "direction"]):
        result = summarize(group.alpha_spearman.fillna(0).to_numpy(), "greater")
        result.update(model_id=keys[0], model=keys[0].split("/")[-1], task=keys[1], direction=keys[2], loo_worst_dropped=group.iloc[result["loo_worst_index"]].label)
        rows.append(result)
    out = pd.DataFrame(rows); out["holm_task_p"] = np.nan
    for _, index in out.groupby(["model_id", "direction"]).groups.items():
        out.loc[index, "holm_task_p"] = holm(out.loc[index, "p_flip_two"])
    return out


def safe_monotonicity_items(add: pd.DataFrame, safe: pd.DataFrame) -> pd.DataFrame:
    maxima = safe[safe.guard_pass].groupby(["model_id", "task", "direction"]).alpha.max().rename("alpha_safe").reset_index()
    eligible = add.merge(maxima, on=["model_id", "task", "direction"], how="inner", validate="many_to_one")
    eligible = eligible[eligible.alpha <= eligible.alpha_safe]
    rows = []
    for keys, group in eligible.groupby(["model_id", "task", "direction", "label"]):
        rho = stats.spearmanr(np.r_[0.0, group.alpha.to_numpy()], np.r_[0.0, group.induced_dod.to_numpy()]).statistic
        rows.append({"model_id": keys[0], "task": keys[1], "direction": keys[2], "label": keys[3], "alpha_safe": group.alpha_safe.iloc[0], "alpha_spearman": rho})
    return pd.DataFrame(rows)


def selected_safe_holm(frame: pd.DataFrame) -> pd.DataFrame:
    """Holm-correct the two task endpoints after outcome-blind safe-alpha selection."""
    output = frame.copy()
    output["selected_safe_holm_p"] = np.nan
    selected = output[output.selected_safe]
    for _, index in selected.groupby(["model_id", "direction"]).groups.items():
        output.loc[index, "selected_safe_holm_p"] = holm(output.loc[index, "p_flip_two"])
    return output


def write_report(abl: pd.DataFrame, add: pd.DataFrame, mono: pd.DataFrame, mono_tests: pd.DataFrame, safe: pd.DataFrame, path: Path) -> None:
    lines = [
        "# RQ3-B primary intervention report", "",
        "Primary inference uses only cells passing both fluency (<10% PPL increase) and base-collapse (<=2 item SD) guards.", "",
        "## ABL", "", "| Model | Task | direction | delta DoD | 95% CI | p flip | Holm | residual ratio (boot CI) | guard | sign +/− |", "|---|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in abl.itertuples(index=False):
        lines.append(f"| {row.model} | {row.task} | {row.direction} | {row.mean:.4f} | [{row.ci_lo:.4f}, {row.ci_hi:.4f}] | {row.p_flip_two:.4f} | {row.holm_task_p:.4f} | {row.residual_ratio:.3f} [{row.residual_ratio_boot_lo:.3f}, {row.residual_ratio_boot_hi:.3f}] | {bool(row.guard_pass)} | {row.positive}/{row.negative} |")
    lines += ["", "## ADD", "", "| Model | Task | direction | alpha | induced DoD | 95% CI | p flip | fixed-alpha Holm | selected-safe Holm | sign +/− | mean item rho(alpha) |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    mean_mono = mono.groupby(["model_id", "task", "direction"]).alpha_spearman.mean()
    for row in add.itertuples(index=False):
        rho = mean_mono.get((row.model_id, row.task, row.direction), np.nan)
        selected_holm = "—" if pd.isna(row.selected_safe_holm_p) else f"{row.selected_safe_holm_p:.4f}"
        lines.append(f"| {row.model} | {row.task} | {row.direction} | {row.alpha:g} | {row.mean:.4f} | [{row.ci_lo:.4f}, {row.ci_hi:.4f}] | {row.p_flip_two:.4f} | {row.holm_task_p:.4f} | {selected_holm} | {row.positive}/{row.negative} | {rho:.3f} |")
    lines += ["", "## Safe-alpha selection", "", "| Model | Task | direction | alpha | fluency | base | selected |", "|---|---:|---|---:|---|---|---|"]
    for row in safe.itertuples(index=False):
        lines.append(f"| {row.model_id.split('/')[-1]} | {row.task} | {row.direction} | {row.alpha:g} | {bool(row.fluency_pass)} | {bool(row.base_pass)} | {bool(row.selected_safe)} |")
    lines += ["", "## Dose-response monotonicity", "", "| Model | Task | direction | mean item rho | p flip | Holm |", "|---|---:|---|---:|---:|---:|"]
    for row in mono_tests.itertuples(index=False):
        lines.append(f"| {row.model} | {row.task} | {row.direction} | {row.mean:.3f} | {row.p_flip_two:.4f} | {row.holm_task_p:.4f} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--steer-subdir", default="steer")
    parser.add_argument("--guards-subdir", default="guards")
    parser.add_argument("--analysis-subdir", default="analysis")
    args = parser.parse_args()
    behavior = pd.concat((pd.read_csv(path) for path in sorted((args.input_dir / "behavior").glob("rq3_behavior_*.csv")) if "__chung-" not in path.name), ignore_index=True)
    shards = sorted((args.input_dir / args.steer_subdir).glob("*/task*/*.csv"))
    if not shards: raise FileNotFoundError("No intervention shards")
    intervention = pd.concat((pd.read_csv(path) for path in shards), ignore_index=True)
    baseline = baseline_cells(behavior)
    sensitivity = prompt_sensitivity(intervention, behavior)
    abl_items, add_items = item_effects(intervention, baseline)
    abl_tests, add_tests, monotonic = test_tables(abl_items, add_items)
    abl_raw_tests, add_raw_tests = raw_test_tables(abl_items, add_items)
    abl_guard, add_guard, safe = guard_tables(abl_items, add_items, baseline, args.input_dir / args.guards_subdir)
    monotonic_all = monotonic
    monotonic = safe_monotonicity_items(add_items, safe)
    mono_tests = monotonicity_tests(monotonic) if not monotonic.empty else pd.DataFrame(columns=["model_id", "model", "task", "direction", "mean", "p_flip_two", "holm_task_p"])
    abl_pass = abl_guard.groupby(["model_id", "task", "direction"]).guard_pass.all().rename("guard_pass").reset_index()
    abl_tests = abl_tests.merge(abl_pass, on=["model_id", "task", "direction"], validate="one_to_one")
    add_tests = add_tests.merge(safe[["model_id", "task", "direction", "alpha", "guard_pass", "selected_safe"]], on=["model_id", "task", "direction", "alpha"], validate="one_to_one")
    add_tests = selected_safe_holm(add_tests)
    out = args.input_dir / args.analysis_subdir; out.mkdir(parents=True, exist_ok=True)
    abl_items.to_csv(out / "rq3b_abl_item_effects.csv", index=False)
    add_items.to_csv(out / "rq3b_add_item_effects.csv", index=False)
    abl_tests.to_csv(out / "rq3b_abl_tests.csv", index=False)
    add_tests.to_csv(out / "rq3b_add_tests.csv", index=False)
    abl_raw_tests.to_csv(out / "rq3b_abl_raw_tests.csv", index=False)
    add_raw_tests.to_csv(out / "rq3b_add_raw_tests.csv", index=False)
    monotonic_all.to_csv(out / "rq3b_add_monotonicity_all_grid.csv", index=False)
    monotonic.to_csv(out / "rq3b_add_monotonicity.csv", index=False)
    mono_tests.to_csv(out / "rq3b_add_monotonicity_tests.csv", index=False)
    abl_guard.to_csv(out / "rq3b_abl_guard_items.csv", index=False)
    add_guard.to_csv(out / "rq3b_add_guard_items.csv", index=False)
    safe.to_csv(out / "rq3b_safe_alpha.csv", index=False)
    sensitivity.to_csv(out / "rq3b_prompt_sensitivity.csv", index=False)
    write_report(abl_tests, add_tests, monotonic, mono_tests, safe, out / "rq3b_report.md")
    return 0


if __name__ == "__main__": raise SystemExit(main())
