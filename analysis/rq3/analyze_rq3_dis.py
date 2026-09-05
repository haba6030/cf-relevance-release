#!/usr/bin/env python3
"""RQ3 H3f: manipulation gates, dissociation, beta calibration and TOST."""

from __future__ import annotations

import argparse
import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260824


def attach_pc_decode_eligibility(frame: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    """Attach the preregistered P/C D gate without inventing a PC gate.

    The specification defines self-decoding gates for P and C, but no composite-PC
    gate.  PC results therefore remain descriptive instead of being silently passed
    or failed by an ad-hoc rule.
    """
    required = {"model_id", "task", "P_decode_gate", "C_decode_gate"}
    missing = required - set(gates.columns)
    if missing:
        raise ValueError(f"P/C gate table is missing columns: {sorted(missing)}")
    gate = gates[list(required)].copy()
    if gate.duplicated(["model_id", "task"]).any():
        raise ValueError("P/C gate table must have one row per model/task")
    output = frame.merge(gate, on=["model_id", "task"], validate="many_to_one")
    output["inferential_eligible"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
    output["eligibility_status"] = "descriptive_only_no_preregistered_PC_gate"
    p = output.direction.eq("P")
    c = output.direction.eq("C")
    output.loc[p, "inferential_eligible"] = output.loc[p, "P_decode_gate"].astype(bool)
    output.loc[c, "inferential_eligible"] = output.loc[c, "C_decode_gate"].astype(bool)
    output.loc[p, "eligibility_status"] = np.where(
        output.loc[p, "P_decode_gate"].astype(bool), "eligible", "inferential_arm_ineligible"
    )
    output.loc[c, "eligibility_status"] = np.where(
        output.loc[c, "C_decode_gate"].astype(bool), "eligible", "inferential_arm_ineligible"
    )
    return output


@lru_cache(maxsize=None)
def sign_matrix(n: int) -> np.ndarray:
    """Cache the exact sign grid reused by every H3f endpoint."""
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)), dtype=np.float64)


def signflip(values: np.ndarray, alternative: str = "two-sided") -> float:
    values = np.asarray(values, dtype=float); observed = values.mean()
    signs = sign_matrix(len(values))
    null = (signs * values).mean(axis=1)
    if alternative == "greater": return float((null >= observed - 1e-15).mean())
    if alternative == "less": return float((null <= observed + 1e-15).mean())
    return float((np.abs(null) >= abs(observed) - 1e-15).mean())


def holm(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy()); adjusted = np.empty(len(values)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values.iloc[index] * (len(values) - rank)))
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def summarize_effect(values: np.ndarray, alternative: str) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) != 15 or not np.isfinite(values).all():
        raise ValueError("DIS endpoint requires 15 finite item effects")
    t = stats.ttest_1samp(values, 0.0); ci = stats.t.interval(.95, 14, loc=values.mean(), scale=stats.sem(values))
    rng = np.random.default_rng(SEED); boot = rng.choice(values, size=(2000, 15), replace=True).mean(axis=1)
    nonzero = values[values != 0]; positives = int((nonzero > 0).sum())
    sign_two = stats.binomtest(positives, len(nonzero), .5, alternative="two-sided").pvalue if len(nonzero) else 1.0
    sign_direction = stats.binomtest(positives, len(nonzero), .5, alternative=alternative).pvalue if len(nonzero) else 1.0
    loo = [signflip(np.delete(values, index)) for index in range(15)]
    return {"mean": values.mean(), "t": t.statistic, "p_t_two": t.pvalue, "ci_lo": ci[0], "ci_hi": ci[1], "boot_lo": np.quantile(boot, .025), "boot_hi": np.quantile(boot, .975), "p_flip_two": signflip(values), "p_flip_directional": signflip(values, alternative), "p_exact_sign_two": sign_two, "p_exact_sign_directional": sign_direction, "positive": positives, "negative": int((nonzero < 0).sum()), "zero": int((values == 0).sum()), "loo_worst_p": max(loo), "loo_worst_index": int(np.argmax(loo))}


def baseline_behavior(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["gap"] = d.lp_conj_h3 / d.nt_conj_h3 - d.lp_conj_h2 / d.nt_conj_h2
    d["gap_raw"] = d.lp_conj_h3 - d.lp_conj_h2
    d["base_raw"] = d.lp_base
    return d.groupby(["model_id", "task", "label", "prefix"])[["gap", "gap_raw", "base_raw"]].mean().reset_index()


def baseline_norming(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby(["model_id", "label", "prefix", "hypothesis"])[["rho", "C"]].mean().reset_index()


def dis_prompt_sensitivity(behavior: pd.DataFrame, baseline_raw: pd.DataFrame) -> pd.DataFrame:
    """Preserve the sign count over all 12 frame/question variants for H3f."""
    changed = behavior.copy()
    changed["gap"] = changed.lp_conj_h3 / changed.nt_conj_h3 - changed.lp_conj_h2 / changed.nt_conj_h2
    base = baseline_raw.copy()
    base["gap"] = base.lp_conj_h3 / base.nt_conj_h3 - base.lp_conj_h2 / base.nt_conj_h2
    base = base[base.prefix == "I"].groupby(
        ["model_id", "task", "label", "frame", "qvar"], as_index=False
    ).gap.mean().rename(columns={"gap": "baseline_gap"})
    changed = changed.merge(
        base, on=["model_id", "task", "label", "frame", "qvar"], validate="many_to_one"
    )
    changed["effect"] = changed.gap - changed.baseline_gap
    rows = []
    for keys, group in changed.groupby(["model_id", "task", "label", "direction", "mode", "alpha"]):
        rows.append(
            {
                "model_id": keys[0], "task": keys[1], "label": keys[2],
                "direction": keys[3], "mode": keys[4], "alpha": keys[5],
                "hypothesized_sign": "positive",
                "negative_variants": int((group.effect < 0).sum()),
                "positive_variants": int((group.effect > 0).sum()),
                "zero_variants": int((group.effect == 0).sum()),
                "n_variants": len(group),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or not (result.n_variants == 12).all():
        raise RuntimeError("DIS prompt-sensitivity cells must each contain 12 variants")
    return result


def build_cells(behavior: pd.DataFrame, norming: pd.DataFrame, base_b: pd.DataFrame, base_n: pd.DataFrame) -> pd.DataFrame:
    behavior = behavior.copy()
    behavior["gap"] = behavior.lp_conj_h3 / behavior.nt_conj_h3 - behavior.lp_conj_h2 / behavior.nt_conj_h2
    behavior["gap_raw"] = behavior.lp_conj_h3 - behavior.lp_conj_h2
    behavior["base_raw_plus"] = behavior.lp_base_plus
    behavior["base_raw_minus"] = behavior.lp_base_minus
    behavior = behavior.groupby(["model_id", "task", "label", "direction", "mode", "alpha"])[["gap", "gap_raw", "base_raw_plus", "base_raw_minus"]].mean().reset_index()
    base_i = base_b[base_b.prefix == "I"].drop(columns="prefix").rename(columns={"gap": "baseline_gap", "gap_raw": "baseline_gap_raw", "base_raw": "baseline_base_raw"})
    behavior = behavior.merge(base_i, on=["model_id", "task", "label"], validate="many_to_one")
    behavior["induced_dod"] = behavior.gap - behavior.baseline_gap
    behavior["induced_dod_raw"] = behavior.gap_raw - behavior.baseline_gap_raw
    behavior["base_shift"] = np.maximum(
        (behavior.base_raw_plus - behavior.baseline_base_raw).abs(),
        (behavior.base_raw_minus - behavior.baseline_base_raw).abs().fillna(0.0),
    )

    manipulated = norming[norming.stage == "ADD"].groupby(["model_id", "task", "label", "direction", "mode", "alpha", "hypothesis"])[["rho", "C"]].mean().reset_index()
    base_i_n = base_n[base_n.prefix == "I"].drop(columns="prefix").rename(columns={"rho": "rho0", "C": "C0"})
    manipulated = manipulated.merge(base_i_n, on=["model_id", "label", "hypothesis"], validate="many_to_one")
    manipulated["delta_rho"] = manipulated.rho - manipulated.rho0
    manipulated["delta_C"] = manipulated.C - manipulated.C0
    wide = manipulated.pivot(
        index=["model_id", "task", "label", "direction", "mode", "alpha"],
        columns="hypothesis", values=["delta_rho", "delta_C"],
    )
    wide.columns = [f"{metric}_{hypothesis}" for metric, hypothesis in wide.columns]
    wide = wide.reset_index()
    wide["delta_rho_diff"] = wide.delta_rho_h3 - wide.delta_rho_h2
    wide["delta_C_diff"] = wide.delta_C_h3 - wide.delta_C_h2
    wide["delta_rho_common"] = (wide.delta_rho_h3 + wide.delta_rho_h2) / 2
    wide["delta_C_common"] = (wide.delta_C_h3 + wide.delta_C_h2) / 2
    scale_rows = []
    for model_id, group in base_n.groupby("model_id"):
        centered = group.copy()
        centered[["rho", "C"]] = centered[["rho", "C"]] - centered.groupby("label")[["rho", "C"]].transform("mean")
        rho_sd = float(centered.rho.std(ddof=1)); C_sd = float(centered.C.std(ddof=1)); correlation = float(centered[["rho", "C"]].corr().iloc[0, 1])
        scale_rows.append({"model_id": model_id, "rho_scale": max(rho_sd, 1e-12), "C_scale": max(C_sd, 1e-12), "PC_alignment": 1.0 if correlation >= 0 else -1.0})
    wide = wide.merge(pd.DataFrame(scale_rows), on="model_id", validate="many_to_one")
    wide["delta_PC_diff"] = wide.delta_rho_diff / wide.rho_scale + wide.PC_alignment * wide.delta_C_diff / wide.C_scale
    wide["delta_PC_common"] = wide.delta_rho_common / wide.rho_scale + wide.PC_alignment * wide.delta_C_common / wide.C_scale
    return behavior.merge(wide, on=["model_id", "task", "label", "direction", "mode", "alpha"], validate="one_to_one")


def safe_alpha_table(cells: pd.DataFrame, base_b: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    """Select alpha from preregistered side-effect guards only."""
    threshold = (
        base_b[base_b.prefix == "I"].groupby(["model_id", "task"]).base_raw
        .std(ddof=1).mul(2).rename("base_limit").reset_index()
    )
    fluency_files = sorted((input_dir / "guards").glob("*/fluency.csv"))
    if not fluency_files:
        raise FileNotFoundError("No fluency guard files")
    fluency = pd.concat((pd.read_csv(path) for path in fluency_files), ignore_index=True)
    fluency = fluency[fluency["mode"] == "ADD"][["model_id", "task", "direction", "alpha", "passes", "increase"]].rename(columns={"passes": "fluency_pass", "increase": "fluency_increase"})
    safe = cells.groupby(["model_id", "task", "direction", "mode", "alpha"], as_index=False).base_shift.max()
    safe = safe.merge(threshold, on=["model_id", "task"], validate="many_to_one")
    safe = safe.merge(fluency, on=["model_id", "task", "direction", "alpha"], how="left", validate="many_to_one")
    safe["fluency_pass"] = safe.fluency_pass.fillna(False)
    safe["base_pass"] = safe.base_shift <= safe.base_limit
    safe["guard_pass"] = safe.base_pass & safe.fluency_pass
    safe["selected_safe"] = False
    passing = safe[safe.guard_pass]
    if not passing.empty:
        selected = passing.groupby(["model_id", "task", "direction", "mode"]).alpha.idxmax()
        safe.loc[selected, "selected_safe"] = True
    return safe


def specificity_table(norming: pd.DataFrame, base_n: pd.DataFrame, primary_safe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = primary_safe[
        primary_safe.direction.astype(str).str.startswith("R_")
        & primary_safe.selected_safe
    ]
    for gate in selected.itertuples(index=False):
        manipulated = norming[
            (norming.model_id == gate.model_id) & (norming.task == gate.task) &
            (norming.direction == gate.direction) & (norming["mode"] == "common") &
            (norming.stage == "ADD") &
            np.isclose(norming.alpha, gate.alpha)
        ].groupby(["label", "hypothesis"])[["rho", "C"]].mean().reset_index()
        baseline = base_n[(base_n.model_id == gate.model_id) & (base_n.prefix == "I")].drop(columns=["model_id", "prefix"])
        merged = manipulated.merge(baseline, on=["label", "hypothesis"], suffixes=("", "0"), validate="one_to_one")
        for metric in ("rho", "C"):
            for hypothesis in ("h2", "h3"):
                values = (merged.loc[merged.hypothesis == hypothesis, metric] - merged.loc[merged.hypothesis == hypothesis, metric + "0"]).to_numpy()
                ci = stats.t.interval(.95, len(values) - 1, loc=values.mean(), scale=stats.sem(values))
                rows.append({"model_id": gate.model_id, "task": gate.task, "direction": gate.direction, "stage": "ADD", "alpha": gate.alpha, "metric": metric, "hypothesis": hypothesis, "mean_change": values.mean(), "ci_lo": ci[0], "ci_hi": ci[1], "ci_contains_zero": ci[0] <= 0 <= ci[1], "p_flip_two": signflip(values)})
    r_ablation = norming[(norming.stage == "ABL") & norming.direction.astype(str).str.startswith("R_")]
    for (model_id, task, direction), group in r_ablation.groupby(["model_id", "task", "direction"]):
        manipulated = group.groupby(["label", "hypothesis"])[["rho", "C"]].mean().reset_index()
        baseline = base_n[(base_n.model_id == model_id) & (base_n.prefix == "I")].drop(columns=["model_id", "prefix"])
        merged = manipulated.merge(baseline, on=["label", "hypothesis"], suffixes=("", "0"), validate="one_to_one")
        for metric in ("rho", "C"):
            for hypothesis in ("h2", "h3"):
                values = (merged.loc[merged.hypothesis == hypothesis, metric] - merged.loc[merged.hypothesis == hypothesis, metric + "0"]).to_numpy()
                ci = stats.t.interval(.95, len(values) - 1, loc=values.mean(), scale=stats.sem(values))
                rows.append({"model_id": model_id, "task": task, "direction": direction, "stage": "ABL", "alpha": np.nan, "metric": metric, "hypothesis": hypothesis, "mean_change": values.mean(), "ci_lo": ci[0], "ci_hi": ci[1], "ci_contains_zero": ci[0] <= 0 <= ci[1], "p_flip_two": signflip(values)})
    return pd.DataFrame(rows)


def abl_norming_checks(norming: pd.DataFrame, base_n: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = base_n[base_n.prefix == "I"].pivot(index=["model_id", "label"], columns="hypothesis", values=["rho", "C"])
    base.columns = [f"base_{metric}_{hypothesis}" for metric, hypothesis in base.columns]; base = base.reset_index()
    manipulated = norming[(norming.stage == "ABL") & norming.direction.isin(["P", "C", "PC"])].groupby(["model_id", "task", "label", "direction", "hypothesis"])[["rho", "C"]].mean().reset_index()
    wide = manipulated.pivot(index=["model_id", "task", "label", "direction"], columns="hypothesis", values=["rho", "C"])
    wide.columns = [f"abl_{metric}_{hypothesis}" for metric, hypothesis in wide.columns]; wide = wide.reset_index().merge(base, on=["model_id", "label"], validate="many_to_one")
    wide["rho_attenuation"] = (wide.base_rho_h3 - wide.base_rho_h2).abs() - (wide.abl_rho_h3 - wide.abl_rho_h2).abs()
    wide["C_attenuation"] = (wide.base_C_h3 - wide.base_C_h2).abs() - (wide.abl_C_h3 - wide.abl_C_h2).abs()
    rows = []
    for keys, group in wide.groupby(["model_id", "task", "direction"]):
        if keys[2] == "P": values = group.rho_attenuation.to_numpy()
        elif keys[2] == "C": values = group.C_attenuation.to_numpy()
        else: values = ((group.rho_attenuation / max(group[["base_rho_h2", "base_rho_h3"]].to_numpy().std(ddof=1), 1e-12)) + (group.C_attenuation / max(group[["base_C_h2", "base_C_h3"]].to_numpy().std(ddof=1), 1e-12))).to_numpy()
        summary = summarize_effect(values, "greater")
        summary.update(model_id=keys[0], model=keys[0].split("/")[-1], task=keys[1], direction=keys[2], loo_worst_dropped=group.iloc[summary["loo_worst_index"]].label)
        rows.append(summary)
    tests = pd.DataFrame(rows); tests["holm_task_p"] = np.nan
    for _, index in tests.groupby(["model_id", "direction"]).groups.items(): tests.loc[index, "holm_task_p"] = holm(tests.loc[index, "p_flip_two"])
    return wide, tests


def endpoint_table(cells: pd.DataFrame, safe: pd.DataFrame) -> pd.DataFrame:
    """Safe-alpha P/C manipulation checks and behavioral endpoints."""
    rows = []
    for gate in safe[safe.selected_safe].itertuples(index=False):
        group = cells[
            (cells.model_id == gate.model_id) & (cells.task == gate.task) &
            (cells.direction == gate.direction) & (cells["mode"] == gate.mode) &
            np.isclose(cells.alpha, gate.alpha)
        ]
        if len(group) != 15:
            raise RuntimeError(f"Safe DIS endpoint does not have 15 items: {gate}")
        suffix = "diff" if gate.mode == "differential" else "common"
        target_column = {"P": f"delta_rho_{suffix}", "C": f"delta_C_{suffix}", "PC": f"delta_PC_{suffix}"}[gate.direction]
        target = group[target_column].to_numpy(); behavior = group.induced_dod.to_numpy()
        behavior_raw = group.induced_dod_raw.to_numpy()
        target_summary = {f"target_{key}": value for key, value in summarize_effect(target, "greater").items()}
        behavior_summary = {f"dod_{key}": value for key, value in summarize_effect(behavior, "greater").items()}
        behavior_raw_summary = {f"dod_raw_{key}": value for key, value in summarize_effect(behavior_raw, "greater").items()}
        rows.append({
            "model_id": gate.model_id, "model": gate.model_id.split("/")[-1], "task": gate.task,
            "direction": gate.direction, "mode": gate.mode, "alpha_safe": gate.alpha,
            "mean_target_change": target.mean(), "target_p_two": signflip(target),
            "target_p_positive": signflip(target, "greater"), "target_positive": int((target > 0).sum()),
            "mean_induced_dod": behavior.mean(), "dod_p_two": signflip(behavior),
            "dod_p_positive": signflip(behavior, "greater"), "dod_positive": int((behavior > 0).sum()),
            **target_summary, **behavior_summary, **behavior_raw_summary,
            "target_loo_worst_dropped": group.iloc[int(target_summary["target_loo_worst_index"])].label,
            "dod_loo_worst_dropped": group.iloc[int(behavior_summary["dod_loo_worst_index"])].label,
            "dod_raw_loo_worst_dropped": group.iloc[int(behavior_raw_summary["dod_raw_loo_worst_index"])].label,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "model_id", "model", "task", "direction", "mode", "alpha_safe",
            "target_holm_task_p", "dod_holm_task_p",
        ])
    out["target_holm_task_p"] = np.nan; out["dod_holm_task_p"] = np.nan
    for _, index in out.groupby(["model_id", "direction", "mode"]).groups.items():
        out.loc[index, "target_holm_task_p"] = holm(out.loc[index, "target_p_two"])
        out.loc[index, "dod_holm_task_p"] = holm(out.loc[index, "dod_p_two"])
    return out


def item_slopes(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in cells.groupby(["model_id", "task", "label", "direction", "mode"]):
        # Include the registered alpha=0 origin. OLS includes an intercept.
        suffix = "diff" if keys[4] == "differential" else "common"
        rho_change = group[f"delta_rho_{suffix}"].to_numpy(); C_change = group[f"delta_C_{suffix}"].to_numpy()
        x = np.r_[0.0, rho_change]
        y = np.r_[0.0, group.induced_dod.to_numpy()]
        beta_rho = stats.linregress(x, y).slope if np.ptp(x) > 0 else np.nan
        rho_linear_r2 = stats.linregress(x, y).rvalue ** 2 if np.ptp(x) > 0 else np.nan
        rho_quadratic = np.polyfit(x, y, 2)[0] if len(np.unique(x)) >= 3 else np.nan
        xc = np.r_[0.0, C_change]
        beta_c = stats.linregress(xc, y).slope if np.ptp(xc) > 0 else np.nan
        C_linear_r2 = stats.linregress(xc, y).rvalue ** 2 if np.ptp(xc) > 0 else np.nan
        C_quadratic = np.polyfit(xc, y, 2)[0] if len(np.unique(xc)) >= 3 else np.nan
        target_change = {
            "P": rho_change,
            "C": C_change,
            "PC": group[f"delta_PC_{suffix}"].to_numpy(),
        }[keys[3]]
        rows.append({
            "model_id": keys[0], "task": keys[1], "label": keys[2], "direction": keys[3], "mode": keys[4],
            "beta_rho": beta_rho, "beta_C": beta_c,
            "rho_linear_r2": rho_linear_r2, "C_linear_r2": C_linear_r2,
            "rho_quadratic": rho_quadratic, "C_quadratic": C_quadratic,
            "rho_alpha_spearman": stats.spearmanr(np.r_[0.0, group.alpha], np.r_[0.0, rho_change]).statistic,
            "C_alpha_spearman": stats.spearmanr(np.r_[0.0, group.alpha], np.r_[0.0, C_change]).statistic,
            "target_alpha_spearman": stats.spearmanr(np.r_[0.0, group.alpha], np.r_[0.0, target_change]).statistic,
            "dod_alpha_spearman": stats.spearmanr(np.r_[0.0, group.alpha], np.r_[0.0, group.induced_dod]).statistic,
        })
    return pd.DataFrame(rows)


def manipulation_monotonicity(slopes: pd.DataFrame) -> pd.DataFrame:
    if slopes.empty:
        return pd.DataFrame(columns=[
            "model_id", "model", "task", "direction", "mode", "mean",
            "p_flip_two", "holm_task_p", "monotonicity_gate",
        ])
    rows = []
    for keys, group in slopes.groupby(["model_id", "task", "direction", "mode"]):
        values = group.target_alpha_spearman.fillna(0).to_numpy()
        result = summarize_effect(values, "greater")
        result.update(
            model_id=keys[0], model=keys[0].split("/")[-1], task=keys[1],
            direction=keys[2], mode=keys[3],
            loo_worst_dropped=group.iloc[result["loo_worst_index"]].label,
        )
        rows.append(result)
    output = pd.DataFrame(rows)
    output["holm_task_p"] = np.nan
    for _, index in output.groupby(["model_id", "direction", "mode"]).groups.items():
        output.loc[index, "holm_task_p"] = holm(output.loc[index, "p_flip_two"])
    output["monotonicity_gate"] = (output["mean"] > 0) & (output.holm_task_p < .05)
    return output


def bootstrap_ratio(predicted: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED); ratios = []
    for _ in range(5000):
        index = rng.integers(0, len(predicted), len(predicted))
        denominator = observed[index].mean()
        ratios.append(predicted[index].mean() / denominator if abs(denominator) > 1e-12 else np.nan)
    return tuple(np.nanquantile(ratios, [.025, .975]))


def tost(values: np.ndarray, margin: float) -> float:
    values = np.asarray(values, dtype=float); mean = values.mean(); se = stats.sem(values); df = len(values) - 1
    if not np.isfinite(margin) or margin <= 0 or se <= 0: return np.nan
    p_lower = 1 - stats.t.cdf((mean + margin) / se, df)
    p_upper = stats.t.cdf((mean - margin) / se, df)
    return float(max(p_lower, p_upper))


def summaries(cells: pd.DataFrame, slopes: pd.DataFrame, base_b: pd.DataFrame, base_n: pd.DataFrame, safe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, task), group in cells.groupby(["model_id", "task"]):
        p = group[(group.direction == "P") & (group["mode"] == "differential")]
        chosen = safe[(safe.model_id == model_id) & (safe.task == task) & (safe.direction == "P") & (safe["mode"] == "differential") & safe.selected_safe]
        if chosen.empty:
            continue
        alpha_safe = float(chosen.iloc[0].alpha)
        at_max = p[np.isclose(p.alpha, alpha_safe)].sort_values("label")
        slope = slopes[(slopes.model_id == model_id) & (slopes.task == task) & (slopes.direction == "P") & (slopes["mode"] == "differential")].sort_values("label")
        labels = at_max.label.tolist()
        base_norm = base_n[(base_n.model_id == model_id) & base_n.label.isin(labels)]
        norm_wide = base_norm.pivot(index="label", columns=["prefix", "hypothesis"], values="rho")
        # H3f-2 explicitly recalibrates H2a. H2a's observed context contrast
        # is R-N (the original context/no-context comparison), not the new
        # R-I prerequisite contrast used to isolate relevance in H3-0.
        observed_dp = ((norm_wide[("R", "h3")] - norm_wide[("N", "h3")]) - (norm_wide[("R", "h2")] - norm_wide[("N", "h2")])).reindex(labels).to_numpy()
        behavior = base_b[(base_b.model_id == model_id) & (base_b.task == task) & base_b.label.isin(labels)]
        bw = behavior.pivot(index="label", columns="prefix", values="gap")
        observed_dod = (bw.R - bw.N).reindex(labels).to_numpy()
        beta = slope.beta_rho.to_numpy(); predicted = beta * observed_dp
        complete_beta = len(beta) == 15 and np.isfinite(beta).all()
        if complete_beta:
            beta_ci = stats.t.interval(.95, len(beta) - 1, loc=beta.mean(), scale=stats.sem(beta))
            if abs(observed_dod.mean()) > 1e-12:
                ratio = predicted.mean() / observed_dod.mean()
                ratio_ci = bootstrap_ratio(predicted, observed_dod)
            else:
                ratio, ratio_ci = np.nan, (np.nan, np.nan)
        else:
            ratio, ratio_ci, beta_ci = np.nan, (np.nan, np.nan), (np.nan, np.nan)
        observed_max = float(np.max(np.abs(observed_dp)))
        reached = float(np.max(np.abs(at_max.delta_rho_diff)))
        currency = stats.spearmanr(at_max.delta_rho_diff, observed_dp).statistic
        delta = abs(observed_dod.mean()) / 2
        mean_beta = beta.mean() if complete_beta else np.nan
        epsilon = delta / abs(mean_beta) if abs(mean_beta) > 1e-12 else np.nan
        rows.append({
            "model_id": model_id, "model": model_id.split("/")[-1], "task": task,
            "calibration_contrast": "R-N (H2a/RQ1)",
            "alpha_safe": alpha_safe,
            "mean_beta_P": mean_beta, "beta_ci_lo": beta_ci[0], "beta_ci_hi": beta_ci[1],
            "observed_mean_DoD": observed_dod.mean(), "observed_mean_deltaP": observed_dp.mean(),
            "predicted_mean_DoD": predicted.mean(), "explanation_ratio": ratio,
            "ratio_ci_lo": ratio_ci[0], "ratio_ci_hi": ratio_ci[1],
            "manipulation_reached": reached, "required_reach": 1.5 * observed_max,
            "reach_gate": reached >= 1.5 * observed_max,
            "currency_spearman": currency, "currency_gate": currency >= .3,
            "mean_monotonicity": slope.rho_alpha_spearman.mean(),
            "induced_dod_p_two": signflip(at_max.induced_dod.to_numpy()),
            "induced_dod_p_positive": signflip(at_max.induced_dod.to_numpy(), "greater"),
            "epsilon_rho": epsilon, "h2a_tost_p": tost(observed_dp, epsilon),
            "conversion_available": bool(complete_beta and np.isfinite(beta_ci).all() and (beta_ci[0] > 0 or beta_ci[1] < 0)),
        })
    output = pd.DataFrame(rows)
    output["h2a_tost_holm_p"] = np.nan
    if not output.empty:
        for _, index in output.groupby("model_id").groups.items():
            finite = output.loc[index, "h2a_tost_p"].dropna().index
            if len(finite):
                output.loc[finite, "h2a_tost_holm_p"] = holm(output.loc[finite, "h2a_tost_p"])
    return output


def write_report(summary: pd.DataFrame, endpoints: pd.DataFrame, specificity: pd.DataFrame, abl_norming: pd.DataFrame, path: Path) -> None:
    lines = ["# RQ3 H3f DIS and calibration", "", "Only the maximum alpha passing fluency and base-collapse guards is used for endpoint tests. H3f-2 calibration uses the registered H2a/RQ1 R-N observational contrast; R-I remains the relevance-specific H3-0/intervention contrast.", "", "| Model | Task | contrast | safe alpha | target monotonic | beta P | beta 95% CI | reach | currency rho | ratio | ratio 95% CI | DoD p | epsilon rho | TOST p | TOST Holm | conversion |", "|---|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.model} | {row.task} | {row.calibration_contrast} | {row.alpha_safe:g} | {bool(row.target_monotonicity_gate)} | {row.mean_beta_P:.4f} | [{row.beta_ci_lo:.4f}, {row.beta_ci_hi:.4f}] | {row.reach_gate} | {row.currency_spearman:.3f} | {row.explanation_ratio:.3f} | [{row.ratio_ci_lo:.3f}, {row.ratio_ci_hi:.3f}] | {row.induced_dod_p_two:.4f} | {row.epsilon_rho:.3f} | {row.h2a_tost_p:.4f} | {row.h2a_tost_holm_p:.4f} | {row.conversion_available} |")
    lines += ["", "## Safe-alpha manipulation endpoints", "", "P/C rows failing their preregistered self-decoding gate are retained but marked inferentially ineligible. The specification defines no composite-PC decoding gate, so PC is descriptive rather than silently assigned an ad-hoc pass/fail rule.", "", "| Model | Task | direction | eligibility | mode | alpha | target change | target p | target monotonic | induced DoD | DoD p |", "|---|---:|---|---|---|---:|---:|---:|---|---:|---:|"]
    for row in endpoints.itertuples(index=False):
        lines.append(f"| {row.model} | {row.task} | {row.direction} | {row.eligibility_status} | {row.mode} | {row.alpha_safe:g} | {row.mean_target_change:.4f} | {row.target_holm_task_p:.4f} | {bool(row.target_monotonicity_gate)} | {row.mean_induced_dod:.4f} | {row.dod_holm_task_p:.4f} |")
    lines += ["", "## R-direction norming specificity", "", "Each row passes specificity when its 95% CI contains zero; all four rows in both ABL and safe-alpha ADD must pass per model/task/direction.", "", "| Model | Task | direction | stage | alpha | metric | hypothesis | change | 95% CI | contains zero |", "|---|---:|---|---|---:|---|---|---:|---:|---|"]
    for row in specificity.itertuples(index=False):
        alpha = "—" if pd.isna(row.alpha) else f"{row.alpha:g}"
        lines.append(f"| {row.model_id.split('/')[-1]} | {row.task} | {row.direction} | {row.stage} | {alpha} | {row.metric} | {row.hypothesis} | {row.mean_change:.4f} | [{row.ci_lo:.4f}, {row.ci_hi:.4f}] | {bool(row.ci_contains_zero)} |")
    lines += ["", "## P/C/PC ablation manipulation checks", "", "Positive attenuation means ablation reduced the encoded norming contrast.", "", "| Model | Task | direction | attenuation | 95% CI | Holm p |", "|---|---:|---|---:|---:|---:|"]
    for row in abl_norming.itertuples(index=False):
        lines.append(f"| {row.model} | {row.task} | {row.direction} | {row.mean:.4f} | [{row.ci_lo:.4f}, {row.ci_hi:.4f}] | {row.holm_task_p:.4f} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--pc-gate-file", type=Path,
        help="P/C self-decoding gate table (default: INPUT_DIR/analysis/h3b_h3_geometry/h3b_geometry_summary.csv).",
    )
    args = parser.parse_args()
    behavior_files = [p for p in sorted((args.input_dir / "behavior").glob("rq3_behavior_*.csv")) if "__chung-" not in p.name]
    base_behavior_raw = pd.concat((pd.read_csv(p) for p in behavior_files), ignore_index=True)
    base_behavior = baseline_behavior(base_behavior_raw)
    norm_files = sorted((args.input_dir / "norming").glob("norm_rq3_*.csv"))
    base_norm = baseline_norming(pd.concat((pd.read_csv(p) for p in norm_files), ignore_index=True))
    bfiles = sorted((args.input_dir / "dis").glob("*/task*/*_behavior.csv")); nfiles = sorted((args.input_dir / "dis").glob("*/task*/*_norming.csv"))
    if not bfiles or not nfiles: raise FileNotFoundError("No DIS shards")
    behavior = pd.concat((pd.read_csv(p) for p in bfiles), ignore_index=True)
    norming = pd.concat((pd.read_csv(p) for p in nfiles), ignore_index=True)
    sensitivity = dis_prompt_sensitivity(behavior, base_behavior_raw)
    cells = build_cells(behavior, norming, base_behavior, base_norm)
    safe = safe_alpha_table(cells, base_behavior, args.input_dir)
    safe_max = safe[safe.guard_pass].groupby(["model_id", "task", "direction", "mode"]).alpha.max().rename("alpha_safe").reset_index()
    analysis_cells = cells.merge(safe_max, on=["model_id", "task", "direction", "mode"], how="inner", validate="many_to_one")
    analysis_cells = analysis_cells[analysis_cells.alpha <= analysis_cells.alpha_safe]
    slopes = item_slopes(analysis_cells)
    target_monotonicity = manipulation_monotonicity(slopes)
    pc_gate_path = args.pc_gate_file or args.input_dir / "analysis" / "h3b_h3_geometry" / "h3b_geometry_summary.csv"
    pc_gates = pd.read_csv(pc_gate_path)
    primary_safe = pd.read_csv(args.input_dir / "analysis" / "rq3b_safe_alpha.csv")
    specificity = specificity_table(norming, base_norm, primary_safe)
    abl_norm_items, abl_norm_tests = abl_norming_checks(norming, base_norm)
    endpoints = endpoint_table(cells, safe)
    if endpoints.empty:
        endpoints["target_monotonicity_mean"] = np.nan
        endpoints["target_monotonicity_holm_p"] = np.nan
        endpoints["target_monotonicity_gate"] = False
    else:
        endpoints = endpoints.merge(
            target_monotonicity[["model_id", "task", "direction", "mode", "mean", "holm_task_p", "monotonicity_gate"]].rename(
                columns={"mean": "target_monotonicity_mean", "holm_task_p": "target_monotonicity_holm_p", "monotonicity_gate": "target_monotonicity_gate"}
            ),
            on=["model_id", "task", "direction", "mode"], validate="one_to_one",
        )
    endpoints = attach_pc_decode_eligibility(endpoints, pc_gates)
    target_monotonicity = attach_pc_decode_eligibility(target_monotonicity, pc_gates)
    abl_norm_tests = attach_pc_decode_eligibility(abl_norm_tests, pc_gates)
    summary = summaries(analysis_cells, slopes, base_behavior, base_norm, safe)
    if summary.empty:
        summary = pd.DataFrame(columns=[
            "model_id", "task", "h2a_tost_holm_p", "target_monotonicity_mean",
            "target_monotonicity_holm_p", "target_monotonicity_gate",
        ])
    else:
        summary = summary.merge(
            target_monotonicity[(target_monotonicity.direction == "P") & (target_monotonicity["mode"] == "differential")][
                ["model_id", "task", "mean", "holm_task_p", "monotonicity_gate"]
            ].rename(columns={"mean": "target_monotonicity_mean", "holm_task_p": "target_monotonicity_holm_p", "monotonicity_gate": "target_monotonicity_gate"}),
            on=["model_id", "task"], how="left", validate="one_to_one",
        )
        summary = summary.merge(
            pc_gates[["model_id", "task", "P_decode_gate"]],
            on=["model_id", "task"], validate="one_to_one",
        )
        summary["inferential_eligible"] = summary.P_decode_gate.astype(bool)
        summary["eligibility_status"] = np.where(
            summary.inferential_eligible, "eligible", "inferential_arm_ineligible"
        )
        summary = summary[summary.inferential_eligible].copy()
    out = args.input_dir / "analysis"; out.mkdir(exist_ok=True)
    cells.to_csv(out / "rq3f_cells.csv", index=False); slopes.to_csv(out / "rq3f_item_slopes.csv", index=False); summary.to_csv(out / "rq3f_summary.csv", index=False)
    safe.to_csv(out / "rq3f_safe_alpha.csv", index=False); specificity.to_csv(out / "rq3f_R_specificity.csv", index=False)
    endpoints.to_csv(out / "rq3f_endpoints.csv", index=False)
    target_monotonicity.to_csv(out / "rq3f_manipulation_monotonicity.csv", index=False)
    sensitivity.to_csv(out / "rq3f_prompt_sensitivity.csv", index=False)
    abl_norm_items.to_csv(out / "rq3f_abl_norming_items.csv", index=False); abl_norm_tests.to_csv(out / "rq3f_abl_norming_tests.csv", index=False)
    write_report(summary, endpoints, specificity, abl_norm_tests, out / "rq3f_report.md")
    return 0


if __name__ == "__main__": raise SystemExit(main())
