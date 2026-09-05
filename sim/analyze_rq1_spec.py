#!/usr/bin/env python3
"""Analyze the cached-model run of the frozen RQ1/H2 specification."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260821


def exact_signflip(values: np.ndarray, alternative: str = "two-sided") -> float:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sign-flip input contains non-finite values")
    observed = values.mean()
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = (signs * values).mean(axis=1)
    if alternative == "greater":
        return float((null >= observed - 1e-15).mean())
    return float((np.abs(null) >= abs(observed) - 1e-15).mean())


def bootstrap_ci(values: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) != 15 or not np.isfinite(values).all():
        raise ValueError(f"Expected 15 finite item effects, got n={len(values)}")
    n = len(values)
    t = stats.ttest_1samp(values, 0.0)
    ci = stats.t.interval(0.95, n - 1, loc=values.mean(), scale=stats.sem(values))
    boot = bootstrap_ci(values)
    positive = int((values > 0).sum())
    sign_p = stats.binomtest(positive, n, 0.5, alternative="greater").pvalue
    loo = []
    for index in range(n):
        loo.append(exact_signflip(np.delete(values, index)))
    return {
        "n": n,
        "mean": values.mean(),
        "t": t.statistic,
        "p_t_two": t.pvalue,
        "ci_lo": ci[0],
        "ci_hi": ci[1],
        "boot_lo": boot[0],
        "boot_hi": boot[1],
        "p_flip_two": exact_signflip(values),
        "p_flip_one_positive": exact_signflip(values, "greater"),
        "positive": positive,
        "negative": int((values < 0).sum()),
        "ties": int((values == 0).sum()),
        "p_sign_one_positive": sign_p,
        "loo_worst_p_flip_two": max(loo),
        "loo_worst_dropped_index": int(np.argmax(loo)),
    }


def holm(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy())
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, values.iloc[index] * (len(values) - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def read_all(directory: Path, pattern: str) -> pd.DataFrame:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {directory / pattern}")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def logprob_items(data: pd.DataFrame, raw: bool = False) -> pd.DataFrame:
    frames = []
    for arm in ("conj", "bare"):
        d = data.copy()
        for hypothesis in ("h2", "h3"):
            if raw:
                d[hypothesis] = d[f"lp_{arm}_{hypothesis}"] - d.lp_base
            else:
                d[hypothesis] = (
                    d[f"lp_{arm}_{hypothesis}"] / d[f"nt_{arm}_{hypothesis}"]
                    - d.lp_base / d.nt_base
                )
        grouped = (
            d.groupby(["model_id", "model", "task", "scenario", "context"])[
                ["h2", "h3"]
            ]
            .mean()
            .unstack("context")
        )
        grouped.columns = ["h2_no", "h2_yes", "h3_no", "h3_yes"]
        grouped = grouped.reset_index()
        grouped["d_h2"] = grouped.h2_yes - grouped.h2_no
        grouped["d_h3"] = grouped.h3_yes - grouped.h3_no
        grouped["dod"] = grouped.d_h3 - grouped.d_h2
        grouped["arm"] = arm
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def fc_items(data: pd.DataFrame, readout: str) -> pd.DataFrame:
    columns = [f"cf_{readout}_h2", f"cf_{readout}_h3"]
    grouped = (
        data.groupby(
            ["model_id", "model", "task", "arm", "scenario", "context"]
        )[columns]
        .mean()
        .unstack("context")
    )
    grouped.columns = ["h2_no", "h2_yes", "h3_no", "h3_yes"]
    grouped = grouped.reset_index()
    grouped["d_h2"] = grouped.h2_yes - grouped.h2_no
    grouped["d_h3"] = grouped.h3_yes - grouped.h3_no
    grouped["dod"] = grouped.d_h3 - grouped.d_h2
    grouped["readout"] = readout
    return grouped


def cell_tests(items: pd.DataFrame, metric: str, raw_items=None) -> pd.DataFrame:
    rows = []
    for keys, group in items.groupby(["model_id", "model", "task", "arm"]):
        model_id, model, task, arm = keys
        result = summarize(group.dod.to_numpy())
        result.update(model_id=model_id, model=model, task=task, arm=arm, metric=metric)
        dropped = group.iloc[result["loo_worst_dropped_index"]].scenario
        result["loo_worst_dropped"] = dropped
        if raw_items is not None:
            raw = raw_items[
                (raw_items.model_id == model_id)
                & (raw_items.task == task)
                & (raw_items.arm == arm)
            ].sort_values("scenario")
            result["raw_mean"] = raw.dod.mean()
            result["raw_p_flip_two"] = exact_signflip(raw.dod.to_numpy())
        rows.append(result)
    out = pd.DataFrame(rows)
    out["holm_task_p_flip_two"] = np.nan
    for _, index in out.groupby(["model_id", "arm", "metric"]).groups.items():
        out.loc[index, "holm_task_p_flip_two"] = holm(out.loc[index, "p_flip_two"])
    return out


def paired_tests(items: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    if factor == "task":
        index = ["model_id", "model", "scenario", "arm"]
        wide = items.pivot_table(index=index, columns="task", values="dod").reset_index()
        for keys, group in wide.groupby(["model_id", "model", "arm"]):
            values = (group[3] - group[2]).to_numpy()
            result = summarize(values)
            result.update(model_id=keys[0], model=keys[1], arm=keys[2], contrast="3-minus-2")
            rows.append(result)
    elif factor == "arm":
        index = ["model_id", "model", "scenario", "task"]
        wide = items.pivot_table(index=index, columns="arm", values="dod").reset_index()
        for keys, group in wide.groupby(["model_id", "model", "task"]):
            values = (group.conj - group.bare).to_numpy()
            result = summarize(values)
            result.update(model_id=keys[0], model=keys[1], task=keys[2], contrast="conj-minus-bare")
            rows.append(result)
    else:
        raise ValueError(factor)
    return pd.DataFrame(rows)


def position_gate(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for readout in ("letter", "content_norm"):
        for hypothesis in ("h2", "h3"):
            subset = data[data[f"cf_{readout}_{hypothesis}"].notna()].copy()
            subset["position_prediction"] = (
                subset[f"pos_{hypothesis}"] > subset.pos_base
            ).astype(int)
            subset["correct"] = (
                subset[f"cf_{readout}_{hypothesis}"] == subset.position_prediction
            )
            for keys, group in subset.groupby(["model_id", "model", "task", "arm"]):
                accuracy = group.correct.mean()
                predictability = max(accuracy, 1.0 - accuracy)
                rows.append(
                    dict(
                        model_id=keys[0],
                        model=keys[1],
                        task=keys[2],
                        arm=keys[3],
                        readout=readout,
                        hypothesis=hypothesis,
                        position_only_accuracy=accuracy,
                        position_only_predictability=predictability,
                        position_direction="direct" if accuracy >= 0.5 else "reverse",
                        contaminated=predictability > 0.60,
                        n=len(group),
                    )
                )
    return pd.DataFrame(rows)


def metric_correlations(log_items: pd.DataFrame, fc_sets: dict) -> pd.DataFrame:
    rows = []
    log = log_items[log_items.arm == "conj"]
    for readout, fc in fc_sets.items():
        fc = fc[fc.arm == "conj"]
        joined = log.merge(
            fc,
            on=["model_id", "model", "task", "scenario", "arm"],
            suffixes=("_log", "_fc"),
        )
        for keys, group in joined.groupby(["model_id", "model", "task"]):
            rho, p = stats.spearmanr(group.dod_log, group.dod_fc)
            rows.append(
                dict(
                    model_id=keys[0], model=keys[1], task=keys[2], readout=readout,
                    spearman=rho, p=p, n=len(group)
                )
            )
    return pd.DataFrame(rows)


def direct_cf_h3_tests(fc_sets: dict, gates: pd.DataFrame) -> pd.DataFrame:
    """Exploratory direct test: does context increase CF for h3 versus base?"""
    rows = []
    for readout, items in fc_sets.items():
        subset = items[items.arm == "conj"].copy()
        subset["delta_cf_h3"] = subset.h3_yes - subset.h3_no
        for keys, group in subset.groupby(["model_id", "model", "task"]):
            result = summarize(group.delta_cf_h3.to_numpy())
            result.update(
                model_id=keys[0],
                model=keys[1],
                task=keys[2],
                readout=readout,
                cf_h3_no=group.h3_no.mean(),
                cf_h3_yes=group.h3_yes.mean(),
            )
            rows.append(result)

    out = pd.DataFrame(rows)
    gate_columns = [
        "model_id", "task", "readout", "position_only_predictability",
        "position_direction", "contaminated",
    ]
    h3_gates = gates[(gates.arm == "conj") & (gates.hypothesis == "h3")][gate_columns]
    return out.merge(h3_gates, on=["model_id", "task", "readout"], how="left")


def norming_analysis(norm: pd.DataFrame, log_items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    averaged = norm.groupby(
        ["model_id", "model", "scenario", "hypothesis", "context"]
    )[["rho", "C"]].mean().reset_index()
    wide = averaged.pivot_table(
        index=["model_id", "model", "scenario"],
        columns=["hypothesis", "context"],
        values=["rho", "C"],
    )
    wide.columns = ["_".join(column) for column in wide.columns]
    wide = wide.reset_index()
    summaries = []
    for measure in ("rho", "C"):
        for hypothesis in ("h2", "h3"):
            column = f"d_{measure}_{hypothesis}"
            wide[column] = wide[f"{measure}_{hypothesis}_yes"] - wide[f"{measure}_{hypothesis}_no"]
            for keys, group in wide.groupby(["model_id", "model"]):
                result = summarize(group[column].to_numpy())
                result.update(model_id=keys[0], model=keys[1], measure=measure, hypothesis=hypothesis)
                summaries.append(result)
        wide[f"change_{measure}_contrast"] = wide[f"d_{measure}_h3"] - wide[f"d_{measure}_h2"]
        wide[f"level_{measure}_contrast"] = (
            (wide[f"{measure}_h3_no"] + wide[f"{measure}_h3_yes"]) / 2
            - (wide[f"{measure}_h2_no"] + wide[f"{measure}_h2_yes"]) / 2
        )

    connections = []
    log = log_items[log_items.arm == "conj"]
    joined = log.merge(wide, on=["model_id", "model", "scenario"])
    for keys, group in joined.groupby(["model_id", "model", "task"]):
        for measure in ("rho", "C"):
            for path in ("change", "level"):
                rho, p = stats.spearmanr(group[f"{path}_{measure}_contrast"], group.dod)
                connections.append(
                    dict(
                        model_id=keys[0], model=keys[1], task=keys[2],
                        measure=measure, path=path, spearman=rho, p=p, n=len(group)
                    )
                )
    return pd.DataFrame(summaries), pd.DataFrame(connections)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    outdir = args.input_dir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    log = read_all(args.input_dir / "logprob", "rq1_logprob_*.csv")
    fc = read_all(args.input_dir / "forcedchoice", "rq1_forcedchoice_*.csv")
    norm = read_all(args.input_dir / "norming", "norm_*.csv")
    log_items = logprob_items(log)
    log_raw = logprob_items(log, raw=True)
    log_tests = cell_tests(log_items, "logprob_norm", log_raw)
    h1a = paired_tests(log_items[log_items.arm == "conj"], "task")
    h1b = paired_tests(log_items, "arm")

    fc_sets = {
        "letter": fc_items(fc, "letter"),
        "content_norm": fc_items(fc, "content_norm"),
    }
    fc_raw = fc_items(fc, "content_raw")
    fc_tests = pd.concat(
        [
            cell_tests(fc_sets["letter"], "forcedchoice_letter"),
            cell_tests(fc_sets["content_norm"], "forcedchoice_content_norm", fc_raw),
        ],
        ignore_index=True,
    )
    gates = position_gate(fc)
    direct_cf_h3 = direct_cf_h3_tests(fc_sets, gates)
    correlations = metric_correlations(log_items, fc_sets)
    norm_summary, norm_connections = norming_analysis(norm, log_items)

    outputs = {
        "logprob_item_effects.csv": log_items,
        "logprob_cell_tests.csv": log_tests,
        "h1a_task_tests.csv": h1a,
        "h1b_arm_tests.csv": h1b,
        "forcedchoice_item_effects.csv": pd.concat(fc_sets.values(), ignore_index=True),
        "forcedchoice_cell_tests.csv": fc_tests,
        "position_gate.csv": gates,
        "direct_cf_h3_tests.csv": direct_cf_h3,
        "metric_correlations.csv": correlations,
        "norming_summary.csv": norm_summary,
        "norming_connections.csv": norm_connections,
    }
    for name, frame in outputs.items():
        frame.to_csv(outdir / name, index=False)

    report = [
        "# Frozen-spec cached-model RQ1/H2 report",
        "",
        f"Models: {log.model_id.nunique()}; items: {log.scenario.nunique()}.",
        "Decision test: exact within-item sign flip, two-sided. Positive DoD is predicted.",
        "Holm correction is across the preregistered 2-way/3-way cells within a model, not across models.",
        "",
        "## H1 log-probability cells",
        "",
        markdown_table(
            log_tests,
            ["model", "task", "arm", "mean", "ci_lo", "ci_hi", "p_flip_two", "holm_task_p_flip_two", "positive", "negative", "loo_worst_p_flip_two", "raw_p_flip_two"],
        ),
        "",
        "## H1a task contrast",
        "",
        markdown_table(h1a, ["model", "arm", "mean", "ci_lo", "ci_hi", "p_flip_two"]),
        "",
        "## H1b conjunction-minus-bare contrast",
        "",
        markdown_table(h1b, ["model", "task", "mean", "ci_lo", "ci_hi", "p_flip_two"]),
        "",
        "## Forced-choice cells",
        "",
        markdown_table(
            fc_tests,
            ["model", "metric", "task", "arm", "mean", "ci_lo", "ci_hi", "p_flip_two", "holm_task_p_flip_two", "positive", "negative", "ties"],
        ),
        "",
        "## Position-contamination gate",
        "",
        markdown_table(gates, ["model", "task", "arm", "readout", "hypothesis", "position_only_accuracy", "position_only_predictability", "position_direction", "contaminated"]),
        "",
        "## Exploratory direct context effect on h3 conjunction-fallacy rate",
        "",
        "Post-hoc test of delta CF_h3 = CF_h3(context) - CF_h3(no context). "
        "Positive is predicted; exact one- and two-sided sign-flip p-values are shown. "
        "No multiplicity correction is applied; every model-by-task cell is reported. "
        "Contaminated readouts must not support a conjunction-fallacy claim.",
        "",
        markdown_table(
            direct_cf_h3,
            [
                "model", "readout", "task", "cf_h3_no", "cf_h3_yes", "mean",
                "ci_lo", "ci_hi", "p_flip_two", "p_flip_one_positive",
                "positive", "negative", "ties", "position_only_predictability",
                "position_direction", "contaminated",
            ],
        ),
        "",
        "## H1c metric correlations",
        "",
        markdown_table(correlations, ["model", "task", "readout", "spearman", "p", "n"]),
        "",
        "## H2a norming shifts",
        "",
        markdown_table(norm_summary, ["model", "measure", "hypothesis", "mean", "ci_lo", "ci_hi", "p_flip_two", "positive", "negative"]),
        "",
        "## H2b change/level connections",
        "",
        markdown_table(norm_connections, ["model", "task", "measure", "path", "spearman", "p", "n"]),
        "",
        "H2 null results are not equivalence evidence; no TOST margin is available.",
    ]
    (outdir / "significance_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"saved {outdir / 'significance_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
