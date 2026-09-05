"""Re-run the canonical RQ1/RQ2 tests unchanged on the **165-item bank** (no GPU needed).

    python3 additional_Qs/analysis/10_rq12_on_165.py

The 165-item front-end run was analysed for H3-0 and H3a only, but its behaviour and norming
outputs contain **everything** the canonical RQ1/RQ2 tests use.

    behavior : lp_base · lp_conj_h2 · lp_conj_h3 · lp_bare_h2 · lp_bare_h3  (conjunction and bare arms)
    norming  : rho (posterior P) · C (confirmation)                          (the two H2 paths)

So the same questions can be asked at n=165 without a new GPU run. This script does that,
**without touching the canonical outputs**; it writes only to a new directory.

Kept identical to the canonical analysis
--------------
The DoD definition is that of `sim/analyze_rq1_spec.py:88`.

    h = lp_arm_h / nt_arm_h  -  lp_base / nt_base        (token-normalised, base subtracted)
    d_h2 = h2(context) - h2(no context) ;  d_h3 = h3(context) - h3(no context)
    dod  = d_h3 - d_h2                                    predicted positive

Prefixes `N` (no context) vs `R` (relevant context) map onto the canonical context no/yes.
Holm is applied, as in the canonical analysis, across the two tasks within (model × arm).

Differences from the canonical analysis; read together with the results
-----------------------------------
1. **Item provenance differs.** The canonical 15 items are Chung 4 + edited 3 + model-written 8;
   the 165 items are **all model-written**. This is a generalisation, not a replication of Chung.
2. **Six models.** DeepSeek-R1-Distill and Qwen3-4B-Thinking are absent from the canonical eight.
3. **The items were screened.** The 165 items passed the `C(h3)` ceiling (G2); the canonical
   15 items were not subject to that screen.
4. **The sign-flip test is approximate.** 2^165 cannot be enumerated, so Monte Carlo is used.

Why the H2b bootstrap has two levels
---------------------------------------
"The movement of P and C carries the CF effect" is measured by an item-level Spearman correlation.
One correlation cannot settle it, **because there are two questions and their answers can differ.**

  (1) Is the correlation non-zero within this cell?          → resample **items** within the cell
  (2) Is it consistently non-zero across cells?              → resample the **correlations** across units

A small correlation that excludes zero in no cell but has **the same sign in every cell** makes (2)
exclude zero. That observation means **"a small offset is consistently present"**, not "there is a
relation", so the two levels are reported separately to prevent misreading.

**(1) Item resampling within a cell** (`boot_spearman`), the same procedure as the canonical
`sim/make_30_spec_results_nb.py`: resample items with replacement, recompute the correlation each time,
and take the 2.5/97.5 percentiles. This is the CI of that cell's correlation.

**(2) Resampling across units.** The per-cell correlations are resampled to give a CI for their mean.
Here **the choice of unit decides the conclusion.** The canonical notebook wrote:

    16 cells are 8 models × 2 tasks, but the two tasks' item profiles correlate at rho≈.96, so they are
    near-duplicates and pseudo-replication narrows the CI. The effective unit is the model, and there
    zero is not excluded. **The model level is canonical.**

So both the cell level (n=12) and the model level (n=6) are reported, **and the model level is read as the verdict.**

**Equivalence is decided at neither level.** Both ask "is it zero", not "is it negligibly small";
the latter needs a preregistered TOST margin, which this run does not have.
rho^2 is reported as a descriptive size measure, not a test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
SEED = 20260824
DRAWS = 200_000


def short(x: str) -> str:
    return (str(x).split("/")[-1]
            .replace("Qwen2.5-Math-1.5B", "Qwen2.5-Math")
            .replace("Qwen3-4B-Instruct-2507", "Qwen3-Instruct")
            .replace("gemma-3-4b-it", "Gemma-3-4B")
            .replace("Phi-4-mini-instruct", "Phi-Instruct")
            .replace("Phi-4-mini-reasoning", "Phi-Reasoning"))


def holm(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy())
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values.iloc[index] * (len(values) - rank)))
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def signflip_p(x: np.ndarray, rng: np.random.Generator) -> float:
    """Item-level two-sided sign-flip test; Monte Carlo instead of exact enumeration for large n."""
    observed = abs(x.mean())
    if len(x) <= 20:
        total = hits = 0
        for mask in range(1 << len(x)):
            signs = np.array([1 if mask >> i & 1 else -1 for i in range(len(x))])
            hits += abs((x * signs).mean()) >= observed - 1e-15
            total += 1
        return hits / total
    signs = rng.choice([-1.0, 1.0], size=(DRAWS, len(x)))
    null = np.abs((signs * x).mean(axis=1))
    return float((null >= observed - 1e-15).sum() + 1) / (DRAWS + 1)


def dod_items(beh: pd.DataFrame, ctx_prefix: str) -> pd.DataFrame:
    """Same computation as the canonical `logprob_items`; only the context prefix is a parameter."""
    frames = []
    d0 = beh[beh.prefix.isin(["N", ctx_prefix])].copy()
    for arm in ("conj", "bare"):
        d = d0.copy()
        for h in ("h2", "h3"):
            d[h] = d[f"lp_{arm}_{h}"] / d[f"nt_{arm}_{h}"] - d.lp_base / d.nt_base
        g = (d.groupby(["model_id", "task", "label", "prefix"])[["h2", "h3"]]
               .mean().unstack("prefix"))
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        g = g.reset_index()
        g["dod"] = ((g[f"h3_{ctx_prefix}"] - g["h3_N"])
                    - (g[f"h2_{ctx_prefix}"] - g["h2_N"]))
        g["arm"] = arm
        frames.append(g[["model_id", "task", "label", "arm", "dod"]])
    return pd.concat(frames, ignore_index=True)


def cell_tests(items: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (m, t, arm), g in items.groupby(["model_id", "task", "arm"]):
        x = g.dod.dropna().to_numpy()
        boot = np.array([np.random.default_rng(SEED + i).choice(x, len(x)).mean()
                         for i in range(2000)])
        rows.append({"model_id": m, "model": short(m), "task": t, "arm": arm,
                     "n": len(x), "mean": x.mean(),
                     "ci_lo": np.quantile(boot, .025), "ci_hi": np.quantile(boot, .975),
                     "p_flip_two": signflip_p(x, rng),
                     "positive": int((x > 0).sum()), "negative": int((x < 0).sum())})
    out = pd.DataFrame(rows)
    out["holm_task_p"] = np.nan
    for _, idx in out.groupby(["model_id", "arm"]).groups.items():
        out.loc[idx, "holm_task_p"] = holm(out.loc[idx, "p_flip_two"])
    return out


def h2a_tests(norm: pd.DataFrame, ctx_prefix: str) -> pd.DataFrame:
    """H2a: does context move posterior P (rho) and confirmation C? Item-level paired differences."""
    rows = []
    n = norm[norm.prefix.isin(["N", ctx_prefix])]
    for (m, h, measure), g in [((m, h, meas), gg)
                               for (m, h), gg in n.groupby(["model_id", "hypothesis"])
                               for meas in ("rho", "C")]:
        w = (g.groupby(["label", "prefix"])[measure].mean().unstack("prefix"))
        if ctx_prefix not in w or "N" not in w:
            continue
        x = (w[ctx_prefix] - w["N"]).dropna().to_numpy()
        boot = np.array([np.random.default_rng(SEED + i).choice(x, len(x)).mean()
                         for i in range(2000)])
        rows.append({"model_id": m, "model": short(m), "hypothesis": h,
                     "measure": measure, "n": len(x), "mean": x.mean(),
                     "ci_lo": np.quantile(boot, .025), "ci_hi": np.quantile(boot, .975),
                     "excludes_zero": bool(np.quantile(boot, .025) > 0
                                           or np.quantile(boot, .975) < 0)})
    return pd.DataFrame(rows)


def boot_spearman(x: np.ndarray, y: np.ndarray, B: int = 2000,
                  seed: int = SEED) -> tuple[float, float]:
    """95% percentile CI of Spearman ρ from **resampling items within** a cell.

    Same procedure as `boot_spearman` in the canonical `sim/make_30_spec_results_nb.py`, but
    n=165 × 2,000 draws × 48 combinations makes repeated `scipy.stats.spearmanr` calls slow, so
    the same definition (Pearson on ranks, ties as mean ranks) is computed in vectorised form.
    """
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    rx = rankdata(x[idx], axis=1).astype(float)
    ry = rankdata(y[idx], axis=1).astype(float)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    denom = np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))
    vals = np.where(denom > 0, (rx * ry).sum(axis=1) / np.where(denom > 0, denom, 1), np.nan)
    vals = vals[np.isfinite(vals)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def h2b_connections(norm: pd.DataFrame, items: pd.DataFrame,
                    ctx_prefix: str) -> pd.DataFrame:
    """H2b: does the movement of P and C carry the CF effect?

    Uses the two paths of the canonical `sim/analyze_rq1_spec.py:262` unchanged.
        change contrast (mediation)  : [context change of h3] - [context change of h2]
        level  contrast (confounding): [mean level of h3] - [mean level of h2]
    Each is related to the conjunction-arm DoD by an item-level Spearman correlation.
    """
    from scipy import stats
    n = norm[norm.prefix.isin(["N", ctx_prefix])]
    avg = n.groupby(["model_id", "label", "hypothesis", "prefix"])[["rho", "C"]].mean()
    wide = avg.unstack(["hypothesis", "prefix"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    for measure in ("rho", "C"):
        for h in ("h2", "h3"):
            wide[f"d_{measure}_{h}"] = (wide[f"{measure}_{h}_{ctx_prefix}"]
                                        - wide[f"{measure}_{h}_N"])
        wide[f"change_{measure}"] = wide[f"d_{measure}_h3"] - wide[f"d_{measure}_h2"]
        wide[f"level_{measure}"] = (
            (wide[f"{measure}_h3_N"] + wide[f"{measure}_h3_{ctx_prefix}"]) / 2
            - (wide[f"{measure}_h2_N"] + wide[f"{measure}_h2_{ctx_prefix}"]) / 2)

    joined = items[items.arm == "conj"].merge(wide, on=["model_id", "label"])
    rows = []
    for (m, t), g in joined.groupby(["model_id", "task"]):
        for measure in ("rho", "C"):
            for path in ("change", "level"):
                x = g[f"{path}_{measure}"].to_numpy(float)
                y = g.dod.to_numpy(float)
                r, p = stats.spearmanr(x, y)
                lo, hi = boot_spearman(x, y)
                rows.append({"model_id": m, "model": short(m), "task": t,
                             "measure": measure, "path": path,
                             "spearman": r, "p": p, "n": len(g),
                             "ci_lo": lo, "ci_hi": hi,
                             "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=PROJ / "additional_Qs" / "rq3_run_server")
    ap.add_argument("--output-dir", type=Path,
                    default=PROJ / "additional_Qs" / "rq12_on_165")
    ap.add_argument("--context-prefix", default="R", choices=["R", "I"],
                    help="prefix mapped onto the canonical context=yes")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    beh = pd.concat((pd.read_csv(p) for p in
                     sorted((a.run / "behavior").glob("rq3_behavior_*.csv"))),
                    ignore_index=True)
    norm = pd.concat((pd.read_csv(p) for p in
                      sorted((a.run / "norming").glob("norm_rq3_*.csv"))),
                     ignore_index=True)
    print(f"behaviour {len(beh)} rows · norming {len(norm)} rows · "
          f"items {beh.label.nunique()} · models {beh.model_id.nunique()}")
    print(f"context prefix = {a.context_prefix} (no context = N)")
    print()

    items = dod_items(beh, a.context_prefix)
    tests = cell_tests(items, rng)
    tests.to_csv(a.output_dir / "h1_cell_tests_165.csv", index=False)
    items.to_csv(a.output_dir / "h1_item_effects_165.csv", index=False)

    print("=" * 84)
    print("H1: does context push the relevance conjunct harder (DoD > 0 predicted)")
    for arm, lab in [("conj", "conjunction (conj)"), ("bare", "bare proposition (bare)")]:
        s = tests[tests.arm == arm]
        print(f"\n  ── {lab}")
        print(s[["model", "task", "n", "mean", "ci_lo", "ci_hi",
                 "p_flip_two", "holm_task_p", "positive", "negative"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"     positive cells {(s['mean'] > 0).sum()}/{len(s)} · "
              f"Holm-significant {(s.holm_task_p < .05).sum()}/{len(s)}")

    # H1a: task factor. Do two- and three-option tasks differ? (0/8 significant in the canonical run)
    rows = []
    for m, gg in items[items.arm == "conj"].groupby("model_id"):
        w = gg.pivot_table(index="label", columns="task", values="dod").dropna()
        diff = (w[2] - w[3]).to_numpy()
        rows.append({"model_id": m, "model": short(m), "n": len(diff),
                     "mean": diff.mean(), "p_flip_two": signflip_p(diff, rng)})
    h1a = pd.DataFrame(rows)
    h1a["holm_p"] = holm(h1a.p_flip_two)
    h1a.to_csv(a.output_dir / "h1a_task_tests_165.csv", index=False)
    print("\n" + "=" * 84)
    print("H1a: task factor. Two-option (L2C) vs three-option (L3C), item-level paired difference")
    print(h1a[["model", "n", "mean", "p_flip_two", "holm_p"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  Holm-significant in {int((h1a.holm_p < .05).sum())}/{len(h1a)} models")
    print("  Note: read the size next to the H1 main effect; the mean DoD is 0.356 whereas")
    print("    the task difference is |0.002| to |0.033|: significant but an order of magnitude smaller.")

    # H1b: is the conjunction-arm DoD smaller than the bare-arm DoD? (16/16 in the canonical run)
    w = tests.pivot_table(index=["model", "task"], columns="arm", values="mean")
    if {"conj", "bare"} <= set(w.columns):
        print("\n" + "=" * 84)
        print("H1b: is the conjunction DoD smaller than the bare DoD")
        w["conj<bare"] = w.conj < w.bare
        print(w.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"  cells with conj < bare: {int(w['conj<bare'].sum())}/{len(w)}")

    h2a = h2a_tests(norm, a.context_prefix)
    h2a.to_csv(a.output_dir / "h2a_tests_165.csv", index=False)
    print("\n" + "=" * 84)
    print("H2a: does context move posterior P (rho) and confirmation C")
    print(h2a[["model", "hypothesis", "measure", "n", "mean", "ci_lo", "ci_hi",
               "excludes_zero"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  contrasts whose CI excludes zero: {int(h2a.excludes_zero.sum())}/{len(h2a)}")

    h2b = h2b_connections(norm, items, a.context_prefix)
    h2b.to_csv(a.output_dir / "h2b_connections_165.csv", index=False)
    print("\n" + "=" * 84)
    print("H2b: does the movement of P and C carry the CF effect")
    print("\n[1] Item resampling within cells: ρ per cell with its 95% CI (same procedure as the canonical boot_spearman)")
    for path, plab in [("change", "mediation"), ("level", "confounding")]:
        s = h2b[h2b.path == path].copy()
        s["measure_"] = np.where(s.measure == "rho", "P", "C")
        v = (s.sort_values(["measure_", "model", "task"])
              [["model", "task", "measure_", "spearman", "ci_lo", "ci_hi",
                "ci_includes_zero"]])
        print(f"\n  ── {plab}({path})")
        print(v.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
        print(f"     cells whose CI includes zero: {int(v.ci_includes_zero.sum())}/{len(v)}")

    print("\n[2] Aggregation across units: CI of the mean of the resampled ρ values")
    print("    The canonical analysis treats the two tasks as pseudo-replicates and takes the **model** as the unit.")
    def bci(v):
        r = np.random.default_rng(SEED)
        m = np.array([v[r.integers(0, len(v), len(v))].mean() for _ in range(4000)])
        return np.percentile(m, [2.5, 97.5])

    hdr = (f"{'path':<6}{'m':<4}{'mean ρ':>9}{'ρ²':>7}"
           f"{'cell-level CI':>20}{'0':>5}{'model-level CI':>20}{'0':>5}{'cells incl 0':>11}")
    print("\n" + hdr)
    for path, plab in [("change", "mediation"), ("level", "confounding")]:
        for measure in ("rho", "C"):
            s = h2b[(h2b.path == path) & (h2b.measure == measure)]
            cell = s.spearman.to_numpy()
            per = s.groupby("model_id").spearman.mean().to_numpy()
            lc, hc = bci(cell); lm, hm = bci(per)
            print(f"{plab:<6}{('P' if measure=='rho' else 'C'):<4}{cell.mean():>+9.4f}"
                  f"{cell.mean()**2:>7.4f}"
                  f"{f'[{lc:+.3f},{hc:+.3f}]':>20}{'incl' if lc <= 0 <= hc else 'excl':>4}"
                  f"{f'[{lm:+.3f},{hm:+.3f}]':>20}{'incl' if lm <= 0 <= hm else 'excl':>4}"
                  f"{int(s.ci_includes_zero.sum()):>8}/{len(s)}")
    print("\n  ρ² approximates rank-based explained variance. It is a descriptive size measure, not a test.")

    print("\n  Note: read the sign too. As the canonical notebook records, mediation requires a **positive** sign,")
    print("    because items where context raised ρ(h3) more should show a larger DoD.")
    for path, plab in [("change", "mediation"), ("level", "confounding")]:
        for measure in ("rho", "C"):
            s = h2b[(h2b.path == path) & (h2b.measure == measure)].spearman
            print(f"    {plab} {'P' if measure=='rho' else 'C'}: positive {int((s>0).sum())}/{len(s)} · "
                  f"range [{s.min():+.3f}, {s.max():+.3f}]")
    print("\n  Note: 'negligibly small' is an equivalence claim and is not decided by this table.")
    print("    A TOST margin must be preregistered, and this run has none.")

    meta = {"context_prefix": a.context_prefix, "seed": SEED,
            "monte_carlo_draws": DRAWS, "n_items": int(beh.label.nunique()),
            "models": sorted(beh.model_id.unique()),
            "note": "Item provenance, models and screening differ from the canonical 15-item run. Generalisation, not replication."}
    (a.output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"\noutput -> {a.output_dir.relative_to(PROJ)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
