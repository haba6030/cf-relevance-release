"""Run the H3-0 gate **for any number of items**.

    python3 additional_Qs/analysis/07_h30_anyN.py --input-dir additional_Qs/rq3_run

## Why the original cannot be used as is

`analysis/rq3/analyze_rq3_h30.py` is specific to the active 15 items, in two places.

**(1) The item-count check** is the superficial constraint.

    if n != 15 or not np.isfinite(values).all():
        raise ValueError(f"Expected 15 finite item values, got {n}")

**(2) The 2^n blow-up of the exact sign-flip test** is the real one.

    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(values))))

It materialises **every sign assignment in memory**: 32,768 rows (4 MB) at n=15, but
4.7×10^21 rows at n=72, which kills the process with an OOM (exit 137 was observed).
The `n != 15` check was the guard against that blow-up.

## What changed

Above a threshold item count the script switches to a **Monte Carlo sign-flip test**, the standard
alternative to the exact permutation test: random signs approximate the null distribution.

    n <= MAX_EXACT (20)   original code (2^20 = 1,048,576 assignments, exact)
    n >  MAX_EXACT        approximated with N_MC (default 200,000) random sign vectors

**Because the method differs from the canonical one, the output records it**: `h30_gate.json` carries
`signflip_method`, and for Monte Carlo the number of draws and the seed.
Monte Carlo p-values carry sampling error: for a true p near 0.05, the standard error with
200,000 draws is about 0.0005.

All other computations (t test, bootstrap, LOO, sign test, Holm) use the original code.
Output names and columns are unchanged, so downstream steps read them as before. **The original file is not modified.**
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "analysis" / "rq3"))

import analyze_rq3_h30 as H  # noqa: E402

# `main()` below replaces H.exact_signflip with this file's wrapper. For small n the wrapper
# calls H.exact_signflip again, which would recurse into itself, so the canonical function
# is pinned before the swap. This keeps the full-run (n=165) Monte Carlo path and the
# LIMIT=3 smoke test recommended by the server guide under the same statistical contract.
_ORIGINAL_EXACT_SIGNFLIP = H.exact_signflip

MAX_EXACT = 20          # 2^20 = 1,048,576 assignments; memory cannot hold more
N_MC = 200_000          # number of Monte Carlo draws
_method = {"mode": "exact", "n_mc": None, "seed": H.SEED}


def signflip(values, alternative: str = "two-sided") -> float:
    """Exact sign-flip for small n, Monte Carlo for large n."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= MAX_EXACT:
        return _ORIGINAL_EXACT_SIGNFLIP(values, alternative)
    observed = values.mean()
    rng = np.random.default_rng(H.SEED)
    null = np.empty(N_MC, dtype=float)
    step = 20_000                                   # chunk to bound memory
    for lo in range(0, N_MC, step):
        hi = min(lo + step, N_MC)
        s = rng.choice((-1.0, 1.0), size=(hi - lo, n))
        null[lo:hi] = (s * values).mean(axis=1)
    if alternative == "greater":
        return float((null >= observed - 1e-15).mean())
    return float((np.abs(null) >= abs(observed) - 1e-15).mean())


def monte_carlo_with_loo(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Compute the full-sample p and every leave-one-out p from one set of sign draws.

    A sign vector with one column removed is still uniform on {−1,+1}^(n−1), so this equals a fresh
    N_MC-draw Monte Carlo null for each LOO fold. Sharing the draws does not change the marginal
    distribution of any p; it only removes the 165-fold redundant computation.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    observed = values.mean()
    observed_loo = (values.sum() - values) / (n - 1)
    count_two = 0
    count_positive = 0
    count_loo = np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(H.SEED)
    step = 20_000
    for lo in range(0, N_MC, step):
        size = min(step, N_MC - lo)
        signs = rng.choice((-1.0, 1.0), size=(size, n))
        signed = signs * values
        totals = signed.sum(axis=1)
        null = totals / n
        count_two += int((np.abs(null) >= abs(observed) - 1e-15).sum())
        count_positive += int((null >= observed - 1e-15).sum())
        null_loo = (totals[:, None] - signed) / (n - 1)
        count_loo += (np.abs(null_loo) >= np.abs(observed_loo)[None, :] - 1e-15).sum(axis=0)
    return count_two / N_MC, count_positive / N_MC, count_loo / N_MC


def summarize(values) -> dict:
    """The original `summarize` with only the item-count check and the sign-flip call replaced."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 3 or not np.isfinite(values).all():
        raise ValueError(f"Expected >=3 finite item values, got {n}")
    t = stats.ttest_1samp(values, 0.0)
    ci = stats.t.interval(0.95, n - 1, loc=values.mean(), scale=stats.sem(values))
    rng = np.random.default_rng(H.SEED)
    boot = rng.choice(values, size=(2000, n), replace=True).mean(axis=1)
    if n <= MAX_EXACT:
        loo = np.asarray([signflip(np.delete(values, i)) for i in range(n)])
        p_flip_two = signflip(values)
        p_flip_positive = signflip(values, "greater")
    else:
        p_flip_two, p_flip_positive, loo = monte_carlo_with_loo(values)
    nonzero = values[values != 0]
    sign_two = (stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5,
                                alternative="two-sided").pvalue if len(nonzero) else 1.0)
    sign_positive = (stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5,
                                     alternative="greater").pvalue if len(nonzero) else 1.0)
    return {
        "n": n, "mean": values.mean(), "t": t.statistic, "p_t_two": t.pvalue,
        "ci_lo": ci[0], "ci_hi": ci[1],
        "boot_lo": np.quantile(boot, 0.025), "boot_hi": np.quantile(boot, 0.975),
        # Key names must match the original **exactly**; gate_table and write_report look them up
        "p_flip_two": p_flip_two,
        "p_flip_one_positive": p_flip_positive,
        "positive": int((values > 0).sum()), "negative": int((values < 0).sum()),
        "zero": int((values == 0).sum()),
        "p_exact_sign_two": sign_two, "p_exact_sign_positive": sign_positive,
        "loo_worst_p_flip_two": max(loo), "loo_worst_index": int(np.argmax(loo)),
    }


def main() -> int:
    global MAX_EXACT, N_MC
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--max-exact", type=int, default=MAX_EXACT)
    ap.add_argument("--n-mc", type=int, default=N_MC)
    a, rest = ap.parse_known_args()
    MAX_EXACT, N_MC = a.max_exact, a.n_mc

    if 15 not in H.summarize.__code__.co_consts:
        print("Warning: the hard-coded 15 is not found in the original; check whether it changed.",
              file=sys.stderr)
    H.summarize = summarize
    H.exact_signflip = signflip          # also used inside tests()
    print(f"summarize/exact_signflip replaced (exact test for n<={MAX_EXACT}, "
          f"Monte Carlo with {N_MC:,} draws above)", file=sys.stderr)

    sys.argv = [sys.argv[0], "--input-dir", str(a.input_dir), *rest]
    rc = H.main()

    # record which method produced the values
    gj = a.input_dir / "analysis" / "h30_gate.json"
    if rc == 0 and gj.exists():
        d = json.loads(gj.read_text("utf-8"))
        d["signflip_method"] = {
            "exact_up_to_n": MAX_EXACT, "monte_carlo_samples": N_MC, "seed": H.SEED,
            "note": ("Above exact_up_to_n items a Monte Carlo approximation replaces the exact "
                     "sign-flip test. The canonical 15-item run is exact."),
        }
        gj.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"signflip_method recorded in h30_gate.json", file=sys.stderr)
        # Align the n=15/exact wording of the canonical report with the actual contract. The tables
        # and gate computations are exactly those produced through the monkeypatch above.
        report = a.input_dir / "analysis" / "h30_report.md"
        tests_csv = a.input_dir / "analysis" / "h30_tests.csv"
        if report.exists() and tests_csv.exists():
            n_actual = int(H.pd.read_csv(tests_csv).n.iloc[0])
            body = report.read_text("utf-8").replace(
                "Unit: item (n=15).", f"Unit: item (n={n_actual})."
            )
            if n_actual > MAX_EXACT:
                body = body.replace(
                    "Primary test: exact two-sided sign-flip;",
                    f"Primary test: Monte Carlo two-sided sign-flip "
                    f"({N_MC:,} draws, seed={H.SEED});",
                )
            report.write_text(body, encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
