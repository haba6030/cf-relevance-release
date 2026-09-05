"""Shared helpers for the reproduction notebooks.

Every notebook does `from repro_utils import *`. The module locates the repository root, fixes the
six-model set the paper reports, and records reported-vs-produced comparisons in a `Checks` table
that each notebook saves as `notebooks/_checks_<NN>.csv` (the summary notebook aggregates them).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── repository root: the directory that holds results/ and additional_Qs/ ──────────────────────
ROOT = Path.cwd().resolve()
while not ((ROOT / "results").exists() and (ROOT / "additional_Qs").exists()):
    if ROOT.parent == ROOT:
        raise RuntimeError("run the notebooks from inside the release repository")
    ROOT = ROOT.parent

# canonical run directories (see REPRODUCTION_MANIFEST.md)
RQ1 = ROOT / "results/rq1_spec_cached_20260821/analysis"            # 15-item behaviour and norms
PCR = ROOT / "results/rq3_pcr_locked_20260825/analysis"             # 15-item prefix decomposition, decoding, separation
GRID = ROOT / "results/rq3_intervention_final_20260827/analysis"    # 15-item preregistered intervention grid
S165 = ROOT / "additional_Qs/rq3_run_server/analysis"               # 165-item decomposition, decoding, separation
B165 = ROOT / "additional_Qs/rq12_on_165"                            # 165-item behaviour and norms
CF = ROOT / "additional_Qs/rq3_cf_rescue_20260830/analysis"        # conditional intervention, raw analysis
CFR = ROOT / "additional_Qs/cf_rescue_reanalysis"                   # conditional intervention, passing-cell re-aggregation
CFP = ROOT / "additional_Qs/cf_rescue_fluency_posthoc"              # conditional intervention, post-hoc fluency
MAN = ROOT / "additional_Qs/rq3_norm_manipulation_20260902/analysis"  # manipulation check
VEC = ROOT / "additional_Qs/rpc_vector_control"                     # vector-level projection removal
FIGS = ROOT / "docs/paper/figs/paper2"

# ── the six models of the paper ─────────────────────────────────────────────────────────────────
SIX = ["Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Instruct-2507",
       "google/gemma-3-4b-it", "microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning"]
SHORT = {"Qwen/Qwen2.5-Math-1.5B": "Qwen2.5-Math", "Qwen/Qwen3-4B": "Qwen3-4B",
         "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-Instruct", "google/gemma-3-4b-it": "Gemma-3",
         "microsoft/Phi-4-mini-instruct": "Phi-instruct", "microsoft/Phi-4-mini-reasoning": "Phi-reasoning"}
SIX_SHORT = [SHORT[m] for m in SIX]


def six(df: pd.DataFrame, col: str = "model_id") -> pd.DataFrame:
    """Restrict a table to the six models, matching either the full id or the bare model name."""
    names = {m.split("/")[-1] for m in SIX}
    keep = df[col].isin(SIX) | df[col].isin(names)
    return df[keep].copy()


def read(path: Path, **kw) -> pd.DataFrame:
    """pd.read_csv that also opens the gzipped copies the release ships for large files."""
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    return pd.read_csv(path, **kw)


def rel(path: Path) -> str:
    return str(Path(path).relative_to(ROOT))


# ── bootstrap over model means (the paper's model-level interval, B = 4000) ─────────────────────
def model_level_ci(values: pd.Series, B: int = 4000, seed: int = 20260821) -> tuple[float, float, float]:
    """values: one number per model (wordings already averaged). Returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(B, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ── reported-vs-produced bookkeeping ────────────────────────────────────────────────────────────
def _decimals(s: str) -> int:
    m = re.search(r"\.(\d+)", s)
    return len(m.group(1)) if m else 0


def _num(s: str) -> float:
    return float(s.replace("−", "-").replace("+", "").replace("%", "").strip())


class Checks:
    def __init__(self, notebook: str):
        self.notebook = notebook
        self.rows: list[dict] = []

    def num(self, cid: str, where: str, what: str, reported: str, produced: float, source: str = "", note: str = "",
            approx: float | None = None):
        """Match to the reported precision: the produced value, rounded half-up to the reported digits, equals the
        reported one (values exactly on a rounding boundary count as a match). `approx` marks a seed-dependent
        quantity: within that tolerance it is reported as ≈ instead of ✅/❌."""
        from decimal import Decimal, ROUND_HALF_UP
        d = _decimals(reported)
        r = _num(reported)
        q = Decimal(1).scaleb(-d)
        rounded = float(Decimal(str(float(produced))).quantize(q, rounding=ROUND_HALF_UP))
        ok = abs(rounded - r) < 1e-12 or abs(float(produced) - r) <= 0.5 * 10 ** (-d) + 1e-12
        status = "✅" if ok else "❌"
        if approx is not None and not ok and abs(float(produced) - r) <= approx:
            status = "≈"
            note = (note + "; " if note else "") + f"seed-dependent bootstrap, within {approx:g} of the reported bound"
        self._add(cid, where, what, reported, f"{produced:+.{d + 1}f}" if d else f"{produced:.1f}", status, source, note)

    def frac(self, cid: str, where: str, what: str, reported: str, k: int, n: int, source: str = "", note: str = ""):
        ok = reported.replace(" ", "") == f"{int(k)}/{int(n)}"
        self._add(cid, where, what, reported, f"{int(k)}/{int(n)}", "✅" if ok else "❌", source, note)

    def rng(self, cid: str, where: str, what: str, reported: str, lo: float, hi: float, pct: bool = False,
            decimals: int = 0, source: str = "", note: str = ""):
        """reported like '58–85' (percent) or '0.58–0.85'."""
        a, b = [_num(x) for x in re.findall(r"[-+−]?\d*\.?\d+", reported.replace("%", ""))[:2]]
        if pct:
            lo, hi = 100 * lo, 100 * hi
        ok = round(lo, decimals) == round(a, decimals) and round(hi, decimals) == round(b, decimals)
        fmt = f"{{:.{decimals}f}}"
        self._add(cid, where, what, reported, f"{fmt.format(lo)}–{fmt.format(hi)}", "✅" if ok else "❌", source, note)

    def presence(self, cid: str, where: str, what: str, reported: str, produced: str, source: str = "", note: str = ""):
        self._add(cid, where, what, reported, produced, "⚠", source, note or "presence check; not a numeric match")

    def blocked(self, cid: str, where: str, what: str, reported: str, note: str):
        self._add(cid, where, what, reported, "—", "⛔", "", note)

    def _add(self, cid, where, what, reported, produced, status, source, note):
        self.rows.append(dict(id=cid, section=where, analysis=what, reported=reported, produced=produced,
                              status=status, source=source, note=note))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def show(self):
        from IPython.display import display
        df = self.frame()
        counts = df.status.value_counts().to_dict()
        print(f"{self.notebook}: " + " · ".join(f"{k} {v}" for k, v in counts.items()))
        display(df.drop(columns=["source"]).style.hide(axis="index"))
        return df

    def save(self, nn: str):
        out = ROOT / "notebooks" / f"_checks_{nn}.csv"
        self.frame().to_csv(out, index=False)
        print(f"saved {rel(out)}")


def show_figure(name: str, width: int = 900):
    from IPython.display import Image, display
    p = FIGS / name
    if p.exists():
        display(Image(filename=str(p), width=width))
    else:
        print(f"figure missing: {rel(p)}")
