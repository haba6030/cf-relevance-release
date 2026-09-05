#!/usr/bin/env python3
"""Write REPRODUCTION_REPORT.md from the check tables the executed notebooks saved (notebooks/_checks_0*.csv)."""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parts = [pd.read_csv(p).assign(notebook=Path(p).stem.replace("_checks_", "")) for p in sorted(glob.glob(str(ROOT / "notebooks" / "_checks_0[1-5].csv")))]
df = pd.concat(parts, ignore_index=True)
df["note"] = df.note.fillna("")
counts = df.status.value_counts()

LEGEND = {"✅": "reproduced to the reported precision", "≈": "seed-dependent bootstrap bound, within 0.01 of the reported one",
          "⚠": "presence or qualitative check, not a numeric match", "❌": "does not match the printed value",
          "⛔": "cannot be recomputed from the released files"}
NB = {"01": "01_behavior_and_norms", "02": "02_representation", "03": "03_conditional_intervention",
      "04": "04_manipulation_check", "05": "05_preregistered_grid"}


def table(sub: pd.DataFrame) -> str:
    rows = ["| id | where | analysis | reported | produced | status | note |", "|---|---|---|---|---|---|---|"]
    for r in sub.itertuples():
        rows.append(f"| {r.id} | {r.section} | {r.analysis} | {r.reported} | {r.produced} | {r.status} | {r.note} |".replace("\n", " "))
    return "\n".join(rows)


out = [f"# Reproduction report\n",
       "Every number the paper reports, checked against the value recomputed or loaded from the released files by the "
       "executed notebooks in `notebooks/`. Regenerate with `python3 notebooks/build_notebooks.py --execute && python3 notebooks/make_report.py`.\n",
       "## Verdicts\n",
       "| status | meaning | count |", "|---|---|---|"]
for k, v in LEGEND.items():
    out.append(f"| {k} | {v} | {int(counts.get(k, 0))} |")
out.append(f"| | total | {len(df)} |\n")
out.append("## Items that did not reproduce exactly (❌), are approximate (≈) or cannot be recomputed (⛔)\n")
out.append(table(df[df.status.isin(["❌", "≈", "⛔"])]))
out.append("\n## Full table by notebook\n")
for nn, name in NB.items():
    sub = df[df.notebook == nn]
    if len(sub):
        c = sub.status.value_counts().to_dict()
        out.append(f"### `{name}.ipynb` — " + " · ".join(f"{k} {v}" for k, v in c.items()) + "\n")
        out.append(table(sub) + "\n")
(ROOT / "REPRODUCTION_REPORT.md").write_text("\n".join(out))
print("wrote REPRODUCTION_REPORT.md:", counts.to_dict())
