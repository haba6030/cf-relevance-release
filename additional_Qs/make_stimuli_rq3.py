"""Expand the `additional_Qs` items into the RQ3 stimulus format (item × prefix N/I/R).

    python3 additional_Qs/make_stimuli_rq3.py
    python3 additional_Qs/make_stimuli_rq3.py --items additional_Qs/selected.txt

The output `additional_Qs/stimuli_RQ3_additional.csv` has **the same columns** as `data/stimuli_RQ3.csv`.
If the columns differ, the RQ3 analysis scripts cannot read it.

Prefixes follow the RQ3 design document §1.1.

    N  no context          intro = e
    I  irrelevant context  intro = ctx_irr   (carrier filled with org_irr)
    R  relevant context    intro = ctx       (carrier filled with org)

`I` and `R` form a **minimal pair** differing only in the ORG slot; `build.py` already guarantees
that through `chung_frame.build()`. This script only expands.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent

PREFIXES = [("N", "ctx_none", "no context"), ("I", "ctx_irr", "irrelevant context"), ("R", "ctx", "relevant context")]

COLUMNS = ["label", "kind", "scenario_id", "paradigm", "tgt", "prefix", "prefix_name",
           "intro", "intro_words", "org_used", "org_source", "e", "h1", "h2", "h3", "QUD",
           "opt_base", "opt_c2", "opt_c3", "opt_b2", "opt_b3",
           "norm_source", "P_h2", "P_h3", "C_h2", "C_h3"]


def bare(tgt: str, frag: str) -> str:
    """Bare arm: prepend the subject of h1 to h_i (spec §4.4)."""
    return f"{tgt} {str(frag).strip().rstrip('.')}."


def conj(h1: str, frag: str) -> str:
    return f"{h1.rstrip('.')} and {str(frag).strip().rstrip('.')}."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, help="one scenario_id per line; default: all of accepted/")
    ap.add_argument("--out", type=Path, default=ROOT / "stimuli_RQ3_additional.csv")
    a = ap.parse_args()

    keep = None
    if a.items:
        keep = {ln.strip() for ln in a.items.read_text("utf-8").splitlines() if ln.strip()}

    rows = []
    n_items = 0
    for p in sorted((ROOT / "accepted").glob("*.json")):
        it = json.loads(p.read_text("utf-8"))
        if keep is not None and it["scenario_id"] not in keep:
            continue
        n_items += 1
        for tag, field, name in PREFIXES:
            intro = it[field]
            rows.append(dict(
                label=it["scenario_id"], kind="additional", scenario_id=it["scenario_id"],
                paradigm=it.get("paradigm", "MA"), tgt=it["tgt"],
                prefix=tag, prefix_name=name, intro=intro,
                intro_words=len(re.findall(r"[A-Za-z][A-Za-z']*", intro)),
                org_used=(it["org"] if tag == "R" else it["org_irr"] if tag == "I" else ""),
                org_source="additional_Qs", e=it["e"], h1=it["h1"], h2=it["h2"], h3=it["h3"],
                QUD=it["Q"],
                opt_base=it["h1"], opt_c2=conj(it["h1"], it["h2"]), opt_c3=conj(it["h1"], it["h3"]),
                opt_b2=bare(it["tgt"], it["h2"]), opt_b3=bare(it["tgt"], it["h3"]),
                norm_source="declared", P_h2=it["P_h2"], P_h3=it["P_h3"],
                C_h2=it["C_h2"], C_h3=it["C_h3"]))

    if not rows:
        sys.exit("no items")
    df = pd.DataFrame(rows)[COLUMNS]

    # --- checks: grid and minimal pair --------------------------------------
    bad = []
    if len(df) != n_items * 3 or set(df.prefix) != {"N", "I", "R"}:
        bad.append(f"grid is not items × {{N,I,R}}: {len(df)} rows / {n_items} items")
    for sid, g in df.groupby("scenario_id"):
        d = {r.prefix: r.intro for r in g.itertuples()}
        if d["N"].strip() != g.e.iloc[0].strip():
            bad.append(f"{sid}: N intro differs from e")
        if d["I"] == d["R"]:
            bad.append(f"{sid}: I and R are identical")
        # I and R must differ in the ORG slot only
        oi, orr = g[g.prefix == "I"].org_used.iloc[0], g[g.prefix == "R"].org_used.iloc[0]
        if d["I"].replace(oi, orr) != d["R"]:
            bad.append(f"{sid}: I and R are not related by a single ORG-slot substitution")
    if bad:
        for b in bad:
            print("  " + b, file=sys.stderr)
        sys.exit(f"{len(bad)} failures")

    df.to_csv(a.out, index=False)
    print(f"{n_items} items × 3 prefixes = {len(df)} rows -> {a.out}")
    print(f"  intro word count: N {df[df.prefix=='N'].intro_words.mean():.1f} · "
          f"I {df[df.prefix=='I'].intro_words.mean():.1f} · "
          f"R {df[df.prefix=='R'].intro_words.mean():.1f}")
    d = df.pivot_table(index="scenario_id", columns="prefix", values="intro_words")
    print(f"  I vs R word-count difference: mean {(d.R-d.I).abs().mean():.2f} · max {(d.R-d.I).abs().max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
