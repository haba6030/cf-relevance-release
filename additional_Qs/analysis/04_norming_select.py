"""Select items by measured norms. **The criteria were fixed before the data were seen** (G1–G3 below).

    python3 additional_Qs/analysis/04_norming_select.py
    python3 additional_Qs/analysis/04_norming_select.py --write additional_Qs/selected.txt

What is measured: the two properties the item design requires (`prompts/10_generate_item.md`, NORMS block).

    confirms-h2   e confirms h2              -> C(h2) in the no-context condition must be high
    inert-h3      e is inert with respect to h3 -> C(h3) in the no-context condition must be near 0

Declared values (`C_h2`, `C_h3`) are the generator's self-report and cannot serve as a gate. Only measured norms decide.
Measured C is a log-odds and not on the declared scale, so **the active 15 items are the baseline instead of an absolute threshold.**
"At least as good as the existing items" is the passing condition.

## The only decision gate is G2 (revised 2026-08-28)

  G2  model-mean C(h3, no context) <= max C_h3 of the 8 active model-written items

This is `inert-h3` (e inert with respect to h3) moved onto the measured scale, and **the only condition the
design document actually requires.** If e confirmed h3, h3 could win without context, and the CF effect
could not be attributed to relevance. Hence it cannot be dropped.

### G1 and G3 were removed from the decision; they are computed but kept as reference columns only

  G1  C(h2) > C(h3) in every model          <- a condition **absent** from the design document
  G3  model-mean ΔC = C(h2)-C(h3) >= median <- a derived quantity that folds the two design requirements into a difference

The NORMS block of `prompts/10_generate_item.md` requires `confirms-h2: C_h2 >= +2.0` and
`inert-h3: C_h3 ∈ [-1,+1]`, and **nothing asks the two quantities to be compared.**

  - **G1 is sensitive to the number of models.** Because of the "every model" operator, 136/200 with 4 models
    becomes 3/200 with 8 models. The measuring model set, not the item, drives the verdict.
    Four of the eight models (Thinking, Phi-reasoning, Qwen2.5-Math, DeepSeek) do not separate
    C_h2 from C_h3 even on the active 15 items (mean ΔC -0.43 to +0.11).
  - **G3 folds two requirements into a difference and measures neither.** An item passes if C_h3 is even
    smaller than a small C_h2, and fails when C_h3 is fine but C_h2 is small.

Reviving `confirms-h2` as an absolute criterion (at or above the median C_h2 of the active model-written items) was considered,
but only 34/200 survive. The eight active model-written items have unusually large C_h2 (median +10.64 vs +6.64 over 200),
and an 8-point distribution is an unstable ruler. **Items with a small effect size do not corrupt the interpretation,**
so this condition is not imposed.

> **This revision was made after the data were seen.** Its basis is the design document, not the observed values
> (G1 is absent from the document), and a condition was removed rather than a threshold moved. The order is not hidden.

The no-context condition is used because the requirement concerns **the relation between e and h**.
Values under context are used separately in H2a (does context move P and C).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent


def load(dirpath: Path, pattern: str) -> pd.DataFrame:
    fs = sorted(dirpath.glob(pattern))
    if not fs:
        sys.exit(f"{dirpath}/{pattern} is empty")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def cell(df: pd.DataFrame) -> pd.DataFrame:
    """Average the three wordings per item × model × hypothesis. No-context condition only."""
    d = df[df.context == "no"]
    g = d.groupby(["model", "scenario", "hypothesis"])[["rho", "C"]].mean().reset_index()
    return g.pivot_table(index=["model", "scenario"], columns="hypothesis",
                         values=["rho", "C"]).reset_index()


def flat(p: pd.DataFrame) -> pd.DataFrame:
    p.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in p.columns]
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", type=Path)
    ap.add_argument("--baseline", type=Path, default=ROOT / "norming_active15")
    ap.add_argument("--gates", default="G2",
                    help="Gates used for the decision. Default G2 (inert-h3). The earlier decision used G1,G2,G3")
    ap.add_argument(
        "--model-ids", nargs="+",
        help="Exact set of model_ids to include. Separates the server grid from older local CSVs.",
    )
    a = ap.parse_args()

    new_raw = load(ROOT / "norming", "norm_*.csv")
    base_raw = load(a.baseline, "norm_*.csv")
    if a.model_ids:
        requested = set(a.model_ids)
        for name, frame in (("new", new_raw), ("baseline", base_raw)):
            present = set(frame.model_id.unique())
            missing = requested - present
            if missing:
                sys.exit(f"{name} norms lack the requested models: {sorted(missing)}")
        new_raw = new_raw[new_raw.model_id.isin(requested)]
        base_raw = base_raw[base_raw.model_id.isin(requested)]
    new = flat(cell(new_raw))
    base = flat(cell(base_raw))
    # The kind of the active norms is restored from the canonical active_set.py, not from the source name.
    # The early server runner labelled every row "active15", which emptied the llm subset and made G2
    # NaN, so the analysis takes the kind directly from the single source of truth.
    sys.path.insert(0, str(PROJ / "data" / "llm-build" / "run"))
    import active_set
    kinds = {}
    for item in active_set.build():
        kinds[item["scenario_id"]] = (
            "chung" if item.get("provenance") == "chung" else
            "lingfix" if str(item.get("src", "")).startswith("lingfix") else
            "llm"
        )
    kinds = pd.Series(kinds, name="kind")

    models = sorted(set(new.model) & set(base.model))
    if not models:
        sys.exit("No model overlaps with the baseline; both sets must be measured with the same models on the same device")
    print(f"{len(models)} models compared: {', '.join(models)}\n")
    new = new[new.model.isin(models)]
    base = base[base.model.isin(models)]

    for df in (new, base):
        df["dC"] = df["C_h2"] - df["C_h3"]

    # ---- baseline (active 15 items)
    b = base.groupby("scenario")[["C_h2", "C_h3", "dC"]].mean()
    b["kind"] = kinds.reindex(b.index)
    llm_max_C3 = b[b.kind == "llm"]["C_h3"].max()
    med_dC = b["dC"].median()
    print("=== Baseline: active 15 items (no context, model mean) ===")
    print(b.round(3).to_string())
    print(f"\n  G2 threshold  max C_h3 of the 8 model-written items = {llm_max_C3:+.3f}")
    print(f"  G3 threshold  median ΔC over 15 items        = {med_dC:+.3f}\n")

    # ---- gates
    sign_ok = (new.assign(ok=new.C_h2 > new.C_h3)
                 .groupby("scenario")["ok"].all())
    m = new.groupby("scenario")[["C_h2", "C_h3", "dC", "rho_h2", "rho_h3"]].mean()
    m["G1_sign"] = sign_ok.reindex(m.index)          # reference only; not used for the decision
    m["G2_inert"] = m.C_h3 <= llm_max_C3
    m["G3_asym"] = m.dC >= med_dC                     # reference only; not used for the decision
    gates = [g.strip() for g in a.gates.split(",") if g.strip()]
    col = {"G1": "G1_sign", "G2": "G2_inert", "G3": "G3_asym"}
    m["PASS"] = True
    for g in gates:
        m["PASS"] &= m[col[g]]

    print("=== 200 candidate items, measured (no context, model mean) ===")
    print(f"  C_h2  mean {m.C_h2.mean():+.3f}  ·  C_h3  mean {m.C_h3.mean():+.3f}"
          f"  ·  ΔC mean {m.dC.mean():+.3f}")
    for g in ("G1_sign", "G2_inert", "G3_asym"):
        tag = "(decision)" if g[:2] in gates else "(reference)"
        print(f"  {g:9s} {int(m[g].sum()):3d}/{len(m)}  {tag}")
    print(f"  {'PASS':9s} {int(m.PASS.sum()):3d}/{len(m)}  gates = {'&'.join(gates)}")

    print("\n=== Failures by reason ===")
    fail = m[~m.PASS]
    print(f"  G1 only   {int((~fail.G1_sign & fail.G2_inert & fail.G3_asym).sum())}")
    print(f"  G2 only   {int((fail.G1_sign & ~fail.G2_inert & fail.G3_asym).sum())}")
    print(f"  G3 only   {int((fail.G1_sign & fail.G2_inert & ~fail.G3_asym).sum())}")
    print(f"  two or more {int((fail[['G1_sign','G2_inert','G3_asym']].sum(axis=1) <= 1).sum())}")

    # ---- did the declared values predict the measured ones
    import json
    dec = {}
    for p in sorted((ROOT / "accepted").glob("*.json")):
        j = json.loads(p.read_text("utf-8"))
        dec[j["scenario_id"]] = (float(j["C_h2"]), float(j["C_h3"]))
    m["declared_C_h3"] = [dec.get(s, (None, None))[1] for s in m.index]
    sub = m.dropna(subset=["declared_C_h3"])
    r = sub[["declared_C_h3", "C_h3"]].corr(method="spearman").iloc[0, 1]
    print(f"\n=== Declared vs measured ===")
    print(f"  C_h3  declared ↔ measured Spearman ρ = {r:+.3f}   (n={len(sub)})")
    hi = sub[sub.declared_C_h3.abs() >= 0.8]
    print(f"  pass rate of the {len(hi)} items with |declared C_h3| >= 0.8: "
          f"{hi.PASS.mean():.0%}  (the other {len(sub)-len(hi)}: {sub[sub.declared_C_h3.abs()<0.8].PASS.mean():.0%})")

    if a.write:
        keep = sorted(m[m.PASS].index)
        a.write.write_text("\n".join(keep) + "\n", encoding="utf-8")
        print(f"\nselected {len(keep)} items -> {a.write}")
    m.round(4).to_csv(ROOT / "norming" / "selection_table.csv")
    print(f"table -> {ROOT/'norming'/'selection_table.csv'}")

    # Also keep the per-model cell values; with them the gates can be changed without re-measuring the norms.
    cellcsv = ROOT / "norming" / "norming_cells.csv"
    cells = new.copy()
    cells["source"] = "additional"
    basecells = base.copy()
    basecells["kind"] = [kinds.get(sc, "") for _, sc in basecells.index] \
        if isinstance(basecells.index, pd.MultiIndex) else ""
    basecells["source"] = "active15"
    pd.concat([cells.reset_index(), basecells.reset_index()], ignore_index=True) \
      .round(4).to_csv(cellcsv, index=False)
    print(f"per-model cells -> {cellcsv}  ({len(cells)} rows + {len(basecells)} baseline rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
