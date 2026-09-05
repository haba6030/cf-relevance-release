#!/usr/bin/env python3
"""Slot JSON -> carrier assembly -> mechanical checks. Inherits the `data/llm-build` specification unchanged.

    python3 additional_Qs/build.py                 # build and check every item in slots/
    python3 additional_Qs/build.py --promote       # move passing items to accepted/
    python3 additional_Qs/build.py --csv           # export a CSV in the active-15-item schema

This script does three things only.
  1. Build ctx and ctx_irr with `sim/chung_frame.build()` (where the no-free-writing rule is enforced)
  2. Run the rules of `data/llm-build/checks/auto_checks.py` unchanged (no new rules)
  3. Put the existing active 15 items into the corpus so unique-item and domain-cap are checked globally

Judge calls (J1–J5) and measured norming are not here; only items passing the mechanical gate proceed to them.
"""
from __future__ import annotations
import argparse, csv, json, shutil, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
sys.path.insert(0, str(PROJ / "sim"))
sys.path.insert(0, str(PROJ / "data" / "llm-build" / "checks"))
import chung_frame                                   # noqa: E402
import auto_checks as AC                             # noqa: E402

SLOTS, BUILT, ACCEPTED, REJECTED = (ROOT / d for d in ("slots", "built", "accepted", "rejected"))
STIMULI = PROJ / "data" / "stimuli_00_all.csv"

REQUIRED = ("scenario_id domain h3_class tgt consultant e h1 h2 h3 org org_irr Q "
            "ov_word fil_word P_h2 P_h3 C_h2 C_h3").split()


def existing_items() -> list[dict]:
    """Load the active 15 items in the check schema, so duplicates and domain caps are measured globally."""
    out = []
    for r in csv.DictReader(STIMULI.open(encoding="utf-8")):
        out.append(dict(scenario_id=r["label"], domain=r.get("domain", ""),
                        paradigm=r.get("paradigm", "MA"), provenance=r.get("provenance", ""),
                        tgt=r["tgt"], consultant=r.get("consultant", ""), e=r.get("e", ""),
                        h1=r["h1"], h2=r["h2"], h3=r["h3"], Q=r.get("QUD", ""),
                        ctx=r.get("ctx", ""), ctx_irr=r.get("ctx_irr", ""),
                        org=r.get("org", ""), h3_class=r.get("h3_class", "")))
    return out


def build_one(s: dict) -> dict:
    """Fill ctx and ctx_irr into a slot dict to make a complete item."""
    missing = [k for k in REQUIRED if k not in s or s[k] == ""]
    if missing:
        raise ValueError(f"{s.get('scenario_id','?')}: missing fields {missing}")
    it = dict(s)
    it["paradigm"] = s.get("paradigm", "MA")
    it["ctx"] = chung_frame.build(s["consultant"], s["org"], s["tgt"], s["e"])
    it["ctx_irr"] = chung_frame.build(s["consultant"], s["org_irr"], s["tgt"], s["e"])
    it["ctx_none"] = s["e"]
    it["src"] = s.get("src", "additional_Qs")
    it["ctx_status"] = "new"
    it["provenance"] = "new"
    return it


def run_checks(new: list[dict], with_existing: bool = True):
    """Apply the auto_checks rules to new items; the corpus includes the existing items."""
    pool = new + (existing_items() if with_existing else [])
    corpus = {k: Counter() for k in ("pairs", "e", "h1", "h2", "h3", "domain")}
    for it in pool:
        corpus["pairs"][(it.get("h2", "").strip().lower(), it.get("h3", "").strip().lower())] += 1
        for f in ("e", "h1", "h2", "h3"):
            corpus[f][(it.get(f) or "").strip().lower()] += 1
        corpus["domain"][(it.get("domain") or "").strip().lower()] += 1

    res = []
    for it in new:
        allbad = AC.check_item(it, corpus)
        hard = [(r, m) for r, m in allbad if not r.endswith("-w")]
        warn = [(r, m) for r, m in allbad if r.endswith("-w")]
        res.append((it, hard, warn))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true", help="move passing items to accepted/")
    ap.add_argument("--csv", action="store_true", help="export accepted/ in the stimuli CSV schema")
    ap.add_argument("--slots", type=Path, help="slot directory (default additional_Qs/slots)")
    ap.add_argument("--tokenizers", action="store_true",
                    help="re-check length-match with real subword tokenizers (word counts miss these)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    for d in (BUILT, ACCEPTED, REJECTED):
        d.mkdir(exist_ok=True)

    slots_dir = a.slots or SLOTS
    slots = sorted(slots_dir.glob("*.json"))
    if not slots:
        print(f"{slots_dir} is empty", file=sys.stderr); return 2

    new = []
    for p in slots:
        raw = json.loads(p.read_text("utf-8"))
        for s in (raw if isinstance(raw, list) else [raw]):
            it = build_one(s)
            (BUILT / f"{it['scenario_id']}.json").write_text(
                json.dumps(it, ensure_ascii=False, indent=1), encoding="utf-8")
            new.append(it)

    res = run_checks(new)
    npass = 0
    rules, warns = Counter(), Counter()
    for it, hard, warn in res:
        sid = it["scenario_id"]
        if hard:
            for r, m in hard: rules[r] += 1
            if not a.quiet:
                print(f"FAIL  {sid}")
                for r, m in hard: print(f"        {r}  {m}")
                for r, m in warn: print(f"        (warning) {r}  {m}")
        else:
            npass += 1
            for r, m in warn: warns[r] += 1
            if not a.quiet:
                print(f"PASS  {sid}" + (f"   ({len(warn)} warnings)" if warn else ""))
                for r, m in warn: print(f"        (warning) {r}  {m}")

    print("\nSet-level checks:")
    for m in AC.check_balance(new):
        print(f"  {m}")
    doms = Counter((it["domain"] or "").lower() for it in new)
    over = {d: c for d, c in doms.items() if c > 3}
    print(f"  {len(doms)} domains" + (f" · over cap: {over}" if over else " · no domain-cap violation"))

    print(f"\n{npass}/{len(new)} passed  (yield {npass/len(new):.0%})")
    if rules:
        print("violated rules: " + ", ".join(f"{k}×{v}" for k, v in rules.most_common()))
    if warns:
        print("warnings on passing items: " + ", ".join(f"{k}×{v}" for k, v in warns.most_common()))

    if a.tokenizers:
        print("\nlength-match with real tokenizers (what the word-count gate misses):")
        ok = [it for it, hard, _ in res if not hard]
        for m in AC.check_tokenizers(ok):
            print(f"  {m}")
        if not any("length-match[" in m for m in AC.check_tokenizers(ok)):
            print("  no violations")

    if a.promote:
        for it, hard, _ in res:
            dst = (ACCEPTED if not hard else REJECTED) / f"{it['scenario_id']}.json"
            dst.write_text(json.dumps(it, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\naccepted/ {len(list(ACCEPTED.glob('*.json')))} · rejected/ {len(list(REJECTED.glob('*.json')))}")

    if a.csv:
        items = [json.loads(p.read_text("utf-8")) for p in sorted(ACCEPTED.glob("*.json"))]
        cols = list(csv.DictReader(STIMULI.open(encoding="utf-8")).fieldnames)
        out = ROOT / "additional_stimuli.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for it in items:
                row = {c: "" for c in cols}
                row.update(label=it["scenario_id"], kind="additional", scenario_id=it["scenario_id"],
                           paradigm="MA", domain=it["domain"], h3_class=it["h3_class"],
                           tgt=it["tgt"], consultant=it["consultant"], org=it["org"],
                           org_irr=it["org_irr"], e=it["e"], h1=it["h1"], h2=it["h2"], h3=it["h3"],
                           QUD=it["Q"], ov_word=it["ov_word"], fil_word=it["fil_word"],
                           ctx_none=it["e"], ctx=it["ctx"], ctx_irr=it["ctx_irr"],
                           opt_base=it["h1"], opt_h2=f"{it['h1'][:-1]} and {it['h2']}.",
                           opt_h3=f"{it['h1'][:-1]} and {it['h3']}.",
                           norm_source="declared", P_h2=it["P_h2"], P_h3=it["P_h3"],
                           C_h2=it["C_h2"], C_h3=it["C_h3"], check="PASS",
                           provenance="additional_Qs", src=it.get("src", ""))
                w.writerow(row)
        print(f"\nCSV -> {out}  ({len(items)} items)")
    return 0 if npass == len(new) else 1


if __name__ == "__main__":
    sys.exit(main())
