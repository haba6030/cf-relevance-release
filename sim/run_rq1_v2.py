#!/usr/bin/env python3
"""Metric B (log-probability) main run: spec §4.1.

    python3 sim/run_rq1_v2.py --dry            # print sample prompts only
    python3 sim/run_rq1_v2.py                  # full run -> sim/rq1_active2.csv

Design (spec §4.1 grid)
    15 items × 2 contexts × 2 frames × 6 questions × 2 tasks (two/three options) = 720 prompts / model
    **5 strings** scored per prompt: h1, h1∧h2, h1∧h3, {subject} h2, {subject} h3

One run yields all four cells: L2C, L3C, L2B, L3B.
The conjunction and bare arms come from the same prompt, so their between-item correlation
(smoke test S1) can be measured in paired form.

Sign convention (spec §2.1): positive is the hypothesised direction.
    m_i = lp(h1∧h_i)/n_i - lp(h1)/n_1
    DoD = (m3_context - m3_no_context) - (m2_context - m2_no_context)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJ / "data" / "llm-build" / "run"))

from scenarios9 import no_context, options   # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = PROJ / "sim" / "rq1_active2.csv"

# spec §4.1: six question wordings per task; the first is Chung's.
QVAR = {
    2: ["Which of the two statements is more probable?",
        "Which is more probable?",
        "Of the two statements below, which is more likely to be true?",
        "Choose the more probable statement.",
        "Which of the following two is more probable?",
        "Answer with the more probable statement."],
    3: ["Rank the three statements from most to least probable.",
        "Which of the three statements is most probable?",
        "Which statement is most probable?",
        "Which is most likely to be true?",
        "Rank the following by probability. The most probable is:",
        "Answer with the most probable statement."],
}
FRAME = ["Consider the following text:\n\n{intro}\n\n{q}\n", "{intro}\n\n{q}\n"]


def strings(it: dict) -> dict:
    """The 5 strings to score; the conjunction and bare arms share the base (spec §4.0)."""
    o = options(it)                      # base / h2 / h3  = conjunction arm
    tgt = it["tgt"].rstrip(".")          # in `V.` the period is part of the name
    if it["tgt"].endswith("."):
        tgt = it["tgt"]
    return {"base": o["base"],
            "conj_h2": o["h2"], "conj_h3": o["h3"],
            "bare_h2": f"{tgt} {it['h2']}.", "bare_h3": f"{tgt} {it['h3']}."}


def items():
    import active_set
    out = []
    for it in active_set.build():
        kind = ("chung" if it["provenance"] == "chung" else
                "lingfix" if str(it.get("src", "")).startswith("lingfix") else "llm")
        out.append((it, kind))
    return out


def make(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(DEV).eval()

    @torch.no_grad()
    def score(prompt, cont):
        pn = tok(prompt, return_tensors="pt").input_ids.shape[1]
        f = tok(prompt + cont, return_tensors="pt").input_ids.to(DEV)
        lg = mdl(f).logits[0, :-1].log_softmax(-1)
        tgt = f[0, 1:]
        pos = torch.arange(pn - 1, tgt.shape[0], device=DEV)
        lp = lg[pos, tgt[pos]]
        return float(lp.sum()), int(lp.shape[0])
    return score


def run(model_name, score, its):
    rows, t0 = [], time.time()
    for it, kind in its:
        S = strings(it)
        for ctx in ("no", "yes"):
            intro = no_context(it) if ctx == "no" else it["ctx"]
            for task in (2, 3):
                for fi, fr in enumerate(FRAME):
                    for qi, q in enumerate(QVAR[task]):
                        p = fr.format(intro=intro, q=q)
                        r = dict(model=model_name.split("/")[-1],
                                 scenario=it["scenario_id"], label=it.get("label", ""),
                                 kind=kind, paradigm=it["paradigm"],
                                 context=ctx, task=task, frame=fi, qvar=qi)
                        for k, v in S.items():
                            lp, n = score(p, " " + v)
                            r[f"lp_{k}"], r[f"nt_{k}"] = lp, n
                        rows.append(r)
        print(f"    {it['scenario_id']:22s} {time.time()-t0:6.0f}s", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def dry(its):
    it, _ = its[0]
    S = strings(it)
    for ctx in ("no", "yes"):
        intro = no_context(it) if ctx == "no" else it["ctx"]
        for task in (2, 3):
            print(f"\n{'='*76}\n[context={ctx} · task={task} options · FRAME 1 · question 1]\n{'='*76}")
            print(FRAME[0].format(intro=intro, q=QVAR[task][0]))
            print("  ── the 5 continuations to score ──")
            for k, v in S.items():
                print(f"    {k:8s} {v}")
    print(f"\n{'='*76}")
    print(f"items {len(its)} × 2 contexts × 2 tasks × {len(FRAME)} frames × 6 questions "
          f"= {len(its)*2*2*len(FRAME)*6} prompts / model")
    print(f"5 strings per prompt -> {len(its)*2*2*len(FRAME)*6*5} scorings / model")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    its = items()
    if a.dry:
        dry(its)
        return 0

    print(f"items {len(its)} · device {DEV} · models {len(a.models)}", file=sys.stderr)
    frames = []
    for m in a.models:
        print(f"  {m}", file=sys.stderr)
        frames.append(run(m, make(m), its))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"saved {a.out}  ({len(df)} rows × {len(df.columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
