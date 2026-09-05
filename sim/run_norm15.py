#!/usr/bin/env python3
"""Norming run (= RQ2a = H2): spec §4.3, H2/H3 design §1.

    python3 sim/run_norm15.py --dry     # sample prompts
    python3 sim/run_norm15.py           # full run -> sim/norm15.csv

Chung's norming experiment ported to the model; only the dependent variables change, to posterior and confirmation.

    rho = log p("yes") - log p("no")              first token
    C   = log p("strength") - log p("weak")       first token

**The introduction is character-identical to the CF task.** A different stimulus is not "the same manipulation"
and the mediation test (H2b-1) would not hold (H2/H3 design §1.2.1).

The C question **refers to** e rather than restating it; e was already read in the introduction.
Three wordings; **the answer tokens are fixed** so the variants remain comparable.
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

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = PROJ / "sim" / "norm15.csv"

# answer tokens fixed, only the question varies, so the variants remain comparable
QP = ["Is it probable that {hyp}? Answer yes or no.",
      "Would you say that {hyp}? Answer yes or no.",
      "Based on the above, is it likely that {hyp}? Answer yes or no."]
# **refers to** e without restating it; it was already read in the introduction.
QC = ["Consider the hypothesis that {hyp}.\n"
      "Does the fact that {ev} strengthen or weaken that hypothesis? "
      "Answer strengthen or weaken.",
      "Consider the hypothesis that {hyp}.\n"
      "Given that {ev}, does that hypothesis strengthen or weaken? "
      "Answer strengthen or weaken.",
      "Consider the hypothesis that {hyp}.\n"
      "How does the fact that {ev} affect that hypothesis — strengthen or weaken? "
      "Answer strengthen or weaken."]


def hyp_str(it: dict, k: str) -> str:
    """`{tgt} {h_k}`; avoids a doubled period when the name ends in one, as in `V.`."""
    return f"{it['tgt']} {it[k]}".replace("V. V.", "V.")


def prompts(it: dict, k: str, ctx: str, vi: int) -> tuple[str, str]:
    """Introduction identical to the CF task (spec §4.0): no context = e, context = carrier."""
    intro = it["e"] if ctx == "no" else it["ctx"]
    h = hyp_str(it, k)
    ev = it["e"].rstrip(".")                 # referred to, so drop the period
    p = f"{intro}\n\n" + QP[vi].format(hyp=h)
    c = f"{intro}\n\n" + QC[vi].format(hyp=h, ev=ev)
    return p, c


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

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def two_way(prompt, a, b):
        x = tok(prompt, return_tensors="pt").to(DEV)
        lg = mdl(**x).logits[0, -1].log_softmax(-1)
        ia = tok.encode(a, add_special_tokens=False)[0]
        ib = tok.encode(b, add_special_tokens=False)[0]
        return float(lg[ia] - lg[ib])
    return chat, two_way


def run(model_name, chat, two_way, its):
    rows, t0 = [], time.time()
    for it, kind in its:
        for k in ("h2", "h3"):
            for ctx in ("no", "yes"):
                for vi in range(len(QP)):
                    pp, cp = prompts(it, k, ctx, vi)
                    rows.append(dict(
                        model=model_name.split("/")[-1], scenario=it["scenario_id"],
                        label=it.get("label", ""), kind=kind, hypothesis=k,
                        context=ctx, variant=vi,
                        rho=two_way(chat(pp), "yes", "no"),
                        C=two_way(chat(cp), "strength", "weak")))
        print(f"    {it['scenario_id']:22s} {time.time()-t0:6.0f}s", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def dry(its):
    it, _ = its[0]
    for ctx in ("no", "yes"):
        pp, cp = prompts(it, "h3", ctx, 0)
        print(f"\n{'='*76}\n[context={ctx} · h3 · wording 1]\n{'='*76}")
        print(f"--- rho prompt (answers: yes / no) ---\n{pp}")
        print(f"\n--- C prompt (answers: strengthen / weaken) ---\n{cp}")
    print(f"\n{'='*76}")
    n = len(its) * 2 * 2 * len(QP)
    print(f"items {len(its)} × 2 hypotheses × 2 contexts × {len(QP)} wordings × 2 measures "
          f"= {n*2} prompts / model")


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
    print(f"items {len(its)} · device {DEV}", file=sys.stderr)
    frames = []
    for m in a.models:
        print(f"  {m}", file=sys.stderr)
        chat, two_way = make(m)
        frames.append(run(m, chat, two_way, its))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"saved {a.out}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
