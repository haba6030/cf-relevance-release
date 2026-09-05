"""Measured norming: rho and C for the items in `additional_Qs/accepted/`.

    python3 additional_Qs/run_norming.py --dry
    python3 additional_Qs/run_norming.py --models Qwen/Qwen3-1.7B --limit 5
    python3 additional_Qs/run_norming.py --models Qwen/Qwen3-1.7B Qwen/Qwen3-4B

Prompt wordings and answer tokens **reuse `sim/run_norm15.py` unchanged**; only the item loader differs.
New wordings would make the values incomparable with the active 15 items.

    rho = log p("yes") - log p("no")            first token
    C   = log p("strength") - log p("weak")     first token

Output is `additional_Qs/norming/norm_<model>.csv`, appended after every item
(an interrupted run can be resumed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
sys.path.insert(0, str(PROJ / "sim"))
sys.path.insert(0, str(PROJ / "data" / "llm-build" / "run"))

from run_norm15 import QP, prompts  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = ROOT / "norming"


def load_items(limit: int | None = None, source: str = "additional") -> list[dict]:
    """`additional` = additional_Qs/accepted/, `active15` = the active 15 items.

    The baseline (G2/G3 thresholds) comes from the active 15 items, so they must be measured **on the same device at the same precision**.
    `sim/run_norm15.py` is MPS-only (`mps` else `cpu`) and unusable on the server, hence this port.
    Item content and prompt wordings come from that file unchanged.
    """
    if source == "active15":
        sys.path.insert(0, str(PROJ / "data" / "llm-build" / "run"))
        import active_set
        its = []
        for it in active_set.build():
            it["label"] = it.get("label") or it["scenario_id"]
            # selection gate G2 uses the distribution of the 8 active model-written items as baseline.
            # preserve the kind as in the canonical active_set.py rather than the source name ("active15").
            it["kind"] = (
                "chung" if it.get("provenance") == "chung" else
                "lingfix" if str(it.get("src", "")).startswith("lingfix") else
                "llm"
            )
            its.append(it)
        return its[:limit] if limit else its
    its = []
    for p in sorted((ROOT / "accepted").glob("*.json")):
        it = json.loads(p.read_text("utf-8"))
        it["label"] = it["scenario_id"]
        its.append(it)
    return its[:limit] if limit else its


def make(model_name: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, dtype=td).to(DEV).eval()

    def chat(text):
        try:
            return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except TypeError:      # template does not accept enable_thinking
            return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True)

    @torch.no_grad()
    def two_way(prompt, a, b):
        x = tok(prompt, return_tensors="pt").to(DEV)
        lg = mdl(**x).logits[0, -1].float().log_softmax(-1)
        ia = tok.encode(a, add_special_tokens=False)[0]
        ib = tok.encode(b, add_special_tokens=False)[0]
        return float(lg[ia] - lg[ib])

    return chat, two_way, mdl


def run_model(model_name: str, its: list[dict], dtype: str, resume: bool,
              out_dir: Path, kind: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = model_name.split("/")[-1].replace(".", "_")
    out = out_dir / f"norm_{tag}.csv"
    done = set()
    if resume and out.exists():
        # one item is hypothesis 2 × context 2 × variant 3 = 12 rows.
        # keep only complete 12-row blocks so an item whose append was interrupted
        # is not mistaken for finished. A restart guard; it does not change the scoring.
        existing = pd.read_csv(out, on_bad_lines="skip")
        counts = existing.groupby("scenario").size()
        done = set(counts[counts == 12].index)
        if len(existing) != 12 * len(done):
            clean = existing[existing.scenario.isin(done)]
            temporary = out.with_suffix(".csv.tmp")
            clean.to_csv(temporary, index=False)
            temporary.replace(out)
        print(f"    resuming; {len(done)} items already done", file=sys.stderr)

    todo = [it for it in its if it["scenario_id"] not in done]
    if not todo:
        print("    nothing to do", file=sys.stderr)
        return out

    chat, two_way, mdl = make(model_name, dtype)
    t0 = time.time()
    first = not out.exists()
    for n, it in enumerate(todo, 1):
        rows = []
        for k in ("h2", "h3"):
            for ctx in ("no", "yes"):
                for vi in range(len(QP)):
                    pp, cp = prompts(it, k, ctx, vi)
                    rows.append(dict(
                        model=model_name.split("/")[-1], model_id=model_name,
                        scenario=it["scenario_id"],
                        label=it["label"], kind=it.get("kind", kind), hypothesis=k,
                        context=ctx, variant=vi,
                        rho=two_way(chat(pp), "yes", "no"),
                        C=two_way(chat(cp), "strength", "weak")))
        pd.DataFrame(rows).to_csv(out, mode="a", header=first, index=False)
        first = False
        el = time.time() - t0
        print(f"    [{n:3d}/{len(todo)}] {it['scenario_id']:24s} "
              f"{el:6.0f}s  ({el/n:4.1f}s per item · {el/n*(len(todo)-n)/60:5.1f} min left)",
              file=sys.stderr, flush=True)
    del mdl
    if DEV == "mps":
        torch.mps.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-1.7B"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--source", default="additional", choices=["additional", "active15"],
                    help="active15 = the active 15 items (baseline for the G2/G3 thresholds)")
    ap.add_argument("--outdir", type=Path)
    a = ap.parse_args()

    its = load_items(a.limit, a.source)
    outdir = a.outdir or (OUTDIR if a.source == "additional" else ROOT / "norming_active15")
    n = len(its) * 2 * 2 * len(QP) * 2
    print(f"items {len(its)} · prompts {n}/model · device {DEV} · dtype {a.dtype}", file=sys.stderr)
    if a.dry:
        it = its[0]
        pp, cp = prompts(it, "h3", "yes", 0)
        print(f"--- rho ---\n{pp}\n\n--- C ---\n{cp}")
        return 0
    for m in a.models:
        print(f"  {m}", file=sys.stderr)
        out = run_model(m, its, a.dtype, not a.no_resume, outdir, a.source)
        print(f"  -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
