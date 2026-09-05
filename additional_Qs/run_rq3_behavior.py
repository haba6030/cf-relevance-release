"""RQ3 behaviour and norming (MPS/CPU version).

    python3 additional_Qs/run_rq3_behavior.py --models Qwen/Qwen3-1.7B --limit 3
    python3 additional_Qs/run_rq3_behavior.py --models Qwen/Qwen3-4B microsoft/Phi-4-mini-instruct

The prompt contract **reuses `analysis/rq3/rq3_prompting.py` unchanged**
(`rq3-v2-raw-sequence-20260825`): raw causal-LM text, no chat template.
Only two things differ from the canonical runner `analysis/rq3/run_rq3_behavior_gpu.py`.

  1. CUDA is not required (the canonical runner refuses on purpose to block CPU fallback)
  2. The stimulus file is an argument (the canonical runner pins `data/stimuli_RQ3.csv` by SHA-256)

**Grid, wordings and scoring are untouched.** Changing them would break comparability with the canonical run.

    behaviour  items × 3 prefixes × 2 tasks × 2 frames × 6 questions = 72N prompts, 5 strings each
    norming    items × 3 prefixes × 2 hypotheses × 2 frames × 3 variants × 2 measures = 72N prompts, 2 continuations each
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
sys.path.insert(0, str(PROJ / "analysis" / "rq3"))

from rq3_prompting import (  # noqa: E402
    FRAME, PROMPT_CONTRACT_VERSION, QVAR, answer_continuations,
    behavior_prompt, norming_prompt,
)

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
OPTS = ("base", "conj_h2", "conj_h3", "bare_h2", "bare_h3")


def make(model_name: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, dtype=td).to(DEV).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.no_grad()
    def score(prompt: str, conts: list[str]) -> list[tuple[float, int]]:
        """Summed log-probability and token count of each continuation by teacher forcing, in one batch."""
        pn = tok(prompt, return_tensors="pt").input_ids.shape[1]
        ids = [tok(prompt + c, return_tensors="pt").input_ids[0] for c in conts]
        L = max(len(x) for x in ids)
        batch = torch.full((len(ids), L), pad, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):
            batch[i, : len(x)] = x
            att[i, : len(x)] = 1
        batch, att = batch.to(DEV), att.to(DEV)
        lg = mdl(batch, attention_mask=att).logits[:, :-1].float().log_softmax(-1)
        tgt = batch[:, 1:]
        lp = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return [(float(lp[i, pn - 1: len(x) - 1].sum()), int(len(x) - pn))
                for i, x in enumerate(ids)]

    return score, mdl


def run_model(model_name: str, stim: pd.DataFrame, dtype: str, out_dir: Path,
              resume: bool) -> None:
    tag = model_name.split("/")[-1].replace(".", "_")
    (out_dir / "behavior").mkdir(parents=True, exist_ok=True)
    (out_dir / "norming").mkdir(parents=True, exist_ok=True)
    fb = out_dir / "behavior" / f"behavior_{tag}.csv"
    fn = out_dir / "norming" / f"norm_{tag}.csv"

    done = set()
    if resume and (fb.exists() or fn.exists()):
        b = pd.read_csv(fb, on_bad_lines="skip") if fb.exists() else pd.DataFrame(columns=["label"])
        n = pd.read_csv(fn, on_bad_lines="skip") if fn.exists() else pd.DataFrame(columns=["label"])
        # an item is complete only with all 72 behaviour rows and 72 norming rows.
        bc = b.groupby("label").size()
        nc = n.groupby("label").size()
        db = set(bc[bc == 72].index)
        dn = set(nc[nc == 72].index)
        done = db & dn
        # items interrupted between the two appends, or partial rows, are removed from both files
        # and recomputed; otherwise a restart duplicates rows in one file.
        if len(b) != 72 * len(done) or len(n) != 72 * len(done):
            for frame, path in ((b, fb), (n, fn)):
                clean = frame[frame.label.isin(done)]
                temporary = path.with_suffix(".csv.tmp")
                clean.to_csv(temporary, index=False)
                temporary.replace(path)
        print(f"    resuming; {len(done)} items done", file=sys.stderr)

    labels = [x for x in stim.label.unique() if x not in done]
    if not labels:
        print("    nothing to do", file=sys.stderr)
        return

    score, mdl = make(model_name, dtype)
    short = model_name.split("/")[-1]
    t0 = time.time()
    hb, hn = not fb.exists(), not fn.exists()

    for n, lab in enumerate(labels, 1):
        g = stim[stim.label == lab]
        brows, nrows = [], []
        for s in g.itertuples(index=False):
            conts = [" " + s.opt_base, " " + s.opt_c2, " " + s.opt_c3,
                     " " + s.opt_b2, " " + s.opt_b3]
            meta = dict(model=short, model_id=model_name, scenario=s.scenario_id,
                        label=s.label, kind=s.kind, paradigm=s.paradigm, prefix=s.prefix)
            # --- behaviour
            for task in (2, 3):
                for fi in range(len(FRAME)):
                    for qi in range(len(QVAR[task])):
                        p = behavior_prompt(s.intro, task, fi, qi)
                        r = dict(meta, task=task, frame=fi, qvar=qi,
                                 prompt_contract=PROMPT_CONTRACT_VERSION)
                        for k, (lp, ntk) in zip(OPTS, score(p, conts)):
                            r[f"lp_{k}"], r[f"nt_{k}"] = lp, ntk
                        brows.append(r)
            # --- norming
            for hyp in ("h2", "h3"):
                hs = f"{s.tgt} {getattr(s, hyp)}".rstrip(".")
                for metric in ("P", "C"):
                    a, b = answer_continuations(metric)
                    for fi in range(len(FRAME)):
                        for vi in range(3):
                            p = norming_prompt(s.intro, hs, metric, fi, vi)
                            (la, na), (lbb, nb) = score(p, [a, b])
                            nrows.append(dict(meta, hypothesis=hyp, metric=metric,
                                              frame=fi, variant=vi,
                                              lp_pos=la, nt_pos=na, lp_neg=lbb, nt_neg=nb,
                                              score=la - lbb,
                                              prompt_contract=PROMPT_CONTRACT_VERSION))
        pd.DataFrame(brows).to_csv(fb, mode="a", header=hb, index=False)
        pd.DataFrame(nrows).to_csv(fn, mode="a", header=hn, index=False)
        hb = hn = False
        el = time.time() - t0
        print(f"    [{n:3d}/{len(labels)}] {lab:24s} {el:6.0f}s "
              f"({el/n:5.1f}s per item · {el/n*(len(labels)-n)/60:5.1f} min left)",
              file=sys.stderr, flush=True)
    del mdl
    if DEV == "mps":
        torch.mps.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-1.7B"])
    ap.add_argument("--stimuli", type=Path, default=ROOT / "stimuli_RQ3_additional.csv")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "rq3_run")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    stim = pd.read_csv(a.stimuli)
    if a.limit:
        keep = list(stim.label.unique())[: a.limit]
        stim = stim[stim.label.isin(keep)]
    n_items = stim.label.nunique()
    if len(stim) != n_items * 3 or set(stim.prefix) != {"N", "I", "R"}:
        sys.exit("stimulus grid is not items × {N,I,R}")
    print(f"items {n_items} · rows {len(stim)} · device {DEV} · dtype {a.dtype}", file=sys.stderr)
    print(f"contract {PROMPT_CONTRACT_VERSION}", file=sys.stderr)
    print(f"behaviour {n_items*3*2*2*6} prompts · norming {n_items*3*2*2*2*3} prompts / model",
          file=sys.stderr)
    for m in a.models:
        print(f"  {m}", file=sys.stderr)
        run_model(m, stim, a.dtype, a.output_dir, not a.no_resume)
    print(f"  -> {a.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
