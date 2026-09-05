"""RQ3 representation extraction (MPS/CPU version): store the residual stream per layer.

    python3 additional_Qs/run_rq3_representation.py --models Qwen/Qwen3-1.7B --limit 3

**The storage layout matches** the canonical `analysis/rq3/run_rq3_representation_gpu.py`.

    task{2,3}_cellmean.npy         (label, prefix, option, hidden_state, dimension)
        read position = last token of the appended option, averaged over the 12 prompt variants (2 frames × 6 questions)
    task{2,3}_prompt_cellmean.npy  (label, prefix, hidden_state, dimension)
        read position = token just before the option, same 12-variant average
    task{2,3}_variant_sd_norm.npy  (label, prefix, option, hidden_state)
        norm of the SD across the 12 variants; records variant noise
    task{2,3}_metadata.json        axis order, labels, prefixes, options, contract

`hidden_state` 0 is the embedding output and $l\\ge1$ the output of decoder block $l$.
The prompt contract is `analysis/rq3/rq3_prompting.py`, unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
sys.path.insert(0, str(PROJ / "analysis" / "rq3"))

from rq3_prompting import (  # noqa: E402
    FRAME, PROMPT_CONTRACT_VERSION, QVAR, behavior_prompt,
)

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
PREFIXES = ("N", "I", "R")
OPTS = ("base", "conj_h2", "conj_h3", "bare_h2", "bare_h3")
OPTCOL = {"base": "opt_base", "conj_h2": "opt_c2", "conj_h3": "opt_c3",
          "bare_h2": "opt_b2", "bare_h3": "opt_b3"}


def atomic_npy(path: Path, value: np.ndarray) -> None:
    """Write so that only a complete array appears at the canonical path."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def make(model_name: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=td, output_hidden_states=True).to(DEV).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.no_grad()
    def states(prompt: str, conts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Run the continuations as one batch.

        Returns (per-option last-token states per layer `(n_cont, L, d)`,
                per-layer state of the token before the option `(L, d)`).
        Right padding with an attention mask, so earlier-token states do not depend on batching.
        """
        pn = tok(prompt, return_tensors="pt").input_ids.shape[1]
        ids = [tok(prompt + c, return_tensors="pt").input_ids[0] for c in conts]
        L = max(len(x) for x in ids)
        batch = torch.full((len(ids), L), pad, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):
            batch[i, : len(x)] = x
            att[i, : len(x)] = 1
        batch, att = batch.to(DEV), att.to(DEV)
        hs = mdl(batch, attention_mask=att, output_hidden_states=True).hidden_states
        H = torch.stack(hs, dim=1)                       # (n, L_layer, T, d)
        idx = torch.tensor([len(x) - 1 for x in ids], device=DEV)
        last = H[torch.arange(len(ids), device=DEV), :, idx].float().cpu().numpy()
        prev = H[0, :, pn - 1].float().cpu().numpy()
        return last, prev

    return states, mdl


def run_model(model_name: str, stim: pd.DataFrame, dtype: str, out_dir: Path,
              tasks: tuple[int, ...], resume: bool) -> None:
    tag = model_name.split("/")[-1].replace("/", "__")
    act = out_dir / "activations" / model_name.replace("/", "__")
    act.mkdir(parents=True, exist_ok=True)
    labels = list(dict.fromkeys(stim.label))
    states, mdl = make(model_name, dtype)

    for task in tasks:
        cm_path = act / f"task{task}_cellmean.npy"
        prompt_path = act / f"task{task}_prompt_cellmean.npy"
        sd_path = act / f"task{task}_variant_sd_norm.npy"
        metadata_path = act / f"task{task}_metadata.json"
        complete_paths = (cm_path, prompt_path, sd_path, metadata_path)
        if resume and all(path.exists() for path in complete_paths):
            try:
                metadata = json.loads(metadata_path.read_text("utf-8"))
                complete = (
                    metadata.get("prompt_contract") == PROMPT_CONTRACT_VERSION
                    and metadata.get("labels") == labels
                    and metadata.get("shape", [None])[0] == len(labels)
                )
            except (OSError, json.JSONDecodeError):
                complete = False
        else:
            complete = False
        if complete:
            print(f"    task{task} already present; skipping", file=sys.stderr)
            continue
        cells, prompts_, sds = [], [], []
        t0 = time.time()
        for n, lab in enumerate(labels, 1):
            g = stim[stim.label == lab].set_index("prefix")
            per_pref, per_pref_prompt, per_pref_sd = [], [], []
            for pf in PREFIXES:
                s = g.loc[pf]
                conts = [" " + str(s[OPTCOL[ok]]) for ok in OPTS]
                vs, ps = [], []
                for fi in range(len(FRAME)):
                    for qi in range(len(QVAR[task])):
                        p = behavior_prompt(s["intro"], task, fi, qi)
                        last, prev = states(p, conts)      # (5, L, d), (L, d)
                        vs.append(last)
                        ps.append(prev)
                V = np.stack(vs)                           # (12, 5, L, d)
                per_pref.append(V.mean(0))                                     # (5, L, d)
                per_pref_sd.append(np.linalg.norm(V.std(0, ddof=1), axis=-1))  # (5, L)
                per_pref_prompt.append(np.stack(ps).mean(0))                   # (L, d)
            cells.append(np.stack(per_pref))                # (3, 5, L, d)
            prompts_.append(np.stack(per_pref_prompt))      # (3, L, d)
            sds.append(np.stack(per_pref_sd))               # (3, 5, L)
            el = time.time() - t0
            print(f"    [task{task}] [{n:3d}/{len(labels)}] {lab:24s} {el:6.0f}s "
                  f"({el/n:5.1f}s per item · {el/n*(len(labels)-n)/60:5.1f} min left)",
                  file=sys.stderr, flush=True)

        A = np.stack(cells).astype(np.float32)
        P = np.stack(prompts_).astype(np.float32)
        S = np.stack(sds).astype(np.float32)
        atomic_npy(cm_path, A)
        atomic_npy(prompt_path, P)
        atomic_npy(sd_path, S)
        metadata = json.dumps(dict(
            model=model_name, task=task, shape=list(A.shape),
            axis_order=["label", "prefix", "option", "hidden_state", "dimension"],
            labels=labels, prefixes=list(PREFIXES), options=list(OPTS),
            hidden_state_definition="0=embedding output; l>=1=decoder block output",
            read_position="last token of attached option",
            cell_variant_count=len(FRAME) * len(QVAR[task]),
            prompt_contract=PROMPT_CONTRACT_VERSION, dtype="float32",
            prompt_final_shape=list(P.shape),
            prompt_final_definition="last token before attached option; averaged over variants",
        ), ensure_ascii=False, indent=2) + "\n"
        metadata_tmp = metadata_path.with_suffix(".json.tmp")
        metadata_tmp.write_text(metadata, encoding="utf-8")
        metadata_tmp.replace(metadata_path)
        print(f"    -> {cm_path}  {A.shape}  {A.nbytes/2**20:.0f}MiB", file=sys.stderr)

    # manifest: downstream analyses locate the model **through this file**.
    # `analyze_rq3_context_decoding_robustness.py` scans manifests/representation_*.json and
    # recovers the activation directory from the file name; without it there are zero cells and it
    # dies with `KeyError: 'family'` (observed).
    man = out_dir / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    tasks_meta = {}
    for task in tasks:
        mp = act / f"task{task}_metadata.json"
        if mp.exists():
            tasks_meta[str(task)] = json.loads(mp.read_text("utf-8"))
    (man / f"representation_{model_name.replace('/', '__')}.json").write_text(
        json.dumps(dict(stage="rq3-norming-activation", model=model_name, dtype=dtype,
                        cpu_offload_allowed=False, prompt_contract=PROMPT_CONTRACT_VERSION,
                        device=DEV, tasks=tasks_meta),
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    del mdl
    if DEV == "mps":
        torch.mps.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-1.7B"])
    ap.add_argument("--stimuli", type=Path, default=ROOT / "stimuli_RQ3_additional.csv")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "rq3_run")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--tasks", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    stim = pd.read_csv(a.stimuli)
    if a.limit:
        stim = stim[stim.label.isin(list(stim.label.unique())[: a.limit])]
    n = stim.label.nunique()
    print(f"items {n} · device {DEV} · dtype {a.dtype} · tasks {a.tasks}", file=sys.stderr)
    print(f"{n*3*12*len(a.tasks)} batched forwards / model (5 options per batch)",
          file=sys.stderr)
    for m in a.models:
        print(f"  {m}", file=sys.stderr)
        run_model(m, stim, a.dtype, a.output_dir, tuple(a.tasks), not a.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
