#!/usr/bin/env python3
"""Conditional CF rescue (ADD) and knockout (ABL) on the locked RQ3 harness.

Selection and evaluation are cross-framed: frame 0 selects the baseline state
for evaluation on frame 1, and vice versa.  A held item's directions and scale
are always estimated from the other 164 items.
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cf_rescue_common import (
    ABL_DOSES, ADD_ALPHAS, CONTRACT, DIRECTION_NAMES, FRONT, MODELS, SEED,
    baseline_rows, fold_directions, load_stimuli, safe_name, selected_layers,
)

import sys
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "analysis" / "rq3"))
from analyze_rq3_representation import norm_arrays  # noqa: E402
from run_rq3_behavior_gpu import BudgetedScorer, FRAME  # noqa: E402
from run_rq3_intervention_gpu import decoder_layers, intro_start_token  # noqa: E402
from rq3_prompting import PROMPT_CONTRACT_VERSION, behavior_prompt  # noqa: E402
from rq3_decoder_contract import H3_DECODER_CONTRACT  # noqa: E402


class SelectedDirectionHook(AbstractContextManager):
    """Selected-layer implementation of the canonical ADD/ABL equations."""

    def __init__(self, model, direction: np.ndarray, layer: int, mode: str,
                 magnitude: float = 0.0, dose: float = 1.0) -> None:
        if mode not in {"ADD", "ABL"}:
            raise ValueError(mode)
        vector = np.asarray(direction, dtype=np.float64)
        self.direction = vector / max(float(np.linalg.norm(vector)), 1e-12)
        self.model, self.layer, self.mode = model, int(layer), mode
        self.magnitude, self.dose = float(magnitude), float(dose)
        self.start_positions: list[int] | None = None
        self.handle = None

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor) or tensor.ndim != 3:
            return tensor
        vector = torch.as_tensor(self.direction, device=tensor.device, dtype=tensor.dtype)
        if self.mode == "ABL":
            score = torch.einsum("bsd,d->bs", tensor, vector)
            return tensor - self.dose * score[..., None] * vector
        if self.start_positions is None or len(self.start_positions) != tensor.shape[0]:
            raise RuntimeError("ADD hook requires one left-padding-aware start per batch row")
        positions = torch.arange(tensor.shape[1], device=tensor.device)[None, :]
        starts = torch.as_tensor(self.start_positions, device=tensor.device)[:, None]
        mask = (positions >= starts).to(tensor.dtype)
        return tensor + self.magnitude * mask[..., None] * vector

    def hook(self, _module, _inputs, output):
        if torch.is_tensor(output):
            return self.transform(output)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (self.transform(output[0]), *output[1:])
        if isinstance(output, list) and output and torch.is_tensor(output[0]):
            return [self.transform(output[0]), *output[1:]]
        raise RuntimeError(f"Unexpected selected-layer output: {type(output)}")

    def __enter__(self):
        layers = decoder_layers(self.model)
        module = self.model.get_input_embeddings() if self.layer == 0 else layers[self.layer - 1]
        self.handle = module.register_forward_hook(self.hook)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
        return False


@torch.inference_mode()
def score_condition(scorer: BudgetedScorer, controller: SelectedDirectionHook,
                    prompt: str, start: int, continuations: dict[str, str]) -> dict:
    jobs = []
    for name, continuation in continuations.items():
        prompt_ids = scorer.tok(prompt, return_tensors="pt").input_ids[0]
        full_ids = scorer.tok(prompt + continuation, return_tensors="pt").input_ids[0]
        prompt_len = int(prompt_ids.numel()); nt = int(full_ids.numel()) - prompt_len
        if nt <= 0 or not torch.equal(full_ids[:prompt_len], prompt_ids):
            raise RuntimeError("Token boundary changed during CF rescue scoring")
        jobs.append((name, full_ids, prompt_len, nt))
    padded = scorer.tok.pad(
        {"input_ids": [full for _, full, _, _ in jobs]}, padding=True,
        return_tensors="pt",
    )
    input_ids = padded["input_ids"].to(scorer.input_device)
    attention_mask = padded["attention_mask"].to(scorer.input_device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    padded_length = input_ids.shape[1]
    controller.start_positions = [padded_length - len(full) + start for _, full, _, _ in jobs]
    keep = max(nt for _, _, _, nt in jobs) + 1
    logits = scorer._forward(input_ids, attention_mask, position_ids, keep)
    result: dict[str, float | int] = {}
    for index, (name, full_ids, prompt_len, nt) in enumerate(jobs):
        begin = keep - nt - 1
        token_logits = logits[index, begin: begin + nt]
        targets = full_ids[prompt_len:].to(logits.device)
        lp = token_logits.log_softmax(-1).gather(-1, targets[:, None]).sum()
        result[f"lp_{name}"] = float(lp.cpu()); result[f"nt_{name}"] = nt
    controller.start_positions = None
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--front", type=Path, default=FRONT)
    ap.add_argument("--tasks", nargs="+", type=int, choices=(2, 3), default=(2, 3))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--qvar", type=int, default=0, choices=range(6))
    ap.add_argument("--max-gpu-mib", type=int, default=24_000)
    args = ap.parse_args()
    if args.max_gpu_mib > 24_000 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid GPU ceiling or shard")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(int(os.environ.get("RQ3_TORCH_THREADS", "1")))

    stimuli = load_stimuli(); labels = stimuli.label.drop_duplicates().tolist()
    work = [i for i in range(len(labels)) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        work = work[:args.limit]
    baseline, selection = baseline_rows(args.model)
    slug = safe_name(args.model)
    norming = pd.read_csv(args.front / "norming" / f"norm_rq3_{slug}.csv")
    rho, confirmation = norm_arrays(norming, labels)
    gate_file = args.front / "analysis" / "rq3a_context_decoder_robustness.csv"
    scorer = BudgetedScorer(args.model, args.max_gpu_mib, reserve_mib=1_000)
    completed: list[int] = []
    try:
        for task in args.tasks:
            layers = selected_layers(gate_file, args.model, task)
            array = np.load(
                args.front / "activations" / slug / f"task{task}_cellmean.npy",
                mmap_mode="r",
            )
            meta = json.loads((args.front / "activations" / slug /
                               f"task{task}_metadata.json").read_text())
            if meta["labels"] != labels:
                raise RuntimeError("Activation/stimulus label order mismatch")
            out_dir = args.output_dir / "shards" / slug / f"task{task}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for held in work:
                label = labels[held]; out_path = out_dir / f"{label}.csv"
                if out_path.exists():
                    old = pd.read_csv(out_path, nrows=1)
                    if "experiment_contract" in old and set(old.experiment_contract) == {CONTRACT}:
                        print(f"SKIP task={task} item={label}", flush=True); continue
                    raise RuntimeError(f"Refusing to mix a legacy shard: {out_path}")
                train = np.flatnonzero(np.arange(len(labels)) != held)
                layer = int(layers[held])
                directions, diagnostics = fold_directions(
                    array, rho, confirmation, train, layer,
                    SEED + task * 10_000 + held,
                )
                stimulus = stimuli[(stimuli.label == label) & (stimuli.prefix == "R")].iloc[0]
                rows = []
                for select_frame, eval_frame in ((0, 1), (1, 0)):
                    sel = selection[(selection.task == task) & (selection.label == label)
                                    & (selection.frame == select_frame)]
                    if len(sel) != 1:
                        raise RuntimeError("Missing unique selection-frame baseline")
                    selection_cf = float(sel.iloc[0].selection_cf)
                    stage = "ADD" if selection_cf <= 0 else "ABL"
                    base = baseline[(baseline.task == task) & (baseline.label == label)
                                    & (baseline.prefix == "R")
                                    & (baseline.frame == eval_frame)
                                    & (baseline.qvar == args.qvar)]
                    base_i = baseline[(baseline.task == task) & (baseline.label == label)
                                      & (baseline.prefix == "I")
                                      & (baseline.frame == eval_frame)
                                      & (baseline.qvar == args.qvar)]
                    if len(base) != 1 or len(base_i) != 1:
                        raise RuntimeError("Missing unique evaluation-frame baseline")
                    prompt = behavior_prompt(stimulus.intro, task, eval_frame, args.qvar)
                    start = intro_start_token(scorer.tok, FRAME[eval_frame], prompt)
                    continuations = {
                        "base": " " + stimulus.opt_base,
                        "conj_h2": " " + stimulus.opt_c2,
                        "conj_h3": " " + stimulus.opt_c3,
                    }
                    strengths = ADD_ALPHAS if stage == "ADD" else ABL_DOSES
                    for direction_name in DIRECTION_NAMES:
                        for strength in strengths:
                            kwargs = ({"magnitude": strength * diagnostics["scale"]}
                                      if stage == "ADD" else {"dose": strength})
                            with SelectedDirectionHook(
                                scorer.model, directions[direction_name], layer, stage, **kwargs
                            ) as controller:
                                values = score_condition(
                                    scorer, controller, prompt, start, continuations
                                )
                            rows.append({
                                "model_id": args.model, "task": task, "label": label,
                                "held": held, "layer": layer,
                                "select_frame": select_frame, "eval_frame": eval_frame,
                                "qvar": args.qvar, "selection_cf": selection_cf,
                                "selection_state": "absent" if stage == "ADD" else "present",
                                "stage": stage, "direction": direction_name,
                                "alpha": strength if stage == "ADD" else np.nan,
                                "dose": strength if stage == "ABL" else np.nan,
                                "baseline_cf_eval": float(base.iloc[0].cf),
                                "baseline_gap_R_eval": float(base.iloc[0].gap),
                                "baseline_gap_I_eval": float(base_i.iloc[0].gap),
                                "baseline_base_eval": float(base.iloc[0].base),
                                **values, **diagnostics,
                                "experiment_contract": CONTRACT,
                                "prompt_contract": PROMPT_CONTRACT_VERSION,
                                "decoder_contract": H3_DECODER_CONTRACT,
                            })
                frame = pd.DataFrame(rows)
                temporary = out_path.with_suffix(".csv.tmp")
                frame.to_csv(temporary, index=False); temporary.replace(out_path)
                scorer.check_budget(f"cf-rescue-task{task}-{label}")
                print(f"SAVED task={task} item={label} rows={len(frame)}", flush=True)
            completed.append(task)
        memory = scorer.memory_manifest()
    finally:
        scorer.close()
    manifests = args.output_dir / "manifests"; manifests.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "rq3-cf-rescue-knockout", "model": args.model,
        "tasks": completed, "shard_index": args.shard_index,
        "num_shards": args.num_shards, "limit": args.limit,
        "add_alphas": list(ADD_ALPHAS), "abl_doses": list(ABL_DOSES),
        "qvar": args.qvar, "items_total": len(labels),
        "selection": "opposite frame mean over six locked baseline qvars",
        "evaluation": "other frame, fixed qvar; cross-swapped",
        "directions": list(DIRECTION_NAMES), "experiment_contract": CONTRACT,
        "prompt_contract": PROMPT_CONTRACT_VERSION,
        "decoder_contract": H3_DECODER_CONTRACT, "memory": memory,
    }
    name = f"cf_rescue_{slug}_shard{args.shard_index}of{args.num_shards}.json"
    (manifests / name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
