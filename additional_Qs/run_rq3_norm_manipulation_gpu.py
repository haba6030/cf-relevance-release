#!/usr/bin/env python3
"""Re-measure P/C norming under the exact 0830 conditional-CF edits.

The edit direction, selected layer, LOIO fold, seed, cross-frame stage
assignment, and ADD/ABL equations are imported from the executed 0830 runner.
Only the scored prompts change: the current RQ3-v2 norming prompts are scored
under the edit.  The old ``sim/run_norm15.py`` chat prompts are intentionally
not used because the executed directions were trained from
``rq3_run_server/norming`` under the RQ3-v2 raw-sequence contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJ / "analysis" / "rq3"))

from cf_rescue_common import (  # noqa: E402
    FRONT, MODELS, SEED, baseline_rows, fold_directions, load_stimuli,
    safe_name, selected_layers,
)
from run_rq3_cf_rescue_gpu import SelectedDirectionHook  # noqa: E402
from analyze_rq3_representation import norm_arrays  # noqa: E402
from run_rq3_behavior_gpu import BudgetedScorer  # noqa: E402
from run_rq3_intervention_gpu import intro_start_token  # noqa: E402
from rq3_prompting import (  # noqa: E402
    FRAME, PROMPT_CONTRACT_VERSION, answer_continuations, norming_prompt,
)


CONTRACT = "rq3-0830-edit-norming-manipulation-check-v1"
PRIMARY = PROJ / "additional_Qs" / "cf_rescue_reanalysis" / "cells_annotated.csv"
POSTHOC = PROJ / "additional_Qs" / "cf_rescue_fluency_posthoc" / "cells_annotated.csv"
CONTROL_FOR = {
    "R_given_PC": ("random_R_given_PC", "shuffle_R_given_PC"),
    "PC_given_R": ("random_PC_given_R", "shuffle_PC_given_R"),
}
OUTPUT_COLUMNS = (
    "label", "prefix", "hypothesis", "metric", "frame", "variant",
    "baseline_score", "lp_pos", "lp_neg", "edited_score", "model_id",
    "task", "held", "layer", "select_frame", "eval_frame", "selection_cf",
    "stage", "target_direction", "direction", "control_role", "strength",
    "selection_source", "pc_orientation", "scale", "experiment_contract",
    "prompt_contract",
)


def truth(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def target_grid(include_posthoc: bool = True) -> pd.DataFrame:
    """Programmatically recover the 10 primary combinations and all strengths."""
    primary = pd.read_csv(PRIMARY)
    primary["pass"] = truth(primary["pass"])
    primary = primary[primary["pass"]].copy()
    primary["selection_source"] = "0830_primary_pass"
    frames = [primary]
    if include_posthoc:
        posthoc = pd.read_csv(POSTHOC)
        posthoc["pass"] = truth(posthoc["pass"])
        posthoc["effect_pass"] = truth(posthoc["effect_pass"])
        posthoc = posthoc[posthoc.effect_pass & ~posthoc["pass"]].copy()
        posthoc["selection_source"] = "0901_posthoc_fluency_six"
        frames.append(posthoc)
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["strength"] = data.alpha.fillna(data.dose)
    keep = ["model_id", "task", "stage", "direction", "strength", "selection_source"]
    data = data[keep].drop_duplicates().sort_values(keep[:-1]).reset_index(drop=True)
    if not set(data.direction).issubset(CONTROL_FOR):
        raise RuntimeError("Unexpected target direction in selected intervention grid")
    primary_combos = data[data.selection_source == "0830_primary_pass"].drop_duplicates(
        ["model_id", "task", "stage", "direction"]
    )
    if len(primary_combos) != 10 or len(data[data.selection_source == "0830_primary_pass"]) != 19:
        raise RuntimeError("The locked primary grid must contain 10 combinations / 19 strengths")
    if include_posthoc and len(data[data.selection_source == "0901_posthoc_fluency_six"]) != 6:
        raise RuntimeError("The optional post-hoc grid must contain exactly six strengths")
    return data


def baseline_index(norming: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model_id", "label", "prefix", "hypothesis", "metric", "frame",
        "variant", "score", "prompt_contract",
    }
    if required - set(norming):
        raise RuntimeError(f"Norming baseline columns missing: {sorted(required-set(norming))}")
    if set(norming.prompt_contract) != {PROMPT_CONTRACT_VERSION}:
        raise RuntimeError("Norming prompt contract differs from the direction-training contract")
    out = norming[norming.prefix.isin(("I", "R"))].copy()
    keys = ["label", "prefix", "hypothesis", "metric", "frame", "variant"]
    if out.duplicated(keys).any():
        raise RuntimeError("Norming baseline has duplicate exact cells")
    return out.set_index(keys)


@torch.inference_mode()
def score_prompt_block(
    scorer: BudgetedScorer,
    controller: SelectedDirectionHook,
    cells: list[dict],
) -> list[dict]:
    """Score all 24 P/C cells for one edit in a single padded forward pass."""
    prompts = [cell["prompt"] for cell in cells]
    encoded = [scorer.tok(prompt, return_tensors="pt").input_ids[0] for prompt in prompts]
    padded = scorer.tok.pad({"input_ids": encoded}, padding=True, return_tensors="pt")
    input_ids = padded.input_ids.to(scorer.input_device)
    attention_mask = padded.attention_mask.to(scorer.input_device)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    width = int(input_ids.shape[1])
    controller.start_positions = [
        width - len(ids) + int(cell["intro_start"]) for ids, cell in zip(encoded, cells)
    ]
    logits = scorer._forward(input_ids, attention_mask, position_ids, 1)[:, -1]
    log_probs = logits.log_softmax(-1)
    rows = []
    for index, cell in enumerate(cells):
        pos, neg = answer_continuations(cell["metric"])
        pos_ids = scorer.tok.encode(pos, add_special_tokens=False)
        neg_ids = scorer.tok.encode(neg, add_special_tokens=False)
        if len(pos_ids) != 1 or len(neg_ids) != 1:
            raise RuntimeError(f"Norming answers must remain one token: {pos!r}, {neg!r}")
        lp_pos = float(log_probs[index, pos_ids[0]].cpu())
        lp_neg = float(log_probs[index, neg_ids[0]].cpu())
        row = {k: v for k, v in cell.items() if k not in {"prompt", "intro_start"}}
        row.update(lp_pos=lp_pos, lp_neg=lp_neg, edited_score=lp_pos-lp_neg)
        rows.append(row)
    controller.start_positions = None
    return rows


def build_prompt_cells(
    scorer: BudgetedScorer, stimuli: pd.DataFrame, eval_frame: int, base: pd.DataFrame
) -> list[dict]:
    cells: list[dict] = []
    for prefix in ("I", "R"):
        stimulus = stimuli[stimuli.prefix == prefix].iloc[0]
        for hypothesis in ("h2", "h3"):
            hyp = f"{stimulus.tgt} {stimulus[hypothesis]}".rstrip(".")
            for metric in ("P", "C"):
                for variant in range(3):
                    prompt = norming_prompt(stimulus.intro, hyp, metric, eval_frame, variant)
                    key = (stimulus.label, prefix, hypothesis, metric, eval_frame, variant)
                    if key not in base.index:
                        raise RuntimeError(f"Missing exact baseline norming cell: {key}")
                    cells.append({
                        "label": stimulus.label, "prefix": prefix,
                        "hypothesis": hypothesis, "metric": metric,
                        "frame": eval_frame, "variant": variant,
                        "baseline_score": float(base.loc[key, "score"]), "prompt": prompt,
                        "intro_start": intro_start_token(scorer.tok, FRAME[eval_frame], prompt),
                    })
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--front", type=Path, default=FRONT)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-gpu-mib", type=int, default=24_000)
    ap.add_argument("--exclude-posthoc", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.shard_index < args.num_shards or args.max_gpu_mib > 24_000:
        raise ValueError("Invalid shard or GPU ceiling")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(int(os.environ.get("RQ3_TORCH_THREADS", "1")))

    grid = target_grid(not args.exclude_posthoc)
    grid = grid[grid.model_id == args.model].copy()
    if grid.empty:
        print(f"SKIP model={args.model}: no selected manipulation-check cells", flush=True)
        return 0
    all_stimuli = load_stimuli(); labels = all_stimuli.label.drop_duplicates().tolist()
    work = [i for i in range(len(labels)) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        work = work[:args.limit]
    _, selection = baseline_rows(args.model)
    slug = safe_name(args.model)
    baseline_tag = args.model.split("/")[-1].replace(".", "_")
    baseline_norming = pd.read_csv(args.front / "norming" / f"norm_{baseline_tag}.csv")
    direction_norming = pd.read_csv(args.front / "norming" / f"norm_rq3_{slug}.csv")
    base = baseline_index(baseline_norming)
    rho, confirmation = norm_arrays(direction_norming, labels)
    gate_file = args.front / "analysis" / "rq3a_context_decoder_robustness.csv"
    scorer = BudgetedScorer(args.model, args.max_gpu_mib, reserve_mib=1_000)
    completed_tasks: list[int] = []
    try:
        for task in sorted(grid.task.unique()):
            task_grid = grid[grid.task == task]
            layers = selected_layers(gate_file, args.model, int(task))
            array = np.load(args.front / "activations" / slug / f"task{task}_cellmean.npy", mmap_mode="r")
            metadata = json.loads((args.front / "activations" / slug /
                                   f"task{task}_metadata.json").read_text())
            if metadata["labels"] != labels:
                raise RuntimeError("Activation/stimulus label order mismatch")
            out_dir = args.output_dir / "shards" / slug / f"task{task}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for held in work:
                label = labels[held]; out_path = out_dir / f"{label}.csv"
                if out_path.exists():
                    old = pd.read_csv(out_path, nrows=1)
                    contract_ok = (
                        "experiment_contract" in old
                        and (old.empty or set(old.experiment_contract) == {CONTRACT})
                    )
                    if contract_ok:
                        print(f"SKIP task={task} item={label}", flush=True); continue
                    raise RuntimeError(f"Refusing to mix a legacy shard: {out_path}")
                train = np.flatnonzero(np.arange(len(labels)) != held)
                layer = int(layers[held])
                directions, diagnostics = fold_directions(
                    array, rho, confirmation, train, layer, SEED + int(task)*10_000 + held
                )
                item_stimuli = all_stimuli[all_stimuli.label == label]
                rows: list[dict] = []
                for select_frame, eval_frame in ((0, 1), (1, 0)):
                    sel = selection[(selection.task == task) & (selection.label == label)
                                    & (selection.frame == select_frame)]
                    if len(sel) != 1:
                        raise RuntimeError("Missing unique selection-frame baseline")
                    stage = "ADD" if float(sel.iloc[0].selection_cf) <= 0 else "ABL"
                    eligible = task_grid[task_grid.stage == stage]
                    if eligible.empty:
                        continue
                    cells = build_prompt_cells(scorer, item_stimuli, eval_frame, base)
                    for spec in eligible.itertuples(index=False):
                        actual = (spec.direction, *CONTROL_FOR[spec.direction])
                        for actual_direction in actual:
                            kwargs = ({"magnitude": spec.strength * diagnostics["scale"]}
                                      if stage == "ADD" else {"dose": spec.strength})
                            with SelectedDirectionHook(
                                scorer.model, directions[actual_direction], layer, stage, **kwargs
                            ) as controller:
                                scored = score_prompt_block(scorer, controller, cells)
                            for row in scored:
                                row.update({
                                    "model_id": args.model, "task": int(task), "held": held,
                                    "layer": layer, "select_frame": select_frame,
                                    "eval_frame": eval_frame, "selection_cf": float(sel.iloc[0].selection_cf),
                                    "stage": stage, "target_direction": spec.direction,
                                    "direction": actual_direction,
                                    "control_role": ("target" if actual_direction == spec.direction else
                                                     "random" if actual_direction.startswith("random") else "shuffle"),
                                    "strength": float(spec.strength),
                                    "selection_source": spec.selection_source,
                                    "pc_orientation": diagnostics["pc_orientation"],
                                    "scale": diagnostics["scale"],
                                    "experiment_contract": CONTRACT,
                                    "prompt_contract": PROMPT_CONTRACT_VERSION,
                                })
                                rows.append(row)
                # A selected model/task can have no eligible stage for an item in
                # either cross-frame arm.  Keep a header-only completion shard so
                # restart safety does not confuse that designed absence with a
                # crashed job.
                frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
                temporary = out_path.with_suffix(".csv.tmp")
                frame.to_csv(temporary, index=False); temporary.replace(out_path)
                scorer.check_budget(f"norm-manip-task{task}-{label}")
                print(f"SAVED task={task} item={label} rows={len(frame)}", flush=True)
            completed_tasks.append(int(task))
        memory = scorer.memory_manifest()
    finally:
        scorer.close()

    manifests = args.output_dir / "manifests"; manifests.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "rq3-0830-edit-norming-manipulation-check",
        "model": args.model, "tasks": completed_tasks,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "limit": args.limit, "items_total": len(labels),
        "target_rows": int(len(grid)), "include_posthoc": not args.exclude_posthoc,
        "theta_primary": 0.25, "theta_sensitivity": 0.5,
        "selection": "same cross-frame item-level ADD/ABL state as 0830",
        "directions": "same fold_directions/seed/layer/scale as 0830; target+matched random+shuffle",
        "norming": "RQ3-v2 raw-sequence prompts and rq3_run_server baseline used to train directions",
        "recommendation_correction": (
            "0901 document named legacy sim/run_norm15/additional_Qs/norming, but executed 0830 "
            "directions use rq3_run_server/norming under the RQ3-v2 contract"
        ),
        "experiment_contract": CONTRACT, "prompt_contract": PROMPT_CONTRACT_VERSION,
        "memory": memory,
    }
    name = f"norm_manip_{slug}_shard{args.shard_index}of{args.num_shards}.json"
    (manifests / name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
