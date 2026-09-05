#!/usr/bin/env python3
"""Fixed WikiText fluency guard for conditional CF rescue/knockout.

Target directions are guarded on every 165-item LOIO fold.  Random and shuffle
controls are guarded on a deterministic 15-fold diagnostic subset; causal pass
criteria depend only on the full target-direction guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cf_rescue_common import (
    ABL_DOSES, ADD_ALPHAS, CONTRACT, DIRECTION_NAMES, FRONT, MODELS, PROJ, SEED,
    fold_directions, load_stimuli, safe_name, selected_layers,
)

sys.path.insert(0, str(PROJ / "analysis" / "rq3"))
from analyze_rq3_representation import norm_arrays  # noqa: E402
from run_rq3_behavior_gpu import BudgetedScorer  # noqa: E402
from rq3_decoder_contract import H3_DECODER_CONTRACT  # noqa: E402
from rq3_prompting import PROMPT_CONTRACT_VERSION  # noqa: E402
from run_rq3_cf_rescue_gpu import SelectedDirectionHook  # noqa: E402


WIKITEXT_SHA256 = "413428c311d70417df523fc14d830311355886b5bf6b78324fba13b608e35e49"
TARGETS = ("R_given_PC", "PC_given_R")
# Exactly 15 labels spanning the locked ordering, fixed before outcomes.
CONTROL_HELDS = set(np.linspace(0, 164, 15, dtype=int).tolist())


def perplexity_metrics(nll: float, count: int, baseline_log_ppl: float) -> tuple[float, float, bool]:
    log_ppl = nll / count
    delta = log_ppl - baseline_log_ppl
    ppl = math.exp(log_ppl) if log_ppl < 700 else float("inf")
    increase = math.expm1(delta) if delta < 700 else float("inf")
    return ppl, increase, bool(np.isfinite(increase) and increase < .10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--front", type=Path, default=FRONT)
    ap.add_argument("--tasks", nargs="+", type=int, choices=(2, 3), default=(2, 3))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-gpu-mib", type=int, default=24_000)
    args = ap.parse_args()
    if args.max_gpu_mib > 24_000 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid GPU ceiling or shard")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(int(os.environ.get("RQ3_TORCH_THREADS", "1")))

    stimuli = load_stimuli()
    labels = stimuli.label.drop_duplicates().tolist()
    work = [i for i in range(165) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        work = work[:args.limit]
    slug = safe_name(args.model)
    norming = pd.read_csv(args.front / "norming" / f"norm_rq3_{slug}.csv")
    rho, confirmation = norm_arrays(norming, labels)
    gate_file = args.front / "analysis" / "rq3a_context_decoder_robustness.csv"

    scorer = BudgetedScorer(args.model, args.max_gpu_mib, reserve_mib=1_000)
    text_path = PROJ / "data" / "wikitext_rq3_guard.txt"
    observed_hash = hashlib.sha256(text_path.read_bytes()).hexdigest()
    if observed_hash != WIKITEXT_SHA256:
        raise RuntimeError(f"WikiText hash changed: {observed_hash}")
    tokens = scorer.tok(
        text_path.read_text(encoding="utf-8"), add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0][:1001]
    if len(tokens) < 1001:
        raise RuntimeError("WikiText guard has fewer than 1,001 model tokens")
    baseline_nll, baseline_count = scorer.sequence_nll(tokens)
    baseline_log_ppl = baseline_nll / baseline_count
    baseline_ppl = math.exp(baseline_log_ppl)
    completed: list[int] = []
    try:
        for task in args.tasks:
            layers = selected_layers(gate_file, args.model, task)
            array = np.load(
                args.front / "activations" / slug / f"task{task}_cellmean.npy",
                mmap_mode="r",
            )
            task_dir = args.output_dir / "guard_shards" / slug / f"task{task}"
            task_dir.mkdir(parents=True, exist_ok=True)
            for held in work:
                label = labels[held]
                out_path = task_dir / f"{label}.csv"
                if out_path.exists():
                    old = pd.read_csv(out_path, nrows=1)
                    if "experiment_contract" in old and set(old.experiment_contract) == {CONTRACT}:
                        print(f"SKIP guard task={task} item={label}", flush=True)
                        continue
                    raise RuntimeError(f"Refusing to mix legacy guard shard: {out_path}")
                train = np.flatnonzero(np.arange(165) != held)
                layer = int(layers[held])
                directions, diagnostics = fold_directions(
                    array, rho, confirmation, train, layer,
                    SEED + task * 10_000 + held,
                )
                names = DIRECTION_NAMES if held in CONTROL_HELDS else TARGETS
                rows = []
                for direction_name in names:
                    direction = directions[direction_name]
                    for alpha in ADD_ALPHAS:
                        with SelectedDirectionHook(
                            scorer.model, direction, layer, "ADD",
                            magnitude=alpha * diagnostics["scale"],
                        ) as controller:
                            controller.start_positions = [0]
                            nll, count = scorer.sequence_nll(tokens)
                        ppl, increase, passes = perplexity_metrics(
                            nll, count, baseline_log_ppl
                        )
                        rows.append({
                            "model_id": args.model, "task": task, "held": held,
                            "label": label, "layer": layer, "mode": "ADD",
                            "direction": direction_name, "alpha": alpha,
                            "dose": np.nan, "ppl": ppl, "increase": increase,
                            "passes": passes, "baseline_ppl": baseline_ppl,
                            "experiment_contract": CONTRACT,
                        })
                    for dose in ABL_DOSES:
                        with SelectedDirectionHook(
                            scorer.model, direction, layer, "ABL", dose=dose,
                        ):
                            nll, count = scorer.sequence_nll(tokens)
                        ppl, increase, passes = perplexity_metrics(
                            nll, count, baseline_log_ppl
                        )
                        rows.append({
                            "model_id": args.model, "task": task, "held": held,
                            "label": label, "layer": layer, "mode": "ABL",
                            "direction": direction_name, "alpha": np.nan,
                            "dose": dose, "ppl": ppl, "increase": increase,
                            "passes": passes, "baseline_ppl": baseline_ppl,
                            "experiment_contract": CONTRACT,
                        })
                frame = pd.DataFrame(rows)
                temporary = out_path.with_suffix(".csv.tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(out_path)
                scorer.check_budget(f"cf-guard-task{task}-{label}")
                print(f"SAVED guard task={task} item={label} rows={len(frame)}", flush=True)
            completed.append(task)
        memory = scorer.memory_manifest()
    finally:
        scorer.close()

    manifests = args.output_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "rq3-cf-rescue-fluency", "model": args.model,
        "tasks": completed, "shard_index": args.shard_index,
        "num_shards": args.num_shards, "limit": args.limit,
        "tokens": baseline_count, "baseline_ppl": baseline_ppl,
        "wikitext_sha256": WIKITEXT_SHA256,
        "target_folds_per_task": 165,
        "control_diagnostic_folds": sorted(CONTROL_HELDS),
        "threshold": "PPL increase < 10%; all target folds",
        "experiment_contract": CONTRACT,
        "prompt_contract": PROMPT_CONTRACT_VERSION,
        "decoder_contract": H3_DECODER_CONTRACT, "memory": memory,
    }
    name = f"cf_guard_{slug}_shard{args.shard_index}of{args.num_shards}.json"
    (manifests / name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
