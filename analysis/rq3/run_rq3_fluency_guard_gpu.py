#!/usr/bin/env python3
"""RQ3 section 6.5: fixed 1,000-token WikiText fluency guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from analyze_rq3_representation import item_did, norm_arrays
    from run_rq3_behavior_gpu import BudgetedScorer, load_stimuli, safe_name
    from run_rq3_intervention_gpu import (ALPHAS, LAMBDAS, SEED, DirectionHooks, context_direction,
        denoised_direction, orthogonalized_direction, random_directions, random_subspace,
        regression_directions, shuffled_context_direction, shuffled_direction,
        subspace_add_vectors, subspace_directions)
    from rq3_prompting import PROMPT_CONTRACT_VERSION
    from rq3_decoder_contract import H3_DECODER_CONTRACT, H3_GATE_FAMILY, load_h3_decoder_gate
except ModuleNotFoundError:
    from sim.analyze_rq3_representation import item_did, norm_arrays
    from sim.run_rq3_behavior_gpu import BudgetedScorer, load_stimuli, safe_name
    from sim.run_rq3_intervention_gpu import (ALPHAS, LAMBDAS, SEED, DirectionHooks, context_direction,
        denoised_direction, orthogonalized_direction, random_directions, random_subspace,
        regression_directions, shuffled_context_direction, shuffled_direction,
        subspace_add_vectors, subspace_directions)
    from sim.rq3_prompting import PROMPT_CONTRACT_VERSION
    from sim.rq3_decoder_contract import H3_DECODER_CONTRACT, H3_GATE_FAMILY, load_h3_decoder_gate

PROJ = Path(__file__).resolve().parents[2]
WIKITEXT_SHA256 = "413428c311d70417df523fc14d830311355886b5bf6b78324fba13b608e35e49"


def perplexity_metrics(nll: float, count: int, baseline_log_ppl: float) -> tuple[float, float, bool]:
    log_ppl = nll / count; delta = log_ppl - baseline_log_ppl
    ppl = math.exp(log_ppl) if log_ppl < 700 else float("inf")
    increase = math.expm1(delta) if delta < 700 else float("inf")
    return ppl, increase, bool(np.isfinite(increase) and increase < .10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gpu-mib", type=int, default=24_000)
    parser.add_argument(
        "--include-decode-fail",
        action="store_true",
        help="Exploratory only: run tasks that failed the preregistered decoding gate.",
    )
    parser.add_argument("--guards-subdir", default="guards")
    parser.add_argument("--manifest-prefix", default="fluency")
    parser.add_argument("--decoder-gate-file", type=Path)
    parser.add_argument("--gate-family", default=H3_GATE_FAMILY)
    parser.add_argument(
        "--direction-mode",
        choices=("registered_did", "h3_context", "h3_subspace"),
        default="h3_context",
    )
    parser.add_argument("--ablate-scope", default="selected", choices=("all", "selected", "both"))
    parser.add_argument("--subspace-k", type=int, default=2)
    parser.add_argument("--random-subspaces", type=int, default=8)
    parser.add_argument("--allow-retired-one-vs-three", action="store_true")
    args = parser.parse_args()
    if args.max_gpu_mib > 24_000: raise ValueError("GPU ceiling cannot exceed 24,000 MiB")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(int(os.environ.get("RQ3_TORCH_THREADS", "4")))
    stimuli = load_stimuli(None); labels = list(dict.fromkeys(stimuli.label.tolist())); slug = safe_name(args.model)
    norming = pd.read_csv(args.output_dir / "norming" / f"norm_rq3_{slug}.csv"); rho, confirmation = norm_arrays(norming, labels)
    if args.direction_mode == "registered_did" and not args.allow_retired_one_vs_three:
        raise ValueError("registered_did uses the retired one-positive/three-negative decoder")
    gate_path = args.decoder_gate_file or args.output_dir / "analysis" / "rq3a_context_decoder_robustness.csv"
    if args.direction_mode in ("h3_context", "h3_subspace"):
        gates = load_h3_decoder_gate(gate_path)
    else:
        gates = pd.read_csv(gate_path)
        if args.gate_family is not None:
            if "family" not in gates.columns: raise ValueError(f"Gate file has no family column: {gate_path}")
            gates = gates[gates.family == args.gate_family]
    gates = gates[gates.model_id == args.model]
    gates = gates[~gates.decode_gate] if args.include_decode_fail else gates[gates.decode_gate]
    if gates.empty:
        gate_kind = "decode-fail" if args.include_decode_fail else "decode-pass"
        print(f"SKIP no {gate_kind} task for {args.model}")
        return 0
    output = args.output_dir / args.guards_subdir / slug
    manifest_path = args.output_dir / "manifests" / f"{args.manifest_prefix}_{slug}.json"
    folds_path = output / "fluency_folds.csv"
    aggregate_path = output / "fluency.csv"
    safe_path = output / "alpha_safe.csv"
    if args.direction_mode == "h3_subspace":
        directions_per_fold = 12 + args.random_subspaces
    else:
        directions_per_fold = 9 if args.direction_mode == "h3_context" else 8
    # H3f uses P/C/PC ADD arms, so those directions need the same preregistered
    # WikiText guard as H3e and its controls.  Omitting them makes every H3f
    # safe-alpha cell ineligible at analysis time.
    if args.direction_mode == "h3_subspace":
        add_directions_per_fold = directions_per_fold
    else:
        add_directions_per_fold = 9 if args.direction_mode == "h3_context" else 8
    n_scopes = 2 if args.ablate_scope == "both" else 1
    expected_fold_rows = (1 + 15 * (directions_per_fold * n_scopes * len(LAMBDAS) + add_directions_per_fold * len(ALPHAS))) * len(gates)
    existing_contract = None
    existing_decoder_contract = None
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
            existing_contract = existing_manifest.get("prompt_contract")
            existing_decoder_contract = existing_manifest.get("decoder_contract")
        except (OSError, json.JSONDecodeError):
            existing_contract = None
    if (
        existing_contract == PROMPT_CONTRACT_VERSION
        and existing_decoder_contract == H3_DECODER_CONTRACT
        and folds_path.exists()
        and aggregate_path.exists()
        and safe_path.exists()
        and sum(1 for _ in folds_path.open(encoding="utf-8")) == expected_fold_rows + 1
    ):
        print(f"SKIP complete fluency model={args.model}")
        return 0
    scorer = BudgetedScorer(args.model, args.max_gpu_mib, reserve_mib=1_000)
    wikitext_path = PROJ / "data" / "wikitext_rq3_guard.txt"
    observed_wikitext_sha256 = hashlib.sha256(wikitext_path.read_bytes()).hexdigest()
    if observed_wikitext_sha256 != WIKITEXT_SHA256:
        raise RuntimeError(
            f"WikiText guard hash changed: {observed_wikitext_sha256} != {WIKITEXT_SHA256}"
        )
    text = wikitext_path.read_text(encoding="utf-8")
    tokens = scorer.tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0][:1001]
    if len(tokens) < 1001: raise RuntimeError("WikiText guard has fewer than 1,001 model tokens")
    base_nll, base_count = scorer.sequence_nll(tokens); baseline_log_ppl = base_nll / base_count; baseline_ppl = math.exp(baseline_log_ppl)
    rows = []
    try:
        for gate in gates.itertuples(index=False):
            task = int(gate.task); selected_layers = np.asarray(json.loads(gate.selected_layers), dtype=int)
            array = np.load(args.output_dir / "activations" / slug / f"task{task}_cellmean.npy", mmap_mode="r")
            prompt_array = np.load(args.output_dir / "activations" / slug / f"task{task}_prompt_cellmean.npy", mmap_mode="r")
            did = item_did(array)
            rows.append({"model_id": args.model, "task": task, "held": -1, "label": "baseline", "layer": -1, "mode": "baseline", "direction": "none", "alpha": 0.0, "dose": np.nan, "ablate_scope": "", "ppl": baseline_ppl, "increase": 0.0, "passes": True})
            for held, label in enumerate(labels):
                train = np.asarray([index for index in range(15) if index != held]); layer = int(selected_layers[held])
                regression = regression_directions(array, rho, confirmation, train)
                add_vectors: dict[str, np.ndarray] = {}
                if args.direction_mode == "h3_context":
                    v_h3 = context_direction(array, train, (2,))
                    directions = {
                        "R_h3": v_h3,
                        "R_h2": context_direction(array, train, (1,)),
                        "R_pooled": context_direction(array, train, (1, 2)),
                        "R_base": context_direction(array, train, (0,)),
                        "P": regression["P"], "C": regression["C"], "PC": regression["PC"],
                        "random": random_directions(v_h3.shape, SEED + held + task * 100),
                        "shuffle": shuffled_context_direction(array, train, (2,), SEED + held + task * 1000),
                    }
                    add_names = (
                        "R_h3", "R_h2", "R_pooled", "R_base",
                        "P", "C", "PC", "random", "shuffle",
                    )
                elif args.direction_mode == "h3_subspace":
                    # must equal the h3_subspace direction set of the intervention script.
                    # otherwise some arm has no safe alpha and every cell is ineligible at analysis time.
                    v_h3 = context_direction(array, train, (2,))
                    k = args.subspace_k
                    held_contrast = np.asarray(
                        array[held, 2, 2] - array[held, 1, 2], dtype=np.float64
                    )
                    basis = subspace_directions(array, train, (2,), k)
                    directions = {
                        "R_h3": v_h3,
                        f"R_h3_k{k}": basis,
                        "R_h3_denoised": denoised_direction(array, train, (2,), k),
                        "R_h3_orth_PC": orthogonalized_direction(v_h3, (regression["P"], regression["C"])),
                        "R_h2": context_direction(array, train, (1,)),
                        "R_pooled": context_direction(array, train, (1, 2)),
                        "R_base": context_direction(array, train, (0,)),
                        "P": regression["P"], "C": regression["C"], "PC": regression["PC"],
                        "random": random_directions(v_h3.shape, SEED + held + task * 100),
                        "shuffle": shuffled_context_direction(array, train, (2,), SEED + held + task * 1000),
                    }
                    add_vectors[f"R_h3_k{k}"] = subspace_add_vectors(basis, held_contrast)
                    for index in range(args.random_subspaces):
                        name = f"random_k{k}_{index:02d}"
                        control = random_subspace(
                            (v_h3.shape[0], v_h3.shape[-1]), k,
                            SEED + held + task * 100 + index * 7919)
                        directions[name] = control
                        add_vectors[name] = subspace_add_vectors(control, held_contrast)
                    # P, C and PC also enter the ADD guard; H3f pushes those arms.
                    add_names = tuple(directions)
                else:
                    directions = {
                        "R": did[train].mean(axis=0), "R_unique": regression["R_unique"], "R_prompt": (prompt_array[train, 2] - prompt_array[train, 1]).mean(axis=0), "P": regression["P"], "C": regression["C"], "PC": regression["PC"],
                        "random": random_directions(did.shape[1:], SEED + held + task * 100),
                        "shuffle": shuffled_direction(array, train, SEED + held + task * 1000),
                    }
                    add_names = tuple(directions)
                states = np.asarray(array[train, :, :, layer], dtype=np.float32)
                scale = float(np.linalg.norm(states.std(axis=0, ddof=1).mean(axis=(0, 1))))
                prompt_states = np.asarray(prompt_array[train, :, layer], dtype=np.float32)
                prompt_scale = float(np.linalg.norm(prompt_states.std(axis=0, ddof=1).mean(axis=0)))
                scopes = ("all", "selected") if args.ablate_scope == "both" else (args.ablate_scope,)
                for name, direction in directions.items():
                    for scope in scopes:
                        for dose in LAMBDAS:
                            with DirectionHooks(scorer.model, direction, "ablate", dose=dose,
                                                ablate_scope=scope, selected_layer=layer):
                                nll, count = scorer.sequence_nll(tokens)
                            ppl, increase, passes = perplexity_metrics(nll, count, baseline_log_ppl)
                            rows.append({"model_id": args.model, "task": task, "held": held, "label": label, "layer": layer, "mode": "ABL", "direction": name, "alpha": np.nan, "dose": dose, "ablate_scope": scope, "ppl": ppl, "increase": increase, "passes": passes})
                for name in add_names:
                    for alpha in ALPHAS:
                        direction_scale = prompt_scale if name == "R_prompt" else scale
                        with DirectionHooks(
                            scorer.model,
                            directions[name],
                            "add",
                            selected_layer=layer,
                            magnitude=alpha * direction_scale,
                            add_vectors=add_vectors.get(name),
                        ) as controller:
                            controller.start_position = 0; nll, count = scorer.sequence_nll(tokens)
                        ppl, increase, passes = perplexity_metrics(nll, count, baseline_log_ppl)
                        rows.append({"model_id": args.model, "task": task, "held": held, "label": label, "layer": layer, "mode": "ADD", "direction": name, "alpha": alpha, "dose": np.nan, "ablate_scope": "", "ppl": ppl, "increase": increase, "passes": passes})
                scorer.check_budget(f"fluency-task{task}-held{held}")
        memory = scorer.memory_manifest()
    finally:
        scorer.close()
    output.mkdir(parents=True, exist_ok=True)
    detailed = pd.DataFrame(rows); detailed.to_csv(output / "fluency_folds.csv", index=False)
    baseline_rows = detailed[detailed["mode"] == "baseline"]
    abl = detailed[detailed["mode"] == "ABL"].groupby(["model_id", "task", "mode", "direction", "dose", "ablate_scope"], as_index=False).agg(alpha=("alpha", "first"), ppl=("ppl", "max"), increase=("increase", "max"), passes=("passes", "all"), n_folds=("held", "count"))
    add = detailed[detailed["mode"] == "ADD"].groupby(["model_id", "task", "mode", "direction", "alpha"], as_index=False).agg(ppl=("ppl", "max"), increase=("increase", "max"), passes=("passes", "all"), n_folds=("held", "count"))
    aggregate = pd.concat([baseline_rows, abl, add], ignore_index=True, sort=False); aggregate.to_csv(output / "fluency.csv", index=False)
    safe = add[add.passes].groupby(["task", "direction"]).alpha.max().reset_index(name="alpha_safe")
    safe.to_csv(output / "alpha_safe.csv", index=False)
    manifest_path.write_text(json.dumps({"stage": "rq3-exploratory-decode-fail-fluency" if args.include_decode_fail else "rq3-fluency", "model": args.model, "tokens": base_count, "baseline_ppl": baseline_ppl, "wikitext_sha256": WIKITEXT_SHA256, "folds_per_task": 15, "aggregation": "worst increase; all folds must pass", "decoder_gate_file": str(gate_path), "gate_family": args.gate_family, "direction_mode": args.direction_mode, "decoder_contract": H3_DECODER_CONTRACT if args.direction_mode in ("h3_context", "h3_subspace") else "retired-one-vs-three", "subspace_k": args.subspace_k if args.direction_mode == "h3_subspace" else None, "random_subspaces": args.random_subspaces if args.direction_mode == "h3_subspace" else 0, "ablate_scope": args.ablate_scope, "abl_doses": list(LAMBDAS), "prompt_contract": PROMPT_CONTRACT_VERSION, "memory": memory}, ensure_ascii=False, indent=2) + "\n")
    return 0


SEED = 20260824
if __name__ == "__main__": raise SystemExit(main())
