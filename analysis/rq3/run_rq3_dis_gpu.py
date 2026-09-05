#!/usr/bin/env python3
"""RQ3 H3f manipulation checks, dissociation, and rho-to-DoD calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from analyze_rq3_representation import norm_arrays
    from run_rq3_behavior_gpu import BudgetedScorer, FRAME, QVAR, load_stimuli, safe_name
    from run_rq3_intervention_gpu import ALPHAS, DirectionHooks, context_direction, intro_start_token, regression_directions
    from run_rq3_representation_gpu import hypothesis_text
    from rq3_prompting import (
        PROMPT_CONTRACT_VERSION, answer_continuations, behavior_prompt,
        norming_prompt,
    )
    from rq3_decoder_contract import H3_DECODER_CONTRACT, load_h3_decoder_gate
except ModuleNotFoundError:
    from sim.analyze_rq3_representation import norm_arrays
    from sim.run_rq3_behavior_gpu import BudgetedScorer, FRAME, QVAR, load_stimuli, safe_name
    from sim.run_rq3_intervention_gpu import ALPHAS, DirectionHooks, context_direction, intro_start_token, regression_directions
    from sim.run_rq3_representation_gpu import hypothesis_text
    from sim.rq3_prompting import (
        PROMPT_CONTRACT_VERSION, answer_continuations, behavior_prompt,
        norming_prompt,
    )
    from sim.rq3_decoder_contract import H3_DECODER_CONTRACT, load_h3_decoder_gate


def activation_scale(array: np.ndarray, train: np.ndarray, layer: int) -> float:
    states = np.asarray(array[train, :, :, layer], dtype=np.float32)
    return float(np.linalg.norm(states.std(axis=0, ddof=1).mean(axis=(0, 1))))


def injection_sign(mode: str, hypothesis: str) -> float:
    if mode == "common":
        return 1.0
    return 1.0 if hypothesis == "h3" else -1.0


def behavior_rows(
    model_id: str,
    scorer: BudgetedScorer,
    stimulus,
    task: int,
    directions: dict[str, np.ndarray],
    selected_layer: int,
    scale: float,
) -> pd.DataFrame:
    rows = []
    for direction_name in ("P", "C", "PC"):
        for mode in ("differential", "common"):
            for alpha in ALPHAS:
                for frame_index, frame in enumerate(FRAME):
                    prompt = behavior_prompt(stimulus.intro, task, frame_index, 0)
                    start = intro_start_token(scorer.tok, frame, prompt)
                    for qvar_index, question in enumerate(QVAR[task]):
                        prompt = behavior_prompt(
                            stimulus.intro, task, frame_index, qvar_index
                        )
                        values = {
                            "model_id": model_id, "label": stimulus.label,
                            "scenario": stimulus.scenario_id, "task": task,
                            "frame": frame_index, "qvar": qvar_index,
                            "prefix": "I", "direction": direction_name,
                            "mode": mode, "alpha": alpha, "activation_scale": scale,
                            "selected_layer": selected_layer,
                        }
                        for hypothesis, option in (("h2", "opt_c2"), ("h3", "opt_c3")):
                            magnitude = injection_sign(mode, hypothesis) * alpha * scale
                            with DirectionHooks(
                                scorer.model, directions[direction_name], "add",
                                selected_layer=selected_layer, magnitude=magnitude,
                            ) as controller:
                                controller.start_position = start
                                lp, nt = scorer.score_one(prompt, " " + getattr(stimulus, option))
                            values[f"lp_conj_{hypothesis}"] = lp
                            values[f"nt_conj_{hypothesis}"] = nt
                        # The differential manipulation has no unique sign for
                        # h1.  Score both signs so the preregistered base-collapse
                        # guard is conservative; common injection needs one sign.
                        base_signs = (1.0, -1.0) if mode == "differential" else (1.0,)
                        for base_sign in base_signs:
                            with DirectionHooks(
                                scorer.model, directions[direction_name], "add",
                                selected_layer=selected_layer,
                                magnitude=base_sign * alpha * scale,
                            ) as controller:
                                controller.start_position = start
                                lp, nt = scorer.score_one(prompt, " " + stimulus.opt_base)
                            suffix = "plus" if base_sign > 0 else "minus"
                            values[f"lp_base_{suffix}"] = lp
                            values[f"nt_base_{suffix}"] = nt
                        rows.append(values)
    return pd.DataFrame(rows)


def norming_rows(
    model_id: str,
    scorer: BudgetedScorer,
    stimulus,
    task: int,
    directions: dict[str, np.ndarray],
    selected_layer: int,
    scales: dict[str, float],
) -> pd.DataFrame:
    rows = []
    conditions = [
        ("P", "differential"), ("P", "common"),
        ("C", "differential"), ("C", "common"),
        ("PC", "differential"), ("PC", "common"),
        *((name, "common") for name in directions if name.startswith("R_")),
    ]
    for direction_name, mode in conditions:
        scale = scales[direction_name]
        for alpha in ALPHAS:
            for hypothesis in ("h2", "h3"):
                hyp = hypothesis_text(stimulus, hypothesis)
                magnitude = injection_sign(mode, hypothesis) * alpha * scale
                with DirectionHooks(
                    scorer.model, directions[direction_name], "add",
                    selected_layer=selected_layer, magnitude=magnitude,
                ) as controller:
                    for frame_index in range(len(FRAME)):
                        for variant in range(3):
                            rho_prompt = norming_prompt(
                                stimulus.intro, hyp, "P", frame_index, variant
                            )
                            c_prompt = norming_prompt(
                                stimulus.intro, hyp, "C", frame_index, variant
                            )
                            controller.start_position = intro_start_token(
                                scorer.tok, FRAME[frame_index], rho_prompt
                            )
                            rho_score = [
                                scorer.score_one(rho_prompt, answer)
                                for answer in answer_continuations("P")
                            ]
                            controller.start_position = intro_start_token(
                                scorer.tok, FRAME[frame_index], c_prompt
                            )
                            c_score = [
                                scorer.score_one(c_prompt, answer)
                                for answer in answer_continuations("C")
                            ]
                            if rho_score[0][1] != rho_score[1][1] or c_score[0][1] != c_score[1][1]:
                                raise RuntimeError("Norming answer candidates must be token-length matched")
                            rows.append(
                                {
                                    "model_id": model_id, "label": stimulus.label,
                                    "scenario": stimulus.scenario_id, "task": task,
                                    "prefix": "I", "direction": direction_name, "mode": mode,
                                    "stage": "ADD", "alpha": alpha, "hypothesis": hypothesis,
                                    "frame": frame_index, "variant": variant,
                                    "rho": rho_score[0][0] - rho_score[1][0],
                                    "C": c_score[0][0] - c_score[1][0],
                                    "rho_yes_nt": rho_score[0][1], "rho_no_nt": rho_score[1][1],
                                    "C_strengthen_nt": c_score[0][1], "C_weaken_nt": c_score[1][1],
                                    "activation_scale": scale, "selected_layer": selected_layer,
                                    "prompt_contract": PROMPT_CONTRACT_VERSION,
                                    "decoder_contract": H3_DECODER_CONTRACT,
                                }
                            )
    return pd.DataFrame(rows)


def norming_ablation_rows(
    model_id: str,
    scorer: BudgetedScorer,
    stimulus,
    task: int,
    directions: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for direction_name in directions:
        with DirectionHooks(scorer.model, directions[direction_name], "ablate"):
            for hypothesis in ("h2", "h3"):
                hyp = hypothesis_text(stimulus, hypothesis)
                for frame_index in range(len(FRAME)):
                    for variant in range(3):
                        rho_prompt = norming_prompt(
                            stimulus.intro, hyp, "P", frame_index, variant
                        )
                        c_prompt = norming_prompt(
                            stimulus.intro, hyp, "C", frame_index, variant
                        )
                        rho_score = [
                            scorer.score_one(rho_prompt, answer)
                            for answer in answer_continuations("P")
                        ]
                        c_score = [
                            scorer.score_one(c_prompt, answer)
                            for answer in answer_continuations("C")
                        ]
                        if rho_score[0][1] != rho_score[1][1] or c_score[0][1] != c_score[1][1]:
                            raise RuntimeError("Norming answer candidates must be token-length matched")
                        rows.append({"model_id": model_id, "label": stimulus.label, "scenario": stimulus.scenario_id, "task": task, "prefix": "I", "direction": direction_name, "mode": "ablate", "stage": "ABL", "alpha": np.nan, "hypothesis": hypothesis, "frame": frame_index, "variant": variant, "rho": rho_score[0][0] - rho_score[1][0], "C": c_score[0][0] - c_score[1][0], "rho_yes_nt": rho_score[0][1], "rho_no_nt": rho_score[1][1], "C_strengthen_nt": c_score[0][1], "C_weaken_nt": c_score[1][1], "activation_scale": np.nan, "selected_layer": -1, "prompt_contract": PROMPT_CONTRACT_VERSION, "decoder_contract": H3_DECODER_CONTRACT})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", type=int, choices=(2, 3))
    parser.add_argument(
        "--items", nargs="+",
        help=("Optional disjoint item-label shard.  The held-out index and its "
              "14-item training fold remain defined against the full stimulus set."),
    )
    parser.add_argument("--max-gpu-mib", type=int, default=24_000)
    parser.add_argument("--manifest-prefix", default="dis")
    args = parser.parse_args()
    if args.max_gpu_mib > 24_000:
        raise ValueError("GPU ceiling cannot exceed 24,000 MiB")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(int(os.environ.get("RQ3_TORCH_THREADS", "4")))
    stimuli = load_stimuli(None); labels = list(dict.fromkeys(stimuli.label.tolist()))
    if args.items:
        unknown = sorted(set(args.items) - set(labels))
        if unknown:
            raise ValueError(f"Unknown item labels: {unknown}")
        selected_items = set(args.items)
    else:
        selected_items = None
    slug = safe_name(args.model)
    norming = pd.read_csv(args.output_dir / "norming" / f"norm_rq3_{slug}.csv")
    rho, confirmation = norm_arrays(norming, labels)
    gate_path = args.output_dir / "analysis" / "rq3a_context_decoder_robustness.csv"
    gates = load_h3_decoder_gate(gate_path)
    gates = gates[(gates.model_id == args.model) & gates.decode_gate]
    if args.tasks:
        gates = gates[gates.task.isin(args.tasks)]
    if gates.empty:
        print(f"SKIP no decode-gate task for {args.model}"); return 0
    scorer = BudgetedScorer(args.model, args.max_gpu_mib, reserve_mib=1_000)
    completed = []
    try:
        for gate in gates.itertuples(index=False):
            task = int(gate.task); selected = np.asarray(json.loads(gate.selected_layers), dtype=int)
            array = np.load(args.output_dir / "activations" / slug / f"task{task}_cellmean.npy", mmap_mode="r")
            prompt_array = np.load(args.output_dir / "activations" / slug / f"task{task}_prompt_cellmean.npy", mmap_mode="r")
            shard_dir = args.output_dir / "dis" / slug / f"task{task}"; shard_dir.mkdir(parents=True, exist_ok=True)
            for held, label in enumerate(labels):
                if selected_items is not None and label not in selected_items:
                    continue
                behavior_path = shard_dir / f"{label}_behavior.csv"; norm_path = shard_dir / f"{label}_norming.csv"
                if behavior_path.exists() and norm_path.exists():
                    existing = [pd.read_csv(path, nrows=1) for path in (behavior_path, norm_path)]
                    if all(
                        "prompt_contract" in frame
                        and set(frame.prompt_contract) == {PROMPT_CONTRACT_VERSION}
                        and "decoder_contract" in frame
                        and set(frame.decoder_contract) == {H3_DECODER_CONTRACT}
                        for frame in existing
                    ):
                        print(f"SKIP DIS task={task} item={label}"); continue
                    raise RuntimeError(
                        f"Legacy DIS shards would be mixed with {PROMPT_CONTRACT_VERSION}: "
                        f"{behavior_path.parent}. Use a fresh output directory."
                    )
                train = np.asarray([i for i in range(15) if i != held]); layer = int(selected[held])
                regression = regression_directions(array, rho, confirmation, train)
                directions = {
                    "R_h3": context_direction(array, train, (2,)),
                    "R_h2": context_direction(array, train, (1,)),
                    "R_pooled": context_direction(array, train, (1, 2)),
                    "R_base": context_direction(array, train, (0,)),
                    "P": regression["P"], "C": regression["C"], "PC": regression["PC"],
                }
                scale = activation_scale(array, train, layer)
                scales = {name: scale for name in directions}
                stimulus = stimuli[(stimuli.label == label) & (stimuli.prefix == "I")].iloc[0]
                behavior = behavior_rows(args.model, scorer, stimulus, task, directions, layer, scale)
                behavior["prompt_contract"] = PROMPT_CONTRACT_VERSION
                behavior["decoder_contract"] = H3_DECODER_CONTRACT
                manipulated_norming = pd.concat([norming_rows(args.model, scorer, stimulus, task, directions, layer, scales), norming_ablation_rows(args.model, scorer, stimulus, task, directions)], ignore_index=True)
                for frame, path in ((behavior, behavior_path), (manipulated_norming, norm_path)):
                    temporary = path.with_suffix(path.suffix + ".tmp"); frame.to_csv(temporary, index=False); temporary.replace(path)
                scorer.check_budget(f"DIS-task{task}-{label}")
                print(f"SAVED DIS task={task} item={label}", flush=True)
            completed.append(task)
        memory = scorer.memory_manifest()
    finally:
        scorer.close()
    manifest = {"stage": "rq3-dis", "model": args.model, "tasks": completed, "items": args.items, "alphas": list(ALPHAS), "prompt_contract": PROMPT_CONTRACT_VERSION, "decoder_contract": H3_DECODER_CONTRACT, "decoder_gate_file": str(gate_path), "memory": memory}
    (args.output_dir / "manifests" / f"{args.manifest_prefix}_{slug}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
