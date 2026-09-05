#!/usr/bin/env python3
"""RQ3 steps 3--4: I-inclusive norming and residual-stream extraction.

Runs only after H3-0 eligibility has been established.  The activation tensor
uses the option-final token, all hidden states (embedding output + block
outputs), and online averages the 12 frame/question variants within each cell.
The same extraction forward passes also recompute continuation log-probability;
these are checked against the frozen behavioral output.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from run_rq3_behavior_gpu import (
        BudgetedScorer, FRAME, QVAR, build_rows, load_stimuli, safe_name,
    )
    from rq3_prompting import (
        PROMPT_CONTRACT_VERSION, answer_continuations, behavior_prompt,
        norming_prompt,
    )
except ModuleNotFoundError:  # import as sim.run_rq3_representation_gpu in tests
    from sim.run_rq3_behavior_gpu import (
        BudgetedScorer, FRAME, QVAR, build_rows, load_stimuli, safe_name,
    )
    from sim.rq3_prompting import (
        PROMPT_CONTRACT_VERSION, answer_continuations, behavior_prompt,
        norming_prompt,
    )
OPTIONS = ("base", "conj_h2", "conj_h3", "bare_h2", "bare_h3")
PREFIXES = ("N", "I", "R")


def atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp.replace(path)


def hypothesis_text(row, hypothesis: str) -> str:
    return f"{row.tgt} {getattr(row, hypothesis)}".replace("V. V.", "V.").rstrip(".")


def score_prompt_pairs(
    scorer: BudgetedScorer,
    prompts: list[str],
    metric: str,
    batch_size: int,
) -> list[tuple[float, float, int, int]]:
    """Score the complete positive/negative answer sequences for each prompt."""
    positive, negative = answer_continuations(metric)
    paired_prompts = [prompt for prompt in prompts for _ in range(2)]
    continuations = [answer for _ in prompts for answer in (positive, negative)]
    scored = scorer.score_pairs(paired_prompts, continuations, batch_size=batch_size)
    paired = [
        (scored[index][0], scored[index + 1][0], scored[index][1], scored[index + 1][1])
        for index in range(0, len(scored), 2)
    ]
    unequal = [(index, row[2], row[3]) for index, row in enumerate(paired) if row[2] != row[3]]
    if unequal:
        raise RuntimeError(
            f"{metric} answer candidates are not token-length matched; "
            f"sum-logprob contrast would be biased: {unequal[:5]}"
        )
    return paired


def run_norming(
    model_name: str,
    scorer: BudgetedScorer,
    stimuli: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    rho_prompts, c_prompts, rows = [], [], []
    for stimulus in stimuli.itertuples(index=False):
        for hypothesis in ("h2", "h3"):
            hyp = hypothesis_text(stimulus, hypothesis)
            for frame_index in range(len(FRAME)):
                for variant in range(3):
                    rho_prompts.append(
                        norming_prompt(stimulus.intro, hyp, "P", frame_index, variant)
                    )
                    c_prompts.append(
                        norming_prompt(stimulus.intro, hyp, "C", frame_index, variant)
                    )
                    rows.append(
                        dict(
                            model=model_name.split("/")[-1],
                            model_id=model_name,
                            scenario=stimulus.scenario_id,
                            label=stimulus.label,
                            kind=stimulus.kind,
                            prefix=stimulus.prefix,
                            hypothesis=hypothesis,
                            frame=frame_index,
                            variant=variant,
                            prompt_contract=PROMPT_CONTRACT_VERSION,
                        )
                    )
    if len(rows) != 540:
        raise RuntimeError(f"Norming grid is {len(rows)}, expected 540")
    rho_scores = score_prompt_pairs(scorer, rho_prompts, "P", batch_size)
    c_scores = score_prompt_pairs(scorer, c_prompts, "C", batch_size)
    for row, rho, confirmation in zip(rows, rho_scores, c_scores):
        row["rho"] = rho[0] - rho[1]
        row["C"] = confirmation[0] - confirmation[1]
        row.update(
            rho_yes_lp=rho[0], rho_no_lp=rho[1],
            rho_yes_nt=rho[2], rho_no_nt=rho[3],
            C_strengthen_lp=confirmation[0], C_weaken_lp=confirmation[1],
            C_strengthen_nt=confirmation[2], C_weaken_nt=confirmation[3],
        )
    return pd.DataFrame(rows)


def activation_task(
    model_name: str,
    scorer: BudgetedScorer,
    stimuli: pd.DataFrame,
    task: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict]:
    labels = list(dict.fromkeys(stimuli.label.tolist()))
    label_index = {value: index for index, value in enumerate(labels)}
    prefix_index = {value: index for index, value in enumerate(PREFIXES)}
    option_index = {value: index for index, value in enumerate(OPTIONS)}
    accumulator = None
    prompt_accumulator = None
    squared_norms = None
    prompt_squared_norms = None
    counts = np.zeros((len(labels), len(PREFIXES), len(OPTIONS)), dtype=np.int16)
    prompt_counts = np.zeros((len(labels), len(PREFIXES)), dtype=np.int16)
    logprob_rows = []
    started = time.time()
    prompts_done = 0

    for stimulus in stimuli.itertuples(index=False):
        continuations = {
            "base": " " + stimulus.opt_base,
            "conj_h2": " " + stimulus.opt_c2,
            "conj_h3": " " + stimulus.opt_c3,
            "bare_h2": " " + stimulus.opt_b2,
            "bare_h3": " " + stimulus.opt_b3,
        }
        for frame_index, frame in enumerate(FRAME):
            for qvar_index, question in enumerate(QVAR[task]):
                prompt = behavior_prompt(
                    stimulus.intro, task, frame_index, qvar_index
                )
                meta = dict(
                    model=model_name.split("/")[-1],
                    model_id=model_name,
                    scenario=stimulus.scenario_id,
                    label=stimulus.label,
                    kind=stimulus.kind,
                    paradigm=stimulus.paradigm,
                    prefix=stimulus.prefix,
                    task=task,
                    frame=frame_index,
                    qvar=qvar_index,
                )
                for option_index_in_prompt, (option, continuation) in enumerate(continuations.items()):
                    lp, nt, hidden, prompt_hidden = scorer.score_hidden(prompt, continuation)
                    if accumulator is None:
                        nlayers = len(hidden)
                        hidden_size = int(hidden[0].shape[0])
                        if any(value.shape != (hidden_size,) for value in hidden):
                            raise RuntimeError("Hidden-state dimensions differ across layers")
                        accumulator = np.zeros(
                            (len(labels), len(PREFIXES), len(OPTIONS), nlayers, hidden_size),
                            dtype=np.float32,
                        )
                        prompt_accumulator = np.zeros(
                            (len(labels), len(PREFIXES), nlayers, hidden_size),
                            dtype=np.float32,
                        )
                        squared_norms = np.zeros((len(labels), len(PREFIXES), len(OPTIONS), nlayers), dtype=np.float64)
                        prompt_squared_norms = np.zeros((len(labels), len(PREFIXES), nlayers), dtype=np.float64)
                    vector = np.stack(hidden, axis=0)
                    li = label_index[stimulus.label]
                    pi = prefix_index[stimulus.prefix]
                    oi = option_index[option]
                    accumulator[li, pi, oi] += vector
                    assert squared_norms is not None
                    squared_norms[li, pi, oi] += np.square(vector.astype(np.float64)).sum(axis=-1)
                    counts[li, pi, oi] += 1
                    if option_index_in_prompt == 0:
                        assert prompt_accumulator is not None
                        prompt_accumulator[li, pi] += np.stack(prompt_hidden, axis=0)
                        assert prompt_squared_norms is not None
                        prompt_vector = np.stack(prompt_hidden, axis=0)
                        prompt_squared_norms[li, pi] += np.square(prompt_vector.astype(np.float64)).sum(axis=-1)
                        prompt_counts[li, pi] += 1
                    meta[f"lp_{option}"] = lp
                    meta[f"nt_{option}"] = nt
                logprob_rows.append(meta)
                prompts_done += 1
                if prompts_done == 1 or prompts_done % 25 == 0 or prompts_done == 540:
                    scorer.check_budget(f"activation-task{task}-{prompts_done}")
                    elapsed = time.time() - started
                    eta = (540 - prompts_done) * elapsed / prompts_done
                    print(
                        f"  activation task={task} {prompts_done}/540 "
                        f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                        flush=True,
                    )
    if accumulator is None or prompt_accumulator is None or squared_norms is None or prompt_squared_norms is None or not np.all(counts == 12) or not np.all(prompt_counts == 12):
        raise RuntimeError(f"Activation cell counts are not all 12: {np.unique(counts)}")
    accumulator /= counts[..., None, None]
    prompt_accumulator /= prompt_counts[..., None, None]
    variance_trace = (squared_norms - counts[..., None] * np.square(accumulator.astype(np.float64)).sum(axis=-1)) / (counts[..., None] - 1)
    prompt_variance_trace = (prompt_squared_norms - prompt_counts[..., None] * np.square(prompt_accumulator.astype(np.float64)).sum(axis=-1)) / (prompt_counts[..., None] - 1)
    variant_sd_norm = np.sqrt(np.maximum(variance_trace, 0)).astype(np.float32)
    prompt_variant_sd_norm = np.sqrt(np.maximum(prompt_variance_trace, 0)).astype(np.float32)
    metadata = {
        "model": model_name,
        "task": task,
        "shape": list(accumulator.shape),
        "axis_order": ["label", "prefix", "option", "hidden_state", "dimension"],
        "labels": labels,
        "prefixes": list(PREFIXES),
        "options": list(OPTIONS),
        "hidden_state_definition": "0=embedding output; l>=1=decoder block output",
        "read_position": "last token of attached option",
        "cell_variant_count": 12,
        "prompt_contract": PROMPT_CONTRACT_VERSION,
        "dtype": "float32",
        "prompt_final_shape": list(prompt_accumulator.shape),
        "prompt_final_definition": "last token before attached option; averaged over 12 variants",
        "variant_sd_norm_summary": {"mean": float(variant_sd_norm.mean()), "median": float(np.median(variant_sd_norm)), "max": float(variant_sd_norm.max())},
        "prompt_variant_sd_norm_summary": {"mean": float(prompt_variant_sd_norm.mean()), "median": float(np.median(prompt_variant_sd_norm)), "max": float(prompt_variant_sd_norm.max())},
    }
    return accumulator, prompt_accumulator, variant_sd_norm, prompt_variant_sd_norm, pd.DataFrame(logprob_rows), metadata


def behavior_crosscheck(extracted: pd.DataFrame, behavior_path: Path, task: int) -> dict:
    behavior = pd.read_csv(behavior_path)
    behavior = behavior[behavior.task == task]
    keys = ["model_id", "label", "prefix", "task", "frame", "qvar"]
    score_columns = [f"lp_{option}" for option in OPTIONS] + [f"nt_{option}" for option in OPTIONS]
    joined = extracted.merge(behavior[keys + score_columns], on=keys, suffixes=("_extract", "_behavior"), validate="one_to_one")
    if len(joined) != 540:
        raise RuntimeError(f"Behavior cross-check joined {len(joined)} rows, expected 540")
    lp_differences = []
    nt_mismatches = 0
    for option in OPTIONS:
        lp_differences.extend(
            (joined[f"lp_{option}_extract"] - joined[f"lp_{option}_behavior"]).abs().tolist()
        )
        nt_mismatches += int(
            (joined[f"nt_{option}_extract"] != joined[f"nt_{option}_behavior"]).sum()
        )
    return {
        "rows": len(joined),
        "max_abs_logprob_difference": max(lp_differences),
        "mean_abs_logprob_difference": float(np.mean(lp_differences)),
        "token_count_mismatches": nt_mismatches,
    }


def batched_behavior_logprobs(
    model_name: str,
    scorer: BudgetedScorer,
    stimuli: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    """Reproduce the frozen behavior scorer's complete ordering and batches."""
    rows = build_rows(model_name, stimuli)
    if len(rows) != 1_080:
        raise RuntimeError(f"Batched cross-check grid is {len(rows)}, expected 1080")
    return pd.DataFrame(scorer.score(rows, batch_size))


def frame_crosscheck(left: pd.DataFrame, right: pd.DataFrame, task: int) -> dict:
    """Measure numerical drift between two in-memory scoring paths."""
    keys = ["model_id", "label", "prefix", "task", "frame", "qvar"]
    score_columns = [f"lp_{option}" for option in OPTIONS] + [f"nt_{option}" for option in OPTIONS]
    joined = left.merge(
        right[keys + score_columns], on=keys, suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(joined) != 540:
        raise RuntimeError(f"In-memory cross-check joined {len(joined)}, expected 540")
    differences = []
    mismatches = 0
    for option in OPTIONS:
        differences.extend(
            (joined[f"lp_{option}_left"] - joined[f"lp_{option}_right"]).abs().tolist()
        )
        mismatches += int(
            (joined[f"nt_{option}_left"] != joined[f"nt_{option}_right"]).sum()
        )
    return {
        "task": task,
        "rows": len(joined),
        "max_abs_logprob_difference": max(differences),
        "mean_abs_logprob_difference": float(np.mean(differences)),
        "token_count_mismatches": mismatches,
    }


def validate_crosscheck(crosscheck: dict, task: int) -> None:
    if crosscheck.get("rows") != 540:
        raise RuntimeError(f"Behavior cross-check row count failed task={task}: {crosscheck}")
    if crosscheck.get("token_count_mismatches") != 0:
        raise RuntimeError(f"Token-count mismatch task={task}: {crosscheck}")
    difference = crosscheck.get("max_abs_logprob_difference")
    if difference is None or not np.isfinite(difference) or difference > 1e-4:
        raise RuntimeError(f"Behavior log-probability cross-check failed task={task}: {crosscheck}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--behavior-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-gpu-mib", type=int, default=24_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(args.threads)
    stimuli = load_stimuli(None)
    slug = safe_name(args.model)
    norm_path = args.output_dir / "norming" / f"norm_rq3_{slug}.csv"
    activation_dir = args.output_dir / "activations" / slug
    activation_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifests" / f"representation_{slug}.json"
    if manifest_path.exists():
        try:
            existing_contract = json.loads(manifest_path.read_text()).get("prompt_contract")
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Unreadable representation manifest: {manifest_path}") from error
        if existing_contract != PROMPT_CONTRACT_VERSION:
            raise RuntimeError(
                f"Refusing to overwrite/mix legacy representation output ({existing_contract!r}) "
                f"with {PROMPT_CONTRACT_VERSION}. Use a fresh output directory."
            )
    scorer = BudgetedScorer(args.model, args.max_gpu_mib, reserve_mib=1_000)
    started = time.time()
    task_manifests = {}
    try:
        if not norm_path.exists() or sum(1 for _ in norm_path.open()) != 541:
            norming = run_norming(args.model, scorer, stimuli, args.batch_size)
            atomic_frame(norming, norm_path)
        else:
            print(f"SKIP complete {norm_path}")

        # The original behavior run interleaves task 2 and task 3 within the
        # 1,080-row grid.  Reproduce the entire ordering before filtering so
        # padded batch shapes and floating-point kernels are exactly aligned.
        batch_logprobs = batched_behavior_logprobs(
            args.model, scorer, stimuli, args.batch_size
        )

        for task in (2, 3):
            array_path = activation_dir / f"task{task}_cellmean.npy"
            prompt_array_path = activation_dir / f"task{task}_prompt_cellmean.npy"
            variant_sd_path = activation_dir / f"task{task}_variant_sd_norm.npy"
            prompt_variant_sd_path = activation_dir / f"task{task}_prompt_variant_sd_norm.npy"
            meta_path = activation_dir / f"task{task}_metadata.json"
            extract_path = activation_dir / f"task{task}_logprob_crosscheck.csv"
            activation_logprob_path = activation_dir / f"task{task}_activation_logprob.csv"
            if array_path.exists() and prompt_array_path.exists() and variant_sd_path.exists() and prompt_variant_sd_path.exists() and meta_path.exists() and extract_path.exists() and activation_logprob_path.exists():
                print(f"SKIP complete task={task}")
                task_manifests[str(task)] = json.loads(meta_path.read_text())
                validate_crosscheck(task_manifests[str(task)].get("behavior_crosscheck", {}), task)
                continue
            array, prompt_array, variant_sd, prompt_variant_sd, extracted, metadata = activation_task(args.model, scorer, stimuli, task)
            batch_extracted = batch_logprobs[batch_logprobs.task == task].copy()
            crosscheck = behavior_crosscheck(batch_extracted, args.behavior_file, task)
            validate_crosscheck(crosscheck, task)
            metadata["behavior_crosscheck"] = crosscheck
            metadata["activation_forward_batch_drift"] = frame_crosscheck(
                extracted, batch_extracted, task
            )
            atomic_npy(array, array_path)
            atomic_npy(prompt_array, prompt_array_path)
            atomic_npy(variant_sd, variant_sd_path)
            atomic_npy(prompt_variant_sd, prompt_variant_sd_path)
            atomic_frame(batch_extracted, extract_path)
            atomic_frame(extracted, activation_logprob_path)
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            task_manifests[str(task)] = metadata
    finally:
        memory = scorer.memory_manifest()
        scorer.close()

    manifest = {
        "stage": "rq3-norming-activation",
        "model": args.model,
        "dtype": "float32",
        "cpu_offload_allowed": True,
        "elapsed_seconds": time.time() - started,
        "norming_rows": 540,
        "prompt_contract": PROMPT_CONTRACT_VERSION,
        "tasks": task_manifests,
        "memory": memory,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(manifest_path)
    print(f"COMPLETE representation model={args.model} elapsed={time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
