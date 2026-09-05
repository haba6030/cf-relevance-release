#!/usr/bin/env python3
"""RQ3-B primary intervention: ABL and ADD with preregistered controls.

The runner is item-sharded and restartable.  Directions for held item i are
estimated from the other 14 items.  ABL removes one direction at every hidden
state/all token positions.  ADD injects at the fold-selected layer from the
beginning of the intro through the attached option.
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

try:
    from analyze_rq3_representation import (
        centered_activation, centered_design, item_did, norm_arrays, ridge_gcv, unit,
    )
    from run_rq3_behavior_gpu import BudgetedScorer, FRAME, QVAR, load_stimuli, safe_name
    from rq3_prompting import PROMPT_CONTRACT_VERSION, behavior_prompt
    from rq3_decoder_contract import H3_DECODER_CONTRACT, H3_GATE_FAMILY, load_h3_decoder_gate
except ModuleNotFoundError:
    from sim.analyze_rq3_representation import (
        centered_activation, centered_design, item_did, norm_arrays, ridge_gcv, unit,
    )
    from sim.run_rq3_behavior_gpu import BudgetedScorer, FRAME, QVAR, load_stimuli, safe_name
    from sim.rq3_prompting import PROMPT_CONTRACT_VERSION, behavior_prompt
    from sim.rq3_decoder_contract import H3_DECODER_CONTRACT, H3_GATE_FAMILY, load_h3_decoder_gate

ALPHAS = (0.5, 1.0, 2.0, 4.0)
# ABL dose factor, mirroring the ADD alpha grid (lambda=1.0 is the old behaviour).
LAMBDAS = (0.25, 0.5, 1.0)
OPTIONS = ("base", "conj_h2", "conj_h3")
SEED = 20260824


def decoder_layers(model) -> list:
    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
    ]
    for path in candidates:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
        except AttributeError:
            continue
        return list(value)
    raise RuntimeError(f"Cannot locate decoder layers for {type(model).__name__}")


class DirectionHooks(AbstractContextManager):
    def __init__(
        self,
        model,
        directions: np.ndarray,
        mode: str,
        selected_layer: int | None = None,
        magnitude: float = 0.0,
        add_vectors: np.ndarray | None = None,
        dose: float = 1.0,
        ablate_scope: str = "all",
    ) -> None:
        self.model = model
        self.directions = directions
        # push direction for subspace ADD; None means a one-dimensional edit that uses directions as is.
        self.add_vectors = directions if add_vectors is None else add_vectors
        self.mode = mode
        self.selected_layer = selected_layer
        self.magnitude = magnitude
        # ABL dose: 1.0 is full projection removal (old behaviour), 0 is no edit.
        self.dose = float(dose)
        # ABL layer scope: "all" is preregistered (every layer and position); "selected" matches the ADD layer.
        self.ablate_scope = ablate_scope
        self.start_position = 0
        self.handles = []

    def transform(self, tensor: torch.Tensor, hidden_index: int) -> torch.Tensor:
        """Handles both a one-dimensional direction and a k-dimensional subspace.

        If `self.directions[hidden_index]` is (d,) this is the one-dimensional edit; if (k, d) a subspace.
        In the latter, ABL projects out the whole span and ADD pushes along one direction within it.
        """
        if not torch.is_tensor(tensor) or tensor.ndim != 3:
            return tensor
        if (self.mode == "ablate" and self.ablate_scope == "selected"
                and hidden_index != self.selected_layer):
            return tensor
        block = torch.as_tensor(
            self.directions[hidden_index], device=tensor.device, dtype=tensor.dtype
        )
        eps = torch.finfo(tensor.dtype).eps
        if block.ndim == 1:
            vector = block / block.norm().clamp_min(eps)
            if self.mode == "ablate":
                return tensor - self.dose * torch.einsum("bsd,d->bs", tensor, vector)[..., None] * vector
            if self.mode == "add" and hidden_index == self.selected_layer:
                output = tensor.clone()
                output[:, self.start_position :] += self.magnitude * vector
                return output
            return tensor
        if self.mode == "ablate":
            coefficients = torch.einsum("bsd,kd->bsk", tensor, block)
            return tensor - self.dose * torch.einsum("bsk,kd->bsd", coefficients, block)
        if self.mode == "add" and hidden_index == self.selected_layer:
            vector = torch.as_tensor(
                self.add_vectors[hidden_index], device=tensor.device, dtype=tensor.dtype
            )
            vector = vector / vector.norm().clamp_min(eps)
            output = tensor.clone()
            output[:, self.start_position :] += self.magnitude * vector
            return output
        return tensor

    def hook(self, hidden_index: int):
        def apply(_module, _inputs, output):
            if torch.is_tensor(output):
                return self.transform(output, hidden_index)
            if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                return (self.transform(output[0], hidden_index), *output[1:])
            if isinstance(output, list) and output and torch.is_tensor(output[0]):
                return [self.transform(output[0], hidden_index), *output[1:]]
            raise RuntimeError(f"Unexpected hook output at hidden state {hidden_index}: {type(output)}")
        return apply

    def __enter__(self):
        layers = decoder_layers(self.model)
        if len(layers) + 1 != self.directions.shape[0]:
            raise RuntimeError(
                f"Layer mismatch: modules={len(layers)} hidden_states={self.directions.shape[0]}"
            )
        self.handles.append(
            self.model.get_input_embeddings().register_forward_hook(self.hook(0))
        )
        for index, layer in enumerate(layers, start=1):
            if self.mode == "ablate" or index == self.selected_layer:
                self.handles.append(layer.register_forward_hook(self.hook(index)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def regression_directions(
    array: np.ndarray,
    rho: np.ndarray,
    confirmation: np.ndarray,
    train: np.ndarray,
) -> dict[str, np.ndarray]:
    layers, dimension = array.shape[-2:]
    output = {
        name: np.empty((layers, dimension), dtype=np.float32)
        for name in ("R_unique", "P", "C")
    }
    x = centered_design(rho, confirmation, train)
    for layer in range(layers):
        activation = centered_activation(array, train, layer)
        coefficient, _ = ridge_gcv(x, activation)
        output["R_unique"][layer] = coefficient[3]
        output["P"][layer] = coefficient[4]
        output["C"][layer] = coefficient[5]
    correlation = float(np.corrcoef(x[:, 4], x[:, 5])[0, 1])
    alignment = 1.0 if correlation >= 0 else -1.0
    p_unit = output["P"] / np.maximum(np.linalg.norm(output["P"], axis=1, keepdims=True), 1e-12)
    c_unit = output["C"] / np.maximum(np.linalg.norm(output["C"], axis=1, keepdims=True), 1e-12)
    output["PC"] = (p_unit + alignment * c_unit).astype(np.float32)
    return output


def shuffled_direction(array: np.ndarray, train: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for item in train:
        shuffled = array[item, rng.permutation(3)]
        values.append(shuffled[2, 2] - shuffled[1, 2] - shuffled[2, 1] + shuffled[1, 1])
    return np.mean(values, axis=0)


def random_directions(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def context_direction(array: np.ndarray, train: np.ndarray, options: tuple[int, ...]) -> np.ndarray:
    """Fold-trained R-minus-I direction for the requested continuation family."""
    r_states = np.asarray(array[train, 2][:, options], dtype=np.float32).mean(axis=1)
    i_states = np.asarray(array[train, 1][:, options], dtype=np.float32).mean(axis=1)
    return (r_states - i_states).mean(axis=0)


def orthonormal_basis(rows: np.ndarray) -> np.ndarray:
    """Orthonormalise row vectors to (k, d); linearly dependent rows are dropped."""
    basis: list[np.ndarray] = []
    for row in np.asarray(rows, dtype=np.float64):
        vector = row.copy()
        for existing in basis:
            vector -= (vector @ existing) * existing
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            basis.append(vector / norm)
    return np.stack(basis, axis=0) if basis else np.zeros((0, rows.shape[-1]))


def subspace_directions(array: np.ndarray, train: np.ndarray, options: tuple[int, ...], k: int) -> np.ndarray:
    """Per-layer k-dimensional subspace basis -> (layers, k, d).

    **v_R is the first basis vector**, so that k=1 equals R_h3 of h3_context exactly.
    The rest are the top PCs of the **residual after removing v_R**, fit on the training items only. Using
    the PCs of the raw contrast matrix would be wrong: |cos(v_R, PC1)| averages 0.81 (0.97 for gemma), so
    the orthogonal component is small and misaligned. Taking them from the residual matches the definition
    "principal dimensions of the variance v_R does not capture" (intervention guide 2.2).
    """
    layers = array.shape[-2]
    blocks = []
    for layer in range(layers):
        contrast = np.asarray(
            array[train, 2][:, options, layer].mean(axis=1)
            - array[train, 1][:, options, layer].mean(axis=1),
            dtype=np.float64,
        )
        mean = contrast.mean(axis=0)
        mean = mean / max(np.linalg.norm(mean), 1e-12)
        residual = contrast - np.outer(contrast @ mean, mean)
        _, _, right = np.linalg.svd(residual, full_matrices=False)
        candidates = np.vstack([mean[None, :], right[: max(k - 1, 0)]])
        basis = orthonormal_basis(candidates)[:k]
        if len(basis) < k:
            basis = np.vstack([basis, np.zeros((k - len(basis), array.shape[-1]))])
        blocks.append(basis)
    return np.stack(blocks, axis=0).astype(np.float32)


def denoised_direction(array: np.ndarray, train: np.ndarray, options: tuple[int, ...], k: int) -> np.ndarray:
    """Arm B: v_R projected onto the top-k principal-component space, a single direction -> (layers, d).

    The same direction the AUC-vs-k curve of `analyze_rq3_intervention_design.py` measures.
    There k=2 beat the raw v_R in 9 of 12 cells (mean 0.806 → 0.816).
    **The PCs come from the raw contrast matrix without removing v_R**: the object being projected is v_R itself,
    so PCs of the residual could not contain it and the definition would fail.
    """
    layers = array.shape[-2]
    blocks = []
    for layer in range(layers):
        contrast = np.asarray(
            array[train, 2][:, options, layer].mean(axis=1)
            - array[train, 1][:, options, layer].mean(axis=1),
            dtype=np.float64,
        )
        _, _, right = np.linalg.svd(contrast, full_matrices=False)
        basis = right[:k]
        projected = basis.T @ (basis @ contrast.mean(axis=0))
        blocks.append(projected / max(np.linalg.norm(projected), 1e-12))
    return np.stack(blocks, axis=0).astype(np.float32)


def orthogonalized_direction(target: np.ndarray, nuisances: tuple[np.ndarray, ...]) -> np.ndarray:
    """Arm D: remove from the target its projection onto the span of the nuisances, per layer -> (layers, d).

    Generalises the orthogonalised direction of Han, Lim, Kong & Jo (ICML 2026), "Dual Mechanisms of Value
    Expression" §3.2, to two nuisances. That paper uses the direction **for intervention**, asking whether
    one mechanism remains causal independently of the other: by ablation rather than by a geometric
    ratio. Since our H3f currently rests on a single purified result in the condition-contrast space,
    this arm poses the same question in activation space.

    Two caveats. (1) v_P and v_C did not pass self-target decoding (P 1/12, C 0/12), so
    **a positive result in this arm is weak evidence.** (2) The measured cos(arm D, arm A) at the selected
    layer averages **0.947**, above 0.98 for Qwen2.5-Math and Phi, so **in about half the cells it is
    effectively the same direction as arm A.** A negative result must not be read as "independent of P and C".
    This cosine is always reported with the result.
    """
    layers = target.shape[0]
    blocks = []
    for layer in range(layers):
        basis = orthonormal_basis(np.stack([np.asarray(n[layer], dtype=np.float64) for n in nuisances]))
        vector = np.asarray(target[layer], dtype=np.float64)
        if len(basis):
            vector = vector - basis.T @ (basis @ vector)
        blocks.append(vector / max(np.linalg.norm(vector), 1e-12))
    return np.stack(blocks, axis=0).astype(np.float32)


def random_subspace(shape: tuple[int, int], k: int, seed: int) -> np.ndarray:
    """Random orthonormal subspace of the same k -> (layers, k, d). Notebook §4-C, proposal 2."""
    rng = np.random.default_rng(seed)
    layers, dimension = shape
    return np.stack(
        [orthonormal_basis(rng.normal(size=(k, dimension))) for _ in range(layers)],
        axis=0,
    ).astype(np.float32)


def subspace_add_vectors(subspace: np.ndarray, held_contrast: np.ndarray) -> np.ndarray:
    """ADD push direction: P_k d_held per layer -> (layers, d)."""
    return np.stack(
        [basis.T @ (basis @ np.asarray(held_contrast[layer], dtype=np.float64))
         for layer, basis in enumerate(subspace)],
        axis=0,
    ).astype(np.float32)


def shuffled_context_direction(
    array: np.ndarray,
    train: np.ndarray,
    options: tuple[int, ...],
    seed: int,
) -> np.ndarray:
    """Within-item N/I/R shuffle control matched to a context decoder."""
    rng = np.random.default_rng(seed)
    values = []
    for item in train:
        shuffled = array[item, rng.permutation(3)]
        values.append(shuffled[2, options].mean(axis=0) - shuffled[1, options].mean(axis=0))
    return np.mean(values, axis=0)


def intro_start_token(tokenizer, frame: str, prompt: str) -> int:
    prefix = frame.split("{intro}", 1)[0]
    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    if prefix_ids and prompt_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("Could not align intro start with prompt tokens")
    return len(prefix_ids)


def score_prompt(
    scorer: BudgetedScorer,
    controller: DirectionHooks,
    stimulus,
    task: int,
    frame_index: int,
    qvar_index: int,
) -> dict:
    frame = FRAME[frame_index]
    prompt = behavior_prompt(stimulus.intro, task, frame_index, qvar_index)
    controller.start_position = intro_start_token(scorer.tok, frame, prompt)
    continuations = {
        "base": " " + stimulus.opt_base,
        "conj_h2": " " + stimulus.opt_c2,
        "conj_h3": " " + stimulus.opt_c3,
    }
    row = {
        "label": stimulus.label,
        "scenario": stimulus.scenario_id,
        "prefix": stimulus.prefix,
        "task": task,
        "frame": frame_index,
        "qvar": qvar_index,
    }
    for option, continuation in continuations.items():
        lp, nt = scorer.score_one(prompt, continuation)
        row[f"lp_{option}"] = lp
        row[f"nt_{option}"] = nt
    return row


def score_frame_variants(
    scorer: BudgetedScorer,
    controller: DirectionHooks,
    stimulus,
    task: int,
    frame_index: int,
) -> list[dict]:
    """Score six question variants and three continuations in one batch."""
    frame = FRAME[frame_index]
    prompts = [
        behavior_prompt(stimulus.intro, task, frame_index, qvar_index)
        for qvar_index in range(6)
    ]
    starts = {intro_start_token(scorer.tok, frame, prompt) for prompt in prompts}
    if len(starts) != 1:
        raise RuntimeError("Intro start changed across question variants")
    controller.start_position = starts.pop()
    continuation_map = {
        "base": " " + stimulus.opt_base,
        "conj_h2": " " + stimulus.opt_c2,
        "conj_h3": " " + stimulus.opt_c3,
    }
    pair_prompts, pair_continuations, pair_keys = [], [], []
    for qvar_index, prompt in enumerate(prompts):
        for option, continuation in continuation_map.items():
            pair_prompts.append(prompt)
            pair_continuations.append(continuation)
            pair_keys.append((qvar_index, option))
    scored = scorer.score_pairs(pair_prompts, pair_continuations, batch_size=18)
    rows = [
        {
            "label": stimulus.label,
            "scenario": stimulus.scenario_id,
            "prefix": stimulus.prefix,
            "task": task,
            "frame": frame_index,
            "qvar": qvar_index,
        }
        for qvar_index in range(6)
    ]
    for (qvar_index, option), (lp, nt) in zip(pair_keys, scored):
        rows[qvar_index][f"lp_{option}"] = lp
        rows[qvar_index][f"nt_{option}"] = nt
    return rows


def run_item(
    model_id: str,
    scorer: BudgetedScorer,
    stimuli: pd.DataFrame,
    array: np.ndarray,
    prompt_array: np.ndarray,
    rho: np.ndarray,
    confirmation: np.ndarray,
    task: int,
    held: int,
    selected_layer: int,
    direction_mode: str = "h3_context",
    subspace_k: int = 1,
    random_subspaces: int = 0,
    ablate_scope: str = "all",
) -> pd.DataFrame:
    labels = list(dict.fromkeys(stimuli.label.tolist()))
    label = labels[held]
    decoder_contract = (
        H3_DECODER_CONTRACT if direction_mode in ("h3_context", "h3_subspace")
        else "retired-one-vs-three"
    )
    train = np.asarray([index for index in range(len(labels)) if index != held])
    regression = regression_directions(array, rho, confirmation, train)
    add_vectors: dict[str, np.ndarray] = {}
    if direction_mode == "h3_subspace":
        # notebook §4-C proposals 1 and 2. k=1 equals R_h3 of h3_context (verified cos = 1.000000).
        held_contrast = np.asarray(array[held, 2, 2] - array[held, 1, 2], dtype=np.float64)
        basis = subspace_directions(array, train, (2,), subspace_k)
        v_h3 = context_direction(array, train, (2,))
        directions = {
            # arm A: the raw v_R. It must be in the same run in k=2 mode for the A/B/C/D comparison to hold.
            "R_h3": v_h3,
            f"R_h3_k{subspace_k}": basis,
            "R_h2": context_direction(array, train, (1,)),
            "R_pooled": context_direction(array, train, (1, 2)),
            "R_base": context_direction(array, train, (0,)),
            "P": regression["P"], "C": regression["C"], "PC": regression["PC"],
            "shuffle": shuffled_context_direction(array, train, (2,), SEED + held + task * 1000),
        }
        # arm B: v_R projected onto the top-k PC space, a **single direction**.
        # the arm the AUC-vs-k curve supports directly; its ABL burden is one-dimensional like arm A.
        directions["R_h3_denoised"] = denoised_direction(array, train, (2,), subspace_k)
        # arm D: R with the span of v_P and v_C removed; the standard causal-independence test of Han et al.
        directions["R_h3_orth_PC"] = orthogonalized_direction(
            v_h3, (regression["P"], regression["C"]))
        # one-dimensional random control; arms A, B and D are one-dimensional, so a k-dimensional random subspace is not matched.
        directions["random"] = random_directions(v_h3.shape, SEED + held + task * 100)
        add_vectors[f"R_h3_k{subspace_k}"] = subspace_add_vectors(basis, held_contrast)
        for index in range(random_subspaces):
            name = f"random_k{subspace_k}_{index:02d}"
            control = random_subspace(
                (basis.shape[0], basis.shape[-1]), subspace_k,
                SEED + held + task * 100 + index * 7919)
            directions[name] = control
            add_vectors[name] = subspace_add_vectors(control, held_contrast)
        add_names = tuple(
            ["R_h3", f"R_h3_k{subspace_k}", "R_h3_denoised", "R_h3_orth_PC",
             "R_h2", "R_pooled", "R_base", "random", "shuffle"]
            + [f"random_k{subspace_k}_{index:02d}" for index in range(random_subspaces)]
        )
    elif direction_mode == "h3_context":
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
        add_names = ("R_h3", "R_h2", "R_pooled", "R_base", "random", "shuffle")
    else:
        did = item_did(array)
        v_r = did[train].mean(axis=0)
        v_prompt = (prompt_array[train, 2] - prompt_array[train, 1]).mean(axis=0)
        directions = {
            "R": v_r,
            "R_unique": regression["R_unique"],
            "R_prompt": v_prompt,
            "P": regression["P"], "C": regression["C"], "PC": regression["PC"],
            "random": random_directions(v_r.shape, SEED + held + task * 100),
            "shuffle": shuffled_direction(array, train, SEED + held + task * 1000),
        }
        add_names = ("R", "R_unique", "R_prompt", "random", "shuffle")
    rows = []
    selected_stimuli = stimuli[(stimuli.label == label) & stimuli.prefix.isin(["I", "R"])]
    scopes = ("all", "selected") if ablate_scope == "both" else (ablate_scope,)
    for name, direction in directions.items():
        for scope in scopes:
            for dose in LAMBDAS:
                # the preregistered cell is (scope="all", dose=1.0); the rest are declared secondary arms.
                with DirectionHooks(scorer.model, direction, mode="ablate", dose=dose,
                                    ablate_scope=scope, selected_layer=selected_layer,
                                    add_vectors=add_vectors.get(name)) as controller:
                    for stimulus in selected_stimuli.itertuples(index=False):
                        for frame_index in range(2):
                            for row in score_frame_variants(
                                scorer, controller, stimulus, task, frame_index
                            ):
                                row.update(
                                    model_id=model_id, stage="ABL", direction=name,
                                    alpha=np.nan, dose=dose, ablate_scope=scope,
                                    preregistered=bool(scope == "all" and dose == 1.0),
                                    prompt_contract=PROMPT_CONTRACT_VERSION,
                                    decoder_contract=decoder_contract,
                                )
                                rows.append(row)

    # Registered activation unit: norm of the across-item SD vector, computed
    # on training items only at the fold-selected hidden state.
    training_states = np.asarray(array[train, :, :, selected_layer], dtype=np.float32)
    scale = float(np.linalg.norm(training_states.std(axis=0, ddof=1).mean(axis=(0, 1))))
    prompt_training_states = np.asarray(prompt_array[train, :, selected_layer], dtype=np.float32)
    prompt_scale = float(np.linalg.norm(prompt_training_states.std(axis=0, ddof=1).mean(axis=0)))
    stimulus_i = selected_stimuli[selected_stimuli.prefix == "I"].iloc[0]
    for name in add_names:
        direction = directions[name]
        direction_scale = prompt_scale if name == "R_prompt" else scale
        for alpha in ALPHAS:
            with DirectionHooks(
                scorer.model,
                direction,
                mode="add",
                selected_layer=selected_layer,
                magnitude=alpha * direction_scale,
                add_vectors=add_vectors.get(name),
            ) as controller:
                for frame_index in range(2):
                    for row in score_frame_variants(
                        scorer, controller, stimulus_i, task, frame_index
                    ):
                        row.update(
                            model_id=model_id,
                            stage="ADD",
                            direction=name,
                            alpha=alpha,
                            dose=np.nan, ablate_scope="", preregistered=False,
                            activation_scale=direction_scale,
                            selected_layer=selected_layer,
                            prompt_contract=PROMPT_CONTRACT_VERSION,
                            decoder_contract=decoder_contract,
                        )
                        rows.append(row)
    expected = len(directions) * len(scopes) * len(LAMBDAS) * 2 * 12 + len(add_names) * len(ALPHAS) * 12
    if len(rows) != expected:
        raise RuntimeError(f"Intervention item rows {len(rows)} != {expected}")
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
    parser.add_argument(
        "--include-decode-fail",
        action="store_true",
        help="Exploratory only: run tasks that failed the preregistered decoding gate.",
    )
    parser.add_argument("--steer-subdir", default="steer")
    parser.add_argument("--manifest-prefix", default="intervention")
    parser.add_argument("--decoder-gate-file", type=Path)
    parser.add_argument("--gate-family", default=H3_GATE_FAMILY)
    parser.add_argument("--direction-mode",
                        choices=("registered_did", "h3_context", "h3_subspace"),
                        default="h3_context")
    parser.add_argument("--subspace-k", type=int, default=2,
                        help="subspace dimension for h3_subspace; k=1 equals h3_context (notebook §4-C C-1).")
    parser.add_argument("--ablate-scope", default="selected", choices=("all", "selected", "both"),
                        help="selected = match the ADD layer (default). all = preregistered, every layer. both = both")
    parser.add_argument("--random-subspaces", type=int, default=8,
                        help="number of random k-dimensional subspace controls (m in §4-C C-3).")
    parser.add_argument(
        "--allow-retired-one-vs-three",
        action="store_true",
        help="Diagnostic reproduction only; never use for canonical downstream results.",
    )
    args = parser.parse_args()
    if args.max_gpu_mib > 24_000:
        raise ValueError("GPU ceiling cannot exceed 24,000 MiB")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Runtime-only control for multi-GPU queues.  Four simultaneous NumPy/PyTorch
    # workers otherwise oversubscribe the 40 host threads and leave every GPU
    # idle.  It does not alter the model, stimuli, folds, or arithmetic dtype.
    torch.set_num_threads(int(os.environ.get("RQ3_TORCH_THREADS", "4")))
    stimuli = load_stimuli(None)
    slug = safe_name(args.model)
    labels = list(dict.fromkeys(stimuli.label.tolist()))
    if args.items:
        unknown = sorted(set(args.items) - set(labels))
        if unknown:
            raise ValueError(f"Unknown item labels: {unknown}")
        selected_items = set(args.items)
    else:
        selected_items = None
    norming = pd.read_csv(args.output_dir / "norming" / f"norm_rq3_{slug}.csv")
    if args.direction_mode == "registered_did" and not args.allow_retired_one_vs_three:
        raise ValueError(
            "registered_did is tied to the retired one-positive/three-negative decoder. "
            "Use h3_context, or pass --allow-retired-one-vs-three for diagnostic reproduction only."
        )
    gate_path = args.decoder_gate_file or args.output_dir / "analysis" / "rq3a_context_decoder_robustness.csv"
    if args.direction_mode in ("h3_context", "h3_subspace"):
        # h3_subspace uses the same canonical gate as h3_context; the two modes coincide at k=1.
        gates = load_h3_decoder_gate(gate_path)
    else:
        gates = pd.read_csv(gate_path)
        if args.gate_family is not None:
            if "family" not in gates.columns:
                raise ValueError(f"Gate file has no family column: {gate_path}")
            gates = gates[gates.family == args.gate_family]
    gates = gates[gates.model_id == args.model]
    gates = gates[~gates.decode_gate] if args.include_decode_fail else gates[gates.decode_gate]
    if args.tasks:
        gates = gates[gates.task.isin(args.tasks)]
    if gates.empty:
        gate_kind = "decode-fail" if args.include_decode_fail else "decode-pass"
        print(f"SKIP no {gate_kind} task for {args.model}")
        return 0
    scorer = BudgetedScorer(args.model, args.max_gpu_mib, reserve_mib=1_000)
    rho, confirmation = norm_arrays(norming, labels)
    completed = []
    try:
        for gate in gates.itertuples(index=False):
            task = int(gate.task)
            selected = np.asarray(json.loads(gate.selected_layers), dtype=int)
            array = np.load(
                args.output_dir / "activations" / slug / f"task{task}_cellmean.npy",
                mmap_mode="r",
            )
            prompt_array = np.load(args.output_dir / "activations" / slug / f"task{task}_prompt_cellmean.npy", mmap_mode="r")
            shard_dir = args.output_dir / args.steer_subdir / slug / f"task{task}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            for held, label in enumerate(labels):
                if selected_items is not None and label not in selected_items:
                    continue
                path = shard_dir / f"{label}.csv"
                if path.exists():
                    existing = pd.read_csv(path, nrows=1)
                    if (
                        "prompt_contract" in existing
                        and set(existing.prompt_contract) == {PROMPT_CONTRACT_VERSION}
                        and "decoder_contract" in existing
                        and set(existing.decoder_contract) == {H3_DECODER_CONTRACT}
                    ):
                        print(f"SKIP task={task} item={label}")
                        continue
                    raise RuntimeError(
                        f"Legacy intervention shard would be mixed with {PROMPT_CONTRACT_VERSION}: "
                        f"{path}. Use a fresh output directory."
                    )
                result = run_item(
                    args.model, scorer, stimuli, array, prompt_array, rho, confirmation,
                    task, held, int(selected[held]), args.direction_mode,
                    args.subspace_k, args.random_subspaces, args.ablate_scope,
                )
                temporary = path.with_suffix(".csv.tmp")
                result.to_csv(temporary, index=False); temporary.replace(path)
                scorer.check_budget(f"intervention-task{task}-{label}")
                print(f"SAVED task={task} item={label} rows={len(result)}", flush=True)
            completed.append(task)
        memory = scorer.memory_manifest()
    finally:
        scorer.close()
    manifest = {
        "stage": "rq3-exploratory-decode-fail-intervention" if args.include_decode_fail else "rq3-primary-intervention",
        "model": args.model,
        "tasks": completed,
        "items": args.items,
        "alphas": list(ALPHAS),
        "gpu_ceiling_mib": args.max_gpu_mib,
        "decoder_gate_file": str(gate_path),
        "gate_family": args.gate_family,
        "direction_mode": args.direction_mode,
        "subspace_k": args.subspace_k,
        "random_subspaces": args.random_subspaces,
        "ablate_scope": args.ablate_scope,
        "abl_doses": list(LAMBDAS),
        "decoder_contract": (H3_DECODER_CONTRACT
                             if args.direction_mode in ("h3_context", "h3_subspace")
                             else "retired-one-vs-three"),
        "prompt_contract": PROMPT_CONTRACT_VERSION,
        "memory": memory,
        "runtime_contract": globals().get("RUNTIME_CONTRACT", "canonical"),
        "runtime_direction_set": globals().get("RUNTIME_DIRECTION_SET", "all"),
        "runtime_qvars": globals().get("RUNTIME_QVARS", 6),
    }
    path = args.output_dir / "manifests" / f"{args.manifest_prefix}_{slug}.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
