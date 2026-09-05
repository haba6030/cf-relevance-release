#!/usr/bin/env python3
"""Exploratory nested-LOIO R-vs-I decoding by continuation family."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from rq3_decoder_contract import H3_DECODER_CONTRACT
except ModuleNotFoundError:
    from sim.rq3_decoder_contract import H3_DECODER_CONTRACT

SEED = 20260824
I, R = 1, 2
FAMILIES = {
    "conjunction": (1, 2),
    "conjunction_h2": (1,),
    "conjunction_h3": (2,),
    "base": (0,),
}


def unit_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def means(array: np.ndarray, items: np.ndarray, layer: int, options: tuple[int, ...]) -> np.ndarray:
    """[item, prefix, hidden], averaging only the requested continuations."""
    return np.asarray(array[items, :, :, layer][:, :, options, :], dtype=np.float32).mean(axis=2)


def score_item(states: np.ndarray, direction: np.ndarray, options: tuple[int, ...]) -> tuple[list[int], list[float]]:
    positive = states[R, list(options)] @ direction
    negative = states[I, list(options)] @ direction
    return [1] * len(options) + [0] * len(options), np.r_[positive, negative].astype(float).tolist()


def nested_loio(array: np.ndarray, options: tuple[int, ...]) -> tuple[np.ndarray, float, np.ndarray]:
    n, _, _, layer_count, _ = array.shape
    items = np.arange(n)
    selected, labels, scores = [], [], []
    for outer in range(n):
        train = items[items != outer]
        layer_auc = []
        for layer in range(layer_count):
            cell = means(array, train, layer, options)
            contrast = cell[:, R] - cell[:, I]
            total = contrast.sum(axis=0)
            inner_labels, inner_scores = [], []
            for position, inner in enumerate(train):
                direction = unit_rows(((total - contrast[position]) / (n - 2))[None])[0]
                y, s = score_item(array[inner, :, :, layer], direction, options)
                inner_labels.extend(y); inner_scores.extend(s)
            layer_auc.append(roc_auc_score(inner_labels, inner_scores))
        chosen = int(np.argmax(layer_auc)); selected.append(chosen)
        cell = means(array, train, chosen, options)
        direction = unit_rows((cell[:, R] - cell[:, I]).mean(axis=0)[None])[0]
        y, s = score_item(array[outer, :, :, chosen], direction, options)
        labels.extend(y); scores.extend(s)
    labels_array = np.asarray(labels)
    return np.asarray(selected), float(roc_auc_score(labels_array, scores)), labels_array


def auc_rows(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    positive, negative = scores[:, labels == 1], scores[:, labels == 0]
    delta = positive[:, :, None] - negative[:, None, :]
    return (delta > 0).mean(axis=(1, 2)) + 0.5 * (delta == 0).mean(axis=(1, 2))


def nulls(
    array: np.ndarray,
    options: tuple[int, ...],
    selected: np.ndarray,
    labels: np.ndarray,
    observed: float,
    random_n: int = 1000,
    permutation_n: int = 200,
) -> dict:
    rng = np.random.default_rng(SEED)
    n, _, _, _, dimension = array.shape
    per_item = 2 * len(options)
    random_scores = np.empty((random_n, n * per_item), dtype=np.float32)
    for held in range(n):
        directions = unit_rows(rng.normal(size=(random_n, dimension)).astype(np.float32))
        states = np.r_[array[held, R, list(options), selected[held]], array[held, I, list(options), selected[held]]]
        random_scores[:, held * per_item : (held + 1) * per_item] = directions @ states.T
    random_auc = auc_rows(labels, random_scores)

    permutations = np.stack([[rng.permutation(3) for _ in range(n)] for _ in range(permutation_n)])
    permutation_scores = np.empty((permutation_n, n * per_item), dtype=np.float32)
    items = np.arange(n)
    item_index = np.broadcast_to(items, (permutation_n, n))
    for held in range(n):
        layer = int(selected[held]); cell = means(array, items, layer, options)
        permuted_r = cell[item_index, permutations[:, :, R]]
        permuted_i = cell[item_index, permutations[:, :, I]]
        keep = items != held
        directions = unit_rows((permuted_r[:, keep] - permuted_i[:, keep]).mean(axis=1))
        states = np.r_[array[held, R, list(options), layer], array[held, I, list(options), layer]]
        permutation_scores[:, held * per_item : (held + 1) * per_item] = directions @ states.T
    permutation_auc = auc_rows(labels, permutation_scores)
    return {
        "random_n": random_n,
        "random_95": float(np.quantile(random_auc, .95)),
        "random_p": float((1 + np.sum(random_auc >= observed)) / (random_n + 1)),
        "permutation_n": permutation_n,
        "permutation_95": float(np.quantile(permutation_auc, .95)),
        "permutation_p": float((1 + np.sum(permutation_auc >= observed)) / (permutation_n + 1)),
    }


def labels_for(family: str) -> tuple[str, str]:
    return {
        "conjunction": ("R,h2 + R,h3", "I,h2 + I,h3"),
        "conjunction_h2": ("R,h2", "I,h2"),
        "conjunction_h3": ("R,h3", "I,h3"),
        "base": ("R,base", "I,base"),
    }[family]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Analysis output directory (default: INPUT_DIR/analysis).",
    )
    args = parser.parse_args()
    rows = []
    for manifest_path in sorted((args.input_dir / "manifests").glob("representation_*.json")):
        manifest = json.loads(manifest_path.read_text()); model_id = manifest["model"]
        model_slug = manifest_path.stem.removeprefix("representation_")
        for task in (2, 3):
            path = args.input_dir / "activations" / model_slug / f"task{task}_cellmean.npy"
            if not path.exists(): continue
            array = np.load(path, mmap_mode="r")
            for family, options in FAMILIES.items():
                selected, auc, labels = nested_loio(array, options)
                reference = nulls(array, options, selected, labels, auc)
                positive, negative = labels_for(family)
                rows.append({
                    "model_id": model_id, "model": model_id.split("/")[-1], "task": task,
                    "family": family, "positive_label": positive, "negative_label": negative,
                    "decoder_contract": H3_DECODER_CONTRACT,
                    "task_training": "separate",
                    "auc_nested_loio": auc, "selected_layers": json.dumps(selected.tolist()),
                    "selected_layer_mode": Counter(selected).most_common(1)[0][0],
                    "decode_gate": bool(auc > .5 and auc > reference["random_95"] and auc > reference["permutation_95"]),
                    **reference,
                })
                print(model_id, task, family, f"AUC={auc:.4f}", flush=True)
    result = pd.DataFrame(rows)
    output = args.output_dir or args.input_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "rq3a_context_decoder_robustness.csv", index=False)
    summary = result.groupby("family").agg(mean_auc=("auc_nested_loio", "mean"), passed=("decode_gate", "sum"), total=("decode_gate", "size"))
    (output / "rq3a_context_decoder_robustness.md").write_text(
        "# Exploratory R-vs-I context decoding\n\n" + summary.to_markdown() + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
