#!/usr/bin/env python3
"""Hierarchical RQ3 P/C detection and same-layer geometry.

This is a post-hoc diagnostic analysis fixed in
``EXPERIMENT_RQ3_PC_REANALYSIS.md``.  It deliberately separates total,
same-environment change, and unique signals instead of treating one fully
residualized binary AUC as evidence for all three claims.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:  # package import in tests, script-local import on the command line
    from sim.analyze_rq3_h3b_geometry import residualize
    from sim.analyze_rq3_representation import finite_corr, norm_arrays, unit
    from sim.rq3_decoder_contract import load_h3_decoder_gate
except ModuleNotFoundError:  # pragma: no cover
    from analyze_rq3_h3b_geometry import residualize
    from analyze_rq3_representation import finite_corr, norm_arrays, unit
    from rq3_decoder_contract import load_h3_decoder_gate


SEED = 20260825
I, R = 1, 2
CONJUNCTION = (1, 2)
BARE = (3, 4)


def format_states(array: np.ndarray, form: str) -> np.ndarray:
    """Return prefix-major P/C states as [item, six cells, layer, dim]."""
    options = BARE if form == "bare" else CONJUNCTION
    selected = np.take(array, options, axis=2)  # item, prefix, hypothesis, layer, dim
    return np.asarray(selected, dtype=np.float32).reshape(
        selected.shape[0], 6, selected.shape[3], selected.shape[4]
    )


def change_states(array: np.ndarray, contrast: str) -> np.ndarray:
    """Return conjunction R-I h3 or its h3-h2 difference as [item,1,L,D]."""
    h3 = np.asarray(array[:, R, 2] - array[:, I, 2], dtype=np.float32)
    if contrast == "delta_h3":
        result = h3
    elif contrast == "did":
        h2 = np.asarray(array[:, R, 1] - array[:, I, 1], dtype=np.float32)
        result = h3 - h2
    else:  # pragma: no cover - callers use a closed set
        raise ValueError(contrast)
    return result[:, None]


def change_target(values: np.ndarray, contrast: str) -> np.ndarray:
    h3 = values[:, R, 1] - values[:, I, 1]
    result = h3 if contrast == "delta_h3" else h3 - (values[:, R, 0] - values[:, I, 0])
    return np.asarray(result, dtype=float)[:, None]


def _center(
    x_train: np.ndarray,
    x_held: np.ndarray,
    y_train: np.ndarray,
    y_held: np.ndarray,
    cell_center: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cell_center:
        x_mean = x_train.mean(axis=0, keepdims=True)
        y_mean = y_train.mean(axis=0, keepdims=True)
    else:
        x_mean = x_train.mean(axis=(0, 1), keepdims=True)
        y_mean = y_train.mean(axis=(0, 1), keepdims=True)
    return x_train - x_mean, x_held - x_mean, y_train - y_mean, y_held - y_mean


def _residualize_control(
    x_train: np.ndarray,
    x_held: np.ndarray,
    y_train: np.ndarray,
    y_held: np.ndarray,
    z_train: np.ndarray,
    z_held: np.ndarray,
    cell_center: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fold-wise FWL residualization after applying the same centering rule."""
    if cell_center:
        z_mean = z_train.mean(axis=0, keepdims=True)
    else:
        z_mean = z_train.mean(axis=(0, 1), keepdims=True)
    z_train = z_train - z_mean
    z_held = z_held - z_mean
    denominator = float(np.sum(z_train * z_train))
    if denominator <= 1e-12:
        ratio = 0.0
        return x_train, x_held, y_train, y_held, ratio
    beta_y = float(np.sum(z_train * y_train) / denominator)
    beta_x = np.einsum("ik,ikd->d", z_train, x_train) / denominator
    original = float(np.sum(y_train * y_train))
    y_train_res = y_train - z_train * beta_y
    y_held_res = y_held - z_held * beta_y
    x_train_res = x_train - z_train[:, :, None] * beta_x
    x_held_res = x_held - z_held[:, :, None] * beta_x
    ratio = float(np.sum(y_train_res * y_train_res) / max(original, 1e-12))
    return x_train_res, x_held_res, y_train_res, y_held_res, ratio


def fit_fold_direction(
    x_train: np.ndarray,
    x_held: np.ndarray,
    y_train: np.ndarray,
    y_held: np.ndarray,
    *,
    cell_center: bool,
    z_train: np.ndarray | None = None,
    z_held: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a one-component covariance direction and calibrate it on train rows."""
    x_train, x_held, y_train, y_held = _center(
        x_train, x_held, y_train, y_held, cell_center
    )
    variance_ratio = 1.0
    if z_train is not None and z_held is not None:
        x_train, x_held, y_train, y_held, variance_ratio = _residualize_control(
            x_train, x_held, y_train, y_held, z_train, z_held, cell_center
        )
    flat_x = x_train.reshape(-1, x_train.shape[-1])
    flat_y = y_train.reshape(-1)
    direction = unit(flat_x.T @ flat_y)
    train_score = flat_x @ direction
    denominator = float(train_score @ train_score)
    slope = float(train_score @ flat_y / denominator) if denominator > 1e-12 else 0.0
    prediction = (x_held.reshape(-1, x_held.shape[-1]) @ direction * slope).reshape(y_held.shape)
    return direction, prediction, y_held, variance_ratio


def exact_sign_flip(values: np.ndarray, alternative: str = "greater") -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    codes = np.arange(1 << n, dtype=np.uint32)[:, None]
    bits = ((codes >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)
    null = ((bits * 2 - 1).astype(float) @ values) / n
    observed = float(values.mean())
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-15))
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-15))


def summarize_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    variance_ratios: list[float],
    *,
    bootstrap: int = 5000,
) -> dict[str, float | bool | int]:
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    contributions = np.mean(targets * predictions, axis=1)
    n = len(contributions)
    mean = float(contributions.mean())
    sd = float(contributions.std(ddof=1))
    se = sd / np.sqrt(n)
    t_value = mean / se if se > 0 else (np.inf if mean > 0 else 0.0)
    p_one = float(stats.t.sf(t_value, n - 1))
    p_two = float(2 * stats.t.sf(abs(t_value), n - 1))
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, n, size=(bootstrap, n))
    boot_target = targets[indices].reshape(bootstrap, -1)
    boot_prediction = predictions[indices].reshape(bootstrap, -1)
    boot_target -= boot_target.mean(axis=1, keepdims=True)
    boot_prediction -= boot_prediction.mean(axis=1, keepdims=True)
    boot_denominator = np.sqrt(
        np.sum(boot_target**2, axis=1) * np.sum(boot_prediction**2, axis=1)
    )
    boot_r = np.divide(
        np.sum(boot_target * boot_prediction, axis=1),
        boot_denominator,
        out=np.full(bootstrap, np.nan),
        where=boot_denominator > 1e-12,
    )
    target_flat = targets.reshape(-1)
    prediction_flat = predictions.reshape(-1)
    pearson = finite_corr(target_flat, prediction_flat)
    spearman = finite_corr(target_flat, prediction_flat, "spearman")
    q2 = float(1 - np.sum((target_flat - prediction_flat) ** 2) / max(np.sum(target_flat**2), 1e-12))
    p_flip_one = exact_sign_flip(contributions, "greater")
    p_flip_two = exact_sign_flip(contributions, "two-sided")
    return {
        "n_items": n,
        "n_rows": target_flat.size,
        "pearson": pearson,
        "spearman": spearman,
        "q2": q2,
        "mean_covariance": mean,
        "t": t_value,
        "p_t_one": p_one,
        "p_t_two": p_two,
        "p_flip_one": p_flip_one,
        "p_flip_two": p_flip_two,
        "pearson_boot_lo": float(np.nanquantile(boot_r, 0.025)),
        "pearson_boot_hi": float(np.nanquantile(boot_r, 0.975)),
        "residual_variance_ratio": float(np.mean(variance_ratios)),
        "gate": bool(np.isfinite(pearson) and pearson > 0 and p_flip_one < 0.05),
    }


def loio_evaluate(
    states: np.ndarray,
    target: np.ndarray,
    selected_layers: np.ndarray,
    *,
    cell_center: bool,
    control: np.ndarray | None = None,
    bootstrap: int = 5000,
) -> tuple[dict, pd.DataFrame, list[np.ndarray]]:
    n = len(states)
    targets, predictions, directions, ratios, fold_rows = [], [], [], [], []
    for held in range(n):
        train = np.flatnonzero(np.arange(n) != held)
        layer = int(selected_layers[held])
        direction, prediction, held_target, ratio = fit_fold_direction(
            states[train, :, layer],
            states[[held], :, layer],
            target[train],
            target[[held]],
            cell_center=cell_center,
            z_train=None if control is None else control[train],
            z_held=None if control is None else control[[held]],
        )
        targets.append(held_target[0])
        predictions.append(prediction[0])
        directions.append(direction)
        ratios.append(ratio)
        fold_rows.append(
            {
                "held_item": held,
                "layer": layer,
                "target_values": json.dumps(held_target[0].tolist()),
                "predictions": json.dumps(prediction[0].tolist()),
                "covariance_contribution": float(np.mean(held_target[0] * prediction[0])),
                "residual_variance_ratio": ratio,
            }
        )
    summary = summarize_predictions(np.stack(targets), np.stack(predictions), ratios, bootstrap=bootstrap)
    summary["selected_layer_mode"] = Counter(selected_layers.tolist()).most_common(1)[0][0]
    return summary, pd.DataFrame(fold_rows), directions


def all_layer_curve(
    states: np.ndarray,
    target: np.ndarray,
    *,
    cell_center: bool,
) -> pd.DataFrame:
    """Vectorized fixed-layer LOIO curve without outcome-based layer selection."""
    n, _, layers, _ = states.shape
    held_targets: list[np.ndarray] = []
    held_predictions: list[np.ndarray] = []
    for held in range(n):
        train = np.flatnonzero(np.arange(n) != held)
        x_train = states[train]
        x_held = states[[held]]
        y_train = target[train]
        y_held = target[[held]]
        if cell_center:
            x_mean = x_train.mean(axis=0, keepdims=True)
            y_mean = y_train.mean(axis=0, keepdims=True)
        else:
            x_mean = x_train.mean(axis=(0, 1), keepdims=True)
            y_mean = y_train.mean(axis=(0, 1), keepdims=True)
        xr = x_train - x_mean
        xh = x_held - x_mean
        yr = y_train - y_mean
        yh = y_held - y_mean
        direction = np.einsum("ikld,ik->ld", xr, yr)
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / np.maximum(norms, 1e-12)
        train_score = np.einsum("ikld,ld->ikl", xr, direction)
        numerator = np.einsum("ikl,ik->l", train_score, yr)
        denominator = np.einsum("ikl,ikl->l", train_score, train_score)
        slope = numerator / np.maximum(denominator, 1e-12)
        prediction = np.einsum("ikld,ld->ikl", xh, direction) * slope[None, None, :]
        held_targets.append(yh[0])
        held_predictions.append(prediction[0])
    targets = np.stack(held_targets)
    predictions = np.stack(held_predictions)  # item, cell, layer
    rows = []
    for layer in range(layers):
        y = targets.reshape(-1)
        pred = predictions[:, :, layer].reshape(-1)
        contributions = np.mean(targets * predictions[:, :, layer], axis=1)
        mean_covariance = float(contributions.mean())
        standard_error = float(contributions.std(ddof=1) / np.sqrt(n))
        t_value = mean_covariance / standard_error if standard_error > 0 else 0.0
        rows.append(
            {
                "layer": layer,
                "pearson": finite_corr(y, pred),
                "spearman": finite_corr(y, pred, "spearman"),
                "q2": float(1 - np.sum((y - pred) ** 2) / max(np.sum(y**2), 1e-12)),
                "mean_covariance": mean_covariance,
                "p_t_one": float(stats.t.sf(t_value, n - 1)),
            }
        )
    return pd.DataFrame(rows)


def transfer_evaluate(
    source: np.ndarray,
    destination: np.ndarray,
    target: np.ndarray,
    selected_layers: np.ndarray,
    *,
    bootstrap: int = 5000,
) -> dict:
    """Train a cell-centered direction in source and score destination."""
    targets, predictions, ratios = [], [], []
    n = len(source)
    for held in range(n):
        train = np.flatnonzero(np.arange(n) != held)
        layer = int(selected_layers[held])
        source_train = source[train, :, layer]
        destination_train = destination[train, :, layer]
        destination_held = destination[[held], :, layer]
        y_train = target[train]
        y_held = target[[held]]
        source_mean = source_train.mean(axis=0, keepdims=True)
        destination_mean = destination_train.mean(axis=0, keepdims=True)
        y_mean = y_train.mean(axis=0, keepdims=True)
        sx = source_train - source_mean
        dx = destination_held - destination_mean
        sy = y_train - y_mean
        dy = y_held - y_mean
        direction = unit(sx.reshape(-1, sx.shape[-1]).T @ sy.reshape(-1))
        train_score = sx.reshape(-1, sx.shape[-1]) @ direction
        denominator = float(train_score @ train_score)
        slope = float(train_score @ sy.reshape(-1) / denominator) if denominator > 1e-12 else 0.0
        prediction = (dx.reshape(-1, dx.shape[-1]) @ direction * slope).reshape(dy.shape)
        targets.append(dy[0]); predictions.append(prediction[0]); ratios.append(1.0)
    return summarize_predictions(np.stack(targets), np.stack(predictions), ratios, bootstrap=bootstrap)


def geometry_rows(
    array: np.ndarray,
    selected: np.ndarray,
    direction_sets: dict[str, list[np.ndarray]],
    gates: dict[str, bool],
) -> tuple[pd.DataFrame, dict]:
    delta = change_states(array, "delta_h3")[:, 0]
    rows = []
    for held in range(len(array)):
        train = np.flatnonzero(np.arange(len(array)) != held)
        layer = int(selected[held])
        v_r = unit(delta[train, layer].mean(axis=0))
        record: dict[str, float | int] = {"held_item": held, "layer": layer}
        valid = []
        for name, vectors in direction_sets.items():
            vector = unit(vectors[held])
            record[f"cos_R_{name}"] = float(v_r @ vector)
            if name in ("delta_P", "delta_C") and gates.get(name, False):
                valid.append(vector)
        remaining = residualize(v_r, valid)
        record["validated_span_size"] = len(valid)
        record["validated_unique_norm_ratio"] = float(np.linalg.norm(remaining))
        rows.append(record)
    frame = pd.DataFrame(rows)
    summary = {
        "validated_span_size": int(frame.validated_span_size.iloc[0]),
        "validated_unique_norm_ratio": float(frame.validated_unique_norm_ratio.mean()),
    }
    for column in [name for name in frame if name.startswith("cos_R_")]:
        values = frame[column].to_numpy()
        summary[column] = float(values.mean())
        summary[f"{column}_min"] = float(values.min())
        summary[f"{column}_max"] = float(values.max())
    return frame, summary


def add_identity(frame: pd.DataFrame, **values: object) -> pd.DataFrame:
    result = frame.copy()
    for key, value in reversed(list(values.items())):
        result.insert(0, key, value)
    return result


def _shown(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".3f")


def write_report(
    tier1: pd.DataFrame,
    tier2: pd.DataFrame,
    tier3: pd.DataFrame,
    format_transfer: pd.DataFrame,
    task_transfer: pd.DataFrame,
    target_layer_transfer: pd.DataFrame,
    geometry: pd.DataFrame,
    path: Path,
) -> None:
    t1 = tier1[(tier1.form == "bare") & (tier1.analysis == "item_deviation")]
    t2 = tier2[tier2.contrast == "delta_h3"]
    t3 = tier3[tier3.contrast == "delta_h3"]
    lines = [
        "# RQ3 hierarchical P/C reanalysis",
        "",
        "This is a diagnostic reanalysis fixed after the data were inspected. The inference unit is the 15 items; "
        "continuous item-LOIO prediction is used instead of a binary AUC. `gate` is Pearson r>0 and item exact "
        "sign-flip one-sided p<.05. No multi-cell correction, so model × task pass counts are descriptive.",
        "",
        "## Tier 1: total per-item P/C signal in the bare arm",
        "",
        f"Passed: {int(t1.gate.sum())}/{len(t1)} cells. `item_deviation` removes the prefix × hypothesis common "
        "mean within the training fold, excluding plain condition identification.",
        "",
        _shown(t1, ["model", "task", "target", "pearson", "pearson_boot_lo", "pearson_boot_hi", "q2", "p_flip_one", "gate"]),
        "",
        "### Difference between raw_total and item_deviation",
        "",
        _shown(tier1.groupby(["form", "analysis", "target"], as_index=False).agg(pearson=("pearson", "mean"), passed=("gate", "sum"), cells=("gate", "size")), ["form", "analysis", "target", "pearson", "passed", "cells"]),
        "",
        "## Tier 2: P/C change within the same conjunction environment",
        "",
        f"h3 R−I change passed: {int(t2.gate.sum())}/{len(t2)} cells.",
        "",
        _shown(t2, ["model", "task", "target", "pearson", "pearson_boot_lo", "pearson_boot_hi", "q2", "p_flip_one", "gate"]),
        "",
        "## Tier 3: unique change with the other construct removed",
        "",
        f"h3 unique change passed: {int(t3.gate.sum())}/{len(t3)} cells. A residual variance ratio <.20 means insufficient identification.",
        "",
        _shown(t3, ["model", "task", "target", "control", "pearson", "residual_variance_ratio", "p_flip_one", "gate"]),
        "",
        "## Tier 4: format and task transfer",
        "",
        _shown(format_transfer.groupby(["source_form", "destination_form", "target"], as_index=False).agg(pearson=("pearson", "mean"), passed=("gate", "sum"), cells=("gate", "size")), ["source_form", "destination_form", "target", "pearson", "passed", "cells"]),
        "",
        _shown(task_transfer.groupby(["source_task", "destination_task", "form", "target"], as_index=False).agg(pearson=("pearson", "mean"), passed=("gate", "sum"), cells=("gate", "size")), ["source_task", "destination_task", "form", "target", "pearson", "passed", "cells"]),
        "",
        "### Choose the P/C layer in one task and validate in the other",
        "",
        "`layer_only` picks only the layer from the source task and learns the direction anew by LOIO in the destination task. "
        "`direction_and_layer` learns the direction in the source task as well and applies it to the destination. The two tasks "
        "share the same items and norming targets, so this is not a fully independent replication.",
        "",
        _shown(target_layer_transfer.groupby(["tier", "family", "target", "mode"], as_index=False).agg(pearson=("pearson", "mean"), passed=("gate", "sum"), cells=("gate", "size")), ["tier", "family", "target", "mode", "pearson", "passed", "cells"]),
        "",
        "## Tier 5: H3b geometry with the detection gate applied",
        "",
        _shown(geometry, ["model", "task", "delta_P_gate", "delta_C_gate", "validated_span_size", "validated_unique_norm_ratio", "cos_R_delta_P", "cos_R_delta_C"]),
        "",
        "`validated_unique_norm_ratio` is the norm ratio of v_R after removing only those P/C directions of the same "
        "conjunction h3 change that passed the self-target gate. Cosines with directions that did not pass are "
        "reported but not read as evidence of orthogonality.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    root = args.input_dir
    out = root / "analysis" / "pc_hierarchy"
    out.mkdir(parents=True, exist_ok=True)
    decoder = load_h3_decoder_gate(
        root / "analysis" / "rq3a_context_decoder_robustness.csv",
        require_both_tasks=True,
    )
    decoder = decoder[decoder.decode_gate.astype(bool)].set_index(["model_id", "task"])

    tier1_rows: list[dict] = []
    tier2_rows: list[dict] = []
    tier3_rows: list[dict] = []
    format_rows: list[dict] = []
    task_rows: list[dict] = []
    target_layer_rows: list[dict] = []
    layer_tables: list[pd.DataFrame] = []
    fold_tables: list[pd.DataFrame] = []
    geometry_summaries: list[dict] = []
    geometry_folds: list[pd.DataFrame] = []

    for manifest_path in sorted((root / "manifests").glob("representation_*.json")):
        manifest = json.loads(manifest_path.read_text())
        model_id = manifest["model"]
        model = model_id.split("/")[-1]
        slug = manifest_path.stem.removeprefix("representation_")
        norming = pd.read_csv(root / "norming" / f"norm_rq3_{slug}.csv")
        task_data: dict[int, dict] = {}
        for task in (2, 3):
            key = (model_id, task)
            if key not in decoder.index:
                continue
            row = decoder.loc[key]
            selected = np.asarray(json.loads(row.selected_layers), dtype=int)
            array = np.load(root / "activations" / slug / f"task{task}_cellmean.npy", mmap_mode="r")
            metadata = json.loads((root / "activations" / slug / f"task{task}_metadata.json").read_text())
            p_values, c_values = norm_arrays(norming, metadata["labels"])
            targets = {"P": p_values, "C": c_values}
            states = {form: format_states(array, form) for form in ("bare", "conjunction")}
            flat_targets = {name: values.reshape(len(values), 6) for name, values in targets.items()}
            task_data[task] = {
                "array": array,
                "selected": selected,
                "states": states,
                "targets": targets,
                "flat_targets": flat_targets,
                "curves": {},
            }

            direction_sets: dict[str, list[np.ndarray]] = {}
            direction_gates: dict[str, bool] = {}
            # Tier 1: both forms are retained so raw vs cell-deviation and transfer can be compared.
            for form in ("bare", "conjunction"):
                for target_name in ("P", "C"):
                    for analysis, cell_center in (("raw_total", False), ("item_deviation", True)):
                        summary, folds, vectors = loio_evaluate(
                            states[form], flat_targets[target_name], selected,
                            cell_center=cell_center, bootstrap=args.bootstrap,
                        )
                        identity = {"model_id": model_id, "model": model, "task": task, "form": form, "analysis": analysis, "target": target_name}
                        tier1_rows.append({**identity, **summary})
                        fold_tables.append(add_identity(folds, tier="tier1", **identity))
                        if analysis == "item_deviation":
                            direction_sets[f"{form}_{target_name}"] = vectors

            # Tier 2 and 3: same-environment changes.
            for contrast in ("delta_h3", "did"):
                x_change = change_states(array, contrast)
                change_targets = {name: change_target(values, contrast) for name, values in targets.items()}
                for target_name in ("P", "C"):
                    summary, folds, vectors = loio_evaluate(
                        x_change, change_targets[target_name], selected,
                        cell_center=False, bootstrap=args.bootstrap,
                    )
                    identity = {"model_id": model_id, "model": model, "task": task, "contrast": contrast, "target": target_name}
                    tier2_rows.append({**identity, **summary})
                    fold_tables.append(add_identity(folds, tier="tier2", **identity))
                    if contrast == "delta_h3":
                        direction_sets[f"delta_{target_name}"] = vectors
                        direction_gates[f"delta_{target_name}"] = bool(summary["gate"])

                    control_name = "C" if target_name == "P" else "P"
                    unique, unique_folds, unique_vectors = loio_evaluate(
                        x_change, change_targets[target_name], selected,
                        cell_center=False, control=change_targets[control_name],
                        bootstrap=args.bootstrap,
                    )
                    unique_identity = {**identity, "control": control_name}
                    tier3_rows.append({**unique_identity, **unique})
                    fold_tables.append(add_identity(unique_folds, tier="tier3", **unique_identity))
                    if contrast == "delta_h3":
                        direction_sets[f"unique_{target_name}"] = unique_vectors

            # Fixed all-layer curves: no best-layer number is used as a same-task confirmatory result.
            for form in ("bare", "conjunction"):
                for target_name in ("P", "C"):
                    curve = all_layer_curve(states[form], flat_targets[target_name], cell_center=True)
                    task_data[task]["curves"][("tier1", form, target_name)] = curve
                    layer_tables.append(add_identity(curve, model_id=model_id, model=model, task=task, tier="tier1", family=form, target=target_name))
            for target_name in ("P", "C"):
                curve = all_layer_curve(change_states(array, "delta_h3"), change_target(targets[target_name], "delta_h3"), cell_center=False)
                task_data[task]["curves"][("tier2", "delta_h3", target_name)] = curve
                layer_tables.append(add_identity(curve, model_id=model_id, model=model, task=task, tier="tier2", family="delta_h3", target=target_name))

            # Within-task format transfer at the same R-selected layers.
            for source_form in ("bare", "conjunction"):
                destination_form = "conjunction" if source_form == "bare" else "bare"
                for target_name in ("P", "C"):
                    result = transfer_evaluate(
                        states[source_form], states[destination_form], flat_targets[target_name], selected,
                        bootstrap=args.bootstrap,
                    )
                    format_rows.append({"model_id": model_id, "model": model, "task": task, "source_form": source_form, "destination_form": destination_form, "target": target_name, **result})

            folds, geometry = geometry_rows(array, selected, direction_sets, direction_gates)
            geometry_folds.append(add_identity(folds, model_id=model_id, model=model, task=task))
            geometry_summaries.append(
                {
                    "model_id": model_id,
                    "model": model,
                    "task": task,
                    "h3_decoder_auc": float(row.auc_nested_loio),
                    "selected_layer_mode": Counter(selected.tolist()).most_common(1)[0][0],
                    "delta_P_gate": direction_gates.get("delta_P", False),
                    "delta_C_gate": direction_gates.get("delta_C", False),
                    **geometry,
                }
            )
            print(model_id, task, "tiers 1-3/geometry complete", flush=True)

        # Cross-task direction transfer, evaluated on the destination task's held item.
        if set(task_data) == {2, 3}:
            for source_task, destination_task in ((2, 3), (3, 2)):
                source = task_data[source_task]
                destination = task_data[destination_task]
                for form in ("bare", "conjunction"):
                    for target_name in ("P", "C"):
                        result = transfer_evaluate(
                            source["states"][form], destination["states"][form],
                            source["flat_targets"][target_name], source["selected"],
                            bootstrap=args.bootstrap,
                        )
                        task_rows.append({"model_id": model_id, "model": model, "source_task": source_task, "destination_task": destination_task, "form": form, "target": target_name, **result})

                # Target-specific layer selection is allowed only across tasks.  We report both
                # destination-trained and fully transferred directions at that source-selected layer.
                families = [
                    ("tier1", "bare", target_name) for target_name in ("P", "C")
                ] + [
                    ("tier1", "conjunction", target_name) for target_name in ("P", "C")
                ] + [
                    ("tier2", "delta_h3", target_name) for target_name in ("P", "C")
                ]
                for tier, family, target_name in families:
                    source_curve = source["curves"][(tier, family, target_name)]
                    finite = source_curve[np.isfinite(source_curve.pearson)]
                    selected_layer = int(finite.loc[finite.pearson.idxmax(), "layer"])
                    fixed = np.full(15, selected_layer, dtype=int)
                    if tier == "tier1":
                        source_states = source["states"][family]
                        destination_states = destination["states"][family]
                        source_target = source["flat_targets"][target_name]
                        destination_target = destination["flat_targets"][target_name]
                        cell_center = True
                    else:
                        source_states = change_states(source["array"], family)
                        destination_states = change_states(destination["array"], family)
                        source_target = change_target(source["targets"][target_name], family)
                        destination_target = change_target(destination["targets"][target_name], family)
                        cell_center = False
                    destination_result, _, _ = loio_evaluate(
                        destination_states, destination_target, fixed,
                        cell_center=cell_center, bootstrap=args.bootstrap,
                    )
                    transferred_result = transfer_evaluate(
                        source_states, destination_states, source_target, fixed,
                        bootstrap=args.bootstrap,
                    )
                    common = {
                        "model_id": model_id,
                        "model": model,
                        "tier": tier,
                        "family": family,
                        "target": target_name,
                        "source_task": source_task,
                        "destination_task": destination_task,
                        "selected_layer": selected_layer,
                        "source_selection_pearson": float(source_curve.loc[source_curve.layer == selected_layer, "pearson"].iloc[0]),
                    }
                    target_layer_rows.append({**common, "mode": "layer_only", **destination_result})
                    target_layer_rows.append({**common, "mode": "direction_and_layer", **transferred_result})
        print(model_id, "task transfer complete", flush=True)

    tier1 = pd.DataFrame(tier1_rows)
    tier2 = pd.DataFrame(tier2_rows)
    tier3 = pd.DataFrame(tier3_rows)
    format_transfer = pd.DataFrame(format_rows)
    task_transfer = pd.DataFrame(task_rows)
    target_layer_transfer = pd.DataFrame(target_layer_rows)
    layer_curves = pd.concat(layer_tables, ignore_index=True)
    folds = pd.concat(fold_tables, ignore_index=True)
    geometry = pd.DataFrame(geometry_summaries)
    geometry_fold_frame = pd.concat(geometry_folds, ignore_index=True)

    # Cross-task layer selection is descriptive and explicitly keeps selection and evaluation tasks separate.
    layer_transfer_rows = []
    for (model_id, tier, family, target), group in layer_curves.groupby(["model_id", "tier", "family", "target"]):
        if set(group.task) != {2, 3}:
            continue
        for source_task, destination_task in ((2, 3), (3, 2)):
            source = group[group.task == source_task]
            destination = group[group.task == destination_task].set_index("layer")
            best = int(source.loc[source.pearson.idxmax(), "layer"])
            layer_transfer_rows.append({
                "model_id": model_id,
                "model": model_id.split("/")[-1],
                "tier": tier,
                "family": family,
                "target": target,
                "source_task": source_task,
                "destination_task": destination_task,
                "selected_layer": best,
                "source_pearson": float(source.loc[source.layer == best, "pearson"].iloc[0]),
                "destination_pearson": float(destination.loc[best, "pearson"]),
            })
    layer_transfer = pd.DataFrame(layer_transfer_rows)

    outputs = {
        "tier1_total.csv": tier1,
        "tier2_change.csv": tier2,
        "tier3_unique.csv": tier3,
        "tier4_format_transfer.csv": format_transfer,
        "tier4_task_transfer.csv": task_transfer,
        "tier4_target_layer_transfer.csv": target_layer_transfer,
        "layer_curves.csv": layer_curves,
        "layer_transfer.csv": layer_transfer,
        "tier5_geometry.csv": geometry,
        "tier5_geometry_folds.csv": geometry_fold_frame,
        "fold_predictions.csv": folds,
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    write_report(tier1, tier2, tier3, format_transfer, task_transfer, target_layer_transfer, geometry, out / "PC_HIERARCHY_REPORT.md")
    (out / "analysis_manifest.json").write_text(json.dumps({
        "status": "complete",
        "seed": SEED,
        "items": 15,
        "models": int(tier1.model_id.nunique()),
        "tasks": [2, 3],
        "bootstrap": args.bootstrap,
        "multiple_testing_adjustment": None,
        "plan": "EXPERIMENT_RQ3_PC_REANALYSIS.md",
        "positive_control": "not_run_existing_activations_already_pass_continuous_cross_task_checks",
        "outputs": list(outputs) + ["PC_HIERARCHY_REPORT.md", "POSITIVE_CONTROL_DECISION.md"],
    }, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
