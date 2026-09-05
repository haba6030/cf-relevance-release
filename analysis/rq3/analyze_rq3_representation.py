#!/usr/bin/env python3
"""RQ3-A analysis: LOIO decoding (D) and P/C/R geometry (G)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 20260824
LAMBDA_GRID = np.logspace(-8, 4, 13)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else np.zeros_like(vector)


def auc_for_item(states: np.ndarray, direction: np.ndarray) -> tuple[list[int], list[float]]:
    """Retired one-versus-three rank labelling. Not used for new outputs.

    It takes `R,h3` alone as positive and pools `R,h2`/`I,h3`/`I,h2` as negatives.
    It mixes the prefix contrast (R vs I) and the conjunct contrast (h3 vs h2) in one ranking problem and
    is coarse with only four ranks per item. The canonical construction is the conjunct-fixed binary decoder,
    `family == "conjunction_h3"` in `analyze_rq3_context_decoding_robustness.py`.
    Assets built with this function live in `archive/rq3_old_label_20260825/` (not released).
    """
    # states: [prefix N/I/R, option base/c2/c3/b2/b3, d]
    # Positive R,c3; negatives R,c2 and I,c3/I,c2.
    pi = {"N": 0, "I": 1, "R": 2}
    oi = {"base": 0, "c2": 1, "c3": 2, "b2": 3, "b3": 4}
    cells = [("R", "c3"), ("R", "c2"), ("I", "c3"), ("I", "c2")]
    scores = [float(states[pi[p], oi[o]] @ direction) for p, o in cells]
    return [1, 0, 0, 0], scores


def item_did(array: np.ndarray) -> np.ndarray:
    r"""Old difference-in-differences direction. Not the canonical direction; kept for comparison only.

    $a_{R,h_3}-a_{I,h_3}-a_{R,h_2}+a_{I,h_2}$, the design paired with the one-versus-three labelling above.
    The canonical direction is `run_rq3_intervention_gpu.context_direction(array, train, (2,))`, i.e.
    $\overline{a_{R,h_3}-a_{I,h_3}}$.
    """
    # [item, hidden_state, dimension]
    return (
        array[:, 2, 2] - array[:, 1, 2]
        - array[:, 2, 1] + array[:, 1, 1]
    )


def contrast_definition(name: str) -> tuple[int, int, int, int]:
    definitions = {
        "primary_RI_conjunction": (2, 1, 2, 1),
        "RN_conjunction": (2, 0, 2, 1),
        "RI_bare": (2, 1, 4, 3),
    }
    return definitions[name]


def item_contrast(array: np.ndarray, name: str) -> np.ndarray:
    high, low, positive, negative = contrast_definition(name)
    return array[:, high, positive] - array[:, low, positive] - array[:, high, negative] + array[:, low, negative]


def auc_for_variant(states: np.ndarray, direction: np.ndarray, name: str) -> tuple[list[int], list[float]]:
    """Variant of the retired one-versus-three labelling. Not canonical."""
    high, low, positive, negative = contrast_definition(name)
    cells = [(high, positive), (high, negative), (low, positive), (low, negative)]
    return [1, 0, 0, 0], [float(states[prefix, option] @ direction) for prefix, option in cells]


def nested_variant_selection(array: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    contrast = item_contrast(array, name)
    n, layers, _ = contrast.shape
    selected, all_labels, all_scores = [], [], []
    for outer in range(n):
        train = [index for index in range(n) if index != outer]
        inner_auc = []
        for layer in range(layers):
            y_layer, s_layer = [], []
            total = contrast[train, layer].sum(axis=0)
            for inner in train:
                direction = unit((total - contrast[inner, layer]) / (n - 2))
                y, scores = auc_for_variant(array[inner, :, :, layer], direction, name)
                y_layer.extend(y); s_layer.extend(scores)
            inner_auc.append(roc_auc_score(y_layer, s_layer))
        layer = int(np.argmax(inner_auc)); selected.append(layer)
        direction = unit(contrast[train, layer].mean(axis=0))
        y, scores = auc_for_variant(array[outer, :, :, layer], direction, name)
        all_labels.extend(y); all_scores.extend(scores)
    return np.asarray(selected), float(roc_auc_score(all_labels, all_scores))


def nested_prompt_selection(array: np.ndarray) -> tuple[np.ndarray, float]:
    """LOIO R-vs-I decode at the preregistered prompt-final read position."""
    contrast = array[:, 2] - array[:, 1]
    n, layers, _ = contrast.shape
    selected, labels, scores = [], [], []
    for outer in range(n):
        train = [index for index in range(n) if index != outer]
        inner_auc = []
        for layer in range(layers):
            y_inner, s_inner = [], []
            total = contrast[train, layer].sum(axis=0)
            for inner in train:
                direction = unit((total - contrast[inner, layer]) / (n - 2))
                y_inner.extend([1, 0])
                s_inner.extend([float(array[inner, 2, layer] @ direction), float(array[inner, 1, layer] @ direction)])
            inner_auc.append(roc_auc_score(y_inner, s_inner))
        layer = int(np.argmax(inner_auc)); selected.append(layer)
        direction = unit(contrast[train, layer].mean(axis=0))
        labels.extend([1, 0]); scores.extend([float(array[outer, 2, layer] @ direction), float(array[outer, 1, layer] @ direction)])
    return np.asarray(selected), float(roc_auc_score(labels, scores))


def loio_layer_curve(array: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    did = item_did(array)
    n, layers, _ = did.shape
    directions = np.empty_like(did)
    rows = []
    for layer in range(layers):
        labels, scores = [], []
        total = did[:, layer].sum(axis=0)
        for held in range(n):
            direction = unit((total - did[held, layer]) / (n - 1))
            directions[held, layer] = direction
            y, s = auc_for_item(array[held, :, :, layer], direction)
            labels.extend(y)
            scores.extend(s)
        rows.append({"layer": layer, "auc_loio": roc_auc_score(labels, scores)})
    return pd.DataFrame(rows), directions


def nested_layer_selection(array: np.ndarray) -> tuple[np.ndarray, float, list[int], list[float]]:
    did = item_did(array)
    n, layers, _ = did.shape
    selected, all_labels, all_scores = [], [], []
    for outer in range(n):
        train = [index for index in range(n) if index != outer]
        inner_auc = []
        for layer in range(layers):
            y_layer, s_layer = [], []
            train_total = did[train, layer].sum(axis=0)
            for inner in train:
                direction = unit((train_total - did[inner, layer]) / (n - 2))
                y, scores = auc_for_item(array[inner, :, :, layer], direction)
                y_layer.extend(y)
                s_layer.extend(scores)
            inner_auc.append(roc_auc_score(y_layer, s_layer))
        layer_star = int(np.argmax(inner_auc))
        selected.append(layer_star)
        direction = unit(did[train, layer_star].mean(axis=0))
        y, scores = auc_for_item(array[outer, :, :, layer_star], direction)
        all_labels.extend(y)
        all_scores.extend(scores)
    return np.asarray(selected), roc_auc_score(all_labels, all_scores), all_labels, all_scores


def decoding_nulls(
    array: np.ndarray,
    selected: np.ndarray,
    observed_auc: float,
    random_n: int = 1000,
    permutation_n: int = 200,
) -> dict:
    rng = np.random.default_rng(SEED)
    did = item_did(array)
    n, _, dimension = did.shape
    random_auc = []
    for _ in range(random_n):
        labels, scores = [], []
        for held in range(n):
            direction = unit(rng.normal(size=dimension))
            y, s = auc_for_item(array[held, :, :, selected[held]], direction)
            labels.extend(y)
            scores.extend(s)
        random_auc.append(roc_auc_score(labels, scores))
    permutation_auc = []
    for _ in range(permutation_n):
        labels, scores = [], []
        # Registered null: shuffle N/I/R context labels independently within
        # each item, then rebuild the complete DiD.  This retains all option
        # and item structure; it is not merely a global sign flip.
        permutations = np.stack([rng.permutation(3) for _ in range(n)])
        permuted_did = np.empty_like(did)
        for item in range(n):
            shuffled = array[item, permutations[item]]
            permuted_did[item] = (
                shuffled[2, 2] - shuffled[1, 2]
                - shuffled[2, 1] + shuffled[1, 1]
            )
        for held in range(n):
            train = np.arange(n) != held
            layer = selected[held]
            direction = unit(permuted_did[train, layer].mean(axis=0))
            y, s = auc_for_item(array[held, :, :, layer], direction)
            labels.extend(y)
            scores.extend(s)
        permutation_auc.append(roc_auc_score(labels, scores))
    random_auc = np.asarray(random_auc)
    permutation_auc = np.asarray(permutation_auc)
    return {
        "random_n": random_n,
        "random_p": float((1 + (random_auc >= observed_auc).sum()) / (random_n + 1)),
        "random_95": float(np.quantile(random_auc, 0.95)),
        "random_mean": float(random_auc.mean()),
        "permutation_n": permutation_n,
        "permutation_p": float((1 + (permutation_auc >= observed_auc).sum()) / (permutation_n + 1)),
        "permutation_95": float(np.quantile(permutation_auc, 0.95)),
        "permutation_mean": float(permutation_auc.mean()),
    }


def norm_arrays(norming: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    averaged = norming.groupby(["label", "prefix", "hypothesis"])[["rho", "C"]].mean()
    rho = np.empty((len(labels), 3, 2), dtype=float)
    confirmation = np.empty_like(rho)
    for item, label in enumerate(labels):
        for prefix, p in zip(("N", "I", "R"), range(3)):
            for hypothesis, h in zip(("h2", "h3"), range(2)):
                row = averaged.loc[(label, prefix, hypothesis)]
                rho[item, p, h] = row.rho
                confirmation[item, p, h] = row.C
    return rho, confirmation


def centered_design(rho: np.ndarray, confirmation: np.ndarray, items: np.ndarray) -> np.ndarray:
    blocks = []
    for item in items:
        rows = []
        for prefix in range(3):
            for hypothesis in range(2):
                is_i = float(prefix == 1)
                is_r = float(prefix == 2)
                is_c3 = float(hypothesis == 1)
                rows.append([is_i, is_r, is_c3, is_r * is_c3, rho[item, prefix, hypothesis], confirmation[item, prefix, hypothesis]])
        block = np.asarray(rows, dtype=float)
        blocks.append(block - block.mean(axis=0, keepdims=True))
    return np.concatenate(blocks, axis=0)


def centered_activation(array: np.ndarray, items: np.ndarray, layer: int) -> np.ndarray:
    blocks = []
    for item in items:
        # Prefix-major and then hypothesis c2/c3, matching centered_design.
        block = array[item, :, :, layer][:, [1, 2], :].reshape(6, -1)
        blocks.append(block - block.mean(axis=0, keepdims=True))
    return np.concatenate(blocks, axis=0)


def ridge_gcv(x: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, float]:
    xtx = x.T @ x
    xta = x.T @ a
    a2 = float(np.sum(a * a))
    best = None
    for lam in LAMBDA_GRID:
        inverse = np.linalg.inv(xtx + lam * np.eye(xtx.shape[0]))
        coefficient = inverse @ xta
        sse = a2 - 2 * float(np.sum(coefficient * xta)) + float(np.sum(coefficient * (xtx @ coefficient)))
        edf = float(np.trace(xtx @ inverse))
        gcv = sse / max((len(x) - edf) ** 2, 1e-12)
        if best is None or gcv < best[0]:
            best = (gcv, coefficient, float(lam))
    assert best is not None
    return best[1], best[2]


def reverse_ridge_gcv(a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit activation -> scalar target in the dual, choosing lambda by GCV."""
    kernel = a @ a.T
    eigenvalues = np.linalg.eigvalsh(kernel)
    scale = max(float(np.trace(kernel) / max(len(kernel), 1)), 1e-12)
    identity = np.eye(len(kernel))
    best = None
    for multiplier in LAMBDA_GRID:
        lam = float(multiplier * scale)
        dual = np.linalg.solve(kernel + lam * identity, y)
        fitted = kernel @ dual
        sse = float(np.sum((y - fitted) ** 2))
        edf = float(np.sum(eigenvalues / (eigenvalues + lam)))
        gcv = sse / max((len(y) - edf) ** 2, 1e-12)
        if best is None or gcv < best[0]:
            best = (gcv, a.T @ dual, lam)
    assert best is not None
    return best[1], best[2]


def finite_corr(left: np.ndarray, right: np.ndarray, method: str = "pearson") -> float:
    value = pd.Series(left).corr(pd.Series(right), method=method)
    return float(value) if np.isfinite(value) else np.nan


def pc_direction_robustness(
    array: np.ndarray,
    rho: np.ndarray,
    confirmation: np.ndarray,
    selected: np.ndarray,
) -> pd.DataFrame:
    """LOIO reverse-regression and high/low-tertile robustness estimates."""
    n = len(array)
    output = []
    for target_name, target_column in (("P", 4), ("C", 5)):
        held_targets, reverse_scores, tertile_scores = [], [], []
        cos_reverse, cos_tertile, lambdas = [], [], []
        for held in range(n):
            train = np.asarray([index for index in range(n) if index != held])
            layer = int(selected[held])
            x_train = centered_design(rho, confirmation, train)
            a_train = centered_activation(array, train, layer)
            x_held = centered_design(rho, confirmation, np.asarray([held]))
            a_held = centered_activation(array, np.asarray([held]), layer)
            nuisance_columns = [index for index in range(x_train.shape[1]) if index != target_column]
            nuisance_train = x_train[:, nuisance_columns]
            nuisance_held = x_held[:, nuisance_columns]

            target_nuisance = np.linalg.lstsq(
                nuisance_train, x_train[:, target_column], rcond=None
            )[0]
            activation_nuisance = np.linalg.lstsq(nuisance_train, a_train, rcond=None)[0]
            y_train = x_train[:, target_column] - nuisance_train @ target_nuisance
            y_held = x_held[:, target_column] - nuisance_held @ target_nuisance
            residual_train = a_train - nuisance_train @ activation_nuisance
            residual_held = a_held - nuisance_held @ activation_nuisance

            reverse, lam = reverse_ridge_gcv(residual_train, y_train)
            low, high = np.quantile(y_train, (1 / 3, 2 / 3))
            low_rows = residual_train[y_train <= low]
            high_rows = residual_train[y_train >= high]
            if not len(low_rows) or not len(high_rows):
                raise RuntimeError("Tertile direction has an empty high/low group")
            tertile = high_rows.mean(axis=0) - low_rows.mean(axis=0)
            primary, _ = ridge_gcv(x_train, a_train)
            primary_direction = primary[target_column]

            held_targets.extend(y_held.tolist())
            reverse_scores.extend((residual_held @ unit(reverse)).tolist())
            tertile_scores.extend((residual_held @ unit(tertile)).tolist())
            cos_reverse.append(float(unit(primary_direction) @ unit(reverse)))
            cos_tertile.append(float(unit(primary_direction) @ unit(tertile)))
            lambdas.append(lam)

        target_values = np.asarray(held_targets)
        reverse_values = np.asarray(reverse_scores)
        tertile_values = np.asarray(tertile_scores)
        output.append(
            {
                "target": target_name,
                "n_held_rows": len(target_values),
                "reverse_pearson": finite_corr(target_values, reverse_values),
                "reverse_spearman": finite_corr(target_values, reverse_values, "spearman"),
                "tertile_pearson": finite_corr(target_values, tertile_values),
                "tertile_spearman": finite_corr(target_values, tertile_values, "spearman"),
                "primary_reverse_cos_mean": float(np.mean(cos_reverse)),
                "primary_reverse_cos_min": float(np.min(cos_reverse)),
                "primary_reverse_cos_max": float(np.max(cos_reverse)),
                "primary_tertile_cos_mean": float(np.mean(cos_tertile)),
                "primary_tertile_cos_min": float(np.min(cos_tertile)),
                "primary_tertile_cos_max": float(np.max(cos_tertile)),
                "reverse_lambda_median": float(np.median(lambdas)),
            }
        )
    return pd.DataFrame(output)


def diagnostic_table(rho: np.ndarray, confirmation: np.ndarray) -> dict:
    items = np.arange(rho.shape[0])
    x = centered_design(rho, confirmation, items)
    corr = float(np.corrcoef(x[:, 4], x[:, 5])[0, 1])
    condition = float(np.linalg.cond(x))
    # Residual variance after prefix/hypothesis/relatedness design columns.
    base = x[:, :4]
    residual_ratios = []
    vifs = []
    for target_index in (4, 5):
        target = x[:, target_index]
        residual = target - base @ np.linalg.lstsq(base, target, rcond=None)[0]
        residual_ratios.append(float(np.var(residual) / np.var(target)))
        other = np.delete(x, target_index, axis=1)
        fitted = other @ np.linalg.lstsq(other, target, rcond=None)[0]
        r2 = 1 - np.sum((target - fitted) ** 2) / np.sum((target - target.mean()) ** 2)
        vifs.append(float(1 / max(1 - r2, 1e-12)))
    return {
        "corr_rho_C": corr,
        "condition_number": condition,
        "rho_residual_variance_ratio": residual_ratios[0],
        "C_residual_variance_ratio": residual_ratios[1],
        "rho_vif": vifs[0],
        "C_vif": vifs[1],
    }


def cross_decoding(array: np.ndarray, rho: np.ndarray, confirmation: np.ndarray, selected: np.ndarray) -> pd.DataFrame:
    """LOIO 4x4 cross-decoding matrix at fold-selected layers."""
    direction_names = ("R", "R_unique", "P", "C")
    target_names = ("R", "R_unique", "P", "C")
    labels = {target: [] for target in target_names}
    scores = {(direction, target): [] for direction in direction_names for target in target_names}
    did = item_did(array); n = len(array)
    for held in range(n):
        train = np.asarray([index for index in range(n) if index != held]); layer = int(selected[held])
        coefficient, _ = ridge_gcv(centered_design(rho, confirmation, train), centered_activation(array, train, layer))
        directions = {"R": did[train, layer].mean(axis=0), "R_unique": coefficient[3], "P": coefficient[4], "C": coefficient[5]}
        block = array[held, :, :, layer]
        states = block[1:3][:, [1, 2], :].reshape(4, -1)
        r_target = np.asarray([0, 0, 0, 1], dtype=int)
        p_values = rho[held, 1:3].reshape(-1); c_values = confirmation[held, 1:3].reshape(-1)
        p_cut = float(np.median(rho[train, 1:3])); c_cut = float(np.median(confirmation[train, 1:3]))
        targets = {"R": r_target, "R_unique": r_target, "P": (p_values > p_cut).astype(int), "C": (c_values > c_cut).astype(int)}
        for target, values in targets.items():
            labels[target].extend(values.tolist())
            for direction, vector in directions.items():
                scores[(direction, target)].extend((states @ unit(vector)).tolist())
    rows = []
    for target in target_names:
        y = np.asarray(labels[target])
        for direction in direction_names:
            value = roc_auc_score(y, scores[(direction, target)]) if len(np.unique(y)) == 2 else np.nan
            rows.append({"direction": direction, "target": target, "auc": value, "n": len(y), "positive": int(y.sum())})
    return pd.DataFrame(rows)


def evidence_decode_gates(
    array: np.ndarray,
    rho: np.ndarray,
    confirmation: np.ndarray,
    selected: np.ndarray,
    random_n: int = 1000,
) -> dict:
    """LOIO self-target gates for v_P and v_C used by RQ3 §6.1."""
    rng = np.random.default_rng(SEED)
    labels = {"P": [], "C": []}
    observed_scores = {"P": [], "C": []}
    held_states = []
    n = len(array)
    for held in range(n):
        train = np.asarray([index for index in range(n) if index != held])
        layer = int(selected[held])
        coefficient, _ = ridge_gcv(
            centered_design(rho, confirmation, train),
            centered_activation(array, train, layer),
        )
        states = array[held, :, :, layer][1:3][:, [1, 2], :].reshape(4, -1)
        held_states.append(states)
        for name, values, column in (
            ("P", rho, 4),
            ("C", confirmation, 5),
        ):
            held_values = values[held, 1:3].reshape(-1)
            threshold = float(np.median(values[train, 1:3]))
            labels[name].extend((held_values > threshold).astype(int).tolist())
            observed_scores[name].extend((states @ unit(coefficient[column])).tolist())

    output = {}
    dimension = array.shape[-1]
    for name in ("P", "C"):
        y = np.asarray(labels[name])
        observed = roc_auc_score(y, observed_scores[name]) if len(np.unique(y)) == 2 else np.nan
        null = []
        for _ in range(random_n):
            scores = []
            for states in held_states:
                scores.extend((states @ unit(rng.normal(size=dimension))).tolist())
            null.append(roc_auc_score(y, scores))
        null = np.asarray(null)
        threshold = float(np.quantile(null, .95))
        output[f"{name}_decode_auc"] = float(observed)
        output[f"{name}_decode_random_95"] = threshold
        output[f"{name}_decode_random_p"] = float((1 + np.sum(null >= observed)) / (random_n + 1))
        output[f"{name}_decode_gate"] = bool(np.isfinite(observed) and observed > .5 and observed > threshold)
    return output


def geometry(
    array: np.ndarray,
    rho: np.ndarray,
    confirmation: np.ndarray,
    selected: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    did = item_did(array)
    n, layers, _ = did.shape
    rows = []
    selected_rows = []
    for held in range(n):
        train = np.asarray([i for i in range(n) if i != held])
        x = centered_design(rho, confirmation, train)
        for layer in range(layers):
            a = centered_activation(array, train, layer)
            coefficient, lam = ridge_gcv(x, a)
            v_r = unit(did[train, layer].mean(axis=0))
            v_unique = coefficient[3]
            v_p = coefficient[4]
            v_c = coefficient[5]
            record = {
                "held_item": held,
                "layer": layer,
                "lambda": lam,
                "cos_R_P": float(unit(v_r) @ unit(v_p)),
                "cos_R_C": float(unit(v_r) @ unit(v_c)),
                "cos_P_C": float(unit(v_p) @ unit(v_c)),
                "unique_norm_ratio": float(np.linalg.norm(v_unique) / max(np.linalg.norm(v_r), 1e-12)),
            }
            rows.append(record)
            if layer == selected[held]:
                selected_rows.append(record)
    selected_frame = pd.DataFrame(selected_rows)
    summary = {}
    rng = np.random.default_rng(SEED)
    for column in ("cos_R_P", "cos_R_C", "cos_P_C", "unique_norm_ratio"):
        values = selected_frame[column].to_numpy()
        boot = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        summary[column] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max()), "boot_lo": float(np.quantile(boot, .025)), "boot_hi": float(np.quantile(boot, .975))}

    dimension = array.shape[-1]
    left = rng.normal(size=(1000, dimension)); right = rng.normal(size=(1000, dimension))
    left /= np.linalg.norm(left, axis=1, keepdims=True); right /= np.linalg.norm(right, axis=1, keepdims=True)
    summary["random_abs_cos_95"] = float(np.quantile(np.abs(np.sum(left * right, axis=1)), .95))

    pr_values = []
    for held in range(n):
        train = np.asarray([index for index in range(n) if index != held])
        layer = int(selected[held])
        matrix = did[train, layer] - did[train, layer].mean(axis=0, keepdims=True)
        singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        eigen = singular**2 / max(len(train) - 1, 1)
        pr_values.append(float(eigen.sum() ** 2 / max(np.sum(eigen**2), 1e-12)))
    pr_values = np.asarray(pr_values)
    pr_boot = rng.choice(pr_values, size=(2000, len(pr_values)), replace=True).mean(axis=1)
    summary["participation_ratio"] = {
        "mean": float(pr_values.mean()),
        "sd": float(pr_values.std(ddof=1)),
        "min": float(pr_values.min()),
        "max": float(pr_values.max()),
        "boot_lo": float(np.quantile(pr_boot, .025)),
        "boot_hi": float(np.quantile(pr_boot, .975)),
    }
    summary["participation_ratio_layer_mode"] = int(Counter(selected).most_common(1)[0][0])
    return pd.DataFrame(rows), summary


def geometry_shuffle_null(
    array: np.ndarray,
    rho: np.ndarray,
    confirmation: np.ndarray,
    selected: np.ndarray,
    observed: dict,
    permutations: int = 200,
) -> dict:
    """Within-item/context and across-item target shuffles for cosine nulls."""
    rng = np.random.default_rng(SEED); n = len(array)
    null = {"cos_R_P": [], "cos_R_C": [], "cos_P_C": []}
    for _ in range(permutations):
        prefix_permutations = np.stack([rng.permutation(3) for _ in range(n)])
        permuted_did = np.empty_like(item_did(array))
        for item in range(n):
            shuffled = array[item, prefix_permutations[item]]
            permuted_did[item] = shuffled[2, 2] - shuffled[1, 2] - shuffled[2, 1] + shuffled[1, 1]
        target_permutation = rng.permutation(n)
        permuted_confirmation = confirmation[target_permutation]
        fold_values = {key: [] for key in null}
        for held in range(n):
            train = np.asarray([index for index in range(n) if index != held]); layer = int(selected[held])
            x = centered_design(rho, confirmation, train)
            coefficient, _ = ridge_gcv(x, centered_activation(array, train, layer))
            x_pc = centered_design(rho, permuted_confirmation, train)
            coefficient_pc, _ = ridge_gcv(x_pc, centered_activation(array, train, layer))
            v_r = unit(permuted_did[train, layer].mean(axis=0)); v_p = unit(coefficient[4]); v_c = unit(coefficient[5]); v_c_perm = unit(coefficient_pc[5])
            fold_values["cos_R_P"].append(float(v_r @ v_p))
            fold_values["cos_R_C"].append(float(v_r @ v_c))
            fold_values["cos_P_C"].append(float(v_p @ v_c_perm))
        for key in null:
            null[key].append(float(np.mean(fold_values[key])))
    result = {"geometry_shuffle_n": permutations}
    for key, values in null.items():
        values = np.asarray(values); observed_value = float(observed[key]["mean"])
        result[f"geometry_{key}_shuffle_abs95"] = float(np.quantile(np.abs(values), .95))
        result[f"geometry_{key}_shuffle_p"] = float((1 + np.sum(np.abs(values) >= abs(observed_value))) / (permutations + 1))
    return result


def analyze_model(output_dir: Path, manifest_path: Path) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame], list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    manifest = json.loads(manifest_path.read_text())
    model_id = manifest["model"]
    slug = manifest_path.stem.removeprefix("representation_")
    norming = pd.read_csv(output_dir / "norming" / f"norm_rq3_{slug}.csv")
    d_rows, curves, geometries, robustness, cross_matrices, pc_robustness = [], [], [], [], [], []
    for task in (2, 3):
        meta_path = output_dir / "activations" / slug / f"task{task}_metadata.json"
        metadata = json.loads(meta_path.read_text())
        labels = metadata["labels"]
        array = np.load(output_dir / "activations" / slug / f"task{task}_cellmean.npy", mmap_mode="r")
        prompt_array = np.load(output_dir / "activations" / slug / f"task{task}_prompt_cellmean.npy", mmap_mode="r")
        curve, _ = loio_layer_curve(array)
        selected, selected_auc, _, _ = nested_layer_selection(array)
        null = decoding_nulls(array, selected, selected_auc)
        rho, confirmation = norm_arrays(norming, labels)
        diagnostics = diagnostic_table(rho, confirmation)
        geometry_rows, geometry_summary = geometry(array, rho, confirmation, selected)
        geometry_null = geometry_shuffle_null(array, rho, confirmation, selected, geometry_summary)
        cross = cross_decoding(array, rho, confirmation, selected)
        cross.insert(0, "task", task); cross.insert(0, "model_id", model_id); cross_matrices.append(cross)
        evidence_gates = evidence_decode_gates(array, rho, confirmation, selected)
        pc_robust = pc_direction_robustness(array, rho, confirmation, selected)
        pc_robust.insert(0, "task", task); pc_robust.insert(0, "model_id", model_id); pc_robustness.append(pc_robust)
        gate = (
            selected_auc > 0.5
            and selected_auc > null["random_95"]
            and selected_auc > null["permutation_95"]
        )
        for name in ("primary_RI_conjunction", "RN_conjunction", "RI_bare"):
            robust_selected, robust_auc = nested_variant_selection(array, name)
            robustness.append({"model_id": model_id, "model": model_id.split("/")[-1], "task": task, "variant": name, "auc_nested_loio": robust_auc, "selected_layer_mode": Counter(robust_selected).most_common(1)[0][0]})
        centered = np.asarray(array) - np.asarray(array).mean(axis=(1, 2), keepdims=True)
        centered_selected, centered_auc = nested_variant_selection(centered, "primary_RI_conjunction")
        robustness.append({"model_id": model_id, "model": model_id.split("/")[-1], "task": task, "variant": "primary_layer_mean_centered", "auc_nested_loio": centered_auc, "selected_layer_mode": Counter(centered_selected).most_common(1)[0][0]})
        prompt_selected, prompt_auc = nested_prompt_selection(prompt_array)
        robustness.append({"model_id": model_id, "model": model_id.split("/")[-1], "task": task, "variant": "prompt_final_R_vs_I", "auc_nested_loio": prompt_auc, "selected_layer_mode": Counter(prompt_selected).most_common(1)[0][0]})
        d_rows.append(
            {
                "model_id": model_id,
                "model": model_id.split("/")[-1],
                "task": task,
                "auc_nested_loio": selected_auc,
                "selected_layers": json.dumps(selected.tolist()),
                "selected_layer_mode": Counter(selected).most_common(1)[0][0],
                "decode_gate": gate,
                **evidence_gates,
                **null,
                **diagnostics,
                **{f"geometry_{key}": value if not isinstance(value, dict) else value["mean"] for key, value in geometry_summary.items()},
                **{f"geometry_{key}_boot_lo": value["boot_lo"] for key, value in geometry_summary.items() if isinstance(value, dict)},
                **{f"geometry_{key}_boot_hi": value["boot_hi"] for key, value in geometry_summary.items() if isinstance(value, dict)},
                **geometry_null,
            }
        )
        curve.insert(0, "task", task)
        curve.insert(0, "model_id", model_id)
        curves.append(curve)
        geometry_rows.insert(0, "task", task)
        geometry_rows.insert(0, "model_id", model_id)
        geometries.append(geometry_rows)
    return d_rows, curves, geometries, robustness, cross_matrices, pc_robustness


def write_report(decoding: pd.DataFrame, pc_robustness: pd.DataFrame, path: Path) -> None:
    lines = [
        "# RQ3-A D/G report",
        "",
        "All test-item scores use directions trained without that item. Layer selection is nested inside each outer fold.",
        "D gate requires nested-LOIO AUC > .5 and above both random-direction and within-item label-permutation 95th percentiles.",
        "",
        "| Model | Task | R AUC | random 95 | p | perm 95 | p | R gate | P AUC/gate | C AUC/gate | corr(rho,C) | rho residual | C residual | cos R/P (shuffle p) | cos R/C (shuffle p) | cos P/C (shuffle p) | PR |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decoding.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.task} | {row.auc_nested_loio:.3f} | {row.random_95:.3f} | {row.random_p:.4f} | "
            f"{row.permutation_95:.3f} | {row.permutation_p:.4f} | {row.decode_gate} | "
            f"{row.P_decode_auc:.3f}/{row.P_decode_gate} | {row.C_decode_auc:.3f}/{row.C_decode_gate} | {row.corr_rho_C:.3f} | "
            f"{row.rho_residual_variance_ratio:.3f} | {row.C_residual_variance_ratio:.3f} | "
            f"{row.geometry_cos_R_P:.3f} ({row.geometry_cos_R_P_shuffle_p:.4f}) | {row.geometry_cos_R_C:.3f} ({row.geometry_cos_R_C_shuffle_p:.4f}) | "
            f"{row.geometry_cos_P_C:.3f} ({row.geometry_cos_P_C_shuffle_p:.4f}) | {row.geometry_participation_ratio:.2f} |"
        )
    lines += [
        "",
        "## P/C direction robustness",
        "",
        "Reverse ridge predicts the held-out continuous norming target from activation. Tertile is the registered high-minus-low diff-in-means direction. All nuisance fits and directions are LOIO.",
        "",
        "| Model | Task | Target | Reverse Pearson | Reverse Spearman | Tertile Pearson | Tertile Spearman | cos(primary, reverse) | cos(primary, tertile) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pc_robustness.itertuples(index=False):
        lines.append(
            f"| {row.model_id.split('/')[-1]} | {row.task} | {row.target} | "
            f"{row.reverse_pearson:.3f} | {row.reverse_spearman:.3f} | "
            f"{row.tertile_pearson:.3f} | {row.tertile_spearman:.3f} | "
            f"{row.primary_reverse_cos_mean:.3f} | {row.primary_tertile_cos_mean:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    raise SystemExit(
        "This script builds rq3a_decoding_gates.csv with the retired one-versus-three labelling. "
        "The canonical gate is rq3a_context_decoder_robustness.csv (family == 'conjunction_h3') produced by "
        "analysis/rq3/analyze_rq3_context_decoding_robustness.py. "
        "The old outputs are archived and not released. "
        "Function-level reuse (ridge_gcv, norm_arrays, centered_*, ...) remains valid."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    manifests = sorted((args.input_dir / "manifests").glob("representation_*.json"))
    if not manifests:
        raise FileNotFoundError("No representation manifests")
    d_rows, curves, geometries, robustness, cross_matrices, pc_robustness = [], [], [], [], [], []
    for manifest in manifests:
        model_rows, model_curves, model_geometries, model_robustness, model_cross, model_pc_robustness = analyze_model(args.input_dir, manifest)
        d_rows.extend(model_rows)
        curves.extend(model_curves)
        geometries.extend(model_geometries)
        robustness.extend(model_robustness)
        cross_matrices.extend(model_cross)
        pc_robustness.extend(model_pc_robustness)
    out = args.input_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    decoding = pd.DataFrame(d_rows)
    decoding.to_csv(out / "rq3a_decoding_gates.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(out / "rq3a_layer_curves.csv", index=False)
    pd.concat(geometries, ignore_index=True).to_csv(out / "rq3a_geometry_loio.csv", index=False)
    pd.DataFrame(robustness).to_csv(out / "rq3a_robustness.csv", index=False)
    pd.concat(cross_matrices, ignore_index=True).to_csv(out / "rq3a_cross_decoding.csv", index=False)
    pc_robustness_frame = pd.concat(pc_robustness, ignore_index=True)
    pc_robustness_frame.to_csv(out / "rq3a_pc_direction_robustness.csv", index=False)
    write_report(decoding, pc_robustness_frame, out / "rq3a_report.md")
    gate = {
        "eligible_intervention_model_tasks": [
            {"model_id": row.model_id, "task": int(row.task)}
            for row in decoding.itertuples(index=False)
            if row.decode_gate
        ]
    }
    (out / "rq3a_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n")
    print(decoding[["model", "task", "auc_nested_loio", "decode_gate"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
