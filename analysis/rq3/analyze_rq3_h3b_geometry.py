#!/usr/bin/env python3
"""H3b geometry using the revised h3-only R/I decoder.

The registered H3b question is representational geometry (R, P, C), not the
historically named ``rq3b_*`` ABL/ADD intervention files.  This analysis uses
the fold-specific layers selected by the conjunction_h3 decoder and never
uses the held item to estimate a direction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:  # package import in tests, script-local import on the command line
    from sim.analyze_rq3_representation import (
        centered_activation,
        centered_design,
        evidence_decode_gates,
        norm_arrays,
        reverse_ridge_gcv,
        ridge_gcv,
        unit,
    )
    from sim.rq3_decoder_contract import load_h3_decoder_gate
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI runs
    from analyze_rq3_representation import (
        centered_activation,
        centered_design,
        evidence_decode_gates,
        norm_arrays,
        reverse_ridge_gcv,
        ridge_gcv,
        unit,
    )
    from rq3_decoder_contract import load_h3_decoder_gate

SEED = 20260824
I, R, H3 = 1, 2, 2


def h3_contrast(array: np.ndarray) -> np.ndarray:
    """Per-item R,h3 minus I,h3 vector: [item, layer, dimension]."""
    return np.asarray(array[:, R, H3] - array[:, I, H3], dtype=np.float32)


def residualize(vector: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    """Remove the Euclidean projection onto the span of nonzero controls."""
    basis = [unit(control) for control in controls if np.linalg.norm(control) > 0]
    if not basis:
        return vector.copy()
    matrix = np.stack(basis, axis=1)
    return vector - matrix @ np.linalg.lstsq(matrix, vector, rcond=None)[0]


def fold_directions(
    array: np.ndarray,
    rho: np.ndarray,
    confirmation: np.ndarray,
    selected: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    contrasts = h3_contrast(array)
    n = len(array)
    rows: list[dict] = []
    vectors: dict[str, list[np.ndarray]] = {name: [] for name in ("R_h3", "R_unique", "P", "C", "P_dec", "C_dec")}
    for held in range(n):
        train = np.flatnonzero(np.arange(n) != held)
        layer = int(selected[held])
        coefficient, lam = ridge_gcv(
            centered_design(rho, confirmation, train),
            centered_activation(array, train, layer),
        )
        raw_r = contrasts[train, layer].mean(axis=0)
        v_p, v_c = coefficient[4], coefficient[5]
        # decoding: activation -> scalar target. H3b is a representation hypothesis,
        # so both formats are reported; neither is treated as canonical yet.
        design_train = centered_design(rho, confirmation, train)
        activation_train = centered_activation(array, train, layer)
        v_p_dec, _ = reverse_ridge_gcv(activation_train, design_train[:, 4])
        v_c_dec, _ = reverse_ridge_gcv(activation_train, design_train[:, 5])
        current = {
            "R_h3": raw_r,
            # The registered b-3 quantity: the part of raw R outside span(P,C).
            "R_unique": residualize(raw_r, [v_p, v_c]),
            "P": v_p,
            "C": v_c,
            "P_dec": v_p_dec,
            "C_dec": v_c_dec,
        }
        for name, vector in current.items():
            vectors[name].append(unit(vector))
        unique_dec = residualize(raw_r, [v_p_dec, v_c_dec])
        rows.append(
            {
                "held_item": held,
                "layer": layer,
                "lambda": lam,
                "cos_R_P_dec": float(unit(raw_r) @ unit(v_p_dec)),
                "cos_R_C_dec": float(unit(raw_r) @ unit(v_c_dec)),
                "cos_P_C_dec": float(unit(v_p_dec) @ unit(v_c_dec)),
                "unique_norm_ratio_dec": float(
                    np.linalg.norm(unique_dec) / max(np.linalg.norm(raw_r), 1e-12)
                ),
                "cos_R_P": float(unit(current["R_h3"]) @ unit(current["P"])),
                "cos_R_C": float(unit(current["R_h3"]) @ unit(current["C"])),
                "cos_P_C": float(unit(current["P"]) @ unit(current["C"])),
                "unique_norm_ratio": float(
                    np.linalg.norm(current["R_unique"]) / max(np.linalg.norm(raw_r), 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows), vectors


def cross_decode(
    array: np.ndarray,
    rho: np.ndarray,
    confirmation: np.ndarray,
    selected: np.ndarray,
    vectors: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    """LOIO cross-decoding; each target uses its natural held-item rows."""
    direction_names = ("R_h3", "R_unique", "P", "C")
    target_names = ("R_h3", "R_unique", "P", "C")
    labels = {name: [] for name in target_names}
    scores = {(d, t): [] for d in direction_names for t in target_names}
    n = len(array)
    for held in range(n):
        train = np.flatnonzero(np.arange(n) != held)
        layer = int(selected[held])

        # The revised decoder's exact binary task: R,h3=1; I,h3=0.
        r_states = np.asarray(array[held, [R, I], H3, layer])
        r_labels = np.asarray([1, 0], dtype=int)
        for target in ("R_h3", "R_unique"):
            labels[target].extend(r_labels.tolist())
            for direction in direction_names:
                scores[(direction, target)].extend((r_states @ vectors[direction][held]).tolist())

        # P/C targets use all six conjunction cells, thresholded only on train items.
        states = np.asarray(array[held, :, :, layer])[:, [1, 2], :].reshape(6, -1)
        for target, values in (("P", rho), ("C", confirmation)):
            held_values = values[held].reshape(-1)
            threshold = float(np.median(values[train]))
            target_labels = (held_values > threshold).astype(int)
            labels[target].extend(target_labels.tolist())
            for direction in direction_names:
                scores[(direction, target)].extend((states @ vectors[direction][held]).tolist())

    rows = []
    for target in target_names:
        y = np.asarray(labels[target])
        for direction in direction_names:
            auc = roc_auc_score(y, scores[(direction, target)]) if len(np.unique(y)) == 2 else np.nan
            rows.append(
                {
                    "direction": direction,
                    "target": target,
                    "auc": float(auc),
                    "n": len(y),
                    "positive": int(y.sum()),
                }
            )
    return pd.DataFrame(rows)


def geometry_summary(folds: pd.DataFrame, contrasts: np.ndarray, selected: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    result: dict[str, float] = {}
    for column in ("cos_R_P_dec", "cos_R_C_dec", "cos_P_C_dec", "unique_norm_ratio_dec",
                   "cos_R_P", "cos_R_C", "cos_P_C", "unique_norm_ratio"):
        values = folds[column].to_numpy()
        boot = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        result[column] = float(values.mean())
        result[f"{column}_boot_lo"] = float(np.quantile(boot, 0.025))
        result[f"{column}_boot_hi"] = float(np.quantile(boot, 0.975))

    pr = []
    for held in range(len(contrasts)):
        train = np.flatnonzero(np.arange(len(contrasts)) != held)
        layer = int(selected[held])
        matrix = contrasts[train, layer] - contrasts[train, layer].mean(axis=0, keepdims=True)
        singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        eigen = singular**2 / max(len(train) - 1, 1)
        pr.append(float(eigen.sum() ** 2 / max(np.sum(eigen**2), 1e-12)))
    pr = np.asarray(pr)
    boot = rng.choice(pr, size=(5000, len(pr)), replace=True).mean(axis=1)
    result["participation_ratio"] = float(pr.mean())
    result["participation_ratio_boot_lo"] = float(np.quantile(boot, 0.025))
    result["participation_ratio_boot_hi"] = float(np.quantile(boot, 0.975))

    dimension = contrasts.shape[-1]
    left = rng.normal(size=(5000, dimension)); right = rng.normal(size=(5000, dimension))
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    result["random_abs_cos_95"] = float(np.quantile(np.abs(np.sum(left * right, axis=1)), 0.95))
    return result


def shuffle_null(
    array: np.ndarray,
    rho: np.ndarray,
    confirmation: np.ndarray,
    selected: np.ndarray,
    observed: dict,
    permutations: int = 200,
) -> dict:
    """Break R/P/C correspondence while retaining item and activation structure."""
    rng = np.random.default_rng(SEED)
    n = len(array)
    contrasts = h3_contrast(array)
    null = {name: [] for name in ("cos_R_P_dec", "cos_R_C_dec", "cos_P_C_dec",
                                  "cos_R_P", "cos_R_C", "cos_P_C")}
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=n)
        p_perm = rng.permutation(n)
        c_perm = rng.permutation(n)
        values = {name: [] for name in null}
        for held in range(n):
            train = np.flatnonzero(np.arange(n) != held)
            layer = int(selected[held])
            v_r = unit((contrasts[train, layer] * signs[train, None]).mean(axis=0))
            p_coef, _ = ridge_gcv(
                centered_design(rho[p_perm], confirmation, train),
                centered_activation(array, train, layer),
            )
            c_coef, _ = ridge_gcv(
                centered_design(rho, confirmation[c_perm], train),
                centered_activation(array, train, layer),
            )
            v_p, v_c = unit(p_coef[4]), unit(c_coef[5])
            activation_train = centered_activation(array, train, layer)
            p_dec, _ = reverse_ridge_gcv(activation_train, centered_design(rho[p_perm], confirmation, train)[:, 4])
            c_dec, _ = reverse_ridge_gcv(activation_train, centered_design(rho, confirmation[c_perm], train)[:, 5])
            v_p_dec, v_c_dec = unit(p_dec), unit(c_dec)
            values["cos_R_P_dec"].append(float(v_r @ v_p_dec))
            values["cos_R_C_dec"].append(float(v_r @ v_c_dec))
            values["cos_P_C_dec"].append(float(v_p_dec @ v_c_dec))
            values["cos_R_P"].append(float(v_r @ v_p))
            values["cos_R_C"].append(float(v_r @ v_c))
            values["cos_P_C"].append(float(v_p @ v_c))
        for name in null:
            null[name].append(float(np.mean(values[name])))
    output = {"shuffle_n": permutations}
    for name, values in null.items():
        values = np.asarray(values)
        output[f"{name}_shuffle_abs95"] = float(np.quantile(np.abs(values), 0.95))
        output[f"{name}_shuffle_p"] = float(
            (1 + np.sum(np.abs(values) >= abs(float(observed[name])))) / (permutations + 1)
        )
    return output


def write_report(summary: pd.DataFrame, cross: pd.DataFrame, path: Path) -> None:
    shown = summary[
        [
            "model", "task", "h3_decoder_auc", "selected_layer_mode",
            "cos_R_P", "cos_R_P_shuffle_p", "cos_R_C", "cos_R_C_shuffle_p",
            "cos_P_C", "cos_P_C_shuffle_p", "unique_norm_ratio",
            "participation_ratio", "P_decode_auc", "P_decode_gate",
            "C_decode_auc", "C_decode_gate",
        ]
    ].copy()
    shown.columns = [
        "Model", "Task", "h3 AUC", "Layer", "cos R/P", "shuffle p", "cos R/C",
        "shuffle p ", "cos P/C", "shuffle p  ", "unique/R", "PR", "P AUC", "P gate",
        "C AUC", "C gate",
    ]
    lines = [
        "# H3b: representational geometry of R/P/C on h3 only",
        "",
        "The R direction is learned per fold as `R,h3 - I,h3`; the held-out item is excluded from every direction estimate and layer choice.",
        "H3b is descriptive. Instead of a binary pass verdict it reports cosine bootstrap CIs and a shuffle null that breaks the pairing.",
        "",
        shown.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Cross-decoding (mean AUC per model × task cell)",
        "",
        cross.groupby(["direction", "target"], as_index=False).auc.mean().pivot(index="direction", columns="target", values="auc").to_markdown(floatfmt=".3f"),
        "",
        "`unique/R` is the norm of the orthogonal residual of the raw h3 R−I vector after removing its projection onto span(P,C), divided by the raw norm. PR is the participation ratio of the per-item h3 R−I contrasts.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="H3b output directory (default: INPUT_DIR/analysis/h3b_h3_geometry).",
    )
    parser.add_argument(
        "--decoder-gate-file",
        type=Path,
        help="Canonical h3 decoder gate (default: INPUT_DIR/analysis/rq3a_context_decoder_robustness.csv).",
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument(
        "--reuse-null",
        action="store_true",
        help="Reuse existing shuffle and P/C gate columns when only b-3/cross-decoding code changed.",
    )
    args = parser.parse_args()
    root = args.input_dir
    output = args.output_dir or root / "analysis" / "h3b_h3_geometry"
    previous_path = output / "h3b_geometry_summary.csv"
    previous = {}
    if args.reuse_null and previous_path.exists():
        old = pd.read_csv(previous_path)
        previous = {(row.model_id, int(row.task)): row._asdict() for row in old.itertuples(index=False)}
    decoder = load_h3_decoder_gate(
        args.decoder_gate_file
        or root / "analysis" / "rq3a_context_decoder_robustness.csv",
        require_both_tasks=True,
    )
    decoder = decoder[decoder.decode_gate.astype(bool)].set_index(["model_id", "task"])
    summaries, fold_tables, cross_tables = [], [], []
    for manifest_path in sorted((root / "manifests").glob("representation_*.json")):
        manifest = json.loads(manifest_path.read_text())
        model_id = manifest["model"]
        slug = manifest_path.stem.removeprefix("representation_")
        norming = pd.read_csv(root / "norming" / f"norm_rq3_{slug}.csv")
        for task in (2, 3):
            key = (model_id, task)
            if key not in decoder.index:
                continue
            row = decoder.loc[key]
            selected = np.asarray(json.loads(row.selected_layers), dtype=int)
            array = np.load(root / "activations" / slug / f"task{task}_cellmean.npy", mmap_mode="r")
            metadata = json.loads((root / "activations" / slug / f"task{task}_metadata.json").read_text())
            rho, confirmation = norm_arrays(norming, metadata["labels"])
            folds, vectors = fold_directions(array, rho, confirmation, selected)
            folds.insert(0, "task", task); folds.insert(0, "model_id", model_id)
            fold_tables.append(folds)
            current_cross = cross_decode(array, rho, confirmation, selected, vectors)
            current_cross.insert(0, "task", task); current_cross.insert(0, "model_id", model_id)
            cross_tables.append(current_cross)
            summary = geometry_summary(folds, h3_contrast(array), selected)
            if key in previous:
                old = previous[key]
                summary.update({name: old[name] for name in old if name.startswith("cos_") and "shuffle_" in name})
                summary["shuffle_n"] = int(old["shuffle_n"])
                gates = {name: old[name] for name in old if name.startswith(("P_decode_", "C_decode_"))}
            else:
                summary.update(shuffle_null(array, rho, confirmation, selected, summary, args.permutations))
                gates = evidence_decode_gates(array, rho, confirmation, selected)
            summaries.append(
                {
                    "model_id": model_id,
                    "model": model_id.split("/")[-1],
                    "task": task,
                    "h3_decoder_auc": float(row.auc_nested_loio),
                    "h3_decoder_gate": bool(row.decode_gate),
                    "selected_layer_mode": Counter(selected).most_common(1)[0][0],
                    **summary,
                    **gates,
                }
            )
            print(model_id, task, "H3b complete", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    fold_frame = pd.concat(fold_tables, ignore_index=True)
    cross_frame = pd.concat(cross_tables, ignore_index=True)
    summary_frame.to_csv(output / "h3b_geometry_summary.csv", index=False)
    fold_frame.to_csv(output / "h3b_geometry_folds.csv", index=False)
    cross_frame.to_csv(output / "h3b_cross_decoding.csv", index=False)
    write_report(summary_frame, cross_frame, output / "H3B_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
