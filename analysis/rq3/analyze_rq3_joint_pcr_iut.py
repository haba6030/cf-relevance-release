#!/usr/bin/env python3
"""Joint P/C/R residualization and intersection-union test for RQ3.

The analysis is deliberately aligned with the amended canonical R contract:
each item contributes the h3 conjunction change ``A(R,h3)-A(I,h3)``.  At the
same fold-specific layer, that change is modeled by the three-column design

    [1, P(R,h3)-P(I,h3), C(R,h3)-C(I,h3)].

For each target, Frisch--Waugh--Lovell residualization removes the other two
columns from both the target and activation.  The resulting direction is fit
on 14 items and evaluated on the held-out item.  Task 2 and task 3 never share
training rows.  Layers come from the pre-existing nested-LOIO canonical R
decoder; the present outcomes are not used for layer selection.

The cell-level alternative is the conjunction

    R_perp_PC > 0 AND P_perp_RC > 0 AND C_perp_RP > 0,

where ``> 0`` means positive held-out self-decoding covariance.  Its IUT
p-value is max(p_R, p_P, p_C).  These are raw one-sided item sign-flip
p-values; Holm is neither computed nor used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sim.analyze_rq3_context_decoding_robustness import nested_loio
    from sim.analyze_rq3_pc_hierarchy import exact_sign_flip
    from sim.analyze_rq3_representation import finite_corr, norm_arrays, unit
    from sim.rq3_decoder_contract import (
        H3_DECODER_CONTRACT,
        H3_NEGATIVE_LABEL,
        H3_POSITIVE_LABEL,
    )
except ModuleNotFoundError:  # pragma: no cover - direct CLI execution
    from analyze_rq3_context_decoding_robustness import nested_loio
    from analyze_rq3_pc_hierarchy import exact_sign_flip
    from analyze_rq3_representation import finite_corr, norm_arrays, unit
    from rq3_decoder_contract import (
        H3_DECODER_CONTRACT,
        H3_NEGATIVE_LABEL,
        H3_POSITIVE_LABEL,
    )


ANALYSIS_VERSION = "rq3-joint-pcr-iut-canonical-h3-20260825"
SEED = 20260825
I, R, H3 = 1, 2, 2
TARGETS = ("R", "P", "C")
TARGET_CONTROLS = {"R": ("P", "C"), "P": ("R", "C"), "C": ("R", "P")}


def fwl_fold(
    activation_train: np.ndarray,
    activation_held: np.ndarray,
    design_train: np.ndarray,
    design_held: np.ndarray,
    target_index: int,
) -> dict[str, np.ndarray | float]:
    """Fit one target after removing the other two design columns.

    No intercept is added silently: the first design column *is* the R
    intercept/mean context-change target.  Thus P and C explicitly control R,
    while R explicitly controls P and C.
    """
    if design_train.ndim != 2 or design_train.shape[1] < 2:
        raise ValueError("Joint FWL design must have at least two columns")
    if not 0 <= target_index < design_train.shape[1]:
        raise ValueError("target_index is outside the joint design")
    controls = [index for index in range(design_train.shape[1]) if index != target_index]
    z_train = design_train[:, controls]
    z_held = design_held[:, controls]
    y_train = design_train[:, target_index]
    y_held = design_held[:, target_index]

    beta_y = np.linalg.lstsq(z_train, y_train, rcond=None)[0]
    beta_a = np.linalg.lstsq(z_train, activation_train, rcond=None)[0]
    y_train_res = y_train - z_train @ beta_y
    y_held_res = y_held - z_held @ beta_y
    a_train_res = activation_train - z_train @ beta_a
    a_held_res = activation_held - z_held @ beta_a

    raw_direction = a_train_res.T @ y_train_res
    direction = unit(raw_direction)
    train_score = a_train_res @ direction
    denominator = float(train_score @ train_score)
    slope = float(train_score @ y_train_res / denominator) if denominator > 1e-12 else 0.0
    prediction = a_held_res @ direction * slope
    original_ss = float(y_train @ y_train)
    residual_ratio = float(y_train_res @ y_train_res / max(original_ss, 1e-12))
    return {
        "target": y_held_res,
        "prediction": prediction,
        "direction": direction,
        "direction_raw_norm": float(np.linalg.norm(raw_direction)),
        "residual_variance_ratio": residual_ratio,
        "design_rank": int(np.linalg.matrix_rank(design_train)),
        "design_condition_number": float(np.linalg.cond(design_train)),
    }


def evaluate_nested_loio(
    activation: np.ndarray,
    design: np.ndarray,
    selected_layers: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all three unique directions with identical outer folds/layers."""
    if activation.ndim != 3:
        raise ValueError(f"activation must be [item,layer,dim], got {activation.shape}")
    if design.shape != (len(activation), 3):
        raise ValueError(f"design must be [item,3], got {design.shape}")
    if selected_layers.shape != (len(activation),):
        raise ValueError("selected_layers must contain one layer per held item")

    fold_rows: list[dict] = []
    for held in range(len(activation)):
        train = np.flatnonzero(np.arange(len(activation)) != held)
        layer = int(selected_layers[held])
        if not 0 <= layer < activation.shape[1]:
            raise ValueError(f"Selected layer {layer} is outside 0..{activation.shape[1]-1}")
        for target_index, target_name in enumerate(TARGETS):
            fit = fwl_fold(
                np.asarray(activation[train, layer], dtype=np.float64),
                np.asarray(activation[[held], layer], dtype=np.float64),
                design[train],
                design[[held]],
                target_index,
            )
            held_target = float(np.asarray(fit["target"])[0])
            prediction = float(np.asarray(fit["prediction"])[0])
            fold_rows.append(
                {
                    "held_item": held,
                    "layer": layer,
                    "target": target_name,
                    "controls": "+".join(TARGET_CONTROLS[target_name]),
                    "held_residual_target": held_target,
                    "held_prediction": prediction,
                    "covariance_contribution": held_target * prediction,
                    "direction_raw_norm": fit["direction_raw_norm"],
                    "residual_variance_ratio": fit["residual_variance_ratio"],
                    "design_rank": fit["design_rank"],
                    "design_condition_number": fit["design_condition_number"],
                }
            )

    folds = pd.DataFrame(fold_rows)
    summary_rows = []
    for target_name, group in folds.groupby("target", sort=False):
        contribution = group.covariance_contribution.to_numpy(float)
        target = group.held_residual_target.to_numpy(float)
        prediction = group.held_prediction.to_numpy(float)
        summary_rows.append(
            {
                "target": target_name,
                "controls": "+".join(TARGET_CONTROLS[target_name]),
                "n_items": len(group),
                "mean_covariance": float(contribution.mean()),
                "pearson": finite_corr(target, prediction),
                "spearman": finite_corr(target, prediction, "spearman"),
                "q2": float(1 - np.sum((target - prediction) ** 2) / max(np.sum(target**2), 1e-12)),
                "p_flip_one": exact_sign_flip(contribution, "greater"),
                "p_flip_two": exact_sign_flip(contribution, "two-sided"),
                "positive_items": int((contribution > 0).sum()),
                "negative_items": int((contribution < 0).sum()),
                "residual_variance_ratio": float(group.residual_variance_ratio.mean()),
                "direction_raw_norm": float(group.direction_raw_norm.mean()),
                "design_rank_min": int(group.design_rank.min()),
                "design_condition_number_median": float(group.design_condition_number.median()),
            }
        )
    return pd.DataFrame(summary_rows), folds


def h30_no_holm(root: Path) -> pd.DataFrame:
    """Return the exact preregistered normalized conjunction R-I rows, raw p only."""
    tests = pd.read_csv(root / "analysis" / "h30_tests.csv")
    selected = tests[
        (tests.arm == "conj")
        & (tests.metric == "normalized")
        & (tests.contrast == "R-I")
        & tests.task.isin((2, 3))
    ].copy()
    if selected.duplicated(["model_id", "task"]).any():
        raise RuntimeError("H3-0 table has duplicate model/task cells")
    selected["raw_gate"] = (selected["mean"] > 0) & (selected["p_flip_two"] < 0.05)
    model_gate = selected.groupby("model_id").agg(
        tasks=("task", "nunique"),
        raw_h30_iut_p=("p_flip_two", "max"),
        raw_h30_all_tasks=("raw_gate", "all"),
    )
    if not ((model_gate.tasks == 2) & model_gate.raw_h30_all_tasks).all():
        failed = model_gate[~((model_gate.tasks == 2) & model_gate.raw_h30_all_tasks)]
        raise RuntimeError(f"Not all requested models pass raw no-Holm H3-0:\n{failed}")
    return selected


def bootstrap_iut_ci(folds: pd.DataFrame, bootstrap: int) -> pd.DataFrame:
    """Item bootstrap CIs for each component's mean covariance (descriptive)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for target, group in folds.groupby("target", sort=False):
        values = group.sort_values("held_item").covariance_contribution.to_numpy(float)
        indices = rng.integers(0, len(values), size=(bootstrap, len(values)))
        means = values[indices].mean(axis=1)
        rows.append(
            {
                "target": target,
                "boot_cov_lo": float(np.quantile(means, 0.025)),
                "boot_cov_hi": float(np.quantile(means, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    cells: pd.DataFrame,
    models: pd.DataFrame,
    global_row: dict,
    raw_h30: pd.DataFrame,
    path: Path,
) -> None:
    shown = cells.copy()
    shown["component"] = shown.target.map(
        {"R": "R_perp_PC", "P": "P_perp_RC", "C": "C_perp_RP"}
    )
    component_table = shown[
        [
            "model", "task", "component", "pearson", "mean_covariance",
            "boot_cov_lo", "boot_cov_hi", "p_flip_one", "residual_variance_ratio",
            "selected_layer_mode", "component_pass",
        ]
    ]
    cell_table = (
        shown.pivot(index=["model", "task"], columns="target", values="p_flip_one")
        .reset_index()
        .merge(
            shown.groupby(["model", "task"], as_index=False).agg(
                iut_p=("cell_iut_p", "first"),
                all_three=("cell_iut_pass", "first"),
            ),
            on=["model", "task"],
        )
    )
    h30_table = raw_h30.pivot(index="model", columns="task", values="p_flip_two").reset_index()
    h30_table.columns = ["model", "task2_raw_p", "task3_raw_p"]
    h30_table["IUT_over_tasks"] = h30_table[["task2_raw_p", "task3_raw_p"]].max(axis=1)
    h30_table["eligible_no_Holm"] = h30_table.IUT_over_tasks < 0.05
    passed_cells = int(shown.groupby(["model", "task"]).cell_iut_pass.first().sum())
    passed_models = int(models.both_tasks_pass.sum())
    lines = [
        "# RQ3 joint P/C/R residualisation and IUT",
        "",
        "## Summary",
        "",
        f"- Model × task cells where all three unique components are detected: **{passed_cells}/14**.",
        f"- Models passing in both the two- and three-option tasks: **{passed_models}/7**.",
        "- Without Holm, the H3-0 entry gate passes **7/7 models**, DeepSeek included.",
        "",
        "## H3-0 entry gate without Holm",
        "",
        "Both tasks are required (intersection), so the per-model entry p is the maximum of the two raw p-values. "
        "Holm-corrected values are not used.",
        "",
        h30_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fixed test",
        "",
        "- Input contrast: per-item conjunction `A(R,h3)-A(I,h3)`; h2, bare and N are not mixed in.",
        "- Design: `DeltaA = beta_R*1 + beta_P*DeltaP + beta_C*DeltaC`.",
        "- Unique directions: the other two columns are removed by FWL from both target and activations within the training fold.",
        "- Generalisation: item LOIO; the two tasks are fit and evaluated separately.",
        "- Layer: the layer chosen by the canonical `R,h3` vs `I,h3` nested LOIO in the same outer fold.",
        "- Component test: one-sided exact sign-flip p over 15 items of the held-out self-decoding covariance.",
        "- Joint hypothesis: `H1 = R_perp_PC AND P_perp_RC AND C_perp_RP`; `p_IUT=max(p_R,p_P,p_C)`.",
        "- Holm and other multiplicity corrections are neither computed nor used.",
        "",
        "`!=0` is operationalised as **held-out self-target covariance > 0**, not as the training norm of the direction. "
        "The learned direction is oriented by the target sign, so negative generalisation is a failure, not a separate significant direction.",
        "R is the intercept component of the change-vector regression, so the **uncentred "
        "held-out covariance and its sign-flip p** are the primary statistics rather than a mean-removing Pearson r.",
        "",
        "## Model × task IUT",
        "",
        cell_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## The three components in detail",
        "",
        component_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Per-model IUT requiring both tasks",
        "",
        models.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Intersection over all seven models",
        "",
        f"IUT p over all models, tasks and components = **{global_row['all_models_iut_p']:.4f}**; "
        f"verdict = **{'pass' if global_row['all_models_iut_pass'] else 'fail'}**.",
        "",
        "## Interpretive limits",
        "",
        "This test asks for a unique held-out linear signal not linearly explained by the other two measured "
        "components in the same hidden-state space. It does not establish causal independence, probabilistic "
        "independence or exact Euclidean orthogonality. P and C are measured from the model's own norming responses.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    output = root / "analysis" / "joint_pcr_iut_no_holm"
    output.mkdir(parents=True, exist_ok=True)

    raw_h30 = h30_no_holm(root)
    eligible = sorted(raw_h30.model_id.unique())
    if len(eligible) != 7:
        raise RuntimeError(f"Expected exactly 7 raw-H3-0 eligible models, found {len(eligible)}")

    old_decoder_path = root / "analysis" / "rq3a_context_decoder_robustness.csv"
    old_decoder = pd.read_csv(old_decoder_path) if old_decoder_path.exists() else pd.DataFrame()
    summary_tables, fold_tables, selection_rows = [], [], []
    for model_id in eligible:
        slug = model_id.replace("/", "__")
        manifest_path = root / "manifests" / f"representation_{slug}.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing representation manifest for raw-H3-0 model: {model_id}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("model") != model_id:
            raise RuntimeError(f"Representation manifest model mismatch: {manifest_path}")
        norming = pd.read_csv(root / "norming" / f"norm_rq3_{slug}.csv")
        model = model_id.split("/")[-1]
        for task in (2, 3):
            activation_path = root / "activations" / slug / f"task{task}_cellmean.npy"
            metadata_path = root / "activations" / slug / f"task{task}_metadata.json"
            array = np.load(activation_path, mmap_mode="r")
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("task") != task or metadata.get("labels") is None:
                raise RuntimeError(f"Task metadata mismatch: {model_id} task{task}")
            p_values, c_values = norm_arrays(norming, metadata["labels"])
            activation_delta = np.asarray(array[:, R, H3] - array[:, I, H3], dtype=np.float32)
            design = np.column_stack(
                [
                    np.ones(len(array), dtype=float),
                    p_values[:, R, 1] - p_values[:, I, 1],
                    c_values[:, R, 1] - c_values[:, I, 1],
                ]
            )
            if np.linalg.matrix_rank(design) != 3:
                raise RuntimeError(f"Joint PCR design is rank deficient: {model_id} task{task}")

            # Recompute the established canonical nested-LOIO layer selections for
            # every model.  This adds DeepSeek without mutating the old decoder CSV.
            selected, auc, _ = nested_loio(array, (H3,))
            previous = old_decoder[
                (old_decoder.model_id == model_id)
                & (old_decoder.task == task)
                & (old_decoder.family == "conjunction_h3")
            ] if len(old_decoder) else pd.DataFrame()
            if len(previous) > 1:
                raise RuntimeError(f"Duplicate previous canonical decoder row: {model_id} task{task}")
            matches_previous = np.nan
            if len(previous) == 1:
                previous_layers = np.asarray(json.loads(previous.iloc[0].selected_layers), dtype=int)
                matches_previous = bool(np.array_equal(selected, previous_layers))
                if not matches_previous:
                    raise RuntimeError(f"Canonical layer recomputation drifted: {model_id} task{task}")
            selection_rows.append(
                {
                    "model_id": model_id,
                    "model": model,
                    "task": task,
                    "decoder_contract": H3_DECODER_CONTRACT,
                    "positive_label": H3_POSITIVE_LABEL,
                    "negative_label": H3_NEGATIVE_LABEL,
                    "auc_nested_loio": auc,
                    "selected_layers": json.dumps(selected.tolist()),
                    "selected_layer_mode": Counter(selected.tolist()).most_common(1)[0][0],
                    "matches_previous_selection": matches_previous,
                }
            )
            summary, folds = evaluate_nested_loio(activation_delta, design, selected)
            summary = summary.merge(bootstrap_iut_ci(folds, args.bootstrap), on="target")
            summary.insert(0, "task", task)
            summary.insert(0, "model", model)
            summary.insert(0, "model_id", model_id)
            summary["selected_layer_mode"] = Counter(selected.tolist()).most_common(1)[0][0]
            summary["component_pass"] = (
                (summary.mean_covariance > 0) & (summary.p_flip_one < 0.05)
            )
            iut_p = float(summary.p_flip_one.max())
            summary["cell_iut_p"] = iut_p
            summary["cell_iut_pass"] = bool(summary.component_pass.all() and iut_p < 0.05)
            folds.insert(0, "task", task)
            folds.insert(0, "model", model)
            folds.insert(0, "model_id", model_id)
            summary_tables.append(summary)
            fold_tables.append(folds)
            print(
                f"{model:36s} task{task} AUC_R={auc:.4f} "
                f"p_R/P/C={summary.set_index('target').p_flip_one.to_dict()} IUT={iut_p:.4f}",
                flush=True,
            )

    cells = pd.concat(summary_tables, ignore_index=True)
    folds = pd.concat(fold_tables, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    model_rows = []
    for (model_id, model), group in cells.groupby(["model_id", "model"], sort=False):
        task_iut = group.groupby("task").cell_iut_p.first()
        task_pass = group.groupby("task").cell_iut_pass.first()
        model_iut = float(task_iut.max())
        model_rows.append(
            {
                "model_id": model_id,
                "model": model,
                "task2_iut_p": float(task_iut.loc[2]),
                "task3_iut_p": float(task_iut.loc[3]),
                "model_iut_p": model_iut,
                "both_tasks_pass": bool(task_pass.all() and model_iut < 0.05),
            }
        )
    models = pd.DataFrame(model_rows)
    all_models_iut_p = float(models.model_iut_p.max())
    global_row = {
        "all_models_iut_p": all_models_iut_p,
        "all_models_iut_pass": bool(models.both_tasks_pass.all() and all_models_iut_p < 0.05),
        "models": len(models),
        "tasks": 2,
        "components": 3,
    }

    raw_h30.to_csv(output / "h30_raw_no_holm.csv", index=False)
    selections.to_csv(output / "canonical_r_layer_selections.csv", index=False)
    cells.to_csv(output / "component_tests.csv", index=False)
    folds.to_csv(output / "fold_predictions.csv", index=False)
    models.to_csv(output / "model_iut.csv", index=False)
    (output / "global_iut.json").write_text(json.dumps(global_row, ensure_ascii=False, indent=2) + "\n")
    write_report(cells, models, global_row, raw_h30, output / "REPORT.md")
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "models": eligible,
        "tasks_separate": True,
        "contrast": "A(R,h3)-A(I,h3)",
        "design_columns": ["R=intercept", "delta_P_h3", "delta_C_h3"],
        "layer_policy": "canonical R,h3-vs-I,h3 nested-LOIO selected layer per outer fold",
        "test": "one-sided exact sign-flip of held-out covariance contributions",
        "iut": "max of three raw component p-values; no Holm",
        "outputs": [
            "h30_raw_no_holm.csv", "canonical_r_layer_selections.csv",
            "component_tests.csv", "fold_predictions.csv", "model_iut.csv",
            "global_iut.json", "REPORT.md",
        ],
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
