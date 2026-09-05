from __future__ import annotations

import numpy as np

try:
    from analyze_rq3_joint_pcr_iut import TARGETS, evaluate_nested_loio, fwl_fold
except ModuleNotFoundError:
    from sim.analyze_rq3_joint_pcr_iut import TARGETS, evaluate_nested_loio, fwl_fold


def test_fwl_recovers_each_unique_axis() -> None:
    rng = np.random.default_rng(7)
    n = 15
    p = np.linspace(-1.5, 1.5, n)
    c = np.sin(np.linspace(-2.0, 2.0, n))
    design = np.column_stack([np.ones(n), p, c])
    basis = np.eye(3)
    activation = design @ basis + rng.normal(scale=0.01, size=(n, 3))
    for target_index in range(3):
        fit = fwl_fold(activation[:-1], activation[-1:], design[:-1], design[-1:], target_index)
        direction = np.asarray(fit["direction"])
        assert abs(direction[target_index]) > 0.95
        assert float(np.asarray(fit["prediction"])[0] * np.asarray(fit["target"])[0]) > 0


def test_joint_iut_uses_identical_fold_layers_for_all_targets() -> None:
    rng = np.random.default_rng(11)
    n, layers = 15, 2
    p = np.linspace(-2, 2, n)
    c = np.cos(np.linspace(-2, 2, n))
    design = np.column_stack([np.ones(n), p, c])
    activation = np.empty((n, layers, 4))
    activation[:, 0] = rng.normal(size=(n, 4))
    activation[:, 1, :3] = design
    activation[:, 1, 3] = rng.normal(scale=0.01, size=n)
    selected = np.ones(n, dtype=int)
    summary, folds = evaluate_nested_loio(activation, design, selected)
    assert set(summary.target) == set(TARGETS)
    assert (folds.groupby("held_item").layer.nunique() == 1).all()
    assert (summary.p_flip_one < 0.05).all()


def test_component_controls_are_symmetric() -> None:
    rng = np.random.default_rng(19)
    design = np.column_stack([np.ones(14), rng.normal(size=14), rng.normal(size=14)])
    activation = rng.normal(size=(14, 5))
    held_design = np.asarray([[1.0, 0.2, -0.4]])
    held_activation = rng.normal(size=(1, 5))
    fits = [fwl_fold(activation, held_activation, design, held_design, index) for index in range(3)]
    assert all(fit["design_rank"] == 3 for fit in fits)
    assert all(np.isfinite(fit["design_condition_number"]) for fit in fits)
