from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

try:
    from analyze_rq3_dis import (
        attach_pc_decode_eligibility,
        build_cells,
        dis_prompt_sensitivity,
        endpoint_table,
        manipulation_monotonicity,
        summaries as dis_summaries,
    )
except ModuleNotFoundError:
    from sim.analyze_rq3_dis import (
        attach_pc_decode_eligibility,
        build_cells,
        dis_prompt_sensitivity,
        endpoint_table,
        manipulation_monotonicity,
        summaries as dis_summaries,
    )
try:
    from analyze_rq3_h30 import item_effects, tests as h30_tests
except ModuleNotFoundError:
    from sim.analyze_rq3_h30 import item_effects, tests as h30_tests
try:
    from analyze_rq3_h3b_geometry import cross_decode as h3b_cross_decode, h3_contrast, residualize
except ModuleNotFoundError:
    from sim.analyze_rq3_h3b_geometry import cross_decode as h3b_cross_decode, h3_contrast, residualize
try:
    from analyze_rq3_pc_hierarchy import (
        exact_sign_flip,
        format_states,
        loio_evaluate as pc_hierarchy_loio,
    )
except ModuleNotFoundError:
    from sim.analyze_rq3_pc_hierarchy import (
        exact_sign_flip,
        format_states,
        loio_evaluate as pc_hierarchy_loio,
    )
try:
    from analyze_rq3_intervention import monotonicity_tests, safe_monotonicity_items, selected_safe_holm
except ModuleNotFoundError:
    from sim.analyze_rq3_intervention import monotonicity_tests, safe_monotonicity_items, selected_safe_holm
try:
    from analyze_rq3_representation import (
        cross_decoding,
        centered_design,
        evidence_decode_gates,
        nested_prompt_selection,
        nested_variant_selection,
        pc_direction_robustness,
    )
except ModuleNotFoundError:
    from sim.analyze_rq3_representation import (
        cross_decoding,
        centered_design,
        evidence_decode_gates,
        nested_prompt_selection,
        nested_variant_selection,
        pc_direction_robustness,
    )
try:
    from run_rq3_fluency_guard_gpu import perplexity_metrics
except ModuleNotFoundError:
    from sim.run_rq3_fluency_guard_gpu import perplexity_metrics
try:
    from run_rq3_intervention_gpu import (
        DirectionHooks,
        context_direction,
        regression_directions,
        subspace_add_vectors,
    )
except ModuleNotFoundError:
    from sim.run_rq3_intervention_gpu import (
        DirectionHooks,
        context_direction,
        regression_directions,
        subspace_add_vectors,
    )
try:
    from run_rq3_representation_gpu import validate_crosscheck
except ModuleNotFoundError:
    from sim.run_rq3_representation_gpu import validate_crosscheck
try:
    import run_rq3_behavior_gpu as behavior_runner
except ModuleNotFoundError:
    import sim.run_rq3_behavior_gpu as behavior_runner


def test_h30_emits_normalized_and_raw_statistics() -> None:
    rows = []
    for item in range(15):
        for task in (2, 3):
            for prefix in ("N", "I", "R"):
                boost = .25 * (prefix == "R")
                rows.append({"model_id": "m", "model": "m", "task": task, "label": f"i{item}", "scenario": f"s{item}", "prefix": prefix, "lp_base": -3 - item / 20, "nt_base": 2, "lp_conj_h2": -4, "nt_conj_h2": 3, "lp_conj_h3": -3 + boost, "nt_conj_h3": 4, "lp_bare_h2": -2, "nt_bare_h2": 2, "lp_bare_h3": -1 + boost, "nt_bare_h3": 2})
    items = item_effects(pd.DataFrame(rows)); result = h30_tests(items)
    assert set(items.metric) == {"normalized", "raw"}
    assert len(result) == 24
    assert items.additivity_error.abs().max() < 1e-12
    assert {"p_exact_sign_two", "loo_worst_p_flip_two", "boot_lo", "boot_hi"} <= set(result.columns)


def test_representation_robustness_and_cross_decode_shapes() -> None:
    rng = np.random.default_rng(4)
    array = rng.normal(size=(15, 3, 5, 4, 10))
    array[:, 2, 2] += 1.0
    prompt = rng.normal(size=(15, 3, 4, 10)); prompt[:, 2] += .5
    rho = rng.normal(size=(15, 3, 2)); confirmation = rng.normal(size=(15, 3, 2))
    selected, auc = nested_variant_selection(array, "primary_RI_conjunction")
    prompt_selected, prompt_auc = nested_prompt_selection(prompt)
    cross = cross_decoding(array, rho, confirmation, selected)
    evidence = evidence_decode_gates(array, rho, confirmation, selected, random_n=20)
    assert selected.shape == prompt_selected.shape == (15,)
    assert 0 <= auc <= 1 and 0 <= prompt_auc <= 1
    assert cross.shape == (16, 5)
    assert set(cross.direction) == {"R", "R_unique", "P", "C"}
    assert {"P_decode_gate", "C_decode_gate", "P_decode_random_95", "C_decode_random_95"} <= set(evidence)


def test_pc_reverse_and_tertile_robustness_are_loio() -> None:
    rng = np.random.default_rng(24)
    array = rng.normal(size=(15, 3, 5, 3, 12))
    rho = rng.normal(size=(15, 3, 2))
    confirmation = rng.normal(size=(15, 3, 2))
    for item in range(15):
        for prefix in range(3):
            for hypothesis in range(2):
                array[item, prefix, hypothesis + 1, :, 0] += rho[item, prefix, hypothesis]
                array[item, prefix, hypothesis + 1, :, 1] += confirmation[item, prefix, hypothesis]
    result = pc_direction_robustness(array, rho, confirmation, np.ones(15, dtype=int))
    assert set(result.target) == {"P", "C"}
    assert result.n_held_rows.eq(90).all()
    assert np.isfinite(result[["reverse_pearson", "tertile_pearson"]]).all().all()


def test_pc_composite_uses_within_item_centered_correlation() -> None:
    rng = np.random.default_rng(31)
    array = rng.normal(size=(15, 3, 5, 2, 8))
    rho = rng.normal(size=(15, 3, 2))
    confirmation = rng.normal(size=(15, 3, 2))
    train = np.arange(15)
    result = regression_directions(array, rho, confirmation, train)
    design = centered_design(rho, confirmation, train)
    sign = 1.0 if np.corrcoef(design[:, 4], design[:, 5])[0, 1] >= 0 else -1.0
    p = result["P"] / np.maximum(np.linalg.norm(result["P"], axis=1, keepdims=True), 1e-12)
    c = result["C"] / np.maximum(np.linalg.norm(result["C"], axis=1, keepdims=True), 1e-12)
    assert np.allclose(result["PC"], p + sign * c)


def test_safe_monotonicity_uses_only_guarded_range() -> None:
    add = pd.DataFrame([{"model_id": "m", "task": 2, "direction": "R", "label": f"i{item}", "alpha": alpha, "induced_dod": alpha + item / 1000} for item in range(15) for alpha in (.5, 1.0, 2.0, 4.0)])
    safe = pd.DataFrame([{"model_id": "m", "task": 2, "direction": "R", "alpha": alpha, "guard_pass": alpha <= 2} for alpha in (.5, 1.0, 2.0, 4.0)])
    item = safe_monotonicity_items(add, safe); test = monotonicity_tests(item)
    assert item.alpha_safe.eq(2).all()
    assert test.iloc[0]["mean"] == 1


def test_add_holm_uses_selected_endpoints_even_when_alphas_differ() -> None:
    frame = pd.DataFrame([
        {"model_id": "m", "task": 2, "direction": "R", "alpha": 2.0, "p_flip_two": .02, "selected_safe": True},
        {"model_id": "m", "task": 3, "direction": "R", "alpha": 1.0, "p_flip_two": .03, "selected_safe": True},
        {"model_id": "m", "task": 2, "direction": "R", "alpha": 1.0, "p_flip_two": .90, "selected_safe": False},
    ])
    result = selected_safe_holm(frame)
    assert np.allclose(result.loc[result.selected_safe, "selected_safe_holm_p"], [.04, .04])
    assert pd.isna(result.loc[~result.selected_safe, "selected_safe_holm_p"]).all()


def test_dis_cell_builder_and_endpoint_statistics() -> None:
    labels = [f"i{i}" for i in range(15)]
    base_b = pd.DataFrame([{"model_id": "m", "task": 2, "label": label, "prefix": prefix, "gap": .1 * (prefix == "R"), "gap_raw": .2 * (prefix == "R"), "base_raw": -4 - index / 10} for index, label in enumerate(labels) for prefix in ("N", "I", "R")])
    base_n = pd.DataFrame([{"model_id": "m", "label": label, "prefix": prefix, "hypothesis": hypothesis, "rho": index / 20 + (hypothesis == "h3"), "C": index / 30 + (hypothesis == "h2")} for index, label in enumerate(labels) for prefix in ("N", "I", "R") for hypothesis in ("h2", "h3")])
    behavior_rows, norming_rows = [], []
    for index, label in enumerate(labels):
        for direction in ("P", "C", "PC"):
            for mode in ("differential", "common"):
                for alpha in (.5, 1.0):
                    behavior_rows.append({"model_id": "m", "task": 2, "label": label, "direction": direction, "mode": mode, "alpha": alpha, "lp_conj_h2": -2 - alpha / 10, "nt_conj_h2": 2, "lp_conj_h3": -2 + alpha / 10, "nt_conj_h3": 2, "lp_base_plus": -4 - index / 10, "lp_base_minus": -4 - index / 10})
                    for hypothesis in ("h2", "h3"):
                        sign = 1 if mode == "common" or hypothesis == "h3" else -1
                        baseline = base_n[(base_n.label == label) & (base_n.prefix == "I") & (base_n.hypothesis == hypothesis)].iloc[0]
                        norming_rows.append({"model_id": "m", "task": 2, "label": label, "direction": direction, "mode": mode, "stage": "ADD", "alpha": alpha, "hypothesis": hypothesis, "rho": baseline.rho + sign * alpha, "C": baseline.C + sign * alpha / 2})
    cells = build_cells(pd.DataFrame(behavior_rows), pd.DataFrame(norming_rows), base_b, base_n)
    safe = cells[["model_id", "task", "direction", "mode", "alpha"]].drop_duplicates(); safe["selected_safe"] = safe.alpha.eq(1.0)
    endpoint = endpoint_table(cells, safe)
    assert len(cells) == 15 * 3 * 2 * 2
    assert len(endpoint) == 3 * 2
    assert {"target_holm_task_p", "dod_raw_loo_worst_dropped", "target_boot_lo"} <= set(endpoint.columns)


def test_h3f_calibration_uses_h2a_r_minus_n_contrast() -> None:
    labels = [f"i{i}" for i in range(15)]
    cells = pd.DataFrame([
        {"model_id": "m", "task": 2, "label": label, "direction": "P", "mode": "differential", "alpha": alpha, "delta_rho_diff": alpha, "induced_dod": .1 * alpha}
        for label in labels for alpha in (.5, 1.0)
    ])
    slopes = pd.DataFrame([
        {"model_id": "m", "task": 2, "label": label, "direction": "P", "mode": "differential", "beta_rho": .1, "rho_alpha_spearman": 1.0}
        for label in labels
    ])
    base_norm = pd.DataFrame([
        {"model_id": "m", "label": label, "prefix": prefix, "hypothesis": hypothesis, "rho": ({"N": 0.0, "I": 1.8, "R": 2.0}[prefix] if hypothesis == "h3" else 0.0)}
        for label in labels for prefix in ("N", "I", "R") for hypothesis in ("h2", "h3")
    ])
    base_behavior = pd.DataFrame([
        {"model_id": "m", "task": 2, "label": label, "prefix": prefix, "gap": {"N": 0.0, "I": .9, "R": 1.0}[prefix]}
        for label in labels for prefix in ("N", "I", "R")
    ])
    safe = pd.DataFrame([{"model_id": "m", "task": 2, "direction": "P", "mode": "differential", "alpha": 1.0, "selected_safe": True}])
    result = dis_summaries(cells, slopes, base_behavior, base_norm, safe).iloc[0]
    assert result.calibration_contrast == "R-N (H2a/RQ1)"
    assert np.isclose(result.observed_mean_deltaP, 2.0)
    assert np.isclose(result.observed_mean_DoD, 1.0)


def test_h3f_target_monotonicity_is_an_explicit_gate() -> None:
    slopes = pd.DataFrame([
        {"model_id": "m", "task": task, "label": f"i{item}", "direction": "P", "mode": "differential", "target_alpha_spearman": 1.0}
        for task in (2, 3) for item in range(15)
    ])
    result = manipulation_monotonicity(slopes)
    assert result.monotonicity_gate.all()
    assert (result.holm_task_p < .05).all()


def test_dis_prompt_sensitivity_keeps_all_twelve_variants() -> None:
    baseline = pd.DataFrame([
        {"model_id": "m", "task": 2, "label": "i", "prefix": "I", "frame": frame, "qvar": qvar, "lp_conj_h2": -2.0, "nt_conj_h2": 2, "lp_conj_h3": -2.0, "nt_conj_h3": 2}
        for frame in range(2) for qvar in range(6)
    ])
    changed = pd.DataFrame([
        {"model_id": "m", "task": 2, "label": "i", "direction": "P", "mode": "differential", "alpha": 1.0, "frame": frame, "qvar": qvar, "lp_conj_h2": -2.0, "nt_conj_h2": 2, "lp_conj_h3": -1.0, "nt_conj_h3": 2}
        for frame in range(2) for qvar in range(6)
    ])
    result = dis_prompt_sensitivity(changed, baseline).iloc[0]
    assert result.n_variants == 12 and result.positive_variants == 12


def test_h3f_pc_decode_eligibility_does_not_invent_a_composite_gate() -> None:
    frame = pd.DataFrame([
        {"model_id": "m", "task": 2, "direction": direction}
        for direction in ("P", "C", "PC")
    ])
    gates = pd.DataFrame([{
        "model_id": "m", "task": 2,
        "P_decode_gate": True, "C_decode_gate": False,
    }])
    result = attach_pc_decode_eligibility(frame, gates).set_index("direction")
    assert bool(result.loc["P", "inferential_eligible"])
    assert not bool(result.loc["C", "inferential_eligible"])
    assert pd.isna(result.loc["PC", "inferential_eligible"])
    assert result.loc["PC", "eligibility_status"] == "descriptive_only_no_preregistered_PC_gate"


def test_perplexity_overflow_is_a_failed_guard() -> None:
    ppl, increase, passes = perplexity_metrics(1000.0, 1, 1.0)
    assert np.isinf(ppl) and np.isinf(increase) and not passes


def test_context_direction_uses_only_train_items_and_requested_options() -> None:
    array = np.zeros((3, 3, 5, 2, 2), dtype=np.float32)
    array[0, 2, 2] = 4; array[0, 1, 2] = 1
    array[1, 2, 2] = 6; array[1, 1, 2] = 2
    array[2, 2, 2] = 1000  # held item must not affect the direction
    observed = context_direction(array, np.asarray([0, 1]), (2,))
    np.testing.assert_allclose(observed, np.full((2, 2), 3.5))
    np.testing.assert_allclose(context_direction(array, np.asarray([0, 1]), (1,)), 0)


def test_h3b_uses_h3_context_contrast_and_prefix_major_pc_rows() -> None:
    array = np.zeros((3, 3, 5, 1, 4), dtype=np.float32)
    array[:, 2, 2, 0, 0] = [3, 4, 5]
    array[:, 1, 2, 0, 0] = 1
    np.testing.assert_allclose(h3_contrast(array)[:, 0, 0], [2, 3, 4])

    # Cross-decoding must align prefix-major activation rows with rho/C.reshape().
    rho = np.arange(18, dtype=float).reshape(3, 3, 2)
    confirmation = rho[::-1].copy()
    selected = np.zeros(3, dtype=int)
    vectors = {name: [np.eye(4)[0] for _ in range(3)] for name in ("R_h3", "R_unique", "P", "C")}
    cross = h3b_cross_decode(array, rho, confirmation, selected, vectors)
    assert cross.shape == (16, 5)
    assert set(cross.direction) == {"R_h3", "R_unique", "P", "C"}

    unique = residualize(np.asarray([1.0, 2.0, 3.0]), [np.asarray([1.0, 0.0, 0.0])])
    np.testing.assert_allclose(unique, [0.0, 2.0, 3.0])
    assert np.linalg.norm(unique) <= np.linalg.norm([1.0, 2.0, 3.0])


def test_pc_hierarchy_detects_continuous_signal_with_item_loio() -> None:
    rng = np.random.default_rng(20260825)
    target = rng.normal(size=(15, 6))
    states = rng.normal(scale=.15, size=(15, 6, 2, 12)).astype(np.float32)
    states[..., 0] += target[:, :, None]
    result, folds, directions = pc_hierarchy_loio(
        states, target, np.ones(15, dtype=int), cell_center=True, bootstrap=100
    )
    assert result["pearson"] > .8
    assert result["p_flip_one"] < .05
    assert result["gate"]
    assert len(folds) == len(directions) == 15


def test_pc_hierarchy_cell_order_and_sign_flip_are_correct() -> None:
    array = np.arange(2 * 3 * 5 * 1 * 1, dtype=np.float32).reshape(2, 3, 5, 1, 1)
    bare = format_states(array, "bare")[:, :, 0, 0]
    conjunction = format_states(array, "conjunction")[:, :, 0, 0]
    np.testing.assert_array_equal(bare[0], [3, 4, 8, 9, 13, 14])
    np.testing.assert_array_equal(conjunction[0], [1, 2, 6, 7, 11, 12])
    assert exact_sign_flip(np.ones(15), "greater") == 1 / (2**15)


def test_behavior_activation_crosscheck_fails_early() -> None:
    validate_crosscheck({"rows": 540, "token_count_mismatches": 0, "max_abs_logprob_difference": 1e-5}, 2)
    try:
        validate_crosscheck({"rows": 540, "token_count_mismatches": 0, "max_abs_logprob_difference": 2e-4}, 2)
    except RuntimeError as error:
        assert "log-probability" in str(error)
    else:
        raise AssertionError("A bad behavior/activation cross-check must fail immediately")


def test_gpu_memory_observation_maps_pid_to_physical_device(monkeypatch) -> None:
    uuid = "GPU-test-uuid"

    def fake_run(command, **_kwargs):
        if "--query-compute-apps=gpu_uuid,pid,used_memory" in command:
            return SimpleNamespace(stdout=f"{uuid}, {os.getpid()}, 8644\n{uuid}, 999, 100\n")
        if "--query-gpu=uuid,memory.used" in command:
            return SimpleNamespace(stdout=f"{uuid}, 8760\nGPU-other, 14000\n")
        raise AssertionError(command)

    monkeypatch.setattr(behavior_runner.subprocess, "run", fake_run)
    assert behavior_runner.gpu_memory_observation() == (8644, 8760)


def test_subspace_add_uses_a_layerwise_vector_not_the_basis_matrix() -> None:
    basis = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    held = np.asarray([[2.0, 3.0, 5.0, 7.0], [11.0, 13.0, 17.0, 19.0]])
    add_vectors = subspace_add_vectors(basis, held)
    assert add_vectors.shape == (2, 4)
    controller = DirectionHooks(
        None,
        basis,
        "add",
        selected_layer=0,
        magnitude=2.0,
        add_vectors=add_vectors,
    )
    output = controller.transform(torch.zeros(1, 3, 4), hidden_index=0)
    assert output.shape == (1, 3, 4)
    assert torch.isfinite(output).all()
    assert torch.linalg.vector_norm(output[0, 0]).item() == pytest.approx(2.0)
