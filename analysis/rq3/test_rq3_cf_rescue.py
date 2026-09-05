from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "additional_Qs"))

from cf_rescue_common import fold_directions  # noqa: E402
from run_rq3_cf_rescue_gpu import SelectedDirectionHook  # noqa: E402


def synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    n, layers, dim = 28, 3, 24
    array = rng.normal(scale=.1, size=(n, 3, 5, layers, dim)).astype(np.float32)
    p = rng.normal(size=(n, 3, 2)); c = .7 * p + rng.normal(scale=.4, size=(n, 3, 2))
    r_axis = rng.normal(size=dim); r_axis /= np.linalg.norm(r_axis)
    pc_axis = .75 * r_axis + rng.normal(scale=.3, size=dim); pc_axis /= np.linalg.norm(pc_axis)
    p_did = (p[:, 2, 1] - p[:, 1, 1]) - (p[:, 2, 0] - p[:, 1, 0])
    c_did = (c[:, 2, 1] - c[:, 1, 1]) - (c[:, 2, 0] - c[:, 1, 0])
    signal = r_axis[None] + ((p_did + c_did) / 2)[:, None] * pc_axis[None]
    array[:, 2, 2, 1] += signal.astype(np.float32)
    return array, p, c


def test_fold_directions_are_deterministic_unit_and_distinct() -> None:
    array, p, c = synthetic(); train = np.arange(27)
    first, diagnostics = fold_directions(array, p, c, train, 1, 123)
    second, _ = fold_directions(array, p, c, train, 1, 123)
    assert set(first) == set(second)
    for name in first:
        assert np.isclose(np.linalg.norm(first[name]), 1.0)
        assert np.allclose(first[name], second[name])
    assert abs(first["R_given_PC"] @ first["PC_given_R"]) < .999
    assert diagnostics["pc_basis_rank"] == 2
    assert diagnostics["scale"] > 0
    assert 0 <= diagnostics["r_residual_norm_ratio"] <= 1 + 1e-8
    assert 0 <= diagnostics["pc_residual_norm_ratio"] <= 1 + 1e-8


def test_selected_hook_add_respects_per_row_start() -> None:
    hook = SelectedDirectionHook(None, np.array([1.0, 0.0]), 1, "ADD", magnitude=2.0)
    hook.start_positions = [1, 2]
    x = torch.zeros((2, 4, 2))
    y = hook.transform(x)
    assert torch.equal(y[0, :, 0], torch.tensor([0.0, 2.0, 2.0, 2.0]))
    assert torch.equal(y[1, :, 0], torch.tensor([0.0, 0.0, 2.0, 2.0]))
    assert torch.count_nonzero(y[:, :, 1]) == 0


def test_selected_hook_full_ablation_and_partial_dose() -> None:
    x = torch.tensor([[[3.0, 4.0]]])
    full = SelectedDirectionHook(None, np.array([1.0, 0.0]), 1, "ABL", dose=1.0)
    half = SelectedDirectionHook(None, np.array([1.0, 0.0]), 1, "ABL", dose=.5)
    assert torch.allclose(full.transform(x), torch.tensor([[[0.0, 4.0]]]))
    assert torch.allclose(half.transform(x), torch.tensor([[[1.5, 4.0]]]))

