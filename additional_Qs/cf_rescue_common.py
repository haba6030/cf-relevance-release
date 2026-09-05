"""Shared definitions for the RQ3 CF rescue/knockout follow-up.

The follow-up keeps the 165-item RQ3 harness but changes the causal question.
For each held-out item, directions are estimated on the other 164 items at the
decoder-selected layer.  ``R_given_PC`` is the activation DiD mean after the
P/C span is projected out; ``PC_given_R`` is the shared-PC covariance direction
after the R mean direction is projected out.  These are exactly the two vector
controls reported by ``15_rpc_vector_control.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJ = Path(__file__).resolve().parents[1]
FRONT = PROJ / "additional_Qs" / "rq3_run_server"
STIMULI = PROJ / "additional_Qs" / "stimuli_RQ3_selected165.csv"
SEED = 20260830

MODELS = (
    "Qwen/Qwen2.5-Math-1.5B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "microsoft/Phi-4-mini-instruct",
    "microsoft/Phi-4-mini-reasoning",
    "google/gemma-3-4b-it",
)

N, I, R = 0, 1, 2
H2, H3 = 1, 2
ADD_ALPHAS = (0.5, 1.0, 2.0)
# dose=2 would reflect a coordinate rather than ablate it, so it is excluded.
ABL_DOSES = (0.5, 1.0)
DIRECTION_NAMES = (
    "R_given_PC", "PC_given_R",
    "random_R_given_PC", "random_PC_given_R",
    "shuffle_R_given_PC", "shuffle_PC_given_R",
)
CONTRACT = "rq3-cf-rescue-knockout-crossframe-v1"


def unit(value: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-10:
        raise RuntimeError(f"Degenerate direction {name}: norm={norm}")
    return value / norm


def orthonormal_rows(rows: list[np.ndarray]) -> np.ndarray:
    basis: list[np.ndarray] = []
    for row in rows:
        vector = np.asarray(row, dtype=np.float64).copy()
        for existing in basis:
            vector -= float(vector @ existing) * existing
        norm = float(np.linalg.norm(vector))
        if norm > 1e-10:
            basis.append(vector / norm)
    if not basis:
        return np.zeros((0, len(rows[0])), dtype=np.float64)
    return np.stack(basis)


def remove_span(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    return value - basis.T @ (basis @ value) if len(basis) else value


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def behavior_path(model_id: str) -> Path:
    return FRONT / "behavior" / f"rq3_behavior_{safe_name(model_id)}.csv"


def load_stimuli() -> pd.DataFrame:
    data = pd.read_csv(STIMULI)
    labels = data.label.drop_duplicates().tolist()
    if len(labels) != 165 or len(data) != 495 or set(data.prefix) != {"N", "I", "R"}:
        raise RuntimeError("The rescue run requires the locked 165 x {N,I,R} stimulus grid")
    return data


def baseline_rows(model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return prompt rows and frame-mean selection scores from the locked run."""
    data = pd.read_csv(behavior_path(model_id))
    required = {
        "label", "task", "prefix", "frame", "qvar", "prompt_contract",
        "lp_base", "nt_base", "lp_conj_h2", "nt_conj_h2",
        "lp_conj_h3", "nt_conj_h3",
    }
    if required - set(data):
        raise RuntimeError(f"Baseline behavior columns missing: {sorted(required - set(data))}")
    if set(data.prompt_contract) != {"rq3-v2-raw-sequence-20260825"}:
        raise RuntimeError("Prompt contract mismatch in baseline behavior")
    data["base"] = data.lp_base / data.nt_base
    data["h2"] = data.lp_conj_h2 / data.nt_conj_h2
    data["h3"] = data.lp_conj_h3 / data.nt_conj_h3
    data["cf"] = data.h3 - data.base
    data["gap"] = data.h3 - data.h2
    selection = (
        data[data.prefix == "R"]
        .groupby(["model_id", "task", "label", "frame"], as_index=False)
        .agg(selection_cf=("cf", "mean"), selection_gap=("gap", "mean"))
    )
    return data, selection


def norm_did(values: np.ndarray) -> np.ndarray:
    """(R-I) at h3 minus (R-I) at h2, one scalar per item."""
    # Norm arrays contain only [h2, h3]; activation arrays additionally contain
    # base at continuation index 0.  Keep the two index systems explicit.
    return ((values[:, R, 1] - values[:, I, 1])
            - (values[:, R, 0] - values[:, I, 0]))


def activation_did(array: np.ndarray, layer: int) -> np.ndarray:
    return np.asarray(
        (array[:, R, H3, layer] - array[:, I, H3, layer])
        - (array[:, R, H2, layer] - array[:, I, H2, layer]),
        dtype=np.float64,
    )


def shared_pc(p: np.ndarray, c: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fold-fitted shared PC scalar for every item; held values use train scaling."""
    p_train, c_train = p[train], c[train]
    p_mean, c_mean = float(p_train.mean()), float(c_train.mean())
    p_sd, c_sd = float(p_train.std(ddof=1)), float(c_train.std(ddof=1))
    if p_sd <= 1e-12 or c_sd <= 1e-12:
        raise RuntimeError("P/C DiD has zero training-fold variance")
    pz = (p - p_mean) / p_sd
    cz = (c - c_mean) / c_sd
    correlation = float(np.corrcoef(pz[train], cz[train])[0, 1])
    orientation = 1.0 if correlation >= 0 else -1.0
    return ((pz + orientation * cz) / np.sqrt(2.0), {
        "pc_corr_train": correlation, "pc_orientation": orientation,
        "p_train_sd": p_sd, "c_train_sd": c_sd,
    })


def fold_directions(
    array: np.ndarray,
    p_values: np.ndarray,
    c_values: np.ndarray,
    train: np.ndarray,
    layer: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Estimate target and constraint-matched controls on one LOIO fold."""
    states = activation_did(array, layer)
    p_did, c_did = norm_did(p_values), norm_did(c_values)
    pc, pc_meta = shared_pc(p_did, c_did, train)
    x = states[train]
    centered = x - x.mean(axis=0, keepdims=True)
    p_centered = p_did[train] - p_did[train].mean()
    c_centered = c_did[train] - c_did[train].mean()
    pc_centered = pc[train] - pc[train].mean()

    v_p = unit(centered.T @ p_centered, name="v_P")
    v_c = unit(centered.T @ c_centered, name="v_C")
    pc_basis = orthonormal_rows([v_p, v_c])
    v_r = unit(x.mean(axis=0), name="v_R")
    r_unique_raw = remove_span(x.mean(axis=0), pc_basis)
    r_unique = unit(r_unique_raw, name="R_given_PC")
    v_pc = centered.T @ pc_centered
    r_basis = v_r[None, :]
    pc_unique_raw = remove_span(v_pc, r_basis)
    pc_unique = unit(pc_unique_raw, name="PC_given_R")

    rng = np.random.default_rng(seed)
    random_r = unit(remove_span(rng.normal(size=x.shape[1]), pc_basis), name="random_R")
    random_pc = unit(remove_span(rng.normal(size=x.shape[1]), r_basis), name="random_PC")
    signs = rng.choice((-1.0, 1.0), size=len(train))
    shuffle_r = unit(remove_span((x * signs[:, None]).mean(axis=0), pc_basis), name="shuffle_R")
    permuted = rng.permutation(pc_centered)
    shuffle_pc = unit(remove_span(centered.T @ permuted, r_basis), name="shuffle_PC")

    directions = {
        "R_given_PC": r_unique,
        "PC_given_R": pc_unique,
        "random_R_given_PC": random_r,
        "random_PC_given_R": random_pc,
        "shuffle_R_given_PC": shuffle_r,
        "shuffle_PC_given_R": shuffle_pc,
    }
    training_states = np.asarray(array[train, :, :, layer], dtype=np.float32)
    scale = float(np.linalg.norm(training_states.std(axis=0, ddof=1).mean(axis=(0, 1))))
    diagnostics = {
        **pc_meta,
        "scale": scale,
        "cos_R_PC": float(abs(v_r @ unit(v_pc, name="v_PC"))),
        "r_residual_norm_ratio": float(
            np.linalg.norm(r_unique_raw) / max(np.linalg.norm(x.mean(axis=0)), 1e-12)
        ),
        "pc_residual_norm_ratio": float(
            np.linalg.norm(pc_unique_raw) / max(np.linalg.norm(v_pc), 1e-12)
        ),
        "cos_unique": float(abs(r_unique @ pc_unique)),
        "pc_basis_rank": float(len(pc_basis)),
    }
    return directions, diagnostics


def selected_layers(gate_file: Path, model_id: str, task: int) -> np.ndarray:
    gate = pd.read_csv(gate_file)
    gate = gate[(gate.model_id == model_id) & (gate.task == task)
                & (gate.family == "conjunction_h3")]
    if len(gate) != 1 or not bool(gate.iloc[0].decode_gate):
        raise RuntimeError(f"No unique passing conjunction_h3 gate: {model_id} task{task}")
    layers = np.asarray(json.loads(gate.iloc[0].selected_layers), dtype=int)
    if len(layers) != 165:
        raise RuntimeError(f"Expected 165 held-item layers, found {len(layers)}")
    return layers
