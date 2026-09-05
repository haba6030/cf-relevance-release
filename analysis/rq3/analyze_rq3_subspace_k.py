#!/usr/bin/env python3
"""Diagnostics motivating the intervention subspace dimension k (CPU).

Four quantities are measured.

1. **Is there real rotation?** The share of the per-item contrast matrix D taken by the common axis v_R,
   and the size of the structure that remains.
2. **How many axes are identifiable?** Residual PCs are re-estimated per LOIO fold and the mean |cos|
   across folds measured. The random expectation is sqrt(2/(pi d)).
3. **Does reading saturate with k?** The share of the held-out item's contrast energy captured by
   a k-dimensional subspace fit on the training items, against a random subspace of the same k
   as control.
4. **How much does the ADD push direction tilt with k?** The push magnitude is already
   unit-normalised by `run_rq3_intervention_gpu.py`, so k changes only
   the direction.

The basis is built as in `run_rq3_intervention_gpu.subspace_directions`:
the first vector is v_R and the rest are the top PCs of the training contrast matrix.
Hence k=1 equals `--direction-mode h3_context` exactly.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

OPTIONS = (0, 1, 2)          # base, conj_h2, conj_h3; same as OPTIONS in the intervention script
CONJ_H3 = 2
KS = (1, 2, 3, 5)
RANKS = (1, 2, 3, 4)


def orthonormal(rows: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(rows.T)
    return q.T[np.abs(np.diag(r)) > 1e-9]


def basis_k(train_contrast: np.ndarray, k: int) -> np.ndarray:
    _, _, right = np.linalg.svd(train_contrast, full_matrices=False)
    candidates = np.vstack([train_contrast.mean(axis=0)[None, :], right[: max(k - 1, 0)]])
    return orthonormal(candidates)[:k]


def analyze(run: Path) -> pd.DataFrame:
    gate = pd.read_csv(run / 'analysis' / 'rq3a_context_decoder_robustness.csv')
    gate = gate[(gate.family == 'conjunction_h3')
                & (gate.decode_gate.astype(str).str.lower().isin(['true', '1']))]
    rng = np.random.default_rng(0)
    rows = []
    for _, cell in gate.iterrows():
        slug = cell.model_id.replace('/', '__')
        layer = int(pd.Series(json.loads(cell.selected_layers)).mode().iloc[0])
        arr = np.load(run / 'activations' / slug / f'task{int(cell.task)}_cellmean.npy', mmap_mode='r')
        contrast = np.asarray(arr[:, 2][:, OPTIONS, layer].mean(axis=1)
                              - arr[:, 1][:, OPTIONS, layer].mean(axis=1), dtype=np.float64)
        held_all = np.asarray(arr[:, 2, CONJ_H3, layer] - arr[:, 1, CONJ_H3, layer], dtype=np.float64)
        n, d = contrast.shape

        v_r = contrast.mean(axis=0); v_r /= np.linalg.norm(v_r)
        along = ((contrast @ v_r) ** 2).sum() / (contrast ** 2).sum()
        spectrum = np.linalg.svd(contrast, compute_uv=False)

        capture = {k: [] for k in KS}
        capture_random = {k: [] for k in KS}
        pcs = {r: [] for r in RANKS}
        shares = {r: [] for r in RANKS}
        tilt = {k: [] for k in KS if k > 1}
        for i in range(n):
            train = [j for j in range(n) if j != i]
            train_contrast, held = contrast[train], held_all[i]
            energy = (held ** 2).sum()
            bases = {k: basis_k(train_contrast, k) for k in KS}
            for k, basis in bases.items():
                capture[k].append(((basis @ held) ** 2).sum() / energy)
                capture_random[k].append(((orthonormal(rng.normal(size=(k, d)))[:k] @ held) ** 2).sum() / energy)
            push = {k: bases[k].T @ (bases[k] @ held) for k in KS}
            unit = lambda v: v / np.linalg.norm(v)
            for k in tilt:
                tilt[k].append(unit(push[1]) @ unit(push[k]))
            mean_train = train_contrast.mean(axis=0); mean_train /= np.linalg.norm(mean_train)
            residual = train_contrast - np.outer(train_contrast @ mean_train, mean_train)
            _, s, right = np.linalg.svd(residual, full_matrices=False)
            eig = s ** 2 / (s ** 2).sum()
            for r in RANKS:
                pcs[r].append(right[r - 1]); shares[r].append(eig[r - 1])

        record = dict(model=str(cell.model).split('/')[-1], task=int(cell.task), layer=layer, d=d,
                      common_axis_share=along,
                      spectrum_effective_dim=(spectrum.sum() ** 2) / (spectrum ** 2).sum(),
                      random_cos_expectation=np.sqrt(2 / (np.pi * d)))
        for r in RANKS:
            stacked = np.array(pcs[r]); gram = np.abs(stacked @ stacked.T)
            record[f'pc{r}_fold_stability'] = (gram.sum() - np.trace(gram)) / (n * (n - 1))
            record[f'pc{r}_variance_share'] = float(np.mean(shares[r]))
        for k in KS:
            record[f'heldout_capture_k{k}'] = float(np.mean(capture[k]))
            record[f'heldout_capture_random_k{k}'] = float(np.mean(capture_random[k]))
        for k in tilt:
            record[f'push_cos_k1_k{k}'] = float(np.mean(tilt[k]))
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    frame = analyze(args.input_dir)
    out = args.input_dir / 'analysis' / f'rq3x_subspace_k{args.suffix}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(frame.to_string(index=False))
    print('\nmean over 12 cells')
    print(frame.drop(columns=['model', 'task', 'layer', 'd']).mean().to_string())
    print(f'\n{out}')


if __name__ == '__main__':
    main()
