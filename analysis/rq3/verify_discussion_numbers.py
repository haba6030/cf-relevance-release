#!/usr/bin/env python3
"""Recompute the geometry numbers cited in the Discussion and Appendix from the raw activations (CPU).

`analyze_rq3_subspace_k.py` hard-codes `KS = (1, 2, 3, 5)` and so cannot produce the k=14 row of the
appendix table or `push_cos_k1_k14`. No stored output held the LOO cosine (74.3°) either.
This script computes both. Existing files are not modified.

    python3 analysis/rq3/verify_discussion_numbers.py \
        --input-dir results/rq3_intervention_final_20260827

The basis is built as in `analyze_rq3_subspace_k.py`. Capture at k=1,2,3,5 matches the stored values
to all decimals; only the random columns differ slightly because the RNG stream differs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

OPTIONS = (0, 1, 2)
CONJ_H3 = 2
KS = (1, 2, 3, 5, 14)


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
    unit = lambda v: v / np.linalg.norm(v)
    rows = []
    for _, cell in gate.iterrows():
        slug = cell.model_id.replace('/', '__')
        layer = int(pd.Series(json.loads(cell.selected_layers)).mode().iloc[0])
        arr = np.load(run / 'activations' / slug / f'task{int(cell.task)}_cellmean.npy', mmap_mode='r')
        contrast = np.asarray(arr[:, 2][:, OPTIONS, layer].mean(axis=1)
                              - arr[:, 1][:, OPTIONS, layer].mean(axis=1), dtype=np.float64)
        held_all = np.asarray(arr[:, 2, CONJ_H3, layer] - arr[:, 1, CONJ_H3, layer], dtype=np.float64)
        n, d = contrast.shape

        capture = {k: [] for k in KS}
        capture_random = {k: [] for k in KS}
        tilt = {k: [] for k in KS if k > 1}
        loo_cos = []
        for i in range(n):
            train = [j for j in range(n) if j != i]
            train_contrast, held = contrast[train], held_all[i]
            energy = (held ** 2).sum()
            bases = {k: basis_k(train_contrast, k) for k in KS}
            for k, basis in bases.items():
                capture[k].append(((basis @ held) ** 2).sum() / energy)
                capture_random[k].append(
                    ((orthonormal(rng.normal(size=(k, d)))[:k] @ held) ** 2).sum() / energy)
            push = {k: bases[k].T @ (bases[k] @ held) for k in KS}
            for k in tilt:
                tilt[k].append(unit(push[1]) @ unit(push[k]))
            # the appendix's 74.3°: cosine between the 14-item training mean axis and the held-out item's own contrast.
            # the held-out side is also the three-option mean contrast (the h3-only contrast gives 0.385, 67.4°).
            loo_cos.append(unit(train_contrast.mean(axis=0)) @ unit(contrast[i]))

        record = dict(model=str(cell.model).split('/')[-1], task=int(cell.task), layer=layer, d=d, n=n,
                      loo_cos_mean=float(np.mean(loo_cos)),
                      loo_cos_negative_folds=int(np.sum(np.array(loo_cos) < 0)))
        for k in KS:
            record[f'heldout_capture_k{k}'] = float(np.mean(capture[k]))
            record[f'heldout_capture_random_k{k}'] = float(np.mean(capture_random[k]))
        for k in tilt:
            record[f'push_cos_k1_k{k}'] = float(np.mean(tilt[k]))
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True, type=Path)
    ap.add_argument('--suffix', default='_k14_verify')
    a = ap.parse_args()
    f = analyze(a.input_dir)
    out = a.input_dir / 'analysis' / f'rq3x_subspace_k{a.suffix}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(out, index=False)

    print(f'{len(f)} cells\n')
    print('appendix table, recomputed')
    print(f"{'k':>3} {'capture':>9} {'random':>9} {'ratio':>8}")
    for k in KS:
        c = f[f'heldout_capture_k{k}'].mean()
        r = f[f'heldout_capture_random_k{k}'].mean()
        print(f'{k:>3} {c:>9.3f} {r:>9.5f} {c / r:>7.0f}x')
    print()
    for k in KS[1:]:
        print(f'push_cos_k1_k{k} = {f[f"push_cos_k1_k{k}"].mean():.4f}')
    m = f.loo_cos_mean.mean()
    print(f'\nmean LOO cosine {m:.4f} -> {np.degrees(np.arccos(m)):.1f}°'
          f'  · negative folds {int(f.loo_cos_negative_folds.sum())} / {int(f.n.sum())}')
    print(f'\n{out}')


if __name__ == '__main__':
    main()
