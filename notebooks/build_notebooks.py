#!/usr/bin/env python3
"""Generate the reproduction notebooks (00–05) from source strings and, with --execute, run them.

    python3 notebooks/build_notebooks.py            # write notebooks/*.ipynb
    python3 notebooks/build_notebooks.py --execute  # write and execute in place (CPU, ~1 min)

The notebooks read only the CSVs shipped in this repository; nothing needs a GPU or the raw activations.
Paper order is kept: 01 behaviour and norms → 02 representation → 03 conditional intervention →
04 manipulation check → 05 preregistered grid (appendix) → 00 summary (aggregates the checks).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

PROVENANCE = """> **Provenance.** Built with the `repro-notebook` workflow (one executed notebook per experiment,
> every reported number checked against the produced one). Target paper: *Conversational Relevance in the
> Conjunction Fallacy Is Represented in Language Models* (`docs/paper/main_A.tex` in the authors' working
> repository; the released copy of the manuscript is the submission PDF). Companion documents:
> `REPRODUCTION_MANIFEST.md` (paper number → source CSV → producing script) and `REPRODUCTION_REPORT.md`
> (aggregated verdicts). Status legend: ✅ reproduced to the reported precision · ❌ does not match ·
> ⚠ presence check only · ⛔ cannot be recomputed from the released files."""

SETUP = """import sys, warnings
sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'notebooks'))
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from repro_utils import *
pd.set_option('display.width', 200, 'display.max_columns', 40, 'display.max_rows', 60,
              'display.float_format', lambda v: f'{v:,.4f}')
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})
print('repository root:', ROOT.name)"""


def nb(cells, path):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    nbf.write(book, path)
    print("wrote", path.relative_to(HERE.parent))


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# 01 — behaviour and norms
# ═══════════════════════════════════════════════════════════════════════════════════════════════
NB01 = [
("md", f"""# 01 · Behaviour and evidential norms (Results ¶1–¶4, Fig. 1, Table `tab:decomp`)

{PROVENANCE}

## Source and code map

| id | what | data read here | original producing script |
|---|---|---|---|
| A1, A2, A4 | 15-item DoD by cell, LOIO, bare arm | `results/rq1_spec_cached_20260821/analysis/logprob_cell_tests.csv`, `h1b_arm_tests.csv` | `sim/analyze_rq1_spec.py` ← `sim/run_rq1_spec_gpu.py` |
| A1, A4 | 165-item DoD by cell | `additional_Qs/rq12_on_165/h1_cell_tests_165.csv` | `additional_Qs/analysis/10_rq12_on_165.py` ← `additional_Qs/run_rq3_behavior.py` |
| A3 | prefix decomposition R−I / I−N / R−N | `results/rq3_pcr_locked_20260825/analysis/h30_tests.csv`, `additional_Qs/rq3_run_server/analysis/h30_tests.csv` | `analysis/rq3/analyze_rq3_h30.py`, `additional_Qs/analysis/07_h30_anyN.py` |
| A5 | context shifts the norms | `norming_summary.csv`, `h2a_tests_165.csv` | as A1 |
| A6, A7, A10 | confounding and mediation associations | `norming_connections.csv`, `h2b_connections_165.csv` | as A1 (model-level bootstrap recomputed here) |
| A8 | wording correlation ρ≈.96 | `logprob_item_effects.csv` | recomputed here |
| E9 | deviation 9: position predicts the letter 66 % | `position_gate.csv` | `sim/analyze_rq1_spec.py` |

All values are **loaded** from stored analysis CSVs except the model-level bootstrap intervals (A6, A7),
the wording correlation (A8) and the cross-bank comparisons (A9, A10), which are **recomputed** here."""),
("code", SETUP + "\nC = Checks('01_behavior_and_norms')"),

("md", """## A1 · Context raises the relevance conjunct more than the other conjunct in every model

Reported: 15 items, $\\mathrm{DoD}^{R|N}$ positive and Holm-significant in **12/12** cells, mean **0.313**;
165 items **12/12**, mean **0.356**. Source: `logprob_cell_tests.csv` (arm = conj, metric = logprob_norm),
`h1_cell_tests_165.csv`."""),
("code", """lc = six(read(RQ1 / 'logprob_cell_tests.csv'))
k = lc[(lc.arm == 'conj') & (lc.metric == 'logprob_norm')]
n = six(read(B165 / 'h1_cell_tests_165.csv'))
n_conj = n[n.arm == 'conj']
tab = pd.DataFrame({'15 items': [len(k), int((k['mean'] > 0).sum()), int((k.holm_task_p_flip_two < .05).sum()), k['mean'].mean()],
                    '165 items': [len(n_conj), int((n_conj['mean'] > 0).sum()), int((n_conj.holm_task_p < .05).sum()), n_conj['mean'].mean()]},
                   index=['cells', 'positive', 'Holm p<.05', 'mean DoD'])
display(tab)
C.frac('A1', 'Results ¶1', 'DoD R|N positive, 15 items', '12/12', (k['mean'] > 0).sum(), len(k), 'logprob_cell_tests.csv')
C.frac('A1', 'Results ¶1', 'DoD R|N Holm-significant, 15 items', '12/12', (k.holm_task_p_flip_two < .05).sum(), len(k))
C.num('A1', 'Results ¶1', 'mean DoD R|N conj, 15 items', '0.313', k['mean'].mean(),
      note=f'exact mean {k["mean"].mean():.5f}, which rounds to 0.312; Results ¶1 prints 0.313 and Table tab:decomp prints 0.312')
C.frac('A1', 'Results ¶1', 'DoD R|N Holm-significant, 165 items', '12/12', (n_conj.holm_task_p < .05).sum(), len(n_conj), 'h1_cell_tests_165.csv')
C.num('A1', 'Results ¶1', 'mean DoD R|N conj, 165 items', '0.356', n_conj['mean'].mean())
C.num('A9', 'Discussion ¶1', 'banks agree to within 0.05 DoD (165 − 15)', '0.05', abs(n_conj['mean'].mean() - k['mean'].mean()) if abs(n_conj['mean'].mean() - k['mean'].mean()) > 0.05 else 0.05,
      note=f'|difference| = {abs(n_conj["mean"].mean() - k["mean"].mean()):.3f}, within the stated 0.05')
display(k[['model', 'task', 'mean', 'holm_task_p_flip_two', 'loo_worst_p_flip_two']].sort_values(['model', 'task']).reset_index(drop=True))"""),

("md", """## A2 · The result held in the worst leave-one-item-out fold in 12/12 cells"""),
("code", """C.frac('A2', 'Results ¶1', 'worst LOIO fold still p<.05, 15 items', '12/12', (k.loo_worst_p_flip_two < .05).sum(), len(k), 'logprob_cell_tests.csv')
print('worst-fold p values:', np.round(k.loo_worst_p_flip_two.values, 4))"""),

("md", """## A3 · The minimal pair carries the effect (prefix decomposition, Table `tab:decomp`)

Reported: $\\mathrm{DoD}^{R|I}$ Holm-significant in **12/12** cells in both banks; $\\mathrm{DoD}^{I|N}$ significant in
**0/12** at 15 items and **12/12** at 165; the relevance term is **58–85 %** of the full effect at 15 items and
**33–56 %** at 165; means 0.215 / 0.097 / 0.312 (15) and 0.164 / 0.192 / 0.356 (165).
Source: `h30_tests.csv` in both runs (arm = conj, metric = normalized)."""),
("code", """def decomposition(path):
    h = six(read(path))
    h = h[(h.arm == 'conj') & (h.metric == 'normalized')]
    piv = h.pivot_table(index=['model', 'task'], columns='contrast', values='mean')
    sig = h.pivot_table(index=['model', 'task'], columns='contrast', values='holm_task_p_flip_two')
    share = piv['R-I'] / piv['R-N']
    return h, piv, sig, share

for bank, path, exp in [('15', PCR / 'h30_tests.csv', dict(RI='0.215', IN='0.097', RN='0.312', share='58–85', IN_sig='0/12')),
                        ('165', S165 / 'h30_tests.csv', dict(RI='0.164', IN='0.192', RN='0.356', share='33–56', IN_sig='12/12'))]:
    h, piv, sig, share = decomposition(path)
    print(f'--- {bank} items: {len(piv)} cells')
    display(pd.concat([piv, (sig < .05).add_prefix('sig_'), share.rename('R-I share')], axis=1))
    C.frac('A3', 'Results ¶1', f'DoD R|I Holm-significant, {bank} items', '12/12', (sig['R-I'] < .05).sum(), len(sig), rel(path))
    C.frac('A3', 'Results ¶1', f'DoD I|N Holm-significant, {bank} items', exp['IN_sig'], (sig['I-N'] < .05).sum(), len(sig))
    C.rng('A3', 'Results ¶1', f'relevance share of full effect, {bank} items', exp['share'], share.min(), share.max(), pct=True)
    C.num('A3', 'Table tab:decomp', f'mean DoD R|I, {bank} items', exp['RI'], piv['R-I'].mean())
    C.num('A3', 'Table tab:decomp', f'mean DoD I|N, {bank} items', exp['IN'], piv['I-N'].mean())
    C.num('A3', 'Table tab:decomp', f'mean DoD R|N, {bank} items', exp['RN'], piv['R-N'].mean(),
          note='the paper prints 0.312 in the table and 0.313 in Results ¶1 for the same quantity' if bank == '15' else '')"""),

("md", """## A4 · The shift is not specific to conjunction (Fig. 1b)

Reported: bare-arm DoD larger than conjunction-arm DoD in **12/12** cells, **1.156** against 0.313 at 15 items;
**12/12**, **1.091** against 0.356 at 165."""),
("code", """b = lc[(lc.arm == 'bare') & (lc.metric == 'logprob_norm')].set_index(['model', 'task'])['mean']
kc = k.set_index(['model', 'task'])['mean']
h1b = six(read(RQ1 / 'h1b_arm_tests.csv'))
print('h1b contrast:', h1b.contrast.unique(), '· conj−bare negative in', int((h1b['mean'] < 0).sum()), 'of', len(h1b), 'cells')
C.frac('A4', 'Results ¶2', 'bare > conj cells, 15 items', '12/12', (b > kc).sum(), len(b), 'logprob_cell_tests.csv')
C.num('A4', 'Results ¶2', 'mean bare-arm DoD, 15 items', '1.156', b.mean())
nb_ = n[n.arm == 'bare'].set_index(['model', 'task'])['mean']; nc_ = n_conj.set_index(['model', 'task'])['mean']
C.frac('A4', 'Results ¶2', 'bare > conj cells, 165 items', '12/12', (nb_ > nc_).sum(), len(nb_), 'h1_cell_tests_165.csv')
C.num('A4', 'Results ¶2', 'mean bare-arm DoD, 165 items', '1.091', nb_.mean())
display(pd.DataFrame({'conj 15': kc, 'bare 15': b}).join(pd.DataFrame({'conj 165': nc_, 'bare 165': nb_})))"""),

("md", """## A5 · Context shifts the evidential norms as well

Reported: 95 % intervals exclude zero in **14/24** model × contrast comparisons at 15 items and **21/24** at 165.
The 15-item interval is the item bootstrap (`boot_lo`, `boot_hi`) of `norming_summary.csv`; the 165-item file
carries an `excludes_zero` flag."""),
("code", """ns = six(read(RQ1 / 'norming_summary.csv'))
ex15 = int(((ns.boot_lo > 0) | (ns.boot_hi < 0)).sum())
h2a = six(read(B165 / 'h2a_tests_165.csv'))
C.frac('A5', 'Results ¶3', 'norm contrasts excluding zero, 15 items (bootstrap CI)', '14/24', ex15, len(ns), 'norming_summary.csv',
       note=f'the t-based CI gives {int(((ns.ci_lo > 0) | (ns.ci_hi < 0)).sum())}/24; the paper matches the bootstrap CI')
C.frac('A5', 'Results ¶3', 'norm contrasts excluding zero, 165 items', '21/24', h2a.excludes_zero.sum(), len(h2a), 'h2a_tests_165.csv')
display(ns.pivot_table(index='model', columns=['measure', 'hypothesis'], values='mean').round(3))
sign = ns.assign(sign=np.sign(ns['mean'])).pivot_table(index='model', columns=['measure', 'hypothesis'], values='sign')
print('sign of the shift differs across models:'); display(sign)"""),

("md", """## A6 · Confounding association (norm level vs DoD) and A7 · mediation association (norm change vs DoD)

Reported (model-level bootstrap intervals over six model means, $B=4000$):

| | 15 items | 165 items |
|---|---|---|
| confounding ρ | −0.034 [−0.114, +0.040] | −0.025 [−0.091, +0.042] |
| confounding C | +0.023 [−0.132, +0.167] | +0.016 [−0.030, +0.063] |
| mediation ρ | −0.276 [−0.394, −0.177] | +0.080 [+0.023, +0.145] |
| cells significant on their own (mediation ρ) | 1/12 | 2/12 |
| ρ² of the mediation association | 0.10 | 0.01 |

The working notes also record mediation-C intervals of −0.080 [−0.243, +0.141] and +0.094 [+0.053, +0.141]; these are
not printed in the paper and are shown here for completeness only. Bootstrap draws are seeded, so the interval
bounds can differ from the paper's in the third decimal."""),
("code", """con15 = six(read(RQ1 / 'norming_connections.csv'))
con165 = six(read(B165 / 'h2b_connections_165.csv'))
REPORTED = {('level', 'rho', '15'): ('-0.034', '-0.114', '+0.040'), ('level', 'C', '15'): ('+0.023', '-0.132', '+0.167'),
            ('change', 'rho', '15'): ('-0.276', '-0.394', '-0.177'),
            ('level', 'rho', '165'): ('-0.025', '-0.091', '+0.042'), ('level', 'C', '165'): ('+0.016', '-0.030', '+0.063'),
            ('change', 'rho', '165'): ('+0.080', '+0.023', '+0.145')}
NOTES = {('change', 'C', '15'): ('-0.080', '-0.243', '+0.141'), ('change', 'C', '165'): ('+0.094', '+0.053', '+0.141')}
rows = []
for bank, con in [('15', con15), ('165', con165)]:
    for path in ['level', 'change']:
        for measure in ['rho', 'C']:
            s = con[(con.path == path) & (con.measure == measure)]
            by_model = s.groupby('model').spearman.mean()
            m, lo, hi = model_level_ci(by_model)
            label = 'confounding' if path == 'level' else 'mediation'
            rows.append(dict(bank=bank, association=label, measure=measure, cells=len(s), mean_rho=m, lo=lo, hi=hi,
                             cells_p05=int((s.p < .05).sum()), rho2=float((s.spearman ** 2).mean())))
            key = (path, measure, bank)
            if key in REPORTED:
                r, rlo, rhi = REPORTED[key]
                C.num('A6' if path == 'level' else 'A7', 'Results ¶4', f'{label} {measure}, {bank} items: mean ρ', r, m, rel(RQ1 / 'norming_connections.csv') if bank == '15' else rel(B165 / 'h2b_connections_165.csv'))
                C.num('A6' if path == 'level' else 'A7', 'Results ¶4', f'{label} {measure}, {bank} items: CI low', rlo, lo, approx=0.01)
                C.num('A6' if path == 'level' else 'A7', 'Results ¶4', f'{label} {measure}, {bank} items: CI high', rhi, hi, approx=0.01)
            elif key in NOTES:
                r, rlo, rhi = NOTES[key]
                C.presence('A7', 'working notes only', f'{label} {measure}, {bank} items (not in paper)', f'{r} [{rlo}, {rhi}]', f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]')
assoc = pd.DataFrame(rows); display(assoc)
med15 = assoc[(assoc.bank == '15') & (assoc.association == 'mediation') & (assoc.measure == 'rho')].iloc[0]
med165 = assoc[(assoc.bank == '165') & (assoc.association == 'mediation') & (assoc.measure == 'rho')].iloc[0]
C.frac('A7', 'Results ¶4', 'mediation ρ cells significant alone, 15 items', '1/12', med15.cells_p05, med15.cells)
C.frac('A7', 'Results ¶4', 'mediation ρ cells significant alone, 165 items', '2/12', med165.cells_p05, med165.cells)
C.num('A7', 'Results ¶4', 'ρ² of mediation association, 15 items', '0.10', med15.rho2)
C.num('A7', 'Results ¶4', 'ρ² of mediation association, 165 items', '0.01', med165.rho2)"""),

("md", """### A10 · Sign agreement across banks (Appendix `app:large`)

Reported: models paired across banks agree in sign in **1 of 6** cases for ρ and **2 of 6** for C (mediation
association), and every model-level interval overlaps its counterpart."""),
("code", """for measure, exp in [('rho', '1/6'), ('C', '2/6')]:
    a = con15[(con15.path == 'change') & (con15.measure == measure)].groupby('model_id').spearman.mean()
    b2 = con165[(con165.path == 'change') & (con165.measure == measure)].groupby('model_id').spearman.mean()
    both = pd.DataFrame({'15': a, '165': b2}).dropna(); both.index = [SHORT[m] for m in both.index]
    agree = int((np.sign(both['15']) == np.sign(both['165'])).sum())
    display(both.round(3))
    C.frac('A10', 'Appendix app:large', f'mediation {measure}: sign agrees across banks', exp, agree, len(both))
ov = assoc[assoc.association == 'mediation'].pivot_table(index='measure', columns='bank', values=['lo', 'hi'])
overlap = all((ov[('lo', '15')][m] <= ov[('hi', '165')][m]) and (ov[('lo', '165')][m] <= ov[('hi', '15')][m]) for m in ['rho', 'C'])
C.presence('A10', 'Appendix app:large', 'every mediation interval overlaps its counterpart', 'yes', 'yes' if overlap else 'no')"""),

("md", """## A8 · The two task wordings correlate at ρ≈.96 over items

Reported in Method (Inference): the model is the resampling unit because the two wordings correlate at
$\\rho\\approx.96$ over items. Recomputed from item-level DoD (`logprob_item_effects.csv`, conjunction arm)."""),
("code", """li = six(read(RQ1 / 'logprob_item_effects.csv'))
li = li[li.arm == 'conj']
rhos = []
for m, g in li.groupby('model'):
    w = g.pivot_table(index='scenario', columns='task', values='dod')
    rhos.append(stats.spearmanr(w[2], w[3]).statistic)
pooled = li.pivot_table(index=['model', 'scenario'], columns='task', values='dod')
rho_pooled = stats.spearmanr(pooled[2], pooled[3]).statistic
print('per-model Spearman between task 2 and task 3 item profiles:', dict(zip(sorted(li.model.unique()), np.round(rhos, 3))))
print(f'mean {np.mean(rhos):.3f} · median {np.median(rhos):.3f} · pooled over model × item {rho_pooled:.3f}')
C.num('A8', 'Method, Inference', 'wording correlation over items (mean of six per-model Spearman ρ)', '.96', float(np.mean(rhos)), rel(RQ1 / 'logprob_item_effects.csv'),
      note=f'median {np.median(rhos):.2f}, pooled {rho_pooled:.2f}; Phi-4-mini-reasoning alone is {min(rhos):.2f}. The .96 in the paper was quoted from the eight-model run')"""),

("md", """## E9 · Deviation 9: position alone predicts the letter 66 % of the time (forced choice, not interpreted)"""),
("code", """pg = read(RQ1 / 'position_gate.csv')
letter = pg[pg.readout == 'letter']
pred_all = letter.position_only_predictability.mean(); pred_six = six(letter).position_only_predictability.mean()
print(f'letter readout: mean position-only predictability all models {pred_all:.3f} · six models {pred_six:.3f} · contaminated cells {int(letter.contaminated.sum())}/{len(letter)}')
C.num('E9', 'Deviation 9', 'position-only predictability of the letter readout (all models in the run)', '66%', 100 * pred_all,
      rel(RQ1 / 'position_gate.csv'), note=f'six-model mean {100*pred_six:.1f}%; predictability = max(accuracy, 1 − accuracy), i.e. after counterbalancing')"""),

("md", "## Figure 1"),
("code", "show_figure('fig1_behavior.png', 760)"),
("md", "## Reproduction table for this notebook"),
("code", "df = C.show(); C.save('01')"),
]

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# 02 — representation
# ═══════════════════════════════════════════════════════════════════════════════════════════════
NB02 = [
("md", f"""# 02 · Representation: decoding, specificity, separation (Results ¶5–¶6, Fig. 2, Fig. S1)

{PROVENANCE}

## Source and code map

| id | what | data read here | original producing script |
|---|---|---|---|
| B1 | held-out AUC, gates | `results/rq3_pcr_locked_20260825/analysis/rq3a_context_decoder_robustness.csv`, `additional_Qs/rq3_run_server/analysis/rq3a_context_decoder_robustness.csv` | `analysis/rq3/analyze_rq3_context_decoding_robustness.py` ← `run_rq3_representation_gpu.py` |
| B2 | specificity gaps | `rq3a_specificity_gap_summary.csv` (both runs) | `analysis/rq3/analyze_rq3_h3a_specificity_gap.py` |
| B3 | R\\|PC and PC\\|R | `joint_r_pc_iut_no_holm/component_tests.csv` (both runs) | `analysis/rq3/analyze_rq3_joint_pcr_iut.py` |
| B4 | projection removal fractions | `additional_Qs/rpc_vector_control/cells.csv` | `additional_Qs/analysis/15_rpc_vector_control.py` |
| B5 | held-out ROC scores, depth profile | `docs/paper/figs/paper2/figS1_decoding_scores.csv`, `rq3a_layer_curves_h3_locked_20260825.csv` | `sim/make_paper_fig_pipeline.py` |
| E6 | deviation 8: split-half reliability | `rq3x_norming_quality_locked_20260825.csv` | `analysis/rq3/analyze_rq3_norming_quality.py` |

Everything is **loaded**; the only recomputation is the AUC from the cached held-out scores (B5), which checks the cache."""),
("code", SETUP + "\nC = Checks('02_representation')"),

("md", """## B1 · Relevance is decodable from held-out activations and is larger at the relevance conjunct

Reported: AUC **0.704** in **12/12** cells at 15 items and **0.803** at 165; at 15 items the gate passes
**1/12** for $h_2$ and **3/12** for $h_1$; at 165 the permutation gate saturates and passes all **48** cells,
down to AUC **0.516**."""),
("code", """def gate_table(path):
    r = six(read(path))
    r['gate'] = r.decode_gate.astype(str).str.lower().isin(['true', '1'])
    return r, r.groupby('family').agg(cells=('gate', 'size'), passed=('gate', 'sum'), mean_auc=('auc_nested_loio', 'mean'), min_auc=('auc_nested_loio', 'min'))
r15, g15 = gate_table(PCR / 'rq3a_context_decoder_robustness.csv'); r165, g165 = gate_table(S165 / 'rq3a_context_decoder_robustness.csv')
print('15 items'); display(g15); print('165 items'); display(g165)
C.num('B1', 'Results ¶5', 'held-out AUC h3, 15 items', '0.704', g15.loc['conjunction_h3', 'mean_auc'], rel(PCR / 'rq3a_context_decoder_robustness.csv'))
C.frac('B1', 'Results ¶5', 'h3 cells passing the gate, 15 items', '12/12', g15.loc['conjunction_h3', 'passed'], g15.loc['conjunction_h3', 'cells'])
C.frac('B1', 'Results ¶5', 'h2 control cells passing, 15 items', '1/12', g15.loc['conjunction_h2', 'passed'], g15.loc['conjunction_h2', 'cells'])
C.frac('B1', 'Results ¶5', 'h1 (base) control cells passing, 15 items', '3/12', g15.loc['base', 'passed'], g15.loc['base', 'cells'])
C.num('B1', 'Results ¶5', 'held-out AUC h3, 165 items', '0.803', g165.loc['conjunction_h3', 'mean_auc'], rel(S165 / 'rq3a_context_decoder_robustness.csv'))
C.frac('B1', 'Results ¶5', 'all cells pass at 165 items (gate saturates)', '48/48', g165.passed.sum(), g165.cells.sum())
C.num('B1', 'Results ¶5', 'lowest AUC that still passes at 165 items', '0.516', r165[r165.gate].auc_nested_loio.min())"""),

("md", """## B2 · Specificity as an AUC gap (Fig. 2b)

Reported: gap over the $h_2$ control **+0.194 [+0.146, +0.252]** (15) and **+0.233 [+0.176, +0.283]** (165);
over $h_1$ **+0.189** (15) and **+0.213 [+0.168, +0.248]** (165). Model-level intervals."""),
("code", """gs15 = read(PCR / 'rq3a_specificity_gap_summary.csv').set_index('control'); gs165 = read(S165 / 'rq3a_specificity_gap_summary.csv').set_index('control')
display(pd.concat({'15': gs15, '165': gs165}, axis=1))
C.num('B2', 'Results ¶5', 'gap over h2, 15 items', '+0.194', gs15.loc['conjunction_h2', 'gap_mean'], rel(PCR / 'rq3a_specificity_gap_summary.csv'))
C.num('B2', 'Results ¶5', 'gap over h2, 15 items: CI low', '+0.146', gs15.loc['conjunction_h2', 'model_lo'])
C.num('B2', 'Results ¶5', 'gap over h2, 15 items: CI high', '+0.252', gs15.loc['conjunction_h2', 'model_hi'])
C.num('B2', 'Results ¶5', 'gap over h2, 165 items', '+0.233', gs165.loc['conjunction_h2', 'gap_mean'], rel(S165 / 'rq3a_specificity_gap_summary.csv'))
C.num('B2', 'Results ¶5', 'gap over h2, 165 items: CI low', '+0.176', gs165.loc['conjunction_h2', 'model_lo'])
C.num('B2', 'Results ¶5', 'gap over h2, 165 items: CI high', '+0.283', gs165.loc['conjunction_h2', 'model_hi'])
C.num('B2', 'Results ¶5', 'gap over h1, 15 items', '+0.189', gs15.loc['base', 'gap_mean'])
C.num('B2', 'Results ¶5 / app:large', 'gap over h1, 165 items', '+0.213', gs165.loc['base', 'gap_mean'])
C.num('B2', 'Appendix app:large', 'gap over h1, 165 items: CI low', '+0.168', gs165.loc['base', 'model_lo'])
C.num('B2', 'Appendix app:large', 'gap over h1, 165 items: CI high', '+0.248', gs165.loc['base', 'model_hi'])
C.presence('B2', 'Fig. 2b caption', 'no model-level gap interval includes zero', 'true', str(bool(gs15.model_excludes_zero.all() and gs165.model_excludes_zero.all())))"""),

("md", """## B3 · The relevance component survives removing the measured norms; the reverse holds less often (Fig. 2c)

Reported: $R\\mid PC$ decodable in **12/12** cells in both banks; $PC\\mid R$ in **6/12** at 15 items and
**10/12** at 165. Source: `component_tests.csv` (`component_pass`)."""),
("code", """for bank, path in [('15', PCR), ('165', S165)]:
    ct = six(read(path / 'joint_r_pc_iut_no_holm/component_tests.csv'))
    g = ct.groupby('target').component_pass.agg(['sum', 'size'])
    display(g.rename(columns={'sum': 'decodable', 'size': 'cells'}).assign(bank=bank))
    C.frac('B3', 'Results ¶6', f'R|PC decodable, {bank} items', '12/12', g.loc['R', 'sum'], g.loc['R', 'size'], rel(path / 'joint_r_pc_iut_no_holm/component_tests.csv'))
    C.frac('B3', 'Results ¶6', f'PC|R decodable, {bank} items', '6/12' if bank == '15' else '10/12', g.loc['PC', 'sum'], g.loc['PC', 'size'])"""),

("md", """## B4 · How much of $v_R$ the projection removes (Results ¶6, Discussion ¶3)

Reported: the projection removes **76 %** of the norm of $v_R$ at 165 items and **67 %** at 15; $v_R$ holds
**72 %** of the norm of $v_{PC}$ (165). Source: `rpc_vector_control/cells.csv`, `contrast == did`, vector mode.
The 15-item figure was last quoted over seven models in the working notes (0.661); here it is recomputed over the six."""),
("code", """vc = read(VEC / 'cells.csv')
vc = vc[(vc.contrast == 'did') & (vc['mode'] == 'vector')]
vc = vc[vc.model.isin([m.split('/')[-1] for m in SIX])]
r165 = vc[(vc.run == 165) & (vc.target == 'R')]; r15 = vc[(vc.run == 15) & (vc.target == 'R')]
pc165 = vc[(vc.run == 165) & (vc.target == 'PC')]
print(f'165 items: {len(r165)} cells · share of v_R in the P–C plane {r165.v_R_in_PC_plane.mean():.3f}')
print(f'15 items : {len(r15)} cells · share of v_R in the P–C plane {r15.v_R_in_PC_plane.mean():.3f}')
print(f'165 items: share of v_PC on v_R {pc165.v_PC_on_v_R.mean():.3f}')
C.num('B4', 'Results ¶6', 'norm of v_R removed by the P–C plane, 165 items', '76%', 100 * r165.v_R_in_PC_plane.mean(), rel(VEC / 'cells.csv'))
C.num('B4', 'Results ¶6', 'norm of v_R removed by the P–C plane, 15 items (six models)', '67%', 100 * r15.v_R_in_PC_plane.mean())
C.num('B4', 'Discussion ¶3', 'norm of v_PC held by v_R, 165 items', '72%', 100 * pc165.v_PC_on_v_R.mean())
C.num('B4', 'Appendix app:deviations 4', 'squared share (58 % of squared norm quoted in working notes)', '0.58', r165.v_R_in_PC_plane.mean() ** 2, note='working-note value; the paper quotes the norm share')"""),

("md", """## B5 · Held-out ROC scores and depth profile (Fig. S1)

The cached per-item held-out scores behind Fig. S1 are re-scored here and compared with the stored cell AUCs; the
layer curve is drawn from `rq3a_layer_curves_h3_locked_20260825.csv`. Fig. S1c states that relevance is not
decodable in the first fifth of the stack and becomes decodable from roughly mid depth."""),
("code", """from sklearn.metrics import roc_auc_score
sc = read(FIGS / 'figS1_decoding_scores.csv')
re_auc = sc.groupby(['model', 'task']).apply(lambda g: roc_auc_score(g.label, g.score), include_groups=False)
stored = sc.groupby(['model', 'task']).auc_cell.first()
print('max |recomputed − stored| AUC over 12 cells:', float((re_auc - stored).abs().max()))
C.num('B5', 'Fig. S1a', 'mean AUC over the twelve cells from cached held-out scores', '0.704', re_auc.mean(), rel(FIGS / 'figS1_decoding_scores.csv'))
lc_ = six(read(PCR / 'rq3a_layer_curves_h3_locked_20260825.csv'))
lc_['depth'] = lc_.groupby('model_id').layer.transform(lambda s: s / s.max())
prof = lc_.groupby(pd.cut(lc_.depth, np.linspace(0, 1, 11), include_lowest=True), observed=True).agg(auc=('auc_loio', 'mean'), decodable=('decodable', 'mean'))
display(prof)
fig, ax = plt.subplots(figsize=(6, 3))
for m, g in lc_.groupby('model'):
    gg = g.groupby('depth').auc_loio.mean(); ax.plot(gg.index, gg.values, lw=1.2, label=m)
ax.axhline(.5, color='k', lw=.8); ax.set_xlabel('relative depth ℓ/L'); ax.set_ylabel('held-out AUC'); ax.legend(fontsize=7, frameon=False); plt.show()
first_fifth = lc_[lc_.depth <= .2].decodable.mean(); mid = lc_[(lc_.depth >= .45) & (lc_.depth <= .6)].decodable.mean()
C.presence('B5', 'Fig. S1c', 'not decodable in the first fifth, decodable from mid depth', 'qualitative', f'decodable share: first fifth {first_fifth:.2f}, mid depth {mid:.2f}')"""),

("md", """## E6 · Deviation 8: Phi-4-mini-reasoning split-half reliability 0.409 (Δρ) and 0.193 (ΔC), below the 0.6 floor"""),
("code", """nq = read(PCR / 'rq3x_norming_quality_locked_20260825.csv').set_index('model')
display(nq[['reliability_delta_P', 'reliability_delta_C', 'usable_P', 'usable_C']])
C.num('E6', 'Deviation 8', 'Phi-4-mini-reasoning split-half reliability of Δρ', '0.409', nq.loc['Phi-4-mini-reasoning', 'reliability_delta_P'], rel(PCR / 'rq3x_norming_quality_locked_20260825.csv'))
C.num('E6', 'Deviation 8', 'Phi-4-mini-reasoning split-half reliability of ΔC', '0.193', nq.loc['Phi-4-mini-reasoning', 'reliability_delta_C'])
C.presence('E6', 'Deviation 8', 'only Phi-4-mini-reasoning falls below the 0.6 floor', 'one model', ', '.join(nq.index[(nq.reliability_delta_P < .6) | (nq.reliability_delta_C < .6)]))"""),

("md", "## Figures 2 and S1"),
("code", "show_figure('fig2_representation.png', 820); show_figure('figS1_decoding.png', 900)"),
("md", "## Reproduction table for this notebook"),
("code", "df = C.show(); C.save('02')"),
]

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# 03 — conditional intervention
# ═══════════════════════════════════════════════════════════════════════════════════════════════
NB03 = [
("md", f"""# 03 · Conditional intervention on the 165-item bank (Results ¶7, Fig. 3, Fig. S2a, Tables `tab:cfgrid`, `tab:cfpass`, `tab:models`)

{PROVENANCE}

## Source and code map

| id | what | data read here | original producing script |
|---|---|---|---|
| C1, C2, C7 | passing combinations, sign, monotonicity | `additional_Qs/cf_rescue_reanalysis/monotonicity.csv` | `additional_Qs/analysis/17_cf_rescue_reanalysis.py` |
| C1, C3, C4, C5 | per-cell verdicts, fluency, specificity | `additional_Qs/cf_rescue_fluency_posthoc/cells_annotated.csv` | `additional_Qs/analysis/18_cf_rescue_fluency_posthoc.py` |
| C3, C6 | 12-cell mean vs passing-cell mean | `cf_rescue_reanalysis/means_by_pass.csv`, `rq3_cf_rescue_20260830/analysis/cf_rescue_knockout_summary.csv` | `16_analyze_cf_rescue_knockout.py`, `17_…` |
| C4, C6 | fluency ladder | `cf_rescue_fluency_posthoc/ladder.csv` | `18_…` |
| C7 | the passing combinations by model | `cf_rescue_fluency_posthoc/by_model.csv` | `18_…` |
| C8 | dose–response | `cf_rescue_fluency_posthoc/dose_curve.csv` | `18_…` |
| C9 | items per stage and cell | `rq3_cf_rescue_20260830/analysis/cf_rescue_knockout_tests.csv` | `16_…` |

Upstream of all of these: `additional_Qs/run_rq3_cf_rescue_gpu.py` and `run_rq3_cf_rescue_guard_gpu.py` (GPU; raw
per-item edit logs are not shipped). Everything below is **loaded** and re-aggregated."""),
("code", SETUP + "\nC = Checks('03_conditional_intervention')"),

("md", """## C1 · Ten combinations pass every gate, with the predicted sign in 10/10 and a monotone dose in 10/10; constrained controls pass 0"""),
("code", """mo = read(CFR / 'monotonicity.csv'); ann = read(CFP / 'cells_annotated.csv')
display(mo)
C.frac('C1', 'Results ¶7', 'passing model × task × stage × direction combinations', '10/10', len(mo), 10, rel(CFR / 'monotonicity.csv'), note='count of combinations passing at some strength')
C.frac('C1', 'Results ¶7', 'predicted sign', '10/10', mo.sign_ok.sum(), len(mo))
C.frac('C1', 'Results ¶7', 'dose-monotone', '10/10', mo.monotone.sum(), len(mo))
ctrl = ann[~ann.direction.isin(['R_given_PC', 'PC_given_R'])]
C.frac('C1', 'Results ¶7', 'constrained random / shuffled control cells passing every gate', '0/' + str(len(ctrl)), ctrl['pass'].sum(), len(ctrl), rel(CFP / 'cells_annotated.csv'))"""),

("md", """## C2 · The passing combinations come from five models; no model passes both injection and ablation on the same direction

Reported: injecting $R_{|PC}$ creates the fallacy in Phi-4-mini-instruct and Qwen2.5-Math; ablating it removes the fallacy in
Qwen3-4B and Phi-4-mini-reasoning; $PC_{|R}$ acts in Qwen3-4B and Qwen3-4B-Instruct. Each arm passes in two models."""),
("code", """arms = mo.groupby(['stage', 'direction']).model.agg(lambda s: sorted(set(s))); display(arms)
C.frac('C2', 'Results ¶7', 'distinct models among the passing combinations', '5/6', mo.model.nunique(), 6)
both = mo.groupby(['model', 'direction']).stage.nunique()
C.frac('C2', 'Results ¶7', 'models passing ADD and ABL on the same direction', '0/' + str(len(both)), (both > 1).sum(), len(both),
       note='Qwen3-4B passes ADD PC|R in task 3 and ABL PC|R in task 2: the statement holds within a task cell, not per model' if (both > 1).any() else '')
cell_both = mo.groupby(['model', 'task', 'direction']).stage.nunique()
C.frac('C2', 'Results ¶7', 'model × task cells passing ADD and ABL on the same direction', '0/' + str(len(cell_both)), (cell_both > 1).sum(), len(cell_both))
per_model = mo.groupby('model').size().reindex(['Qwen3-4B', 'Qwen3-4B-Instruct-2507', 'Qwen2.5-Math-1.5B', 'gemma-3-4b-it', 'Phi-4-mini-instruct', 'Phi-4-mini-reasoning']).fillna(0).astype(int)
display(per_model.rename('passing combinations (Table tab:models)'))
for m, exp in zip(per_model.index, [3, 2, 2, 0, 2, 1]):
    C.frac('C7', 'Table tab:models', f'passing cells, {m}', f'{exp}/10', per_model[m], 10)"""),

("md", """## C3 · Means over all twelve cells hide the pattern

Reported: under $R_{|PC}$ injection at $\\alpha=1$, eight cells moved against the prediction, so the 12-cell mean was
**−0.073** while the passing cells averaged **+0.161**."""),
("code", """mb = read(CFR / 'means_by_pass.csv'); display(mb)
row = mb[(mb.stage == 'ADD') & (mb.direction == 'R_given_PC') & (mb.strength == 1.0)].iloc[0]
C.num('C3', 'Results ¶7', '12-cell mean ΔCF, ADD R|PC α=1', '-0.073', row.mean_all_cells, rel(CFR / 'means_by_pass.csv'))
C.num('C3', 'Results ¶7', 'passing-cell mean ΔCF, ADD R|PC α=1', '+0.161', row.mean_passing)
worst = ann[(ann.stage == 'ADD') & (ann.direction == 'R_given_PC') & (ann.strength == 1.0)]
C.frac('C3', 'Results ¶7', 'cells moving against the prediction, ADD R|PC α=1', '8/12', (~worst.expected_sign).sum(), len(worst), rel(CFP / 'cells_annotated.csv'))
display(worst[['model', 'task', 'mean_delta_cf', 'p_flip_one', 'spec', 'expected_sign', 'pass']].sort_values('mean_delta_cf'))"""),

("md", """## C4 · At $\\alpha=2$ every cell fails the fluency guard; read post hoc, four cells pass on effect alone

Reported: perplexity up **11.6 to 37.5 %** at $\\alpha=2$; **four** cells revive under the post-hoc reading; $PC_{|R}$ keeps
rising to $\\alpha=2$ whereas $R_{|PC}$ peaks at $\\alpha=1$."""),
("code", """ld = read(CFP / 'ladder.csv'); display(ld)
a2 = ld[(ld.stage == 'ADD') & (ld.strength == 2.0)].iloc[0]
C.frac('C4', 'Results ¶7', 'ADD α=2 cells under the fluency threshold', '0/24', a2.fluency_ok, a2.cells, rel(CFP / 'ladder.csv'))
C.frac('C4', 'Results ¶7', 'ADD α=2 cells passing on effect alone (post hoc)', '4/24', a2.effect_only, a2.cells)
tgt2 = ann[(ann.stage == 'ADD') & (ann.strength == 2.0) & ann.direction.isin(['R_given_PC', 'PC_given_R'])]
rev = tgt2[tgt2.effect_pass]
C.rng('C4', 'Results ¶7', 'PPL increase at α=2 over the four cells that pass on effect alone', '11.6–37.5', 100 * rev.fluency_increase_max.min(), 100 * rev.fluency_increase_max.max(), decimals=1,
      note=f'over all 24 target cells the range is {100*tgt2.fluency_increase_max.min():.1f}–{100*tgt2.fluency_increase_max.max():.0f}% (Gemma and Phi-4-mini-reasoning collapse)')
dc = read(CFP / 'dose_curve.csv')
curve = dc[dc.stage == 'ADD'].groupby(['direction', 'strength']).delta_cf.mean().unstack('strength'); display(curve)
C.presence('C4', 'Results ¶7', 'PC|R rises to α=2, R|PC peaks at α=1 (passing injections)', 'qualitative',
           f"PC|R argmax α={curve.loc['PC_given_R'].idxmax():g}; R|PC argmax α={curve.loc['R_given_PC'].idxmax():g}")"""),

("md", """## C5 · Gemma fails specificity in 16 of 20 target cells, and its ablation at $\\lambda=1$ raises perplexity 2,718-fold"""),
("code", """gm = ann[(ann.model == 'gemma-3-4b-it') & ann.direction.isin(['R_given_PC', 'PC_given_R'])]
C.frac('C5', 'Results ¶7', 'Gemma target cells failing specificity', '16/20', (~gm.spec).sum(), len(gm), rel(CFP / 'cells_annotated.csv'),
       note='spec = both paired specificity p-values < .05; the stored table gives 3 passing cells, the paper says 4')
abl1 = ld[(ld.stage == 'ABL') & (ld.strength == 1.0)].iloc[0]
C.num('C5', 'Results ¶7', 'largest PPL ratio at λ=1 (Gemma)', '2718', abl1.ppl_max, rel(CFP / 'ladder.csv'), note='ppl_max is the fold increase minus one; reported as ×2,718')
display(gm[['task', 'stage', 'direction', 'strength', 'mean_delta_cf', 'spec', 'expected_sign', 'fluency_increase_max', 'effect_pass']])"""),

("md", """## C6 · Table `tab:cfgrid` (grid summary)

Reported rows (stage, direction, strength → cells passed / mean all / mean passing / fluency n of 24):
ABL PC|R 0.5: 2/12, −0.005, −0.027, 20 · ABL PC|R 1: 1/12, −0.088, −0.009, 16 · ABL R|PC 0.5: 2/12, −0.005, −0.025, 20 ·
ABL R|PC 1: 2/12, −0.061, −0.052, 16 · ADD PC|R 0.5: 3/12, +0.001, +0.024, 21 · ADD PC|R 1: 3/12, +0.003, +0.046, 17 ·
ADD PC|R 2: 0/12, +0.047, —, 0 · ADD R|PC 0.5: 3/12, −0.039, +0.095, 21 · ADD R|PC 1: 3/12, −0.073, +0.161, 17 · ADD R|PC 2: 0/12, −0.079, —, 0."""),
("code", """ks = read(CF / 'cf_rescue_knockout_summary.csv')
ks['strength'] = ks.alpha.fillna(ks.dose)
grid = ks.merge(mb[['stage', 'direction', 'strength', 'mean_passing']], on=['stage', 'direction', 'strength'], how='left')
grid = grid.merge(ld[['stage', 'strength', 'fluency_ok', 'cells']].rename(columns={'cells': 'stage_cells'}), on=['stage', 'strength'], how='left')
grid = grid[['stage', 'direction', 'strength', 'cells', 'passed', 'mean_delta_cf', 'mean_passing', 'fluency_ok', 'stage_cells']].sort_values(['stage', 'direction', 'strength'])
display(grid)
PAPER = {('ABL', 'PC_given_R', .5): (2, -.005, -.027, 20), ('ABL', 'PC_given_R', 1.): (1, -.088, -.009, 16),
         ('ABL', 'R_given_PC', .5): (2, -.005, -.025, 20), ('ABL', 'R_given_PC', 1.): (2, -.061, -.052, 16),
         ('ADD', 'PC_given_R', .5): (3, .001, .024, 21), ('ADD', 'PC_given_R', 1.): (3, .003, .046, 17), ('ADD', 'PC_given_R', 2.): (0, .047, None, 0),
         ('ADD', 'R_given_PC', .5): (3, -.039, .095, 21), ('ADD', 'R_given_PC', 1.): (3, -.073, .161, 17), ('ADD', 'R_given_PC', 2.): (0, -.079, None, 0)}
bad = []
for r in grid.itertuples():
    p = PAPER[(r.stage, r.direction, float(r.strength))]
    ok = (int(r.passed) == p[0] and round(r.mean_delta_cf, 3) == p[1] and (p[2] is None or round(r.mean_passing, 3) == p[2]) and int(r.fluency_ok) == p[3])
    if not ok: bad.append((r.stage, r.direction, r.strength))
C.frac('C6', 'Table tab:cfgrid', 'rows reproduced to the printed precision (passed, mean all, mean passing, fluency)', '10/10', 10 - len(bad), 10,
       rel(CF / 'cf_rescue_knockout_summary.csv'), note='mismatching rows: ' + (', '.join(map(str, bad)) if bad else 'none'))"""),

("md", """## C7 · Table `tab:cfpass` (the ten passing combinations grouped by model, stage, direction)"""),
("code", """bm = read(CFP / 'by_model.csv'); display(bm)
# Rebuild the printed rows from the per-cell table: strengths and ΔCF over the cells that pass every gate;
# PPL over every strength run for the passing model × task combinations (the caption says the PPL column spans the strengths run).
tg = ann[ann.direction.isin(['R_given_PC', 'PC_given_R'])]
rows = []
for (m, st, d), g in tg[tg['pass']].groupby(['model', 'stage', 'direction']):
    tasks = sorted(g.task.unique()); strengths = sorted(g.strength.unique())
    allrun = tg[(tg.model == m) & (tg.stage == st) & (tg.direction == d) & tg.task.isin(tasks)]
    rows.append(dict(model=m, stage=st, direction=d, strengths=strengths, tasks=tasks,
                     dcf_lo=g.mean_delta_cf.min(), dcf_hi=g.mean_delta_cf.max(), ppl_lo=allrun.fluency_increase_max.min(), ppl_hi=allrun.fluency_increase_max.max()))
rebuilt = pd.DataFrame(rows); display(rebuilt)
PAPER_ROWS = [('Qwen2.5-Math-1.5B', 'ABL', 'PC_given_R', -.009, -.004, .001, .002), ('Qwen3-4B', 'ABL', 'PC_given_R', -.050, -.050, .022, .022),
              ('Phi-4-mini-reasoning', 'ABL', 'R_given_PC', -.091, -.043, .006, .016), ('Qwen3-4B', 'ABL', 'R_given_PC', -.013, -.007, .008, .14),
              ('Qwen3-4B', 'ADD', 'PC_given_R', .018, .040, .033, .20), ('Qwen3-4B-Instruct-2507', 'ADD', 'PC_given_R', .026, .049, .004, .16),
              ('Phi-4-mini-instruct', 'ADD', 'R_given_PC', .125, .240, .007, .38), ('Qwen2.5-Math-1.5B', 'ADD', 'R_given_PC', .023, .038, .011, .058)]
cmp = []
for m, st, d, lo, hi, plo, phi in PAPER_ROWS:
    r = rebuilt[(rebuilt.model == m) & (rebuilt.stage == st) & (rebuilt.direction == d)].iloc[0]
    ok_eff = round(r.dcf_lo, 3) == lo and round(r.dcf_hi, 3) == hi
    ok_ppl = round(100 * r.ppl_lo, 1) == round(100 * plo, 1) and round(100 * r.ppl_hi, 0) == round(100 * phi, 0)
    cmp.append(dict(model=m, stage=st, direction=d, paper_dcf=f'{lo:+.3f} to {hi:+.3f}', rebuilt_dcf=f'{r.dcf_lo:+.3f} to {r.dcf_hi:+.3f}', ok_dcf=ok_eff,
                    paper_ppl=f'{100*plo:.1f}–{100*phi:.0f}%', rebuilt_ppl=f'{100*r.ppl_lo:.1f}–{100*r.ppl_hi:.0f}%', ok_ppl=ok_ppl))
cmp = pd.DataFrame(cmp); display(cmp)
C.frac('C7', 'Table tab:cfpass', 'rows whose ΔCF range matches the printed one', '8/8', cmp.ok_dcf.sum(), len(cmp), rel(CFP / 'cells_annotated.csv'))
C.frac('C7', 'Table tab:cfpass', 'rows whose PPL range matches the printed one', '8/8', cmp.ok_ppl.sum(), len(cmp),
       note='mismatches: ' + (', '.join(f'{r.model} {r.stage}' for r in cmp[~cmp.ok_ppl].itertuples()) or 'none'))
C.frac('C8', 'Fig. S2a caption', 'injection combinations (model × task) passing at some strength', '6/6', (mo.stage == 'ADD').sum(), 6, rel(CFR / 'monotonicity.csv'))"""),

("md", """## C9 · Items per stage and cell range from 45 to 131 (Appendix `app:cf`)"""),
("code", """kt = read(CF / 'cf_rescue_knockout_tests.csv'); t = kt[kt.direction.isin(['R_given_PC', 'PC_given_R'])]
C.rng('C9', 'Appendix app:cf', 'items per stage and cell', '45–131', t.n.min(), t.n.max(), source=rel(CF / 'cf_rescue_knockout_tests.csv'))
display(t.groupby(['model_id', 'task', 'stage']).n.first().unstack('stage'))"""),

("md", "## Figures 3 and S2"),
("code", "show_figure('fig3_cf_intervention.png', 900); show_figure('figS2_intervention_detail.png', 900)"),
("md", "## Reproduction table for this notebook"),
("code", "df = C.show(); C.save('03')"),
]

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# 04 — manipulation check
# ═══════════════════════════════════════════════════════════════════════════════════════════════
NB04 = [
("md", f"""# 04 · Manipulation check under the conditional edit (Results ¶8, Fig. S2b–c, Table `tab:manip`)

{PROVENANCE}

## Source and code map

| id | what | data read here | original producing script |
|---|---|---|---|
| D1, D4 | joint IUT verdicts per strength cell | `additional_Qs/rq3_norm_manipulation_20260902/analysis/norm_manipulation_joint_verdicts.csv` | `additional_Qs/analysis/19_analyze_norm_manipulation.py` ← `run_rq3_norm_manipulation_gpu.py` |
| D2 | per-metric TOST / superiority, controls | `norm_manipulation_tests.csv` | `19_…` |
| D2 | PPL at α=2 | `additional_Qs/cf_rescue_fluency_posthoc/cells_annotated.csv` | `18_cf_rescue_fluency_posthoc.py` |
| D3 | separate P and C readouts of PC\\|R | `nonjoint_fig23b/fig23b_nonjoint_readouts.csv` | `21_analyze_fig23b_nonjoint.py` |
| D4, D6 | equivalence margins | `norm_manipulation_margins.csv` | `19_…` |

Everything is **loaded**."""),
("code", SETUP + "\nC = Checks('04_manipulation_check')"),

("md", """## D1 · The relevance edit leaves the norms inside their preregistered margin; the evidential edit does not move them as predicted

Reported: $R_{|PC}$ both norms equivalent in **8/12** strength cells; $PC_{|R}$ both moving as predicted in **0/13**;
every $R_{|PC}$ ablation cell passes (**5/5**, including $\\lambda=1$); all four failures are injections at $\\alpha\\ge1$."""),
("code", """jv = read(MAN / 'norm_manipulation_joint_verdicts.csv'); jv['model'] = jv.model_id.str.split('/').str[-1]
tally = jv.groupby(['target_direction', 'joint_criterion']).joint_pass.agg(['sum', 'size']); display(tally)
r = jv[jv.target_direction == 'R_given_PC']; pc = jv[jv.target_direction == 'PC_given_R']
C.frac('D1', 'Results ¶8', 'R|PC: P and C both equivalent', '8/12', r.joint_pass.sum(), len(r), rel(MAN / 'norm_manipulation_joint_verdicts.csv'))
C.frac('D1', 'Results ¶8', 'PC|R: P and C both move as predicted', '0/13', pc.joint_pass.sum(), len(pc))
rabl = r[r.stage == 'ABL']; C.frac('D1', 'Results ¶8', 'R|PC ablation cells passing', '5/5', rabl.joint_pass.sum(), len(rabl))
fails = r[~r.joint_pass]; C.frac('D1', 'Results ¶8', 'R|PC failures that are injections at α≥1', '4/4', ((fails.stage == 'ADD') & (fails.strength >= 1)).sum(), len(fails))
display(r[['model', 'task', 'stage', 'strength', 'P_mean', 'P_p', 'C_mean', 'C_p', 'joint_iut_p', 'joint_pass']].sort_values(['stage', 'strength', 'model']))"""),

("md", """## D2 · The constrained controls separate two causes of leakage

Reported: at $\\alpha=0.5$ nothing moves the norms; at $\\alpha=1$ only the target moves them and random directions stay
equivalent in **12/12**; at $\\alpha=2$ random directions move them too and perplexity rises **35.7 %**.
(The paper attributes the leakage to $\\cos(v_R, v_P)=.335$; that cosine is not stored in any released file.)"""),
("code", """tt = read(MAN / 'norm_manipulation_tests.csv'); rc = tt[tt.target_direction == 'R_given_PC']
piv = rc.groupby(['stage', 'strength', 'control_role']).equivalent.agg(['sum', 'size']); piv['ratio'] = piv['sum'].astype(str) + '/' + piv['size'].astype(str)
display(piv['ratio'].unstack('control_role'))
rnd1 = rc[(rc.strength == 1.0) & (rc.control_role == 'random')]
C.frac('D2', 'Results ¶8', 'random directions equivalent at strength 1 (P and C tests, injection and ablation cells)', '12/12', rnd1.equivalent.sum(), len(rnd1), rel(MAN / 'norm_manipulation_tests.csv'))
rnd05 = rc[(rc.strength == 0.5) & (rc.control_role != 'target')]; tg05 = rc[(rc.strength == 0.5) & (rc.control_role == 'target')]
C.frac('D2', 'Results ¶8', 'at strength 0.5 nothing moves the norms (target tests equivalent)', f'{len(tg05)}/{len(tg05)}', tg05.equivalent.sum(), len(tg05))
rnd2 = rc[(rc.strength == 2.0) & (rc.control_role == 'random')]
C.frac('D2', 'Results ¶8', 'at α=2 random directions move the norms too (equivalent tests)', '0/' + str(len(rnd2)), rnd2.equivalent.sum(), len(rnd2))
ld = read(CFP / 'ladder.csv'); a2 = ld[(ld.stage == 'ADD') & (ld.strength == 2.0)].iloc[0]
C.num('D2', 'Results ¶8', 'perplexity increase at α=2 (median over the 24 injection cells)', '35.7%', 100 * a2.ppl_median, rel(CFP / 'ladder.csv'),
      note=f'max {100*a2.ppl_max:.0f}% (Phi-4-mini-reasoning); the paper quotes the median')
C.blocked('D5', 'Results ¶8 / Discussion ¶2', 'cos(v_R, v_P) = .335', '.335', 'not stored in any released CSV; would need the activations (not shipped)')"""),

("md", """## D3 · Under $PC_{|R}$ the two norms move in opposite directions

Reported: $C$ rose as predicted in **6/13** cells; $\\rho$ fell in all **nine** Qwen3 injection cells, against the prediction;
controls at chance, random **0/13** and shuffle **2/13**. Source: separate P and C readouts (no AND gate)."""),
("code", """nj = read(MAN / 'nonjoint_fig23b/fig23b_nonjoint_readouts.csv'); nj['model'] = nj.model_id.str.split('/').str[-1]
pc_pass = nj.pivot_table(index='control_role', columns='metric', values='pass_raw_p', aggfunc='sum'); pc_n = nj.pivot_table(index='control_role', columns='metric', values='pass_raw_p', aggfunc='size')
display(pc_pass.astype(int).astype(str) + '/' + pc_n.astype(str))
tgt = nj[nj.control_role == 'target']
C.frac('D3', 'Results ¶8', 'PC|R target cells where C rises as predicted', '6/13', tgt[tgt.metric == 'C'].pass_raw_p.sum(), (tgt.metric == 'C').sum(), rel(MAN / 'nonjoint_fig23b/fig23b_nonjoint_readouts.csv'))
q = tgt[(tgt.metric == 'P') & (tgt.stage == 'ADD') & tgt.model.str.startswith('Qwen3')]
C.frac('D3', 'Results ¶8', 'Qwen3 injection cells where ρ moves against the prediction', '9/9', (q.oriented_delta_did < 0).sum(), len(q))
for role, exp in [('random', '0/13'), ('shuffle', '2/13')]:
    c = nj[nj.control_role == role]
    per_metric = c.groupby('metric').pass_raw_p.sum().to_dict()
    C.frac('D3', 'Results ¶8', f'{role} control cells passing the oriented test on C (separate readout)', exp, per_metric.get('C', 0), (c.metric == 'C').sum(), rel(MAN / 'nonjoint_fig23b/fig23b_nonjoint_readouts.csv'),
           note=f'per-metric counts P {per_metric.get("P", 0)}, C {per_metric.get("C", 0)}; joint (both norms) {int(c.groupby(["model_id", "task", "stage", "strength"]).pass_raw_p.all().sum())}')
display(tgt.pivot_table(index=['model', 'task', 'stage', 'strength'], columns='metric', values='oriented_delta_did'))"""),

("md", """## D4 · Table `tab:manip` (all 25 strength cells) and D6 · the equivalence margins

Reported margins: Qwen2.5-Math $\\delta_\\rho=0.025$, $\\delta_C=0.009$; Qwen3-4B $0.218$, $0.118$; Qwen3-4B-Instruct $0.285$, $0.257$.
The paper prints no margins for the two Phi models although they appear in the table; they are shown here."""),
("code", """display(jv[['model', 'task', 'stage', 'target_direction', 'strength', 'P_mean', 'P_p', 'C_mean', 'C_p', 'joint_pass']].sort_values(['target_direction', 'model', 'task', 'stage', 'strength']).reset_index(drop=True))
C.frac('D4', 'Table tab:manip', 'strength cells in the table', '25/25', len(jv), 25)
mg = read(MAN / 'norm_manipulation_margins.csv'); mg['model'] = mg.model_id.str.split('/').str[-1]; display(mg.pivot_table(index='model', columns='metric', values='margin'))
for m, dp, dc in [('Qwen2.5-Math-1.5B', '0.025', '0.009'), ('Qwen3-4B', '0.218', '0.118'), ('Qwen3-4B-Instruct-2507', '0.285', '0.257')]:
    C.num('D6', 'Table tab:manip caption', f'margin δ_ρ, {m}', dp, mg[(mg.model == m) & (mg.metric == 'P')].margin.iloc[0], rel(MAN / 'norm_manipulation_margins.csv'))
    C.num('D6', 'Table tab:manip caption', f'margin δ_C, {m}', dc, mg[(mg.model == m) & (mg.metric == 'C')].margin.iloc[0])
thetas = sorted(tt.theta.unique())
C.presence('D7', 'Deviation 6', 'θ=0.5 sensitivity margin was fixed; results not printed in the paper', 'θ ∈ {0.25, 0.5}', f'θ values present in tests.csv: {thetas}')"""),

("md", "## Figure S2"),
("code", "show_figure('figS2_intervention_detail.png', 900)"),
("md", "## Reproduction table for this notebook"),
("code", "df = C.show(); C.save('04')"),
]

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# 05 — preregistered grid (appendix)
# ═══════════════════════════════════════════════════════════════════════════════════════════════
NB05 = [
("md", f"""# 05 · The preregistered 15-item intervention grid and its geometry (Appendices `app:grid`, `app:intervention`, `app:models`, deviations)

{PROVENANCE}

## Source and code map

| id | what | data read here | original producing script |
|---|---|---|---|
| E1 | ablation / injection cell tests, guards | `results/rq3_intervention_final_20260827/analysis/rq3b_abl_tests.csv`, `rq3b_add_tests.csv` | `analysis/rq3/analyze_rq3_intervention.py` ← `run_rq3_intervention_gpu.py`, `run_rq3_fluency_guard_gpu.py` |
| E1 | manipulation check of the grid (norm shifts under ablation) | `rq3f_R_specificity.csv`, `rq3f_abl_norming_tests.csv` | `analysis/rq3/analyze_rq3_dis.py` ← `run_rq3_dis_gpu.py` |
| E2, E3, E5 | LOO cosine, k-sweep capture, decoding vs capture | `rq3x_subspace_k_k14_verify.csv`, `rq3x_subspace_k_k2.csv`, `rq3a_context_decoder_robustness.csv` | `analysis/rq3/verify_discussion_numbers.py`, `analyze_rq3_subspace_k.py` (need the activations to re-run; outputs shipped) |
| E11 | directions and doses of the grid | `rq3b_abl_tests.csv`, `rq3b_add_tests.csv` | as E1 |
| E8, E12 | compute and environment | `results/*/manifests/*.json` | run manifests |

Everything is **loaded**; the cell-level aggregates of E1 are recomputed from the stored tests."""),
("code", SETUP + "\nC = Checks('05_preregistered_grid')"),

("md", """## E1 · The preregistered grid returned a null

Reported (Appendix `app:intervention`): ablation and injection each supported the hypothesis in **1 of 10** adjudicated cells;
layer-matched ablation *raises* the DoD (**+0.009**) and injection lowers it in all four arms (**−0.019 to −0.032**); the only
directions that move as predicted are **two controls**; the preregistered ablation cell clears the guards in **0 of 40** cells.

Adjudication here: Holm-corrected $p<.05$, predicted sign, guard passed. Layer-matched ablation is `ablate_scope == selected`
at $\\lambda=1$; injection is the `selected_safe` strength per cell. The four arms are `R_h3`, `R_h3_denoised`, `R_h3_k2`, `R_h3_orth_PC`."""),
("code", """abl = read(GRID / 'rq3b_abl_tests.csv'); add = read(GRID / 'rq3b_add_tests.csv')
ARMS = ['R_h3', 'R_h3_denoised', 'R_h3_k2', 'R_h3_orth_PC']
sel = abl[(abl.ablate_scope == 'selected') & (abl.dose == 1.0)]
def supported(d, sign):
    return int(((d.holm_task_p < .05) & (np.sign(d['mean']) == sign) & d.guard_pass.astype(bool)).sum())
a = sel[sel.direction == 'R_h3']; i = add[(add.direction == 'R_h3') & (add.selected_safe == True)]
C.frac('E1', 'Appendix app:intervention', 'ablation of v_R supports the hypothesis (adjudicated cells)', '1/10', supported(a, -1), len(a), rel(GRID / 'rq3b_abl_tests.csv'))
C.frac('E1', 'Appendix app:intervention', 'injection of v_R supports the hypothesis (adjudicated cells)', '1/10', supported(i, 1), len(i), rel(GRID / 'rq3b_add_tests.csv'))
C.num('E1', 'Appendix app:intervention', 'layer-matched ablation, mean ΔDoD over 10 cells (R_h3, λ=1)', '+0.009', a['mean'].mean(),
      note='recomputed mean over the 10 model × task cells; the paper\\'s +0.009 may average a different cell set')
inj = pd.Series({d: add[(add.direction == d) & (add.selected_safe == True)]['mean'].mean() for d in ARMS}); display(inj.rename('mean ΔDoD, injection at selected safe α'))
C.rng('E1', 'Appendix app:intervention', 'injection lowers the DoD in all four arms (range of arm means)', '-0.032–-0.019', inj.min(), inj.max(), decimals=3,
      note='all four arm means are negative; the printed range differs from the recomputed one')
pre = abl[(abl.ablate_scope == 'all') & (abl.dose == 1.0) & abl.direction.isin(ARMS)]
C.frac('E1', 'Appendix app:intervention', 'preregistered ablation cells clearing both guards (4 arms × 10 cells)', '0/40', pre.guard_pass.astype(bool).sum(), len(pre))
g = sel.groupby('direction').apply(lambda d: pd.Series({'mean': d['mean'].mean(), 'supported': supported(d, -1)}), include_groups=False)
display(g.sort_values('mean'))
C.frac('E1', 'Appendix app:intervention', 'directions moving as predicted under layer-matched ablation are controls', '2/2', int((g.supported > 0).sum() - g.loc[ARMS].supported.gt(0).sum()), int((g.supported > 0).sum()),
       note='directions with ≥1 supported cell: ' + ', '.join(g.index[g.supported > 0]))"""),

("md", """### The manipulation check of the grid does not pass

Reported: re-asking the norming questions with the hook in place, *no cell* has all four intervals containing zero (**0 of 20**);
ablating $v_{R,h_3}$, the $h_2$ control, the bare control, $v_P$, $v_C$ or their span all shift $C$ by **+2.5 to +3.8**.
The released `rq3f_R_specificity.csv` holds the ablation-induced norm shifts per (model, task, direction, measure, hypothesis)."""),
("code", """rs = read(GRID / 'rq3f_R_specificity.csv'); display(rs.head(3))
print('directions:', sorted(rs.direction.unique()), '· stages:', sorted(rs.stage.unique()), '· metrics:', sorted(rs.metric.unique()))
# four intervals per (model, task, direction, stage): metric × hypothesis; a cell passes if all four contain zero
zero = rs.groupby(['model_id', 'task', 'direction', 'stage']).ci_contains_zero.all()
per = zero.groupby(['direction', 'stage']).agg(['sum', 'size']); display(per)
tot = zero.xs('R_h3', level='direction')
C.frac('E1', 'Appendix app:intervention', 'grid manipulation check for v_R: cells (10 ablation + 10 injection) with all four norm intervals containing zero', '0/20', tot.sum(), len(tot),
       rel(GRID / 'rq3f_R_specificity.csv'))
cshift = rs[(rs.stage == 'ABL') & (rs.metric == 'C') & (rs.hypothesis == 'h3')].groupby('direction').mean_change.mean(); display(cshift.rename('mean ΔC(h3) under ablation'))
nt = read(GRID / 'rq3f_abl_norming_tests.csv')
allC = pd.concat([cshift[['R_h3', 'R_h2', 'R_base']], nt.groupby('direction')['mean'].mean()])
display(allC.rename('mean C shift under ablation, six directions'))
C.rng('E1', 'Appendix app:intervention', 'C shift under ablation across v_R, h2, bare, v_P, v_C, span (range of direction means)', '2.5–3.8', allC.min(), allC.max(), decimals=1,
      note='rq3f_abl_norming_tests.csv stores the P/C/PC ablation shifts on a different scale (means 0.8–1.2); the printed range is not recovered from the release')
"""),

("md", """## E2 · Fifteen items read a direction but do not steer one: the LOO cosine is +0.271 (74.3°), opposite side in 14 of 150 folds

Source: `rq3x_subspace_k_k14_verify.csv`, produced by `verify_discussion_numbers.py` from the canonical activations
(the activations are not shipped, the output is). Ten cells: five models × two wordings that passed the decoder gate in this run."""),
("code", """k14 = read(GRID / 'rq3x_subspace_k_k14_verify.csv'); display(k14[['model', 'task', 'layer', 'n', 'loo_cos_mean', 'loo_cos_negative_folds', 'heldout_capture_k1', 'heldout_capture_k14', 'push_cos_k1_k14']])
m = k14.loo_cos_mean.mean()
C.num('E2', 'Appendix app:intervention', 'mean LOO cosine between the training axis and the held-out contrast', '+0.271', m, rel(GRID / 'rq3x_subspace_k_k14_verify.csv'))
C.num('E2', 'Appendix app:intervention', 'the same cosine as an angle (degrees)', '74.3', float(np.degrees(np.arccos(m))))
C.frac('E2', 'Appendix app:intervention', 'folds where the axis falls on the opposite side', '14/150', k14.loo_cos_negative_folds.sum(), k14.n.sum())
C.num('E2', 'Appendix app:intervention', 'the component an edit can move is a fifth of the contrast (capture at k=1)', '0.178', k14.heldout_capture_k1.mean())"""),

("md", """## E3 · Higher-dimensional subspaces do not recover the contrast (appendix table)

Reported: capture 0.178 / 0.212 / 0.242 / 0.289 at $k=1,2,5,14$; random 0.00039 / 0.00097 / 0.00218 / 0.00592; ratios
451× / 219× / 111× / 49×; push direction within cosine 0.705 of the one-dimensional one. Averaged over the ten cells."""),
("code", """rows = []
for kk in [1, 2, 3, 5, 14]:
    c = k14[f'heldout_capture_k{kk}'].mean(); r = k14[f'heldout_capture_random_k{kk}'].mean()
    rows.append(dict(k=kk, capture=c, random=r, ratio=c / r))
tab = pd.DataFrame(rows).set_index('k'); display(tab.style.format({'capture': '{:.3f}', 'random': '{:.5f}', 'ratio': '{:.0f}×'}))
for kk, cap, rnd, ratio in [(1, '0.178', '0.00039', '451'), (2, '0.212', '0.00097', '219'), (5, '0.242', '0.00218', '111'), (14, '0.289', '0.00592', '49')]:
    C.num('E3', 'Appendix table (k-sweep)', f'capture at k={kk}', cap, tab.loc[kk, 'capture'], rel(GRID / 'rq3x_subspace_k_k14_verify.csv'),
          note='the k=2 row in the paper (0.212, 219×) does not match the stored k=2 capture' if kk == 2 else '')
    C.num('E3', 'Appendix table (k-sweep)', f'random capture at k={kk}', rnd, tab.loc[kk, 'random'],
          note='the random column depends on the RNG stream; no stored file reproduces the printed random values')
    C.num('E3', 'Appendix table (k-sweep)', f'ratio at k={kk}', ratio, tab.loc[kk, 'ratio'], note='follows the random column')
C.num('E3', 'Appendix app:intervention', 'push direction cosine between k=1 and k=14', '0.705', k14.push_cos_k1_k14.mean())
k2 = read(GRID / 'rq3x_subspace_k_k2.csv'); print('k=2 capture in the separate k2 file:', round(k2.heldout_capture_k2.mean(), 4), '(same construction)')"""),

("md", """## E5 · Decoding and capture are dissociated: Spearman ρ = +0.20, p = 0.58

Reported: the cell with the lowest capture (0.073) decodes at 0.711 and the one with the highest (0.279) at 0.671.
This recomputation settles a discrepancy in the working notes, which quoted +0.31, p = .33 from an older run."""),
("code", """gate = read(GRID / 'rq3a_context_decoder_robustness.csv'); gate = gate[gate.family == 'conjunction_h3']; gate['model_s'] = gate.model.str.split('/').str[-1]
mm = k14.merge(gate[['model_s', 'task', 'auc_nested_loio']], left_on=['model', 'task'], right_on=['model_s', 'task'])
rho, p = stats.spearmanr(mm.heldout_capture_k1, mm.auc_nested_loio)
display(mm[['model', 'task', 'heldout_capture_k1', 'auc_nested_loio']].sort_values('heldout_capture_k1'))
C.num('E5', 'Appendix app:intervention', 'Spearman ρ between capture and decoding AUC', '+0.20', rho, rel(GRID / 'rq3x_subspace_k_k14_verify.csv'))
C.num('E5', 'Appendix app:intervention', 'its p value', '0.58', p)
lo = mm.loc[mm.heldout_capture_k1.idxmin()]; hi = mm.loc[mm.heldout_capture_k1.idxmax()]
C.num('E5', 'Appendix app:intervention', 'lowest-capture cell: capture', '0.073', lo.heldout_capture_k1); C.num('E5', 'Appendix app:intervention', 'lowest-capture cell: AUC', '0.711', lo.auc_nested_loio)
C.num('E5', 'Appendix app:intervention', 'highest-capture cell: capture', '0.279', hi.heldout_capture_k1); C.num('E5', 'Appendix app:intervention', 'highest-capture cell: AUC', '0.671', hi.auc_nested_loio)"""),

("md", """## E4 · Variance decomposition (item × model 20.8 % / 24.4 %, arm main effect 0.04–0.11 %) and E7 · retired decoder labelling (AUC 0.559)

Neither quantity has a stored output or a producing script in the released files."""),
("code", """C.blocked('E4', 'Appendix app:intervention', 'variance decomposition of the ablation/injection effect (item × model share)', '20.8% / 24.4%; arm 0.04–0.11%',
          'no stored output or script located; would require the per-item edit tables rq3b_*_item_effects and a decomposition not in the release')
C.blocked('E7', 'Deviation 1', 'retired one-versus-three rank AUC', '0.559 (6/12)', 'the retired gate file is not part of the canonical run; the value exists only in an internal notebook output')
C.blocked('E1', 'Appendix app:grid', 'the manipulation check denominator "20 cells" and the "+2.5 to +3.8" range across six directions', '0/20; +2.5 to +3.8',
          'partially reproduced above from rq3f_R_specificity.csv; the exact cell set of the printed statement is not recoverable from the release')"""),

("md", """## E11 · The grid: twenty ablation directions, seventeen injection directions, λ ∈ {0.25, 0.5, 1}, α ∈ {0.5, 1, 2, 4}"""),
("code", """C.frac('E11', 'Appendix app:grid', 'ablation directions', '20/20', abl.direction.nunique(), 20, rel(GRID / 'rq3b_abl_tests.csv'))
C.frac('E11', 'Appendix app:grid', 'injection directions', '17/17', add.direction.nunique(), 17, rel(GRID / 'rq3b_add_tests.csv'))
C.presence('E11', 'Appendix app:grid', 'ablation doses λ', '{0.25, 0.5, 1.0}', str(sorted(abl.dose.unique())))
C.presence('E11', 'Appendix app:grid', 'injection strengths α', '{0.5, 1, 2, 4}', str(sorted(add.alpha.unique())))
print('ablation directions:', sorted(abl.direction.unique()))"""),

("md", """## E8, E12 · Models, assets and compute (Appendix `app:models`)

Reported: torch 2.9.1 (CUDA 12.8), transformers 4.57.1; one NVIDIA TITAN RTX (24 GB); the preregistered grid took 56 GPU-hours
summed over five models at peak allocation 18.8 GB; pinned model revisions. Read from the run manifests (server paths scrubbed)."""),
("code", """import json, glob
man = [json.load(open(p)) for p in sorted(glob.glob(str(GRID.parent / 'manifests' / '*.json')))]
def walk(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: yield v
            yield from walk(v, key)
    elif isinstance(o, list):
        for v in o: yield from walk(v, key)
versions = {(v, t) for m in man for v in walk(m, 'torch_version') for t in walk(m, 'transformers_version')}
print('torch / transformers versions in manifests:', versions)
C.presence('E12', 'Appendix app:models', 'pinned library versions', 'torch 2.9.1 / transformers 4.57.1', '; '.join(sorted(f'{a} / {b}' for a, b in versions)))
peak_mib = max([x for m in man for x in walk(m, 'peak_nvidia_mib') if isinstance(x, (int, float))] or [0])
print(f'peak allocation over the shipped manifests: {peak_mib:.0f} MiB = {peak_mib/1024:.2f} GiB')
C.blocked('E8', 'Appendix app:models', 'GPU-hours of the preregistered grid', '56 h over five models', 'the shipped manifests record memory and versions but no wall-clock time')
C.num('E8', 'Appendix app:models', 'peak GPU allocation (MiB / 1000, as printed)', '18.8', peak_mib / 1000, note=f'{peak_mib:.0f} MiB = {peak_mib/1024:.2f} GiB; the paper divides MiB by 1000')
revs = {m.get('model'): r for m in man for r in walk(m, 'model_revision')}
C.presence('E12', 'Appendix app:models', 'pinned model revisions', '1cfa9a7, cdbee75, 4a83ca6, 093f9f3, cfbefac, 0e3b1e2', '; '.join(f'{k}: {str(v)[:7]}' for k, v in revs.items() if k))"""),

("md", "## Reproduction table for this notebook"),
("code", "df = C.show(); C.save('05')"),
]

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# 00 — summary
# ═══════════════════════════════════════════════════════════════════════════════════════════════
NB00 = [
("md", f"""# 00 · Summary: every reported number at a glance

{PROVENANCE}

This notebook aggregates the check tables written by notebooks 01–05 (`notebooks/_checks_0*.csv`) and shows the paper's
figures. Run the five experiment notebooks first (`python3 notebooks/build_notebooks.py --execute` does so in order)."""),
("code", SETUP),
("code", """import glob
parts = [pd.read_csv(p).assign(notebook=Path(p).stem.replace('_checks_', '')) for p in sorted(glob.glob(str(ROOT / 'notebooks' / '_checks_0[1-5].csv')))]
allc = pd.concat(parts, ignore_index=True)
counts = allc.status.value_counts()
display(Markdown('### Verdicts\\n\\n' + ' · '.join(f'**{k}** {v}' for k, v in counts.items()) + f'  (of {len(allc)} checks)'))
display(allc.groupby(['notebook', 'status']).size().unstack(fill_value=0))"""),
("md", "## Main-text numbers (Results ¶1–¶8)"),
("code", """main = allc[allc.section.str.startswith('Results')]
display(main[['id', 'section', 'analysis', 'reported', 'produced', 'status']].style.hide(axis='index'))"""),
("md", "## Appendix, tables, figures and deviations"),
("code", """rest = allc[~allc.section.str.startswith('Results')]
display(rest[['id', 'section', 'analysis', 'reported', 'produced', 'status', 'note']].style.hide(axis='index'))"""),
("md", "## Everything that did not reproduce exactly (❌), could not be recomputed (⛔) or is a presence check (⚠)"),
("code", """odd = allc[allc.status != '✅']
display(odd[['notebook', 'id', 'analysis', 'reported', 'produced', 'status', 'note']].style.hide(axis='index'))
allc.to_csv(ROOT / 'notebooks' / '_checks_all.csv', index=False); print('saved notebooks/_checks_all.csv')"""),
("md", "## Figures as printed"),
("code", """for f, w in [('fig1_behavior.png', 760), ('fig2_representation.png', 820), ('fig3_cf_intervention.png', 900), ('figS1_decoding.png', 900), ('figS2_intervention_detail.png', 900)]:
    display(Markdown(f'**{f}**')); show_figure(f, w)"""),
]

BOOKS = {"01_behavior_and_norms": NB01, "02_representation": NB02, "03_conditional_intervention": NB03,
         "04_manipulation_check": NB04, "05_preregistered_grid": NB05, "00_summary": NB00}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    names = a.only or list(BOOKS)
    for name in names:
        nb(BOOKS[name], HERE / f"{name}.ipynb")
    if a.execute:
        for name in names:
            cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                   "--ExecutePreprocessor.timeout=900", str(HERE / f"{name}.ipynb")]
            print("executing", name); r = subprocess.run(cmd, cwd=HERE.parent, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr[-3000:]); sys.exit(f"execution failed: {name}")
        print("all notebooks executed")


if __name__ == "__main__":
    main()
