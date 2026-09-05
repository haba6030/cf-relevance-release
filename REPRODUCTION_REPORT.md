# Reproduction report

Every number the paper reports, checked against the value recomputed or loaded from the released files by the executed notebooks in `notebooks/`. Regenerate with `python3 notebooks/build_notebooks.py --execute && python3 notebooks/make_report.py`.

## Verdicts

| status | meaning | count |
|---|---|---|
| ✅ | reproduced to the reported precision | 135 |
| ≈ | seed-dependent bootstrap bound, within 0.01 of the reported one | 5 |
| ⚠ | presence or qualitative check, not a numeric match | 12 |
| ❌ | does not match the printed value | 18 |
| ⛔ | cannot be recomputed from the released files | 5 |
| | total | 175 |

## Items that did not reproduce exactly (❌), are approximate (≈) or cannot be recomputed (⛔)

| id | where | analysis | reported | produced | status | note |
|---|---|---|---|---|---|---|
| A1 | Results ¶1 | mean DoD R|N conj, 15 items | 0.313 | +0.3125 | ❌ | exact mean 0.31246, which rounds to 0.312; Results ¶1 prints 0.313 and Table tab:decomp prints 0.312 |
| A6 | Results ¶4 | confounding rho, 15 items: CI high | +0.040 | +0.0423 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A6 | Results ¶4 | confounding C, 15 items: CI high | +0.167 | +0.1690 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A7 | Results ¶4 | mediation rho, 15 items: CI low | -0.394 | -0.3958 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A7 | Results ¶4 | mediation rho, 15 items: CI high | -0.177 | -0.1801 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A6 | Results ¶4 | confounding rho, 165 items: CI low | -0.091 | -0.0929 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A8 | Method, Inference | wording correlation over items (mean of six per-model Spearman ρ) | .96 | +0.929 | ❌ | median 0.95, pooled 0.92; Phi-4-mini-reasoning alone is 0.77. The .96 in the paper was quoted from the eight-model run |
| C2 | Results ¶7 | models passing ADD and ABL on the same direction | 0/7 | 1/7 | ❌ | Qwen3-4B passes ADD PC|R in task 3 and ABL PC|R in task 2: the statement holds within a task cell, not per model |
| C5 | Results ¶7 | Gemma target cells failing specificity | 16/20 | 17/20 | ❌ | spec = both paired specificity p-values < .05; the stored table gives 3 passing cells, the paper says 4 |
| C7 | Table tab:cfpass | rows whose PPL range matches the printed one | 8/8 | 5/8 | ❌ | mismatches: Qwen3-4B ABL, Qwen3-4B ABL, Qwen2.5-Math-1.5B ADD |
| D5 | Results ¶8 / Discussion ¶2 | cos(v_R, v_P) = .335 | .335 | — | ⛔ | not stored in any released CSV; would need the activations (not shipped) |
| E1 | Appendix app:intervention | layer-matched ablation, mean ΔDoD over 10 cells (R_h3, λ=1) | +0.009 | +0.0129 | ❌ | recomputed mean over the 10 model × task cells; the paper's +0.009 may average a different cell set |
| E1 | Appendix app:intervention | injection lowers the DoD in all four arms (range of arm means) | -0.032–-0.019 | -0.038–-0.025 | ❌ | all four arm means are negative; the printed range differs from the recomputed one |
| E1 | Appendix app:intervention | directions moving as predicted under layer-matched ablation are controls | 2/2 | 2/3 | ❌ | directions with ≥1 supported cell: C, R_h3, random_k2_00 |
| E1 | Appendix app:intervention | C shift under ablation across v_R, h2, bare, v_P, v_C, span (range of direction means) | 2.5–3.8 | 0.8–3.9 | ❌ | rq3f_abl_norming_tests.csv stores the P/C/PC ablation shifts on a different scale (means 0.8–1.2); the printed range is not recovered from the release |
| E3 | Appendix table (k-sweep) | random capture at k=1 | 0.00039 | +0.000363 | ❌ | the random column depends on the RNG stream; no stored file reproduces the printed random values |
| E3 | Appendix table (k-sweep) | ratio at k=1 | 451 | 491.3 | ❌ | follows the random column |
| E3 | Appendix table (k-sweep) | capture at k=2 | 0.212 | +0.2042 | ❌ | the k=2 row in the paper (0.212, 219×) does not match the stored k=2 capture |
| E3 | Appendix table (k-sweep) | random capture at k=2 | 0.00097 | +0.000753 | ❌ | the random column depends on the RNG stream; no stored file reproduces the printed random values |
| E3 | Appendix table (k-sweep) | ratio at k=2 | 219 | 271.3 | ❌ | follows the random column |
| E3 | Appendix table (k-sweep) | random capture at k=5 | 0.00218 | +0.002132 | ❌ | the random column depends on the RNG stream; no stored file reproduces the printed random values |
| E3 | Appendix table (k-sweep) | ratio at k=5 | 111 | 113.5 | ❌ | follows the random column |
| E3 | Appendix table (k-sweep) | random capture at k=14 | 0.00592 | +0.005725 | ❌ | the random column depends on the RNG stream; no stored file reproduces the printed random values |
| E3 | Appendix table (k-sweep) | ratio at k=14 | 49 | 50.4 | ❌ | follows the random column |
| E4 | Appendix app:intervention | variance decomposition of the ablation/injection effect (item × model share) | 20.8% / 24.4%; arm 0.04–0.11% | — | ⛔ | no stored output or script located; would require the per-item edit tables rq3b_*_item_effects and a decomposition not in the release |
| E7 | Deviation 1 | retired one-versus-three rank AUC | 0.559 (6/12) | — | ⛔ | the retired gate file is not part of the canonical run; the value exists only in an internal notebook output |
| E1 | Appendix app:grid | the manipulation check denominator "20 cells" and the "+2.5 to +3.8" range across six directions | 0/20; +2.5 to +3.8 | — | ⛔ | partially reproduced above from rq3f_R_specificity.csv; the exact cell set of the printed statement is not recoverable from the release |
| E8 | Appendix app:models | GPU-hours of the preregistered grid | 56 h over five models | — | ⛔ | the shipped manifests record memory and versions but no wall-clock time |

## Full table by notebook

### `01_behavior_and_norms.ipynb` — ✅ 44 · ≈ 5 · ⚠ 3 · ❌ 2

| id | where | analysis | reported | produced | status | note |
|---|---|---|---|---|---|---|
| A1 | Results ¶1 | DoD R|N positive, 15 items | 12/12 | 12/12 | ✅ |  |
| A1 | Results ¶1 | DoD R|N Holm-significant, 15 items | 12/12 | 12/12 | ✅ |  |
| A1 | Results ¶1 | mean DoD R|N conj, 15 items | 0.313 | +0.3125 | ❌ | exact mean 0.31246, which rounds to 0.312; Results ¶1 prints 0.313 and Table tab:decomp prints 0.312 |
| A1 | Results ¶1 | DoD R|N Holm-significant, 165 items | 12/12 | 12/12 | ✅ |  |
| A1 | Results ¶1 | mean DoD R|N conj, 165 items | 0.356 | +0.3560 | ✅ |  |
| A9 | Discussion ¶1 | banks agree to within 0.05 DoD (165 − 15) | 0.05 | +0.050 | ✅ | |difference| = 0.044, within the stated 0.05 |
| A2 | Results ¶1 | worst LOIO fold still p<.05, 15 items | 12/12 | 12/12 | ✅ |  |
| A3 | Results ¶1 | DoD R|I Holm-significant, 15 items | 12/12 | 12/12 | ✅ |  |
| A3 | Results ¶1 | DoD I|N Holm-significant, 15 items | 0/12 | 0/12 | ✅ |  |
| A3 | Results ¶1 | relevance share of full effect, 15 items | 58–85 | 58–85 | ✅ |  |
| A3 | Table tab:decomp | mean DoD R|I, 15 items | 0.215 | +0.2149 | ✅ |  |
| A3 | Table tab:decomp | mean DoD I|N, 15 items | 0.097 | +0.0975 | ✅ |  |
| A3 | Table tab:decomp | mean DoD R|N, 15 items | 0.312 | +0.3123 | ✅ | the paper prints 0.312 in the table and 0.313 in Results ¶1 for the same quantity |
| A3 | Results ¶1 | DoD R|I Holm-significant, 165 items | 12/12 | 12/12 | ✅ |  |
| A3 | Results ¶1 | DoD I|N Holm-significant, 165 items | 12/12 | 12/12 | ✅ |  |
| A3 | Results ¶1 | relevance share of full effect, 165 items | 33–56 | 33–56 | ✅ |  |
| A3 | Table tab:decomp | mean DoD R|I, 165 items | 0.164 | +0.1637 | ✅ |  |
| A3 | Table tab:decomp | mean DoD I|N, 165 items | 0.192 | +0.1923 | ✅ |  |
| A3 | Table tab:decomp | mean DoD R|N, 165 items | 0.356 | +0.3560 | ✅ |  |
| A4 | Results ¶2 | bare > conj cells, 15 items | 12/12 | 12/12 | ✅ |  |
| A4 | Results ¶2 | mean bare-arm DoD, 15 items | 1.156 | +1.1556 | ✅ |  |
| A4 | Results ¶2 | bare > conj cells, 165 items | 12/12 | 12/12 | ✅ |  |
| A4 | Results ¶2 | mean bare-arm DoD, 165 items | 1.091 | +1.0912 | ✅ |  |
| A5 | Results ¶3 | norm contrasts excluding zero, 15 items (bootstrap CI) | 14/24 | 14/24 | ✅ | the t-based CI gives 13/24; the paper matches the bootstrap CI |
| A5 | Results ¶3 | norm contrasts excluding zero, 165 items | 21/24 | 21/24 | ✅ |  |
| A6 | Results ¶4 | confounding rho, 15 items: mean ρ | -0.034 | -0.0342 | ✅ |  |
| A6 | Results ¶4 | confounding rho, 15 items: CI low | -0.114 | -0.1140 | ✅ |  |
| A6 | Results ¶4 | confounding rho, 15 items: CI high | +0.040 | +0.0423 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A6 | Results ¶4 | confounding C, 15 items: mean ρ | +0.023 | +0.0229 | ✅ |  |
| A6 | Results ¶4 | confounding C, 15 items: CI low | -0.132 | -0.1321 | ✅ |  |
| A6 | Results ¶4 | confounding C, 15 items: CI high | +0.167 | +0.1690 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A7 | Results ¶4 | mediation rho, 15 items: mean ρ | -0.276 | -0.2762 | ✅ |  |
| A7 | Results ¶4 | mediation rho, 15 items: CI low | -0.394 | -0.3958 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A7 | Results ¶4 | mediation rho, 15 items: CI high | -0.177 | -0.1801 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A7 | working notes only | mediation C, 15 items (not in paper) | -0.080 [-0.243, +0.141] | -0.080 [-0.243, +0.141] | ⚠ | presence check; not a numeric match |
| A6 | Results ¶4 | confounding rho, 165 items: mean ρ | -0.025 | -0.0255 | ✅ |  |
| A6 | Results ¶4 | confounding rho, 165 items: CI low | -0.091 | -0.0929 | ≈ | seed-dependent bootstrap, within 0.01 of the reported bound |
| A6 | Results ¶4 | confounding rho, 165 items: CI high | +0.042 | +0.0420 | ✅ |  |
| A6 | Results ¶4 | confounding C, 165 items: mean ρ | +0.016 | +0.0165 | ✅ |  |
| A6 | Results ¶4 | confounding C, 165 items: CI low | -0.030 | -0.0297 | ✅ |  |
| A6 | Results ¶4 | confounding C, 165 items: CI high | +0.063 | +0.0629 | ✅ |  |
| A7 | Results ¶4 | mediation rho, 165 items: mean ρ | +0.080 | +0.0799 | ✅ |  |
| A7 | Results ¶4 | mediation rho, 165 items: CI low | +0.023 | +0.0230 | ✅ |  |
| A7 | Results ¶4 | mediation rho, 165 items: CI high | +0.145 | +0.1450 | ✅ |  |
| A7 | working notes only | mediation C, 165 items (not in paper) | +0.094 [+0.053, +0.141] | +0.094 [+0.053, +0.142] | ⚠ | presence check; not a numeric match |
| A7 | Results ¶4 | mediation ρ cells significant alone, 15 items | 1/12 | 1/12 | ✅ |  |
| A7 | Results ¶4 | mediation ρ cells significant alone, 165 items | 2/12 | 2/12 | ✅ |  |
| A7 | Results ¶4 | ρ² of mediation association, 15 items | 0.10 | +0.098 | ✅ |  |
| A7 | Results ¶4 | ρ² of mediation association, 165 items | 0.01 | +0.013 | ✅ |  |
| A10 | Appendix app:large | mediation rho: sign agrees across banks | 1/6 | 1/6 | ✅ |  |
| A10 | Appendix app:large | mediation C: sign agrees across banks | 2/6 | 2/6 | ✅ |  |
| A10 | Appendix app:large | every mediation interval overlaps its counterpart | yes | no | ⚠ | presence check; not a numeric match |
| A8 | Method, Inference | wording correlation over items (mean of six per-model Spearman ρ) | .96 | +0.929 | ❌ | median 0.95, pooled 0.92; Phi-4-mini-reasoning alone is 0.77. The .96 in the paper was quoted from the eight-model run |
| E9 | Deviation 9 | position-only predictability of the letter readout (all models in the run) | 66% | 66.0 | ✅ | six-model mean 64.7%; predictability = max(accuracy, 1 − accuracy), i.e. after counterbalancing |

### `02_representation.ipynb` — ✅ 28 · ⚠ 3

| id | where | analysis | reported | produced | status | note |
|---|---|---|---|---|---|---|
| B1 | Results ¶5 | held-out AUC h3, 15 items | 0.704 | +0.7037 | ✅ |  |
| B1 | Results ¶5 | h3 cells passing the gate, 15 items | 12/12 | 12/12 | ✅ |  |
| B1 | Results ¶5 | h2 control cells passing, 15 items | 1/12 | 1/12 | ✅ |  |
| B1 | Results ¶5 | h1 (base) control cells passing, 15 items | 3/12 | 3/12 | ✅ |  |
| B1 | Results ¶5 | held-out AUC h3, 165 items | 0.803 | +0.8031 | ✅ |  |
| B1 | Results ¶5 | all cells pass at 165 items (gate saturates) | 48/48 | 48/48 | ✅ |  |
| B1 | Results ¶5 | lowest AUC that still passes at 165 items | 0.516 | +0.5164 | ✅ |  |
| B2 | Results ¶5 | gap over h2, 15 items | +0.194 | +0.1937 | ✅ |  |
| B2 | Results ¶5 | gap over h2, 15 items: CI low | +0.146 | +0.1463 | ✅ |  |
| B2 | Results ¶5 | gap over h2, 15 items: CI high | +0.252 | +0.2519 | ✅ |  |
| B2 | Results ¶5 | gap over h2, 165 items | +0.233 | +0.2327 | ✅ |  |
| B2 | Results ¶5 | gap over h2, 165 items: CI low | +0.176 | +0.1756 | ✅ |  |
| B2 | Results ¶5 | gap over h2, 165 items: CI high | +0.283 | +0.2830 | ✅ |  |
| B2 | Results ¶5 | gap over h1, 15 items | +0.189 | +0.1893 | ✅ |  |
| B2 | Results ¶5 / app:large | gap over h1, 165 items | +0.213 | +0.2127 | ✅ |  |
| B2 | Appendix app:large | gap over h1, 165 items: CI low | +0.168 | +0.1676 | ✅ |  |
| B2 | Appendix app:large | gap over h1, 165 items: CI high | +0.248 | +0.2475 | ✅ |  |
| B2 | Fig. 2b caption | no model-level gap interval includes zero | true | True | ⚠ | presence check; not a numeric match |
| B3 | Results ¶6 | R|PC decodable, 15 items | 12/12 | 12/12 | ✅ |  |
| B3 | Results ¶6 | PC|R decodable, 15 items | 6/12 | 6/12 | ✅ |  |
| B3 | Results ¶6 | R|PC decodable, 165 items | 12/12 | 12/12 | ✅ |  |
| B3 | Results ¶6 | PC|R decodable, 165 items | 10/12 | 10/12 | ✅ |  |
| B4 | Results ¶6 | norm of v_R removed by the P–C plane, 165 items | 76% | 76.2 | ✅ |  |
| B4 | Results ¶6 | norm of v_R removed by the P–C plane, 15 items (six models) | 67% | 67.0 | ✅ |  |
| B4 | Discussion ¶3 | norm of v_PC held by v_R, 165 items | 72% | 72.1 | ✅ |  |
| B4 | Appendix app:deviations 4 | squared share (58 % of squared norm quoted in working notes) | 0.58 | +0.581 | ✅ | working-note value; the paper quotes the norm share |
| B5 | Fig. S1a | mean AUC over the twelve cells from cached held-out scores | 0.704 | +0.7037 | ✅ |  |
| B5 | Fig. S1c | not decodable in the first fifth, decodable from mid depth | qualitative | decodable share: first fifth 0.56, mid depth 1.00 | ⚠ | presence check; not a numeric match |
| E6 | Deviation 8 | Phi-4-mini-reasoning split-half reliability of Δρ | 0.409 | +0.4090 | ✅ |  |
| E6 | Deviation 8 | Phi-4-mini-reasoning split-half reliability of ΔC | 0.193 | +0.1931 | ✅ |  |
| E6 | Deviation 8 | only Phi-4-mini-reasoning falls below the 0.6 floor | one model | Phi-4-mini-reasoning | ⚠ | presence check; not a numeric match |

### `03_conditional_intervention.ipynb` — ✅ 23 · ❌ 3 · ⚠ 1

| id | where | analysis | reported | produced | status | note |
|---|---|---|---|---|---|---|
| C1 | Results ¶7 | passing model × task × stage × direction combinations | 10/10 | 10/10 | ✅ | count of combinations passing at some strength |
| C1 | Results ¶7 | predicted sign | 10/10 | 10/10 | ✅ |  |
| C1 | Results ¶7 | dose-monotone | 10/10 | 10/10 | ✅ |  |
| C1 | Results ¶7 | constrained random / shuffled control cells passing every gate | 0/240 | 0/240 | ✅ |  |
| C2 | Results ¶7 | distinct models among the passing combinations | 5/6 | 5/6 | ✅ |  |
| C2 | Results ¶7 | models passing ADD and ABL on the same direction | 0/7 | 1/7 | ❌ | Qwen3-4B passes ADD PC|R in task 3 and ABL PC|R in task 2: the statement holds within a task cell, not per model |
| C2 | Results ¶7 | model × task cells passing ADD and ABL on the same direction | 0/10 | 0/10 | ✅ |  |
| C7 | Table tab:models | passing cells, Qwen3-4B | 3/10 | 3/10 | ✅ |  |
| C7 | Table tab:models | passing cells, Qwen3-4B-Instruct-2507 | 2/10 | 2/10 | ✅ |  |
| C7 | Table tab:models | passing cells, Qwen2.5-Math-1.5B | 2/10 | 2/10 | ✅ |  |
| C7 | Table tab:models | passing cells, gemma-3-4b-it | 0/10 | 0/10 | ✅ |  |
| C7 | Table tab:models | passing cells, Phi-4-mini-instruct | 2/10 | 2/10 | ✅ |  |
| C7 | Table tab:models | passing cells, Phi-4-mini-reasoning | 1/10 | 1/10 | ✅ |  |
| C3 | Results ¶7 | 12-cell mean ΔCF, ADD R|PC α=1 | -0.073 | -0.0728 | ✅ |  |
| C3 | Results ¶7 | passing-cell mean ΔCF, ADD R|PC α=1 | +0.161 | +0.1610 | ✅ |  |
| C3 | Results ¶7 | cells moving against the prediction, ADD R|PC α=1 | 8/12 | 8/12 | ✅ |  |
| C4 | Results ¶7 | ADD α=2 cells under the fluency threshold | 0/24 | 0/24 | ✅ |  |
| C4 | Results ¶7 | ADD α=2 cells passing on effect alone (post hoc) | 4/24 | 4/24 | ✅ |  |
| C4 | Results ¶7 | PPL increase at α=2 over the four cells that pass on effect alone | 11.6–37.5 | 11.6–37.5 | ✅ | over all 24 target cells the range is 11.6–500% (Gemma and Phi-4-mini-reasoning collapse) |
| C4 | Results ¶7 | PC|R rises to α=2, R|PC peaks at α=1 (passing injections) | qualitative | PC|R argmax α=2; R|PC argmax α=1 | ⚠ | presence check; not a numeric match |
| C5 | Results ¶7 | Gemma target cells failing specificity | 16/20 | 17/20 | ❌ | spec = both paired specificity p-values < .05; the stored table gives 3 passing cells, the paper says 4 |
| C5 | Results ¶7 | largest PPL ratio at λ=1 (Gemma) | 2718 | 2718.0 | ✅ | ppl_max is the fold increase minus one; reported as ×2,718 |
| C6 | Table tab:cfgrid | rows reproduced to the printed precision (passed, mean all, mean passing, fluency) | 10/10 | 10/10 | ✅ | mismatching rows: none |
| C7 | Table tab:cfpass | rows whose ΔCF range matches the printed one | 8/8 | 8/8 | ✅ |  |
| C7 | Table tab:cfpass | rows whose PPL range matches the printed one | 8/8 | 5/8 | ❌ | mismatches: Qwen3-4B ABL, Qwen3-4B ABL, Qwen2.5-Math-1.5B ADD |
| C8 | Fig. S2a caption | injection combinations (model × task) passing at some strength | 6/6 | 6/6 | ✅ |  |
| C9 | Appendix app:cf | items per stage and cell | 45–131 | 45–131 | ✅ |  |

### `04_manipulation_check.ipynb` — ✅ 19 · ⛔ 1 · ⚠ 1

| id | where | analysis | reported | produced | status | note |
|---|---|---|---|---|---|---|
| D1 | Results ¶8 | R|PC: P and C both equivalent | 8/12 | 8/12 | ✅ |  |
| D1 | Results ¶8 | PC|R: P and C both move as predicted | 0/13 | 0/13 | ✅ |  |
| D1 | Results ¶8 | R|PC ablation cells passing | 5/5 | 5/5 | ✅ |  |
| D1 | Results ¶8 | R|PC failures that are injections at α≥1 | 4/4 | 4/4 | ✅ |  |
| D2 | Results ¶8 | random directions equivalent at strength 1 (P and C tests, injection and ablation cells) | 12/12 | 12/12 | ✅ |  |
| D2 | Results ¶8 | at strength 0.5 nothing moves the norms (target tests equivalent) | 10/10 | 10/10 | ✅ |  |
| D2 | Results ¶8 | at α=2 random directions move the norms too (equivalent tests) | 0/2 | 0/2 | ✅ |  |
| D2 | Results ¶8 | perplexity increase at α=2 (median over the 24 injection cells) | 35.7% | +35.71 | ✅ | max 500% (Phi-4-mini-reasoning); the paper quotes the median |
| D5 | Results ¶8 / Discussion ¶2 | cos(v_R, v_P) = .335 | .335 | — | ⛔ | not stored in any released CSV; would need the activations (not shipped) |
| D3 | Results ¶8 | PC|R target cells where C rises as predicted | 6/13 | 6/13 | ✅ |  |
| D3 | Results ¶8 | Qwen3 injection cells where ρ moves against the prediction | 9/9 | 9/9 | ✅ |  |
| D3 | Results ¶8 | random control cells passing the oriented test on C (separate readout) | 0/13 | 0/13 | ✅ | per-metric counts P 0, C 0; joint (both norms) 0 |
| D3 | Results ¶8 | shuffle control cells passing the oriented test on C (separate readout) | 2/13 | 2/13 | ✅ | per-metric counts P 2, C 2; joint (both norms) 0 |
| D4 | Table tab:manip | strength cells in the table | 25/25 | 25/25 | ✅ |  |
| D6 | Table tab:manip caption | margin δ_ρ, Qwen2.5-Math-1.5B | 0.025 | +0.0252 | ✅ |  |
| D6 | Table tab:manip caption | margin δ_C, Qwen2.5-Math-1.5B | 0.009 | +0.0090 | ✅ |  |
| D6 | Table tab:manip caption | margin δ_ρ, Qwen3-4B | 0.218 | +0.2178 | ✅ |  |
| D6 | Table tab:manip caption | margin δ_C, Qwen3-4B | 0.118 | +0.1182 | ✅ |  |
| D6 | Table tab:manip caption | margin δ_ρ, Qwen3-4B-Instruct-2507 | 0.285 | +0.2845 | ✅ |  |
| D6 | Table tab:manip caption | margin δ_C, Qwen3-4B-Instruct-2507 | 0.257 | +0.2572 | ✅ |  |
| D7 | Deviation 6 | θ=0.5 sensitivity margin was fixed; results not printed in the paper | θ ∈ {0.25, 0.5} | θ values present in tests.csv: [0.25] | ⚠ | presence check; not a numeric match |

### `05_preregistered_grid.ipynb` — ✅ 21 · ❌ 13 · ⛔ 4 · ⚠ 4

| id | where | analysis | reported | produced | status | note |
|---|---|---|---|---|---|---|
| E1 | Appendix app:intervention | ablation of v_R supports the hypothesis (adjudicated cells) | 1/10 | 1/10 | ✅ |  |
| E1 | Appendix app:intervention | injection of v_R supports the hypothesis (adjudicated cells) | 1/10 | 1/10 | ✅ |  |
| E1 | Appendix app:intervention | layer-matched ablation, mean ΔDoD over 10 cells (R_h3, λ=1) | +0.009 | +0.0129 | ❌ | recomputed mean over the 10 model × task cells; the paper's +0.009 may average a different cell set |
| E1 | Appendix app:intervention | injection lowers the DoD in all four arms (range of arm means) | -0.032–-0.019 | -0.038–-0.025 | ❌ | all four arm means are negative; the printed range differs from the recomputed one |
| E1 | Appendix app:intervention | preregistered ablation cells clearing both guards (4 arms × 10 cells) | 0/40 | 0/40 | ✅ |  |
| E1 | Appendix app:intervention | directions moving as predicted under layer-matched ablation are controls | 2/2 | 2/3 | ❌ | directions with ≥1 supported cell: C, R_h3, random_k2_00 |
| E1 | Appendix app:intervention | grid manipulation check for v_R: cells (10 ablation + 10 injection) with all four norm intervals containing zero | 0/20 | 0/20 | ✅ |  |
| E1 | Appendix app:intervention | C shift under ablation across v_R, h2, bare, v_P, v_C, span (range of direction means) | 2.5–3.8 | 0.8–3.9 | ❌ | rq3f_abl_norming_tests.csv stores the P/C/PC ablation shifts on a different scale (means 0.8–1.2); the printed range is not recovered from the release |
| E2 | Appendix app:intervention | mean LOO cosine between the training axis and the held-out contrast | +0.271 | +0.2708 | ✅ |  |
| E2 | Appendix app:intervention | the same cosine as an angle (degrees) | 74.3 | +74.29 | ✅ |  |
| E2 | Appendix app:intervention | folds where the axis falls on the opposite side | 14/150 | 14/150 | ✅ |  |
| E2 | Appendix app:intervention | the component an edit can move is a fifth of the contrast (capture at k=1) | 0.178 | +0.1783 | ✅ |  |
| E3 | Appendix table (k-sweep) | capture at k=1 | 0.178 | +0.1783 | ✅ |  |
| E3 | Appendix table (k-sweep) | random capture at k=1 | 0.00039 | +0.000363 | ❌ | the random column depends on the RNG stream; no stored file reproduces the printed random values |
| E3 | Appendix table (k-sweep) | ratio at k=1 | 451 | 491.3 | ❌ | follows the random column |
| E3 | Appendix table (k-sweep) | capture at k=2 | 0.212 | +0.2042 | ❌ | the k=2 row in the paper (0.212, 219×) does not match the stored k=2 capture |
| E3 | Appendix table (k-sweep) | random capture at k=2 | 0.00097 | +0.000753 | ❌ | the random column depends on the RNG stream; no stored file reproduces the printed random values |
| E3 | Appendix table (k-sweep) | ratio at k=2 | 219 | 271.3 | ❌ | follows the random column |
| E3 | Appendix table (k-sweep) | capture at k=5 | 0.242 | +0.2420 | ✅ |  |
| E3 | Appendix table (k-sweep) | random capture at k=5 | 0.00218 | +0.002132 | ❌ | the random column depends on the RNG stream; no stored file reproduces the printed random values |
| E3 | Appendix table (k-sweep) | ratio at k=5 | 111 | 113.5 | ❌ | follows the random column |
| E3 | Appendix table (k-sweep) | capture at k=14 | 0.289 | +0.2886 | ✅ |  |
| E3 | Appendix table (k-sweep) | random capture at k=14 | 0.00592 | +0.005725 | ❌ | the random column depends on the RNG stream; no stored file reproduces the printed random values |
| E3 | Appendix table (k-sweep) | ratio at k=14 | 49 | 50.4 | ❌ | follows the random column |
| E3 | Appendix app:intervention | push direction cosine between k=1 and k=14 | 0.705 | +0.7047 | ✅ |  |
| E5 | Appendix app:intervention | Spearman ρ between capture and decoding AUC | +0.20 | +0.200 | ✅ |  |
| E5 | Appendix app:intervention | its p value | 0.58 | +0.580 | ✅ |  |
| E5 | Appendix app:intervention | lowest-capture cell: capture | 0.073 | +0.0730 | ✅ |  |
| E5 | Appendix app:intervention | lowest-capture cell: AUC | 0.711 | +0.7111 | ✅ |  |
| E5 | Appendix app:intervention | highest-capture cell: capture | 0.279 | +0.2787 | ✅ |  |
| E5 | Appendix app:intervention | highest-capture cell: AUC | 0.671 | +0.6711 | ✅ |  |
| E4 | Appendix app:intervention | variance decomposition of the ablation/injection effect (item × model share) | 20.8% / 24.4%; arm 0.04–0.11% | — | ⛔ | no stored output or script located; would require the per-item edit tables rq3b_*_item_effects and a decomposition not in the release |
| E7 | Deviation 1 | retired one-versus-three rank AUC | 0.559 (6/12) | — | ⛔ | the retired gate file is not part of the canonical run; the value exists only in an internal notebook output |
| E1 | Appendix app:grid | the manipulation check denominator "20 cells" and the "+2.5 to +3.8" range across six directions | 0/20; +2.5 to +3.8 | — | ⛔ | partially reproduced above from rq3f_R_specificity.csv; the exact cell set of the printed statement is not recoverable from the release |
| E11 | Appendix app:grid | ablation directions | 20/20 | 20/20 | ✅ |  |
| E11 | Appendix app:grid | injection directions | 17/17 | 17/17 | ✅ |  |
| E11 | Appendix app:grid | ablation doses λ | {0.25, 0.5, 1.0} | [0.25, 0.5, 1.0] | ⚠ | presence check; not a numeric match |
| E11 | Appendix app:grid | injection strengths α | {0.5, 1, 2, 4} | [0.5, 1.0, 2.0, 4.0] | ⚠ | presence check; not a numeric match |
| E12 | Appendix app:models | pinned library versions | torch 2.9.1 / transformers 4.57.1 | 2.9.1+cu128 / 4.57.1 | ⚠ | presence check; not a numeric match |
| E8 | Appendix app:models | GPU-hours of the preregistered grid | 56 h over five models | — | ⛔ | the shipped manifests record memory and versions but no wall-clock time |
| E8 | Appendix app:models | peak GPU allocation (MiB / 1000, as printed) | 18.8 | +18.81 | ✅ | 18808 MiB = 18.37 GiB; the paper divides MiB by 1000 |
| E12 | Appendix app:models | pinned model revisions | 1cfa9a7, cdbee75, 4a83ca6, 093f9f3, cfbefac, 0e3b1e2 | Qwen/Qwen2.5-Math-1.5B: 4a83ca6; Qwen/Qwen3-4B-Instruct-2507: cdbee75; Qwen/Qwen3-4B: 1cfa9a7; google/gemma-3-4b-it: 093f9f3; microsoft/Phi-4-mini-instruct: cfbefac; microsoft/Phi-4-mini-reasoning: 0e3b1e2 | ⚠ | presence check; not a numeric match |
