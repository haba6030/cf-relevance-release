# Reproduction manifest: every reported number, its source file and its producing script

Authoritative source: the submitted manuscript. Six models only. The verdict for each row is in
`REPRODUCTION_REPORT.md` and in the notebook named in the id prefix (A → `01_behavior_and_norms`, B → `02_representation`,
C → `03_conditional_intervention`, D → `04_manipulation_check`, E → `05_preregistered_grid`).
Two banks: the 15-item anchored bank (`results/…`) and the 165-item generalization bank (`additional_Qs/…`).

Legend for **status**: `csv` = value is read from a stored analysis CSV; `nb` = value must be recomputed
in the notebook from a stored CSV (aggregation the paper does but no script stores); `act` = needs the
raw activations (`.npy`, 1.2 GB / 12 GB, not shipped); `??` = no stored output or code located yet.

---

## Experiment A — Behavior and norms

| id | §/where | Reported | Source file | Producing script (← upstream) | status |
|---|---|---|---|---|---|
| A1 | Results ¶1, Fig 1a | 15 items: DoD^{R\|N} conj positive & Holm 12/12, mean **0.313**; 165: 12/12, mean **0.356** | `results/rq1_spec_cached_20260821/analysis/logprob_cell_tests.csv` (arm=conj, metric=logprob_norm, 6 models) · `additional_Qs/rq12_on_165/h1_cell_tests_165.csv` | `sim/analyze_rq1_spec.py` ← `sim/run_rq1_spec_gpu.py` · `additional_Qs/analysis/10_rq12_on_165.py` ← `additional_Qs/run_rq3_behavior.py` | csv |
| A2 | Results ¶1 | worst LOIO fold significant 12/12 | same, `loo_worst_p_flip_two` | same | csv |
| A3 | Results ¶1, Tab. `tab:decomp` | DoD^{R\|I} Holm 12/12 both banks; DoD^{I\|N} 0/12 (15) · 12/12 (165); share 58–85 % · 33–56 %; means 0.215/0.097/0.312 and 0.164/0.192/0.356 | `results/rq3_pcr_locked_20260825/analysis/h30_tests.csv` · `additional_Qs/rq3_run_server/analysis/h30_tests.csv` (+`h30_gate.csv`) | `analysis/rq3/analyze_rq3_h30.py` ← `analysis/rq3/run_rq3_behavior_gpu.py` · `additional_Qs/analysis/07_h30_anyN.py` | csv |
| A4 | Results ¶2, Fig 1b | bare > conj 12/12; **1.156** vs 0.313 (15), **1.091** vs 0.356 (165) | `logprob_cell_tests.csv` (arm=bare) + `h1b_arm_tests.csv` · `h1_cell_tests_165.csv` | as A1 | csv |
| A5 | Results ¶3 | norms shift: CI excludes 0 in **14/24** (15) · **21/24** (165) | `results/rq1_spec_cached_20260821/analysis/norming_summary.csv` (boot_lo/hi) · `additional_Qs/rq12_on_165/h2a_tests_165.csv` | `sim/analyze_rq1_spec.py` ← `sim/run_rq1_spec_gpu.py` (norming arm, prompts from `sim/run_norm15.py`) · `10_rq12_on_165.py` | csv |
| A6 | Results ¶4 | confounding ρ −0.034 [−0.114,+0.040], C +0.023 [−0.132,+0.167] (15); −0.025 [−0.091,+0.042], +0.016 [−0.030,+0.063] (165) | `norming_connections.csv` (path=level) · `h2b_connections_165.csv` | model-level bootstrap B=4000 over 6 model means | nb |
| A7 | Results ¶4 | mediation ρ −0.276 [−0.394,−0.177] (15) · +0.080 [+0.023,+0.145] (165); C −0.080 [−0.243,+0.141] · +0.094 [+0.053,+0.141]; cells sig 1/12 · 2/12; ρ² 0.10 · 0.01; sign agrees in 1/6 models | same (path=change) | nb | nb |

## Experiment B — Representation

| id | §/where | Reported | Source file | Producing script | status |
|---|---|---|---|---|---|
| B1 | Results ¶5, Fig 2a, Fig S1 | held-out AUC **0.704** 12/12 (15) · **0.803** (165); gate: h2 1/12, h1 3/12 at 15; all 48 cells pass at 165, min 0.516 | `results/rq3_pcr_locked_20260825/analysis/rq3a_context_decoder_robustness.csv` · `additional_Qs/rq3_run_server/analysis/rq3a_context_decoder_robustness.csv` | `analysis/rq3/analyze_rq3_context_decoding_robustness.py` ← `run_rq3_representation_gpu.py` / `additional_Qs/run_rq3_representation.py` | csv |
| B2 | Results ¶5, Fig 2b | gap over h2 **+0.194** [+0.146,+0.252] · **+0.233** [+0.176,+0.283]; over h1 +0.189 · +0.213 [+0.168,+0.248] | `rq3a_specificity_gap_summary.csv` (both runs) | `analysis/rq3/analyze_rq3_h3a_specificity_gap.py` | csv |
| B3 | Results ¶6, Fig 2c | R\|PC 12/12 both; PC\|R **6/12** (15) · **10/12** (165) | `analysis/joint_r_pc_iut_no_holm/component_tests.csv` (both runs) | `analysis/rq3/analyze_rq3_joint_pcr_iut.py` | csv |
| B4 | Results ¶6, Discussion ¶3, App. large | projection removes **76 %** of ‖v_R‖ (165) · **67 %** (15); v_R holds **72 %** of ‖v_PC‖ | `additional_Qs/rpc_vector_control/cells.csv` (`contrast==did`, `v_R_in_PC_plane`, `v_PC_on_v_R`) | `additional_Qs/analysis/15_rpc_vector_control.py` (needs activations to re-run; CSV stored) | csv (15-item value recomputed over the six models) |
| B5 | Fig S1 | 12 ROC curves, null 95th pct, depth profile | `docs/paper/figs/paper2/figS1_decoding_scores.csv` (cache) · `rq3a_layer_curves_h3_locked_20260825.csv` | `sim/make_paper_fig_pipeline.py` (recomputes from activations; cache shipped) | csv/act |

## Experiment C — Conditional intervention (165 items, 0830 pipeline)

| id | §/where | Reported | Source file | Producing script | status |
|---|---|---|---|---|---|
| C1 | Results ¶7, Fig 3 | **10** combinations pass; sign 10/10; monotone 10/10; constrained controls 0 | `additional_Qs/cf_rescue_reanalysis/monotonicity.csv` · `additional_Qs/cf_rescue_fluency_posthoc/cells_annotated.csv` | `additional_Qs/analysis/16_analyze_cf_rescue_knockout.py` → `17_cf_rescue_reanalysis.py` → `18_cf_rescue_fluency_posthoc.py` ← `additional_Qs/run_rq3_cf_rescue_gpu.py` + `run_rq3_cf_rescue_guard_gpu.py` (`cf_rescue_common.py`) | csv |
| C2 | Results ¶7 | five models; no model passes ADD and ABL on same direction; model lists per arm | `monotonicity.csv`, `cf_rescue_fluency_posthoc/by_model.csv` | same | csv |
| C3 | Results ¶7 | ADD R\|PC α=1: 12-cell mean **−0.073**, passing mean **+0.161**, 8 cells against | `cf_rescue_reanalysis/means_by_pass.csv`, `cells_annotated.csv` | same | csv |
| C4 | Results ¶7, Tab. `tab:cfgrid` | α=2 fluency 0/24, PPL +11.6–37.5 %; 4 cells revive post hoc; PC\|R rises to α=2 | `cf_rescue_fluency_posthoc/ladder.csv`, `dose_curve.csv`, `cells_annotated.csv` | `18_cf_rescue_fluency_posthoc.py` | csv |
| C5 | Results ¶7, Tab. `tab:cfpass` | Gemma fails specificity 16/20; λ=1 PPL ×2,718 | `cells_annotated.csv`, `ladder.csv` (`ppl_max`) | same | csv |
| C6 | Tab. `tab:cfgrid` | 10 rows: cells passed, mean all, mean passing, fluency n/24 | `rq3_cf_rescue_20260830/analysis/cf_rescue_knockout_summary.csv` + `means_by_pass.csv` + `ladder.csv` | 16 · 17 · 18 | csv |
| C7 | Tab. `tab:cfpass`, Tab. `tab:models` | 8 grouped rows; passing cells per model 3/2/2/0/2/1 | `by_model.csv`, `monotonicity.csv` | 18 · 17 | csv |
| C8 | Fig S2a | dose–response of passing injections | `dose_curve.csv` | `sim/make_paper_fig3_0830.py` | csv |

## Experiment D — Manipulation check (deviation 6)

| id | §/where | Reported | Source file | Producing script | status |
|---|---|---|---|---|---|
| D1 | Results ¶8, Fig S2b | R\|PC equivalent **8/12**; PC\|R predicted **0/13**; ABL 5/5; 4 failures all ADD α≥1 | `additional_Qs/rq3_norm_manipulation_20260902/analysis/norm_manipulation_joint_verdicts.csv` | `additional_Qs/analysis/19_analyze_norm_manipulation.py` ← `additional_Qs/run_rq3_norm_manipulation_gpu.py` | csv |
| D2 | Results ¶8 | random stays equivalent 12/12 at α=1; α=2 PPL +35.7 % | `norm_manipulation_tests.csv` (control_role) · `cells_annotated.csv` (PPL) | 19 · 18 | csv |
| D3 | Results ¶8, Fig S2c | PC\|R: C rises 6/13, P falls in 9 Qwen3 ADD cells; random 0/13, shuffle 2/13 | `analysis/nonjoint_fig23b/fig23b_nonjoint_readouts.csv` · `norm_manipulation_tests.csv` | `additional_Qs/analysis/21_analyze_fig23b_nonjoint.py` · 19 | csv |
| D4 | Tab. `tab:manip` | 25 rows ΔDID_ρ, p_ρ, ΔDID_C, p_C, pass; margins δ per model | `norm_manipulation_joint_verdicts.csv`, `norm_manipulation_margins.csv` | 19 | csv |
| D5 | Results ¶8, Discussion | cos(v_R, v_P) = **.335** | ?? not in any stored CSV (`cf_rescue_knockout_cells.csv` has cos_R_PC=0.721 with the pooled PC, not v_P) | ?? | ?? |

## Experiment E — Preregistered 15-item grid and its geometry (Appendix `app:intervention`, `app:grid`, deviations)

| id | §/where | Reported | Source file | Producing script | status |
|---|---|---|---|---|---|
| E1 | App. intervention ¶1–2 | ABL/ADD each support in 1/10 adjudicated cells; manipulation check 0/20; C shifts +2.5 to +3.8 across directions; layer-matched ABL +0.009; injection −0.019 to −0.032; preregistered ABL clears guards 0/40 | `results/rq3_intervention_final_20260827/analysis/rq3b_{abl,add}_tests.csv`, `rq3f_*.csv`, `rq3b_report.md`, `rq3f_report.md` | `analysis/rq3/analyze_rq3_intervention.py`, `analyze_rq3_dis.py` ← `run_rq3_intervention_gpu.py`, `run_rq3_dis_gpu.py`, `run_rq3_fluency_guard_gpu.py` | csv, **aggregates not yet re-derived** (must be recomputed in notebook and compared) |
| E2 | App. intervention | LOIO cosine **+0.271 = 74.3°**; opposite side in **14/150** folds | `results/rq3_intervention_final_20260827/analysis/rq3x_subspace_k_k14_verify.csv` | `analysis/rq3/verify_discussion_numbers.py` (needs the activations to re-run; output shipped) | csv |
| E3 | App. table (k-sweep) | capture 0.178/0.212/0.242/0.289; random 0.00039/…; ratio 451×/219×/111×/49×; push cos 0.705; "ten model–wording cells" | `results/rq3_pcr_locked_20260825/analysis/rq3x_subspace_k_locked_20260825.csv` (k=1,2,3,5) + `verify_discussion_numbers.py` (k=14) | `analyze_rq3_subspace_k.py`, `verify_discussion_numbers.py` | csv/act ⚠ paper averages 10 cells; stored file has 12 rows |
| E4 | App. intervention | variance decomposition: item×model 20.8 % (ABL) · 24.4 % (ADD); arm main effect 0.04–0.11 % | **no stored output, no script found** | ?? | ?? |
| E5 | App. intervention | decoding AUC vs capture uncorrelated: Spearman **+0.20, p = 0.58**; cells 0.073→0.711, 0.279→0.671 | `rq3x_subspace_k_*.csv` + `rq3a_context_decoder_robustness.csv` | recomputed in notebook 05: **+0.200, p = 0.580**, matching the paper (an internal note had quoted a stale +0.31) | nb ✅ |
| E6 | Deviation 8 | Phi-4-mini-reasoning split-half 0.409 (Δρ), 0.193 (ΔC) | `results/rq3_pcr_locked_20260825/analysis/rq3x_norming_quality_locked_20260825.csv` | `analysis/rq3/analyze_rq3_norming_quality.py` | csv |
| E7 | Deviation 1 | retired 1-vs-3 AUC 0.559 (6/12) → 0.704 (12/12) | internal notebook output only; the retired gate file is not part of the canonical run | `analysis/rq3/analyze_rq3_representation.py` (retired labeling) | ?? |
| E8 | App. models | compute 56 GPU-h, peak 18.8 GB; model revisions; torch 2.9.1 / transformers 4.57.1 | `results/rq3_intervention_final_20260827/manifests/*.json` | — | csv (manifests contain server paths → scrub) |

## Items and prompts (Appendices `app:items`, `app:prompts`, `app:build`)

| id | Reported | Source | Script |
|---|---|---|---|
| S1 | 15 items, ORG fillers, norms | `data/stimuli_RQ3.csv` → `docs/paper/gen/items.tex` | `docs/paper/gen/make_appendix_items.py`; `analysis/rq3/make_stimuli_rq3.py` |
| S2 | 165 items | `additional_Qs/stimuli_RQ3_selected165.csv` | `additional_Qs/build.py`, `make_stimuli_rq3.py`, `analysis/04_norming_select.py` |
| S3 | generation prompts, judge rubric, accepted/rejected | `data/llm-build/{00_item_spec.md,prompts/,protocol/,checks/auto_checks.py,items/accepted,items/rejected}`; `additional_Qs/{accepted,rejected,slots}` | — |
| S4 | prompt wordings (frames, 6+6 task wordings, 3+3 norm wordings) | `analysis/rq3/rq3_prompting.py`, `sim/run_rq1_v2.py` (FRAME, QVAR), `sim/run_norm15.py` (QP) | — |
| S5 | WikiText 1,000-token fluency text | `data/wikitext_rq3_guard.txt` | `analysis/rq3/fetch_wikitext_rq3.py` |

---

## Flags

**(a) Reported but no stored output / code located** — E4 (variance decomposition), D5 (cos .335), E7 (retired AUC 0.559), the GPU-hours of E8. Marked ⛔ in the report.

**(b) Values that do not reproduce exactly** — listed with notes in `REPRODUCTION_REPORT.md` (18 of 175 checks): the 15-item mean DoD (exact 0.3125, printed 0.313 in Results and 0.312 in the table), the wording correlation (.96 printed, .93 recomputed over six models), Gemma's specificity failures (16/20 printed, 17/20 stored), three PPL ranges in Table `tab:cfpass`, the +0.009 and −0.019 to −0.032 of the grid appendix, the k=2 row and the random column of the k-sweep table, and the 'no model passes ADD and ABL on the same direction' statement, which holds per model × task cell but not per model (Qwen3-4B).

**(c) Values recomputed over the six-model subset** — B4 (15-item 67 % reproduces over six models), E3 (k-table averaged over the ten cells that passed the decoder gate in the grid run).

**(d) Code present but not reported (excluded from the release)** — internal variant scripts,
`analyze_rq3_common_axis*`, `analyze_rq3_pc_hierarchy.py` outputs (`pc_hierarchy/`), `analyze_rq3_variants_h3*`,
`additional_Qs/analysis/{01,02,03,05,06,08,09,11,12,13,14,20}_*.py`, `rq3_norm_component_20260902/`,
`additional_Qs/rq3_intervention_server/` (165-item preregistered-grid intervention, not in the paper),
TED-Q transfer (`run_rq3_transfer_gpu.py`), forced-choice readout (computed by `analyze_rq1_spec.py`, not interpreted).
`analyze_rq3_pc_hierarchy.py`, `analyze_rq3_h3b_geometry.py` and `analyze_rq3_representation.py` ship because other released scripts import them.

---

## Verifier notes (independent re-read of `main_A.tex`, 2026-09-05)

~165 numeric values across 38 rows checked: **0 value/sign mismatches.** Corrections and additions:

- **A7 over-inclusion.** The C-mediation intervals (−0.080 [−0.243,+0.141] · +0.094 [+0.053,+0.141]) are in
  the working notes, **not in the paper**. Keep them in the notebook as working-note values, labelled as such.
  The paper's cross-bank sign agreement is 1/6 for ρ **and 2/6 for C** (App. large).
- **Paper-internal inconsistency.** DoD^{R|N} at 15 items is 0.313 in Results ¶1 and 0.312 in Tab. `tab:decomp`.
  The notebook reports the recomputed value to three decimals so the tex can be made consistent.
  App. grid says "three refinements", App. intervention says "four refinements" (text only, no number to reproduce).
- **E5 discrepancy resolved in the paper's favour.** Spearman(capture_k1, AUC) recomputed from
  `rq3x_subspace_k_k14_verify.csv` × `rq3a_context_decoder_robustness.csv` = **+0.200, p = 0.580** (10 cells),
  lowest capture 0.073 → AUC 0.711, highest 0.279 → 0.671. An internal note that quoted +0.31, p=.33 was stale.
- **E2/E3 are stored**: `results/rq3_intervention_final_20260827/analysis/rq3x_subspace_k_k14_verify.csv`
  (10 cells; loo_cos 0.2708 → 74.3°, 14 negative folds; capture k1 0.178 / k5 0.242 / k14 0.289; push cos k14 0.705).
  ⚠ k=2 row: paper 0.212 · 0.00097 · 219×, stored 0.204 · 0.00088 · 232×.

Additional reported claims to carry as rows (added to the notebooks as presence/derivation checks):

| id | Reported | Source | status |
|---|---|---|---|
| A8 | wordings correlate ρ≈.96 over items (Method, Results ¶8) | item-level DoD by task in `logprob_item_effects.csv` / `h30_item_effects.csv` | nb |
| A9 | "held at 15 and 165 to within 0.05 DoD" (0.356−0.313) | A1 | nb |
| A10 | mediation intervals overlap across banks; sign agrees 1/6 (ρ), 2/6 (C) | A7 | nb |
| C9 | items per stage and cell range 45–131 | `cf_rescue_knockout_tests.csv` (`n`) | csv |
| C10 | full `tab:cfgrid` (10 rows × 4 values) and `tab:cfpass` (8 rows: strengths, tasks, ΔCF range, PPL range) | `cf_rescue_knockout_summary.csv`, `means_by_pass.csv`, `ladder.csv`, `by_model.csv`, `cells_annotated.csv` | csv |
| C11 | Fig S2a: six combinations pass at some injection strength | `dose_curve.csv` | csv |
| D6 | margins δ_ρ, δ_C: Qwen2.5-Math 0.025/0.009; Qwen3-4B 0.218/0.118; Qwen3-4B-Instruct 0.285/0.257 (Phi models not printed in the paper) | `norm_manipulation_margins.csv` | csv |
| D7 | deviation 6: θ=0.5 sensitivity margin — no result row in paper | `norm_manipulation_tests.csv` (theta column, if run) | ?? |
| E9 | deviation 9: position predicts the letter 66 % after counterbalancing | `results/rq1_spec_cached_20260821/analysis/position_gate.csv` | csv |
| E10 | deviation 8: 0.6 split-half floor | E6 | csv |
| E11 | App. grid: 20 ablation directions, 17 injection; λ∈{0.25,0.5,1}, α∈{0.5,1,2,4} | `rq3b_{abl,add}_tests.csv` unique directions/doses | csv |
| S6 | R vs I differ by ≤3 words in 50–57; 4 Chung + 3 edited + 8 generated; 180 released items | `data/stimuli_RQ3.csv`, `stimuli_RQ3_selected165.csv` | nb |
| E12 | models table: params 4.0/4.0/1.5/4.3/3.8/3.8 B, five fp16 + Gemma fp32, one TITAN RTX 24 GB, 56 GPU-h over five models | `results/*/manifests/*.json` | csv (presence) |
