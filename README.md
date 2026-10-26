# Conversational Relevance in the Conjunction Fallacy Is Represented Independently in Language Models

Code, stimuli, derived data and executed notebooks for the paper of the same title, by Jinil Kim and Jaejun Ryu
(Seoul National University), presented at the NeurIPS 2026 Workshop on Linguistic Principles for Foundation Models
(LP4FM). Paper: https://openreview.net/forum?id=aVWJk84zeg (non-archival).

The paper asks whether the *conversational relevance* account of the conjunction fallacy holds in six open-weight
language models: whether context raises the relevant conjunct (behaviour), whether relevance is a linearly decodable
component of the residual stream that survives removing the posterior-probability and confirmation directions
(representation), and whether editing that residual component changes the fallacy while leaving the evidential norms
inside an equivalence margin fixed before the manipulation check (intervention and manipulation check).

**Everything the paper reports can be recomputed here from the shipped CSVs without a GPU.** The GPU stages that
produced those CSVs are included as scripts with their exact commands (Section 4), but their raw outputs
(activations, per-item edit logs) are too large to ship.

## 1. Quick start: reproduce every number in the paper

```bash
git clone https://github.com/haba6030/cf-relevance-release.git && cd cf-relevance-release
python3 -m pip install numpy pandas scipy scikit-learn matplotlib nbformat nbconvert jupyter
python3 notebooks/build_notebooks.py --execute     # ~1 minute on a laptop, CPU only
```

The notebooks are also shipped **already executed**, so they can simply be opened:

| notebook | paper section | what it checks |
|---|---|---|
| `notebooks/00_summary.ipynb` | all | every reported number at a glance, with verdict counts, plus the figures |
| `notebooks/01_behavior_and_norms.ipynb` | Results ¶1–¶4, Fig. 1, Table `tab:decomp` | context effect, prefix decomposition, bare arm, norm shifts, confounding and mediation associations |
| `notebooks/02_representation.ipynb` | Results ¶5–¶6, Fig. 2, Fig. S1 | held-out decoding, specificity gaps, R\|PC and PC\|R separation, projection fractions |
| `notebooks/03_conditional_intervention.ipynb` | Results ¶7, Fig. 3, Fig. S2a, Tables `tab:cfgrid`, `tab:cfpass`, `tab:models` | the ten passing combinations, cell means, fluency, Gemma |
| `notebooks/04_manipulation_check.ipynb` | Results ¶8, Fig. S2b–c, Table `tab:manip` | equivalence and superiority tests, controls, margins |
| `notebooks/05_preregistered_grid.ipynb` | Appendices `app:grid`, `app:intervention`, `app:models`, deviations | the null of the preregistered grid and its geometry |

Each notebook opens with a **source and code map** (paper number → CSV read → script that produced it) and ends with a
table of reported vs produced values. `notebooks/_checks_all.csv` collects those tables across the six notebooks, with
one row per checked number. As of this release: 135 of 175 checks reproduce to the printed precision, 5 are
seed-dependent bootstrap bounds within 0.01, 12 are qualitative or presence checks, 18 do not match exactly and 5
cannot be recomputed from the released files. The last two groups carry a note in that file.

## 2. Layout

The directory layout of the working repository is kept, because the scripts locate their inputs by relative paths
anchored at the repository root.

```
data/                      15-item bank (stimuli_RQ3.csv), its construction record (llm-build/), WikiText fluency text
additional_Qs/             165-item bank (stimuli_RQ3_selected165.csv), item generation (build.py, slots/, accepted/),
                           its GPU runners, the conditional intervention and manipulation check, and their analyses
sim/                       15-item behaviour + norming runner and analysis (RQ1), figure scripts, Chung carrier
analysis/rq3/              15-item prefix decomposition, activation extraction, decoding, separation, preregistered grid
results/                   15-item runs: behaviour/norming CSVs, analysis outputs, run manifests (no activations)
additional_Qs/rq3_run_server/          165-item behaviour/norming CSVs and analysis outputs
additional_Qs/rq3_cf_rescue_20260830/  conditional intervention analysis (raw shards not shipped)
additional_Qs/rq3_norm_manipulation_20260902/  manipulation-check analysis
docs/paper/figs/paper2/    the figures as printed (fig1, fig2, fig3, figS1, figS2) and the appendix item table source
notebooks/                 reproduction notebooks (this README, Section 1)
```

Two large cell-level tables are shipped gzipped (`cf_rescue_knockout_cells.csv.gz`, `norm_manipulation_cells.csv.gz`);
`pandas.read_csv` opens them directly.

## 3. The six models and the two item banks

| model | params | precision | HF revision |
|---|---|---|---|
| Qwen/Qwen3-4B | 4.0B | fp16 | 1cfa9a7 |
| Qwen/Qwen3-4B-Instruct-2507 | 4.0B | fp16 | cdbee75 |
| Qwen/Qwen2.5-Math-1.5B | 1.5B | fp16 | 4a83ca6 |
| google/gemma-3-4b-it | 4.3B | fp32 | 093f9f3 |
| microsoft/Phi-4-mini-instruct | 3.8B | fp16 | cfbefac |
| microsoft/Phi-4-mini-reasoning | 3.8B | fp16 | 0e3b1e2 |

* **15-item anchored bank** — `data/stimuli_RQ3.csv` (45 rows = 15 items × prefixes N/I/R). Four items adapt
  Chung et al. (2024), three edit Tentori et al. (2013) hypotheses into the Chung carrier, eight were generated
  under `data/llm-build/prompts/10_generate_item.md` and screened by `data/llm-build/checks/auto_checks.py` and the
  judge protocol in `data/llm-build/protocol/`. Prefix `R` and `I` differ only in the ORG noun phrase
  (`sim/chung_frame.py`).
* **165-item generalisation bank** — `additional_Qs/stimuli_RQ3_selected165.csv` (495 rows). 200 candidates
  (`additional_Qs/slots/*.json`, assembled and checked by `additional_Qs/build.py` into `accepted/`) were screened by
  measured norms (`additional_Qs/analysis/04_norming_select.py`, gate G2) to 165.
* Prompt wordings: `analysis/rq3/rq3_prompting.py` (behaviour and norming, contract `rq3-v2-raw-sequence-20260825`),
  `sim/run_rq1_v2.py` and `sim/run_norm15.py` (15-item RQ1 run).

## 4. Regenerating the results from scratch (GPU)

Every run used one 24 GB GPU with `torch==2.9.1` (CUDA 12.8) and `transformers==4.57.1`; `requirements.txt` pins
them. `<M>` is one of the six model ids above. Directories named below are the ones the shipped CSVs came from;
use a new `--output-dir` to avoid overwriting them.

**15-item bank: behaviour and norms (Results ¶1–¶4)**
```bash
python3 analysis/rq3/make_stimuli_rq3.py                       # data/stimuli_RQ3.csv from data/stimuli_00_all.csv
python3 analysis/rq3/fetch_wikitext_rq3.py                     # data/wikitext_rq3_guard.txt
python3 sim/run_rq1_spec_gpu.py --model <M> --output-dir results/rq1_spec_cached_20260821      # per model
python3 sim/analyze_rq1_spec.py --input-dir results/rq1_spec_cached_20260821                    # logprob_cell_tests.csv, norming_*.csv
```

**15-item bank: prefix decomposition, activations, decoding, separation (Results ¶1, ¶5–¶6)**
```bash
RUN=results/rq3_pcr_locked_20260825
python3 analysis/rq3/run_rq3_behavior_gpu.py       --model <M> --output-dir $RUN
python3 analysis/rq3/analyze_rq3_h30.py            --input-dir $RUN                            # h30_tests.csv
python3 analysis/rq3/run_rq3_representation_gpu.py --model <M> --output-dir $RUN               # activations/<model>/task{2,3}_cellmean.npy, norming/
python3 analysis/rq3/analyze_rq3_context_decoding_robustness.py --input-dir $RUN               # rq3a_context_decoder_robustness.csv
python3 analysis/rq3/analyze_rq3_h3a_specificity_gap.py         --input-dir $RUN               # rq3a_specificity_gap_*.csv
python3 analysis/rq3/analyze_rq3_joint_pcr_iut.py               --input-dir $RUN               # joint_r_pc_iut_no_holm/component_tests.csv
python3 analysis/rq3/analyze_rq3_norming_quality.py             --input-dir $RUN               # split-half reliability (deviation 8)
```
The queue scripts (`analysis/rq3/queue_rq3_*_gpu.py`, `sim/queue_rq1_spec_gpu.py`) run the per-model commands over
several GPUs with restart-safe manifests.

**15-item bank: preregistered intervention grid (Appendix)**
```bash
RUN=results/rq3_intervention_final_20260827      # behaviour, norming and activations copied from the run above
python3 analysis/rq3/run_rq3_fluency_guard_gpu.py --model <M> --output-dir $RUN --direction-mode h3_subspace --subspace-k 2 --ablate-scope both
python3 analysis/rq3/run_rq3_intervention_gpu.py  --model <M> --output-dir $RUN --direction-mode h3_subspace --subspace-k 2 --ablate-scope both
python3 analysis/rq3/run_rq3_dis_gpu.py           --model <M> --output-dir $RUN                # manipulation check of the grid
python3 analysis/rq3/analyze_rq3_intervention.py        --input-dir $RUN                       # rq3b_{abl,add}_tests.csv
python3 analysis/rq3/analyze_rq3_dis.py                 --input-dir $RUN                       # rq3f_*.csv
python3 analysis/rq3/analyze_rq3_intervention_design.py --input-dir $RUN                       # subspace diagnostics
python3 analysis/rq3/verify_discussion_numbers.py       --input-dir $RUN                       # rq3x_subspace_k_k14_verify.csv (LOO cosine, k-sweep)
```
`analysis/rq3/analyze_rq3_subspace_k.py` reads its run directory from a constant inside the file.

**165-item bank: items, behaviour, norms, decoding, separation (Appendix `app:large`, Results ¶1–¶6 dark bars)**
```bash
python3 additional_Qs/build.py --promote && python3 additional_Qs/build.py --csv                 # slots/ -> accepted/ -> stimuli_RQ3_additional.csv
python3 additional_Qs/run_norming.py --models <M> --dtype float16 --source additional --outdir additional_Qs/norming
python3 additional_Qs/run_norming.py --models <M> --dtype float16 --source active15  --outdir additional_Qs/norming_active15
python3 additional_Qs/analysis/04_norming_select.py --write additional_Qs/selected.txt          # gate G2 -> 165 items
python3 additional_Qs/make_stimuli_rq3.py --items additional_Qs/selected.txt --out additional_Qs/stimuli_RQ3_selected165.csv
RUN=additional_Qs/rq3_run_server
python3 additional_Qs/run_rq3_behavior.py       --models <M> --dtype float16 --stimuli additional_Qs/stimuli_RQ3_selected165.csv --output-dir $RUN
python3 additional_Qs/run_rq3_representation.py --models <M> --dtype float16 --stimuli additional_Qs/stimuli_RQ3_selected165.csv --output-dir $RUN
python3 additional_Qs/normalize_rq3_outputs.py  --run-dir $RUN
python3 additional_Qs/analysis/07_h30_anyN.py   --input-dir $RUN                                # h30_tests.csv (Monte Carlo sign-flip at n=165)
python3 analysis/rq3/analyze_rq3_context_decoding_robustness.py --input-dir $RUN
python3 analysis/rq3/analyze_rq3_h3a_specificity_gap.py         --input-dir $RUN
python3 analysis/rq3/analyze_rq3_joint_pcr_iut.py               --input-dir $RUN
python3 additional_Qs/analysis/10_rq12_on_165.py                                                # additional_Qs/rq12_on_165/*.csv
python3 additional_Qs/analysis/15_rpc_vector_control.py                                         # additional_Qs/rpc_vector_control/cells.csv
```
Gemma is run with `--dtype float32` throughout (fp16 produced degenerate logits).

**Conditional intervention and manipulation check (Results ¶7–¶8, Appendix `app:cf`)**
```bash
python3 additional_Qs/queue_rq3_cf_rescue.py --output-dir additional_Qs/rq3_cf_rescue_20260830 --devices 0,1,2,3
#   (drives run_rq3_cf_rescue_gpu.py and run_rq3_cf_rescue_guard_gpu.py per model and shard; run_rq3_cf_rescue_linger.sh wraps it)
python3 additional_Qs/analysis/16_analyze_cf_rescue_knockout.py --input-dir additional_Qs/rq3_cf_rescue_20260830
python3 additional_Qs/analysis/17_cf_rescue_reanalysis.py                                       # additional_Qs/cf_rescue_reanalysis/
python3 additional_Qs/analysis/18_cf_rescue_fluency_posthoc.py                                  # additional_Qs/cf_rescue_fluency_posthoc/
python3 additional_Qs/queue_rq3_norm_manipulation.py --output-dir additional_Qs/rq3_norm_manipulation_20260902 --devices 0,1,2,3
python3 additional_Qs/analysis/19_analyze_norm_manipulation.py --input-dir additional_Qs/rq3_norm_manipulation_20260902
python3 additional_Qs/analysis/21_analyze_fig23b_nonjoint.py   --input-dir additional_Qs/rq3_norm_manipulation_20260902
```
The design and gates are in the module docstrings of `additional_Qs/cf_rescue_common.py` and
`additional_Qs/run_rq3_norm_manipulation_gpu.py`; the equations are in the paper's Method section.

**Figures and the appendix item table**
```bash
python3 sim/make_paper_figs_2runs.py        # fig1_behavior.png, fig2_representation.png
python3 sim/make_paper_fig3_0830.py         # fig3_cf_intervention.png, figS2_intervention_detail.png
python3 sim/make_paper_fig_pipeline.py      # figS1_decoding.png (recomputes the held-out ROC from the activations; the cached scores are shipped)
python3 docs/paper/gen/make_appendix_items.py   # docs/paper/gen/items.tex
```

Tests: `python3 -m pytest analysis/rq3/test_rq3_contracts.py analysis/rq3/test_rq3_joint_pcr_iut.py -q`.

## 5. What is not included, and why

| omitted | size | how to regenerate |
|---|---|---|
| residual-stream activations (`*/activations/*.npy`) | 1.2 GB (15 items) + 12 GB (165 items) | `run_rq3_representation_gpu.py` / `run_rq3_representation.py` (Section 4) |
| per-item edit logs of the conditional intervention and manipulation check (`shards/`, `guard_shards/`) | 130 MB | `queue_rq3_cf_rescue.py`, `queue_rq3_norm_manipulation.py` |
| the manipulation-check shards of the preregistered grid (`dis/`) | not retained | `run_rq3_dis_gpu.py` |
| the item-generation judge transcripts and an internal audit memo | — | the judge prompts and protocol are in `data/llm-build/` |
| external pragmatics corpora used to calibrate the judges | — | see `data/llm-build/protocol/calibration.md`; not redistributed for licence reasons |
| the 165-item preregistered-grid run (`rq3_intervention_server`) | 94 MB | not reported in the paper |

Five reported quantities cannot be recomputed from the released files and are marked ⛔ in the report:
the variance decomposition of the grid effect (item × model share), the retired one-versus-three decoder AUC (0.559),
$\cos(v_R, v_P)=.335$, the GPU-hours of the grid, and the exact cell set behind the grid's "0 of 20" manipulation
check. Each would need the activations or logs listed above.

## 6. Path placeholders

This repository was first released anonymously for double-blind review. Paths, user names and the original repository
URL were removed from every file at that time (`<REPO>`, `<HOME>`, `<HF_CACHE>` placeholders), and the placeholders are
kept so the files match what the reviewers saw. Comments and generated reports were translated into English. The
`_release_manifest.json` lists every file copied from the working repository.

## 7. Licence

Code: MIT. Stimuli and derived data: CC BY 4.0, except the texts adapted from Chung et al. (2024) and
Tentori et al. (2013) and the WikiText excerpt, which remain under their original terms (see `LICENSE`).

## 8. Citation

```bibtex
@inproceedings{kim2026relevance,
  title     = {Conversational Relevance in the Conjunction Fallacy Is Represented Independently in Language Models},
  author    = {Kim, Jinil and Ryu, Jaejun},
  booktitle = {NeurIPS 2026 Workshop on Linguistic Principles for Foundation Models (LP4FM)},
  year      = {2026},
  note      = {Non-archival}
}
```
