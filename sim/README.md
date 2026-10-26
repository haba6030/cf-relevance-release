# sim/: 15-item behaviour, norms, and figures

Run from the repository root. `<M>` is one of the six model ids in the top-level README.

**Behaviour and norms (Results ¶1–¶4)**
```bash
python3 analysis/rq3/make_stimuli_rq3.py        # data/stimuli_RQ3.csv
python3 analysis/rq3/fetch_wikitext_rq3.py      # data/wikitext_rq3_guard.txt
python3 sim/run_rq1_spec_gpu.py --model <M> --output-dir results/rq1_spec_cached_20260821
python3 sim/analyze_rq1_spec.py --input-dir results/rq1_spec_cached_20260821   # logprob_cell_tests.csv, norming_*.csv
```
`sim/queue_rq1_spec_gpu.py` runs the per-model command over several GPUs.

**Figures and the appendix item table**
```bash
python3 sim/make_paper_figs_2runs.py            # fig1_behavior.png, fig2_representation.png
python3 sim/make_paper_fig3_0830.py             # fig3_cf_intervention.png, figS2_intervention_detail.png
python3 sim/make_paper_fig_pipeline.py          # figS1_decoding.png (needs activations; cached scores are shipped)
python3 docs/paper/gen/make_appendix_items.py   # docs/paper/gen/items.tex
```
Use a new `--output-dir` to keep the shipped outputs.
