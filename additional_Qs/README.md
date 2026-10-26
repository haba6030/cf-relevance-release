# additional_Qs/: 165-item bank, conditional intervention, manipulation check

Run from the repository root. `<M>` is one of the six model ids in the top-level README; use `--dtype float32` for Gemma.

**Items and norms**
```bash
python3 additional_Qs/build.py --promote && python3 additional_Qs/build.py --csv
python3 additional_Qs/run_norming.py --models <M> --dtype float16 --source additional --outdir additional_Qs/norming
python3 additional_Qs/run_norming.py --models <M> --dtype float16 --source active15  --outdir additional_Qs/norming_active15
python3 additional_Qs/analysis/04_norming_select.py --write additional_Qs/selected.txt      # gate G2 -> 165 items
python3 additional_Qs/make_stimuli_rq3.py --items additional_Qs/selected.txt --out additional_Qs/stimuli_RQ3_selected165.csv
```

**Behaviour, decoding, separation (Results ¶1–¶6, dark bars)**
```bash
RUN=additional_Qs/rq3_run_server; S=additional_Qs/stimuli_RQ3_selected165.csv
python3 additional_Qs/run_rq3_behavior.py       --models <M> --dtype float16 --stimuli $S --output-dir $RUN
python3 additional_Qs/run_rq3_representation.py --models <M> --dtype float16 --stimuli $S --output-dir $RUN
python3 additional_Qs/normalize_rq3_outputs.py  --run-dir $RUN
python3 additional_Qs/analysis/07_h30_anyN.py   --input-dir $RUN
python3 analysis/rq3/analyze_rq3_context_decoding_robustness.py --input-dir $RUN
python3 analysis/rq3/analyze_rq3_h3a_specificity_gap.py         --input-dir $RUN
python3 analysis/rq3/analyze_rq3_joint_pcr_iut.py               --input-dir $RUN
python3 additional_Qs/analysis/10_rq12_on_165.py
python3 additional_Qs/analysis/15_rpc_vector_control.py
```

**Conditional intervention and manipulation check (Results ¶7–¶8)**
```bash
python3 additional_Qs/queue_rq3_cf_rescue.py --output-dir additional_Qs/rq3_cf_rescue_20260830 --devices 0,1,2,3
python3 additional_Qs/analysis/16_analyze_cf_rescue_knockout.py --input-dir additional_Qs/rq3_cf_rescue_20260830
python3 additional_Qs/analysis/17_cf_rescue_reanalysis.py
python3 additional_Qs/analysis/18_cf_rescue_fluency_posthoc.py
python3 additional_Qs/queue_rq3_norm_manipulation.py --output-dir additional_Qs/rq3_norm_manipulation_20260902 --devices 0,1,2,3
python3 additional_Qs/analysis/19_analyze_norm_manipulation.py --input-dir additional_Qs/rq3_norm_manipulation_20260902
python3 additional_Qs/analysis/21_analyze_fig23b_nonjoint.py   --input-dir additional_Qs/rq3_norm_manipulation_20260902
```
The gates are documented in the docstrings of `cf_rescue_common.py` and `run_rq3_norm_manipulation_gpu.py`.
The two cell tables are gzipped; `pandas.read_csv` opens them directly.
