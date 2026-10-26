# analysis/rq3/: 15-item decoding, separation, and the preregistered grid

Run from the repository root. `<M>` is one of the six model ids in the top-level README.

**Prefix decomposition, activations, decoding, separation (Results ¶1, ¶5–¶6)**
```bash
RUN=results/rq3_pcr_locked_20260825
python3 analysis/rq3/run_rq3_behavior_gpu.py       --model <M> --output-dir $RUN
python3 analysis/rq3/run_rq3_representation_gpu.py --model <M> --output-dir $RUN
python3 analysis/rq3/analyze_rq3_h30.py                         --input-dir $RUN
python3 analysis/rq3/analyze_rq3_context_decoding_robustness.py --input-dir $RUN
python3 analysis/rq3/analyze_rq3_h3a_specificity_gap.py         --input-dir $RUN
python3 analysis/rq3/analyze_rq3_joint_pcr_iut.py               --input-dir $RUN
python3 analysis/rq3/analyze_rq3_norming_quality.py             --input-dir $RUN
```

**Preregistered intervention grid (Appendix)**
```bash
RUN=results/rq3_intervention_final_20260827   # behaviour, norming, activations copied from the run above
ARGS="--direction-mode h3_subspace --subspace-k 2 --ablate-scope both"
python3 analysis/rq3/run_rq3_fluency_guard_gpu.py --model <M> --output-dir $RUN $ARGS
python3 analysis/rq3/run_rq3_intervention_gpu.py  --model <M> --output-dir $RUN $ARGS
python3 analysis/rq3/run_rq3_dis_gpu.py           --model <M> --output-dir $RUN
python3 analysis/rq3/analyze_rq3_intervention.py        --input-dir $RUN
python3 analysis/rq3/analyze_rq3_dis.py                 --input-dir $RUN
python3 analysis/rq3/analyze_rq3_intervention_design.py --input-dir $RUN
python3 analysis/rq3/verify_discussion_numbers.py       --input-dir $RUN
```
`analyze_rq3_subspace_k.py` reads its run directory from a constant in the file. The `queue_rq3_*_gpu.py` scripts
spread the per-model commands over several GPUs.

Tests: `python3 -m pytest analysis/rq3/test_rq3_contracts.py analysis/rq3/test_rq3_joint_pcr_iut.py -q`
