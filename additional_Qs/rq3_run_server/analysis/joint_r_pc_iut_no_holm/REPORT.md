# RQ3 joint residualisation of R against the shared PC, and IUT

## Summary

- Cells where both `R_perp_PC` and `PC_perp_R` are detected: **10/14**.
- Cells passing the full hypothesis including `P not perpendicular C`: **10/14**.
- Models passing the full hypothesis in both tasks: **5/7**.

## Consolidated verdict table

Parentheses hold raw p-values. `2/3` shows the two independently fit and evaluated task cells side by side, not a pooled value.

| Model                  | R_perp_PC (2/3)               | PC_perp_R (2/3)               | P_not_orthogonal_C        | Full IUT (2/3)                | Both tasks   |
|:-----------------------|:------------------------------|:------------------------------|:--------------------------|:------------------------------|:-------------|
| Qwen2.5-Math-1.5B      | PASS (<.0001) / PASS (<.0001) | PASS (<.0001) / PASS (<.0001) | PASS (r=+0.163, p=0.0468) | PASS (0.0468) / PASS (0.0468) | PASS         |
| Qwen3-4B               | PASS (<.0001) / PASS (<.0001) | PASS (<.0001) / PASS (<.0001) | PASS (r=+0.600, p=<.0001) | PASS (<.0001) / PASS (<.0001) | PASS         |
| Qwen3-4B-Instruct-2507 | PASS (<.0001) / PASS (<.0001) | PASS (0.0124) / PASS (0.0111) | PASS (r=+0.658, p=<.0001) | PASS (0.0124) / PASS (0.0111) | PASS         |
| gemma-3-4b-it          | PASS (<.0001) / PASS (<.0001) | PASS (0.0004) / PASS (0.0005) | PASS (r=+0.565, p=<.0001) | PASS (0.0004) / PASS (0.0005) | PASS         |
| Phi-4-mini-instruct    | PASS (<.0001) / PASS (<.0001) | PASS (<.0001) / PASS (<.0001) | PASS (r=+0.658, p=<.0001) | PASS (<.0001) / PASS (<.0001) | PASS         |
| Phi-4-mini-reasoning   | PASS (<.0001) / PASS (<.0001) | FAIL (0.5271) / FAIL (0.4259) | FAIL (r=+0.018, p=0.8375) | FAIL (0.8375) / FAIL (0.8375) | FAIL         |

## PC construction and test contract

- delta-P and delta-C are standardised separately on the 14 training items of each fold.
- The shared target is `PC1=(zP + sign(corr(P,C))*zC)/sqrt(2)`, the exact PC1 of the two standardised variables.
- The input activations are the conjunction contrast `A(R,h3)-A(I,h3)` as before; tasks 2 and 3 are separate.
- `deltaA = beta_R*1 + beta_PC*PC1` is fit at the layer chosen by the canonical R nested LOIO.
- The direction IUT is `max(p_R_perp_PC, p_PC_perp_R)`.
- The full IUT is the maximum of those and the two-sided p of the P/C covariance.
- All p-values are raw exact item sign-flip p; Holm is not used.

## Check of the assumption P not perpendicular C

| model                  |   pc_correlation |   pc_covariance |   pc_association_p_two |   pc1_full_sample_variance_share | pc_association_pass   |
|:-----------------------|-----------------:|----------------:|-----------------------:|---------------------------------:|:----------------------|
| Qwen2.5-Math-1.5B      |           0.1633 |          0.0031 |                 0.0468 |                           0.5816 | True                  |
| Qwen3-4B               |           0.6000 |          0.8106 |                 0.0000 |                           0.8000 | True                  |
| Qwen3-4B-Instruct-2507 |           0.6582 |          2.1686 |                 0.0000 |                           0.8291 | True                  |
| gemma-3-4b-it          |           0.5646 |          2.0563 |                 0.0000 |                           0.7823 | True                  |
| Phi-4-mini-instruct    |           0.6579 |          0.2578 |                 0.0000 |                           0.8290 | True                  |
| Phi-4-mini-reasoning   |           0.0181 |          0.0106 |                 0.8375 |                           0.5090 | False                 |

The sign of the correlation only orients the C axis in PC1; the assumption itself is two-sided, so a significant negative correlation also counts as 'non-orthogonal'.

## Model × task IUT

| model                  |   task |     PC |      R |   direction_iut_p |   full_iut_p | directions_pass   | full_hypothesis_pass   |
|:-----------------------|-------:|-------:|-------:|------------------:|-------------:|:------------------|:-----------------------|
| Phi-4-mini-instruct    |      2 | 0.0000 | 0.0000 |            0.0000 |       0.0000 | True              | True                   |
| Phi-4-mini-instruct    |      3 | 0.0000 | 0.0000 |            0.0000 |       0.0000 | True              | True                   |
| Phi-4-mini-reasoning   |      2 | 0.5271 | 0.0000 |            0.5271 |       0.8375 | False             | False                  |
| Phi-4-mini-reasoning   |      3 | 0.4259 | 0.0000 |            0.4259 |       0.8375 | False             | False                  |
| Qwen2.5-Math-1.5B      |      2 | 0.0000 | 0.0000 |            0.0000 |       0.0468 | True              | True                   |
| Qwen2.5-Math-1.5B      |      3 | 0.0000 | 0.0000 |            0.0000 |       0.0468 | True              | True                   |
| Qwen3-4B               |      2 | 0.0001 | 0.0000 |            0.0001 |       0.0001 | True              | True                   |
| Qwen3-4B               |      3 | 0.0001 | 0.0000 |            0.0001 |       0.0001 | True              | True                   |
| Qwen3-4B-Instruct-2507 |      2 | 0.0124 | 0.0000 |            0.0124 |       0.0124 | True              | True                   |
| Qwen3-4B-Instruct-2507 |      3 | 0.0111 | 0.0000 |            0.0111 |       0.0111 | True              | True                   |
| gemma-3-4b-it          |      2 | 0.0004 | 0.0000 |            0.0004 |       0.0004 | True              | True                   |
| gemma-3-4b-it          |      3 | 0.0005 | 0.0000 |            0.0005 |       0.0005 | True              | True                   |

## Results by direction

| model                  |   task | target   |   mean_covariance |   boot_cov_lo |   boot_cov_hi |   p_flip_one |   pearson |   residual_variance_ratio | component_pass   |
|:-----------------------|-------:|:---------|------------------:|--------------:|--------------:|-------------:|----------:|--------------------------:|:-----------------|
| Qwen2.5-Math-1.5B      |      2 | R        |            0.5700 |        0.4974 |        0.6458 |       0.0000 |   -0.1482 |                    1.0000 | True             |
| Qwen2.5-Math-1.5B      |      2 | PC       |            0.3096 |        0.2071 |        0.4216 |       0.0000 |    0.4934 |                    1.0000 | True             |
| Qwen2.5-Math-1.5B      |      3 | R        |            0.5840 |        0.5102 |        0.6597 |       0.0000 |   -0.1453 |                    1.0000 | True             |
| Qwen2.5-Math-1.5B      |      3 | PC       |            0.3220 |        0.2177 |        0.4378 |       0.0000 |    0.5030 |                    1.0000 | True             |
| Qwen3-4B               |      2 | R        |            0.6869 |        0.6162 |        0.7581 |       0.0000 |    0.1268 |                    1.0000 | True             |
| Qwen3-4B               |      2 | PC       |            0.2601 |        0.1241 |        0.4083 |       0.0001 |    0.3439 |                    1.0000 | True             |
| Qwen3-4B               |      3 | R        |            0.6820 |        0.6122 |        0.7527 |       0.0000 |    0.1561 |                    1.0000 | True             |
| Qwen3-4B               |      3 | PC       |            0.2877 |        0.1392 |        0.4521 |       0.0001 |    0.3619 |                    1.0000 | True             |
| Qwen3-4B-Instruct-2507 |      2 | R        |            0.6710 |        0.5993 |        0.7462 |       0.0000 |   -0.0538 |                    1.0000 | True             |
| Qwen3-4B-Instruct-2507 |      2 | PC       |            0.1153 |        0.0146 |        0.2171 |       0.0124 |    0.1911 |                    1.0000 | True             |
| Qwen3-4B-Instruct-2507 |      3 | R        |            0.6771 |        0.6054 |        0.7510 |       0.0000 |   -0.0434 |                    1.0000 | True             |
| Qwen3-4B-Instruct-2507 |      3 | PC       |            0.1109 |        0.0152 |        0.2078 |       0.0111 |    0.1871 |                    1.0000 | True             |
| gemma-3-4b-it          |      2 | R        |            0.7192 |        0.6546 |        0.7862 |       0.0000 |    0.0403 |                    1.0000 | True             |
| gemma-3-4b-it          |      2 | PC       |            0.1636 |        0.0659 |        0.2570 |       0.0004 |    0.2416 |                    1.0000 | True             |
| gemma-3-4b-it          |      3 | R        |            0.7187 |        0.6523 |        0.7862 |       0.0000 |    0.0561 |                    1.0000 | True             |
| gemma-3-4b-it          |      3 | PC       |            0.1539 |        0.0601 |        0.2452 |       0.0005 |    0.2483 |                    1.0000 | True             |
| Phi-4-mini-instruct    |      2 | R        |            0.7368 |        0.6719 |        0.8037 |       0.0000 |    0.0864 |                    1.0000 | True             |
| Phi-4-mini-instruct    |      2 | PC       |            0.5949 |        0.4312 |        0.7785 |       0.0000 |    0.5781 |                    1.0000 | True             |
| Phi-4-mini-instruct    |      3 | R        |            0.7424 |        0.6777 |        0.8089 |       0.0000 |    0.1029 |                    1.0000 | True             |
| Phi-4-mini-instruct    |      3 | PC       |            0.5944 |        0.4319 |        0.7801 |       0.0000 |    0.5763 |                    1.0000 | True             |
| Phi-4-mini-reasoning   |      2 | R        |            0.5691 |        0.4991 |        0.6430 |       0.0000 |    0.0692 |                    1.0000 | True             |
| Phi-4-mini-reasoning   |      2 | PC       |           -0.0023 |       -0.0709 |        0.0636 |       0.5271 |   -0.0058 |                    1.0000 | False            |
| Phi-4-mini-reasoning   |      3 | R        |            0.5752 |        0.5047 |        0.6501 |       0.0000 |    0.0806 |                    1.0000 | True             |
| Phi-4-mini-reasoning   |      3 | PC       |            0.0065 |       -0.0601 |        0.0723 |       0.4259 |    0.0146 |                    1.0000 | False            |

## Per-model IUT requiring both tasks

| model_id                       | model                  |   task2_full_iut_p |   task3_full_iut_p |   model_iut_p | both_tasks_pass   |
|:-------------------------------|:-----------------------|-------------------:|-------------------:|--------------:|:------------------|
| Qwen/Qwen2.5-Math-1.5B         | Qwen2.5-Math-1.5B      |             0.0468 |             0.0468 |        0.0468 | True              |
| Qwen/Qwen3-4B                  | Qwen3-4B               |             0.0001 |             0.0001 |        0.0001 | True              |
| Qwen/Qwen3-4B-Instruct-2507    | Qwen3-4B-Instruct-2507 |             0.0124 |             0.0111 |        0.0124 | True              |
| google/gemma-3-4b-it           | gemma-3-4b-it          |             0.0004 |             0.0005 |        0.0005 | True              |
| microsoft/Phi-4-mini-instruct  | Phi-4-mini-instruct    |             0.0000 |             0.0000 |        0.0000 | True              |
| microsoft/Phi-4-mini-reasoning | Phi-4-mini-reasoning   |             0.8375 |             0.8375 |        0.8375 | False             |

## Intersection over the fixed set of all seven models

Full IUT p over all models and tasks = **0.8375**; verdict = **fail**.

## Interpretive limits

This analysis tests the weaker representational model in which P and C share one linear common axis. Passing means R and the shared PC each retain a held-out self-signal after linear control for the other. It does not mean causal independence or a zero Euclidean inner product. In models where the P/C non-orthogonality assumption fails, a PC1-direction result is not read confirmatorily as a 'shared P/C component'.
