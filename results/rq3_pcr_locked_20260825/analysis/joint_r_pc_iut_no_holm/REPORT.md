# RQ3 joint residualisation of R against the shared PC, and IUT

## Summary

- Cells where both `R_perp_PC` and `PC_perp_R` are detected: **6/14**.
- Cells passing the full hypothesis including `P not perpendicular C`: **6/14**.
- Models passing the full hypothesis in both tasks: **3/7**.

## Consolidated verdict table

Parentheses hold raw p-values. `2/3` shows the two independently fit and evaluated task cells side by side, not a pooled value.

| Model                         | R_perp_PC (2/3)               | PC_perp_R (2/3)               | P_not_orthogonal_C        | Full IUT (2/3)                | Both tasks   |
|:------------------------------|:------------------------------|:------------------------------|:--------------------------|:------------------------------|:-------------|
| Qwen2.5-Math-1.5B             | PASS (<.0001) / PASS (0.0002) | FAIL (0.0659) / FAIL (0.3193) | FAIL (r=-0.010, p=0.9533) | FAIL (0.9533) / FAIL (0.9533) | FAIL         |
| Qwen3-4B                      | PASS (<.0001) / PASS (<.0001) | PASS (0.0175) / PASS (0.0177) | PASS (r=+0.836, p=0.0149) | PASS (0.0175) / PASS (0.0177) | PASS         |
| Qwen3-4B-Instruct-2507        | PASS (0.0002) / PASS (0.0001) | FAIL (0.1621) / FAIL (0.3282) | PASS (r=+0.663, p=0.0098) | FAIL (0.1621) / FAIL (0.3282) | FAIL         |
| DeepSeek-R1-Distill-Qwen-1.5B | PASS (0.0092) / PASS (0.0009) | FAIL (0.9378) / FAIL (0.9868) | FAIL (r=+0.146, p=0.5535) | FAIL (0.9378) / FAIL (0.9868) | FAIL         |
| gemma-3-4b-it                 | PASS (<.0001) / PASS (<.0001) | PASS (0.0258) / PASS (0.0016) | PASS (r=+0.635, p=0.0180) | PASS (0.0258) / PASS (0.0180) | PASS         |
| Phi-4-mini-instruct           | PASS (<.0001) / PASS (<.0001) | PASS (0.0073) / PASS (0.0149) | PASS (r=+0.864, p=0.0034) | PASS (0.0073) / PASS (0.0149) | PASS         |
| Phi-4-mini-reasoning          | PASS (<.0001) / PASS (<.0001) | FAIL (0.0833) / FAIL (0.1921) | FAIL (r=-0.000, p=0.9995) | FAIL (0.9995) / FAIL (0.9995) | FAIL         |

## PC construction and test contract

- delta-P and delta-C are standardised separately on the 14 training items of each fold.
- The shared target is `PC1=(zP + sign(corr(P,C))*zC)/sqrt(2)`, the exact PC1 of the two standardised variables.
- The input activations are the conjunction contrast `A(R,h3)-A(I,h3)` as before; tasks 2 and 3 are separate.
- `deltaA = beta_R*1 + beta_PC*PC1` is fit at the layer chosen by the canonical R nested LOIO.
- The direction IUT is `max(p_R_perp_PC, p_PC_perp_R)`.
- The full IUT is the maximum of those and the two-sided p of the P/C covariance.
- All p-values are raw exact item sign-flip p; Holm is not used.

## Check of the assumption P not perpendicular C

| model                         |   pc_correlation |   pc_covariance |   pc_association_p_two |   pc1_full_sample_variance_share | pc_association_pass   |
|:------------------------------|-----------------:|----------------:|-----------------------:|---------------------------------:|:----------------------|
| Qwen2.5-Math-1.5B             |          -0.0102 |         -0.0001 |                 0.9533 |                           0.5051 | False                 |
| Qwen3-4B                      |           0.8364 |          1.7910 |                 0.0149 |                           0.9182 | True                  |
| Qwen3-4B-Instruct-2507        |           0.6626 |          4.2254 |                 0.0098 |                           0.8313 | True                  |
| DeepSeek-R1-Distill-Qwen-1.5B |           0.1464 |          0.0020 |                 0.5535 |                           0.5732 | False                 |
| gemma-3-4b-it                 |           0.6354 |          2.2765 |                 0.0180 |                           0.8177 | True                  |
| Phi-4-mini-instruct           |           0.8638 |          0.1906 |                 0.0034 |                           0.9319 | True                  |
| Phi-4-mini-reasoning          |          -0.0002 |         -0.0001 |                 0.9995 |                           0.5001 | False                 |

The sign of the correlation only orients the C axis in PC1; the assumption itself is two-sided, so a significant negative correlation also counts as 'non-orthogonal'.

## Model × task IUT

| model                         |   task |     PC |      R |   direction_iut_p |   full_iut_p | directions_pass   | full_hypothesis_pass   |
|:------------------------------|-------:|-------:|-------:|------------------:|-------------:|:------------------|:-----------------------|
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | 0.9378 | 0.0092 |            0.9378 |       0.9378 | False             | False                  |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | 0.9868 | 0.0009 |            0.9868 |       0.9868 | False             | False                  |
| Phi-4-mini-instruct           |      2 | 0.0073 | 0.0000 |            0.0073 |       0.0073 | True              | True                   |
| Phi-4-mini-instruct           |      3 | 0.0149 | 0.0000 |            0.0149 |       0.0149 | True              | True                   |
| Phi-4-mini-reasoning          |      2 | 0.0833 | 0.0000 |            0.0833 |       0.9995 | False             | False                  |
| Phi-4-mini-reasoning          |      3 | 0.1921 | 0.0001 |            0.1921 |       0.9995 | False             | False                  |
| Qwen2.5-Math-1.5B             |      2 | 0.0659 | 0.0001 |            0.0659 |       0.9533 | False             | False                  |
| Qwen2.5-Math-1.5B             |      3 | 0.3193 | 0.0002 |            0.3193 |       0.9533 | False             | False                  |
| Qwen3-4B                      |      2 | 0.0175 | 0.0000 |            0.0175 |       0.0175 | True              | True                   |
| Qwen3-4B                      |      3 | 0.0177 | 0.0001 |            0.0177 |       0.0177 | True              | True                   |
| Qwen3-4B-Instruct-2507        |      2 | 0.1621 | 0.0002 |            0.1621 |       0.1621 | False             | False                  |
| Qwen3-4B-Instruct-2507        |      3 | 0.3282 | 0.0001 |            0.3282 |       0.3282 | False             | False                  |
| gemma-3-4b-it                 |      2 | 0.0258 | 0.0001 |            0.0258 |       0.0258 | True              | True                   |
| gemma-3-4b-it                 |      3 | 0.0016 | 0.0001 |            0.0016 |       0.0180 | True              | True                   |

## Results by direction

| model                         |   task | target   |   mean_covariance |   boot_cov_lo |   boot_cov_hi |   p_flip_one |   pearson |   residual_variance_ratio | component_pass   |
|:------------------------------|-------:|:---------|------------------:|--------------:|--------------:|-------------:|----------:|--------------------------:|:-----------------|
| Qwen2.5-Math-1.5B             |      2 | R        |            0.5109 |        0.2977 |        0.7841 |       0.0001 |   -0.2304 |                    1.0000 | True             |
| Qwen2.5-Math-1.5B             |      2 | PC       |            0.0700 |       -0.0127 |        0.1538 |       0.0659 |    0.1765 |                    1.0000 | False            |
| Qwen2.5-Math-1.5B             |      3 | R        |            0.4210 |        0.2749 |        0.5702 |       0.0002 |   -0.1904 |                    1.0000 | True             |
| Qwen2.5-Math-1.5B             |      3 | PC       |            0.0250 |       -0.0825 |        0.1210 |       0.3193 |    0.0932 |                    1.0000 | False            |
| Qwen3-4B                      |      2 | R        |            0.4652 |        0.2721 |        0.6912 |       0.0000 |   -0.2304 |                    1.0000 | True             |
| Qwen3-4B                      |      2 | PC       |            0.5691 |        0.0877 |        1.1998 |       0.0175 |    0.5046 |                    1.0000 | True             |
| Qwen3-4B                      |      3 | R        |            0.5325 |        0.3050 |        0.8041 |       0.0001 |   -0.2071 |                    1.0000 | True             |
| Qwen3-4B                      |      3 | PC       |            0.4461 |        0.0747 |        0.9333 |       0.0177 |    0.3368 |                    1.0000 | True             |
| Qwen3-4B-Instruct-2507        |      2 | R        |            0.5583 |        0.3210 |        0.8279 |       0.0002 |   -0.5706 |                    1.0000 | True             |
| Qwen3-4B-Instruct-2507        |      2 | PC       |            0.5230 |       -0.0720 |        1.5436 |       0.1621 |    0.4759 |                    1.0000 | False            |
| Qwen3-4B-Instruct-2507        |      3 | R        |            0.6161 |        0.3560 |        0.9221 |       0.0001 |   -0.5707 |                    1.0000 | True             |
| Qwen3-4B-Instruct-2507        |      3 | PC       |            0.0521 |       -0.1455 |        0.2637 |       0.3282 |    0.2507 |                    1.0000 | False            |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | R        |            0.2565 |        0.0230 |        0.6732 |       0.0092 |    0.1016 |                    1.0000 | True             |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | PC       |           -0.0842 |       -0.1807 |        0.0170 |       0.9378 |   -0.4568 |                    1.0000 | False            |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | R        |            0.2788 |        0.0553 |        0.6692 |       0.0009 |    0.0949 |                    1.0000 | True             |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | PC       |           -0.0909 |       -0.1622 |       -0.0240 |       0.9868 |   -0.4959 |                    1.0000 | False            |
| gemma-3-4b-it                 |      2 | R        |            0.6114 |        0.4097 |        0.8113 |       0.0001 |   -0.4072 |                    1.0000 | True             |
| gemma-3-4b-it                 |      2 | PC       |            0.4378 |        0.0570 |        0.8685 |       0.0258 |    0.4583 |                    1.0000 | True             |
| gemma-3-4b-it                 |      3 | R        |            0.6813 |        0.4629 |        0.8935 |       0.0001 |   -0.4729 |                    1.0000 | True             |
| gemma-3-4b-it                 |      3 | PC       |            0.6567 |        0.2476 |        1.1261 |       0.0016 |    0.5862 |                    1.0000 | True             |
| Phi-4-mini-instruct           |      2 | R        |            0.5081 |        0.3328 |        0.7292 |       0.0000 |    0.0037 |                    1.0000 | True             |
| Phi-4-mini-instruct           |      2 | PC       |            0.4111 |        0.1285 |        0.7399 |       0.0073 |    0.4658 |                    1.0000 | True             |
| Phi-4-mini-instruct           |      3 | R        |            0.5648 |        0.3872 |        0.7787 |       0.0000 |    0.1173 |                    1.0000 | True             |
| Phi-4-mini-instruct           |      3 | PC       |            0.4067 |        0.1075 |        0.7435 |       0.0149 |    0.4636 |                    1.0000 | True             |
| Phi-4-mini-reasoning          |      2 | R        |            0.4179 |        0.2703 |        0.5850 |       0.0000 |   -0.4634 |                    1.0000 | True             |
| Phi-4-mini-reasoning          |      2 | PC       |            0.0469 |       -0.0079 |        0.1115 |       0.0833 |    0.3726 |                    1.0000 | False            |
| Phi-4-mini-reasoning          |      3 | R        |            0.3918 |        0.2260 |        0.5863 |       0.0001 |   -0.5417 |                    1.0000 | True             |
| Phi-4-mini-reasoning          |      3 | PC       |            0.0364 |       -0.0269 |        0.1111 |       0.1921 |    0.2323 |                    1.0000 | False            |

## Per-model IUT requiring both tasks

| model_id                                  | model                         |   task2_full_iut_p |   task3_full_iut_p |   model_iut_p | both_tasks_pass   |
|:------------------------------------------|:------------------------------|-------------------:|-------------------:|--------------:|:------------------|
| Qwen/Qwen2.5-Math-1.5B                    | Qwen2.5-Math-1.5B             |             0.9533 |             0.9533 |        0.9533 | False             |
| Qwen/Qwen3-4B                             | Qwen3-4B                      |             0.0175 |             0.0177 |        0.0177 | True              |
| Qwen/Qwen3-4B-Instruct-2507               | Qwen3-4B-Instruct-2507        |             0.1621 |             0.3282 |        0.3282 | False             |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-1.5B |             0.9378 |             0.9868 |        0.9868 | False             |
| google/gemma-3-4b-it                      | gemma-3-4b-it                 |             0.0258 |             0.0180 |        0.0258 | True              |
| microsoft/Phi-4-mini-instruct             | Phi-4-mini-instruct           |             0.0073 |             0.0149 |        0.0149 | True              |
| microsoft/Phi-4-mini-reasoning            | Phi-4-mini-reasoning          |             0.9995 |             0.9995 |        0.9995 | False             |

## Intersection over the fixed set of all seven models

Full IUT p over all models and tasks = **0.9995**; verdict = **fail**.

## Interpretive limits

This analysis tests the weaker representational model in which P and C share one linear common axis. Passing means R and the shared PC each retain a held-out self-signal after linear control for the other. It does not mean causal independence or a zero Euclidean inner product. In models where the P/C non-orthogonality assumption fails, a PC1-direction result is not read confirmatorily as a 'shared P/C component'.
