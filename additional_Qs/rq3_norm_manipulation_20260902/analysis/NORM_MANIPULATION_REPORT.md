# P/C manipulation check of the 0830 intervention

The RQ3-v2 norms P/C were re-measured under the same LOIO direction, layer, seed and ADD/ABL hook.
R|PC passes if both P and C are TOST-equivalent; PC|R passes if both oriented P and C show superiority.
θ is fixed at .25. The joint hypothesis over the two components is an IUT, so the joint p is the maximum of the two raw p-values. Holm-corrected values are neither computed nor used.

- R|PC: P and C both equivalent in 8/12 strength cells
- PC|R: P and C both move in the predicted direction in 0/13 strength cells

## θ=.25: R|PC and PC|R

| model | task | stage | direction | strength | source | P Δdid | P raw p | C Δdid | C raw p | joint IUT raw p | pass |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-Math-1.5B | 2 | ABL | PC_given_R | 0.5 | 0830_primary_pass | +0.0026 | 0.9209 | +0.0004 | 0.6852 | 0.9209 | no |
| Qwen2.5-Math-1.5B | 2 | ABL | PC_given_R | 1 | 0830_primary_pass | +0.0050 | 0.9242 | +0.0005 | 0.6891 | 0.9242 | no |
| Qwen3-4B | 2 | ABL | PC_given_R | 0.5 | 0830_primary_pass | -0.0381 | 0.001837 | -0.0146 | 0.05146 | 0.05146 | no |
| Qwen3-4B | 2 | ABL | PC_given_R | 1 | 0901_posthoc_fluency_six | -0.1729 | 6.359e-06 | -0.0123 | 0.2149 | 0.2149 | no |
| Qwen3-4B | 3 | ADD | PC_given_R | 0.5 | 0830_primary_pass | -0.0634 | 0.9999 | +0.0207 | 0.0004316 | 0.9999 | no |
| Qwen3-4B | 3 | ADD | PC_given_R | 1 | 0830_primary_pass | -0.1071 | 0.9998 | +0.0381 | 0.0003643 | 0.9998 | no |
| Qwen3-4B | 3 | ADD | PC_given_R | 2 | 0901_posthoc_fluency_six | -0.1715 | 0.9999 | +0.0670 | 0.0003404 | 0.9999 | no |
| Qwen3-4B-Instruct-2507 | 2 | ADD | PC_given_R | 0.5 | 0830_primary_pass | -0.0268 | 0.9044 | +0.0687 | 0.009927 | 0.9044 | no |
| Qwen3-4B-Instruct-2507 | 2 | ADD | PC_given_R | 1 | 0830_primary_pass | -0.0730 | 0.9572 | +0.1074 | 0.03804 | 0.9572 | no |
| Qwen3-4B-Instruct-2507 | 2 | ADD | PC_given_R | 2 | 0901_posthoc_fluency_six | -0.2215 | 0.9928 | +0.0429 | 0.344 | 0.9928 | no |
| Qwen3-4B-Instruct-2507 | 3 | ADD | PC_given_R | 0.5 | 0830_primary_pass | -0.0483 | 0.998 | +0.0358 | 0.0263 | 0.998 | no |
| Qwen3-4B-Instruct-2507 | 3 | ADD | PC_given_R | 1 | 0830_primary_pass | -0.0985 | 0.9972 | +0.0501 | 0.1093 | 0.9972 | no |
| Qwen3-4B-Instruct-2507 | 3 | ADD | PC_given_R | 2 | 0901_posthoc_fluency_six | -0.1774 | 0.9847 | -0.0151 | 0.5714 | 0.9847 | no |
| Qwen2.5-Math-1.5B | 3 | ADD | R_given_PC | 0.5 | 0830_primary_pass | -0.0162 | 0.0003657 | +0.0035 | 0.001379 | 0.001379 | yes |
| Qwen2.5-Math-1.5B | 3 | ADD | R_given_PC | 1 | 0830_primary_pass | -0.0402 | 0.9931 | +0.0060 | 0.2103 | 0.9931 | no |
| Qwen3-4B | 2 | ABL | R_given_PC | 1 | 0901_posthoc_fluency_six | +0.0004 | 3.401e-24 | +0.0531 | 6.906e-11 | 6.906e-11 | yes |
| Qwen3-4B | 3 | ABL | R_given_PC | 0.5 | 0830_primary_pass | +0.0019 | 1.94e-101 | +0.0004 | 1.068e-100 | 1.068e-100 | yes |
| Qwen3-4B | 3 | ABL | R_given_PC | 1 | 0830_primary_pass | +0.0004 | 1.338e-70 | +0.0007 | 1.725e-86 | 1.338e-70 | yes |
| Phi-4-mini-instruct | 2 | ADD | R_given_PC | 0.5 | 0830_primary_pass | -0.0100 | 4.888e-16 | -0.0382 | 5.82e-09 | 5.82e-09 | yes |
| Phi-4-mini-instruct | 2 | ADD | R_given_PC | 1 | 0830_primary_pass | -0.0713 | 0.001406 | -0.1567 | 0.9961 | 0.9961 | no |
| Phi-4-mini-instruct | 2 | ADD | R_given_PC | 2 | 0901_posthoc_fluency_six | -0.2790 | 0.9414 | -0.2550 | 1 | 1 | no |
| Phi-4-mini-instruct | 3 | ADD | R_given_PC | 0.5 | 0830_primary_pass | -0.0280 | 2.158e-12 | -0.0263 | 6.127e-11 | 6.127e-11 | yes |
| Phi-4-mini-instruct | 3 | ADD | R_given_PC | 1 | 0830_primary_pass | -0.0953 | 0.01642 | -0.1277 | 0.9027 | 0.9027 | no |
| Phi-4-mini-reasoning | 3 | ABL | R_given_PC | 0.5 | 0830_primary_pass | -0.0143 | 1.119e-25 | +0.0048 | 6.374e-20 | 6.374e-20 | yes |
| Phi-4-mini-reasoning | 3 | ABL | R_given_PC | 1 | 0830_primary_pass | -0.0256 | 6.664e-11 | +0.0105 | 4.468e-10 | 4.468e-10 | yes |

## Equivalence margins

δ = per-model median(|baseline context DID|) × θ with θ fixed at .25. The margin is used only in the R|PC TOST, not in the PC|R superiority test.

| model_id                       | metric   |   theta |   anchor_median_abs_context_did |   margin |   n_anchor_items |
|:-------------------------------|:---------|--------:|--------------------------------:|---------:|-----------------:|
| Qwen/Qwen2.5-Math-1.5B         | P        |    0.25 |                        0.100911 | 0.025228 |              165 |
| Qwen/Qwen2.5-Math-1.5B         | C        |    0.25 |                        0.035807 | 0.008952 |              165 |
| Qwen/Qwen3-4B                  | P        |    0.25 |                        0.871093 | 0.217773 |              165 |
| Qwen/Qwen3-4B                  | C        |    0.25 |                        0.472655 | 0.118164 |              165 |
| Qwen/Qwen3-4B-Instruct-2507    | P        |    0.25 |                        1.13802  | 0.284505 |              165 |
| Qwen/Qwen3-4B-Instruct-2507    | C        |    0.25 |                        1.02865  | 0.257161 |              165 |
| microsoft/Phi-4-mini-instruct  | P        |    0.25 |                        0.697917 | 0.174479 |              165 |
| microsoft/Phi-4-mini-instruct  | C        |    0.25 |                        0.40625  | 0.101562 |              165 |
| microsoft/Phi-4-mini-reasoning | P        |    0.25 |                        0.585402 | 0.146351 |              165 |
| microsoft/Phi-4-mini-reasoning | C        |    0.25 |                        0.451015 | 0.112754 |              165 |

## Interpretive limits

The target combinations were chosen after the 0830 results were seen. This check qualifies the interpretation of those positive CF interventions; it is not an independent confirmatory run.
The p-values in the table are the uncorrected p of the preregistered t-based TOST / one-sided superiority tests for P and C. Joint IUT p = max(P p, C p); no Holm correction is applied.
