# RQ3 H3f DIS and calibration

Only the maximum alpha passing fluency and base-collapse guards is used for endpoint tests. H3f-2 calibration uses the registered H2a/RQ1 R-N observational contrast; R-I remains the relevance-specific H3-0/intervention contrast.

| Model | Task | contrast | safe alpha | target monotonic | beta P | beta 95% CI | reach | currency rho | ratio | ratio 95% CI | DoD p | epsilon rho | TOST p | TOST Holm | conversion |
|---|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-4B-Instruct-2507 | 2 | R-N (H2a/RQ1) | 1 | True | -0.1569 | [-0.2508, -0.0630] | False | 0.350 | -3.540 | [-9.688, -1.553] | 0.0001 | 0.821 | 0.9994 | 1.0000 | True |

## Safe-alpha manipulation endpoints

P/C rows failing their preregistered self-decoding gate are retained but marked inferentially ineligible. The specification defines no composite-PC decoding gate, so PC is descriptive rather than silently assigned an ad-hoc pass/fail rule.

| Model | Task | direction | eligibility | mode | alpha | target change | target p | target monotonic | induced DoD | DoD p |
|---|---:|---|---|---|---:|---:|---:|---|---:|---:|
| Qwen2.5-Math-1.5B | 2 | C | inferential_arm_ineligible | common | 1 | -0.2790 | 0.0006 | False | -0.0866 | 0.0603 |
| Qwen2.5-Math-1.5B | 2 | C | inferential_arm_ineligible | differential | 1 | -0.5123 | 0.0018 | False | -0.1465 | 0.0218 |
| Qwen2.5-Math-1.5B | 2 | P | inferential_arm_ineligible | common | 0.5 | 0.2802 | 0.0001 | True | -0.0167 | 0.5084 |
| Qwen2.5-Math-1.5B | 2 | P | inferential_arm_ineligible | differential | 0.5 | 0.5528 | 0.0001 | True | 0.1005 | 0.0988 |
| Qwen2.5-Math-1.5B | 2 | PC | descriptive_only_no_preregistered_PC_gate | common | 1 | 2.9706 | 0.0001 | True | -0.0897 | 0.0547 |
| Qwen2.5-Math-1.5B | 2 | PC | descriptive_only_no_preregistered_PC_gate | differential | 1 | 4.1943 | 0.0003 | True | 0.0300 | 0.6891 |
| Qwen2.5-Math-1.5B | 3 | C | inferential_arm_ineligible | common | 1 | 0.0455 | 0.6326 | False | -0.0684 | 0.0603 |
| Qwen2.5-Math-1.5B | 3 | C | inferential_arm_ineligible | differential | 1 | 0.2465 | 0.3132 | False | -0.2355 | 0.0048 |
| Qwen2.5-Math-1.5B | 3 | P | inferential_arm_ineligible | common | 1 | 0.5724 | 0.0001 | True | -0.0626 | 0.1094 |
| Qwen2.5-Math-1.5B | 3 | P | inferential_arm_ineligible | differential | 1 | 0.8585 | 0.0001 | True | 0.0513 | 0.4771 |
| Qwen2.5-Math-1.5B | 3 | PC | descriptive_only_no_preregistered_PC_gate | common | 1 | 3.5348 | 0.0001 | True | -0.0805 | 0.0547 |
| Qwen2.5-Math-1.5B | 3 | PC | descriptive_only_no_preregistered_PC_gate | differential | 1 | 5.5355 | 0.0002 | True | -0.1220 | 0.1555 |
| Qwen3-4B | 2 | C | inferential_arm_ineligible | common | 1 | 4.4595 | 0.0002 | True | 0.0785 | 0.2810 |
| Qwen3-4B | 2 | C | inferential_arm_ineligible | differential | 1 | 7.3995 | 0.0004 | True | -0.1347 | 0.1158 |
| Qwen3-4B | 2 | P | inferential_arm_ineligible | common | 0.5 | 0.4432 | 0.0135 | False | -0.0332 | 0.2983 |
| Qwen3-4B | 2 | P | inferential_arm_ineligible | differential | 0.5 | 0.7984 | 0.0361 | False | -0.0556 | 0.3519 |
| Qwen3-4B | 2 | PC | descriptive_only_no_preregistered_PC_gate | common | 0.5 | 2.4470 | 0.0001 | True | 0.0048 | 0.8793 |
| Qwen3-4B | 2 | PC | descriptive_only_no_preregistered_PC_gate | differential | 0.5 | 4.2194 | 0.0001 | True | -0.1509 | 0.0695 |
| Qwen3-4B | 3 | C | inferential_arm_ineligible | common | 0.5 | 2.3164 | 0.0002 | True | 0.0070 | 0.8364 |
| Qwen3-4B | 3 | C | inferential_arm_ineligible | differential | 0.5 | 3.8177 | 0.0004 | True | -0.1601 | 0.0002 |
| Qwen3-4B | 3 | P | inferential_arm_ineligible | common | 0.5 | 0.5579 | 0.0135 | False | -0.0537 | 0.0531 |
| Qwen3-4B | 3 | P | inferential_arm_ineligible | differential | 0.5 | 1.0985 | 0.0361 | False | -0.0785 | 0.3516 |
| Qwen3-4B | 3 | PC | descriptive_only_no_preregistered_PC_gate | common | 0.5 | 2.5477 | 0.0001 | True | -0.0475 | 0.2462 |
| Qwen3-4B | 3 | PC | descriptive_only_no_preregistered_PC_gate | differential | 0.5 | 4.3444 | 0.0001 | True | -0.2520 | 0.0007 |
| Qwen3-4B-Instruct-2507 | 2 | C | inferential_arm_ineligible | common | 0.5 | 4.2256 | 0.0001 | True | 0.0485 | 0.0516 |
| Qwen3-4B-Instruct-2507 | 2 | C | inferential_arm_ineligible | differential | 0.5 | 7.6071 | 0.0001 | True | -0.0331 | 0.6777 |
| Qwen3-4B-Instruct-2507 | 2 | P | eligible | common | 1 | 2.8310 | 0.0005 | True | -0.1004 | 0.0136 |
| Qwen3-4B-Instruct-2507 | 2 | P | eligible | differential | 1 | 4.7733 | 0.0018 | True | -0.6355 | 0.0001 |
| Qwen3-4B-Instruct-2507 | 2 | PC | descriptive_only_no_preregistered_PC_gate | common | 0.5 | 3.0236 | 0.0001 | True | 0.0053 | 0.7961 |
| Qwen3-4B-Instruct-2507 | 2 | PC | descriptive_only_no_preregistered_PC_gate | differential | 0.5 | 5.5973 | 0.0001 | True | -0.4285 | 0.0001 |
| Qwen3-4B-Instruct-2507 | 3 | C | inferential_arm_ineligible | common | 1 | 3.7612 | 0.0001 | True | 0.0874 | 0.0516 |
| Qwen3-4B-Instruct-2507 | 3 | C | inferential_arm_ineligible | differential | 1 | 8.3926 | 0.0001 | True | -0.1002 | 0.6777 |
| Qwen3-4B-Instruct-2507 | 3 | P | inferential_arm_ineligible | common | 1 | 2.5353 | 0.0002 | True | -0.1239 | 0.0085 |
| Qwen3-4B-Instruct-2507 | 3 | P | inferential_arm_ineligible | differential | 1 | 5.4966 | 0.0001 | True | -0.5156 | 0.0001 |
| Qwen3-4B-Instruct-2507 | 3 | PC | descriptive_only_no_preregistered_PC_gate | common | 0.5 | 2.3008 | 0.0001 | True | -0.0141 | 0.7960 |
| Qwen3-4B-Instruct-2507 | 3 | PC | descriptive_only_no_preregistered_PC_gate | differential | 0.5 | 5.0631 | 0.0001 | True | -0.3534 | 0.0001 |
| gemma-3-4b-it | 2 | C | inferential_arm_ineligible | common | 4 | -0.0244 | 0.0002 | False | -0.0020 | 0.3131 |
| gemma-3-4b-it | 2 | C | inferential_arm_ineligible | differential | 4 | -0.0489 | 0.0002 | False | -0.0184 | 0.0001 |
| gemma-3-4b-it | 2 | P | inferential_arm_ineligible | common | 4 | -0.0213 | 0.0001 | False | -0.0001 | 1.0000 |
| gemma-3-4b-it | 2 | P | inferential_arm_ineligible | differential | 4 | -0.0426 | 0.0001 | False | 0.0048 | 0.0917 |
| gemma-3-4b-it | 2 | PC | descriptive_only_no_preregistered_PC_gate | common | 4 | 0.0023 | 0.0204 | True | -0.0034 | 0.0936 |
| gemma-3-4b-it | 2 | PC | descriptive_only_no_preregistered_PC_gate | differential | 4 | 0.0047 | 0.0208 | True | -0.0200 | 0.0001 |
| gemma-3-4b-it | 3 | C | inferential_arm_ineligible | common | 4 | 0.0013 | 0.7025 | False | -0.0013 | 0.3834 |
| gemma-3-4b-it | 3 | C | inferential_arm_ineligible | differential | 4 | 0.0025 | 0.7091 | False | -0.0197 | 0.0001 |
| gemma-3-4b-it | 3 | P | inferential_arm_ineligible | common | 4 | 0.0029 | 0.2124 | False | -0.0006 | 1.0000 |
| gemma-3-4b-it | 3 | P | inferential_arm_ineligible | differential | 4 | 0.0057 | 0.2148 | False | 0.0061 | 0.0917 |
| gemma-3-4b-it | 3 | PC | descriptive_only_no_preregistered_PC_gate | common | 4 | 0.0094 | 0.0001 | True | -0.0029 | 0.1132 |
| gemma-3-4b-it | 3 | PC | descriptive_only_no_preregistered_PC_gate | differential | 4 | 0.0189 | 0.0001 | True | -0.0190 | 0.0012 |
| Phi-4-mini-instruct | 2 | C | inferential_arm_ineligible | common | 1 | 0.5283 | 0.5044 | True | -0.0304 | 0.4806 |
| Phi-4-mini-instruct | 2 | C | inferential_arm_ineligible | differential | 1 | -0.1673 | 0.7698 | False | 0.1865 | 0.0258 |
| Phi-4-mini-instruct | 2 | P | inferential_arm_ineligible | common | 1 | -1.2208 | 0.0100 | False | -0.0105 | 1.0000 |
| Phi-4-mini-instruct | 2 | P | inferential_arm_ineligible | differential | 1 | -1.3705 | 0.2153 | False | -0.2538 | 0.0010 |
| Phi-4-mini-instruct | 2 | PC | descriptive_only_no_preregistered_PC_gate | common | 1 | -0.3003 | 0.3649 | False | -0.0620 | 0.0385 |
| Phi-4-mini-instruct | 2 | PC | descriptive_only_no_preregistered_PC_gate | differential | 1 | -2.2836 | 0.0001 | False | -0.0744 | 0.3870 |
| Phi-4-mini-instruct | 3 | C | inferential_arm_ineligible | common | 1 | 0.1523 | 0.7385 | True | -0.0052 | 0.8665 |
| Phi-4-mini-instruct | 3 | C | inferential_arm_ineligible | differential | 1 | -0.8076 | 0.4357 | False | 0.1670 | 0.0289 |
| Phi-4-mini-instruct | 3 | P | inferential_arm_ineligible | common | 1 | -0.6659 | 0.1210 | False | -0.0052 | 1.0000 |
| Phi-4-mini-instruct | 3 | P | inferential_arm_ineligible | differential | 1 | -1.2441 | 0.2153 | False | -0.2706 | 0.0006 |
| Phi-4-mini-instruct | 3 | PC | descriptive_only_no_preregistered_PC_gate | common | 1 | 0.2005 | 0.4210 | False | -0.0407 | 0.1740 |
| Phi-4-mini-instruct | 3 | PC | descriptive_only_no_preregistered_PC_gate | differential | 1 | -1.7195 | 0.0001 | False | -0.1016 | 0.3870 |

## R-direction norming specificity

Each row passes specificity when its 95% CI contains zero; all four rows in both ABL and safe-alpha ADD must pass per model/task/direction.

| Model | Task | direction | stage | alpha | metric | hypothesis | change | 95% CI | contains zero |
|---|---:|---|---|---:|---|---|---:|---:|---|
| Qwen2.5-Math-1.5B | 2 | R_base | ADD | 1 | rho | h2 | -0.8347 | [-1.1019, -0.5676] | False |
| Qwen2.5-Math-1.5B | 2 | R_base | ADD | 1 | rho | h3 | -0.7237 | [-0.9466, -0.5007] | False |
| Qwen2.5-Math-1.5B | 2 | R_base | ADD | 1 | C | h2 | 0.6280 | [0.4707, 0.7853] | False |
| Qwen2.5-Math-1.5B | 2 | R_base | ADD | 1 | C | h3 | 0.7174 | [0.5724, 0.8623] | False |
| Qwen2.5-Math-1.5B | 2 | R_h2 | ADD | 1 | rho | h2 | -0.6266 | [-0.8333, -0.4199] | False |
| Qwen2.5-Math-1.5B | 2 | R_h2 | ADD | 1 | rho | h3 | -0.5348 | [-0.6895, -0.3800] | False |
| Qwen2.5-Math-1.5B | 2 | R_h2 | ADD | 1 | C | h2 | -0.3141 | [-0.4659, -0.1623] | False |
| Qwen2.5-Math-1.5B | 2 | R_h2 | ADD | 1 | C | h3 | -0.2740 | [-0.4062, -0.1418] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3 | ADD | 1 | rho | h2 | 0.4693 | [0.2904, 0.6481] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3 | ADD | 1 | rho | h3 | 0.4385 | [0.2440, 0.6331] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3 | ADD | 1 | C | h2 | 0.4547 | [0.3220, 0.5875] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3 | ADD | 1 | C | h3 | 0.4498 | [0.3229, 0.5767] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_denoised | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_denoised | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_denoised | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_denoised | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_k2 | ADD | 0.5 | rho | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_k2 | ADD | 0.5 | rho | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_k2 | ADD | 0.5 | C | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_k2 | ADD | 0.5 | C | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_orth_PC | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_orth_PC | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_orth_PC | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3_orth_PC | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 2 | R_pooled | ADD | 1 | rho | h2 | 0.0284 | [-0.1801, 0.2370] | True |
| Qwen2.5-Math-1.5B | 2 | R_pooled | ADD | 1 | rho | h3 | 0.0308 | [-0.1614, 0.2230] | True |
| Qwen2.5-Math-1.5B | 2 | R_pooled | ADD | 1 | C | h2 | 0.2401 | [0.1113, 0.3688] | False |
| Qwen2.5-Math-1.5B | 2 | R_pooled | ADD | 1 | C | h3 | 0.2474 | [0.1251, 0.3698] | False |
| Qwen2.5-Math-1.5B | 3 | R_base | ADD | 1 | rho | h2 | 0.0279 | [-0.2498, 0.3055] | True |
| Qwen2.5-Math-1.5B | 3 | R_base | ADD | 1 | rho | h3 | 0.0804 | [-0.1636, 0.3244] | True |
| Qwen2.5-Math-1.5B | 3 | R_base | ADD | 1 | C | h2 | 0.5420 | [0.3901, 0.6939] | False |
| Qwen2.5-Math-1.5B | 3 | R_base | ADD | 1 | C | h3 | 0.5973 | [0.4271, 0.7675] | False |
| Qwen2.5-Math-1.5B | 3 | R_h2 | ADD | 1 | rho | h2 | 0.0906 | [-0.0822, 0.2633] | True |
| Qwen2.5-Math-1.5B | 3 | R_h2 | ADD | 1 | rho | h3 | 0.1063 | [-0.0460, 0.2586] | True |
| Qwen2.5-Math-1.5B | 3 | R_h2 | ADD | 1 | C | h2 | 0.0002 | [-0.0858, 0.0862] | True |
| Qwen2.5-Math-1.5B | 3 | R_h2 | ADD | 1 | C | h3 | 0.0241 | [-0.0702, 0.1183] | True |
| Qwen2.5-Math-1.5B | 3 | R_h3 | ADD | 1 | rho | h2 | 0.9774 | [0.7794, 1.1754] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3 | ADD | 1 | rho | h3 | 0.9207 | [0.7209, 1.1205] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3 | ADD | 1 | C | h2 | 0.4624 | [0.3091, 0.6156] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3 | ADD | 1 | C | h3 | 0.4970 | [0.3387, 0.6554] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_denoised | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_denoised | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_denoised | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_denoised | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_k2 | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_k2 | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_k2 | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_k2 | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_orth_PC | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_orth_PC | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_orth_PC | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3_orth_PC | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen2.5-Math-1.5B | 3 | R_pooled | ADD | 1 | rho | h2 | 0.7434 | [0.5770, 0.9099] | False |
| Qwen2.5-Math-1.5B | 3 | R_pooled | ADD | 1 | rho | h3 | 0.6733 | [0.5382, 0.8083] | False |
| Qwen2.5-Math-1.5B | 3 | R_pooled | ADD | 1 | C | h2 | 0.4068 | [0.2689, 0.5446] | False |
| Qwen2.5-Math-1.5B | 3 | R_pooled | ADD | 1 | C | h3 | 0.4375 | [0.3013, 0.5737] | False |
| Qwen3-4B | 2 | R_h3 | ADD | 1 | rho | h2 | 1.4825 | [0.6995, 2.2654] | False |
| Qwen3-4B | 2 | R_h3 | ADD | 1 | rho | h3 | 1.9156 | [1.2727, 2.5586] | False |
| Qwen3-4B | 2 | R_h3 | ADD | 1 | C | h2 | 3.9872 | [3.1355, 4.8388] | False |
| Qwen3-4B | 2 | R_h3 | ADD | 1 | C | h3 | 3.9082 | [3.0000, 4.8165] | False |
| Qwen3-4B | 2 | R_h3_denoised | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_denoised | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_denoised | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_denoised | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_k2 | ADD | 0.5 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_k2 | ADD | 0.5 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_k2 | ADD | 0.5 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_k2 | ADD | 0.5 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_orth_PC | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_orth_PC | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_orth_PC | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_h3_orth_PC | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 2 | R_pooled | ADD | 0.5 | rho | h2 | 0.5148 | [0.1743, 0.8552] | False |
| Qwen3-4B | 2 | R_pooled | ADD | 0.5 | rho | h3 | 0.7025 | [0.4084, 0.9966] | False |
| Qwen3-4B | 2 | R_pooled | ADD | 0.5 | C | h2 | 1.7192 | [1.2068, 2.2316] | False |
| Qwen3-4B | 2 | R_pooled | ADD | 0.5 | C | h3 | 1.4778 | [1.0370, 1.9186] | False |
| Qwen3-4B | 3 | R_h3 | ADD | 1 | rho | h2 | 1.2291 | [0.3722, 2.0860] | False |
| Qwen3-4B | 3 | R_h3 | ADD | 1 | rho | h3 | 1.6171 | [0.7525, 2.4818] | False |
| Qwen3-4B | 3 | R_h3 | ADD | 1 | C | h2 | 3.3165 | [2.4725, 4.1606] | False |
| Qwen3-4B | 3 | R_h3 | ADD | 1 | C | h3 | 3.2100 | [2.2243, 4.1958] | False |
| Qwen3-4B | 3 | R_h3_denoised | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_denoised | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_denoised | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_denoised | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_k2 | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_k2 | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_k2 | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_k2 | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_orth_PC | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_orth_PC | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_orth_PC | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_h3_orth_PC | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B | 3 | R_pooled | ADD | 1 | rho | h2 | 0.7319 | [0.0379, 1.4260] | False |
| Qwen3-4B | 3 | R_pooled | ADD | 1 | rho | h3 | 1.1339 | [0.4747, 1.7931] | False |
| Qwen3-4B | 3 | R_pooled | ADD | 1 | C | h2 | 2.9917 | [2.4944, 3.4890] | False |
| Qwen3-4B | 3 | R_pooled | ADD | 1 | C | h3 | 2.8355 | [2.1382, 3.5328] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_base | ADD | 1 | rho | h2 | 1.0451 | [-0.0024, 2.0926] | True |
| Qwen3-4B-Instruct-2507 | 2 | R_base | ADD | 1 | rho | h3 | 0.9316 | [-0.3094, 2.1725] | True |
| Qwen3-4B-Instruct-2507 | 2 | R_base | ADD | 1 | C | h2 | 1.7780 | [0.5582, 2.9977] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_base | ADD | 1 | C | h3 | 2.4231 | [0.9737, 3.8724] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h2 | ADD | 1 | rho | h2 | -4.7925 | [-7.0161, -2.5689] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h2 | ADD | 1 | rho | h3 | -4.5829 | [-6.5304, -2.6355] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h2 | ADD | 1 | C | h2 | -7.1899 | [-10.6562, -3.7237] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h2 | ADD | 1 | C | h3 | -6.0328 | [-8.3928, -3.6729] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3 | ADD | 0.5 | rho | h2 | -0.7522 | [-1.3982, -0.1062] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3 | ADD | 0.5 | rho | h3 | -0.7472 | [-1.3723, -0.1221] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3 | ADD | 0.5 | C | h2 | -0.1317 | [-0.6770, 0.4137] | True |
| Qwen3-4B-Instruct-2507 | 2 | R_h3 | ADD | 0.5 | C | h3 | -0.0691 | [-0.8343, 0.6961] | True |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_denoised | ADD | 0.5 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_denoised | ADD | 0.5 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_denoised | ADD | 0.5 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_denoised | ADD | 0.5 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_k2 | ADD | 0.5 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_k2 | ADD | 0.5 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_k2 | ADD | 0.5 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_k2 | ADD | 0.5 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_orth_PC | ADD | 0.5 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_orth_PC | ADD | 0.5 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_orth_PC | ADD | 0.5 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3_orth_PC | ADD | 0.5 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_pooled | ADD | 0.5 | rho | h2 | -1.2011 | [-1.8137, -0.5885] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_pooled | ADD | 0.5 | rho | h3 | -1.1710 | [-1.8510, -0.4909] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_pooled | ADD | 0.5 | C | h2 | -1.4378 | [-2.4316, -0.4439] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_pooled | ADD | 0.5 | C | h3 | -1.4576 | [-2.3846, -0.5305] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_base | ADD | 1 | rho | h2 | 2.1710 | [0.6461, 3.6958] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_base | ADD | 1 | rho | h3 | 2.6294 | [0.9085, 4.3504] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_base | ADD | 1 | C | h2 | 2.2873 | [0.7466, 3.8279] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_base | ADD | 1 | C | h3 | 3.6681 | [1.5201, 5.8161] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h2 | ADD | 1 | rho | h2 | -3.7029 | [-5.8402, -1.5657] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h2 | ADD | 1 | rho | h3 | -3.4401 | [-5.3884, -1.4917] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h2 | ADD | 1 | C | h2 | -5.6076 | [-8.6396, -2.5755] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h2 | ADD | 1 | C | h3 | -4.1672 | [-6.2260, -2.1084] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3 | ADD | 1 | rho | h2 | 1.6621 | [0.1061, 3.2180] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3 | ADD | 1 | rho | h3 | 1.6480 | [0.2950, 3.0010] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3 | ADD | 1 | C | h2 | 0.9963 | [-0.0644, 2.0571] | True |
| Qwen3-4B-Instruct-2507 | 3 | R_h3 | ADD | 1 | C | h3 | 2.2708 | [0.7636, 3.7781] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_denoised | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_denoised | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_denoised | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_denoised | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_k2 | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_k2 | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_k2 | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_k2 | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_orth_PC | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_orth_PC | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_orth_PC | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3_orth_PC | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_pooled | ADD | 1 | rho | h2 | 0.8203 | [-0.4637, 2.1043] | True |
| Qwen3-4B-Instruct-2507 | 3 | R_pooled | ADD | 1 | rho | h3 | 0.7710 | [-0.2917, 1.8337] | True |
| Qwen3-4B-Instruct-2507 | 3 | R_pooled | ADD | 1 | C | h2 | 0.0026 | [-0.8036, 0.8089] | True |
| Qwen3-4B-Instruct-2507 | 3 | R_pooled | ADD | 1 | C | h3 | 0.9246 | [0.0390, 1.8101] | False |
| gemma-3-4b-it | 2 | R_base | ADD | 4 | rho | h2 | -0.0109 | [-0.0160, -0.0059] | False |
| gemma-3-4b-it | 2 | R_base | ADD | 4 | rho | h3 | -0.0109 | [-0.0160, -0.0058] | False |
| gemma-3-4b-it | 2 | R_base | ADD | 4 | C | h2 | -0.0361 | [-0.0450, -0.0272] | False |
| gemma-3-4b-it | 2 | R_base | ADD | 4 | C | h3 | -0.0362 | [-0.0453, -0.0272] | False |
| gemma-3-4b-it | 2 | R_h2 | ADD | 4 | rho | h2 | -0.0025 | [-0.0082, 0.0033] | True |
| gemma-3-4b-it | 2 | R_h2 | ADD | 4 | rho | h3 | -0.0023 | [-0.0079, 0.0033] | True |
| gemma-3-4b-it | 2 | R_h2 | ADD | 4 | C | h2 | -0.0202 | [-0.0231, -0.0173] | False |
| gemma-3-4b-it | 2 | R_h2 | ADD | 4 | C | h3 | -0.0203 | [-0.0232, -0.0173] | False |
| gemma-3-4b-it | 2 | R_h3 | ADD | 4 | rho | h2 | 0.0322 | [0.0306, 0.0339] | False |
| gemma-3-4b-it | 2 | R_h3 | ADD | 4 | rho | h3 | 0.0329 | [0.0311, 0.0348] | False |
| gemma-3-4b-it | 2 | R_h3 | ADD | 4 | C | h2 | -0.0260 | [-0.0303, -0.0218] | False |
| gemma-3-4b-it | 2 | R_h3 | ADD | 4 | C | h3 | -0.0259 | [-0.0300, -0.0217] | False |
| gemma-3-4b-it | 2 | R_h3_denoised | ADD | 4 | rho | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_denoised | ADD | 4 | rho | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_denoised | ADD | 4 | C | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_denoised | ADD | 4 | C | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_k2 | ADD | 4 | rho | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_k2 | ADD | 4 | rho | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_k2 | ADD | 4 | C | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_k2 | ADD | 4 | C | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_orth_PC | ADD | 4 | rho | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_orth_PC | ADD | 4 | rho | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_orth_PC | ADD | 4 | C | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_h3_orth_PC | ADD | 4 | C | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 2 | R_pooled | ADD | 4 | rho | h2 | 0.0272 | [0.0246, 0.0298] | False |
| gemma-3-4b-it | 2 | R_pooled | ADD | 4 | rho | h3 | 0.0279 | [0.0255, 0.0304] | False |
| gemma-3-4b-it | 2 | R_pooled | ADD | 4 | C | h2 | -0.0281 | [-0.0321, -0.0242] | False |
| gemma-3-4b-it | 2 | R_pooled | ADD | 4 | C | h3 | -0.0280 | [-0.0319, -0.0241] | False |
| gemma-3-4b-it | 3 | R_base | ADD | 4 | rho | h2 | 0.0042 | [-0.0023, 0.0108] | True |
| gemma-3-4b-it | 3 | R_base | ADD | 4 | rho | h3 | 0.0044 | [-0.0022, 0.0110] | True |
| gemma-3-4b-it | 3 | R_base | ADD | 4 | C | h2 | -0.0279 | [-0.0336, -0.0221] | False |
| gemma-3-4b-it | 3 | R_base | ADD | 4 | C | h3 | -0.0280 | [-0.0338, -0.0222] | False |
| gemma-3-4b-it | 3 | R_h2 | ADD | 4 | rho | h2 | -0.0108 | [-0.0176, -0.0040] | False |
| gemma-3-4b-it | 3 | R_h2 | ADD | 4 | rho | h3 | -0.0108 | [-0.0176, -0.0040] | False |
| gemma-3-4b-it | 3 | R_h2 | ADD | 4 | C | h2 | -0.0300 | [-0.0331, -0.0270] | False |
| gemma-3-4b-it | 3 | R_h2 | ADD | 4 | C | h3 | -0.0301 | [-0.0332, -0.0270] | False |
| gemma-3-4b-it | 3 | R_h3 | ADD | 4 | rho | h2 | 0.0409 | [0.0390, 0.0427] | False |
| gemma-3-4b-it | 3 | R_h3 | ADD | 4 | rho | h3 | 0.0416 | [0.0395, 0.0436] | False |
| gemma-3-4b-it | 3 | R_h3 | ADD | 4 | C | h2 | -0.0269 | [-0.0302, -0.0236] | False |
| gemma-3-4b-it | 3 | R_h3 | ADD | 4 | C | h3 | -0.0267 | [-0.0298, -0.0236] | False |
| gemma-3-4b-it | 3 | R_h3_denoised | ADD | 4 | rho | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_denoised | ADD | 4 | rho | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_denoised | ADD | 4 | C | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_denoised | ADD | 4 | C | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_k2 | ADD | 4 | rho | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_k2 | ADD | 4 | rho | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_k2 | ADD | 4 | C | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_k2 | ADD | 4 | C | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_orth_PC | ADD | 4 | rho | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_orth_PC | ADD | 4 | rho | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_orth_PC | ADD | 4 | C | h2 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_h3_orth_PC | ADD | 4 | C | h3 | nan | [nan, nan] | False |
| gemma-3-4b-it | 3 | R_pooled | ADD | 4 | rho | h2 | 0.0342 | [0.0316, 0.0369] | False |
| gemma-3-4b-it | 3 | R_pooled | ADD | 4 | rho | h3 | 0.0349 | [0.0323, 0.0374] | False |
| gemma-3-4b-it | 3 | R_pooled | ADD | 4 | C | h2 | -0.0308 | [-0.0341, -0.0276] | False |
| gemma-3-4b-it | 3 | R_pooled | ADD | 4 | C | h3 | -0.0307 | [-0.0338, -0.0276] | False |
| Phi-4-mini-instruct | 2 | R_base | ADD | 1 | rho | h2 | 0.0531 | [-0.6430, 0.7493] | True |
| Phi-4-mini-instruct | 2 | R_base | ADD | 1 | rho | h3 | 0.3488 | [-0.0341, 0.7317] | True |
| Phi-4-mini-instruct | 2 | R_base | ADD | 1 | C | h2 | 1.1622 | [0.6333, 1.6911] | False |
| Phi-4-mini-instruct | 2 | R_base | ADD | 1 | C | h3 | 1.3104 | [0.7206, 1.9001] | False |
| Phi-4-mini-instruct | 2 | R_h2 | ADD | 1 | rho | h2 | 0.5703 | [-0.0780, 1.2187] | True |
| Phi-4-mini-instruct | 2 | R_h2 | ADD | 1 | rho | h3 | 0.6596 | [0.1666, 1.1526] | False |
| Phi-4-mini-instruct | 2 | R_h2 | ADD | 1 | C | h2 | 1.5292 | [1.0584, 2.0000] | False |
| Phi-4-mini-instruct | 2 | R_h2 | ADD | 1 | C | h3 | 1.7195 | [1.3472, 2.0917] | False |
| Phi-4-mini-instruct | 2 | R_h3 | ADD | 1 | rho | h2 | 1.2499 | [0.6247, 1.8750] | False |
| Phi-4-mini-instruct | 2 | R_h3 | ADD | 1 | rho | h3 | 1.5921 | [1.0860, 2.0981] | False |
| Phi-4-mini-instruct | 2 | R_h3 | ADD | 1 | C | h2 | 1.7802 | [1.2004, 2.3600] | False |
| Phi-4-mini-instruct | 2 | R_h3 | ADD | 1 | C | h3 | 2.3214 | [1.8210, 2.8218] | False |
| Phi-4-mini-instruct | 2 | R_h3_denoised | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_denoised | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_denoised | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_denoised | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_k2 | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_k2 | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_k2 | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_k2 | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_orth_PC | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_orth_PC | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_orth_PC | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_h3_orth_PC | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 2 | R_pooled | ADD | 1 | rho | h2 | 1.2725 | [0.6981, 1.8469] | False |
| Phi-4-mini-instruct | 2 | R_pooled | ADD | 1 | rho | h3 | 1.5565 | [1.1011, 2.0118] | False |
| Phi-4-mini-instruct | 2 | R_pooled | ADD | 1 | C | h2 | 1.9352 | [1.4083, 2.4620] | False |
| Phi-4-mini-instruct | 2 | R_pooled | ADD | 1 | C | h3 | 2.4382 | [2.0055, 2.8709] | False |
| Phi-4-mini-instruct | 3 | R_base | ADD | 1 | rho | h2 | 0.5680 | [-0.1099, 1.2459] | True |
| Phi-4-mini-instruct | 3 | R_base | ADD | 1 | rho | h3 | 0.8808 | [0.4095, 1.3521] | False |
| Phi-4-mini-instruct | 3 | R_base | ADD | 1 | C | h2 | 1.4700 | [0.9573, 1.9827] | False |
| Phi-4-mini-instruct | 3 | R_base | ADD | 1 | C | h3 | 1.7837 | [1.2439, 2.3235] | False |
| Phi-4-mini-instruct | 3 | R_h2 | ADD | 1 | rho | h2 | 0.7715 | [0.3062, 1.2368] | False |
| Phi-4-mini-instruct | 3 | R_h2 | ADD | 1 | rho | h3 | 0.8256 | [0.4313, 1.2199] | False |
| Phi-4-mini-instruct | 3 | R_h2 | ADD | 1 | C | h2 | 1.5776 | [1.2542, 1.9009] | False |
| Phi-4-mini-instruct | 3 | R_h2 | ADD | 1 | C | h3 | 1.8362 | [1.4807, 2.1917] | False |
| Phi-4-mini-instruct | 3 | R_h3 | ADD | 1 | rho | h2 | 1.1646 | [0.5778, 1.7514] | False |
| Phi-4-mini-instruct | 3 | R_h3 | ADD | 1 | rho | h3 | 1.7820 | [1.2286, 2.3353] | False |
| Phi-4-mini-instruct | 3 | R_h3 | ADD | 1 | C | h2 | 1.4345 | [1.0935, 1.7754] | False |
| Phi-4-mini-instruct | 3 | R_h3 | ADD | 1 | C | h3 | 2.1814 | [1.8239, 2.5389] | False |
| Phi-4-mini-instruct | 3 | R_h3_denoised | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_denoised | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_denoised | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_denoised | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_k2 | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_k2 | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_k2 | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_k2 | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_orth_PC | ADD | 1 | rho | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_orth_PC | ADD | 1 | rho | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_orth_PC | ADD | 1 | C | h2 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_h3_orth_PC | ADD | 1 | C | h3 | nan | [nan, nan] | False |
| Phi-4-mini-instruct | 3 | R_pooled | ADD | 1 | rho | h2 | 1.1970 | [0.6862, 1.7078] | False |
| Phi-4-mini-instruct | 3 | R_pooled | ADD | 1 | rho | h3 | 1.7038 | [1.2311, 2.1766] | False |
| Phi-4-mini-instruct | 3 | R_pooled | ADD | 1 | C | h2 | 1.6185 | [1.3102, 1.9268] | False |
| Phi-4-mini-instruct | 3 | R_pooled | ADD | 1 | C | h3 | 2.3173 | [1.9856, 2.6489] | False |
| Qwen2.5-Math-1.5B | 2 | R_base | ABL | — | rho | h2 | 0.0716 | [-0.0499, 0.1932] | True |
| Qwen2.5-Math-1.5B | 2 | R_base | ABL | — | rho | h3 | 0.0931 | [-0.0237, 0.2099] | True |
| Qwen2.5-Math-1.5B | 2 | R_base | ABL | — | C | h2 | 0.5936 | [0.4615, 0.7258] | False |
| Qwen2.5-Math-1.5B | 2 | R_base | ABL | — | C | h3 | 0.5929 | [0.4438, 0.7419] | False |
| Qwen2.5-Math-1.5B | 2 | R_h2 | ABL | — | rho | h2 | -0.5594 | [-0.7717, -0.3471] | False |
| Qwen2.5-Math-1.5B | 2 | R_h2 | ABL | — | rho | h3 | -0.4435 | [-0.6027, -0.2843] | False |
| Qwen2.5-Math-1.5B | 2 | R_h2 | ABL | — | C | h2 | -0.3260 | [-0.4796, -0.1724] | False |
| Qwen2.5-Math-1.5B | 2 | R_h2 | ABL | — | C | h3 | -0.2819 | [-0.4412, -0.1226] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3 | ABL | — | rho | h2 | 0.3400 | [0.2218, 0.4581] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3 | ABL | — | rho | h3 | 0.4958 | [0.4273, 0.5643] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3 | ABL | — | C | h2 | 0.9732 | [0.8686, 1.0777] | False |
| Qwen2.5-Math-1.5B | 2 | R_h3 | ABL | — | C | h3 | 0.9565 | [0.8461, 1.0670] | False |
| Qwen2.5-Math-1.5B | 2 | R_pooled | ABL | — | rho | h2 | -0.0773 | [-0.1843, 0.0297] | True |
| Qwen2.5-Math-1.5B | 2 | R_pooled | ABL | — | rho | h3 | 0.0912 | [0.0265, 0.1559] | False |
| Qwen2.5-Math-1.5B | 2 | R_pooled | ABL | — | C | h2 | 0.4721 | [0.3806, 0.5635] | False |
| Qwen2.5-Math-1.5B | 2 | R_pooled | ABL | — | C | h3 | 0.4725 | [0.3687, 0.5763] | False |
| Qwen2.5-Math-1.5B | 3 | R_base | ABL | — | rho | h2 | 0.0205 | [-0.1494, 0.1904] | True |
| Qwen2.5-Math-1.5B | 3 | R_base | ABL | — | rho | h3 | 0.0512 | [-0.1151, 0.2175] | True |
| Qwen2.5-Math-1.5B | 3 | R_base | ABL | — | C | h2 | 0.8453 | [0.7239, 0.9668] | False |
| Qwen2.5-Math-1.5B | 3 | R_base | ABL | — | C | h3 | 0.8509 | [0.7097, 0.9922] | False |
| Qwen2.5-Math-1.5B | 3 | R_h2 | ABL | — | rho | h2 | -0.5786 | [-0.8237, -0.3335] | False |
| Qwen2.5-Math-1.5B | 3 | R_h2 | ABL | — | rho | h3 | -0.4877 | [-0.6716, -0.3038] | False |
| Qwen2.5-Math-1.5B | 3 | R_h2 | ABL | — | C | h2 | -0.0240 | [-0.1354, 0.0873] | True |
| Qwen2.5-Math-1.5B | 3 | R_h2 | ABL | — | C | h3 | 0.0097 | [-0.0998, 0.1193] | True |
| Qwen2.5-Math-1.5B | 3 | R_h3 | ABL | — | rho | h2 | 0.6262 | [0.4878, 0.7647] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3 | ABL | — | rho | h3 | 0.8274 | [0.7415, 0.9132] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3 | ABL | — | C | h2 | 0.8804 | [0.7399, 1.0210] | False |
| Qwen2.5-Math-1.5B | 3 | R_h3 | ABL | — | C | h3 | 0.8898 | [0.7731, 1.0065] | False |
| Qwen2.5-Math-1.5B | 3 | R_pooled | ABL | — | rho | h2 | 0.0726 | [-0.0666, 0.2118] | True |
| Qwen2.5-Math-1.5B | 3 | R_pooled | ABL | — | rho | h3 | 0.2690 | [0.1689, 0.3690] | False |
| Qwen2.5-Math-1.5B | 3 | R_pooled | ABL | — | C | h2 | 0.6693 | [0.5319, 0.8068] | False |
| Qwen2.5-Math-1.5B | 3 | R_pooled | ABL | — | C | h3 | 0.6973 | [0.5768, 0.8179] | False |
| Qwen3-4B | 2 | R_base | ABL | — | rho | h2 | 5.5483 | [4.1223, 6.9744] | False |
| Qwen3-4B | 2 | R_base | ABL | — | rho | h3 | 5.4594 | [4.0624, 6.8564] | False |
| Qwen3-4B | 2 | R_base | ABL | — | C | h2 | 1.1632 | [0.1908, 2.1355] | False |
| Qwen3-4B | 2 | R_base | ABL | — | C | h3 | 1.2123 | [0.4634, 1.9612] | False |
| Qwen3-4B | 2 | R_h2 | ABL | — | rho | h2 | -3.4548 | [-4.2864, -2.6232] | False |
| Qwen3-4B | 2 | R_h2 | ABL | — | rho | h3 | -2.6188 | [-3.4994, -1.7382] | False |
| Qwen3-4B | 2 | R_h2 | ABL | — | C | h2 | -0.0679 | [-1.0324, 0.8966] | True |
| Qwen3-4B | 2 | R_h2 | ABL | — | C | h3 | 0.1753 | [-0.7321, 1.0826] | True |
| Qwen3-4B | 2 | R_h3 | ABL | — | rho | h2 | -2.1077 | [-3.0287, -1.1867] | False |
| Qwen3-4B | 2 | R_h3 | ABL | — | rho | h3 | -2.8601 | [-3.9848, -1.7354] | False |
| Qwen3-4B | 2 | R_h3 | ABL | — | C | h2 | -1.5929 | [-2.4570, -0.7287] | False |
| Qwen3-4B | 2 | R_h3 | ABL | — | C | h3 | -1.1710 | [-2.1459, -0.1960] | False |
| Qwen3-4B | 2 | R_pooled | ABL | — | rho | h2 | -1.4852 | [-2.3938, -0.5767] | False |
| Qwen3-4B | 2 | R_pooled | ABL | — | rho | h3 | -2.0215 | [-3.2953, -0.7478] | False |
| Qwen3-4B | 2 | R_pooled | ABL | — | C | h2 | -1.3051 | [-2.2161, -0.3940] | False |
| Qwen3-4B | 2 | R_pooled | ABL | — | C | h3 | -0.9856 | [-2.0191, 0.0478] | True |
| Qwen3-4B | 3 | R_base | ABL | — | rho | h2 | 7.4211 | [6.2342, 8.6080] | False |
| Qwen3-4B | 3 | R_base | ABL | — | rho | h3 | 7.9221 | [6.7854, 9.0588] | False |
| Qwen3-4B | 3 | R_base | ABL | — | C | h2 | 2.0937 | [0.9152, 3.2722] | False |
| Qwen3-4B | 3 | R_base | ABL | — | C | h3 | 2.4801 | [1.5135, 3.4466] | False |
| Qwen3-4B | 3 | R_h2 | ABL | — | rho | h2 | -1.9101 | [-2.7716, -1.0486] | False |
| Qwen3-4B | 3 | R_h2 | ABL | — | rho | h3 | -1.2202 | [-2.0763, -0.3641] | False |
| Qwen3-4B | 3 | R_h2 | ABL | — | C | h2 | 1.1461 | [0.2711, 2.0211] | False |
| Qwen3-4B | 3 | R_h2 | ABL | — | C | h3 | 1.3260 | [0.4963, 2.1558] | False |
| Qwen3-4B | 3 | R_h3 | ABL | — | rho | h2 | -3.3633 | [-4.8167, -1.9099] | False |
| Qwen3-4B | 3 | R_h3 | ABL | — | rho | h3 | -4.6952 | [-6.2516, -3.1389] | False |
| Qwen3-4B | 3 | R_h3 | ABL | — | C | h2 | -1.7075 | [-2.6547, -0.7603] | False |
| Qwen3-4B | 3 | R_h3 | ABL | — | C | h3 | -1.2412 | [-2.2007, -0.2817] | False |
| Qwen3-4B | 3 | R_pooled | ABL | — | rho | h2 | -2.5028 | [-3.7859, -1.2198] | False |
| Qwen3-4B | 3 | R_pooled | ABL | — | rho | h3 | -2.9241 | [-4.2645, -1.5836] | False |
| Qwen3-4B | 3 | R_pooled | ABL | — | C | h2 | -1.8715 | [-2.7492, -0.9939] | False |
| Qwen3-4B | 3 | R_pooled | ABL | — | C | h3 | -1.4832 | [-2.3960, -0.5703] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_base | ABL | — | rho | h2 | 2.8804 | [1.5402, 4.2205] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_base | ABL | — | rho | h3 | 2.6834 | [1.4732, 3.8935] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_base | ABL | — | C | h2 | 2.1066 | [0.0831, 4.1301] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_base | ABL | — | C | h3 | 2.8920 | [1.7008, 4.0832] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h2 | ABL | — | rho | h2 | -1.5754 | [-3.2577, 0.1069] | True |
| Qwen3-4B-Instruct-2507 | 2 | R_h2 | ABL | — | rho | h3 | -1.8954 | [-3.6691, -0.1217] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h2 | ABL | — | C | h2 | -4.7173 | [-6.8986, -2.5361] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h2 | ABL | — | C | h3 | -3.2246 | [-4.7733, -1.6759] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3 | ABL | — | rho | h2 | -1.9554 | [-4.6053, 0.6944] | True |
| Qwen3-4B-Instruct-2507 | 2 | R_h3 | ABL | — | rho | h3 | -2.6576 | [-4.6929, -0.6222] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3 | ABL | — | C | h2 | -5.0479 | [-6.8927, -3.2031] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_h3 | ABL | — | C | h3 | -4.1908 | [-5.5784, -2.8032] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_pooled | ABL | — | rho | h2 | -2.3501 | [-5.2289, 0.5287] | True |
| Qwen3-4B-Instruct-2507 | 2 | R_pooled | ABL | — | rho | h3 | -2.7421 | [-4.6903, -0.7939] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_pooled | ABL | — | C | h2 | -5.6056 | [-7.3477, -3.8635] | False |
| Qwen3-4B-Instruct-2507 | 2 | R_pooled | ABL | — | C | h3 | -5.1958 | [-6.5915, -3.8001] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_base | ABL | — | rho | h2 | 3.8302 | [2.5653, 5.0952] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_base | ABL | — | rho | h3 | 3.8062 | [2.5398, 5.0727] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_base | ABL | — | C | h2 | 1.6864 | [-0.3754, 3.7481] | True |
| Qwen3-4B-Instruct-2507 | 3 | R_base | ABL | — | C | h3 | 2.5317 | [1.3858, 3.6776] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h2 | ABL | — | rho | h2 | -4.3615 | [-5.4842, -3.2388] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h2 | ABL | — | rho | h3 | -4.3951 | [-6.1407, -2.6496] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h2 | ABL | — | C | h2 | -3.7721 | [-6.0093, -1.5349] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h2 | ABL | — | C | h3 | -2.0601 | [-3.6444, -0.4759] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3 | ABL | — | rho | h2 | -1.5459 | [-3.8872, 0.7955] | True |
| Qwen3-4B-Instruct-2507 | 3 | R_h3 | ABL | — | rho | h3 | -2.2479 | [-4.1346, -0.3613] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3 | ABL | — | C | h2 | -5.7980 | [-7.5412, -4.0547] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_h3 | ABL | — | C | h3 | -4.9833 | [-6.3267, -3.6400] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_pooled | ABL | — | rho | h2 | -1.6290 | [-3.8373, 0.5792] | True |
| Qwen3-4B-Instruct-2507 | 3 | R_pooled | ABL | — | rho | h3 | -1.9218 | [-3.7126, -0.1311] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_pooled | ABL | — | C | h2 | -6.4129 | [-8.1757, -4.6502] | False |
| Qwen3-4B-Instruct-2507 | 3 | R_pooled | ABL | — | C | h3 | -5.8457 | [-7.3649, -4.3265] | False |
| gemma-3-4b-it | 2 | R_base | ABL | — | rho | h2 | -4.4520 | [-7.1986, -1.7054] | False |
| gemma-3-4b-it | 2 | R_base | ABL | — | rho | h3 | -0.9281 | [-3.4291, 1.5729] | True |
| gemma-3-4b-it | 2 | R_base | ABL | — | C | h2 | 13.0277 | [9.6287, 16.4266] | False |
| gemma-3-4b-it | 2 | R_base | ABL | — | C | h3 | 15.4446 | [12.9042, 17.9850] | False |
| gemma-3-4b-it | 2 | R_h2 | ABL | — | rho | h2 | -1.3778 | [-5.1716, 2.4160] | True |
| gemma-3-4b-it | 2 | R_h2 | ABL | — | rho | h3 | 2.1883 | [-1.4513, 5.8278] | True |
| gemma-3-4b-it | 2 | R_h2 | ABL | — | C | h2 | 16.2687 | [14.4256, 18.1117] | False |
| gemma-3-4b-it | 2 | R_h2 | ABL | — | C | h3 | 18.6627 | [17.7764, 19.5490] | False |
| gemma-3-4b-it | 2 | R_h3 | ABL | — | rho | h2 | -4.1078 | [-6.9934, -1.2221] | False |
| gemma-3-4b-it | 2 | R_h3 | ABL | — | rho | h3 | -0.5247 | [-2.4150, 1.3657] | True |
| gemma-3-4b-it | 2 | R_h3 | ABL | — | C | h2 | 11.6978 | [10.2428, 13.1528] | False |
| gemma-3-4b-it | 2 | R_h3 | ABL | — | C | h3 | 14.1082 | [12.7702, 15.4461] | False |
| gemma-3-4b-it | 2 | R_pooled | ABL | — | rho | h2 | 1.8496 | [-1.6317, 5.3308] | True |
| gemma-3-4b-it | 2 | R_pooled | ABL | — | rho | h3 | 5.4373 | [3.2354, 7.6391] | False |
| gemma-3-4b-it | 2 | R_pooled | ABL | — | C | h2 | 6.5828 | [4.8765, 8.2891] | False |
| gemma-3-4b-it | 2 | R_pooled | ABL | — | C | h3 | 8.9847 | [7.1674, 10.8019] | False |
| gemma-3-4b-it | 3 | R_base | ABL | — | rho | h2 | -7.0579 | [-12.0316, -2.0841] | False |
| gemma-3-4b-it | 3 | R_base | ABL | — | rho | h3 | -3.5618 | [-8.1028, 0.9792] | True |
| gemma-3-4b-it | 3 | R_base | ABL | — | C | h2 | 6.6047 | [4.0044, 9.2050] | False |
| gemma-3-4b-it | 3 | R_base | ABL | — | C | h3 | 9.0341 | [7.2771, 10.7911] | False |
| gemma-3-4b-it | 3 | R_h2 | ABL | — | rho | h2 | -0.8483 | [-3.8384, 2.1418] | True |
| gemma-3-4b-it | 3 | R_h2 | ABL | — | rho | h3 | 2.6928 | [-0.2686, 5.6541] | True |
| gemma-3-4b-it | 3 | R_h2 | ABL | — | C | h2 | 15.2316 | [12.3106, 18.1525] | False |
| gemma-3-4b-it | 3 | R_h2 | ABL | — | C | h3 | 17.6222 | [15.2940, 19.9503] | False |
| gemma-3-4b-it | 3 | R_h3 | ABL | — | rho | h2 | 0.2493 | [-2.7295, 3.2281] | True |
| gemma-3-4b-it | 3 | R_h3 | ABL | — | rho | h3 | 3.8417 | [2.1125, 5.5709] | False |
| gemma-3-4b-it | 3 | R_h3 | ABL | — | C | h2 | 10.6003 | [9.0903, 12.1102] | False |
| gemma-3-4b-it | 3 | R_h3 | ABL | — | C | h3 | 13.0138 | [11.3658, 14.6619] | False |
| gemma-3-4b-it | 3 | R_pooled | ABL | — | rho | h2 | 3.9824 | [0.7458, 7.2190] | False |
| gemma-3-4b-it | 3 | R_pooled | ABL | — | rho | h3 | 7.5883 | [5.5611, 9.6154] | False |
| gemma-3-4b-it | 3 | R_pooled | ABL | — | C | h2 | 6.2855 | [4.5712, 7.9998] | False |
| gemma-3-4b-it | 3 | R_pooled | ABL | — | C | h3 | 8.7060 | [6.8762, 10.5359] | False |
| Phi-4-mini-instruct | 2 | R_base | ABL | — | rho | h2 | 0.6750 | [-0.0686, 1.4185] | True |
| Phi-4-mini-instruct | 2 | R_base | ABL | — | rho | h3 | 0.3629 | [-0.4583, 1.1841] | True |
| Phi-4-mini-instruct | 2 | R_base | ABL | — | C | h2 | 1.6148 | [1.1037, 2.1260] | False |
| Phi-4-mini-instruct | 2 | R_base | ABL | — | C | h3 | 1.6973 | [1.2353, 2.1593] | False |
| Phi-4-mini-instruct | 2 | R_h2 | ABL | — | rho | h2 | -0.3568 | [-1.0217, 0.3081] | True |
| Phi-4-mini-instruct | 2 | R_h2 | ABL | — | rho | h3 | -0.8520 | [-1.4929, -0.2110] | False |
| Phi-4-mini-instruct | 2 | R_h2 | ABL | — | C | h2 | 0.2079 | [-0.3303, 0.7461] | True |
| Phi-4-mini-instruct | 2 | R_h2 | ABL | — | C | h3 | 0.6163 | [0.0977, 1.1350] | False |
| Phi-4-mini-instruct | 2 | R_h3 | ABL | — | rho | h2 | 2.5483 | [1.4564, 3.6401] | False |
| Phi-4-mini-instruct | 2 | R_h3 | ABL | — | rho | h3 | 3.5491 | [2.7488, 4.3494] | False |
| Phi-4-mini-instruct | 2 | R_h3 | ABL | — | C | h2 | 2.3904 | [1.9314, 2.8493] | False |
| Phi-4-mini-instruct | 2 | R_h3 | ABL | — | C | h3 | 2.9985 | [2.5172, 3.4799] | False |
| Phi-4-mini-instruct | 2 | R_pooled | ABL | — | rho | h2 | 1.5646 | [0.7038, 2.4254] | False |
| Phi-4-mini-instruct | 2 | R_pooled | ABL | — | rho | h3 | 2.3402 | [1.5213, 3.1590] | False |
| Phi-4-mini-instruct | 2 | R_pooled | ABL | — | C | h2 | 1.8357 | [1.3566, 2.3147] | False |
| Phi-4-mini-instruct | 2 | R_pooled | ABL | — | C | h3 | 2.4273 | [2.0360, 2.8186] | False |
| Phi-4-mini-instruct | 3 | R_base | ABL | — | rho | h2 | 0.2111 | [-0.5103, 0.9325] | True |
| Phi-4-mini-instruct | 3 | R_base | ABL | — | rho | h3 | -0.1789 | [-0.9320, 0.5742] | True |
| Phi-4-mini-instruct | 3 | R_base | ABL | — | C | h2 | 1.7131 | [1.1274, 2.2989] | False |
| Phi-4-mini-instruct | 3 | R_base | ABL | — | C | h3 | 2.0108 | [1.5176, 2.5041] | False |
| Phi-4-mini-instruct | 3 | R_h2 | ABL | — | rho | h2 | -0.3733 | [-0.9514, 0.2048] | True |
| Phi-4-mini-instruct | 3 | R_h2 | ABL | — | rho | h3 | -0.9022 | [-1.4907, -0.3138] | False |
| Phi-4-mini-instruct | 3 | R_h2 | ABL | — | C | h2 | 0.2519 | [-0.1534, 0.6571] | True |
| Phi-4-mini-instruct | 3 | R_h2 | ABL | — | C | h3 | 0.5820 | [0.1453, 1.0187] | False |
| Phi-4-mini-instruct | 3 | R_h3 | ABL | — | rho | h2 | 2.7081 | [1.7062, 3.7100] | False |
| Phi-4-mini-instruct | 3 | R_h3 | ABL | — | rho | h3 | 3.6803 | [2.8780, 4.4825] | False |
| Phi-4-mini-instruct | 3 | R_h3 | ABL | — | C | h2 | 2.6465 | [2.2495, 3.0435] | False |
| Phi-4-mini-instruct | 3 | R_h3 | ABL | — | C | h3 | 3.2953 | [2.8323, 3.7584] | False |
| Phi-4-mini-instruct | 3 | R_pooled | ABL | — | rho | h2 | 2.0230 | [1.2257, 2.8203] | False |
| Phi-4-mini-instruct | 3 | R_pooled | ABL | — | rho | h3 | 2.5900 | [1.7344, 3.4455] | False |
| Phi-4-mini-instruct | 3 | R_pooled | ABL | — | C | h2 | 2.3563 | [1.8754, 2.8372] | False |
| Phi-4-mini-instruct | 3 | R_pooled | ABL | — | C | h3 | 2.9350 | [2.5527, 3.3172] | False |

## P/C/PC ablation manipulation checks

Positive attenuation means ablation reduced the encoded norming contrast.

| Model | Task | direction | attenuation | 95% CI | Holm p |
|---|---:|---|---:|---:|---:|
| Qwen2.5-Math-1.5B | 2 | C | 0.0450 | [-0.0070, 0.0971] | 0.1752 |
| Qwen2.5-Math-1.5B | 2 | P | 0.0285 | [-0.0630, 0.1200] | 0.5148 |
| Qwen2.5-Math-1.5B | 2 | PC | 0.0668 | [-0.2334, 0.3669] | 1.0000 |
| Qwen2.5-Math-1.5B | 3 | C | 0.0523 | [-0.0170, 0.1216] | 0.1752 |
| Qwen2.5-Math-1.5B | 3 | P | 0.0547 | [-0.0223, 0.1317] | 0.3018 |
| Qwen2.5-Math-1.5B | 3 | PC | 0.0439 | [-0.3367, 0.4245] | 1.0000 |
| Qwen3-4B | 2 | C | 0.9945 | [0.5485, 1.4405] | 0.0009 |
| Qwen3-4B | 2 | P | -1.4841 | [-2.5679, -0.4002] | 0.0188 |
| Qwen3-4B | 2 | PC | 0.6443 | [0.3439, 0.9447] | 0.0009 |
| Qwen3-4B | 3 | C | 0.7741 | [0.2867, 1.2614] | 0.0061 |
| Qwen3-4B | 3 | P | -1.3375 | [-2.3459, -0.3292] | 0.0188 |
| Qwen3-4B | 3 | PC | 0.6239 | [0.3687, 0.8792] | 0.0006 |
| Qwen3-4B-Instruct-2507 | 2 | C | 1.0557 | [0.4487, 1.6628] | 0.0063 |
| Qwen3-4B-Instruct-2507 | 2 | P | 0.5188 | [-1.0018, 2.0395] | 0.4905 |
| Qwen3-4B-Instruct-2507 | 2 | PC | 0.6954 | [0.3280, 1.0628] | 0.0018 |
| Qwen3-4B-Instruct-2507 | 3 | C | 1.0022 | [0.3637, 1.6407] | 0.0065 |
| Qwen3-4B-Instruct-2507 | 3 | P | 1.0557 | [-0.2849, 2.3964] | 0.2303 |
| Qwen3-4B-Instruct-2507 | 3 | PC | 0.6458 | [0.3156, 0.9761] | 0.0018 |
| gemma-3-4b-it | 2 | C | 3.2130 | [1.8437, 4.5823] | 0.0001 |
| gemma-3-4b-it | 2 | P | 4.2399 | [1.9957, 6.4841] | 0.0001 |
| gemma-3-4b-it | 2 | PC | 2.2387 | [1.2424, 3.2350] | 0.0001 |
| gemma-3-4b-it | 3 | C | 3.2155 | [1.8455, 4.5855] | 0.0001 |
| gemma-3-4b-it | 3 | P | 4.2633 | [2.0301, 6.4965] | 0.0001 |
| gemma-3-4b-it | 3 | PC | 2.2378 | [1.2414, 3.2342] | 0.0001 |
| Phi-4-mini-instruct | 2 | C | 0.7101 | [0.2602, 1.1599] | 0.0035 |
| Phi-4-mini-instruct | 2 | P | 0.3406 | [-0.1732, 0.8544] | 0.3975 |
| Phi-4-mini-instruct | 2 | PC | 1.7426 | [0.8570, 2.6283] | 0.0003 |
| Phi-4-mini-instruct | 3 | C | 0.8291 | [0.4159, 1.2422] | 0.0021 |
| Phi-4-mini-instruct | 3 | P | 0.2809 | [-0.4544, 1.0162] | 0.4999 |
| Phi-4-mini-instruct | 3 | PC | 1.7524 | [0.8916, 2.6132] | 0.0002 |
