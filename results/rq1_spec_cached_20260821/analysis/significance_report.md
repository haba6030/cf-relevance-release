# Frozen-spec cached-model RQ1/H2 report

Models: 8; items: 15.
Decision test: exact within-item sign flip, two-sided. Positive DoD is predicted.
Holm correction is across the preregistered 2-way/3-way cells within a model, not across models.

## H1 log-probability cells

| model                         |   task | arm   |   mean |   ci_lo |   ci_hi |   p_flip_two |   holm_task_p_flip_two |   positive |   negative |   loo_worst_p_flip_two |   raw_p_flip_two |
|:------------------------------|-------:|:------|-------:|--------:|--------:|-------------:|-----------------------:|-----------:|-----------:|-----------------------:|-----------------:|
| Qwen2.5-Math-1.5B             |      2 | bare  | 0.8463 |  0.4745 |  1.218  |       0.0006 |                 0.0012 |         14 |          1 |                 0.0012 |           0.0004 |
| Qwen2.5-Math-1.5B             |      2 | conj  | 0.2272 |  0.0564 |  0.3979 |       0.0076 |                 0.0153 |         12 |          3 |                 0.0153 |           0.0056 |
| Qwen2.5-Math-1.5B             |      3 | bare  | 0.7824 |  0.3975 |  1.1672 |       0.0009 |                 0.0012 |         14 |          1 |                 0.0017 |           0.0004 |
| Qwen2.5-Math-1.5B             |      3 | conj  | 0.1936 |  0.0458 |  0.3414 |       0.0077 |                 0.0153 |         12 |          3 |                 0.0154 |           0.0059 |
| Qwen3-4B                      |      2 | bare  | 1.1995 |  0.6595 |  1.7395 |       0.0004 |                 0.0007 |         13 |          2 |                 0.0007 |           0.0004 |
| Qwen3-4B                      |      2 | conj  | 0.2809 |  0.0732 |  0.4886 |       0.0123 |                 0.0157 |         13 |          2 |                 0.0247 |           0.0163 |
| Qwen3-4B                      |      3 | bare  | 1.0972 |  0.5741 |  1.6202 |       0.0008 |                 0.0008 |         13 |          2 |                 0.0016 |           0.0007 |
| Qwen3-4B                      |      3 | conj  | 0.2934 |  0.0908 |  0.4959 |       0.0079 |                 0.0157 |         13 |          2 |                 0.0157 |           0.0095 |
| Qwen3-4B-Instruct-2507        |      2 | bare  | 1.1653 |  0.5905 |  1.7402 |       0.0007 |                 0.0015 |         14 |          1 |                 0.0015 |           0.0004 |
| Qwen3-4B-Instruct-2507        |      2 | conj  | 0.2577 |  0.0648 |  0.4507 |       0.013  |                 0.0195 |         12 |          3 |                 0.026  |           0.0097 |
| Qwen3-4B-Instruct-2507        |      3 | bare  | 1.1336 |  0.5636 |  1.7035 |       0.001  |                 0.0015 |         13 |          2 |                 0.002  |           0.0005 |
| Qwen3-4B-Instruct-2507        |      3 | conj  | 0.252  |  0.0743 |  0.4298 |       0.0098 |                 0.0195 |         12 |          3 |                 0.0195 |           0.0067 |
| Qwen3-4B-Thinking-2507        |      2 | bare  | 0.7633 |  0.4602 |  1.0664 |       0.0001 |                 0.0002 |         14 |          1 |                 0.0002 |           0.0001 |
| Qwen3-4B-Thinking-2507        |      2 | conj  | 0.1358 |  0.008  |  0.2636 |       0.0413 |                 0.068  |         11 |          4 |                 0.0809 |           0.0314 |
| Qwen3-4B-Thinking-2507        |      3 | bare  | 0.7716 |  0.4316 |  1.1117 |       0.0002 |                 0.0002 |         14 |          1 |                 0.0005 |           0.0002 |
| Qwen3-4B-Thinking-2507        |      3 | conj  | 0.1449 |  0.014  |  0.2757 |       0.034  |                 0.068  |         11 |          4 |                 0.0674 |           0.0244 |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | bare  | 0.5534 |  0.1506 |  0.9562 |       0.0056 |                 0.0111 |         12 |          3 |                 0.0112 |           0.002  |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | conj  | 0.1977 |  0.0221 |  0.3733 |       0.022  |                 0.0439 |         12 |          3 |                 0.0439 |           0.0154 |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | bare  | 0.4977 |  0.1509 |  0.8445 |       0.0056 |                 0.0111 |         12 |          3 |                 0.0111 |           0.0032 |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | conj  | 0.174  |  0.0187 |  0.3293 |       0.027  |                 0.0439 |         11 |          4 |                 0.0541 |           0.0215 |
| gemma-3-4b-it                 |      2 | bare  | 1.2173 |  0.6209 |  1.8136 |       0.001  |                 0.0017 |         13 |          2 |                 0.0021 |           0.0009 |
| gemma-3-4b-it                 |      2 | conj  | 0.3451 |  0.1193 |  0.571  |       0.0043 |                 0.0087 |         11 |          4 |                 0.0087 |           0.0049 |
| gemma-3-4b-it                 |      3 | bare  | 1.2098 |  0.576  |  1.8436 |       0.0009 |                 0.0017 |         14 |          1 |                 0.0017 |           0.0005 |
| gemma-3-4b-it                 |      3 | conj  | 0.2854 |  0.1015 |  0.4692 |       0.0048 |                 0.0087 |         11 |          4 |                 0.0096 |           0.004  |
| Phi-4-mini-instruct           |      2 | bare  | 0.9993 |  0.5735 |  1.425  |       0.0007 |                 0.0011 |         14 |          1 |                 0.0015 |           0.0005 |
| Phi-4-mini-instruct           |      2 | conj  | 0.2931 |  0.1654 |  0.4208 |       0.0005 |                 0.0009 |         12 |          3 |                 0.001  |           0.0005 |
| Phi-4-mini-instruct           |      3 | bare  | 1.0939 |  0.6424 |  1.5454 |       0.0005 |                 0.0011 |         14 |          1 |                 0.0011 |           0.0004 |
| Phi-4-mini-instruct           |      3 | conj  | 0.309  |  0.1803 |  0.4378 |       0.0004 |                 0.0009 |         13 |          2 |                 0.0009 |           0.0004 |
| Phi-4-mini-reasoning          |      2 | bare  | 1.6108 |  0.8121 |  2.4096 |       0.0007 |                 0.0015 |         14 |          1 |                 0.0015 |           0.0005 |
| Phi-4-mini-reasoning          |      2 | conj  | 0.4741 |  0.126  |  0.8223 |       0.009  |                 0.009  |         12 |          3 |                 0.0181 |           0.0029 |
| Phi-4-mini-reasoning          |      3 | bare  | 1.5113 |  0.6207 |  2.402  |       0.0032 |                 0.0032 |         12 |          3 |                 0.0063 |           0.0021 |
| Phi-4-mini-reasoning          |      3 | conj  | 0.538  |  0.2287 |  0.8473 |       0.0032 |                 0.0063 |         13 |          2 |                 0.0063 |           0.0022 |

## H1a task contrast

| model                         | arm   |    mean |   ci_lo |   ci_hi |   p_flip_two |
|:------------------------------|:------|--------:|--------:|--------:|-------------:|
| Qwen2.5-Math-1.5B             | conj  | -0.0336 | -0.081  |  0.0139 |       0.1277 |
| Qwen3-4B                      | conj  |  0.0125 | -0.0246 |  0.0496 |       0.4787 |
| Qwen3-4B-Instruct-2507        | conj  | -0.0057 | -0.0449 |  0.0336 |       0.7571 |
| Qwen3-4B-Thinking-2507        | conj  |  0.009  | -0.0188 |  0.0369 |       0.5026 |
| DeepSeek-R1-Distill-Qwen-1.5B | conj  | -0.0237 | -0.0786 |  0.0312 |       0.3753 |
| gemma-3-4b-it                 | conj  | -0.0598 | -0.1295 |  0.0099 |       0.0886 |
| Phi-4-mini-instruct           | conj  |  0.0159 | -0.0063 |  0.0382 |       0.1457 |
| Phi-4-mini-reasoning          | conj  |  0.0638 | -0.1704 |  0.298  |       0.5631 |

## H1b conjunction-minus-bare contrast

| model                         |   task |    mean |   ci_lo |   ci_hi |   p_flip_two |
|:------------------------------|-------:|--------:|--------:|--------:|-------------:|
| Qwen2.5-Math-1.5B             |      2 | -0.6191 | -0.9037 | -0.3345 |       0.0007 |
| Qwen2.5-Math-1.5B             |      3 | -0.5888 | -0.8883 | -0.2893 |       0.0008 |
| Qwen3-4B                      |      2 | -0.9186 | -1.4104 | -0.4267 |       0.0004 |
| Qwen3-4B                      |      3 | -0.8038 | -1.2738 | -0.3338 |       0.0013 |
| Qwen3-4B-Instruct-2507        |      2 | -0.9076 | -1.4014 | -0.4139 |       0.0002 |
| Qwen3-4B-Instruct-2507        |      3 | -0.8815 | -1.376  | -0.3871 |       0.0007 |
| Qwen3-4B-Thinking-2507        |      2 | -0.6275 | -0.8896 | -0.3654 |       0.0001 |
| Qwen3-4B-Thinking-2507        |      3 | -0.6268 | -0.9254 | -0.3282 |       0.0001 |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | -0.3557 | -0.6517 | -0.0597 |       0.0219 |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | -0.3236 | -0.6101 | -0.0372 |       0.0298 |
| gemma-3-4b-it                 |      2 | -0.8721 | -1.4156 | -0.3287 |       0.002  |
| gemma-3-4b-it                 |      3 | -0.9244 | -1.4917 | -0.3571 |       0.0015 |
| Phi-4-mini-instruct           |      2 | -0.7062 | -1.0702 | -0.3422 |       0.0012 |
| Phi-4-mini-instruct           |      3 | -0.7849 | -1.1768 | -0.393  |       0.001  |
| Phi-4-mini-reasoning          |      2 | -1.1367 | -1.9939 | -0.2795 |       0.0129 |
| Phi-4-mini-reasoning          |      3 | -0.9734 | -1.7683 | -0.1785 |       0.0228 |

## Forced-choice cells

| model                         | metric                    |   task | arm   |    mean | ci_lo   | ci_hi   |   p_flip_two |   holm_task_p_flip_two |   positive |   negative |   ties |
|:------------------------------|:--------------------------|-------:|:------|--------:|:--------|:--------|-------------:|-----------------------:|-----------:|-----------:|-------:|
| Qwen2.5-Math-1.5B             | forcedchoice_letter       |      2 | bare  |  0.1111 | -0.0213 | 0.2436  |       0.1133 |                 0.1133 |         11 |          4 |      0 |
| Qwen2.5-Math-1.5B             | forcedchoice_letter       |      2 | conj  |  0.0222 | -0.0314 | 0.0759  |       0.5312 |                 0.5312 |          5 |          4 |      6 |
| Qwen2.5-Math-1.5B             | forcedchoice_letter       |      3 | bare  |  0.1222 | 0.0238  | 0.2207  |       0.0225 |                 0.045  |         11 |          4 |      0 |
| Qwen2.5-Math-1.5B             | forcedchoice_letter       |      3 | conj  |  0.037  | -0.0139 | 0.0879  |       0.1611 |                 0.3223 |          9 |          6 |      0 |
| Qwen3-4B                      | forcedchoice_letter       |      2 | bare  |  0.1778 | 0.0584  | 0.2972  |       0.0117 |                 0.0117 |          9 |          2 |      4 |
| Qwen3-4B                      | forcedchoice_letter       |      2 | conj  |  0.2111 | 0.0750  | 0.3472  |       0.0017 |                 0.0034 |         12 |          1 |      2 |
| Qwen3-4B                      | forcedchoice_letter       |      3 | bare  |  0.2296 | 0.1110  | 0.3483  |       0.002  |                 0.004  |         14 |          1 |      0 |
| Qwen3-4B                      | forcedchoice_letter       |      3 | conj  |  0.1315 | 0.0277  | 0.2352  |       0.0207 |                 0.0207 |         10 |          5 |      0 |
| Qwen3-4B-Instruct-2507        | forcedchoice_letter       |      2 | bare  |  0.1833 | 0.0131  | 0.3535  |       0.0454 |                 0.0454 |         10 |          2 |      3 |
| Qwen3-4B-Instruct-2507        | forcedchoice_letter       |      2 | conj  |  0.0389 | -0.1016 | 0.1794  |       0.6161 |                 0.6161 |          7 |          7 |      1 |
| Qwen3-4B-Instruct-2507        | forcedchoice_letter       |      3 | bare  |  0.2556 | 0.1026  | 0.4085  |       0.0025 |                 0.005  |         13 |          2 |      0 |
| Qwen3-4B-Instruct-2507        | forcedchoice_letter       |      3 | conj  |  0.0778 | -0.0073 | 0.1629  |       0.0691 |                 0.1382 |         10 |          4 |      1 |
| Qwen3-4B-Thinking-2507        | forcedchoice_letter       |      2 | bare  |  0.3167 | 0.1883  | 0.4451  |       0.0005 |                 0.0005 |         12 |          1 |      2 |
| Qwen3-4B-Thinking-2507        | forcedchoice_letter       |      2 | conj  |  0.1778 | 0.0124  | 0.3431  |       0.0317 |                 0.0317 |         10 |          2 |      3 |
| Qwen3-4B-Thinking-2507        | forcedchoice_letter       |      3 | bare  |  0.3778 | 0.2567  | 0.4989  |       0.0001 |                 0.0001 |         15 |          0 |      0 |
| Qwen3-4B-Thinking-2507        | forcedchoice_letter       |      3 | conj  |  0.1907 | 0.0902  | 0.2913  |       0.0005 |                 0.001  |         12 |          2 |      1 |
| DeepSeek-R1-Distill-Qwen-1.5B | forcedchoice_letter       |      2 | bare  |  0.0556 | -0.0378 | 0.1489  |       0.2715 |                 0.2715 |          7 |          4 |      4 |
| DeepSeek-R1-Distill-Qwen-1.5B | forcedchoice_letter       |      2 | conj  |  0.0889 | 0.0225  | 0.1552  |       0.0229 |                 0.0459 |         10 |          2 |      3 |
| DeepSeek-R1-Distill-Qwen-1.5B | forcedchoice_letter       |      3 | bare  |  0.0704 | 0.0078  | 0.1330  |       0.0337 |                 0.0674 |          9 |          3 |      3 |
| DeepSeek-R1-Distill-Qwen-1.5B | forcedchoice_letter       |      3 | conj  |  0.0333 | -0.0219 | 0.0886  |       0.2456 |                 0.2456 |          9 |          5 |      1 |
| gemma-3-4b-it                 | forcedchoice_letter       |      2 | bare  |  0.1278 | -0.0320 | 0.2875  |       0.1259 |                 0.1259 |          9 |          5 |      1 |
| gemma-3-4b-it                 | forcedchoice_letter       |      2 | conj  |  0.05   | -0.1495 | 0.2495  |       0.7023 |                 0.7023 |          6 |          8 |      1 |
| gemma-3-4b-it                 | forcedchoice_letter       |      3 | bare  |  0.2593 | 0.1365  | 0.3820  |       0.0013 |                 0.0027 |         12 |          3 |      0 |
| gemma-3-4b-it                 | forcedchoice_letter       |      3 | conj  |  0.1648 | 0.0171  | 0.3125  |       0.033  |                 0.0659 |          8 |          7 |      0 |
| Phi-4-mini-instruct           | forcedchoice_letter       |      2 | bare  |  0.1722 | 0.0119  | 0.3325  |       0.0625 |                 0.0625 |          5 |          1 |      9 |
| Phi-4-mini-instruct           | forcedchoice_letter       |      2 | conj  |  0.0056 | -0.0421 | 0.0532  |       1      |                 1      |          4 |          4 |      7 |
| Phi-4-mini-instruct           | forcedchoice_letter       |      3 | bare  |  0.2574 | 0.1234  | 0.3914  |       0.0005 |                 0.001  |         12 |          1 |      2 |
| Phi-4-mini-instruct           | forcedchoice_letter       |      3 | conj  |  0.0907 | 0.0275  | 0.1540  |       0.0044 |                 0.0088 |         11 |          1 |      3 |
| Phi-4-mini-reasoning          | forcedchoice_letter       |      2 | bare  |  0.1111 | 0.0541  | 0.1681  |       0.002  |                 0.0039 |         12 |          1 |      2 |
| Phi-4-mini-reasoning          | forcedchoice_letter       |      2 | conj  |  0.0111 | -0.1121 | 0.1343  |       0.9292 |                 0.9292 |          7 |          6 |      2 |
| Phi-4-mini-reasoning          | forcedchoice_letter       |      3 | bare  |  0.1352 | 0.0127  | 0.2577  |       0.0277 |                 0.0277 |         10 |          4 |      1 |
| Phi-4-mini-reasoning          | forcedchoice_letter       |      3 | conj  |  0.0463 | -0.0452 | 0.1378  |       0.3239 |                 0.6477 |          8 |          6 |      1 |
| Qwen2.5-Math-1.5B             | forcedchoice_content_norm |      2 | bare  |  0.1833 | -0.0225 | 0.3891  |       0.1016 |                 0.2031 |          7 |          2 |      6 |
| Qwen2.5-Math-1.5B             | forcedchoice_content_norm |      2 | conj  |  0      |         |         |       1      |                 1      |          0 |          0 |     15 |
| Qwen2.5-Math-1.5B             | forcedchoice_content_norm |      3 | bare  |  0.063  | -0.0962 | 0.2221  |       0.4492 |                 0.4492 |          6 |          3 |      6 |
| Qwen2.5-Math-1.5B             | forcedchoice_content_norm |      3 | conj  |  0.0037 | -0.0435 | 0.0509  |       0.8125 |                 1      |          4 |          2 |      9 |
| Qwen3-4B                      | forcedchoice_content_norm |      2 | bare  |  0.1278 | 0.0309  | 0.2247  |       0.0039 |                 0.0078 |         10 |          0 |      5 |
| Qwen3-4B                      | forcedchoice_content_norm |      2 | conj  |  0      |         |         |       1      |                 1      |          0 |          0 |     15 |
| Qwen3-4B                      | forcedchoice_content_norm |      3 | bare  |  0.1648 | -0.0267 | 0.3564  |       0.0703 |                 0.0703 |          6 |          4 |      5 |
| Qwen3-4B                      | forcedchoice_content_norm |      3 | conj  | -0.0019 | -0.0154 | 0.0117  |       1      |                 1      |          2 |          1 |     12 |
| Qwen3-4B-Instruct-2507        | forcedchoice_content_norm |      2 | bare  |  0.0889 | -0.0953 | 0.2730  |       0.3564 |                 0.6875 |         10 |          2 |      3 |
| Qwen3-4B-Instruct-2507        | forcedchoice_content_norm |      2 | conj  |  0.0056 | -0.0064 | 0.0175  |       1      |                 1      |          1 |          0 |     14 |
| Qwen3-4B-Instruct-2507        | forcedchoice_content_norm |      3 | bare  |  0.0796 | -0.0725 | 0.2318  |       0.3438 |                 0.6875 |          5 |          5 |      5 |
| Qwen3-4B-Instruct-2507        | forcedchoice_content_norm |      3 | conj  |  0.0074 | -0.0095 | 0.0243  |       0.5    |                 1      |          3 |          1 |     11 |
| Qwen3-4B-Thinking-2507        | forcedchoice_content_norm |      2 | bare  |  0.1    | -0.1374 | 0.3374  |       0.4111 |                 0.4111 |          7 |          4 |      4 |
| Qwen3-4B-Thinking-2507        | forcedchoice_content_norm |      2 | conj  |  0.0056 | -0.0064 | 0.0175  |       1      |                 1      |          1 |          0 |     14 |
| Qwen3-4B-Thinking-2507        | forcedchoice_content_norm |      3 | bare  |  0.2222 | 0.0197  | 0.4247  |       0.0254 |                 0.0508 |          9 |          2 |      4 |
| Qwen3-4B-Thinking-2507        | forcedchoice_content_norm |      3 | conj  | -0.0148 | -0.0461 | 0.0164  |       0.5    |                 1      |          1 |          2 |     12 |
| DeepSeek-R1-Distill-Qwen-1.5B | forcedchoice_content_norm |      2 | bare  |  0.0833 | -0.0400 | 0.2067  |       0.2109 |                 0.2148 |          8 |          2 |      5 |
| DeepSeek-R1-Distill-Qwen-1.5B | forcedchoice_content_norm |      2 | conj  |  0      |         |         |       1      |                 1      |          0 |          0 |     15 |
| DeepSeek-R1-Distill-Qwen-1.5B | forcedchoice_content_norm |      3 | bare  |  0.1185 | -0.0347 | 0.2717  |       0.1074 |                 0.2148 |          6 |          4 |      5 |
| DeepSeek-R1-Distill-Qwen-1.5B | forcedchoice_content_norm |      3 | conj  |  0      |         |         |       1      |                 1      |          0 |          0 |     15 |
| gemma-3-4b-it                 | forcedchoice_content_norm |      2 | bare  |  0.1778 | -0.0779 | 0.4335  |       0.1743 |                 0.1743 |          8 |          4 |      3 |
| gemma-3-4b-it                 | forcedchoice_content_norm |      2 | conj  | -0.0333 | -0.1069 | 0.0403  |       0.5    |                 0.5295 |          3 |          3 |      9 |
| gemma-3-4b-it                 | forcedchoice_content_norm |      3 | bare  |  0.2704 | 0.1024  | 0.4383  |       0.0052 |                 0.0105 |         11 |          3 |      1 |
| gemma-3-4b-it                 | forcedchoice_content_norm |      3 | conj  |  0.0963 | -0.0763 | 0.2689  |       0.2648 |                 0.5295 |          9 |          5 |      1 |
| Phi-4-mini-instruct           | forcedchoice_content_norm |      2 | bare  |  0.2778 | 0.0609  | 0.4947  |       0.0156 |                 0.0156 |          7 |          0 |      8 |
| Phi-4-mini-instruct           | forcedchoice_content_norm |      2 | conj  |  0.0056 | -0.0218 | 0.0330  |       1      |                 1      |          1 |          2 |     12 |
| Phi-4-mini-instruct           | forcedchoice_content_norm |      3 | bare  |  0.3407 | 0.2021  | 0.4794  |       0.0002 |                 0.0005 |         13 |          0 |      2 |
| Phi-4-mini-instruct           | forcedchoice_content_norm |      3 | conj  |  0.0333 | -0.0291 | 0.0957  |       0.3594 |                 0.7188 |          6 |          3 |      6 |
| Phi-4-mini-reasoning          | forcedchoice_content_norm |      2 | bare  |  0.0333 | -0.1027 | 0.1693  |       0.667  |                 0.667  |          8 |          4 |      3 |
| Phi-4-mini-reasoning          | forcedchoice_content_norm |      2 | conj  |  0.0667 | 0.0110  | 0.1224  |       0.041  |                 0.082  |         10 |          2 |      3 |
| Phi-4-mini-reasoning          | forcedchoice_content_norm |      3 | bare  |  0.1796 | 0.0650  | 0.2943  |       0.0037 |                 0.0073 |         11 |          2 |      2 |
| Phi-4-mini-reasoning          | forcedchoice_content_norm |      3 | conj  |  0.0222 | -0.0254 | 0.0699  |       0.3789 |                 0.3789 |          5 |          7 |      3 |

## Position-contamination gate

| model                         |   task | arm   | readout      | hypothesis   |   position_only_accuracy |   position_only_predictability | position_direction   | contaminated   |
|:------------------------------|-------:|:------|:-------------|:-------------|-------------------------:|-------------------------------:|:---------------------|:---------------|
| Qwen2.5-Math-1.5B             |      2 | bare  | letter       | h2           |                   0.6194 |                         0.6194 | direct               | True           |
| Qwen2.5-Math-1.5B             |      2 | conj  | letter       | h2           |                   0.9222 |                         0.9222 | direct               | True           |
| Qwen2.5-Math-1.5B             |      3 | bare  | letter       | h2           |                   0.5741 |                         0.5741 | direct               | False          |
| Qwen2.5-Math-1.5B             |      3 | conj  | letter       | h2           |                   0.7287 |                         0.7287 | direct               | True           |
| Qwen3-4B                      |      2 | bare  | letter       | h2           |                   0.3583 |                         0.6417 | reverse              | True           |
| Qwen3-4B                      |      2 | conj  | letter       | h2           |                   0.1806 |                         0.8194 | reverse              | True           |
| Qwen3-4B                      |      3 | bare  | letter       | h2           |                   0.4296 |                         0.5704 | reverse              | False          |
| Qwen3-4B                      |      3 | conj  | letter       | h2           |                   0.6176 |                         0.6176 | direct               | True           |
| Qwen3-4B-Instruct-2507        |      2 | bare  | letter       | h2           |                   0.4611 |                         0.5389 | reverse              | False          |
| Qwen3-4B-Instruct-2507        |      2 | conj  | letter       | h2           |                   0.4556 |                         0.5444 | reverse              | False          |
| Qwen3-4B-Instruct-2507        |      3 | bare  | letter       | h2           |                   0.3926 |                         0.6074 | reverse              | True           |
| Qwen3-4B-Instruct-2507        |      3 | conj  | letter       | h2           |                   0.4694 |                         0.5306 | reverse              | False          |
| Qwen3-4B-Thinking-2507        |      2 | bare  | letter       | h2           |                   0.3083 |                         0.6917 | reverse              | True           |
| Qwen3-4B-Thinking-2507        |      2 | conj  | letter       | h2           |                   0.175  |                         0.825  | reverse              | True           |
| Qwen3-4B-Thinking-2507        |      3 | bare  | letter       | h2           |                   0.3796 |                         0.6204 | reverse              | True           |
| Qwen3-4B-Thinking-2507        |      3 | conj  | letter       | h2           |                   0.562  |                         0.562  | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | bare  | letter       | h2           |                   0.6444 |                         0.6444 | direct               | True           |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | conj  | letter       | h2           |                   0.8306 |                         0.8306 | direct               | True           |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | bare  | letter       | h2           |                   0.2824 |                         0.7176 | reverse              | True           |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | conj  | letter       | h2           |                   0.6111 |                         0.6111 | direct               | True           |
| gemma-3-4b-it                 |      2 | bare  | letter       | h2           |                   0.4444 |                         0.5556 | reverse              | False          |
| gemma-3-4b-it                 |      2 | conj  | letter       | h2           |                   0.5111 |                         0.5111 | direct               | False          |
| gemma-3-4b-it                 |      3 | bare  | letter       | h2           |                   0.4676 |                         0.5324 | reverse              | False          |
| gemma-3-4b-it                 |      3 | conj  | letter       | h2           |                   0.6444 |                         0.6444 | direct               | True           |
| Phi-4-mini-instruct           |      2 | bare  | letter       | h2           |                   0.5167 |                         0.5167 | direct               | False          |
| Phi-4-mini-instruct           |      2 | conj  | letter       | h2           |                   0.9583 |                         0.9583 | direct               | True           |
| Phi-4-mini-instruct           |      3 | bare  | letter       | h2           |                   0.512  |                         0.512  | direct               | False          |
| Phi-4-mini-instruct           |      3 | conj  | letter       | h2           |                   0.863  |                         0.863  | direct               | True           |
| Phi-4-mini-reasoning          |      2 | bare  | letter       | h2           |                   0.3639 |                         0.6361 | reverse              | True           |
| Phi-4-mini-reasoning          |      2 | conj  | letter       | h2           |                   0.4194 |                         0.5806 | reverse              | False          |
| Phi-4-mini-reasoning          |      3 | bare  | letter       | h2           |                   0.4815 |                         0.5185 | reverse              | False          |
| Phi-4-mini-reasoning          |      3 | conj  | letter       | h2           |                   0.6407 |                         0.6407 | direct               | True           |
| Qwen2.5-Math-1.5B             |      2 | bare  | letter       | h3           |                   0.7417 |                         0.7417 | direct               | True           |
| Qwen2.5-Math-1.5B             |      2 | conj  | letter       | h3           |                   0.9333 |                         0.9333 | direct               | True           |
| Qwen2.5-Math-1.5B             |      3 | bare  | letter       | h3           |                   0.5593 |                         0.5593 | direct               | False          |
| Qwen2.5-Math-1.5B             |      3 | conj  | letter       | h3           |                   0.7361 |                         0.7361 | direct               | True           |
| Qwen3-4B                      |      2 | bare  | letter       | h3           |                   0.2361 |                         0.7639 | reverse              | True           |
| Qwen3-4B                      |      2 | conj  | letter       | h3           |                   0.125  |                         0.875  | reverse              | True           |
| Qwen3-4B                      |      3 | bare  | letter       | h3           |                   0.3259 |                         0.6741 | reverse              | True           |
| Qwen3-4B                      |      3 | conj  | letter       | h3           |                   0.5148 |                         0.5148 | direct               | False          |
| Qwen3-4B-Instruct-2507        |      2 | bare  | letter       | h3           |                   0.5139 |                         0.5139 | direct               | False          |
| Qwen3-4B-Instruct-2507        |      2 | conj  | letter       | h3           |                   0.3694 |                         0.6306 | reverse              | True           |
| Qwen3-4B-Instruct-2507        |      3 | bare  | letter       | h3           |                   0.3241 |                         0.6759 | reverse              | True           |
| Qwen3-4B-Instruct-2507        |      3 | conj  | letter       | h3           |                   0.3861 |                         0.6139 | reverse              | True           |
| Qwen3-4B-Thinking-2507        |      2 | bare  | letter       | h3           |                   0.2111 |                         0.7889 | reverse              | True           |
| Qwen3-4B-Thinking-2507        |      2 | conj  | letter       | h3           |                   0.1361 |                         0.8639 | reverse              | True           |
| Qwen3-4B-Thinking-2507        |      3 | bare  | letter       | h3           |                   0.3148 |                         0.6852 | reverse              | True           |
| Qwen3-4B-Thinking-2507        |      3 | conj  | letter       | h3           |                   0.537  |                         0.537  | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | bare  | letter       | h3           |                   0.6611 |                         0.6611 | direct               | True           |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | conj  | letter       | h3           |                   0.8194 |                         0.8194 | direct               | True           |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | bare  | letter       | h3           |                   0.2491 |                         0.7509 | reverse              | True           |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | conj  | letter       | h3           |                   0.6111 |                         0.6111 | direct               | True           |
| gemma-3-4b-it                 |      2 | bare  | letter       | h3           |                   0.4361 |                         0.5639 | reverse              | False          |
| gemma-3-4b-it                 |      2 | conj  | letter       | h3           |                   0.5083 |                         0.5083 | direct               | False          |
| gemma-3-4b-it                 |      3 | bare  | letter       | h3           |                   0.338  |                         0.662  | reverse              | True           |
| gemma-3-4b-it                 |      3 | conj  | letter       | h3           |                   0.6046 |                         0.6046 | direct               | True           |
| Phi-4-mini-instruct           |      2 | bare  | letter       | h3           |                   0.5472 |                         0.5472 | direct               | False          |
| Phi-4-mini-instruct           |      2 | conj  | letter       | h3           |                   0.9667 |                         0.9667 | direct               | True           |
| Phi-4-mini-instruct           |      3 | bare  | letter       | h3           |                   0.537  |                         0.537  | direct               | False          |
| Phi-4-mini-instruct           |      3 | conj  | letter       | h3           |                   0.8157 |                         0.8157 | direct               | True           |
| Phi-4-mini-reasoning          |      2 | bare  | letter       | h3           |                   0.3306 |                         0.6694 | reverse              | True           |
| Phi-4-mini-reasoning          |      2 | conj  | letter       | h3           |                   0.3639 |                         0.6361 | reverse              | True           |
| Phi-4-mini-reasoning          |      3 | bare  | letter       | h3           |                   0.5046 |                         0.5046 | direct               | False          |
| Phi-4-mini-reasoning          |      3 | conj  | letter       | h3           |                   0.6157 |                         0.6157 | direct               | True           |
| Qwen2.5-Math-1.5B             |      2 | bare  | content_norm | h2           |                   0.4639 |                         0.5361 | reverse              | False          |
| Qwen2.5-Math-1.5B             |      2 | conj  | content_norm | h2           |                   0.5    |                         0.5    | direct               | False          |
| Qwen2.5-Math-1.5B             |      3 | bare  | content_norm | h2           |                   0.4731 |                         0.5269 | reverse              | False          |
| Qwen2.5-Math-1.5B             |      3 | conj  | content_norm | h2           |                   0.5009 |                         0.5009 | direct               | False          |
| Qwen3-4B                      |      2 | bare  | content_norm | h2           |                   0.6333 |                         0.6333 | direct               | True           |
| Qwen3-4B                      |      2 | conj  | content_norm | h2           |                   0.5    |                         0.5    | direct               | False          |
| Qwen3-4B                      |      3 | bare  | content_norm | h2           |                   0.5491 |                         0.5491 | direct               | False          |
| Qwen3-4B                      |      3 | conj  | content_norm | h2           |                   0.5    |                         0.5    | direct               | False          |
| Qwen3-4B-Instruct-2507        |      2 | bare  | content_norm | h2           |                   0.5694 |                         0.5694 | direct               | False          |
| Qwen3-4B-Instruct-2507        |      2 | conj  | content_norm | h2           |                   0.5028 |                         0.5028 | direct               | False          |
| Qwen3-4B-Instruct-2507        |      3 | bare  | content_norm | h2           |                   0.5046 |                         0.5046 | direct               | False          |
| Qwen3-4B-Instruct-2507        |      3 | conj  | content_norm | h2           |                   0.5028 |                         0.5028 | direct               | False          |
| Qwen3-4B-Thinking-2507        |      2 | bare  | content_norm | h2           |                   0.6083 |                         0.6083 | direct               | True           |
| Qwen3-4B-Thinking-2507        |      2 | conj  | content_norm | h2           |                   0.5028 |                         0.5028 | direct               | False          |
| Qwen3-4B-Thinking-2507        |      3 | bare  | content_norm | h2           |                   0.5491 |                         0.5491 | direct               | False          |
| Qwen3-4B-Thinking-2507        |      3 | conj  | content_norm | h2           |                   0.5056 |                         0.5056 | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | bare  | content_norm | h2           |                   0.5528 |                         0.5528 | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | conj  | content_norm | h2           |                   0.5    |                         0.5    | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | bare  | content_norm | h2           |                   0.512  |                         0.512  | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | conj  | content_norm | h2           |                   0.5    |                         0.5    | direct               | False          |
| gemma-3-4b-it                 |      2 | bare  | content_norm | h2           |                   0.5472 |                         0.5472 | direct               | False          |
| gemma-3-4b-it                 |      2 | conj  | content_norm | h2           |                   0.4278 |                         0.5722 | reverse              | False          |
| gemma-3-4b-it                 |      3 | bare  | content_norm | h2           |                   0.5148 |                         0.5148 | direct               | False          |
| gemma-3-4b-it                 |      3 | conj  | content_norm | h2           |                   0.4657 |                         0.5343 | reverse              | False          |
| Phi-4-mini-instruct           |      2 | bare  | content_norm | h2           |                   0.5056 |                         0.5056 | direct               | False          |
| Phi-4-mini-instruct           |      2 | conj  | content_norm | h2           |                   0.5028 |                         0.5028 | direct               | False          |
| Phi-4-mini-instruct           |      3 | bare  | content_norm | h2           |                   0.5009 |                         0.5009 | direct               | False          |
| Phi-4-mini-instruct           |      3 | conj  | content_norm | h2           |                   0.5231 |                         0.5231 | direct               | False          |
| Phi-4-mini-reasoning          |      2 | bare  | content_norm | h2           |                   0.5639 |                         0.5639 | direct               | False          |
| Phi-4-mini-reasoning          |      2 | conj  | content_norm | h2           |                   0.5611 |                         0.5611 | direct               | False          |
| Phi-4-mini-reasoning          |      3 | bare  | content_norm | h2           |                   0.4787 |                         0.5213 | reverse              | False          |
| Phi-4-mini-reasoning          |      3 | conj  | content_norm | h2           |                   0.5944 |                         0.5944 | direct               | False          |
| Qwen2.5-Math-1.5B             |      2 | bare  | content_norm | h3           |                   0.4222 |                         0.5778 | reverse              | False          |
| Qwen2.5-Math-1.5B             |      2 | conj  | content_norm | h3           |                   0.5    |                         0.5    | direct               | False          |
| Qwen2.5-Math-1.5B             |      3 | bare  | content_norm | h3           |                   0.4361 |                         0.5639 | reverse              | False          |
| Qwen2.5-Math-1.5B             |      3 | conj  | content_norm | h3           |                   0.5028 |                         0.5028 | direct               | False          |
| Qwen3-4B                      |      2 | bare  | content_norm | h3           |                   0.6639 |                         0.6639 | direct               | True           |
| Qwen3-4B                      |      2 | conj  | content_norm | h3           |                   0.5    |                         0.5    | direct               | False          |
| Qwen3-4B                      |      3 | bare  | content_norm | h3           |                   0.5611 |                         0.5611 | direct               | False          |
| Qwen3-4B                      |      3 | conj  | content_norm | h3           |                   0.5046 |                         0.5046 | direct               | False          |
| Qwen3-4B-Instruct-2507        |      2 | bare  | content_norm | h3           |                   0.5694 |                         0.5694 | direct               | False          |
| Qwen3-4B-Instruct-2507        |      2 | conj  | content_norm | h3           |                   0.5    |                         0.5    | direct               | False          |
| Qwen3-4B-Instruct-2507        |      3 | bare  | content_norm | h3           |                   0.4593 |                         0.5407 | reverse              | False          |
| Qwen3-4B-Instruct-2507        |      3 | conj  | content_norm | h3           |                   0.5028 |                         0.5028 | direct               | False          |
| Qwen3-4B-Thinking-2507        |      2 | bare  | content_norm | h3           |                   0.6194 |                         0.6194 | direct               | True           |
| Qwen3-4B-Thinking-2507        |      2 | conj  | content_norm | h3           |                   0.5    |                         0.5    | direct               | False          |
| Qwen3-4B-Thinking-2507        |      3 | bare  | content_norm | h3           |                   0.538  |                         0.538  | direct               | False          |
| Qwen3-4B-Thinking-2507        |      3 | conj  | content_norm | h3           |                   0.513  |                         0.513  | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | bare  | content_norm | h3           |                   0.5278 |                         0.5278 | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | conj  | content_norm | h3           |                   0.5    |                         0.5    | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | bare  | content_norm | h3           |                   0.4954 |                         0.5046 | reverse              | False          |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | conj  | content_norm | h3           |                   0.5    |                         0.5    | direct               | False          |
| gemma-3-4b-it                 |      2 | bare  | content_norm | h3           |                   0.4972 |                         0.5028 | reverse              | False          |
| gemma-3-4b-it                 |      2 | conj  | content_norm | h3           |                   0.3611 |                         0.6389 | reverse              | True           |
| gemma-3-4b-it                 |      3 | bare  | content_norm | h3           |                   0.4907 |                         0.5093 | reverse              | False          |
| gemma-3-4b-it                 |      3 | conj  | content_norm | h3           |                   0.4454 |                         0.5546 | reverse              | False          |
| Phi-4-mini-instruct           |      2 | bare  | content_norm | h3           |                   0.5278 |                         0.5278 | direct               | False          |
| Phi-4-mini-instruct           |      2 | conj  | content_norm | h3           |                   0.5    |                         0.5    | direct               | False          |
| Phi-4-mini-instruct           |      3 | bare  | content_norm | h3           |                   0.5213 |                         0.5213 | direct               | False          |
| Phi-4-mini-instruct           |      3 | conj  | content_norm | h3           |                   0.5194 |                         0.5194 | direct               | False          |
| Phi-4-mini-reasoning          |      2 | bare  | content_norm | h3           |                   0.5583 |                         0.5583 | direct               | False          |
| Phi-4-mini-reasoning          |      2 | conj  | content_norm | h3           |                   0.6167 |                         0.6167 | direct               | True           |
| Phi-4-mini-reasoning          |      3 | bare  | content_norm | h3           |                   0.5    |                         0.5    | direct               | False          |
| Phi-4-mini-reasoning          |      3 | conj  | content_norm | h3           |                   0.5741 |                         0.5741 | direct               | False          |

## Exploratory direct context effect on h3 conjunction-fallacy rate

Post-hoc test of delta CF_h3 = CF_h3(context) - CF_h3(no context). Positive is predicted; exact one- and two-sided sign-flip p-values are shown. No multiplicity correction is applied; every model-by-task cell is reported. Contaminated readouts must not support a conjunction-fallacy claim.

| model                         | readout      |   task |   cf_h3_no |   cf_h3_yes |    mean | ci_lo   | ci_hi   |   p_flip_two |   p_flip_one_positive |   positive |   negative |   ties |   position_only_predictability | position_direction   | contaminated   |
|:------------------------------|:-------------|-------:|-----------:|------------:|--------:|:--------|:--------|-------------:|----------------------:|-----------:|-----------:|-------:|-------------------------------:|:---------------------|:---------------|
| Qwen2.5-Math-1.5B             | letter       |      2 |     0.5389 |      0.4611 | -0.0778 | -0.1394 | -0.0162 |       0.0156 |                1      |          0 |          7 |      8 |                         0.9333 | direct               | True           |
| Qwen2.5-Math-1.5B             | letter       |      3 |     0.3407 |      0.3426 |  0.0019 | -0.0466 | 0.0503  |       1      |                0.5    |          7 |          5 |      3 |                         0.7361 | direct               | True           |
| Qwen3-4B                      | letter       |      2 |     0.5056 |      0.5667 |  0.0611 | -0.0252 | 0.1474  |       0.1973 |                0.0986 |          7 |          3 |      5 |                         0.875  | reverse              | True           |
| Qwen3-4B                      | letter       |      3 |     0.3148 |      0.4481 |  0.1333 | 0.0527  | 0.2139  |       0.0037 |                0.0019 |         12 |          3 |      0 |                         0.5148 | direct               | False          |
| Qwen3-4B-Instruct-2507        | letter       |      2 |     0.25   |      0.3444 |  0.0944 | -0.0614 | 0.2503  |       0.2407 |                0.1204 |         11 |          3 |      1 |                         0.6306 | reverse              | True           |
| Qwen3-4B-Instruct-2507        | letter       |      3 |     0.287  |      0.3111 |  0.0241 | -0.0412 | 0.0893  |       0.4681 |                0.2341 |          8 |          6 |      1 |                         0.6139 | reverse              | True           |
| Qwen3-4B-Thinking-2507        | letter       |      2 |     0.3944 |      0.4778 |  0.0833 | -0.0529 | 0.2196  |       0.2422 |                0.1211 |          8 |          1 |      6 |                         0.8639 | reverse              | True           |
| Qwen3-4B-Thinking-2507        | letter       |      3 |     0.1093 |      0.2648 |  0.1556 | 0.0793  | 0.2318  |       0.0012 |                0.0006 |         12 |          1 |      2 |                         0.537  | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B | letter       |      2 |     0.5167 |      0.5222 |  0.0056 | -0.0630 | 0.0742  |       1      |                0.5    |          5 |          4 |      6 |                         0.8194 | direct               | True           |
| DeepSeek-R1-Distill-Qwen-1.5B | letter       |      3 |     0.45   |      0.4352 | -0.0148 | -0.0758 | 0.0461  |       0.6562 |                0.7148 |          4 |          5 |      6 |                         0.6111 | direct               | True           |
| gemma-3-4b-it                 | letter       |      2 |     0.2278 |      0.4    |  0.1722 | -0.0052 | 0.3496  |       0.0527 |                0.0264 |         10 |          3 |      2 |                         0.5083 | direct               | False          |
| gemma-3-4b-it                 | letter       |      3 |     0.2815 |      0.413  |  0.1315 | 0.0384  | 0.2246  |       0.0049 |                0.0025 |         12 |          3 |      0 |                         0.6046 | direct               | True           |
| Phi-4-mini-instruct           | letter       |      2 |     0.5222 |      0.5    | -0.0222 | -0.0547 | 0.0103  |       0.2891 |                0.9648 |          2 |          6 |      7 |                         0.9667 | direct               | True           |
| Phi-4-mini-instruct           | letter       |      3 |     0.3074 |      0.3648 |  0.0574 | 0.0037  | 0.1112  |       0.0352 |                0.0176 |          9 |          2 |      4 |                         0.8157 | direct               | True           |
| Phi-4-mini-reasoning          | letter       |      2 |     0.6056 |      0.6111 |  0.0056 | -0.1012 | 0.1123  |       1      |                0.5    |          6 |          7 |      2 |                         0.6361 | reverse              | True           |
| Phi-4-mini-reasoning          | letter       |      3 |     0.3778 |      0.487  |  0.1093 | 0.0387  | 0.1799  |       0.0079 |                0.004  |         11 |          3 |      1 |                         0.6157 | direct               | True           |
| Qwen2.5-Math-1.5B             | content_norm |      2 |     1      |      1      |  0      |         |         |       1      |                1      |          0 |          0 |     15 |                         0.5    | direct               | False          |
| Qwen2.5-Math-1.5B             | content_norm |      3 |     0.9778 |      0.9981 |  0.0204 | -0.0193 | 0.0600  |       0.5    |                0.25   |          2 |          0 |     13 |                         0.5028 | direct               | False          |
| Qwen3-4B                      | content_norm |      2 |     1      |      1      |  0      |         |         |       1      |                1      |          0 |          0 |     15 |                         0.5    | direct               | False          |
| Qwen3-4B                      | content_norm |      3 |     0.9944 |      0.9926 | -0.0019 | -0.0154 | 0.0117  |       1      |                0.625  |          2 |          1 |     12 |                         0.5046 | direct               | False          |
| Qwen3-4B-Instruct-2507        | content_norm |      2 |     1      |      1      |  0      |         |         |       1      |                1      |          0 |          0 |     15 |                         0.5    | direct               | False          |
| Qwen3-4B-Instruct-2507        | content_norm |      3 |     0.9944 |      0.9963 |  0.0019 | -0.0104 | 0.0141  |       1      |                0.5    |          2 |          1 |     12 |                         0.5028 | direct               | False          |
| Qwen3-4B-Thinking-2507        | content_norm |      2 |     1      |      1      |  0      |         |         |       1      |                1      |          0 |          0 |     15 |                         0.5    | direct               | False          |
| Qwen3-4B-Thinking-2507        | content_norm |      3 |     0.9944 |      0.9722 | -0.0222 | -0.0567 | 0.0123  |       0.5    |                1      |          0 |          2 |     13 |                         0.513  | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B | content_norm |      2 |     1      |      1      |  0      |         |         |       1      |                1      |          0 |          0 |     15 |                         0.5    | direct               | False          |
| DeepSeek-R1-Distill-Qwen-1.5B | content_norm |      3 |     1      |      1      |  0      |         |         |       1      |                1      |          0 |          0 |     15 |                         0.5    | direct               | False          |
| gemma-3-4b-it                 | content_norm |      2 |     0.8778 |      0.8444 | -0.0333 | -0.1166 | 0.0500  |       0.5156 |                0.8281 |          3 |          4 |      8 |                         0.6389 | reverse              | True           |
| gemma-3-4b-it                 | content_norm |      3 |     0.8278 |      0.763  | -0.0648 | -0.1775 | 0.0479  |       0.2661 |                0.8798 |          6 |          8 |      1 |                         0.5546 | reverse              | False          |
| Phi-4-mini-instruct           | content_norm |      2 |     0.9722 |      0.9833 |  0.0111 | -0.0184 | 0.0406  |       0.75   |                0.375  |          2 |          1 |     12 |                         0.5    | direct               | False          |
| Phi-4-mini-instruct           | content_norm |      3 |     0.9389 |      0.9593 |  0.0204 | -0.0284 | 0.0692  |       0.5781 |                0.2891 |          4 |          3 |      8 |                         0.5194 | direct               | False          |
| Phi-4-mini-reasoning          | content_norm |      2 |     0.8222 |      0.9    |  0.0778 | 0.0028  | 0.1527  |       0.0625 |                0.0312 |         11 |          1 |      3 |                         0.6167 | direct               | True           |
| Phi-4-mini-reasoning          | content_norm |      3 |     0.8519 |      0.9185 |  0.0667 | 0.0206  | 0.1127  |       0.0098 |                0.0049 |         11 |          1 |      3 |                         0.5741 | direct               | False          |

## H1c metric correlations

| model                         |   task | readout      | spearman   | p      |   n |
|:------------------------------|-------:|:-------------|:-----------|:-------|----:|
| Qwen2.5-Math-1.5B             |      2 | letter       | 0.0037     | 0.9895 |  15 |
| Qwen2.5-Math-1.5B             |      3 | letter       | 0.0340     | 0.9044 |  15 |
| Qwen3-4B                      |      2 | letter       | -0.2688    | 0.3326 |  15 |
| Qwen3-4B                      |      3 | letter       | 0.0214     | 0.9395 |  15 |
| Qwen3-4B-Instruct-2507        |      2 | letter       | 0.2359     | 0.3973 |  15 |
| Qwen3-4B-Instruct-2507        |      3 | letter       | 0.0500     | 0.8594 |  15 |
| Qwen3-4B-Thinking-2507        |      2 | letter       | 0.0108     | 0.9696 |  15 |
| Qwen3-4B-Thinking-2507        |      3 | letter       | -0.0805    | 0.7755 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | letter       | -0.1304    | 0.6431 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | letter       | 0.1968     | 0.4821 |  15 |
| gemma-3-4b-it                 |      2 | letter       | 0.3695     | 0.1753 |  15 |
| gemma-3-4b-it                 |      3 | letter       | 0.5679     | 0.0272 |  15 |
| Phi-4-mini-instruct           |      2 | letter       | 0.0283     | 0.9204 |  15 |
| Phi-4-mini-instruct           |      3 | letter       | 0.1153     | 0.6824 |  15 |
| Phi-4-mini-reasoning          |      2 | letter       | -0.2148    | 0.4420 |  15 |
| Phi-4-mini-reasoning          |      3 | letter       | 0.3143     | 0.2539 |  15 |
| Qwen2.5-Math-1.5B             |      2 | content_norm |            |        |  15 |
| Qwen2.5-Math-1.5B             |      3 | content_norm | -0.0162    | 0.9543 |  15 |
| Qwen3-4B                      |      2 | content_norm |            |        |  15 |
| Qwen3-4B                      |      3 | content_norm | 0.3427     | 0.2111 |  15 |
| Qwen3-4B-Instruct-2507        |      2 | content_norm | -0.3093    | 0.2620 |  15 |
| Qwen3-4B-Instruct-2507        |      3 | content_norm | 0.1925     | 0.4919 |  15 |
| Qwen3-4B-Thinking-2507        |      2 | content_norm | -0.3712    | 0.1732 |  15 |
| Qwen3-4B-Thinking-2507        |      3 | content_norm | 0.0587     | 0.8353 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | content_norm |            |        |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | content_norm |            |        |  15 |
| gemma-3-4b-it                 |      2 | content_norm | 0.2380     | 0.3930 |  15 |
| gemma-3-4b-it                 |      3 | content_norm | 0.6452     | 0.0094 |  15 |
| Phi-4-mini-instruct           |      2 | content_norm | 0.2247     | 0.4208 |  15 |
| Phi-4-mini-instruct           |      3 | content_norm | -0.0315    | 0.9113 |  15 |
| Phi-4-mini-reasoning          |      2 | content_norm | 0.1807     | 0.5193 |  15 |
| Phi-4-mini-reasoning          |      3 | content_norm | 0.2106     | 0.4511 |  15 |

## H2a norming shifts

| model                         | measure   | hypothesis   |    mean |    ci_lo |   ci_hi |   p_flip_two |   positive |   negative |
|:------------------------------|:----------|:-------------|--------:|---------:|--------:|-------------:|-----------:|-----------:|
| Qwen2.5-Math-1.5B             | rho       | h2           |  0.1603 |  -0.1707 |  0.4913 |       0.3105 |         10 |          5 |
| Qwen3-4B                      | rho       | h2           | -6.606  | -10.8151 | -2.3969 |       0.006  |          1 |         14 |
| Qwen3-4B-Instruct-2507        | rho       | h2           | -7.7316 | -12.0313 | -3.4319 |       0.003  |          3 |         12 |
| Qwen3-4B-Thinking-2507        | rho       | h2           | -1.757  |  -2.6861 | -0.8278 |       0.0013 |          3 |         12 |
| DeepSeek-R1-Distill-Qwen-1.5B | rho       | h2           |  0.9451 |   0.6211 |  1.269  |       0.0001 |         14 |          1 |
| gemma-3-4b-it                 | rho       | h2           |  0.3899 |  -1.6967 |  2.4764 |       0.6984 |          7 |          8 |
| Phi-4-mini-instruct           | rho       | h2           | -2.2    |  -3.3375 | -1.0625 |       0.001  |          3 |         12 |
| Phi-4-mini-reasoning          | rho       | h2           | -0.8931 |  -2.7355 |  0.9493 |       0.3232 |          5 |         10 |
| Qwen2.5-Math-1.5B             | rho       | h3           |  0.2896 |  -0.0444 |  0.6236 |       0.0845 |         10 |          5 |
| Qwen3-4B                      | rho       | h3           |  1.6041 |  -3.4285 |  6.6368 |       0.4969 |          8 |          7 |
| Qwen3-4B-Instruct-2507        | rho       | h3           | -0.3635 |  -2.998  |  2.2709 |       0.7679 |          8 |          7 |
| Qwen3-4B-Thinking-2507        | rho       | h3           | -0.4968 |  -0.937  | -0.0565 |       0.0304 |          4 |         11 |
| DeepSeek-R1-Distill-Qwen-1.5B | rho       | h3           |  0.9648 |   0.6656 |  1.2639 |       0.0002 |         14 |          1 |
| gemma-3-4b-it                 | rho       | h3           | -0.2108 |  -4.3251 |  3.9035 |       0.9249 |          8 |          7 |
| Phi-4-mini-instruct           | rho       | h3           | -0.5465 |  -1.2893 |  0.1963 |       0.139  |          4 |         11 |
| Phi-4-mini-reasoning          | rho       | h3           | -2.279  |  -3.944  | -0.614  |       0.0053 |          4 |         11 |
| Qwen2.5-Math-1.5B             | C         | h2           |  0.8396 |   0.3156 |  1.3635 |       0.0056 |         12 |          3 |
| Qwen3-4B                      | C         | h2           | -4.3123 |  -6.3871 | -2.2376 |       0.0003 |          1 |         14 |
| Qwen3-4B-Instruct-2507        | C         | h2           | -2.9656 |  -5.196  | -0.7353 |       0.0009 |          2 |         13 |
| Qwen3-4B-Thinking-2507        | C         | h2           |  1.0721 |   0.6111 |  1.5331 |       0.0005 |         13 |          2 |
| DeepSeek-R1-Distill-Qwen-1.5B | C         | h2           |  0.178  |  -0.1312 |  0.4872 |       0.2338 |         10 |          5 |
| gemma-3-4b-it                 | C         | h2           | -4.1912 |  -7.6463 | -0.7362 |       0.0222 |          1 |         14 |
| Phi-4-mini-instruct           | C         | h2           | -2.4347 |  -3.3917 | -1.4778 |       0.0001 |          0 |         15 |
| Phi-4-mini-reasoning          | C         | h2           | -1.4115 |  -2.4599 | -0.3631 |       0.0137 |          4 |         11 |
| Qwen2.5-Math-1.5B             | C         | h3           |  0.8412 |   0.2866 |  1.3958 |       0.0074 |         12 |          3 |
| Qwen3-4B                      | C         | h3           | -0.728  |  -3.4684 |  2.0124 |       0.5754 |          7 |          8 |
| Qwen3-4B-Instruct-2507        | C         | h3           | -2.7922 |  -5.8633 |  0.2789 |       0.0723 |          4 |         11 |
| Qwen3-4B-Thinking-2507        | C         | h3           |  1.1026 |   0.5959 |  1.6093 |       0.0011 |         13 |          2 |
| DeepSeek-R1-Distill-Qwen-1.5B | C         | h3           |  0.2065 |  -0.076  |  0.489  |       0.135  |         10 |          5 |
| gemma-3-4b-it                 | C         | h3           |  1.3737 |  -2.0367 |  4.784  |       0.4019 |          9 |          6 |
| Phi-4-mini-instruct           | C         | h3           | -1.9865 |  -3.098  | -0.8749 |       0.0004 |          2 |         13 |
| Phi-4-mini-reasoning          | C         | h3           | -1.6177 |  -2.9477 | -0.2876 |       0.0217 |          4 |         11 |

## H2b change/level connections

| model                         |   task | measure   | path   |   spearman |      p |   n |
|:------------------------------|-------:|:----------|:-------|-----------:|-------:|----:|
| Qwen2.5-Math-1.5B             |      2 | rho       | change |    -0.4929 | 0.062  |  15 |
| Qwen2.5-Math-1.5B             |      2 | rho       | level  |     0.0714 | 0.8003 |  15 |
| Qwen2.5-Math-1.5B             |      2 | C         | change |     0.4393 | 0.1014 |  15 |
| Qwen2.5-Math-1.5B             |      2 | C         | level  |     0.1    | 0.7229 |  15 |
| Qwen2.5-Math-1.5B             |      3 | rho       | change |    -0.5321 | 0.0412 |  15 |
| Qwen2.5-Math-1.5B             |      3 | rho       | level  |     0.0536 | 0.8496 |  15 |
| Qwen2.5-Math-1.5B             |      3 | C         | change |     0.3536 | 0.1961 |  15 |
| Qwen2.5-Math-1.5B             |      3 | C         | level  |     0.1071 | 0.7039 |  15 |
| Qwen3-4B                      |      2 | rho       | change |    -0.1714 | 0.5413 |  15 |
| Qwen3-4B                      |      2 | rho       | level  |    -0.0929 | 0.742  |  15 |
| Qwen3-4B                      |      2 | C         | change |    -0.2    | 0.4748 |  15 |
| Qwen3-4B                      |      2 | C         | level  |    -0.225  | 0.4201 |  15 |
| Qwen3-4B                      |      3 | rho       | change |    -0.1821 | 0.5159 |  15 |
| Qwen3-4B                      |      3 | rho       | level  |    -0.2214 | 0.4277 |  15 |
| Qwen3-4B                      |      3 | C         | change |    -0.15   | 0.5936 |  15 |
| Qwen3-4B                      |      3 | C         | level  |    -0.275  | 0.3212 |  15 |
| Qwen3-4B-Instruct-2507        |      2 | rho       | change |    -0.1143 | 0.6851 |  15 |
| Qwen3-4B-Instruct-2507        |      2 | rho       | level  |    -0.1964 | 0.4829 |  15 |
| Qwen3-4B-Instruct-2507        |      2 | C         | change |    -0.0179 | 0.9496 |  15 |
| Qwen3-4B-Instruct-2507        |      2 | C         | level  |    -0.2393 | 0.3904 |  15 |
| Qwen3-4B-Instruct-2507        |      3 | rho       | change |    -0.1464 | 0.6025 |  15 |
| Qwen3-4B-Instruct-2507        |      3 | rho       | level  |    -0.1821 | 0.5159 |  15 |
| Qwen3-4B-Instruct-2507        |      3 | C         | change |     0.0643 | 0.8199 |  15 |
| Qwen3-4B-Instruct-2507        |      3 | C         | level  |    -0.1929 | 0.491  |  15 |
| Qwen3-4B-Thinking-2507        |      2 | rho       | change |    -0.2643 | 0.3412 |  15 |
| Qwen3-4B-Thinking-2507        |      2 | rho       | level  |    -0.1643 | 0.5585 |  15 |
| Qwen3-4B-Thinking-2507        |      2 | C         | change |     0.275  | 0.3212 |  15 |
| Qwen3-4B-Thinking-2507        |      2 | C         | level  |    -0.0857 | 0.7613 |  15 |
| Qwen3-4B-Thinking-2507        |      3 | rho       | change |    -0.2286 | 0.4126 |  15 |
| Qwen3-4B-Thinking-2507        |      3 | rho       | level  |    -0.1607 | 0.5672 |  15 |
| Qwen3-4B-Thinking-2507        |      3 | C         | change |     0.2321 | 0.4051 |  15 |
| Qwen3-4B-Thinking-2507        |      3 | C         | level  |     0.0143 | 0.9597 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | rho       | change |     0.5214 | 0.0462 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | rho       | level  |    -0.0536 | 0.8496 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | C         | change |    -0.0321 | 0.9095 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      2 | C         | level  |     0.2286 | 0.4126 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | rho       | change |     0.5536 | 0.0323 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | rho       | level  |    -0.0107 | 0.9698 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | C         | change |     0.0464 | 0.8695 |  15 |
| DeepSeek-R1-Distill-Qwen-1.5B |      3 | C         | level  |     0.1429 | 0.6115 |  15 |
| gemma-3-4b-it                 |      2 | rho       | change |    -0.3786 | 0.1641 |  15 |
| gemma-3-4b-it                 |      2 | rho       | level  |     0.1036 | 0.7134 |  15 |
| gemma-3-4b-it                 |      2 | C         | change |    -0.2286 | 0.4126 |  15 |
| gemma-3-4b-it                 |      2 | C         | level  |     0.2429 | 0.3831 |  15 |
| gemma-3-4b-it                 |      3 | rho       | change |    -0.3464 | 0.2059 |  15 |
| gemma-3-4b-it                 |      3 | rho       | level  |    -0.0036 | 0.9899 |  15 |
| gemma-3-4b-it                 |      3 | C         | change |    -0.1571 | 0.576  |  15 |
| gemma-3-4b-it                 |      3 | C         | level  |     0.1    | 0.7229 |  15 |
| Phi-4-mini-instruct           |      2 | rho       | change |    -0.3036 | 0.2714 |  15 |
| Phi-4-mini-instruct           |      2 | rho       | level  |     0.0286 | 0.9195 |  15 |
| Phi-4-mini-instruct           |      2 | C         | change |    -0.1607 | 0.5672 |  15 |
| Phi-4-mini-instruct           |      2 | C         | level  |     0.0536 | 0.8496 |  15 |
| Phi-4-mini-instruct           |      3 | rho       | change |    -0.3107 | 0.2597 |  15 |
| Phi-4-mini-instruct           |      3 | rho       | level  |     0.0071 | 0.9798 |  15 |
| Phi-4-mini-instruct           |      3 | C         | change |    -0.1679 | 0.5499 |  15 |
| Phi-4-mini-instruct           |      3 | C         | level  |     0.0857 | 0.7613 |  15 |
| Phi-4-mini-reasoning          |      2 | rho       | change |    -0.3179 | 0.2483 |  15 |
| Phi-4-mini-reasoning          |      2 | rho       | level  |     0.075  | 0.7905 |  15 |
| Phi-4-mini-reasoning          |      2 | C         | change |    -0.3107 | 0.2597 |  15 |
| Phi-4-mini-reasoning          |      2 | C         | level  |     0.1571 | 0.576  |  15 |
| Phi-4-mini-reasoning          |      3 | rho       | change |    -0.0179 | 0.9496 |  15 |
| Phi-4-mini-reasoning          |      3 | rho       | level  |    -0.0536 | 0.8496 |  15 |
| Phi-4-mini-reasoning          |      3 | C         | change |    -0.425  | 0.1143 |  15 |
| Phi-4-mini-reasoning          |      3 | C         | level  |     0.3607 | 0.1866 |  15 |

H2 null results are not equivalence evidence; no TOST margin is available.
