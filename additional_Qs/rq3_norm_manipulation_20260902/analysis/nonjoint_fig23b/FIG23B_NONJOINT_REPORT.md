# Fig. 23B: separate P and C readouts of the shared PC|R intervention

This analysis does not re-run the shared PC|R intervention; it recomputes one-sided tests of P and C on CPU from the stored 165-item item-level ΔDID.
No joint IUT gate over P and C and no Holm correction are used. Note that the two tests are two readouts of the same PC|R intervention, not separate P|R and C|R direction edits.

## Pass counts

| readout | target | random | shuffle |
|---|---:|---:|---:|
| P | 2/13 | 0/13 | 2/13 |
| C | 6/13 | 0/13 | 2/13 |

## Target cells

| model | task | stage | strength | source | readout | oriented ΔDID | raw one-sided p | pass | n |
|---|---:|---|---:|---|---|---:|---:|---|---:|
| Qwen2.5-Math-1.5B | 2 | ABL | 0.5 | 0830_primary_pass | C | -0.0004 | 0.6852 | no | 56 |
| Qwen2.5-Math-1.5B | 2 | ABL | 0.5 | 0830_primary_pass | P | -0.0026 | 0.9209 | no | 56 |
| Qwen2.5-Math-1.5B | 2 | ABL | 1 | 0830_primary_pass | C | -0.0005 | 0.6891 | no | 56 |
| Qwen2.5-Math-1.5B | 2 | ABL | 1 | 0830_primary_pass | P | -0.0050 | 0.9242 | no | 56 |
| Qwen3-4B | 2 | ABL | 0.5 | 0830_primary_pass | C | +0.0146 | 0.05146 | no | 131 |
| Qwen3-4B | 2 | ABL | 0.5 | 0830_primary_pass | P | +0.0381 | 0.001837 | yes | 131 |
| Qwen3-4B | 2 | ABL | 1 | 0901_posthoc_fluency_six | C | +0.0123 | 0.2149 | no | 131 |
| Qwen3-4B | 2 | ABL | 1 | 0901_posthoc_fluency_six | P | +0.1729 | 6.359e-06 | yes | 131 |
| Qwen3-4B | 3 | ADD | 0.5 | 0830_primary_pass | C | +0.0207 | 0.0004316 | yes | 56 |
| Qwen3-4B | 3 | ADD | 0.5 | 0830_primary_pass | P | -0.0634 | 0.9999 | no | 56 |
| Qwen3-4B | 3 | ADD | 1 | 0830_primary_pass | C | +0.0381 | 0.0003643 | yes | 56 |
| Qwen3-4B | 3 | ADD | 1 | 0830_primary_pass | P | -0.1071 | 0.9998 | no | 56 |
| Qwen3-4B | 3 | ADD | 2 | 0901_posthoc_fluency_six | C | +0.0670 | 0.0003404 | yes | 56 |
| Qwen3-4B | 3 | ADD | 2 | 0901_posthoc_fluency_six | P | -0.1715 | 0.9999 | no | 56 |
| Qwen3-4B-Instruct-2507 | 2 | ADD | 0.5 | 0830_primary_pass | C | +0.0687 | 0.009927 | yes | 54 |
| Qwen3-4B-Instruct-2507 | 2 | ADD | 0.5 | 0830_primary_pass | P | -0.0268 | 0.9044 | no | 54 |
| Qwen3-4B-Instruct-2507 | 2 | ADD | 1 | 0830_primary_pass | C | +0.1074 | 0.03804 | yes | 54 |
| Qwen3-4B-Instruct-2507 | 2 | ADD | 1 | 0830_primary_pass | P | -0.0730 | 0.9572 | no | 54 |
| Qwen3-4B-Instruct-2507 | 2 | ADD | 2 | 0901_posthoc_fluency_six | C | +0.0429 | 0.344 | no | 54 |
| Qwen3-4B-Instruct-2507 | 2 | ADD | 2 | 0901_posthoc_fluency_six | P | -0.2215 | 0.9928 | no | 54 |
| Qwen3-4B-Instruct-2507 | 3 | ADD | 0.5 | 0830_primary_pass | C | +0.0358 | 0.0263 | yes | 65 |
| Qwen3-4B-Instruct-2507 | 3 | ADD | 0.5 | 0830_primary_pass | P | -0.0483 | 0.998 | no | 65 |
| Qwen3-4B-Instruct-2507 | 3 | ADD | 1 | 0830_primary_pass | C | +0.0501 | 0.1093 | no | 65 |
| Qwen3-4B-Instruct-2507 | 3 | ADD | 1 | 0830_primary_pass | P | -0.0985 | 0.9972 | no | 65 |
| Qwen3-4B-Instruct-2507 | 3 | ADD | 2 | 0901_posthoc_fluency_six | C | -0.0151 | 0.5714 | no | 65 |
| Qwen3-4B-Instruct-2507 | 3 | ADD | 2 | 0901_posthoc_fluency_six | P | -0.1774 | 0.9847 | no | 65 |

## Decision contract

The null hypothesis of each readout is that its mean effect in the predicted direction is at most zero. ADD/ABL and the sign of C are oriented as in the original intervention contract. Each P or C cell passes individually at uncorrected one-sided t p<.05. There is no AND gate between P and C.
The effect is ΔDID = [(R−I)@h3−(R−I)@h2]_edited − [(R−I)@h3−(R−I)@h2]_baseline; the two cross-frame repetitions are averaged within item.
