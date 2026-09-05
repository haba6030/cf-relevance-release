# Conditional CF rescue and knockout

Selection uses one baseline frame and evaluation uses the other; the split is cross-swapped. 
The two cross-frame estimates are averaged within item before inference (maximum n=165). 
Primary outcome is normalized `conj_h3 - base`; DoD is secondary. Raw p-values are reported without Holm.

| stage | direction | alpha | dose | cells passed | mean ΔCF | mean crossing |
|---|---|---:|---:|---:|---:|---:|
| ABL | PC_given_R |  | 0.5 | 2/12 | -0.0047 | 2.2% |
| ABL | PC_given_R |  | 1 | 1/12 | -0.0884 | 7.4% |
| ABL | R_given_PC |  | 0.5 | 2/12 | -0.0051 | 3.3% |
| ABL | R_given_PC |  | 1 | 2/12 | -0.0615 | 8.4% |
| ADD | PC_given_R | 0.5 |  | 3/12 | +0.0009 | 4.1% |
| ADD | PC_given_R | 1 |  | 3/12 | +0.0029 | 8.5% |
| ADD | PC_given_R | 2 |  | 0/12 | +0.0474 | 15.5% |
| ADD | R_given_PC | 0.5 |  | 3/12 | -0.0392 | 5.9% |
| ADD | R_given_PC | 1 |  | 3/12 | -0.0728 | 10.3% |
| ADD | R_given_PC | 2 |  | 0/12 | -0.0791 | 12.7% |

A cell passes only if the direction gate, one-sided item sign-flip, 
angle/constraint-matched random and shuffle comparisons, full-165-fold WikiText fluency guard, 
and the base-collapse guard all pass.
