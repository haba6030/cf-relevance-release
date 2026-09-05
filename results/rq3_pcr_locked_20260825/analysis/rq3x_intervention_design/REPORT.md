# Intervention redesign diagnostics: one dimension is not enough

The current H3d/e ablates (ABL) and injects (ADD) **one direction** `v_R`, the mean of the 15
per-item R,h3 − I,h3 contrasts. Measuring how many dimensions those 15 vectors span shows
that they are not summarised by a single direction.

| Model | Task | Layer | PR | PC1 | PC1–2 | PC1–3 | PC1–4 | PC1–6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Math-1.5B | 2 | 14 | 5.27 | 0.384 | 0.529 | 0.608 | 0.663 | 0.760 |
| Qwen2.5-Math-1.5B | 3 | 20 | 7.79 | 0.250 | 0.417 | 0.542 | 0.622 | 0.745 |
| Qwen3-4B-Instruct-2507 | 2 | 22 | 3.80 | 0.470 | 0.631 | 0.700 | 0.763 | 0.853 |
| Qwen3-4B-Instruct-2507 | 3 | 20 | 3.49 | 0.501 | 0.646 | 0.711 | 0.767 | 0.854 |
| Qwen3-4B | 2 | 22 | 5.24 | 0.360 | 0.547 | 0.655 | 0.725 | 0.824 |
| Qwen3-4B | 3 | 20 | 4.15 | 0.443 | 0.600 | 0.687 | 0.753 | 0.848 |
| gemma-3-4b-it | 2 | 34 | 4.60 | 0.380 | 0.619 | 0.696 | 0.751 | 0.828 |
| gemma-3-4b-it | 3 | 34 | 3.96 | 0.446 | 0.646 | 0.716 | 0.763 | 0.835 |
| Phi-4-mini-instruct | 2 | 10 | 3.49 | 0.479 | 0.686 | 0.778 | 0.817 | 0.880 |
| Phi-4-mini-instruct | 3 | 10 | 3.48 | 0.478 | 0.692 | 0.782 | 0.824 | 0.884 |
| Phi-4-mini-reasoning | 2 | 11 | 5.47 | 0.323 | 0.561 | 0.654 | 0.721 | 0.808 |
| Phi-4-mini-reasoning | 3 | 13 | 5.05 | 0.369 | 0.570 | 0.666 | 0.726 | 0.812 |

## v_R is not PC1

`v_R` is the **mean** of the 15 contrasts and PC1 the **direction of maximal variance**. Close, but not identical.

| Model | Task | \|cos(v_R, PC1)\| | \|cos(v_R, PC2)\| | share of v_R inside PC1–4 |
|---|---:|---:|---:|---:|
| Qwen2.5-Math-1.5B | 2 | 0.819 | 0.452 | 0.987 |
| Qwen2.5-Math-1.5B | 3 | 0.911 | 0.290 | 0.970 |
| Qwen3-4B-Instruct-2507 | 2 | 0.902 | 0.391 | 0.991 |
| Qwen3-4B-Instruct-2507 | 3 | 0.939 | 0.299 | 0.990 |
| Qwen3-4B | 2 | 0.918 | 0.233 | 0.984 |
| Qwen3-4B | 3 | 0.926 | 0.280 | 0.987 |
| gemma-3-4b-it | 2 | 0.951 | 0.248 | 0.984 |
| gemma-3-4b-it | 3 | 0.987 | 0.051 | 0.990 |
| Phi-4-mini-instruct | 2 | 0.933 | 0.285 | 0.994 |
| Phi-4-mini-instruct | 3 | 0.928 | 0.308 | 0.994 |
| Phi-4-mini-reasoning | 2 | 0.919 | 0.045 | 0.976 |
| Phi-4-mini-reasoning | 3 | 0.834 | 0.370 | 0.978 |

Mean |cos(v_R, PC1)| = **0.914**, about 24 degrees apart. Still, 98.5% of `v_R` lies in the subspace spanned by the top four PCs.

**So 'k=1 equals the current intervention' cannot be claimed.** For the subspace to contain the current
direction, `v_R` must be placed in the basis explicitly.

**PC1 explains 40.7% of the variance on average.** Four PCs reach 74.1%, six reach 82.8%. The participation ratio is 3.48–7.79, nowhere near 1 in any cell.

**The current intervention therefore pushes less than half of the relevance contrast.** This is why the
failure of ABL/ADD cannot be read as 'relevance is not causal': the one-dimensional premise itself
does not fit the data.

## How many dimensions does reading need? Measured directly by decoding AUC

PC1 at 41% is indirect evidence. **Measured directly in the same test (LOIO decoding AUC):**

| Model | Task | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 | k=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Math-1.5B | 2 | 0.702 | 0.769 | 0.756 | 0.760 | 0.747 | 0.742 | 0.724 |
| Qwen2.5-Math-1.5B | 3 | 0.662 | 0.764 | 0.716 | 0.738 | 0.760 | 0.751 | 0.751 |
| Qwen3-4B-Instruct-2507 | 2 | 0.764 | 0.827 | 0.831 | 0.818 | 0.827 | 0.831 | 0.818 |
| Qwen3-4B-Instruct-2507 | 3 | 0.831 | 0.849 | 0.844 | 0.840 | 0.853 | 0.844 | 0.849 |
| Qwen3-4B | 2 | 0.724 | 0.800 | 0.791 | 0.791 | 0.809 | 0.804 | 0.791 |
| Qwen3-4B | 3 | 0.747 | 0.818 | 0.809 | 0.818 | 0.813 | 0.813 | 0.818 |
| gemma-3-4b-it | 2 | 0.773 | 0.831 | 0.827 | 0.827 | 0.831 | 0.836 | 0.840 |
| gemma-3-4b-it | 3 | 0.827 | 0.827 | 0.827 | 0.822 | 0.822 | 0.827 | 0.822 |
| Phi-4-mini-instruct | 2 | 0.836 | 0.858 | 0.858 | 0.853 | 0.858 | 0.862 | 0.862 |
| Phi-4-mini-instruct | 3 | 0.827 | 0.876 | 0.880 | 0.876 | 0.871 | 0.871 | 0.871 |
| Phi-4-mini-reasoning | 2 | 0.778 | 0.782 | 0.796 | 0.809 | 0.787 | 0.733 | 0.764 |
| Phi-4-mini-reasoning | 3 | 0.711 | 0.787 | 0.787 | 0.791 | 0.778 | 0.791 | 0.778 |

Mean k=1 0.765 · k=2 0.816 · k=3 0.810 · k=4 0.812 · k=6 0.813 · k=8 0.809 · k=10 0.807.

**From k=1 to k=2 the AUC rises 0.765 → 0.816 and is flat afterwards.**
k=2 beats k=1 in 11 of 12 cells. **Even reading needs more than one dimension**, and
it saturates at two, consistent with PC1–2 explaining 59.5%.

> **Where reading saturates is not where writing suffices.** A classifier can succeed on part of the signal,
> but changing behaviour requires touching enough of the component the model actually uses. The intervention's
> k should therefore be explored from 2 upward, with the curve above as a lower bound.

## Encoding or decoding form for the intervention

§3-D showed the two forms diverge at mean cos 0.46. However,
**encoding is the right form for intervention.** Decoding weights include components that cancel noise
to read the target well; pushing along them pushes the cancelling component, not the signal
(Haufe et al. 2014). H3b (representation) and H3d/e (causation) call for different forms.

Fortunately the current `v_R` = difference in means is already an encoding form, so **the direction
definition need not change.** What must change is the dimension.

That the two forms diverge this much also means **the activation covariance is strongly anisotropic**
(under isotropy `w` and `beta` coincide). One-dimensional interventions are especially fragile in anisotropic spaces.
