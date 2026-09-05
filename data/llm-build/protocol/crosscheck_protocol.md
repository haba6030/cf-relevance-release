# Cross-check protocol: LLM-as-a-judge

The rules for accepting or rejecting an item. **Fixed before items are written** and not changed afterwards.
Moving a threshold post hoc turns validation into selection bias.

This document and all of `prompts/2*.md` are released with the paper.

---

## 1. Judge pool: frontier models

### 1.1 Why not a local 8B model

**Judges must be stronger than the models under test.** Answerhood (J1) and logical independence (J2) carry
the validity of the stimulus design; if those judgments come from models of the same level as the subjects,
one gets "stimuli that models of that class deem valid", not "valid stimuli".

The pilot in §3.1 shows it numerically: `llama3.1:8b` failed with discrimination AUC 0.748 (threshold .75)
and lexical contamination +0.32 (threshold .30). **A local 8B model is not a judge route.**

### 1.2 Judge composition

| role | model | note |
|---|---|---|
| **generation** | one Claude Opus model | `prompts/10_generate_item.md` |
| **judging** | Claude Opus, a GPT model, a third frontier model; at least 3 families | a judge that coincides with the generator is excluded on that item |
| **forbidden** | `Qwen3-*`, `Phi-4-*`, `aya-expanse-8b`, `LMT-60-4B`, `translategemma-4b-it` | **models under test**; self-preference |
| **forbidden** | local 8B models such as `llama3.1:8b`, `qwen3:8b` | §1.1 |

The judge unit is not the model but **`judge_id` = (model × prompt variant × seed)**.

```
models          at least 3 different families
prompts         2 variants (original / instruction blocks reordered)
seeds/samples   2
-----------------------------------------
judge_id    3 × 2 × 2 = 12
```

Seed and prompt variants measure **within-model instability**; differences between models measure **between-model disagreement**.
Inter-rater reliability is reported for the two separately.

### 1.3 Self-preference: when generator and judge coincide

Both generation and judging use frontier models, so **the same model may appear on both sides.** Then

1. that model's judgments are **excluded** on the items it generated,
2. acceptance rates before and after exclusion are **both reported**,
3. and per-model acceptance differences are reported (does a model pass its own items more often).

### 1.4 Cost and execution

24 items × 6 judgments × 12 judge_ids × 2 blinding orders ≈ **3,500 calls**.
At 1–2k tokens per prompt this is affordable, and
API access is not a blocker.

`run/run_judges.py` selects the route with `--backend {ollama,anthropic,openai}`.
API keys are read from environment variables and never stored.

## 2. Blinding and counterbalancing

| operation | content |
|---|---|
| **label blinding** | judges never see `h2`/`h3`; options are shown as `A`/`B` with the assignment randomised per item |
| **order balance** | every item is sent in **both** orders `(A=h2, B=h3)` and `(A=h3, B=h2)`; position bias cancels, and a judge whose verdict flips between the two is void on that item |
| **field blocking** | `notes`, `bridge`, `cost`, `src`, `generator` and the target norms `P_*`/`C_*` are never sent |
| **item order** | randomised per judge to block order effects between neighbouring items |

The assignment map (`blind_map_id` → actual assignment) is opened only after all judging is complete.

## 3. Pre-calibration: is the judge qualified?

**Before items are sent**, check whether the judge measures answerhood or lexical overlap.
Data: external pragmatics corpora (not redistributed). Procedure: `protocol/calibration.md`.

| check | data | pass criterion |
|---|---|---|
| **convergence** | stratified sample of 300 human `c2_answer_compat` labels from QUDeval | Spearman ρ between J1 answerhood and human labels ≥ .45; exact agreement on the 3-point scale ≥ 55% |
| **discrimination** | 20 `C-neg-constructed` (overlap / answerhood 0) + 20 implicature pairs (no overlap / answerhood 1) | the two sets must separate; AUC ≥ .75 |
| **lexical contamination** | on those 40, the correlation between judge `answerhood` and **machine-counted** overlap | \|ρ\| ≤ .30; above it the judge measures vocabulary, not answerhood → **exclude**. The judge's own `overlap` rating is not used (§3.1) |

### 3.1 Judge qualification: run twice, and **the first run's test was itself wrong**

#### Round 1 (confounded two-cell set)

| model | parsed | discrimination AUC | lexical contamination |
|---|---:|---:|---:|
| `llama3.1:8b` | 40/40 | 0.573 | ρ=+0.26 "pass" |
| `qwen3:8b` | 40/40 | 0.698 | ρ=−0.43 "exclude" |

**These verdicts are unusable.** The calibration set had only the two cells [overlap 0 / answerhood 1] and [overlap ≥1 / answerhood 3],
so **overlap and true answerhood were completely confounded**. On such a set a judge that discriminates answerhood well
shows a large negative ρ; the ρ=−0.434 "exclusion" of `qwen3:8b` was evidence of **discrimination**, not contamination.

#### Round 2 (balanced 2×2 set + J1 anchor fix): `llama3.1:8b`, 80 items

| | no overlap | overlap | diff |
|---|---:|---:|---:|
| truth = answers (n=20/20) | 1.95 | 2.30 | **+0.35** |
| truth = does not answer (n=20/20) | 1.05 | 1.35 | **+0.30** |

*(answerhood score 3 = answers … 1 = does not answer)*

| metric | value | threshold | verdict |
|---|---:|---|---|
| discrimination AUC (four cells) | **0.748** | ≥.75 | narrow fail |
| **lexical contamination (mean within-stratum overlap effect)** | **+0.32** | ≤0.30 | **exceeded → exclude** |
| judge `overlap` rating vs machine | ρ=+0.17, direction agreement 52% | — | still noise |

**This time contamination is measured cleanly.** Holding true answerhood fixed and adding lexical overlap
raises the judge's answerhood score **by a consistent +0.3 in both strata**. That is the exact form of
"measuring vocabulary, not answerhood", invisible in the confounded ρ of round 1.

#### Isolating the effect of the prompt fix

On the same two cells (`pos` vs `neg_constructed`) alone: **0.573 → 0.657**.
Fixing one J1 anchor example alone moved the AUC by +0.08. The rest of the difference is the changed set, and
**the two numbers must not be compared directly**: prompt and data changed together.

> **Verdict: `llama3.1:8b` is excluded.** Discrimination misses narrowly, but contamination exceeds the threshold, and above all
> its direction is the very confound this design removes. Preregistered thresholds are not moved post hoc.


Details are in the internal review memo of 2026-08-20 (not released).

## 4. Aggregation and acceptance

### 4.1 Aggregation rules

| judgment type | aggregation |
|---|---|
| ordinal (J1 answerhood and overlap, J5 naturalness) | median per `judge_id` → median per model → overall median |
| nominal (J2 entailment, J3 asymmetry) | majority, except items in §4.2 that require unanimity |
| numeric (J4 P, C) | median of per-model medians; no means, since one judge's 100 would swamp the rest |
| free text (`*_why`, `demand_quote`) | not aggregated; read by a person |

### 4.2 Acceptance table (after the audit of 2026-08-20)

An item is accepted only if it **passes every row**; otherwise it goes to `items/rejected/` with a reason.

> **Fix the aggregation unit first.**
> Verdict per `judge_id` → **majority per model** → aggregation across models. "Unanimous" means **unanimous across models**,
> **not** across judge_ids. Requiring unanimity over 12 judge_ids × 2 blinding orders = 24 verdicts on
> 6 criteria would pass **0.3 of 24 items** at a 3% per-verdict error rate
> (internal audit A-3); at the model level the same error rate passes 13.9.

| row | condition | aggregation | change at audit |
|---|---|---|---|
| A0 | no `auto_checks.py` violation (`*w` warnings allowed) | machine | confirms-h2/inert-h3/prob-range demoted to warnings |
| ~~A1~~ | ~~J1: h3 answerhood = 1~~ | — | **retired**: tautological since $Q$ is derived from $h_3$ |
| ~~A2~~ | ~~J1: h2 answerhood = 3~~ | — | **retired**, same reason |
| ~~A3~~ | ~~J1: overlap rating~~ | — | retired (uncorrelated in the pilot) |
| **A4a** | J1R: recovered question `SAME`/`NESTED` relative to our $Q$ | majority of models | **redesigned**: separate call without exposing $Q$ |
| **A4b** | J1R: median `confidence` ≥ 3 | median | new |
| **A4c** | J1R: `bridge_knowledge` not `none` | majority of models | **new: proof that a world-knowledge bridge exists** |
| A5 | J2: `r_23`, `r_32`, `r_12`, `r_13` = INDEPENDENT | **unanimous across models** | unit changed |
| A6 | J2: `e_2` = `e_3` = NEITHER | majority of models | — |
| **A7** | J3: probability shift \|pp\| ≤ **13.9** (both h2 and h3) | median | **recalibrated to Chung's observed maximum** |
| A8 | J3: `new_target_info` empty | majority + human check | relaxed from unanimity |
| A9 | J3: no vocabulary in `asymmetry_route` | majority + human check | relaxed from unanimity |
| A10 | J3: no `demand_suspicion` = STRONG | majority of models | relaxed |
| **A18** | J3: `v3_bears_on_A/B` both false, no `v3_other_difference` | majority of models | **new: validates `ctx_irr`** |
| ~~A11~~ | ~~J4: C band~~ | — | **moved to measured norming** |
| ~~A12~~ | ~~J4: P band~~ | — | same |
| A13 | J5: naturalness ≥ 3 | median | — |
| ~~A14~~ | ~~J5: `banned_forms` empty~~ | — | **deleted**: comes from the RQ3 stimulus-bank specification; J5 records only |
| A15 | J5: `already_mentioned` = NEITHER | **unanimous across models** | unit changed |
| A16 | human sign-off: the authors confirm A5 and A15 directly | human | — |
| **A17** | set level: no $h_3$ predicate type exceeds half | machine (class-balance) | **new: predicate-type confound** |
| **A19** | `ov_word`/`fil_word` form a minimal pair | machine (minimal-pair) | **new** |

**What the LLM judges actually decide is A4a–c, A5, A6, A7–A10, A13, A15 and A18.**
The rest is machine (A0, A17, A19), measurement (former A11/A12) and human (A16).

#### Basis of the A7 threshold

10 pp was set without a basis and **rejects 3 of the 8 Chung stimulus cells**
(`data/chung2024/readable/norming.csv`: violin h3 +13.9, swiss-man h3 +12.4,
swedish-woman h2 +11.9). A threshold that rejects the seed paper's stimuli is not a threshold.

**Redefinition**: the $|\Delta P|$ of our items must not exceed the **observed maximum of 13.9 pp** among Chung's
normed stimuli. In addition, fitting Chung's model `lm(judgment ~ HYPOTHESIS * CONTEXT)` to the whole
item set must give **a non-significant CONTEXT main effect**
(Chung: probability $p=.361$, confirmation $p=.060$). Drawing the threshold from the seed data removes its arbitrariness.

### 4.3 Parse failures

If a judge's output does not parse as JSON it is **recorded and kept** with `parse_ok=false`.
The rule of proposal §5.1.4 applies unchanged.

- Never discard silently. Parse-failure rates are reported per judge and per item.
- A judge whose failure rate exceeds 20% is excluded, with results before and after exclusion both reported.
- Failures are not random; they cluster on hard items, which are exactly the items of interest.

## 5. Separating judge failure from item failure

The same failure signal has two possible causes. The rule for telling them apart is fixed in advance.

| signal | read as item failure when | read as judge failure when |
|---|---|---|
| A1/A2 fail | it happens **on some items only** | it happens on almost **every item** (calibrated judges included) |
| A11 fail | some items | `C_h3` never near 0 on any item: the phenomenon reported earlier |
| low inter-rater reliability | — | always the judge |

**If A11 fails on every item, fixing items is pointless.** Then the J4 gate is switched off,
$C$ is assigned by a person and confirmed only by measured norming.
This branch is written down in advance because saying "the judge was odd" after the measurements arrive
would be post-hoc rationalisation.

## 6. Reliability reporting

Regardless of acceptance, the following is reported for **every item and every criterion**.

- **Krippendorff's α**: ordinal distance for ordinal criteria, nominal for nominal ones; per criterion.
  - α ≥ .80 usable, .67 ≤ α < .80 tentative, α < .67 = **that criterion's verdicts are not used in the paper's claims**
- **Within- vs between-model** variance decomposition. If disagreement across seeds/prompt variants exceeds
  disagreement across models, the judgment itself is noise.
- **Position bias**: the rate of verdicts flipping between the `(A=h2)` and `(A=h3)` presentations.
- **Per-judge exclusion and reasons**, with acceptance rates before and after.
- **Acceptance rate**: the four-stage funnel generation → A0 → A1–A15 → human sign-off.
  These numbers replace the yield arithmetic of proposal §2.16 with measurements.

## 7. Preregistered items

Before item generation starts, the following are fixed and pinned by commit hash.

1. every threshold in the §4.2 acceptance table
2. the §1.2 judge pool
3. the §3 calibration pass criteria
4. the §6 α thresholds
5. the target of 24 items, 15 of them new
6. the branching rule of §5

**Every later change is labelled exploratory and reported as such.**
