# Item specification (v1): conjunction-fallacy items with an attached context

This document is **the sole basis of the generation and judge prompts**. To change a prompt, change this first.
This document and all of `prompts/` are released with the paper.

Basis: proposal §0.4 (construct D1), §2.12 (mechanical generation), §2.13 (norming), §5.1 (measurement),
and the internal data-status notebook §4-A′, §5-B (noise diagnostics).

---

## 1. Inheriting the Chung carrier: this is not free writing

RQ1 stimuli must be **based on** the Chung et al. (2024) paradigm. New items **inherit** the task structure
and string carrier of the Chung items; only the slot contents are new.

### 1.1 The two task structures, verified from the raw data

Read directly from Chung's published data files; this is a reading, not a guess.

| | Exp1 | Exp2 |
|---|---|---|
| format | **two-option forced choice** | **three-way comparison (ranking)** |
| options | `h1` vs `h1 ∧ h_alt` | `base(h1)` · `h1 ∧ h2` · `h1 ∧ h3` |
| $h_2$/$h_3$ | manipulated **between** items (`alternative_hypothesis` column) | **within** item, all three at once |
| context | `context ∈ {true,false}` | same |
| raw data | `exp1.csv` (2,550 rows), `sim/chung_items.json` | `exp2.csv` (8,386 rows), `trial_type=survey-sort` |
| response object | `"Dan is <b>an expert mountaineer</b> and <b>owns an umbrella</b>."` | a permutation of `['base','h2','h3']` |

`options()` in `sim/scenarios9.py` holds the Exp2 three-way comparison and `sim/chung_items.json` the Exp1
two-option choice. **New items must be usable in both structures**: all of
`h1`, `h2`, `h3` must be present; none may be omitted.

### 1.2 The context carrier, fixed byte for byte

The four contexts in Chung's appendix **differ in four slots only and are otherwise identical.** Verified by
`sim/chung_frame.py` (`python3 sim/chung_frame.py` → all four PASS).

```
{CONSULTANT} is a consultant doing research for {ORG}, trying to discover new target
groups in Europe for the company to market to. She calls a randomly selected person,
{TGT}, and starts asking {TGT} questions. She finds out that {E}
```

| scenario | CONSULTANT | ORG |
|---|---|---|
| violin | Adina | `an umbrella company` |
| swiss-man | Natalie | `a car company` |
| italian-undergrad | Maria | `an American travel agency` |
| swedish-woman | Becky | `a shampoo company` |

> **The `ctx` of a new item is built by filling the slots of this carrier.** It passes through `sim/chung_frame.py:build()`,
> and the **carrier-intact** check of `auto_checks.py` verifies the four fixed spans as strings.
> A freely written paragraph does not pass.

Fixing the carrier buys three things.
1. **Comparability.** A different carrier is no longer "the same manipulation" as Chung's human data.
2. **Less noise.** Context length, syntax and speaker role become **constants** across items and leave the between-item variance.
   (The earlier G9 "ctx 45–65 words" is superseded by the carrier and retired.)
3. **Fewer degrees of freedom.** The generator can touch only the `ORG` slot, so a lexical-overlap (no-overlap)
   violation can arise in **one place** only.

### 1.3 Writing a new item is therefore an `ORG`-slot design problem

Chung filled `ORG` by **naming the very object $h_3$ refers to**
(`an umbrella company` ↔ `owns an umbrella`). This is the source of no-overlap violations; 2 of the 4 overlap.

We fill it by **describing a situation in which $h_3$ matters**.

```
h3  : owns an umbrella
Chung: "an umbrella company"                          <- linked lexically; violates no-overlap
ours : "a firm that surveys who copes with sudden downpours"   <- linked by knowledge; passes no-overlap
```

The carrier is unchanged and only one noun phrase differs. Being a **minimal pair**, no length or syntax confound enters.

### 1.4 Which "phrases" are inherited: summary

| inheritance | what |
|---|---|
| **verbatim** | the four fixed spans of the context carrier (§1.2); the option-conjunction format `"{h1} and {h_i}."` |
| **form only** | `h1` = reference-class setter, `h2` = confirmed conjunct, `h3` = relevance conjunct, `e` = evidence sentence |
| **newly written** | the `ORG` slot, the `CONSULTANT` and `TGT` names, the content of `e`, `h1`, `h2`, `h3` |

### 1.5 Shape of an item

The Tentori, Crupi & Russo (2013) M-A paradigm with the context of Chung et al. (2024) attached.
It must be **field-compatible** with the structure in `sim/scenarios9.py`.

```
e    evidence sentence        "Dan has a degree in violin performance."
h1   reference-class setter   "Dan is an expert mountaineer."
h2   confirmed conjunct       "gives music lessons"      <- confirmed by e but with a low posterior
h3   relevance conjunct       "owns an umbrella"          <- inert with respect to e but answers the QUD
ctx  relevance context        the carrier (§1.2) with the ORG slot filled
```

The options are `base = h1`, `option2 = h1 ∧ h2`, `option3 = h1 ∧ h3` (Exp2 three-way comparison).
In the Exp1 arm only `base` and `option_i` are shown and $i$ is manipulated between items.

**An item therefore needs exactly one asymmetry:**

> without context $h_2$ must win; with context $h_3$ must win.
> The former is produced by **confirmation**, the latter by **answerhood**. The two forces must attach to different conjuncts.

An item without this asymmetry contributes zero to the DV however natural it is, and only adds variance.
This is the leading candidate for why one model split 5/4 at `n=9`.

### 1.6 The DV: pin down Chung's estimand first

Reading Chung's analysis script (lines 191–225) literally, the Exp2 DV is this.

```r
conj_error_h2 <- match("base", judgment) > match("h2", judgment)
conj_error_h3 <- match("base", judgment) > match("h3", judgment)
glmer(conj_error ~ 1 + context_coded * conj_error_hyp + (1 | subjectid), family="binomial")
```

That is, **each conjunction is compared with `base`** to give a binary fallacy indicator, and the `context × hypothesis`
**interaction** is examined. The human target of +4.9 pp is that interaction.

> **Note: this does not compare $h_2$ and $h_3$ directly.** `base` is the reference point.
> Item design must target this estimand, which is why `h1` (= base) is **a slot that cannot be dropped**.

## 2. Slot definitions

| field | type | description |
|---|---|---|
| `scenario_id` | str | kebab-case, globally unique |
| `domain` | str | topic area; used for fold assignment and duplicate control |
| `paradigm` | `"MA"` | this specification covers M-A only (A-B with `e=null` in v2) |
| `tgt` | str | name of the target person |
| `e` | str | evidence sentence; a single finite clause beginning with `tgt` |
| `h1` | str | reference-class setter; **a group-restricting statement, not an everyday hobby** (§2.5) |
| `h2` | str | verb phrase only; attaches to the subject of `h1`; no period |
| `h3` | str | verb phrase only; same |
| `Q` | str | the QUD the context raises, as an explicit question; **never appears in `ctx`** |
| `ctx` | str | relevance context paragraph; **built by filling the carrier (§1.2)** |
| `consultant` | str | the `CONSULTANT` slot of the carrier |
| `org` | str | the `ORG` slot of the carrier; **the real free variable of new-item design** |
| `ctx_irr` | str | irrelevant-QUD control (RQ2b); same length and structure |
| `ov_word` | str | content word of `h3` inserted in the RQ2b lexical-overlap arm |
| `fil_word` | str | its minimal-pair filler (same part of speech, syllable count, frequency band) |
| `P_h2`,`P_h3` | float | target posterior $P(h_i\mid e)$ |
| `C_h2`,`C_h3` | float | target confirmation $C(h_i,e)$ on Tentori's $[-10,10]$ scale |
| `notes` | str | rationale recorded at generation (never shown to judges) |

## 3. Hard constraints: violation discards the item

Machine-checkable constraints are decided by `checks/auto_checks.py`; the rest fall to the judges (J1–J5).

### Scope of the rules: two tiers

The checker is **a tool for building items consistent with prior work**, not for auditing prior work.
Hence two tiers.

| tier | applies to | basis |
|---|---|---|
| **GATE** | every item (Chung included) | only rules the Chung stimuli **actually satisfy 4/4**; the basis is the practice of the prior work itself |
| **GEN_ONLY** | **newly written items only** | Chung stimuli are replication targets and out of scope; the basis is **measurement properties observed** on the Chung stimuli |

GEN_ONLY is not an exemption. An exemption says "the rule is right but this item is excused"; here the
rule **does not apply in the first place**, because editing the Chung stimuli would end the replication.
Violations are still printed, tagged `[out of scope, reference]`.

### G rules: machine checks (`checks/auto_checks.py`)

**GATE: every item**

| id | constraint | basis | Chung 4 |
|---|---|---|---|
| **carrier-intact** | `ctx` contains all four fixed spans of the Chung carrier (`sim/chung_frame.py`) | §1.2, **carrier inheritance** | 4/4 |
| **org-slot** | the `ORG` slot is recoverable and matches the `org` field | carrier integrity | 4/4 |
| **no-complement** | `h3` is not the negation (complement) of `h2` | complements are two cells of one question, so no answerhood contrast is possible | 4/4 |
| **evidence-present** | `paradigm=MA` implies evidence `e` is present | information match of the no-context condition | 4/4 |
| **unique-item** | the `(h2,h3)` pair, `h2`, `h3`, `e`, `h1` are unique across the item set | swiss-man/swiss-person duplicate | 4/4 |
| **qud-implicit** | the string `Q` does not appear verbatim in `ctx` | the context must raise the QUD **implicitly** | 4/4 |
| **control-length** | token count of `ctx_irr` within ±10% of `ctx` | RQ2b minimal pair | n/a |
| **no-name-leak** | the `tgt` name is diagnostic of neither `h2` nor `h3` | the `surgeon`/`is male` leak | 4/4 |
| **domain-cap** | at most 3 items per `domain` | topic clustering | set level |

**GEN_ONLY: newly generated items only**

| id | constraint | basis |
|---|---|---|
| **no-overlap** | content words of `ctx`−`e` ∩ `h2`/`h3` = ∅ | lexical priming does not cancel in the difference |
| **org-no-overlap** | `ORG` content words do not overlap `h2`/`h3` | the only place no-overlap can arise |
| **overlap-symmetric** | `ctx`∩`h2` and `ctx`∩`h3` have equal overlap counts | symmetry |
| **length-match** | `h2`/`h3` token counts differ by ≤ 1, each ≥ 3 tokens, ratio ≤ 1.34 | $1/n$ scale asymmetry of the normalised DV |

> **no-overlap is the most important line of this specification.** If the context pre-mentions a content word of $h_3$,
> the conditional surprisal of $h_3$ drops for that reason alone, and the difference does not cancel it.
>
> **For newly written items no-overlap is a hard gate.**
> Chung's originals are replication targets, out of scope, and are not edited.
>
> To make $h_3$ relevant while respecting no-overlap, a **world-knowledge bridge** is needed.
> `umbrella company` ↔ `owns an umbrella` links lexically (violation); `a firm that surveys who
> is well prepared for sudden rain` ↔ `owns an umbrella` links by knowledge (pass).
> This rewriting is the most expensive step of generation and is handled by `prompts/11_relexicalize.md`.

### J rules: judge checks (LLM-as-a-judge plus human sign-off)

| id | constraint | judge |
|---|---|---|
| **J1a** | `h3` answers `Q` (eliminates at least one live cell) | answerhood |
| **J1b** | `h2` does **not** answer `Q` | answerhood |
| **J1c** | the QUD a reader of `ctx` would raise is `Q` (recovery match) | answerhood |
| **J2a** | `h2` and `h3` are logically independent (neither entails or excludes the other) | independence |
| **J2b** | `h1` entails neither `h2` nor `h3` | independence |
| **J2c** | `e` neither confirms nor disconfirms `h3` | independence |
| **J3a** | `ctx` gives **no evidence about the probability** of `h2` or `h3` (context invariance) | confound |
| **J3b** | `ctx` gives no information about the target beyond `e` | confound |
| **J3c** | `ctx_irr` is irrelevant to both `h2` and `h3` | confound |
| **J4** | $P(h_2\mid e) < P(h_3\mid e)$ with a gap ≥ .15; $C(h_2,e)$ clearly positive, $|C(h_3,e)| \le 1$ | norming |
| **J5a** | `ctx` reads as natural discourse, not as an instruction | naturalness |
| **J5b** | the speaker of `ctx` is cooperative (do not copy maxim-violating materials) | naturalness |

> **J3a is the premise of RQ2a.** If the context changes probabilities, the claim "relevance is independent of
> probability" cannot stand. The premise failure reported earlier may originate here.

## 4. Target norms: why these values

The actual distribution of the seven Tentori human norms (`HUMAN_NORMS` in `sim/scenarios9.py`):

| | $P(h_2)$ | $P(h_3)$ | $C(h_2)$ | $C(h_3)$ |
|---|---|---|---|---|
| range | .16–.77 | .12–.83 | 2.4–5.6 | −4.1–0.2 |
| structure | | | **all positive** | **all ≈0 or negative** |

The ordering of $P$ is not constant (`italian-undergrad` has $P_{h2}>P_{h3}$). **What is invariant is $C$:**
$h_2$ is confirmed and $h_3$ is not. Hence the **hard conditions concern $C$** and
$P$ is a preference.

```
hard        C(h2,e) ≥ +2.0      C(h3,e) ∈ [−1, +1]
preference  P(h3|e) − P(h2|e) ≥ .15
forbidden   P(h2|e) or P(h3|e) below .05 or above .95 (floor and ceiling effects)
```

## 5. How many items

Currently 9; target **24** (existing 9 + new 15).

The basis is power. At $n=9$ the smallest one-sided sign-test $p$ is $2^{-9}=.002$, and Holm over 5 models
makes $.002\times5=.01$ the best case: **only a perfect 9/9 survives.**
In fact only `Qwen3-4B` reached 9/0; every other model fell after Holm.
At $n=24$ the sign-test floor is $6\times10^{-8}$, and a realistic 18/24 consistency
remains significant after correction.

Of the 24, **9 are rewrites of existing items to the no-overlap standard**. Rewritten items and originals are
not counted as the same item, and results on the originals are reported as robustness only.

## 6. What this specification does not guarantee

1. **No human norms.** The 15 new items do not inherit Tentori's norms. Target $P$ and $C$ are values
   declared at generation, and only model norming confirms where they land.
   → Stated as a limitation; small-scale human norming to be added where possible.
2. **Judge norming (J4) is circular.** Items built by asking a model for $P$ and $C$ cannot validate the
   model's $P$/$C$ structure (§2.13.4). J4 is **a screening gate, not a measurement.**
   Reported norming values must be the independent measured ones.
3. **Complementarity cannot be decided by machine** (§2.12.3). Whether two arbitrary noun phrases are complements
   over $CS$ is not determined by the string. J2 checks this only in a **relaxed form**,
   and the final sign-off is human.
4. **Respecting no-overlap lowers naturalness.** A world-knowledge bridge is always less smooth than a lexical one.
   J5 measures that cost, and the naturalness scores of passing items are reported.
