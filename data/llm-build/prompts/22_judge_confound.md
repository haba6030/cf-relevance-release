# J3: confound audit (context invariance, leakage, asymmetry, irrelevant control)

> **`ctx_irr` was added in the audit of 2026-08-20.** The irrelevant-QUD control arm of RQ2b
> appeared in no prompt and would have run **unvalidated**
> (internal audit C-1). Q6 covers it.

**This judge protects the premise of RQ2a.** If the relevance context changes probabilities, the claim
"relevance is independent of probability" cannot stand. Among the candidate causes of the premise failure
reported earlier (the models' $P\cdot C$ structure disagreeing with human norms), **the ones fixable at the item level** are here.

The context-invariance norming of proposal §2.13.2(4) recast as a judgment.

---

## SYSTEM

```
You audit an experimental stimulus for leakage. Your job is to find reasons the
stimulus might produce the intended result for an unintended cause.

Be adversarial. A stimulus that survives you is one that will survive a reviewer.
Do not soften a finding because the item is otherwise well made.

Output exactly one JSON object and nothing else.
```

## USER

````
# Version 1 -- the participant sees only this

{E}

# Version 2 -- the participant sees only this

{CTX}

# Version 3 -- a control version, seen by a different participant

{CTX_IRR}

# Two statements that will be evaluated under each version

  A. {STATEMENT_A}
  B. {STATEMENT_B}

# Audit

Q1  PROBABILITY SHIFT. Suppose you had to say how likely A is. Would your number be
    different after reading Version 2 than after reading Version 1? Same for B.
    Report the direction and the size in percentage points, as your best estimate.
    Zero is a legitimate answer and is what the design wants.

Q2  TARGET INFORMATION. Does Version 2 tell you anything about the person that
    Version 1 does not? List it. The only fact about the person that may appear in
    Version 2 is the one in Version 1.

Q3  ASYMMETRY. Does Version 2 draw attention to A more than to B, or to B more than
    to A, by any route OTHER than raising a question that one of them settles?
    Routes to check: shared vocabulary, shared topic, word order, recency,
    emotional charge, concreteness.

Q4  DEMAND. Would a participant reading Version 2 suspect that the experiment wants
    a particular answer? Quote the phrase that gives it away, if any.

Q5  SPEAKER. Is the person described in Version 2 behaving cooperatively and
    ordinarily, or evasively, oddly, or with a hidden motive?

Q6  TWO AXES. Version 1 gives a fact about the person. Version 2 describes an
    organisation with a purpose.
    (a) In one short phrase, what subject matter connects the fact in Version 1 to
        statement A? (If nothing connects them, say "nothing".)
    (b) In one short phrase, what subject matter connects the organisation's purpose
        in Version 2 to statement B? (If nothing connects them, say "nothing".)
    (c) Are (a) and (b) the same subject matter?
    They must NOT be. If confirmation and relevance run through the same topic, the
    item cannot separate them.

Q7  CONTROL VERSION. Version 3 is meant to be a paragraph that raises a question
    bearing on NEITHER A nor B. Check it.
    (a) Does Version 3 raise a question that A would help answer?
    (b) Does Version 3 raise a question that B would help answer?
    (c) Does Version 3 differ from Version 2 in anything other than the
        organisation's purpose? Length, tone, and what is said about the person
        should all be the same.

# Output

{
  "A_prob_shift_pp": <number, negative means Version 2 lowers it>,
  "B_prob_shift_pp": <number>,
  "prob_shift_reason": "<at most 25 words, or empty string if both are 0>",
  "new_target_info": ["<fact>", ...],
  "asymmetry": "NONE"|"FAVOURS_A"|"FAVOURS_B",
  "asymmetry_route": "<which route, or empty string>",
  "demand_suspicion": "NONE"|"MILD"|"STRONG",
  "demand_quote": "<quoted phrase, or empty string>",
  "speaker": "COOPERATIVE"|"ODD"|"EVASIVE",
  "axis_a": "<subject matter linking Version 1 to A, or 'nothing'>",
  "axis_b": "<subject matter linking Version 2 to B, or 'nothing'>",
  "axes_same": true|false,
  "v3_bears_on_A": true|false,
  "v3_bears_on_B": true|false,
  "v3_other_difference": "<what else differs, or empty string>"
}
````

---

## Pass conditions

| | condition |
|---|---|
| J3a | judge medians of `|A_prob_shift_pp|` and `|B_prob_shift_pp|` **both ≤ 10** |
| J3b | `new_target_info` empty for every judge |
| J3c | `asymmetry` = `NONE` by majority, and **no** judge gives `vocabulary` as `asymmetry_route` |
| J3d | no judge gives `demand_suspicion` = `STRONG` |
| J3e | `speaker` = `COOPERATIVE` by majority |
| J3h | `axes_same` = `false` (**unanimous**): the confirmation axis and the relevance axis must be distinct |
| J3f | `v3_bears_on_A` = `v3_bears_on_B` = `false` (majority) |
| J3g | `v3_other_difference` empty (majority) |

> **J3a is an equivalence claim, so its threshold must be fixed in advance.** 10 pp is conservative
> relative to the spread of Tentori's $P$ norms (.16–.83); it was set before the pilot
> and not changed afterwards. Relaxing it post hoc would turn an equivalence test into wishful thinking.
>
> **The `vocabulary` clause of J3c overlaps with no-overlap and J1d, a triple check by design.** Lexical leakage
is the one fatal confound in this design; the machine check, the answerhood judge and the confound judge
> must each catch it independently.
