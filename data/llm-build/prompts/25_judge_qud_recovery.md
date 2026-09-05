# J1R: QUD recovery (reconstruct the question from the context alone)

**Split off from J1 after the audit** (internal audit A-1, A-2).

## Why it was split off

The old J1 showed $Q$ at the start and then asked at the end to "recover the question from the context alone".
**Exposure cannot be undone by an instruction to ignore what was just read.** That item measured compliance,
not recovery.

And since $Q$ is **derived mechanically** from $h_3$ (proposal §2.12.2), $Q$ and $h_3$ share vocabulary
by definition. In all 8 generated items $Q \cap h_3$ is 2–3 words and
$Q \cap h_2$ is empty, so **showing $Q$ lets string matching solve the answerhood judgment.**

> **This prompt shows neither $Q$ nor $h_2$ nor $h_3$.**
> It gives the context paragraph alone. That is the one point in this design that actually needs validation:
> **does the `ORG` slot actually raise the intended question?**

---

## SYSTEM

```
You read a short paragraph and say what question it puts on the table.

A paragraph puts a question on the table when someone pursuing the concerns described
in it would need that question answered in order to get on with what they are doing.
You are not asked whether the paragraph is interesting or well written.

Output exactly one JSON object and nothing else.
```

## USER

````
# Paragraph

{CTX}

# What to do

Q1  Write the single question this paragraph most puts on the table about the person
    it names. Write it as a yes/no question about that person.

Q2  Write two more questions it puts on the table, in descending order of how strongly
    the paragraph raises them.

Q3  How confident are you that Q1 is the question the writer of the paragraph intended
    to raise? 1 = guessing, 5 = certain.

Q4  Name the piece of ordinary world knowledge a reader needs in order to get from what
    the paragraph says to your Q1. If none is needed -- if the paragraph simply states
    the question in other words -- say "none".

# Output

{
  "q1": "<a yes/no question about the named person>",
  "q2": "<second question>",
  "q3_third": "<third question>",
  "confidence": 1|2|3|4|5,
  "bridge_knowledge": "<one sentence, or 'none'>"
}
````

---

## Scoring, by a separate judge

Whether the recovered question denotes the same partition as our $Q$ is asked **in a separate call**.
A judge scoring its own recovery is not scoring.

### Matching prompt (SYSTEM)

```
You compare two yes/no questions and say whether they divide the possibilities the
same way. Two questions match when knowing the answer to one always tells you the
answer to the other. Different wording does not matter.

Output exactly one JSON object and nothing else.
```

### Matching prompt (USER)

````
  Question 1: {Q}
  Question 2: {RECOVERED_Q1}

Do these two questions divide the possibilities the same way?

  SAME       knowing the answer to one settles the other
  NESTED     one is a special case of the other
  RELATED    same topic, different partitions
  UNRELATED  neither bears on the other

{
  "relation": "SAME"|"NESTED"|"RELATED"|"UNRELATED",
  "why": "<at most 20 words>"
}
````

---

## Pass conditions (decision table A4)

| | condition | aggregation |
|---|---|---|
| A4a | recovered question is `SAME` as or `NESTED` in our $Q$ | majority of models |
| A4b | median `confidence` ≥ 3 | median |
| A4c | `bridge_knowledge` is **not** `none` | majority of models |

> **A4c is the heart of this judgment.** If `bridge_knowledge` is `none`, the context raised the question
> **lexically**, meaning no-overlap was passed at the string level only.
> Our design must link the context to $h_3$ **through world knowledge**:
> not `an umbrella company` but `a firm that surveys who copes with sudden downpours`.
> The bridge exists only if the judge can put it into words.

## What this judgment replaces

The old A1 (h3 answerhood=1) and A2 (h2 answerhood=3) are **retired.** Because $Q$ was built from $h_3$,
that $h_3$ answers $Q$ is a consequence of how we wrote it, not a fact the judge confirmed.
`20_judge_answerhood.md` is kept **for judge calibration only**.
