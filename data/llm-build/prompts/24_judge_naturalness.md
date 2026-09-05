# J5: naturalness and format compliance

Respecting no-overlap (lexical separation) stiffens the context. **Measuring and reporting** that cost is
this judge's purpose; hiding it would put an unverified "our stimuli are natural" into the paper.

Formal properties (part-of-speech frame match, banned forms) are **recorded** as well; recorded, not gated,
because those criteria come from the RQ3 stimulus-bank specification and are not gates for RQ1 items.
(`AUDIT_criteria.md` §G).

---

## SYSTEM

```
You rate experimental materials for readability and form. You are a copy editor, not
a participant: do not try to answer the item.

Output exactly one JSON object and nothing else.
```

## USER

````
# Paragraph

{CTX}

# Two phrases that will be appended to a sentence about the same person

  A. {STATEMENT_A_VP}
  B. {STATEMENT_B_VP}

# Full sentence they will appear in

  base    : {H1_FULL}
  with A  : {H1_STEM} and {STATEMENT_A_VP}.
  with B  : {H1_STEM} and {STATEMENT_B_VP}.

# Rate

Q1  Does the paragraph read like ordinary English prose someone might actually
    write? 1 = stilted or machine-made, 5 = entirely natural.

Q2  Do "with A" and "with B" read as equally ordinary sentences? If one is more
    awkward than the other, say which. Unequal awkwardness is a confound: the
    smoother sentence will be preferred for reasons unrelated to the manipulation.

Q3  Do A and B have the same grammatical shape -- same kind of verb, same number
    of things after it?

Q4  Does either A or B contain: a modal verb; a vague quantifier (most, many, few,
    several, often, usually); progressive aspect; a negation; a comma; the word
    "and" or "or"; a gradable adjective with no number attached?

Q5  Is either A or B about something the paragraph already told you?

# Output

{
  "ctx_naturalness": 1|2|3|4|5,
  "ctx_stilted_phrase": "<the least natural phrase, or empty string>",
  "sentence_balance": "EQUAL"|"A_SMOOTHER"|"B_SMOOTHER",
  "same_grammatical_shape": true|false,
  "shape_difference": "<what differs, or empty string>",
  "banned_forms": ["<form found, with the phrase>", ...],
  "already_mentioned": "NEITHER"|"A"|"B"|"BOTH"
}
````

---

## Pass conditions

| | condition |
|---|---|
| J5a | median `ctx_naturalness` ≥ **3** |
| J5b | `sentence_balance` = `EQUAL` by majority |
| ~~J5c~~ | ~~`same_grammatical_shape`~~ → **recorded only** |
| ~~J5d~~ | ~~`banned_forms` empty~~ → **recorded only** |
| J5e | `already_mentioned` = `NEITHER`, **unanimous** |

> **The threshold of 3 for J5a is a compromise.** Requiring 5 would leave almost no item that respects
> no-overlap. The `ctx_naturalness` distribution of passing items is reported as is, beside the same
> score for Chung's four originals; if our stimuli are less natural than the originals, the number
> should say so.
>
> **J5e is unanimous because it is the last line of defence for no-overlap.** The machine check sees string
> matches only and misses synonym leakage such as `umbrella` ↔ `rain gear`. Among the non-human judges,
> this is the only place that can catch it.
