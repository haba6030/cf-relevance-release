# P-RELEX: `ORG` slot rewriting prompt

Only the `ORG` slot is rewritten. The rest of the Chung carrier is fixed byte for byte, and so are
`e`, `h1`, `h2`, `h3`: editing a conjunct makes a different item and breaks comparability with earlier results.

**Two kinds of input.**

| input | what is done |
|---|---|
| **newly generated item** (`no-overlap`/`org-no-overlap` violation) | rewrite `ORG` to remove the lexical overlap |
| **carrier-departing item** (`carrier-intact` violation) | fold the purpose statement that spilled into a second sentence back into the `ORG` noun phrase, restoring the carrier |

> **Chung stimuli are not targets of this prompt.** `no-overlap`, `org-no-overlap`, `overlap-symmetric` and `length-match` belong to the `GEN_ONLY` tier and
> do not apply to Chung items. Replication targets are used exactly as published.

---

## SYSTEM

```
You repair one noun phrase inside a fixed template. Everything outside that noun
phrase is frozen and must be reproduced exactly, to the byte.

Output exactly one JSON object and nothing else.
```

## USER

````
# The frozen carrier -- reproduce every word outside the slots exactly

  {CONSULTANT} is a consultant doing research for {ORG}, trying to discover new
  target groups in Europe for the company to market to. She calls a randomly
  selected person, {TGT}, and starts asking {TGT} questions. She finds out that {E}

# Frozen fields -- do not alter, do not paraphrase, do not repunctuate

  CONSULTANT : {CONSULTANT}
  TGT        : {TGT}
  E          : {E}
  h1         : {H1}
  h2         : {H2}
  h3         : {H3}
  Q          : {Q}

# The slot to repair

  current ORG : {ORG}
  current full context (for reference only) :
  {CTX}

# What is wrong

  Content words shared by ORG and h3 : {OVERLAP_H3}
  Content words shared by ORG and h2 : {OVERLAP_H2}
  Carrier spans missing from the current context : {MISSING_SPANS}

  Shared vocabulary makes h3 cheaper to produce after the context for reasons of
  lexical activation rather than relevance, and the measurement then cannot tell the
  two apart. Taking a difference across contexts does not cancel it.

  A missing carrier span means the context was written freehand instead of filled
  into the template. Fold whatever it said into ORG.

# What the new ORG must satisfy

  no-shared-word  Shares NO content word with h2 or with h3. Function words, {CONSULTANT} and
      {TGT} are exempt.
  worth-knowing  Still makes h3 worth knowing: someone pursuing this organisation's purpose
      would want the answer to Q. The link runs through world knowledge, not through
      shared vocabulary.
  not-evidential  Does NOT make h3 more probable. Changing what is asked is allowed. Changing
      what is known about the target is not.
  h2-irrelevant  Leaves h2 beside the point for this organisation.
  single-np  Is a single noun phrase, 4-14 words, with its own article, no full stop inside.
      It may carry one relative clause. It may NOT become a second sentence.
  qud-unspoken  Does not contain Q verbatim.

# How the link is made without sharing words

  Name the SITUATION in which h3 matters, not the OBJECT that h3 names.

    h3 = "owns an umbrella"
      forbidden : "an umbrella company"
      allowed   : "a firm that surveys who copes with sudden downpours"

    h3 = "has a driving license"
      forbidden : "a car company", "a driving school"
      allowed   : "a chain that lets customers collect vehicles themselves"

    h3 = "is allergic to shellfish"
      forbidden : "a seafood wholesaler", "a shellfish restaurant chain"
      allowed   : "a caterer deciding which dishes are safe to hand round"

  A near miss, kept here because it is instructive. For h3 = "has brown hair" we once
  wrote ORG = "a brand choosing which of two dye tones to stock". It passes no-shared-word, but it
  fails worth-knowing: what someone buys dye for tells you what colour they WANT, not what colour
  they HAVE. The bridge has to point at the state h3 asserts, not at an adjacent one.

# Output

{
  "org": "<the repaired noun phrase>",
  "org_irr": "<the same slot filled with a purpose unrelated to BOTH h2 and h3;
              similar length; used as the RQ2b irrelevant-QUD control>",
  "bridge": "<one sentence: what piece of world knowledge links ORG to h3>",
  "cost": "<one sentence: what got less natural than the original slot, or 'none'>"
}
````

---

## Operating notes

- The model does not produce `ctx`; `sim/chung_frame.py:build()` assembles it from `org`,
  so the carrier can never be damaged.
- Run `auto_checks.py` **again** after assembly; a new slot often creates new overlap.
- `bridge` is never shown to judge J1; it would give the answer away.
- Every `cost` is collected and reported. Keeping lexical separation makes the `ORG` slot long and bureaucratic;
  that is **the real price of this design** and must not be hidden.
- If no-shared-word still fails after three attempts, **discard the item.** Some $h_3$ cannot be relexicalised
  (proposal §2.14.5), and finding that out early is a benefit of the pipeline.
