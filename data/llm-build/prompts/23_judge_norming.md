# J4: prior norm estimate (a screening gate, not a measurement)

A **pre-screen** for whether $P(h_i\mid e)$ and $C(h_i,e)$ can plausibly land in the target bands of
`00_item_spec.md` §4. Actual norming is done by measured model norming,
with human norming added where possible.

> **These judgments must not be reported as the paper's norming values.** Validating the models' $P\cdot C$
> structure with items built by asking a model is circular (§2.13.4). J4 only discards hopeless items
> before the expensive measurement and claims nothing more.

Follows the elicitation format of Tentori (2013). **The call is split in four.**

| call | asks | hypothesis shown |
|---|---|---|
| J4P-A | posterior probability | A only |
| J4P-B | posterior probability | B only |
| J4C-A | confirmation | A only |
| J4C-B | confirmation | B only |

**Why four.** Before the audit A and B were placed side by side in one call and both answered
(internal audit B-3), a procedure both prior studies explicitly avoided.

- §2.13.1: the $\pi$ block and the $\rho$ block must be separated to avoid anchoring carry-over.
- Tentori (2013): respondents were split so that each saw $h_2$ **or** $h_3$ for a scenario, never both.
- Chung et al. (2024) p.9: *"For each scenario, the participants were asked about
  either h2 or h3."*

The cost doubles but procedural correspondence is restored. `{HYP_VP}` / `{HYP_FULL}` in the two blocks below
hold **one hypothesis only**.

---

## Call 1: posterior probability $P$

### SYSTEM

```
You estimate how likely a description is to hold of a person, given what is known
about them. Answer with numbers. Do not explain unless asked.

Output exactly one JSON object and nothing else.
```

### USER

````
Imagine 100 people picked at random, all of whom satisfy the following:

  {H1_FULL}
  {E}

Of those 100 people, how many would you expect to satisfy the following?

  {HYP_VP}

Answer with one integer between 0 and 100.

{
  "out_of_100": <int>
}
````

---

## Call 2: confirmation $C$

### SYSTEM

```
You judge whether a piece of news makes a hypothesis more or less credible. You are
not judging whether the hypothesis is likely overall; you are judging the change.

Output exactly one JSON object and nothing else.
```

### USER

````
You know this about a person:

  {H1_FULL}

Consider one hypothesis about that same person:

  {HYP_FULL}

You now learn one new fact:

  {E}

Say how much that new fact changes the credibility of that hypothesis, on a scale
from -10 to +10.

  +10  the new fact makes it far more credible
    0  the new fact makes no difference either way
  -10  the new fact makes it far less credible

Zero is a normal answer. Most facts are irrelevant to most hypotheses.

{
  "confirmation": <number in [-10, 10]>,
  "why": "<at most 15 words>"
}
````

---

## Pass conditions

After unblinding, by judge median:

| | condition | basis |
|---|---|---|
| J4a | $C(h_2,e) \ge +2.0$ | all 7 Tentori norms have $C_{h2}\in[2.4,5.6]$ |
| J4b | $C(h_3,e) \in [-1,+1]$ | 6 of the 7 norms lie in $[-0.6,0.2]$ |
| J4c | $P(h_2\mid e),\,P(h_3\mid e) \in [.05,.95]$ | avoid floor and ceiling |
| J4d *(preference)* | $P(h_3\mid e)-P(h_2\mid e) \ge .15$ | not hard: 2 of the 7 norms have the opposite sign |

> **J4a/J4b are hard and J4d a preference** because what is invariant in the human norms is $C$, not $P$
> (`00_item_spec.md` §4 table). Screening on the ordering of $P$ would reject legitimate items such as
> Tentori's own `italian-undergrad`.

## Signs of judge failure

If many judges cannot place $C(h_3,e)$ near 0, the problem may not be the item but **a model that does
not separate confirmation from posterior probability**, the very phenomenon reported earlier.
In that case the J4 gate is powerless; a person assigns $C$ and only measurement confirms it.
**Judge failure and item failure are recorded separately** (`protocol/crosscheck_protocol.md` §5).
