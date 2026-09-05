# J2: logical independence and entailment

> **Blinding.** $h_2$/$h_3$ are randomly assigned to S2/S3 (`blind_map` in `run_judges.py`).
> Before the audit `H2_FULL`/`H3_FULL` were inserted in fixed order, so **J2 alone was unblinded**
> (`AUDIT_criteria.md` B-4).

A conjunction-fallacy item requires $h_1$, $h_2$, $h_3$ to be **logically independent**.
If one entails another, $P(h_1\wedge h_i)$ equals $P(h_1)$ or 0 and the contrast collapses.

Proposal §2.12.6 lists this among **the three properties a machine cannot check** (the others being
the cell window and complementarity), hence the two-layer scheme of judge majority plus human sign-off.

---

## SYSTEM

```
You judge logical relations between statements. Judge only what follows as a matter
of meaning, not what is likely, typical, or usually true of people.

"X is a surgeon" does not entail "X is male", however the statistics run. It is not
even close: entailment means no situation makes the first true and the second false.

Output exactly one JSON object and nothing else.
```

## USER

````
# Statements about one person

  S1. {H1_FULL}
  S2. {STATEMENT_A_FULL}
  S3. {STATEMENT_B_FULL}

# Background fact given about the same person

  E.  {E}

# Decide each relation

For each ordered pair below, choose one label:

  ENTAILS       the first cannot be true while the second is false
  EXCLUDES      the two cannot both be true
  INDEPENDENT   all four combinations of true/false are possible

  r_12 : S1 to S2
  r_13 : S1 to S3
  r_23 : S2 to S3
  r_32 : S3 to S2

(S2 and S3 are presented in a random order. Do not assume either is the "main" one.)

Then, for the background fact E:

  e_2 : does E make S2 CERTAIN, IMPOSSIBLE, or NEITHER?
  e_3 : does E make S3 CERTAIN, IMPOSSIBLE, or NEITHER?

Then say whether S2 and S3 are the two answers to one and the same question -- that is,
whether one is simply the denial of the other, however worded. This is a defect: it means
no context can bear on one without bearing on the other.

Then one more thing. The item assumes that a question of the form

  {Q}

has exactly two ways of being settled, and that they exhaust the possibilities.
Say whether that is so, and if not, name a case that falls outside both.

# Output

{
  "r_12": "ENTAILS"|"EXCLUDES"|"INDEPENDENT",
  "r_13": "ENTAILS"|"EXCLUDES"|"INDEPENDENT",
  "r_23": "ENTAILS"|"EXCLUDES"|"INDEPENDENT",
  "r_32": "ENTAILS"|"EXCLUDES"|"INDEPENDENT",
  "e_2":  "CERTAIN"|"IMPOSSIBLE"|"NEITHER",
  "e_3":  "CERTAIN"|"IMPOSSIBLE"|"NEITHER",
  "s2_s3_are_complements": true|false,
  "question_exhaustive": true|false,
  "uncovered_case": "<a case outside both options, or empty string>"
}
````

---

## Pass conditions

| | condition |
|---|---|
| J2a | `r_23` = `r_32` = `INDEPENDENT`, **unanimous** |
| J2e | `s2_s3_are_complements` = `false`, **unanimous** (double-checked against machine rule G18) |
| J2b | `r_12` = `r_13` = `INDEPENDENT`, unanimous |
| J2c | `e_2` = `e_3` = `NEITHER` |
| J2d | `question_exhaustive` = `true` by majority |

> **J2a and J2b require unanimity, not a majority.** Entailment is not a matter on which judgments should split;
> if they split, the sentence is ambiguous, and ambiguous sentences are not used as stimuli.
>
> **J2d is for reference.** As §2.12.3 notes, whether two noun phrases are complements is decided neither by
> the string nor reliably by asking a model. A `false` is reviewed by a person; a `true` does not
> establish complementarity.
