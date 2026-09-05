# J1: answerhood judgment, **used for judge calibration only**

> **Since the audit of 2026-08-20 this prompt is not used to judge items.**
> Because $Q$ is derived mechanically from $h_3$, $Q\cap h_3 \ne \emptyset$ and $Q\cap h_2=\emptyset$ hold
> in 8/8 items, so **string matching solves it.** A1 and A2 of the acceptance table were retired
> and replaced by `25_judge_qud_recovery.md`. This file remains in use only for the judge
> qualification check of `protocol/calibration.md` (the calibration items do not have this problem
> because their $Q$ is not derived from $h_3$).

The **target judge** for construct D1. Proposal §0.4: relevance is a binary relation between $h$ and the top QUD $Q$;
$h$ is relevant to $Q$ when it **eliminates at least one live cell of $Q$**.

The scale is `criteria2` (answer compatibility) of QUDeval (Wu et al. 2023), unchanged,
so that the judge can be **pre-calibrated on the 2,190 human annotations** of the QUDeval answerhood set
(`protocol/calibration.md`).

**Blinding is mandatory.** The judge must not know which option is $h_2$ and which is $h_3$. The caller
randomises the `A`/`B` assignment per item and stores the map separately.

---

## SYSTEM

```
You judge whether a statement answers a question. You are not judging whether the
statement is true, likely, interesting, or well written.

A statement ANSWERS a question when knowing the statement rules out at least one of
the ways the question could have been settled. A statement that is about the same
topic, or that mentions the same things, does not thereby answer the question.

Output exactly one JSON object and nothing else.
```

## USER

````
# Question

{Q}

# Two statements about the same person

  A. {STATEMENT_A}
  B. {STATEMENT_B}

# Context in which the question arose

{CTX}

# What to decide

For A, and separately for B, decide how the statement stands to the question.

  1  ANSWERS      Knowing it settles the question, or rules out at least one way
                  the question could have been settled.
  2  PARTIAL      It bears on the question but leaves every way of settling it open.
  3  DOES NOT     Knowing it leaves the question exactly where it was.

Then, independently, rate how much vocabulary the statement shares with the
context, on the same three-point scale:

  1  HIGH   several content words in common
  2  SOME   one content word in common
  3  NONE   no content word in common

These two ratings must be made separately. A statement can share every word with the
context and still not answer the question. Here is a real example of exactly that:

    Question : What do you think of my painting?
    Statement: I painted the walls.
    answerhood = 3 (does not answer)   overlap = 1 (high)

And a statement can share no word and still answer:

    Question : Is Smith seeing anyone?
    Statement: He has been making a lot of trips to New York lately.
    answerhood = 1 (answers)   overlap = 3 (none)

    It answers because the natural reading is that Smith is seeing someone in New York,
    and that rules out "no one". An answer does not have to be literal, and it does not
    have to be certain. If the ordinary reader would come away knowing more about how
    the question stands, it answers.

Reserve 2 for statements that bear on the question without settling any of its ways:

    Question : Is Smith seeing anyone?
    Statement: He has seemed cheerful this month.

# Output

{
  "A_answerhood": 1|2|3,
  "A_overlap": 1|2|3,
  "A_why": "<at most 25 words, naming the cell that is or is not ruled out>",
  "B_answerhood": 1|2|3,
  "B_overlap": 1|2|3,
  "B_why": "<at most 25 words>",
}
````

---

## Pass conditions (aggregated in `protocol/crosscheck_protocol.md`)

An item passes J1 if, after unblinding:

| | condition |
|---|---|
| J1a | median `answerhood` of $h_3$ = **1** |
| J1b | median `answerhood` of $h_2$ = **3** |
| ~~J1c~~ | **split off** to `25_judge_qud_recovery.md`. Showing $Q$ and then asking to recover $Q$ measures compliance, not recovery |
| ~~J1d~~ | ~~`overlap` = 3 for both~~; **dropped from the gate.** In the pilot the judges' overlap ratings were uncorrelated with the machine check (rho=−0.02). Recorded for diagnosis only |

> **J1d diverging from J1a is the key information of this judgment.** Only `answerhood`=1 together with
> `overlap`=3 licenses "answerhood, not vocabulary". If the two move together, the judge is measuring
> lexical overlap rather than answerhood and is excluded
> (discriminant-validity check in `protocol/calibration.md`).
