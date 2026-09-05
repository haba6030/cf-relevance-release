# Judge pre-calibration: does the judge measure answerhood or vocabulary?

The procedure behind `crosscheck_protocol.md` §3. Run **before any new item is sent to a judge**.

One question is asked.

> Is the judge's "does this sentence answer this question" verdict **separable from lexical overlap?**

If not, that judge cannot validate `no-overlap` (lexical separation between context and conjunct): it carries
the very confound we are trying to remove.

---

## 1. Building the calibration set

```bash
bash     data/additional_linguistics/fetch.sh
python3  data/additional_linguistics/extract.py
python3  data/llm-build/run/build_calibration_set.py
# -> data/llm-build/items/calibration_set.json
```

A **2×2 of answerhood × lexical overlap**. This is the most important design decision in this document.

| | answerhood 1 (answers) | answerhood 3 (does not answer) |
|---|---|---|
| **no overlap** | `C-pos` 20: original pairs from the implicature set, 0 shared words | `C-negx-noov` 20: responses from other dialogues, 0 shared words |
| **overlap** | `C-pos-ov` 20: original pairs, ≥1 shared word | `C-negx-ov` 20: responses from other dialogues, ≥1 shared word |

Added to these are a stratified human-labelled sample `C-human` of 300 (QUDeval `c2_answer_compat` 1/2/3, 100 each) and
14 naturally occurring maxim violations `C-neg` (`lmprag_maxims`). 394 in total.

### Why it must be 2×2: what the first version got wrong

**The first version used only two cells, `C-pos` (overlap 0 / answerhood 1) and `C-negx-ov` (overlap ≥1 / answerhood 3).**
That **completely confounds** lexical overlap with true answerhood: overlap 0 ⟺ answerhood 1. On such a set

> **the better a judge discriminates answerhood, the larger its "lexical contamination" correlation.**

In the pilot of 2026-08-20 `qwen3:8b` was ruled "contaminated, excluded" at ρ = **−0.434**,
which was evidence of **discrimination**, not contamination. The test fired on the opposite signal
to the one it was meant to catch.

With all four cells filled, overlap can vary while true answerhood is held fixed, and contamination is
defined as a **within-stratum effect**:

$$\text{contamination} = \tfrac12\Big[\big(\bar s_{1,\text{ov}}-\bar s_{1,\text{no}}\big)
+\big(\bar s_{3,\text{ov}}-\bar s_{3,\text{no}}\big)\Big]$$

For a judge that measures answerhood this value should be 0.

### The natural prototype

Item 2 of `lmprag_maxims` is a real instance of [high overlap / answerhood 0].

```
Q : What do you think of my painting?
A : I painted the walls.
```

It shares `paint` but eliminates no cell of $Q$.

> **The weakness first.** Such items are rare in natural data: **1** of the 20 `lmprag_maxims` items.
> Hence `C-negx-ov` and `C-negx-noov` are **constructed negatives**
> (responses from other dialogues attached). It is a weakness of the same kind as the constructed negatives
> elsewhere in the project and is reported the same way.

## 2. Running the judgment

Use `prompts/20_judge_answerhood.md` **unchanged**; a prompt tweaked for calibration makes the
calibration meaningless. For calibration items the original discourse (news paragraph / dialogue prefix) fills `CTX`,
and an unrelated sentence from the same discourse fills `STATEMENT_B` to keep the format.

## 3. Pass criteria

| check | metric | threshold | on failure |
|---|---|---|---|
| **convergence** | Spearman ρ between judge answerhood and human labels on C-human | ≥ .45 | exclude |
| | exact agreement on the 3-point scale | ≥ 55% | exclude |
| **discrimination** | AUC of the answerhood-1 cells vs the answerhood-3 cells | ≥ .75 | exclude |
| **lexical contamination** | mean **within-stratum** overlap effect (formula above) | \|diff\| ≤ 0.30 | **exclude** |
| **responsiveness** | parse success rate | ≥ 80% | exclude |

> **The overall correlation ignoring strata is not used.** Unless the 2×2 is balanced it is confounded, and
> even when balanced the within-stratum effect is easier to read. `run_calibration.py` prints the overall
> correlation too, flagged "confounded, do not use".

The ρ ≥ .45 threshold looks low, but QUDeval's own inter-annotator agreement is imperfect and
the news-discourse domain costs transfer. **A higher threshold would likely exclude every judge,
and "no judge is usable" is itself a legitimate conclusion.**

## 4. If every judge fails

That is **information**, not a setback. The judges are frontier models (`protocol` §1.2), so total failure
should be rare; if it happens there are three routes.

1. **Fix the prompt.** In the pilot, changing one J1 anchor example moved the AUC on the same data
   from 0.573 to 0.657. Much of a qualification failure comes from the prompt, not the judge.
   **But a changed prompt restarts calibration from scratch**:
   tuning the prompt on the same data until it passes is overfitting, not calibration.
   The number of revisions and every round's metrics are reported.
2. **Humans judge.** 24 items × J1, J2 is manageable. With a single author, self-scoring bias is stated
   as a limitation; where possible a colleague serves as second judge and κ is reported.
3. **Fall back to a design without judges.** Control `no-overlap` by the machine check alone and claim answerhood
   **by manipulation only**; then run the RQ2b 2×2 to show the separation of answerhood from lexical overlap
   **in behavioural data**. In that case LLM-as-a-judge does not appear in the paper.

> **On every route, approvals from a judge that failed qualification are not reported.**
> Including failed judgments "for reference" is the worst option: readers read them as evidence.

## 5. Reporting

Calibration results are reported regardless of item acceptance.

- the four metrics per judge
- excluded judges and reasons
- distribution of answerhood verdicts on C-neg / C-pos (overlapping if they overlap)
- the 3×3 confusion matrix against QUDeval human labels
