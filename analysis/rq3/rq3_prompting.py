"""Single prompt/label contract for every RQ3 model-facing measurement.

Version 1 mixed raw causal prompts (behavior/activation) with chat-templated
first-token prompts (P/C norming).  It also asked for ``strengthen/weaken``
while scoring ``strength/weak``.  Version 2 keeps the causal-LM interface used
by the behavioral experiment, uses the same two stimulus frames everywhere,
and scores the complete answer continuations named in the question.
"""

from __future__ import annotations

from typing import Literal


PROMPT_CONTRACT_VERSION = "rq3-v2-raw-sequence-20260825"
TASKS = (2, 3)
PREFIXES = ("N", "I", "R")
OPTIONS = ("base", "conj_h2", "conj_h3", "bare_h2", "bare_h3")

# Task 2 and task 3 remain separate experimental arms.  The wording is frozen
# from docs/EXPERIMENT_spec.md; only construction is centralized here.
QVAR = {
    2: (
        "Which of the two statements is more probable?",
        "Which is more probable?",
        "Of the two statements below, which is more likely to be true?",
        "Choose the more probable statement.",
        "Which of the following two is more probable?",
        "Answer with the more probable statement.",
    ),
    3: (
        "Rank the three statements from most to least probable.",
        "Which of the three statements is most probable?",
        "Which statement is most probable?",
        "Which is most likely to be true?",
        "Rank the following by probability. The most probable is:",
        "Answer with the most probable statement.",
    ),
}

# Both behavior and norming use these raw causal-LM frames.  No component may
# apply a chat template on top of them.
FRAME = (
    "Consider the following text:\n\n{intro}\n\n{q}\n",
    "{intro}\n\n{q}\n",
)

QP = (
    "Is it probable that {hyp}? Answer yes or no.\nAnswer:",
    "Would you say that {hyp}? Answer yes or no.\nAnswer:",
    "Based on the text above, is it likely that {hyp}? Answer yes or no.\nAnswer:",
)

# The evidence is already present in ``intro``.  Refer to it rather than
# repeating it, which would create a repetition/lexical-priming confound.
QC = (
    "Consider the hypothesis that {hyp}.\nDoes the evidence described in the text strengthen or weaken that hypothesis? Answer strengthen or weaken.\nAnswer:",
    "Consider the hypothesis that {hyp}.\nGiven the evidence described above, does that hypothesis strengthen or weaken? Answer strengthen or weaken.\nAnswer:",
    "Consider the hypothesis that {hyp}.\nHow does the evidence described above affect that hypothesis — strengthen or weaken? Answer strengthen or weaken.\nAnswer:",
)

ANSWER_CONTINUATIONS = {
    "P": (" yes", " no"),
    "C": (" strengthen", " weaken"),
}


def validate_task(task: int) -> int:
    if task not in TASKS:
        raise ValueError(f"RQ3 task must be one of {TASKS}, got {task}")
    return task


def render_frame(intro: str, question: str, frame_index: int) -> str:
    if frame_index not in range(len(FRAME)):
        raise ValueError(f"Unknown RQ3 frame {frame_index}")
    return FRAME[frame_index].format(intro=intro, q=question)


def behavior_prompt(intro: str, task: int, frame_index: int, qvar_index: int) -> str:
    validate_task(task)
    if qvar_index not in range(len(QVAR[task])):
        raise ValueError(f"Unknown task-{task} question variant {qvar_index}")
    return render_frame(intro, QVAR[task][qvar_index], frame_index)


def norming_prompt(
    intro: str,
    hypothesis: str,
    metric: Literal["P", "C"],
    frame_index: int,
    variant: int,
) -> str:
    questions = QP if metric == "P" else QC if metric == "C" else None
    if questions is None:
        raise ValueError(f"Unknown norming metric {metric!r}")
    if variant not in range(len(questions)):
        raise ValueError(f"Unknown {metric} norming variant {variant}")
    return render_frame(intro, questions[variant].format(hyp=hypothesis), frame_index)


def answer_continuations(metric: Literal["P", "C"]) -> tuple[str, str]:
    try:
        return ANSWER_CONTINUATIONS[metric]
    except KeyError as error:
        raise ValueError(f"Unknown norming metric {metric!r}") from error

