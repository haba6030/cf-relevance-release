from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

try:
    from rq3_decoder_contract import (
        H3_DECODER_CONTRACT,
        load_h3_decoder_gate,
    )
except ModuleNotFoundError:
    from sim.rq3_decoder_contract import (
        H3_DECODER_CONTRACT,
        load_h3_decoder_gate,
    )
try:
    from rq3_prompting import (
        PROMPT_CONTRACT_VERSION,
        answer_continuations,
        behavior_prompt,
        norming_prompt,
    )
except ModuleNotFoundError:
    from sim.rq3_prompting import (
        PROMPT_CONTRACT_VERSION,
        answer_continuations,
        behavior_prompt,
        norming_prompt,
    )


def test_task2_task3_prompts_are_distinct_arms() -> None:
    task2 = behavior_prompt("INTRO", 2, 0, 0)
    task3 = behavior_prompt("INTRO", 3, 0, 0)
    assert "two statements" in task2
    assert "three statements" in task3
    assert task2 != task3
    assert PROMPT_CONTRACT_VERSION in "rq3-v2-raw-sequence-20260825"


def test_norming_prompt_does_not_repeat_evidence_and_answers_match_question() -> None:
    evidence = "The analyst found a red umbrella."
    prompt = norming_prompt(evidence, "Dan owns an umbrella", "C", 1, 0)
    assert prompt.count(evidence) == 1
    assert "strengthen or weaken" in prompt
    assert answer_continuations("C") == (" strengthen", " weaken")
    assert answer_continuations("P") == (" yes", " no")
    assert "<|" not in prompt  # raw causal-LM interface, not a chat template


def test_retired_one_vs_three_gate_is_rejected(tmp_path) -> None:
    path = tmp_path / "retired.csv"
    pd.DataFrame(
        [{
            "model_id": "m", "task": 2, "selected_layers": "[1]",
            "decode_gate": True,
        }]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="retired rq3a_decoding_gates"):
        load_h3_decoder_gate(path)


def test_canonical_gate_keeps_tasks_separate(tmp_path) -> None:
    path = tmp_path / "canonical.csv"
    pd.DataFrame(
        [
            {"model_id": "m", "task": task, "family": "conjunction_h3",
             "positive_label": "R,h3", "negative_label": "I,h3",
             "selected_layers": "[1]", "decode_gate": True}
            for task in (2, 3)
        ]
    ).to_csv(path, index=False)
    frame = load_h3_decoder_gate(path, require_both_tasks=True)
    assert set(frame.task) == {2, 3}
    assert set(frame.decoder_contract) == {H3_DECODER_CONTRACT}


def test_relocated_rq3_scripts_resolve_the_repository_root() -> None:
    rq3 = Path(__file__).resolve().parent
    repo = rq3.parents[1]
    for name in (
        "accelerate_rq3_task_gpu.py",
        "fetch_wikitext_rq3.py",
        "queue_rq3_behavior_gpu.py",
        "queue_rq3_h3_decoder_gpu.py",
        "queue_rq3_intervention_gpu.py",
        "queue_rq3_pcr_separability_gpu.py",
        "queue_rq3_representation_gpu.py",
        "queue_rq3_transfer_gpu.py",
        "run_rq3_fluency_guard_gpu.py",
        "run_rq3_pcr_separability_analysis.py",
    ):
        source = (rq3 / name).read_text()
        assert "Path(__file__).resolve().parent.parent" not in source, name
        assert "Path(__file__).resolve().parents[2]" in source, name
    from run_rq3_behavior_gpu import PROJ, STIMULI
    assert PROJ == repo
    assert STIMULI == repo / "data" / "stimuli_RQ3.csv"


def test_h3_subspace_execution_contract_is_forwarded_end_to_end() -> None:
    rq3 = Path(__file__).resolve().parent
    fluency = (rq3 / "run_rq3_fluency_guard_gpu.py").read_text()
    intervention = (rq3 / "run_rq3_intervention_gpu.py").read_text()
    queue = (rq3 / "queue_rq3_intervention_gpu.py").read_text()
    assert 'choices=("registered_did", "h3_context", "h3_subspace")' in fluency
    assert "args.subspace_k, args.random_subspaces, args.ablate_scope," in intervention
    assert 'direction_mode in ("h3_context", "h3_subspace")' in intervention
    assert "--random-subspaces {args.random_subspaces} --ablate-scope {args.ablate_scope}" in queue
