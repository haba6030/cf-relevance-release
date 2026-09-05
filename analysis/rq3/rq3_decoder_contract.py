"""Canonical decoder-label contract for RQ3 and downstream provenance checks.

The first RQ3 implementation used a retired one-positive/three-negative ranking
task.  Downstream stages must instead use the amended, task-specific binary
contrast with the continuation held fixed: R,h3 versus I,h3.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


H3_DECODER_CONTRACT = "rq3-h3-context-r-vs-i-20260825"
H3_GATE_FAMILY = "conjunction_h3"
H3_POSITIVE_LABEL = "R,h3"
H3_NEGATIVE_LABEL = "I,h3"
RETIRED_DECODER_DESCRIPTION = "R,h3=1 versus R,h2/I,h3/I,h2=0"


def load_h3_decoder_gate(path: Path, *, require_both_tasks: bool = False) -> pd.DataFrame:
    """Load and validate the only decoder gate allowed for canonical follow-ups."""
    frame = pd.read_csv(path)
    required = {
        "model_id", "task", "family", "positive_label", "negative_label",
        "selected_layers", "decode_gate",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Decoder gate {path} lacks canonical provenance columns: {sorted(missing)}. "
            "The retired rq3a_decoding_gates.csv cannot be used downstream."
        )
    frame = frame[frame.family == H3_GATE_FAMILY].copy()
    if frame.empty:
        raise ValueError(f"No {H3_GATE_FAMILY!r} rows in decoder gate {path}")
    bad = frame[
        (frame.positive_label != H3_POSITIVE_LABEL)
        | (frame.negative_label != H3_NEGATIVE_LABEL)
        | ~frame.task.isin((2, 3))
    ]
    if not bad.empty:
        columns = ["model_id", "task", "positive_label", "negative_label"]
        raise ValueError(
            "Non-canonical decoder labels found; expected R,h3=1 and I,h3=0 "
            f"within each task:\n{bad[columns].to_string(index=False)}"
        )
    duplicates = frame.duplicated(["model_id", "task"], keep=False)
    if duplicates.any():
        raise ValueError("Decoder gate has duplicate model/task rows")
    if require_both_tasks:
        incomplete = {
            model: sorted(group.task.astype(int).tolist())
            for model, group in frame.groupby("model_id")
            if set(group.task.astype(int)) != {2, 3}
        }
        if incomplete:
            raise ValueError(f"Decoder gate does not keep task 2/3 as complete separate cells: {incomplete}")
    frame["decoder_contract"] = H3_DECODER_CONTRACT
    return frame

