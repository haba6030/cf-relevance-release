"""Read the canonical decoding gate: a thin wrapper around `rq3_decoder_contract`.

**The canonical contract is `analysis/rq3/rq3_decoder_contract.py`.** This module wraps it and
offers only the short call `canonical_gates(analysis_dir, model_id)`. The contract constants and
validation logic live there, so **new code should use `rq3_decoder_contract` directly.**

The retired `analysis/rq3a_decoding_gates.csv` is a product of the one-versus-three labelling and
has been archived. No code should read that gate again.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from rq3_decoder_contract import H3_GATE_FAMILY, load_h3_decoder_gate
except ModuleNotFoundError:  # pragma: no cover
    from sim.rq3_decoder_contract import H3_GATE_FAMILY, load_h3_decoder_gate

CANONICAL_GATE = "rq3a_context_decoder_robustness.csv"
CANONICAL_FAMILY = H3_GATE_FAMILY
RETIRED_GATE = "rq3a_decoding_gates.csv"


def canonical_gates(analysis_dir: Path, model_id: str | None = None, *, passed_only: bool = True) -> pd.DataFrame:
    """The canonical gate table, filtered to `model_id` if given."""
    path = Path(analysis_dir) / CANONICAL_GATE
    if not path.exists():
        retired = Path(analysis_dir) / RETIRED_GATE
        hint = " (only the retired rq3a_decoding_gates.csv exists)" if retired.exists() else ""
        raise FileNotFoundError(f"canonical gate missing: {path}{hint}")
    table = load_h3_decoder_gate(path)
    if model_id is not None:
        table = table[table.model_id == model_id]
    if passed_only:
        table = table[table.decode_gate.astype(str).str.lower().isin(("true", "1"))]
    return table
