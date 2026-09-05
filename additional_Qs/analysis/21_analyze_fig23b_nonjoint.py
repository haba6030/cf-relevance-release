#!/usr/bin/env python3
"""Recompute Fig. 23B as separate P and C readout tests on CPU.

This is deliberately not a P|R/C|R intervention analysis.  Every row comes
from the already executed shared PC|R edit; only the P and C manipulation
readouts are tested separately, without the former P&C IUT gate.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def load_base():
    path = Path(__file__).with_name("19_analyze_norm_manipulation.py")
    spec = importlib.util.spec_from_file_location("norm_manipulation_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def report(rows: pd.DataFrame) -> str:
    target = rows[rows.control_role == "target"].copy()
    counts = (rows.groupby(["control_role", "metric"]).pass_raw_p
              .agg(["sum", "count"]).reset_index())
    lines = [
        "# Fig. 23B: separate P and C readouts of the shared PC|R intervention", "",
        "This analysis does not re-run the shared PC|R intervention; it recomputes one-sided tests of P and C "
        "on CPU from the stored 165-item item-level ΔDID.",
        "No joint IUT gate over P and C and no Holm correction are used. "
        "Note that the two tests are two readouts of the same PC|R intervention, not separate P|R and C|R direction edits.", "",
        "## Pass counts", "",
        "| readout | target | random | shuffle |",
        "|---|---:|---:|---:|",
    ]
    indexed = counts.set_index(["control_role", "metric"])
    for metric in ("P", "C"):
        values = [f"{int(indexed.loc[(role, metric), 'sum'])}/{int(indexed.loc[(role, metric), 'count'])}"
                  for role in ("target", "random", "shuffle")]
        lines.append(f"| {metric} | {values[0]} | {values[1]} | {values[2]} |")
    lines += [
        "", "## Target cells", "",
        "| model | task | stage | strength | source | readout | oriented ΔDID | raw one-sided p | pass | n |",
        "|---|---:|---|---:|---|---|---:|---:|---|---:|",
    ]
    for row in target.sort_values(["model_id", "task", "stage", "strength", "metric"]).itertuples():
        lines.append(
            f"| {row.model_id.split('/')[-1]} | {row.task} | {row.stage} | {row.strength:g} | "
            f"{row.selection_source} | {row.metric} | {row.oriented_delta_did:+.4f} | "
            f"{row.raw_one_sided_p:.4g} | {'yes' if row.pass_raw_p else 'no'} | {row.n} |"
        )
    lines += [
        "", "## Decision contract", "",
        "The null hypothesis of each readout is that its mean effect in the predicted direction is at most zero. "
        "ADD/ABL and the sign of C are oriented as in the original intervention contract. Each P or C cell "
        "passes individually at uncorrected one-sided t p<.05. There is no AND gate between P and C.",
        "The effect is ΔDID = [(R−I)@h3−(R−I)@h2]_edited − "
        "[(R−I)@h3−(R−I)@h2]_baseline; the two cross-frame repetitions are averaged within item.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild item effects from the executed prompt-level cells, then rerun the
    # locked one-sided tests.  This avoids deriving component p-values from the
    # already-combined joint p.
    cells = BASE.load_cells(args.input_dir)
    items = BASE.item_effects(cells)
    margins = BASE.baseline_margins(items)
    tests = BASE.analyze(items, margins)
    rows = tests[(tests.theta == .25) &
                 (tests.target_direction == "PC_given_R")].copy()
    rows = rows.rename(columns={
        "expected_oriented_mean": "oriented_delta_did",
        "superiority_p": "raw_one_sided_p",
        "superiority_pass": "pass_raw_p",
    })
    keep = [
        "model_id", "task", "stage", "strength", "selection_source",
        "direction", "control_role", "metric", "n", "oriented_delta_did",
        "raw_one_sided_p", "pass_raw_p", "items", "min_frames", "max_frames",
    ]
    rows = rows[keep].sort_values([
        "control_role", "model_id", "task", "stage", "strength", "metric"
    ]).reset_index(drop=True)
    target = rows[rows.control_role == "target"]
    assert len(target) == 26
    passed = target.groupby("metric").pass_raw_p.sum()
    assert int(passed["P"]) == 2 and int(passed["C"]) == 6

    rows.to_csv(args.output_dir / "fig23b_nonjoint_readouts.csv", index=False)
    (args.output_dir / "FIG23B_NONJOINT_REPORT.md").write_text(
        report(rows), encoding="utf-8"
    )
    print("Fig23B non-joint CPU analysis complete: P=2/13, C=6/13")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
