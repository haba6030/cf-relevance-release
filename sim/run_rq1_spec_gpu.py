#!/usr/bin/env python3
"""Run the frozen RQ1/H2 specification on one locally cached model and one GPU.

Outputs per model:
  logprob/       L2C, L3C, L2B, L3B (720 prompt rows; five strings each)
  forcedchoice/  F2C, F3C, F2B, F3B (3,600 prompt rows; letter/content reads)
  norming/       NP2, NP3, NC2, NC3 (180 rows; two forward passes each)

The grids and prompt text follow docs/EXPERIMENT_spec.md sections 4.1--4.3.  This
runner deliberately refuses to run without CUDA so a queued GPU experiment can
never silently fall back to CPU.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJ / "data" / "llm-build" / "run"))

from run_norm15 import prompts as norm_prompts  # noqa: E402
from run_rq1_v2 import FRAME, QVAR, items as active_items, strings  # noqa: E402
from scenarios9 import no_context  # noqa: E402

ARMS = ("conj", "bare")
HYPOTHESES = ("h2", "h3")
LETTERS = ("A", "B", "C")


def safe_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_name)


def atomic_csv(rows: list[dict], path: Path, expected: int) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete output for {path}: {len(rows)} != {expected}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def output_complete(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        return sum(1 for _ in path.open(encoding="utf-8")) == expected + 1
    except OSError:
        return False


class GPUScorer:
    def __init__(self, model_name: str, device: str, dtype: torch.dtype) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"GPU device required, got {device}")
        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        if next(self.model.parameters()).device.type != "cuda":
            raise RuntimeError("Model was not placed on CUDA")

    def _forward(self, input_ids, attention_mask, position_ids, keep: int):
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        try:
            logits = self.model(**kwargs, logits_to_keep=keep).logits
        except TypeError:
            logits = self.model(**kwargs).logits
        if logits.shape[1] > keep:
            logits = logits[:, -keep:]
        return logits.float()

    def continuation_jobs(self, rows: list[dict]) -> list[dict]:
        jobs = []
        for row_index, row in enumerate(rows):
            prompt_ids = self.tok(row["prompt"], return_tensors="pt").input_ids[0]
            prompt_len = int(prompt_ids.numel())
            for key, continuation in row["continuations"].items():
                full_ids = self.tok(
                    row["prompt"] + continuation, return_tensors="pt"
                ).input_ids[0]
                nt = int(full_ids.numel()) - prompt_len
                if nt <= 0 or not torch.equal(full_ids[:prompt_len], prompt_ids):
                    raise RuntimeError(
                        f"Token boundary changed: row={row_index}, continuation={key}"
                    )
                jobs.append(
                    dict(
                        row_index=row_index,
                        key=key,
                        input_ids=full_ids,
                        prompt_len=prompt_len,
                        nt=nt,
                    )
                )
        return jobs

    @torch.inference_mode()
    def score_continuations(
        self, rows: list[dict], batch_size: int, progress_label: str
    ) -> None:
        jobs = self.continuation_jobs(rows)
        started = time.time()
        for offset in range(0, len(jobs), batch_size):
            batch = jobs[offset : offset + batch_size]
            padded = self.tok.pad(
                {"input_ids": [job["input_ids"] for job in batch]},
                padding=True,
                return_tensors="pt",
            )
            input_ids = padded["input_ids"].to(self.device)
            attention_mask = padded["attention_mask"].to(self.device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            keep = max(job["nt"] for job in batch) + 1
            logits = self._forward(input_ids, attention_mask, position_ids, keep)
            for batch_index, job in enumerate(batch):
                nt = job["nt"]
                start = keep - nt - 1
                token_logits = logits[batch_index, start : start + nt]
                targets = job["input_ids"][job["prompt_len"] :].to(self.device)
                token_lp = token_logits.log_softmax(-1).gather(
                    -1, targets[:, None]
                ).squeeze(-1)
                row = rows[job["row_index"]]["meta"]
                row[f"lp_{job['key']}"] = float(token_lp.sum().cpu())
                row[f"nt_{job['key']}"] = nt
            done = min(offset + len(batch), len(jobs))
            if done == len(jobs) or done % max(batch_size * 20, 1) == 0:
                elapsed = time.time() - started
                eta = (len(jobs) - done) * elapsed / done
                print(
                    f"  {progress_label} {done}/{len(jobs)} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )

    def first_token_ids(self, labels: tuple[str, ...], leading_space: bool) -> list[int]:
        ids = []
        for label in labels:
            text = (" " if leading_space else "") + label
            encoded = self.tok.encode(text, add_special_tokens=False)
            if not encoded:
                raise RuntimeError(f"No token for answer label {text!r}")
            if len(encoded) != 1:
                print(
                    f"WARNING answer label {text!r} has {len(encoded)} tokens; "
                    "using its first token as preregistered",
                    flush=True,
                )
            ids.append(encoded[0])
        return ids

    @torch.inference_mode()
    def score_next_tokens(
        self,
        prompts: list[str],
        token_ids: list[int],
        batch_size: int,
        progress_label: str,
    ) -> list[list[float]]:
        output: list[list[float]] = []
        started = time.time()
        for offset in range(0, len(prompts), batch_size):
            text_batch = prompts[offset : offset + batch_size]
            padded = self.tok(text_batch, padding=True, return_tensors="pt")
            input_ids = padded["input_ids"].to(self.device)
            attention_mask = padded["attention_mask"].to(self.device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            logits = self._forward(input_ids, attention_mask, position_ids, 1)[:, -1]
            lp = logits.log_softmax(-1)[:, token_ids].cpu()
            output.extend(lp.tolist())
            done = min(offset + len(text_batch), len(prompts))
            if done == len(prompts) or done % max(batch_size * 20, 1) == 0:
                elapsed = time.time() - started
                eta = (len(prompts) - done) * elapsed / done
                print(
                    f"  {progress_label} {done}/{len(prompts)} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )
        return output

    def close(self) -> None:
        del self.model
        del self.tok
        gc.collect()
        torch.cuda.empty_cache()


def build_logprob_rows(model_name: str, its) -> list[dict]:
    rows = []
    for item, kind in its:
        candidates = strings(item)
        for context in ("no", "yes"):
            intro = no_context(item) if context == "no" else item["ctx"]
            for task in (2, 3):
                for frame_index, frame in enumerate(FRAME):
                    for qvar_index, question in enumerate(QVAR[task]):
                        prompt = frame.format(intro=intro, q=question)
                        rows.append(
                            {
                                "meta": dict(
                                    model=model_name.split("/")[-1],
                                    model_id=model_name,
                                    scenario=item["scenario_id"],
                                    label=item.get("label", ""),
                                    kind=kind,
                                    paradigm=item["paradigm"],
                                    context=context,
                                    task=task,
                                    frame=frame_index,
                                    qvar=qvar_index,
                                ),
                                "prompt": prompt,
                                "continuations": {
                                    key: " " + value for key, value in candidates.items()
                                },
                            }
                        )
    return rows


def arm_options(item: dict, arm: str) -> dict[str, str]:
    candidates = strings(item)
    return {
        "base": candidates["base"],
        "h2": candidates[f"{arm}_h2"],
        "h3": candidates[f"{arm}_h3"],
    }


def fc_prompt(intro: str, question: str, order: tuple[str, ...], opts: dict) -> str:
    displayed = "\n".join(
        f"{LETTERS[index]}. {opts[key]}" for index, key in enumerate(order)
    )
    return (
        "Consider the following text:\n\n"
        f"{intro}\n\n{question}\n{displayed}\nAnswer:"
    )


def build_forcedchoice_rows(model_name: str, its) -> list[dict]:
    rows = []
    for item, kind in its:
        for context in ("no", "yes"):
            intro = no_context(item) if context == "no" else item["ctx"]
            for arm in ARMS:
                opts = arm_options(item, arm)
                for alternative in HYPOTHESES:
                    keys = ("base", alternative)
                    for order in itertools.permutations(keys):
                        positions = {key: order.index(key) for key in order}
                        for qvar_index, question in enumerate(QVAR[2]):
                            rows.append(
                                {
                                    "meta": dict(
                                        model=model_name.split("/")[-1],
                                        model_id=model_name,
                                        scenario=item["scenario_id"],
                                        label=item.get("label", ""),
                                        kind=kind,
                                        paradigm=item["paradigm"],
                                        context=context,
                                        task=2,
                                        arm=arm,
                                        alternative=alternative,
                                        order="|".join(order),
                                        qvar=qvar_index,
                                        pos_base=positions["base"],
                                        pos_h2=positions.get("h2", -1),
                                        pos_h3=positions.get("h3", -1),
                                    ),
                                    "prompt": fc_prompt(intro, question, order, opts),
                                    "order": order,
                                    "continuations": {
                                        key: " " + opts[key] for key in order
                                    },
                                }
                            )
                keys = ("base", "h2", "h3")
                for order in itertools.permutations(keys):
                    positions = {key: order.index(key) for key in order}
                    for qvar_index, question in enumerate(QVAR[3]):
                        rows.append(
                            {
                                "meta": dict(
                                    model=model_name.split("/")[-1],
                                    model_id=model_name,
                                    scenario=item["scenario_id"],
                                    label=item.get("label", ""),
                                    kind=kind,
                                    paradigm=item["paradigm"],
                                    context=context,
                                    task=3,
                                    arm=arm,
                                    alternative="both",
                                    order="|".join(order),
                                    qvar=qvar_index,
                                    pos_base=positions["base"],
                                    pos_h2=positions["h2"],
                                    pos_h3=positions["h3"],
                                ),
                                "prompt": fc_prompt(intro, question, order, opts),
                                "order": order,
                                "continuations": {
                                    key: " " + opts[key] for key in order
                                },
                            }
                        )
    return rows


def score_forcedchoice(
    scorer: GPUScorer, rows: list[dict], batch_size: int
) -> list[dict]:
    letter_ids = scorer.first_token_ids(LETTERS, leading_space=True)
    letter_scores = scorer.score_next_tokens(
        [row["prompt"] for row in rows], letter_ids, batch_size, "FC-letter"
    )
    for row, scores in zip(rows, letter_scores):
        meta = row["meta"]
        for index, value in enumerate(scores):
            meta[f"lp_letter_{LETTERS[index]}"] = value
        for key in ("base", "h2", "h3"):
            if key in row["order"]:
                meta[f"lp_letter_{key}"] = scores[row["order"].index(key)]
            else:
                meta[f"lp_letter_{key}"] = math.nan

    content_rows = [
        {
            "meta": row["meta"],
            "prompt": row["prompt"],
            "continuations": {
                f"content_{key}": continuation
                for key, continuation in row["continuations"].items()
            },
        }
        for row in rows
    ]
    scorer.score_continuations(content_rows, batch_size, "FC-content")

    for row in rows:
        meta = row["meta"]
        for hypothesis in HYPOTHESES:
            if hypothesis not in row["order"]:
                meta[f"cf_letter_{hypothesis}"] = math.nan
                meta[f"cf_content_norm_{hypothesis}"] = math.nan
                meta[f"cf_content_raw_{hypothesis}"] = math.nan
                continue
            meta[f"cf_letter_{hypothesis}"] = int(
                meta[f"lp_letter_{hypothesis}"] > meta["lp_letter_base"]
            )
            meta[f"cf_content_norm_{hypothesis}"] = int(
                meta[f"lp_content_{hypothesis}"] / meta[f"nt_content_{hypothesis}"]
                > meta["lp_content_base"] / meta["nt_content_base"]
            )
            meta[f"cf_content_raw_{hypothesis}"] = int(
                meta[f"lp_content_{hypothesis}"] > meta["lp_content_base"]
            )
    return [row["meta"] for row in rows]


def build_norm_rows(model_name: str, scorer: GPUScorer, its, batch_size: int) -> list[dict]:
    rows = []
    rho_prompts = []
    c_prompts = []
    for item, kind in its:
        for hypothesis in HYPOTHESES:
            for context in ("no", "yes"):
                for variant in range(3):
                    rho_prompt, c_prompt = norm_prompts(
                        item, hypothesis, context, variant
                    )

                    def chat(text: str) -> str:
                        return scorer.tok.apply_chat_template(
                            [{"role": "user", "content": text}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )

                    rho_prompts.append(chat(rho_prompt))
                    c_prompts.append(chat(c_prompt))
                    rows.append(
                        dict(
                            model=model_name.split("/")[-1],
                            model_id=model_name,
                            scenario=item["scenario_id"],
                            label=item.get("label", ""),
                            kind=kind,
                            hypothesis=hypothesis,
                            context=context,
                            variant=variant,
                        )
                    )
    rho_ids = scorer.first_token_ids(("yes", "no"), leading_space=False)
    c_ids = scorer.first_token_ids(("strength", "weak"), leading_space=False)
    rho_scores = scorer.score_next_tokens(
        rho_prompts, rho_ids, batch_size, "norm-rho"
    )
    c_scores = scorer.score_next_tokens(c_prompts, c_ids, batch_size, "norm-C")
    for row, rho, confirmation in zip(rows, rho_scores, c_scores):
        row["rho"] = rho[0] - rho[1]
        row["C"] = confirmation[0] - confirmation[1]
    return rows


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(args.threads)
    its = active_items()
    if len(its) != 15:
        raise RuntimeError(f"Expected 15 active items, got {len(its)}")
    slug = safe_name(args.model)
    paths = {
        "logprob": args.output_dir / "logprob" / f"rq1_logprob_{slug}.csv",
        "forcedchoice": args.output_dir
        / "forcedchoice"
        / f"rq1_forcedchoice_{slug}.csv",
        "norming": args.output_dir / "norming" / f"norm_{slug}.csv",
    }
    expected = {"logprob": 720, "forcedchoice": 3600, "norming": 180}
    if all(output_complete(paths[key], expected[key]) for key in paths):
        print(f"SKIP complete model={args.model}", flush=True)
        return

    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(
        f"LOAD model={args.model} device={args.device} dtype={args.dtype} "
        f"batch={args.batch_size}",
        flush=True,
    )
    scorer = GPUScorer(args.model, args.device, dtype)
    started = time.time()
    try:
        if not output_complete(paths["logprob"], expected["logprob"]):
            rows = build_logprob_rows(args.model, its)
            scorer.score_continuations(rows, args.batch_size, "logprob")
            atomic_csv(
                [row["meta"] for row in rows],
                paths["logprob"],
                expected["logprob"],
            )
            print(f"SAVED {paths['logprob']}", flush=True)
        else:
            print(f"SKIP {paths['logprob']}", flush=True)

        if not output_complete(paths["forcedchoice"], expected["forcedchoice"]):
            rows = build_forcedchoice_rows(args.model, its)
            scored = score_forcedchoice(scorer, rows, args.batch_size)
            atomic_csv(scored, paths["forcedchoice"], expected["forcedchoice"])
            print(f"SAVED {paths['forcedchoice']}", flush=True)
        else:
            print(f"SKIP {paths['forcedchoice']}", flush=True)

        if not output_complete(paths["norming"], expected["norming"]):
            rows = build_norm_rows(args.model, scorer, its, args.batch_size)
            atomic_csv(rows, paths["norming"], expected["norming"])
            print(f"SAVED {paths['norming']}", flush=True)
        else:
            print(f"SKIP {paths['norming']}", flush=True)
    finally:
        scorer.close()

    manifest = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "offline": True,
        "n_items": 15,
        "logprob_rows": expected["logprob"],
        "forcedchoice_rows": expected["forcedchoice"],
        "norming_rows": expected["norming"],
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.output_dir / "manifests" / f"{slug}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(manifest_path)
    print(f"COMPLETE model={args.model} elapsed={time.time()-started:.0f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and threads must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
