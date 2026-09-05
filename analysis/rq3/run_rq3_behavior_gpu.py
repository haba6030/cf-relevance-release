#!/usr/bin/env python3
"""RQ3 step 1: N/I/R behavioral log-probability grid on one cached model.

This implements docs/EXPERIMENT_RQ3.md sections 1.1, 4.1, 5.0 and 8.1 step 1.
It deliberately has no forced-choice arm.  Each model produces

    15 items x 3 prefixes x 2 tasks x 2 frames x 6 questions = 1,080 rows

with five scored continuations per row.  Models are loaded offline in fp32.
When a model does not fit wholly inside the requested GPU budget, Transformers'
device-map loader offloads layers to CPU while keeping CUDA inference active.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from rq3_prompting import (
        FRAME, OPTIONS, PROMPT_CONTRACT_VERSION, QVAR, behavior_prompt,
    )
except ModuleNotFoundError:  # import as sim.run_rq3_behavior_gpu in tests
    from sim.rq3_prompting import (
        FRAME, OPTIONS, PROMPT_CONTRACT_VERSION, QVAR, behavior_prompt,
    )

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
STIMULI = PROJ / "data" / "stimuli_RQ3.csv"
STIMULI_SHA256 = "3cd24e18e959c212608012ece4ef1c288b2eebb4993fd7b927d087e5ea841764"

CONTINUATIONS = OPTIONS


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value)


def output_complete(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        return sum(1 for _ in path.open(encoding="utf-8")) == expected + 1
    except OSError:
        return False


def atomic_csv(rows: list[dict], path: Path, expected: int) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete output: {len(rows)} != {expected}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def gpu_memory_observation() -> tuple[int, int] | None:
    """Return (this PID MiB, total MiB on its physical GPU) from nvidia-smi."""
    try:
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        devices = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    own_by_uuid: dict[str, int] = {}
    for line in processes.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid, mib = int(fields[1]), int(fields[2])
        except ValueError:
            continue
        if pid == os.getpid():
            own_by_uuid[fields[0]] = own_by_uuid.get(fields[0], 0) + mib
    if not own_by_uuid:
        return None
    total_by_uuid: dict[str, int] = {}
    for line in devices.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            total_by_uuid[fields[0]] = int(fields[1])
        except ValueError:
            continue
    if any(uuid not in total_by_uuid for uuid in own_by_uuid):
        return None
    return max(own_by_uuid.values()), max(total_by_uuid[uuid] for uuid in own_by_uuid)


def gpu_process_mib() -> int | None:
    observation = gpu_memory_observation()
    return None if observation is None else observation[0]


class BudgetedScorer:
    def __init__(self, model_name: str, max_gpu_mib: int, reserve_mib: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing CPU-only fallback")
        if max_gpu_mib > 24_000:
            raise ValueError("The user-imposed hard ceiling is 24,000 MiB per GPU")
        self.max_gpu_mib = max_gpu_mib
        self.model_name = model_name
        self.load_budget_mib = max_gpu_mib - reserve_mib
        if self.load_budget_mib < 1024:
            raise ValueError("GPU budget is too small after reserve")

        # Leave the requested reserve for activations while allowing the CUDA
        # allocator to use most (not all) of the hard ceiling.  The remaining
        # 750 MiB is held outside the allocator: roughly 250 MiB for the CUDA
        # context and 500 MiB for the queue's permitted pre-launch device use.
        # The physical-device check below is the final hard guard.
        allocator_budget_mib = max(self.load_budget_mib, max_gpu_mib - 750)
        total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
        fraction = min(1.0, allocator_budget_mib / total_mib)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        torch.cuda.reset_peak_memory_stats(0)

        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={0: f"{self.load_budget_mib}MiB", "cpu": "200GiB"},
        ).eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        device_map = getattr(self.model, "hf_device_map", {})
        if not any(str(device).startswith("cuda") or device == 0 for device in device_map.values()):
            raise RuntimeError(f"No model layer placed on CUDA: {device_map}")
        self.peak_nvidia_mib = 0
        self.peak_device_nvidia_mib = 0
        self.check_budget("after-load")

    def check_budget(self, where: str) -> None:
        observation = gpu_memory_observation()
        if observation is not None:
            observed, device_observed = observation
            self.peak_nvidia_mib = max(self.peak_nvidia_mib, observed)
            self.peak_device_nvidia_mib = max(self.peak_device_nvidia_mib, device_observed)
            if observed > self.max_gpu_mib or device_observed > self.max_gpu_mib:
                raise RuntimeError(
                    f"GPU memory ceiling exceeded at {where}: "
                    f"process={observed} MiB device_total={device_observed} MiB "
                    f"> {self.max_gpu_mib} MiB"
                )

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

    def chat_prompt(self, text: str) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": text}],
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": text}], **kwargs
            )

    def first_token_ids(self, labels: tuple[str, ...]) -> list[int]:
        ids = []
        for label in labels:
            encoded = self.tok.encode(label, add_special_tokens=False)
            if not encoded:
                raise RuntimeError(f"No token for answer label {label!r}")
            if len(encoded) != 1:
                print(
                    f"WARNING answer label {label!r} has {len(encoded)} tokens; "
                    "using its first token as preregistered",
                    flush=True,
                )
            ids.append(encoded[0])
        return ids

    @torch.inference_mode()
    def score_next_tokens(
        self, prompts: list[str], token_ids: list[int], batch_size: int, label: str
    ) -> list[list[float]]:
        output: list[list[float]] = []
        started = time.time()
        for offset in range(0, len(prompts), batch_size):
            texts = prompts[offset : offset + batch_size]
            padded = self.tok(texts, padding=True, return_tensors="pt")
            input_ids = padded["input_ids"].to(self.input_device)
            attention_mask = padded["attention_mask"].to(self.input_device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            logits = self._forward(input_ids, attention_mask, position_ids, 1)[:, -1]
            output.extend(logits.log_softmax(-1)[:, token_ids].cpu().tolist())
            done = min(offset + len(texts), len(prompts))
            if offset == 0 or done == len(prompts) or done % max(batch_size * 25, 1) == 0:
                self.check_budget(f"{label}-{done}")
                elapsed = time.time() - started
                eta = (len(prompts) - done) * elapsed / done
                print(f"  {label} {done}/{len(prompts)} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
        return output

    @torch.inference_mode()
    def score_hidden(self, prompt: str, continuation: str) -> tuple[float, int, list[np.ndarray], list[np.ndarray]]:
        """Score one continuation and return option-final and prompt-final states."""
        # Imported lazily at module level below would make the behavior-only
        # runner depend on NumPy anyway through pandas; keeping this method
        # generic is useful to the representation runner.
        import numpy as np

        prompt_ids = self.tok(prompt, return_tensors="pt").input_ids[0]
        full_ids = self.tok(prompt + continuation, return_tensors="pt").input_ids[0]
        prompt_len = int(prompt_ids.numel())
        nt = int(full_ids.numel()) - prompt_len
        if nt <= 0 or not torch.equal(full_ids[:prompt_len], prompt_ids):
            raise RuntimeError("Token boundary changed during activation extraction")
        input_ids = full_ids[None, :].to(self.input_device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        keep = nt + 1
        try:
            result = self.model(**kwargs, logits_to_keep=keep)
        except TypeError:
            result = self.model(**kwargs)
        logits = result.logits
        if logits.shape[1] > keep:
            logits = logits[:, -keep:]
        token_logits = logits[0, :nt].float()
        targets = full_ids[prompt_len:].to(token_logits.device)
        lp = token_logits.log_softmax(-1).gather(-1, targets[:, None]).sum()
        option_hidden = [
            state[0, -1].detach().float().cpu().numpy().astype(np.float32, copy=False)
            for state in result.hidden_states
        ]
        prompt_hidden = [
            state[0, prompt_len - 1].detach().float().cpu().numpy().astype(np.float32, copy=False)
            for state in result.hidden_states
        ]
        return float(lp.cpu()), nt, option_hidden, prompt_hidden

    @torch.inference_mode()
    def prompt_hidden(self, prompt: str) -> list[np.ndarray]:
        """Return every layer at the final prompt token, without generation."""
        import numpy as np

        encoded = self.tok(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.input_device)
        attention_mask = encoded.attention_mask.to(self.input_device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        try:
            result = self.model(**kwargs, logits_to_keep=1)
        except TypeError:
            result = self.model(**kwargs)
        return [
            state[0, -1].detach().float().cpu().numpy().astype(np.float32, copy=False)
            for state in result.hidden_states
        ]

    @torch.inference_mode()
    def score_one(self, prompt: str, continuation: str) -> tuple[float, int]:
        """Score one continuation; intervention hooks may be active."""
        prompt_ids = self.tok(prompt, return_tensors="pt").input_ids[0]
        full_ids = self.tok(prompt + continuation, return_tensors="pt").input_ids[0]
        prompt_len = int(prompt_ids.numel())
        nt = int(full_ids.numel()) - prompt_len
        if nt <= 0 or not torch.equal(full_ids[:prompt_len], prompt_ids):
            raise RuntimeError("Token boundary changed during intervention scoring")
        input_ids = full_ids[None, :].to(self.input_device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        logits = self._forward(input_ids, attention_mask, position_ids, nt + 1)
        token_logits = logits[0, :nt]
        targets = full_ids[prompt_len:].to(token_logits.device)
        value = token_logits.log_softmax(-1).gather(-1, targets[:, None]).sum()
        return float(value.cpu()), nt

    @torch.inference_mode()
    def score_pairs(
        self,
        prompts: list[str],
        continuations: list[str],
        batch_size: int,
    ) -> list[tuple[float, int]]:
        """Score aligned prompt/continuation pairs under one hook setting."""
        if len(prompts) != len(continuations):
            raise ValueError("prompts and continuations must have equal length")
        jobs = []
        for prompt, continuation in zip(prompts, continuations):
            prompt_ids = self.tok(prompt, return_tensors="pt").input_ids[0]
            full_ids = self.tok(prompt + continuation, return_tensors="pt").input_ids[0]
            prompt_len = int(prompt_ids.numel())
            nt = int(full_ids.numel()) - prompt_len
            if nt <= 0 or not torch.equal(full_ids[:prompt_len], prompt_ids):
                raise RuntimeError("Token boundary changed during batched scoring")
            jobs.append((full_ids, prompt_len, nt))

        scores: list[tuple[float, int]] = []
        for offset in range(0, len(jobs), batch_size):
            batch = jobs[offset : offset + batch_size]
            padded = self.tok.pad(
                {"input_ids": [full_ids for full_ids, _, _ in batch]},
                padding=True,
                return_tensors="pt",
            )
            input_ids = padded["input_ids"].to(self.input_device)
            attention_mask = padded["attention_mask"].to(self.input_device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            keep = max(nt for _, _, nt in batch) + 1
            logits = self._forward(input_ids, attention_mask, position_ids, keep)
            for index, (full_ids, prompt_len, nt) in enumerate(batch):
                start = keep - nt - 1
                token_logits = logits[index, start : start + nt]
                targets = full_ids[prompt_len:].to(logits.device)
                value = token_logits.log_softmax(-1).gather(-1, targets[:, None]).sum()
                scores.append((float(value.cpu()), nt))
        return scores

    @torch.inference_mode()
    def score_next_one(self, prompt: str, token_ids: list[int]) -> list[float]:
        encoded = self.tok(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.input_device)
        attention_mask = encoded.attention_mask.to(self.input_device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        logits = self._forward(input_ids, attention_mask, position_ids, 1)[0, -1]
        return logits.log_softmax(-1)[token_ids].float().cpu().tolist()

    @torch.inference_mode()
    def sequence_nll(self, token_ids: torch.Tensor, chunk_tokens: int = 1_000) -> tuple[float, int]:
        """Teacher-forced NLL, using the full fixed guard context when possible."""
        token_ids = token_ids.detach().cpu().long()
        total, count, start = 0.0, 0, 0
        while start < len(token_ids) - 1:
            end = min(len(token_ids), start + chunk_tokens + 1)
            chunk = token_ids[start:end][None, :].to(self.input_device)
            attention_mask = torch.ones_like(chunk)
            position_ids = attention_mask.long().cumsum(-1) - 1
            logits = self._forward(chunk, attention_mask, position_ids, chunk.shape[1])
            prediction = logits[0, :-1]
            targets = chunk[0, 1:].to(prediction.device)
            nll = -prediction.log_softmax(-1).gather(-1, targets[:, None]).sum()
            total += float(nll.cpu()); count += int(targets.numel())
            start = end - 1
        return total, count

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
    def score(self, rows: list[dict], batch_size: int) -> list[dict]:
        jobs = self.continuation_jobs(rows)
        started = time.time()
        for offset in range(0, len(jobs), batch_size):
            batch = jobs[offset : offset + batch_size]
            padded = self.tok.pad(
                {"input_ids": [job["input_ids"] for job in batch]},
                padding=True,
                return_tensors="pt",
            )
            input_ids = padded["input_ids"].to(self.input_device)
            attention_mask = padded["attention_mask"].to(self.input_device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            keep = max(job["nt"] for job in batch) + 1
            logits = self._forward(input_ids, attention_mask, position_ids, keep)
            logits_device = logits.device
            for batch_index, job in enumerate(batch):
                nt = job["nt"]
                start = keep - nt - 1
                token_logits = logits[batch_index, start : start + nt]
                targets = job["input_ids"][job["prompt_len"] :].to(logits_device)
                token_lp = token_logits.log_softmax(-1).gather(
                    -1, targets[:, None]
                ).squeeze(-1)
                meta = rows[job["row_index"]]["meta"]
                meta[f"lp_{job['key']}"] = float(token_lp.sum().cpu())
                meta[f"nt_{job['key']}"] = nt
            done = min(offset + len(batch), len(jobs))
            if offset == 0 or done == len(jobs) or done % max(batch_size * 25, 1) == 0:
                self.check_budget(f"batch-{done}")
                elapsed = time.time() - started
                eta = (len(jobs) - done) * elapsed / done
                print(
                    f"  behavior {done}/{len(jobs)} elapsed={elapsed:.0f}s "
                    f"eta={eta:.0f}s gpu_mib={gpu_process_mib()}",
                    flush=True,
                )
        return [row["meta"] for row in rows]

    def memory_manifest(self) -> dict:
        self.check_budget("manifest")
        if self.peak_nvidia_mib <= 0:
            raise RuntimeError("No trustworthy nvidia-smi memory observation was recorded")
        if self.peak_device_nvidia_mib <= 0:
            raise RuntimeError("No trustworthy physical-device memory observation was recorded")
        return {
            "user_ceiling_mib": self.max_gpu_mib,
            "loader_budget_mib": self.load_budget_mib,
            "peak_nvidia_mib": self.peak_nvidia_mib,
            "peak_device_nvidia_mib": self.peak_device_nvidia_mib,
            "peak_torch_allocated_mib": torch.cuda.max_memory_allocated(0) / 2**20,
            "peak_torch_reserved_mib": torch.cuda.max_memory_reserved(0) / 2**20,
            "device_map": {k: str(v) for k, v in self.model.hf_device_map.items()},
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
        }

    def close(self) -> None:
        del self.model
        del self.tok
        gc.collect()
        torch.cuda.empty_cache()


def load_stimuli(labels: list[str] | None) -> pd.DataFrame:
    observed_sha256 = hashlib.sha256(STIMULI.read_bytes()).hexdigest()
    if observed_sha256 != STIMULI_SHA256:
        raise RuntimeError(
            f"RQ3 stimuli hash changed: {observed_sha256} != {STIMULI_SHA256}"
        )
    data = pd.read_csv(STIMULI)
    if labels:
        missing = sorted(set(labels) - set(data.label))
        if missing:
            raise ValueError(f"Unknown labels: {missing}")
        data = data[data.label.isin(labels)].copy()
    expected = len(data.label.unique()) * 3
    if len(data) != expected or set(data.prefix) != {"N", "I", "R"}:
        raise RuntimeError("Stimulus grid is not item x {N,I,R}")
    return data


def build_rows(model_name: str, stimuli: pd.DataFrame) -> list[dict]:
    rows = []
    for stimulus in stimuli.itertuples(index=False):
        continuations = {
            "base": " " + stimulus.opt_base,
            "conj_h2": " " + stimulus.opt_c2,
            "conj_h3": " " + stimulus.opt_c3,
            "bare_h2": " " + stimulus.opt_b2,
            "bare_h3": " " + stimulus.opt_b3,
        }
        for task in (2, 3):
            for frame_index, frame in enumerate(FRAME):
                for qvar_index, question in enumerate(QVAR[task]):
                    rows.append(
                        {
                            "meta": dict(
                                model=model_name.split("/")[-1],
                                model_id=model_name,
                                scenario=stimulus.scenario_id,
                                label=stimulus.label,
                                kind=stimulus.kind,
                                paradigm=stimulus.paradigm,
                                prefix=stimulus.prefix,
                                task=task,
                                frame=frame_index,
                                qvar=qvar_index,
                            ),
                            "prompt": behavior_prompt(
                                stimulus.intro, task, frame_index, qvar_index
                            ),
                            "continuations": continuations,
                        }
                    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-gpu-mib", type=int, default=24_000)
    parser.add_argument("--reserve-mib", type=int, default=1_000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and threads must be positive")
    if args.max_gpu_mib > 24_000:
        parser.error("--max-gpu-mib cannot exceed the user's 24,000 MiB ceiling")
    return args


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(args.threads)
    stimuli = load_stimuli(args.labels)
    n_items = stimuli.label.nunique()
    expected = n_items * 3 * 2 * 2 * 6
    slug = safe_name(args.model)
    suffix = "" if not args.labels else "__" + "_".join(args.labels)
    output_path = args.output_dir / "behavior" / f"rq3_behavior_{slug}{suffix}.csv"
    manifest_dir = args.output_dir / "manifests"
    manifest_path = manifest_dir / f"behavior_{slug}{suffix}.json"
    if output_complete(output_path, expected) and manifest_path.exists():
        print(f"SKIP complete {output_path}")
        return 0

    print(
        f"LOAD model={args.model} dtype=float32 items={n_items} "
        f"batch={args.batch_size} gpu_ceiling={args.max_gpu_mib}MiB",
        flush=True,
    )
    scorer = BudgetedScorer(args.model, args.max_gpu_mib, args.reserve_mib)
    started = time.time()
    memory = {}
    try:
        rows = build_rows(args.model, stimuli)
        scored = scorer.score(rows, args.batch_size)
        atomic_csv(scored, output_path, expected)
        memory = scorer.memory_manifest()
    finally:
        scorer.close()

    manifest = {
        "stage": "rq3-behavior",
        "model": args.model,
        "dtype": "float32",
        "cpu_offload_allowed": True,
        "n_items": n_items,
        "labels": args.labels or "all",
        "prefixes": ["N", "I", "R"],
        "rows": expected,
        "batch_size": args.batch_size,
        "elapsed_seconds": time.time() - started,
        "memory": memory,
        "offline": True,
        "stimuli_sha256": STIMULI_SHA256,
        "prompt_contract": PROMPT_CONTRACT_VERSION,
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(manifest_path)
    print(f"COMPLETE {args.model} rows={expected} elapsed={time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
