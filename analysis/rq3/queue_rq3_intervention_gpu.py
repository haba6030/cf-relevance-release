#!/usr/bin/env python3
"""Four-GPU RQ3-B queue: primary ABL/ADD, DIS, then analyses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from rq3_prompting import PROMPT_CONTRACT_VERSION
    from rq3_decoder_contract import H3_DECODER_CONTRACT, H3_GATE_FAMILY, load_h3_decoder_gate
except ModuleNotFoundError:
    from sim.rq3_prompting import PROMPT_CONTRACT_VERSION
    from sim.rq3_decoder_contract import H3_DECODER_CONTRACT, H3_GATE_FAMILY, load_h3_decoder_gate


def stamp() -> str: return datetime.now().astimezone().isoformat(timespec="seconds")
def slug(model: str) -> str: return model.replace("/", "__")


def gpu_used_mib() -> dict[str, int]:
    result = None
    for attempt in range(3):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
            break
        except (OSError, subprocess.SubprocessError) as error:
            print(f"WARNING nvidia-smi memory query failed ({attempt + 1}/3): {error}", flush=True)
            time.sleep(2)
    if result is None:
        return {}
    return {parts[0]: int(parts[1]) for parts in ([x.strip() for x in line.split(",")] for line in result.stdout.splitlines())}


def write_status(status: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n"); tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--gate-file", type=Path, required=True); parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--direction-mode", default="h3_subspace",
                        choices=("h3_context", "h3_subspace"))
    parser.add_argument("--ablate-scope", default="selected", choices=("all", "selected", "both"))
    parser.add_argument("--subspace-k", type=int, default=2)
    parser.add_argument("--random-subspaces", type=int, default=8)
    parser.add_argument("--max-gpu-mib", type=int, default=24_000); args = parser.parse_args()
    if args.max_gpu_mib > 24_000: raise ValueError("GPU ceiling cannot exceed 24,000 MiB")
    project = Path(__file__).resolve().parents[2]
    gate = load_h3_decoder_gate(args.gate_file, require_both_tasks=True)
    eligible = gate[gate.decode_gate.astype(bool)]
    models = sorted(eligible.model_id.unique().tolist())
    if not models: print("No decode-gate models; intervention queue has nothing to run"); return 0
    def complete(model: str) -> bool:
        manifests = [
            args.output_dir / "manifests" / f"{stage}_{slug(model)}.json"
            for stage in ("intervention", "dis", "fluency")
        ]
        if not all(path.exists() for path in manifests):
            return False
        try:
            loaded = [json.loads(path.read_text()) for path in manifests]
            if not all(item.get("prompt_contract") == PROMPT_CONTRACT_VERSION for item in loaded):
                return False
            if not all(item.get("decoder_contract") == H3_DECODER_CONTRACT for item in loaded):
                return False
            intervention = next((item for item in loaded if "intervention" in str(item.get("stage"))), {})
            fluency = next((item for item in loaded if "fluency" in str(item.get("stage"))), {})
            for item in (intervention, fluency):
                if item.get("direction_mode") != args.direction_mode:
                    return False
                if item.get("subspace_k") != args.subspace_k:
                    return False
                if item.get("random_subspaces") != args.random_subspaces:
                    return False
                if item.get("ablate_scope") != args.ablate_scope:
                    return False
        except (OSError, json.JSONDecodeError):
            return False
        return (
            (args.output_dir / "guards" / slug(model) / "fluency.csv").exists()
            and (args.output_dir / "guards" / slug(model) / "fluency_folds.csv").exists()
        )
    status_path = args.output_dir / "intervention_queue_status.json"
    status = {"stage": "rq3-intervention", "started_at": stamp(), "finished_at": None, "gpu_ceiling_mib": args.max_gpu_mib, "models": {m: {"state": "complete_existing" if complete(m) else "queued", "attempts": 0} for m in models}}
    write_status(status, status_path)
    env_base = os.environ.copy(); env_base.update({"HF_HOME": str(args.hf_home), "HF_HUB_CACHE": str(args.hf_home / "hub"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    pending, free, running, failures = [m for m in models if not complete(m)], list(args.devices), {}, 0
    logs = args.output_dir / "logs"; logs.mkdir(exist_ok=True)
    while pending or running:
        while pending and free:
            usage = gpu_used_mib(); idle = next((d for d in free if usage.get(d, 10**9) <= 500), None)
            if idle is None: status["waiting_for_idle_gpu"] = {"at": stamp(), "used_mib": usage}; write_status(status, status_path); break
            model = pending.pop(0); free.remove(idle); entry = status["models"][model]; entry["attempts"] += 1
            log_path = logs / f"intervention_{slug(model)}.log"; handle = log_path.open("a", encoding="utf-8")
            # A separate worker script is unnecessary: invoke the three Python stages via
            # this interpreter in a subprocess command list handled by bash -c.
            command_text = (
                f"{sys.executable} -u {project / 'analysis' / 'rq3' / 'run_rq3_fluency_guard_gpu.py'} --model {model} --output-dir {args.output_dir} --max-gpu-mib {args.max_gpu_mib} --decoder-gate-file {args.gate_file} --gate-family {H3_GATE_FAMILY} --direction-mode {args.direction_mode} --subspace-k {args.subspace_k} --random-subspaces {args.random_subspaces} --ablate-scope {args.ablate_scope} && "
                f"{sys.executable} -u {project / 'analysis' / 'rq3' / 'run_rq3_intervention_gpu.py'} --model {model} --output-dir {args.output_dir} --max-gpu-mib {args.max_gpu_mib} --decoder-gate-file {args.gate_file} --gate-family {H3_GATE_FAMILY} --direction-mode {args.direction_mode} --subspace-k {args.subspace_k} --random-subspaces {args.random_subspaces} --ablate-scope {args.ablate_scope} && "
                f"{sys.executable} -u {project / 'analysis' / 'rq3' / 'run_rq3_dis_gpu.py'} --model {model} --output-dir {args.output_dir} --max-gpu-mib {args.max_gpu_mib}"
            )
            env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = idle
            handle.write(f"\n[{stamp()}] START gpu={idle} {command_text}\n"); handle.flush()
            process = subprocess.Popen(["bash", "-lc", command_text], cwd=project, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[process.pid] = dict(process=process, model=model, device=idle, log=handle)
            entry.update(state="running", started_at=stamp(), gpu=idle, pid=process.pid, log=str(log_path)); write_status(status, status_path)
        if not running:
            if pending: time.sleep(30); continue
            break
        time.sleep(10)
        for pid, job in list(running.items()):
            code = job["process"].poll()
            if code is None: continue
            job["log"].write(f"[{stamp()}] EXIT {code}\n"); job["log"].close(); entry = status["models"][job["model"]]; entry.update(finished_at=stamp(), exit_code=code)
            if code == 0 and complete(job["model"]): entry["state"] = "complete"
            elif entry["attempts"] <= 1: entry["state"] = "retry_queued"; pending.append(job["model"])
            else: entry["state"] = "failed"; failures += 1
            free.append(job["device"]); free.sort(key=args.devices.index); del running[pid]; write_status(status, status_path)
    if failures == 0:
        analysis_results = {}
        for name, script in (("primary", "analyze_rq3_intervention.py"), ("dis", "analyze_rq3_dis.py")):
            log_path = logs / f"rq3b_{name}_analysis.log"
            with log_path.open("a", encoding="utf-8") as handle:
                result = subprocess.run([sys.executable, str(project / "analysis" / "rq3" / script), "--input-dir", str(args.output_dir)], cwd=project, env=env_base, stdout=handle, stderr=subprocess.STDOUT)
            analysis_results[name] = {"exit_code": result.returncode, "log": str(log_path)}; failures += int(result.returncode != 0)
        status["analysis"] = analysis_results
    status.update(finished_at=stamp(), failure_count=failures, state="complete" if failures == 0 else "complete_with_failures"); write_status(status, status_path)
    return int(failures > 0)


if __name__ == "__main__": raise SystemExit(main())
