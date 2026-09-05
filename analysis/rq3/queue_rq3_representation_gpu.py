#!/usr/bin/env python3
"""Four-GPU RQ3 representation queue, restricted by the H3-0 gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from rq3_prompting import PROMPT_CONTRACT_VERSION
except ModuleNotFoundError:
    from sim.rq3_prompting import PROMPT_CONTRACT_VERSION


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slug(model: str) -> str:
    return model.replace("/", "__")


def complete(output_dir: Path, model: str) -> bool:
    s = slug(model)
    checks = [
        output_dir / "norming" / f"norm_rq3_{s}.csv",
        output_dir / "activations" / s / "task2_cellmean.npy",
        output_dir / "activations" / s / "task3_cellmean.npy",
        output_dir / "activations" / s / "task2_prompt_cellmean.npy",
        output_dir / "activations" / s / "task3_prompt_cellmean.npy",
        output_dir / "activations" / s / "task2_variant_sd_norm.npy",
        output_dir / "activations" / s / "task3_variant_sd_norm.npy",
        output_dir / "activations" / s / "task2_prompt_variant_sd_norm.npy",
        output_dir / "activations" / s / "task3_prompt_variant_sd_norm.npy",
        output_dir / "manifests" / f"representation_{s}.json",
    ]
    if not all(path.exists() for path in checks):
        return False
    try:
        manifest = json.loads(checks[-1].read_text())
        norm_rows = sum(1 for _ in checks[0].open(encoding="utf-8")) - 1
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("prompt_contract") == PROMPT_CONTRACT_VERSION
        and norm_rows == 540
    )


def gpu_used_mib() -> dict[str, int]:
    result = None
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=10,
            )
            break
        except (OSError, subprocess.SubprocessError) as error:
            print(f"WARNING nvidia-smi memory query failed ({attempt + 1}/3): {error}", flush=True)
            time.sleep(2)
    if result is None:
        return {}
    output = {}
    for line in result.stdout.splitlines():
        index, used = [part.strip() for part in line.split(",")]
        output[index] = int(used)
    return output


def write_status(status: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--gate-file", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--max-gpu-mib", type=int, default=24_000)
    args = parser.parse_args()
    if args.max_gpu_mib > 24_000:
        raise ValueError("GPU ceiling cannot exceed 24,000 MiB")
    project = Path(__file__).resolve().parents[2]
    models = json.loads(args.gate_file.read_text())["eligible_models"]
    if not models:
        print("No H3-0 eligible models; representation queue has nothing to run")
        return 0
    status_path = args.output_dir / "representation_queue_status.json"
    status = {
        "stage": "rq3-representation",
        "started_at": stamp(),
        "finished_at": None,
        "gpu_ceiling_mib": args.max_gpu_mib,
        "models": {model: {"state": "complete_existing" if complete(args.output_dir, model) else "queued", "attempts": 0} for model in models},
    }
    write_status(status, status_path)
    env_base = os.environ.copy()
    env_base.update({
        "HF_HOME": str(args.hf_home), "HF_HUB_CACHE": str(args.hf_home / "hub"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    pending = [model for model in models if not complete(args.output_dir, model)]
    free = list(args.devices)
    running = {}
    failures = 0
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    while pending or running:
        while pending and free:
            usage = gpu_used_mib()
            idle = next((device for device in free if usage.get(device, 10**9) <= 500), None)
            if idle is None:
                status["waiting_for_idle_gpu"] = {"at": stamp(), "used_mib": usage}
                write_status(status, status_path)
                break
            model, device = pending.pop(0), idle
            free.remove(device)
            entry = status["models"][model]
            entry["attempts"] += 1
            log_path = logs / f"representation_{slug(model)}.log"
            behavior = args.output_dir / "behavior" / f"rq3_behavior_{slug(model)}.csv"
            command = [
                sys.executable, "-u", str(project / "analysis" / "rq3" / "run_rq3_representation_gpu.py"),
                "--model", model, "--output-dir", str(args.output_dir),
                "--behavior-file", str(behavior), "--batch-size", "16", "--threads", "4",
                "--max-gpu-mib", str(args.max_gpu_mib),
            ]
            env = env_base.copy(); env["CUDA_VISIBLE_DEVICES"] = device
            log = log_path.open("a", encoding="utf-8")
            log.write(f"\n[{stamp()}] START gpu={device} {' '.join(command)}\n"); log.flush()
            process = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[process.pid] = dict(process=process, model=model, device=device, log=log)
            entry.update(state="running", started_at=stamp(), gpu=device, pid=process.pid, log=str(log_path))
            write_status(status, status_path)
        if not running:
            if pending:
                time.sleep(30)
                continue
            break
        time.sleep(10)
        for pid, job in list(running.items()):
            code = job["process"].poll()
            if code is None: continue
            job["log"].write(f"[{stamp()}] EXIT {code}\n"); job["log"].close()
            entry = status["models"][job["model"]]
            entry.update(finished_at=stamp(), exit_code=code)
            if code == 0 and complete(args.output_dir, job["model"]): entry["state"] = "complete"
            elif entry["attempts"] <= 1: entry["state"] = "retry_queued"; pending.append(job["model"])
            else: entry["state"] = "failed"; failures += 1
            free.append(job["device"]); free.sort(key=args.devices.index)
            del running[pid]; write_status(status, status_path)
    if failures == 0:
        analysis_log = logs / "rq3a_analysis.log"
        status["analysis"] = {"state": "running", "started_at": stamp(), "log": str(analysis_log)}
        write_status(status, status_path)
        with analysis_log.open("a", encoding="utf-8") as handle:
            # `analyze_rq3_representation.py` intentionally exits because it
            # creates the retired one-positive/three-negative decoder.  New
            # representation runs must produce only the contract-validated
            # R,h3-versus-I,h3 gate.
            commands = [
                [sys.executable, str(project / "analysis" / "rq3" / "analyze_rq3_context_decoding_robustness.py"), "--input-dir", str(args.output_dir)],
            ]
            return_codes = []
            for command in commands:
                handle.write(f"\nRUN {' '.join(command)}\n"); handle.flush()
                completed = subprocess.run(command, cwd=project, env=env_base, stdout=handle, stderr=subprocess.STDOUT)
                return_codes.append(completed.returncode)
                if completed.returncode:
                    break
            result = type("AnalysisResult", (), {"returncode": max(return_codes, default=1)})()
        status["analysis"].update(state="complete" if result.returncode == 0 else "failed", finished_at=stamp(), exit_code=result.returncode)
        failures += int(result.returncode != 0)
    status.update(finished_at=stamp(), failure_count=failures, state="complete" if failures == 0 else "complete_with_failures")
    write_status(status, status_path)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
