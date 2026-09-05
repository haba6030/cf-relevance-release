#!/usr/bin/env python3
"""Restartable four-GPU queue for RQ3 behavioral step and H3-0 gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MODELS = [
    "microsoft/Phi-4-mini-reasoning",
    "google/gemma-3-4b-it",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen2.5-Math-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]


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
        # Unknown usage is never treated as idle: the caller's missing-device
        # default blocks launches until a trustworthy query succeeds.
        return {}
    output = {}
    for line in result.stdout.splitlines():
        index, used = [part.strip() for part in line.split(",")]
        output[index] = int(used)
    return output


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    return value.replace("/", "__")


def complete(output_dir: Path, model: str) -> bool:
    path = output_dir / "behavior" / f"rq3_behavior_{safe_name(model)}.csv"
    manifest = output_dir / "manifests" / f"behavior_{safe_name(model)}.json"
    return path.exists() and manifest.exists() and sum(1 for _ in path.open(encoding="utf-8")) == 1081


def write_status(status: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--max-gpu-mib", type=int, default=24_000)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_gpu_mib > 24_000:
        raise ValueError("GPU ceiling cannot exceed 24,000 MiB")
    project = Path(__file__).resolve().parents[2]
    runner = project / "analysis" / "rq3" / "run_rq3_behavior_gpu.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    status_path = args.output_dir / "queue_status.json"
    status = {
        "stage": "rq3-behavior-h30",
        "started_at": stamp(),
        "finished_at": None,
        "pid": os.getpid(),
        "devices": args.devices,
        "gpu_ceiling_mib": args.max_gpu_mib,
        "dtype": "float32-with-cpu-offload",
        "models": {model: {"state": "queued", "attempts": 0} for model in args.models},
    }
    for model in args.models:
        if complete(args.output_dir, model):
            status["models"][model]["state"] = "complete_existing"
    write_status(status, status_path)

    env_base = os.environ.copy()
    env_base.update(
        {
            "HF_HOME": str(args.hf_home),
            "HF_HUB_CACHE": str(args.hf_home / "hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    pending = [model for model in args.models if not complete(args.output_dir, model)]
    free_devices = list(args.devices)
    running: dict[int, dict] = {}
    failures = 0
    while pending or running:
        while pending and free_devices:
            usage = gpu_used_mib()
            idle = next((device for device in free_devices if usage.get(device, 10**9) <= 500), None)
            if idle is None:
                status["waiting_for_idle_gpu"] = {"at": stamp(), "used_mib": usage}
                write_status(status, status_path)
                break
            model = pending.pop(0)
            device = idle
            free_devices.remove(device)
            entry = status["models"][model]
            entry["attempts"] += 1
            log_path = logs / f"behavior_{safe_name(model)}.log"
            command = [
                sys.executable,
                "-u",
                str(runner),
                "--model",
                model,
                "--output-dir",
                str(args.output_dir),
                "--batch-size",
                str(args.batch_size),
                "--threads",
                str(args.threads_per_worker),
                "--max-gpu-mib",
                str(args.max_gpu_mib),
            ]
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            log = log_path.open("a", encoding="utf-8")
            log.write(f"\n[{stamp()}] START gpu={device} {' '.join(command)}\n")
            log.flush()
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
            if code is None:
                continue
            job["log"].write(f"[{stamp()}] EXIT {code}\n")
            job["log"].close()
            model, device = job["model"], job["device"]
            entry = status["models"][model]
            entry.update(finished_at=stamp(), exit_code=code)
            if code == 0 and complete(args.output_dir, model):
                entry["state"] = "complete"
            elif entry["attempts"] <= args.retries:
                entry["state"] = "retry_queued"
                pending.append(model)
            else:
                entry["state"] = "failed"
                failures += 1
            free_devices.append(device)
            free_devices.sort(key=args.devices.index)
            del running[pid]
            write_status(status, status_path)

    if failures == 0:
        analysis_log_path = logs / "h30_analysis.log"
        status["analysis"] = {"state": "running", "started_at": stamp(), "log": str(analysis_log_path)}
        write_status(status, status_path)
        with analysis_log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(project / "analysis" / "rq3" / "analyze_rq3_h30.py"), "--input-dir", str(args.output_dir)],
                cwd=project,
                env=env_base,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        status["analysis"].update(
            state="complete" if result.returncode == 0 else "failed",
            finished_at=stamp(),
            exit_code=result.returncode,
        )
        failures += int(result.returncode != 0)
    status.update(
        finished_at=stamp(),
        failure_count=failures,
        state="complete" if failures == 0 else "complete_with_failures",
    )
    write_status(status, status_path)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
