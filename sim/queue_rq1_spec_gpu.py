#!/usr/bin/env python3
"""Restartable four-GPU queue for the frozen RQ1/H2 specification."""

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
    "Qwen/Qwen2.5-Math-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "microsoft/Phi-4-mini-instruct",
    "microsoft/Phi-4-mini-reasoning",
    "google/gemma-3-4b-it",
]


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_name(model: str) -> str:
    return model.replace("/", "__")


def write_status(status: dict, path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def completed(output_dir: Path, model: str) -> bool:
    slug = safe_name(model)
    checks = [
        (output_dir / "logprob" / f"rq1_logprob_{slug}.csv", 721),
        (output_dir / "forcedchoice" / f"rq1_forcedchoice_{slug}.csv", 3601),
        (output_dir / "norming" / f"norm_{slug}.csv", 181),
    ]
    for path, lines in checks:
        if not path.exists():
            return False
        try:
            if sum(1 for _ in path.open(encoding="utf-8")) != lines:
                return False
        except OSError:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parent.parent
    runner = project / "sim" / "run_rq1_spec_gpu.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    status_path = args.output_dir / "queue_status.json"
    status = {
        "started_at": stamp(),
        "finished_at": None,
        "pid": os.getpid(),
        "condition": (
            "docs/EXPERIMENT_spec.md frozen grid; active15; CUDA fp16; "
            "L2/L3 x conjunction/bare x logprob/forced-choice; letter/content; norming"
        ),
        "requested_count_text": 7,
        "resolved_model_count": len(args.models),
        "count_note": "The explicit list contains eight models; all eight are queued.",
        "devices": args.devices,
        "models": {model: {"state": "queued", "attempts": 0} for model in args.models},
    }
    for model in args.models:
        if completed(args.output_dir, model):
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

    pending = [model for model in args.models if not completed(args.output_dir, model)]
    free_devices = list(args.devices)
    running: dict[int, dict] = {}
    failures = 0

    while pending or running:
        while pending and free_devices:
            model = pending.pop(0)
            device = free_devices.pop(0)
            entry = status["models"][model]
            entry["attempts"] += 1
            log_path = logs / f"{safe_name(model)}.log"
            command = [
                sys.executable,
                "-u",
                str(runner),
                "--model",
                model,
                "--output-dir",
                str(args.output_dir),
                "--device",
                "cuda:0",
                "--batch-size",
                str(args.batch_size),
                "--threads",
                str(args.threads_per_worker),
            ]
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            log = log_path.open("a", encoding="utf-8")
            log.write(f"\n[{stamp()}] START gpu={device} {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[process.pid] = {
                "process": process,
                "model": model,
                "device": device,
                "log": log,
                "log_path": str(log_path),
            }
            entry.update(
                {
                    "state": "running",
                    "started_at": stamp(),
                    "gpu": device,
                    "pid": process.pid,
                    "log": str(log_path),
                }
            )
            write_status(status, status_path)

        if not running:
            break
        time.sleep(10)
        for pid, job in list(running.items()):
            code = job["process"].poll()
            if code is None:
                continue
            job["log"].write(f"[{stamp()}] EXIT {code}\n")
            job["log"].close()
            model = job["model"]
            device = job["device"]
            entry = status["models"][model]
            entry.update({"finished_at": stamp(), "exit_code": code})
            if code == 0 and completed(args.output_dir, model):
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

    status["finished_at"] = stamp()
    status["failure_count"] = failures
    if failures == 0:
        analysis_log_path = logs / "analysis.log"
        analysis_command = [
            sys.executable,
            str(project / "sim" / "analyze_rq1_spec.py"),
            "--input-dir",
            str(args.output_dir),
        ]
        status["analysis"] = {
            "state": "running", "started_at": stamp(), "log": str(analysis_log_path)
        }
        write_status(status, status_path)
        with analysis_log_path.open("a", encoding="utf-8") as analysis_log:
            analysis = subprocess.run(
                analysis_command,
                cwd=project,
                env=env_base,
                stdout=analysis_log,
                stderr=subprocess.STDOUT,
            )
        status["analysis"].update(
            {
                "state": "complete" if analysis.returncode == 0 else "failed",
                "finished_at": stamp(),
                "exit_code": analysis.returncode,
            }
        )
        failures += int(analysis.returncode != 0)
    status["failure_count"] = failures
    status["state"] = "complete" if failures == 0 else "complete_with_failures"
    write_status(status, status_path)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
