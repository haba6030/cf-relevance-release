#!/usr/bin/env python3
"""Logout-safe four-GPU queue for the 0901 P/C manipulation check."""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_rq3_norm_manipulation_gpu import CONTRACT, PROJ, safe_name, target_grid


PYTHON = Path(sys.executable)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def gpu_env(device: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(device),
        "HF_HOME": "<HF_CACHE>",
        "HF_HUB_CACHE": "<HF_CACHE>",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "RQ3_TORCH_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def runner_command(root: Path, model: str, shard: int, count: int, ceiling: int,
                   *, limit: int | None = None) -> list[str]:
    command = [
        str(PYTHON), "-u", str(PROJ / "additional_Qs" / "run_rq3_norm_manipulation_gpu.py"),
        "--model", model, "--output-dir", str(root),
        "--shard-index", str(shard), "--num-shards", str(count),
        "--max-gpu-mib", str(ceiling),
    ]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--max-gpu-mib", type=int, default=24_000)
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    if args.max_gpu_mib > 24_000 or args.num_shards < 1:
        raise ValueError("Invalid GPU ceiling or shard count")

    root = args.output_dir.resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = root / "queue_status.json"
    grid = target_grid(True)
    models = list(dict.fromkeys(grid.model_id.tolist()))
    status = {
        "state": "running", "stage": "smoke", "started_at": now(), "finished_at": None,
        "experiment_contract": CONTRACT, "source_commit": "1581758d9c972e1249024cd5afaee924ba510884",
        "devices": args.devices, "max_gpu_mib": args.max_gpu_mib,
        "num_shards": args.num_shards, "models": models,
        "target_strength_cells": len(grid), "primary_strength_cells": 19,
        "posthoc_strength_cells": 6, "smoke": {}, "jobs": [], "analysis": None,
        "eta": {"basis": "pending smoke measurement", "estimated_finish": None},
        "error": None,
    }
    atomic_json(status_path, status)

    # One model, one held item.  The locked first item has an eligible task-3
    # ADD arm for Qwen2.5-Math, so this checks a real edited cell, not just load.
    smoke_root = root / "smoke"
    smoke_model = "Qwen/Qwen2.5-Math-1.5B"
    smoke_log = logs / "smoke_qwen_math_item0.log"
    smoke_command = runner_command(smoke_root, smoke_model, 0, 1, args.max_gpu_mib, limit=1)
    smoke_started = time.monotonic()
    with smoke_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] START {' '.join(smoke_command)}\n"); handle.flush()
        result = subprocess.run(smoke_command, cwd=PROJ, env=gpu_env(args.devices[0]),
                                stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[{now()}] EXIT {result.returncode}\n")
    smoke_seconds = time.monotonic() - smoke_started
    smoke_rows = 0
    for path in (smoke_root / "shards").glob("*/task*/*.csv"):
        try:
            smoke_rows += len(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            pass
    status["smoke"] = {
        "state": "complete" if result.returncode == 0 and smoke_rows > 0 else "failed",
        "model": smoke_model, "held_index": 0, "elapsed_seconds": round(smoke_seconds, 3),
        "rows": smoke_rows, "exit_code": result.returncode, "log": str(smoke_log),
    }
    if result.returncode or smoke_rows <= 0:
        status.update(state="failed", stage="smoke", finished_at=now(),
                      error="one-item smoke produced no edited rows or exited nonzero")
        atomic_json(status_path, status)
        return 1

    status["stage"] = "gpu"
    # Smoke includes model load and is deliberately pessimistic as an item-rate
    # estimate.  Keep it in status, but replace the completion ETA with measured
    # full-shard throughput as soon as jobs finish.
    status["eta"] = {
        "basis": "one-item smoke (load-inclusive upper bound; pending full shard)",
        "estimated_finish": None, "smoke_seconds": round(smoke_seconds, 3),
    }

    jobs: queue.Queue[tuple[str, int]] = queue.Queue()
    order = [
        "Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Instruct-2507",
        "microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning",
        "Qwen/Qwen2.5-Math-1.5B",
    ]
    for shard in range(args.num_shards):
        for model in order:
            if model not in models:
                continue
            jobs.put((model, shard))
            status["jobs"].append({
                "model": model, "shard": shard, "state": "pending", "device": None,
                "started_at": None, "finished_at": None, "elapsed_seconds": None,
                "attempts": 0, "exit_code": None,
                "log": str(logs / f"{safe_name(model)}_shard{shard}.log"),
            })
    atomic_json(status_path, status)
    lock = threading.Lock()
    failed = threading.Event()

    def save() -> None:
        atomic_json(status_path, status)

    def update_eta() -> None:
        completed = [j for j in status["jobs"] if j["state"] == "complete"]
        if not completed:
            return
        durations = [float(j["elapsed_seconds"]) for j in completed]
        pending_count = sum(j["state"] in {"pending", "running"} for j in status["jobs"])
        seconds = (pending_count / max(len(args.devices), 1)) * float(sum(durations) / len(durations))
        finish = datetime.now().astimezone().timestamp() + seconds
        status["eta"] = {
            "basis": f"mean of {len(completed)} completed full shards",
            "remaining_seconds": round(seconds),
            "estimated_finish": datetime.fromtimestamp(finish).astimezone().isoformat(timespec="seconds"),
        }

    def worker(device: int) -> None:
        while not failed.is_set():
            try:
                model, shard = jobs.get_nowait()
            except queue.Empty:
                return
            with lock:
                entry = next(j for j in status["jobs"] if j["model"] == model and j["shard"] == shard)
                entry.update(state="running", device=device, started_at=now(), attempts=entry["attempts"] + 1)
                save()
            command = runner_command(root, model, shard, args.num_shards, args.max_gpu_mib)
            started = time.monotonic()
            log_path = Path(entry["log"])
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{now()}] START {' '.join(command)}\n"); handle.flush()
                result = subprocess.run(command, cwd=PROJ, env=gpu_env(device),
                                        stdout=handle, stderr=subprocess.STDOUT)
                handle.write(f"[{now()}] EXIT {result.returncode}\n")
            elapsed = time.monotonic() - started
            with lock:
                entry.update(state="complete" if result.returncode == 0 else "failed",
                             finished_at=now(), elapsed_seconds=round(elapsed, 3),
                             exit_code=result.returncode)
                update_eta(); save()
            jobs.task_done()
            if result.returncode:
                failed.set()
                return

    threads = [threading.Thread(target=worker, args=(device,), daemon=False) for device in args.devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failed.is_set():
        status.update(state="failed", stage="gpu", finished_at=now(), error="GPU shard failed")
        save()
        return 1

    expected = int(grid[["model_id", "task"]].drop_duplicates().shape[0] * 165)
    observed = len(list((root / "shards").glob("*/task*/*.csv")))
    manifests = len(list((root / "manifests").glob("*.json")))
    if observed != expected or manifests != len(status["jobs"]):
        status.update(state="failed", stage="verify", finished_at=now(),
                      error=f"shards {observed}/{expected}; manifests {manifests}/{len(status['jobs'])}")
        save()
        return 1

    status["stage"] = "analysis"
    save()
    analysis_log = logs / "analysis.log"
    command = [str(PYTHON), str(PROJ / "additional_Qs" / "analysis" /
                                "19_analyze_norm_manipulation.py"), "--input-dir", str(root)]
    with analysis_log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] START {' '.join(command)}\n"); handle.flush()
        result = subprocess.run(command, cwd=PROJ, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[{now()}] EXIT {result.returncode}\n")
    status["analysis"] = {"exit_code": result.returncode, "log": str(analysis_log)}
    if result.returncode:
        status.update(state="failed", stage="analysis", finished_at=now(), error="analysis failed")
        save()
        return 1
    status.update(state="complete", stage="complete", finished_at=now(), eta={
        "basis": "completed", "remaining_seconds": 0, "estimated_finish": now(),
    })
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
