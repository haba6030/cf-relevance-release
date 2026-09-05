#!/usr/bin/env python3
"""Four-GPU restart-safe queue for the conditional CF rescue experiment."""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from cf_rescue_common import CONTRACT, MODELS, PROJ, safe_name


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--devices", nargs="+", type=int, default=(0, 1, 2, 3))
    ap.add_argument("--max-gpu-mib", type=int, default=24_000)
    ap.add_argument("--num-shards", type=int, default=2)
    args = ap.parse_args()
    root = args.output_dir.resolve(); logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
    status_path = root / "queue_status.json"; lock = threading.Lock()
    status = {
        "state": "running", "stage": "gpu", "started_at": now(), "finished_at": None,
        "experiment_contract": CONTRACT, "models": list(MODELS),
        "devices": args.devices, "max_gpu_mib": args.max_gpu_mib,
        "num_shards": args.num_shards, "eta_hours": 3.8,
        "jobs": [], "guard_jobs": [], "analysis": None, "error": None,
    }

    def save() -> None:
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(status_path)

    jobs: queue.Queue[tuple[str, int]] = queue.Queue()
    # Put one shard of every model in front to avoid a slow model becoming a
    # final single-GPU tail; second shards follow in the same order.
    order = (
        "google/gemma-3-4b-it", "Qwen/Qwen3-4B",
        "Qwen/Qwen3-4B-Instruct-2507", "microsoft/Phi-4-mini-reasoning",
        "Qwen/Qwen2.5-Math-1.5B", "microsoft/Phi-4-mini-instruct",
    )
    for shard in range(args.num_shards):
        for model in order:
            jobs.put((model, shard))
            status["jobs"].append({
                "model": model, "shard": shard, "state": "pending",
                "device": None, "started_at": None, "finished_at": None,
                "exit_code": None, "log": str(logs / f"{safe_name(model)}_shard{shard}.log"),
            })
    save()

    failed = threading.Event()

    def worker(device: int) -> None:
        while not failed.is_set():
            try:
                model, shard = jobs.get_nowait()
            except queue.Empty:
                return
            with lock:
                entry = next(x for x in status["jobs"] if x["model"] == model and x["shard"] == shard)
                entry.update(state="running", device=device, started_at=now()); save()
            command = [
                sys.executable, "-u", str(PROJ / "additional_Qs" / "run_rq3_cf_rescue_gpu.py"),
                "--model", model, "--output-dir", str(root),
                "--shard-index", str(shard), "--num-shards", str(args.num_shards),
                "--max-gpu-mib", str(args.max_gpu_mib),
            ]
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(device)
            env.update({
                "HF_HOME": "<HF_CACHE>",
                "HF_HUB_CACHE": "<HF_CACHE>",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "RQ3_TORCH_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            })
            log_path = Path(entry["log"])
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"[{now()}] START {' '.join(command)}\n"); log.flush()
                result = subprocess.run(command, cwd=PROJ, env=env, stdout=log, stderr=subprocess.STDOUT)
                log.write(f"[{now()}] EXIT {result.returncode}\n")
            with lock:
                entry.update(state="complete" if result.returncode == 0 else "failed",
                             finished_at=now(), exit_code=result.returncode); save()
            jobs.task_done()
            if result.returncode:
                failed.set(); return

    threads = [threading.Thread(target=worker, args=(device,), daemon=False) for device in args.devices]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if failed.is_set():
        with lock:
            status.update(state="failed", stage="gpu", finished_at=now(),
                          error="one or more GPU shards failed"); save()
        return 1

    expected = len(MODELS) * 2 * 165
    observed = len(list((root / "shards").glob("*/task*/*.csv")))
    if observed != expected:
        status.update(state="failed", stage="verify", finished_at=now(),
                      error=f"shard count {observed} != {expected}"); save(); return 1

    # The fixed WikiText guard is also item-sharded and restart-safe.  Run it
    # only after all behavioral shards so status cleanly distinguishes stages.
    status["stage"] = "fluency_guard"; save()
    guard_queue: queue.Queue[tuple[str, int]] = queue.Queue()
    for shard in range(args.num_shards):
        for model in order:
            guard_queue.put((model, shard))
            status["guard_jobs"].append({
                "model": model, "shard": shard, "state": "pending",
                "device": None, "started_at": None, "finished_at": None,
                "exit_code": None,
                "log": str(logs / f"guard_{safe_name(model)}_shard{shard}.log"),
            })
    save()
    guard_failed = threading.Event()

    def guard_worker(device: int) -> None:
        while not guard_failed.is_set():
            try:
                model, shard = guard_queue.get_nowait()
            except queue.Empty:
                return
            with lock:
                entry = next(x for x in status["guard_jobs"]
                             if x["model"] == model and x["shard"] == shard)
                entry.update(state="running", device=device, started_at=now()); save()
            command = [
                sys.executable, "-u",
                str(PROJ / "additional_Qs" / "run_rq3_cf_rescue_guard_gpu.py"),
                "--model", model, "--output-dir", str(root),
                "--shard-index", str(shard), "--num-shards", str(args.num_shards),
                "--max-gpu-mib", str(args.max_gpu_mib),
            ]
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(device)
            env.update({
                "HF_HOME": "<HF_CACHE>",
                "HF_HUB_CACHE": "<HF_CACHE>",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "RQ3_TORCH_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            })
            log_path = Path(entry["log"])
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"[{now()}] START {' '.join(command)}\n"); log.flush()
                result = subprocess.run(
                    command, cwd=PROJ, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                log.write(f"[{now()}] EXIT {result.returncode}\n")
            with lock:
                entry.update(state="complete" if result.returncode == 0 else "failed",
                             finished_at=now(), exit_code=result.returncode); save()
            guard_queue.task_done()
            if result.returncode:
                guard_failed.set(); return

    threads = [threading.Thread(target=guard_worker, args=(device,), daemon=False)
               for device in args.devices]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if guard_failed.is_set():
        status.update(state="failed", stage="fluency_guard", finished_at=now(),
                      error="one or more fluency guard shards failed"); save(); return 1
    observed_guards = len(list((root / "guard_shards").glob("*/task*/*.csv")))
    if observed_guards != expected:
        status.update(state="failed", stage="verify_guard", finished_at=now(),
                      error=f"guard shard count {observed_guards} != {expected}"); save(); return 1
    status["stage"] = "analysis"; save()
    analysis_log = logs / "analysis.log"
    command = [
        sys.executable, str(PROJ / "additional_Qs" / "analysis" /
                            "16_analyze_cf_rescue_knockout.py"),
        "--input-dir", str(root), "--project", str(PROJ),
    ]
    with analysis_log.open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJ, stdout=log, stderr=subprocess.STDOUT)
    status["analysis"] = {"exit_code": result.returncode, "log": str(analysis_log)}
    if result.returncode:
        status.update(state="failed", stage="analysis", finished_at=now(),
                      error="analysis failed"); save(); return 1
    status.update(state="complete", stage="complete", finished_at=now()); save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
