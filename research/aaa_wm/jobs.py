"""A tiny RAM-safe job runner for long experiment queues on FLOWBOX.

Each job runs through ``capped.sh`` (a cgroup scope with MemoryMax, no swap, low CPU/IO
priority). At most ``--cpu`` CPU jobs and ``--gpu`` GPU jobs run at once, the sum of running
caps never exceeds ``--budget``, and a job starts only when its ``needs`` files exist. Finished
jobs (their ``output`` exists) are skipped, so a queue can be resumed after an interruption.

    python -m research.aaa_wm.jobs queue.json --cpu 2 --gpu 1 --budget 9G
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _bytes(s: str) -> int:
    return int(float(s[:-1]) * {"M": 2**20, "G": 2**30}[s[-1]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--cpu", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--budget", default="9G")
    args = ap.parse_args()
    jobs = json.loads(Path(args.queue).read_text())
    running: dict[str, tuple[subprocess.Popen[bytes], dict]] = {}
    pending = [j for j in jobs if not Path(j["output"]).exists()]
    budget = _bytes(args.budget)
    while pending or running:
        for name, (proc, job) in list(running.items()):
            if proc.poll() is not None:
                print(time.strftime("%H:%M:%S"), "done", name, "exit", proc.returncode, flush=True)
                del running[name]
        used = sum(_bytes(j["mem"]) for _, j in running.values())
        for job in list(pending):
            kind = job.get("kind", "cpu")
            n_kind = sum(1 for _, j in running.values() if j.get("kind", "cpu") == kind)
            if n_kind >= (args.gpu if kind == "gpu" else args.cpu) or used + _bytes(job["mem"]) > budget:
                continue
            if not all(Path(p).exists() for p in job.get("needs", [])):
                continue
            cmd = [str(HERE / "capped.sh"), job["mem"], job["log"], *job["cmd"]]
            running[job["name"]] = (subprocess.Popen(cmd, cwd=job.get("cwd")), job)
            used += _bytes(job["mem"])
            pending.remove(job)
            print(time.strftime("%H:%M:%S"), "start", job["name"], flush=True)
        time.sleep(20)
    print("QUEUE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
