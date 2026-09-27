"""A tiny RAM-safe job runner for long experiment queues on FLOWBOX.

Each job runs through ``capped.sh`` (a cgroup scope with MemoryMax, no swap, low CPU/IO
priority). At most ``--cpu`` CPU jobs and ``--gpu`` GPU jobs run at once, the sum of running
caps never exceeds ``--budget``, and a job starts only when its ``needs`` files exist. Finished
jobs (their ``output`` exists) are skipped, so a queue can be resumed after an interruption.
The queue file is re-read every loop, so jobs can be appended while it runs. A running job holds
``<output>.lock`` with its PID, and a restarted runner never starts a job whose lock names a
live process.

    python -m research.aaa_wm.jobs queue.json --cpu 2 --gpu 1 --budget 9G
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _bytes(s: str) -> int:
    return int(float(s[:-1]) * {"M": 2**20, "G": 2**30}[s[-1]])


class Adopted:
    """A job started by an earlier runner (its lock names a live PID): counted and polled, not restarted."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            self.returncode = 0
            return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--cpu", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--budget", default="9G")
    ap.add_argument(
        "--vram", default="5000M", help="GPU memory budget; a job's 'vram' field (default 1G for gpu jobs)"
    )
    args = ap.parse_args()
    running: dict[str, tuple[subprocess.Popen[bytes], dict]] = {}
    started: set[str] = set()
    budget = _bytes(args.budget)
    vbudget = _bytes(args.vram)

    def vram(job: dict) -> int:
        return _bytes(job.get("vram", "1G")) if job.get("kind") == "gpu" else 0

    def locked(job: dict) -> bool:
        lock = Path(job["output"] + ".lock")
        if not lock.exists():
            return False
        try:
            os.kill(int(lock.read_text().strip()), 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            return False

    def refresh() -> list[dict]:
        jobs = json.loads(Path(args.queue).read_text())
        return [
            j for j in jobs if j["name"] not in started and not Path(j["output"]).exists() and not locked(j)
        ]

    for job in json.loads(Path(args.queue).read_text()):
        if not Path(job["output"]).exists() and locked(job):
            running[job["name"]] = (Adopted(int(Path(job["output"] + ".lock").read_text())), job)  # type: ignore[assignment]
            started.add(job["name"])
            print(time.strftime("%H:%M:%S"), "adopted", job["name"], flush=True)
    pending = refresh()
    while pending or running:
        for name, (proc, job) in list(running.items()):
            if proc.poll() is not None:
                print(time.strftime("%H:%M:%S"), "done", name, "exit", proc.returncode, flush=True)
                Path(job["output"] + ".lock").unlink(missing_ok=True)
                del running[name]
        used = sum(_bytes(j["mem"]) for _, j in running.values())
        vused = sum(vram(j) for _, j in running.values())
        for job in list(pending):
            kind = job.get("kind", "cpu")
            n_kind = sum(1 for _, j in running.values() if j.get("kind", "cpu") == kind)
            if (
                n_kind >= (args.gpu if kind == "gpu" else args.cpu)
                or used + _bytes(job["mem"]) > budget
                or vused + vram(job) > vbudget
            ):
                continue
            if not all(Path(p).exists() for p in job.get("needs", [])):
                continue
            cmd = [str(HERE / "capped.sh"), job["mem"], job["log"], *job["cmd"]]
            proc = subprocess.Popen(cmd, cwd=job.get("cwd"))
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["output"] + ".lock").write_text(str(proc.pid))
            running[job["name"]] = (proc, job)
            started.add(job["name"])
            used += _bytes(job["mem"])
            vused += vram(job)
            pending.remove(job)
            print(time.strftime("%H:%M:%S"), "start", job["name"], flush=True)
        time.sleep(20)
        pending = [j for j in pending if j["name"] not in started] + [
            j for j in refresh() if j["name"] not in {p["name"] for p in pending}
        ]
    print("QUEUE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
