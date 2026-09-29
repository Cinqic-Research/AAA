"""RAM-safe, resumable job runner for long experiment queues on FLOWBOX.

Each command runs through capped.sh. The runner and wrapper share an exclusive flock on
<output>.lock; the wrapper records the command's exit status atomically in
<output>.status.json. An output is reusable only when that record matches the job and the
recorded output digest. Incomplete outputs are retained beside the requested output.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STATUS_SCHEMA = 1


def _bytes(s: str) -> int:
    return int(float(s[:-1]) * {"M": 2**20, "G": 2**30}[s[-1]])


def _job_hash(job: dict[str, Any]) -> str:
    encoded = json.dumps(job, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _status_path(job: dict[str, Any]) -> Path:
    return Path(str(job["output"]) + ".status.json")


def _lock_path(job: dict[str, Any]) -> Path:
    return Path(str(job["output"]) + ".lock")


def _path_identity(path: str | Path) -> str:
    return str(Path(path).resolve())


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _valid_status(job: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any] | None:
    record = _read_json(_status_path(job))
    if not record:
        return None
    if record.get("schema") != STATUS_SCHEMA or record.get("job_hash") != _job_hash(job):
        return None
    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        return None
    if run_id is not None and record["run_id"] != run_id:
        return None
    code = record.get("exit_code")
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    return record


def completed_successfully(
    proc: subprocess.Popen[bytes] | Adopted | None,
    job: dict[str, Any],
    *,
    run_id: str | None = None,
) -> bool:
    """Return true only for a matching zero-exit wrapper record and unchanged output."""
    if isinstance(proc, Adopted):
        if proc.returncode != 0:
            return False
        run_id = proc.run_id
    elif proc is not None and proc.returncode != 0:
        return False
    record = _valid_status(job, run_id=run_id)
    if not record or record.get("exit_code") != 0:
        return False
    output = Path(job["output"])
    actual_digest = _sha256(output)
    return actual_digest is not None and record.get("output_sha256") == actual_digest and output.is_file()


class Adopted:
    """Monitor a job from a prior runner through its shared lock and status record.

    Legacy PID-only locks are monitored for liveness but cannot establish success because
    they contain neither an exit status nor a wrapper completion record.
    """

    def __init__(
        self,
        pid: int | None = None,
        *,
        lock_fd: int | None = None,
        job_hash: str | None = None,
        run_id: str | None = None,
        job: dict[str, Any] | None = None,
    ) -> None:
        self.pid = pid
        self.lock_fd = lock_fd
        self.job_hash = job_hash
        self.run_id = run_id
        self.job = job
        self.returncode: int | None = None
        self.status: dict[str, Any] | None = None

    def _finish_from_status(self) -> int:
        if self.job is None or self.job_hash is None or self.run_id is None:
            self.returncode = 1
            return self.returncode
        record = _valid_status(self.job, run_id=self.run_id)
        if record is None or (self.job_hash is not None and record.get("job_hash") != self.job_hash):
            self.returncode = 1
        else:
            self.status = record
            self.returncode = record["exit_code"]
        return self.returncode

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self.pid is not None:
            try:
                os.kill(self.pid, 0)
                return None
            except PermissionError:
                return None
            except (ProcessLookupError, OSError):
                self.returncode = 1
                if self.lock_fd is not None:
                    os.close(self.lock_fd)
                    self.lock_fd = None
                return self.returncode
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            try:
                record = _read_lock_record(self.lock_fd)
                if record is not None:
                    self.job_hash = record.get("job_hash")
                    self.run_id = record.get("run_id")
                return self._finish_from_status()
            finally:
                os.close(self.lock_fd)
                self.lock_fd = None
        self.returncode = 1
        return self.returncode


def _read_lock_record(fd: int) -> dict[str, Any] | None:
    try:
        raw = os.pread(fd, 4096, 0).decode("utf-8").strip()
    except (OSError, UnicodeError):
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _legacy_pid(fd: int) -> int | None:
    try:
        value = os.pread(fd, 128, 0).decode("ascii").strip()
        pid = int(value)
        return pid if pid > 0 else None
    except (OSError, UnicodeError, ValueError):
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _open_lock(job: dict[str, Any]) -> int:
    path = _lock_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, 0o600)


def _try_existing_lock(job: dict[str, Any]) -> Adopted | None:
    """Return an adopter if another process holds the lock; otherwise close and return None."""
    fd = _open_lock(job)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        record = _read_lock_record(fd)
        if record and isinstance(record.get("pid"), int):
            adopter = Adopted(
                lock_fd=fd,
                job_hash=record.get("job_hash"),
                run_id=record.get("run_id"),
                job=job,
            )
            return adopter
        pid = _legacy_pid(fd)
        os.close(fd)
        if pid is not None and _pid_is_alive(pid):
            return Adopted(pid=pid, job=job)
        # A held lock may be between truncation and its new metadata write. Wait for
        # release, then read the final record instead of guessing from partial contents.
        fd = _open_lock(job)
        return Adopted(lock_fd=fd, job=job)
    else:
        legacy_pid = _legacy_pid(fd)
        if legacy_pid is not None and _pid_is_alive(legacy_pid):
            return Adopted(pid=legacy_pid, lock_fd=fd, job=job)
        os.close(fd)
        return None


def _archive(path: Path, label: str, run_id: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    destination = path.with_name(path.name + f".{label}-{run_id}")
    path.replace(destination)


def _prepare_output(job: dict[str, Any], run_id: str) -> None:
    """Move any non-reusable result and stale status aside before starting a fresh run."""
    output = Path(job["output"])
    status = _status_path(job)
    if completed_successfully(None, job):
        return
    _archive(output, "incomplete", run_id)
    _archive(status, "stale-status", run_id)


def _quarantine_failed_result(job: dict[str, Any], run_id: str | None) -> bool:
    """Archive a failed run's output while holding its lock, if no newer run owns it."""
    fd = _open_lock(job)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        current_lock = _read_lock_record(fd)
        if run_id is not None and current_lock is not None and current_lock.get("run_id") != run_id:
            return False
        current_status = _read_json(_status_path(job))
        if run_id is not None and current_status is not None and current_status.get("run_id") != run_id:
            return False
        archive_id = run_id or uuid.uuid4().hex
        _archive(Path(job["output"]), "failed-output", archive_id)
        _archive(_status_path(job), "failed-status", archive_id)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _write_lock_record(fd: int, job: dict[str, Any], run_id: str) -> None:
    identity = _job_hash(job)
    record = {
        "schema": 1,
        "job_hash": identity,
        "run_id": run_id,
        "pid": os.getpid(),
    }
    raw = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, raw)
    os.fsync(fd)


def _claim_for_launch(job: dict[str, Any], run_id: str) -> tuple[int | None, Adopted | None, bool]:
    """Try to claim output. Result is (fd, adopter, already_complete)."""
    fd = _open_lock(job)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        record = _read_lock_record(fd)
        if record and isinstance(record.get("pid"), int):
            return (
                None,
                Adopted(
                    lock_fd=fd,
                    job_hash=record.get("job_hash"),
                    run_id=record.get("run_id"),
                    job=job,
                ),
                False,
            )
        pid = _legacy_pid(fd)
        if pid is not None and _pid_is_alive(pid):
            os.close(fd)
            return None, Adopted(pid=pid, job=job), False
        return None, Adopted(lock_fd=fd, job=job), False
    legacy_pid = _legacy_pid(fd)
    if legacy_pid is not None and _pid_is_alive(legacy_pid):
        return None, Adopted(pid=legacy_pid, lock_fd=fd, job=job), False
    if completed_successfully(None, job):
        os.close(fd)
        return None, None, True
    _prepare_output(job, run_id)
    _write_lock_record(fd, job, run_id)
    return fd, None, False


def _need_state(
    job: dict[str, Any],
    *,
    queue_outputs: set[str],
    succeeded_outputs: set[str],
    failed_outputs: set[str],
) -> str:
    """Return ready, waiting, or failed for the job's declared dependency paths."""
    waiting = False
    for raw_path in job.get("needs", []):
        path = _path_identity(raw_path)
        if path in failed_outputs:
            return "failed"
        if (path in queue_outputs and path not in succeeded_outputs) or not Path(raw_path).exists():
            waiting = True
    return "waiting" if waiting else "ready"


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
    running: dict[str, tuple[subprocess.Popen[bytes] | Adopted, dict[str, Any], str | None]] = {}
    started: set[str] = set()
    failed: list[str] = []
    succeeded_outputs: set[str] = set()
    failed_outputs: set[str] = set()
    budget = _bytes(args.budget)
    vbudget = _bytes(args.vram)

    def vram(job: dict[str, Any]) -> int:
        return _bytes(job.get("vram", "1G")) if job.get("kind") == "gpu" else 0

    initial = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    for job in initial:
        if completed_successfully(None, job):
            started.add(job["name"])
            succeeded_outputs.add(_path_identity(job["output"]))
            continue
        adopted = _try_existing_lock(job)
        if adopted is not None:
            running[job["name"]] = (adopted, job, adopted.run_id)
            started.add(job["name"])
            print(time.strftime("%H:%M:%S"), "adopted", job["name"], flush=True)

    while True:
        for name, (proc, job, run_id) in list(running.items()):
            code = proc.poll()
            if code is not None:
                print(time.strftime("%H:%M:%S"), "done", name, "exit", code, flush=True)
                if completed_successfully(proc, job, run_id=run_id):
                    succeeded_outputs.add(_path_identity(job["output"]))
                else:
                    if not _quarantine_failed_result(job, run_id):
                        print(time.strftime("%H:%M:%S"), "could-not-quarantine", name, flush=True)
                    failed_outputs.add(_path_identity(job["output"]))
                    failed.append(name)
                del running[name]

        try:
            queued = json.loads(Path(args.queue).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print("QUEUE_FAILED", f"invalid queue: {error}", flush=True)
            return 1
        queue_outputs = {_path_identity(job["output"]) for job in queued}
        pending = []
        for job in queued:
            if job["name"] in started:
                continue
            if completed_successfully(None, job):
                started.add(job["name"])
                succeeded_outputs.add(_path_identity(job["output"]))
                continue
            need_state = _need_state(
                job,
                queue_outputs=queue_outputs,
                succeeded_outputs=succeeded_outputs,
                failed_outputs=failed_outputs,
            )
            if need_state == "failed":
                started.add(job["name"])
                failed.append(job["name"])
                failed_outputs.add(_path_identity(job["output"]))
                print(time.strftime("%H:%M:%S"), "blocked-by-failed-need", job["name"], flush=True)
                continue
            pending.append(job)

        if not pending and not running:
            break

        used = sum(_bytes(j["mem"]) for _, j, _ in running.values())
        vused = sum(vram(j) for _, j, _ in running.values())
        for job in pending:
            if job["name"] in started:
                continue
            kind = job.get("kind", "cpu")
            n_kind = sum(1 for _, j, _ in running.values() if j.get("kind", "cpu") == kind)
            if (
                n_kind >= (args.gpu if kind == "gpu" else args.cpu)
                or used + _bytes(job["mem"]) > budget
                or vused + vram(job) > vbudget
            ):
                continue
            if (
                _need_state(
                    job,
                    queue_outputs=queue_outputs,
                    succeeded_outputs=succeeded_outputs,
                    failed_outputs=failed_outputs,
                )
                != "ready"
            ):
                continue
            run_id = uuid.uuid4().hex
            fd, adopter, already_complete = _claim_for_launch(job, run_id)
            if already_complete:
                started.add(job["name"])
                succeeded_outputs.add(_path_identity(job["output"]))
                continue
            if adopter is not None:
                running[job["name"]] = (adopter, job, adopter.run_id)
                started.add(job["name"])
                used += _bytes(job["mem"])
                vused += vram(job)
                print(time.strftime("%H:%M:%S"), "adopted", job["name"], flush=True)
                continue
            if fd is None:
                continue
            identity = _job_hash(job)
            status_path = _status_path(job)
            try:
                cmd = [
                    str(HERE / "capped.sh"),
                    job["mem"],
                    job["log"],
                    str(status_path.resolve()),
                    str(Path(job["output"]).resolve()),
                    run_id,
                    identity,
                    *job["cmd"],
                ]
                proc = subprocess.Popen(cmd, cwd=job.get("cwd"), pass_fds=(fd,))
            except OSError as error:
                print(time.strftime("%H:%M:%S"), "start-failed", job["name"], error, flush=True)
                failed.append(job["name"])
                failed_outputs.add(_path_identity(job["output"]))
                started.add(job["name"])
            else:
                running[job["name"]] = (proc, job, run_id)
                started.add(job["name"])
                used += _bytes(job["mem"])
                vused += vram(job)
                print(time.strftime("%H:%M:%S"), "start", job["name"], flush=True)
            finally:
                os.close(fd)

        time.sleep(1)

    if failed:
        print("QUEUE_FAILED", ",".join(failed), flush=True)
        return 1
    print("QUEUE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
