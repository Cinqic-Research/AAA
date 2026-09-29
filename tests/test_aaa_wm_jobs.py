"""Focused race and recovery regressions for the AAA-WM queue runner."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from research.aaa_wm import jobs


class Jobs(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "result.json"
        self.job = {
            "name": "candidate",
            "mem": "1G",
            "log": str(self.root / "run.log"),
            "output": str(self.output),
            "cmd": ["python", "-c", "pass"],
        }

    def _write_lock(self, run_id: str = "run-1", job: dict | None = None) -> int:
        target = job or self.job
        fd = jobs._open_lock(target)
        fcntl.flock(fd, fcntl.LOCK_EX)
        jobs._write_lock_record(fd, target, run_id)
        return fd

    def _write_status(self, exit_code: int, run_id: str = "run-1", job: dict | None = None) -> None:
        target = job or self.job
        output_path = Path(target["output"])
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest() if output_path.is_file() else None
        status = {
            "schema": jobs.STATUS_SCHEMA,
            "job_hash": jobs._job_hash(target),
            "run_id": run_id,
            "pid": 456,
            "exit_code": exit_code,
            "output_sha256": digest,
        }
        jobs._status_path(target).write_text(json.dumps(status), encoding="utf-8")

    def test_partial_output_is_archived_before_fresh_launch(self) -> None:
        self.output.write_text("{partial", encoding="utf-8")

        fd, adopter, already_complete = jobs._claim_for_launch(self.job, "run-fresh")

        self.assertIsNotNone(fd)
        self.assertIsNone(adopter)
        self.assertFalse(already_complete)
        self.assertFalse(self.output.exists())
        archived = list(self.root.glob("result.json.incomplete-*"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_text(encoding="utf-8"), "{partial")
        self.assertFalse(jobs._status_path(self.job).exists())
        os.close(fd)

    def test_adopted_nonzero_exit_is_failure_even_with_partial_output(self) -> None:
        self.output.write_text("{partial", encoding="utf-8")
        owner_fd = self._write_lock()
        observer_fd = jobs._open_lock(self.job)
        adopted = jobs.Adopted(
            lock_fd=observer_fd,
            job_hash=jobs._job_hash(self.job),
            run_id="run-1",
            job=self.job,
        )
        self._write_status(7)

        self.assertIsNone(adopted.poll())
        fcntl.flock(owner_fd, fcntl.LOCK_UN)
        os.close(owner_fd)
        self.assertEqual(adopted.poll(), 7)
        self.assertFalse(jobs.completed_successfully(adopted, self.job))

    def test_adopted_completion_without_status_is_failure(self) -> None:
        self.output.write_text("partial but present", encoding="utf-8")
        owner_fd = self._write_lock()
        observer_fd = jobs._open_lock(self.job)
        adopted = jobs.Adopted(
            lock_fd=observer_fd,
            job_hash=jobs._job_hash(self.job),
            run_id="run-1",
            job=self.job,
        )

        self.assertIsNone(adopted.poll())
        fcntl.flock(owner_fd, fcntl.LOCK_UN)
        os.close(owner_fd)
        self.assertEqual(adopted.poll(), 1)
        self.assertFalse(jobs.completed_successfully(adopted, self.job))

    def test_output_digest_change_invalidates_success(self) -> None:
        self.output.write_text("complete", encoding="utf-8")
        self._write_status(0)

        self.assertTrue(jobs.completed_successfully(None, self.job))
        self.output.write_text("changed after completion", encoding="utf-8")
        self.assertFalse(jobs.completed_successfully(None, self.job))

    def test_live_legacy_pid_lock_is_adopted_after_flock_claim(self) -> None:
        lock_path = jobs._lock_path(self.job)
        lock_path.write_text(f"{os.getpid()}\n", encoding="ascii")

        fd, adopted, already_complete = jobs._claim_for_launch(self.job, "must-not-launch")

        self.assertIsNone(fd)
        self.assertIsNotNone(adopted)
        self.assertEqual(adopted.pid, os.getpid())
        self.assertIsNotNone(adopted.lock_fd)
        self.assertFalse(already_complete)
        self.assertEqual(adopted.poll(), None)
        self.assertEqual(lock_path.read_text(encoding="ascii"), f"{os.getpid()}\n")
        os.close(adopted.lock_fd)

    def test_failed_producer_blocks_dependents_and_archives_partial_output(self) -> None:
        consumer_output = self.root / "consumer.json"
        consumer = {
            "name": "consumer",
            "mem": "1G",
            "log": str(self.root / "consumer.log"),
            "output": str(consumer_output),
            "cmd": ["python", "-c", "pass"],
            "needs": [str(self.output)],
        }
        queue_path = self.root / "queue.json"
        queue_path.write_text(json.dumps([self.job, consumer]), encoding="utf-8")
        launched: list[list[str]] = []

        class FailedProcess:
            returncode = 7

            def poll(self) -> int:
                return self.returncode

        def fake_popen(command: list[str], **kwargs: object) -> FailedProcess:
            launched.append(command)
            self.output.write_text("{partial", encoding="utf-8")
            return FailedProcess()

        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", ["aaa-wm-jobs", str(queue_path)]),
            patch.object(jobs.subprocess, "Popen", fake_popen),
            patch.object(jobs.time, "sleep", lambda _: None),
            redirect_stdout(stdout),
        ):
            result = jobs.main()

        self.assertEqual(result, 1)
        self.assertEqual(len(launched), 1)
        self.assertIn("QUEUE_FAILED candidate,consumer", stdout.getvalue())
        self.assertFalse(self.output.exists())
        self.assertEqual(len(list(self.root.glob("result.json.failed-output-*"))), 1)
        self.assertFalse(consumer_output.exists())

    def test_needs_distinguish_queue_outputs_from_external_files(self) -> None:
        output = self.root / "producer.json"
        output.write_text("partial", encoding="utf-8")
        consumer = {"needs": [str(output)]}
        path = jobs._path_identity(output)
        self.assertEqual(
            jobs._need_state(
                consumer,
                queue_outputs={path},
                succeeded_outputs=set(),
                failed_outputs=set(),
            ),
            "waiting",
        )
        self.assertEqual(
            jobs._need_state(
                consumer,
                queue_outputs={path},
                succeeded_outputs=set(),
                failed_outputs={path},
            ),
            "failed",
        )
        self.assertEqual(
            jobs._need_state(
                consumer,
                queue_outputs=set(),
                succeeded_outputs=set(),
                failed_outputs=set(),
            ),
            "ready",
        )

    def test_late_completed_producer_releases_its_dependent(self) -> None:
        consumer_output = self.root / "consumer.json"
        consumer = {
            "name": "consumer",
            "mem": "1G",
            "log": str(self.root / "consumer.log"),
            "output": str(consumer_output),
            "cmd": ["python", "-c", "consumer"],
            "needs": [str(self.output)],
        }
        queue_path = self.root / "queue.json"
        queue_path.write_text(json.dumps([self.job, consumer]), encoding="utf-8")
        original_claim = jobs._claim_for_launch
        launched: list[str] = []

        def late_complete(job: dict, run_id: str) -> tuple[int | None, jobs.Adopted | None, bool]:
            if job["name"] != self.job["name"]:
                return original_claim(job, run_id)
            self.output.write_text("producer complete", encoding="utf-8")
            self._write_status(0, run_id="external-run")
            return None, None, True

        class CompleteProcess:
            returncode = 0

            def poll(self) -> int:
                return 0

        def complete_consumer(command: list[str], **kwargs: object) -> CompleteProcess:
            self.assertEqual(command[-1], "consumer")
            launched.append(command[-1])
            consumer_output.write_text("consumer complete", encoding="utf-8")
            status = {
                "schema": jobs.STATUS_SCHEMA,
                "job_hash": jobs._job_hash(consumer),
                "run_id": command[5],
                "pid": 789,
                "exit_code": 0,
                "output_sha256": hashlib.sha256(consumer_output.read_bytes()).hexdigest(),
            }
            Path(command[3]).write_text(json.dumps(status), encoding="utf-8")
            return CompleteProcess()

        sleep_count = 0

        def bounded_sleep(_: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count > 5:
                raise AssertionError("dependent remained blocked after its producer completed")

        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", ["aaa-wm-jobs", str(queue_path)]),
            patch.object(jobs, "_claim_for_launch", side_effect=late_complete),
            patch.object(jobs.subprocess, "Popen", side_effect=complete_consumer),
            patch.object(jobs.time, "sleep", side_effect=bounded_sleep),
            redirect_stdout(stdout),
        ):
            result = jobs.main()

        self.assertEqual(result, 0, stdout.getvalue())
        self.assertEqual(launched, ["consumer"])
        self.assertTrue(jobs.completed_successfully(None, consumer))

    def test_competing_launchers_cannot_both_claim_output_lock(self) -> None:
        start = threading.Barrier(2)
        attempted = threading.Barrier(2)
        outcomes: list[tuple[int | None, jobs.Adopted | None, bool]] = []

        def contender() -> None:
            start.wait()
            fd, adopted, already_complete = jobs._claim_for_launch(self.job, "competing-run")
            outcomes.append((fd, adopted, already_complete))
            attempted.wait()
            if fd is not None:
                os.close(fd)
            if adopted is not None and adopted.lock_fd is not None:
                os.close(adopted.lock_fd)

        threads = [threading.Thread(target=contender) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(not complete for _, _, complete in outcomes))
        self.assertEqual(sum(fd is not None for fd, _, _ in outcomes), 1)

    def test_cap_wrapper_records_actual_nonzero_status_atomically(self) -> None:
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        fake_systemd_run = fake_bin / "systemd-run"
        fake_systemd_run.write_text(
            "#!/usr/bin/env bash\nprintf '{partial' >\"$TEST_OUTPUT\"\nexit 7\n",
            encoding="utf-8",
        )
        fake_systemd_run.chmod(0o755)
        status_path = jobs._status_path(self.job)
        env = os.environ.copy()
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["TEST_OUTPUT"] = str(self.output)
        wrapper = Path(jobs.HERE / "capped.sh")

        result = subprocess.run(
            [
                str(wrapper),
                "1G",
                self.job["log"],
                str(status_path),
                str(self.output),
                "wrapper-run",
                jobs._job_hash(self.job),
                "true",
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 7, result.stderr)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["exit_code"], 7)
        self.assertEqual(status["run_id"], "wrapper-run")
        self.assertEqual(status["job_hash"], jobs._job_hash(self.job))
        self.assertEqual(status["output_sha256"], hashlib.sha256(b"{partial").hexdigest())

    def test_wrapper_holds_inherited_lock_until_status_is_written(self) -> None:
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        fake_systemd_run = fake_bin / "systemd-run"
        fake_systemd_run.write_text(
            '#!/usr/bin/env bash\ntouch "$TEST_STARTED"\n'
            'while [[ ! -f "$TEST_GATE" ]]; do sleep 0.01; done\n'
            "printf 'complete' >\"$TEST_OUTPUT\"\n",
            encoding="utf-8",
        )
        fake_systemd_run.chmod(0o755)
        run_id = "inherited-lock-run"
        lock_fd = jobs._open_lock(self.job)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        jobs._write_lock_record(lock_fd, self.job, run_id)
        env = os.environ.copy()
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["TEST_OUTPUT"] = str(self.output)
        started = self.root / "systemd-started"
        gate = self.root / "allow-exit"
        env["TEST_STARTED"] = str(started)
        env["TEST_GATE"] = str(gate)
        wrapper = Path(jobs.HERE / "capped.sh")
        proc = subprocess.Popen(
            [
                str(wrapper),
                "1G",
                self.job["log"],
                str(jobs._status_path(self.job)),
                str(self.output),
                run_id,
                jobs._job_hash(self.job),
                "true",
            ],
            env=env,
            pass_fds=(lock_fd,),
        )
        os.close(lock_fd)
        observer_fd = jobs._open_lock(self.job)
        try:
            deadline = time.monotonic() + 5
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(started.exists(), "fake systemd-run did not start")
            with self.assertRaises(BlockingIOError):
                fcntl.flock(observer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            gate.touch()
            returncode = proc.wait(timeout=10)
        try:
            fcntl.flock(observer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(returncode, 0)
            self.assertTrue(jobs.completed_successfully(None, self.job))
        finally:
            os.close(observer_fd)


if __name__ == "__main__":
    unittest.main()
