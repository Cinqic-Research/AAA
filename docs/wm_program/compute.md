# Compute, memory and the 2026-09-26 freeze incident

FLOWBOX: Ryzen 7 5700G (16 threads), 14 GiB usable RAM, 2 GiB swap file, RTX 2060 6 GB (Turing, fp16
tensor cores, no bf16), driver 595.91.07. Work, data and checkpoints live on the `Cinqic Storage`
HDD (a 5,400 rpm SATA drive), never on the 97%-full NVMe.

## The desktop freezes

The user reported that the PC froze whenever work started. The kernel journals of the affected
boots show why:

- **Boot 2026-09-26 01:56-09:16.** Global OOM at 04:16. At that moment six experiment `python`
  processes held about 1.5 GB each (parallel WM-S evaluations), one held 3.1 GB (a GPU trainer
  holding the behavior dataset) and one 0.65 GB: about 12 GB on a 14 GB machine. Free swap was
  132 kB. The system first thrashed swap (the freeze), then the kernel OOM killer chose the
  **desktop app** (`claude-desktop`, oom_score_adj 300), not the experiments. A second OOM at 09:16
  preceded a reboot. Earlier in the same boot, nine parallel grid jobs, each spawning sandboxed
  CPython subprocesses, drove the load average to 34.
- An `NVRM: Xid 43` at 02:33 came from a deliberately bad benchmark index in my own GPU test (a
  device-side assertion). It did not recur.

**Cause: my experiment scheduling.** Too many memory-heavy processes ran with no per-job limit,
and the kernel's global OOM policy protects experiments over the desktop.

**Fix (in the repository).**

- [`research/aaa_wm/capped.sh`](../../research/aaa_wm/capped.sh) runs every job in a user cgroup
  scope with `MemoryMax`, no swap, `CPUWeight`/`IOWeight` 20, `nice 15` and idle I/O priority.
  A job that exceeds its cap is killed alone, inside its own scope. This was verified several
  times the same day, and the desktop was never affected.
- [`research/aaa_wm/jobs.py`](../../research/aaa_wm/jobs.py) schedules queues. It enforces a total
  cap budget (9.5 GB), CPU/GPU concurrency limits and prerequisites. It acquires an exclusive
  `flock` on each persistent output lock before launch and passes that lock through
  [`capped.sh`](../../research/aaa_wm/capped.sh). The wrapper records the job hash, run id, command
  exit code and output digest in an atomic status file. A restart adopts a running job through that
  lock and reuses its output only after the matching zero-exit status and digest verify. Stale or
  failed output is archived; a failed queue producer does not release dependent jobs to consume its
  partial file. Legacy PID-only locks are waited on and cannot establish success.
- In-process caches are bounded. Measured peaks set the caps:

| Job | Measured peak RSS | Cap |
|---|---:|---:|
| WM-S library table build (200K programs, 5.2M observations) | 5.16 GB | 6 GB |
| opaque.v0 neural arm training (dataset + CUDA context) | > 4 GB (killed at 4 GB) | 6 GB |
| WM-S evaluation (4,000 episodes) | < 2.5 GB | 2.5 GB |
| LM training, 5M-13M params (block sampler) | about 2.2 GB | 3 GB |
| corpus build (MinHash over about 700K documents) | < 4 GB | 4 GB |
| byte tokenization (streamed) / BPE tokenization (streamed) | 1.4 GB / 2.2 GB | 2.5 / 3 GB |

Caps protect the desktop only if the *sum* of real use stays below physical RAM. That is why every
job goes through one runner and one budget. The system tools `earlyoom` or `systemd-oomd`, with
a preference for killing batch jobs, would add a second line of defense. Installing them changes
system settings, so that is a decision for Markus, not something I did.

Two self-inflicted scheduling bugs were found and fixed the same day. Queue scripts waited with
`pgrep -f` on text that also appeared in my own shell commands, so they deadlocked. `pkill -f`
killed its own shell three times. [`research/aaa_wm/killjobs.sh`](../../research/aaa_wm/killjobs.sh)
excludes its own process tree.

## Throughput decisions (measured)

- **opaque.v0 neural arms** (about 0.98M-parameter transformer, batch 256, about 113 tokens): fp32
  0.098 s/step, fp16 autocast 0.052 s/step. AMP was validated on a 1,500-step run (loss 2.84 against
  2.80, equal held-out accuracy) before use. The model is memory-bound, not FLOP-bound: the 2060
  measured 5.2 TFLOP/s fp32 and 21.9 TFLOP/s fp16 on 4096^2 matmuls.
- **WM-S planning** is CPU-bound. Subset validation dominated (4.7 s/episode). A cached skeleton
  validator, proven equivalent by a test, brought it to 0.38 s/episode.
- **LM training** was I/O-bound at 33K tokens/s: random windows from multi-GB memmaps on the HDD at
  idle I/O priority. A contiguous in-RAM block sampler reaches 174K tokens/s (bytes, 3.2M
  params) and 163K tokens/s (BPE-8K). The GPU is then the bottleneck.

## Storage

The HDD had about 376 GB free at the start. Raw corpora take about 5 GB, processed corpus text about
1.8 GB, token arrays about 5 GB, the behavior dataset about 0.1 GB compressed, and checkpoints
at most 4 MB (world-model arms) to about 110 MB (the largest LM planned).
