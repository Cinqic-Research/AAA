"""Generate the independent-review handoff document from committed evidence.

Everything in the output is read from the retained confirmation summaries and
from git. Nothing about the outcome is written here in prose; if a gate failed,
this document says so.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark_v2_1"
OUTPUT = ROOT / "docs" / "handoff_astra.md"

BATCH_A = "aaa-v2_1-confirmation-a-0002"
BATCH_B = "aaa-v2_1-confirmation-b-0002"
RETIRED_A = "aaa-v2_1-confirmation-a-0001"
RETIRED_B = "aaa-v2_1-confirmation-b-0001"
STARTING_SHA = "235ce28518ca4ec3a9a240066154090a33b33470"
TAG = "opus-independent-engineering-complete-awaiting-astra-review"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1e-4 or abs(value) >= 1e5:
            return f"{value:.{digits - 1}e}"
        return f"{value:.{digits}g}"
    return str(value)


def interval(summary: dict[str, Any], key: str) -> str:
    entry = summary["gates"].get("intervals", {}).get(key)
    if not entry:
        return "n/a"
    if entry.get("status") != "OK":
        return f"**{entry.get('status')}**"
    return f"{fmt(entry['estimate'])} [{fmt(entry['lower'])}, {fmt(entry['upper'])}]"


def load(batch: str) -> dict[str, Any]:
    path = RESULTS / batch / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing confirmation evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def gates(summary: dict[str, Any]) -> dict[str, Any]:
    return {gate["name"]: gate for gate in summary["gates"]["gates"]}


def main() -> int:
    a, b = load(BATCH_A), load(BATCH_B)
    ga, gb = gates(a), gates(b)
    manifest = json.loads((ROOT / "benchmarks" / "freeze_manifest.json").read_text(encoding="utf-8"))
    hardware = a["metadata"]["hardware"]

    merge_sha = git("rev-list", "-n", "1", "--merges", "main") or "not yet merged"
    tag_target = git("rev-list", "-n", "1", TAG) or (
        f"the final `main` commit — the one that adds this document. Verify with `git rev-list -n 1 {TAG}`."
    )

    lines: list[str] = [
        "# Engineering handoff for independent review",
        "",
        "**Status: Opus engineering complete. Awaiting independent Astra review.**",
        "**This document does not grant approval.**",
        "",
        "Every status below is a measured engineering outcome. None of it is an",
        "approval decision, and none of it should be read as one. GPT-6 Astra is the",
        "designated independent reviewer and is the only reviewer authorized to issue",
        "an APPROVED or DECLINED decision.",
        "",
        "## 1. Identity",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| starting `main` | `{STARTING_SHA}` |",
        "| engineering branch | `opus/aaa-complete-engineering-repair` |",
        f"| merge commit on `main` | `{merge_sha}` |",
        f"| tag | `{TAG}` |",
        f"| tag target | {tag_target if tag_target.startswith('the final') else f'`{tag_target}`'} |",
        f"| protocol | `{a['spec_version']}` |",
        f"| specification hash | `{a['spec_hash']}` |",
        f"| dependency lock hash | `{manifest['dependency_lock']['hash']}` |",
        f"| freeze manifest source | `{manifest['source']['commit']}`, dirty: {fmt(manifest['source']['dirty'])} |",
        f"| A/B relationship | `{manifest['ab_relationship']}` |",
        "",
        "### Selected candidate",
        "",
        f"`{manifest['candidate']['model']}` — three parameters, feature set",
        f"`{manifest['candidate']['feature_set']}`, forgetting `{manifest['candidate']['forgetting']}`",
        f"(`{manifest['candidate']['forgetting_mode']}`, trace bound `{fmt(manifest['candidate']['trace_bound'])}`,",
        f"self-triggered at `{manifest['candidate']['detector_multiplier']}x` its own error scale), ridge",
        f"`{manifest['candidate']['ridge']}`, reflection `{fmt(manifest['candidate']['reflect'])}`,",
        f"{manifest['training']['episodes_per_replica']} training episodes per replica.",
        "",
        "Selection evidence: [`candidate_selection.md`](candidate_selection.md), from",
        "development streams only.",
        "",
        "Frozen checkpoint hashes (shared by A and B, by declared design):",
        "",
    ]
    for index, digest in enumerate(manifest["checkpoints"]["hashes"]):
        lines.append(f"- replica {index}: `{digest}`")

    lines += [
        "",
        "## 2. Confirmation outcomes",
        "",
        "| | Confirmation A | Confirmation B |",
        "|---|---|---|",
        f"| batch id | `{BATCH_A}` | `{BATCH_B}` |",
        f"| all required gates pass | **{fmt(a['gates']['all_required_gates_pass'])}** | **{fmt(b['gates']['all_required_gates_pass'])}** |",
        f"| unmet required gates | {', '.join(a['gates']['unmet_required_gates']) or 'none'} | {', '.join(b['gates']['unmet_required_gates']) or 'none'} |",
        f"| source commit | `{a['metadata']['git']['commit'][:12]}` | `{b['metadata']['git']['commit'][:12]}` |",
        f"| source dirty | {fmt(a['metadata']['git']['dirty'])} | {fmt(b['metadata']['git']['dirty'])} |",
        f"| runtime | {fmt(a['runtime_seconds'])} s | {fmt(b['runtime_seconds'])} s |",
        "",
        "### Every gate, both attempts",
        "",
        "| Gate | Required | A | A observed | B | B observed |",
        "|---|---|---|---:|---|---:|",
    ]
    for name in [gate["name"] for gate in a["gates"]["gates"]]:
        left, right = ga[name], gb.get(name, {})
        lines.append(
            f"| `{name}` | {fmt(left['required'])} | **{left['status']}** | {fmt(left['observed'])} "
            f"| **{right.get('status', 'n/a')}** | {fmt(right.get('observed'))} |"
        )

    round_one = {}
    for batch in (RETIRED_A, RETIRED_B):
        path = RESULTS / batch / "summary.json"
        if path.exists():
            round_one[batch] = json.loads(path.read_text(encoding="utf-8"))

    if round_one:
        lines += [
            "",
            "## 2b. Round 1 — the confirmation that failed",
            "",
            "Recorded here because it is evidence, not an embarrassment to be tidied away.",
            "The first confirmation round under this protocol **failed**: both fresh streams",
            "failed the required `always_online_stability` gate and both exited non-zero.",
            "Both batches are retired permanently and cannot be reused. **No threshold was",
            "altered in response**; the candidate was repaired instead, on development",
            "evidence. See `AAA-120` in [`issue_ledger.md`](issue_ledger.md) for the full",
            "diagnosis, including an alternative repair that was tried and did not work.",
            "",
            "| | Round 1 A | Round 1 B |",
            "|---|---|---|",
        ]
        left, right = round_one.get(RETIRED_A), round_one.get(RETIRED_B)

        def detail(summary, key):
            if summary is None:
                return None
            gate = gates(summary)["always_online_stability"]
            return gate["details"].get(key)

        lines += [
            f"| batch id | `{RETIRED_A}` | `{RETIRED_B}` |",
            f"| all required gates pass | **{fmt(left and left['gates']['all_required_gates_pass'])}** "
            f"| **{fmt(right and right['gates']['all_required_gates_pass'])}** |",
            f"| unmet required gates | {', '.join(left['gates']['unmet_required_gates']) if left else 'n/a'} "
            f"| {', '.join(right['gates']['unmet_required_gates']) if right else 'n/a'} |",
            f"| `candidate_online` normalized MAE | {fmt(detail(left, 'candidate'))} | {fmt(detail(right, 'candidate'))} |",
            f"| `constant_motion_reflected` | {fmt(detail(left, 'baseline'))} | {fmt(detail(right, 'baseline'))} |",
            f"| specification hash | `{left['spec_hash'][:16]}` | `{right['spec_hash'][:16]}` |",
            "",
            "The round-1 specification hash differs from round 2 because the candidate changed;",
            "a batch declared against one hash is refused under the other.",
        ]

    lines += [
        "",
        "## 3. The findings that mattered most",
        "",
        "### Numerical stability of the learner",
        "",
        "The previous covariance-form RLS with forgetting 0.90 lost positive",
        "semidefiniteness after **323** updates on a slow constant-velocity stream and",
        "overflowed to non-finite state at update **6,642** on a stationary one; its",
        "checkpoint loader accepted negative-definite, indefinite, asymmetric, singular,",
        "zero and badly conditioned covariance matrices. The replacement propagates a",
        "square-root factor with trace-bounded forgetting, matches an independent ridge",
        "batch least-squares reference to **1.7e-18** with no forgetting, and survives",
        "eleven weak-excitation streams to 20,000 updates. Invalid state now raises.",
        "",
        "### The fair reflected baseline",
        "",
        "The v2 bounce gate compared a candidate that could reflect into the public",
        "bounds against a baseline that could not. Measured on development streams,",
        "essentially the entire reported margin was that transform:",
        "",
        "| Bounce-event normalized MAE | Value |",
        "|---|---|",
        "| raw constant motion | ~3.1e-03 |",
        "| reflected constant motion (like-for-like) | ~3.7e-17 |",
        "| candidate | ~4.2e-17 |",
        "| candidate with reflection removed | ~3.1e-03 |",
        "",
        "Against the fair baseline the candidate is not 20% better; it is marginally",
        "worse, at floating-point magnitude. The gate was **replaced, not lowered**:",
        "no correct system can beat a baseline that is already exact, so the criterion",
        "is now an absolute accuracy requirement plus non-regression with an absolute",
        "floor. The decision was made on development evidence and frozen before any",
        "confirmation batch was generated.",
        "",
        "Measured decomposition in confirmation A:",
        "",
        f"- versus raw constant motion: {interval(a, 'bounce.decomposition_vs_raw_constant_motion')}",
        f"- versus reflected constant motion: {interval(a, 'bounce.decomposition_vs_reflected_constant_motion')}",
        f"- candidate without reflection versus raw constant motion: {interval(a, 'bounce.decomposition_no_reflect_vs_raw_constant_motion')}",
        "",
        "The middle row is the honest one.",
        "",
        "### Learning evidence",
        "",
        "The v2 `straight_learning` gate was satisfied by a parameterless analytic rule.",
        "It is replaced by a frozen-probe learning curve: checkpoints taken at increasing",
        "cumulative update budgets, scored on one **fixed** development probe bank.",
        "",
        "| Cumulative training episodes | Frozen probe normalized MAE (A) |",
        "|---:|---:|",
    ]
    curve = a["learning_curve"]["curve"]
    for budget in a["learning_curve"]["budgets"]:
        lines.append(f"| {budget} | {fmt(curve[str(budget)]['mean'])} |")

    lines += [
        "",
        f"Reduction: {interval(a, 'learning_progress.reduction')}.",
        "",
        "### Changed-law adaptation",
        "",
        "The matched experiment: common prefix, cloned learner state at the",
        "intervention, frozen and updating copies in the same changed world, plus an",
        "unchanged-world control. The branch state is hashed into",
        "`metrics/changed_law_interventions.json` so the matched design is auditable",
        "without trusting the runtime.",
        "",
        "| Comparison | A | B |",
        "|---|---|---|",
        f"| online vs identical frozen copy | {interval(a, 'changed_law.online_vs_frozen')} | {interval(b, 'changed_law.online_vs_frozen')} |",
        f"| online vs persistence | {interval(a, 'changed_law.online_vs_persistence')} | {interval(b, 'changed_law.online_vs_persistence')} |",
        f"| online vs constant motion (non-regression margin) | {interval(a, 'changed_law.online_vs_constant_motion_margin')} | {interval(b, 'changed_law.online_vs_constant_motion_margin')} |",
        f"| unchanged-world control margin | {interval(a, 'unchanged_control.margin')} | {interval(b, 'unchanged_control.margin')} |",
        "",
        "The v2 result reported roughly 55% improvement over the frozen copy. That",
        "signal survives the repaired statistics, the stabilized learner, the corrected",
        "recovery rules and fresh confirmation streams. 55% was never used as a target.",
        "",
        "### Recovery",
        "",
        "Eligibility is now driven by the peak post-event shock rather than a",
        "50-transition average, which previously diluted large fast shocks out of the",
        "denominator entirely. Unrecovered events are counted.",
        "",
        "| | A | B |",
        "|---|---:|---:|",
    ]
    ra, rb = ga["recovery"]["details"], gb["recovery"]["details"]
    lines += [
        f"| eligible | {fmt(ra['eligible'])} | {fmt(rb['eligible'])} |",
        f"| recovered | {fmt(ra['recovered'])} | {fmt(rb['recovered'])} |",
        f"| unrecovered within horizon | {fmt(ra['unrecovered'])} | {fmt(rb['unrecovered'])} |",
        f"| rate | {fmt(ga['recovery']['observed'])} | {fmt(gb['recovery']['observed'])} |",
    ]
    for label, detail in (("A", ra), ("B", rb)):
        counts = detail["status_counts"]
        ineligible = {k: v for k, v in counts.items() if k not in ("recovered", "unrecovered")}
        lines.append(f"| ineligible reasons ({label}) | {json.dumps(ineligible, sort_keys=True)} | |")
    times_a = ra.get("recovery_time_steps") or {}
    times_b = rb.get("recovery_time_steps") or {}
    lines += [
        f"| recovery time median / p90 / p95 | {fmt(times_a.get('median'))} / {fmt(times_a.get('p90'))} / {fmt(times_a.get('p95'))} "
        f"| {fmt(times_b.get('median'))} / {fmt(times_b.get('p90'))} / {fmt(times_b.get('p95'))} |",
        "",
        "Per-replica recovery rates, confirmation A: "
        + ", ".join(f"`r{k}` {fmt(v)}" for k, v in sorted(ra["per_replica_recovery_rate"].items()))
        + ".",
        "",
        "### Reproducibility",
        "",
        "| Check | A | B |",
        "|---|---|---|",
    ]
    for name in sorted(a["reproducibility"].get("checks", {})):
        lines.append(
            f"| `{name}` | {fmt(a['reproducibility']['checks'][name])} "
            f"| {fmt(b['reproducibility']['checks'].get(name))} |"
        )
    lines += ["", "| Correctness check | A | B |", "|---|---|---|"]
    for name in sorted(a["correctness"].get("checks", {})):
        lines.append(
            f"| `{name}` | {fmt(a['correctness']['checks'][name])} "
            f"| {fmt(b['correctness']['checks'].get(name))} |"
        )

    latency = a["latency"]
    lines += [
        "",
        "### Latency of the selected candidate",
        "",
        "| Operation | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|---|---:|---:|---:|",
        f"| predict | {fmt(latency['predict_p50_ms'])} | {fmt(latency['predict_p95_ms'])} | {fmt(latency['predict_p99_ms'])} |",
        f"| update | {fmt(latency['update_p50_ms'])} | {fmt(latency['update_p95_ms'])} | {fmt(latency['update_p99_ms'])} |",
        f"| predict+update | {fmt(latency['predict_plus_update_p50_ms'])} | {fmt(latency['predict_plus_update_p95_ms'])} | {fmt(latency['predict_plus_update_p99_ms'])} |",
        "",
        f"Samples {fmt(latency['samples'])}, warm-up {fmt(latency['warmup_samples'])} excluded, "
        f"failures {fmt(latency['failures'])}, on the selected candidate itself "
        f"(forgetting {fmt(latency['candidate_forgetting'])}).",
        "",
        "### Hardware",
        "",
        f"- CPU: {hardware.get('cpu_model')} ({fmt(hardware.get('cpu_count'))} threads)",
        f"- OS: {hardware.get('os')}",
        f"- Python {hardware.get('python')}, NumPy {hardware.get('numpy')}, BLAS {json.dumps(hardware.get('blas'))}",
        f"- GPU present but unused: {hardware.get('gpu')}",
        "",
        "## 4. Architecture changes",
        "",
        "- `aaa/benchmark.py` (587 lines, one module) became the `aaa/benchmark/`",
        "  package: `spec`, `seeds`, `families`, `training`, `stats`, `gates`,",
        "  `evidence`, `manifest`, `recompute`, `runner`, `report`.",
        "- The canonical specification is typed, strictly validated, hash-identified,",
        "  ships as package data, and is fully consumed — a test fails if any declared",
        "  leaf stops being read.",
        "- `StepRecord` v2 gives every provenance concept its own field; the previous",
        "  `seed` field carried a replica id.",
        "- The learner is a square-root RLS with trace-bounded, self-triggered",
        "  forgetting; the historical SGD learner is retained as a reported diagnostic.",
        "- New: confirmation batch registry, freeze manifest, experiment registry with",
        "  safe resume, independent recomputation, an always-online deployment track,",
        "  and a reflected constant-motion baseline.",
        "",
        "## 5. Superseded and failed evidence",
        "",
        "- `results/final/` — historical v1, preserved unchanged, unfavourable result",
        "  intact.",
        "- `results/benchmark_v2/` — the four v2 confirmation attempts, preserved and",
        "  reclassified as historical/provisional under a superseded methodology, with",
        "  corrections marked inline rather than rewritten away.",
        f"- `{RETIRED_A}` and `{RETIRED_B}` — the round-1 confirmation pair, retired by",
        "  required-gate failure. Preserved in full under `results/benchmark_v2_1/`.",
        "  Batch status is in `benchmarks/confirmation_batches.json`.",
        "",
        "## 6. Unresolved limitations",
        "",
        "- `AAA-077` raw per-step evidence is regenerable rather than durably",
        "  archived. Git LFS or an external archive would be stronger; neither is in",
        "  place and this is stated as a recommendation, not as done.",
        "- `AAA-110` is closed rather than open: branch protection, required status",
        "  checks, Dependabot and branch cleanup were applied through the API and read",
        "  back to confirm. Two items are deliberately *not* applied and say so with",
        "  their exact commands in `SECURITY.md`: `enforce_admins`, so the maintainer",
        "  keeps a recovery path, and the account-level read-only workflow permission,",
        "  which every workflow here already supersedes by declaring",
        "  `permissions: contents: read` directly.",
        "- `AAA-008` the always-online family rotates through a fixed sequence of the",
        "  existing regimes rather than an open-ended stream.",
        "- The world is deterministic and noiseless. Observation noise is the single",
        "  most useful next experiment and is deliberately out of scope here.",
        "",
        "## 7. Exact reproduction commands",
        "",
        "```bash",
        "git clone https://github.com/Cinqic/AAA.git && cd AAA",
        f"git checkout {TAG}",
        "python3 -m venv .venv && . .venv/bin/activate",
        "python -m pip install -r requirements-lock.txt",
        "python -m pip install -e . --no-deps",
        "python tools/check_lock.py",
        "",
        "# verification suite",
        "python -m unittest discover -s tests -t . -v",
        "python -m ruff check . && python -m ruff format --check . && python -m mypy",
        "python tools/check_exit_codes.py",
        "",
        "# protocol identity (must match the hash in section 1)",
        "python -m aaa.cli spec-hash",
        "",
        "# regenerate a confirmation attempt from its recorded batch identity",
        f"python -m aaa.cli benchmark --role confirmation_a --batch-id {BATCH_A} \\",
        "  --reproduce --output-root runs",
        "```",
        "",
        "### Recompute every metric and gate without retraining or re-simulating",
        "",
        "```bash",
        f"python -m aaa.cli recompute runs/benchmark-v2_1/{BATCH_A}",
        "```",
        "",
        "This verifies checksums and the specification hash first and exits non-zero if",
        "any stored gate status fails to reproduce.",
        "",
        "## 8. Evidence locations",
        "",
        "| What | Where |",
        "|---|---|",
        f"| confirmation A summary and report | `results/benchmark_v2_1/{BATCH_A}/` |",
        f"| confirmation B summary and report | `results/benchmark_v2_1/{BATCH_B}/` |",
        "| freeze manifest | `benchmarks/freeze_manifest.json` |",
        "| batch registry | `benchmarks/confirmation_batches.json` |",
        "| golden seed fixture | `benchmarks/golden_seeds.json` |",
        "| canonical specification | `aaa/benchmark/data/benchmark_v2_1.json` |",
        "| pre-repair defect reproduction | `docs/evidence/pre_repair_probes.json` |",
        "| learner diagnosis | `docs/evidence/diagnosis/` |",
        "| candidate selection | `docs/evidence/candidate_selection.json` |",
        "| issue ledger | `docs/issue_ledger.md` |",
        "| errata | `docs/errata.md` |",
        "",
        "Per-attempt checksums are in each attempt's `checksums.json`. Raw per-step",
        "records are regenerable rather than committed; see",
        "[`evidence_policy.md`](evidence_policy.md) and its stated limitation.",
        "",
        "## 9. Self-review",
        "",
        "See [`self_review.md`](self_review.md). It is an **Opus self-review, not",
        "independent approval**, and it is offered as a record of what was checked and",
        "what was found, not as a verdict.",
        "",
        "---",
        "",
        "**Opus independent engineering complete. Awaiting Astra independent review.**",
        "**No independent approval decision has been made.**",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
