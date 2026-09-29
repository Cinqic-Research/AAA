#!/usr/bin/env bash
# Run one experiment job inside a user cgroup with a hard memory cap and low CPU/IO priority.
#
#   research/aaa_wm/capped.sh <MemoryMax> <log> <status-json> <output> <run-id> <job-hash> <command...>
#
# Why: on FLOWBOX (14 GiB RAM, 2 GiB swap) parallel experiment processes exhausted memory on
# 2026-09-26; the system thrashed swap (desktop freeze) and the kernel OOM killer then chose
# the desktop app. A capped job is killed alone, inside its own scope, before it can starve the
# desktop, and nice/ionice keep the UI responsive. Keep the sum of caps <= 9G and at most three
# CPU-heavy jobs at once.
set -uo pipefail
mem="$1"
log="$2"
status_path="$3"
output_path="$4"
run_id="$5"
job_hash="$6"
shift 6
mkdir -p -- "$(dirname -- "$status_path")" "$(dirname -- "$log")"

write_completion_status() {
  local rc=$?
  trap - EXIT
  local output_sha=""
  local tmp="$status_path.tmp.$run_id.$$"
  if [[ -f "$output_path" ]]; then
    output_sha="$(sha256sum -- "$output_path" 2>/dev/null | awk '{print $1}')" || output_sha=""
  fi
  if [[ "$output_sha" =~ ^[[:xdigit:]]{64}$ ]]; then
    printf '{"schema":1,"job_hash":"%s","run_id":"%s","pid":%d,"exit_code":%d,"output_sha256":"%s"}\n' \
      "$job_hash" "$run_id" "$$" "$rc" "$output_sha" >"$tmp" &&
      mv -f -- "$tmp" "$status_path"
  else
    printf '{"schema":1,"job_hash":"%s","run_id":"%s","pid":%d,"exit_code":%d,"output_sha256":null}\n' \
      "$job_hash" "$run_id" "$$" "$rc" >"$tmp" &&
      mv -f -- "$tmp" "$status_path"
  fi
  rm -f -- "$tmp"
  exit "$rc"
}
trap write_completion_status EXIT

systemd-run --user --scope --quiet \
  -p MemoryMax="$mem" -p MemorySwapMax=0 -p CPUWeight=20 -p IOWeight=20 \
  nice -n 15 ionice -c 3 "$@" >"$log" 2>&1
