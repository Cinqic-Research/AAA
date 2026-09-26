#!/usr/bin/env bash
# Run one experiment job inside a user cgroup with a hard memory cap and low CPU/IO priority.
#
#   research/aaa_wm/capped.sh <MemoryMax, e.g. 2500M> <log file> <command...>
#
# Why: on FLOWBOX (14 GiB RAM, 2 GiB swap) parallel experiment processes exhausted memory on
# 2026-09-26; the system thrashed swap (desktop freeze) and the kernel OOM killer then chose
# the desktop app. A capped job is killed alone, inside its own scope, before it can starve the
# desktop, and nice/ionice keep the UI responsive. Keep the sum of caps <= 9G and at most three
# CPU-heavy jobs at once.
set -euo pipefail
mem="$1"; log="$2"; shift 2
exec systemd-run --user --scope --quiet \
  -p MemoryMax="$mem" -p MemorySwapMax=0 -p CPUWeight=20 -p IOWeight=20 \
  nice -n 15 ionice -c 3 "$@" >"$log" 2>&1
