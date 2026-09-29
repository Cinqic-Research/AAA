#!/usr/bin/env bash
# Kill processes whose command line matches PATTERN, never this script or its ancestors.
#   research/aaa_wm/killjobs.sh PATTERN
set -u
pat="$1"
skip=" $$ $PPID "
p=$PPID
while [ "$p" -gt 1 ]; do p=$(ps -o ppid= -p "$p" | tr -d ' '); skip="$skip$p "; done
for pid in $(pgrep -f -- "$pat"); do
  case "$skip" in *" $pid "*) continue ;; esac
  echo "kill $pid $(ps -o args= -p "$pid" | cut -c1-100)"
  kill "$pid" 2>/dev/null
done
