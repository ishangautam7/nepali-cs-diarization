#!/usr/bin/env bash
# Run the M2a confirmatory analysis exactly as frozen: the commands stored in planning/FREEZE_M2A.json
# (confirm.py, then attribution.py), in order, on the box. Both programs re-check every frozen hash and input path
# and refuse to run if anything differs. Each test set is scored once (PREREGISTRATION.md §7); do not re-run
# after a success.
# Usage on the box: nohup bash ops/run_confirm_m2a.sh > ~/logs/confirm_m2a.log 2>&1 &
set -eo pipefail
cd "$(dirname "$0")/.."
source ops/env.sh
out=$HOME/data/d1/results/confirm
if [ -f "$out/attribution_m2a.json" ]; then
  echo "refusing: $out/attribution_m2a.json exists; the confirmatory run has already been done"
  exit 1
fi
mkdir -p "$out"
python3 - <<'PY' > "$out/commands.txt"
import json
for c in json.load(open("planning/FREEZE_M2A.json"))["commands"]:
    print(c)
PY
while IFS= read -r cmd; do
  echo "== $(date -u +%FT%TZ) $cmd"
  bash -c "$cmd" < /dev/null
done < "$out/commands.txt"
echo "CONFIRM_M2A_DONE $(date -u +%FT%TZ)"
