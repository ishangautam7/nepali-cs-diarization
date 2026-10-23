#!/usr/bin/env bash
# Run M3 jobs from a queue file, at most N at a time (PREREGISTRATION_H4.md §7, frozen commands in FREEZE_H4.json).
# Each queue line: "<fold|all> <seed> <w_dist> <ablation> <embed-splits...>", e.g. "0 0 0.3 none validation train".
# A job trains the model (skipped if its meta.json exists), then writes its dense L = 1 s grid with m3_embed
# (each split skipped if already written). Only splits named on the line are embedded; "-" embeds nothing.
# Usage on the box:  bash ops/run_m3_queue.sh <queue-file> <N> > ~/logs/m3_queue.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
QUEUE=$1
N=${2:-2}
source ops/env.sh
mkdir -p ~/logs/m3
run_one() {
  set -- $1
  local fold=$1 seed=$2 w=$3 abl=$4
  shift 4
  local sel tag
  if [ "$fold" = "all" ]; then sel="--all"; tag="all"; else sel="--fold $fold"; tag="f$fold"; fi
  tag="${tag}_s${seed}_w${w}_${abl}"
  echo "$(date -u +%FT%TZ) train $tag"
  if ! .venv-e3/bin/python -m csdiar.mitigations.m3_train $sel --seed "$seed" --w-dist "$w" --ablation "$abl" \
      > ~/logs/m3/"$tag".log 2>&1; then
    echo "$(date -u +%FT%TZ) FAIL train $tag"; return 0
  fi
  if [ "$1" != "-" ]; then
    echo "$(date -u +%FT%TZ) embed $tag ($*)"
    if ! .venv-e3/bin/python -m csdiar.mitigations.m3_embed --ckpt ~/data/m3/runs/"$tag"/best.pt \
        --name "m3_$tag" --splits "$@" > ~/logs/m3/"$tag"_embed.log 2>&1; then
      echo "$(date -u +%FT%TZ) FAIL embed $tag"; return 0
    fi
  fi
  echo "$(date -u +%FT%TZ) done $tag"
}
export -f run_one
grep -v -E '^\s*(#|$)' "$QUEUE" | xargs -P "$N" -I{} bash -c 'run_one "$@"' _ {}
echo "QUEUE_DONE"
