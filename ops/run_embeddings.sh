#!/usr/bin/env bash
# Dense window embeddings of all D1 clips for one model (e1, e2 or e3): L = 1.0 s (primary), then 0.5 s and
# 1.5 s (sensitivity, PLAN.md §2). Finished splits are skipped, so a re-run resumes.
# While it runs it reserves GPU memory in ~/run/gpu_reserve_gb (other GPU users on the box keep that much free)
# and sets the default 16 GB back on exit.
# Usage on the box: nohup bash ops/run_embeddings.sh e3 > ~/logs/emb_e3.log 2>&1 &
set -eo pipefail
cd "$(dirname "$0")/.."
source ops/env.sh
model="$1"
py=.venv-pyannote4/bin/python
[ "$model" = e3 ] && py=.venv-e3/bin/python
mkdir -p ~/run
echo "${RESERVE_GB:-24}" > ~/run/gpu_reserve_gb
trap 'echo 16 > ~/run/gpu_reserve_gb' EXIT
for win in 1.0 0.5 1.5; do
  "$py" -W ignore -m csdiar.models.embed --model "$model" --root ~/data/d1 --out ~/data/d1/emb --win "$win"
done
echo "EMB_DONE $model"
