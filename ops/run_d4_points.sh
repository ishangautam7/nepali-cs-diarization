#!/usr/bin/env bash
# D4 point-anchored window embeddings (H1 windows at L = 0.5 / 1.0 / 1.5 s) for E1-E3, for both pairs, from the
# boundary version given as $1 (default lex_d4v1). Waits until ops/run_d4_features.sh releases its lock, so the
# two GPU chains never run at once. Reserves GPU memory while it runs and sets the default 16 GB back on exit.
# Usage on the box: nohup bash ops/run_d4_points.sh lex_d4v1 > ~/logs/d4_points.log 2>&1 &
set -eo pipefail
cd "$(dirname "$0")/.."
source ops/env.sh
tag=${1:-lex_d4v1}
mkdir -p ~/run
exec 9> ~/run/d4_features.lock
echo "$(date -u +%FT%TZ) waiting for the D4 feature chain"
flock 9
echo "${RESERVE_GB:-24}" > ~/run/gpu_reserve_gb
trap 'echo 16 > ~/run/gpu_reserve_gb' EXIT
for m in e1 e2 e3; do
  py=.venv-pyannote4/bin/python
  [ "$m" = e3 ] && py=.venv-e3/bin/python
  for dir in hi-en bn-en; do
    root=$HOME/data/d4/$dir
    "$py" -W ignore -m csdiar.models.point_embed --model "$m" --root "$root" --points "$root/boundaries/$tag" \
      --out "$root/point_emb/$tag" --splits train test
  done
done
echo "D4_POINTS_DONE $(date -u +%FT%TZ)"
