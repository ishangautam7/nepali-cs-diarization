#!/usr/bin/env bash
# Run the pre-registered D1 sensitivity variants (PREREGISTRATION.md §5 items 2, 3, 8; planning/DEVIATIONS.md entry 9)
# end to end on the box, with the reviewed scripts/sensitivity_variants.py:
#   1. the three variant boundary runs (CPU) -> ~/data/d1/boundaries/lex_v1b_<variant>;
#   2. once the D4 alignment has finished (so the GPU is not shared by three research jobs at once), the point
#      embeddings of each variant for E3 (.venv-e3) and E1, E2 (.venv-pyannote4), as in ops/rebuild_features.sh, for
#      the train and test splits -> ~/data/d1/point_emb/lex_v1b_<variant>; the research VRAM reserve is held at
#      RESERVE_GB (default 24) meanwhile and put back to 16 afterwards, as the other GPU scripts do;
#   3. the analysis of base (the rebuilt, unvaried lex_v1b) and of each variant -> ~/data/d1/results/sensitivity/.
# Each step stops the chain on failure. Nothing is deleted; a variant whose boundary folder exists is refused by
# the script itself. Usage on the box:
#   cd ~/nepali-cs-diarization; nohup bash ops/run_sensitivity_variants.sh > ~/logs/sens_variants.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
source ops/env.sh
PY4=.venv-pyannote4/bin/python
PYE3=.venv-e3/bin/python
D1=$HOME/data/d1
VARIANTS=(retag_ia low_align loan_lang)
RESERVE_GB=${RESERVE_GB:-24}
say() { echo "$(date -u +%FT%TZ) $*"; }

for v in "${VARIANTS[@]}"; do
  say "boundaries $v"
  "$PY4" scripts/sensitivity_variants.py boundaries --variant "$v"
done

say "waiting for the D4 alignment (~/logs/d4_align.log) to finish before using the GPU"
for _ in $(seq 1 480); do
  if grep -q -E 'D4_ALIGN_DONE|Traceback|failed' ~/logs/d4_align.log 2>/dev/null; then break; fi
  sleep 60
done

echo "$RESERVE_GB" > ~/run/gpu_reserve_gb
trap 'echo 16 > ~/run/gpu_reserve_gb' EXIT
for v in "${VARIANTS[@]}"; do
  for m in e3 e1 e2; do
    py=$PY4
    if [ "$m" = e3 ]; then py=$PYE3; fi
    say "point_embed $v $m"
    "$py" -W ignore -m csdiar.models.point_embed --model "$m" --root "$D1" --points "$D1/boundaries/lex_v1b_$v" \
      --out "$D1/point_emb/lex_v1b_$v" --splits train test
  done
done
echo 16 > ~/run/gpu_reserve_gb

for v in base "${VARIANTS[@]}"; do
  say "analyse $v"
  "$PY4" scripts/sensitivity_variants.py analyse --variant "$v"
done
say "SENS_DONE"
