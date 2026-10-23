#!/usr/bin/env bash
# D4 (MUCS 2021) features for H7: ingest, then GPU word alignment, VAD and dense L = 1.0 s window embeddings
# (E1-E3) for each language pair. All of this is independent of the language-tagging rules, which are fixed
# with H7's pre-registration before any D4 outcome is computed. Finished outputs are skipped, so a re-run resumes.
# Reserves GPU memory in ~/run/gpu_reserve_gb while it runs and sets the default 16 GB back on exit.
# Usage on the box: nohup bash ops/run_d4_features.sh > ~/logs/d4_features.log 2>&1 &
set -eo pipefail
cd "$(dirname "$0")/.."
source ops/env.sh
mkdir -p ~/run
exec 9> ~/run/d4_features.lock
flock -n 9 || { echo "another run_d4_features.sh is running; not starting"; exit 1; }
echo "${RESERVE_GB:-24}" > ~/run/gpu_reserve_gb
trap 'echo 16 > ~/run/gpu_reserve_gb' EXIT
py=.venv-pyannote4/bin/python
for spec in Hindi-English:hi-en:hin Bengali-English:bn-en:ben; do
  IFS=: read -r name dir lang <<< "$spec"
  root=$HOME/data/d4/$dir
  [ -f "$root/manifest.csv" ] || "$py" -m csdiar.data.ingest_d4 --raw ~/data/d4/raw --pair "$name" --out "$root"
  "$py" -W ignore -m csdiar.data.align --root "$root" --out "$root/align" --splits train test --lang "$lang"
  "$py" -W ignore -m csdiar.data.vad --root "$root" --out "$root/vad" --splits train test
done
for m in e1 e2 e3; do
  pm=$py
  [ "$m" = e3 ] && pm=.venv-e3/bin/python
  for dir in hi-en bn-en; do
    "$pm" -W ignore -m csdiar.models.embed --model "$m" --root "$HOME/data/d4/$dir" --out "$HOME/data/d4/$dir/emb" \
      --win 1.0 --splits train test
  done
done
echo "D4_FEATURES_DONE"
