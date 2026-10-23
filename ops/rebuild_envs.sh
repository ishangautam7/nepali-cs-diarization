#!/usr/bin/env bash
# Rebuild every research environment and the public inputs on a FRESH box (after bootstrap_al2023.sh,
# box_mount.sh and a pinned uv in ~/.local/bin). Each step logs to ~/logs/<step>.log; a failed step is
# reported and the chain goes on, so one bad download does not block the rest. Safe to re-run.
#   env_pyannote4 (micromamba, ~/opt/ffmpeg, .venv-pyannote4) -> .venv-data -> WavLM SV checkpoint (Google
#   Drive, SHA-256 checked) -> env_e3 -> env_p1 -> env_audit -> env_r -> D1 parquet (public, pinned) -> D4 (OpenSLR)
# Gated inputs (pyannote Community-1, IndicVoices) need `hf auth login` on the box first; DISPLACE 2024
# zips need the organisers' password. Neither is attempted here.
# Usage on the box:  nohup bash ops/rebuild_envs.sh > ~/logs/rebuild_envs.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
UV=$HOME/.local/bin/uv
mkdir -p ~/logs ~/data/models/unispeech_sv
step() { # name command...
  local name=$1; shift
  echo "$(date -u +%FT%TZ) start $name"
  if "$@" > ~/logs/"$name".log 2>&1; then echo "$(date -u +%FT%TZ) ok $name"
  else echo "$(date -u +%FT%TZ) FAIL $name (see ~/logs/$name.log)"; fi
}
venv_data() {
  [ -x .venv-data/bin/python ] || "$UV" venv --quiet --python /usr/bin/python3.12 .venv-data
  VIRTUAL_ENV="$PWD/.venv-data" "$UV" pip install --quiet "pyarrow==21.0.0" "huggingface_hub==0.34.4" \
    "pytest==8.4.1" "gdown==6.4.2"
  .venv-data/bin/python -c "import pyarrow, huggingface_hub; print('pyarrow', pyarrow.__version__, 'hf', huggingface_hub.__version__)"
}
wavlm_ckpt() {
  local ckpt=~/data/models/unispeech_sv/wavlm_large_finetune.pth
  [ -s "$ckpt" ] || .venv-data/bin/gdown --quiet 1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP -O "$ckpt"
  echo "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b  $ckpt" | sha256sum -c -
}
step env_pyannote4 bash ops/env_pyannote4.sh
step venv_data venv_data
step wavlm_ckpt wavlm_ckpt
step env_e3 bash ops/env_e3.sh
step env_p1 bash ops/env_p1.sh
step env_audit bash ops/env_audit.sh
step env_r bash ops/env_r.sh
step d1_download .venv-data/bin/python scripts/download_d1.py --out ~/data/d1_hf
step d4_download bash ops/get_d4.sh
echo "REBUILD_ENVS_DONE"
