#!/usr/bin/env bash
# Build the main model environment on the GPU box (PLAN.md §8, "env-pyannote4"). Safe to re-run.
#   1. micromamba (pinned release, checked against its published SHA-256)
#   2. FFmpeg 7.1 shared libraries from conda-forge into ~/opt/ffmpeg (Amazon Linux 2023 has no FFmpeg;
#      torchcodec, which pyannote.audio 4 uses for decoding, needs them)
#   3. .venv-pyannote4: torch/torchaudio 2.8.0 built for CUDA 12.8 (Blackwell sm_120), torchcodec 0.7.0
#      (the torchcodec line for torch 2.8), pyannote.audio 4.0.7, pyannote.metrics 4.2.0,
#      speechbrain 1.1.2, ctc-forced-aligner 1.0.2, soundfile 0.14.0
#   4. smoke test: CUDA matmul on the GPU, and every package imports
# Usage on the box:  bash ops/env_pyannote4.sh > ~/logs/env_pyannote4.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
MAMBA_VERSION=2.9.0-0
OPT=$HOME/opt
UV=$HOME/.local/bin/uv
mkdir -p "$OPT"

echo "== 1. micromamba $MAMBA_VERSION"
if [ ! -x "$OPT/micromamba" ]; then
  base="https://github.com/mamba-org/micromamba-releases/releases/download/$MAMBA_VERSION"
  curl -fsSL --retry 3 -o "$OPT/micromamba.download" "$base/micromamba-linux-64"
  expected=$(curl -fsSL --retry 3 "$base/micromamba-linux-64.sha256" | awk '{print $1}')
  echo "$expected  $OPT/micromamba.download" | sha256sum -c -
  mv "$OPT/micromamba.download" "$OPT/micromamba"
  chmod +x "$OPT/micromamba"
fi

echo "== 2. FFmpeg shared libraries"
if ! ls "$OPT"/ffmpeg/lib/libavcodec.so.* >/dev/null 2>&1; then
  "$OPT/micromamba" create -y -q -r "$OPT/mamba-root" -p "$OPT/ffmpeg" -c conda-forge "ffmpeg=7.1"
fi
ls "$OPT"/ffmpeg/lib/libavcodec.so.* | head -1

echo "== 3. Python environment"
[ -x .venv-pyannote4/bin/python ] || "$UV" venv --quiet --python /usr/bin/python3.12 .venv-pyannote4
export VIRTUAL_ENV="$PWD/.venv-pyannote4"
cat > .venv-pyannote4/constraints.txt <<'EOF'
torch==2.8.0
torchaudio==2.8.0
torchcodec==0.7.0
EOF
"$UV" pip install --quiet --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchaudio==2.8.0"
"$UV" pip install --quiet -c .venv-pyannote4/constraints.txt \
  "torchcodec==0.7.0" "pyannote.audio==4.0.7" "pyannote.metrics==4.2.0" "speechbrain==1.1.2" \
  "ctc-forced-aligner==1.0.2" "soundfile==0.14.0" "uroman==1.3.1.1" "silero-vad==6.2.3" \
  "praat-parselmouth==0.4.7"

echo "== 4. smoke test"
source ops/env.sh
.venv-pyannote4/bin/python - <<'PY'
import torch
x = torch.randn(4096, 4096, device="cuda")
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0), "matmul ok", bool(torch.isfinite((x @ x).sum())))
import torchaudio, torchcodec, pyannote.audio, pyannote.metrics, speechbrain, ctc_forced_aligner, soundfile
for m in (torchaudio, torchcodec, pyannote.audio, pyannote.metrics, speechbrain, soundfile):
    print(m.__name__, getattr(m, "__version__", "?"))
print("ctc_forced_aligner import ok")
PY
echo "ENV_DONE"
