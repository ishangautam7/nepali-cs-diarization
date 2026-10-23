#!/usr/bin/env bash
# Build the P1-AHC environment on the GPU box (PLAN.md §5, P1-AHC). Safe to re-run.
# P1-AHC needs pyannote.audio (Community-1 segmentation and clustering) and the E3/M3 embedding (s3prl WavLM) in
# one process. .venv-pyannote4 has no s3prl and .venv-e3 has no pyannote, and both serve frozen analyses or
# running jobs, so neither is modified: this is a separate environment with the same pins as both
# (torch/torchaudio 2.8.0 for CUDA 12.8, torchcodec 0.7.0, pyannote.audio 4.0.7, pyannote.metrics 4.2.0,
# speechbrain 1.1.2, s3prl 0.4.18 with numpy < 2.3 as in .venv-e3, soundfile 0.14.0).
# Usage on the box:  bash ops/env_p1.sh > ~/logs/env_p1.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
UV=$HOME/.local/bin/uv
[ -x .venv-p1/bin/python ] || "$UV" venv --quiet --python /usr/bin/python3.12 .venv-p1
export VIRTUAL_ENV="$PWD/.venv-p1"
printf 'torch==2.8.0\ntorchaudio==2.8.0\ntorchcodec==0.7.0\n' > .venv-p1/constraints.txt
"$UV" pip install --quiet --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchaudio==2.8.0"
"$UV" pip install --quiet -c .venv-p1/constraints.txt "torchcodec==0.7.0" "pyannote.audio==4.0.7" \
  "pyannote.metrics==4.2.0" "speechbrain==1.1.2" "s3prl==0.4.18" "soundfile==0.14.0" "numpy<2.3"
"$UV" pip freeze > .venv-p1/freeze.txt
source ops/env.sh
.venv-p1/bin/python - <<'PY'
import numpy, torch, pyannote.audio, s3prl, speechbrain
print("numpy", numpy.__version__, "torch", torch.__version__, "pyannote.audio", pyannote.audio.__version__,
      "speechbrain", speechbrain.__version__, "cuda", torch.cuda.is_available())
PY
echo "ENV_P1_DONE"
