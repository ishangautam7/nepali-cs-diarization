#!/usr/bin/env bash
# Build the E3 environment on the GPU box (PLAN.md §5: E3 = WavLM-Large + ECAPA-TDNN, UniSpeech SV checkpoint).
# Kept separate from .venv-pyannote4 because s3prl brings its own dependency set.
#   .venv-e3: torch/torchaudio 2.8.0 (CUDA 12.8, Blackwell), s3prl 0.4.18, soundfile 0.14.0, numpy
#   checkpoint: ~/data/models/unispeech_sv/wavlm_large_finetune.pth (UniSpeech "WavLM large, Fix pre-train: No",
#   Vox1-O EER 0.431%; sha256 51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b)
# Usage on the box:  bash ops/env_e3.sh > ~/logs/env_e3.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
UV=$HOME/.local/bin/uv
CKPT=$HOME/data/models/unispeech_sv/wavlm_large_finetune.pth
[ -x .venv-e3/bin/python ] || "$UV" venv --quiet --python /usr/bin/python3.12 .venv-e3
export VIRTUAL_ENV="$PWD/.venv-e3"
printf 'torch==2.8.0\ntorchaudio==2.8.0\n' > .venv-e3/constraints.txt
"$UV" pip install --quiet --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchaudio==2.8.0"
"$UV" pip install --quiet -c .venv-e3/constraints.txt "s3prl==0.4.18" "soundfile==0.14.0" "numpy<2.3"
echo "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b  $CKPT" | sha256sum -c -
source ops/env.sh
.venv-e3/bin/python -c "import torch, s3prl; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 's3prl ok')"
echo "ENV_E3_DONE"
