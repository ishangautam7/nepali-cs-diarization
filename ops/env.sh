# Source this on the GPU box before running anything:  source ops/env.sh
# FFmpeg shared libraries (conda-forge, installed by ops/env_pyannote4.sh) are needed by torchcodec.
export LD_LIBRARY_PATH="$HOME/opt/ffmpeg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Model and dataset caches live on the durable EBS disk; /mnt/nvme is scratch only (wiped on stop).
export HF_HOME="$HOME/data/hf"
export TORCH_HOME="$HOME/data/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME" /mnt/nvme/tmp 2>/dev/null || true
export TMPDIR=/mnt/nvme/tmp
