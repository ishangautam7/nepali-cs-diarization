#!/usr/bin/env bash
# Build the speaker-purity audit environment on the GPU box (PREREGISTRATION.md §5 item 4). Safe to re-run.
# The audit embeds clips with two families that are not under test:
#   - NeMo TitaNet-L  (nvidia/speakerverification_en_titanet_large, CC BY 4.0), run on the GPU through NeMo;
#   - WeSpeaker CAM++ (Wespeaker/wespeaker-voxceleb-campplus, Apache-2.0), run through its official ONNX
#     export with onnxruntime on the CPU, so no WeSpeaker training stack is installed.
#   1. .venv-audit: torch/torchaudio 2.8.0 for CUDA 12.8 (as in .venv-pyannote4), nemo-toolkit[asr] 2.7.3
#      (the last 2.x release), onnxruntime 1.31.0, soundfile 0.14.0; the resolved set is written to
#      .venv-audit/freeze.txt
#   2. both checkpoints at pinned Hugging Face revisions into ~/data/models, with SHA-256 printed
#   3. smoke test: each model embeds 3 s of noise; prints the embedding sizes (TitaNet-L 192, CAM++ 512)
# Usage on the box:  bash ops/env_audit.sh > ~/logs/env_audit.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
UV=$HOME/.local/bin/uv
TITANET_REV=0dc382f40121a5fbd34db10a2bb04d826c2be6a8
CAMPP_REV=acf623ad8ca746e50baa432255cf8fc57c669c45

echo "== 1. Python environment"
[ -x .venv-audit/bin/python ] || "$UV" venv --quiet --python /usr/bin/python3.12 .venv-audit
export VIRTUAL_ENV="$PWD/.venv-audit"
cat > .venv-audit/constraints.txt <<'EOF'
torch==2.8.0
torchaudio==2.8.0
EOF
"$UV" pip install --quiet --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchaudio==2.8.0"
"$UV" pip install --quiet -c .venv-audit/constraints.txt \
  "nemo-toolkit[asr]==2.7.3" "onnxruntime==1.31.0" "soundfile==0.14.0"
"$UV" pip freeze > .venv-audit/freeze.txt
wc -l .venv-audit/freeze.txt

echo "== 2. checkpoints"
source ops/env.sh
.venv-audit/bin/python - "$TITANET_REV" "$CAMPP_REV" <<'PY'
import hashlib, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
models = Path.home() / "data/models"
for repo, fname, rev, sub in [
    ("nvidia/speakerverification_en_titanet_large", "speakerverification_en_titanet_large.nemo", sys.argv[1], "titanet-large"),
    ("Wespeaker/wespeaker-voxceleb-campplus", "voxceleb_CAM++.onnx", sys.argv[2], "wespeaker-campplus"),
]:
    p = Path(hf_hub_download(repo, fname, revision=rev, local_dir=models / sub))
    print(repo, rev, p, hashlib.sha256(p.read_bytes()).hexdigest())
PY

echo "== 3. smoke test"
.venv-audit/bin/python - <<'PY'
import numpy as np, torch
from csdiar.data.audit_embed import load_campplus, load_titanet, embed_campplus, embed_titanet
wav = (0.05 * np.random.default_rng(0).standard_normal(48000)).astype(np.float32)
t = load_titanet("cuda")
c = load_campplus()
et, ec = embed_titanet(t, [wav], "cuda"), embed_campplus(c, [wav])
print("torch", torch.__version__, torch.cuda.get_device_name(0), "| titanet", et.shape, "| campplus", ec.shape)
assert et.shape == (1, 192) and ec.shape == (1, 512)
PY
echo "ENV_DONE"
