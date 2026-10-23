#!/usr/bin/env bash
# Rebuild the derived D1 features and M3 inputs on a fresh GPU box after the 2026-10-19 box loss
# (planning/DEVIATIONS.md entry 6), with the code frozen in planning/FREEZE_H4.json: all 18 guarded files and every
# unguarded producer used here are unchanged since the freeze commit cd22caf. It starts where ops/rebuild_envs.sh
# stops (the environments, the WavLM SV checkpoint, the public D1 parquet in ~/data/d1_hf and D4).
#
# Two phases, so M3 training (the long pole) can start as early as possible:
#   m3-inputs  D1 ingest, alignment, VAD, boundaries (lex_v1b), clean windows (lex_v1), purity-audit embeddings and
#              subset, the D5 pool from IndicVoices Nepali (gated), M3 window labels, dev trials, audio memmaps.
#              Then: nohup bash ops/run_m3_queue.sh ops/queues/m3_h4.txt 2 > ~/logs/m3_queue.log 2>&1 &
#   features   the dense L = 1 s grids of E1, E2 and E3, VoxLingua107 LID, and point embeddings for E1, E2 and E3:
#              what confirm_h4 and secondary_h4 read. It needs only the boundaries from m3-inputs, so it can run
#              alongside it; pyannote Community-1 (gated; E2 is its embedding) is fetched last, after every step
#              that needs no Hugging Face login.
# Only L = 1 s grids are built: H4 and its secondaries read nothing else (the 0.5 s and 1.5 s grids served the
# finished M2a length sensitivity), and each window length is a separate run of csdiar.models.embed.
#
# Each step logs to ~/logs/rebuild/<step>.log and, when it succeeds, writes ~/data/.rebuild_features/<step>.done;
# a step with a done-marker is skipped, so a re-run resumes after the last finished step. The chain stops at the
# first failure. Nothing is deleted. Gated downloads use Ishan's Hugging Face login on the box, from either
# ~/data/hf/token (`source ops/env.sh` first) or ~/.cache/huggingface/token (a plain `hf auth login`).
# GPU steps hold ~/run/gpu_reserve_gb at RESERVE_GB (default 24) and put 16 back afterwards, as run_embeddings.sh.
#
# Usage on the box:  nohup bash ops/rebuild_features.sh m3-inputs > ~/logs/rebuild_features.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source ops/env.sh
PHASE=${1:-}
case "$PHASE" in
  m3-inputs|features) ;;
  *) echo "usage: bash ops/rebuild_features.sh m3-inputs|features"; exit 2 ;;
esac

PY4=.venv-pyannote4/bin/python      # alignment, VAD, boundaries, windows, purity audit, LID, E1/E2 (as originally)
PYE3=.venv-e3/bin/python            # the E3 grid and E3 point embeddings
PYAUDIT=.venv-audit/bin/python      # TitaNet / CAM++ audit embeddings
PYIO=.venv-audit/bin/python         # parquet + audio I/O (ingest, m3_data): the rebuilt env with pyarrow+soundfile+numpy
PYDATA=.venv-data/bin/python        # Hugging Face downloads (huggingface_hub 0.34.4)
D1=$HOME/data/d1
M3=$HOME/data/m3
D5=$HOME/data/d5
IV=$HOME/data/indicvoices_nepali
IV_REPO=ai4bharat/IndicVoices
IV_REV=c96f9088f138cf89d419da7e8e643e1f05c00a87       # DECISIONS.md: 74 parquet files under nepali/, 28.2 GB
C1=$HOME/data/models/pyannote-community-1
C1_REPO=pyannote/speaker-diarization-community-1
C1_REV=3533c8cf8e369892e6b79ff1bf80f7b0286a54ee       # DECISIONS.md / PREREGISTRATION.md (E2)
LEX=(--loans planning/lexicon/loans.txt --names planning/lexicon/names.txt)
SPLITS=(--splits validation train test)
LOGS=$HOME/logs/rebuild
DONE=$HOME/data/.rebuild_features
RESERVE_GB=${RESERVE_GB:-24}
mkdir -p "$LOGS" "$DONE" ~/run

say() { echo "$(date -u +%FT%TZ) $*"; }
die() { say "STOP: $*"; exit 1; }
gpu_on()  { echo "$RESERVE_GB" > ~/run/gpu_reserve_gb; }
gpu_off() { echo 16 > ~/run/gpu_reserve_gb; }
trap gpu_off EXIT

for py in "$PY4" "$PYE3" "$PYAUDIT" "$PYDATA"; do
  [ -x "$py" ] || die "missing interpreter $py (run ops/rebuild_envs.sh first)"
done

# step <name> <cpu|gpu> <command...>: skipped when its done-marker exists; stops the chain on failure.
step() {
  local name=$1 kind=$2
  shift 2
  if [ -e "$DONE/$name.done" ]; then say "skip  $name (done)"; return 0; fi
  say "start $name"
  if [ "$kind" = gpu ]; then gpu_on; fi
  if "$@" > "$LOGS/$name.log" 2>&1; then
    if [ "$kind" = gpu ]; then gpu_off; fi
    date -u +%FT%TZ > "$DONE/$name.done"
    say "ok    $name"
  else
    die "$name failed (see $LOGS/$name.log)"
  fi
}

# Point huggingface_hub at Ishan's login wherever `hf auth login` wrote it (the token itself is never printed).
hf_token() {
  local p
  for p in "$HF_HOME/token" "$HOME/.cache/huggingface/token"; do
    if [ -s "$p" ]; then export HF_TOKEN_PATH="$p"; return 0; fi
  done
  die "no Hugging Face login on the box: run 'hf auth login' there (needed for the gated $1)"
}

fetch_indicvoices() {
  "$PYDATA" - "$IV" "$IV_REPO" "$IV_REV" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
dest, repo, rev = sys.argv[1:4]
snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=dest, allow_patterns=["nepali/*.parquet"])
n = len(list(Path(dest, "nepali").glob("*.parquet")))
print("parquet files under nepali/:", n)
sys.exit(0 if n == 74 else 1)
PY
}

fetch_community1() {
  "$PYDATA" - "$C1" "$C1_REPO" "$C1_REV" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
dest, repo, rev = sys.argv[1:4]
snapshot_download(repo, revision=rev, local_dir=dest)
ok = Path(dest, "embedding").is_dir()
print("embedding/ present:", ok)
sys.exit(0 if ok else 1)
PY
}

if [ "$PHASE" = m3-inputs ]; then
  [ -d "$HOME/data/d1_hf/data" ] || die "missing ~/data/d1_hf/data (ops/rebuild_envs.sh downloads it)"
  [ -s "$HOME/data/models/unispeech_sv/wavlm_large_finetune.pth" ] || die "missing the WavLM SV checkpoint"
  step d1_ingest cpu "$PYIO" -m csdiar.data.ingest_nepali_cs --shards "$HOME/data/d1_hf/data" --out "$D1"
  step d1_align gpu "$PY4" -W ignore -m csdiar.data.align --root "$D1" --out "$D1/align" "${SPLITS[@]}"
  step d1_vad gpu "$PY4" -m csdiar.data.vad --root "$D1" --out "$D1/vad" "${SPLITS[@]}"
  step d1_boundaries cpu "$PY4" -m csdiar.data.boundaries --root "$D1" --align "$D1/align" --vad "$D1/vad" \
    --out "$D1/boundaries/lex_v1b" "${LEX[@]}" "${SPLITS[@]}"
  step d1_windows cpu "$PY4" -m csdiar.data.windows --root "$D1" --align "$D1/align" --vad "$D1/vad" \
    --out "$D1/windows/lex_v1" "${LEX[@]}" "${SPLITS[@]}"
  step d1_audit_titanet gpu "$PYAUDIT" -m csdiar.data.audit_embed --model titanet --root "$D1" --vad "$D1/vad" \
    --out "$D1/audit" "${SPLITS[@]}"
  step d1_audit_campplus cpu "$PYAUDIT" -m csdiar.data.audit_embed --model campplus --root "$D1" --vad "$D1/vad" \
    --out "$D1/audit" "${SPLITS[@]}"
  step d1_purity cpu "$PY4" -m csdiar.analysis.purity_audit --root "$D1" --emb "$D1/audit" --out "$D1/results/purity"
  [ -e "$DONE/d5_fetch.done" ] || hf_token "IndicVoices Nepali dataset (the D5 pool)"
  step d5_fetch cpu fetch_indicvoices
  step d5_pool cpu "$PYIO" -m csdiar.mitigations.m3_data d5 --src "$IV/nepali" --out "$D5"
  step m3_d1_windows cpu "$PYIO" -m csdiar.mitigations.m3_data d1 --root "$D1" --align "$D1/align" \
    --boundaries "$D1/boundaries/lex_v1b" --audit-emb "$D1/audit" --purity "$D1/results/purity" "${LEX[@]}" --out "$M3"
  step m3_dev_trials cpu "$PYIO" -m csdiar.mitigations.m3_data dev --windows "$D1/windows/lex_v1" --out "$M3"
  step m3_audio cpu "$PYIO" -m csdiar.mitigations.m3_data audio --root "$D1" --d5 "$D5" --out "$M3" --which d1 d5
  say "M3_INPUTS_DONE"
fi

if [ "$PHASE" = features ]; then
  [ -e "$DONE/d1_boundaries.done" ] || die "run the m3-inputs phase first (boundaries are missing)"
  # Everything that needs no Hugging Face login first; the gated Community-1 download (E2) comes last.
  step d1_emb_e3_L1 gpu "$PYE3" -W ignore -m csdiar.models.embed --model e3 --root "$D1" --out "$D1/emb" --win 1.0
  step d1_lid gpu "$PY4" -m csdiar.models.lid --root "$D1" --out "$D1/lid" "${SPLITS[@]}"
  step d1_point_e3 gpu "$PYE3" -W ignore -m csdiar.models.point_embed --model e3 --root "$D1" \
    --points "$D1/boundaries/lex_v1b" --out "$D1/point_emb/lex_v1b" "${SPLITS[@]}"
  step d1_emb_e1_L1 gpu "$PY4" -W ignore -m csdiar.models.embed --model e1 --root "$D1" --out "$D1/emb" --win 1.0
  step d1_point_e1 gpu "$PY4" -W ignore -m csdiar.models.point_embed --model e1 --root "$D1" \
    --points "$D1/boundaries/lex_v1b" --out "$D1/point_emb/lex_v1b" "${SPLITS[@]}"
  [ -e "$DONE/c1_fetch.done" ] || hf_token "pyannote Community-1 model (E2)"
  step c1_fetch cpu fetch_community1
  step d1_emb_e2_L1 gpu "$PY4" -W ignore -m csdiar.models.embed --model e2 --root "$D1" --out "$D1/emb" --win 1.0
  step d1_point_e2 gpu "$PY4" -W ignore -m csdiar.models.point_embed --model e2 --root "$D1" \
    --points "$D1/boundaries/lex_v1b" --out "$D1/point_emb/lex_v1b" "${SPLITS[@]}"
  say "FEATURES_DONE"
fi
