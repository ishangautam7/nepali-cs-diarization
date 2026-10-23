#!/usr/bin/env bash
# Restore the research data that the host's 30-minute backup job copies onto a NEW GPU box, so the study resumes
# where it stopped instead of rebuilding everything (which would change the files FREEZE_H4.json pins and force another
# amendment). Run it on the backup host from the repo root, ONCE, before anything on the new box resumes: it
# overwrites files on the box with their backed-up copies. The new box first needs ops/bootstrap_al2023.sh,
# ops/box_mount.sh, the repo copy at ~/nepali-cs-diarization and its environments (ops/rebuild_envs.sh), and
# ops/ssh_config needs a Host block for its IP.
#
#   bash ops/restore_box_from_backup.sh ec2-user@<new-ip> [backup-dir]
#
# 1. Refuses unless the box has the repo copy and all five Python environments.
# 2. Copies <backup>/data/ into the box's ~/data/ and <backup>/logs/ into ~/logs/restored/. No --delete: nothing on the
#    box is removed.
# 3. Writes the ops/rebuild_features.sh done-marker of every step whose outputs were all restored, so the rebuild then
#    re-runs only what the backup leaves out (the D1 and D5 WAVs, the M3 audio memmaps, the gated downloads), all of
#    which reproduce byte-for-byte. Re-running a restored step instead would rewrite its files (gzip headers carry a
#    timestamp), so their pinned hashes would no longer match.
# 4. Compares each environment with its package list in ops/locks/venv-*.freeze.txt (what this study runs with, frozen
#    on 2026-10-19) and prints every package that differs. The env scripts pin only the main packages, so a later
#    rebuild can pick up newer releases of the rest: install the locked version of each differing package before
#    anything resumes (torch and torchaudio come from https://download.pytorch.org/whl/cu128, as in ops/env_*.sh).
# Then on the box, from ~/nepali-cs-diarization:
#   nohup bash ops/rebuild_features.sh m3-inputs ... (needs `hf auth login` for IndicVoices), then `... features`;
#   python3.12 scripts/amend_freeze_h4_rebuild.py verify    # every file FREEZE_H4.json pins must match
#   nohup bash ops/run_m3_queue.sh ops/queues/m3_h4.txt 2 ...  # finished jobs and grid splits are skipped
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${1:?usage: bash ops/restore_box_from_backup.sh ec2-user@<ip> [backup-dir]}
SRC=${2:-$HOME/nepdiar-box-backup}
SSH="ssh -F ops/ssh_config -o ConnectTimeout=20"
[ -d "$SRC/data" ] || { echo "no backup at $SRC/data"; exit 1; }
$SSH "$HOST" 'cd ~/nepali-cs-diarization && for v in pyannote4 e3 audit data p1; do [ -x .venv-$v/bin/python ] || exit 1; done' \
  || { echo "the box needs the repo copy at ~/nepali-cs-diarization and all five environments first (ops/rebuild_envs.sh)"; exit 1; }

rsync -a -e "$SSH" "$SRC/data/" "$HOST:data/"
if [ -d "$SRC/logs" ]; then rsync -a -e "$SSH" "$SRC/logs/" "$HOST:logs/restored/"; fi

$SSH "$HOST" 'bash -s' <<'EOF'
set -u
D=$HOME/data
M=$D/.rebuild_features
mkdir -p "$M"
mark() {   # mark <step> <output>...: write the done-marker only if every listed output was restored
  local step=$1 f
  shift
  for f in "$@"; do
    if [ ! -s "$f" ]; then echo "re-run: $step ($f not in the backup)"; return 0; fi
  done
  date -u +%FT%TZ > "$M/$step.done"
  echo "restored: $step"
}
S="validation train test"
mark d1_align $(for s in $S; do echo "$D/d1/align/$s.jsonl"; done)
mark d1_vad $(for s in $S; do echo "$D/d1/vad/$s.npz"; done)
mark d1_boundaries "$D/d1/boundaries/lex_v1b/summary.json" \
  $(for k in points matches; do for s in $S; do echo "$D/d1/boundaries/lex_v1b/${k}_$s.csv.gz"; done; done)
mark d1_windows "$D/d1/windows/lex_v1/summary.json" $(for s in $S; do echo "$D/d1/windows/lex_v1/windows_$s.csv.gz"; done)
mark d1_audit_titanet $(for s in $S; do echo "$D/d1/audit/titanet/$s.npz"; done)
mark d1_audit_campplus $(for s in $S; do echo "$D/d1/audit/campplus/$s.npz"; done)
mark d1_purity "$D/d1/results/purity/subset.json"
mark m3_d1_windows "$D/m3/d1_windows.npz"
mark m3_dev_trials "$D/m3/dev_trials.npz"
for m in e3 e1 e2; do
  mark "d1_emb_${m}_L1" $(for s in $S; do echo "$D/d1/emb/${m}_L1/$s.npz"; done)
  mark "d1_point_$m" "$D/d1/point_emb/lex_v1b/$m/train_L1.npz"
done
mark d1_lid $(for s in $S; do echo "$D/d1/lid/$s.npz"; done)
echo "left for rebuild_features.sh: d1_ingest, d5_fetch, d5_pool, m3_audio, c1_fetch and any step listed as re-run above"

echo "--- environments against their package lists (ops/locks/venv-*.freeze.txt)"
cd "$HOME/nepali-cs-diarization"
for lock in ops/locks/venv-*.freeze.txt; do
  v=${lock#ops/locks/venv-}
  v=${v%.freeze.txt}
  d=$(diff <(grep -v '^#' "$lock") <("$HOME/.local/bin/uv" pip freeze --python ".venv-$v/bin/python" 2>/dev/null) | grep -E '^[<>]')
  if [ -z "$d" ]; then echo ".venv-$v: matches"; else echo ".venv-$v differs (< locked, > installed):"; echo "$d"; fi
done
EOF
