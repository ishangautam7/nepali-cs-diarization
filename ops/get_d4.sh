#!/usr/bin/env bash
# D4 for H7 (PLAN.md §3): MUCS 2021 sub-task 2, OpenSLR 104 (CC BY-SA 4.0), Hindi-English and Bengali-English
# train and test archives (real spoken-tutorial recordings). The EU mirror is about 5x faster from us-east-1
# than the primary. Resumable (wget -c), each archive gzip-tested, SHA-256 recorded in SHA256SUMS.
# The blind test set has no public transcripts at the time of writing and is not downloaded.
# Usage on the box:  nohup bash ops/get_d4.sh > ~/logs/d4_download.log 2>&1 &
set -uo pipefail
DEST=${1:-$HOME/data/d4/raw}
MIRROR=https://openslr.elda.org/resources/104
mkdir -p "$DEST"
cd "$DEST"
for f in Hindi-English_test Bengali-English_test Hindi-English_train Bengali-English_train; do
  ( if wget -q -c "$MIRROR/$f.tar.gz" && gzip -t "$f.tar.gz"; then echo "$(date -u +%H:%M) $f ok"
    else echo "$(date -u +%H:%M) $f FAILED"; fi ) &
done
wait
sha256sum ./*.tar.gz > SHA256SUMS
echo "D4_DONE"
