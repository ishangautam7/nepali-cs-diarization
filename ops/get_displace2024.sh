#!/usr/bin/env bash
# Fetch the two password-protected DISPLACE 2024 supervised archives (Zenodo record 14097187) into
# ~/data/displace/zips and check them against Zenodo's own size and MD5. Nothing is extracted or listed: the
# archives stay encrypted until Ishan puts the password in ~/.config/csdiar/keys.env (PREREGISTRATION_M2B.md §2).
# Resumable and safe to re-run; writes FETCH_DONE (name, size, md5) once both files verify. Deletes nothing.
# Usage on the box:  nohup bash ops/get_displace2024.sh > ~/logs/displace_fetch.log 2>&1 < /dev/null &
set -euo pipefail
RECORD=14097187
DEST=${DISPLACE_ZIPS:-$HOME/data/displace/zips}
FILES=(Track1_SD_supervised.zip Track2_LD_supervised.zip)
mkdir -p "$DEST"
cd "$DEST"
curl -fsSL --retry 5 -o files.json "https://zenodo.org/api/records/$RECORD/files"

meta() { # name -> "size md5 url" from Zenodo's file list
  python3 -c 'import json, sys
name = sys.argv[1]
e = next(x for x in json.load(open("files.json"))["entries"] if x["key"] == name)
algo, _, digest = e["checksum"].partition(":")
assert algo == "md5", e["checksum"]
print(e["size"], digest, e["links"]["content"])' "$1"
}

for f in "${FILES[@]}"; do
  read -r size md5 url < <(meta "$f")
  if [ -s "$f" ] && [ "$(stat -c %s "$f")" = "$size" ] && echo "$md5  $f" | md5sum -c --quiet -; then
    echo "$(date -u +%FT%TZ) $f already verified"
    continue
  fi
  echo "$(date -u +%FT%TZ) fetching $f ($size bytes)"
  curl -fL --silent --show-error --retry 10 --retry-delay 30 -C - -o "$f.part" "$url"
  [ "$(stat -c %s "$f.part")" = "$size" ] || { echo "size mismatch for $f.part"; exit 1; }
  echo "$md5  $f.part" | md5sum -c --quiet - || { echo "md5 mismatch for $f.part (left in place)"; exit 1; }
  mv "$f.part" "$f"
  echo "$(date -u +%FT%TZ) verified $f"
done
for f in "${FILES[@]}"; do
  printf '%s\t%s\t%s\n' "$f" "$(stat -c %s "$f")" "$(md5sum "$f" | cut -d' ' -f1)"
done > FETCH_DONE
echo "$(date -u +%FT%TZ) FETCH_DONE"
