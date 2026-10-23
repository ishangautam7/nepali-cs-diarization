#!/usr/bin/env bash
# Build the arXiv source package for paper/main.tex and test-compile it the way arXiv does.
#
#   1. Compile paper/main.tex with pdflatex + BibTeX to produce main.bbl.
#   2. Pack main.tex, main.bbl and the figures the paper actually includes (no .bib, no unused
#      PNGs, no build files) into paper/arxiv/arxiv_source.tar.gz, deterministically.
#   3. Unpack the tarball in a fresh directory and compile it with pdflatex only (no BibTeX, no
#      refs.bib), as arXiv does when a .bbl is present. Fail on any error or undefined reference.
#   4. Keep that PDF as paper/arxiv/Gautam_codeswitch_diarization.pdf, the version readers get.
#
# Needs TeX Live (pdflatex, bibtex, IEEEtran, cite, psnfss Times/Courier/Helvetica) and GNU tar.
# Override the tools with PDFLATEX=... BIBTEX=... if they are not on PATH.
set -euo pipefail

PDFLATEX=${PDFLATEX:-pdflatex}
BIBTEX=${BIBTEX:-bibtex}
# Fixed PDF dates (2026-10-22 00:00 UTC) make rebuilds byte-identical; override for a new version.
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1792627200}
here=$(cd "$(dirname "$0")" && pwd)
paper=$(cd "$here/.." && pwd)
out_tar="$here/arxiv_source.tar.gz"
out_pdf="$here/Gautam_codeswitch_diarization.pdf"

work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

latex() { # dir: run pdflatex once, fail with the log tail on error
  (cd "$1" && "$PDFLATEX" -interaction=nonstopmode -halt-on-error main.tex > pdflatex.out 2>&1) \
    || { tail -40 "$1/pdflatex.out"; exit 1; }
}

# The figures main.tex includes, in order (explicit paths with extensions).
mapfile -t figs < <(grep -o '\\includegraphics\(\[[^]]*\]\)\?{[^}]*}' "$paper/main.tex" | sed 's/.*{\(.*\)}/\1/')
[ "${#figs[@]}" -gt 0 ] || { echo "no \\includegraphics found" >&2; exit 1; }

# 1. Full build to get main.bbl.
mkdir -p "$work/full"
cp "$paper/main.tex" "$paper/refs.bib" "$work/full/"
for f in "${figs[@]}"; do mkdir -p "$work/full/$(dirname "$f")"; cp "$paper/$f" "$work/full/$f"; done
latex "$work/full"
(cd "$work/full" && "$BIBTEX" main > bibtex.out 2>&1) || { cat "$work/full/bibtex.out"; exit 1; }
latex "$work/full"; latex "$work/full"

# 2. Stage exactly the files arXiv needs and pack them reproducibly.
mkdir -p "$work/pkg"
cp "$work/full/main.tex" "$work/full/main.bbl" "$work/pkg/"
for f in "${figs[@]}"; do mkdir -p "$work/pkg/$(dirname "$f")"; cp "$work/full/$f" "$work/pkg/$f"; done
(cd "$work/pkg" && tar --sort=name --mtime='2026-10-22 00:00Z' --owner=0 --group=0 --numeric-owner \
  --mode='u+rw,go+rX,go-w' -cf - main.tex main.bbl "${figs[@]}") | gzip -n -9 > "$out_tar"

# 3. Compile the unpacked tarball with pdflatex alone.
mkdir -p "$work/check"
tar -xzf "$out_tar" -C "$work/check"
latex "$work/check"; latex "$work/check"; latex "$work/check"
log="$work/check/main.log"
if grep -Eq '(Reference|Citation) .* undefined|There were undefined references|Rerun to get' "$log"; then
  grep -E '(Reference|Citation) .* undefined|There were undefined references|Rerun to get' "$log" >&2; exit 1
fi

# 4. Keep the checked PDF and report.
cp "$work/check/main.pdf" "$out_pdf"
echo "package: $out_tar"
tar -tzvf "$out_tar"
echo "sha256:  $(sha256sum "$out_tar" | cut -d' ' -f1)"
echo "pdf:     $out_pdf ($(grep -o 'Output written on main.pdf ([0-9]* pages' "$log" | grep -o '[0-9]* pages'))"
grep -E 'Overfull|LaTeX Warning' "$log" || echo "no overfull boxes or LaTeX warnings"
