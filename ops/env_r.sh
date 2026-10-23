#!/usr/bin/env bash
# R environment for the mixed-model sensitivity analyses (PREREGISTRATION.md §6; csdiar/analysis/glmm.R):
# conda-forge r-base 4.5, lme4 2.0-6 and jsonlite, in ~/opt/r-glmm, built with the pinned micromamba from
# ops/env_pyannote4.sh. The exact build list (URLs and MD5s) is written to ops/locks/r-glmm.explicit.txt
# on the box; copy it into the repository after a rebuild. Safe to re-run.
# Usage on the box:  bash ops/env_r.sh > ~/logs/env_r.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
OPT=$HOME/opt
ENV=$OPT/r-glmm
if [ ! -x "$ENV/bin/Rscript" ]; then
  "$OPT/micromamba" create -y -q -r "$OPT/mamba-root" -p "$ENV" -c conda-forge --override-channels \
    "r-base=4.5" "r-lme4=2.0_6" "r-jsonlite"
fi
mkdir -p ops/locks
"$OPT/micromamba" env export -r "$OPT/mamba-root" -p "$ENV" --explicit --md5 > ops/locks/r-glmm.explicit.txt
"$ENV/bin/Rscript" -e 'cat(R.version.string, "| lme4", format(packageVersion("lme4")), "| Matrix",
  format(packageVersion("Matrix")), "| jsonlite", format(packageVersion("jsonlite")), "\n")'
echo "ENV_R_DONE"
