#!/usr/bin/env bash
# One-time OS setup for the g7e GPU box running Amazon Linux 2023. Safe to re-run:
# every step is skipped when its result is already present.
#   1. NVIDIA driver (provides nvidia-smi) from AWS's amazonlinux-nvidia repo. It ships the
#      open kernel modules that Blackwell GPUs require. Skipped if a driver already works.
#   2. Base packages: Python 3.12, compilers, git, tmux, htop, rsync; libsndfile and sox if available.
#   3. fastfetch, pinned to a release and checked against its published SHA-256.
# Run as ec2-user (needs sudo):  bash bootstrap_al2023.sh
# If it prints REBOOT_REQUIRED, run `sudo reboot` so the new kernel module loads.
set -euo pipefail

FASTFETCH_VERSION=2.70.0
# The "polyfilled" build targets older glibc; Amazon Linux 2023 ships glibc 2.34.
FASTFETCH_RPM=fastfetch-linux-amd64-polyfilled.rpm
FASTFETCH_SHA256=64508f6e43ff1468834c85d9771bdf0a8ef70c81bef94e4fbea5677785930e19

reboot_required=0

echo "== 1. NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "driver already working: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
else
  sudo dnf install -y nvidia-release
  sudo dnf install -y nvidia-driver-cuda
  # Load the module now if the build matches the running kernel; otherwise a reboot is needed.
  if sudo modprobe nvidia 2>/dev/null && nvidia-smi >/dev/null 2>&1; then
    echo "driver loaded without reboot"
  else
    reboot_required=1
  fi
fi

echo "== 2. Base packages"
sudo dnf install -y python3.12 python3.12-devel python3.12-pip gcc gcc-c++ make git tmux htop rsync tar xz
for pkg in libsndfile sox; do
  sudo dnf install -y "$pkg" || echo "WARN: $pkg is not in the Amazon Linux repos; it will come from the Python environments instead"
done

echo "== 3. fastfetch $FASTFETCH_VERSION"
if command -v fastfetch >/dev/null 2>&1; then
  echo "fastfetch already installed: $(fastfetch --version)"
else
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL --retry 3 -o "$tmp/$FASTFETCH_RPM" \
    "https://github.com/fastfetch-cli/fastfetch/releases/download/$FASTFETCH_VERSION/$FASTFETCH_RPM"
  echo "$FASTFETCH_SHA256  $tmp/$FASTFETCH_RPM" | sha256sum -c -
  sudo dnf install -y "$tmp/$FASTFETCH_RPM"
  fastfetch --version
fi

if [ "$reboot_required" -eq 1 ]; then
  echo "REBOOT_REQUIRED"
else
  nvidia-smi
fi
echo "BOOTSTRAP_DONE"
