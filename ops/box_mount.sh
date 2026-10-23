#!/usr/bin/env bash
# Mount the EC2 local NVMe instance store at /mnt/nvme, used only as fast scratch (caches, temporary
# features). Instance store is wiped on stop/start (kept across reboot), so run this after every boot
# and never keep anything there that cannot be regenerated. Durable data lives on the EBS root volume.
# The disk is formatted ONLY when it carries no filesystem at all (a blank instance store); anything
# with a filesystem is mounted as is, never reformatted.
set -euo pipefail
mnt=/mnt/nvme
if mountpoint -q "$mnt"; then echo "already mounted: $mnt"; df -h "$mnt" | tail -1; exit 0; fi
dev=$(ls /dev/disk/by-id/nvme-Amazon_EC2_NVMe_Instance_Storage_* 2>/dev/null | grep -v -- '-part' | head -1 || true)
[ -n "$dev" ] || { echo "no instance store on this instance; nothing to mount"; exit 0; }
dev=$(readlink -f "$dev")
if [ -z "$(sudo blkid -o value -s TYPE "$dev" 2>/dev/null || true)" ]; then
  echo "formatting blank instance store $dev as xfs"
  sudo mkfs.xfs -q "$dev"
fi
sudo mkdir -p "$mnt"
sudo mount -o noatime "$dev" "$mnt"
sudo chown "$(id -u):$(id -g)" "$mnt"
df -h "$mnt" | tail -1
