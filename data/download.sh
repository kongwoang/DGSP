#!/usr/bin/env bash
# Fetch the test splits into a data root (each dataset's licence applies; nuScenes and Argoverse-HD need manual steps).
#   nuscenes      v1.0-trainval metadata + CAM_FRONT keyframes: put v1.0-trainval_meta.tgz and v1.0-trainval*_blobs.tgz
#                 (nuscenes.org, account required) into <root>/downloads/nuscenes first
#   av2           Argoverse 2 sensor val logs of splits/av2_val.txt (public bucket, s5cmd): ring_front_center,
#                 calibration, poses, annotations
#   argoverse_hd  Argoverse-HD annotations + Argoverse 1.1 tracking val: put Argoverse-HD-Full.zip and
#                 tracking_val_v1.1.tar.gz into <root>/downloads first
# usage: bash data/download.sh nuscenes|av2|argoverse_hd|all [root (default: datasets)]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); what=${1:-}; root=${2:-datasets}
[[ -z "$what" || "$what" == -h || "$what" == --help ]] && { sed -n 2,9p "$0"; exit 0; }
mkdir -p "$root/downloads"
nuscenes() {
  local src=$root/downloads/nuscenes dst=$root/nuscenes; mkdir -p "$dst"
  [[ -f $src/v1.0-trainval_meta.tgz ]] || { echo "put the nuScenes v1.0-trainval archives in $src"; exit 1; }
  tar -xzf "$src/v1.0-trainval_meta.tgz" -C "$dst"
  for f in "$src"/v1.0-trainval*_blobs.tgz; do tar -xzf "$f" -C "$dst" --wildcards 'samples/CAM_FRONT/*'; done
}
av2() {
  command -v s5cmd > /dev/null || { echo "install s5cmd (github.com/peak/s5cmd)"; exit 1; }
  while read -r log; do
    [[ -z "$log" ]] && continue
    local d=$root/av2/sensor/val/$log; mkdir -p "$d"
    for f in annotations.feather city_SE3_egovehicle.feather "calibration/*"; do
      s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/$log/$f" "$d/$(dirname "$f")/" > /dev/null
    done
    s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/val/$log/sensors/cameras/ring_front_center/*" \
      "$d/sensors/cameras/ring_front_center/" > /dev/null
  done < "$here/splits/av2_val.txt"
}
argoverse_hd() {
  mkdir -p "$root/argoverse_hd" "$root/argoverse/argoverse-tracking"
  [[ -f $root/downloads/Argoverse-HD-Full.zip ]] || { echo "put Argoverse-HD-Full.zip (Argoverse-HD project page) in $root/downloads"; exit 1; }
  [[ -f $root/downloads/tracking_val_v1.1.tar.gz ]] || { echo "put tracking_val_v1.1.tar.gz (argoverse.org) in $root/downloads"; exit 1; }
  unzip -q -o "$root/downloads/Argoverse-HD-Full.zip" "Argoverse-HD/annotations/*" -d "$root/downloads/ahd"
  rm -rf "$root/argoverse_hd/annotations"; mv "$root/downloads/ahd/Argoverse-HD/annotations" "$root/argoverse_hd/annotations"
  tar -xzf "$root/downloads/tracking_val_v1.1.tar.gz" -C "$root/argoverse/"
}
case $what in
  nuscenes) nuscenes ;; av2) av2 ;; argoverse_hd) argoverse_hd ;; all) nuscenes; av2; argoverse_hd ;;
  *) echo "unknown: $what"; exit 1 ;;
esac
