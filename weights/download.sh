#!/usr/bin/env bash
# YOLOv8s (official Ultralytics COCO release) and the DGSP weights (GitHub release v1.0); md5 of each file is checked.
set -euo pipefail
cd "$(dirname "$0")"
REL=https://github.com/kongwoang/DGSP/releases/download/v1.0
md5() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
get() {  # file url md5
  [ -f "$1" ] || curl -fL -o "$1" "$2"
  [ "$(md5 "$1")" = "$3" ] || { echo "md5 mismatch: $1" >&2; exit 1; }
}
get yolov8s.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt 0a1d5d0619d5a24848a267ec1e3a3118
get looming.pt           $REL/looming.pt           1d9112abccf10951811ff2642b4d017b
get refiner.pt           $REL/refiner.pt           1c84be3045ab9e86da0509e2195e7a77
get adascale_nuscenes.pt $REL/adascale_nuscenes.pt 718ef04e6a9f0815124c802e7f247fc6
get adascale_av2.pt      $REL/adascale_av2.pt      d8d74e24f33f3b8672b7d727fe1b4dd8
get bar_nuscenes.pt      $REL/bar_nuscenes.pt      4c67b4c14cd914daa30a89d167e2e585
get bar_av2.pt           $REL/bar_av2.pt           7817f7d620b7d3a0657500f62b3157ec
get chanakya_nuscenes.pt $REL/chanakya_nuscenes.pt 5ec9dd03d61ada1bd9070516b774ea2b
get chanakya_av2.pt      $REL/chanakya_av2.pt      a69ed42e1f2a4b8314fad5987ae1ba81
echo "weights ready"
