#!/usr/bin/env bash
# Official code of four baselines at fixed commits (not vendored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
fetch() {  # url dir commit
  [ -d "third_party/$2/.git" ] || git clone --quiet "$1" "third_party/$2"
  git -C "third_party/$2" checkout --quiet "$3"
  echo "$2 @ $(git -C "third_party/$2" rev-parse --short HEAD)"
}
fetch https://github.com/PuAnysh/UFPMP-Det.git UFPMP-Det 6809b4f   # UFP packing (baselines/ufp.py)
fetch https://github.com/tchittesh/fovea.git fovea c8707bd             # FOVEA grid generator (baselines/fovea.py)
fetch https://github.com/ViGeng/bgt-ada.git bgt-ada 6669ab0            # BAR estimator, Chanakya state features (bar.py, chanakya.py)
