"""Check a data root against the split lists: every test unit present with its images and annotations.
usage: python -m data.prepare --data <root>"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.registry import LAYOUT, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    root = Path(ap.parse_args().data)
    bad = 0
    for (ds, split), (_, sub) in LAYOUT.items():
        names = manifest(ds, split)
        if ds == "nuscenes":
            ok = (root / sub / "v1.0-trainval" / "scene.json").exists() and (root / sub / "samples" / "CAM_FRONT").is_dir()
            scenes = {s["name"] for s in json.loads((root / sub / "v1.0-trainval" / "scene.json").read_text())} if ok else set()
            miss = [n for n in names if n not in scenes]
        elif ds == "av2":
            miss = [n for n in names if not (root / sub / n / "sensors" / "cameras" / "ring_front_center").is_dir()
                    or not (root / sub / n / "annotations.feather").exists()]
        else:
            ann = root / "argoverse_hd" / "annotations" / "val.json"
            logs = [ln.split()[1] for ln in (Path(__file__).parent / "splits" / "argoverse_hd_val.txt").read_text().split("\n") if ln.strip()]
            miss = ([] if ann.exists() else ["annotations/val.json"]) + [g for g in logs if not (root / sub / g).is_dir()]
        bad += len(miss)
        print(f"{ds:13s} {len(names):4d} units, missing {len(miss)}" + (f": {miss[:5]}" if miss else ""))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
