"""Dataset / split -> units, with the repository's data layout (see data/README.md)."""
from __future__ import annotations

from pathlib import Path

from . import argoverse_hd, av2, nuscenes

SPLITS = Path(__file__).resolve().parent / "splits"
# (dataset, split) -> manifest file, data sub-directory
LAYOUT = {
    ("nuscenes", "test"): ("nuscenes_test.txt", "nuscenes"),
    ("av2", "test"): ("av2_val.txt", "av2/sensor/val"),
    ("argoverse_hd", "test"): ("argoverse_hd_val.txt", "argoverse/argoverse-tracking/val"),
}


def manifest(dataset: str, split: str) -> list[str]:
    f, _ = LAYOUT[(dataset, split)]
    return [ln.split()[0] for ln in (SPLITS / f).read_text().split("\n") if ln.strip()]


def open_units(dataset: str, split: str, data_root, limit: int = 0, only=None):
    """Yields (unit, Camera, frames) in sorted order; limit: first N units; only: a set of unit names."""
    data_root = Path(data_root)
    names = manifest(dataset, split)
    if only:
        names = [n for n in names if n in set(only)]
    if limit:
        names = names[:limit]
    _, sub = LAYOUT[(dataset, split)]
    if dataset == "nuscenes":
        yield from nuscenes.units(data_root / sub, split, names)
    elif dataset == "av2":
        yield from av2.units(data_root / sub, names, ego_mask=True)
    else:
        keep = set(names)
        for u in argoverse_hd.units(data_root / "argoverse_hd" / "annotations" / "val.json", data_root / sub):
            if u[0] in keep:
                yield u
