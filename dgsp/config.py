"""The DGSP configuration (the only place where its values are set) and the switches of the ablation table."""
from __future__ import annotations

from dataclasses import dataclass, replace

R_GRID = (12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0)    # field grid: ranges ahead [m]
Y_GRID = (-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0)    # field grid: lateral offsets [m, + left]
CAR = (1.8, 1.6)                                             # probe object (width, height) [m]


@dataclass(frozen=True)
class Config:
    rounds: int = 3                  # round cap of the loop
    closing_speed: float = 3.0       # closing speed of the probe object [m/s]
    margin: float = 0.4              # field: a cell is sensitive if the probe's demand reaches (1 - margin) x tau_next
    context: float = 2.0             # support / track crop = box size x context
    window_long_side: int = 960      # crops are detected on the stride-32 grid of the frame letterboxed to this side
    track: bool = True               # first round also re-checks the previous frame's verified support objects
    guard: str | None = "filtered"   # truncation guard: "filtered" (skips boxes that cannot matter), "plain", None
    max_grow: int = 2                # guard: re-detections per truncated box
    retro: bool = True               # same-scale history for newly verified objects (looming chain)
    retro_range: float = 40.0        # ... for objects in the ego corridor within this range [m]
    field: bool = True               # field crops
    support: str = "support"         # support crops: "support" | "pivots" | "none"
    footprint: str = "exact"         # field footprint: "exact" (ego -> camera transform) | "pinhole" (camera height only)
    r_grid: tuple = R_GRID           # field ranges (extended when the range limit is lifted, see eval/run.py SETTINGS)


DGSP = Config()
DGSP_FAST = replace(DGSP, rounds=1)
# the loop driven by the TTC-warning planner (other planners): no track seeds, static probe, pinhole footprint
DGSP_TTC = replace(DGSP, closing_speed=0.0, track=False, footprint="pinhole")

ABLATIONS = {                         # ablation table; "w/o refiner" = DGSP's action without the refiner (record field a_none)
    "support only": replace(DGSP, field=False),
    "field only": replace(DGSP, support="none"),
    "pivots": replace(DGSP, support="pivots"),
    "single round": replace(DGSP, rounds=1),
    "w/o guard": replace(DGSP, guard=None),
}
METHODS = {"DGSP": DGSP, "DGSP-fast": DGSP_FAST, **ABLATIONS}

# detector and data
KEEP_CONF = 0.25                                             # detections below this confidence are discarded
DRAFT_CANVAS = {"nuscenes": (192, 320), "av2": (320, 256), "argoverse_hd": (224, 320)}   # (height, width) of u320
NATIVE_LONG = {"nuscenes": 1600, "av2": 2048, "argoverse_hd": 1920}
