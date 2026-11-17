"""nuScenes CAM_FRONT keyframes (read from the JSON tables; no devkit needed)."""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dgsp.lift import Calib, Camera

from .common import GEOM_FIELDS, fill_range_rate, frame_geometry, yaw_rates

FRAME_DT = 0.5               # keyframes are 2 Hz
IMG_W, IMG_H = 1600, 900
CAM_HEIGHT = 1.51            # CAM_FRONT height above the road used by the lift [m]
EVAL_PREFIXES = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.trailer", "vehicle.construction",
                 "human.pedestrian", "vehicle.motorcycle", "vehicle.bicycle")


def coarse_class(category: str) -> str:
    if category.startswith("human."):
        return "person"
    if category.startswith(("vehicle.bicycle", "vehicle.motorcycle")):
        return "cyclist"
    return "vehicle"


def _rot(q):
    return Rotation.from_quat(np.roll(q, -1)).as_matrix()          # w, x, y, z -> x, y, z, w


class NuScenes:
    def __init__(self, dataroot, version: str = "v1.0-trainval"):
        self.root = Path(dataroot)
        t = self.root / version
        self._t = {n: {r["token"]: r for r in json.loads((t / f"{n}.json").read_text())}
                   for n in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor", "instance", "category")}
        self._ann = {}
        for a in json.loads((t / "sample_annotation.json").read_text()):
            self._ann.setdefault(a["sample_token"], []).append(a)
        cam = {k for k, v in self._t["sensor"].items() if v["channel"] == "CAM_FRONT"}
        self._cam = {}
        for sd in self._t["sample_data"].values():
            if sd["is_key_frame"] and self._t["calibrated_sensor"][sd["calibrated_sensor_token"]]["sensor_token"] in cam:
                self._cam[sd["sample_token"]] = sd
        self.by_name = {v["name"]: k for k, v in self._t["scene"].items()}

    @functools.lru_cache(maxsize=256)
    def samples(self, scene: str) -> tuple:
        out, tok = [], self._t["scene"][scene]["first_sample_token"]
        while tok:
            out.append(tok)
            tok = self._t["sample"][tok]["next"]
        return tuple(out)

    def image_path(self, tok) -> Path:
        return self.root / self._cam[tok]["filename"]

    def _cs(self, tok):
        return self._t["calibrated_sensor"][self._cam[tok]["calibrated_sensor_token"]]

    def camera(self, scene: str) -> Camera:
        """Intrinsics and camera -> ego transform of the scene's first keyframe."""
        cs = self._cs(self.samples(scene)[0])
        return Camera(Calib(np.array(cs["camera_intrinsic"], dtype=float)), CAM_HEIGHT, FRAME_DT,
                      _rot(cs["rotation"]), np.array(cs["translation"], dtype=float))

    def _pose(self, tok):
        p = self._t["ego_pose"][self._cam[tok]["ego_pose_token"]]
        return _rot(p["rotation"]), np.array(p["translation"], dtype=float)

    def speeds(self, scene: str) -> np.ndarray:
        """Ego speed per keyframe (centred difference of the ego poses)."""
        toks = self.samples(scene)
        pos = np.array([self._pose(tk)[1] for tk in toks])
        ts = np.array([self._t["sample"][tk]["timestamp"] / 1e6 for tk in toks])
        v = np.zeros(len(toks))
        for i in range(len(toks)):
            a, b = max(0, i - 1), min(len(toks) - 1, i + 1)
            dt = ts[b] - ts[a]
            v[i] = np.linalg.norm(pos[b] - pos[a]) / dt if dt > 1e-6 else 0.0
        return v

    def geometry(self, scene: str) -> np.ndarray:
        """Annotated road users of every keyframe: ego-frame footprint and the projected 2D box."""
        toks = self.samples(scene)
        rows = []
        for fi, tk in enumerate(toks):
            Rg, tg = self._pose(tk)
            cs = self._cs(tk)
            Rc, tc = _rot(cs["rotation"]), np.array(cs["translation"], dtype=float)
            K = np.array(cs["camera_intrinsic"], dtype=float)
            for a in self._ann.get(tk, []):
                cat = self._t["category"][self._t["instance"][a["instance_token"]]["category_token"]]["name"]
                if not cat.startswith(EVAL_PREFIXES) or a["num_lidar_pts"] + a["num_radar_pts"] < 1:
                    continue
                centre = np.array(a["translation"], dtype=float)
                w, l, h = (float(x) for x in a["size"])
                Rb = _rot(a["rotation"])
                corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
                foot = np.stack([np.concatenate([Rb[:2, :2] @ c, [-h / 2]]) + centre for c in corners])
                foot_ego = (Rg.T @ (foot - tg).T).T
                top = foot.copy()
                top[:, 2] += h
                allc = np.vstack([foot, top])
                cam = (Rc.T @ ((Rg.T @ (allc - tg).T).T - tc).T).T
                front = cam[:, 2] > 0.5
                if front.sum() < 4:
                    continue
                uv = (K @ cam[front].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                x1, y1 = uv.min(0)
                x2, y2 = uv.max(0)
                x1, x2 = np.clip([x1, x2], 0, IMG_W - 1)
                y1, y2 = np.clip([y1, y2], 0, IMG_H - 1)
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                rows.append((self._t["scene"][scene]["name"], fi, a["instance_token"], cat.split(".")[0], coarse_class(cat),
                             x1, y1, x2, y2, foot_ego[:, 0].min(), foot_ego[:, 0].mean(), foot_ego[:, 1].mean(),
                             foot_ego[:, 1].min(), foot_ego[:, 1].max()))
        if not rows:
            return np.zeros(0, dtype=GEOM_FIELDS)
        inst = {t: i for i, t in enumerate(sorted({r[2] for r in rows}))}
        out = np.zeros(len(rows), dtype=GEOM_FIELDS)
        for i, r in enumerate(rows):
            out[i]["seq"], out[i]["frame"], out[i]["track_id"] = r[0], r[1], inst[r[2]]
            out[i]["type"], out[i]["coarse"] = r[3], r[4]
            for j, c in enumerate(("x1", "y1", "x2", "y2")):
                out[i][c] = r[5 + j]
            for j, c in enumerate(("long_near", "long_cen", "lat_cen", "lat_min", "lat_max")):
                out[i][c] = r[9 + j]
            out[i]["ttc"] = np.inf
        fill_range_rate(out, halfwidth=1, dt=FRAME_DT)
        return out


def units(root, split: str, scenes=None):
    """nuScenes scenes of a split manifest (data/splits/nuscenes_<split>.txt), in sorted order."""
    db = NuScenes(root)
    for name in sorted(scenes):
        sc = db.by_name[name]
        toks = db.samples(sc)
        geom, speeds = db.geometry(sc), db.speeds(sc)
        yaw = yaw_rates([db._pose(t)[0] for t in toks], [db._t["sample"][t]["timestamp"] / 1e6 for t in toks])

        def frames(toks=toks, geom=geom, speeds=speeds, yaw=yaw):
            for fr in range(len(toks)):
                yield fr, str(db.image_path(toks[fr])), float(speeds[fr]), frame_geometry(geom, fr), True, float(yaw[fr])
        yield name, db.camera(sc), frames
