"""Argoverse 2 sensor dataset, ring_front_center: annotation (LiDAR) timestamps at 5 Hz (every 2nd), each paired with
the nearest camera image; ground truth motion-compensated from the LiDAR to the camera timestamp."""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from dgsp.lift import Calib, Camera

from .common import GEOM_FIELDS, fill_range_rate, frame_geometry, yaw_rates

CAM = "ring_front_center"
EGO_MASK_Y0 = 1830.0          # rows >= this show the ego vehicle's hood
CAM_HEIGHT = 1.7506           # ring_front_center height above the road [m] (from the calibration and the road plane)
COARSE = {"REGULAR_VEHICLE": "vehicle", "LARGE_VEHICLE": "vehicle", "BUS": "vehicle", "BOX_TRUCK": "vehicle",
          "TRUCK": "vehicle", "TRUCK_CAB": "vehicle", "VEHICULAR_TRAILER": "vehicle", "SCHOOL_BUS": "vehicle",
          "ARTICULATED_BUS": "vehicle", "MOTORCYCLE": "cyclist", "BICYCLE": "cyclist", "BICYCLIST": "cyclist",
          "MOTORCYCLIST": "cyclist", "WHEELED_RIDER": "cyclist", "PEDESTRIAN": "person"}


def _R(qw, qx, qy, qz):
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


class AV2Log:
    def __init__(self, root, log: str, stride: int = 2):
        self.root, self.log = Path(root), log
        d = self.root / log
        self.ann = pd.read_feather(d / "annotations.feather")
        ext = pd.read_feather(d / "calibration" / "egovehicle_SE3_sensor.feather").set_index("sensor_name").loc[CAM]
        intr = pd.read_feather(d / "calibration" / "intrinsics.feather").set_index("sensor_name").loc[CAM]
        self.Rc, self.tc = _R(ext.qw, ext.qx, ext.qy, ext.qz), np.array([ext.tx_m, ext.ty_m, ext.tz_m], float)
        self.K = np.array([[intr.fx_px, 0, intr.cx_px], [0, intr.fy_px, intr.cy_px], [0, 0, 1]], float)
        self.W, self.H = int(intr.width_px), int(intr.height_px)
        pose = pd.read_feather(d / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
        self.pose_ts = pose.timestamp_ns.to_numpy(np.int64)
        self.pose = pose
        self.img_dir = d / "sensors" / "cameras" / CAM
        self.img_ts = np.array(sorted(int(p.stem) for p in self.img_dir.glob("*.jpg")), np.int64)
        self.frame_ts = np.sort(self.ann.timestamp_ns.unique())[::stride]
        self.dt = float(np.median(np.diff(self.frame_ts))) * 1e-9 if len(self.frame_ts) > 1 else 0.2

    def _pose(self, t):
        i = int(np.clip(np.searchsorted(self.pose_ts, t), 1, len(self.pose_ts) - 1))
        j = i if abs(self.pose_ts[i] - t) < abs(self.pose_ts[i - 1] - t) else i - 1
        r = self.pose.iloc[j]
        return _R(r.qw, r.qx, r.qy, r.qz), np.array([r.tx_m, r.ty_m, r.tz_m], float)

    def image_ts(self, fr: int) -> int:
        t = self.frame_ts[fr]
        return int(self.img_ts[np.argmin(np.abs(self.img_ts - t))])

    def image_path(self, fr: int) -> Path:
        return self.img_dir / f"{self.image_ts(fr)}.jpg"

    def speeds(self) -> np.ndarray:
        pos = np.array([self._pose(t)[1] for t in self.frame_ts])
        if len(pos) < 2:
            return np.zeros(len(pos))
        v = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) / (np.diff(self.frame_ts) * 1e-9)
        return np.r_[v[0], v]

    def camera(self, ego_mask: bool = True) -> Camera:
        """The front camera (height CAM_HEIGHT above the road); the hood band masked."""
        return Camera(Calib(self.K), CAM_HEIGHT, self.dt, self.Rc, self.tc, EGO_MASK_Y0 if ego_mask else None)

    @functools.lru_cache(maxsize=1)
    def geometry(self) -> np.ndarray:
        rows, tracks = [], {}
        for fr, t in enumerate(self.frame_ts):
            a = self.ann[(self.ann.timestamp_ns == t) & self.ann.category.isin(list(COARSE)) & (self.ann.num_interior_pts >= 1)]
            if not len(a):
                continue
            Rl, tl = self._pose(t)
            Rm, tm = self._pose(self.image_ts(fr))
            for r in a.itertuples():
                Rb = _R(r.qw, r.qx, r.qy, r.qz)
                c = np.array([r.tx_m, r.ty_m, r.tz_m])
                l, w, h = r.length_m, r.width_m, r.height_m
                corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
                foot = np.stack([Rb @ np.array([x, y, -h / 2]) + c for x, y in corners])
                top = foot + Rb @ np.array([0, 0, h])
                allc = np.vstack([foot, top])
                comp = lambda p: ((Rl @ p.T).T + tl - tm) @ Rm           # noqa: E731  ego(lidar) -> ego(camera)
                foot_e, allc_e = comp(foot), comp(allc)
                cam = (allc_e - self.tc) @ self.Rc
                front = cam[:, 2] > 0.5
                if front.sum() < 4:
                    continue
                uv = (self.K @ cam[front].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                x1, y1 = uv.min(0)
                x2, y2 = uv.max(0)
                x1, x2 = np.clip([x1, x2], 0, self.W - 1)
                y1, y2 = np.clip([y1, y2], 0, self.H - 1)
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                tid = tracks.setdefault(r.track_uuid, len(tracks))
                rows.append((self.log, fr, tid, r.category.lower(), COARSE[r.category], x1, y1, x2, y2,
                             foot_e[:, 0].min(), foot_e[:, 0].mean(), foot_e[:, 1].mean(), foot_e[:, 1].min(), foot_e[:, 1].max()))
        if not rows:
            return np.zeros(0, dtype=GEOM_FIELDS)
        out = np.zeros(len(rows), dtype=GEOM_FIELDS)
        for i, r in enumerate(rows):
            out[i]["seq"], out[i]["frame"], out[i]["track_id"] = r[0][:16], r[1], r[2]
            out[i]["type"], out[i]["coarse"] = r[3], r[4]
            for j, col in enumerate(("x1", "y1", "x2", "y2")):
                out[i][col] = r[5 + j]
            for j, col in enumerate(("long_near", "long_cen", "lat_cen", "lat_min", "lat_max")):
                out[i][col] = r[9 + j]
            out[i]["ttc"] = np.inf
        fill_range_rate(out, halfwidth=1, dt=self.dt)
        return out


def units(root, logs, ego_mask: bool = True):
    """AV2 logs of a split manifest under root (sensor/val)."""
    for log in sorted(logs):
        L = AV2Log(root, log)

        def frames(L=L):
            geom, speeds = L.geometry(), L.speeds()
            yaw = yaw_rates([L._pose(t)[0] for t in L.frame_ts], L.frame_ts * 1e-9)
            for fr in range(len(L.frame_ts)):
                yield fr, str(L.image_path(fr)), float(speeds[fr]), frame_geometry(geom, fr), True, float(yaw[fr])
        yield log, L.camera(ego_mask), frames
