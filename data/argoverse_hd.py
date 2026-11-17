"""Argoverse-HD (Argoverse 1.1 tracking sequences, ring_front_center 30 Hz, 1920 x 1200). Every image of a sequence is
run (for the histories); a frame has ground truth iff a 3D amodal LiDAR sweep lies within 50 ms."""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dgsp.lift import Calib, Camera

from .common import GEOM_FIELDS, fill_range_rate, frame_geometry, yaw_rates

CAM = "ring_front_center"
COARSE = {"VEHICLE": "vehicle", "LARGE_VEHICLE": "vehicle", "BUS": "vehicle", "TRAILER": "vehicle", "SCHOOL_BUS": "vehicle",
          "EMERGENCY_VEHICLE": "vehicle", "MOPED": "cyclist", "MOTORCYCLE": "cyclist", "BICYCLE": "cyclist",
          "BICYCLIST": "cyclist", "MOTORCYCLIST": "cyclist", "PEDESTRIAN": "person"}

CAM_HEIGHT = 1.6957          # ring_front_center height above the road [m] (from the calibration and the road plane)


def _R(qw, qx, qy, qz):
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


class AV1Log:
    def __init__(self, root, log: str, image_names):
        """root: the Argoverse 1.1 tracking directory holding `log`; image_names: the sequence's images in order."""
        self.root, self.log = Path(root), log
        d = self.root / log
        cal = json.loads((d / "vehicle_calibration_info.json").read_text())
        c = next(x["value"] for x in cal["camera_data_"] if x["key"].endswith(CAM))
        q = c["vehicle_SE3_camera_"]["rotation"]["coefficients"]
        self.Rc, self.tc = _R(*q), np.asarray(c["vehicle_SE3_camera_"]["translation"], float)
        self.K = np.array([[c["focal_length_x_px_"], 0, c["focal_center_x_px_"]],
                           [0, c["focal_length_y_px_"], c["focal_center_y_px_"]], [0, 0, 1]], float)
        self.W, self.H = 1920, 1200
        self.img_dir = d / CAM
        self.names = list(image_names)
        self.frame_ts = np.array([int(n.rsplit("_", 1)[1].split(".")[0]) for n in self.names], np.int64)
        self.dt = float(np.median(np.diff(self.frame_ts))) * 1e-9 if len(self.frame_ts) > 1 else 1 / 30
        ps = sorted((d / "poses").glob("city_SE3_egovehicle_*.json"))
        self.pose_ts = np.array([int(p.stem.rsplit("_", 1)[1]) for p in ps], np.int64)
        self._pose_files = ps
        lab = d / "per_sweep_annotations_amodal"
        ls = sorted(lab.glob("tracked_object_labels_*.json")) if lab.exists() else []
        self.sweep_ts = np.array([int(p.stem.rsplit("_", 1)[1]) for p in ls], np.int64)
        self._label_files = ls

    @functools.lru_cache(maxsize=None)
    def _pose_i(self, i):
        p = json.loads(self._pose_files[i].read_text())
        return _R(*p["rotation"]), np.asarray(p["translation"], float)

    def _pose(self, t):
        i = int(np.clip(np.searchsorted(self.pose_ts, t), 1, len(self.pose_ts) - 1))
        j = i if abs(self.pose_ts[i] - t) < abs(self.pose_ts[i - 1] - t) else i - 1
        return self._pose_i(j)

    def image_path(self, fr):
        return self.img_dir / self.names[fr]

    def speeds(self):
        pos = np.array([self._pose(t)[1] for t in self.frame_ts])
        if len(pos) < 2:
            return np.zeros(len(pos))
        k = 3                                         # centred difference over 3 frames smooths the 30 Hz pose jitter
        v = np.zeros(len(pos))
        for i in range(len(pos)):
            a, b = max(0, i - k), min(len(pos) - 1, i + k)
            v[i] = np.linalg.norm(pos[b, :2] - pos[a, :2]) / max((self.frame_ts[b] - self.frame_ts[a]) * 1e-9, 1e-3)
        return v

    def camera(self) -> Camera:
        return Camera(Calib(self.K), CAM_HEIGHT, self.dt, self.Rc, self.tc)

    def gt_frames(self, tol_ns=50_000_000):
        out = {}
        if not len(self.sweep_ts):
            return out
        for fr, t in enumerate(self.frame_ts):
            j = int(np.argmin(np.abs(self.sweep_ts - t)))
            if abs(int(self.sweep_ts[j]) - int(t)) <= tol_ns:
                out[fr] = j
        return out

    @functools.lru_cache(maxsize=1)
    def geometry(self):
        rows, tracks = [], {}
        for fr, j in self.gt_frames().items():
            Rl, tl = self._pose(int(self.sweep_ts[j]))
            Rm, tm = self._pose(int(self.frame_ts[fr]))
            for o in json.loads(self._label_files[j].read_text()):
                cls = COARSE.get(o.get("label_class", ""))
                if cls is None:
                    continue
                c = np.array([o["center"]["x"], o["center"]["y"], o["center"]["z"]], float)
                r = o["rotation"]
                Rb = _R(r["w"], r["x"], r["y"], r["z"])
                l, w, h = o["length"], o["width"], o["height"]
                corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
                foot = np.stack([Rb @ np.array([x, y, -h / 2]) + c for x, y in corners])
                allc = np.vstack([foot, foot + Rb @ np.array([0, 0, h])])
                comp = lambda p: ((Rl @ p.T).T + tl - tm) @ Rm                       # noqa: E731
                foot_e, allc_e = comp(foot), comp(allc)
                cam = (allc_e - self.tc) @ self.Rc
                front = cam[:, 2] > 0.5
                if front.sum() < 4:
                    continue
                uv = (self.K @ cam[front].T).T
                uv = uv[:, :2] / uv[:, 2:3]
                x1, y1 = np.clip(uv.min(0), 0, [self.W - 1, self.H - 1])
                x2, y2 = np.clip(uv.max(0), 0, [self.W - 1, self.H - 1])
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                tid = tracks.setdefault(o.get("track_label_uuid", id(o)), len(tracks))
                rows.append((fr, tid, o["label_class"].lower(), cls, x1, y1, x2, y2, foot_e[:, 0].min(), foot_e[:, 0].mean(),
                             foot_e[:, 1].mean(), foot_e[:, 1].min(), foot_e[:, 1].max()))
        if not rows:
            return np.zeros(0, dtype=GEOM_FIELDS)
        out = np.zeros(len(rows), dtype=GEOM_FIELDS)
        for i, r in enumerate(rows):
            out[i]["seq"], out[i]["frame"], out[i]["track_id"] = self.log[:16], r[0], r[1]
            out[i]["type"], out[i]["coarse"] = r[2], r[3]
            for k, col in enumerate(("x1", "y1", "x2", "y2")):
                out[i][col] = r[4 + k]
            for k, col in enumerate(("long_near", "long_cen", "lat_cen", "lat_min", "lat_max")):
                out[i][col] = r[8 + k]
            out[i]["ttc"] = np.inf
        # labelled sweeps are ~10 Hz (every 3rd frame): range rate with dt per frame, window +-3 frames
        fill_range_rate(out, halfwidth=3, dt=1 / 30)
        return out


def sequences(annotation_json):
    """(sequence id, log, image names in order) of an Argoverse-HD annotation file (val.json)."""
    d = json.loads(Path(annotation_json).read_text())
    by = {}
    for im in d["images"]:
        by.setdefault(im["sid"], []).append(im)
    return [(sid, d["seq_dirs"][sid].split("/")[1], [i["name"] for i in sorted(ims, key=lambda x: x["fid"])])
            for sid, ims in sorted(by.items())]


def units(annotation_json, tracking_root):
    """Argoverse-HD sequences as units 'seqNN'; tracking_root: the Argoverse 1.1 tracking directory of the split."""
    for sid, log, names in sequences(annotation_json):
        L = AV1Log(tracking_root, log, names)

        def frames(L=L, names=names):
            geom, speeds = L.geometry(), L.speeds()
            gtf = L.gt_frames()
            yaw = yaw_rates([L._pose(t)[0] for t in L.frame_ts], L.frame_ts * 1e-9)
            for fr in range(len(names)):
                g = frame_geometry(geom, fr) if fr in gtf else geom[:0]
                yield fr, str(L.image_path(fr)), float(speeds[fr]), g, fr in gtf, float(yaw[fr])
        yield f"seq{sid:02d}", L.camera(), frames
