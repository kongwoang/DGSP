"""Lift: verified boxes -> ego-frame geometry (z, lat_min, lat_max, ttc). Monocular range from the ground-plane and
class-height cues; TTC from the learned looming estimate on a same-fidelity 3-observation chain; range refiner on the
crop-verified objects of the action's support."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# Typical physical heights [m] used as the class-conditional size prior.
HEIGHT_PRIOR = {"vehicle": 1.55, "person": 1.72, "cyclist": 1.72}


@dataclass(frozen=True)
class Calib:
    """Pinhole intrinsics; only fx, fy, cx, cy are used."""
    K: np.ndarray

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])


@dataclass
class Camera:
    """One unit's camera: intrinsics, camera height used by the lift, frame interval and camera -> ego (R, t).

    R, t take camera coordinates (x right, y down, z forward) to ego coordinates (x forward, y left, z up)."""
    calib: Calib
    cam_h: float
    dt: float
    R: np.ndarray
    t: np.ndarray
    mask_y0: float | None = None     # ego-hood mask: boxes at least half below this image row carry no geometry


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 4) x (m, 4) -> (n, m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def associate(curr: np.ndarray, prev: np.ndarray, thr: float = 0.3) -> np.ndarray:
    """Greedy IoU association of current to previous boxes: index of the previous box per current box, or -1."""
    out = -np.ones(len(curr), int)
    if len(prev) == 0 or len(curr) == 0:
        return out
    iou = box_iou(curr, prev)
    taken = set()
    for ci in np.argsort(-iou.max(axis=1)):
        for pj in np.argsort(-iou[ci]):
            if iou[ci, pj] < thr:
                break
            if pj not in taken:
                taken.add(int(pj))
                out[ci] = pj
                break
    return out


def ttc_from_scale(curr_h: np.ndarray, prev_h: np.ndarray, dt: float, max_ttc: float = 1e3) -> np.ndarray:
    """Scale-change TTC: a box growing by a factor s over dt hits in dt / (s - 1)."""
    ttc = np.full(len(curr_h), np.inf)
    ok = np.isfinite(prev_h) & (prev_h > 1) & (curr_h > 1)
    growth = np.zeros_like(ttc)
    growth[ok] = curr_h[ok] / prev_h[ok]
    expanding = ok & (growth > 1.0 + 1e-3)
    ttc[expanding] = dt / (growth[expanding] - 1.0)
    return np.minimum(ttc, max_ttc)


def _range_height(y1, y2, coarse, calib: Calib, min_px: float = 2.0):
    h = np.maximum(y2 - y1, min_px)
    prior = np.array([HEIGHT_PRIOR.get(c, 1.6) for c in coarse], dtype=np.float64)
    return calib.fy * prior / h


def _ego_x(R, t, X, Y, Z):
    return R[0, 0] * X + R[0, 1] * Y + R[0, 2] * Z + t[0]


def _ego_y(R, t, X, Y, Z):
    return R[1, 0] * X + R[1, 1] * Y + R[1, 2] * Z + t[1]


def lift(xyxy, coarse, prev_xyxy, cam: Camera, min_px: float = 2.0) -> dict:
    """Per-detection ego-frame geometry: fused range z (and its ground / height cues), lateral extent, scale-change TTC.

    Ground cue z_g = f_y h_cam / D, with D the pixel distance of the box bottom below the ego horizon (the road normal
    seen from the camera); height cue z_h = f_y H_class / h_box; fused with weight clip((D - 5) / 25, 0, 1) on the
    ground cue, then the bottom-centre point and the two bottom corners are expressed in the ego frame. The TTC is the
    scale-change TTC against the IoU-associated box of `prev_xyxy` (None: no previous frame)."""
    xyxy = np.asarray(xyxy).astype(np.float64)
    n = len(xyxy)
    if n == 0:
        z = np.zeros(0)
        return {"z": z, "z_ground": z, "z_height": z, "lat_min": z, "lat_max": z, "ttc": z, "box_h": z}
    x1, y1, x2, y2 = xyxy.T
    prev_h = np.full(n, np.nan)
    if prev_xyxy is not None:
        prev = np.asarray(prev_xyxy).astype(np.float64)
        j = associate(xyxy, prev)
        prev_h[j >= 0] = prev[j[j >= 0], 3] - prev[j[j >= 0], 1]
    ttc = ttc_from_scale(y2 - y1, prev_h, cam.dt)
    calib = cam.calib
    R = np.asarray(cam.R, dtype=np.float64)
    t = np.asarray(cam.t, dtype=np.float64)
    n0, n1, n2 = -R[2, 0], -R[2, 1], -R[2, 2]
    uc = 0.5 * (x1 + x2)
    dv = y2 - calib.cy
    D = n0 * ((calib.fy / calib.fx) * (uc - calib.cx)) + n1 * dv + n2 * calib.fy
    zg = calib.fy * cam.cam_h / np.maximum(D, min_px)
    zh = _range_height(y1, y2, coarse, calib)
    w = np.clip((D - 5.0) / 25.0, 0.0, 1.0)
    zc = np.clip(w * zg + (1 - w) * zh, 0.5, 200.0)
    yb = (y2 - calib.cy) * zc / calib.fy
    xl = (x1 - calib.cx) * zc / calib.fx
    xr = (x2 - calib.cx) * zc / calib.fx
    xc = (uc - calib.cx) * zc / calib.fx
    y_left, y_right = _ego_y(R, t, xl, yb, zc), _ego_y(R, t, xr, yb, zc)
    zgc, zhc = np.clip(zg, 0.5, 200.0), np.clip(zh, 0.5, 200.0)
    g = {"z": _ego_x(R, t, xc, yb, zc),
         "z_ground": _ego_x(R, t, (uc - calib.cx) * zgc / calib.fx, (y2 - calib.cy) * zgc / calib.fy, zgc),
         "z_height": _ego_x(R, t, (uc - calib.cx) * zhc / calib.fx, (y2 - calib.cy) * zhc / calib.fy, zhc),
         "lat_min": np.minimum(y_left, y_right), "lat_max": np.maximum(y_left, y_right), "ttc": ttc, "box_h": y2 - y1}
    if cam.mask_y0 is not None:
        # a box lying at least half inside the hood band (rows >= mask_y0) is the ego vehicle itself: no range, no TTC
        m = (y2 - np.maximum(y1, cam.mask_y0)).clip(0, None) / np.maximum(y2 - y1, 1e-6) >= 0.5
        if m.any():
            for key in ("z", "z_ground", "z_height", "ttc"):
                arr = np.asarray(g[key], float).copy()
                arr[m] = np.inf
                g[key] = arr
    return g


def mono(dets: dict, prev: dict | None, cam: "Camera"):
    """The (z, lat_min, lat_max, ttc) tuple a planner reads, from detections and the previous frame's detections."""
    g = lift(dets["xyxy"], dets["coarse"], None if prev is None else prev["xyxy"], cam)
    return tuple(np.asarray(g[k], float) for k in ("z", "lat_min", "lat_max", "ttc"))


SIZE, CONTEXT = 64, 1.5


def crop64(img, box):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * CONTEXT, (y2 - y1) * CONTEXT
    side = max(w, h, 8.0)
    X1, Y1, X2, Y2 = int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)
    H, W = img.shape[:2]
    pad = cv2.copyMakeBorder(img, max(-Y1, 0), max(Y2 - H, 0), max(-X1, 0), max(X2 - W, 0), cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    oy, ox = max(-Y1, 0), max(-X1, 0)
    patch = pad[Y1 + oy:Y2 + oy, X1 + ox:X2 + ox]
    return cv2.resize(patch, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


CLASS_ID = {"vehicle": 0, "person": 1, "cyclist": 2}


class LoomingNet(nn.Module):
    def __init__(self, n_feat: int, imagenet_init: bool = False):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_init else None)
        m.fc = nn.Identity()
        self.cnn = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = nn.Sequential(nn.Linear(3 * 512 + n_feat, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))

    def forward(self, x, f):
        """x: (n, 3, 3, 64, 64) float in [0, 1] (time t, t-1, t-2; BGR channels), f: (n, n_feat)"""
        n = x.shape[0]
        x = x.reshape(n * 3, 3, 64, 64)
        x = Fn.interpolate(x.flip(1), size=128, mode="bilinear", align_corners=False)
        e = self.cnn((x - self.mu) / self.sd).reshape(n, 3 * 512)
        return self.head(torch.cat([e, f], 1))[:, 0]


def chains(X, s1, s2):
    """Same-fidelity 3-observation chains (i, j1, j2): box i of X associated with box j1 of the previous frame's
    history s1, itself associated with box j2 of s2 (two frames back)."""
    if s1 is None or s2 is None or not len(s1) or not len(s2) or not len(X):
        return []
    j1 = associate(X, s1)
    a12 = associate(s1, s2)
    return [(i, int(j1[i]), int(a12[j1[i]])) for i in range(len(X)) if j1[i] >= 0 and a12[j1[i]] >= 0]


class Looming:
    """Looming estimate inference. `checkpoint`: weights/looming.pt (state dict, feature normalisation)."""

    def __init__(self, checkpoint: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.net = LoomingNet(int(ck["n_feat"]))
        self.net.load_state_dict(ck["state"])
        self.net = self.net.to(self.device).eval()
        self.fmu, self.fsd = ck["fmu"].numpy(), ck["fsd"].numpy()
        self.count = 0

    def lifter(self, hist: dict, img, imgs, cam):
        """lift(dets) -> (z, lat_min, lat_max, ttc) for detections of the pipeline whose history is `hist`
        (s1, s2: its boxes at t-1, t-2); imgs = [image at t-2, image at t-1]."""
        def f(full):
            X = np.asarray(full["xyxy"], float).reshape(-1, 4)
            g = lift(X, full["coarse"], None, cam)
            z, lo, hi = (np.asarray(g[k], float) for k in ("z", "lat_min", "lat_max"))
            ttc = np.full(len(X), np.inf)
            chn = chains(X, hist.get("s1"), hist.get("s2")) if len(imgs) == 2 else []
            self.count += len(chn)
            if not chn:
                return z, lo, hi, ttc
            H, W = img.shape[:2]
            dt = cam.dt
            idx = np.array([c[0] for c in chn])
            zz = lift(X[idx], np.asarray(full["coarse"])[idx], None, cam)
            cr, fe = [], []
            for n, (i, j1, j2) in enumerate(chn):
                b1, b2 = hist["s1"][j1], hist["s2"][j2]
                x1, y1, x2, y2 = X[i]
                zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                hh = (y2 - y1, b1[3] - b1[1], b2[3] - b2[1])
                cr.append(np.stack([crop64(img, X[i]), crop64(imgs[-1], b1), crop64(imgs[-2], b2)]))
                fe.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(full["conf"][i]),
                           float(CLASS_ID[str(full["coarse"][i])]), float(z[i]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))),
                           float(np.log(max(hh[0] / max(hh[1], 1e-3), 1e-3))), float(np.log(max(hh[1] / max(hh[2], 1e-3), 1e-3))),
                           dt, 1.0])
            xin = torch.from_numpy(np.stack(cr)).permute(0, 1, 4, 2, 3).float().to(self.device) / 255.0
            fin = torch.from_numpy(((np.array(fe, np.float32) - self.fmu) / self.fsd).astype(np.float32)).to(self.device)
            with torch.no_grad():
                rh = np.clip(self.net(xin, fin).float().cpu().numpy(), 0, 2)
            for n, (i, _, _) in enumerate(chn):
                r = rh[n]
                ttc[i] = min(1 / r, 1e3) if r > 1e-3 / dt else np.inf
            return z, lo, hi, ttc
        return f


class RefinerNet(nn.Module):
    def __init__(self, n_feat: int, n_out: int = 3, imagenet_init: bool = False):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_init else None)
        m.fc = nn.Identity()
        self.cnn = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = nn.Sequential(nn.Linear(512 + n_feat, 256), nn.ReLU(inplace=True), nn.Linear(256, n_out))

    def forward(self, x, f):
        x = Fn.interpolate(x.flip(1), size=128, mode="bilinear", align_corners=False)
        return self.head(torch.cat([self.cnn((x - self.mu) / self.sd), f], 1))


class Refiner:
    """Refiner inference. `checkpoint`: weights/refiner.pt."""

    def __init__(self, checkpoint: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.net = RefinerNet(len(ck["fmu"]))
        self.net.load_state_dict(ck["state"])
        self.net = self.net.to(self.device).eval()
        self.fmu, self.fsd = ck["fmu"].numpy(), ck["fsd"].numpy()
        self.count = 0

    def refine(self, scene, support, img, cam):
        """Refined (z, lat_min, lat_max, ttc) of `scene`; only crop-verified (from_full) objects of `support` change."""
        tt = [np.asarray(a, float).copy() for a in scene["mono"]]
        sup = [p_ for p_ in support if scene["from_full"][p_]]
        self.count += len(sup)
        if not sup:
            return tuple(tt)
        H, W = img.shape[:2]
        zz = lift(scene["xyxy"][sup], scene["coarse"][sup], None, cam)
        Fm = []
        for n, p_ in enumerate(sup):
            x1, y1, x2, y2 = scene["xyxy"][p_]
            zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
            Fm.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                       float(CLASS_ID[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh,
                       float(np.log(max(zg, .5) / max(zh, .5))), 1.0])
        Fm = ((np.array(Fm, np.float32) - self.fmu) / self.fsd).astype(np.float32)
        x = torch.from_numpy(np.stack([crop64(img, scene["xyxy"][p_]) for p_ in sup])).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        with torch.no_grad():
            o = self.net(x, torch.from_numpy(Fm).to(self.device)).float().cpu().numpy().reshape(len(sup), -1)
        for n, p_ in enumerate(sup):
            c = float(np.exp(o[n, 0]))
            for a in tt[:3]:
                a[p_] *= c
        return tuple(tt)
