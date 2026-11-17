"""Detection: YOLO forward with the letterbox / candidate threshold / class-aware NMS the method uses, the aligned crop
windows and the truncation guard."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision

from .config import KEEP_CONF
from .field import box_crop

# COCO ids of traffic participants and their coarse class
COCO_TRAFFIC = {0: "person", 1: "cyclist", 2: "vehicle", 3: "cyclist", 5: "vehicle", 7: "vehicle"}
COCO_KEEP = sorted(COCO_TRAFFIC)
PAD = 114


@dataclass(frozen=True)
class Mode:
    """Network input canvas (height, width) and the NMS settings."""
    name: str
    net_h: int
    net_w: int
    conf: float = 0.10    # candidate threshold before NMS
    iou: float = 0.65
    max_det: int = 100


def letterbox(img: np.ndarray, net_h: int, net_w: int):
    """Resize keeping aspect, pad to (net_h, net_w). Returns image, scale, (pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = min(net_h / h, net_w / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padx, pady = (net_w - new_w) / 2, (net_h - new_h) / 2
    top, bottom = int(round(pady - 0.1)), int(round(pady + 0.1))
    left, right = int(round(padx - 0.1)), int(round(padx + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
    return out, r, (left, top)


class Detector:
    """A YOLOv8 model (FP16 on CUDA) returning traffic-class detections in original image pixels."""

    def __init__(self, weights: str, device: str = "cuda:0", half: bool = True):
        from ultralytics import YOLO
        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"
        self._keep = torch.tensor(COCO_KEEP, device=self.device)
        self.net = YOLO(weights).model.to(self.device).eval()
        if self.half:
            self.net = self.net.half()

    @torch.no_grad()
    def detect(self, img_bgr: np.ndarray, mode: Mode) -> dict:
        lb, r, pad = letterbox(img_bgr, mode.net_h, mode.net_w)
        x = torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))).to(self.device, non_blocking=True)
        x = ((x.half() if self.half else x.float()) / 255.0).unsqueeze(0)
        out = self.net(x)
        raw = out[0] if isinstance(out, (list, tuple)) else out
        return self._post(raw, mode, r, pad, img_bgr.shape)

    def _post(self, raw, mode: Mode, r, pad, img_shape) -> dict:
        p = raw[0].float()                                        # (4 + nc, N)
        boxes_xywh, cls_scores = p[:4], p[4:]
        traffic = cls_scores[self._keep]
        best, best_i = traffic.max(0)
        keep = best > mode.conf
        if int(keep.sum()) == 0:
            return {"xyxy": np.zeros((0, 4), np.float32), "conf": np.zeros(0, np.float32), "coarse": np.array([], dtype="U8")}
        boxes_xywh = boxes_xywh[:, keep].T
        best, best_i = best[keep], best_i[keep]
        xy, wh = boxes_xywh[:, :2], boxes_xywh[:, 2:]
        xyxy = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)
        sel = torchvision.ops.batched_nms(xyxy, best, self._keep[best_i], mode.iou)[: mode.max_det]
        xyxy = xyxy[sel]
        xyxy[:, [0, 2]] -= pad[0]
        xyxy[:, [1, 3]] -= pad[1]
        xyxy /= r
        h0, w0 = img_shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, w0 - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, h0 - 1)
        coarse = np.array([COCO_TRAFFIC[int(c)] for c in self._keep[best_i[sel]].cpu().numpy()])
        return {"xyxy": xyxy.cpu().numpy().astype(np.float32), "conf": best[sel].cpu().numpy().astype(np.float32),
                "coarse": coarse}


def keep_confident(d: dict, thr: float = KEEP_CONF) -> dict:
    k = d["conf"] >= thr
    return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}


def wholeframe_mode(dataset: str, long_side: int, img_w: int, img_h: int) -> Mode:
    """Whole-frame canvas at a long side (fixed canvases at the usual long sides; others: the aspect rounded up to /32)."""
    if dataset == "argoverse_hd":
        return Mode(f"ahd_u{long_side}", int(math.ceil(long_side * 1200 / 1920 / 32)) * 32, long_side)
    if dataset == "nuscenes":
        table = {640: (384, 640), 960: (576, 960), 1280: (736, 1280), 1536: (864, 1536), 1600: (928, 1600)}
        h, w = table.get(long_side, (int(math.ceil(long_side * 900 / 1600 / 32)) * 32, long_side))
        return Mode(f"ng{h}x{w}", h, w)
    table = {640: (640, 512), 960: (960, 768), 1280: (1280, 992), 1536: (1536, 1184), 2048: (2048, 1568)}
    h, w = table.get(long_side, (long_side, int(math.ceil(long_side * 1550 / 2048 / 32)) * 32))
    return Mode(f"ag{h}x{w}", h, w)


class Windows:
    """Crop detection on the stride-32 grid of the frame letterboxed to the window long side (u960): a crop is widened to
    the grid cells it touches and the detector runs on that window of the letterbox at the window's own size. Counts
    network pixels (window areas); identical windows within a frame are detected once. Keeps the previous frame's
    letterbox for retro look-ups."""

    def __init__(self, detector: Detector, dataset: str, long_side: int = 960, keep_conf: float = KEEP_CONF):
        self.det, self.dataset, self.long_side, self.keep_conf = detector, dataset, long_side, keep_conf
        self.prev_lb = None
        self.lb = None

    def new_frame(self, img):
        H, W = img.shape[:2]
        self.W, self.H = W, H
        md = wholeframe_mode(self.dataset, self.long_side, W, H)
        self.prev_lb = self.lb
        self.lb, self.r, (self.px0, self.py0) = letterbox(img, md.net_h, md.net_w)
        self.FH, self.FW = md.net_h, md.net_w
        self.px, self._cache = 0.0, {}

    def window(self, cr):
        """Letterbox window (x1, y1, x2, y2) of a crop in native pixels."""
        x1, y1 = int(cr[0]), int(cr[1])
        xe, ye = min(max(int(math.ceil(cr[2])), x1 + 8), self.W), min(max(int(math.ceil(cr[3])), y1 + 8), self.H)
        wx1 = min(max(0, int(math.floor((x1 * self.r + self.px0) / 32)) * 32), self.FW - 32)
        wy1 = min(max(0, int(math.floor((y1 * self.r + self.py0) / 32)) * 32), self.FH - 32)
        wx2 = min(self.FW, max(wx1 + 32, int(math.ceil((xe * self.r + self.px0) / 32)) * 32))
        wy2 = min(self.FH, max(wy1 + 32, int(math.ceil((ye * self.r + self.py0) / 32)) * 32))
        return wx1, wy1, wx2, wy2

    def native(self, cr):
        """The window of a crop, in native pixels (what the detector actually sees)."""
        wx1, wy1, wx2, wy2 = self.window(cr)
        return np.array([max((wx1 - self.px0) / self.r, 0.0), max((wy1 - self.py0) / self.r, 0.0),
                         min((wx2 - self.px0) / self.r, float(self.W)), min((wy2 - self.py0) / self.r, float(self.H))])

    def _detect(self, lb, w):
        wx1, wy1, wx2, wy2 = w
        h_, w_ = wy2 - wy1, wx2 - wx1
        d = self.det.detect(np.ascontiguousarray(lb[wy1:wy2, wx1:wx2]), Mode(f"a{h_}x{w_}", h_, w_))
        xy = (np.asarray(d["xyxy"], float).reshape(-1, 4) + np.array([wx1, wy1, wx1, wy1], float)
              - np.array([self.px0, self.py0, self.px0, self.py0], float)) / self.r
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, self.W - 1)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, self.H - 1)
        return keep_confident({"xyxy": xy, "conf": np.asarray(d["conf"], float), "coarse": np.asarray(d["coarse"])}, self.keep_conf)

    def __call__(self, cr) -> dict:
        w = self.window(cr)
        self.px += (w[2] - w[0]) * (w[3] - w[1])
        if w not in self._cache:
            self._cache[w] = self._detect(self.lb, w)
        return {k: x.copy() for k, x in self._cache[w].items()}

    def previous(self, cr) -> dict:
        """Detect the same window on the previous frame's letterbox (retro); counted, never cached."""
        w = self.window(cr)
        self.px += (w[2] - w[0]) * (w[3] - w[1])
        return self._detect(self.prev_lb, w)


def truncated(xyxy, crop, W, H, tol=2.0, top_only=False):
    """Boxes touching a crop edge that is not an image edge. top_only: (top-edge truncation, any other edge)."""
    x1, y1, x2, y2 = crop
    b = np.asarray(xyxy, float).reshape(-1, 4)
    top = (b[:, 1] - y1 < tol) & (y1 > tol)
    other = (((b[:, 0] - x1 < tol) & (x1 > tol)) | ((x2 - b[:, 2] < tol + 1) & (x2 < W - tol - 1)) |
             ((y2 - b[:, 3] < tol + 1) & (y2 < H - tol - 1)))
    return (top, other) if top_only else (top | other)


def grow(box, crop, W, H):
    """A crop around a truncated box, extended by the box's own size on each truncated side, with context 2."""
    x1, y1, x2, y2 = box
    w, h, tol = x2 - x1, y2 - y1, 2.0
    ex1 = x1 - w if (x1 - crop[0] < tol and crop[0] > tol) else x1
    ex2 = x2 + w if (crop[2] - x2 < tol + 1) else x2
    ey1 = y1 - h if (y1 - crop[1] < tol and crop[1] > tol) else y1
    ey2 = y2 + h if (crop[3] - y2 < tol + 1) else y2
    return box_crop(np.array([ex1, ey1, ex2, ey2]), W, H)


def guarded(detect, rect_of, W, H, max_grow=2, cannot_matter=None):
    """Truncation guard: det(crop) -> (detections, grown crops). A detection touching a non-image edge of the window
    (rect_of(crop)) is replaced by the detection of the same object (intersection over the truncated box >= 0.5) in a
    crop grown around it, at most max_grow times, or dropped. cannot_matter(box, rect) -> True keeps a truncated box as
    detected (filtered guard); such boxes are reported in skipped."""
    skipped = set()

    def det(crop):
        d = {k: np.asarray(x) for k, x in detect(crop).items()}
        if not len(d["conf"]):
            return d, []
        rect = rect_of(crop)
        top, other = truncated(d["xyxy"], rect, W, H, top_only=True)
        tr = other | top
        if cannot_matter is not None:
            for i in np.flatnonzero(tr):
                if cannot_matter(d["xyxy"][i], rect):
                    tr[i] = False
                    skipped.add(tuple(np.asarray(d["xyxy"][i], float).tolist()))
        out = {k: x[~tr] for k, x in d.items()}
        extra, live = [], [(d["xyxy"][i], crop) for i in np.flatnonzero(tr)]
        for step in range(max_grow):
            if not live:
                break
            gs = [grow(b, rect_of(c), W, H) for b, c in live]
            extra += gs
            nxt = []
            for (b, _), g in zip(live, gs):
                dg = {k: np.asarray(x) for k, x in detect(g).items()}
                if not len(dg["conf"]):
                    continue
                ix = np.clip(np.minimum(dg["xyxy"][:, 2], b[2]) - np.maximum(dg["xyxy"][:, 0], b[0]), 0, None)
                iy = np.clip(np.minimum(dg["xyxy"][:, 3], b[3]) - np.maximum(dg["xyxy"][:, 1], b[1]), 0, None)
                io = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
                j = int(np.argmax(io))
                if io[j] < 0.5:
                    continue
                if not truncated(dg["xyxy"][j:j + 1], rect_of(g), W, H)[0] or step == max_grow - 1:
                    out = {k: np.concatenate([out[k], dg[k][j:j + 1]]) for k in out}
                else:
                    nxt.append((dg["xyxy"][j], g))
            live = nxt
        return out, extra
    det.skipped = skipped
    return det
