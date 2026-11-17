"""UFP: Unified Foreground Packing of UFPMP-Det (official module, third_party/UFPMP-Det, see fetch_third_party.sh). The
draft's boxes (conf >= 0.10) are packed into a mosaic (box scale 2.0); one detector pass on the mosaic resized to fit
512 x 512 (rounded up to /32); a detection belongs to a chip if IoF > 0.9 and is mapped back; per-class NMS 0.6. A box
touching its chip's inner edge is re-detected in a context-2 crop (same object, IoU >= 0.3). Draft objects whose centre
lies in a chip are replaced; same lift and refiner as every row."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

from dgsp.crops import Mode, keep_confident
from dgsp.field import box_crop, centres_in
from dgsp.lift import box_iou
from dgsp.config import DGSP, NATIVE_LONG
from dgsp.loop import Pipeline
from dgsp import planner as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "UFPMP-Det" / "UFPMP-Det-Tools"))


def iof(a, b):
    l, t, r, d = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if l >= r or t >= d:
        return 0
    return (r - l) * (d - t) / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def nms(dets, thresh):
    dets = np.array(dets)
    x1, y1, x2, y2, s = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas, order, keep = (x2 - x1 + 1) * (y2 - y1 + 1), s.argsort()[::-1], []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        order = order[np.where(w * h / (areas[i] + areas[order[1:]] - w * h) <= thresh)[0] + 1]
    return keep


def mosaic_of(chips, img, w, h):
    out = np.zeros((math.ceil(h), math.ceil(w), 3))
    for x1, y1, cw, ch, nx, ny, sf in chips:
        if cw and ch:
            out[ny:ny + ch * sf, nx:nx + cw * sf] = cv2.resize(img[y1:y1 + ch, x1:x1 + cw], (cw * sf, ch * sf))
    return out.astype(np.uint8)


class UFP(Pipeline):
    box_scale, side = 2.0, 512

    def __init__(self, dataset, detector, looming, params=P.P0):
        super().__init__(DGSP, dataset, detector, looming, params=params)

    def crop_detect(self, img, c):
        """Detect a crop at the verification scale (crop size x 960 / native long side, rounded up to /32)."""
        H, W = img.shape[:2]
        s = 960 / NATIVE_LONG[self.dataset]
        x1, y1 = int(c[0]), int(c[1])
        x2, y2 = min(max(int(math.ceil(c[2])), x1 + 8), W), min(max(int(math.ceil(c[3])), y1 + 8), H)
        nh, nw = max(32, int(math.ceil((y2 - y1) * s / 32)) * 32), max(32, int(math.ceil((x2 - x1) * s / 32)) * 32)
        d = keep_confident(self.det.detect(img[y1:y2, x1:x2], Mode(f"c{nh}x{nw}", nh, nw)))
        return dict(d, xyxy=d["xyxy"] + np.array([x1, y1, x1, y1], float)), nh * nw

    def step(self, img, v, draft=None):
        from ufp import UnifiedForegroundPacking
        H, W = img.shape[:2]
        draft = draft if draft is not None else self.draft(img)
        raw = np.asarray(self.det.detect(img, self.draft_mode)["xyxy"], float).reshape(-1, 4)    # conf >= 0.10
        full, px, srcs = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}, 0.0, []
        rec, mw, mh = UnifiedForegroundPacking(raw.copy(), self.box_scale, input_shape=[W, H]) if len(raw) else ([], 0, 0)
        if rec and mw >= 1 and mh >= 1:
            chips = [[math.floor(x) for x in c] for c in rec]
            srcs = [np.array([c[0], c[1], c[0] + c[2], c[1] + c[3]], float) for c in chips]
            mos = mosaic_of(chips, img, mw, mh)
            r = self.side / max(mos.shape[:2])
            nh, nw = int(math.ceil(mos.shape[0] * r / 32) * 32), int(math.ceil(mos.shape[1] * r / 32) * 32)
            px += nh * nw
            d = keep_confident(self.det.detect(mos, Mode(f"ufp{nh}x{nw}", nh, nw)))
            per = {}
            for (x1, y1, x2, y2), s, cl in zip(d["xyxy"], d["conf"], d["coarse"]):
                for ox, oy, cw, ch, nx, ny, sf in chips:
                    if iof([x1, y1, x2, y2], [nx, ny, nx + cw * sf, ny + ch * sf]) > 0.9:
                        a, b = (x1 - nx) / sf + ox, (y1 - ny) / sf + oy
                        per.setdefault(str(cl), []).append([a, b, a + (x2 - x1) / sf, b + (y2 - y1) / sf, float(s), ox, oy, ox + cw, oy + ch])
                        break
            bx, cf, cs = [], [], []
            for cl, lst in per.items():
                arr = np.array(lst)
                for i in nms(arr[:, :5], 0.6):
                    b, src = arr[i, :4], arr[i, 5:9]
                    if any(abs(b[k] - src[k]) < 2 and abs(src[k] - (0, 0, W, H)[k]) > 2 for k in range(4)):   # truncated by its chip
                        g, gpx = self.crop_detect(img, box_crop(b, W, H))
                        px += gpx
                        if len(g["conf"]):
                            io = box_iou(b[None], g["xyxy"])[0]
                            j = int(np.argmax(io))
                            if io[j] >= 0.3:
                                bx.append(g["xyxy"][j]); cf.append(float(g["conf"][j])); cs.append(str(g["coarse"][j]))
                        continue
                    bx.append(b); cf.append(float(arr[i, 4])); cs.append(cl)
            full = {"xyxy": np.array(bx, float).reshape(-1, 4), "conf": np.array(cf, float), "coarse": np.array(cs) if cs else np.zeros(0, "<U8")}
        out = centres_in(draft["xyxy"], srcs)
        fz = self.looming.lifter(self.hist, img, self.imgs, self.cam)(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
        scene = {"xyxy": np.r_[draft["xyxy"][~out], full["xyxy"]], "conf": np.r_[draft["conf"][~out], full["conf"]],
                 "coarse": np.r_[draft["coarse"][~out], full["coarse"]], "mono": tuple(np.r_[a[~out], b] for a, b in zip(draft["mono"], fz)),
                 "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}
        self.hist["s2"], self.hist["s1"] = self.hist["s1"], np.asarray(scene["xyxy"], float).reshape(-1, 4)[scene["from_full"]]
        self.imgs = (self.imgs + [img])[-2:]
        return {"scene": scene, "px": (self.draft_mode.net_h * self.draft_mode.net_w + px) / (W * H)}
