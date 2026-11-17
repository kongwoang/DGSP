"""AdaScale: a ResNet-18 on the 320 x 180 thumbnail predicts the best scale of the ladder (320 ... 960); the row runs one
whole-frame pass one ladder step above the prediction (ladder extended to the native size, capped at 1280). At 320 it is
the draft. Weights: weights/adascale_nuscenes.pt (nuScenes), weights/adascale_av2.pt (AV2 and Argoverse-HD). px counts
the draft pass plus the chosen pass."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from baselines.wholeframe import WholeFrame
from dgsp.config import DRAFT_CANVAS, NATIVE_LONG
from dgsp.crops import Mode, keep_confident, wholeframe_mode

SCALES = (320, 480, 640, 800, 960)
STEP, TOP = 1, 1280


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Linear(512, len(SCALES))
        self.m = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                       # x: B x 3 x 180 x 320, RGB in [0, 1]
        return self.m((x - self.mu) / self.sd)


def scale_mode(dataset, s, W, H):
    """AdaScale's own canvases on its ladder (nuScenes 16:9, AV2 portrait 4:5), the whole-frame canvases above it."""
    if s in SCALES and dataset == "nuscenes":
        return Mode(f"as{s}", int(round(s * 9 / 16 / 32)) * 32, s)
    if s in SCALES and dataset == "av2":
        return Mode(f"as{s}", s, int(round(0.8 * s / 32)) * 32)
    return wholeframe_mode(dataset, s, W, H)


class AdaScale(WholeFrame):
    def __init__(self, dataset, detector, looming, params=None, weights_dir="weights"):
        super().__init__(dataset, detector, looming, "adascale")
        self.net = Net().to(detector.device).eval()
        name = "adascale_nuscenes.pt" if dataset == "nuscenes" else "adascale_av2.pt"
        self.net.load_state_dict(torch.load(Path(weights_dir) / name, map_location="cpu", weights_only=True))
        nw = NATIVE_LONG[dataset]
        self.ladder = sorted(set(list(SCALES) + list(range(1120, nw, 160)) + [nw]))

    def step(self, img, v, draft=None):
        H, W = img.shape[:2]
        with torch.no_grad():
            x = torch.from_numpy(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()).permute(2, 0, 1)[None]
            i0 = int(self.net(x.float().to(self.det.device) / 255.0).argmax(1).item())
        top = self.ladder.index(TOP) if TOP in self.ladder else len(self.ladder) - 1
        s = self.ladder[int(np.clip(i0 + STEP, 0, top))]
        h0, w0 = DRAFT_CANVAS[self.dataset]
        if s == SCALES[0]:
            d, px = draft, h0 * w0
        else:
            md = scale_mode(self.dataset, s, W, H)
            d, px = keep_confident(self.det.detect(img, md)), h0 * w0 + md.net_h * md.net_w
        d = {k: d[k] for k in ("xyxy", "conf", "coarse")}
        scene = dict(d, mono=self.looming.lifter(self.hist, img, self.imgs, self.cam)(d), from_full=np.ones(len(d["conf"]), bool))
        self.hist["s2"], self.hist["s1"] = self.hist["s1"], np.asarray(d["xyxy"], float).reshape(-1, 4)
        self.imgs = (self.imgs + [img])[-2:]
        return {"scene": scene, "px": px / (W * H), "scale": s}
