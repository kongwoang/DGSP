"""Budget-Adaptive Routing (BAR, official estimator code: third_party/bgt-ada, see fetch_third_party.sh): the skipping
estimator (MobileNetV2-Lite, 128 px) scores the raw frame; a frame is routed when the score reaches the threshold, and then
acts on the 1280 whole frame (+ lift, refiner; the 1280 arm runs on every frame so its history is continuous), otherwise on
the draft. Weights weights/bar_nuscenes.pt (nuScenes) and weights/bar_av2.pt (AV2, Argoverse-HD); px = draft (+ 1280 pass)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from baselines.wholeframe import WholeFrame
from dgsp.config import DRAFT_CANVAS

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = {"nuscenes": -1.882492184638977, "av2": 37.062679290771484}   # score quantile routing 50 % of frames


def estimator(dataset, device, weights_dir):
    sys.path.insert(0, str(ROOT / "third_party" / "bgt-ada"))
    from src.models.cnn import MobileNetV2LiteEstimator
    est = MobileNetV2LiteEstimator(image_size=128, device=str(device))
    est._setup_model()                                         # official architecture and transforms
    name = "bar_nuscenes.pt" if dataset == "nuscenes" else "bar_av2.pt"
    est.model.load_state_dict(torch.load(Path(weights_dir) / name, map_location="cpu", weights_only=True))
    est.model.eval()
    return est


class BAR:
    side = 1280

    def __init__(self, dataset, detector, looming, params=None, weights_dir=ROOT / "weights"):
        self.dataset = dataset
        self.est = estimator(dataset, detector.device, weights_dir)
        self.arm = WholeFrame(dataset, detector, looming, self.side)
        self.thr = THRESHOLD["nuscenes" if dataset == "nuscenes" else "av2"]

    def new_unit(self, cam):
        self.arm.new_unit(cam)

    def step(self, img, v, draft=None):
        from PIL import Image
        H, W = img.shape[:2]
        with torch.no_grad():
            s = float(self.est.model(self.est.transform(Image.fromarray(np.ascontiguousarray(img[:, :, ::-1])))[None].to(self.est.device)).item())
        lm = self.arm.looming
        n0 = lm.count
        arm = self.arm.step(img, v)
        h0, w0 = DRAFT_CANVAS[self.dataset]
        if s >= self.thr:
            return {"scene": arm["scene"], "px": h0 * w0 / (W * H) + arm["px"], "routed": True}
        lm.count = n0                                          # lift cost counts only on routed frames
        return {"scene": dict(draft, from_full=np.zeros(len(draft["conf"]), bool)), "px": h0 * w0 / (W * H), "routed": False}
