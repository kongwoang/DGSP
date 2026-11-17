"""Chanakya-style policy: an MLP (two hidden layers of 128) on the draft's causal state chooses, per frame, the draft or one
whole frame at 640 / 960 / 1280 (+ lift, refiner; every arm runs on every frame so its history is continuous). State = the
21 statistics of the draft detections (official feature code: third_party/bgt-ada), ego speed, the draft's required
deceleration, the draft's action and its previous action. Weights weights/chanakya_nuscenes.pt / chanakya_av2.pt (AV2,
Argoverse-HD); px = draft (+ the chosen pass)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from baselines.wholeframe import WholeFrame
from dgsp import planner as P
from dgsp.config import DRAFT_CANVAS

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = (0, 640, 960, 1280)


class Chanakya:
    def __init__(self, dataset, detector, looming, params=None, weights_dir=ROOT / "weights"):
        sys.path.insert(0, str(ROOT / "third_party" / "bgt-ada"))
        from src.features import BASE_FEATURES, extract_detection_features
        self.feats, self.names = extract_detection_features, BASE_FEATURES
        ck = torch.load(Path(weights_dir) / ("chanakya_nuscenes.pt" if dataset == "nuscenes" else "chanakya_av2.pt"), map_location="cpu",
                        weights_only=True)
        h = int(ck["hidden"])
        self.net = torch.nn.Sequential(torch.nn.Linear(len(ck["mu"]), h), torch.nn.ReLU(), torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 4))
        self.net.load_state_dict(ck["state"]); self.net.eval()
        self.mu, self.sd = ck["mu"].numpy(), ck["sd"].numpy()
        self.dataset, self.arms = dataset, {s: WholeFrame(dataset, detector, looming, s) for s in ACTIONS[1:]}

    def new_unit(self, cam):
        self.prev = None
        for a in self.arms.values():
            a.new_unit(cam)

    def step(self, img, v, draft=None):
        H, W = img.shape[:2]
        b = np.asarray(draft["xyxy"], float).reshape(-1, 4)
        f = self.feats([(float(x[0]), float(x[1]), float(x[2] - x[0]), float(x[3] - x[1]), float(c), str(k))
                        for x, c, k in zip(b, draft["conf"], draft["coarse"])])
        a_req = float(P.required_decel(*draft["mono"], v)[0])
        a_d = P.action(a_req)
        x = np.array([[float(f[k]) for k in self.names] + [v, a_req, float(a_d), float(-1 if self.prev is None else self.prev)]], np.float32)
        self.prev = a_d
        with torch.no_grad():
            s = ACTIONS[int(self.net(torch.from_numpy((x - self.mu) / self.sd)).argmax(1).item())]
        lm, out, n = self.arms[ACTIONS[1]].looming, {}, {0: 0}
        n0 = lm.count
        for k, a in self.arms.items():
            c = lm.count
            out[k] = a.step(img, v)
            n[k] = lm.count - c
        lm.count = n0 + n[s]                                   # lift cost of the chosen pass only
        h0, w0 = DRAFT_CANVAS[self.dataset]
        if s:
            return {"scene": out[s]["scene"], "px": h0 * w0 / (W * H) + out[s]["px"], "size": s}
        return {"scene": dict(draft, from_full=np.zeros(len(draft["conf"]), bool)), "px": h0 * w0 / (W * H), "size": 0}
