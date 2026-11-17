"""AnA regions (own rule): context-2 crops around the draft detections (conf >= 0.25) in the ego corridor within
the range limit with a finite TTC, most urgent first (TTC, then range); iterated 3 times on the merged scene. Same windows, guard, merge and lift as DGSP;
no field, no support crops. Also the shared region-rule runner used by counterfactual.py."""
from __future__ import annotations

import numpy as np

from dgsp import planner as P
from dgsp.crops import guarded
from dgsp.field import box_crop, covered
from dgsp.config import DGSP
from dgsp.loop import Pipeline, _nms, merge


class RegionRule(Pipeline):
    iters = 1

    def __init__(self, dataset, detector, looming, params=P.P0):
        super().__init__(DGSP, dataset, detector, looming, params=params)

    def targets(self, scene, draft, r, v):
        raise NotImplementedError

    def step(self, img, v, draft=None):
        cam, (H, W) = self.cam, img.shape[:2]
        self.win.new_frame(img)
        draft = draft if draft is not None else self.draft(img)
        scene, crops = dict(draft, from_full=np.zeros(len(draft["conf"]), bool)), []
        lift = self.looming.lifter(self.hist, img, self.imgs, cam)
        for r in range(self.iters):
            xyxy, idx = self.targets(scene, draft, r, v)
            new = []
            for k in idx:
                c = box_crop(np.asarray(xyxy[k], float), W, H)
                if not covered(c, crops + new):
                    new.append(c)
            if not new:
                break
            crops = crops + new
            self.win.px = 0.0                                # every pass re-detects all crops (cached): px of the final pass
            detect = guarded(self.win, self.win.native, W, H, 2, lambda b, rc: self.cannot_matter(b, rc, W, H))
            full, grown = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}, []
            for c in crops:
                d, ex = detect(c)
                grown += ex
                full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
            scene = merge(draft, _nms(full), crops, grown, lift)
        ff = np.asarray(scene["from_full"], bool)
        self.hist["s2"], self.hist["s1"] = self.hist["s1"], np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff]
        self.imgs = (self.imgs + [img])[-2:]
        return {"scene": scene, "px": (self.draft_mode.net_h * self.draft_mode.net_w + self.win.px) / (W * H), "crops": crops}


class AnA(RegionRule):
    iters, t = 3, 0.25

    def targets(self, scene, draft, r, v):
        s = draft if r == 0 else scene
        z, lo, hi, ttc = (np.asarray(a, float) for a in s["mono"])
        p = self.p
        ok = (np.asarray(s["conf"], float) >= self.t) & P.in_corridor(lo, hi, p.corridor_half_w) & (z <= p.max_range) & (ttc < 1e18)
        idx = np.flatnonzero(ok)
        return s["xyxy"], idx[np.lexsort((z[idx], ttc[idx]))]
