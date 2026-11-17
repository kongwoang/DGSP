"""SAHI (official library): 640 x 640 slices with overlap 0.1 at input size 640, plus the full-image pass, GREEDYNMM merge
(library defaults), confidence 0.25, the same YOLOv8s weights; every detection lifted (own history) and refined."""
from __future__ import annotations

import math

import numpy as np

from baselines.wholeframe import WholeFrame
from dgsp.crops import COCO_TRAFFIC


class SAHI(WholeFrame):
    side, overlap, size = 640, 0.1, 640

    def __init__(self, dataset, detector, looming, params=None, weights="weights/yolov8s.pt"):
        super().__init__(dataset, detector, looming, "sahi")
        from sahi import AutoDetectionModel
        self.model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=weights, confidence_threshold=0.25,
                                                        device=str(detector.device), image_size=self.size)

    def step(self, img, v, draft=None):
        from sahi.predict import get_sliced_prediction
        from sahi.slicing import get_slice_bboxes
        H, W = img.shape[:2]
        res = get_sliced_prediction(np.ascontiguousarray(img[:, :, ::-1]), self.model, slice_height=self.side, slice_width=self.side,
                                    overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap, perform_standard_pred=True, verbose=0)
        ob = [o for o in res.object_prediction_list if o.category.id in COCO_TRAFFIC]
        d = {"xyxy": np.array([o.bbox.to_xyxy() for o in ob], float).reshape(-1, 4), "conf": np.array([o.score.value for o in ob], float),
             "coarse": np.array([COCO_TRAFFIC[o.category.id] for o in ob]) if ob else np.zeros(0, "<U8")}
        n = len(get_slice_bboxes(H, W, self.side, self.side, overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap))
        z = self.size
        px = n * z * z + z * int(math.ceil(z * min(W, H) / max(W, H) / 32)) * 32
        scene = dict(d, mono=self.looming.lifter(self.hist, img, self.imgs, self.cam)(d), from_full=np.ones(len(d["conf"]), bool))
        self.hist["s2"], self.hist["s1"] = self.hist["s1"], d["xyxy"]
        self.imgs = (self.imgs + [img])[-2:]
        return {"scene": scene, "px": px / (W * H)}
