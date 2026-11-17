"""FOVEA with the official grid generator (third_party/fovea, see fetch_third_party.sh): image-adaptive KDE saliency from
the draft's boxes (KDE_I), separable 31 x 51 grid, attraction_fwhm 4, anti_crop, bandwidth 64, amplitude 1; the frame is
warped to the 1280 whole-frame canvas, detected, boxes unwarped; every detection lifted (own history) and refined.
px counts the draft pass plus the warped canvas."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from baselines.wholeframe import WholeFrame
from dgsp.config import DRAFT_CANVAS
from dgsp.crops import keep_confident, wholeframe_mode

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "fovea" / "fovea"


def grid_module():
    """Load the official grid_generator.py unchanged; its registry imports are satisfied by stubs (decorator only)."""
    if not hasattr(np, "float"):
        np.float = float                                      # the official module uses the removed alias
    class Registry:                                           # noqa: E306
        def __init__(self, name):
            self.name = name

        def register_module(self, *a, **k):
            return lambda c: c
    mods = {n: types.ModuleType(n) for n in ("mmcv", "mmcv.utils", "mmdet", "mmdet.models", "mmdet.models.builder", "fovea_official",
                                             "fovea_official.utils", "fovea_official.utils.vis", "fovea_official.models")}
    mods["mmcv.utils"].Registry = Registry
    mods["mmdet.models.builder"].build = lambda *a, **k: None
    mods["fovea_official"].__path__ = [str(ROOT)]
    mods["fovea_official.utils.vis"].vis_batched_imgs = lambda *a, **k: None
    sys.modules.update(mods)
    spec = importlib.util.spec_from_file_location("fovea_official.models.grid_generator", ROOT / "models" / "grid_generator.py")
    gg = importlib.util.module_from_spec(spec)
    gg.__package__ = "fovea_official.models"
    spec.loader.exec_module(gg)
    return gg


class FOVEA(WholeFrame):
    side, bandwidth, amplitude = 1280, 64, 1

    def __init__(self, dataset, detector, looming, params=None):
        super().__init__(dataset, detector, looming, "fovea")
        self.gg, self.gen = grid_module(), None

    def step(self, img, v, draft=None):
        H, W = img.shape[:2]
        md = wholeframe_mode(self.dataset, self.side, W, H)
        if self.gen is None:
            self.gen = self.gg.PlainKDEGrid(output_shape=(md.net_h, md.net_w), grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                                            anti_crop=True, bandwidth_scale=self.bandwidth, amplitude_scale=self.amplitude)
        device = self.det.device
        xt = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1)[None].float().to(device)
        with torch.no_grad():
            boxes = torch.tensor(np.asarray(draft["xyxy"], float).reshape(-1, 4), dtype=torch.float32, device=device)
            grid = self.gen(xt, [{"pad_shape": (H, W, 3)}], [boxes])
            warped = Fn.grid_sample(xt, grid, align_corners=True)
        wimg = warped[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
        d = keep_confident(self.det.detect(wimg, md))
        if len(d["conf"]):
            bb = self.gg.unwarp_bboxes(torch.tensor(d["xyxy"], dtype=torch.float32, device=device), grid[0], (H, W))
            d = dict(d, xyxy=bb.cpu().numpy().astype(float))
        scene = dict(d, mono=self.looming.lifter(self.hist, img, self.imgs, self.cam)(d), from_full=np.ones(len(d["conf"]), bool))
        self.hist["s2"], self.hist["s1"] = self.hist["s1"], np.asarray(d["xyxy"], float).reshape(-1, 4)
        self.imgs = (self.imgs + [img])[-2:]
        h0, w0 = DRAFT_CANVAS[self.dataset]
        return {"scene": scene, "px": (h0 * w0 + md.net_h * md.net_w) / (W * H)}
