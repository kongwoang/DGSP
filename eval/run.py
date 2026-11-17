"""Run methods on a test split; one record per frame and method: <out>/<dataset>/<setting>/<method>.pkl.

  python -m eval.run --dataset nuscenes --methods DGSP,u960 --data /path/to/data [--units 3] [--setting x2]

A record holds the frame's annotated requirement (braking, TTC warning, curved path) and, for the method's final scene:
the braking action with / without the refiner (a / a_none), the TTC-warning action (a2) and the curved-path action (ac),
px (detector input pixels / native pixels), GF (GFLOPs, lift included), crops, rounds, and which relevant annotated
objects the scene contains.
Settings: ego speed and annotated closing speeds scaled, optionally without the 80 m range limit."""
from __future__ import annotations

import argparse
import pickle
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from baselines import REGISTRY as BASELINES
from baselines.wholeframe import WholeFrame
from data.registry import open_units
from dgsp import planner as P
from dgsp.config import DGSP_TTC, METHODS as CONFIGS, R_GRID
from dgsp.crops import Detector
from dgsp.lift import Looming, Refiner
from dgsp.loop import Pipeline
from eval.metrics import matched

ROOT = Path(__file__).resolve().parents[1]
WHOLE = ["u320", "u320+", "u448", "u512", "u640", "u768", "u960", "u1280", "native"]
METHODS = list(CONFIGS) + ["DGSP (TTC loop)", "DGSP (curved loop)"] + WHOLE + list(BASELINES)
DET_FLOPS, LOOMING_GF, REFINER_GF = 70341.625e-9, 3.573411, 1.191142     # GFLOPs (2 per MAC) per detector input pixel / per object
RX = tuple(sorted(R_GRID + (85.0, 100.0, 115.0, 135.0, 160.0)))
SETTINGS = {"base": (1.0, 80.0, R_GRID), "x1.5": (1.5, 80.0, R_GRID), "x2": (2.0, 80.0, R_GRID),
            "x2-nolimit": (2.0, 1e6, RX)}            # (speed factor, range limit [m], field ranges)


def build(name, ds, det, looming, p, rg):
    cfg = {k: replace(c, r_grid=rg) for k, c in CONFIGS.items()}
    if name in cfg:
        return Pipeline(cfg[name], ds, det, looming, params=p)
    if name == "DGSP (TTC loop)":
        return Pipeline(replace(DGSP_TTC, r_grid=rg), ds, det, looming, planner="ttc", params=p)
    if name == "DGSP (curved loop)":
        return Pipeline(cfg["DGSP"], ds, det, looming, planner="curved", params=p)
    if name in WHOLE:
        return None if name == "u320" else WholeFrame(ds, det, looming, name if name in ("u320+", "native") else int(name[1:]))
    return BASELINES[name](ds, det, looming, p)


def cached_drafts(path):
    """(unit, frame) -> stored nuScenes draft."""
    z = np.load(path)
    out = {}
    for u, fr, s0, n in zip(z["unit"], z["frame"], z["start"], z["count"]):
        sl = slice(int(s0), int(s0) + int(n))
        out[(str(u), int(fr))] = {"xyxy": z["xyxy"][sl].astype(float), "conf": z["conf"][sl].astype(float), "coarse": z["coarse"][sl],
                                  "mono": tuple(z["mono"][sl, i].astype(float) for i in range(4))}
    return out


def requirement(g, v_ego, s, p):
    """Annotated requirement at speed factor s: ego speed and every annotated closing speed raised by (s - 1) v_ego."""
    if not len(g):
        z = np.zeros(0)
        return (z, z, z, z), 0.0, z
    rng, lo, hi, tt = (np.asarray(g[c], float) for c in ("long_near", "lat_min", "lat_max", "ttc"))
    if s != 1.0:
        vcl = np.where(np.isfinite(tt) & (tt > 0), rng / np.where(tt > 0, tt, 1.0), 0.0) + (s - 1.0) * v_ego
        tt = np.where(vcl > 1e-6, rng / np.maximum(vcl, 1e-6), np.inf)
    gt = (rng, lo, hi, tt)
    a_gt, a_obj = P.required_decel(*gt, v_ego * s, p)
    return gt, a_gt, a_obj


def actions(rec, scene, refiner, img, cam, v, k, p, braking_regions=False):
    """Every planner on a final scene; the refiner acts on the planner's support (the braking support for TTC, and for the
    curved planner on the scene of a braking loop)."""
    brk, cur, ttc = P.Braking(p), P.CurvedBraking(k, p), P.TTCWarning(p)
    scene = scene if "from_full" in scene else dict(scene, from_full=np.zeros(len(scene["conf"]), bool))
    refiner.count = 0
    rb = refiner.refine(scene, brk.support(scene["mono"], v), img, cam)
    rec["n_refiner"] = refiner.count
    rc = rb if braking_regions else refiner.refine(scene, cur.support(scene["mono"], v), img, cam)
    rec.update({"a": brk.decide(rb, v), "a_none": brk.decide(scene["mono"], v), "a2": ttc.decide(rb), "ac": cur.decide(rc, v)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nuscenes", "av2", "argoverse_hd"])
    ap.add_argument("--methods", default="DGSP,u960")
    ap.add_argument("--data", required=True, help="data root (see data/README.md)")
    ap.add_argument("--setting", default="base", choices=list(SETTINGS))
    ap.add_argument("--units", type=int, default=0, help="first N units only (0 = all)")
    ap.add_argument("--only", default="", help="comma-separated unit names")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--online-draft", action="store_true", help="nuScenes: run YOLO for the draft instead of the stored draft")
    a = ap.parse_args()
    names = METHODS if a.methods == "all" else [m for m in a.methods.split(",") if m]
    assert all(m in METHODS for m in names), f"unknown method; choose from {METHODS}"
    s, cutoff, rg = SETTINGS[a.setting]
    p = replace(P.P0, max_range=cutoff)
    W8 = ROOT / "weights"
    det = Detector(str(W8 / "yolov8s.pt"), a.device)
    looming, refiner = Looming(str(W8 / "looming.pt"), a.device), Refiner(str(W8 / "refiner.pt"), a.device)
    pipes = {m: build(m, a.dataset, det, looming, p, rg) for m in names}
    drafter = Pipeline(CONFIGS["DGSP"], a.dataset, det, looming)     # the u320 draft shared by every method
    stored = cached_drafts(ROOT / "data" / "nuscenes_draft.npz") if a.dataset == "nuscenes" and not a.online_draft else None
    rows = {m: [] for m in names}
    for unit, cam, frames in open_units(a.dataset, "test", a.data, a.units, [u for u in a.only.split(",") if u] or None):
        for q in [drafter] + list(pipes.values()):
            if q is not None:
                q.new_unit(cam)
        for fr, path, v_ego, g, scored, yaw in frames():
            img = cv2.imread(path)
            H, W = img.shape[:2]
            v, k = v_ego * s, P.curvature(yaw, v_ego)
            gt, a_gt, a_obj = requirement(g, v_ego, s, p)
            rel = np.flatnonzero(a_obj >= p.thr_decel)
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
            draft = stored[(str(unit), int(fr))] if stored is not None else drafter.draft(img)
            dec = int(np.argmax(a_obj)) if a_gt > 0 else -1
            base = {"unit": unit, "frame": fr, "scored": bool(scored), "v": v, "k": k, "a_gt": a_gt, "act_gt": P.action(a_gt, p),
                    "act2_gt": P.TTCWarning(p).decide(gt), "act_gt_c": P.action(P.required_decel(*P.CurvedBraking(k, p).along(gt), v, p)[0], p),
                    "a_draft": P.Braking(p).decide(draft["mono"], v), "found_draft": matched(gtb[rel], gtc[rel], draft["xyxy"], draft["coarse"]) >= 0,
                    "dec_seen": bool(matched(gtb[dec:dec + 1], gtc[dec:dec + 1], draft["xyxy"], draft["coarse"])[0] >= 0) if dec >= 0 else None,
                    "dec_range": float(gt[0][dec]) if dec >= 0 else None, "rel_class": [str(c) for c in gtc[rel]]}
            for m, q in pipes.items():
                rec = dict(base, method=m)
                looming.count = 0
                if q is None:                                         # u320 = the draft
                    o = {"scene": dict(draft, from_full=np.zeros(len(draft["conf"]), bool)),
                         "px": drafter.draft_mode.net_h * drafter.draft_mode.net_w / (W * H)}
                elif type(q) is Pipeline:                             # baselines built on Pipeline take no curvature
                    o = q.step(img, v, k, draft=draft)
                    rec.update({"crops": len(o["crops"]), "guard": len(o["grown"]), "rounds": o["rounds"], "converged": o["converged"]})
                else:
                    o = q.step(img, v, draft=draft)
                rec["n_looming"] = looming.count
                actions(rec, o["scene"], refiner, img, cam, v, k, p, isinstance(q, Pipeline) and q.kind == "braking")
                if q is None:
                    rec["a"] = rec["a_none"] = base["a_draft"]       # the draft is scored without refiner
                    rec["a2"], rec["ac"] = P.TTCWarning(p).decide(draft["mono"]), P.CurvedBraking(k, p).decide(draft["mono"], v)
                    rec["n_refiner"] = 0
                rec["px"] = o["px"]
                rec["gf"] = DET_FLOPS * o["px"] * W * H + LOOMING_GF * rec["n_looming"] + REFINER_GF * rec["n_refiner"]
                rec["found"] = matched(gtb[rel], gtc[rel], o["scene"]["xyxy"], o["scene"]["coarse"]) >= 0
                rows[m].append(rec)
    out = Path(a.out) / a.dataset / a.setting
    out.mkdir(parents=True, exist_ok=True)
    for m, r in rows.items():
        f = out / f"{m.replace(' ', '_').replace('/', '-')}.pkl"
        with open(f, "wb") as fh:
            pickle.dump(r, fh)
        print(f"{m}: {len(r)} frames -> {f}")


if __name__ == "__main__":
    main()
