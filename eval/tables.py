"""Records -> CSV tables (dataset, row, statistic, value, DGSP - row, 95 % interval, mark).

  python -m eval.tables --runs runs --out tables

  main.csv      fidelity, quality and cost, ablation: px, GF, Agr., Rep., Trans. (vs pi(F), F = u960), episode P / R / F1
  recovery.csv  relevant objects missed by the draft that the row recovers (all, vehicles, pedestrians + cyclists)
  regime.csv    Agr. by decisive object (seen / missed), ego-speed tercile, range of the decisive object
  stress.csv    Agr. at raised ego speeds
  planners.csv  curved path (Agr. with its own pi(F), F1) and TTC warning (F1)
Paired unit bootstrap (2,000 draws, seed 0); mark: dagger = DGSP significantly higher, * = lower."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import bootstrap, compare, episodes, recovery, statistics

DATASETS = ("nuscenes", "av2", "argoverse_hd")
REF, F = "DGSP", "u960"
SPEED_CUTS = {"nuscenes": (3.24, 6.72), "av2": (3.60, 7.70), "argoverse_hd": (3.60, 7.70)}   # ego-speed terciles [m/s]
REFINER_GF = 1.191142                                                 # GFLOPs of the refiner per object
NAMES = {"u320": "u320 (draft)", "u960": "u960 (F)", "Counterfactual": "CF",
         "support only": "Support only (w/o field)", "field only": "Field only (w/o support)", "pivots": "Pivots instead of support",
         "single round": "Single round", "w/o guard": "W/o truncation guard", "DGSP w/o refiner": "W/o range refiner"}
GROUPS = {"all": None, "vehicles": {"vehicle"}, "pedestrians + cyclists": {"person", "cyclist"}}


def load(folder):
    """method -> per-frame DataFrame (all frames, sorted by unit / frame)."""
    out = {}
    for f in sorted(Path(folder).glob("*.pkl")):
        r = pickle.load(open(f, "rb"))
        if r:
            out[r[0]["method"]] = pd.DataFrame(r).sort_values(["unit", "frame"]).reset_index(drop=True)
    return out


def frame_table(M):
    """One frame table: annotated requirement, draft, F's action and its transitions, and a_<row> / px_<row> / ..."""
    f = M[F]
    d = f[["unit", "frame", "scored", "v", "act_gt", "act2_gt", "act_gt_c", "a_draft", "found_draft", "dec_seen", "dec_range",
           "rel_class"]].copy()
    d["F"], d["Fc"] = f["a"].to_numpy(int), f["ac"].to_numpy(int)
    prev = d.groupby("unit")["F"].shift(1)
    d["F_changes"] = (prev.notna() & (prev != d["F"])).to_numpy()
    for m, x in M.items():
        assert (x[["unit", "frame"]].to_numpy() == d[["unit", "frame"]].to_numpy()).all(), m
        for c in ("a", "a_none", "a2", "ac", "px", "gf", "found"):
            d[f"{c}|{m}"] = x[c].to_numpy() if c != "found" else list(x[c])
    if REF in M:
        r = f"{REF} w/o refiner"
        d[f"a|{r}"], d[f"px|{r}"] = d[f"a_none|{REF}"], d[f"px|{REF}"]
        d[f"gf|{r}"] = d[f"gf|{REF}"] - REFINER_GF * M[REF]["n_refiner"].to_numpy(float)
    return d[d.scored.astype(bool)].reset_index(drop=True)


def named(rows):
    return [{**r, "row": NAMES.get(r["row"], r["row"])} for r in rows]


def main_tables(runs):
    long, rec, S_all = [], [], {}
    for ds in DATASETS:
        if not (Path(runs) / ds / "base").exists():
            continue
        d = frame_table(load(Path(runs) / ds / "base"))
        units, C = bootstrap(d.unit)
        rows = [c.split("|", 1)[1] for c in d.columns if c.startswith("a|")]
        S = {m: statistics(d, units, C, f"a|{m}", f"px|{m}", "F", gf=f"gf|{m}") for m in rows}
        S_all[ds] = S
        long += [{"dataset": ds, **r} for r in named(compare(S, REF))]
        for grp, cls in GROUPS.items():
            R = {m: recovery(d, units, C, f"found|{m}", cls) for m in rows if not m.endswith("w/o refiner")}
            rec += [{"dataset": ds, "class_group": grp, **r} for r in named(compare({m: {"recovery": v} for m, v in R.items()}, REF))]
    if len(S_all) == len(DATASETS):                         # mean over the splits: mean of the per-split replicates
        common = set.intersection(*(set(S) for S in S_all.values()))
        mean = {m: {k: (np.mean([S_all[ds][m][k][0] for ds in DATASETS]), np.mean([S_all[ds][m][k][1] for ds in DATASETS], 0))
                    for k in S_all[DATASETS[0]][m]} for m in common}
        long += [{"dataset": "mean", **r} for r in named(compare(mean, REF))]
    return pd.DataFrame(long), pd.DataFrame(rec)


def agreement(x, rows, units=None, C=None):
    if units is None:
        units, C = bootstrap(x.unit)
    return {r: {"Agr.": statistics(x, units, C, f"a|{r}", f"px|{r}", "F")["Agr."]} for r in rows if f"a|{r}" in x}


def regime(runs):
    out, stress = [], []
    for ds in DATASETS:
        base = Path(runs) / ds / "base"
        if not base.exists():
            continue
        d = frame_table(load(base))
        units, C = bootstrap(d.unit)                       # resampled over all units of the split, also for a subset
        lo, hi = SPEED_CUTS[ds]
        brake, rng = (d.act_gt >= 1).to_numpy(), pd.to_numeric(d.dec_range, errors="coerce").to_numpy(float)
        cats = {"All": np.ones(len(d), bool), "Seen": brake & d.dec_seen.eq(True).to_numpy(), "Missed": brake & d.dec_seen.eq(False).to_numpy(),
                "Low": (d.v <= lo).to_numpy(), "Mid": ((d.v > lo) & (d.v <= hi)).to_numpy(), "High": (d.v > hi).to_numpy(),
                "<20": brake & (rng < 20), "20-40": brake & (rng >= 20) & (rng < 40), "40-60": brake & (rng >= 40) & (rng < 60),
                ">60": brake & (rng >= 60), "none": ~brake}
        for cat, m in cats.items():
            x = d[m].reset_index(drop=True)
            if len(x):
                out += [{"dataset": ds, "category": cat, "frames": len(x), **r} for r in compare(agreement(x, (REF, "u512", "u640"), units, C), REF)]
        for st in ("x1.5", "x2", "x2-nolimit"):
            if (Path(runs) / ds / st).exists():
                e = frame_table(load(Path(runs) / ds / st))
                stress += [{"dataset": ds, "setting": st.replace("-nolimit", ", no range limit"), "frames": len(e), **r}
                           for r in compare(agreement(e, (REF, "u512", "u640")), REF)]
    return pd.DataFrame(out), pd.DataFrame(stress)


def planners(runs):
    out = []
    for ds in DATASETS:
        if not (Path(runs) / ds / "base").exists():
            continue
        d = frame_table(load(Path(runs) / ds / "base"))
        units, C = bootstrap(d.unit)
        U = lambda v: pd.Series(np.asarray(v, float)).groupby(d.unit.to_numpy()).sum().reindex(units, fill_value=0).to_numpy()  # noqa: E731
        n = U(np.ones(len(d)))
        for part, col, gt, rows, levels in (
                ("curved path", "ac", "act_gt_c", {"Whole frame": "u512", "DGSP, braking": REF, "DGSP, own": "DGSP (curved loop)"}, (1,)),
                ("TTC warning", "a2", "act2_gt", {"u512": "u512", "u960 (F)": "u960", "DGSP, braking": REF,
                                                              "DGSP, own": "DGSP (TTC loop)"}, (1, 2))):
            for lv in levels:
                S = {}
                for lab, r in rows.items():
                    if f"{col}|{r}" not in d:
                        continue
                    e = episodes(d, f"{col}|{r}", gt, lv).reindex(units)
                    fn, ep, fp = (e[c].fillna(0).to_numpy(float) for c in ("missed", "episodes", "false"))
                    tp = ep - fn
                    S[lab] = {"F1": (2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1e-12), (C @ (2 * tp)) / np.maximum(C @ (2 * tp + fp + fn), 1e-12))}
                    if col == "ac":
                        eq = (d[f"ac|{r}"].to_numpy(int) == d["Fc"].to_numpy(int)).astype(float)
                        S[lab]["Agr."] = (eq.mean(), (C @ U(eq)) / np.maximum(C @ n, 1e-12))
                out += [{"dataset": ds, "planner": part, "level": lv, **r} for r in compare(S, "DGSP, own")]
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    L, R = main_tables(a.runs)
    L.to_csv(od / "main.csv", index=False); R.to_csv(od / "recovery.csv", index=False)
    Rg, S = regime(a.runs)
    Rg.to_csv(od / "regime.csv", index=False); S.to_csv(od / "stress.csv", index=False)
    planners(a.runs).to_csv(od / "planners.csv", index=False)
    if len(L):
        W = L[L.statistic.isin(["px", "GF", "Agr.", "F1"])].assign(
            cell=lambda z: np.where(z.statistic == "GF", z.value.round(1), (100 * z.value).round(1)).astype(str) + z.get("mark", "").fillna(""))
        print(W.pivot_table(index="row", columns=["dataset", "statistic"], values="cell", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
