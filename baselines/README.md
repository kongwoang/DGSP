# Baselines

Every baseline row runs through `python -m eval.run --methods <row>` and is scored like DGSP (same lift, refiner and
planner on its final scene; px = detector input pixels / native pixels, draft included where it is used).

| Row (`--methods`) | File | Setting |
|---|---|---|
| u320, u320+, u448 ... u1280, native | wholeframe.py | one pass at that long side; u320+ = the draft's boxes lifted |
| AdaScale | adascale.py | re-implementation: scale net (weights/adascale_*.pt), one ladder step above its prediction, capped at 1280 |
| SAHI | sahi.py | 640 slices, overlap 0.1, input 640, plus the full-image pass (pip `sahi`) |
| FOVEA | fovea.py | KDE from the draft's boxes, 1280 canvas, bandwidth 64, amplitude 1 (official grid generator) |
| UFP | ufp.py | UFPMP-Det packing of the draft's boxes, box scale 2.0, mosaic fit to 512 (official module) |
| AnA | ana.py | crops around the draft detections in the ego corridor, 3 iterations |
| Counterfactual | counterfactual.py | crops around the objects whose removal / range / TTC perturbation changes the action |
| BAR | bar.py | official skipping estimator routes a frame to the 1280 whole frame when its score reaches the threshold, else the draft |
| Chanakya | chanakya.py | its selection rule implemented directly: MLP policy on the draft's state chooses the draft or a 640 / 960 / 1280 whole frame per frame |

FOVEA, UFP and BAR use the official code of those methods, Chanakya the draft-feature code of BAR:
`bash baselines/fetch_third_party.sh` (fixed commits, into `third_party/`). BAR downloads torchvision's MobileNetV2
ImageNet weights once when it builds the estimator (they are then replaced by weights/bar_*.pt).
