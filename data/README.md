# Data

Loaders (nuscenes.py, av2.py, argoverse_hd.py), the test split lists (splits/) and download / check scripts; no data.

```bash
bash data/download.sh all /path/to/data      # see the script header for the manual steps
python -m data.prepare --data /path/to/data  # are the test units' directories and metadata in place?
```

Layout under the data root:

```
nuscenes/                           v1.0-trainval/ (metadata JSON) and samples/CAM_FRONT/
av2/sensor/val/<log>/               Argoverse 2 sensor val logs (ring_front_center, calibration, poses, annotations)
argoverse_hd/annotations/val.json   Argoverse-HD annotations
argoverse/argoverse-tracking/val/   Argoverse 1.1 tracking val logs (the Argoverse-HD sequences)
```

| Split list | Units |
|---|---|
| splits/nuscenes_test.txt | 228 nuScenes scenes |
| splits/av2_val.txt | 150 Argoverse 2 sensor val logs |
| splits/argoverse_hd_val.txt | 24 Argoverse-HD val sequences (unit name, Argoverse 1.1 log) |
| splits/av2_ego_mask.yaml | AV2 ego-hood band: image rows >= 1830 carry no geometry |

Annotated frames: nuScenes 9,181, AV2 11,818, Argoverse-HD 15,006 (the scored frames of every table).

nuscenes_draft.npz: the u320 draft of the nuScenes test frames (boxes, confidence, class, monocular lift), read by
`eval.run` unless `--online-draft` is given.
