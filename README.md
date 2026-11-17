<div align="center">

# Decision-Guided Speculative Perception (DGSP)

**Check Your Blind Spots: Decision-Guided Perception by Probing the Planner with Hypothetical Objects**

[![Paper](https://img.shields.io/badge/Paper-arXiv%20(soon)-b31b1b.svg)](#citation)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg?logo=pytorch&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="assets/teaser.png" width="80%" alt="DGSP teaser">

</div>

DGSP decides where to spend high-resolution computation by asking the **planner**, not the image.
A cheap low-resolution pass gives a draft scene and action. Probing the planner with *hypothetical objects* yields a **decision-sensitivity field**, the places where an unseen object would change the action. Only this field and the **support**, the detections the action rests on, are verified at high resolution, until the action is stable.

## Method

<p align="center"><img src="assets/method.png" width="100%" alt="DGSP overview"></p>

## Results

Agreement with the action of whole-frame inference at 960 px (Agr.), braking-episode F1, and detector-input pixels (% of native).

| Method | nuScenes<br>Agr. / F1 / px | Argoverse 2<br>Agr. / F1 / px | Argoverse-HD<br>Agr. / F1 / px |
|:--|:-:|:-:|:-:|
| Whole frame, 512 px | 97.7 / 62.2 / 10.2 | 96.4 / 71.3 / 6.7 | 94.5 / 39.8 / 7.1 |
| Whole frame, 960 px | 100 / 65.0 / 38.4 | 100 / 75.9 / 23.2 | 100 / 40.9 / 25.3 |
| **DGSP** | **98.7 / 69.0 / 8.4** | **98.2 / 77.5 / 5.9** | **97.5 / 46.7 / 6.9** |

## Quick start

```bash
pip install -r requirements.txt
bash weights/download.sh                       # YOLOv8s + DGSP weights
python -m eval.run --dataset nuscenes --data /path/to/data
python -m eval.tables --runs runs --out tables
```

Datasets: see [`data/README.md`](data/README.md). `--methods all` runs every row of the paper; [`baselines/README.md`](baselines/README.md) covers the baselines.

## Citation

```bibtex
@misc{phamcong2026dgsp,
  title  = {Check Your Blind Spots: Decision-Guided Perception by Probing the Planner with Hypothetical Objects},
  author = {Hoang Pham Cong and Long D. Nguyen and Ho Viet Duc Luong and Nguyen Tran Minh Nhat and Binh P. Nguyen},
  year   = {2026}
}
```

## License

[MIT](LICENSE). Detection uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0) as a dependency.
