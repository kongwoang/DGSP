<div align="center">

# Decision-Guided Speculative Perception (DGSP)

**Check Your Blind Spots: Decision-Guided Perception by Probing the Planner with Hypothetical Objects**

[![arXiv](https://img.shields.io/badge/arXiv-2611.23030-b31b1b.svg)](https://arxiv.org/abs/2611.23030)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg?logo=pytorch&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="assets/teaser.png" width="80%" alt="DGSP teaser">

</div>

DGSP decides where to spend high-resolution computation by asking the **planner**, not the image.
A cheap low-resolution pass gives a draft scene and action. Probing the planner with *hypothetical objects* yields a **decision-sensitivity field**, the places where an unseen object would change the action. Only this field and the **support**, the detections the action rests on, are verified at high resolution, until the action is stable.

## Method

<p align="center"><img src="assets/method.png" width="100%" alt="DGSP overview"></p>

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
  year   = {2026},
  eprint = {2611.23030},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2611.23030}
}
```

## License

[MIT](LICENSE). Detection uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0) as a dependency.
