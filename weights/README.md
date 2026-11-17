# Weights

| File | Content | Source | md5 |
|---|---|---|---|
| yolov8s.pt | YOLOv8s, COCO | official Ultralytics release (`bash weights/download.sh`) | 0a1d5d0619d5a24848a267ec1e3a3118 |
| looming.pt | looming estimate (TTC of verified objects) | GitHub release v1.0 | 1d9112abccf10951811ff2642b4d017b |
| refiner.pt | range refiner | GitHub release v1.0 | 1c84be3045ab9e86da0509e2195e7a77 |
| adascale_nuscenes.pt | AdaScale scale net, nuScenes | GitHub release v1.0 | 718ef04e6a9f0815124c802e7f247fc6 |
| adascale_av2.pt | AdaScale scale net, AV2 and Argoverse-HD | GitHub release v1.0 | d8d74e24f33f3b8672b7d727fe1b4dd8 |
| bar_nuscenes.pt / bar_av2.pt | BAR skipping estimator (AV2 file also for Argoverse-HD) | GitHub release v1.0 | 4c67b4c14cd914daa30a89d167e2e585 / 7817f7d620b7d3a0657500f62b3157ec |
| chanakya_nuscenes.pt / chanakya_av2.pt | Chanakya-style policy (AV2 file also for Argoverse-HD) | GitHub release v1.0 | 5ec9dd03d61ada1bd9070516b774ea2b / a69ed42e1f2a4b8314fad5987ae1ba81 |

`bash weights/download.sh` fetches every file and checks its md5. All files load with `torch.load(..., weights_only=True)`.
