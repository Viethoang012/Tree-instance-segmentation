# Tree-instance-segmentation

Tree crown instance segmentation on aerial RGB tiles.

Pipeline: EDTER edge / extent maps (Stage I + Stage II, flip + multi-scale TTA) → watershed over-segmentation → GCN region merging → region scorer → mask NMS. Output is a COCO-format detections file.

## Layout

```
Tree-instance-segmentation/
├── predict.py              # images -> detections json
├── evaluate.py             # detections json + COCO annotations -> AP
├── visualize.py            # detections (+ ground truth, edge maps) drawn on the tiles
├── requirements.txt
├── treeseg/                # model and pipeline code
└── checkpoints/
    ├── edter_stage1.pth    # EDTER Stage I (global, ViT-L/16)
    ├── edter_stage2.pth    # EDTER Stage II (local, ViT-B/8)
    ├── gcn_r18.pt          # region-merging GCN
    ├── scorer.pkl          # region scorer
    └── torch_hub/hub/checkpoints/alexnet-owt-7be5be79.pth   # AlexNet weights used by LPIPS
```

## Environment

- Python 3.12, CUDA GPU (bf16 support)
- Install dependencies:

```bash
conda create -n treeseg python=3.12 -y
conda activate treeseg
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Data

Images are read from `<data_root>/<split>/images/*.jpg` (1024×1024 tiles). If `<data_root>/<split>/annotations/instances.json` exists, its image ids are used in the output; otherwise ids follow the sorted file order.

Example (ForestSeg-T1):

```
ForestSeg-T1/
└── Val/
    ├── images/*.jpg
    └── annotations/instances.json
```
## Checkpoint (Updating)
### Backup at: https://drive.google.com/drive/folders/1MD_eT9HgyZM6tqfdyJl_pTbrQGe7Fc3f?usp=sharing
## Inference
### Please run it first before evaluation
```bash
python predict.py \
    --data_root /path/to/ForestSeg-T1 \
    --split Val \
    --out outputs/val_dets.json
```

Options:

| argument | description |
|---|---|
| `--data_root` | folder that contains `<split>/images` |
| `--split` | sub-folder name (default `Val`) |
| `--list` | optional text file with image names (without extension), one per line |
| `--ann` | COCO json used for image ids (default `<data_root>/<split>/annotations/instances.json`) |
| `--out` | output detections json |
| `--work` | cache folder for EDTER maps (default `work/<split>`); maps already present are reused |
| `--ckpt` | checkpoint folder (default `checkpoints`) |
| `--procs` | CPU worker processes (default 16) |

Output: a list of COCO detections `{"image_id", "category_id", "segmentation" (RLE), "score", "file_name"}`.

## Evaluation

```bash
python evaluate.py \
    --ann /path/to/ForestSeg-T1/Val/annotations/instances.json \
    --dets outputs/val_dets.json
```

Prints mask AP[.25:.95] (mean over IoU thresholds 0.25:0.05:0.95), AP50 and AP70 (COCO evaluation, maxDets 100).

## Visualization

```bash
python visualize.py \
    --dets outputs/val_dets.json \
    --data_root /path/to/ForestSeg-T1 \
    --split Val \
    --ann /path/to/ForestSeg-T1/Val/annotations/instances.json \
    --maps work/Val/maps \
    --out_dir outputs/vis \
    --max_images 20
```

Each output image shows the predicted crowns, and, when given, the ground-truth crowns (`--ann`) and the EDTER edge map (`--maps`) side by side.

| argument | description |
|---|---|
| `--dets` | detections json from `predict.py` |
| `--data_root`, `--split` | image location, as in `predict.py` |
| `--out_dir` | output folder (one `.jpg` per tile) |
| `--ann` | optional COCO json, adds a ground-truth panel |
| `--maps` | optional EDTER maps folder (`<work>/maps` written by `predict.py`), adds an edge-map panel |
| `--score_thr` | draw only detections with score ≥ this value (default 0) |
| `--images` | optional list of image file names to draw |
| `--max_images` | maximum number of tiles to draw (default 20) |
