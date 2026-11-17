"""Draw predicted tree crowns (and optionally ground truth and EDTER edge maps) on the input tiles."""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as M


def colors(n, seed=0):
    rng = np.random.RandomState(seed)
    hsv = np.stack([rng.randint(0, 180, n), rng.randint(150, 256, n), rng.randint(180, 256, n)], 1).astype(np.uint8)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0].astype(np.float32)


def overlay(img, masks, alpha=0.45, thickness=2, seed=0):
    out = img.astype(np.float32)
    cols = colors(len(masks), seed)
    for m, c in zip(masks, cols):
        out[m] = out[m] * (1 - alpha) + c * alpha
    out = out.astype(np.uint8)
    for m, c in zip(masks, cols):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, tuple(int(v) for v in c), thickness)
    return out


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (12 + 14 * len(text), 36), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dets', required=True, help='detections json from predict.py')
    ap.add_argument('--data_root', required=True, help='folder containing <split>/images')
    ap.add_argument('--split', default='Val')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ann', default=None, help='COCO json: adds a ground-truth panel and maps image ids to files')
    ap.add_argument('--maps', default=None, help='EDTER maps folder (e.g. work/Val/maps): adds an edge-map panel')
    ap.add_argument('--score_thr', type=float, default=0.0, help='only draw detections with score >= this')
    ap.add_argument('--images', nargs='*', default=None, help='image file names to draw (default: all)')
    ap.add_argument('--max_images', type=int, default=20)
    args = ap.parse_args()

    dets = json.load(open(args.dets))
    by_img = defaultdict(list)
    id2file = {}
    for d in dets:
        by_img[d['image_id']].append(d)
        if 'file_name' in d:
            id2file[d['image_id']] = d['file_name']
    gt_by_img = defaultdict(list)
    if args.ann:
        gt = json.load(open(args.ann))
        for im in gt['images']:
            id2file.setdefault(im['id'], im['file_name'])
        for a in gt['annotations']:
            gt_by_img[a['image_id']].append(a)
    file2id = {f: i for i, f in id2file.items()}
    files = args.images or sorted(file2id)
    files = [f if f.lower().endswith('.jpg') else f + '.jpg' for f in files][:args.max_images]
    os.makedirs(args.out_dir, exist_ok=True)

    for f in files:
        iid = file2id.get(f)
        img = cv2.imread(f'{args.data_root}/{args.split}/images/{f}')
        if img is None:
            print(f'skip {f}: image not found')
            continue
        H, W = img.shape[:2]
        ds = sorted([d for d in by_img.get(iid, []) if d['score'] >= args.score_thr], key=lambda d: -d['score'])
        pred = [M.decode(d['segmentation']).astype(bool) for d in ds]
        panels = [label(overlay(img, pred), f'prediction ({len(pred)})')]
        if args.ann:
            g = []
            for a in gt_by_img.get(iid, []):
                seg = a['segmentation']
                rle = M.frPyObjects(seg, H, W) if isinstance(seg, list) else seg
                if isinstance(rle, dict) and isinstance(rle.get('counts'), list):
                    rle = M.frPyObjects(rle, H, W)
                g.append(M.decode(M.merge(rle) if isinstance(rle, list) else rle).astype(bool))
            panels.append(label(overlay(img, g, seed=1), f'ground truth ({len(g)})'))
        if args.maps:
            e = cv2.imread(f'{args.maps}/{f.rsplit(".", 1)[0]}_edge.png', cv2.IMREAD_UNCHANGED)
            if e is not None:
                e = cv2.resize((e.astype(np.float32) / 65535 * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
                panels.append(label(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR), 'edge map'))
        sep = np.full((H, 8, 3), 255, np.uint8)
        canvas = panels[0]
        for p in panels[1:]:
            canvas = np.concatenate([canvas, sep, p], 1)
        out = f'{args.out_dir}/{f.rsplit(".", 1)[0]}.jpg'
        cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(out)


if __name__ == '__main__':
    main()
