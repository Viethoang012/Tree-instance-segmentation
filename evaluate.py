"""COCO mask AP of a detections json against a COCO annotation file.
Reports AP[.25:.95] (mean over IoU 0.25:0.05:0.95), AP50, AP70, AP75 and AP[.50:.95]."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'treeseg'))

from pycocotools.coco import COCO  # noqa: E402

import postproc as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True, help='COCO ground-truth json')
    ap.add_argument('--dets', required=True, help='detections json from predict.py')
    args = ap.parse_args()
    gt = COCO(args.ann)
    dets = json.load(open(args.dets))
    for d in dets:
        d.pop('file_name', None)
    m = P.coco_eval(gt, dets, sorted(gt.getImgIds()))
    print(f"AP[.25:.95] {m['AP_25_95'] * 100:.2f}")
    print(f"AP50        {m['AP50'] * 100:.2f}")
    print(f"AP70        {m['AP70'] * 100:.2f}")




if __name__ == '__main__':
    main()
