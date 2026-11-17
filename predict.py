"""Tree crown instance segmentation: images -> COCO-format detections (json)."""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool, get_context

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('TORCH_HOME', os.path.join(ROOT, 'checkpoints', 'torch_hub'))
os.environ.setdefault('INFER_BATCH', '1')
sys.path.insert(0, os.path.join(ROOT, 'treeseg'))

import cv2  # noqa: E402
import lpips  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cnn_gcn as CG  # noqa: E402
import postproc as P  # noqa: E402
from infer import build, multiscale_infer  # noqa: E402
from pipeline import build_graphs, candidates_job, mask_nms, to_dets  # noqa: E402

CFG = dict(
    scales=(0.5, 0.75),
    merge_levels=(0.6, 0.7, 0.8, 0.85, 0.9),
    nms=0.88,
    min_score=0.15,
)
_NMS = None


def _nms_job(i):
    cands, scores, hw, nms, ms = _NMS
    rs = cands[i]
    if not rs:
        return []
    keep = mask_nms(rs, scores[i], nms, min_score=ms)
    return to_dets(rs, keep, scores[i], 0, hw[i])


def edge_maps(stems, split, maps_dir, ck_dir):
    todo = [s for s in stems if not (os.path.exists(f'{maps_dir}/{s}_edge.png') and os.path.exists(f'{maps_dir}/{s}_ext.png'))]
    if not todo:
        return
    os.makedirs(maps_dir, exist_ok=True)
    model = build(2, f'{ck_dir}/edter_stage2.pth', f'{ck_dir}/edter_stage1.pth')
    t0 = time.time()
    for i, s in enumerate(todo):
        r = multiscale_infer(model, split, s, 2, list(CFG['scales']), True)
        cv2.imwrite(f'{maps_dir}/{s}_edge.png', (r['edge'][0, 0].cpu().numpy() * 65535).astype(np.uint16))
        cv2.imwrite(f'{maps_dir}/{s}_ext.png', (r['extent'][0, 0].cpu().numpy() * 65535).astype(np.uint16))
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f'  edge maps {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)', flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    global _NMS
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='folder containing <split>/images/*.jpg')
    ap.add_argument('--split', default='Val')
    ap.add_argument('--list', default=None, help='optional text file with image names (default: all images)')
    ap.add_argument('--ann', default=None, help='COCO json used for image ids (default: <data_root>/<split>/annotations/instances.json)')
    ap.add_argument('--out', required=True, help='output detections json')
    ap.add_argument('--work', default=None, help='cache folder for edge maps (default: work/<split>)')
    ap.add_argument('--ckpt', default=os.path.join(ROOT, 'checkpoints'))
    ap.add_argument('--procs', type=int, default=16)
    args = ap.parse_args()

    P.IMG_ROOT = os.path.abspath(args.data_root)
    img_dir = f'{P.IMG_ROOT}/{args.split}/images'
    if args.list:
        stems = [l.strip() for l in open(args.list) if l.strip()]
    else:
        stems = sorted(f.rsplit('.', 1)[0] for f in os.listdir(img_dir) if f.lower().endswith('.jpg'))
    ann = args.ann or f'{P.IMG_ROOT}/{args.split}/annotations/instances.json'
    name2img = {im['file_name'].rsplit('.', 1)[0]: im for im in json.load(open(ann))['images']} if os.path.exists(ann) else {}
    maps_dir = os.path.join(args.work or os.path.join('work', args.split), 'maps')
    print(f'{len(stems)} images', flush=True)

    # 1. edge / extent maps
    edge_maps(stems, args.split, maps_dir, args.ckpt)

    # 2. region graphs and merge probabilities
    ck = torch.load(f'{args.ckpt}/gcn_r18.pt', map_location='cpu', weights_only=False)
    p = ck['params']
    model, norm = CG.build_model(ck)
    model = model.cuda()
    net = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
    with Pool(args.procs) as pool:
        graphs = build_graphs(stems, args.split, maps_dir, p, net, pool, ck['edge_feats'])
        CG.add_patches(graphs, args.split, pool)
        probs = CG.predict_all(model, norm, graphs, 'cuda')
        for g in graphs:
            del g['patches']
        print('  merge probabilities done', flush=True)
        # 3. candidate regions
        cands = dict(pool.map(candidates_job, [(g, pr, list(CFG['merge_levels']), p) for g, pr in zip(graphs, probs)],
                              chunksize=4))
    print(f'  candidates: {sum(len(v) for v in cands.values())}', flush=True)

    # 4. region scores
    scorer = pickle.load(open(f'{args.ckpt}/scorer.pkl', 'rb'))
    rr = [cands[s] for s in stems]
    scores = [np.clip(scorer.predict(np.array([r['feats'] for r in rs])), 0, 1) if rs else None for rs in rr]

    # 5. mask NMS
    hw = []
    for s in stems:
        im = name2img.get(s)
        hw.append((im['height'], im['width']) if im else cv2.imread(f'{img_dir}/{s}.jpg', cv2.IMREAD_UNCHANGED).shape[:2])
    _NMS = (rr, scores, hw, CFG['nms'], CFG['min_score'])
    with get_context('fork').Pool(args.procs) as pl:
        parts = pl.map(_nms_job, range(len(stems)), chunksize=4)
    dets = []
    for k, (s, ds) in enumerate(zip(stems, parts)):
        iid = name2img[s]['id'] if s in name2img else k
        for d in ds:
            d['image_id'] = iid
            d['file_name'] = s + '.jpg'
            dets.append(d)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dets, open(args.out, 'w'))
    print(f'{len(dets)} detections -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
