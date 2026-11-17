"""Multi-hypothesis instance extraction from EDTER edge/extent maps.

1. watershed over-segmentation (postproc.make_markers + watershed)
2. a merge hierarchy: greedy boundary merging at several thresholds t_merge (UCM-like levels)
3. every distinct region of every level is a candidate; per-candidate features
4. a gradient-boosted scorer (trained on the Train hold-out ONLY) predicts IoU with the best GT crown
5. greedy mask-NMS on predicted scores -> final instances (scores used for AP ranking)
"""
import argparse
import json
import os
import pickle
import re
from multiprocessing import Pool

import cv2
import numpy as np
from pycocotools import mask as M
from pycocotools.coco import COCO
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import postproc as P
from data import load_inst

LEVELS = (0.0, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
FEAT_NAMES = ['log_area', 'ext_mean', 'ext_min_q10', 'edge_in', 'edge_ring', 'edge_ring_min_q10', 'ring_bg_frac',
              'solidity', 'extent_ratio', 'aspect', 'border', 'level', 'n_parts', 'ring_minus_in', 'eq_diam']
K3 = np.ones((3, 3), np.uint8)


def region_feats(m, edge, ext, fg, sl, level, H, W, n_parts):
    a = m.sum()
    md = cv2.dilate(m.astype(np.uint8), K3, iterations=2).astype(bool)
    ring = md & ~m
    me = cv2.erode(m.astype(np.uint8), K3, iterations=2).astype(bool)
    inn = me if me.any() else m
    e_in = edge[inn]; e_ring = edge[ring] if ring.any() else np.array([0.])
    x_in = ext[m]
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(cnts)) if cnts else None
    ha = cv2.contourArea(hull) if hull is not None else a
    h, w = m.shape
    border = float(sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W)
    ring_bg = (~fg[ring]).mean() if ring.any() else 0.
    return [np.log(a), x_in.mean(), np.quantile(x_in, 0.1), e_in.mean(), e_ring.mean(), np.quantile(e_ring, 0.1), ring_bg,
            a / max(ha, 1), a / (h * w), max(h, w) / max(min(h, w), 1), border, level, n_parts,
            e_ring.mean() - e_in.mean(), np.sqrt(a)]


def candidates(edge, ext, p, levels=LEVELS, pad=3):
    """Returns list of dict(level, mask(bool, cropped), box(y0,x0,y1,x1), feats)."""
    fg = ext > p['t_ext']
    markers = P.make_markers(fg, edge, p)
    elev = ndi.gaussian_filter(edge, p['sigma']) if p['sigma'] > 0 else edge
    lab0 = watershed(elev, markers, mask=fg)
    H, W = edge.shape
    out, seen = [], set()
    for lv in levels:
        lab = P.boundary_merge(lab0, elev, lv, p.get('absorb', 0)) if (lv > 0 or p.get('absorb', 0)) else lab0
        for k, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is None:
                continue
            # padded crop so that ring features see the surroundings
            y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
            x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
            m = lab[y0:y1, x0:x1] == k
            a = int(m.sum())
            if a < p['min_area']:
                continue
            key = (a, sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
            if key in seen:
                continue
            seen.add(key)
            m = ndi.binary_fill_holes(m)
            n_parts = int(lab0[y0:y1, x0:x1][m].max() > 0 and len(np.unique(lab0[y0:y1, x0:x1][m])))
            f = region_feats(m, edge[y0:y1, x0:x1], ext[y0:y1, x0:x1], fg[y0:y1, x0:x1], sl, lv, H, W, n_parts)
            out.append(dict(level=lv, mask=m, box=(y0, x0, y1, x1), feats=f))
    return out


def iou_targets(cands, inst_small):
    """IoU of each candidate with its best-matching GT instance (on the 512 grid)."""
    gareas = np.bincount(inst_small.ravel())
    t = []
    for c in cands:
        y0, x0, y1, x1 = c['box']
        g = inst_small[y0:y1, x0:x1][c['mask']]
        g = g[g > 0]
        if g.size == 0:
            t.append(0.); continue
        ids, inter = np.unique(g, return_counts=True)
        a = c['mask'].sum()
        iou = inter / (a + gareas[ids] - inter)
        t.append(float(iou.max()))
    return t


def _cand_job(args):
    stem, split, pred_dir, p, with_target = args
    e, x = P.load_maps(pred_dir, stem)
    cs = candidates(e, x, p)
    tg = None
    if with_target:
        inst = load_inst(split, stem)
        inst = cv2.resize(inst.astype(np.uint16), e.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.int32)
        tg = iou_targets(cs, inst)
    return stem, cs, tg


def mask_nms(cands, scores, thr, max_det=100, min_score=0.0):
    order = np.argsort(-scores)
    keep, rles = [], []
    for i in order:
        if scores[i] < min_score or len(keep) >= max_det:
            break
        c = cands[i]
        y0, x0, y1, x1 = c['box']
        full = np.zeros(c['img_hw'], np.uint8)
        full[y0:y1, x0:x1] = c['mask']
        r = M.encode(np.asfortranarray(full))
        if rles and M.iou([r], rles, [0] * len(rles)).max() > thr:
            continue
        keep.append(i); rles.append(r)
    return keep


def to_dets(cands, keep, scores, img_id, out_hw):
    H, W = out_hw
    dets = []
    for i in keep:
        c = cands[i]
        h, w = c['img_hw']
        y0, x0, y1, x1 = c['box']
        sy, sx = H / h, W / w
        crop = cv2.resize(c['mask'].astype(np.uint8), (int(round((x1 - x0) * sx)), int(round((y1 - y0) * sy))),
                          interpolation=cv2.INTER_NEAREST)
        full = np.zeros((H, W), np.uint8)
        Y0, X0 = int(round(y0 * sy)), int(round(x0 * sx))
        full[Y0:Y0 + crop.shape[0], X0:X0 + crop.shape[1]] = crop[:H - Y0, :W - X0]
        r = M.encode(np.asfortranarray(full)); r['counts'] = r['counts'].decode(); r['size'] = [int(v) for v in r['size']]
        dets.append(dict(image_id=img_id, category_id=1, segmentation=r, score=float(scores[i])))
    return dets


def collect(pred_dir, split, stems, p, pool, with_target):
    res = {}
    for stem, cs, tg in pool.imap(_cand_job, [(s, split, pred_dir, p, with_target) for s in stems], chunksize=2):
        for c in cs:
            c['img_hw'] = (512, 512)
        res[stem] = (cs, tg)
    return res


def fit_scorer(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0).fit(X, y)


def evaluate(gt, split, res, scorer, nms_thr, pool=None):
    name2img = {os.path.splitext(im['file_name'])[0]: im for im in gt.dataset['images']}
    dets, ids = [], []
    for stem, (cs, _) in res.items():
        im = name2img[stem]
        ids.append(im['id'])
        if not cs:
            continue
        sc = np.clip(scorer.predict(np.array([c['feats'] for c in cs])), 0, 1)
        keep = mask_nms(cs, sc, nms_thr)
        dets += to_dets(cs, keep, sc, im['id'], (im['height'], im['width']))
    m = P.coco_eval(gt, dets, ids)
    m.update(P.f1_at50(gt, dets, ids))
    return m, dets, ids


def src_of(s):
    return re.sub(r'_r\d+_c\d+$', '', s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hold_pred', required=True)
    ap.add_argument('--val_pred', required=True)
    ap.add_argument('--params', required=True, help='json from postproc --tune (base watershed params)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = dict(P.DEFAULT, **json.load(open(args.params))['params'])
    p['t_merge'] = 0
    hold = [l.strip() for l in open('data/train_hold.txt') if l.strip()]
    val = [l.strip() for l in open('data/Val/list.txt') if l.strip()]
    gt_tr, gt_val = COCO(P.ANN.format('Train')), COCO(P.ANN.format('Val'))
    log = open(f'{args.out}/log.txt', 'w')

    def say(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

    with Pool(args.procs) as pool:
        rh = collect(args.hold_pred, 'Train', hold, p, pool, True)
        say('hold candidates', sum(len(v[0]) for v in rh.values()))
        # 2-fold CV over source photos inside the hold-out to choose the NMS threshold honestly
        srcs = sorted({src_of(s) for s in rh})
        cv = {}
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            aps = []
            for f in srcs:
                tr = {s: v for s, v in rh.items() if src_of(s) != f}
                te = {s: v for s, v in rh.items() if src_of(s) == f}
                X = np.array([c['feats'] for cs, _ in tr.values() for c in cs]); y = np.concatenate([t for _, t in tr.values()])
                m, _, _ = evaluate(gt_tr, 'Train', te, fit_scorer(X, y), thr)
                aps.append(m['AP'])
            cv[thr] = float(np.mean(aps))
            say(f'CV nms {thr}: AP {cv[thr]:.4f} folds {aps}')
        thr = max(cv, key=cv.get)
        X = np.array([c['feats'] for cs, _ in rh.values() for c in cs]); y = np.concatenate([t for _, t in rh.values()])
        scorer = fit_scorer(X, y)
        pickle.dump(dict(scorer=scorer, params=p, nms=thr, feats=FEAT_NAMES), open(f'{args.out}/scorer.pkl', 'wb'))
        rv = collect(args.val_pred, 'Val', val, p, pool, False)
        say('val candidates', sum(len(v[0]) for v in rv.values()))
    m, dets, ids = evaluate(gt_val, 'Val', rv, scorer, thr)
    say('VAL', json.dumps(m))
    json.dump(dict(params=p, nms=thr, cv=cv, metrics=m, n_images=len(ids)), open(f'{args.out}/val_metrics.json', 'w'), indent=1)
    json.dump(dets, open(f'{args.out}/val_dets.json', 'w'))


if __name__ == '__main__':
    main()
