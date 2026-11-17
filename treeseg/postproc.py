"""EDTER edge (+extent) maps -> tree instance masks, and COCO-style evaluation.

Pipeline per image (maps are at 512x512 = 0.5x of the 1024 tiles):
  1. fg      = extent > t_ext                                    (tree-crown foreground)
  2. core    = fg & (edge < t_edge), morphological opening (r_open) to cut thin bridges
  3. markers = connected components of core with >= min_seed px
  4. labels  = watershed(edge (smoothed), markers, mask=fg)     (edge pixels go to the adjacent crown)
  5. drop instances < min_area, fill holes, upsample label map to 1024 (nearest)
  6. score   = mean(extent in mask) * (1 - mean(edge in mask))  -> ranking for AP
"""
import argparse
import itertools
import json
import os
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from pycocotools import mask as M
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import ndimage as ndi
from skimage.segmentation import watershed

IMG_ROOT = '/home/hoangdmv/Computer_vision/TreeCog/ForestSeg_subset/ForestSeg-T1'
ANN = IMG_ROOT + '/{}/annotations/instances.json'

DEFAULT = dict(t_ext=0.5, t_edge=0.5, r_open=1, min_seed=10, min_area=30, sigma=1.0)


def load_maps(pred_dir, stem):
    e = cv2.imread(f'{pred_dir}/{stem}_edge.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
    x = cv2.imread(f'{pred_dir}/{stem}_ext.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
    return e, x


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def boundary_merge(lab, edge, t_merge, min_area=0, return_root=False):
    """Greedy region merging (OWT-UCM style): repeatedly merge the two adjacent regions whose shared
    boundary has the lowest mean edge probability, while it is < t_merge. Afterwards regions smaller than
    min_area are absorbed by their weakest-boundary neighbour. Background (0) never merges."""
    a = np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()])
    b = np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()])
    w = np.concatenate([((edge[:, :-1] + edge[:, 1:]) / 2).ravel(), ((edge[:-1, :] + edge[1:, :]) / 2).ravel()])
    sel = (a != b) & (a > 0) & (b > 0)
    a, b, w = a[sel], b[sel], w[sel]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    n = int(lab.max()) + 1
    key = lo.astype(np.int64) * n + hi
    uk, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=w)
    c = np.bincount(inv).astype(np.float64)
    nb = {}
    for k, ss, cc in zip(uk, s, c):
        i, j = int(k // n), int(k % n)
        nb.setdefault(i, {})[j] = [ss, cc]
        nb.setdefault(j, {})[i] = [ss, cc]
    area = np.bincount(lab.ravel(), minlength=n).astype(np.int64)
    parent = np.arange(n)

    def merge(i, j):  # merge j into i
        for k, (ss, cc) in nb.pop(j).items():
            nb[k].pop(j)
            if k == i:
                continue
            if k in nb[i]:
                nb[i][k][0] += ss; nb[i][k][1] += cc
            else:
                nb[i][k] = [ss, cc]
            nb[k][i] = nb[i][k]
        parent[j] = i
        area[i] += area[j]

    if t_merge > 0:
        while True:
            best, bi, bj = t_merge, -1, -1
            for i, d in nb.items():
                for j, (ss, cc) in d.items():
                    if i < j and ss / cc < best:
                        best, bi, bj = ss / cc, i, j
            if bi < 0:
                break
            merge(bi, bj)
    if min_area > 0:
        changed = True
        while changed:
            changed = False
            for j in list(nb.keys()):
                if j in nb and area[j] < min_area and nb[j]:
                    i = min(nb[j], key=lambda k: nb[j][k][0] / nb[j][k][1])
                    merge(i, j); changed = True
    # resolve union-find
    root = parent.copy()
    for _ in range(64):
        nr = root[root]
        if np.array_equal(nr, root):
            break
        root = nr
    return root if return_root else root[lab]


def make_markers(fg, edge, p):
    core = (fg & (edge < p['t_edge'])).astype(np.uint8)
    if p['r_open'] > 0:
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, disk(p['r_open']))
    if p.get('marker', 'cc') == 'dist':
        from skimage.morphology import h_maxima
        D = ndi.distance_transform_edt(core)
        pk = h_maxima(ndi.gaussian_filter(D, 1.0), p['h']) & (core > 0)
        markers, _ = ndi.label(pk)
        return markers
    n, cc, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
    keep = np.zeros(n, np.int32)
    good = np.where(stats[:, cv2.CC_STAT_AREA] >= p['min_seed'])[0]
    good = good[good > 0]
    keep[good] = np.arange(1, len(good) + 1)
    return keep[cc]


def exg_foreground(img):
    """Classical vegetation mask: Excess-Green index 2g-r-b on chromatic coords, Otsu threshold."""
    from skimage.filters import threshold_otsu
    f = img.astype(np.float32)
    s = f.sum(2) + 1e-6
    r, g, b = f[..., 0] / s, f[..., 1] / s, f[..., 2] / s
    exg = 2 * g - r - b
    return (exg > threshold_otsu(exg)).astype(np.float32)


def bright_foreground(img):
    """Classical crown mask: crowns are brighter than the shaded ground -> Otsu on smoothed grey level."""
    from skimage.filters import threshold_otsu
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2)
    return (g > threshold_otsu(g)).astype(np.float32)


def labels_from_maps(edge, ext, p, merge=True):
    """p['fg']: 'extent' (learned EDTER extent head) | 'none' (edge only) | 'exg' (ext already replaced by ExG mask)
    p['method']: 'ws' (marker watershed) | 'cc' (connected components of the non-edge core only)."""
    if p.get('fg', 'extent') == 'none':
        ext = np.ones_like(edge)
    fg = ext > p['t_ext']
    markers = make_markers(fg, edge, p)
    elev = ndi.gaussian_filter(edge, p['sigma']) if p['sigma'] > 0 else edge
    do_merge = merge and (p.get('t_merge', 0) > 0 or p.get('absorb', 0) > 0)
    if p.get('method', 'ws') == 'cc':
        lab = markers
        if do_merge:  # adjacency + boundary strength from the watershed partition, merged ids mapped back onto CCs
            root = boundary_merge(watershed(elev, markers, mask=fg), elev, p.get('t_merge', 0), p.get('absorb', 0),
                                  return_root=True)
            lab = root[markers]
    else:
        lab = watershed(elev, markers, mask=fg)
        if do_merge:
            lab = boundary_merge(lab, elev, p.get('t_merge', 0), p.get('absorb', 0))
    return lab, ext


def instances_from_maps(edge, ext, p, out_hw=(1024, 1024)):
    lab, ext = labels_from_maps(edge, ext, p)
    return labels_to_instances(lab, edge, ext, p, out_hw)


def labels_to_instances(lab, edge, ext, p, out_hw=(1024, 1024)):
    H, W = out_hw
    lab_up = cv2.resize(lab.astype(np.int32).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    res = []
    objs = ndi.find_objects(lab)
    sy, sx = H / lab.shape[0], W / lab.shape[1]
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m_small = lab[sl] == k
        area = m_small.sum()
        if area < p['min_area']:
            continue
        score = float(ext[sl][m_small].mean() * (1 - edge[sl][m_small].mean()))
        ys = slice(int(sl[0].start * sy), int(np.ceil(sl[0].stop * sy)))
        xs = slice(int(sl[1].start * sx), int(np.ceil(sl[1].stop * sx)))
        m = np.zeros((H, W), np.uint8)
        m[ys, xs] = ndi.binary_fill_holes(lab_up[ys, xs] == k)
        rle = M.encode(np.asfortranarray(m))
        rle['counts'] = rle['counts'].decode(); rle['size'] = [int(v) for v in rle['size']]
        res.append(dict(segmentation=rle, score=score))
    return res


def _work(args):
    stem, img_id, hw, pred_dir, p = args
    e, x = load_maps(pred_dir, stem)
    if p.get('fg') in ('exg', 'bright'):
        img = cv2.cvtColor(cv2.imread(f"{IMG_ROOT}/{p['img_split']}/images/{stem}.jpg"), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, e.shape[::-1], interpolation=cv2.INTER_AREA)
        x = exg_foreground(img) if p['fg'] == 'exg' else bright_foreground(img)
    return [dict(image_id=img_id, category_id=1, **r) for r in instances_from_maps(e, x, p, hw)]


def predict(pred_dir, gt, stems, p, pool):
    name2img = {os.path.splitext(im['file_name'])[0]: im for im in gt.dataset['images']}
    jobs = [(s, name2img[s]['id'], (name2img[s]['height'], name2img[s]['width']), pred_dir, p) for s in stems]
    dets = []
    for r in pool.imap(_work, jobs, chunksize=4):
        dets.extend(r)
    return dets, [name2img[s]['id'] for s in stems]


def f1_at50(gt, dets, img_ids):
    """Greedy one-to-one matching at IoU>=0.5 over all images (score-sorted), plus mean IoU of matches."""
    tp = fp = fn = 0
    ious = []
    by = {}
    for d in dets:
        by.setdefault(d['image_id'], []).append(d)
    for i in img_ids:
        g = [a['segmentation'] for a in gt.loadAnns(gt.getAnnIds(imgIds=i))]
        h, w = gt.imgs[i]['height'], gt.imgs[i]['width']
        g = [M.merge(M.frPyObjects(s, h, w)) if isinstance(s, list) else s for s in g]
        d = sorted(by.get(i, []), key=lambda z: -z['score'])
        ds = [z['segmentation'] for z in d]
        if not g or not ds:
            fp += len(ds); fn += len(g); continue
        iou = M.iou(ds, g, [0] * len(g))
        used = np.zeros(len(g), bool)
        for r in range(len(ds)):
            row = np.where(used, -1, iou[r])
            j = row.argmax()
            if row[j] >= 0.5:
                used[j] = True; tp += 1; ious.append(row[j])
            else:
                fp += 1
        fn += (~used).sum()
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(P50=P, R50=R, F1_50=2 * P * R / max(P + R, 1e-8), mIoU_matched=float(np.mean(ious)) if ious else 0.,
                n_pred=len(dets), n_gt=int(tp + fn))


def coco_eval(gt, dets, img_ids, verbose=False):
    if not dets:
        return dict(AP=0.)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, 'segm')
        E.params.imgIds = img_ids
        E.params.maxDets = [1, 10, 100]
        # IoU thresholds 0.25:0.05:0.95 (15 values) so that AP25 and AP@[.25:.95] are available too
        E.params.iouThrs = np.round(np.arange(0.25, 0.951, 0.05), 2)
        E.evaluate(); E.accumulate()
    thr = E.params.iouThrs
    prec = E.eval['precision']  # [T, R, K, A, M]; area idx 0=all,1=small,2=medium,3=large; maxDet idx 2 = 100
    rec = E.eval['recall']      # [T, K, A, M]

    def ap(ts, a=0):
        p = prec[ts, :, :, a, 2]
        p = p[p > -1]
        return float(p.mean()) if p.size else -1.

    def ar(ts, a=0):
        r = rec[ts, :, a, 2]
        r = r[r > -1]
        return float(r.mean()) if r.size else -1.

    t = lambda v: [int(np.argmin(np.abs(thr - v)))]
    std = [i for i, v in enumerate(thr) if v >= 0.5 - 1e-6]   # standard COCO 0.50:0.95
    allt = list(range(len(thr)))                               # 0.25:0.95
    m = dict(AP=ap(std), AP25=ap(t(0.25)), AP50=ap(t(0.5)), AP70=ap(t(0.7)), AP75=ap(t(0.75)), AP_25_95=ap(allt),
             APs=ap(std, 1), APm=ap(std, 2), APl=ap(std, 3), AR100=ar(std), AR100_25_95=ar(allt))
    if verbose:
        print(' '.join(f'{k}={v:.4f}' for k, v in m.items()))
    return m


def tune(pred_dir, split, stems, pool, log=None, base=None, space=None, objective='AP'):
    """Coordinate-descent search of post-processing params on the hold-out set (objective: COCO mask AP)."""
    gt = COCO(ANN.format(split)) if not isinstance(split, COCO) else split
    space = space or dict(t_edge=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], t_ext=[0.3, 0.4, 0.5, 0.6],
                          r_open=[0, 1, 2, 3], min_seed=[5, 10, 20, 40], min_area=[10, 30, 60, 100], sigma=[0, 1.0, 2.0])
    best = dict(DEFAULT, **(base or {}))
    cache = {}

    def score(p):
        key = tuple(sorted(p.items()))
        if key not in cache:
            dets, ids = predict(pred_dir, gt, stems, p, pool)
            cache[key] = coco_eval(gt, dets, ids)[objective]
            msg = f'  {p} -> {objective} {cache[key]:.4f}'
            print(msg, flush=True)
            if log: log.write(msg + '\n'); log.flush()
        return cache[key]

    cur = score(best)
    for rnd in range(2):
        changed = False
        for k, vals in space.items():
            for v in vals:
                p = dict(best, **{k: v})
                s = score(p)
                if s > cur + 1e-4:
                    cur, best, changed = s, p, True
        if not changed:
            break
    return best, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='dir with <stem>_edge.png / <stem>_ext.png')
    ap.add_argument('--split', default='Val')
    ap.add_argument('--list', required=True)
    ap.add_argument('--tune', action='store_true', help='tune params on this set (use the Train hold-out!)')
    ap.add_argument('--params', default=None, help='json file with params (from --tune)')
    ap.add_argument('--out', required=True, help='output json (metrics / params)')
    ap.add_argument('--save_dets', default=None)
    ap.add_argument('--procs', type=int, default=24)
    args = ap.parse_args()
    stems = [l.strip() for l in open(args.list) if l.strip()]
    gt = COCO(ANN.format(args.split))
    with Pool(args.procs) as pool:
        if args.tune:
            log = open(args.out.replace('.json', '_log.txt'), 'w')
            best, ap_ = tune(args.pred, gt, stems, pool, log)
            p = best
        else:
            p = dict(DEFAULT, **json.load(open(args.params))['params']) if args.params else dict(DEFAULT)
        dets, ids = predict(args.pred, gt, stems, p, pool)
    m = coco_eval(gt, dets, ids, verbose=True)
    m.update(f1_at50(gt, dets, ids))
    res = dict(params=p, metrics=m, n_images=len(ids))
    print(json.dumps(res, indent=1))
    json.dump(res, open(args.out, 'w'), indent=1)
    if args.save_dets:
        json.dump(dets, open(args.save_dets, 'w'))


if __name__ == '__main__':
    main()
