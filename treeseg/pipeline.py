import cv2
import numpy as np
from pycocotools import mask as M
from scipy import ndimage as ndi

import gcn_merge as G

K3 = np.ones((3, 3), np.uint8)


def build_graphs(stems, split, maps_dir, p, net, pool, edge_feats, chunk=48):
    gs = []
    for k in range(0, len(stems), chunk):
        part = pool.starmap(G.build_graph, [(s, split, maps_dir, p) for s in stems[k:k + chunk]])
        G.add_lpips(part, net, 'cuda')
        gs += part
    for g in gs:
        g['eattr'] = G.edge_attr(g, edge_feats)
    return gs


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


def graph_feats(g, prob, root, level):
    n = len(root)
    parts = np.bincount(root, minlength=n)
    ri, rj = root[g['ei']], root[g['ej']]
    inner = ri == rj
    cnt_in = np.bincount(ri[inner], minlength=n)
    sum_in = np.bincount(ri[inner], weights=prob[inner], minlength=n)
    min_in = np.ones(n); np.minimum.at(min_in, ri[inner], prob[inner])
    b = ~inner
    bi, bp = np.concatenate([ri[b], rj[b]]), np.concatenate([prob[b], prob[b]])
    cnt_b = np.bincount(bi, minlength=n)
    sum_b = np.bincount(bi, weights=bp, minlength=n)
    max_b = np.zeros(n); np.maximum.at(max_b, bi, bp)
    mean_in = np.where(cnt_in > 0, sum_in / np.maximum(cnt_in, 1), 1.)
    mean_b = np.where(cnt_b > 0, sum_b / np.maximum(cnt_b, 1), 0.)
    return lambda k: [level, np.log(parts[k]), mean_in[k], min_in[k], max_b[k], mean_b[k], cnt_b[k]]


def regions(lab, edge, ext, p, gf, pad=3):
    H, W = lab.shape
    fg = ext > p['t_ext']
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        m = lab[y0:y1, x0:x1] == k
        if m.sum() < p['min_area']:
            continue
        m = ndi.binary_fill_holes(m)
        f = region_feats(m, edge[y0:y1, x0:x1], ext[y0:y1, x0:x1], fg[y0:y1, x0:x1], sl, 0, H, W, 1)
        f = list(f) + gf(k)
        out.append(dict(feats=f, mask=m, box=(y0, x0, y1, x1), img_hw=(H, W)))
    return out


def candidates_job(args):
    g, prob, levels, p = args
    out, seen = [], set()
    for lv in levels:
        if len(g['ei']):
            root = G.merge_labels(g, prob, lv)
            lab = root[g['lab']]
            gf = graph_feats(g, prob, root, lv)
        else:
            lab = g['lab']
            gf = lambda k, lv=lv: [lv, 0., 1., 1., 0., 0., 0]
        for r in regions(lab, g['edge'], g['ext'], p, gf):
            key = (int(r['mask'].sum()),) + tuple(r['box'])
            if key not in seen:
                seen.add(key); r['level'] = lv; out.append(r)
    return g['stem'], out


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
