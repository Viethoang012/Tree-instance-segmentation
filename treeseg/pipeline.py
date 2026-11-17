"""Inference helpers: over-segmentation graphs and candidate regions at several merge levels."""
import numpy as np
from scipy import ndimage as ndi

import gcn_merge as G
from hierarchy import region_feats


def build_graphs(stems, split, maps_dir, p, net, pool, edge_feats, chunk=48):
    """Over-segmentation region graphs (node / edge features + LPIPS) for every image."""
    gs = []
    for k in range(0, len(stems), chunk):
        part = pool.starmap(G.build_graph, [(s, split, maps_dir, p, False) for s in stems[k:k + chunk]])
        G.add_lpips(part, net, 'cuda')
        gs += part
    for g in gs:
        g['eattr'] = G.edge_attr(g, edge_feats)
    return gs


def graph_feats(g, prob, root, level):
    """Per merged region (keyed by its root id): merge level, number of over-segmentation parts, merge probability
    of the internal edges and of the boundary edges to neighbouring regions."""
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
    """Regions of a label map -> list of dict(feats, mask (cropped, map grid), box, img_hw)."""
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
    """Candidate regions of one image: GCN merging at every level in `levels`, duplicates removed."""
    g, prob, levels, p = args
    out, seen = [], set()
    for lv in levels:
        if len(g['ei']):
            root = G.merge_labels(g, prob, lv, return_root=True)
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
