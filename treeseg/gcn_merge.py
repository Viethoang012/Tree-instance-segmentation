"""GCN-based region merging on top of an EDTER over-segmentation.

Graph per image
  node  = one region of the over-segmentation (watershed on the EDTER edge map, distance-transform markers)
          5 node features: log-area, extent (area / bbox area), solidity, aspect ratio, deviation (std of grey level)
  edge  = two regions sharing a boundary; edge attribute = LPIPS distance between the two regions' image patches
          (optional extra edge attributes: mean/min EDTER edge prob along the shared boundary, --edge_feats)
  label = 1 if both regions have the same majority GT crown id (regions whose majority is background: label 0)
Model
  GCN (2 layers, symmetric-normalised adjacency weighted by exp(-lpips / tau)) -> node embeddings
  edge head: MLP([h_i, h_j, |h_i - h_j|, edge_attr]) -> P(merge)
Merging
  union-find over edges with P > t  (t chosen on the Train hold-out, never on Val), optionally iterated.

Usage
  build : python gcn_merge.py build --split Train --list data/train_fit.txt --pred results/s1_final/maps_fit --out graphs/fit
  train : python gcn_merge.py train --train graphs/fit --hold graphs/hold --out gcn_runs/spec
  eval  : python gcn_merge.py eval --model gcn_runs/spec/model.pt --graphs graphs/val --split Val --out gcn_runs/spec/val.json
"""
import argparse
import glob
import json
import os
import re
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from scipy import ndimage as ndi

import postproc as P
from data import load_inst

OVERSEG = dict(P.DEFAULT, fg='extent', method='ws', t_ext=0.5, t_edge=0.3, r_open=0, min_seed=10, min_area=30,
               sigma=1.0, marker='dist', h=1)
PATCH = 64
NODE_FEATS = ['log_area', 'extent', 'solidity', 'aspect', 'deviation']


# ----------------------------------------------------------------------------------------------- graph building
def region_patch(img, m, sl):
    crop = img[sl].copy()
    mm = m[sl]
    crop[~mm] = crop[mm].mean(0) if mm.any() else 0
    return cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA)


def build_graph(stem, split, pred_dir, p, with_gt, lab=None, maps=None):
    edge, ext = P.load_maps(pred_dir, stem) if maps is None else maps
    if lab is None:
        lab, ext = P.labels_from_maps(edge, ext, p, merge=False)
    img = cv2.cvtColor(cv2.imread(f'{P.IMG_ROOT}/{split}/images/{stem}.jpg'), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, edge.shape[::-1], interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    n = int(lab.max()) + 1
    objs = ndi.find_objects(lab)
    feats = np.zeros((n, 5), np.float32)
    patches = np.zeros((n, PATCH, PATCH, 3), np.uint8)
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m = lab[sl] == k
        a = m.sum()
        h, w = m.shape
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(np.concatenate(cnts))) if cnts else a
        feats[k] = [np.log(a), a / (h * w), a / max(hull, 1), max(h, w) / max(min(h, w), 1), grey[sl][m].std()]
        full = np.zeros(lab.shape, bool); full[sl] = m
        patches[k] = region_patch(img, full, sl)
    # adjacency + boundary edge statistics
    a = np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()])
    b = np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()])
    w = np.concatenate([((edge[:, :-1] + edge[:, 1:]) / 2).ravel(), ((edge[:-1, :] + edge[1:, :]) / 2).ravel()])
    sel = (a != b) & (a > 0) & (b > 0)
    a, b, w = a[sel], b[sel], w[sel]
    key = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float32)
    bmean = np.bincount(inv, weights=w) / cnt
    bmin = np.full(len(uk), 1.0, np.float32); np.minimum.at(bmin, inv, w.astype(np.float32))
    ei, ej = (uk // n).astype(np.int64), (uk % n).astype(np.int64)
    bfeat = np.stack([bmean, bmin, np.log(cnt)], 1).astype(np.float32)
    efeat = rich_edge_feats(lab, edge, ext, img, ei, ej, a, b, w, inv)
    y = None
    if with_gt:
        inst = load_inst(split, stem)
        inst = cv2.resize(inst.astype(np.uint16), edge.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.int64)
        G = int(inst.max()) + 1
        mm = lab > 0
        flat = lab[mm].astype(np.int64) * G + inst[mm]
        u, c = np.unique(flat, return_counts=True)
        maj = np.zeros(n, np.int64); best = np.zeros(n)
        for uu, cc in zip(u, c):
            r, g = divmod(int(uu), G)
            if cc > best[r]:
                best[r], maj[r] = cc, g
        y = ((maj[ei] == maj[ej]) & (maj[ei] > 0)).astype(np.float32)
    return dict(stem=stem, lab=lab.astype(np.int32), edge=edge, ext=ext, x=feats, patches=patches,
                ei=ei, ej=ej, bfeat=bfeat, efeat=efeat, y=y)

EFEATS = ['ext_bnd', 'edge_bstd', 'edge_bmax', 'edge_contrast', 'ext_min', 'dL', 'dab', 'sol_union', 'sol_drop',
          'contact', 'log_area_ratio', 'union_elong', 'union_extent']


def rich_edge_feats(lab, edge, ext, img, ei, ej, a, b, w, inv):
    """Extra edge attributes (--edge_feats 2), mostly independent of how confident the EDTER edge map is:
    boundary extent / edge spread, colour difference (Lab), and the shape of the union of the two regions."""
    E = len(ei)
    out = np.zeros((E, len(EFEATS)), np.float32)
    if E == 0:
        return out
    n = int(lab.max()) + 1
    L = lab.ravel()
    area = np.bincount(L, minlength=n).astype(np.float64)
    safe = np.maximum(area, 1)
    ext_r = np.bincount(L, weights=ext.ravel(), minlength=n) / safe
    edge_r = np.bincount(L, weights=edge.ravel(), minlength=n) / safe
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64).reshape(-1, 3)
    col = np.stack([np.bincount(L, weights=lab_img[:, c], minlength=n) / safe for c in range(3)], 1)
    # boundary pixel-pair statistics (pairs as in build_graph: a, b, w, inv)
    H, W = lab.shape
    xe = np.concatenate([((ext[:, :-1] + ext[:, 1:]) / 2).ravel(), ((ext[:-1, :] + ext[1:, :]) / 2).ravel()])
    sel = (np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()]) != np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()]))
    sel &= (np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()]) > 0) & (np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()]) > 0)
    xe = xe[sel]
    cnt = np.bincount(inv, minlength=E).astype(np.float64)
    out[:, 0] = np.bincount(inv, weights=xe, minlength=E) / cnt
    m1 = np.bincount(inv, weights=w, minlength=E) / cnt
    m2 = np.bincount(inv, weights=w * w, minlength=E) / cnt
    out[:, 1] = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    bmax = np.zeros(E); np.maximum.at(bmax, inv, w); out[:, 2] = bmax
    out[:, 3] = m1 - np.maximum(edge_r[ei], edge_r[ej])
    out[:, 4] = np.minimum(ext_r[ei], ext_r[ej])
    out[:, 5] = np.abs(col[ei, 0] - col[ej, 0])
    out[:, 6] = np.hypot(col[ei, 1] - col[ej, 1], col[ei, 2] - col[ej, 2])
    amin, amax = np.minimum(area[ei], area[ej]), np.maximum(area[ei], area[ej])
    out[:, 9] = cnt / np.sqrt(np.maximum(amin, 1))
    out[:, 10] = np.log(np.maximum(amin, 1) / np.maximum(amax, 1))
    objs = ndi.find_objects(lab)

    def solidity(m):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 1.
        return float(m.sum()) / max(cv2.contourArea(cv2.convexHull(np.concatenate(cnts))), 1.)
    sol = {}
    for e in range(E):
        i, j = int(ei[e]), int(ej[e])
        si, sj = objs[i - 1], objs[j - 1]
        y0, y1 = min(si[0].start, sj[0].start), max(si[0].stop, sj[0].stop)
        x0, x1 = min(si[1].start, sj[1].start), max(si[1].stop, sj[1].stop)
        crop = lab[y0:y1, x0:x1]
        mu = ((crop == i) | (crop == j)).astype(np.uint8)
        su = solidity(mu)
        for r in (i, j):
            if r not in sol:
                sol[r] = solidity((crop == r).astype(np.uint8))
        mo = cv2.moments(mu, binaryImage=True)
        if mo['m00'] > 0:
            c20, c02, c11 = mo['mu20'] / mo['m00'], mo['mu02'] / mo['m00'], mo['mu11'] / mo['m00']
            d = np.sqrt(4 * c11 ** 2 + (c20 - c02) ** 2)
            l1, l2 = (c20 + c02 + d) / 2, (c20 + c02 - d) / 2
            el = np.log(np.sqrt(max(l1, 1e-6) / max(l2, 1e-6)))
        else:
            el = 0.
        out[e, 7] = su
        out[e, 8] = min(sol[i], sol[j]) - su
        out[e, 11] = el
        out[e, 12] = mu.sum() / mu.size
    return out


def n_edge(edge_feats):
    return 1 + 3 * (edge_feats >= 1) + len(EFEATS) * (edge_feats >= 2)


def edge_attr(g, edge_feats):
    if not len(g['ei']):
        return np.zeros((0, n_edge(edge_feats)), np.float32)
    ea = [g['lpips'][:, None]]
    if edge_feats >= 1:
        ea.append(g['bfeat'])
    if edge_feats >= 2:
        ea.append(g['efeat'])
    return np.concatenate(ea, 1).astype(np.float32)


@torch.no_grad()
def add_lpips(graphs, net, device, bs=512):
    for g in graphs:
        if len(g['ei']) == 0:
            g['lpips'] = np.zeros(0, np.float32); continue
        pt = torch.from_numpy(g['patches']).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(device)
        out = []
        for s in range(0, len(g['ei']), bs):
            i = torch.from_numpy(g['ei'][s:s + bs]).to(device); j = torch.from_numpy(g['ej'][s:s + bs]).to(device)
            out.append(net(pt[i], pt[j]).flatten().float().cpu())
        g['lpips'] = torch.cat(out).numpy().astype(np.float32)
        del g['patches']


def cmd_build(args):
    import lpips
    from multiprocessing import Pool
    stems = [l.strip() for l in open(args.list) if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    p = dict(OVERSEG, **json.loads(args.params)) if args.params else OVERSEG
    json.dump(p, open(f'{args.out}/params.json', 'w'))
    dev = 'cuda'
    net = lpips.LPIPS(net='alex', verbose=False).to(dev).eval()
    t0 = time.time()
    with Pool(args.procs) as pool:
        for k in range(0, len(stems), 32):
            chunk = stems[k:k + 32]
            gs = pool.starmap(build_graph, [(s, args.split, args.pred, p, args.split == 'Train') for s in chunk])
            add_lpips(gs, net, dev)
            for g in gs:
                np.savez_compressed(f'{args.out}/{g["stem"]}.npz', **{kk: v for kk, v in g.items() if v is not None})
            print(f'{k + len(chunk)}/{len(stems)} {time.time() - t0:.0f}s', flush=True)


# ----------------------------------------------------------------------------------------------- model
class GCNMerge(nn.Module):
    def __init__(self, n_node=5, n_edge=1, hid=64, layers=2, tau=0.3):
        super().__init__()
        self.inp = nn.Linear(n_node, hid)
        self.layers = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau))))
        self.head = nn.Sequential(nn.Linear(3 * hid + n_edge, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, x, ei, ej, eattr):
        n = x.shape[0]
        w = torch.exp(-eattr[:, 0] / self.log_tau.exp())  # LPIPS-weighted adjacency
        row = torch.cat([ei, ej, torch.arange(n, device=x.device)])
        col = torch.cat([ej, ei, torch.arange(n, device=x.device)])
        val = torch.cat([w, w, torch.ones(n, device=x.device)])
        deg = torch.zeros(n, device=x.device).index_add_(0, row, val)
        norm = val / torch.sqrt(deg[row] * deg[col])
        h = F.relu(self.inp(x))
        for lin in self.layers:
            agg = torch.zeros_like(h).index_add_(0, row, norm[:, None] * h[col])
            h = F.relu(lin(agg)) + h
        z = torch.cat([h[ei], h[ej], (h[ei] - h[ej]).abs(), eattr], 1)
        return self.head(z).squeeze(1)


def load_graphs(d, edge_feats):
    out = []
    for f in sorted(glob.glob(f'{d}/*.npz')):
        g = dict(np.load(f, allow_pickle=False))
        g['stem'] = os.path.basename(f)[:-4]
        g['eattr'] = edge_attr(g, edge_feats)
        out.append(g)
    return out


class Norm:
    def __init__(self, graphs, cols=None):
        self.cols = list(range(len(NODE_FEATS))) if cols is None else list(cols)  # node features kept
        X = np.concatenate([g['x'][1:, self.cols] for g in graphs]); E = np.concatenate([g['eattr'] for g in graphs])
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-6
        self.em, self.es = E.mean(0), E.std(0) + 1e-6
        self.em[0], self.es[0] = 0., 1.  # keep raw LPIPS for the adjacency weight

    def __call__(self, g, dev):
        cols = getattr(self, 'cols', None)
        x = g['x'] if cols is None else g['x'][:, cols]
        return (torch.from_numpy((x - self.xm) / self.xs).to(dev), torch.from_numpy(g['ei']).to(dev),
                torch.from_numpy(g['ej']).to(dev), torch.from_numpy((g['eattr'] - self.em) / self.es).to(dev))


def n_node(norm):
    return len(getattr(norm, 'cols', None) or NODE_FEATS)


EVAL_PROCS = int(os.environ.get('EVAL_PROCS', 16))
MERGE_METHOD = 'single'  # default merge algorithm (module-level so existing callers can switch it)


def merge_labels(g, prob, t, return_root=False, method=None):
    """Partition the region graph from edge merge probabilities.
    single : union-find over edges with P > t (single linkage; one wrong edge merges two crowns)
    avg    : agglomerative, merge the pair of clusters with the highest boundary-length-weighted mean P while >= t
    gaec   : greedy additive edge contraction (multicut heuristic) on costs logit(P) - logit(t), summed over the
             edges between two clusters; merge while the best summed cost is > 0
    """
    method = method or MERGE_METHOD
    n = g['x'].shape[0]
    if method == 'single' or len(g['ei']) == 0:
        parent = np.arange(n)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for i, j in zip(g['ei'][prob > t], g['ej'][prob > t]):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
        root = np.array([find(a) for a in range(n)])
        return root if return_root else root[g['lab']]
    import heapq
    eps = 1e-6
    p = np.clip(prob.astype(np.float64), eps, 1 - eps)
    ln = np.exp(g['bfeat'][:, 2]).astype(np.float64)          # shared boundary length (pixel pairs)
    cost = np.log(p / (1 - p)) - np.log(t / (1 - t)) if 0 < t < 1 else np.full(len(p), -1.)
    adj = [dict() for _ in range(n)]                           # adj[a][b] = [sum p*len, sum len, sum cost]
    for i, j, pp, ll, cc in zip(g['ei'], g['ej'], p, ln, cost):
        for a, b in ((i, j), (j, i)):
            v = adj[a].setdefault(b, [0., 0., 0.]); v[0] += pp * ll; v[1] += ll; v[2] += cc
    key = (lambda v: v[0] / v[1]) if method == 'avg' else (lambda v: v[2])
    ok = (lambda k: k >= t) if method == 'avg' else (lambda k: k > 0)
    root = np.arange(n); members = {a: [a] for a in range(n)}; ver = np.zeros(n, np.int64)
    heap = [(-key(v), a, b, 0, 0) for a in range(n) for b, v in adj[a].items() if a < b]
    heapq.heapify(heap)
    while heap:
        k, a, b, va, vb = heapq.heappop(heap)
        if va != ver[a] or vb != ver[b] or root[a] != a or root[b] != b or b not in adj[a]:
            continue
        if not ok(-k):
            break
        if len(members[a]) < len(members[b]):
            a, b = b, a
        for c, v in adj[b].items():                            # merge b into a
            if c == a:
                continue
            w = adj[a].setdefault(c, [0., 0., 0.]); w[0] += v[0]; w[1] += v[1]; w[2] += v[2]
            adj[c][a] = w; del adj[c][b]
        del adj[a][b]; adj[b] = {}
        for m in members[b]:
            root[m] = a
        members[a] += members.pop(b)
        ver[a] += 1
        for c, v in adj[a].items():
            heapq.heappush(heap, (-key(v), min(a, c), max(a, c), ver[min(a, c)], ver[max(a, c)]))
    root[0] = 0
    return root if return_root else root[g['lab']]


def predict_all(model, norm, graphs, dev):
    model.eval()
    with torch.no_grad():
        return [torch.sigmoid(model(*norm(g, dev))).cpu().numpy() if len(g['ei']) else np.zeros(0) for g in graphs]


_EG = None  # (graphs, probs, t, p, hw) shared with forked eval workers


def _eg_job(i):
    graphs, probs, t, p, hw = _EG
    g, pr = graphs[i], probs[i]
    lab = merge_labels(g, pr, t) if t < 1 else g['lab']
    return P.labels_to_instances(lab, g['edge'], g['ext'], p, hw[i])


def eval_graphs(gt, graphs, probs, t, p, pool=None, procs=None):
    """procs: number of forked workers for mask building (default EVAL_PROCS); 0/1 = serial."""
    global _EG
    name2img = {im['file_name'].rsplit('.', 1)[0]: im for im in gt.dataset['images']}
    ims = [name2img[g['stem']] for g in graphs]
    ids = [im['id'] for im in ims]
    hw = [(im['height'], im['width']) for im in ims]
    procs = EVAL_PROCS if procs is None else procs
    if procs and procs > 1 and len(graphs) > 8:
        from multiprocessing import get_context
        _EG = (graphs, probs, t, p, hw)
        with get_context('fork').Pool(procs) as pl:
            res = pl.map(_eg_job, range(len(graphs)), chunksize=4)
        _EG = None
    else:
        _EG = (graphs, probs, t, p, hw)
        res = [_eg_job(i) for i in range(len(graphs))]
        _EG = None
    dets = [dict(image_id=i, category_id=1, **r) for i, rr in zip(ids, res) for r in rr]
    m = P.coco_eval(gt, dets, ids)
    m.update(P.f1_at50(gt, dets, ids))
    return m, dets


def cmd_train(args):
    dev = 'cuda'
    torch.manual_seed(0); np.random.seed(0)
    tr = load_graphs(args.train, args.edge_feats); ho = load_graphs(args.hold, args.edge_feats)
    tr = [g for g in tr if len(g['ei'])]
    cols = [NODE_FEATS.index(c) for c in args.node_feats.split(',')] if args.node_feats else None
    norm = Norm(tr, cols)
    model = GCNMerge(len(norm.cols), tr[0]['eattr'].shape[1], args.hid, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    pos = np.concatenate([g['y'] for g in tr]).mean()
    pw = torch.tensor((1 - pos) / pos, device=dev)
    print(f'train graphs {len(tr)} edges {sum(len(g["ei"]) for g in tr)} pos-rate {pos:.3f}; hold graphs {len(ho)}', flush=True)
    gt_tr = COCO(P.ANN.format('Train'))
    p = json.load(open(f'{args.hold}/params.json'))
    os.makedirs(args.out, exist_ok=True)
    best = (-1, None, None)
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(len(tr)); tot = 0.
        for k in range(0, len(perm), args.bs):
            opt.zero_grad(); loss = 0.
            for i in perm[k:k + args.bs]:
                g = tr[i]
                logit = model(*norm(g, dev))
                loss = loss + F.binary_cross_entropy_with_logits(logit, torch.from_numpy(g['y']).to(dev), pos_weight=pw)
            (loss / len(perm[k:k + args.bs])).backward(); opt.step(); tot += loss.item()
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            probs = predict_all(model, norm, ho, dev)
            yh = np.concatenate([g['y'] for g in ho]); ph = np.concatenate(probs)
            acc = ((ph > 0.5) == (yh > 0.5)).mean()
            res = {t: eval_graphs(gt_tr, ho, probs, t, p)[0][args.objective] for t in map(float, args.ts.split(','))}
            tb = max(res, key=res.get)
            print(f'ep {ep + 1} loss {tot / len(tr):.4f} hold edge-acc {acc:.3f} hold {args.objective}: ' +
                  ' '.join(f'{t}:{v * 100:.2f}' for t, v in res.items()), flush=True)
            if res[tb] > best[0]:
                best = (res[tb], tb, ep + 1)
                torch.save(dict(model=model.state_dict(), norm=vars(norm), hid=args.hid, layers=args.layers,
                                edge_feats=args.edge_feats, t=tb, params=p, epoch=ep + 1), f'{args.out}/model.pt')
    print(f'best hold {args.objective} {best[0] * 100:.2f} at t={best[1]} epoch {best[2]}', flush=True)


def _rebuild(args):
    stem, split, lab, edge, ext, p, with_gt = args
    return build_graph(stem, split, None, p, with_gt, lab=lab, maps=(edge, ext))


def rebuild_round(gs, probs, t, split, p, with_gt, net, dev, pool, edge_feats):
    """Merge with the current probabilities, relabel, and rebuild the graphs (features + LPIPS) on the merged regions."""
    jobs = []
    for g, pr in zip(gs, probs):
        lab = merge_labels(g, pr, t) if len(g['ei']) else g['lab']
        lab = np.unique(lab, return_inverse=True)[1].reshape(lab.shape).astype(np.int32)  # 0 stays 0 (min id)
        jobs.append((g['stem'], split, lab, g['edge'], g['ext'], p, with_gt))
    new = pool.map(_rebuild, jobs, chunksize=4)
    add_lpips(new, net, dev)
    for g in new:
        g['eattr'] = edge_attr(g, edge_feats)
    return new


def cmd_eval(args):
    import lpips
    from multiprocessing import Pool
    dev = 'cuda'
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    gs = load_graphs(args.graphs, ck['edge_feats'])
    norm = Norm.__new__(Norm); norm.__dict__.update(ck['norm'])
    model = GCNMerge(n_node(norm), gs[0]['eattr'].shape[1], ck['hid'], ck['layers']).to(dev)
    model.load_state_dict(ck['model'])
    probs = predict_all(model, norm, gs, dev)
    gt = COCO(P.ANN.format(args.split))
    p = ck['params']
    K = ('AP_25_95', 'AP', 'AP25', 'AP50', 'AP70', 'AP75', 'F1_50')
    m0, _ = eval_graphs(gt, gs, probs, 1.0, p)
    m1, dets = eval_graphs(gt, gs, probs, ck['t'], p)
    print('no-merge     ' + ' '.join(f'{k} {m0[k] * 100:.2f}' for k in K))
    print(f'GCN round 1  ' + ' '.join(f'{k} {m1[k] * 100:.2f}' for k in K) + f'   (t={ck["t"]})', flush=True)
    rounds = {1: m1}
    if args.rounds > 1:
        net = lpips.LPIPS(net='alex', verbose=False).to(dev).eval()
        with Pool(16) as pool:
            for r in range(2, args.rounds + 1):
                gs = rebuild_round(gs, probs, ck['t'], args.split, p, False, net, dev, pool, ck['edge_feats'])
                probs = predict_all(model, norm, gs, dev)
                mr, dr = eval_graphs(gt, gs, probs, ck['t'], p)
                rounds[r] = mr
                print(f'GCN round {r}  ' + ' '.join(f'{k} {mr[k] * 100:.2f}' for k in K), flush=True)
                if mr[args.objective] > m1[args.objective]:
                    m1, dets = mr, dr
    json.dump(dict(no_merge=m0, gcn_merge=m1, rounds=rounds, t=ck['t'], params=p, n_images=len(gs)), open(args.out, 'w'), indent=1)
    json.dump([{k: d[k] for k in ('image_id', 'category_id', 'segmentation', 'score')} for d in dets],
              open(args.out.replace('.json', '_dets.json'), 'w'))  # loadRes adds numpy area/bbox in place


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('--split', required=True); b.add_argument('--list', required=True)
    b.add_argument('--pred', required=True); b.add_argument('--out', required=True); b.add_argument('--params', default=None)
    b.add_argument('--procs', type=int, default=16)
    t = sub.add_parser('train'); t.add_argument('--train', required=True); t.add_argument('--hold', required=True)
    t.add_argument('--out', required=True); t.add_argument('--epochs', type=int, default=40); t.add_argument('--bs', type=int, default=8)
    t.add_argument('--hid', type=int, default=64); t.add_argument('--layers', type=int, default=2)
    t.add_argument('--edge_feats', type=int, default=0, help='1: + boundary edge-prob stats; 2: + rich features (EFEATS)')
    t.add_argument('--objective', default='AP_25_95'); t.add_argument('--eval_every', type=int, default=5)
    t.add_argument('--node_feats', default='', help='comma list from ' + ','.join(NODE_FEATS) + ' (default: all)')
    t.add_argument('--ts', default='0.3,0.4,0.5,0.6,0.7,0.8', help='merge thresholds tried on the hold-out')
    e = sub.add_parser('eval'); e.add_argument('--model', required=True); e.add_argument('--graphs', required=True)
    e.add_argument('--split', required=True); e.add_argument('--out', required=True)
    e.add_argument('--rounds', type=int, default=1); e.add_argument('--objective', default='AP_25_95')
    args = ap.parse_args()
    dict(build=cmd_build, train=cmd_train, eval=cmd_eval)[args.cmd](args)
