import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

import postproc as P

PATCH = 64
NODE_FEATS = ['log_area', 'extent', 'solidity', 'aspect', 'deviation']


def region_patch(img, m, sl):
    crop = img[sl].copy()
    mm = m[sl]
    crop[~mm] = crop[mm].mean(0) if mm.any() else 0
    return cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA)


def build_graph(stem, split, pred_dir, p):
    """Region adjacency graph of the watershed over-segmentation of one image."""
    edge, ext = P.load_maps(pred_dir, stem)
    lab, ext = P.labels_from_maps(edge, ext, p)
    img = cv2.cvtColor(cv2.imread(f'{P.IMG_ROOT}/{split}/images/{stem}.jpg'), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, edge.shape[::-1], interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    n = int(lab.max()) + 1
    feats = np.zeros((n, 5), np.float32)
    patches = np.zeros((n, PATCH, PATCH, 3), np.uint8)
    for k, sl in enumerate(ndi.find_objects(lab), 1):
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
    return dict(stem=stem, lab=lab.astype(np.int32), edge=edge, ext=ext, x=feats, patches=patches,
                ei=ei, ej=ej, bfeat=bfeat)


def n_edge(edge_feats):
    return 1 + 3 * (edge_feats >= 1)


def edge_attr(g, edge_feats):
    if not len(g['ei']):
        return np.zeros((0, n_edge(edge_feats)), np.float32)
    ea = [g['lpips'][:, None]]
    if edge_feats >= 1:
        ea.append(g['bfeat'])
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


class GCNMerge(nn.Module):
    def __init__(self, n_node=5, n_edge=1, hid=64, layers=2, tau=0.3):
        super().__init__()
        self.inp = nn.Linear(n_node, hid)
        self.layers = nn.ModuleList([nn.Linear(hid, hid) for _ in range(layers)])
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau))))
        self.head = nn.Sequential(nn.Linear(3 * hid + n_edge, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, x, ei, ej, eattr):
        n = x.shape[0]
        w = torch.exp(-eattr[:, 0] / self.log_tau.exp())
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


class Norm:
    def __call__(self, g, dev):
        cols = getattr(self, 'cols', None)
        x = g['x'] if cols is None else g['x'][:, cols]
        return (torch.from_numpy((x - self.xm) / self.xs).to(dev), torch.from_numpy(g['ei']).to(dev),
                torch.from_numpy(g['ej']).to(dev), torch.from_numpy((g['eattr'] - self.em) / self.es).to(dev))


def n_node(norm):
    return len(getattr(norm, 'cols', None) or NODE_FEATS)


def merge_labels(g, prob, t):
    """Union-find over edges with merge probability > t; returns the root id of every region."""
    n = g['x'].shape[0]
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, j in zip(g['ei'][prob > t], g['ej'][prob > t]):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(a) for a in range(n)])
