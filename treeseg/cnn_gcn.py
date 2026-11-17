import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy import ndimage as ndi

import gcn_merge as G
import postproc as P

PS = 64


def load_img(split, stem, hw):
    img = cv2.cvtColor(cv2.imread(f'{P.IMG_ROOT}/{split}/images/{stem}.jpg'), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, hw[::-1], interpolation=cv2.INTER_AREA)


def edge_patches(lab, edge, ext, img, ei, ej, pad=6):
    """One 7-channel patch (RGB, edge, extent, mask_i, mask_j) around every pair of adjacent regions."""
    E = len(ei)
    out = np.zeros((E, 7, PS, PS), np.uint8)
    if E == 0:
        return out
    H, W = lab.shape
    stack = np.concatenate([img, (edge[..., None] * 255).astype(np.uint8), (ext[..., None] * 255).astype(np.uint8)], 2)
    objs = ndi.find_objects(lab)
    for e in range(E):
        i, j = int(ei[e]), int(ej[e])
        si, sj = objs[i - 1], objs[j - 1]
        y0, y1 = min(si[0].start, sj[0].start), max(si[0].stop, sj[0].stop)
        x0, x1 = min(si[1].start, sj[1].start), max(si[1].stop, sj[1].stop)
        side = max(y1 - y0, x1 - x0) + 2 * pad
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        Y0, X0 = cy - side // 2, cx - side // 2
        Y1, X1 = Y0 + side, X0 + side
        crop = np.zeros((side, side, 7), np.uint8)
        ya, yb, xa, xb = max(Y0, 0), min(Y1, H), max(X0, 0), min(X1, W)
        crop[ya - Y0:yb - Y0, xa - X0:xb - X0, :5] = stack[ya:yb, xa:xb]
        l = lab[ya:yb, xa:xb]
        crop[ya - Y0:yb - Y0, xa - X0:xb - X0, 5] = (l == i) * 255
        crop[ya - Y0:yb - Y0, xa - X0:xb - X0, 6] = (l == j) * 255
        r = cv2.resize(crop[..., :4], (PS, PS), interpolation=cv2.INTER_AREA)
        r2 = cv2.resize(crop[..., 4:], (PS, PS), interpolation=cv2.INTER_AREA)
        out[e] = np.concatenate([r, r2], 2).transpose(2, 0, 1)
    return out


def _patch_job(args):
    g, split = args
    img = load_img(split, g['stem'], g['lab'].shape)
    return edge_patches(g['lab'], g['edge'], g['ext'], img, g['ei'], g['ej'])


def add_patches(graphs, split, pool):
    for g, pt in zip(graphs, pool.map(_patch_job, [({k: g[k] for k in ('stem', 'lab', 'edge', 'ext', 'ei', 'ej')}, split)
                                                     for g in graphs], chunksize=4)):
        g['patches'] = pt


class R18MaskPool(nn.Module):
    """ResNet18 features of the RGB patch, pooled inside each of the two region masks."""
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, emb=32):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.register_buffer('mean', torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.STD).view(1, 3, 1, 1))
        self.out = nn.Sequential(nn.Linear(4 * 256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, emb))

    def forward(self, p):
        rgb = F.interpolate(p[:, :3].float().div(255.), size=128, mode='bilinear', align_corners=False)
        f = self.body((rgb - self.mean) / self.std)
        m = F.adaptive_avg_pool2d(p[:, 5:7].float().div(255.), f.shape[-2:])
        den = m.sum((2, 3)).clamp_min(1e-3)
        fi = (f * m[:, :1]).sum((2, 3)) / den[:, :1]
        fj = (f * m[:, 1:]).sum((2, 3)) / den[:, 1:]
        return self.out(torch.cat([fi, fj, (fi - fj).abs(), fi * fj], 1))


class CNNGCN(nn.Module):
    def __init__(self, n_node, n_edge, hid=64, layers=2, emb=32):
        super().__init__()
        self.cnn = R18MaskPool(emb)
        self.gcn = G.GCNMerge(n_node, n_edge + emb, hid, layers)

    def forward(self, x, ei, ej, eattr, patches):
        return self.gcn(x, ei, ej, torch.cat([eattr, self.cnn(patches)], 1))


@torch.no_grad()
def predict_all(model, norm, graphs, dev):
    """Merge probability of every edge of every graph."""
    model.eval()
    out = []
    for g in graphs:
        if not len(g['ei']):
            out.append(np.zeros(0)); continue
        x, ei, ej, ea = norm(g, dev)
        out.append(torch.sigmoid(model(x, ei, ej, ea, torch.from_numpy(g['patches']).to(dev))).cpu().numpy())
    return out


def build_model(ck):
    norm = G.Norm.__new__(G.Norm); norm.__dict__.update(ck['norm'])
    model = CNNGCN(G.n_node(norm), G.n_edge(ck['edge_feats']), ck['hid'], ck['layers'], ck['emb'])
    model.load_state_dict(ck['model'])
    return model, norm
