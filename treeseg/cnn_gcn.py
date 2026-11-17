"""CNN + GCN region merging, trained end-to-end.

For every graph edge (two adjacent over-segmentation regions i, j) a 64x64 patch is cropped around the union of the
two regions, with channels  RGB | EDTER edge | EDTER extent | mask_i | mask_j  (uint8, 7 x 64 x 64).
A small CNN embeds the patch (EMB dims); the embedding is appended to the usual edge attributes
(LPIPS, boundary edge stats [, rich features]) and fed to the GCN edge head (gcn_merge.GCNMerge).
CNN and GCN are trained jointly with the same merge labels (both regions share the majority GT crown).

Usage
  patches: python cnn_gcn.py patches --graphs graphs_s2_h05/fit --split Train
  train  : python cnn_gcn.py train --train graphs_s2_h05/fit --hold graphs_s2_h05/hold --out gcn_runs/cnn_s2h05 --edge_feats 1
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi

import gcn_merge as G
import postproc as P

PS = 64
CHANNELS = ['r', 'g', 'b', 'edge', 'ext', 'mask_i', 'mask_j']


# ----------------------------------------------------------------------------------------------- patches
def load_img(split, stem, hw):
    img = cv2.cvtColor(cv2.imread(f'{P.IMG_ROOT}/{split}/images/{stem}.jpg'), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, hw[::-1], interpolation=cv2.INTER_AREA)


def edge_patches(lab, edge, ext, img, ei, ej, pad=6):
    E = len(ei)
    out = np.zeros((E, len(CHANNELS), PS, PS), np.uint8)
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


def _patch_file_job(args):
    f, split, out = args
    g = dict(np.load(f))
    g['stem'] = os.path.basename(f)[:-4]
    np.save(f'{out}/{g["stem"]}.npy', _patch_job((g, split)))


def cmd_patches(args):
    from multiprocessing import Pool
    out = args.graphs.rstrip('/') + '_patches'
    os.makedirs(out, exist_ok=True)
    fs = sorted(glob.glob(f'{args.graphs}/*.npz')); t0 = time.time()
    with Pool(args.procs) as pool:
        pool.map(_patch_file_job, [(f, args.split, out) for f in fs], chunksize=4)
    print(f'{len(fs)} graphs -> {out} in {time.time() - t0:.0f}s', flush=True)


def load_graphs(d, edge_feats):
    gs = G.load_graphs(d, edge_feats)
    pd = d.rstrip('/') + '_patches'
    for g in gs:
        g['patches'] = np.load(f'{pd}/{g["stem"]}.npy')
    return gs


# ----------------------------------------------------------------------------------------------- model
class PatchCNN(nn.Module):
    def __init__(self, cin, emb=32, w=32):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(cin, w), blk(w, 2 * w), blk(2 * w, 3 * w), blk(3 * w, 4 * w))
        self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(4 * w, emb))

    def forward(self, x):
        return self.out(self.f(x).mean((2, 3)))


class R18MaskPool(nn.Module):
    """ImageNet ResNet18 (conv1..layer3, stride 16) on the RGB patch upsampled to 128x128; features are averaged
    inside mask_i and mask_j (masks are NOT fed to the backbone) -> MLP([f_i, f_j, |f_i-f_j|, f_i*f_j]) -> emb."""
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, emb=32, ft='none', pretrained=True):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.ft = ft
        for n, prm in self.body.named_parameters():
            prm.requires_grad = (ft == 'layer3' and n.startswith('6.')) or ft == 'all'
        self.register_buffer('mean', torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.STD).view(1, 3, 1, 1))
        self.out = nn.Sequential(nn.Linear(4 * 256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, emb))

    def train(self, mode=True):
        super().train(mode)
        for m in self.body.modules():  # frozen BN statistics (ImageNet) always
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, p):
        rgb = F.interpolate(p[:, :3].float().div(255.), size=128, mode='bilinear', align_corners=False)
        f = self.body((rgb - self.mean) / self.std)                       # (E, 256, 8, 8)
        m = F.adaptive_avg_pool2d(p[:, 5:7].float().div(255.), f.shape[-2:])  # (E, 2, 8, 8) soft masks
        den = m.sum((2, 3)).clamp_min(1e-3)
        fi = (f * m[:, :1]).sum((2, 3)) / den[:, :1]
        fj = (f * m[:, 1:]).sum((2, 3)) / den[:, 1:]
        return self.out(torch.cat([fi, fj, (fi - fj).abs(), fi * fj], 1))


class CNNGCN(nn.Module):
    def __init__(self, n_node, n_edge, chans, hid=64, layers=2, emb=32, w=32, backbone='small', ft='none'):
        super().__init__()
        self.chans = list(chans)
        self.backbone = backbone
        self.cnn = PatchCNN(len(self.chans), emb, w) if backbone == 'small' else R18MaskPool(emb, ft, pretrained=False)
        self.gcn = G.GCNMerge(n_node, n_edge + emb, hid, layers)

    def forward(self, x, ei, ej, eattr, patches):
        if self.backbone == 'small':
            z = self.cnn(patches[:, self.chans].float().div(255.))
        else:
            z = self.cnn(patches)
        return self.gcn(x, ei, ej, torch.cat([eattr, z], 1))


def augment(p):
    """Random flips / 90-degree rotations of the whole patch batch (geometry-only, keeps masks aligned)."""
    if np.random.rand() < .5:
        p = p.flip(3)
    if np.random.rand() < .5:
        p = p.flip(2)
    return torch.rot90(p, int(np.random.randint(4)), (2, 3))


def forward_graph(model, norm, g, dev, aug=False):
    x, ei, ej, ea = norm(g, dev)
    pt = torch.from_numpy(g['patches']).to(dev)
    if aug:
        pt = augment(pt)
    return model(x, ei, ej, ea, pt)


@torch.no_grad()
def predict_all(model, norm, graphs, dev):
    model.eval()
    return [torch.sigmoid(forward_graph(model, norm, g, dev)).cpu().numpy() if len(g['ei']) else np.zeros(0) for g in graphs]


def build_model(ck):
    norm = G.Norm.__new__(G.Norm); norm.__dict__.update(ck['norm'])
    model = CNNGCN(G.n_node(norm), G.n_edge(ck['edge_feats']), ck['chans'], ck['hid'], ck['layers'], ck['emb'], ck['width'],
                   ck.get('backbone', 'small'), ck.get('ft', 'none'))
    model.load_state_dict(ck['model'])
    return model, norm


# ----------------------------------------------------------------------------------------------- training
def cmd_train(args):
    from pycocotools.coco import COCO
    dev = 'cuda'
    torch.manual_seed(0); np.random.seed(0)
    tr = [g for g in load_graphs(args.train, args.edge_feats) if len(g['ei'])]
    ho = load_graphs(args.hold, args.edge_feats)
    norm = G.Norm(tr)
    chans = [CHANNELS.index(c) for c in args.chans.split(',')]
    model = CNNGCN(len(norm.cols), tr[0]['eattr'].shape[1], chans, args.hid, args.layers, args.emb, args.width,
                   args.backbone, args.ft)
    if args.backbone == 'r18':
        model.cnn = R18MaskPool(args.emb, args.ft, pretrained=True)
    model = model.to(dev)
    bb = [q for n, q in model.named_parameters() if n.startswith('cnn.body.') and q.requires_grad]
    rest = [q for n, q in model.named_parameters() if not n.startswith('cnn.body.') and q.requires_grad]
    groups = [dict(params=rest, lr=args.lr)] + ([dict(params=bb, lr=args.lr * args.bb_lr)] if bb else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = args.epochs * ((len(tr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g['lr'] for g in groups], total_steps=steps, pct_start=0.1)
    print(f'trainable params: head/gcn {sum(q.numel() for q in rest)}, backbone {sum(q.numel() for q in bb)}', flush=True)
    pos = np.concatenate([g['y'] for g in tr]).mean()
    pw = torch.tensor((1 - pos) / pos, device=dev)
    print(f'train graphs {len(tr)} edges {sum(len(g["ei"]) for g in tr)} pos {pos:.3f}; hold {len(ho)}; chans {args.chans}', flush=True)
    gt_tr = COCO(P.ANN.format('Train'))
    p = json.load(open(f'{args.hold}/params.json'))
    ts = [float(v) for v in args.ts.split(',')]
    os.makedirs(args.out, exist_ok=True)
    best = (-1, None, None)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(len(tr)); tot = 0.
        for k in range(0, len(perm), args.bs):
            opt.zero_grad(); loss = 0.
            for i in perm[k:k + args.bs]:
                g = tr[i]
                logit = forward_graph(model, norm, g, dev, aug=True)
                loss = loss + F.binary_cross_entropy_with_logits(logit, torch.from_numpy(g['y']).to(dev), pos_weight=pw)
            (loss / len(perm[k:k + args.bs])).backward(); opt.step(); sched.step(); tot += loss.item()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            probs = predict_all(model, norm, ho, dev)
            yh = np.concatenate([g['y'] for g in ho]); ph = np.concatenate(probs)
            acc = ((ph > 0.5) == (yh > 0.5)).mean()
            res = {t: G.eval_graphs(gt_tr, ho, probs, t, p)[0]['AP_25_95'] for t in ts}
            tb = max(res, key=res.get)
            print(f'ep {ep + 1} loss {tot / len(tr):.4f} hold edge-acc {acc:.3f} hold AP_25_95: ' +
                  ' '.join(f'{t}:{v * 100:.2f}' for t, v in res.items()) + f'  ({time.time() - t0:.0f}s)', flush=True)
            if res[tb] > best[0]:
                best = (res[tb], tb, ep + 1)
                torch.save(dict(model=model.state_dict(), norm=vars(norm), hid=args.hid, layers=args.layers, emb=args.emb,
                                width=args.width, chans=chans, edge_feats=args.edge_feats, t=tb, params=p, epoch=ep + 1,
                                cnn=True, backbone=args.backbone, ft=args.ft), f'{args.out}/model.pt')
    print(f'best hold AP_25_95 {best[0] * 100:.2f} at t={best[1]} epoch {best[2]}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('patches'); a.add_argument('--graphs', required=True); a.add_argument('--split', default='Train')
    a.add_argument('--procs', type=int, default=24)
    t = sub.add_parser('train'); t.add_argument('--train', required=True); t.add_argument('--hold', required=True)
    t.add_argument('--out', required=True); t.add_argument('--epochs', type=int, default=20); t.add_argument('--bs', type=int, default=8)
    t.add_argument('--lr', type=float, default=2e-3); t.add_argument('--hid', type=int, default=64)
    t.add_argument('--layers', type=int, default=2); t.add_argument('--emb', type=int, default=32)
    t.add_argument('--width', type=int, default=32); t.add_argument('--edge_feats', type=int, default=1)
    t.add_argument('--chans', default=','.join(CHANNELS)); t.add_argument('--eval_every', type=int, default=2)
    t.add_argument('--ts', default='0.6,0.7,0.8,0.85,0.9')
    t.add_argument('--backbone', default='small', choices=['small', 'r18'])
    t.add_argument('--ft', default='none', choices=['none', 'layer3', 'all'], help='r18: which backbone layers to fine-tune')
    t.add_argument('--bb_lr', type=float, default=0.1, help='backbone lr = lr * bb_lr')
    args = ap.parse_args()
    dict(patches=cmd_patches, train=cmd_train)[args.cmd](args)
