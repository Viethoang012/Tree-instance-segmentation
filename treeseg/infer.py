"""Sliding-window inference for EDTER Stage I / Stage II, and dumping of probability maps."""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data import BASE_SCALE, DATA_ROOT, load_img, normalize, read_list
from edter_model import EDTERGlobal, EDTERLocal


def windows(n, crop, stride):
    g = max(n - crop + stride - 1, 0) // stride + 1
    out = []
    for i in range(g):
        a = min(i * stride, n - crop)
        out.append(max(a, 0))
    return sorted(set(out))


FLIPS = [lambda t: t, lambda t: t.flip(-1), lambda t: t.flip(-2)]  # each is its own inverse
BATCH = int(os.environ.get('INFER_BATCH', 16))


@torch.no_grad()
def slide_infer(model, img, stage=1, crop=320, stride=192, flip_tta=False, batch=None):
    """img: (1,3,H,W) normalized. Returns dict of probability maps (1,1,H,W): edge, extent (stage 1 model).
    All windows (x flips) are run in batches of `batch` crops."""
    batch = batch or BATCH
    B, _, H, W = img.shape
    assert B == 1
    edge = img.new_zeros((1, 1, H, W))
    ext = img.new_zeros((1, 1, H, W))
    cnt = img.new_zeros((1, 1, H, W))
    gm = model.global_model if stage == 2 else model
    jobs = [(y, x, f) for y in windows(H, crop, stride) for x in windows(W, crop, stride)
            for f in (FLIPS if flip_tta else FLIPS[:1])]
    for k in range(0, len(jobs), batch):
        chunk = jobs[k:k + batch]
        ci = torch.cat([f(img[:, :, y:y + crop, x:x + crop]) for y, x, f in chunk])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            g = gm(ci, with_aux=False)
            if stage == 2:
                e = model(ci, global_features=g['global_features'], with_aux=False)['fuse_edge']
            else:
                e = g['edge']
        e = torch.sigmoid(e.float()); xx = torch.sigmoid(g['extent'].float())
        for j, (y, x, f) in enumerate(chunk):
            edge[:, :, y:y + crop, x:x + crop] += f(e[j:j + 1])
            ext[:, :, y:y + crop, x:x + crop] += f(xx[j:j + 1])
            cnt[:, :, y:y + crop, x:x + crop] += 1
    return dict(edge=edge / cnt, extent=ext / cnt)


def prep_image(split, stem, device, scale=BASE_SCALE):
    img = load_img(split, stem)
    H, W = img.shape[:2]
    img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return normalize(img)[None].to(device)


def multiscale_infer(model, split, stem, stage, scales, flip_tta):
    """Average edge/extent maps over several input scales, resampled to the BASE_SCALE grid."""
    acc = None
    for sc in scales:
        x = prep_image(split, stem, 'cuda', sc)
        r = slide_infer(model, x, stage, flip_tta=flip_tta)
        if acc is None:
            H = int(round(x.shape[2] / sc * BASE_SCALE)); W = int(round(x.shape[3] / sc * BASE_SCALE))
            acc = {k: 0. for k in r}
        for k, v in r.items():
            acc[k] = acc[k] + (F.interpolate(v, size=(H, W), mode='area' if sc > BASE_SCALE else 'bilinear') if v.shape[-2:] != (H, W) else v)
    return {k: v / len(scales) for k, v in acc.items()}


def build(stage, ckpt, ckpt_global=None, device='cuda'):
    g = EDTERGlobal(img_size=320)
    if stage == 1:
        g.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
        return g.to(device).eval()
    g.load_state_dict(torch.load(ckpt_global, map_location='cpu')['model'])
    m = EDTERLocal(g)
    missing, unexp = m.load_state_dict(torch.load(ckpt, map_location='cpu')['model'], strict=False)
    assert not unexp and all(k.startswith('global_model.') for k in missing), (missing, unexp)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, default=1)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ckpt_global', default=None)
    ap.add_argument('--split', default='Val')
    ap.add_argument('--list', default=None, help='list of stems (default: <split>/list.txt)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--scales', default='', help='comma list of input scales (e.g. 0.5,0.75,1.0); output stays at BASE_SCALE')
    args = ap.parse_args()
    stems = read_list(args.list or f'{DATA_ROOT}/{args.split}/list.txt')
    model = build(args.stage, args.ckpt, args.ckpt_global)
    os.makedirs(args.out, exist_ok=True)
    for i, s in enumerate(stems):
        if args.scales:
            r = multiscale_infer(model, args.split, s, args.stage, [float(v) for v in args.scales.split(',')], args.tta)
        else:
            r = slide_infer(model, prep_image(args.split, s, 'cuda'), args.stage, flip_tta=args.tta)
        e = (r['edge'][0, 0].cpu().numpy() * 65535).astype(np.uint16)
        x = (r['extent'][0, 0].cpu().numpy() * 65535).astype(np.uint16)
        cv2.imwrite(f'{args.out}/{s}_edge.png', e)
        cv2.imwrite(f'{args.out}/{s}_ext.png', x)
        if i % 100 == 0:
            print(i, len(stems), flush=True)


if __name__ == '__main__':
    main()
