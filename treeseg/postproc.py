import contextlib
import io

import cv2
import numpy as np
from pycocotools.cocoeval import COCOeval
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

IMG_ROOT = None  # set by the caller: folder that contains <split>/images


def load_maps(pred_dir, stem):
    e = cv2.imread(f'{pred_dir}/{stem}_edge.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
    x = cv2.imread(f'{pred_dir}/{stem}_ext.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
    return e, x


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def make_markers(fg, edge, p):
    core = (fg & (edge < p['t_edge'])).astype(np.uint8)
    if p['r_open'] > 0:
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, disk(p['r_open']))
    if p.get('marker', 'cc') == 'dist':
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


def labels_from_maps(edge, ext, p):
    fg = ext > p['t_ext']
    markers = make_markers(fg, edge, p)
    elev = ndi.gaussian_filter(edge, p['sigma']) if p['sigma'] > 0 else edge
    return watershed(elev, markers, mask=fg), ext


def coco_eval(gt, dets, img_ids):
    if not dets:
        return dict(AP_25_95=0., AP50=0., AP70=0.)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, 'segm')
        E.params.imgIds = img_ids
        E.params.maxDets = [1, 10, 100]
        E.params.iouThrs = np.round(np.arange(0.25, 0.951, 0.05), 2)
        E.evaluate(); E.accumulate()
    thr = E.params.iouThrs
    prec = E.eval['precision']

    def ap(ts):
        p = prec[ts, :, :, 0, 2]
        p = p[p > -1]
        return float(p.mean()) if p.size else -1.

    t = lambda v: [int(np.argmin(np.abs(thr - v)))]
    return dict(AP_25_95=ap(list(range(len(thr)))), AP50=ap(t(0.5)), AP70=ap(t(0.7)))
