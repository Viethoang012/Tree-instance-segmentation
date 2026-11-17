import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_ROOT = '/home/hoangdmv/Computer_vision/TreeCog/ForestSeg_subset/ForestSeg-T1'
DATA_ROOT = '/home/hoangdmv/Computer_vision/TreeCog/edter_forest/data'
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # EDTER img_norm_cfg (ImageNet)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BASE_SCALE = 0.5  # 1024x1024 tiles are processed at 512x512


def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


def load_img(split, stem):
    return cv2.cvtColor(cv2.imread(f'{IMG_ROOT}/{split}/images/{stem}.jpg'), cv2.COLOR_BGR2RGB)


def load_inst(split, stem):
    return cv2.imread(f'{DATA_ROOT}/{split}/inst/{stem}.png', cv2.IMREAD_UNCHANGED).astype(np.int32)


def inst_to_edge(inst):
    """Boundary map: pixel whose 4-neighbour carries a different instance id (tree/tree and tree/bg)."""
    e = np.zeros(inst.shape, bool)
    dv = inst[1:] != inst[:-1]
    dh = inst[:, 1:] != inst[:, :-1]
    e[1:] |= dv
    e[:-1] |= dv
    e[:, 1:] |= dh
    e[:, :-1] |= dh
    return e


def normalize(img):
    x = (img.astype(np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class ForestEdgeTrain(Dataset):
    def __init__(self, stems, split='Train', crop=320, scale_range=(0.4, 0.65), length=None):
        self.stems, self.split, self.crop, self.scale_range = stems, split, crop, scale_range
        self.length = length or len(stems)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random
        stem = self.stems[i % len(self.stems)]
        img, inst = load_img(self.split, stem), load_inst(self.split, stem)
        s = rng.uniform(*self.scale_range)
        H, W = img.shape[:2]
        h, w = max(self.crop, int(round(H * s))), max(self.crop, int(round(W * s)))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        inst = cv2.resize(inst.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        y, x = rng.randint(0, h - self.crop + 1), rng.randint(0, w - self.crop + 1)
        img, inst = img[y:y + self.crop, x:x + self.crop], inst[y:y + self.crop, x:x + self.crop]
        if rng.rand() < 0.5:
            img, inst = img[:, ::-1], inst[:, ::-1]
        if rng.rand() < 0.5:
            img, inst = img[::-1], inst[::-1]
        k = rng.randint(4)
        img, inst = np.rot90(img, k), np.rot90(inst, k)
        # mild photometric jitter (aerial imagery: brightness/contrast/saturation)
        img = img.astype(np.float32)
        img = img * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
        gray = img.mean(2, keepdims=True)
        img = gray + (img - gray) * rng.uniform(0.8, 1.2)
        img = np.clip(img, 0, 255)
        edge = inst_to_edge(inst)
        ext = inst > 0
        return (normalize(img), torch.from_numpy(edge[None].astype(np.float32)),
                torch.from_numpy(ext[None].astype(np.float32)))
