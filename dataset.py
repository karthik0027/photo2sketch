from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pencil_sketch(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    return cv2.divide(gray, 255 - blurred, scale=256)


class PhotoSketchDataset(Dataset):

    def __init__(self, root: str, size: int = 256, augment: bool = True):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        bgr = cv2.imread(str(self.paths[idx]))
        bgr = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.augment and np.random.rand() < 0.5:
            bgr = cv2.flip(bgr, 1)

        sketch = pencil_sketch(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        photo = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1.0
        sketch = torch.from_numpy(sketch).unsqueeze(0).float() / 127.5 - 1.0
        return photo, sketch
