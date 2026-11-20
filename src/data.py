"""Data loading: RGB image -> CIELAB -> (L input, ab target)."""

import json
import os

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

from .config import AB_SCALE, IMG_SIZE, L_SCALE, SEED

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_rgb(path, size=IMG_SIZE):
    """Load an image as float RGB in [0, 1], resized to (size, size)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def rgb_to_model_io(rgb):
    """Split an RGB image into the normalized L input and ab target."""
    lab = rgb2lab(rgb)
    l_norm = (lab[:, :, 0] / L_SCALE)[..., np.newaxis]   # (H, W, 1) in [0, 1]
    ab_norm = lab[:, :, 1:] / AB_SCALE                   # (H, W, 2) in ~[-1, 1]
    return l_norm.astype(np.float32), ab_norm.astype(np.float32)


def model_io_to_rgb(l_norm, ab_norm):
    """Recombine the L channel with (predicted) ab channels into RGB [0, 1]."""
    lab = np.zeros(l_norm.shape[:2] + (3,), dtype=np.float64)
    lab[:, :, 0] = np.squeeze(l_norm) * L_SCALE
    lab[:, :, 1:] = ab_norm * AB_SCALE
    return np.clip(lab2rgb(lab), 0.0, 1.0)


def list_images(data_dir):
    return sorted(f for f in os.listdir(data_dir) if f.lower().endswith(IMG_EXTS))


def make_split(data_dir, n_train, seed=SEED, split_file=None):
    """Shuffle file names once and split them into train / test sets.

    The split is saved to JSON so evaluation always uses images the model
    has never seen during training.
    """
    files = list_images(data_dir)
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    split = {"train": files[:n_train], "test": files[n_train:]}
    if split_file:
        os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)
        with open(split_file, "w") as f:
            json.dump(split, f, indent=1)
    return split


def load_split(split_file):
    with open(split_file) as f:
        return json.load(f)


def load_arrays(data_dir, filenames, verbose=True):
    """Load a list of files into X (L) and Y (ab) arrays."""
    X, Y = [], []
    for i, name in enumerate(filenames, 1):
        try:
            l_norm, ab_norm = rgb_to_model_io(load_rgb(os.path.join(data_dir, name)))
        except Exception as exc:  # skip unreadable files
            print(f"  skipping {name}: {exc}")
            continue
        X.append(l_norm)
        Y.append(ab_norm)
        if verbose and i % 500 == 0:
            print(f"  processed {i}/{len(filenames)} images")
    return np.stack(X), np.stack(Y)
