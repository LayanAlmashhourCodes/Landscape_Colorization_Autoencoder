"""Evaluate a trained model with PSNR and SSIM on held-out images.

Example:
    python -m src.evaluate --model models/colorizer.keras --samples 200
"""

import argparse
import os

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from . import config
from .colorize import colorize
from .data import load_split


def main():
    import tensorflow as tf

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=config.DEFAULT_MODEL_PATH)
    p.add_argument("--data-dir", default=config.DEFAULT_DATA_DIR)
    p.add_argument("--split", default=None, help="split.json written by train.py")
    p.add_argument("--samples", type=int, default=100)
    args = p.parse_args()

    split_path = args.split or os.path.join(os.path.dirname(args.model) or ".", "split.json")
    test_files = load_split(split_path)["test"]
    rng = np.random.default_rng(config.SEED)
    chosen = rng.choice(test_files, size=min(args.samples, len(test_files)), replace=False)

    model = tf.keras.models.load_model(args.model)
    p_scores, s_scores = [], []
    for name in chosen:
        _, pred, real = colorize(model, os.path.join(args.data_dir, name))
        p_scores.append(psnr(real, pred, data_range=1.0))
        s_scores.append(ssim(real, pred, data_range=1.0, win_size=3, channel_axis=2))

    print(f"images evaluated : {len(chosen)} (never seen in training)")
    print(f"average PSNR     : {np.mean(p_scores):.2f} dB")
    print(f"average SSIM     : {np.mean(s_scores):.4f}")


if __name__ == "__main__":
    main()
