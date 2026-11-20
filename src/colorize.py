"""Colorize any image (or folder of images) with a trained model.

Example:
    python -m src.colorize --model models/colorizer.keras --input my_photo.jpg --output out/
"""

import argparse
import os

import numpy as np
from PIL import Image

from . import config
from .data import IMG_EXTS, load_rgb, model_io_to_rgb, rgb_to_model_io


def colorize(model, path, size=config.IMG_SIZE):
    """Return (grayscale, colorized, original) RGB arrays in [0, 1]."""
    rgb = load_rgb(path, size)              # works for color or grayscale files
    l_norm, _ = rgb_to_model_io(rgb)
    pred_ab = model.predict(l_norm[np.newaxis], verbose=0)[0]
    colored = model_io_to_rgb(l_norm, pred_ab)
    gray = np.repeat(l_norm, 3, axis=2)
    return gray, colored, rgb


def main():
    import tensorflow as tf

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=config.DEFAULT_MODEL_PATH)
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--output", default="outputs")
    args = p.parse_args()

    model = tf.keras.models.load_model(args.model)
    os.makedirs(args.output, exist_ok=True)
    paths = ([os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
              if f.lower().endswith(IMG_EXTS)] if os.path.isdir(args.input) else [args.input])

    for path in paths:
        gray, colored, _ = colorize(model, path)
        strip = np.concatenate([gray, colored], axis=1)
        name = os.path.splitext(os.path.basename(path))[0] + "_colorized.png"
        Image.fromarray((strip * 255).astype(np.uint8)).save(os.path.join(args.output, name))
        print(f"{path} -> {os.path.join(args.output, name)}")


if __name__ == "__main__":
    main()
