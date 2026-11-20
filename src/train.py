"""Train the colorization autoencoder.

Example:
    python -m src.train --data-dir "data/landscape Images/color" --epochs 50
"""

import argparse
import json
import os

from . import config
from .data import load_arrays, make_split
from .model import build_autoencoder


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=config.DEFAULT_DATA_DIR)
    p.add_argument("--n-train", type=int, default=config.TRAIN_IMAGES)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--no-batch-norm", action="store_true", help="train the baseline model")
    p.add_argument("--out", default=config.DEFAULT_MODEL_PATH)
    args = p.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    split = make_split(args.data_dir, args.n_train, split_file=os.path.join(out_dir, "split.json"))
    print(f"train images: {len(split['train'])} | held-out test images: {len(split['test'])}")

    X, Y = load_arrays(args.data_dir, split["train"])
    print(f"X: {X.shape}  Y: {Y.shape}")

    model = build_autoencoder(config.IMG_SIZE, batch_norm=not args.no_batch_norm)
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    history = model.fit(
        X, Y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=config.VAL_SPLIT,
    )

    model.save(args.out)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"saved model -> {args.out}")


if __name__ == "__main__":
    main()
