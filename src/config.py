"""Shared settings for the colorization pipeline."""

IMG_SIZE = 128          # images are resized to IMG_SIZE x IMG_SIZE
L_SCALE = 100.0         # L channel in [0, 100]  -> divide to get [0, 1]
AB_SCALE = 128.0        # a/b channels in ~[-128, 127] -> divide to get ~[-1, 1]

TRAIN_IMAGES = 5000     # number of images used for training (incl. validation)
VAL_SPLIT = 0.1
BATCH_SIZE = 64
EPOCHS = 50
SEED = 42

DEFAULT_DATA_DIR = "data/landscape Images/color"
DEFAULT_MODEL_PATH = "models/colorizer.keras"
