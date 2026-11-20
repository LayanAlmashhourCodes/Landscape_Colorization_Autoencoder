"""Convolutional autoencoder architectures.

Encoder: 4 strided conv layers  128 -> 64 -> 32 -> 16 -> 8  (64/128/256/512 filters)
Decoder: 4 transposed convs     8 -> 16 -> 32 -> 64 -> 128 (256/128/64/64 filters)
Output : 3x3 conv, 2 channels (a, b), tanh  -> values in [-1, 1]
"""

from tensorflow.keras import layers, models

ENC_FILTERS = (64, 128, 256, 512)
DEC_FILTERS = (256, 128, 64, 64)


def build_autoencoder(img_size=128, batch_norm=True):
    """Build the colorization autoencoder.

    batch_norm=False reproduces the first (baseline) model.
    batch_norm=True  is the final model: BatchNormalization after every
    conv layer, which fixed the "gray output" collapse.
    """
    inp = layers.Input(shape=(img_size, img_size, 1), name="L_channel")
    x = inp

    for f in ENC_FILTERS:                       # Encoder (feature extraction)
        x = layers.Conv2D(f, 3, strides=2, padding="same", activation="relu")(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)

    for f in DEC_FILTERS:                       # Decoder (reconstruction)
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same", activation="relu")(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)

    out = layers.Conv2D(2, 3, padding="same", activation="tanh", name="ab_channels")(x)
    name = "colorizer_bn" if batch_norm else "colorizer_baseline"
    return models.Model(inp, out, name=name)
