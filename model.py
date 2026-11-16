import json

from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


def load_config(config_path: str = "config.json") -> dict:

    path = Path(config_path)

    if not path.exists():

        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)


cfg = load_config()
SEED = cfg["SEED"]
H = cfg["H"]
W = cfg["W"]

keras.utils.set_random_seed(SEED)


BASE_FILTERS = 8


def residual_block(x, filters):
    
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
    
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
    
    return layers.Activation("relu")(layers.Add()([x, shortcut]))


def gated_skip(skip, decoder, filters):
    
    width = max(1, filters // 2)
    skip_key = layers.Conv2D(width, 1)(skip)
    decoder_key = layers.Conv2D(width, 1)(decoder)
    match = layers.Activation("relu")(layers.Add()([skip_key, decoder_key]))
    gate = layers.Conv2D(1, 1, activation="sigmoid")(match)
    
    return layers.Multiply()([skip, gate])


def build_model(height=H, width=W, base=BASE_FILTERS):
    
    image = keras.Input((height, width, 1), dtype="float32", name="raw_frame")
    low = keras.ops.min(image, axis=(1, 2), keepdims=True)
    high = keras.ops.max(image, axis=(1, 2), keepdims=True)
    x = (image - low) / keras.ops.maximum(high - low, 1e-7)

    widths = [base, 2 * base, 4 * base, 8 * base]

    # try next
    #widths = [4, 8, 16, 24]

    # try next
    widths = [4, 8, 12, 18]

    # try next 
    #widths = [4, 8, 12]

    #widths = [2, 4, 8]

    stride = 2 ** len(widths)
    dh, dw = (-height) % stride, (-width) % stride
    padding = ((dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2))
    x = layers.ZeroPadding2D(padding, name="pad_to_stride")(x)
    skips = []
    
    for filters in widths:
    
        x = residual_block(x, filters)
        skips.append(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = residual_block(x, 2 * widths[-1])
    x = layers.SpatialDropout2D(0.2)(x)
    
    for filters, skip in zip(reversed(widths), reversed(skips)):
    
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        
        #x = layers.Concatenate()([x, gated_skip(skip, x, filters)])
        x = layers.Add()([x, skip])
        
        x = residual_block(x, filters)

    x = layers.Cropping2D(padding, name="original_grid")(x)
    probability = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="pupil_probability")(x)
    
    return keras.Model(image, probability, name="pupil_attention_unet")


if __name__ == "__main__":

    print("imports succesful\n")
    