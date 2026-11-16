import os
import h5py
import json

import datahandling as DH
import tensorflow as tf
import numpy as np

from pathlib import Path
from tensorflow import keras


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


def image_batch(raw, path):
    
    images = raw.astype(np.float32, copy=False)
    DH.require_finite(images, f"{path} / data")
    
    return images[..., None]


def recording_chunks(recording, pupil_value, binary_only):
    
    with h5py.File(recording.path, "r") as f:
    
        for ids in DH.frame_blocks(recording.frame_ids):
    
            images = image_batch(DH.read_selected_thw(f["data"], ids), recording.path)
            raw = DH.read_selected_thw(f["label"], ids)
            yield images, DH.binary_mask(raw, pupil_value, binary_only, recording.path)


def flip_pair(image, mask):
    
    pair = tf.concat([image, mask], axis=-1)
    pair = tf.image.random_flip_left_right(pair, seed=SEED + 2)
    
    return pair[..., :1], pair[..., 1:]


def make_dataset(files, pupil_value, binary_only, training=False, AUGMENT = False, BATCH_SIZE = 4):
    
    selected = {str(r.path): r for r in files}
    paths = tf.data.Dataset.from_tensor_slices(list(selected))
    n_frames = sum(len(r.frame_ids) for r in files)
    block = tf.TensorSpec((None, H, W, 1), tf.float32)
    
    if training:
    
        paths = paths.shuffle(len(files), seed=SEED, reshuffle_each_iteration=True)

    
    def chunks_for_path(filename):
    
        path = os.fsdecode(np.asarray(filename).item())
        return recording_chunks(selected[path], pupil_value, binary_only)

    
    def one_file(path):
    
        return tf.data.Dataset.from_generator(
            chunks_for_path,
            args=(path,), output_signature=(block, block),
        ).unbatch()

    open_files = min(4, len(files)) if training else 1
    frames = paths.interleave(one_file, cycle_length=open_files, block_length=1,
                              num_parallel_calls=open_files, deterministic=True)
    frames = frames.apply(tf.data.experimental.assert_cardinality(n_frames))
    
    if training:
    
        frames = frames.shuffle(min(64, n_frames), seed=SEED + 1, reshuffle_each_iteration=True)
    
        if AUGMENT:
    
            frames = frames.map(flip_pair, num_parallel_calls=1)
    
    return frames.batch(BATCH_SIZE, drop_remainder=False).prefetch(1)