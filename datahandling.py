import h5py
import numpy as np
import json

from collections import Counter
from dataclasses import dataclass, replace
from itertools import groupby
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

CHUNK_FRAMES = cfg["CHUNK_FRAMES"]

SESSIONS = ("1_0_2", "2_0_1", "2_0_2")


@dataclass(frozen=True)
class Recording:
    
    path: Path
    user: int
    eye: str
    session: str
    frame_ids: tuple[int, ...]  


def index_recordings(root):

    if not root.is_dir():

        raise FileNotFoundError(f"Set DATASET_PATH to the directory containing left/ and right/. Received {root}")

    records, missing = [], []

    for user in range(1, 49):

        for eye in ("left", "right"):

            for session in SESSIONS:

                path = root / eye / f"user{user}_session_{session}.h5"

                if not path.is_file():

                    missing.append(path)
                    continue

                with h5py.File(path, "r") as f:

                    if "data" not in f or "label" not in f:

                        raise ValueError(f"{path} must contain both 'data' and 'label'.")

                    x, y = f["data"], f["label"]

                    if x.ndim != 3 or x.shape[:2] != (W, H) or x.shape[2] == 0:

                        raise ValueError(f"{path}: expected ({W}, {H}, N>0), got {x.shape}.")

                    if y.shape != x.shape:

                        raise ValueError(f"{path}: data {x.shape} and label {y.shape} differ.")

                    if x.dtype.kind not in "buif" or y.dtype.kind not in "buif":

                        raise TypeError(f"{path}: expected numeric pixels and masks, got {x.dtype}, {y.dtype}.")

                    records.append(Recording(path, user, eye, session, tuple(range(x.shape[2]))))

    return records, missing


def split_frames(records, test_session, val_fraction, seed):
    
    if test_session not in SESSIONS or not 0 < val_fraction < 1:
    
        raise ValueError("Choose a listed TEST_SESSION and a VAL_FRACTION between 0 and 1.")
    
    test = [r for r in records if r.session == test_session]
    candidates = [r for r in records if r.session != test_session]
    n_pool = sum(len(r.frame_ids) for r in candidates)
    
    if not test or n_pool < 2:
    
        raise ValueError("The split needs at least one test recording and two non-test frames.")
    
    n_val = min(n_pool - 1, max(1, round(n_pool * val_fraction)))
    is_val = np.zeros(n_pool, dtype=bool)
    is_val[np.random.default_rng(seed).choice(n_pool, n_val, replace=False)] = True

    train, val = [], []
    offset = 0
    
    for r in candidates:
    
        end = offset + len(r.frame_ids)
        ids = np.asarray(r.frame_ids)
        chosen = is_val[offset:end]
    
        if not chosen.all():
    
            train.append(replace(r, frame_ids=tuple(ids[~chosen].tolist())))
    
        if chosen.any():
    
            val.append(replace(r, frame_ids=tuple(ids[chosen].tolist())))
    
        offset = end
    
    return train, val, test



def binary_mask(raw, pupil_value, binary_only, source):
    
    require_finite(raw, source)
    
    if binary_only and not np.all((raw == 0) | (raw == pupil_value)):
    
        values = np.unique(raw)[:12].tolist()
        raise ValueError(f"{source} contains {values}, inconsistent with training codes 0 and {pupil_value}.")
    
    return (raw == pupil_value).astype(np.float32)[..., None]


def require_finite(array, source):
    
    if not np.isfinite(array).all():
    
        raise ValueError(f"Non-finite values in {source}.")
    

def read_thw(dataset, start, stop):
    """Read a W,H,T slab as T,H,W, including a length-one time dimension."""
    
    return dataset[:, :, start:stop].transpose(2, 1, 0)


def frame_blocks(frame_ids):
    """Group selected indices into HDF5 reads spanning at most CHUNK_FRAMES frames."""
    
    for _, group in groupby(frame_ids, key=lambda t: t // CHUNK_FRAMES):
    
        yield np.fromiter(group, dtype=np.intp)


def read_selected_thw(dataset, ids):
    
    start, stop = int(ids[0]), int(ids[-1]) + 1
    block = read_thw(dataset, start, stop)
    
    return block if len(ids) == len(block) else block[ids - start]


def require_finite(array, source):
    
    if not np.isfinite(array).all():
    
        raise ValueError(f"Non-finite values in {source}.")


def inspect_training_masks(files, requested_value):
    
    counts = Counter()
    total, foreground = 0, 0
    preview_frames = []
    
    for r in files:
    
        preview_frame = None
    
        with h5py.File(r.path, "r") as f:
    
            for ids in frame_blocks(r.frame_ids):
    
                raw = read_selected_thw(f["label"], ids)
                require_finite(raw, f"{r.path} / label")
                total += raw.size
    
                if requested_value is None:
    
                    values, frequencies = np.unique(raw, return_counts=True)
                    counts.update(dict(zip(values.tolist(), frequencies.tolist())))
                    seen = set(counts)
    
                    if not (seen <= {0, 1} or seen <= {0, 255}):
    
                        raise ValueError(
                            f"Training labels include {sorted(seen)[:12]} (at {r.path.name}). "
                            "Set PUPIL_VALUE to the raw label that identifies pupil."
                        )
    
                else:
    
                    foreground += int(np.count_nonzero(raw == requested_value))
    
                if preview_frame is None:
    
                    pupil = raw != 0 if requested_value is None else raw == requested_value
                    present = np.flatnonzero(np.any(pupil, axis=(1, 2)))
    
                    if present.size:
    
                        preview_frame = int(ids[present[len(present) // 2]])
    
        if preview_frame is not None:
    
            preview_frames.append((r, preview_frame))
    
    if requested_value is None:
    
        print("Training pixels by raw label:", dict(sorted(counts.items())))
        pupil_value = max(counts)
        foreground = total - counts[0]
    
    else:
    
        pupil_value = requested_value
    
    if not 0 < foreground < total:
    
        raise ValueError(f"Mapping raw pupil value {pupil_value} gives {foreground}/{total} pupil pixels in training.")
    
    print(f"Raw pupil value {pupil_value} covers {100 * foreground / total:.2f}% of training pixels.")
    
    return pupil_value, preview_frames


def binary_mask(raw, pupil_value, binary_only, source):
    
    require_finite(raw, source)
    
    if binary_only and not np.all((raw == 0) | (raw == pupil_value)):
    
        values = np.unique(raw)[:12].tolist()
        raise ValueError(f"{source} contains {values}, inconsistent with training codes 0 and {pupil_value}.")
    
    return (raw == pupil_value).astype(np.float32)[..., None]



if __name__ == "__main__":

    print("all imports successful\n")