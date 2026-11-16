import json
import csv
import datetime

import losses
import make_datasets

import tensorflow as tf
import datahandling as DH
import model as network
import numpy as np

from tensorflow import keras
from datetime import datetime, timezone
from pathlib import Path


def load_config(config_path: str = "config.json") -> dict:

    path = Path(config_path)

    if not path.exists():

        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)


cfg = load_config()


SEED = cfg["SEED"]

# Dataset configuration
DATASET_PATH = Path(cfg["DATASET_PATH"])
TEST_SESSION = cfg["TEST_SESSION"]
H = cfg["H"]
W = cfg["W"]
PUPIL_VALUE = cfg["PUPIL_VALUE"]  
CHUNK_FRAMES = cfg["CHUNK_FRAMES"]

# Training configuration
REQUIRE_GPU = cfg["REQUIRE_GPU"]
logical_gpus = tf.config.list_logical_devices("GPU")
TRAIN_DEVICE = logical_gpus[0].name if logical_gpus else "/CPU:0"
probe_on_gpu = cfg["PROBE_ON_GPU"]

VAL_FRACTION = cfg["VAL_FRACTION"]
BATCH_SIZE = cfg["BATCH_SIZE"]
EPOCHS = cfg["EPOCHS"]
BASE_FILTERS = cfg["BASE_FILTERS"]
LEARNING_RATE = cfg["LEARNING_RATE"]
THRESHOLD = cfg["THRESHOLD"]
AUGMENT = cfg["AUGMENT"]

# Output dir
OUTPUT_ROOT = Path(cfg["OUTPUT_ROOT"])


def training_pipeline():

    records, missing = DH.index_recordings(DATASET_PATH)
    
    sanity_checks(missing)

    train_files, val_files, test_files, splits = get_datafiles(records)

        
    pupil_id, _ = DH.inspect_training_masks(train_files, PUPIL_VALUE)
    binary_only = PUPIL_VALUE is None

    train_ds = make_datasets.make_dataset(train_files, pupil_id, binary_only, training=True)
    val_ds = make_datasets.make_dataset(val_files, pupil_id, binary_only)

    model = get_model(val_ds)

    run_dir, checkpoint = init_run_dir(pupil_id, binary_only, splits)
    monitor = "val_pupil_iou"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(checkpoint), monitor=monitor, mode="max", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="max", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=10),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    test(checkpoint, test_files, pupil_id, binary_only, run_dir)

    test_per_user(run_dir, test_files, pupil_id, binary_only)

    return


def sanity_checks(missing):
   
    if missing:
    
        print(f"Missing {len(missing)} of 288 expected files. First missing paths:")

        for path in missing[:5]:
    
            print(" ", path.relative_to(DATASET_PATH))
    return 


def get_datafiles(records):
    
    train_files, val_files, test_files = DH.split_frames(records, TEST_SESSION, VAL_FRACTION, SEED)
    splits = {"train": train_files, "val": val_files, "test": test_files}
    
    print(f"{'Split':<8}{'Users':>7}{'Files':>8}{'Frames':>10}")
    
    for name, files in splits.items():
    
        print(f"{name:<8}{len({r.user for r in files}):>7}{len(files):>8}{sum(len(r.frame_ids) for r in files):>10}")
    
    print(f"Validation sampling: {VAL_FRACTION:.0%} of non-test frames, seed {SEED}.")
    print("Held-out session:", TEST_SESSION)
    
    return train_files, val_files, test_files, splits


def get_model(val_ds):

    with tf.device(TRAIN_DEVICE):
        
        model = network.build_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=losses.bce_dice_loss,
            metrics=[keras.metrics.BinaryIoU(target_class_ids=[1], threshold=THRESHOLD, name="pupil_iou")],
            jit_compile=False,
        )

    # Exercise the input, forward pass, and loss on one real frame before fitting.
    probe_x, probe_y = next(iter(val_ds))
    probe_prob = model(probe_x[:1], training=False)

    tf.debugging.assert_equal(tf.shape(probe_prob), tf.shape(probe_y[:1]))
    tf.debugging.assert_all_finite(losses.bce_dice_loss(probe_y[:1], probe_prob), "Non-finite probe loss.")

    print(f"{model.name} · {model.count_params():,} parameters · output per frame {tuple(probe_prob.shape[1:])}")
    print("Model output device:", probe_prob.device)

    return model


def init_run_dir(pupil_id, binary_only, splits):

    run_dir = OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "best.keras"
    print("Run directory:", run_dir.resolve())

    with (run_dir / "split.csv").open("w", newline="") as out:
        
        writer = csv.writer(out)
        writer.writerow(["split", "user", "eye", "session", "relative_path", "frame_index"])
        
        for split, files in splits.items():
        
            for r in files:
        
                relative_path = r.path.relative_to(DATASET_PATH)
                writer.writerows((split, r.user, r.eye, r.session, relative_path, t) for t in r.frame_ids)


    settings = dict(
        test_session=TEST_SESSION, validation_fraction=VAL_FRACTION, split_unit="frame",
        pupil_value=float(pupil_id), automatic_binary_encoding=binary_only, threshold=THRESHOLD,
        height=H, width=W, seed=SEED, batch_size=BATCH_SIZE, epochs=EPOCHS,
        base_filters=BASE_FILTERS, learning_rate=LEARNING_RATE, horizontal_flip=AUGMENT,
        tensorflow_version=tf.__version__, training_device=TRAIN_DEVICE, gpu_probe_passed=probe_on_gpu,
    )
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    return run_dir, checkpoint


def test(checkpoint, test_files, pupil_id, binary_only, run_dir):

    with tf.device(TRAIN_DEVICE):

        best = keras.models.load_model(checkpoint)
    
    test_ds = make_datasets.make_dataset(test_files, pupil_id, binary_only)
    
    scores = {k: float(v) for k, v in best.evaluate(test_ds, return_dict=True).items()}
    iou = scores["pupil_iou"]
    scores["pupil_dice"] = 2.0 * iou / (1.0 + iou)
    (run_dir / "test_metrics.json").write_text(json.dumps(scores, indent=2))
    
    print(f"Session {TEST_SESSION} · pupil IoU {iou:.4f} · pupil Dice {scores['pupil_dice']:.4f}")



def test_per_user(run_dir, test_files, pupil_id, binary_only):

    with tf.device(TRAIN_DEVICE):
        best = keras.models.load_model(run_dir / "best.keras", compile=False)


    @tf.function(reduce_retracing=True)
    def test_batch_stats(images, masks):
       
        probs = best(images, training=False)
        predicted = probs >= THRESHOLD
        actual = masks > 0.5
        intersection = tf.math.count_nonzero(actual & predicted)
        union = tf.math.count_nonzero(actual | predicted)
        loss_sum = tf.reduce_sum(losses.bce_dice_loss(masks, probs))  # Sum of per-frame losses
       
        return intersection, union, loss_sum


    def overlap_scores(intersection, union):
       
        if union == 0:
       
            return 0.0, 0.0  # Same empty-union convention as the original test cell
       
        return intersection / union, 2 * intersection / (union + intersection)

    rows = []
    print(f"Held-out session {TEST_SESSION} | threshold {THRESHOLD}")
    print(f"{'User':<9} {'Frames':>6} {'IoU':>9} {'Dice':>9} {'Loss':>9}")

    for uid in sorted({r.user for r in test_files}):

        recordings = [r for r in test_files if r.user == uid]
        dataset = make_datasets.make_dataset(recordings, pupil_id, binary_only, training=False)
        intersection = union = frames = 0
        loss_sum = 0.0

        for images, masks in dataset:

            batch_i, batch_u, batch_loss = test_batch_stats(images, masks)
            intersection += int(batch_i.numpy())
            union += int(batch_u.numpy())
            loss_sum += float(batch_loss.numpy())
            frames += int(images.shape[0])

        iou, dice = overlap_scores(intersection, union)
        loss = loss_sum / frames
        rows.append(dict(user=uid, frames=frames, intersection=intersection, union=union,
                        pupil_iou=iou, pupil_dice=dice, loss=loss))
        print(f"{'user' + str(uid):<9} {frames:>6} {iou:>9.2%} {dice:>9.2%} {loss:>9.4f}")

    # Equal weight per user for the macro scores; summed pixels for the pooled scores.
    total_frames = sum(r["frames"] for r in rows)
    pooled_iou, pooled_dice = overlap_scores(
        sum(r["intersection"] for r in rows),
        sum(r["union"] for r in rows),
    )
    scores = dict(
        loss=sum(r["loss"] * r["frames"] for r in rows) / total_frames,
        pupil_iou=pooled_iou,
        pupil_dice=pooled_dice,
        mean_user_iou=float(np.mean([r["pupil_iou"] for r in rows])),
        mean_user_dice=float(np.mean([r["pupil_dice"] for r in rows])),
    )

    print(f"\nMean across {len(rows)} users   "
        f"IoU {scores['mean_user_iou']:.2%} | Dice {scores['mean_user_dice']:.2%}")
    print(f"Pooled over {total_frames} frames   "
        f"IoU {pooled_iou:.2%} | Dice {pooled_dice:.2%} | Loss {scores['loss']:.4f}")

    with (run_dir / "test_metrics_by_user.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "test_metrics.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":

    print("Starting...")
    training_pipeline()
    print("Training and test completed\n")
    
