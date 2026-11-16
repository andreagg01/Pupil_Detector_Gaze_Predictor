import tensorflow as tf

from tensorflow import keras


@keras.utils.register_keras_serializable(package="near_eye")
def bce_dice_loss(y_true, y_prob):

    y_true = tf.cast(y_true, tf.float32)
    y_prob = tf.cast(y_prob, tf.float32)

    # binary_crossentropy reduces the single channel, leaving B,H,W.
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_prob), axis=(1, 2))
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_prob, axis=axes)
    size = tf.reduce_sum(y_true + y_prob, axis=axes)
    dice = (2.0 * intersection + 1.0) / (size + 1.0)

    return 0.5 * bce + 0.5 * (1.0 - dice)
