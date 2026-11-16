from time import perf_counter

import cv2
import h5py
import numpy as np
from scipy.optimize import least_squares


def camera_parameters(path, h, w):

    params = {}
    
    if path:
    
        with np.load(path, allow_pickle=False) as file:
    
            params = {key: file[key] for key in file.files}
    
    default_K = [[w, 0, (w - 1) / 2], [0, w, (h - 1) / 2], [0, 0, 1]]
    K = np.asarray(params.get("K", default_K), dtype=float)
    
    if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
    
            or not np.allclose(K[2], [0, 0, 1])):
    
        raise ValueError("K must be a valid 3x3 camera matrix.")
    
    if "image_size" in params:
    
        size = np.asarray(params["image_size"], dtype=float)
        if "K" not in params or size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
    
            raise ValueError("image_size requires K and a positive (width, height).")
    
        sx, sy = np.array([w, h]) / size
        K = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]]) @ K
    
    dist = params.get("dist")
    if dist is not None:
    
        dist = np.asarray(dist, dtype=float).ravel()
    
        if "K" not in params or dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
    
            raise ValueError("Supply K and valid OpenCV distortion coefficients together.")
    
    eye = params.get("eye_center_scaled")
    if eye is not None:
    
        eye = np.asarray(eye, dtype=float).reshape(-1)
    
        if eye.shape != (3,) or not np.isfinite(eye).all() or eye[2] <= 0:
    
            raise ValueError("eye_center_scaled must be a finite 3-vector with positive z.")
    
        eye = eye / eye[2]
    
    rotation = np.asarray(params.get("gaze_rotation", np.eye(3)), dtype=float)
    if (rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
    
        raise ValueError("gaze_rotation must be a proper 3x3 rotation.")
    
    mode = "fixation-calibrated gaze" if "gaze_rotation" in params else "optical-axis estimate"
    print(f"Output: {mode} | intrinsics: " + ("supplied" if "K" in params else "assumed"))
    
    return K, dist, eye, rotation



def ellipse_observation(ellipse, K):

    """Back-project the fitted ellipse to the notebook's two optical-axis candidates."""
    center, radii, R = ellipse
    T = np.eye(3)
    T[:2, :2], T[:2, 2] = R, -R @ center
    B = T @ K
    Q = B.T @ np.diag([1/radii[0]**2, 1/radii[1]**2, -1]) @ B
    (lo, mid, hi), V = np.linalg.eigh(Q)
    
    if not lo < 0 < mid:
    
        return None
    
    a, b = np.sqrt(np.maximum([hi - mid, mid - lo], 0) / (hi - lo))
    normals = b * V[:, 0] + np.array([[a], [-a]]) * V[:, 2]
    
    try:
    
        rays = np.linalg.solve(Q, normals.T).T
    
    except np.linalg.LinAlgError:
    
        return None
    
    if not np.isfinite(rays).all() or np.any(np.abs(rays[:, 2]) < 1e-12):
    
        return None
    
    rays = rays / rays[:, 2:3]
    normals[np.sum(normals * rays, axis=1) > 0] *= -1
    
    return a * V[:, 1], normals, rays



def fit_eye_center(planes, focal_px):

    planes = np.asarray(planes)
    A, rhs = planes[:, :2], -planes[:, 2]
    xy, _, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)

    if rank < 2 or sv[1] <= 1e-6 * sv[0]:

        return None

    xy = least_squares(lambda p: A @ p - rhs, xy, loss="soft_l1", f_scale=1/focal_px).x

    if sv[1] < 0.05 * sv[0]:

        print("Weak eye-center estimate. More varied eye directions improve conditioning.")

    return np.r_[xy, 1.0]



def fit_pupil(mask, K, dist):
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
    
        return None
    
    border = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(border)
    if (len(border) < 5 or cv2.contourArea(border) <= 0 or x == 0 or y == 0
    
            or x+w == mask.shape[1] or y+h == mask.shape[0]):
    
        return None
    
    points = border.astype(np.float32)
    if dist is not None:
    
        points = cv2.undistortPoints(points, K, dist, P=K)
    
    try:
    
        center, diameters, angle = cv2.fitEllipse(points)
    
    except cv2.error:
    
        return None
    
    radii = np.asarray(diameters) / 2
    if min(radii) <= 0 or not np.isfinite([*center, *radii, angle]).all():
    
        return None
    
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    R = np.array([[c, s], [-s, c]])
    local = (points[:, 0] - center) @ R.T / radii
    
    if np.percentile(np.abs(np.linalg.norm(local, axis=1) - 1), 90) > 0.20:
    
        return None
    
    return center, radii, R



def load_predictor(path, threshold, warmup):
    
    import tensorflow as tf

    start = perf_counter()
    model = tf.keras.models.load_model(path, compile=False)
    shape = model.input_shape
    
    if (not isinstance(shape, tuple) or len(shape) != 4 or shape[-1] != 1
            or None in shape[1:3] or tuple(model.output_shape[1:]) != tuple(shape[1:])):
    
        raise ValueError("Expected a fixed-size grayscale input and one pupil probability per pixel.")
    
    h, w = shape[1:3]
    spec = tf.TensorSpec((1, h, w, 1), tf.float32)

    @tf.function(input_signature=[spec])
    def infer(batch):
    
        prob = model(batch, training=False)
        return tf.cast(prob[0, :, :, 0] >= threshold, tf.uint8)

    probe = tf.zeros(spec.shape, dtype=tf.float32)
    
    for _ in range(warmup):
    
        result = infer(probe)
        result.numpy()
    
    print(f"Model: {path} | {model.count_params():,} parameters")
    print(f"Model load + {warmup} warm-up calls: {perf_counter() - start:.2f} s")
    print(f"TensorFlow GPUs: {[g.name for g in tf.config.list_physical_devices('GPU')]}")
    print(f"Mask tensor device: {result.device} | input: {w} x {h} | batch: 1")
    return infer, h, w
