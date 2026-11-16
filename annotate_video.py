"""Overlay the predicted pupil, fitted ellipse and estimated direction on a recorded video.

    python annotate_pupil_video.py eye.mov best.keras eye_overlay.mp4
    python annotate_pupil_video.py eye.mov best.keras eye_overlay.mp4 --roi 220 160 600 450

Put benchmark_pupil_gaze.py beside this file. Use your TensorFlow environment with
OpenCV, NumPy, SciPy, h5py, and FFmpeg/ffprobe (FFmpeg 7+ with libx264 and fsync).
ROI is X Y WIDTH HEIGHT in the upright input video. The default is the entire frame.
One recording should show the same eye in a fixed position relative to the camera.

Optional --calibration eye.npz accepts K, dist, eye_center_scaled and gaze_rotation
from the benchmark. If K was calibrated on the upright, uncropped phone image,
include image_size=(width,height) for that image. Its aspect ratio must match
the video. Without image_size, K is interpreted on the final 346x260 model grid.
The camera axes are x right, y down, z into the scene. Without gaze_rotation the
green arrow represents the projected optical axis in the camera view.
"""

import argparse
import json
import cv2
import subprocess

import numpy as np

from contextlib import contextmanager
from fractions import Fraction

from pathlib import Path
from tempfile import TemporaryDirectory

from utility_functions import (
    camera_parameters, ellipse_observation, fit_eye_center, fit_pupil, load_predictor,
)


HEIGHT, WIDTH = 260, 346


def inspect_video(path):

    entries = ("stream=width,height,time_base,start_pts,duration_ts,avg_frame_rate,"
               "sample_aspect_ratio:stream_side_data=rotation:stream_tags=rotate:"
               "frame=best_effort_timestamp,duration,pkt_duration:format=start_time")

    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "V:0", "-show_entries",
        entries, "-of", "json", str(path),
    ])

    info = json.loads(raw)

    if not info.get("streams") or not info.get("frames"):

        raise ValueError("The recording contains no decodable video frames.")

    stream = info["streams"][0]
    side = stream.get("side_data_list", [])
    angle = float(next((s["rotation"] for s in side if "rotation" in s),
                       stream.get("tags", {}).get("rotate", 0)))

    if not np.isclose(angle / 90, round(angle / 90)):

        raise ValueError("This recording has a non-right-angle display rotation.")

    width, height = stream["width"], stream["height"]
    sar = stream.get("sample_aspect_ratio", "1:1")
    sar = Fraction(sar.replace(":", "/")) if sar not in ("N/A", "0:1") else Fraction(1)

    if round(angle / 90) % 2:

        width, height = height, width
        sar = 1 / sar

    frames = info["frames"]

    if any("best_effort_timestamp" not in f for f in frames):

        raise ValueError("The recording is missing frame timestamps.")

    pts = [int(f["best_effort_timestamp"]) for f in frames]

    if any(b <= a for a, b in zip(pts, pts[1:])):

        raise ValueError("The recording contains repeated or backwards frame timestamps.")

    tb = Fraction(stream["time_base"])
    start = info.get("format", {}).get("start_time")
    origin = Fraction(start) if start not in (None, "N/A") else pts[0] * tb
    relative_pts = [p - round(origin / tb) for p in pts]
    duration = int(frames[-1].get("duration") or frames[-1].get("pkt_duration") or 0)

    if duration <= 0 and "duration_ts" in stream:

        duration = int(stream.get("start_pts", pts[0])) + int(stream["duration_ts"]) - pts[-1]

    if duration <= 0:

        if len(pts) > 1:

            duration = pts[-1] - pts[-2]

        else:

            rate = stream.get("avg_frame_rate", "0/0")

            if rate == "0/0" or Fraction(rate) <= 0:

                raise ValueError("The source does not specify a duration for its only frame.")

            duration = max(1, round(1 / (Fraction(rate) * tb)))

    return width, height, sar, relative_pts, tb, duration


@contextmanager
def decoded_frames(path, width, height):
    
    # FFmpeg applies the phone's display rotation before writing these BGR pixels.
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
           "-map", "0:V:0", "-fps_mode", "passthrough", "-c:v", "rawvideo",
           "-threads:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
    
        def frames():
    
            size = width * height * 3
    
            while True:
    
                data = proc.stdout.read(size)
    
                if not data:
    
                    break
    
                if len(data) != size:
    
                    raise ValueError("The decoder returned an incomplete video frame.")
    
                yield np.frombuffer(data, np.uint8).reshape(height, width, 3)
    
            if proc.wait():
    
                raise RuntimeError("FFmpeg could not decode the recording.")

        try:
    
            yield frames()
    
        finally:
    
            proc.stdout.close()
    
            if proc.poll() is None:
    
                proc.terminate()



class EyeView:
    """The crop and resize used by both the network and the camera projection."""

    def __init__(self, width, height, roi, sar):

        self.x, self.y, self.w, self.h = roi or (0, 0, width, height)
        x, y, w, h = self.x, self.y, self.w, self.h

        if min(x, y) < 0 or min(w, h) < 1 or x+w > width or y+h > height:

            raise ValueError(f"The eye ROI must fit inside the upright {width}x{height} frame.")

        scale = min(WIDTH / (w * float(sar)), HEIGHT / h)
        self.wm = min(WIDTH, max(1, round(w * float(sar) * scale)))
        self.hm = min(HEIGHT, max(1, round(h * scale)))
        self.left = (WIDTH - self.wm) // 2
        self.top = (HEIGHT - self.hm) // 2
        self.body = np.s_[self.top:self.top+self.hm, self.left:self.left+self.wm]
        self.roi = np.s_[y:y+h, x:x+w]
        sx, sy = self.wm / w, self.hm / h
        self.matrix = np.array([[sx, 0, self.left + (sx-1)/2 - sx*x],
                                [0, sy, self.top + (sy-1)/2 - sy*y], [0, 0, 1]])


    def batch(self, frame):

        gray = cv2.cvtColor(frame[self.roi], cv2.COLOR_BGR2GRAY).astype(np.float32)
        interp = cv2.INTER_AREA if self.wm <= self.w and self.hm <= self.h else cv2.INTER_LINEAR
        small = cv2.resize(gray, (self.wm, self.hm), interpolation=interp)
        # Padding with the crop minimum preserves the model's internal min/max normalization.
        image = np.full((HEIGHT, WIDTH), small.min(), np.float32)
        image[self.body] = small

        return image[None, :, :, None]


    def to_source(self, pixels):

        return (np.asarray(pixels) - self.matrix[:2, 2]) / self.matrix.diagonal()[:2]



def geometry_settings(path, width, height, sar, view):

    if path:

        with np.load(path, allow_pickle=False) as file:

            from_full_frame = "image_size" in file
            has_k = "K" in file
            calibrated = "gaze_rotation" in file

            if from_full_frame:

                size = np.asarray(file["image_size"])

                if size.shape != (2,) or min(size) <= 0 or not np.isclose(
                        size[0]/size[1], width/height, rtol=0.005):

                    raise ValueError("Calibration image_size must describe the upright, uncropped video.")

    else:

        from_full_frame = has_k = calibrated = False

    if from_full_frame or not has_k:

        K, dist, eye, rotation = camera_parameters(path, height, width)

        if not has_k:

            K[1, 1] *= float(sar)

        K = view.matrix @ K

    else:

        K, dist, eye, rotation = camera_parameters(path, HEIGHT, WIDTH)

    label = "Gaze" if calibrated else "Optical axis"

    if not has_k:

        label += " (approx.)"

    return K, dist, eye, rotation, label



def pupil_ellipse(mask, view, K, dist):

    # Run the benchmark's border rejection on the actual crop, excluding padding.
    shift = np.array([view.left, view.top], dtype=float)
    crop_K = K.copy()
    crop_K[:2, 2] -= shift
    ellipse = fit_pupil(np.ascontiguousarray(mask[view.body]), crop_K, dist)

    if ellipse is None:

        return None

    center, radii, R = ellipse

    return np.asarray(center) + shift, radii, R



def axis_and_ray(observation, eye):

    """Benchmark branch selection, retaining the matching 3-D ray for projection."""
    _, normals, rays = observation

    if np.linalg.norm(normals[0] - normals[1]) < 1e-6:

        return normals[0], rays[0]

    if eye is None:

        return None

    candidates = []
    for normal, ray in zip(normals, rays):

        (depth, offset), _, rank, _ = np.linalg.lstsq(
            np.column_stack((ray, -normal)), eye, rcond=None)

        if rank == 2 and depth > 0 and offset > 0:

            candidates.append((normal, ray))

    return candidates[0] if len(candidates) == 1 else None



def paint(frame, mask, ellipse, view, K, dist, eye, rotation, label):

    frame = frame.copy()
    crop = frame[view.roi]
    mask_full = cv2.resize(mask[view.body], (view.w, view.h), interpolation=cv2.INTER_NEAREST)
    marked = mask_full.astype(bool)
    crop[marked] = 0.58 * crop[marked] + 0.42 * np.array([230, 190, 0])
    height, width = frame.shape[:2]
    pen = max(1, round(min(height, width) / 460))
    font_size = min(1.25, max(0.40, min(height, width)/780))
    status = "No reliable ellipse" if marked.any() else "No pupil mask"
    valid = False

    if ellipse is not None:

        center, radii, R = ellipse
        # The fit lies in the undistorted image. Redistort all overlay geometry.
        angles = np.linspace(0, 2*np.pi, 96, endpoint=False)
        ring = center + (np.column_stack((np.cos(angles), np.sin(angles))) * radii) @ R
        undistorted = np.vstack((ring, center))
        camera_rays = np.linalg.solve(K, np.c_[undistorted, np.ones(len(undistorted))].T).T
        drawn = project(camera_rays, view, K, dist)
        cv2.polylines(frame, [drawn[:-1]], True, (0, 210, 255), pen, cv2.LINE_AA)
        cv2.drawMarker(frame, tuple(drawn[-1]), (0, 210, 255), cv2.MARKER_CROSS,
                       max(7, pen*6), pen, cv2.LINE_AA)
        obs = ellipse_observation(ellipse, K)
        selected = axis_and_ray(obs, eye) if obs is not None else None
        status = "Direction unresolved"

        if selected is not None:

            normal, ray = selected
            direction = rotation @ normal
            shaft = project(np.stack((ray, ray + 0.15 * ray[2] * direction)), view, K, dist)
            cv2.arrowedLine(frame, tuple(shaft[0]), tuple(shaft[1]), (80, 255, 70),
                            pen+1, cv2.LINE_AA, tipLength=0.22)
            status = f"{label}  " + " ".join(f"{v:+.2f}" for v in direction)
            valid = True

    anchor = (max(4, pen*4), max(16, round(24*font_size)))

    for color, thickness in [((20, 20, 20), pen+2), ((250, 250, 250), pen)]:

        cv2.putText(frame, status, anchor, cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    color, thickness, cv2.LINE_AA)

    return frame, valid



def project(xyz, view, K, dist):

    zeros = np.zeros(3)
    pixels = cv2.projectPoints(xyz, zeros, zeros, K, dist)[0].reshape(-1, 2)

    return np.rint(view.to_source(pixels)).astype(np.int32)



def encoder_command(source, width, height, sar, count, tb, last_duration):

    streams = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=codec_name", "-of", "json", str(source),
    ])
    # Keep common phone audio streams bit-for-bit, including their original delay.
    audio = []

    for index, stream in enumerate(json.loads(streams)["streams"]):

        if stream.get("codec_name") in {"aac", "mp3", "alac", "ac3", "eac3"}:

            audio.extend([f"-c:a:{index}", "copy"])

        else:

            audio.extend([f"-c:a:{index}", "aac", f"-b:a:{index}", "192k"])

    # With B frames disabled, the last packet is also the last displayed frame.
    # MP4 uses the gap to the next timestamp for every preceding sample.
    fix_tail = rf"setts=duration=if(eq(N\,{count-1})\,{last_duration}\,DURATION)"
    return [
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-xerror", "-filter_threads", "1",
        "-copyts", "-start_at_zero", "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", "25", "-i", "pipe:0",
        "-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "-1",
        "-vf", f"fsync=file=timing.map,setsar={sar.numerator}/{sar.denominator}",
        "-fps_mode", "passthrough", "-enc_time_base", f"{tb.numerator}/{tb.denominator}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-bf", "0",
        "-pix_fmt", "yuv444p" if width % 2 or height % 2 else "yuv420p",
        "-video_track_timescale", str(tb.denominator), "-movie_timescale", str(tb.denominator),
        "-bsf:v", fix_tail, *audio,
        "-avoid_negative_ts", "disabled", "-movflags", "+faststart", "annotated.mp4",
    ]



def annotate(video, model, output, roi=None, calibration=None, threshold=0.5):

    """Process all frames and return the path to the finished H.264 MP4."""
    video, model, output = (Path(p).expanduser().resolve() for p in (video, model, output))

    if output.suffix.lower() != ".mp4" or output in (video, model):

        raise ValueError("Choose a separate .mp4 output path.")

    if not 0 < threshold < 1:

        raise ValueError("The probability threshold must be between zero and one.")

    for path in (video, model):

        if not path.is_file():

            raise FileNotFoundError(path)

    width, height, sar, pts, tb, tail = inspect_video(video)
    view = EyeView(width, height, roi, sar)
    K, dist, eye, rotation, label = geometry_settings(calibration, width, height, sar, view)
    infer, h, w = load_predictor(model, threshold, warmup=1)

    if (h, w) != (HEIGHT, WIDTH):

        raise ValueError(f"This model expects {(h,w)}. The pupil models use {(HEIGHT, WIDTH)}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    ellipses, planes = [], []
    mask_bytes = (HEIGHT * WIDTH + 7) // 8
    print(f"Predicting {len(pts)} frames. Input for the model is {WIDTH} x {HEIGHT}.", flush=True)

    with TemporaryDirectory(prefix=".pupil-", dir=output.parent) as tmp:

        tmp = Path(tmp)

        with (tmp / "masks.bin").open("wb") as cache, decoded_frames(video, width, height) as frames:

            for frame in frames:

                mask = infer(view.batch(frame)).numpy()
                cache.write(np.packbits(mask).tobytes())
                ellipse = pupil_ellipse(mask, view, K, dist)
                ellipses.append(ellipse)

                if eye is None and ellipse is not None:

                    obs = ellipse_observation(ellipse, K)

                    if obs is not None:

                        planes.append(obs[0])

                if len(ellipses) % 250 == 0:

                    print(f"Predicted {len(ellipses)}/{len(pts)}", flush=True)

        if len(ellipses) != len(pts):

            raise ValueError(f"Decoder returned {len(ellipses)} frames for {len(pts)} timestamps.")

        if eye is None and len(planes) >= 2:

            eye = fit_eye_center(planes, (K[0, 0] + K[1, 1]) / 2)

        if eye is None:

            print("Insufficient eye rotation to estimate a fixed center. Ambiguous directions will be hidden.")

        with (tmp / "timing.map").open("w", encoding="ascii") as mapping:

            for i, timestamp in enumerate(pts):

                mapping.write(f"{i} {timestamp} {tb.numerator}/{tb.denominator}\n")

        cmd = encoder_command(video, width, height, sar, len(pts), tb, tail)
        print("Rendering the annotated recording.", flush=True)
        valid = 0

        with (tmp / "masks.bin").open("rb") as cache, decoded_frames(video, width, height) as frames:

            with subprocess.Popen(cmd, stdin=subprocess.PIPE, cwd=tmp) as encoder:

                try:

                    for frame, ellipse in zip(frames, ellipses, strict=True):

                        packed = np.frombuffer(cache.read(mask_bytes), np.uint8)
                        mask = np.unpackbits(packed, count=HEIGHT*WIDTH).reshape(HEIGHT, WIDTH)
                        overlay, ok = paint(frame, mask, ellipse, view, K, dist, eye, rotation, label)
                        encoder.stdin.write(memoryview(overlay))
                        valid += int(ok)

                    encoder.stdin.close()

                    if encoder.wait():

                        raise RuntimeError("FFmpeg could not finish encoding the MP4.")

                finally:

                    if encoder.poll() is None:

                        encoder.terminate()

        (tmp / "annotated.mp4").replace(output)

    print(f"Saved {output}\n{len(pts)} frames | {sum(e is not None for e in ellipses)} "
          f"ellipse fits | {valid} valid directions")

    return output



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", help="Recorded eye video")
    parser.add_argument("model", help="Pupil segmentation .keras checkpoint")
    parser.add_argument("output", help="Output .mp4 file")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="One-eye region of the upright input, in pixels")
    parser.add_argument("--calibration", type=Path, help="Optional same-eye/camera .npz")
    parser.add_argument("--threshold", type=float, default=0.5, help="Pupil probability cutoff")
    args = parser.parse_args()
    annotate(args.video, args.model, args.output, args.roi, args.calibration, args.threshold)
