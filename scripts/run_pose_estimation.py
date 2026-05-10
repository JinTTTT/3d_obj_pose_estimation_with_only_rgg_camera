"""
iPhone 13 pose estimation using YOLOv8-pose + solvePnP.

Run with:
    python3 scripts/run_pose_estimation.py
"""

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

CAMERA_INDEX         = 2
ROOT                 = Path(__file__).resolve().parents[1]
CALIBRATION_PATH     = ROOT / "config" / "camera_calibration.yaml"
MODEL_WEIGHTS_PATH   = ROOT / "runs" / "run_100" / "weights" / "best.pt"

CONFIDENCE_THRESHOLD = 0.75


def load_camera_calibration(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    camera_matrix    = fs.getNode("camera_matrix").mat()
    dist_coeffs      = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def define_phone_corners_3d():
    """
    3D coordinates of the 8 bounding box corners in meters.
    Must match the exact same order used during training:
      kp0-3: screen face (z negative), kp4-7: back face (z positive)
      within each face: top-left, top-right, bottom-right, bottom-left

    iPhone 13 real dimensions:
      width  = 71.5 mm  → half = 35.75 mm = 0.03575 m
      height = 146.7 mm → half = 73.35 mm = 0.07335 m
      depth  =  7.65 mm → half =  3.825 mm = 0.003825 m
    """
    HALF_WIDTH  = 0.03575
    HALF_HEIGHT = 0.07335
    HALF_DEPTH  = 0.003825

    return np.array([
        [-HALF_WIDTH,  HALF_HEIGHT, -HALF_DEPTH],   # kp0  screen top-left
        [ HALF_WIDTH,  HALF_HEIGHT, -HALF_DEPTH],   # kp1  screen top-right
        [ HALF_WIDTH, -HALF_HEIGHT, -HALF_DEPTH],   # kp2  screen bottom-right
        [-HALF_WIDTH, -HALF_HEIGHT, -HALF_DEPTH],   # kp3  screen bottom-left
        [-HALF_WIDTH,  HALF_HEIGHT,  HALF_DEPTH],   # kp4  back top-left
        [ HALF_WIDTH,  HALF_HEIGHT,  HALF_DEPTH],   # kp5  back top-right
        [ HALF_WIDTH, -HALF_HEIGHT,  HALF_DEPTH],   # kp6  back bottom-right
        [-HALF_WIDTH, -HALF_HEIGHT,  HALF_DEPTH],   # kp7  back bottom-left
    ], dtype=np.float32)


def main():
    camera_matrix, dist_coeffs = load_camera_calibration(CALIBRATION_PATH)
    phone_corners_3d = define_phone_corners_3d()
    yolo_model = YOLO(str(MODEL_WEIGHTS_PATH))

    webcam = cv2.VideoCapture(CAMERA_INDEX)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Controls:  q or Esc to quit")
    last_console_print_time = 0.0

    # Timing accumulators for inference reporting
    yolo_times   = []
    pnp_times    = []
    frame_times  = []
    last_frame_time = time.perf_counter()

    while True:
        frame_ok, frame = webcam.read()
        if not frame_ok:
            break

        now = time.perf_counter()
        frame_times.append(now - last_frame_time)
        last_frame_time = now

        # ── Step 1: run YOLO to detect keypoints ──────────────────────────────
        t0 = time.perf_counter()
        detections = yolo_model(frame, verbose=False)[0]
        yolo_times.append(time.perf_counter() - t0)

        if detections.keypoints is not None and len(detections.keypoints) > 0:
            for keypoint_detection, bounding_box in zip(detections.keypoints, detections.boxes):

                if float(bounding_box.conf) < CONFIDENCE_THRESHOLD:
                    continue

                # detected_corners_2d shape: (8, 2) — pixel positions of 8 keypoints
                detected_corners_2d = keypoint_detection.xy[0].cpu().numpy().astype(np.float32)

                # Skip if any keypoint is at (0, 0) — means YOLO couldn't find it
                if np.any(np.all(detected_corners_2d == 0, axis=1)):
                    continue

                # ── Step 2: solvePnP ──────────────────────────────────────────
                t1 = time.perf_counter()
                success, rvec, tvec = cv2.solvePnP(
                    phone_corners_3d,
                    detected_corners_2d,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                pnp_times.append(time.perf_counter() - t1)

                if not success:
                    continue

                # ── Step 3: draw axes on the phone ────────────────────────────
                cv2.drawFrameAxes(
                    frame, camera_matrix, dist_coeffs, rvec, tvec,
                    0.03,  # axis length = 3 cm
                )

                for i, (px, py) in enumerate(detected_corners_2d):
                    cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(px) + 5, int(py) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                translation = tvec.ravel()
                rotation_deg = np.degrees(rvec.ravel())

                cv2.putText(frame,
                            f"t: x={translation[0]:.3f} y={translation[1]:.3f} z={translation[2]:.3f} m",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"r: {rotation_deg[0]:.1f} {rotation_deg[1]:.1f} {rotation_deg[2]:.1f} deg",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if now - last_console_print_time > 1.0:
                    avg_yolo_ms  = np.mean(yolo_times) * 1000  if yolo_times  else 0
                    avg_pnp_ms   = np.mean(pnp_times)  * 1000  if pnp_times   else 0
                    avg_fps      = 1.0 / np.mean(frame_times)   if frame_times else 0

                    print(f"tvec [m]: x={translation[0]:.4f} y={translation[1]:.4f} z={translation[2]:.4f}")
                    print(f"rvec [deg]: {rotation_deg[0]:.2f} {rotation_deg[1]:.2f} {rotation_deg[2]:.2f}")
                    print(f"Timing — YOLO: {avg_yolo_ms:.1f} ms  PnP: {avg_pnp_ms:.2f} ms  FPS: {avg_fps:.1f}\n")

                    yolo_times.clear()
                    pnp_times.clear()
                    frame_times.clear()
                    last_console_print_time = now

        cv2.imshow("iPhone pose estimation", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
