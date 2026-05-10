# 3D Object Pose Estimation with YOLO + PnP

![Pose estimation demo](assets/pose_estimation_demo.png)

Estimate the 6-DoF pose (position + orientation) of an iPhone 13 from a single RGB camera frame — no markers, no depth sensor.

The approach: train YOLOv8-pose to detect 8 corner keypoints on the phone body, then use OpenCV `solvePnP` to recover the 3D pose from those 2D–3D correspondences. This is the same math that ArUco uses, applied to a real object instead of a printed marker.

---

## Pipeline

```
CAD model (STL)
      │
      ▼
Blender renders                  ← scripts/generate_dataset.py
  (synthetic images + auto labels)
      │
      ▼
Train YOLOv8-pose                ← scripts/train.py
      │
      ▼
Real-time inference              ← scripts/run_pose_estimation.py
  YOLO detects 8 keypoints (2D)
  solvePnP recovers rvec + tvec
```

---

## Project Structure

```
cad/
  iphone13.stl                    iPhone 13 (standard, 2 cameras) CAD model in mm
config/
  camera_calibration.yaml         fx=676.68, fy=677.34, cx=345.79, cy=236.51
dataset/
  images/train/                   997 synthetic renders (PNG)
  images/val/                     188 synthetic renders (PNG)
  labels/train/                   YOLO pose annotations (.txt)
  labels/val/                     YOLO pose annotations (.txt)
  dataset.yaml
  real/images/                    Real webcam captures for fine-tuning
  real/labels/                    Hand-annotated YOLO pose labels (CVAT export)
  real_dataset.yaml
runs/
  run_100/weights/best.pt         Synthetic-trained model
  run_finetune/weights/best.pt    Fine-tuned on real images (active model)
scripts/
  inspect_stl_dimensions.py       Print STL bounding box dimensions (Blender)
  generate_dataset.py             Generate the full synthetic training dataset (Blender)
  split_train_val.py              Move 10% of train images → val
  capture_real_images.py          Capture real webcam images for annotation
  train.py                        Train or fine-tune YOLOv8n-pose
  run_pose_estimation.py          Real-time 6-DoF pose estimation from webcam
tests/
  render_single_image.py          Render one test image to check the pipeline (Blender)
  verify_pose_on_synthetic.py     Verify YOLO + solvePnP on a known Blender render
```

---

## Step-by-Step Guide

### 1. Get the CAD model

Download the iPhone 13 STL from Printables (search "iphone13 stl printables sumit_basra").
Place it at `cad/iphone13.stl`. The model should be the **standard iPhone 13** (2 cameras, not Pro/Max).
Units are in mm, centered at origin, screen face pointing in −Z.

Run `inspect_stl_dimensions.py` to verify dimensions:

```bash
blender --background --python scripts/inspect_stl_dimensions.py
```

Expected output: X ≈ ±36.2 mm, Y ≈ ±73.4 mm, Z ≈ −3.8 to +6.3 mm.

### 2. Camera calibration

Copy your calibration file to `config/camera_calibration.yaml`.
It needs `camera_matrix` and `distortion_coefficients` in OpenCV YAML format.

The calibration is used both by Blender (to match the render FOV to the real camera) and by `solvePnP` at inference time.

### 3. Generate synthetic training data

```bash
blender --background --python scripts/generate_dataset.py
```

This script fixes the phone flat (screen face down) at the origin and orbits a virtual camera around it — full 360° azimuth, 10–80° elevation, 150–350 mm distance — rendering with Cycles + OptiX. For each render it:
1. Projects the 8 bounding box corners to pixel coordinates using `world_to_camera_view`
2. Marks each keypoint as visible (facing camera) or occluded (back face) using face normal dot product
3. Writes a YOLO pose label file alongside the image

Key settings in the script:
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `NUM_RENDERS` | 100 | Images to generate |
| `RENDER_SAMPLES` | 256 | Cycles render samples (quality) |
| `MIN/MAX_DISTANCE_MM` | 150–350 mm | Camera distance range |

Takes about 15 minutes with RTX 4060 + OptiX.

### 4. Split train / val

```bash
python3 scripts/split_train_val.py
```

Moves a random 10% of images and labels from `train/` to `val/`.
Result: ~900 train, ~100 val (exact numbers depend on how many renders succeeded).

### 5. Train YOLO

```bash
python3 scripts/train.py
```

Two-stage training:

**Stage 1 — synthetic pre-training (`run_100`)**

Train YOLOv8n-pose from scratch on ~1000 synthetic Blender renders. 100 epochs, batch 16, image size 640, `lr0=0.01`.

| Epoch | pose_loss | Pose mAP50 |
|-------|-----------|------------|
| 1     | 9.50      | 0.001      |
| 25    | 2.1       | 0.85       |
| 50    | 0.80      | 0.97       |
| 100   | 0.31      | 0.994      |

Model saved to `runs/run_100/weights/best.pt`.

**Stage 2 — real-image fine-tuning (`run_finetune`)**

Fine-tune `run_100/best.pt` on ~20 hand-annotated real webcam photos. 50 epochs, batch 8, `lr0=0.001` (lower to preserve synthetic knowledge). Configure `train.py` to point `DATASET_CONFIG` at `real_dataset.yaml` and `BASE_MODEL` at `run_100/weights/best.pt`.

Model saved to `runs/run_finetune/weights/best.pt` — this is the active model used by `run_pose_estimation.py`.

### 6. Verify on a synthetic image

```bash
# Render one image from a known camera position
blender --background --python tests/verify_pose_on_synthetic.py -- --mode render

# Run YOLO + solvePnP and compare against ground truth
python3 tests/verify_pose_on_synthetic.py --mode predict
```

This is a sanity check. The script renders the phone from a fixed position (80, −250, 120) mm, then runs the full inference pipeline and prints predicted tvec vs known camera position. It also prints per-step inference times (YOLO ms, PnP ms).

### 7. Run real-time pose estimation

```bash
python3 scripts/run_pose_estimation.py
```

Opens webcam index 2 (change `CAMERA_INDEX` at the top of the file if needed).
Detections below 30% confidence are skipped.

Each frame:
1. YOLO detects 8 keypoints (2D corners)
2. `solvePnP` (EPNP solver) recovers rvec + tvec from the 2D–3D correspondences
3. The solution is validated by reprojection error (< 15 px) and distance (5 cm – 2 m)
4. The pose is smoothed over the last 10 frames to reduce jitter
5. The phone's 6-DoF coordinate frame axes are drawn on the object (red=X, green=Y, blue=Z)

Translation (x, y, z in metres) and rotation (axis-angle in degrees) are printed to the terminal at 1 Hz.

---

## Inference Timing

Measured on RTX 4060 (640×480 input):

| Step | Time |
|------|------|
| YOLO keypoint detection | ~8–12 ms |
| solvePnP | < 1 ms |
| **Total per frame** | **~10–13 ms (~80–100 FPS)** |

The bottleneck is YOLO. `solvePnP` is essentially free — it's just solving a small linear system with 8 point correspondences.

---

## Keypoint Definition

8 corners of the phone bounding box, consistent across Blender rendering and solvePnP:

```
kp0  screen top-left      (-HALF_WIDTH,  HALF_HEIGHT, -HALF_DEPTH)
kp1  screen top-right     ( HALF_WIDTH,  HALF_HEIGHT, -HALF_DEPTH)
kp2  screen bottom-right  ( HALF_WIDTH, -HALF_HEIGHT, -HALF_DEPTH)
kp3  screen bottom-left   (-HALF_WIDTH, -HALF_HEIGHT, -HALF_DEPTH)
kp4  back top-left        (-HALF_WIDTH,  HALF_HEIGHT,  HALF_DEPTH)
kp5  back top-right       ( HALF_WIDTH,  HALF_HEIGHT,  HALF_DEPTH)
kp6  back bottom-right    ( HALF_WIDTH, -HALF_HEIGHT,  HALF_DEPTH)
kp7  back bottom-left     (-HALF_WIDTH, -HALF_HEIGHT,  HALF_DEPTH)

HALF_WIDTH = 35.75 mm,  HALF_HEIGHT = 73.35 mm,  HALF_DEPTH = 3.825 mm
```

The order here must match exactly between the Blender render script, the YOLO annotation, and the `phone_corners_3d` array in `run_pose_estimation.py`.

---

## Known Limitations

1. **Limited real training data** — fine-tuning used only ~20 real photos. More diverse real images (different lighting, backgrounds, angles) would improve robustness.

2. **No upside-down detection** — the training camera always looks at the phone from above (elevation 10–80°). The model has not seen the phone from below.

3. **Thin object depth ambiguity** — the phone is only 7.65 mm thick, so the 8 corners are nearly coplanar. The EPNP solver handles this well, but depth (Z) estimation is less accurate than X/Y translation.

---

## Dependencies

```
Python      3.10+
ultralytics 8.x      (pip install ultralytics)
opencv-python        (pip install opencv-python)
numpy
Blender     5.1+     (for data generation only)
```
