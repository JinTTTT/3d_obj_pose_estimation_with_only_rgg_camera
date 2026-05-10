"""
Train YOLOv8-pose to detect iPhone 13 corner keypoints.

Run with:
    python3 scripts/train.py
"""

from ultralytics import YOLO

DATASET_CONFIG = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/dataset/dataset.yaml"

# yolov8n-pose.pt = nano (smallest, fastest) — good for testing
# yolov8s-pose.pt = small — better accuracy, use for real training
yolo_model = YOLO("yolov8n-pose.pt")

yolo_model.train(
    data    = DATASET_CONFIG,
    epochs  = 100,
    imgsz   = 640,
    batch   = 16,
    device  = 0,
    project = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/runs",
    name    = "run_100",
)
