"""
Train YOLOv8-pose to detect iPhone 13 corner keypoints.

Run with:
    python3 scripts/train.py
"""

from ultralytics import YOLO

DATASET_CONFIG = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/dataset/real_dataset.yaml"
BASE_MODEL     = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/runs/run_100/weights/best.pt"

# Fine-tune the synthetic-trained model on real images only.
# Lower lr so we don't destroy what the model already learned.
yolo_model = YOLO(BASE_MODEL)

yolo_model.train(
    data    = DATASET_CONFIG,
    epochs  = 50,
    imgsz   = 640,
    batch   = 8,
    device  = 0,
    lr0     = 0.001,
    project = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/runs",
    name    = "run_finetune",
)
