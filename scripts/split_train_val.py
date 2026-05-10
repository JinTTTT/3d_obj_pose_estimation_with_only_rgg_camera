"""
Split the dataset into train and val sets.

Run with:
    python3 scripts/split_train_val.py

Moves 10% of images (and their label files) from train/ to val/.
"""

import random
import shutil
from pathlib import Path

DATASET_DIR = Path("/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/dataset")
VAL_RATIO   = 0.1

train_images_dir = DATASET_DIR / "images" / "train"
train_labels_dir = DATASET_DIR / "labels" / "train"
val_images_dir   = DATASET_DIR / "images" / "val"
val_labels_dir   = DATASET_DIR / "labels" / "val"

all_train_images = sorted(train_images_dir.glob("*.png"))
num_val_images   = int(len(all_train_images) * VAL_RATIO)
val_image_paths  = random.sample(all_train_images, num_val_images)

for image_path in val_image_paths:
    label_path = train_labels_dir / image_path.with_suffix(".txt").name

    shutil.move(str(image_path), str(val_images_dir / image_path.name))
    shutil.move(str(label_path), str(val_labels_dir / label_path.name))

print(f"Train : {len(list(train_images_dir.glob('*.png')))}")
print(f"Val   : {len(list(val_images_dir.glob('*.png')))}")
