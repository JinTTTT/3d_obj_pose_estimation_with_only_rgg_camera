"""
Generate synthetic training images for YOLOv8-pose.

Run with:
    blender --background --python scripts/generate_dataset.py

Output:
    dataset/images/train/   + dataset/labels/train/
    dataset/images/val/     + dataset/labels/val/
    dataset/dataset.yaml
"""

import bpy
import math
import random
import re
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT            = Path("/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp")
STL_PATH        = ROOT / "cad"    / "iphone13.stl"
CALIBRATION_PATH = ROOT / "config" / "camera_calibration.yaml"
DATASET_DIR     = ROOT / "dataset"

# ── Settings ──────────────────────────────────────────────────────────────────

NUM_RENDERS  = 356   # how many NEW renders to generate
START_INDEX  = 644   # continue numbering from where we left off
RENDER_WIDTH  = 640
RENDER_HEIGHT = 480
RENDER_SAMPLES = 64  # Cycles quality — 64 is good on GPU

# Camera orbit range
MIN_DISTANCE_MM  = 150   # closest camera distance from phone center
MAX_DISTANCE_MM  = 1000  # furthest
MIN_ELEVATION_DEG = 5    # nearly horizontal
MAX_ELEVATION_DEG = 85   # nearly top-down


# ── Camera calibration ────────────────────────────────────────────────────────

def read_calibration_file(path):
    with open(path) as f:
        content = f.read()
    data = re.search(r'camera_matrix:.*?data:\s*\[(.*?)\]', content, re.DOTALL)
    values = [float(v.strip()) for v in data.group(1).split(',')]
    fx, fy = values[0], values[4]
    cx, cy = values[2], values[5]
    return fx, fy, cx, cy


# ── GPU setup ─────────────────────────────────────────────────────────────────

def enable_optix_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"


# ── Model loading and keypoint definition ─────────────────────────────────────

def import_phone_stl():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=str(STL_PATH))
    return next(o for o in bpy.data.objects if o.type == "MESH")


def get_phone_corners(phone_mesh):
    """
    Build the 8 bounding box corners from the phone mesh.
    We construct each corner manually from min/max values so the
    order is always guaranteed.

    Keypoint order (must stay consistent with solvePnP later):
      kp0–3 : screen face (z_min), corners going top-left → top-right → bottom-right → bottom-left
      kp4–7 : back face   (z_max), same corner order
    """
    bound_box_corners = [phone_mesh.matrix_world @ Vector(c) for c in phone_mesh.bound_box]
    xs = [c.x for c in bound_box_corners]
    ys = [c.y for c in bound_box_corners]
    zs = [c.z for c in bound_box_corners]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    phone_corners = [
        Vector((x_min, y_max, z_min)),  # kp0  screen top-left
        Vector((x_max, y_max, z_min)),  # kp1  screen top-right
        Vector((x_max, y_min, z_min)),  # kp2  screen bottom-right
        Vector((x_min, y_min, z_min)),  # kp3  screen bottom-left
        Vector((x_min, y_max, z_max)),  # kp4  back top-left
        Vector((x_max, y_max, z_max)),  # kp5  back top-right
        Vector((x_max, y_min, z_max)),  # kp6  back bottom-right
        Vector((x_min, y_min, z_max)),  # kp7  back bottom-left
    ]

    # Face normals — used to decide if a keypoint is visible or occluded.
    # Screen face points in -Z direction, back face points in +Z direction.
    face_normals = [Vector((0, 0, -1))] * 4 + [Vector((0, 0, 1))] * 4

    phone_center = Vector(((x_min + x_max) / 2,
                           (y_min + y_max) / 2,
                           (z_min + z_max) / 2))

    return phone_corners, face_normals, phone_center


# ── Camera ────────────────────────────────────────────────────────────────────

def add_camera(fov_horizontal):
    bpy.ops.object.camera_add(location=(0, -300, 100))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    camera.data.lens_unit = "FOV"
    camera.data.angle     = fov_horizontal
    return camera


def place_camera_randomly(camera, phone_center):
    """Place the camera at a random position on a sphere around the phone."""
    elevation = math.radians(random.uniform(MIN_ELEVATION_DEG, MAX_ELEVATION_DEG))
    azimuth   = random.uniform(0, 2 * math.pi)
    distance  = random.uniform(MIN_DISTANCE_MM, MAX_DISTANCE_MM)

    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)

    camera.location = phone_center + Vector((x, y, z))
    direction = phone_center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()  # required — otherwise world matrix is stale


# ── World / lighting ──────────────────────────────────────────────────────────

def setup_lighting():
    """
    Sky texture for lighting + solid color for background.
    Returns the nodes we need to update each render.
    """
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output_node   = nodes.new("ShaderNodeOutputWorld")
    bg_node       = nodes.new("ShaderNodeBackground")
    mix_node      = nodes.new("ShaderNodeMixRGB")
    sun_node      = nodes.new("ShaderNodeTexSky")
    bg_color_node = nodes.new("ShaderNodeRGB")
    light_path    = nodes.new("ShaderNodeLightPath")

    links.new(light_path.outputs["Is Camera Ray"], mix_node.inputs["Fac"])
    links.new(sun_node.outputs["Color"],           mix_node.inputs["Color1"])
    links.new(bg_color_node.outputs["Color"],      mix_node.inputs["Color2"])
    links.new(mix_node.outputs["Color"],           bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"],       output_node.inputs["Surface"])

    bpy.context.scene.world = world
    return sun_node, bg_color_node


def randomize_lighting(sun_node, bg_color_node):
    """Change sun direction and background color for this render."""
    sun_node.sun_elevation = random.uniform(0.1, 0.6)
    sun_node.sun_rotation  = random.uniform(0, 2 * math.pi)

    r = random.uniform(0.05, 0.95)
    g = random.uniform(0.05, 0.95)
    b = random.uniform(0.05, 0.95)
    bg_color_node.outputs[0].default_value = (r, g, b, 1.0)


# ── Keypoint projection ───────────────────────────────────────────────────────

def project_corners_to_pixels(scene, camera, phone_corners, face_normals):
    """
    Project each 3D corner to 2D pixel coordinates and decide visibility.

    Visibility values (YOLO convention):
      2 = visible         — in frame and face is pointing toward camera
      1 = occluded        — in frame but face is pointing away (hidden side)
      0 = out of frame    — not in the image at all
    """
    projected = []

    for corner, normal in zip(phone_corners, face_normals):
        co = world_to_camera_view(scene, camera, corner)

        pixel_x = co.x * RENDER_WIDTH
        pixel_y = (1.0 - co.y) * RENDER_HEIGHT

        in_frame = 0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0 and co.z > 0

        if not in_frame:
            projected.append((pixel_x, pixel_y, 0))
            continue

        # dot > 0 means the face normal and the direction to camera point the same way
        direction_to_camera = (camera.location - corner).normalized()
        facing_camera = normal.dot(direction_to_camera) > 0

        projected.append((pixel_x, pixel_y, 2 if facing_camera else 1))

    return projected


# ── YOLO annotation ───────────────────────────────────────────────────────────

def build_yolo_label(projected_corners):
    """
    Build one YOLO pose annotation line.

    Format:
      class  bbox_cx bbox_cy bbox_w bbox_h  kp0_x kp0_y kp0_vis  kp1_x ...

    All pixel coordinates are normalized to [0, 1].
    Returns None if no keypoints are visible at all.
    """
    visible_corners = [(px, py) for px, py, vis in projected_corners if vis > 0]
    if not visible_corners:
        return None

    xs = [p[0] for p in visible_corners]
    ys = [p[1] for p in visible_corners]

    # Bounding box with a small 5% padding
    pad_x = (max(xs) - min(xs)) * 0.05
    pad_y = (max(ys) - min(ys)) * 0.05
    x1 = max(0,            min(xs) - pad_x)
    x2 = min(RENDER_WIDTH, max(xs) + pad_x)
    y1 = max(0,             min(ys) - pad_y)
    y2 = min(RENDER_HEIGHT, max(ys) + pad_y)

    bbox_cx = ((x1 + x2) / 2) / RENDER_WIDTH
    bbox_cy = ((y1 + y2) / 2) / RENDER_HEIGHT
    bbox_w  = (x2 - x1)       / RENDER_WIDTH
    bbox_h  = (y2 - y1)       / RENDER_HEIGHT

    parts = [f"0 {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f}"]
    for px, py, vis in projected_corners:
        parts.append(f"{px/RENDER_WIDTH:.6f} {py/RENDER_HEIGHT:.6f} {vis}")

    return " ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("\nLoading phone model...")
    phone_mesh = import_phone_stl()
    phone_corners, face_normals, phone_center = get_phone_corners(phone_mesh)

    print("Setting up camera and lighting...")
    fx, fy, cx, cy = read_calibration_file(CALIBRATION_PATH)
    fov_horizontal = 2 * math.atan(RENDER_WIDTH / (2 * fx))
    camera = add_camera(fov_horizontal)
    sun_node, bg_color_node = setup_lighting()

    enable_optix_gpu()

    scene = bpy.context.scene
    scene.render.engine       = "CYCLES"
    scene.cycles.device       = "GPU"
    scene.cycles.samples      = RENDER_SAMPLES
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT

    print(f"\nGenerating {NUM_RENDERS} renders starting from index {START_INDEX}\n")

    num_skipped = 0

    for i in range(NUM_RENDERS):
        image_stem = f"render_{START_INDEX + i:05d}"
        image_out  = DATASET_DIR / "images" / "train" / f"{image_stem}.png"
        label_out  = DATASET_DIR / "labels" / "train" / f"{image_stem}.txt"

        place_camera_randomly(camera, phone_center)
        randomize_lighting(sun_node, bg_color_node)

        projected_corners = project_corners_to_pixels(scene, camera, phone_corners, face_normals)
        label_line = build_yolo_label(projected_corners)

        if label_line is None:
            num_skipped += 1
            continue

        scene.render.filepath = str(image_out)
        bpy.ops.render.render(write_still=True)
        label_out.write_text(label_line + "\n")

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{NUM_RENDERS}]  skipped so far: {num_skipped}")

    # dataset.yaml — YOLOv8 needs this to know where the data is
    dataset_yaml = f"""\
path: {DATASET_DIR}
train: images/train
val:   images/val

nc: 1
names: ["iphone"]

kpt_shape: [8, 3]
"""
    (DATASET_DIR / "dataset.yaml").write_text(dataset_yaml)

    print(f"\nDone.  Skipped: {num_skipped}")
    print(f"Total train images: {len(list((DATASET_DIR/'images'/'train').glob('*.png')))}")
    print(f"Now run split_train_val.py to create the train/val split.")


main()
