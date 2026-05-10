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

ROOT             = Path("/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp")
STL_PATH         = ROOT / "cad"    / "iphone13.stl"
CALIBRATION_PATH = ROOT / "config" / "camera_calibration.yaml"
DATASET_DIR      = ROOT / "dataset"

# ── Settings ──────────────────────────────────────────────────────────────────

NUM_RENDERS    = 100
START_INDEX    = 0
RENDER_WIDTH   = 640
RENDER_HEIGHT  = 480
RENDER_SAMPLES = 256

# Camera distance range (matches real-world capture at ~50 cm)
MIN_DISTANCE_MM = 150
MAX_DISTANCE_MM = 350

# How many times to retry placing the camera before skipping a render
MAX_CAMERA_RETRIES = 20


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
    phone_mesh = next(o for o in bpy.data.objects if o.type == "MESH")

    # Principled BSDF material — gives specular highlights and shadows that
    # make the 3D shape of the phone clearly visible
    mat = bpy.data.materials.new("PhoneMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.85, 1.0)  # light gray
    bsdf.inputs["Roughness"].default_value  = 0.6   # matte — no harsh reflections
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    phone_mesh.data.materials.clear()
    phone_mesh.data.materials.append(mat)

    return phone_mesh


def get_phone_corners(phone_mesh):
    """
    Build the 8 bounding box corners from the phone mesh.
    Called every render because the phone rotates each frame.

    Keypoint order (must stay consistent with solvePnP later):
      kp0–3 : screen face (z_min), top-left → top-right → bottom-right → bottom-left
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


def set_phone_flat_screen_up(phone_mesh):
    """Fix the phone lying flat with screen facing upward. Never changes during rendering."""
    # Default STL has screen facing -Z (downward). Keep default rotation.
    phone_mesh.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()


def place_camera_randomly(camera, phone_center):
    """Orbit the camera around the phone — varying angle and direction."""
    elevation = math.radians(random.uniform(10, 80))  # above desk level only
    azimuth   = random.uniform(0, 2 * math.pi)         # full 360° around
    distance  = random.uniform(MIN_DISTANCE_MM, MAX_DISTANCE_MM)

    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)

    camera.location = phone_center + Vector((x, y, z))
    direction = phone_center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def phone_is_fully_in_frame(scene, camera, phone_corners):
    """Return True only if every corner that is in front of the camera is within frame."""
    for corner in phone_corners:
        co = world_to_camera_view(scene, camera, corner)
        if co.z <= 0:
            continue  # behind camera — not visible, don't care
        if not (0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0):
            return False
    return True


# ── World / lighting ──────────────────────────────────────────────────────────

def setup_lighting():
    """
    Sky texture for lighting, black background.
    Returns the sun node so we can randomize it each render.
    """
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output_node = nodes.new("ShaderNodeOutputWorld")
    bg_node     = nodes.new("ShaderNodeBackground")
    mix_node    = nodes.new("ShaderNodeMixRGB")
    sun_node    = nodes.new("ShaderNodeTexSky")
    black_node  = nodes.new("ShaderNodeRGB")
    light_path  = nodes.new("ShaderNodeLightPath")

    # Camera rays see black; lighting rays use the sky texture
    black_node.outputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    links.new(light_path.outputs["Is Camera Ray"], mix_node.inputs["Fac"])
    links.new(sun_node.outputs["Color"],           mix_node.inputs["Color1"])
    links.new(black_node.outputs["Color"],         mix_node.inputs["Color2"])
    links.new(mix_node.outputs["Color"],           bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"],       output_node.inputs["Surface"])
    bg_node.inputs["Strength"].default_value = 0.6

    bpy.context.scene.world = world
    return sun_node


def randomize_lighting(sun_node):
    """Randomize sun position — low to mid angle to avoid overexposure."""
    sun_node.sun_elevation = random.uniform(0.05, 0.5)      # 3° to 29° — never directly overhead
    sun_node.sun_rotation  = random.uniform(0, 2 * math.pi) # all compass directions


# ── Keypoint projection ───────────────────────────────────────────────────────

def project_corners_to_pixels(scene, camera, phone_corners, face_normals):
    """
    Project each 3D corner to 2D pixel coordinates and decide visibility.

    Visibility values (YOLO convention):
      2 = visible   — in frame and face pointing toward camera
      1 = occluded  — in frame but face pointing away
      0 = out of frame
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

        direction_to_camera = (camera.location - corner).normalized()
        facing_camera = normal.dot(direction_to_camera) > 0

        projected.append((pixel_x, pixel_y, 2 if facing_camera else 1))

    return projected


# ── YOLO annotation ───────────────────────────────────────────────────────────

def build_yolo_label(projected_corners):
    """
    Build one YOLO pose annotation line.
    Returns None if no keypoints are visible at all.
    """
    visible_corners = [(px, py) for px, py, vis in projected_corners if vis > 0]
    if not visible_corners:
        return None

    xs = [p[0] for p in visible_corners]
    ys = [p[1] for p in visible_corners]

    pad_x = (max(xs) - min(xs)) * 0.05
    pad_y = (max(ys) - min(ys)) * 0.05
    x1 = max(0,             min(xs) - pad_x)
    x2 = min(RENDER_WIDTH,  max(xs) + pad_x)
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

    print("Setting up camera and lighting...")
    fx, fy, cx, cy = read_calibration_file(CALIBRATION_PATH)
    fov_horizontal = 2 * math.atan(RENDER_WIDTH / (2 * fx))
    camera = add_camera(fov_horizontal)
    sun_node = setup_lighting()

    enable_optix_gpu()

    scene = bpy.context.scene
    scene.render.engine       = "CYCLES"
    scene.cycles.device       = "GPU"
    scene.cycles.samples      = RENDER_SAMPLES
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT

    set_phone_flat_screen_up(phone_mesh)
    phone_corners, face_normals, phone_center = get_phone_corners(phone_mesh)

    print(f"\nGenerating {NUM_RENDERS} renders\n")

    num_saved   = 0
    num_skipped = 0

    while num_saved < NUM_RENDERS:

        # Try multiple camera positions until the whole phone fits in frame
        placed = False
        for _ in range(MAX_CAMERA_RETRIES):
            place_camera_randomly(camera, phone_center)
            if phone_is_fully_in_frame(scene, camera, phone_corners):
                placed = True
                break

        if not placed:
            num_skipped += 1
            continue

        randomize_lighting(sun_node)

        projected_corners = project_corners_to_pixels(scene, camera, phone_corners, face_normals)
        label_line = build_yolo_label(projected_corners)

        if label_line is None:
            num_skipped += 1
            continue

        image_stem = f"render_{START_INDEX + num_saved:05d}"
        image_out  = DATASET_DIR / "images" / "train" / f"{image_stem}.png"
        label_out  = DATASET_DIR / "labels" / "train" / f"{image_stem}.txt"

        scene.render.filepath = str(image_out)
        bpy.ops.render.render(write_still=True)
        label_out.write_text(label_line + "\n")

        num_saved += 1
        if num_saved % 100 == 0:
            print(f"  [{num_saved}/{NUM_RENDERS}]  skipped so far: {num_skipped}")

    dataset_yaml = f"""\
path: {DATASET_DIR}
train: images/train
val:   images/val

nc: 1
names: ["iphone"]

kpt_shape: [8, 3]
"""
    (DATASET_DIR / "dataset.yaml").write_text(dataset_yaml)

    print(f"\nDone. Saved: {num_saved}  Skipped: {num_skipped}")
    print(f"Now run split_train_val.py to create the train/val split.")


main()
