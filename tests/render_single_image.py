"""
Render one image and print its projected keypoint positions to verify the pipeline.

Run with:
    blender --background --python tests/render_single_image.py
"""

import bpy
import math
import re
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

STL_PATH   = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/cad/iphone13.stl"
OUTPUT_IMAGE  = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/test_render.png"
CALIBRATION_PATH = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/config/camera_calibration.yaml"

RENDER_WIDTH  = 640
RENDER_HEIGHT = 480


def read_calibration_file(path):
    """
    Read fx, fy, cx, cy from our OpenCV calibration YAML.
    We parse it manually because cv2 is not available inside Blender.
    """
    with open(path) as f:
        content = f.read()

    # The camera matrix is stored as a flat list of 9 values: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    data = re.search(r'camera_matrix:.*?data:\s*\[(.*?)\]', content, re.DOTALL)
    values = [float(v.strip()) for v in data.group(1).split(',')]

    fx = values[0]
    fy = values[4]
    cx = values[2]
    cy = values[5]

    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    return fx, fy, cx, cy


# ── Step 1: load the phone model ──────────────────────────────────────────────

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.stl_import(filepath=STL_PATH)

phone_mesh = next(o for o in bpy.data.objects if o.type == "MESH")

# ── Step 2: define the 8 keypoints from the bounding box ─────────────────────

bound_box_corners = [phone_mesh.matrix_world @ Vector(c) for c in phone_mesh.bound_box]
xs = [c.x for c in bound_box_corners]
ys = [c.y for c in bound_box_corners]
zs = [c.z for c in bound_box_corners]

x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)
z_min, z_max = min(zs), max(zs)

phone_corners_3d = [
    Vector((x_min, y_max, z_min)),  # kp0  screen top-left
    Vector((x_max, y_max, z_min)),  # kp1  screen top-right
    Vector((x_max, y_min, z_min)),  # kp2  screen bottom-right
    Vector((x_min, y_min, z_min)),  # kp3  screen bottom-left
    Vector((x_min, y_max, z_max)),  # kp4  back top-left
    Vector((x_max, y_max, z_max)),  # kp5  back top-right
    Vector((x_max, y_min, z_max)),  # kp6  back bottom-right
    Vector((x_min, y_min, z_max)),  # kp7  back bottom-left
]

# ── Step 3: set up camera ─────────────────────────────────────────────────────

fx, fy, cx, cy = read_calibration_file(CALIBRATION_PATH)

# Convert fx (pixels) to horizontal field of view (radians) for Blender
# FOV = 2 * arctan(image_width / (2 * fx))
fov_horizontal = 2 * math.atan(RENDER_WIDTH / (2 * fx))
print(f"  FOV horizontal = {math.degrees(fov_horizontal):.1f} degrees")

bpy.ops.object.camera_add(location=(200, -150, 30))
camera = bpy.context.active_object
bpy.context.scene.camera = camera

camera.data.lens_unit = "FOV"
camera.data.angle     = fov_horizontal

# Point the camera toward the phone (which is at the origin)
direction = Vector((0, 0, 0)) - camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

# ── Step 4: lighting + background setup ──────────────────────────────────────
#
# Two-layer world setup:
#   - SKY TEXTURE  →  used for lighting the phone (indirect rays)
#   - SOLID COLOR  →  used for what the camera sees as background
#
# The Light Path node separates the two: "Is Camera Ray" = 1 for background
# pixels the camera sees directly, 0 for lighting rays that bounce off objects.

world = bpy.data.worlds.new("Sky")
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

output_node  = nodes.new("ShaderNodeOutputWorld")
bg_node      = nodes.new("ShaderNodeBackground")
mix_node     = nodes.new("ShaderNodeMixRGB")
sun_node     = nodes.new("ShaderNodeTexSky")
bg_color_node = nodes.new("ShaderNodeRGB")
light_path   = nodes.new("ShaderNodeLightPath")

sun_node.sun_elevation = 0.6
sun_node.sun_rotation  = 0.5

bg_color_node.outputs[0].default_value = (0.8, 0.2, 0.2, 1.0)  # red for easy checking

links.new(light_path.outputs["Is Camera Ray"], mix_node.inputs["Fac"])
links.new(sun_node.outputs["Color"],           mix_node.inputs["Color1"])
links.new(bg_color_node.outputs["Color"],      mix_node.inputs["Color2"])
links.new(mix_node.outputs["Color"],           bg_node.inputs["Color"])
links.new(bg_node.outputs["Background"],       output_node.inputs["Surface"])

bpy.context.scene.world = world

# ── Step 5: render ────────────────────────────────────────────────────────────

scene = bpy.context.scene
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == "OPTIX"

scene.render.engine       = "CYCLES"
scene.cycles.device       = "GPU"
scene.cycles.samples      = 64
scene.render.resolution_x = RENDER_WIDTH
scene.render.resolution_y = RENDER_HEIGHT
scene.render.filepath     = OUTPUT_IMAGE

bpy.ops.render.render(write_still=True)
print(f"Image saved to: {OUTPUT_IMAGE}")

# ── Step 6: print projected keypoint positions ────────────────────────────────

print("\nProjected keypoint positions:")
for i, corner in enumerate(phone_corners_3d):
    co = world_to_camera_view(scene, camera, corner)
    pixel_x = co.x * RENDER_WIDTH
    pixel_y = (1.0 - co.y) * RENDER_HEIGHT
    print(f"  kp{i}: pixel ({pixel_x:.1f}, {pixel_y:.1f})  depth={co.z:.1f} mm")
