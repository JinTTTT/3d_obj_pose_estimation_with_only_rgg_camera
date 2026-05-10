"""
Test pose estimation on a synthetic Blender render.

Step 1 — render one image from a known camera pose (run via Blender)
Step 2 — run YOLO + solvePnP on that image (run via Python)

Step 1:
    blender --background --python tests/verify_pose_on_synthetic.py -- --mode render

Step 2:
    python3 tests/verify_pose_on_synthetic.py --mode predict
"""

import sys
import argparse

ROOT        = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp"
TEST_IMAGE  = f"{ROOT}/test_synthetic.png"
CALIBRATION_PATH = f"{ROOT}/config/camera_calibration.yaml"
MODEL_PATH  = f"{ROOT}/runs/run_100/weights/best.pt"


# ── Shared: 3D keypoint definition (same as run_pose_estimation.py) ───────────

def get_phone_corners_3d():
    import numpy as np
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
    ], dtype="float32")


# ── Step 1: render via Blender ────────────────────────────────────────────────

def render_test_image():
    import bpy, math, re
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view

    # Fixed camera position — we'll compare against this ground truth later
    CAMERA_LOCATION = (80, -250, 120)   # mm

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=f"{ROOT}/cad/iphone13.stl")
    phone_mesh = next(o for o in bpy.data.objects if o.type == "MESH")

    # Load calibration to get FOV
    with open(CALIBRATION_PATH) as f:
        content = f.read()
    data = re.search(r'camera_matrix:.*?data:\s*\[(.*?)\]', content, re.DOTALL)
    values = [float(v.strip()) for v in data.group(1).split(',')]
    fx = values[0]
    fov_horizontal = 2 * math.atan(640 / (2 * fx))

    # Camera
    bpy.ops.object.camera_add(location=CAMERA_LOCATION)
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    camera.data.lens_unit = "FOV"
    camera.data.angle = fov_horizontal

    direction = Vector((0, 0, 0)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    # Lighting
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    out      = nodes.new("ShaderNodeOutputWorld")
    bg       = nodes.new("ShaderNodeBackground")
    sun_node = nodes.new("ShaderNodeTexSky")
    sun_node.sun_elevation = 0.4
    links.new(sun_node.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = world

    # Render
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.filepath = TEST_IMAGE
    bpy.ops.render.render(write_still=True)

    # Print ground truth keypoint positions
    bound_box_corners = [phone_mesh.matrix_world @ Vector(c) for c in phone_mesh.bound_box]
    xs = [c.x for c in bound_box_corners]
    ys = [c.y for c in bound_box_corners]
    zs = [c.z for c in bound_box_corners]
    x_min,x_max = min(xs),max(xs)
    y_min,y_max = min(ys),max(ys)
    z_min,z_max = min(zs),max(zs)

    phone_corners_3d = [
        Vector((x_min, y_max, z_min)), Vector((x_max, y_max, z_min)),
        Vector((x_max, y_min, z_min)), Vector((x_min, y_min, z_min)),
        Vector((x_min, y_max, z_max)), Vector((x_max, y_max, z_max)),
        Vector((x_max, y_min, z_max)), Vector((x_min, y_min, z_max)),
    ]

    print("\nGround truth 2D keypoints (pixels):")
    for i, corner in enumerate(phone_corners_3d):
        co = world_to_camera_view(scene, camera, corner)
        pixel_x = co.x * 640
        pixel_y = (1.0 - co.y) * 480
        print(f"  kp{i}: ({pixel_x:.1f}, {pixel_y:.1f})")

    print(f"\nCamera location (ground truth): {CAMERA_LOCATION} mm")
    print(f"Image saved: {TEST_IMAGE}")


# ── Step 2: predict via YOLO + solvePnP ──────────────────────────────────────

def run_pose_estimation():
    import cv2
    import numpy as np
    import time
    from ultralytics import YOLO

    # Load calibration
    fs = cv2.FileStorage(CALIBRATION_PATH, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs   = fs.getNode("distortion_coefficients").mat()
    fs.release()

    phone_corners_3d = get_phone_corners_3d()
    yolo_model = YOLO(MODEL_PATH)

    image = cv2.imread(TEST_IMAGE)

    t0 = time.perf_counter()
    detections = yolo_model(image, verbose=False)[0]
    yolo_ms = (time.perf_counter() - t0) * 1000

    if detections.keypoints is None or len(detections.keypoints) == 0:
        print("No phone detected in the image.")
        return

    keypoint_detection = detections.keypoints[0]
    detected_corners_2d = keypoint_detection.xy[0].cpu().numpy().astype(np.float32)

    print("\nYOLO detected 2D keypoints (pixels):")
    for i, (px, py) in enumerate(detected_corners_2d):
        print(f"  kp{i}: ({px:.1f}, {py:.1f})")

    t1 = time.perf_counter()
    success, rvec, tvec = cv2.solvePnP(
        phone_corners_3d, detected_corners_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    pnp_ms = (time.perf_counter() - t1) * 1000

    if not success:
        print("solvePnP failed.")
        return

    translation = tvec.ravel()
    rotation_deg = np.degrees(rvec.ravel())

    print(f"\nPredicted tvec [m]: x={translation[0]:.4f}  y={translation[1]:.4f}  z={translation[2]:.4f}")
    print(f"Predicted rvec [deg]: {rotation_deg[0]:.2f}  {rotation_deg[1]:.2f}  {rotation_deg[2]:.2f}")
    print(f"Predicted distance from camera: {np.linalg.norm(translation)*1000:.1f} mm")
    print(f"\nInference timing — YOLO: {yolo_ms:.1f} ms  PnP: {pnp_ms:.2f} ms")

    # Draw and show
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
    for i, (px, py) in enumerate(detected_corners_2d):
        cv2.circle(image, (int(px), int(py)), 5, (0, 255, 0), -1)
        cv2.putText(image, str(i), (int(px)+5, int(py)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    out_path = TEST_IMAGE.replace(".png", "_result.png")
    cv2.imwrite(out_path, image)
    print(f"\nResult image saved: {out_path}")
    cv2.imshow("Synthetic test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

# When running inside Blender, sys.argv contains all of Blender's own arguments.
# We only want the arguments after the "--" separator.
if "--" in sys.argv:
    script_args = sys.argv[sys.argv.index("--") + 1:]
else:
    script_args = sys.argv[1:]

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["render", "predict"], required=True)
args = parser.parse_args(script_args)

if args.mode == "render":
    render_test_image()
else:
    run_pose_estimation()
