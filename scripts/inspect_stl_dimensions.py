"""
Print the bounding box dimensions of the iPhone STL and list the 8 corner keypoints.

Run with:
    blender --background --python scripts/inspect_stl_dimensions.py
"""

import bpy
from mathutils import Vector

STL_PATH = "/home/jtao/workspace/object_detection_ws/3d_obj_pose_estimation_with_yolo_and_pnp/cad/iphone13.stl"

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.stl_import(filepath=STL_PATH)

phone_mesh = next(o for o in bpy.data.objects if o.type == "MESH")

bound_box_corners = [phone_mesh.matrix_world @ Vector(c) for c in phone_mesh.bound_box]

xs = [c.x for c in bound_box_corners]
ys = [c.y for c in bound_box_corners]
zs = [c.z for c in bound_box_corners]

x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)
z_min, z_max = min(zs), max(zs)

print(f"X : {x_min:.2f}  to  {x_max:.2f}   (width  = {x_max - x_min:.2f} mm)")
print(f"Y : {y_min:.2f}  to  {y_max:.2f}   (height = {y_max - y_min:.2f} mm)")
print(f"Z : {z_min:.2f}  to  {z_max:.2f}   (depth  = {z_max - z_min:.2f} mm)")
print()
print("8 bounding box corners (our keypoints):")
print(f"  kp0  screen top-left      ({x_min:.2f}, {y_max:.2f}, {z_min:.2f})")
print(f"  kp1  screen top-right     ({x_max:.2f}, {y_max:.2f}, {z_min:.2f})")
print(f"  kp2  screen bottom-right  ({x_max:.2f}, {y_min:.2f}, {z_min:.2f})")
print(f"  kp3  screen bottom-left   ({x_min:.2f}, {y_min:.2f}, {z_min:.2f})")
print(f"  kp4  back   top-left      ({x_min:.2f}, {y_max:.2f}, {z_max:.2f})")
print(f"  kp5  back   top-right     ({x_max:.2f}, {y_max:.2f}, {z_max:.2f})")
print(f"  kp6  back   bottom-right  ({x_max:.2f}, {y_min:.2f}, {z_max:.2f})")
print(f"  kp7  back   bottom-left   ({x_min:.2f}, {y_min:.2f}, {z_max:.2f})")
