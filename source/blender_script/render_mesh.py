"""Blender render script for TRELLIS-style multiview rendering.

Adapted from TRELLIS/dataset_toolkits/blender_script/render.py for Blender 4.x.
Normalizes the mesh to [-0.5, 0.5]^3, renders multiview images, and exports
the normalized mesh as PLY.

Usage (called by preprocess.py, not directly):
    blender -b -P source/blender_script/render_mesh.py -- \
        --object mesh.obj \
        --views '[{"yaw":0,"pitch":0,"radius":2,"fov":0.698}]' \
        --output_folder /tmp/renders \
        --resolution 512 \
        --engine CYCLES \
        --samples 32 \
        --save_mesh
"""

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, Tuple

import bpy
import numpy as np
from mathutils import Vector


# ── Render init ──────────────────────────────────────────────────────────────


def init_render(engine: str = "CYCLES", resolution: int = 512, samples: int = 32):
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True

    if engine == "BLENDER_EEVEE_NEXT":
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = samples
    else:
        # CYCLES (default)
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        scene.cycles.filter_type = "BOX"
        scene.cycles.filter_width = 1
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transparent_max_bounces = 3
        scene.cycles.transmission_bounces = 3
        scene.cycles.use_denoising = True


# ── Scene management ─────────────────────────────────────────────────────────


def init_scene():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str):
    ext = object_path.split(".")[-1].lower()
    if ext == "obj":
        bpy.ops.wm.obj_import(filepath=object_path)
    elif ext in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif ext == "ply":
        bpy.ops.wm.ply_import(filepath=object_path)
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif ext == "stl":
        bpy.ops.wm.stl_import(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def delete_invisible_objects():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    for col in [c for c in bpy.data.collections if c.hide_viewport]:
        bpy.data.collections.remove(col)


# ── Camera and lighting ──────────────────────────────────────────────────────


def init_camera():
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam


def init_lighting():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Key light
    default_light = bpy.data.objects.new(
        "Default_Light", bpy.data.lights.new("Default_Light", type="POINT")
    )
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)

    # Top light
    top_light = bpy.data.objects.new(
        "Top_Light", bpy.data.lights.new("Top_Light", type="AREA")
    )
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)

    # Bottom light
    bottom_light = bpy.data.objects.new(
        "Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA")
    )
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)


# ── Scene normalization ──────────────────────────────────────────────────────


def scene_bbox() -> Tuple[Vector, Vector]:
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, bpy.types.Mesh):
            continue
        found = True
        for coord in obj.bound_box:
            coord = obj.matrix_world @ Vector(coord)
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("No mesh objects in scene")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Normalize scene to fit in [-0.5, 0.5]^3 unit cube."""
    scene_root_objects = [
        obj for obj in bpy.context.scene.objects.values() if not obj.parent
    ]
    if len(scene_root_objects) > 1:
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    return scale, offset


# ── Mesh processing ──────────────────────────────────────────────────────────


def convert_to_meshes():
    bpy.ops.object.select_all(action="DESELECT")
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objs:
        return
    bpy.context.view_layer.objects.active = mesh_objs[0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")


def triangulate_meshes():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objs:
        return
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def get_transform_matrix(obj) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix


# ── Main ─────────────────────────────────────────────────────────────────────


def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)

    # Initialize
    init_render(engine=arg.engine, resolution=arg.resolution, samples=arg.samples)
    init_scene()
    load_object(arg.object)
    print("[INFO] Scene initialized.")

    # Normalize to unit cube
    scale, offset = normalize_scene()
    print("[INFO] Scene normalized.")

    # Camera and lighting
    cam = init_camera()
    init_lighting()
    print("[INFO] Camera and lighting initialized.")

    # Render views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": [],
    }
    views = json.loads(arg.views)
    for i, view in enumerate(views):
        cam.location = (
            view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"]),
            view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"]),
            view["radius"] * np.sin(view["pitch"]),
        )
        cam.data.lens = 16 / np.tan(view["fov"] / 2)

        bpy.context.scene.render.filepath = os.path.join(
            arg.output_folder, f"{i:03d}.png"
        )
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        to_export["frames"].append({
            "file_path": f"{i:03d}.png",
            "camera_angle_x": view["fov"],
            "transform_matrix": get_transform_matrix(cam),
        })

    # Save camera parameters
    with open(os.path.join(arg.output_folder, "transforms.json"), "w") as f:
        json.dump(to_export, f, indent=4)

    # Export normalized mesh as PLY
    if arg.save_mesh:
        for obj in bpy.context.scene.objects:
            obj.hide_set(False)
        convert_to_meshes()
        triangulate_meshes()
        print("[INFO] Meshes triangulated.")
        bpy.ops.wm.ply_export(filepath=os.path.join(arg.output_folder, "mesh.ply"))

    print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True)
    parser.add_argument("--views", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="/tmp/renders")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--engine", type=str, default="CYCLES",
                        help="CYCLES or BLENDER_EEVEE_NEXT")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--save_mesh", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    main(args)
