"""Headless Blender script to import ms2 + manis and export per-animation GLBs.

Naming convention (compatible with AniMo):
    {ms2_stem}_{manis_stem}_{action_name}.glb
    e.g. aardvark_female__animationmotionextractedbehaviour.manisete15d87f2_aardvark_female_enrichmentboxshake.glb

Usage:
    blender --background --python source/blender_export_glb.py -- \
        --ms2 data/ovl/Aardvark_Male.ovl/aardvark_male_.ms2 \
        --manis data/ovl/Aardvark_Male.ovl/animationmotionextractedlocomotion.manisetcb366443.manis \
        --outdir data/glb_test
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── cobra-tools on sys.path ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COBRA_ROOT = os.path.join(PROJECT_ROOT, "third_party", "cobra-tools")
sys.path.insert(0, COBRA_ROOT)

import bpy  # noqa: E402  (available inside Blender)


# ── register cobra-tools addon so scene.cobra etc. exist ─────────────
def register_cobra_addon():
    """Register the cobra-tools Blender addon."""
    import importlib
    addon = importlib.import_module("__init__")
    addon.register()
    log.info("cobra-tools addon registered")


class Reporter:
    """Minimal reporter expected by cobra-tools import functions."""
    def show_info(self, msg):
        log.info(msg)

    def show_warning(self, msg):
        log.warning(msg)

    def show_error(self, msg):
        log.error(msg)


def clear_scene():
    """Remove all objects, meshes, armatures, actions from the default scene."""
    bpy.ops.wm.read_homefile(use_empty=True)


def import_ms2(filepath: str):
    from plugin.import_ms2 import load as load_ms2
    abs_path = os.path.abspath(filepath)
    log.info(f"Importing MS2: {abs_path}")
    load_ms2(Reporter(), filepath=abs_path)


def import_manis(filepath: str):
    from plugin.import_manis import load as load_manis
    abs_path = os.path.abspath(filepath)
    log.info(f"Importing MANIS: {abs_path}")
    load_manis(Reporter(), filepath=abs_path, disable_ik=True)


def clean_scene_for_export():
    """Remove physics meshes, empties, and non-L0 LODs — keep only armature + L0 meshes."""
    scene = bpy.context.scene

    keep = set()
    for obj in scene.objects:
        if obj.type == 'ARMATURE':
            keep.add(obj.name)
        elif obj.type == 'MESH' and any("_L0" in c.name for c in obj.users_collection):
            keep.add(obj.name)

    to_delete = [obj for obj in scene.objects if obj.name not in keep]
    log.info(f"Cleaning scene: removing {len(to_delete)} objects, keeping {len(keep)}: {sorted(keep)}")

    # Use bpy.data.objects.remove() — reliable in headless mode unlike bpy.ops.object.delete()
    for obj in to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Clean up orphan mesh/empty data
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    # Remove empty collections
    for coll in list(bpy.data.collections):
        if not coll.objects and not coll.children:
            bpy.data.collections.remove(coll)

    remaining = [f"{obj.name}({obj.type})" for obj in scene.objects]
    log.info(f"Scene after cleanup: {len(remaining)} objects: {remaining}")


def get_armature():
    """Find the armature object in the scene."""
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def export_actions(armature, actions, outdir, ms2_stem, manis_stem):
    """Export a list of actions as individual GLBs with AniMo-compatible naming."""
    scene = bpy.context.scene
    exported = 0

    for action in actions:
        action_name = action.name
        safe_action = action_name.replace("@", "_").replace(" ", "_")
        # AniMo naming: {ms2_stem}_{manis_stem}_{action_name}.glb
        out_name = f"{ms2_stem}_{manis_stem}_{safe_action}.glb"
        out_path = os.path.join(outdir, out_name)

        if not armature.animation_data:
            armature.animation_data_create()
        armature.animation_data.action = action

        frame_start, frame_end = action.frame_range
        scene.frame_start = int(frame_start)
        scene.frame_end = int(frame_end)

        bpy.ops.export_scene.gltf(
            filepath=out_path,
            export_format='GLB',
            export_animations=True,
            export_animation_mode='ACTIVE_ACTIONS',
            export_anim_single_armature=True,
            export_nla_strips=False,
        )
        exported += 1

    return exported


def main():
    # Parse args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Export ms2+manis to per-animation GLBs")
    parser.add_argument("--ms2", required=True, help="Path to .ms2 file")
    parser.add_argument("--manis", nargs="+", required=True, help="Paths to .manis files")
    parser.add_argument("--outdir", required=True, help="Output directory for GLBs")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Register addon
    register_cobra_addon()

    # 2. Clear scene
    clear_scene()

    # 3. Import mesh + skeleton
    import_ms2(args.ms2)

    # 4. Clean scene before animations (remove physics, empties, high LODs)
    clean_scene_for_export()

    armature = get_armature()
    if not armature:
        log.error("No armature found in scene!")
        return

    # Derive naming components
    # ms2_stem: "aardvark_female_" (keep trailing underscore for AniMo compat)
    ms2_stem = os.path.splitext(os.path.basename(args.ms2))[0]
    # animal_name for action filtering: "aardvark_female" (stripped)
    animal_prefix = ms2_stem.rstrip("_").lower()

    total_exported = 0

    # 5. Import manis one at a time, track new actions per manis
    for manis_path in args.manis:
        actions_before = set(bpy.data.actions)

        import_manis(manis_path)

        actions_after = set(bpy.data.actions)
        new_actions = actions_after - actions_before

        # Filter: only actions belonging to this animal's armature
        matched = [a for a in new_actions if animal_prefix in a.name.lower()]
        if not matched and new_actions:
            log.warning(f"No actions matching '{animal_prefix}' in {os.path.basename(manis_path)} "
                        f"({len(new_actions)} new actions), skipping")
            continue

        # manis_stem: "animationmotionextractedbehaviour.manisete15d87f2" (without .manis)
        manis_basename = os.path.basename(manis_path)
        manis_stem = manis_basename.rsplit(".manis", 1)[0]

        n = export_actions(armature, matched, args.outdir, ms2_stem, manis_stem)
        total_exported += n
        log.info(f"  {manis_basename}: {n} GLBs exported ({len(new_actions)} new, {len(matched)} matched)")

    log.info(f"Done. {total_exported} GLBs written to {args.outdir}")


if __name__ == "__main__":
    main()
