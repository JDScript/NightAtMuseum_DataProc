"""Pack GLB animation files into HDF5 with train/val split.

Directory layout (input):  data/glb/{animal}/{name}.glb
Each GLB: single skeletal animation with mesh.

HDF5 layout (output):
    {animal}/{glb_name}/glb        (N,) uint8  — raw GLB bytes
    {animal}/{glb_name}/            attrs: frame_count, duration, n_vertices, n_faces
    data_split/train               string[]  — paths like "animal/glb_name"
    data_split/val                 string[]

Split strategy: natsort all sequence paths, fixed-seed shuffle, 80/20 split.

Usage:
    uv run python source/pack_glb_hdf5.py
    uv run python source/pack_glb_hdf5.py --glb-dir data/glb --output data_glb.hdf5 --workers 16
"""

import dataclasses
import json
import random
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import tyro
from natsort import natsorted
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ──────────────────────────────────────────────────────────────────


def read_one_glb(glb_path: Path) -> tuple[str, str, bytes, dict] | None:
    """Read a single GLB. Returns (animal, name, raw_bytes, metadata) or None."""
    try:
        raw = glb_path.read_bytes()
        # Parse JSON chunk for metadata
        meta = {}
        if len(raw) >= 20:
            magic, _ver, _total = struct.unpack_from("<III", raw, 0)
            if magic == 0x46546C67:
                chunk_len, chunk_type = struct.unpack_from("<II", raw, 12)
                if chunk_type == 0x4E4F534A:
                    gltf = json.loads(raw[20 : 20 + chunk_len])
                    # Extract animation info
                    anims = gltf.get("animations", [])
                    accessors = gltf.get("accessors", [])
                    if anims:
                        anim = anims[0]
                        max_time = 0.0
                        max_frames = 0
                        for s in anim.get("samplers", []):
                            idx = s.get("input")
                            if idx is not None and idx < len(accessors):
                                acc = accessors[idx]
                                t = acc.get("max", [0])
                                if isinstance(t, list) and t:
                                    max_time = max(max_time, t[0])
                                max_frames = max(max_frames, acc.get("count", 0))
                        meta["frame_count"] = max_frames
                        meta["duration"] = max_time
                    # Mesh info
                    meshes = gltf.get("meshes", [])
                    if meshes:
                        prim = meshes[0].get("primitives", [{}])[0]
                        pos_idx = prim.get("attributes", {}).get("POSITION")
                        idx_idx = prim.get("indices")
                        if pos_idx is not None and pos_idx < len(accessors):
                            meta["n_vertices"] = accessors[pos_idx].get("count", 0)
                        if idx_idx is not None and idx_idx < len(accessors):
                            meta["n_faces"] = accessors[idx_idx].get("count", 0) // 3

        animal = glb_path.parent.name
        name = glb_path.stem
        return (animal, name, raw, meta)
    except Exception as e:
        print(f"  WARN: failed to read {glb_path}: {e}")
        return None


# ── CLI ──────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """Pack GLB animations into HDF5."""

    glb_dir: Path = PROJECT_ROOT / "data" / "glb"
    """Input directory with {animal}/{name}.glb structure."""

    output: Path = PROJECT_ROOT / "data_glb.hdf5"
    """Output HDF5 file path."""

    workers: int = 8
    """Number of parallel workers for reading GLBs."""

    val_ratio: float = 0.2
    """Fraction of sequences held out for validation."""

    seed: int = 42
    """Random seed for split."""


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: Args):
    # 1. Discover all GLB files, natsort for deterministic ordering
    all_glbs = natsorted(args.glb_dir.glob("*/*.glb"), key=str)
    if not all_glbs:
        print(f"No GLB files found in {args.glb_dir}")
        return
    print(f"Found {len(all_glbs):,} GLB files")

    # 2. Build path list, shuffle with fixed seed, split
    all_paths = [f"{p.parent.name}/{p.stem}" for p in all_glbs]
    indices = list(range(len(all_glbs)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * args.val_ratio))

    train_paths = [all_paths[i] for i in indices[n_val:]]
    val_paths = [all_paths[i] for i in indices[:n_val]]
    print(f"Split: {len(train_paths):,} train, {len(val_paths):,} val")

    # 3. Pack into HDF5 with multiprocess reading
    errors = 0

    with h5py.File(args.output, "w") as hf:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(read_one_glb, p): p for p in all_glbs}
            pbar = tqdm(total=len(all_glbs), desc="Packing", unit="glb")

            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                if result is None:
                    errors += 1
                    continue

                animal, name, raw, meta = result
                grp_path = f"{animal}/{name}"

                grp = hf.create_group(grp_path)
                grp.create_dataset("glb", data=np.frombuffer(raw, dtype=np.uint8))
                for k, v in meta.items():
                    grp.attrs[k] = v

            pbar.close()

        # 4. Write split info
        dt = h5py.string_dtype()
        split_grp = hf.create_group("data_split")
        split_grp.create_dataset("train", data=np.array(train_paths, dtype=object), dtype=dt)
        split_grp.create_dataset("val", data=np.array(val_paths, dtype=object), dtype=dt)

        hf.attrs["n_train"] = len(train_paths)
        hf.attrs["n_val"] = len(val_paths)
        hf.attrs["seed"] = args.seed

    print(f"\nDone: {args.output}")
    print(f"  Train: {len(train_paths):,}  Val: {len(val_paths):,}  Errors: {errors}")
    print(f"  File size: {args.output.stat().st_size / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main(tyro.cli(Args))
