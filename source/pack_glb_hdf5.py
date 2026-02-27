"""Pack GLB animation files into HDF5 with train/val split.

Directory layout (input):  data/glb/{animal}/{name}.glb
Each GLB: single skeletal animation with mesh, stored as raw bytes.

HDF5 layout (output):
    {animal}/{glb_name}/glb        (N,) uint8  — raw GLB bytes
    data_split/train               string[]  — paths like "animal/glb_name"
    data_split/val                 string[]

Split strategy: natsort all sequence paths, fixed-seed shuffle, 80/20 split.

Usage:
    uv run python source/pack_glb_hdf5.py
    uv run python source/pack_glb_hdf5.py --glb-dir data/glb --output data_glb.hdf5 --workers 16
"""

import dataclasses
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import tyro
from natsort import natsorted
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ──────────────────────────────────────────────────────────────────


def read_one_glb(glb_path: Path) -> tuple[str, str, bytes] | None:
    """Read a single GLB. Returns (animal, clip_name, raw_bytes) or None."""
    try:
        raw = glb_path.read_bytes()
        return (glb_path.parent.name, glb_path.stem, raw)
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

    batch_size: int = 256
    """Number of files to read per batch (controls peak memory)."""

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

    # 3. Pack into HDF5 — batched parallel reads to bound memory
    errors = 0

    with h5py.File(args.output, "w") as hf:
        pbar = tqdm(total=len(all_glbs), desc="Packing", unit="glb")

        for batch_start in range(0, len(all_glbs), args.batch_size):
            batch = all_glbs[batch_start : batch_start + args.batch_size]

            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(read_one_glb, batch):
                    pbar.update(1)
                    if result is None:
                        errors += 1
                        continue

                    animal, name, raw = result
                    grp = hf.create_group(f"{animal}/{name}")
                    grp.create_dataset("glb", data=np.frombuffer(raw, dtype=np.uint8), compression="lzf")

        pbar.close()

        # 4. Write split info + metadata
        dt = h5py.string_dtype()
        split_grp = hf.create_group("data_split")
        split_grp.create_dataset("train", data=np.array(train_paths, dtype=object), dtype=dt)
        split_grp.create_dataset("val", data=np.array(val_paths, dtype=object), dtype=dt)

        hf.attrs["format"] = "raw_glb"
        hf.attrs["n_train"] = len(train_paths)
        hf.attrs["n_val"] = len(val_paths)
        hf.attrs["seed"] = args.seed

    print(f"\nDone: {args.output}")
    print(f"  Train: {len(train_paths):,}  Val: {len(val_paths):,}  Errors: {errors}")
    print(f"  File size: {args.output.stat().st_size / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main(tyro.cli(Args))
