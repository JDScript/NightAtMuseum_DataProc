"""Pack pickle mesh sequences into Zarr with train/val split.

Directory layout (input):  data_pickle/{animal}/{name}.pkl
Each pkl: {"vertices": (F, V, 3) float64, "faces": (N, 3) int64}

Zarr layout (output):  data_pickle.zarr/
    {animal}/{seq_name}/vertices   (F, V, 3) float32  — chunked per frame
    {animal}/{seq_name}/faces      (N, 3)    int32
    attrs: train_split, val_split   list[str] — paths like "animal/seq_name"

Split strategy: natsort all sequence paths, fixed-seed shuffle, 80/20 split.

Zarr advantages over HDF5 for this data:
  - Parallel writes: multiple workers can write different arrays simultaneously
  - Per-chunk compression (zstd via blosc): vertices chunked per-frame for random access
  - Directory store: each array is a directory, friendly to parallel file systems (Lustre/GPFS)

Usage:
    uv run python source/pack_pickle_zarr.py
    uv run python source/pack_pickle_zarr.py --pkl-dir data_pickle --output data_pickle.zarr --workers 8
"""

import dataclasses
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tyro
import zarr
from natsort import natsorted
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ──────────────────────────────────────────────────────────────────


def read_one_pkl(pkl_path: Path) -> tuple[str, str, np.ndarray, np.ndarray] | None:
    """Read a single pickle. Returns (animal, seq_name, vertices, faces) or None."""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        verts = data["vertices"].astype(np.float32)  # (F, V, 3)
        faces = data["faces"].astype(np.int32)  # (N, 3)
        animal = pkl_path.parent.name
        seq_name = pkl_path.stem
        return (animal, seq_name, verts, faces)
    except Exception as e:
        print(f"  WARN: failed to read {pkl_path}: {e}")
        return None


def write_one_seq(
    store_path: str,
    animal: str,
    seq_name: str,
    verts: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Write a single sequence into the zarr store (worker-safe for directory store)."""
    root = zarr.open_group(store_path, mode="r+")
    grp = root.require_group(f"{animal}/{seq_name}")

    # Vertices: chunk per frame for efficient single-frame access
    grp.create_array(
        "vertices",
        data=verts,
        chunks=(1, verts.shape[1], 3),
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        overwrite=True,
    )

    # Faces: single chunk (shared across frames, typically small)
    grp.create_array(
        "faces",
        data=faces,
        chunks=faces.shape,
        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
        overwrite=True,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """Pack pickle mesh sequences into Zarr."""

    pkl_dir: Path = PROJECT_ROOT / "data_pickle"
    """Input directory with {animal}/{name}.pkl structure."""

    output: Path = PROJECT_ROOT / "data_pickle.zarr"
    """Output Zarr directory store path."""

    workers: int = 8
    """Number of parallel workers for reading and writing."""

    batch_size: int = 32
    """Number of files to read per batch (controls peak memory)."""

    val_ratio: float = 0.2
    """Fraction of sequences held out for validation."""

    seed: int = 42
    """Random seed for split."""


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: Args):
    # 1. Discover all pickle files, natsort for deterministic ordering
    all_pkls = natsorted(args.pkl_dir.glob("*/*.pkl"), key=str)
    if not all_pkls:
        print(f"No pickle files found in {args.pkl_dir}")
        return
    print(f"Found {len(all_pkls):,} pickle files")

    # 2. Build path list, shuffle with fixed seed, split
    all_paths = [f"{p.parent.name}/{p.stem}" for p in all_pkls]
    indices = list(range(len(all_pkls)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * args.val_ratio))
    train_paths = [all_paths[i] for i in indices[n_val:]]
    val_paths = [all_paths[i] for i in indices[:n_val]]
    print(f"Split: {len(train_paths):,} train, {len(val_paths):,} val")

    # 3. Create zarr store
    store_path = str(args.output)
    root = zarr.open_group(store_path, mode="w")

    # 4. Pack — batched parallel reads, then parallel writes
    errors = 0
    pbar = tqdm(total=len(all_pkls), desc="Packing", unit="seq")

    for batch_start in range(0, len(all_pkls), args.batch_size):
        batch = all_pkls[batch_start : batch_start + args.batch_size]

        # Read batch in parallel
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(read_one_pkl, batch):
                if result is None:
                    errors += 1
                    pbar.update(1)
                    continue
                results.append(result)
                pbar.update(1)

        # Write batch in parallel (zarr directory store supports concurrent writes
        # to different arrays since each writes to separate files)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for animal, seq_name, verts, faces in results:
                fut = pool.submit(write_one_seq, store_path, animal, seq_name, verts, faces)
                futures.append(fut)
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc is not None:
                    print(f"  WARN: write failed: {exc}")
                    errors += 1

    pbar.close()

    # 5. Write split as JSON attrs (avoids zarr v3 unstable string dtype warning)
    root = zarr.open_group(store_path, mode="r+")
    root.attrs["format"] = "mesh_sequence"
    root.attrs["train_split"] = train_paths
    root.attrs["val_split"] = val_paths
    root.attrs["n_train"] = len(train_paths)
    root.attrs["n_val"] = len(val_paths)
    root.attrs["seed"] = args.seed

    # 6. Summary
    print(f"\nDone: {args.output}")
    print(f"  Train: {len(train_paths):,}  Val: {len(val_paths):,}  Errors: {errors}")

    total_bytes = sum(f.stat().st_size for f in Path(store_path).rglob("*") if f.is_file())
    print(f"  Store size: {total_bytes / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main(tyro.cli(Args))
