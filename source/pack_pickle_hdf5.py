"""Pack pickle mesh sequences into HDF5 with train/val split.

Directory layout (input):  data_pickle/{animal}/{name}.pkl
Each pkl: {"vertices": (F, V, 3) float64, "faces": (N, 3) int64}

HDF5 layout (output):
    {animal}/{seq_name}/vertices   (F, V, 3) float32
    {animal}/{seq_name}/faces      (N, 3)    int32
    data_split/train               string[]  — paths like "animal/seq_name"
    data_split/val                 string[]

Split strategy: natsort all sequence paths, fixed-seed shuffle, 80/20 split.

Usage:
    uv run python source/pack_pickle_hdf5.py
    uv run python source/pack_pickle_hdf5.py --pkl-dir data_pickle --output data_pickle.hdf5 --workers 8
"""

import dataclasses
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import tyro
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


# ── CLI ──────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """Pack pickle mesh sequences into HDF5."""

    pkl_dir: Path = PROJECT_ROOT / "data_pickle"
    """Input directory with {animal}/{name}.pkl structure."""

    output: Path = PROJECT_ROOT / "data_pickle.hdf5"
    """Output HDF5 file path."""

    workers: int = 8
    """Number of parallel workers for reading pickles."""

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
    val_indices = set(indices[:n_val])

    train_paths = [all_paths[i] for i in indices[n_val:]]
    val_paths = [all_paths[i] for i in indices[:n_val]]
    print(f"Split: {len(train_paths):,} train, {len(val_paths):,} val")

    # 3. Pack into HDF5 with multiprocess reading
    errors = 0

    with h5py.File(args.output, "w") as hf:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(read_one_pkl, p): p for p in all_pkls}
            pbar = tqdm(total=len(all_pkls), desc="Packing", unit="seq")

            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                if result is None:
                    errors += 1
                    continue

                animal, seq_name, verts, faces = result
                grp_path = f"{animal}/{seq_name}"

                grp = hf.create_group(grp_path)
                grp.create_dataset("vertices", data=verts, compression="gzip", compression_opts=4)
                grp.create_dataset("faces", data=faces, compression="gzip", compression_opts=4)

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
