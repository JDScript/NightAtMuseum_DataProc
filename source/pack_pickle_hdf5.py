"""Pack pickle mesh sequences into HDF5 with train/val split.

Directory layout (input):  data_pickle/{animal}/{name}.pkl
Each pkl: {"vertices": (F, V, 3) float64, "faces": (N, 3) int64}

HDF5 layout (output):
    {animal}/{seq_name}/vertices   (F, V, 3) float32
    {animal}/{seq_name}/faces      (N, 3)    int32
    data_split/train               string[]  — paths like "animal/seq_name"
    data_split/val                 string[]

Split strategy: 80/20 by species (all variants of a species go to the same split).

Usage:
    uv run python source/pack_pickle_hdf5.py
    uv run python source/pack_pickle_hdf5.py --pkl-dir data_pickle --output data_pickle.hdf5 --workers 8
"""

import dataclasses
import pickle
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import tyro
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_species(animal_name: str) -> str:
    for s in ("_Male", "_Female", "_Juvenile"):
        if animal_name.endswith(s):
            return animal_name[: -len(s)]
    return animal_name


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
    """Fraction of species held out for validation."""

    seed: int = 42
    """Random seed for split."""


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: Args):
    # 1. Discover all pickle files
    all_pkls = sorted(args.pkl_dir.glob("*/*.pkl"))
    if not all_pkls:
        print(f"No pickle files found in {args.pkl_dir}")
        return
    print(f"Found {len(all_pkls):,} pickle files")

    # 2. Group by species for split
    species_to_animals = defaultdict(set)
    animal_to_pkls = defaultdict(list)
    for p in all_pkls:
        animal = p.parent.name
        species = get_species(animal)
        species_to_animals[species].add(animal)
        animal_to_pkls[animal].append(p)

    all_species = sorted(species_to_animals.keys())
    rng = random.Random(args.seed)
    rng.shuffle(all_species)

    n_val = max(1, int(len(all_species) * args.val_ratio))
    val_species = set(all_species[:n_val])
    train_species = set(all_species[n_val:])

    train_animals = {a for sp in train_species for a in species_to_animals[sp]}
    val_animals = {a for sp in val_species for a in species_to_animals[sp]}

    train_pkls = [p for p in all_pkls if p.parent.name in train_animals]
    val_pkls = [p for p in all_pkls if p.parent.name in val_animals]

    print(f"Species: {len(all_species)} total, {len(train_species)} train, {len(val_species)} val")
    print(f"Animals: {len(train_animals)} train, {len(val_animals)} val")
    print(f"Sequences: {len(train_pkls):,} train, {len(val_pkls):,} val")

    # 3. Pack into HDF5 with multiprocess reading
    train_paths = []
    val_paths = []
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

                if animal in train_animals:
                    train_paths.append(grp_path)
                else:
                    val_paths.append(grp_path)

            pbar.close()

        # 4. Write split info
        dt = h5py.string_dtype()
        split_grp = hf.create_group("data_split")
        split_grp.create_dataset("train", data=np.array(train_paths, dtype=object), dtype=dt)
        split_grp.create_dataset("val", data=np.array(val_paths, dtype=object), dtype=dt)

        # Metadata
        hf.attrs["n_species_train"] = len(train_species)
        hf.attrs["n_species_val"] = len(val_species)
        hf.attrs["n_train"] = len(train_paths)
        hf.attrs["n_val"] = len(val_paths)
        hf.attrs["seed"] = args.seed

    print(f"\nDone: {args.output}")
    print(f"  Train: {len(train_paths):,}  Val: {len(val_paths):,}  Errors: {errors}")
    print(f"  File size: {args.output.stat().st_size / 1024 / 1024:.0f} MB")


if __name__ == "__main__":
    main(tyro.cli(Args))
