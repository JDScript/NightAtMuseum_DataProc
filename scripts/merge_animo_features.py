"""Merge animo feature rank files into a single HDF5.

Merges both the original rank 3-7 files and the rerun rank 0-3 files.

Usage:
    uv run python scripts/merge_animo_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
from natsort import natsorted
from tqdm import tqdm

FEATURES_DIR = Path("/scratch/zj2631/data/trellis_features")
RERUN_DIR = Path("/scratch/zj2631/data/trellis_features_rerun")
PREPROC_DIR = Path("/scratch/zj2631/data/trellis_preproc")
DATASET = "animo"


def find_rank_files() -> list[Path]:
    """Find all rank HDF5 files from both original and rerun dirs."""
    files = []

    # Original rank 3-7
    for p in sorted(FEATURES_DIR.glob(f"{DATASET}_rank*.hdf5")):
        files.append(p)
        print(f"Found (original): {p.name} ({p.stat().st_size / 1e9:.1f} GB)")

    # Rerun rank files
    for p in sorted(RERUN_DIR.glob(f"{DATASET}_rank*.hdf5")):
        files.append(p)
        print(f"Found (rerun):    {p.name} ({p.stat().st_size / 1e9:.1f} GB)")

    return files


def merge_rank_files():
    out_path = FEATURES_DIR / f"{DATASET}.hdf5"

    # Safety check
    if out_path.exists():
        print(f"WARNING: {out_path} already exists ({out_path.stat().st_size / 1e9:.1f} GB)")
        resp = input("Delete and re-merge? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return
        out_path.unlink()

    rank_files = find_rank_files()
    if not rank_files:
        print("No rank files found!")
        return

    # Merge
    print(f"\nMerging {len(rank_files)} files into {out_path}...")
    total = 0
    with h5py.File(out_path, "w") as merged:
        merged.attrs["model"] = "dinov2_vitl14_reg"
        merged.attrs["dataset"] = DATASET

        for rank_path in rank_files:
            print(f"  Merging {rank_path.name}...")
            count = 0
            with h5py.File(rank_path, "r") as src:
                for animal in tqdm(src.keys(), desc=rank_path.name):
                    dst_animal = merged.require_group(animal)
                    for seq in src[animal].keys():
                        if seq in dst_animal:
                            continue  # skip duplicates
                        src[animal].copy(seq, dst_animal)
                        count += 1
            print(f"    {count} sequences")
            total += count

    print(f"\nMerged: {out_path} ({out_path.stat().st_size / 1e9:.1f} GB)")
    print(f"Total sequences merged: {total}")

    # Verify against manifest
    meta_path = PREPROC_DIR / DATASET / "meta.json"
    if not meta_path.exists():
        print("meta.json not found, skipping verification")
        return

    meta = json.loads(meta_path.read_text())
    source_seqs = set()
    for chunk_file, seq_keys in meta["chunks"].items():
        for key in seq_keys:
            source_seqs.add(key)

    with h5py.File(out_path, "r") as f:
        merged_seqs = set()
        for animal in f.keys():
            for seq in f[animal].keys():
                merged_seqs.add(f"{animal}/{seq}")

    missing = natsorted(source_seqs - merged_seqs)
    extra = natsorted(merged_seqs - source_seqs)

    print(f"\nVerification:")
    print(f"  Source manifest: {len(source_seqs)}")
    print(f"  Merged:          {len(merged_seqs)}")
    print(f"  Missing:         {len(missing)}")
    print(f"  Extra:           {len(extra)}")

    if missing:
        missing_path = FEATURES_DIR / f"{DATASET}_still_missing.txt"
        missing_path.write_text("\n".join(missing) + "\n")
        print(f"  Missing list: {missing_path}")


if __name__ == "__main__":
    merge_rank_files()
