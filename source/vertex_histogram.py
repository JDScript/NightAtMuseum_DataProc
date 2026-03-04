"""Count vertices per sequence across all three datasets and plot histograms.

Multi-threaded: each worker opens its own HDF5 handle.
- mesh_sequence format: reads vertices shape[1] (instant, metadata only)
- raw_glb format: parses GLB with trimesh (CPU-bound, benefits from threading)

Usage:
    uv run python source/vertex_histogram.py
    uv run python source/vertex_histogram.py --workers 16
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("/scratch/zj2631/data")

DATASETS: dict[str, dict] = {
    "dt4d": {"path": DATA_DIR / "dt4d.hdf5", "format": "mesh_sequence"},
    "night_at_museum": {
        "path": DATA_DIR / "night_at_the_museum.hdf5",
        "format": "mesh_sequence",
    },
    "animo": {"path": DATA_DIR / "animo.hdf5", "format": "raw_glb"},
}


def get_keys(hdf5_path: Path) -> list[str]:
    """Read all sequence keys from data_split/{train,val}."""
    with h5py.File(hdf5_path, "r") as hf:
        keys: list[str] = []
        if "data_split" in hf:
            for split in ("train", "val"):
                if split in hf["data_split"]:
                    keys.extend(hf["data_split"][split][()].astype(str).tolist())
        return keys


def _count_mesh_seq(hdf5_path: str, keys: list[str]) -> list[tuple[str, int]]:
    """Count vertices for mesh_sequence format — just read dataset shape."""
    results = []
    with h5py.File(hdf5_path, "r") as hf:
        for key in keys:
            grp = hf.get(key)
            if grp is not None and "vertices" in grp:
                results.append((key, grp["vertices"].shape[1]))
    return results


def _count_glb(hdf5_path: str, keys: list[str]) -> list[tuple[str, int]]:
    """Count vertices for raw_glb format — parse GLB with trimesh."""
    import trimesh

    results = []
    with h5py.File(hdf5_path, "r") as hf:
        for key in keys:
            grp = hf.get(key)
            if grp is None or "glb" not in grp:
                continue
            glb_bytes = bytes(grp["glb"][()])
            scene = trimesh.load(
                io.BytesIO(glb_bytes), file_type="glb", process=False
            )
            total = sum(
                m.vertices.shape[0]
                for m in scene.geometry.values()
                if hasattr(m, "vertices")
            )
            results.append((key, total))
    return results


def process_dataset(
    name: str, info: dict, num_workers: int = 8
) -> np.ndarray:
    """Process one dataset with multiple threads, return vertex counts array."""
    hdf5_path = info["path"]
    fmt = info["format"]
    keys = get_keys(hdf5_path)
    print(f"  {name}: {len(keys)} sequences, format={fmt}")

    fn = _count_mesh_seq if fmt == "mesh_sequence" else _count_glb

    # Shard keys across workers
    chunks = [keys[i::num_workers] for i in range(num_workers)]

    all_results: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(fn, str(hdf5_path), chunk): i
            for i, chunk in enumerate(chunks)
            if chunk  # skip empty chunks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {name}"):
            all_results.extend(future.result())

    counts = np.array([v for _, v in all_results])
    return counts


def print_stats(name: str, counts: np.ndarray):
    """Print summary statistics for vertex counts."""
    print(f"\n{'=' * 50}")
    print(f"  {name}: {len(counts)} sequences")
    print(f"{'=' * 50}")
    print(f"  Min:      {counts.min():>10,}")
    print(f"  Max:      {counts.max():>10,}")
    print(f"  Mean:     {counts.mean():>10,.0f}")
    print(f"  Median:   {np.median(counts):>10,.0f}")
    print(f"  Std:      {counts.std():>10,.0f}")
    print(f"  P5:       {np.percentile(counts, 5):>10,.0f}")
    print(f"  P25:      {np.percentile(counts, 25):>10,.0f}")
    print(f"  P75:      {np.percentile(counts, 75):>10,.0f}")
    print(f"  P95:      {np.percentile(counts, 95):>10,.0f}")
    print(f"  P99:      {np.percentile(counts, 99):>10,.0f}")
    print(f"  Total:    {counts.sum():>14,}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", type=str, default="vertex_histogram.png",
        help="Output plot path",
    )
    args = parser.parse_args()

    print(f"Counting vertices with {args.workers} threads...")

    all_counts: dict[str, np.ndarray] = {}
    for name, info in DATASETS.items():
        t0 = time.time()
        counts = process_dataset(name, info, num_workers=args.workers)
        elapsed = time.time() - t0
        all_counts[name] = counts
        print(f"    -> done in {elapsed:.1f}s")

    # Print statistics
    for name, counts in all_counts.items():
        print_stats(name, counts)

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"dt4d": "#4C72B0", "night_at_museum": "#55A868", "animo": "#C44E52"}

    for ax, (name, counts) in zip(axes, all_counts.items()):
        ax.hist(counts, bins=80, color=colors[name], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{name} (n={len(counts):,})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Vertices per sequence")
        ax.set_ylabel("Count")
        ax.axvline(np.median(counts), color="k", linestyle="--", linewidth=1, label=f"median={np.median(counts):,.0f}")
        ax.axvline(np.mean(counts), color="red", linestyle=":", linewidth=1, label=f"mean={np.mean(counts):,.0f}")
        ax.legend(fontsize=9)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig.suptitle("Vertex Count Distribution", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
