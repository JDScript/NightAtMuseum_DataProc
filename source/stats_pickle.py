"""Statistics and visualization for exported pickle mesh sequences.

Each pickle contains: {"vertices": (F, V, 3), "faces": (N, 3)}
Directory layout: data_pickle/{animal}/{name}.pkl

Usage:
    uv run python source/stats_pickle.py
    uv run python source/stats_pickle.py --pkl-dir data_pickle --outdir figures_pickle
"""

import dataclasses
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Pickle parsing ───────────────────────────────────────────────────────────


def get_pkl_info(pkl_path: Path) -> dict | None:
    """Load a pickle and return shape metadata without keeping the arrays."""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        verts = data["vertices"]  # (frames, vertices, 3)
        faces = data["faces"]  # (faces, 3)
        return {
            "name": pkl_path.stem,
            "frame_count": verts.shape[0],
            "n_vertices": verts.shape[1],
            "n_faces": faces.shape[0],
            "filesize_mb": pkl_path.stat().st_size / (1024 * 1024),
        }
    except Exception:
        return None


# ── Classification helpers ───────────────────────────────────────────────────


def get_species(animal_dir: str) -> str:
    """'Bengal_Tiger_Male' -> 'Bengal_Tiger'."""
    for s in ("_Male", "_Female", "_Juvenile"):
        if animal_dir.endswith(s):
            return animal_dir[: -len(s)]
    return animal_dir


def get_category(animal_dir: str) -> str:
    """'Bengal_Tiger_Male' -> 'Male'."""
    for s in ("Male", "Female", "Juvenile"):
        if animal_dir.endswith(s):
            return s
    return "Other"


# ── CLI ──────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """Statistics and visualization for exported pickle mesh sequences."""

    pkl_dir: Path = PROJECT_ROOT / "data_pickle"
    """Path to the pickle output directory."""

    outdir: Path = PROJECT_ROOT / "figures_pickle"
    """Output directory for figures."""

    workers: int = 8
    """Number of parallel workers."""


# ── Scanning ─────────────────────────────────────────────────────────────────


def _process_one_pkl(pkl_path: Path) -> dict | None:
    """Worker function: parse one pickle and return a record."""
    info = get_pkl_info(pkl_path)
    if info is None:
        return None
    animal_name = pkl_path.parent.name
    return {
        "animal": animal_name,
        "species": get_species(animal_name),
        "category": get_category(animal_name),
        "pkl": pkl_path.name,
        **info,
    }


def scan_pkls(pkl_dir: Path, workers: int) -> list[dict]:
    """Scan all pickle files using multiprocessing."""
    all_pkls = sorted(pkl_dir.glob("*/*.pkl"))
    if not all_pkls:
        return []

    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one_pkl, p): p for p in all_pkls}
        for future in tqdm(as_completed(futures), total=len(all_pkls),
                           desc="Parsing pickles", unit="file"):
            result = future.result()
            if result is not None:
                records.append(result)

    return records


# ── Printing ─────────────────────────────────────────────────────────────────

W = 110


def print_header(title: str):
    print(f"\n{'=' * W}")
    print(f"{title:^{W}}")
    print(f"{'=' * W}")


def print_summary(records: list[dict]):
    """Print text summary tables."""
    species_set = {r["species"] for r in records}
    animals_set = {r["animal"] for r in records}
    categories = defaultdict(list)
    for r in records:
        categories[r["category"]].append(r)

    frames = np.array([r["frame_count"] for r in records])
    verts = np.array([r["n_vertices"] for r in records])
    faces_arr = np.array([r["n_faces"] for r in records])
    sizes = np.array([r["filesize_mb"] for r in records])

    # ── Overview
    print_header("OVERVIEW")
    print(f"  Species:         {len(species_set)}")
    print(f"  Animals:         {len(animals_set)}")
    print(f"  Sequences:       {len(records):,}")
    print(f"  Total size:      {sizes.sum():,.0f} MB ({sizes.sum() / 1024:.1f} GB)")
    print(f"  Frames:          {frames.min()} – {frames.max()}  (mean {frames.mean():.0f}, median {np.median(frames):.0f})")
    print(f"  Vertices/mesh:   {verts.min():,} – {verts.max():,}  (mean {verts.mean():,.0f})")
    print(f"  Faces/mesh:      {faces_arr.min():,} – {faces_arr.max():,}  (mean {faces_arr.mean():,.0f})")

    # ── By category
    print_header("BY CATEGORY")
    print(f"\n{'Category':<12} {'Animals':>8} {'Seqs':>8} {'Frames(mean)':>13} {'Verts(mean)':>12} {'Size(GB)':>10}")
    print("-" * 68)
    for cat in ["Male", "Female", "Juvenile", "Other"]:
        recs = categories.get(cat, [])
        if not recs:
            continue
        n_animals = len({r["animal"] for r in recs})
        avg_f = np.mean([r["frame_count"] for r in recs])
        avg_v = np.mean([r["n_vertices"] for r in recs])
        total_size = sum(r["filesize_mb"] for r in recs) / 1024
        print(f"{cat:<12} {n_animals:>8} {len(recs):>8,} {avg_f:>13.1f} {avg_v:>12,.0f} {total_size:>10.1f}")
    print("-" * 68)
    print(f"{'TOTAL':<12} {len(animals_set):>8} {len(records):>8,} {'':>13} {'':>12} {sizes.sum() / 1024:>10.1f}")

    # ── Top 20 species
    print_header("TOP 20 SPECIES BY SEQUENCE COUNT")
    species_clips = defaultdict(int)
    species_animals = defaultdict(set)
    for r in records:
        species_clips[r["species"]] += 1
        species_animals[r["species"]].add(r["animal"])

    top = sorted(species_clips.items(), key=lambda x: -x[1])[:20]
    print(f"\n{'Species':<35} {'Variants':>9} {'Sequences':>10}")
    print("-" * 58)
    for sp, count in top:
        print(f"{sp:<35} {len(species_animals[sp]):>9} {count:>10,}")


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_all(records: list[dict], outdir: Path):
    """Generate and save all plots."""
    outdir.mkdir(parents=True, exist_ok=True)

    frames = np.array([r["frame_count"] for r in records])
    verts = np.array([r["n_vertices"] for r in records])
    sizes = np.array([r["filesize_mb"] for r in records])

    cats_all = [r["category"] for r in records]
    cat_order = ["Male", "Female", "Juvenile", "Other"]
    cats_present = [c for c in cat_order if c in set(cats_all)]
    colors4 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # ── 1. Frame count distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(frames, bins=80, edgecolor="black", linewidth=0.3, color="#4C72B0")
    ax.set_xlabel("Frame Count")
    ax.set_ylabel("Number of Sequences")
    ax.set_title("Distribution of Mesh Sequence Frame Counts")
    ax.axvline(np.median(frames), color="red", ls="--", label=f"median = {np.median(frames):.0f}")
    ax.axvline(np.mean(frames), color="orange", ls="--", label=f"mean = {np.mean(frames):.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "frame_count_dist.png", dpi=150)
    plt.close(fig)

    # ── 2. Vertex count distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(verts, bins=60, edgecolor="black", linewidth=0.3, color="#DD8452")
    ax.set_xlabel("Vertex Count per Mesh")
    ax.set_ylabel("Number of Sequences")
    ax.set_title("Distribution of Vertex Counts")
    ax.axvline(np.median(verts), color="red", ls="--", label=f"median = {np.median(verts):,.0f}")
    ax.axvline(np.mean(verts), color="orange", ls="--", label=f"mean = {np.mean(verts):,.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "vertex_count_dist.png", dpi=150)
    plt.close(fig)

    # ── 3. Sequences per category (bar chart)
    cat_counts = defaultdict(int)
    for r in records:
        cat_counts[r["category"]] += 1
    c_vals = [cat_counts[c] for c in cats_present]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(cats_present, c_vals, color=colors4[:len(cats_present)],
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, c_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(c_vals) * 0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of Sequences")
    ax.set_title("Sequences by Category")
    fig.tight_layout()
    fig.savefig(outdir / "category_bar.png", dpi=150)
    plt.close(fig)

    # ── 4. Top 30 species (horizontal bar)
    species_clips = defaultdict(int)
    for r in records:
        species_clips[r["species"]] += 1
    top = sorted(species_clips.items(), key=lambda x: -x[1])[:30]
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(names, vals, color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Number of Sequences")
    ax.set_title("Top 30 Species by Sequence Count")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.005, i, f"{v:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "species_top30.png", dpi=150)
    plt.close(fig)

    # ── 5. Frame count by category (box plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    cat_frames = [[r["frame_count"] for r in records if r["category"] == c] for c in cats_present]
    bp = ax.boxplot(cat_frames, labels=cats_present, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors4[:len(cats_present)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Frame Count")
    ax.set_title("Frame Count Distribution by Category")
    fig.tight_layout()
    fig.savefig(outdir / "frames_by_category_box.png", dpi=150)
    plt.close(fig)

    # ── 6. File size distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sizes, bins=80, edgecolor="black", linewidth=0.3, color="#8172B2")
    ax.set_xlabel("File Size (MB)")
    ax.set_ylabel("Number of Sequences")
    ax.set_title("Distribution of Pickle File Sizes")
    ax.axvline(np.median(sizes), color="red", ls="--", label=f"median = {np.median(sizes):.1f} MB")
    ax.axvline(np.mean(sizes), color="orange", ls="--", label=f"mean = {np.mean(sizes):.1f} MB")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "filesize_dist.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures saved to {outdir}/")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: Args):
    records = scan_pkls(args.pkl_dir, args.workers)

    if not records:
        print("No pickle files found.")
        return

    print_summary(records)
    plot_all(records, args.outdir)


if __name__ == "__main__":
    main(tyro.cli(Args))
