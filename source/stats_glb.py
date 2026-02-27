"""Statistics and visualization for exported GLB animations.

Scans data/glb/ directories, parses GLB JSON chunks to extract animation
metadata (duration, frame count), and produces summary tables + matplotlib plots.

Usage:
    uv run python source/stats_glb.py
    uv run python source/stats_glb.py --glb-dir data/glb --outdir figures
"""

import dataclasses
import json
import struct
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── GLB parsing ──────────────────────────────────────────────────────────────


def read_glb_json(filepath: Path) -> dict | None:
    """Read the JSON chunk from a GLB file."""
    try:
        with open(filepath, "rb") as f:
            # GLB header: magic(4) + version(4) + length(4)
            magic, version, length = struct.unpack("<III", f.read(12))
            if magic != 0x46546C67:  # 'glTF'
                return None
            # First chunk: length(4) + type(4) + data
            chunk_len, chunk_type = struct.unpack("<II", f.read(8))
            if chunk_type != 0x4E4F534A:  # 'JSON'
                return None
            return json.loads(f.read(chunk_len))
    except Exception:
        return None


def get_animation_info(glb_path: Path) -> dict | None:
    """Extract animation metadata from a GLB file.

    Returns dict with keys: name, duration, frame_count, fps
    """
    gltf = read_glb_json(glb_path)
    if not gltf:
        return None

    animations = gltf.get("animations", [])
    if not animations:
        return None

    anim = animations[0]  # one animation per GLB
    accessors = gltf.get("accessors", [])

    # Find max timestamp across all animation channels
    max_time = 0.0
    max_frames = 0
    for channel_sampler_idx in set(s.get("input") for s in anim.get("samplers", [])):
        if channel_sampler_idx is not None and channel_sampler_idx < len(accessors):
            acc = accessors[channel_sampler_idx]
            t = acc.get("max", [0])
            if isinstance(t, list) and t:
                max_time = max(max_time, t[0])
            max_frames = max(max_frames, acc.get("count", 0))

    if max_frames == 0:
        return None

    fps = (max_frames - 1) / max_time if max_time > 0 else 0.0

    return {
        "name": anim.get("name", glb_path.stem),
        "duration": max_time,
        "frame_count": max_frames,
        "fps": fps,
    }


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
    """Statistics and visualization for exported GLB animations."""

    glb_dir: Path = PROJECT_ROOT / "data" / "glb"
    """Path to the GLB output directory."""

    outdir: Path = PROJECT_ROOT / "figures"
    """Output directory for figures."""


# ── Scanning ─────────────────────────────────────────────────────────────────


def scan_glbs(glb_dir: Path) -> list[dict]:
    """Scan all GLB files and return list of records."""
    animal_dirs = sorted(d for d in glb_dir.iterdir() if d.is_dir())
    records = []

    for animal_dir in tqdm(animal_dirs, desc="Scanning animals", unit="dir"):
        glb_files = sorted(animal_dir.glob("*.glb"))
        if not glb_files:
            continue

        animal_name = animal_dir.name
        species = get_species(animal_name)
        category = get_category(animal_name)

        for glb_path in glb_files:
            info = get_animation_info(glb_path)
            if info is None:
                continue
            records.append({
                "animal": animal_name,
                "species": species,
                "category": category,
                "glb": glb_path.name,
                **info,
            })

    return records


# ── Printing ─────────────────────────────────────────────────────────────────

W = 100


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

    durations = [r["duration"] for r in records]
    frames = [r["frame_count"] for r in records]

    # ── Overview
    print_header("OVERVIEW")
    print(f"  Species:       {len(species_set)}")
    print(f"  Animals:       {len(animals_set)}")
    print(f"  Animations:    {len(records):,}")
    print(f"  Duration:      {np.min(durations):.2f}s – {np.max(durations):.2f}s  (mean {np.mean(durations):.2f}s, median {np.median(durations):.2f}s)")
    print(f"  Frames:        {np.min(frames)} – {np.max(frames)}  (mean {np.mean(frames):.0f}, median {np.median(frames):.0f})")

    # ── By category
    print_header("BY CATEGORY")
    print(f"\n{'Category':<12} {'Animals':>8} {'Clips':>10} {'Frames (mean)':>14} {'Duration (mean)':>16}")
    print("-" * 64)
    for cat in ["Male", "Female", "Juvenile", "Other"]:
        recs = categories.get(cat, [])
        if not recs:
            continue
        n_animals = len({r["animal"] for r in recs})
        avg_frames = np.mean([r["frame_count"] for r in recs])
        avg_dur = np.mean([r["duration"] for r in recs])
        print(f"{cat:<12} {n_animals:>8} {len(recs):>10,} {avg_frames:>14.1f} {avg_dur:>14.2f}s")
    print("-" * 64)
    print(f"{'TOTAL':<12} {len(animals_set):>8} {len(records):>10,}")

    # ── Top 10 species by clip count
    print_header("TOP 20 SPECIES BY CLIP COUNT")
    species_clips = defaultdict(int)
    species_animals = defaultdict(set)
    for r in records:
        species_clips[r["species"]] += 1
        species_animals[r["species"]].add(r["animal"])

    top = sorted(species_clips.items(), key=lambda x: -x[1])[:20]
    print(f"\n{'Species':<35} {'Variants':>9} {'Clips':>8}")
    print("-" * 55)
    for sp, count in top:
        print(f"{sp:<35} {len(species_animals[sp]):>9} {count:>8,}")


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_all(records: list[dict], outdir: Path):
    """Generate and save all plots."""
    outdir.mkdir(parents=True, exist_ok=True)

    durations = np.array([r["duration"] for r in records])
    frames = np.array([r["frame_count"] for r in records])

    # ── 1. Frame count distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(frames, bins=80, edgecolor="black", linewidth=0.3, color="#4C72B0")
    ax.set_xlabel("Frame Count")
    ax.set_ylabel("Number of Animations")
    ax.set_title("Distribution of Animation Frame Counts")
    ax.axvline(np.median(frames), color="red", linestyle="--", label=f"median = {np.median(frames):.0f}")
    ax.axvline(np.mean(frames), color="orange", linestyle="--", label=f"mean = {np.mean(frames):.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "frame_count_dist.png", dpi=150)
    plt.close(fig)

    # ── 2. Duration distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(durations, bins=80, edgecolor="black", linewidth=0.3, color="#55A868")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Number of Animations")
    ax.set_title("Distribution of Animation Durations")
    ax.axvline(np.median(durations), color="red", linestyle="--", label=f"median = {np.median(durations):.2f}s")
    ax.axvline(np.mean(durations), color="orange", linestyle="--", label=f"mean = {np.mean(durations):.2f}s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "duration_dist.png", dpi=150)
    plt.close(fig)

    # ── 3. Clips per category (bar chart)
    cat_order = ["Male", "Female", "Juvenile", "Other"]
    cat_counts = defaultdict(int)
    for r in records:
        cat_counts[r["category"]] += 1
    cats = [c for c in cat_order if cat_counts[c] > 0]
    counts = [cat_counts[c] for c in cats]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(cats, counts, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(cats)],
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of Animations")
    ax.set_title("Animations by Category")
    fig.tight_layout()
    fig.savefig(outdir / "category_bar.png", dpi=150)
    plt.close(fig)

    # ── 4. Clips per species (horizontal bar, top 30)
    species_clips = defaultdict(int)
    for r in records:
        species_clips[r["species"]] += 1
    top = sorted(species_clips.items(), key=lambda x: -x[1])[:30]
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(names, vals, color="#4C72B0", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Number of Animations")
    ax.set_title("Top 30 Species by Animation Count")
    for i, v in enumerate(vals):
        ax.text(v + 5, i, f"{v:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "species_top30.png", dpi=150)
    plt.close(fig)

    # ── 5. Frame count by category (box plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    cat_frames = [[r["frame_count"] for r in records if r["category"] == c] for c in cats]
    bp = ax.boxplot(cat_frames, labels=cats, patch_artist=True, showfliers=False)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors[:len(cats)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Frame Count")
    ax.set_title("Frame Count Distribution by Category")
    fig.tight_layout()
    fig.savefig(outdir / "frames_by_category_box.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures saved to {outdir}/")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: Args):
    records = scan_glbs(args.glb_dir)

    if not records:
        print("No GLB animations found.")
        return

    print_summary(records)
    plot_all(records, args.outdir)


if __name__ == "__main__":
    main(tyro.cli(Args))
