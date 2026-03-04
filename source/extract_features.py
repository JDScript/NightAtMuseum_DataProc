"""Extract DINOv2 features from rendered multiview images and project onto voxels.

Reads chunk HDF5 files from trellis_preproc/{dataset}/, extracts DINOv2 features
from rendered images, projects them onto voxel positions via camera parameters,
and saves per-dataset HDF5 with voxel indices + features.

Output HDF5 schema:
    {animal}/{seq_name}/
        voxel_indices    (N, 3)    uint8    — grid indices [0, 63]
        patchtokens      (N, 1024) float16  — DINOv2 features per voxel

Usage:
    # Single GPU test:
    uv run python source/extract_features.py --dataset dt4d --limit 2

    # Full run on 8 GPUs:
    torchrun --nproc_per_node=8 source/extract_features.py --dataset animo
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from natsort import natsorted
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Projection ──────────────────────────────────────────────────────────────


def build_intrinsics(fov: float, image_size: int = 518) -> torch.Tensor:
    """Build 3x3 camera intrinsic matrix from FOV (radians)."""
    f = 0.5 * image_size / np.tan(fov / 2)
    cx = cy = image_size / 2
    K = torch.tensor([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], dtype=torch.float32)
    return K


def project_to_uv(
    positions: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project 3D voxel positions to normalized UV coordinates.

    Args:
        positions:  (N, 3) voxel centers in [-0.5, 0.5]^3
        extrinsics: (B, 4, 4) world-to-camera matrices
        intrinsics: (B, 3, 3) camera intrinsic matrices

    Returns:
        uv: (B, N, 2) normalized coords in [-1, 1] for grid_sample
    """
    B = extrinsics.shape[0]
    N = positions.shape[0]

    ones = torch.ones(N, 1, device=positions.device, dtype=positions.dtype)
    pos_h = torch.cat([positions, ones], dim=1)  # (N, 4)

    cam = torch.bmm(extrinsics[:, :3, :], pos_h.T.unsqueeze(0).expand(B, -1, -1))
    proj = torch.bmm(intrinsics, cam)

    uv = proj[:, :2, :] / (proj[:, 2:3, :] + 1e-8)  # (B, 2, N)
    uv = uv.permute(0, 2, 1)  # (B, N, 2)

    image_size = intrinsics[0, 0, 2] * 2  # cx * 2 = image_size
    uv = uv / image_size * 2 - 1

    return uv


# ── HDF5 merge ─────────────────────────────────────────────────────────────


def _merge_into(src: h5py.Group, dst: h5py.Group):
    """Recursively deep-merge src HDF5 group into dst, creating groups as needed."""
    for key in src:
        if isinstance(src[key], h5py.Group):
            _merge_into(src[key], dst.require_group(key))
        else:
            src.copy(key, dst)


# ── Data loading (CPU, thread-safe) ────────────────────────────────────────


def _load_renders_and_cameras(
    grp: h5py.Group, image_size: int = 518
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load rendered PNGs and camera parameters from an HDF5 group.

    Returns:
        images:     (N, 3, H, W) float32 tensor, normalized for DINOv2
        extrinsics: (N, 4, 4) world-to-camera matrices
        intrinsics: (N, 3, 3) camera intrinsic matrices
    """
    transforms_data = json.loads(grp.attrs["transforms"])
    frames = transforms_data["frames"]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    images_list = []
    extrinsics_list = []
    intrinsics_list = []

    renders_grp = grp["renders"]
    for frame in frames:
        file_key = frame["file_path"].replace(".png", "")  # "000"

        if file_key not in renders_grp:
            continue

        # Decode PNG bytes
        png_bytes = bytes(renders_grp[file_key][()])
        img = Image.open(io.BytesIO(png_bytes)).resize(
            (image_size, image_size), Image.LANCZOS
        )
        img = np.array(img).astype(np.float32) / 255.0

        # Alpha compositing: RGB * A
        rgb = img[:, :, :3] * img[:, :, 3:4]
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W)
        img_tensor = normalize(img_tensor)
        images_list.append(img_tensor)

        # Camera extrinsics: c2w -> w2c
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w[:3, 1:3] *= -1  # OpenGL to OpenCV convention
        w2c = torch.inverse(c2w)
        extrinsics_list.append(w2c)

        # Camera intrinsics
        fov = frame["camera_angle_x"]
        K = build_intrinsics(fov, image_size)
        intrinsics_list.append(K)

    if not images_list:
        return (
            torch.empty(0, 3, image_size, image_size),
            torch.empty(0, 4, 4),
            torch.empty(0, 3, 3),
        )

    images = torch.stack(images_list)       # (N, 3, H, W)
    extrinsics = torch.stack(extrinsics_list)  # (N, 4, 4)
    intrinsics = torch.stack(intrinsics_list)  # (N, 3, 3)
    return images, extrinsics, intrinsics


def load_sequence(
    chunk_path: str, group_key: str, image_size: int = 518
) -> dict:
    """Load one sequence from HDF5: voxels + rendered images + cameras.

    CPU-only. Each call opens its own HDF5 handle, safe for concurrent threads.
    """
    with h5py.File(chunk_path, "r") as hf:
        grp = hf[group_key]
        voxel_positions = torch.from_numpy(grp["voxel_positions"][()]).float()
        voxel_indices = grp["voxel_indices"][()].astype(np.uint8)
        images, extrinsics, intrinsics = _load_renders_and_cameras(grp, image_size)
    return {
        "voxel_positions": voxel_positions,
        "voxel_indices": voxel_indices,
        "images": images,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }


# ── Feature extraction (GPU) ──────────────────────────────────────────────


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    data: dict,
    device: torch.device,
    batch_size: int = 16,
    image_size: int = 518,
) -> dict | None:
    """Run DINOv2 on pre-loaded data and project features onto voxels.

    Args:
        data: dict from load_sequence() with CPU tensors.

    Returns dict with 'voxel_indices' (N,3) uint8 and 'patchtokens' (N,1024) float16,
    or None if no views.
    """
    n_patch = image_size // 14  # 37

    voxel_positions = data["voxel_positions"].to(device)
    images = data["images"]
    n_views = images.shape[0]

    if n_views == 0:
        return None

    patchtokens_list = []
    uv_list = []

    for i in range(0, n_views, batch_size):
        batch_imgs = images[i:i + batch_size].to(device)
        batch_ext = data["extrinsics"][i:i + batch_size].to(device)
        batch_int = data["intrinsics"][i:i + batch_size].to(device)
        bs = batch_imgs.shape[0]

        # DINOv2 forward
        features = model(batch_imgs, is_training=True)
        # Skip CLS + register tokens, reshape to spatial
        tokens = features["x_prenorm"][:, model.num_register_tokens + 1:]
        tokens = tokens.permute(0, 2, 1).reshape(bs, -1, n_patch, n_patch)
        # (B, 1024, 37, 37)

        # Project voxels to UV
        uv = project_to_uv(voxel_positions, batch_ext, batch_int)
        # (B, N, 2)

        patchtokens_list.append(tokens)
        uv_list.append(uv)

    # Concatenate across batches
    all_tokens = torch.cat(patchtokens_list, dim=0)  # (n_views, 1024, 37, 37)
    all_uv = torch.cat(uv_list, dim=0)  # (n_views, N, 2)

    # Sample features at projected voxel locations
    sampled = F.grid_sample(
        all_tokens,                    # (n_views, 1024, 37, 37)
        all_uv.unsqueeze(1),           # (n_views, 1, N, 2)
        mode="bilinear",
        align_corners=False,
        padding_mode="zeros",
    )  # (n_views, 1024, 1, N)
    sampled = sampled.squeeze(2).permute(0, 2, 1)  # (n_views, N, 1024)

    # Average across views
    mean_features = sampled.mean(dim=0)  # (N, 1024)

    return {
        "voxel_indices": data["voxel_indices"],
        "patchtokens": mean_features.cpu().half().numpy(),
    }


# ── Manifest ────────────────────────────────────────────────────────────────


def collect_sequences(
    preproc_dir: Path, dataset: str
) -> list[tuple[str, str]]:
    """Collect all (chunk_hdf5_path, group_key) pairs from meta.json.

    Returns a natsorted list of (hdf5_path, group_key) tuples.
    """
    dataset_dir = preproc_dir / dataset
    meta_path = dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text())

    sequences = []
    for chunk_file, seq_keys in meta["chunks"].items():
        chunk_path = str(dataset_dir / chunk_file)
        for key in seq_keys:
            group_key = f"{dataset}/{key}"
            sequences.append((chunk_path, group_key))

    return natsorted(sequences, key=lambda x: x[1])


# ── CLI ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """Extract DINOv2 features and project onto voxels."""

    dataset: str = "dt4d"
    """Dataset name (animo, night_at_museum, dt4d)."""

    preproc_dir: Path = Path("/scratch/zj2631/data/trellis_preproc")
    """Directory containing chunk HDF5 files."""

    output_dir: Path = Path("/scratch/zj2631/data/trellis_features")
    """Output directory for feature HDF5 files."""

    model_name: str = "dinov2_vitl14_reg"
    """DINOv2 model name."""

    batch_size: int = 16
    """Batch size for DINOv2 forward pass (images per batch)."""

    limit: int = 0
    """Process only first N sequences (0 = all). For testing."""


    prefetch: int = 4
    """Number of sequences to prefetch in background (queue depth)."""

    load_workers: int = 4
    """Number of background threads for data loading (PNG decode)."""


def main():
    import tyro

    args = tyro.cli(Args)

    # Distributed setup
    local_rank = 0
    global_rank = 0
    world_size = 1
    if dist.is_available() and "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = global_rank == 0

    if is_main:
        log.info(f"Dataset: {args.dataset}, Model: {args.model_name}")
        log.info(f"World size: {world_size}, Rank: {global_rank}, Local rank: {local_rank}")

    # Load model
    if is_main:
        log.info("Loading DINOv2 model...")
    model = torch.hub.load("facebookresearch/dinov2", args.model_name, verbose=False)
    model.eval().to(device)
    if is_main:
        log.info("Model loaded.")

    # Collect sequences
    if is_main:
        log.info("Collecting sequences...")
    sequences = collect_sequences(args.preproc_dir, args.dataset)
    if is_main:
        log.info(f"Total sequences: {len(sequences):,}")

    if args.limit > 0:
        sequences = sequences[: args.limit]
        if is_main:
            log.info(f"Limited to {len(sequences)} sequences")

    # Shard across GPUs
    my_sequences = sequences[global_rank::world_size]
    if is_main:
        log.info(f"Sequences per GPU: ~{len(my_sequences)}")

    # Output HDF5
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        out_path = args.output_dir / f"{args.dataset}_rank{global_rank}.hdf5"
    else:
        out_path = args.output_dir / f"{args.dataset}.hdf5"

    ok = 0
    errors = 0
    t_start = time.time()

    # ── Prefetch pipeline ──
    # Background threads decode PNGs while GPU processes the current sequence.
    # Queue backpressure (maxsize) prevents loading too far ahead into RAM.
    load_queue: Queue = Queue(maxsize=args.prefetch)

    def _loader(item):
        chunk_path, group_key = item
        try:
            data = load_sequence(chunk_path, group_key)
            load_queue.put((group_key, data))
        except Exception as e:
            log.warning(f"Load failed {group_key}: {e}")
            load_queue.put((group_key, None))

    with h5py.File(out_path, "w") as out_hf:
        out_hf.attrs["model"] = args.model_name
        out_hf.attrs["dataset"] = args.dataset

        with ThreadPoolExecutor(max_workers=args.load_workers) as load_exec:
            # Submit all load tasks; executor + queue backpressure throttle them
            load_exec.map(_loader, my_sequences)

            pbar = tqdm(
                range(len(my_sequences)), desc="Extracting", disable=not is_main
            )
            for _ in pbar:
                group_key, data = load_queue.get()

                if data is None:
                    errors += 1
                    continue

                try:
                    result = extract_features(
                        model, data, device, batch_size=args.batch_size
                    )

                    if result is None:
                        errors += 1
                        continue

                    # Strip dataset prefix for output key
                    out_key = group_key
                    if out_key.startswith(f"{args.dataset}/"):
                        out_key = out_key[len(f"{args.dataset}/"):]

                    out_grp = out_hf.create_group(out_key)
                    out_grp.create_dataset(
                        "voxel_indices", data=result["voxel_indices"]
                    )
                    out_grp.create_dataset(
                        "patchtokens", data=result["patchtokens"]
                    )
                    ok += 1
                    pbar.set_postfix_str(f"ok={ok}")

                except Exception as e:
                    errors += 1
                    log.warning(f"Failed {group_key}: {e}")

    elapsed = time.time() - t_start
    log.info(
        f"Rank {global_rank}: Done in {elapsed:.0f}s | "
        f"OK: {ok} | Errors: {errors} | "
        f"Output: {out_path} ({out_path.stat().st_size / 1024 / 1024:.0f} MB)"
    )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
