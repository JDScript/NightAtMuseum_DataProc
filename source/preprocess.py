"""TRELLIS-style preprocessing: render multiview images + voxelize first-frame meshes.

Reads first-frame meshes from HDF5 datasets, renders them with Blender, voxelizes
the normalized meshes, and packs results into per-chunk HDF5 files.

Processing flow (per SLURM task):
    1. Build manifest from HDF5 files → chunk by chunk_id/num_chunks
    2. For each sequence (parallel via ThreadPoolExecutor):
       a. Export first-frame mesh to .obj/.glb
       b. Blender headless render → multiview PNGs + transforms.json + mesh.ply
       c. Voxelize normalized mesh.ply → sparse voxel positions
    3. Pack results into per-chunk HDF5 in TMPDIR
    4. Move chunk HDF5 to output_dir

Usage:
    # Single test (processes ~11 sequences):
    uv run python source/preprocess.py --chunk-id 0 --num-chunks 10000

    # Full run (called by SLURM sbatch):
    uv run python source/preprocess.py --chunk-id $SLURM_ARRAY_TASK_ID --num-chunks 256
"""

import dataclasses
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import tyro
from natsort import natsorted
from tqdm import tqdm

from mesh_toolkit import MeshRecord, build_manifest, enumerate_sequences, export_for_blender
from sphere_hammersley import generate_views

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLENDER_BIN = PROJECT_ROOT / "third_party" / "blender" / "blender"
RENDER_SCRIPT = PROJECT_ROOT / "source" / "blender_script" / "render_mesh.py"


# ── Voxelization ─────────────────────────────────────────────────────────────


def voxelize_mesh(
    mesh_ply_path: str, resolution: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize a normalized mesh within [-0.5, 0.5]^3.

    Returns:
        positions: (N, 3) float32 — sparse voxel centers in [-0.5, 0.5]^3
        indices:   (N, 3) uint8   — grid indices in [0, resolution-1]
    """
    mesh = o3d.io.read_triangle_mesh(mesh_ply_path)
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1.0 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    grid_indices = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    assert np.all(grid_indices >= 0) and np.all(grid_indices < resolution)
    positions = (grid_indices + 0.5) / resolution - 0.5
    return positions.astype(np.float32), grid_indices.astype(np.uint8)


# ── Per-sequence processing ──────────────────────────────────────────────────


def process_one(
    record: MeshRecord,
    work_dir: Path,
    views_json: str,
    blender_bin: Path,
    render_script: Path,
    engine: str,
    resolution: int,
    samples: int,
    voxel_resolution: int,
    blender_timeout: int,
) -> dict:
    """Process a single sequence: export → render → voxelize.

    Returns a result dict with status and data paths.
    """
    safe_uid = record.uid.replace("/", "__")
    seq_dir = work_dir / safe_uid
    render_dir = seq_dir / "renders"
    t0 = time.time()

    try:
        # 1. Export first-frame mesh
        mesh_path = export_for_blender(record, seq_dir)

        # 2. Blender render
        cmd = [
            str(blender_bin),
            "-b",
            "-P",
            str(render_script),
            "--",
            "--object", str(mesh_path),
            "--views", views_json,
            "--output_folder", str(render_dir),
            "--resolution", str(resolution),
            "--engine", engine,
            "--samples", str(samples),
            "--save_mesh",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=blender_timeout,
        )

        transforms_path = render_dir / "transforms.json"
        if result.returncode != 0 or not transforms_path.exists():
            err = result.stderr.strip()[-300:] if result.stderr else "unknown"
            return {
                "uid": record.uid,
                "status": "render_failed",
                "error": err,
                "time": time.time() - t0,
            }

        # 3. Voxelize
        mesh_ply = render_dir / "mesh.ply"
        if not mesh_ply.exists():
            return {
                "uid": record.uid,
                "status": "no_mesh_ply",
                "time": time.time() - t0,
            }

        positions, indices = voxelize_mesh(str(mesh_ply), voxel_resolution)

        return {
            "uid": record.uid,
            "status": "success",
            "render_dir": str(render_dir),
            "voxel_positions": positions,
            "voxel_indices": indices,
            "time": time.time() - t0,
        }

    except subprocess.TimeoutExpired:
        return {"uid": record.uid, "status": "timeout", "time": blender_timeout}
    except Exception as e:
        return {
            "uid": record.uid,
            "status": "error",
            "error": str(e),
            "time": time.time() - t0,
        }


# ── HDF5 packing ────────────────────────────────────────────────────────────


def pack_result_to_hdf5(
    hf: h5py.File, result: dict, num_views: int, resolution: int
):
    """Pack one processed sequence into the HDF5 file."""
    uid = result["uid"]
    render_dir = Path(result["render_dir"])

    grp = hf.create_group(uid)

    # Renders: store raw PNG bytes (already compressed, no HDF5 gzip needed)
    for i in range(num_views):
        png_path = render_dir / f"{i:03d}.png"
        if png_path.exists():
            png_bytes = png_path.read_bytes()
            grp.create_dataset(
                f"renders/{i:03d}",
                data=np.void(png_bytes),
            )

    # Transforms JSON
    transforms_path = render_dir / "transforms.json"
    if transforms_path.exists():
        grp.attrs["transforms"] = transforms_path.read_text()

    # Voxels
    positions = result.get("voxel_positions")
    indices = result.get("voxel_indices")
    if positions is not None:
        grp.create_dataset("voxel_positions", data=positions)
        grp.create_dataset("voxel_indices", data=indices)

    # Normalized mesh from mesh.ply
    mesh_ply = render_dir / "mesh.ply"
    if mesh_ply.exists():
        mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
        verts = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int32)
        grp.create_dataset("mesh_vertices", data=verts)
        grp.create_dataset("mesh_faces", data=faces)


# ── CLI ──────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """TRELLIS-style preprocessing: render + voxelize first-frame meshes."""

    hdf5_path: Path = Path("/scratch/zj2631/data/animo.hdf5")
    """Path to a single HDF5 file to process."""

    output_dir: Path = Path("/scratch/zj2631/data/trellis_preproc")
    """Output directory for chunk HDF5 files."""

    chunk_id: int | None = None
    """Chunk index for SLURM array jobs (0-based)."""

    num_chunks: int | None = None
    """Total number of chunks for SLURM array jobs."""

    num_views: int = 150
    """Number of multiview renders per mesh (150=TRELLIS default, 36=fast)."""

    resolution: int = 512
    """Render resolution (pixels)."""

    engine: str = "CYCLES"
    """Blender render engine (EEVEE requires GPU/EGL, won't work on CPU nodes)."""

    samples: int = 32
    """Render samples (CYCLES only; EEVEE ignores this)."""

    voxel_resolution: int = 64
    """Voxel grid resolution."""

    workers: int = 2
    """Number of parallel Blender processes."""

    blender_timeout: int = 600
    """Timeout per mesh in seconds (default 10 min)."""

    batch_size: int = 50
    """Sequences per mini-batch (controls TMPDIR disk usage)."""


def main(args: Args):
    # 1. Determine working directory (SLURM_TMPDIR or fallback)
    tmpdir = os.environ.get("SLURM_TMPDIR")
    if not tmpdir:
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        tmpdir = f"/state/partition/job-{job_id}"
        if not Path(tmpdir).exists():
            tmpdir = f"/tmp/preproc_{args.chunk_id or 0}"
    work_dir = Path(tmpdir) / "preproc_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Working directory: {work_dir}")

    # 2. Build manifest from single HDF5 file
    log.info(f"Enumerating sequences from {args.hdf5_path} ...")
    manifest = enumerate_sequences(args.hdf5_path)
    total = len(manifest)
    dataset_name = Path(args.hdf5_path).stem
    log.info(f"Dataset: {dataset_name}, total sequences: {total:,}")

    if args.chunk_id is not None and args.num_chunks is not None:
        # Even distribution via divmod: first `remainder` chunks get base+1,
        # the rest get base.  Every chunk is non-empty when num_chunks <= total.
        base, remainder = divmod(total, args.num_chunks)
        if args.chunk_id < remainder:
            start = args.chunk_id * (base + 1)
            end = start + base + 1
        else:
            start = remainder * (base + 1) + (args.chunk_id - remainder) * base
            end = start + base
        manifest = manifest[start:end]
        log.info(
            f"Chunk {args.chunk_id}/{args.num_chunks}: "
            f"sequences [{start}:{end}] ({len(manifest):,} items)"
        )

    if not manifest:
        log.info("Nothing to process")
        return

    # 3. Setup output: {output_dir}/{dataset_name}/{chunk_label}.hdf5
    dataset_out_dir = args.output_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    chunk_label = f"{dataset_name}_chunk_{args.chunk_id:04d}" if args.chunk_id is not None else f"{dataset_name}_chunk_local"
    hdf5_tmp_path = Path(tmpdir) / f"{chunk_label}.hdf5"
    hdf5_final_path = dataset_out_dir / f"{chunk_label}.hdf5"
    status_path = dataset_out_dir / f"{chunk_label}_status.json"

    # Skip if chunk already completed successfully
    if status_path.exists():
        try:
            prev = json.loads(status_path.read_text())
            if prev.get("ok", 0) == len(manifest) and prev.get("errors", -1) == 0:
                log.info(f"Chunk already complete ({prev['ok']} ok), skipping")
                return
            log.info(
                f"Previous run incomplete (ok={prev.get('ok')}, "
                f"errors={prev.get('errors')}), reprocessing"
            )
        except Exception:
            pass

    # Check blender
    if not BLENDER_BIN.exists():
        log.error(f"Blender not found at {BLENDER_BIN}")
        return

    # 4. Process in mini-batches
    ok = 0
    errors = []
    t_start = time.time()

    with h5py.File(hdf5_tmp_path, "w") as hf:
        # Write metadata attrs
        hf.attrs["format"] = "trellis_preproc"
        hf.attrs["num_views"] = args.num_views
        hf.attrs["resolution"] = args.resolution
        hf.attrs["engine"] = args.engine
        hf.attrs["voxel_resolution"] = args.voxel_resolution
        if args.chunk_id is not None:
            hf.attrs["chunk_id"] = args.chunk_id
            hf.attrs["num_chunks"] = args.num_chunks

        pbar = tqdm(total=len(manifest), desc="Processing", unit="mesh")

        for batch_start in range(0, len(manifest), args.batch_size):
            batch = manifest[batch_start : batch_start + args.batch_size]
            batch_work = work_dir / f"batch_{batch_start}"
            batch_work.mkdir(parents=True, exist_ok=True)

            # Generate views for each sequence
            def _process(record: MeshRecord) -> dict:
                views = generate_views(
                    args.num_views, radius=2.0, fov_deg=40.0, uid=record.uid
                )
                views_json = json.dumps(views)
                return process_one(
                    record=record,
                    work_dir=batch_work,
                    views_json=views_json,
                    blender_bin=BLENDER_BIN,
                    render_script=RENDER_SCRIPT,
                    engine=args.engine,
                    resolution=args.resolution,
                    samples=args.samples,
                    voxel_resolution=args.voxel_resolution,
                    blender_timeout=args.blender_timeout,
                )

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_process, r): r for r in batch}
                for future in as_completed(futures):
                    result = future.result()
                    pbar.update(1)

                    if result["status"] == "success":
                        ok += 1
                        pack_result_to_hdf5(hf, result, args.num_views, args.resolution)
                        pbar.set_postfix_str(
                            f'{result["uid"].split("/")[-1]}: '
                            f'{result["time"]:.0f}s'
                        )
                    else:
                        errors.append(result)
                        pbar.set_postfix_str(
                            f'{result["uid"].split("/")[-1]}: '
                            f'{result["status"]}'
                        )

            # Clean up batch working directory to free TMPDIR space
            shutil.rmtree(batch_work, ignore_errors=True)

        pbar.close()

    # 5. Move HDF5 to final location
    log.info(f"Moving {hdf5_tmp_path} -> {hdf5_final_path}")
    shutil.move(str(hdf5_tmp_path), str(hdf5_final_path))

    # 6. Summary
    elapsed = time.time() - t_start
    hdf5_size = hdf5_final_path.stat().st_size / 1024 / 1024
    log.info("=" * 60)
    log.info(
        f"Done in {elapsed:.0f}s | OK: {ok} | Errors: {len(errors)} | "
        f"HDF5: {hdf5_size:.0f} MB"
    )
    if errors:
        log.info("Failed sequences:")
        for e in errors[:20]:
            log.info(f"  {e['uid']}: {e['status']} - {e.get('error', '')[:100]}")
        if len(errors) > 20:
            log.info(f"  ... and {len(errors) - 20} more")

    # Write status JSON
    status_path.write_text(
        json.dumps(
            {
                "chunk_id": args.chunk_id,
                "total": len(manifest),
                "ok": ok,
                "errors": len(errors),
                "elapsed_s": elapsed,
                "error_details": [
                    {"uid": e["uid"], "status": e["status"], "error": e.get("error", "")}
                    for e in errors
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
