"""Unified first-frame mesh reader for all HDF5 dataset formats.

Supports:
  - GLB format (animo.hdf5): raw GLB bytes stored as uint8
  - Mesh sequence format (night_at_museum.hdf5, dt4d.hdf5): vertices (F,V,3) + faces (N,3)

Usage:
    from mesh_toolkit import build_manifest, export_for_blender

    manifest = build_manifest(Path("/scratch/zj2631/data"))
    for record in manifest:
        path = export_for_blender(record, output_dir=Path("/tmp/meshes"))
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
from natsort import natsorted


@dataclasses.dataclass(frozen=True)
class MeshRecord:
    """Metadata for a single mesh sequence (no data loaded yet)."""

    dataset: str  # e.g. "animo", "night_at_museum", "dt4d"
    hdf5_path: str  # absolute path to the HDF5 file
    group_key: str  # HDF5 group path, e.g. "Lion/run_001"
    format: str  # "glb" or "mesh_sequence"
    uid: str  # unique ID: "{dataset}/{group_key}"


# ── Enumeration ──────────────────────────────────────────────────────────────


def _detect_format(hf: h5py.File) -> str:
    """Detect HDF5 format by checking root attributes or first group."""
    fmt = hf.attrs.get("format", "")
    if fmt == "raw_glb":
        return "glb"
    if fmt == "mesh_sequence":
        return "mesh_sequence"
    # Fallback: check first data group for 'glb' or 'vertices' dataset
    for k1 in hf:
        if k1 == "data_split":
            continue
        grp = hf[k1]
        if isinstance(grp, h5py.Group):
            for k2 in grp:
                sub = grp[k2]
                if isinstance(sub, h5py.Group):
                    if "glb" in sub:
                        return "glb"
                    if "vertices" in sub:
                        return "mesh_sequence"
    raise ValueError("Cannot detect HDF5 format")


def enumerate_sequences(hdf5_path: str | Path, dataset: str | None = None) -> list[MeshRecord]:
    """List all sequences in an HDF5 file using data_split keys.

    Reads sequence keys from ``data_split/train`` + ``data_split/val``
    for a stable, reproducible ordering (HDF5 group iteration order is
    not guaranteed).

    Args:
        hdf5_path: Path to the HDF5 file.
        dataset: Dataset name override. If None, derived from filename stem.
    """
    hdf5_path = str(hdf5_path)
    if dataset is None:
        dataset = Path(hdf5_path).stem

    records = []
    with h5py.File(hdf5_path, "r") as hf:
        fmt = _detect_format(hf)

        # Read keys from data_split (stable, explicit list)
        if "data_split" in hf:
            keys: list[str] = []
            for split in ("train", "val"):
                if split in hf["data_split"]:
                    keys.extend(hf["data_split"][split][()].astype(str).tolist())
            keys = natsorted(keys)
        else:
            # Fallback: walk groups (legacy files without data_split)
            keys = []
            for k1 in hf:
                if k1 == "data_split":
                    continue
                grp1 = hf[k1]
                if not isinstance(grp1, h5py.Group):
                    continue
                for k2 in grp1:
                    if isinstance(grp1[k2], h5py.Group):
                        keys.append(f"{k1}/{k2}")
            keys = natsorted(keys)

        for group_key in keys:
            uid = f"{dataset}/{group_key}"
            records.append(MeshRecord(
                dataset=dataset,
                hdf5_path=hdf5_path,
                group_key=group_key,
                format=fmt,
                uid=uid,
            ))

    return records


def build_manifest(
    data_dir: str | Path,
    hdf5_files: dict[str, str] | None = None,
) -> list[MeshRecord]:
    """Build the full manifest from all HDF5 files in data_dir.

    Args:
        data_dir: Directory containing HDF5 files.
        hdf5_files: Optional dict of {dataset_name: filename}. If None, uses defaults.
    """
    data_dir = Path(data_dir)
    if hdf5_files is None:
        hdf5_files = {
            "animo": "animo.hdf5",
            "night_at_museum": "night_at_museum.hdf5",
            "dt4d": "dt4d.hdf5",
        }

    records: list[MeshRecord] = []
    for dataset, filename in hdf5_files.items():
        path = data_dir / filename
        if not path.exists():
            print(f"  WARN: {path} not found, skipping")
            continue
        seq = enumerate_sequences(path, dataset)
        print(f"  {dataset}: {len(seq):,} sequences")
        records.extend(seq)

    return natsorted(records, key=lambda r: r.uid)


# ── First-frame export ───────────────────────────────────────────────────────


def export_for_blender(record: MeshRecord, output_dir: str | Path) -> Path:
    """Extract first-frame mesh and write to a file Blender can import.

    For GLB format: writes a .glb file (Blender imports directly).
    For mesh sequence: writes a .obj file with vertices[0] and faces.

    Returns the path to the exported file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use uid with slashes replaced for filesystem safety
    safe_name = record.uid.replace("/", "__")

    with h5py.File(record.hdf5_path, "r") as hf:
        grp = hf[record.group_key]

        if record.format == "glb":
            out_path = output_dir / f"{safe_name}.glb"
            glb_bytes = bytes(grp["glb"][()])
            out_path.write_bytes(glb_bytes)
            return out_path

        # mesh_sequence format
        out_path = output_dir / f"{safe_name}.obj"
        vertices = grp["vertices"][0].astype(np.float32)  # (V, 3) first frame
        faces = grp["faces"][()].astype(np.int32)  # (N, 3)
        _write_obj(out_path, vertices, faces)
        return out_path


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a minimal OBJ file."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
