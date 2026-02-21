"""Batch export all animals from OVL dirs to per-animation GLBs using parallel Blender processes.

Usage:
    uv run python source/batch_export.py --workers 8
    uv run python source/batch_export.py --workers 4 --filter "Aardvark*"
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLENDER_BIN = PROJECT_ROOT / "third_party" / "blender" / "blender"
EXPORT_SCRIPT = PROJECT_ROOT / "source" / "blender_export_glb.py"
OVL_DIR = PROJECT_ROOT / "data" / "ovl"
GLB_DIR = PROJECT_ROOT / "data" / "glb"


def discover_animals(ovl_dir: Path, pattern: str = "*") -> list[dict]:
    """Scan OVL directories and return list of animals with their ms2/manis files."""
    animals = []
    for ovl_path in sorted(ovl_dir.glob(f"{pattern}.ovl")):
        if not ovl_path.is_dir():
            continue
        ms2_files = list(ovl_path.glob("*.ms2"))
        if not ms2_files:
            log.warning(f"Skipping {ovl_path.name}: no .ms2 file")
            continue
        # Include all manis except "partials" (additive overlays — incomplete standalone).
        manis_files = sorted(
            f for f in ovl_path.glob("*.manis")
            if "partials" not in f.name.lower()
        )
        if not manis_files:
            log.warning(f"Skipping {ovl_path.name}: no .manis files")
            continue
        animals.append({
            "name": ovl_path.stem,  # e.g. "Aardvark_Male.ovl" -> "Aardvark_Male"
            "ms2": ms2_files[0],
            "manis": manis_files,
            "ovl_dir": ovl_path,
        })
    return animals


def export_animal(animal: dict, glb_dir: Path, blender_bin: Path, export_script: Path) -> dict:
    """Run Blender headless to export one animal. Returns result dict."""
    name = animal["name"]
    outdir = glb_dir / name
    t0 = time.time()

    # Skip if already exported (check if dir exists and has .glb files)
    if outdir.exists() and list(outdir.glob("*.glb")):
        return {"name": name, "status": "skipped", "glbs": len(list(outdir.glob("*.glb"))), "time": 0}

    cmd = [
        str(blender_bin), "--background", "--python", str(export_script),
        "--",
        "--ms2", str(animal["ms2"]),
        "--manis", *[str(m) for m in animal["manis"]],
        "--outdir", str(outdir),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min per animal
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            # Extract error lines from both stdout and stderr
            all_output = result.stdout + "\n" + result.stderr
            err_lines = [l for l in all_output.split("\n") if "Error" in l or "Exception" in l]
            err_summary = "; ".join(err_lines[:3]) if err_lines else result.stderr.strip()[-200:] if result.stderr.strip() else "unknown error"
            return {"name": name, "status": "error", "error": err_summary, "time": elapsed}

        # Count exported GLBs
        glb_count = len(list(outdir.glob("*.glb"))) if outdir.exists() else 0
        return {"name": name, "status": "ok", "glbs": glb_count, "time": elapsed}

    except subprocess.TimeoutExpired:
        return {"name": name, "status": "timeout", "time": 1800}
    except Exception as e:
        return {"name": name, "status": "error", "error": str(e), "time": time.time() - t0}


def main():
    parser = argparse.ArgumentParser(description="Batch export all animals to GLB")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel Blender processes")
    parser.add_argument("--filter", type=str, default="*", help="Glob pattern to filter animals (e.g. 'Aardvark*')")
    parser.add_argument("--ovl-dir", type=Path, default=OVL_DIR)
    parser.add_argument("--glb-dir", type=Path, default=GLB_DIR)
    args = parser.parse_args()

    if not BLENDER_BIN.exists():
        log.error(f"Blender not found at {BLENDER_BIN}")
        sys.exit(1)

    animals = discover_animals(args.ovl_dir, args.filter)
    log.info(f"Found {len(animals)} animals to process with {args.workers} workers")

    args.glb_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    errors = []
    total_glbs = 0
    t_start = time.time()

    pbar = tqdm(total=len(animals), desc="Exporting", unit="animal")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(export_animal, a, args.glb_dir, BLENDER_BIN, EXPORT_SCRIPT): a
            for a in animals
        }
        for future in as_completed(futures):
            result = future.result()
            status = result["status"]
            name = result["name"]

            if status == "ok":
                ok += 1
                total_glbs += result["glbs"]
                pbar.set_postfix_str(f"{name}: {result['glbs']} GLBs in {result['time']:.0f}s")
            elif status == "skipped":
                skipped += 1
                total_glbs += result["glbs"]
                pbar.set_postfix_str(f"{name}: skipped ({result['glbs']} GLBs)")
            else:
                errors.append(result)
                pbar.set_postfix_str(f"{name}: {status}")

            pbar.update(1)

    pbar.close()

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info(f"Done in {elapsed:.0f}s | OK: {ok} | Skipped: {skipped} | Errors: {len(errors)} | Total GLBs: {total_glbs}")
    if errors:
        log.info("Failed animals:")
        for e in errors:
            log.info(f"  {e['name']}: {e.get('error', e['status'])}")


if __name__ == "__main__":
    main()
