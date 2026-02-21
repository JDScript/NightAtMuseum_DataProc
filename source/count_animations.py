"""Count all animations across OVL directories using cobra-tools (no Blender needed).

Usage:
    uv run python source/count_animations.py
    uv run python source/count_animations.py --filter "Bengal*"
    uv run python source/count_animations.py --motion-only
"""

import dataclasses
import sys
from collections import defaultdict
from pathlib import Path

import tyro
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "cobra-tools"))

from generated.formats.manis import ManisFile


# ── Category classification ──────────────────────────────────────────────────

# Two-level classification: motion_type × action_type
# motion_type: how root motion is handled
#   motionextracted  — root motion separated out (clean body anim)
#   notmotionextracted — root motion baked in bones
#   shared/size-variant — generic or size-specific clips
# action_type: what the animal is doing
#   behaviour, locomotion, fighting, pounce, partials, other

MOTION_TYPES = {
    "motionextracted": "motion_extracted",
    "notmotionextracted": "not_motion_extracted",
}

ACTION_TYPES = ["behaviour", "locomotion", "fighting", "fighting_pounce", "pounce", "partials"]


def classify_manis(filename: str) -> tuple[str, str]:
    """Return (motion_type, action_type) from manis filename."""
    name = filename.lower()

    # Determine motion type
    if "notmotionextracted" in name:
        motion = "not_motion_extracted"
    elif "motionextracted" in name:
        motion = "motion_extracted"
    else:
        motion = "other"

    # Determine action type
    if "partials" in name:
        action = "partials"
    elif "fighting_pounce" in name or "fightingpounce" in name:
        action = "fighting_pounce"
    elif "pounce" in name:
        action = "pounce"
    elif "fighting" in name:
        action = "fighting"
    elif "behaviour" in name:
        action = "behaviour"
    elif "locomotion" in name:
        action = "locomotion"
    else:
        action = "other"

    return motion, action


# ── CLI ──────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """Count animations across all animals (no Blender needed)."""

    ovl_dir: Path = PROJECT_ROOT / "data" / "ovl"
    """Path to the root OVL directory."""

    filter: str = "*"
    """Glob pattern to filter animals (e.g. 'Bengal*')."""

    motion_only: bool = False
    """Only count animationmotionextracted* files (the ones we actually export)."""


# ── Scanning ─────────────────────────────────────────────────────────────────


def scan_all(args: Args) -> tuple[dict, list]:
    """Scan OVL dirs, return (animal_stats, errors).

    animal_stats[name] = {
        "manis_files": int,
        "total": int,
        "by_motion": {motion_type: int},
        "by_action": {action_type: int},
        "by_pair":   {(motion, action): int},
    }
    """
    animal_stats: dict[str, dict] = {}
    errors: list[tuple[str, str, str]] = []

    ovl_dirs = sorted(d for d in args.ovl_dir.glob(f"{args.filter}.ovl") if d.is_dir())

    for ovl_path in tqdm(ovl_dirs, desc="Scanning OVL dirs", unit="dir"):
        animal_name = ovl_path.stem

        manis_files = sorted(ovl_path.glob("*.manis"))
        if args.motion_only:
            manis_files = [
                f for f in manis_files
                if "animationmotionextracted" in f.name and "notmotion" not in f.name
            ]
        if not manis_files:
            continue

        stats = {
            "manis_files": len(manis_files),
            "total": 0,
            "by_motion": defaultdict(int),
            "by_action": defaultdict(int),
            "by_pair": defaultdict(int),
        }

        for manis_path in manis_files:
            motion, action = classify_manis(manis_path.name)
            try:
                mf = ManisFile()
                mf.load(str(manis_path))
                count = len(mf.mani_infos)
            except Exception as e:
                errors.append((animal_name, manis_path.name, str(e)))
                continue

            stats["total"] += count
            stats["by_motion"][motion] += count
            stats["by_action"][action] += count
            stats["by_pair"][(motion, action)] += count

        if stats["total"] > 0:
            animal_stats[animal_name] = stats

    return animal_stats, errors


# ── Species aggregation ──────────────────────────────────────────────────────


def get_species(animal_name: str) -> str:
    """'Bengal_Tiger_Male' -> 'Bengal_Tiger', 'Aardvark_Juvenile' -> 'Aardvark'."""
    suffixes = ("_Male", "_Female", "_Juvenile")
    for s in suffixes:
        if animal_name.endswith(s):
            return animal_name[: -len(s)]
    return animal_name


# ── Printing ─────────────────────────────────────────────────────────────────

W = 120  # output width


def print_header(title: str):
    print(f"\n{'=' * W}")
    print(f"{title:^{W}}")
    print(f"{'=' * W}")


def print_motion_x_action_table(animal_stats: dict):
    """Print a motion_type × action_type cross table."""
    print_header("MOTION TYPE × ACTION TYPE")

    # Collect all keys
    all_motions = sorted({m for s in animal_stats.values() for m in s["by_motion"]})
    all_actions = sorted({a for s in animal_stats.values() for a in s["by_action"]})

    # Aggregate
    cross: dict[tuple[str, str], int] = defaultdict(int)
    motion_totals: dict[str, int] = defaultdict(int)
    action_totals: dict[str, int] = defaultdict(int)
    grand_total = 0
    for s in animal_stats.values():
        for (m, a), c in s["by_pair"].items():
            cross[(m, a)] += c
            motion_totals[m] += c
            action_totals[a] += c
            grand_total += c

    # Print
    col_w = max(16, max((len(a) for a in all_actions), default=0) + 2)
    label_w = 24
    print(f"\n{'':>{label_w}}", end="")
    for a in all_actions:
        print(f"{a:>{col_w}}", end="")
    print(f"{'TOTAL':>{col_w}}")
    print("-" * (label_w + col_w * (len(all_actions) + 1)))

    for m in all_motions:
        print(f"{m:>{label_w}}", end="")
        for a in all_actions:
            v = cross.get((m, a), 0)
            print(f"{v:>{col_w},}" if v else f"{'—':>{col_w}}", end="")
        print(f"{motion_totals[m]:>{col_w},}")

    print("-" * (label_w + col_w * (len(all_actions) + 1)))
    print(f"{'TOTAL':>{label_w}}", end="")
    for a in all_actions:
        print(f"{action_totals[a]:>{col_w},}", end="")
    print(f"{grand_total:>{col_w},}")


def print_action_summary(animal_stats: dict):
    """Print action-type summary with animal counts, min/max/avg."""
    print_header("BY ACTION TYPE")

    all_actions = sorted({a for s in animal_stats.values() for a in s["by_action"]})

    print(f"\n{'Action':<20} {'Animals':>8} {'Clips':>10} {'Min':>6} {'Max':>6} {'Avg':>8} {'Median':>8}")
    print("-" * 70)

    total_clips = 0
    for action in all_actions:
        counts = [s["by_action"].get(action, 0) for s in animal_stats.values() if action in s["by_action"]]
        if not counts:
            continue
        counts.sort()
        n_animals = len(counts)
        total = sum(counts)
        total_clips += total
        avg = total / n_animals
        median = counts[len(counts) // 2]
        print(f"{action:<20} {n_animals:>8} {total:>10,} {counts[0]:>6} {counts[-1]:>6} {avg:>8.1f} {median:>8}")

    print("-" * 70)
    print(f"{'ALL':<20} {len(animal_stats):>8} {total_clips:>10,}")


def print_motion_summary(animal_stats: dict):
    """Print motion-type summary."""
    print_header("BY MOTION TYPE")

    all_motions = sorted({m for s in animal_stats.values() for m in s["by_motion"]})

    print(f"\n{'Motion Type':<24} {'Animals':>8} {'Clips':>10} {'Min':>6} {'Max':>6} {'Avg':>8}")
    print("-" * 66)

    for motion in all_motions:
        counts = [s["by_motion"].get(motion, 0) for s in animal_stats.values() if motion in s["by_motion"]]
        if not counts:
            continue
        counts.sort()
        n_animals = len(counts)
        total = sum(counts)
        avg = total / n_animals
        print(f"{motion:<24} {n_animals:>8} {total:>10,} {counts[0]:>6} {counts[-1]:>6} {avg:>8.1f}")


def print_species_summary(animal_stats: dict):
    """Aggregate by species (strip Male/Female/Juvenile suffix)."""
    print_header("BY SPECIES (aggregated across Male/Female/Juvenile)")

    species_data: dict[str, dict] = {}
    for name, s in animal_stats.items():
        sp = get_species(name)
        if sp not in species_data:
            species_data[sp] = {"variants": [], "total": 0, "by_action": defaultdict(int)}
        species_data[sp]["variants"].append(name)
        species_data[sp]["total"] += s["total"]
        for a, c in s["by_action"].items():
            species_data[sp]["by_action"][a] += c

    all_actions = sorted({a for s in animal_stats.values() for a in s["by_action"]})
    col_w = max(10, max((len(a) for a in all_actions), default=0) + 2)
    lbl_w = 35

    print(f"\n{'Species':<{lbl_w}} {'Vars':>5} {'Total':>8}", end="")
    for a in all_actions:
        print(f"{a:>{col_w}}", end="")
    print()
    print("-" * (lbl_w + 5 + 8 + col_w * len(all_actions)))

    for sp in sorted(species_data.keys()):
        d = species_data[sp]
        print(f"{sp:<{lbl_w}} {len(d['variants']):>5} {d['total']:>8,}", end="")
        for a in all_actions:
            v = d["by_action"].get(a, 0)
            print(f"{v:>{col_w},}" if v else f"{'—':>{col_w}}", end="")
        print()

    print("-" * (lbl_w + 5 + 8 + col_w * len(all_actions)))
    print(f"{'TOTAL':<{lbl_w}} {sum(len(d['variants']) for d in species_data.values()):>5}"
          f" {sum(d['total'] for d in species_data.values()):>8,}")
    print(f"\nUnique species: {len(species_data)}")


def print_per_animal(animal_stats: dict):
    """Print per-animal breakdown."""
    print_header("PER ANIMAL")

    all_actions = sorted({a for s in animal_stats.values() for a in s["by_action"]})
    col_w = max(10, max((len(a) for a in all_actions), default=0) + 2)
    lbl_w = 45

    print(f"\n{'Animal':<{lbl_w}} {'Files':>5} {'Total':>8}", end="")
    for a in all_actions:
        print(f"{a:>{col_w}}", end="")
    print()
    print("-" * (lbl_w + 5 + 8 + col_w * len(all_actions)))

    for name in sorted(animal_stats.keys()):
        s = animal_stats[name]
        print(f"{name:<{lbl_w}} {s['manis_files']:>5} {s['total']:>8,}", end="")
        for a in all_actions:
            v = s["by_action"].get(a, 0)
            print(f"{v:>{col_w},}" if v else f"{'—':>{col_w}}", end="")
        print()


def print_errors(errors: list):
    if not errors:
        return
    print_header(f"ERRORS ({len(errors)})")
    for animal, fname, err in errors:
        print(f"  {animal}/{fname}: {err}")


def print_final_summary(animal_stats: dict, n_scanned: int, errors: list):
    print_header("SUMMARY")
    total = sum(s["total"] for s in animal_stats.values())
    total_files = sum(s["manis_files"] for s in animal_stats.values())
    n_species = len({get_species(n) for n in animal_stats})
    print(f"  OVL directories scanned:  {n_scanned}")
    print(f"  Animals with animations:  {len(animal_stats)}")
    print(f"  Unique species:           {n_species}")
    print(f"  Total manis files:        {total_files}")
    print(f"  Total animation clips:    {total:,}")
    if errors:
        print(f"  Errors:                   {len(errors)}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: Args):
    animal_stats, errors = scan_all(args)

    n_scanned = len([d for d in args.ovl_dir.glob(f"{args.filter}.ovl") if d.is_dir()])

    print_motion_x_action_table(animal_stats)
    print_action_summary(animal_stats)
    print_motion_summary(animal_stats)
    print_species_summary(animal_stats)
    print_per_animal(animal_stats)
    print_errors(errors)
    print_final_summary(animal_stats, n_scanned, errors)


if __name__ == "__main__":
    main(tyro.cli(Args))
