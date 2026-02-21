# NightAtMuseum DataProc

Data processing pipeline for 4D generation: extract animated meshes from Planet Zoo game assets and export as per-animation GLB files.

## Data

Source data: OVL archives from Planet Zoo, pre-extracted into `data/ovl/`. Each animal directory contains:
- **1 `.ms2` file** — mesh + skeleton
- **Multiple `.manis` files** — animation sets by category (behaviour, locomotion, fighting, pounce, etc.)
- **`.fgm` / `.tex` files** — material definitions and texture metadata (textures not yet supported)

558 OVL directories, 504 animals with animations, 169 unique species, ~80k animation clips total.

## Setup

### 1. Python environment

```bash
uv sync
```

### 2. Blender

Download [Blender 4.5.7 LTS](https://www.blender.org/download/lts/4-5/) for Linux and extract to `third_party/blender/`.

Install `bitarray` into Blender's bundled Python (required for decompressing animations):

```bash
third_party/blender/4.5/python/bin/python3.11 -m pip install bitarray
```

### 3. Data

Download `export_ovl_loop.7z`, extract to `data/ovl/`.

### 4. cobra-tools

Already included at `third_party/cobra-tools/` (v3.0.2). No extra setup needed.

## Pipeline

### Count animations (no Blender needed)

Preview the full dataset statistics before exporting:

```bash
# Full statistics across all animals
uv run python source/count_animations.py

# Filter to specific animals
uv run python source/count_animations.py --filter "Bengal*"

# Only count motionextracted animations (clean body anim)
uv run python source/count_animations.py --motion-only
```

### Batch export OVL → GLB

Exports each animal's ms2 + manis into per-animation GLB files under `data/glb/`.

```bash
# Full export with 8 parallel Blender processes
uv run python source/batch_export.py --workers 8

# Export a subset
uv run python source/batch_export.py --workers 4 --filter "Aardvark*"

# Single animal test (direct Blender call)
third_party/blender/blender --background --python source/blender_export_glb.py -- \
    --ms2 data/ovl/Aardvark_Male.ovl/aardvark_male_.ms2 \
    --manis data/ovl/Aardvark_Male.ovl/animationmotionextractedbehaviour.manisete15d87f2.manis \
    --outdir data/glb/Aardvark_Male
```

Output naming follows AniMo convention: `{ms2_stem}_{manis_stem}_{action_name}.glb`

```
data/glb/
├── Aardvark_Male/
│   ├── aardvark_male__animationmotionextractedbehaviour.maniseta599a2e7_aardvark_male_enrichmentboxshake.glb
│   ├── aardvark_male__animationmotionextractedlocomotion.manisetcb366443_aardvark_male_walkbase.glb
│   └── ...
├── Bengal_Tiger_Female/
│   └── ...
└── ...
```

## Manis File Categories

Each animal's OVL directory has up to 8 manis files spanning two motion types:

| Motion type | Description | Exported? |
|-------------|-------------|-----------|
| `animationmotionextracted*` | Root motion separated — clean body animation | Yes |
| `animationnotmotionextracted*` | Root motion baked in bones | Yes |
| `*partials*` | Additive overlays (ears, tail only) — incomplete standalone | No |

Action categories within each motion type:
- **behaviour** — eating, enrichment, mating, social
- **locomotion** — walk, run, swim, turn, climb
- **fighting** — attack, flee, taunt
- **pounce** — predator pounce (some animals only)

## File Overview

| File | Description |
|------|-------------|
| `source/batch_export.py` | Multi-threaded batch orchestrator with tqdm progress |
| `source/blender_export_glb.py` | Headless Blender script (ms2+manis → per-animation GLB) |
| `source/count_animations.py` | Count/statistics across all animals (no Blender needed) |
