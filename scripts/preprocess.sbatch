#!/bin/bash
#SBATCH --job-name=trellis_preproc
#SBATCH --account=torch_pr_49_tandon_advanced
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --qos=cpu48 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/preproc_%A_%a.out
#SBATCH --error=logs/preproc_%A_%a.err

# ── Configuration ────────────────────────────────────────────────────────────
# Submit one job per HDF5 file with appropriate --array range:
#   sbatch --array=0-255 scripts/preprocess.sbatch animo
#   sbatch --array=0-100 scripts/preprocess.sbatch night_at_museum
#   sbatch --array=0-7   scripts/preprocess.sbatch dt4d
DATASET="${1:?Usage: sbatch scripts/preprocess.sbatch <animo|night_at_museum|dt4d>}"

WORKERS=2
NUM_VIEWS=64       # 150 = TRELLIS default, 36 = fast
ENGINE="CYCLES"     # EEVEE requires GPU/EGL, won't work on CPU nodes
SAMPLES=32          # only affects CYCLES
IMAGE="/share/apps/images/ubuntu-24.04.3.sif"
DATA_DIR="/scratch/zj2631/data"
OUTPUT_DIR="/scratch/zj2631/data/trellis_preproc"
HDF5_PATH="${DATA_DIR}/${DATASET}.hdf5"

# NUM_CHUNKS = SLURM_ARRAY_TASK_MAX + 1 (derived from --array range)
NUM_CHUNKS=$(( SLURM_ARRAY_TASK_MAX + 1 ))

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "${SLURM_SUBMIT_DIR}/logs"

echo "=== TRELLIS Preprocess ==="
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}, Mem: ${SLURM_MEM_PER_NODE}MB"
echo "Dataset: ${DATASET}, HDF5: ${HDF5_PATH}"
echo "Chunk: ${SLURM_ARRAY_TASK_ID} / ${NUM_CHUNKS}"
echo "Engine: ${ENGINE}, Views: ${NUM_VIEWS}, Samples: ${SAMPLES}"

# Determine TMPDIR for fast local I/O
WORK_DIR="${SLURM_TMPDIR:-/state/partition/job-${SLURM_JOB_ID}}"
echo "Working directory: ${WORK_DIR}"
echo "========================="

cd "${SLURM_SUBMIT_DIR}"

# ── Run ──────────────────────────────────────────────────────────────────────
singularity exec "${IMAGE}" \
    .venv/bin/python source/preprocess.py \
        --hdf5-path "${HDF5_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --chunk-id "${SLURM_ARRAY_TASK_ID}" \
        --num-chunks "${NUM_CHUNKS}" \
        --workers "${WORKERS}" \
        --num-views "${NUM_VIEWS}" \
        --engine "${ENGINE}" \
        --samples "${SAMPLES}"

EXIT_CODE=$?
echo "=== ${DATASET} chunk ${SLURM_ARRAY_TASK_ID} finished with exit code ${EXIT_CODE} ==="
exit ${EXIT_CODE}
