#!/bin/bash
#SBATCH --job-name=glb_export
#SBATCH --account=torch_pr_51_tandon_advanced
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30GB
#SBATCH --time=12:00:00
#SBATCH --array=0-255
#SBATCH --output=logs/export_%A_%a.out
#SBATCH --error=logs/export_%A_%a.err

NUM_CHUNKS=256
WORKERS=4
IMAGE="/share/apps/images/ubuntu-24.04.3.sif"

mkdir -p "${SLURM_SUBMIT_DIR}/logs"

echo "=== SLURM Array Export ==="
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname), CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Chunk: ${SLURM_ARRAY_TASK_ID} / ${NUM_CHUNKS}"
echo "========================="

cd "${SLURM_SUBMIT_DIR}"

singularity exec "${IMAGE}" \
    .venv/bin/python source/batch_export.py \
        --workers "${WORKERS}" \
        --chunk-id "${SLURM_ARRAY_TASK_ID}" \
        --num-chunks "${NUM_CHUNKS}"

echo "=== Chunk ${SLURM_ARRAY_TASK_ID} finished with exit code $? ==="
