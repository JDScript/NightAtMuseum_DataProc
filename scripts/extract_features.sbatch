#!/bin/bash
#SBATCH --job-name=dino_features
#SBATCH --account=torch_pr_51_tandon_advanced
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/features_%j.out
#SBATCH --error=logs/features_%j.err

mkdir -p "${SLURM_SUBMIT_DIR}/logs"

echo "=== DINOv2 Feature Extraction ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}, GPUs/node: ${SLURM_GPUS_ON_NODE}"
echo "Nodelist: ${SLURM_JOB_NODELIST}"
echo "================================="

cd "${SLURM_SUBMIT_DIR}"

# Multi-node torchrun setup
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=29500
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
NNODES=${SLURM_JOB_NUM_NODES}

echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}, GPUS_PER_NODE=${GPUS_PER_NODE}"

echo ""
echo "=== Processing animo ==="
date

srun --ntasks-per-node=1 \
    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --rdzv_id="${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        source/extract_features.py \
            --dataset animo \
            --batch-size 32 \
            --prefetch 16 \
            --load_workers 16 \
            --output_dir "/scratch/zj2631/data/trellis_features_new"

echo "=== animo done (exit code $?) ==="
date
