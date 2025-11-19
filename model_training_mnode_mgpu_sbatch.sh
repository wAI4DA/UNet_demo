#!/bin/bash
#SBATCH --account gpu-wizard #Replace with whatever your GPU project is, if not gpu-wizard
#SBATCH -t 03:00:00 #Change if more time is needed
#SBATCH -J mnmg_UNet_training
#SBATCH -o JOB_LOG_%x_%J.out

#SBATCH --qos=gpu #change to gpuwf if using GPU windfall allocation
#SBATCH --partition=u1-h100

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # BACK TO: one launcher task per node
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2                 # 2 GPUs per node
# NO --gpus-per-task - let all GPUs be visible to the launcher task

#SBATCH --export=ALL

echo "Starting job"

# --- Threading: 2 ranks × 2 threads = 4 CPUs/node ---
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# --- NCCL / rendezvous ---
#export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Network (tweak iface if needed, e.g., ib0, enp175s0f0np0)
# export NCCL_SOCKET_IFNAME=^lo,docker0

# Rendezvous (shared by all nodes)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export RDZV_BACKEND=c10d
export RDZV_ENDPOINT=${MASTER_ADDR}:${MASTER_PORT}
export RDZV_ID=$SLURM_JOB_ID

export WORLD_SIZE=4 #For use on 4 nodes - change as needed

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NODEID=$SLURM_NODEID / SLURM_NNODES=$SLURM_NNODES"

echo "starting at $(date)"
startTime=$(date +%s)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


###############
nvidia-smi
echo $PWD
source /scratch3/BMC/wrfruc/gge/AI/ai4da/load_ai4da.sh

###############

# --- Quick sanity check on *every* node about GPU visibility/binding ---
srun --ntasks-per-node=2 --mpi=none \
     --gres=gpu:2 \
     bash -lc 'echo "Host: $(hostname)"; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true'

# --- Launch: one torchrun per node; each spawns 2 ranks (1 per GPU) ---

srun --ntasks-per-node=2 --mpi=none \
     --gres=gpu:2 \
     --gpu-bind=map_gpu:0,1 \
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node=2 \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend="${RDZV_BACKEND}" \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    --rdzv_id="${RDZV_ID}" \
    model_training_mnode_mgpu.py

stopTime=$(date +%s)
echo "finished at $(date)"
echo "runTime=$((stopTime-startTime))"

