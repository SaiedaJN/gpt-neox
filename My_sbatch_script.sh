#!/bin/bash
#SBATCH --job-name="neox"
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --account=m4093

#activate environment
module load conda
conda activate gpt-neox-conda
module load pytorch

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Your hostfile creation script from above
/global/u1/s/saiedaaz/LLM4VV-finetunning/gpt-neox/Write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=/pscratch/sd/s/saiedaaz/LLM-Project/QwenTraining/hostfiles/hosts_$SLURM_JOBID

# Launch training
python deepy.py train.py /global/homes/s/saiedaaz/LLM4VV-finetunning/gpt-neox/configs/qwen2_5_7b.yml

