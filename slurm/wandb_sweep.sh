#!/bin/bash

#SBATCH --partition=long                                 # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=72:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/o/omar.salemohamed/wandb-sweep-%j.out  # Write the log on scratch

# 1. Load the required modules
module --quiet load python/3.9

# 2. Load environment
source ~/.virtualenvs/tasksim/bin/activate

# # 3. Copy your dataset on the compute node
# cp /network/data/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
wandb agent clip_cl/CL-Similarity/uv2kvsoa

# # 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/<to_save> /network/scratch/<u>/<username>/