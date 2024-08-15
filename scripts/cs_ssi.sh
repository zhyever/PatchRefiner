#!/bin/bash

WEIGHT=('1e-2' '5e-2' '7e-2' '3e-2' '9e-2')
# WEIGHT=('2e-1' '75e-2' '375e-3')

for exp_weight in "${WEIGHT[@]}"
do

# Compose slurm
cat <<<"#!/usr/bin/env bash

#SBATCH --job-name=ssigm-${exp_weight}
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH -o /ibex/ai/home/liz0l/codes/PatchRefiner/work_dir/logs/patchrefiner_zoedepth/pr_ssigm_cs-${exp_weight}.%J.out
#SBATCH -e /ibex/ai/home/liz0l/codes/PatchRefiner/work_dir/logs/patchrefiner_zoedepth/pr_ssigm_cs-${exp_weight}.%J.err

# activate env
conda activate torch2

# set pythonpath
export PYTHONPATH="${PYTHONPATH}:/ibex/ai/home/liz0l/codes/PatchRefiner"
export PYTHONPATH="${PYTHONPATH}:/ibex/ai/home/liz0l/codes/PatchRefiner/external"

# switch to 11.8 manually
module add cuda
module switch cuda/12.2

# cd xxx may not be necessary to cd ...
cd /ibex/ai/home/liz0l/codes/PatchRefiner

# ssigm
bash ./tools/dist_train.sh configs/patchrefiner_zoedepth_online_pesudo/pr_ssi_midas_cs.py 4 --work-dir ./work_dir/zoedepth/cs --log-name ssi_${exp_weight} --tag pr --cfg-options model.edge_loss_weight=${exp_weight} --debug"> ./work_dir/scripts/ssi_${exp_weight}.slurm

# Submit slurm
sbatch ./work_dir/scripts/ssi_${exp_weight}.slurm

done

