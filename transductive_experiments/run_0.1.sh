#!/bin/bash
#SBATCH --job-name=sub_0.1
#SBATCH --output=transductive_0.1_output.txt
#SBATCH --error=transductive_0.1_error.txt
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=16GB

INPUT=subsampling_0.1.yaml

CURRENT_DIR=`pwd`/transductive_experiments
CONFIG=$CURRENT_DIR/$INPUT

export EXPERIMENT_DIR=$CURRENT_DIR

# Activate env
module load anaconda/3
conda activate /home/mila/p/pedro.ferraz/miniconda3/envs/SparseLearnedRandomWalker

# Start Python script
python transductive_experiment.py --config $CONFIG
