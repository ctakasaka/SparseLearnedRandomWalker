#!/bin/bash
#SBATCH --job-name=train_slrw_256
#SBATCH --output=train_slrw_256_output.txt
#SBATCH --error=train_slrw_256_error.txt
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10
#SBATCH --time=72:00:00
#SBATCH --mem=32GB

INPUT=train_hp_tuning_config.yaml

CURRENT_DIR=`pwd`/orion
CONFIG=$CURRENT_DIR/$INPUT

export EXPERIMENT_DIR=$CURRENT_DIR
export ORION_CONFIG=$EXPERIMENT_DIR/orion_config.yaml
export ORION_DB_ADDRESS=$EXPERIMENT_DIR/'orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

# Activate env
module load anaconda/3
conda activate /home/mila/p/pedro.ferraz/miniconda3/envs/SparseLearnedRandomWalker

# Start Orion
echo "Start orion hunt"
orion -v hunt --config $ORION_CONFIG python train.py --config $CONFIG
