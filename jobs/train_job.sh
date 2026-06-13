#!/usr/bin/env bash
#SBATCH --job-name=clap_train
#SBATCH --time=7-00:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=./%x_%A.out

direnv allow . && eval "$(direnv export bash)"

export HF_HOME=/data/chi-gpu1/ge96xah/hf_cache

python -u src/ParaCLAP.py
