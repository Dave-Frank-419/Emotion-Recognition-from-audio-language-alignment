#!/usr/bin/env bash
#SBATCH --job-name=smoke_test
#SBATCH --time=01:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=./%x_%A.out

direnv allow . && eval "$(direnv export bash)"

export HF_HOME=/data/chi-gpu1/ge96xah/hf_cache

python --version
nvidia-smi
python -u src/smoke_test.py
