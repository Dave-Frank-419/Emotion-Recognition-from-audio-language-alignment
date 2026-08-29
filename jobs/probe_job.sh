#!/usr/bin/env bash
#SBATCH --job-name=backbone_probe
#SBATCH --time=01:30:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=./%x_%A.out

cd /data/chi-gpu1/ge96xah/Emotion-Recognition-from-audio-language-alignment
source .devenv/state/venv/bin/activate

export HF_HOME=/data/chi-gpu1/ge96xah/hf_cache

python -u src/backbone_probe.py
