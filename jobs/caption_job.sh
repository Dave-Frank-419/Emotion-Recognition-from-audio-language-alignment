#!/usr/bin/env bash
#SBATCH --job-name=captions
#SBATCH --time=3-00:00:00
#SBATCH --partition=students
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=./%x_%A.out

cd /data/chi-gpu1/ge96xah/Emotion-Recognition-from-audio-language-alignment
source .devenv/state/venv/bin/activate

export HF_HOME=/data/chi-gpu1/ge96xah/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u src/llm_caption_generator.py
