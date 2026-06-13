#!/usr/bin/env bash
#SBATCH --job-name=features_templates
#SBATCH --time=12:00:00
#SBATCH --partition=students
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=./%x_%A.out

direnv allow . && eval "$(direnv export bash)"

python -u src/features.py
python -u src/template_creation.py -dataset /nas/student/DavidFrank/MSP-Podcast/converted -features features.csv --dest templates
