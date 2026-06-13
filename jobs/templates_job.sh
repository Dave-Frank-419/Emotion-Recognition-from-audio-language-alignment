#!/usr/bin/env bash
#SBATCH --job-name=templates
#SBATCH --time=02:00:00
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=./%x_%A.out

direnv allow . && eval "$(direnv export bash)"

python -u src/template_creation.py -dataset /nas/student/DavidFrank/MSP-Podcast/converted -features features.csv --dest templates
