#!/bin/bash
#SBATCH --job-name=specdecode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/SD_%j.out
#SBATCH --error=logs/SD_%j.err

# Email notifications
#SBATCH --mail-user=xc562@cornell.edu   # <-- Replace with your email if needed
#SBATCH --mail-type=END,FAIL            # Notify on job end or failure

# Optional: load modules or source environment
source /share/apps/anaconda3/2022.10/etc/profile.d/conda.sh
conda activate vllm  # or your env name

# Run the experiment Python script
python SD.py

# Move output files into sd_results folder
mv specdecode_summary.csv SD_results/specdecode_summary.csv
mv specdecode_detailed.csv SD_results/specdecode_detailed.csv
