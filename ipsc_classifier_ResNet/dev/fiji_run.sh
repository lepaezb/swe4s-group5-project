#!/bin/bash
#SBATCH --partition=nvidia-a100
#SBATCH --time=1:00:00
#SBATCH --job-name=ipsc_classification_model
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lupa9404@colorado.edu
#SBATCH --output=/scratch/Users/lupa9404/ipsc_classification_model%j.out
#SBATCH --error=/scratch/Users/lupa9404/ipsc_classification_model%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Print the job and runtime details
pwd; hostname; date; uptime; id; df -h

# Activate virtual python environment
module load python/3.11.3
. /scratch/Users/lupa9404/python_env/swe4s/bin/activate

# Run script
python3 ipsc_classification_fiji.py
