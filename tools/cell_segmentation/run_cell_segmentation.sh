#!/bin/bash
#SBATCH --partition=nvidia-a100
#SBATCH --time=1:00:00
#SBATCH --job-name=cell_segmentation_cyto
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lupa9404@colorado.edu
#SBATCH --output=/scratch/Users/lupa9404/swe4s/cell_segmentation/results_cyto_%j.out
#SBATCH --error=/scratch/Users/lupa9404/swe4s/cell_segmentation/results_cyto_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Print the job and runtime details
pwd; hostname; date; uptime; id; df -h

# Activate virtual python environment
module load python/3.9.15
. /scratch/Users/lupa9404/python_env/swe4s/bin/activate

# Verify Python and PyTorch
which python
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA available:', torch.cuda.is_available());"

# Run script
python3 /scratch/Users/lupa9404/swe4s/cell_segmentation/cell_segmentation_cyto.py