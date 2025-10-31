#!/bin/bash

# Job Flags
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=16G

# Set up environment
module load miniforge

echo "Starting FSNet evaluation job..."
# nvidia-smi