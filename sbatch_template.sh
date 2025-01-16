#!/bin/bash

#SBATCH -J GENIE
#SBATCH --array TASKARRAYDEFINITION
#SBATCH -M hpd
#SBATCH --nodelist hpd-node-007
#SBATCH -t 5759
#SBATCH --mem=100G
#ESIS_PRIVATE

module load gpt-hpd

srun python3.11 -u WORKERSCRIPT --mpi 1.1.1.1
