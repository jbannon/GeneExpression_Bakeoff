#!/bin/bash

#SBATCH --job-name=Ipi_RF
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --mem=20GB
#SBATCH --array=0-0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jjb509@nyu.edu
#SBATCH -o ./slurmouts/Ipi/RF_%j.out

module purge


settings=(SKCM.MC)

singularity exec --nv \
            --overlay /scratch/jjb509/GeneExpression_Bakeoff/src/my_overlay.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python permutation_test.py -drug Ipi -model RandomForest -settings ${settings[$SLURM_ARRAY_TASK_ID]}"
