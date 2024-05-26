#!/bin/bash

#SBATCH --job-name=SVM_Array
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --mem=20GB
#SBATCH --array=0-5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jjb509@nyu.edu
#SBATCH -o ./slurmouts/Atezo/output_%j.out

module purge


models=(LogisticRegression RandomForest Poly_SVC RBF_SVC Linear_SVC)

singularity exec --nv \
            --overlay /scratch/jjb509/GeneExpression_Bakeoff/src/my_overlay.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python monte_carlo.py --drug Atezo --model ${models[$SLURM_ARRAY_TASK_ID]} --niter 100"
