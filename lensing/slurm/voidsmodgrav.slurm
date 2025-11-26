#!/bin/bash
#SBATCH --job-name=lens_c0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=3-00:00:00
#SBATCH --partition=batch
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=fmcaporaso@mi.unc.edu.ar


. /etc/profile

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate envpy

for i in {0..9}
do
    printf "\n ####### \n Running config_$i \n ####### \n \n"
    srun python voidsmodgrav.py --sample 'fullsource' --config config_$i.toml -c 56
done
