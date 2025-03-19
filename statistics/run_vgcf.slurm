#!/bin/bash
#SBATCH --job-name=vgcf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=3-00:00:00
#SBATCH --partition=batch
## An email is sent when the job is canceled or finished
## Change "root" for your e-mail address
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=fmcaporaso@mi.unc.edu.ar


. /etc/profile
##source $HOME/.bashrc

##cd /mnt/projects/lensing/HALO_SHAPE/MICEv1.0/codes_HS_micev1.0/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_py39

## vgcf.py hace ambos LCDM y fR...

## Listos
##srun python vgcf.py --sample 'test1' --Rv_min 10.0 --Rv_max 12.0 --z_min 0.1 --z_max 0.2 --ndots 22 --ncores 56
##srun python vgcf.py --sample 'test0' --Rv_min 5.0 --Rv_max 10.0 --z_min 0.1 --z_max 0.2 --ndots 22 --ncores 56
##srun python vgcf.py --sample 'test2' --Rv_min 12.0 --Rv_max 14.0 --z_min 0.1 --z_max 0.2 --ndots 22 --ncores 56
##srun python vgcf.py --sample 'test3' --Rv_min 14.0 --Rv_max 16.0 --z_min 0.1 --z_max 0.2 --ndots 22 --ncores 56
##srun python vgcf.py --sample 'test4' --Rv_min 16.0 --Rv_max 18.0 --z_min 0.1 --z_max 0.2 --ndots 22 --ncores 56
##srun python vgcf.py --sample 'test5' --Rv_min 18.0 --Rv_max 20.0 --z_min 0.1 --z_max 0.2 --ndots 22 --ncores 56
##srun python vgcf.py --sample 'test6' --Rv_min 20.0 --Rv_max 22.0 --z_min 0.1 --z_max 0.2 --ndots 22 --ncores 56
##srun python vgcf.py --sample 'test7' --Rv_min 22.0 --Rv_max 50.0 --z_min 0.1 --z_max 0.2 --ndots 22 --ncores 56

srun python vgcf.py --sim 'both' --sample 'test8_debug' --Rv_min 7.30 --Rv_max 10.0 --z_min 0.45 --z_max 0.46 --ndots 22 --ncores 56
srun python vgcf.py --sim 'both' --sample 'test9_debug' --Rv_min 10.0 --Rv_max 12.0 --z_min 0.45 --z_max 0.46 --ndots 22 --ncores 56
## srun python vgcf.py --sample 'test10' --Rv_min 12.0 --Rv_max 14.0 --z_min 0.45 --z_max 0.5 --ndots 22 --ncores 56
## srun python vgcf.py --sample 'test11' --Rv_min 14.0 --Rv_max 16.0 --z_min 0.45 --z_max 0.5 --ndots 22 --ncores 56
## srun python vgcf.py --sample 'test12' --Rv_min 16.0 --Rv_max 18.0 --z_min 0.45 --z_max 0.5 --ndots 22 --ncores 56
## srun python vgcf.py --sample 'test13' --Rv_min 18.0 --Rv_max 20.0 --z_min 0.45 --z_max 0.5 --ndots 22 --ncores 56
## srun python vgcf.py --sample 'test14' --Rv_min 20.0 --Rv_max 22.0 --z_min 0.45 --z_max 0.5 --ndots 22 --ncores 56
## srun python vgcf.py --sample 'test15' --Rv_min 22.0 --Rv_max 30.0 --z_min 0.45 --z_max 0.5 --ndots 22 --ncores 56
