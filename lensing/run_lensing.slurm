#!/bin/bash
#SBATCH --job-name=vlensing
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

## LCDM
srun python forVoid_MG.py --sample 'lcdm_z02-03_rv10-15' --lens_cat 'voids_LCDM_09.dat' --source_cat 'l768_gr_z04-07_for02-03_19304.fits' --Rv_min 10.0 --Rv_max 15.0 --z_min 0.2 --z_max 0.3 --ndots 30--nk 169 --ncores 56

## fR
srun python forVoid_MG.py --sample 'fR_z02-03_rv10-15' --lens_cat 'voids_fR_09.dat' --source_cat 'l768_mg_z04-07_for02-03_19260.fits' --Rv_min 10.0 --Rv_max 15.0 --z_min 0.2 --z_max 0.3 --ndots 30 --nk 169 --ncores 56

