#!/bin/bash
#SBATCH --job-name=vlen_4
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
srun python forVoid_MG.py --sample 'cut5' --lens_cat 'voids_LCDM_09.dat' --source_cat 'l768_gr_z08-13_for04-05_19513.fits' --Rv_min 5.0 --Rv_max 10.0 --z_min 0.4 --z_max 0.5 --ndots 25 --nk 169 --ncores 56 -r 2 

## fR
srun python forVoid_MG.py --sample 'cut5' --lens_cat 'voids_fR_09.dat' --source_cat 'l768_mg_z08-13_for04-05_19514.fits'  --Rv_min 5.0 --Rv_max 10.0 --z_min 0.4 --z_max 0.5 --ndots 25 --nk 169 --ncores 56 -r 2

