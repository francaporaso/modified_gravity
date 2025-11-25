import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

grav = ['GR', 'fR']
voidtype = ['R', 'S']
redshift = ['010-015', '015-020', '020-025', '025-030', '050-055', '055-060']

R = {}
Sigma = {}
DSigma_t = {}
DSigma_x = {}
covS = {}
covDSt = {}
covDSx = {}
jack = {}

for g in grav:
    R[g] = {}
    Sigma[g] = {}
    DSigma_t[g] = {}
    DSigma_x[g] = {}
    covS[g] = {}
    covDSt[g] = {}
    covDSx[g] = {}
    jack[g] = {}
    
    for t in voidtype:
        R[g][t] = {}
        Sigma[g][t] = {}
        DSigma_t[g][t] = {}
        DSigma_x[g][t] = {}
        covS[g][t] = {}
        covDSt[g][t] = {}
        covDSx[g][t] = {}
        jack[g][t] = {}
        
        for z in redshift:

            with fits.open(f'results/Rv08-30/lensing_{g}_L09_Rv08-30_z{z}_type{t}_binlin.fits') as f:
                R[g][t][z] = np.linspace(f[0].header['RIN'], f[0].header['ROUT'], f[0].header['N'])
                Sigma[g][t][z] = f[1].data.Sigma
                DSigma_t[g][t][z] = f[1].data.DSigma_t
                DSigma_x[g][t][z] = f[1].data.DSigma_x
                covS[g][t][z] = f[2].data
                covDSt[g][t][z] = f[3].data
                covDSx[g][t][z] = f[4].data
                try:
                    jack[g][t][z] = {
                        'Sigma':f[5].data,
                        'DSigma_t':f[6].data,
                        'DSigma_x':f[7].data,
                    }
                except IndexError:
                    print(f"jackknife not saved for {t=}, {z=}, {g=}")


def plot_profiles1():
    fig, axes = plt.subplots(len(voidtype), len(redshift))

    for g in grav:
        for i, t in enumerate(voidtype):
            for j, z in enumerate(redshift):
                
                axes[i,j].errorbar(
                    R[g][t][z], 
                    Sigma[g][t][z],
                    np.sqrt(np.diag(covS[g][t][z])),
                    capsize=3,
                    fmt='.-',
                    label=g+t+z
                )
                axes[i,j].legend()

    plt.show()

def plot_profiles2():
    colors = {
        'GR':plt.get_cmap('Blues', len(redshift)+2),
        'fR':plt.get_cmap('Reds', len(redshift)+2),
    }

    fig, axes = plt.subplots(1,2,sharex=True, sharey=True)
    for g in grav:
        for i, z in enumerate(redshift):
            axes[i].errorbar(
                R[g]['R'][z],
                Sigma[g]['R'][z],
                np.sqrt(np.diag(covS[g]['R'][z])),
                c=colors[g](i+2), capsize=2, fmt='.-', label=g+z
            )

    plt.show()