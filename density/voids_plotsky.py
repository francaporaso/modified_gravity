import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from astropy.cosmology import LambdaCDM
import sys
sys.path.append('/home/fcaporaso/modified_gravity/lensing')
from funcs import lenscat_load

cosmo = LambdaCDM(H0=100.0, Om0=0.3089, Ode0=0.6911)

def rad2ang(rv, redshift,cosmo=cosmo):
    radxmpc = cosmo.arcsec_per_kpc_proper(redshift).to('rad/Mpc').value
    return radxmpc*rv

voids = {
    'LCDM':lenscat_load('voids_LCDM_09.dat',0.0,50.0,0.2,0.21,-1.0,0.0,-1.0,0.0,2.0,1,1)[0],
    'fR':lenscat_load('voids_fR_09.dat',0.0,50.0,0.2,0.21,-1.0,0.0,-1.0,0.0,2.0,1,1)[0],
}

def void_plot(voids=voids):
    ## chequear como poner los radios con la deformacion de los polos... 
    pos = {
        'LCDM':(np.deg2rad(voids['LCDM'][1]), np.sin(np.deg2rad(voids['LCDM'][2]))),
        'fR':(np.deg2rad(voids['fR'][1]), np.sin(np.deg2rad(voids['fR'][2]))),
    }
    
    fig,ax = plt.subplots(figsize=(10,5))
    ax.scatter(
        *pos['LCDM'],
        c='b',alpha=0.2,s=2
    )
    ax.scatter(
        *pos['fR'],
        c='r',alpha=0.2,s=2
    )
    for v in voids['LCDM'].T:
        circ = Circle(
            (np.deg2rad(v[1]), np.sin(np.deg2rad(v[2]))),
            radius=rad2ang(v[0],v[3]),
            color='b', fill=False
        )
        ax.add_patch(circ)
    for v in voids['fR'].T:
        circ = Circle(
            (np.deg2rad(v[1]), np.sin(np.deg2rad(v[2]))),
            radius=rad2ang(v[0],v[3]),
            color='r', fill=False
        )
        ax.add_patch(circ)
        
    return fig, ax
    
if __name__=='__main__':
    void_plot()