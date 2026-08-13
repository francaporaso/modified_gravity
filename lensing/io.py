from astropy.io import fits
from astropy.table import Table
import numpy as np
from pathlib import Path

def read_lens_catalog(filename, cat='sparkling', **kwargs):

    filepath = Path(filename).expanduser()

    if cat == 'sparkling':
        return read_sparkling(filepath, **kwargs)

    elif cat == 'redmapper':
        raise NotImplementedError

def read_source_catalog():
    pass

def read_sparkling(filename,
                   Rv_min, Rv_max, 
                   z_min, z_max, 
                   delta_min, delta_max, 
                   rho1_min=-1.0, rho1_max=0.0, 
                   flag=2,
                   has_id=False, 
                   fullshape=True):

    if has_id:
        RV,RA,DEC,Z,R1,R2 = 1,2,3,4,8,9
    else:
        RV,RA,DEC,Z,R1,R2 = 0,1,2,3,7,8
    # 0:Rv, 1:ra, 2:dec, 3:z, 4:xv, 5:yv, 6:zv, 7:rho1, 8:rho2, 9:logp, 10:diff CdM y CdV, 11:flag
    # CdM: centro de masa
    # CdV: centro del void
    L = np.loadtxt(filename, dtype='f4').T

    mask = (
        (L[RV] >= Rv_min) & (L[RV] < Rv_max) & 
        (L[Z] >= z_min) & (L[Z] < z_max) & 
        (L[R1] >= rho1_min) & (L[R1] < rho1_max) & 
        (L[R2] >= delta_min) & (L[R2] < delta_max) & 
        (L[11] >= flag)
    )

    nvoids = mask.sum()
    if fullshape:
        L = L[:, mask]
    else:
        L = L[[RV,RA,DEC,Z]][:, mask]

    return L, nvoids

