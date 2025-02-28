#--------------------------- Functions --------------------------------------------

import numpy as np
from astropy.coordinates import angular_separation, position_angle
from astropy.constants import G,c,M_sun,pc
from astropy.io import fits

#parameters
cvel = c.value;    # Speed of light (m.s-1)
G    = G.value;    # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value    # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

def cov_matrix(array):
        
    K = len(array)
    Kmean = np.average(array,axis=0)
    bins = array.shape[1]
    
    COV = np.zeros((bins,bins))
    
    for k in range(K):
        dif = (array[k]- Kmean)
        COV += np.outer(dif,dif)        
    
    COV *= (K-1)/K
    return COV

def eq2p2(ra_gal, dec_gal, RA0,DEC0):
    """
    angular separation and position angle from centre (RA0,DEC0) to gal position (ra_gal, dec_gal)
    returns two angles in radians
    all parameters must be in radians.
    """

    ra_prime = ra_gal - RA0

    rad = angular_separation(ra_prime, dec_gal, 0.0, DEC0)
    theta = position_angle(0.0, DEC0, ra_prime, dec_gal).value

    return rad, theta

def lenscat_load(lens_cat,
                 Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, flag,
                 ncores:int, nk:int):

    ## 0:Rv, 1:ra, 2:dec, 3:z, 4:xv, 5:yv, 6:zv, 7:rho1, 8:rho2, 9:logp, 10:diff CdM y CdV, 11:flag
    ## CdM: centro de masa
    ## CdV: centro del void
    try: 
        L = np.loadtxt("/home/fcaporaso/cats/L768/"+lens_cat).T
    except:
        L = np.loadtxt(lens_cat).T
    # if octant: ## octant deprecated
    #     # selecciono los void en un octante
    #     eps = 1.0
    #     L = L[:, (L[1] >= 0.0+eps) & (L[1] <= 90.0-eps) & (L[2]>= 0.0+eps) & (L[2] <= 90.0-eps)]

    sqrt_nk = int(np.sqrt(nk))
    NNN = len(L[0]) ##total number of voids
    ra,dec = L[1],L[2]
    K    = np.zeros((nk+1,NNN))
    K[0] = np.ones(NNN).astype(bool)

    ramin  = np.min(ra)
    cdec   = np.sin(np.deg2rad(dec))
    decmin = np.min(cdec)
    dra    = ((np.max(ra)+1.e-5) - ramin)/sqrt_nk
    ddec   = ((np.max(cdec)+1.e-5) - decmin)/sqrt_nk

    c = 1
    for a in range(sqrt_nk): 
        for d in range(sqrt_nk): 
            mra  = (ra  >= ramin + a*dra)&(ra < ramin + (a+1)*dra) 
            mdec = (cdec >= decmin + d*ddec)&(cdec < decmin + (d+1)*ddec) 
            K[c] = ~(mra&mdec)
            c += 1

    mask = (L[0] >= Rv_min) & (L[0] < Rv_max) & (L[3] >= z_min) & (L[3] < z_max) & (
            L[7] >= rho1_min) & (L[7] < rho1_max) & (L[8] >= rho2_min) & (L[8] < rho2_max) & (L[11] >= flag)

    nvoids = mask.sum()
    L = L[:,mask]
    ## no hace falta con nuevos voids
    # L[1][L[1]<0.0] = L[1][L[1]<0.0] + np.float32(360.0) # corrección ra sources in (0,360)

    if bool(ncores-1):
        if ncores > nvoids:
            ncores = nvoids
        lbins = int(round(nvoids/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)
        K = np.split(K.T, slices)

    return L, K, nvoids

def sourcecat_load(sourcename):
    folder = '/home/fcaporaso/cats/L768/'
    with fits.open(folder+sourcename) as f:
        mask = np.abs(f[1].data.gamma1) < 10.0
        S = f[1].data[mask]

    return S
    # return S.ra_gal, S.dec_gal, S.true_redshift_gal, S.kappa, S.gamma1, S.gamma2
