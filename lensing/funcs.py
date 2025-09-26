#--------------------------- Functions --------------------------------------------
import numpy as np
from astropy.coordinates import angular_separation, position_angle
#from astropy.constants import G,c,M_sun,pc
#from astropy.io import fits
from astropy.table import Table
import healpy as hp
#parameters
# cvel = c.value;    # Speed of light (m.s-1)
# G    = G.value;    # Gravitational constant (m3.kg-1.s-2)
# pc   = pc.value    # 1 pc (m)
# Msun = M_sun.value # Solar mass (kg)

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

def normalize_cov(cov):
    norm_cov = np.zeros_like(cov)
    for i in range(len(cov)):
        for j in range(len(cov)):
            norm_cov[i,j]=cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
    return norm_cov

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

def lenscat_load(name,
                 Rv_min, Rv_max, z_min, z_max, delta_min, delta_max, rho1_min=-1.0, rho1_max=0.0, flag=2,
                 NCHUNKS:int=1, NK:int=1, octant=False, MICE=False, fullshape=True):

    if MICE:
        RV,RA,DEC,Z,R1,R2 = 1,2,3,4,8,9
    else:
        RV,RA,DEC,Z,R1,R2 = 0,1,2,3,7,8
    ## 0:Rv, 1:ra, 2:dec, 3:z, 4:xv, 5:yv, 6:zv, 7:rho1, 8:rho2, 9:logp, 10:diff CdM y CdV, 11:flag
    ## CdM: centro de masa
    ## CdV: centro del void
    try: 
        L = np.loadtxt("/home/fcaporaso/cats/L768/"+name, dtype='f4').T
    except:
        L = np.loadtxt(name, dtype='f4').T
    
    if octant: 
        print(' Using octant '.center(40,'#'), flush=True)
        # selecciono los void en un octante
        eps = 6.0 ## sale de tomar el angulo substendido por el void más grande al redshift más bajo
        L = L[:, (L[RA] >= 0.0+eps) & (L[RA] <= 90.0-eps) & (L[DEC]>= 0.0+eps) & (L[DEC] <= 90.0-eps)]

    sqrt_NK = int(np.sqrt(NK))
    NNN = len(L[0]) ##total number of voids
    ra,dec = L[RA],L[DEC]
    K    = np.zeros((NK+1,NNN), dtype=bool)
    K[0] = np.ones(NNN, dtype=bool)

    ramin  = np.min(ra)
    cdec   = np.sin(np.deg2rad(dec))
    decmin = np.min(cdec)
    dra    = ((np.max(ra)+1.e-5) - ramin)/sqrt_NK
    ddec   = ((np.max(cdec)+1.e-5) - decmin)/sqrt_NK

    c = 1
    for a in range(sqrt_NK): 
        for d in range(sqrt_NK): 
            mra  = (ra  >= ramin + a*dra)&(ra < ramin + (a+1)*dra) 
            mdec = (cdec >= decmin + d*ddec)&(cdec < decmin + (d+1)*ddec) 
            K[c] = ~(mra&mdec)
            c += 1

    mask = (L[RV] >= Rv_min) & (L[RV] < Rv_max) & (L[Z] >= z_min) & (L[Z] < z_max) & (
            L[R1] >= rho1_min) & (L[R1] < rho1_max) & (L[R2] >= delta_min) & (L[R2] < delta_max) & (L[11] >= flag)

    nvoids = mask.sum()
    if fullshape:
        L = L[:, mask]
    else:
        L = L[[RV,RA,DEC,Z]][:, mask]

    if bool(NCHUNKS-1):
        if NCHUNKS > nvoids:
            NCHUNKS = nvoids
        lbins = round(nvoids/NCHUNKS)
        slices = (np.arange(lbins)+1)*NCHUNKS
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)
        K = np.split(K.T, slices)

    return L, K, nvoids

def sourcecat_load(name,NSIDE=64):
    
    folder = '/home/fcaporaso/cats/L768/'
    S = Table.read(folder+name, memmap=True, format='fits')
    
    return S
