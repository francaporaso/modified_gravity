#--------------------------- Functions --------------------------------------------
import numpy as np
from astropy.coordinates import angular_separation, position_angle
#from astropy.constants import G,c,M_sun,pc
#from astropy.io import fits
from astropy.table import Table
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

## TODO
## agregar de nuevo option for octant
def lenscat_load(name,
                 Rv_min, Rv_max, z_min, z_max, delta_min, delta_max, rho1_min=-1.0, rho1_max=0.0, flag=2,
                 ncores:int=1, Nk:int=1, octant=False, MICE=False, fullshape=True):

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

    sqrt_Nk = int(np.sqrt(Nk))
    NNN = len(L[0]) ##total number of voids
    ra,dec = L[RA],L[DEC]
    K    = np.zeros((Nk+1,NNN))
    K[0] = np.ones(NNN).astype(bool)

    ramin  = np.min(ra)
    cdec   = np.sin(np.deg2rad(dec))
    decmin = np.min(cdec)
    dra    = ((np.max(ra)+1.e-5) - ramin)/sqrt_Nk
    ddec   = ((np.max(cdec)+1.e-5) - decmin)/sqrt_Nk

    c = 1
    for a in range(sqrt_Nk): 
        for d in range(sqrt_Nk): 
            mra  = (ra  >= ramin + a*dra)&(ra < ramin + (a+1)*dra) 
            mdec = (cdec >= decmin + d*ddec)&(cdec < decmin + (d+1)*ddec) 
            K[c] = ~(mra&mdec)
            c += 1

    mask = (L[RV] >= Rv_min) & (L[RV] < Rv_max) & (L[Z] >= z_min) & (L[Z] < z_max) & (
            L[R1] >= rho1_min) & (L[R1] < rho1_max) & (L[R2] >= delta_min) & (L[R2] < delta_max) & (L[11] >= flag)

    nvoids = mask.sum()
    if fullshape:
        L = L[:,mask]
    else:
        L = L[[RV,RA,DEC,Z]]

    if bool(ncores-1):
        if ncores > nvoids:
            ncores = nvoids
        lbins = round(nvoids/ncores)
        slices = (np.arange(lbins)+1)*ncores
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)
        K = np.split(K.T, slices)

    return L, K, nvoids

def sourcecat_load(name):
    folder = '/home/fcaporaso/cats/L768/'
    S = Table.read(folder+name, memmap=True, format='fits')

    ra_gal_rad  = np.deg2rad(S['ra_gal'])
    dec_gal_rad = np.deg2rad(S['dec_gal'])
    S['cos_ra_gal']  = np.cos(ra_gal_rad)
    S['sin_ra_gal']  = np.sin(ra_gal_rad)
    S['cos_dec_gal'] = np.cos(dec_gal_rad)
    S['sin_dec_gal'] = np.sin(dec_gal_rad)

    return S
    #return np.vstack([S.ra_gal, S.dec_gal, S.true_redshift_gal, S.kappa, S.gamma1, S.gamma2])


#####################
### BALLTREE - probar
# import sklearn
# S = fits...
# X = np.deg2rad([S.dec_gal, S.ra_gal]).T ## shape = (len(S),2) 
# BT = sklearn(X, leaf_size=2, metric='haversine') ## haversine == great-circle
# idx = BT.query_radius([np.deg2rad([DEC0,RA0])], r=0.8) # r: great-dist en radianes
# catdata = S.ra_gal[idx], S.dec_gal[idx]