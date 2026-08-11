#--------------------------- Functions --------------------------------------------
import numpy as np
from astropy.coordinates import angular_separation, position_angle
#from astropy.constants import G,c,M_sun,pc
from astropy.io import fits
from astropy.table import Table
from kmeans_radec import kmeans_sample
#import healpy as hp
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

def get_jackknife_naive(ra_sample, dec_sample, ra_cl, dec_cl, NK=100):
    '''
    divides the sky in NK = n*m rectangular regions of the same area.
    usefull for rectangular survey geometries. 
    returns index of the jackknife region that the void belongs to.
    '''

    n = 10 if NK>=100 else 5
    m = np.ceil(NK/n).astype(int)
    print(f'{n=}')
    print(f'{m=}')
    print(f'effective NJK: {n*m}')

    ramin  = np.min(ra_sample)
    print(f'{ramin=}')
    cdec   = np.sin(np.deg2rad(dec_sample))
    decmin = np.min(cdec)
    print(f'{decmin=}')
    dra    = (np.max(ra_sample) - ramin)/n
    ddec   = (np.max(cdec) - decmin)/m

    sdec_cl = np.sin(np.deg2rad(dec_cl))

    kidx = np.zeros(len(ra_cl), dtype=int)
    c = 0
    for a in range(n):
        for d in range(m):
            mra  = (ra_cl  >= ramin + a*dra) & (ra_cl < ramin + (a+1)*dra)
            mdec = (sdec_cl >= decmin + d*ddec) & (sdec_cl < decmin + (d+1)*ddec)
            kidx[mra&mdec] = c
            c += 1

    return kidx

def get_jackknife_kmeans(ra_sample, dec_sample, ra_cl, dec_cl, nlenses, NJK):

    #K = np.zeros((NJK+1, nlenses), dtype=bool)
    #K[0] = True
    sam = np.column_stack([ra_sample, dec_sample])
    L = np.column_stack([ra_cl, dec_cl])

    km = kmeans_sample(sam, ncen=NJK, verbose=0)
    labels = km.find_nearest(L)

    #for j in range(1, NJK+1):
    #    K[j] = ~(labels==j-1)

    # return K, labels
    return labels, km

def lenscat_load(name,
                 Rv_min, Rv_max, z_min, z_max, delta_min, delta_max, rho1_min=-1.0, rho1_max=0.0, flag=2,
                 NCHUNKS:int=1, NK:int=1, octant=False, is_MICE=False, fullshape=True):

    if is_MICE:
        RV,RA,DEC,Z,R1,R2 = 1,2,3,4,8,9
    else:
        RV,RA,DEC,Z,R1,R2 = 0,1,2,3,7,8
    ## 0:Rv, 1:ra, 2:dec, 3:z, 4:xv, 5:yv, 6:zv, 7:rho1, 8:rho2, 9:logp, 10:diff CdM y CdV, 11:flag
    ## CdM: centro de masa
    ## CdV: centro del void
    try:
        L = np.loadtxt("/home/fcaporaso/cats/"+name, dtype='f4').T
    except:
        L = np.loadtxt(name, dtype='f4').T

    if octant:
        print(' Using octant '.center(40,'#'), flush=True)
        # selecciono los void en un octante
        eps = 6.0 ## sale de tomar el angulo substendido por el void más grande al redshift más bajo
        L = L[:, (L[RA] >= 0.0+eps) & (L[RA] <= 90.0-eps) & (L[DEC]>= 0.0+eps) & (L[DEC] <= 90.0-eps)]

    mask = (L[RV] >= Rv_min) & (L[RV] < Rv_max) & (L[Z] >= z_min) & (L[Z] < z_max) & (
            L[R1] >= rho1_min) & (L[R1] < rho1_max) & (L[R2] >= delta_min) & (L[R2] < delta_max) & (L[11] >= flag)

    nvoids = mask.sum()
    if fullshape:
        L = L[:, mask]
    else:
        L = L[[RV,RA,DEC,Z]][:, mask]

    #K = get_jackknife_kmeans(L, nvoids=nvoids, NK=NK, RA=RA, DEC=DEC)

    if bool(NCHUNKS-1):
        if NCHUNKS > nvoids:
            NCHUNKS = nvoids
        lbins = round(nvoids/NCHUNKS)
        slices = (np.arange(lbins)+1)*NCHUNKS
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)
        #K = np.split(K.T, slices)

    return L, nvoids

def sourcecat_load_nback(name, nback=30.0, seed=0):
    """Get a subsample of with nback density of sources. Uses random selection of rows with seed. Inefficient for nback near 26.9. For nback bigger use directly sorucecat_load. ONLY WORKS FOR MICECAT (hardcoded)."""
    # nback :: number density of background sources [arcsec^-2]
    # octant surface = 5157.0 deg^2
    LENDATAMICE = 499609996
    n_select = int(nback*5157.0*3600.0)

    rng = np.random.default_rng(seed)
    with fits.open(name, memmap=True) as f:
        j = np.sort(rng.choice(LENDATAMICE, size=n_select, shuffle=False, replace=False))
        sample = f[1].data[j]

    return Table(sample, copy=False)

def sourcecat_load(name):
    """Can only load sourcecat as it is. For different Nback use sourcecat_load_nback"""

    S = Table.read(name, memmap=True, format='fits')
    return S

# ## Cuentas en drive 'IATE/sphere_plane_cut.pdf'
# def get_masked_data_intersection(psi, ra0, dec0, z0):
#     '''
#     objects are selected by intersecting the sphere with a plane
#     and keeping those inside the spherical cap.
#     '''

#     ra0_rad = np.deg2rad(ra0)
#     dec0_rad = np.deg2rad(dec0)
#     cos_dec0 = np.cos(dec0_rad)

#     mask_z = _S['true_redshift_gal']>z0+0.1
#     mask_field = (cos_dec0*np.cos(ra0_rad)*_S['cos_dec_gal']*_S['cos_ra_gal']
#                 + cos_dec0*np.sin(ra0_rad)*_S['cos_dec_gal']*_S['sin_ra_gal']
#                 + np.sin(dec0_rad)*_S['sin_dec_gal'] >= np.sqrt(1-np.sin(np.deg2rad(psi))**2))

#     return _S[mask_field&mask_z]
