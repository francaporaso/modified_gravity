"""Standalone helpers: covariances for analysis, validation tools and the legacy void loader.

The scripts of density/ and statistics/ import this file as a top-level module
(`from funcs import ...`, with lensing/ in sys.path), so it must not import from the
`lensing` package.
"""
import numpy as np

def cov_matrix(array):
    """Jackknife covariance of K samples (rows of `array`)"""

    K = len(array)
    dif = array - np.average(array, axis=0)
    return (K-1)/K * (dif.T @ dif)

def normalize_cov(cov):
    """Correlation matrix of a covariance matrix"""
    sigma = np.sqrt(np.diag(cov))
    return cov / np.outer(sigma, sigma)

## validation: the regions of a rectangular footprint, independent of kmeans_radec
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

    if bool(NCHUNKS-1):
        if NCHUNKS > nvoids:
            NCHUNKS = nvoids
        lbins = round(nvoids/NCHUNKS)
        slices = (np.arange(lbins)+1)*NCHUNKS
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)

    return L, nvoids

# ## Cuentas en drive 'IATE/sphere_plane_cut.pdf'
# ## validation: source selection by intersecting the sphere with a plane, to compare
# ## with the pixel queries of stacking.query_sources
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
