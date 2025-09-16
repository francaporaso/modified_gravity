import numpy as np #also imported in funcs...
from astropy.cosmology import LambdaCDM
from astropy.constants import G,c,M_sun,pc
from astropy.table import Table
from multiprocessing import Pool
from tqdm import tqdm

from funcs import eq2p2, lenscat_load, sourcecat_load

_cosmo = None
_S = None
_bines = None

def init_worker(source_args, profile_args, cosmo_params):

    global _cosmo, _S, _bines # only declare global when intending to modify them

    _cosmo = LambdaCDM(**cosmo_params)
    _S = sourcecat_load(**source_args) 
    ra_gal_rad  = np.deg2rad(_S['ra_gal'])
    dec_gal_rad = np.deg2rad(_S['dec_gal'])
    _S['cos_ra_gal']  = np.cos(ra_gal_rad)
    _S['sin_ra_gal']  = np.sin(ra_gal_rad)
    _S['cos_dec_gal'] = np.cos(dec_gal_rad)
    _S['sin_dec_gal'] = np.sin(dec_gal_rad)

    if profile_args['binning'] == 'log':
        _bines = np.logspace(profile_args['RIN'], profile_args['ROUT'], profile_args['N']+1)
    else:
        _bines = np.linspace(profile_args['RIN'], profile_args['ROUT'], profile_args['N']+1)

def sigma_crit(z_l, z_s):
    d_l  = _cosmo.angular_diameter_distance(z_l).value*pc.value*1.0e6
    d_s  = _cosmo.angular_diameter_distance(z_s).value
    d_ls = _cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    return (((c.value**2.0)/(4.0*np.pi*G.value*d_l))*(d_s/d_ls))*(pc.value**2/M_sun.value)


## Cuentas en drive 'IATE/sphere_plane_cut.pdf'
def get_masked_data(psi, ra0, dec0, z0):
    '''
    objects are selected by intersecting a sphere and a plane
    and keeping those inside the spherical cap.
    '''
    ra0_rad = np.deg2rad(ra0)
    dec0_rad = np.deg2rad(dec0)
    cos_dec0 = np.cos(dec0_rad)

    local_S = _S[(_S['true_redshift_gal']>z0+0.1)]
    mask_field = (cos_dec0*np.cos(ra0_rad)*local_S['cos_dec_gal']*local_S['cos_ra_gal']
                + cos_dec0*np.sin(ra0_rad)*local_S['cos_dec_gal']*local_S['sin_ra_gal'] 
                + np.sin(dec0_rad)*local_S['sin_dec_gal'] >= np.sqrt(1-np.sin(np.deg2rad(psi))**2))
    
    return local_S[mask_field]

## TODO :: descargar el catalogo de nuevo... no tengo guardados los valores de redshift observado (ie con vel peculiares ie RSD)
def partial_profile(inp):    
    assert len(inp) == 4

    Sigma_wsum    = np.zeros(N)
    DSigma_t_wsum = np.zeros(N)
    DSigma_x_wsum = np.zeros(N)
    N_inbin       = np.zeros(N)
    
    ra0, dec0, z0, Rv0 = inp

    # for ni in range(N):
    # adentro del for, mask depende de n... solo quiero las gx en un anillo

    DEGxMPC = _cosmo.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
    psi = DEGxMPC*ROUT*Rv0
    
    catdata = get_masked_data(psi, ra0, dec0, z0)
    sigma_c = sigma_crit(z0, catdata['true_redshift_gal'])/Rv0

    rads, theta = eq2p2(
        np.deg2rad(catdata['ra_gal']), np.deg2rad(catdata['dec_gal']),
        np.deg2rad(ra0), np.deg2rad(dec0)
    )

    ## TODO :: al descargar, cambiarle el signo
    e1 = -catdata['gamma1']
    e2 = -catdata['gamma2']

    #get tangential ellipticities 
    cos2t = np.cos(2.0*theta)
    sin2t = np.sin(2.0*theta)
    et = -(e1*cos2t+e2*sin2t)*sigma_c
    ex = (-e1*sin2t+e2*cos2t)*sigma_c
        
    #get convergence
    k  = catdata['kappa']*sigma_c

    # r = (np.rad2deg(rads)/DEGxMPC)/Rv0
    # bines = linspace() or logspace()
    dig = np.digitize((np.rad2deg(rads)/DEGxMPC)/Rv0, _bines)

    for nbin in range(N):
        mbin = dig == nbin+1              
        Sigma_wsum[nbin]    = k[mbin].sum()
        DSigma_t_wsum[nbin] = et[mbin].sum()
        DSigma_x_wsum[nbin] = ex[mbin].sum()
        N_inbin[nbin]       = np.count_nonzero(mbin) ## idem mbin.sum(), faster
    
    return Sigma_wsum, DSigma_t_wsum, DSigma_x_wsum, N_inbin

def stacking(lens_args, source_args, profile_args, cosmo_params):
    
    N = profile_args['N']
    Nk = profile_args['Nk']
    ncores = profile_args['ncores']

    N_inbin = np.zeros((Nk+1, N))
    Sigma_wsum = np.zeros((Nk+1, N))
    DSigma_t_wsum = np.zeros((Nk+1, N))
    DSigma_x_wsum = np.zeros((Nk+1, N))

    L, K, nvoids = lenscat_load(**lens_args)
    print(f'Nvoids: {nvoids}', flush=True)

    # for i, Li in enumerate(tqdm(L)):
    #     num = len(Li)
    #     inp = np.array([Li.T[1], Li.T[2], Li.T[3], Li.T[0]]).T
    #     with Pool(processes=num) as pool:
    #         resmap = np.array(pool.map(vlen.partial_profile, inp))
    #         pool.close()
    #         pool.join()

    with Pool(processes=ncores, initializer=init_worker, 
              initargs=(source_args, profile_args, cosmo_params)) as pool:
        
        resmap = np.array(pool.map(partial_profile, L.T))
        pool.close()
        pool.join()

    for j,r in enumerate(resmap):
        km = np.tile(K[j], (N,1)).T
        N_inbin += np.tile(r[-1], (Nk+1,1))*km
        Sigma_wsum += np.tile(r[0], (Nk+1,1))*km
        DSigma_t_wsum += np.tile(r[1], (Nk+1,1))*km
        DSigma_x_wsum += np.tile(r[2], (Nk+1,1))*km

    Sigma = Sigma_wsum/N_inbin
    DSigma_t = DSigma_t_wsum/N_inbin
    DSigma_x = DSigma_x_wsum/N_inbin

    return Sigma, DSigma_t, DSigma_x 


if __name__ == '__main__':

    import time

    lens_name = 'voids_LCDM_09.dat'
    Rv_min = 10.0
    Rv_max = 11.0
    z_min = 0.2
    z_max = 0.22
    delta_min = -1.0 # void type
    delta_max = -0.1 # void type

    source_name = 'l768_gr_z04-07_for02-03_19304.fits'

    RIN = 0.1
    ROUT = 1.0
    N = 10
    Nk = 10
    ncores = 8

    lens_args = dict(
        name = lens_name,
        Rv_min = Rv_min,
        Rv_max = Rv_max,
        z_min = z_min,
        z_max = z_max,
        delta_min = delta_min, # void type
        delta_max = delta_max, # void type
        ncores = 1,
        Nk = Nk,
        fullshape=False,
    )

    source_args = dict(
        name = source_name
    )

    profile_args = dict(
        RIN = RIN,
        ROUT = ROUT,
        N = N,
        Nk = Nk,
        ncores = ncores,
        binning = 'lin'
    )

    cosmo_params = dict(
        Om0 = 0.3089,
        Ode0 = 0.6911,
        H0 = 100.0
    )
    print('Start!')
    t1=time.time()
    stacking(lens_args, source_args, profile_args, cosmo_params)
    print('End!')
    print(f'took {(time.time()-t1)/60.0} s')