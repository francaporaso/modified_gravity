from argparse import ArgumentParser
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G,c,M_sun,pc
from astropy.table import Table
import healpy as hp
from multiprocessing import Pool
import numpy as np
import time
from tqdm import tqdm

from funcs import eq2p2, lenscat_load, sourcecat_load

SC_CONSTANT : float = (c.value**2.0/(4.0*np.pi*G.value))*(pc.value/M_sun.value)*1.0e-6

_RIN : float    = None
_ROUT : float   = None
_N : int        = None
_NK : int       = None
_NCORES : int   = None
_S : Table      = None
_binspace = None

_NSIDE : int = None

def init_worker(source_args, profile_args):

    global _S, _NSIDE
    global _RIN, _ROUT, _N, _NK, _NCORES, _binspace

    _RIN    = profile_args['RIN']
    _ROUT   = profile_args['ROUT']
    _N      = profile_args['N']
    _NK     = profile_args['NK']
    _NCORES = profile_args['NCORES']
    _NSIDE = profile_args['NSIDE']
    _binspace = np.linspace if profile_args['binning']=='lin' else np.logspace
    _S = sourcecat_load(**source_args)
    #print(f'worker initialized: {type(_S)}', flush=True)

def sigma_crit(z_l, z_s):
    
    d_l  = cosmo.angular_diameter_distance(z_l).value
    d_s  = cosmo.angular_diameter_distance(z_s).value
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    return SC_CONSTANT*(d_s/(d_ls*d_l))

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

def get_masked_data(psi, ra0, dec0, z0):
    '''
    objects are selected by pixel on a disc of rad=psi+pad.
    '''

    mask_z = _S['true_redshift_gal']>z0+0.1
    pix_idx = hp.query_disc(_NSIDE, vec=hp.ang2vec(ra0, dec0, lonlat=True), radius=np.deg2rad(psi+0.5))
    mask_field = np.isin(_S['pix'], pix_idx)

    return _S[mask_field&mask_z]

## TODO :: descargar el catalogo de nuevo... no tengo guardados los valores de redshift observado (ie con vel peculiares ie RSD)
def partial_profile(inp):    
    
    #print('partial init', flush=True)

    Sigma_wsum    = np.zeros(_N)
    DSigma_t_wsum = np.zeros(_N)
    DSigma_x_wsum = np.zeros(_N)
    N_inbin       = np.zeros(_N)
    
    Rv0, ra0, dec0, z0 = inp
    # for ni in range(N):
    # adentro del for, mask depende de n... solo quiero las gx en un anillo

    DEGxMPC = cosmo.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
    psi = DEGxMPC*_ROUT*Rv0
    
    catdata = get_masked_data(psi, ra0, dec0, z0)
    #print(catdata.info, flush=True)

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
    bines = _binspace(_RIN, _ROUT, _N+1)
    dig = np.digitize((np.rad2deg(rads)/DEGxMPC)/Rv0, bines)

    for nbin in range(N):
        mbin = dig == nbin+1              
        Sigma_wsum[nbin]    = k[mbin].sum()
        DSigma_t_wsum[nbin] = et[mbin].sum()
        DSigma_x_wsum[nbin] = ex[mbin].sum()
        N_inbin[nbin]       = np.count_nonzero(mbin) ## idem mbin.sum(), faster
    
    return Sigma_wsum, DSigma_t_wsum, DSigma_x_wsum, N_inbin

def stacking(source_args, lens_args, profile_args):

    N = profile_args['N']
    NK = profile_args['NK']
    NCORES = profile_args['NCORES']

    N_inbin       = np.zeros((NK+1, N))
    Sigma_wsum    = np.zeros((NK+1, N))
    DSigma_t_wsum = np.zeros((NK+1, N))
    DSigma_x_wsum = np.zeros((NK+1, N))

    L, K, nvoids = lenscat_load(**lens_args)
    K = K[:, :nvoids] # me quedo con los que voy a usar
    print(f'Nvoids: {nvoids}', flush=True)

    print('Starting pool...', flush=True)
    with Pool(processes=NCORES, initializer=init_worker, 
              initargs=(source_args, profile_args)) as pool:

        resmap = list(tqdm(pool.imap_unordered(partial_profile, L.T), total=nvoids))
        pool.close()
        pool.join()

    print('Pool ended, stacking...', flush=True)
    for j,r in enumerate(np.array(resmap)):
        km = np.tile(K[:,j], (N,1)).T
        N_inbin += np.tile(r[-1], (NK+1,1))*km
        Sigma_wsum += np.tile(r[0], (NK+1,1))*km
        DSigma_t_wsum += np.tile(r[1], (NK+1,1))*km
        DSigma_x_wsum += np.tile(r[2], (NK+1,1))*km

    Sigma = Sigma_wsum/N_inbin
    DSigma_t = DSigma_t_wsum/N_inbin
    DSigma_x = DSigma_x_wsum/N_inbin

    return Sigma, DSigma_t, DSigma_x 

def main():

    parser = ArgumentParser()
    parser.add_argument('--lens_name', type=str, default='voids_LCDM_09.dat', action='store')
    parser.add_argument('--source_name', type=str, default='l768_gr_z04-07_for02-03_w-pix_19304.fits', action='store')
    parser.add_argument('--sample', type=str, default='TEST_LCDM_', action='store')
    parser.add_argument('-c','--NCORES', type=int, default=2, action='store')
    parser.add_argument('--Rv_min', type=float, default=1.0, action='store')
    parser.add_argument('--Rv_max', type=float, default=50.0, action='store')
    parser.add_argument('--z_min', type=float, default=0.0, action='store')
    parser.add_argument('--z_max', type=float, default=0.6, action='store')
    parser.add_argument('--delta_min', type=float, default=-1.0, action='store')
    parser.add_argument('--delta_max', type=float, default=100.0, action='store')
    parser.add_argument('--flag', type=float, default=2.0, action='store')
    parser.add_argument('--RIN', type=float, default=0.05, action='store')
    parser.add_argument('--ROUT', type=float, default=5.0, action='store')    
    parser.add_argument('-N','--ndots', type=int, default=22, action='store')    
    parser.add_argument('-K','--NK', type=int, default=100, action='store')    
    parser.add_argument('--addnoise', action='store_true')
    parser.add_argument('--binning', type='str', default='lin', action='store', choices=['lin', 'log'])
    args = parser.parse_args()

    lens_args = dict(
        name = args.lens_name,
        Rv_min = args.Rv_min,
        Rv_max = args.Rv_max,
        z_min = args.z_min,
        z_max = args.z_max,
        delta_min = args.delta_min, # void type
        delta_max = args.delta_max, # void type
        NK = args.NK,
        fullshape=False,
    )

    source_args = dict(
        name = args.source_name,
    )

    profile_args = dict(
        RIN = args.RIN,
        ROUT = args.ROUT,
        N = args.N,
        NK = args.NK,
        NSIDE = 64,
        NCORES = args.NCORES,
        binning = args.binning,
        name = args.sample,
        noise = args.addnoise
    )

    if lens_args['delta_max']<=0:
        voidtype = 'R'
    elif lens_args['delta_min']>=0:
        voidtype = 'S'
    else:
        voidtype = 'all'

    # program arguments
    print(f' {" Program arguments ":=^60}')
    print(' Lens cat '+f'{": ":.>10}{lens_args["name"]}')
    print(' Source cat '+f'{": ":.>8}{source_args["name"]}')
    print(' Output file '+f'{": ":.>7}{profile_args["name"]}')
    print(' NCORES '+f'{": ":.>12}{profile_args["NCORES"]}\n')
    
    # lens arguments
    print(f' {" Void sample ":=^60}')
    print(' Radii '+f'{": ":.>13}[{lens_args["Rv_min"]:.2f}, {lens_args["Rv_max"]:.2f}) Mpc/h')
    print(' Redshift '+f'{": ":.>10}[{lens_args["z_min"]:.2f}, {lens_args["z_max"]:.2f})')
    print(' Type '+f'{": ":.>14}[{lens_args["delta_min"]},{lens_args["delta_max"]}) => {voidtype}\n')
    # print('Octante: '.ljust(15,'.'), f' {args.octant}'.rjust(15,'.'),sep='')
    #print('N voids: '.ljust(15,'.'), f' {nvoids}'.rjust(15,'.'),sep='')

    # profile arguments
    print(f'{" Profile arguments ":=^60}')
    print(' RMIN '+f'{": ":.>14}{profile_args["RIN"]:.2f}')
    print(' RMAX '+f'{": ":.>14}{profile_args["ROUT"]:.2f}')
    print(' N '+f'{": ":.>17}{profile_args["N"]:<2d}')
    print(' NK '+f'{": ":.>16}{profile_args["NK"]:<2d}')
    print(' Shape Noise '+f'{": ":.>7}{profile_args["noise"]}\n')

    res = Table(dict(zip(('Sigma','DSigma_t','DSigma_x'),stacking(source_args, lens_args, profile_args))))
    res.write('test.fits', format='fits', overwrite=True)
    print('Saved in "test.fits"', flush=True)

if __name__ == '__main__':

    lens_name = 'voids_LCDM_09.dat'
    Rv_min = 10.0
    Rv_max = 11.0
    z_min = 0.2
    z_max = 0.22
    delta_min = -1.0 # void type
    delta_max = -0.2 # void type

    source_name = 'l768_gr_z04-07_for02-03_w-pix_19304.fits'
    NSIDE = 64

    RIN = 0.1
    ROUT = 1.0
    N = 10
    NK = 25 ## Debe ser siempre un cuadrado!
    NCORES = 8

    lens_args = dict(
        name = lens_name,
        Rv_min = Rv_min,
        Rv_max = Rv_max,
        z_min = z_min,
        z_max = z_max,
        delta_min = delta_min, # void type
        delta_max = delta_max, # void type
        NCHUNKS = 1,
        NK = NK,
        fullshape=False,
    )

    source_args = dict(
        name = source_name,
    )

    profile_args = dict(
        RIN = RIN,
        ROUT = ROUT,
        N = N,
        NK = NK,
        NSIDE = NSIDE,
        NCORES = NCORES,
        binning = 'lin'
    )

    # cosmo_params = dict(
    #     Om0 = 0.3089,
    #     Ode0 = 0.6911,
    #     H0 = 100.0
    # )

    print('Start!')
    t1=time.time()
    main()
    print('End!')
    print(f'took {(time.time()-t1)/60.0} min')