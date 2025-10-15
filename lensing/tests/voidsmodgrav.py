from argparse import ArgumentParser
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G,c,M_sun,pc
from astropy.io import fits
from astropy.table import Table
import healpy as hp
from multiprocessing import Pool
import numpy as np
import time
import toml
from tqdm import tqdm

import sys
sys.path.append('../')

from funcs import eq2p2, lenscat_load, sourcecat_load, cov_matrix

SC_CONSTANT : float = (c.value**2.0/(4.0*np.pi*G.value))*(pc.value/M_sun.value)*1.0e-6

_RIN : float  = None
_ROUT : float = None
_N : int      = None
_NK : int     = None
_NCORES : int = None
_S : Table    = None
_binspace = None
_NSIDE : int = None

def _init_globals(source_args, profile_args):

    global _S, _NSIDE
    global _RIN, _ROUT, _N, _NK, _NCORES, _binspace

    _RIN    = profile_args['RIN']
    _ROUT   = profile_args['ROUT']
    _N      = profile_args['N']
    _NK     = profile_args['NK']
    _NCORES = profile_args['NCORES']
    _NSIDE = profile_args['NSIDE']
    _binspace = {"lin": lambda s, e, n: np.linspace(s, e, n),
                 "log": lambda s, e, n: np.logspace(np.log10(s), np.log10(e), n)}[profile_args['binning']]
    _S = sourcecat_load(**source_args)

def sigma_crit(z_l, z_s):
    
    d_l  = cosmo.angular_diameter_distance(z_l).value
    d_s  = cosmo.angular_diameter_distance(z_s).value
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    return SC_CONSTANT*(d_s/(d_ls*d_l))

def get_masked_data(psi, ra0, dec0, z0):
    '''
    objects are selected by pixel on a disc of rad=psi+pad.
    pad = 0.1*psi
    '''

    mask_z = _S['true_redshift_gal']>z0+0.1
    pix_idx = hp.query_disc(_NSIDE, vec=hp.ang2vec(ra0, dec0, lonlat=True), radius=np.deg2rad(psi*1.1))
    mask = np.isin(_S['pix'], pix_idx, assume_unique=True)
    return _S[mask&mask_z]

## TODO :: descargar el catalogo de nuevo... no tengo guardados los valores de redshift observado (ie con vel peculiares ie RSD)
def partial_profile(inp):

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

    for nbin in range(_N):
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
    print(' nvoids '+f'{": ":.>12}{nvoids}\n', flush=True)

    extradata = dict(
        nvoids=nvoids,
        z_mean=L[2].mean(),
        Rv_mean=L[0].mean(),
        delta_mean=L[8].mean()
    )

    _init_globals(source_args=source_args, profile_args=profile_args)

    with Pool(processes=NCORES) as pool:
        resmap = list(tqdm(pool.imap_unordered(partial_profile, L[[0,1,2,3]].T), total=nvoids))

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

    return Sigma, DSigma_t, DSigma_x, extradata

def execute_single_simu(config, args, gravity):

    lens_args = dict(
        name = config['sim'][gravity]['lens']['void08'] if args.use08 else config['sim'][gravity]['lens']['void09'] ,
        Rv_min = config['void']['Rv_min'],
        Rv_max = config['void']['Rv_max'],
        z_min = config['void']['z_min'],
        z_max = config['void']['z_max'],
        delta_min = config['void']['delta_min'], # void type
        delta_max = config['void']['delta_max'], # void type
        NK = config['NK'], 
        fullshape=True,
        NCHUNKS=1,
    )

    if config['void']['z_min']>0.1 and config['void']['z_max']<0.2:
        sourcename = config['sim'][gravity][f'z01-02']
    else:
        sourcename = config['sim'][gravity][f'z02-03']

    source_args = dict(
        name = sourcename,
    )

    profile_args = dict(
        RIN = config['prof']['RIN'],
        ROUT = config['prof']['ROUT'],
        N = config['prof']['NDOTS'],
        NK = config['NK'],
        NSIDE = 64, # No tocar! depende del source file...
        NCORES = config ['NCORES'],
        binning = config['BIN'],
        name = args.sample,
        noise = args.addnoise   
    )

    if lens_args['delta_max']<=0:
        voidtype = 'R'
    elif lens_args['delta_min']>=0:
        voidtype = 'S'
    else:
        voidtype = 'mixed'
    
    output_file = (f'results/{profile_args["name"]}-{gravity}_L{lens_args["name"].split("_")[-1][:-4]}_'
                   f'Rv{lens_args["Rv_min"]:02.0f}-{lens_args["Rv_max"]:02.0f}_'
                   f'z{100*lens_args["z_min"]:03.0f}-{100*lens_args["z_max"]:03.0f}_'
                   f'type{voidtype}_bin{profile_args["binning"]}.fits')

    # === program arguments
    print(f' {" Settings ":=^60}')
    print(' Lens cat '+f'{": ":.>10}{lens_args["name"]}')
    print(' Source cat '+f'{": ":.>8}{source_args["name"]}')
    print(' Output file '+f'{": ":.>7}{output_file}')
    print(' NCORES '+f'{": ":.>12}{profile_args["NCORES"]}\n')

    # === profile arguments
    # print(f' {" Profile arguments ":=^60}')
    print(' RMIN '+f'{": ":.>14}{profile_args["RIN"]:.2f}')
    print(' RMAX '+f'{": ":.>14}{profile_args["ROUT"]:.2f}')
    print(' N '+f'{": ":.>17}{profile_args["N"]:<2d}')
    print(' NK '+f'{": ":.>16}{profile_args["NK"]:<2d}')
    print(' Binning '+f'{": ":.>11}{profile_args["binning"]}')
    print(' Shape Noise '+f'{": ":.>7}{profile_args["noise"]}\n')
    
    # === lens arguments
    print(f' {" Void sample ":=^60}')
    print(' Radii '+f'{": ":.>13}[{lens_args["Rv_min"]:.2f}, {lens_args["Rv_max"]:.2f}) Mpc/h')
    print(' Redshift '+f'{": ":.>10}[{lens_args["z_min"]:.2f}, {lens_args["z_max"]:.2f})')
    print(' Type '+f'{": ":.>14}[{lens_args["delta_min"]},{lens_args["delta_max"]}) => {voidtype}')

    return np.nan

    # ==== Calculating profiles
    Sigma, DSigma_t, DSigma_x, extradata = stacking(source_args, lens_args, profile_args)
    cov_S = cov_matrix(Sigma[1:,:])
    cov_DSt = cov_matrix(DSigma_t[1:,:])
    cov_DSx = cov_matrix(DSigma_x[1:,:])
    # =======================

    # ==== Saving
    head=fits.Header()
    head['nvoids']=extradata['nvoids']
    head['lenscat']=lens_args['name']
    head['sourcat']=source_args['name']
    head['Rv_min']=lens_args['Rv_min']
    head['Rv_max']=lens_args['Rv_max']
    head['Rv_mean']=extradata['Rv_mean']
    head['z_min']=lens_args['z_min']
    head['z_max']=lens_args['z_max']
    head['z_mean']=extradata['z_mean']
    head['voidtype']=voidtype
    head['deltamin']=lens_args['delta_min']
    head['deltamax']=lens_args['delta_max']
    head['RIN']=profile_args['RIN']
    head['ROUT']=profile_args['ROUT']
    head['N']=profile_args['N']
    head['NK']=profile_args['NK']
    head['binning']=profile_args['binning']
    head['HISTORY'] = f'{time.asctime()}'

    table = Table({'Sigma':Sigma[0],'DSigma_t':DSigma_t[0],'DSigma_x':DSigma_x[0]})
    cov_hdu = [
        fits.ImageHDU(cov_S, name='cov_Sigma'),
        fits.ImageHDU(cov_DSt, name='cov_DSigma_t'),
        fits.ImageHDU(cov_DSx, name='cov_DSigma_x'),
    ]

    primary_hdu = fits.PrimaryHDU(header=head)
    table_hdu = fits.BinTableHDU(table, name='profiles') 
 
    hdul = fits.HDUList([primary_hdu, table_hdu] + cov_hdu)
    hdul.writeto(output_file, overwrite=False)
    
    print(f' File saved in: {output_file}', flush=True)

def main():

    parser = ArgumentParser()
    parser.add_argument('--sample', type=str, default='TEST', action='store')
    parser.add_argument('-c','--NCORES', type=int, default=8, action='store')
    parser.add_argument('--config', type=str, default='config.toml', action='store')
    parser.add_argument('--use08', action='store_true')
    parser.add_argument('--addnoise', action='store_true')
    args = parser.parse_args()

    config = toml.load(args.config)

    for gravity in ['GR','fR']:
        print(' '+f' EXECUTING {gravity} '.center(60, '$')+' \n')
        execute_single_simu(config, args, gravity)

if __name__ == '__main__':

    print('''
    ▗▖▗▞▀▚▖▄▄▄▄   ▄▄▄ ▄ ▄▄▄▄   ▗▄▄▖
    ▐▌▐▛▀▀▘█   █ ▀▄▄  ▄ █   █ ▐▌   
    ▐▌▝▚▄▄▖█   █ ▄▄▄▀ █ █   █ ▐▌▝▜▌
    ▐▙▄▄▖             █       ▝▚▄▞▘
    '''.center(60,' '),
    flush=True)
    t1=time.time()
    main()
    print('End!')
    print(f'took {(time.time()-t1)/60.0:5.2f} min')
