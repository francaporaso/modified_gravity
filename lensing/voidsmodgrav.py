from argparse import ArgumentParser
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G,c,M_sun,pc
from astropy.io import fits
from astropy.table import Table
import healpy as hp
from multiprocessing import Pool
import numpy as np
import os
import time
import toml
from tqdm import tqdm

from funcs import eq2p2, lenscat_load, sourcecat_load, cov_matrix

SC_CONSTANT : float = (c.value**2.0/(4.0*np.pi*G.value))*(pc.value/M_sun.value)*1.0e-6

_RIN : float  = None
_ROUT : float = None
_N : int      = None
_NK : int     = None
_NCORES : int = None
_S : Table    = None
_PIX_TO_IDX : dict = {}
_binspace = None
_NSIDE : int = None

REDSHIFT = "true_redshift_gal"

def _init_globals(source_args, profile_args):

    global _S, _PIX_TO_IDX
    global _NCORES, _RIN, _ROUT, _N, _NK, _NSIDE
    global _binspace
    global REDSHIFT

    _RIN    = profile_args['RIN']
    _ROUT   = profile_args['ROUT']
    _N      = profile_args['N']
    _NK     = profile_args['NK']
    _NCORES = profile_args['NCORES']
    _NSIDE = profile_args['NSIDE']
    _binspace = (np.linspace if profile_args['binning']=='lin' else
                lambda start,end,n: np.logspace(np.log10(start), np.log10(end), n))
    #_binspace = {"lin": lambda s, e, n: np.linspace(s, e, n),
    #             "log": lambda s, e, n: np.logspace(np.log10(s), np.log10(e), n)}[profile_args['binning']]

    _S = sourcecat_load(**source_args)

    if "true_redshift_gal" not in _S.columns:
        REDSHIFT = "redshift_gal"

    ## making a dict of healpix idx for fast query
    upix, split_idx = np.unique(_S['pix'], return_index=True)
    split_idx = np.append(split_idx, len(_S))
    for i, pix in enumerate(upix):
        _PIX_TO_IDX[int(pix)] = np.arange(split_idx[i], split_idx[i+1])

def check_output_exists(output_file, overwrite=False):

    if os.path.exists(output_file):
        if not overwrite:
            raise FileExistsError(
                f'\n{"="*60}\n'
                f'Output file already exists: {output_file}\n'
                f'Use --overwrite flag to allow overwriting, or choose a different sample name.\n'
                f'{"="*60}'
            )
        else:
            print(f' WARNING: Will overwrite existing file: {output_file}', flush=True)
    return True

def sigma_crit(z_l, z_s):
    
    d_l  = cosmo.angular_diameter_distance(z_l).value
    d_s  = cosmo.angular_diameter_distance(z_s).value
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    return SC_CONSTANT*(d_s/(d_ls*d_l))

def _get_masked_data(psi, ra0, dec0, z0):
    '''
    objects are selected by pixel on a disc of rad=psi+pad.
    pad = 0.1*psi
    returns a copy of the data
    '''

    mask_z = _S[REDSHIFT]>z0+0.1
    pix_idx = hp.query_disc(_NSIDE, vec=hp.ang2vec(ra0, dec0, lonlat=True), radius=np.deg2rad(psi*1.1))
    mask = np.isin(_S['pix'], pix_idx, assume_unique=True)
    return _S[mask&mask_z]

def _get_masked_idx_fast(psi, ra0, dec0, z0):
    '''
    objects are selected by pixel on a disc of rad=psi+pad where pad = 0.1*psi
    uses prebuilt _PIX_TO_INDEX dict
    returns the indices of _S where to select
    '''

    pix_idx = hp.query_disc(
        _NSIDE, 
        vec=hp.ang2vec(ra0, dec0, lonlat=True), 
        radius=np.deg2rad(psi*1.1)
    )
    
    idx_source = np.concatenate([
        _PIX_TO_IDX[p] for p in pix_idx
    ])

    mask_z = _S[REDSHIFT][idx_source] > (z0+0.1)
    
    return idx_source[mask_z]

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
    
    #catdata = _get_masked_data(psi, ra0, dec0, z0)
    idx = _get_masked_idx_fast(psi, ra0, dec0, z0)
    catdata = _S[idx]
    
    sigma_c = sigma_crit(z0, catdata[REDSHIFT])/Rv0

    rads, theta = eq2p2(
        np.deg2rad(catdata['ra_gal']), np.deg2rad(catdata['dec_gal']),
        np.deg2rad(ra0), np.deg2rad(dec0)
    )

    ## Si cosmohub=[19532,19531,19260,19304], cambiarle el signo
    ## Si comoshub=[22833, 22834], el signo ya está cambiado
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
        z_mean=L[3].mean(),
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

    if (config['void']['z_min']>=0.1) and (config['void']['z_max']<=0.2):
        sourcename = config['sim'][gravity]['source'][f'for01-02']
    else:
        sourcename = config['sim'][gravity]['source'][f'for02-03']

    source_args = dict(
        name = sourcename,
    )

    profile_args = dict(
        RIN = config['prof']['RIN'],
        ROUT = config['prof']['ROUT'],
        N = config['prof']['NDOTS'],
        NK = config['NK'],
        NSIDE = 64, # No tocar! depende del source file...
        NCORES = config['NCORES'],
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
    
    output_file = (f'results/lensing_{profile_args["name"]}-{gravity}_L{lens_args["name"].split("_")[-1][:-4]}_'
                   f'Rv{lens_args["Rv_min"]:02.0f}-{lens_args["Rv_max"]:02.0f}_'
                   f'z{100*lens_args["z_min"]:03.0f}-{100*lens_args["z_max"]:03.0f}_'
                   f'type{voidtype}_bin{profile_args["binning"]}.fits')
    
    check_output_exists(output_file, overwrite=args.overwrite)

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

    # ==== Calculating profiles
    Sigma, DSigma_t, DSigma_x, extradata = stacking(source_args, lens_args, profile_args)
    # =======================

    # ==== Saving
    head=fits.Header()
    head.update({
        'nvoids':extradata['nvoids'],
        'lenscat':lens_args['name'],
        'sourcat':source_args['name'],
        'Rv_min':lens_args['Rv_min'],
        'Rv_max':lens_args['Rv_max'],
        'Rv_mean':extradata['Rv_mean'],
        'z_min':lens_args['z_min'],
        'z_max':lens_args['z_max'],
        'z_mean':extradata['z_mean'],
        'voidtype':voidtype,
        'deltamin':lens_args['delta_min'],
        'deltamax':lens_args['delta_max'],
        'RIN':profile_args['RIN'],
        'ROUT':profile_args['ROUT'],
        'N':profile_args['N'],
        'NK':profile_args['NK'],
        'binning':profile_args['binning'],
        'HISTORY':f'{time.asctime()}',
    })

    table = Table({
        'Sigma':Sigma[0],
        'DSigma_t':DSigma_t[0],
        'DSigma_x':DSigma_x[0]
    })

    cov_hdu = [
        fits.ImageHDU(cov_matrix(Sigma[1:,:]), name='cov_Sigma'),
        fits.ImageHDU(cov_matrix(DSigma_t[1:,:]), name='cov_DSigma_t'),
        fits.ImageHDU(cov_matrix(DSigma_x[1:,:]), name='cov_DSigma_x'),
    ]

    jack_hdu = [
        fits.BinTableHDU(
            Table(dict(
                [(str(n), Sigma[n,:]) for n in range(1, profile_args['NK']+1)]
            ))
        ),
        fits.BinTableHDU(
            Table(dict(
                [(str(n), DSigma_t[n,:]) for n in range(1, profile_args['NK']+1)]
            ))
        ),
        fits.BinTableHDU(
            Table(dict(
                [(str(n), DSigma_x[n,:]) for n in range(1, profile_args['NK']+1)]
            ))
        ),
    ]
 
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=head), 
        fits.BinTableHDU(table, name='profiles'),
        *cov_hdu,
        *jack_hdu
    ])
    
    hdul.writeto(output_file, overwrite=args.overwrite)
    print(f' File saved in: {output_file}', flush=True)

def main():

    parser = ArgumentParser()
    parser.add_argument('--sample', type=str, action='store', required=True)
    parser.add_argument('-c','--NCORES', type=int, default=8, action='store')
    parser.add_argument('--config', type=str, default='config.toml', action='store')
    parser.add_argument('--use08', action='store_true')
    parser.add_argument('--addnoise', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fROnly', action='store_true')
    group.add_argument('--GROnly', action='store_true')
    args = parser.parse_args()

    config = toml.load(args.config)

    if config['NCORES'] <= args.NCORES:
        config['NCORES'] = args.NCORES

    if args.GROnly:
        print(' '+f' EXECUTING -GR- ONLY '.center(60, '$')+' \n')
        execute_single_simu(config, args, 'GR')
    elif args.fROnly:
        print(' '+f' EXECUTING -f(R)- ONLY '.center(60, '$')+' \n')
        execute_single_simu(config, args, 'fR')
    else:
        for gravity in ['GR','fR']:
            print(' '+f' EXECUTING -{gravity}- '.center(60, '$')+' \n')
            execute_single_simu(config, args, gravity)

if __name__ == '__main__':

    # print('''
    # ▗▖▗▞▀▚▖▄▄▄▄   ▄▄▄ ▄ ▄▄▄▄   ▗▄▄▖
    # ▐▌▐▛▀▀▘█   █ ▀▄▄  ▄ █   █ ▐▌   
    # ▐▌▝▚▄▄▖█   █ ▄▄▄▀ █ █   █ ▐▌▝▜▌
    # ▐▙▄▄▖             █       ▝▚▄▞▘
    # '''.center(60,' '),
    # flush=True)

    print(' '+f'Start'.center(60,'#'))
    t1=time.time()
    main()
    print(' End!')
    print(f' Took {(time.time()-t1)/60.0:5.2f} min')
