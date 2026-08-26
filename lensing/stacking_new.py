from argparse import ArgumentParser
#from astropy.cosmology import WMAP5 as cosmo ## MICE uses WMAP5
from astropy.cosmology import FlatLambdaCDM ## MICE uses WMAP5
from astropy.constants import G,c,M_sun,pc
from astropy.io import fits
from astropy.table import Table
import healpy as hp
from multiprocessing import get_context
import numpy as np
import os
from time import time, asctime
#import toml
from tqdm import tqdm
from itertools import product

from lensing.funcs import eq2p2, sourcecat_load, cov_matrix, get_jackknife_kmeans
from lensing.io import read_lens_catalog
from lensing.settings import Config

ctx = get_context('fork')

# --- Fixed globals
cfg : Config
#SOURCE = None
PIX_TO_IDX : dict = {}
SC_CONSTANT : float = (c.value**2.0/(4.0*np.pi*G.value))*(pc.value/M_sun.value)*1e-6

# "z_cgal" : true-z
# "z_cgal_v" : spec-z
# "z_desdm_mc" : photo-z
# REDSHIFT = "z_cgal_v" # name of the redshift column in the source file
# -----------------

def init_globals():

    global SOURCE
    global binspace
    global cosmo

    #set cosmology
    cosmo = FlatLambdaCDM(H0=100.0*cfg.h, Om0=cfg.Om0, Ob0=cfg.Ob0)

    # set binning
    binspace = ( np.linspace if cfg.binning=='lin' else np.geomspace )

    # read cat
    SOURCE = sourcecat_load(cfg.sourcename)

def make_pix2idx_dict(source):
    global PIX_TO_IDX
    # making a dict of healpix idx for fast query
    upix, split_idx = np.unique(source[cfg.scols['pix']], return_index=True)
    split_idx = np.append(split_idx, len(source))
    for i, pix in enumerate(upix):
        PIX_TO_IDX[int(pix)] = np.arange(split_idx[i], split_idx[i+1])

def check_output_exists(output_file, overwrite=False):

    if os.path.exists(output_file):
        if not overwrite:
            raise OSError(
                f'\n{"="*60}\n'
                f'Output file already exists: {output_file}\n'
                f'Use --overwrite flag to allow overwriting, or choose a different sample name.\n'
                f'{"="*60}'
            )
        else:
            print(f' WARNING: Will overwrite existing file: {output_file}', flush=True)
    return True

## WARNING :
## cosmo.distance is in physical units (ie Mpc)
## need to divide out by littleh to get Mpc/h
## important only if h!=1
def sigma_crit(z_l, z_s):
    d_l  = cosmo.angular_diameter_distance(z_l).value
    d_s  = cosmo.angular_diameter_distance(z_s).value
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    return SC_CONSTANT*(d_s/(d_ls*d_l))

def get_masked_idx_fast(psi, ra0, dec0, z0):
    '''
    objects are selected by pixel on a disc of rad=psi+pad where pad = 0.1*psi
    uses prebuilt _PIX_TO_INDEX dict
    returns the indices of _S where to select
    '''

    pix_idx = hp.query_disc(
        cfg.NSIDE,
        vec=hp.ang2vec(ra0, dec0, lonlat=True),
        radius=np.deg2rad(psi*1.1)
    )

    idx_arrays = np.concatenate([
        PIX_TO_IDX[p]
        for p in pix_idx
        if p in PIX_TO_IDX
    ])

    mask_z = SOURCE[cfg.scols['redshift']][idx_arrays] > (z0+0.1)

    return idx_arrays[mask_z]

## distance: is needed a cosmo.h dividing sigma_c when h!=1, else is not needed.
## leaving it for general case...
def partial_profile(inp):

    Sigma_wsum    = np.zeros(cfg.NBINS)
    DSigma_t_wsum = np.zeros(cfg.NBINS)
    DSigma_x_wsum = np.zeros(cfg.NBINS)
    N_inbin       = np.zeros(cfg.NBINS)

    Rv0, ra0, dec0, z0 = inp
    # for ni in range(N):
    # adentro del for, mask depende de n... solo quiero las gx en un anillo

    DEGxMPC = cosmo.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
    psi = DEGxMPC * cfg.ROUT * Rv0

    idx = get_masked_idx_fast(psi, ra0, dec0, z0)
    catdata = SOURCE[idx]

    #sigma_c = sigma_crit(z0, catdata[REDSHIFT])/Rv0
    ## dividing by cosmo.h gives the correct units! (Rv0 in Mpc/h but sigma_crit in physical Msun*pc^-2)
    ## factor of almost 1.5 difference!
    sigma_c = sigma_crit(z0, catdata[cfg.scols['redshift']]) / (Rv0*cosmo.h)

    rads, theta = eq2p2(
        np.deg2rad(catdata[cfg.scols['ra']]), np.deg2rad(catdata[cfg.scols['dec']]),
        np.deg2rad(ra0), np.deg2rad(dec0)
    )

    e1 = catdata[cfg.scols['gamma1']]
    e2 = -catdata[cfg.scols['gamma2']]
    if cfg.addnoise:
        e1-=catdata[cfg.scols['eps1']]
        e2+=catdata[cfg.scols['eps2']]

    #get tangential ellipticities
    cos2t = np.cos(2.0*theta)
    sin2t = np.sin(2.0*theta)
    et = -(e1*cos2t+e2*sin2t) * sigma_c
    ex = (-e1*sin2t+e2*cos2t) * sigma_c

    #get convergence
    k  = catdata[cfg.scols['kappa']] * sigma_c

    bines = binspace(cfg.RIN, cfg.ROUT, cfg.NBINS+1)
    dig = np.digitize((np.rad2deg(rads)/DEGxMPC)/Rv0, bines)

    for nbin in range(cfg.NBINS):
        mbin = dig == nbin+1
        Sigma_wsum[nbin]    = k[mbin].sum()
        DSigma_t_wsum[nbin] = et[mbin].sum()
        DSigma_x_wsum[nbin] = ex[mbin].sum()
        N_inbin[nbin]       = np.count_nonzero(mbin) ## idem mbin.sum(), faster

    return Sigma_wsum, DSigma_t_wsum, DSigma_x_wsum, N_inbin

def stacking(rv_min, rv_max, z_min, z_max, delta_min, delta_max, gravity):
    
    lenses = read_lens_catalog(
        filename = cfg.lensname,
        cat='sparkling',
        Rv_min = rv_min, Rv_max = rv_max,
        z_min = z_min, z_max = z_max,
        delta_min = delta_min, delta_max = delta_max,
        flag = cfg.flag,
        has_id = False,
        fullshape = cfg.fullshape
    )
    lensrand = read_lens_catalog(
        filename = cfg.randsname,
        cat='sparkling',
        Rv_min = rv_min, Rv_max = rv_max,
        z_min = z_min, z_max = z_max,
        delta_min = delta_min, delta_max = delta_max,
        flag = cfg.flag,
        has_id = False,
        fullshape = cfg.fullshape
    )

    nvoids = len(lenses.T)
    nrands = len(lensrand.T)

    if delta_max<=0:
        voidtype = 'R'
    elif delta_min>=0:
        voidtype = 'S'
    else:
        voidtype = 'mixed'

    output_filename = (f'results/lensing_{gravity}_{cfg.sample}_N{cfg.NBINS}_'
                   f'Rv{rv_min:02.0f}-{rv_max:02.0f}_'
                   f'z{100*z_min:03.0f}-{100*z_max:03.0f}_'
                   f'type{voidtype}_bin{cfg.binning}')
    if cfg.addnoise:
        output_filename += 'w-noise'
    output_filename += '.fits'

    assert check_output_exists(output_filename, overwrite=cfg.overwrite)

    # === program arguments
    print(f' {" Settings ":=^60}')
    print(' Lens cat '+f'{": ":.>10}{cfg.lensname}')
    print(' Source cat '+f'{": ":.>8}{cfg.sourcename}')
    print(' NCORES '+f'{": ":.>12}{cfg.ncores}\n')

    # === profile arguments
    print(' RMIN '+f'{": ":.>14}{cfg.RIN:.2f}')
    print(' RMAX '+f'{": ":.>14}{cfg.ROUT:.2f}')
    print(' NBINS '+f'{": ":.>13}{cfg.NBINS:<2d}')
    print(' NJK '+f'{": ":.>15}{cfg.NJK:<2d}')
    print(' Binning '+f'{": ":.>11}{cfg.binning}')
    print(' Shape Noise '+f'{": ":.>7}{cfg.addnoise}\n')

    # === lens arguments
    print(f' {" Void sample ":=^60}')
    print(' Radii '+f'{": ":.>13}[{rv_min:.2f}, {rv_max:.2f}) Mpc/h')
    print(' Redshift '+f'{": ":.>10}[{z_min:.2f}, {z_max:.2f})')
    print(' Type '+f'{": ":.>14}[{delta_min},{delta_max}) => {voidtype}')
    print(' # of voids '+f'{": ":.>12}{nvoids}\n', flush=True)

    N_inbin       = np.zeros((cfg.NJK+1, cfg.NBINS))
    Sigma_wsum    = np.zeros((cfg.NJK+1, cfg.NBINS))
    DSigma_t_wsum = np.zeros((cfg.NJK+1, cfg.NBINS))
    DSigma_x_wsum = np.zeros((cfg.NJK+1, cfg.NBINS))
    
    Sigma_wsum_rand = np.zeros((cfg.NJK+1, cfg.NBINS))
    N_inbin_rand = np.zeros((cfg.NJK+1, cfg.NBINS))
    # calculating voids 
    with ctx.Pool(processes=cfg.ncores) as pool:
        resmap = list(
            tqdm(
                pool.imap(
                    partial_profile, 
                    lenses[[
                        cfg.lcols['rv'],
                        cfg.lcols['ra'],
                        cfg.lcols['dec'],
                        cfg.lcols['z']
                    ]].T
                ), 
                total=nvoids
            )
        )

        # calculating randoms
        print('\n >> Calculating profiles for random voids...', flush=True)
        randmap = list(
            tqdm(
                pool.imap(
                    partial_profile,
                    lensrand[[
                        cfg.lcols['rv'],
                        cfg.lcols['ra'],
                        cfg.lcols['dec'],
                        cfg.lcols['z']
                    ]].T
                ),
                total=nrands
            )
        )
    
    print('\n >> Pool ended, stacking...', flush=True)
 
    # -- reducing...
    kappa, gamma_t, gamma_x, nbin = map(
        lambda x: np.vstack(x),
        zip(*resmap)
    )
    
    kappa_rand, _, _, nbin_rand = map(
        lambda x: np.vstack(x),
        zip(*randmap)
    )

    N_inbin[0] = nbin.sum(axis=0)
    Sigma_wsum[0] = kappa.sum(axis=0)
    DSigma_t_wsum[0] = gamma_t.sum(axis=0)
    DSigma_x_wsum[0] = gamma_x.sum(axis=0)

    N_inbin_rand[0] = nbin_rand.sum(axis=0)
    Sigma_wsum_rand[0] = kappa_rand.sum(axis=0)
    
    # calculate jackknife regions and profiles
    jidx = np.arange(1, len(SOURCE)-1, len(SOURCE)//10_000, dtype=int)
    kidx, km = get_jackknife_kmeans(
        ra_sample=SOURCE[cfg.scols['ra']][jidx], 
        dec_sample=SOURCE[cfg.scols['dec']][jidx], 
        ra_cl=lenses[cfg.lcols['ra']],
        dec_cl=lenses[cfg.lcols['dec']],
        nlenses=nvoids, 
        NJK=cfg.NJK
    )
    kidx_rand = km.find_nearest(np.column_stack([lensrand[cfg.lcols['ra']], lensrand[cfg.lcols['dec']]]))
    
    for j, k in enumerate(range(cfg.NJK)):
        mask = (kidx!=k)
        mask_rand = (kidx_rand!=k)

        N_inbin[j+1,:] = nbin[mask].sum(axis=0)
        Sigma_wsum[j+1,:] = kappa[mask].sum(axis=0)
        DSigma_t_wsum[j+1,:] = gamma_t[mask].sum(axis=0)
        DSigma_x_wsum[j+1,:] = gamma_x[mask].sum(axis=0)

        N_inbin_rand[j+1,:] = nbin_rand[mask_rand].sum(axis=0)
        Sigma_wsum_rand[j+1,:] = kappa_rand[mask_rand].sum(axis=0)

    Sigma = Sigma_wsum/N_inbin
    DSigma_t = DSigma_t_wsum/N_inbin
    DSigma_x = DSigma_x_wsum/N_inbin

    Sigma_rand = Sigma_wsum_rand/N_inbin_rand

    extradata = dict(
        nvoids=nvoids,
        z_mean=lenses[cfg.lcols['z']].mean(),
        z_median=lenses[cfg.lcols['z']].median(),
        Rv_mean=lenses[cfg.lcols['rv']].mean(),
        Rv_median=lenses[cfg.lcols['rv']].median(),
        delta_mean=lenses[cfg.lcols['delta']].mean(),
        delta_median=lenses[cfg.lcols['delta']].mean(),
    )

    # return Sigma, DSigma_t, DSigma_x, extradata
    # =======================

    # ==== Saving
    head=fits.Header()
    head.update({
        'nvoids':extradata['nvoids'],
        'lenscat':cfg.lensname,
        'sourcat':cfg.sourcename,
        'Rv_min':rv_min,
        'Rv_max':rv_max,
        'Rv_mean':extradata['Rv_mean'],
        'Rv_med':extradata['Rv_median'],
        'z_min':z_min,
        'z_max':z_max,
        'z_mean':extradata['z_mean'],
        'z_med':extradata['z_median'],
        'voidtype':voidtype,
        'deltamin':delta_min,
        'deltamax':delta_max,
        'deltamea':extradata['delta_mean'],
        'deltamed':extradata['delta_median'],
        'RIN':cfg.RIN,
        'ROUT':cfg.ROUT,
        'NBINS':cfg.NBINS,
        'NJK':cfg.NJK,
        'binning':cfg.binning,
        'HISTORY':f'{asctime()}',
    })

    table = Table({
        'R':binspace(cfg.RIN, cfg.ROUT, cfg.NBINS),
        'Sigma':Sigma[0],
        'DSigma_t':DSigma_t[0],
        'DSigma_x':DSigma_x[0],
        'Sigma_rand':Sigma_rand[0]
    })

    # sigma cov_matrix
    covS = cov_matrix(Sigma[1:,:]-Sigma_rand[1:,:])
    cov_hdu = [
        fits.ImageHDU(covS, name='cov_Sigma'),
        fits.ImageHDU(cov_matrix(DSigma_t[1:,:]), name='cov_DSigma_t'),
        fits.ImageHDU(cov_matrix(DSigma_x[1:,:]), name='cov_DSigma_x'),
        fits.ImageHDU( cov_matrix(Sigma_rand[1:,:]), name='cov_Sigma_rand'),
    ]

    jack_hdu = [
        fits.ImageHDU(Sigma[1:, :], name='jack_Sigma'),
        fits.ImageHDU(DSigma_t[1:, :], name='jack_DSigma_t'),
        fits.ImageHDU(DSigma_x[1:, :], name='jack_DSigma_x'),
        fits.ImageHDU(Sigma_rand[1:, :], name='jack_Sigma_rand'),
    ]

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=head),
        fits.BinTableHDU(table, name='profiles'),
        *cov_hdu,
        *jack_hdu
    ])

    hdul.writeto(output_filename, overwrite=cfg.overwrite)
    print(f' >> File saved in: {output_filename}.\n', flush=True)
    
    return 0


def main():
    global cfg

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='lensing/config.toml', action='store')
    parser.add_argument('--ncores', type=int, action='store')
    parser.add_argument('--gravity', type=str, action='store', nargs='+', required=True)
    parser.add_argument('--use_threshold', type=str, action='store', choices=['08', '09'], default='09')
    args = parser.parse_args()

    print(' Start '.center(15, '='))
    tini = time()
    
    i = 0
    total = 0
    for grav in args.gravity:

        cfg = Config(args.config, grav, args.use_threshold)
        if args.ncores is not None:
            cfg.set_ncores(args.ncores)
        
        if i==0:
            total = len(cfg.zbins)*len(cfg.rvbins)*len(cfg.voidtype)*len(args.gravity)
            print(f' >> Running {len(cfg.zbins)} redshift bin(s) x {len(cfg.rvbins)} radius bin(s), for {len(cfg.voidtype)} void types.')
            print(f' >> Calculating {total} void profiles')

        print(f' >> Running for -{grav.upper()}-')
        init_globals()
        make_pix2idx_dict(SOURCE)

        for ((z_min, z_max), (rv_min, rv_max)) in product(cfg.zbins, cfg.rvbins):
            for void in cfg.voidtype:
                i+=1
                # reset in each loop
                delta_min, delta_max = -1.0, 100.0

                if void=='S':
                    delta_min = 0.0
                elif void=='R':
                    delta_max = 0.0

                print(f' \n[{i}/{total}]')
                check = stacking(rv_min, rv_max, z_min, z_max, delta_min, delta_max, grav.upper())
                assert check == 0, ' >> Something went wrong. << '

    print(' End! '.center(15,'='))
    print(f' >> Took {(time()-tini)/60.0:.3f} min <<\n\n')


if __name__ == '__main__':

    main()
