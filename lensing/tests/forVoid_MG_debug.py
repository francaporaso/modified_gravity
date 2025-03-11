from argparse import ArgumentParser
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord, angular_separation
from astropy.io import fits
import astropy.units as u
from functools import partial
from multiprocessing import Pool
import numpy as np
import os
import time
from tqdm import tqdm

import sys
sys.path.append('/home/fcaporaso/modified_gravity/lensing/')
from funcs import eq2p2, cov_matrix, lenscat_load, sourcecat_load
from funcs import cvel,G,pc,Msun


parser = ArgumentParser()
parser.add_argument('--lens_cat', type=str, default='voids_LCDM_09.dat', action='store')
parser.add_argument('--source_cat', type=str, default='l768_gr_z04-07_for02-03_19304.fits', action='store')
parser.add_argument('--sample', type=str, default='TEST_LCDM_', action='store')
parser.add_argument('-c','--ncores', type=int, default=2, action='store')
parser.add_argument('-r','--n_runslices', type=int, default=1, action='store')
parser.add_argument('--h_cosmo', type=float, default=1.0, action='store')
parser.add_argument('--Om0', type=float, default=0.3089, action='store')
parser.add_argument('--Ode0', type=float, default=0.6911, action='store')
parser.add_argument('--Rv_min', type=float, default=1.0, action='store')
parser.add_argument('--Rv_max', type=float, default=50.0, action='store')
parser.add_argument('--z_min', type=float, default=0.0, action='store')
parser.add_argument('--z_max', type=float, default=0.6, action='store')
parser.add_argument('--rho1_min', type=float, default=-1.0, action='store')
parser.add_argument('--rho1_max', type=float, default=0.0, action='store')
parser.add_argument('--rho2_min', type=float, default=-1.0, action='store')
parser.add_argument('--rho2_max', type=float, default=100.0, action='store')
parser.add_argument('--flag', type=float, default=2.0, action='store')
# parser.add_argument('--octant', action='store_true') ## 'store_true' guarda True SOLO cuando se da --octant
parser.add_argument('--RIN', type=float, default=0.0, action='store')
parser.add_argument('--ROUT', type=float, default=5.0, action='store')    
parser.add_argument('-N','--ndots', type=int, default=22, action='store')    
parser.add_argument('-K','--nk', type=int, default=100, action='store')    
parser.add_argument('--addnoise', action='store_true')
args = parser.parse_args()

cosmo = LambdaCDM(H0=100.0*args.h_cosmo, Om0=args.Om0, Ode0=args.Ode0)

def SigmaCrit(zl, zs):

    global cosmo
    dl  = cosmo.angular_diameter_distance(zl).value
    Dl = dl*1.e6*pc #en m
    ds  = cosmo.angular_diameter_distance(zs).value              #dist ang diam de la fuente
    dls = cosmo.angular_diameter_distance_z1z2(zl, zs).value      #dist ang diam entre fuente y lente
                
    BETA_array = dls / ds

    return (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)

## TODO
## se puede probar definir mask dentro del for loop, 
## y q cargue en memoria solo los q caen en cada anillo
## eso debería ser más eficiente en memoria
def partial_profile(addnoise, S,
                    RA0, DEC0, Z, Rv,
                    RIN, ROUT, ndots):
    
    ndots = int(ndots)
    
    DEGxMPC = cosmo.arcsec_per_kpc_proper(Z).to('deg/Mpc').value
    delta = DEGxMPC*(ROUT*Rv)
    ## great-circle separation of sources from void centre
    ## WARNING: not stable near the edges
    pos_angles = np.arange(0,360,90)*u.deg
    c1 = SkyCoord(RA0, DEC0, unit='deg')
    c2 = c1.directional_offset_by(pos_angles, delta*u.deg)
    mask = (S.true_redshift_gal > (Z+0.1))&(
        S.dec_gal < c2[0].dec.deg)&(S.dec_gal > c2[2].dec.deg)&(
        S.ra_gal < c2[1].ra.deg)&(S.ra_gal > c2[3].ra.deg)
    
    ## solid angle sep with astropy
    ## using in case the other mask fails
    if mask.sum() == 0:
        # print('Failed mask for',RA0,DEC0)
        return [[-np.inf]]
        # sep = angular_separation(
        #         np.deg2rad(RA0), np.deg2rad(DEC0),
        #         np.deg2rad(S.ra_gal), np.deg2rad(S.dec_gal)
        # )
        # mask = (sep < np.deg2rad(delta))&(S.true_redshift_gal >(Z+0.1))
        # assert mask.sum() != 0

    catdata = S[mask]

    sigma_c = SigmaCrit(Z, catdata.true_redshift_gal)
    
    rads, theta = eq2p2(
        np.deg2rad(catdata.ra_gal), np.deg2rad(catdata.dec_gal),
        np.deg2rad(RA0), np.deg2rad(DEC0)
    )
                           
    e1 = -1.0*catdata.gamma1
    e2 = -1.0*catdata.gamma2
    # Add shape noise due to intrisic galaxy shapes        
    if addnoise:
        es1 = -1.*catdata.defl1
        es2 = catdata.defl2
        e1 += es1
        e2 += es2
    
    #get tangential ellipticities 
    cos2t = np.cos(2*theta)
    sin2t = np.sin(2*theta)
    et = -1.0*(e1*cos2t+e2*sin2t)*sigma_c/Rv
    ex = (-e1*sin2t+e2*cos2t)*sigma_c/Rv
           
    #get convergence
    k  = catdata.kappa*sigma_c/Rv

    r = (np.rad2deg(rads)/DEGxMPC)/Rv
    bines = np.linspace(RIN,ROUT,num=ndots+1)
    dig = np.digitize(r,bines)

    SIGMAwsum    = np.zeros(ndots)
    DSIGMAwsum_T = np.zeros(ndots)
    DSIGMAwsum_X = np.zeros(ndots)
    N_inbin      = np.zeros(ndots)

    for nbin in range(ndots):
        mbin = dig == nbin+1              
        SIGMAwsum[nbin]    = k[mbin].sum()
        DSIGMAwsum_T[nbin] = et[mbin].sum()
        DSIGMAwsum_X[nbin] = ex[mbin].sum()
        N_inbin[nbin]      = np.count_nonzero(mbin) ## hace lo mismo q mbin.sum() pero más rápido
    
    return SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin

part_profile_func = partial(partial_profile, args.addnoise, sourcecat_load(args.source_cat))
def partial_profile_unpack(minput):
    return part_profile_func(*minput)

def stacking(RIN, ROUT, ndots, nk, ncores,
             L, K):

    # WHERE THE SUMS ARE GOING TO BE SAVED    
    Ninbin = np.zeros((nk+1,ndots))    
    SIGMAwsum    = np.zeros((nk+1,ndots)) 
    DSIGMAwsum_T = np.zeros((nk+1,ndots)) 
    DSIGMAwsum_X = np.zeros((nk+1,ndots))
                
    discarded = 0

    with Pool(processes=ncores) as pool:

        entrada = np.array([L[1],L[2],L[3],L[0],
                            np.full(len(L.T),RIN), np.full(len(L.T),ROUT), np.full(len(L.T),ndots)]).T

        print('pool init')
        resmap = np.array(pool.map(partial_profile_unpack, entrada))
        pool.close()
        pool.join()
    
        print('saving to arrays')
        for j, res in enumerate(resmap):
            km = np.tile(K[j],(ndots,1)).T
            if np.isfinite(res[0][0]):
                SIGMAwsum    += np.tile(res[0],(nk+1,1))*km
                DSIGMAwsum_T += np.tile(res[1],(nk+1,1))*km
                DSIGMAwsum_X += np.tile(res[2],(nk+1,1))*km
                Ninbin += np.tile(res[3],(nk+1,1))*km
            else:
                discarded += 1 
        print('quiting context manager')

    # COMPUTING PROFILE        
    Ninbin[DSIGMAwsum_T == 0] = 1.
            
    Sigma     = (SIGMAwsum/Ninbin)
    DSigma_T  = (DSIGMAwsum_T/Ninbin)
    DSigma_X  = (DSIGMAwsum_X/Ninbin)

    print('Voids discarded: '.ljust(15,'.'), f' {discarded}'.rjust(15,'.'),sep='')

    return Sigma, DSigma_T, DSigma_X, Ninbin, discarded


def main(args=args):
        
    #reading Lens catalog
    ## ncores=1 means no slicing
    L, K, nvoids = lenscat_load(args.lens_cat,
        args.Rv_min, args.Rv_max, args.z_min, args.z_max, args.rho1_min, args.rho1_max, args.rho2_min, args.rho2_max, args.flag,
        1, args.nk)
    
    # program arguments
    print(' Program arguments '.center(30,"="))
    print('Lens cat: '.ljust(15,'.'), f' {args.lens_cat}'.rjust(15,'.'), sep='')
    print('Sour catalog: '.ljust(15,'.'), f' {args.source_cat.split("_")[-1][:-5]}'.rjust(15,'.'),sep='')
    print('Out: '.ljust(15,'.'), f' {args.sample}'.rjust(15,'.'),sep='')
    print('N cores: '.ljust(15,'.'), f' {args.ncores}'.rjust(15,'.'),sep='')
    print('N slices: '.ljust(15,'.'), f' {args.n_runslices}'.rjust(15,'.'),sep='')
    
    if args.rho2_max<=0:
        tipo = 'R'
    elif args.rho2_min>=0:
        tipo = 'S'
    else:
        tipo = 'all'
    
    # lens arguments
    print(' Void sample '.center(30,"="))
    print('Radii: '.ljust(15,'.'), f' [{args.Rv_min}, {args.Rv_max})'.rjust(15,'.'), sep='')
    print('Redshift: '.ljust(15,'.'), f' [{args.z_min}, {args.z_max})'.rjust(15,'.'),sep='')
    print('Tipo: '.ljust(15,'.'), f' {tipo}'.rjust(15,'.'),sep='')
    # print('Octante: '.ljust(15,'.'), f' {args.octant}'.rjust(15,'.'),sep='')
    print('N voids: '.ljust(15,'.'), f' {nvoids}'.rjust(15,'.'),sep='')
    
    # profile arguments
    print(' Profile arguments '.center(30,"="))
    print('RMIN: '.ljust(15,'.'), f' {args.RIN}'.rjust(15,'.'), sep='')
    print('RMAX: '.ljust(15,'.'), f' {args.ROUT}'.rjust(15,'.'),sep='')
    print('N: '.ljust(15,'.'), f' {args.ndots}'.rjust(15,'.'),sep='')
    print('N jackknife: '.ljust(15,'.'), f' {args.nk}'.rjust(15,'.'),sep='')
    print('Shape Noise: '.ljust(15,'.'), f' {args.addnoise}'.rjust(15,'.'),sep='')
    
    # Defining radial bins
    bines = np.linspace(args.RIN,args.ROUT,num=args.ndots+1)
    R = (bines[:-1] + np.diff(bines)*0.5)

    if not bool(args.n_runslices-1):
        print('Running single slice')
        Sigma, DSigma_T, DSigma_X, Ninbin, discarded = stacking(args.RIN, args.ROUT, args.ndots, args.nk, args.ncores, L, K)

        covS = cov_matrix(Sigma[1:,:])
        covDSt = cov_matrix(DSigma_T[1:,:])
        covDSx = cov_matrix(DSigma_X[1:,:])

    else:
        print('Running multiple slice')

        cuts = np.round(np.linspace(args.RIN, args.ROUT, args.n_runslices+1),2)
        # R = np.array([])
        Sigma = np.array([])
        DSigma_T = np.array([])
        DSigma_X = np.array([])
        Ninbin = np.array([])
        
        n = args.ndots//args.n_runslices

        for j in np.arange(args.n_runslices):
            print(f'RUN {j+1} out of {args.n_runslices} slices')
            
            rmin, rmax = cuts[j], cuts[j+1]
            res_parcial = stacking(rmin, rmax, n, args.nk, L, K)
            # R = np.append(R, res_parcial[0])
            Sigma = np.append(Sigma, res_parcial[0])
            DSigma_T = np.append(DSigma_T, res_parcial[1])
            DSigma_X = np.append(DSigma_X, res_parcial[2])
            Ninbin = np.append(Ninbin, res_parcial[3])
            discarded = res_parcial[-1]

        Sigma = Sigma.reshape(args.nk+1,args.ndots)
        DSigma_T = DSigma_T.reshape(args.nk+1,args.ndots)
        DSigma_X = DSigma_X.reshape(args.nk+1,args.ndots)
        Ninbin = Ninbin.reshape(args.nk+1,args.ndots)

        covS = cov_matrix(Sigma[1:,:])
        covDSt = cov_matrix(DSigma_T[1:,:])
        covDSx = cov_matrix(DSigma_X[1:,:])

    # AVERAGE VOID PARAMETERS AND SAVE IT IN HEADER
    zmean    = np.concatenate([Li[:,3] for Li in L]).mean()
    rvmean   = np.concatenate([Li[:,0] for Li in L]).mean()
    rho2mean = np.concatenate([Li[:,8] for Li in L]).mean()
    
    head = fits.Header()
    head.append(('nvoids',int(nvoids-discarded)))
    head.append(('lens',args.lens_cat))
    head.append(('sour',args.source_cat.split('_')[-1][:-5],'cosmohub stamp')) ## considerando q está separado por '_'
    head.append(('Rv_min',np.round(args.Rv_min,2)))
    head.append(('Rv_max',np.round(args.Rv_max,2)))
    head.append(('Rv_mean',np.round(rvmean,4)))
    head.append(('r1_min',np.round(args.rho1_min,2)))
    head.append(('r1_max',np.round(args.rho1_max,2)))
    head.append(('r2_min',np.round(args.rho2_min,2)))
    head.append(('r2_max',np.round(args.rho2_max,2)))
    head.append(('r2_mean',np.round(rho2mean,4)))
    head.append(('z_min',np.round(args.z_min,2)))
    head.append(('z_max',np.round(args.z_max,2)))
    head.append(('z_mean',np.round(zmean,4)))
    # head.append(('SLCS_INFO'))
    head.append(('RMIN',np.round(args.RIN,4)))
    head.append(('RMAX',np.round(args.ROUT,4)))
    head.append(('ndots',np.round(args.ndots,4)))
    head.append(('nk',np.round(args.nk,4),'jackknife slices'))
    head['HISTORY'] = f'{time.asctime()}'

    table_p = [fits.Column(name='Rp', format='E', array=R),
               fits.Column(name='Sigma', format='E', array=Sigma.flatten()),
               fits.Column(name='DSigma_T', format='E', array=DSigma_T.flatten()),
               fits.Column(name='DSigma_X', format='E', array=DSigma_X.flatten()),
               fits.Column(name='Ninbin', format='E', array=Ninbin.flatten())]

    table_c = [fits.Column(name='covS', format='E', array=covS.flatten()),
               fits.Column(name='covDSt', format='E', array=covDSt.flatten()),
               fits.Column(name='covDSx', format='E', array=covDSx.flatten())]

    tbhdu_p = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))
    tbhdu_c = fits.BinTableHDU.from_columns(fits.ColDefs(table_c))
    
    primary_hdu = fits.PrimaryHDU(header=head)
    
    hdul = fits.HDUList([primary_hdu, tbhdu_p, tbhdu_c])

    output_file = f'results/{args.sample}_{args.lens_cat[6:-4]}_{np.ceil(args.Rv_min).astype(int)}-{np.ceil(args.Rv_max).astype(int)}_z0{int(10.0*args.z_min)}-0{int(10.0*args.z_max)}_type{tipo}.fits'

    hdul.writeto(output_file,overwrite=True)
    print(f'File saved in: {output_file}')
    

if __name__=='__main__':

    tin = time.time()
    print('''
    ▗▖▗▞▀▚▖▄▄▄▄   ▄▄▄ ▄ ▄▄▄▄   ▗▄▄▖
    ▐▌▐▛▀▀▘█   █ ▀▄▄  ▄ █   █ ▐▌   
    ▐▌▝▚▄▄▖█   █ ▄▄▄▀ █ █   █ ▐▌▝▜▌
    ▐▙▄▄▖             █       ▝▚▄▞▘
    ''',
    flush=True)
    main()
    print(' TOTAL TIME '.ljust(15,'.'), f' {np.round((time.time()-tin)/60.,2)} min'.rjust(15,'.'),sep='')
    print(' END :) '.center(30,"="))
