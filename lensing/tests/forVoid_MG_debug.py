from argparse import ArgumentParser
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord
from astropy.constants import G,c,M_sun,pc
from astropy.io import fits
import astropy.units as u
from functools import partial
import sys
sys.path.append('/home/fcaporaso/modified_gravity/lensing')
from funcs import ang_sep, eq2p2, cov_matrix
from multiprocessing import Pool
import numpy as np
import os
import sys
import time
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--lens_cat', type=str, default='voids_LCDM_09.dat', action='store')
parser.add_argument('--source_cat', type=str, default='l768_gr_z04-07_for02-03_19304.fits', action='store')
parser.add_argument('--sample', type=str, default='TEST_LCDM_', action='store')
parser.add_argument('-c','--ncores', type=int, default=2, action='store')
parser.add_argument('-r','--n_runslices', type=int, default=1, action='store')
parser.add_argument('--h_cosmo', type=float, default=1.0, action='store')
parser.add_argument('--Om0', type=float, default=0.3089, action='store')
parser.add_argument('--Ode0', type=float, default=0.6911, action='store')
parser.add_argument('--Rv_min', type=float, default=15.0, action='store')
parser.add_argument('--Rv_max', type=float, default=20.0, action='store')
parser.add_argument('--z_min', type=float, default=0.2, action='store')
parser.add_argument('--z_max', type=float, default=0.3, action='store')
parser.add_argument('--rho1_min', type=float, default=-1.0, action='store')
parser.add_argument('--rho1_max', type=float, default=0.0, action='store')
parser.add_argument('--rho2_min', type=float, default=-1.0, action='store')
parser.add_argument('--rho2_max', type=float, default=100.0, action='store')
parser.add_argument('--flag', type=float, default=2.0, action='store')
parser.add_argument('--octant', action='store_true') ## 'store_true' guarda True SOLO cuando se da --octant
parser.add_argument('--RIN', type=float, default=0.05, action='store')
parser.add_argument('--ROUT', type=float, default=5.0, action='store')    
parser.add_argument('-N','--ndots', type=int, default=22, action='store')    
parser.add_argument('-K','--nk', type=int, default=100, action='store')    
parser.add_argument('--addnoise', action='store_true')
args = parser.parse_args()
cosmo = LambdaCDM(H0=100*args.h_cosmo, Om0=args.Om0, Ode0=args.Ode0)

#parameters
cvel = c.value;    # Speed of light (m.s-1)
G    = G.value;    # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value    # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

def lenscat_load(lens_cat,
                 Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, flag,
                 ncores:int, octant:bool, nk:int):

    ## 0:Rv, 1:ra, 2:dec, 3:z, 4:xv, 5:yv, 6:zv, 7:rho1, 8:rho2, 9:logp, 10:diff CdM y CdV, 11:flag
    ## CdM: centro de masa
    ## CdV: centro del void
    L = np.loadtxt("/home/fcaporaso/cats/L768/"+lens_cat).T

    if octant:
        # selecciono los void en un octante
        eps = 1.0
        L = L[:, (L[1] >= 0.0+eps) & (L[1] <= 90.0-eps) & (L[2]>= 0.0+eps) & (L[2] <= 90.0-eps)]

    sqrt_nk = int(np.sqrt(nk))
    NNN = len(L[0]) ##total number of voids
    ra,dec = L[1]+180.0, L[2]
    K    = np.zeros((nk+1,NNN))
    K[0] = np.ones(NNN).astype(bool)

    ramin  = np.min(ra)
    cdec   = np.sin(np.deg2rad(dec))
    decmin = np.min(cdec)
    dra    = ((np.max(ra)+1.e-5) - ramin)/sqrt_nk
    ddec   = ((np.max(cdec)+1.e-5) - decmin)/sqrt_nk

    c = 1
    for a in range(sqrt_nk): 
        for d in range(sqrt_nk): 
            mra  = (ra  >= ramin + a*dra)&(ra < ramin + (a+1)*dra) 
            mdec = (cdec >= decmin + d*ddec)&(cdec < decmin + (d+1)*ddec) 
            K[c] = ~(mra&mdec)
            c += 1

    mask = (L[0] >= Rv_min) & (L[0] < Rv_max) & (L[3] >= z_min) & (L[3] < z_max) & (
            L[7] >= rho1_min) & (L[7] < rho1_max) & (L[8] >= rho2_min) & (L[8] < rho2_max) & (L[11] >= flag)

    nvoids = mask.sum()
    L = L[:,mask]
    L[1] = L[1] + 180.0

    if bool(ncores-1):
        if ncores > nvoids:
            ncores = nvoids
        lbins = int(round(nvoids/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)
        K = np.split(K.T, slices)

    return L, K, nvoids

def sourcecat_load(sourcename):
    folder = '/home/fcaporaso/cats/L768/'
    with fits.open(folder+sourcename) as f:
        mask = np.abs(f[1].data.gamma1) < 10.0
        S = f[1].data[mask]

    return S
    # return S.ra_gal, S.dec_gal, S.true_redshift_gal, S.kappa, S.gamma1, S.gamma2
        
def SigmaCrit(zl, zs):
    
    dl  = cosmo.angular_diameter_distance(zl).value
    Dl = dl*1.e6*pc #en m
    ds  = cosmo.angular_diameter_distance(zs).value              #dist ang diam de la fuente
    dls = cosmo.angular_diameter_distance_z1z2(zl, zs).value      #dist ang diam entre fuente y lente
                
    BETA_array = dls / ds

    return (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)

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
    
    ## solid angle sep with maria_func
    ## using in case the other mask fails
    if mask.sum() == 0:
        print('Fail for',RA0,DEC0)
        sep = ang_sep(
                np.deg2rad(RA0), np.deg2rad(DEC0),
                np.deg2rad(S.ra_gal), np.deg2rad(S.dec_gal)
        )
        mask = (sep < np.deg2rad(delta))&(S.true_redshift_gal >(Z+0.1))
        assert mask.sum() != 0

    catdata = S[mask]

    sigma_c = SigmaCrit(Z, catdata.true_redshift_gal)
    
    rads, theta, *_ = eq2p2(
        np.deg2rad(catdata.ra_gal), np.deg2rad(catdata.dec_gal),
        np.deg2rad(RA0), np.deg2rad(DEC0)
    )
                           
    e1 = catdata.gamma1
    e2 = -1.*catdata.gamma2
    # Add shape noise due to intrisic galaxy shapes        
    if addnoise:
        es1 = -1.*catdata.defl1
        es2 = catdata.defl2
        e1 += es1
        e2 += es2
    
    #get tangential ellipticities 
    cos2t = np.cos(2*theta)
    sin2t = np.sin(2*theta)
    et = (-e1*cos2t-e2*sin2t)*sigma_c/Rv
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

def stacking(RIN, ROUT, ndots, nk,
             L, K):

    # WHERE THE SUMS ARE GOING TO BE SAVED    
    Ninbin = np.zeros((nk+1,ndots))    
    SIGMAwsum    = np.zeros((nk+1,ndots)) 
    DSIGMAwsum_T = np.zeros((nk+1,ndots)) 
    DSIGMAwsum_X = np.zeros((nk+1,ndots))
                
    for i, Li in enumerate(tqdm(L)):
        num = len(Li)
        if num == 1:
            entrada = [Li[1], Li[2], Li[3], Li[0],
                       RIN, ROUT, ndots]
            resmap = np.array([part_profile_func(entrada)])

        else:
            entrada = np.array([Li.T[1],Li.T[2],Li.T[3],Li.T[0],
                                np.full(num,RIN), np.full(num,ROUT), np.full(num,ndots)]).T
            with Pool(processes=num) as pool:
                resmap = np.array(pool.map(partial_profile_unpack,entrada))
                pool.close()
                pool.join()
        
        for j, res in enumerate(resmap):
            km      = np.tile(K[i][j],(ndots,1)).T
                                
            SIGMAwsum    += np.tile(res[0],(nk+1,1))*km
            DSIGMAwsum_T += np.tile(res[1],(nk+1,1))*km
            DSIGMAwsum_X += np.tile(res[2],(nk+1,1))*km
            Ninbin += np.tile(res[3],(nk+1,1))*km

    # COMPUTING PROFILE        
    Ninbin[DSIGMAwsum_T == 0] = 1.
            
    Sigma     = (SIGMAwsum/Ninbin)
    DSigma_T  = (DSIGMAwsum_T/Ninbin)
    DSigma_X  = (DSIGMAwsum_X/Ninbin)

    return Sigma, DSigma_T, DSigma_X, Ninbin


def main(args=args):
        
    tini = time.time()
    #reading Lens catalog
    L, K, nvoids = lenscat_load(args.lens_cat,
        args.Rv_min, args.Rv_max, args.z_min, args.z_max, args.rho1_min, args.rho1_max, args.rho2_min, args.rho2_max, args.flag,
        args.ncores, args.octant, args.nk)
    
    # program arguments
    print(' Program arguments '.center(30,"="))
    print('Lens catalog: '.ljust(15,'.'), f' {args.lens_cat}'.rjust(15,'.'), sep='')
    # print('Sources catalog: '.ljust(15,'.'), f' {source_cat}'.rjust(15,'.'),sep='')
    print('Output name: '.ljust(15,'.'), f' {args.sample}'.rjust(15,'.'),sep='')
    print('N of cores: '.ljust(15,'.'), f' {args.ncores}'.rjust(15,'.'),sep='')
    print('N of slices: '.ljust(15,'.'), f' {args.n_runslices}'.rjust(15,'.'),sep='')
    
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
    print('Octante: '.ljust(15,'.'), f' {args.octant}'.rjust(15,'.'),sep='')
    print('N voids: '.ljust(15,'.'), f' {nvoids}'.rjust(15,'.'),sep='')
    
    # profile arguments
    print(' Profile arguments '.center(30,"="))
    print('RMIN: '.ljust(15,'.'), f' {args.RIN}'.rjust(15,'.'), sep='')
    print('RMAX: '.ljust(15,'.'), f' {args.ROUT}'.rjust(15,'.'),sep='')
    print('N: '.ljust(15,'.'), f' {args.ndots}'.rjust(15,'.'),sep='')
    print('N jackknife: '.ljust(15,'.'), f' {args.nk}'.rjust(15,'.'),sep='')
    print('Shape Noise: '.ljust(15,'.'), f' {args.addnoise}'.rjust(15,'.'),sep='')

    try:
        os.mkdir('results/')
    except FileExistsError:
        pass
    
    output_file = f'results/'
    # Defining radial bins
    bines = np.linspace(args.RIN,args.ROUT,num=args.ndots+1)
    R = (bines[:-1] + np.diff(bines)*0.5)

    if not bool(args.n_runslices-1):
        Sigma, DSigma_T, DSigma_X, Ninbin = stacking(args.RIN, args.ROUT, args.ndots, args.nk, L, K)

        covS = cov_matrix(Sigma[1:,:])
        covDSt = cov_matrix(DSigma_T[1:,:])
        covDSx = cov_matrix(DSigma_X[1:,:])

    else:
        cuts = np.round(np.linspace(args.RIN, args.ROUT, args.n_runslices+1),2)
        Sigma = np.array([])
        DSigma_T = np.array([])
        DSigma_X = np.array([])
        Ninbin = np.array([])
        
        n = args.ndots//args.n_runslices

        for j in np.arange(args.n_runslices):
            print(f'RUN {j+1} out of {args.n_runslices} slices')
            
            rmin, rmax = cuts[j], cuts[j+1]
            res_parcial = stacking(rmin, rmax, n, args.nk, L, K)

            Sigma = np.append(Sigma, res_parcial[0])
            DSigma_T = np.append(DSigma_T, res_parcial[1])
            DSigma_X = np.append(DSigma_X, res_parcial[2])
            Ninbin = np.append(Ninbin, res_parcial[3])
        
        Sigma = Sigma.reshape(args.nk+1,args.ndots)
        DSigma_T = DSigma_T.reshape(args.nk+1,args.ndots)
        DSigma_X = DSigma_X.reshape(args.nk+1,args.ndots)
        Ninbin = Ninbin.reshape(args.nk+1,args.ndots)

        covS = cov_matrix(Sigma[1:,:])
        covDSt = cov_matrix(DSigma_T[1:,:])
        covDSx = cov_matrix(DSigma_X[1:,:])

    # AVERAGE VOID PARAMETERS AND SAVE IT IN HEADER
    zmean    = np.concatenate([L[i][:,3] for i in range(len(L))]).mean()
    rvmean   = np.concatenate([L[i][:,0] for i in range(len(L))]).mean()
    rho2mean = np.concatenate([L[i][:,8] for i in range(len(L))]).mean()
    
    head = fits.Header()
    head.append(('nvoids',int(nvoids)))
    head.append(('cat',args.lens_cat))
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
    head.append(('SLCS_INFO'))
    head.append(('RMIN',np.round(args.RIN,4)))
    head.append(('RMAX',np.round(args.ROUT,4)))
    head.append(('ndots',np.round(args.ndots,4)))

    table_p = [fits.Column(name='Rp', format='E', array=R),
               fits.Column(name='Sigma', format='E', array=Sigma.flatten()),
               fits.Column(name='DSigma_T', format='E', array=DSigma_T.flatten()),
               fits.Column(name='DSigma_X', format='E', array=DSigma_X.flatten()),
               fits.Column(name='Ninbin', format='E', array=Ninbin.flatten())]

    table_c = [fits.Column(name='Sigma', format='E', array=covS.flatten()),
               fits.Column(name='DSigma_T', format='E', array=covDSt.flatten()),
               fits.Column(name='DSigma_X', format='E', array=covDSx.flatten())]

    tbhdu_p = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))
    tbhdu_c = fits.BinTableHDU.from_columns(fits.ColDefs(table_c))
    
    primary_hdu = fits.PrimaryHDU(header=head)
    
    hdul = fits.HDUList([primary_hdu, tbhdu_p, tbhdu_c])
    
    hdul.writeto(f'{output_file+args.sample}.fits',overwrite=True)
    print(f'File saved... {output_file+args.sample}.fits')
            
    print(f'Partial time: {np.round((time.time()-tini)/60. , 3)} mins')


if __name__=='__main__':

    tin = time.time()
    main()
    print(f'TOTAL TIME: {np.round((time.time()-tin)/60.,2)} min')
