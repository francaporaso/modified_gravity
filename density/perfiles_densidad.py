from argparse import ArgumentParser
from astropy.cosmology import LambdaCDM
from astropy.io import fits
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import sys
from tqdm import tqdm
import time

sys.path.append('/home/fcaporaso/modified_gravity/lensing/')
from funcs import lenscat_load, cov_matrix
sys.path.append('/home/fcaporaso/FlagShip/vgcf/')
from vgcf import ang2xyz

parser = ArgumentParser()
parser.add_argument('--lens_cat', type=str, default='voids_LCDM_09.dat', action='store')
parser.add_argument('--tracer_cat', type=str, default='l768_gr_centrals_19602.fits', action='store')
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
parser.add_argument('--RIN', type=float, default=0.05, action='store')
parser.add_argument('--ROUT', type=float, default=5.0, action='store')    
parser.add_argument('-N','--ndots', type=int, default=22, action='store')    
parser.add_argument('-K','--nk', type=int, default=100, action='store')    
# parser.add_argument('--addnoise', action='store_true')
args = parser.parse_args()

cosmo = LambdaCDM(H0=100*args.h_cosmo, Om0=args.Om0, Ode0=args.Ode0) ## Planck15

def tracercat_load(catname=args.tracer_cat, ## descargar catalogo
                   if_centrals=True, cosmo=cosmo):
    
    folder = '/home/fcaporaso/cats/L768/'
    if if_centrals:    
        with fits.open(catname) as f:
            centrals = f[1].data.kind == 0 ## chequear q sean los centrales!
            z_gal   = f[1].data.true_redshift_gal
            mask_z  = (z_gal >= 0.1) & (z_gal <= 0.5)
            mmm = centrals&mask_z
            ra_gal  = f[1].data.ra_gal[mmm]
            dec_gal = f[1].data.dec_gal[mmm]
            z_gal   = z_gal[mmm]
            lmhalo  = f[1].data.halo_lm[mmm] ## chequear
        
        xh,yh,zh = ang2xyz(ra_gal, dec_gal, z_gal, cosmo=cosmo)
        return xh, yh, zh, lmhalo
    else:
        with fits.open(catname) as f:
            ra_gal  = f[1].data.ra_gal
            dec_gal = f[1].data.dec_gal
            z_gal   = f[1].data.true_redshift_cgal
        
        xh,yh,zh = ang2xyz(ra_gal, dec_gal, z_gal, cosmo=cosmo)
        return xh, yh , zh
    
def mean_density_comovingshell(xh, yh, zh, logmh,
                               m, rv, xv, yv, zv):

    dist_void = np.sqrt(xv**2 + yv**2 + zv**2)
    dist = np.sqrt(xh**2 + yh**2 + zh**2)
    chi_min = dist_void - m*rv
    chi_max = dist_void + m*rv

    lmh = logmh[(dist > chi_min)&(dist < chi_max)]

    vol = (1/8)*(4*np.pi/3)*(chi_max**3 - chi_min**3)
    mass = np.sum(10.0 ** lmh)

    return mass/vol, len(lmh)/vol

def number_density_v2(N, m, xh, yh, zh, lmhalo, rv, xv, yv, zv):
    number_gx = np.zeros(N)
    mass_bin = np.zeros(N)
    vol = np.zeros(N)
    dist = np.sqrt((xh-xv)**2 + (yh-yv)**2 + (zh-zv)**2) ## dist to center of void i
    const = m*rv/N

    mask_mean = (dist < 1.1*m*rv)
    logmass = lmhalo[mask_mean]
    dist = dist[mask_mean]

    mean_den_com, mean_gx_com = mean_density_comovingshell(xh,yh,zh,lmhalo,
                                                           m,rv,xv,yv,zv)
    
    # mass_ball = np.sum( 10.0**(logmass) )
    # vol_ball = (4/3)*np.pi*(5*m*rv)**3
    # mean_gx_ball = np.sum(mask_mean)/vol_ball
    # mean_den_ball = mass_ball/vol_ball
    
    for k in range(N):
        mask = (dist < (k+1)*const) & (dist >= k*const)
        number_gx[k] = mask.sum()
        mass_bin[k] = np.sum( 10.0**(logmass[mask]) )
        vol[k] = (k+1)**3 - k**3
    
    vol *= (4/3)*np.pi*const**3
    
    return number_gx, mass_bin, vol, np.full_like(vol, mean_gx_com), np.full_like(vol, mean_den_com)

partial_func = partial(number_density_v2, args.ndots, args.ROUT, *tracercat_load())
def partial_func_unpack(A):
    return partial_func(*A)

def stacking(N, m, 
             L, K,
             nk):
    
    # COVARIANZA JACKKNIFE
    numbergx = np.zeros((nk+1,N))
    massbin = np.zeros((nk+1,N))
    mu = np.zeros((nk+1,N)) ## vol * denball_5
    mu_gx = np.zeros((nk+1,N)) ## vol * ngal_ball_5
    
    # POISSON
    # numbergx = np.zeros((nvoids,N))
    # mu_gx = np.zeros((nvoids,N)) ## vol * meangxcomsh

    # massbin = np.zeros((nvoids,N))
    # mu = np.zeros((nvoids,N)) ## vol * meandencomsh

    # count = 0
    for i,Li in enumerate(tqdm(L)):
        num=len(Li)
        entrada = np.array([Li.T[1], Li.T[5], Li.T[6], Li.T[7]]).T
        with Pool(processes=num) as pool:
            resmap = pool.map(partial_func_unpack,
                           entrada)
            pool.close()
            pool.join()
        
        for j, res in enumerate(resmap):
            #COVARIANZA JACKKNIFE
            km = np.tile(K[i][j], (N,1)).T
            numbergx += np.tile(res[0], (nk+1,1))*km
            massbin += np.tile(res[1], (nk+1,1))*km
            mu += np.tile(res[2]*res[4], (nk+1,1))*km
            mu_gx += np.tile(res[2]*res[3], (nk+1,1))*km

            #POISSON
            # numbergx[count] = res[0]
            # massbin[count] = res[1]
            # mu[count] = res[2]*res[4]
            # mu_gx[count] = res[2]*res[3]
            # count+=1
    
    # COVARIANZA JACKKNIFE
    delta = massbin/mu - 1
    deltagx = numbergx/mu_gx - 1
    cov_delta = cov_matrix(delta[1:,:])
    cov_deltagx = cov_matrix(deltagx[1:,:])
    return delta[0], deltagx[0], cov_delta, cov_deltagx
    
    # POISSON
    # Ngx = np.sum(numbergx,axis=0)
    # Msum = np.sum(massbin,axis=0)
    # e_M = np.std(massbin,axis=0)
    # mu_sum = np.sum(mu,axis=0)
    # e_mu = np.std(mu,axis=0)
    # 
    # delta = Msum/mu_sum - 1
    # deltagx = Ngx/np.sum(mu_gx,axis=0) - 1
    # e_delta = np.sqrt( e_M**2 + (Msum*e_mu/mu_sum)**2 )/mu_sum
    
    # if lensargs[7]<=0:
    #     t = 'R'
    # elif lensargs[6]>=0:
    #     t = 'S'
    # else:
    #     t = 'all'
    # 
    # np.savetxt(f'density_mice_mdcs_Rv{int(lensargs[0])}-{int(lensargs[1])}_{t}_z0{int(10*lensargs[2])}-0{int(10*lensargs[3])}_{sample}.csv', 
    #         np.column_stack([delta, deltagx, e_delta]), 
    #         delimiter=','
    # )

def main(args=args):

    L,K,nvoids = lenscat_load(args.lens_cat,
            args.Rv_min, args.Rv_max, args.z_min, args.z_max, args.rho1_min, args.rho1_max, args.rho2_min, args.rho2_max, args.flag,
            args.ncores, args.nk)

    if args.rho2_min<=0:
        t = 'R'
    elif args.rho2_max>=0:
        t = 'S'
    else:
        t = 'all'
        
    # program arguments
    print(' Program arguments '.center(30,"="))
    print('Lens catalog: '.ljust(15,'.'), f' {args.lens_cat}'.rjust(15,'.'), sep='')
    # print('Sources catalog: '.ljust(15,'.'), f' {source_cat}'.rjust(15,'.'),sep='')
    print('Output name: '.ljust(15,'.'), f' {args.sample}'.rjust(15,'.'),sep='')
    print('N of cores: '.ljust(15,'.'), f' {args.ncores}'.rjust(15,'.'),sep='')
    print('N of slices: '.ljust(15,'.'), f' {args.n_runslices}'.rjust(15,'.'),sep='')

    # lens arguments
    print(' Void sample '.center(30,"="))
    print('Radii: '.ljust(15,'.'), f' [{args.Rv_min}, {args.Rv_max})'.rjust(15,'.'), sep='')
    print('Redshift: '.ljust(15,'.'), f' [{args.z_min}, {args.z_max})'.rjust(15,'.'),sep='')
    print('Tipo: '.ljust(15,'.'), f' {t}'.rjust(15,'.'),sep='')
    # print('Octante: '.ljust(15,'.'), f' {args.octant}'.rjust(15,'.'),sep='')
    print('N voids: '.ljust(15,'.'), f' {nvoids}'.rjust(15,'.'),sep='')
    
    # profile arguments
    print(' Profile arguments '.center(30,"="))
    print('RMIN: '.ljust(15,'.'), f' {args.RIN}'.rjust(15,'.'), sep='')
    print('RMAX: '.ljust(15,'.'), f' {args.ROUT}'.rjust(15,'.'),sep='')
    print('N: '.ljust(15,'.'), f' {args.ndots}'.rjust(15,'.'),sep='')
    print('N jackknife: '.ljust(15,'.'), f' {args.nk}'.rjust(15,'.'),sep='')
    # print('Shape Noise: '.ljust(15,'.'), f' {args.addnoise}'.rjust(15,'.'),sep='')
    
    delta, deltagx, covdelta, covdeltagx = stacking(N=args.ndots, m=args.ROUT, L=L, K=K, nk=args.nk)
    
    Lrv = np.concatenate([Li[:,0] for Li in L])
    Lz = np.concatenate([Li[:,3] for Li in L])
    
    head = fits.Header()
    head.append(('Nvoids', int(nvoids))) ### TODO cuidado, no son los mismos q lensing xq no hay discarding
    head.append(('lens',args.lens_cat))
    head.append(('sour',args.source_cat[-10:-5],'cosmohub stamp')) ## considerando q los numeros de cosmohub son simepre 5...
    head.append(('Rv_min', Lrv.min()))
    head.append(('Rv_max', Lrv.max()))
    head.append(('z_min', Lz.min()))
    head.append(('z_max', Lz.max()))
    head.append(('rho1_min', args.rho1_min))
    head.append(('rho1_max', args.rho1_max))
    head.append(('rho2_min', args.rho2_min))
    head.append(('rho2_max', args.rho2_max))
    head.append(('rmax', args.ROUT))
    head.append(('ndots',np.round(args.ndots,4)))
    head.append(('nk',np.round(args.nk,4),'jackknife slices'))
    head['HISTORY'] = f'{time.asctime()}'

    primary_hdu = fits.PrimaryHDU(header=head)
    hdul = fits.HDUList([primary_hdu])
    
    rrr = np.linspace(0,args.ROUT,args.ndots+1)
    rrr = rrr[:-1] + np.diff(rrr)*0.5
    
    table_delta = np.array([fits.Column(name='r', format='E', array=rrr),
                      fits.Column(name='delta', format='E', array=delta),
                      fits.Column(name='deltagx', format='E', array=deltagx),
                     ])
    table_cov = np.array([fits.Column(name='cov_delta', format='E', array=covdelta.flatten()),
                          fits.Column(name='cov_deltagx', format='E', array=covdeltagx.flatten()),
                     ])

    hdul.append(fits.BinTableHDU.from_columns(table_delta))
    hdul.append(fits.BinTableHDU.from_columns(table_cov))
    
    output_file = f'results/density_{args.sample}_{args.lens_cat[6:-4]}_{np.ceil(args.Rv_min).astype(int)}-{np.ceil(args.Rv_max).astype(int)}_z0{int(10.0*args.z_min)}-0{int(10.0*args.z_max)}_type{t}.fits'
    hdul.writeto(output_file, overwrite=True)

### -------- RUN
if __name__ == '__main__':

    tin = time.time()
    main()
    print(' TOTAL TIME '.ljust(15,'.'), f' {np.round((time.time()-tin)/60.,2)} min'.rjust(15,'.'),sep='')
    print(' END :) '.center(30,"="))
