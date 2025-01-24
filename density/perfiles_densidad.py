import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import LambdaCDM
from astropy.io import fits
import sys
# sys.path.append('/home/fcaporaso/FlagShip/profiles/')
sys.path.append('/home/fcaporaso/FlagShip/vgcf/')
# from perfiles import lenscat_load
from vgcf import ang2xyz
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import time

cosmo = LambdaCDM(H0=100, Om0=0.3089, Ode0=0.6911) ## Planck15

tin = time.time()

## ------ PARAMS
N = 22 ## Num de puntos del perfil
m = 5 ## dist maxima en R_v del perfil

def lenscat_load(Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, 
                 flag=2.0, lensname="voids_LCDM_09.dat",
                 split=False, NSPLITS=1,
                 nk = 100, 
                 octant=True):

    ## 0:Rv, 1:ra, 2:dec, 3:z, 4:xv, 5:yv, 6:zv, 7:rho1, 8:rho2, 9:logp, 10:diff CdM y CdV, 11:flag
    ## CdM: centro de masa
    ## CdV: centro del void
    L = np.loadtxt("/home/fcaporaso/cats/L768/"+lensname).T

    if octant:
        # selecciono los void en un octante
        eps = 1.0
        L = L[:, (L[1] >= 0.0+eps) & (L[1] <= 90.0-eps) & (L[2]>= 0.0+eps) & (L[2] <= 90.0-eps)]

    sqrt_nk = int(np.sqrt(nk))
    NNN = len(L[0]) ##total number of voids
    ra,dec = L[1],L[2]
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

    if split:
        if NSPLITS > nvoids:
            NSPLITS = nvoids
        lbins = int(round(nvoids/float(NSPLITS), 0))
        slices = ((np.arange(lbins)+1)*NSPLITS).astype(int)
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)
        K = np.split(K.T, slices)

    return L, K, nvoids

def tracercat_load(catname='/home/fcaporaso/cats/L768/l768_gr_halos_{}.fits', ## descargar catalogo
                   if_centrals=True, cosmo=cosmo):
    
    if if_centrals:    
        with fits.open(catname) as f:
            centrals = f[1].data.kind == 0 ## chequear q sean los centrales!
            z_gal   = f[1].data.true_redshift_gal
            mask_z  = (z_gal >= 0.1) & (z_gal <= 0.5)
            mmm = centrals&mask_z
            ra_gal  = f[1].data.ra_gal[mmm]
            dec_gal = f[1].data.dec_gal[mmm]
            z_gal   = z_gal[mmm]
            lmhalo  = f[1].data.lmhalo[mmm] ## chequear
        
        xh,yh,zh = ang2xyz(ra_gal, dec_gal, z_gal, cosmo=cosmo)
        return xh, yh, zh, lmhalo
    else:
        with fits.open(catname) as f:
            ra_gal  = f[1].data.ra_gal
            dec_gal = f[1].data.dec_gal
            z_gal   = f[1].data.true_redshift_cgal
        
        xh,yh,zh = ang2xyz(ra_gal, dec_gal, z_gal, cosmo=cosmo)
        return xh, yh , zh

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

partial_func = partial(number_density_v2, N, m, *tracercat_load())
def partial_func_unpack(A):
    return partial_func(*A)

def saveresults(args,nvoids,sample,
               *results):
    h = fits.Header()
    
    h.append(('Nvoids', int(nvoids)))
    h.append(('Rv_min', args[0]))
    h.append(('Rv_max', args[1]))
    h.append(('z_min', args[2]))
    h.append(('z_max', args[3]))
    h.append(('rho1_min', args[4]))
    h.append(('rho1_max', args[5]))
    h.append(('rho2_min', args[6]))
    h.append(('rho2_max', args[7]))
    h.append(('rmax', m))

    primary_hdu = fits.PrimaryHDU(header=h)
    hdul = fits.HDUList([primary_hdu])
    
    rrr = np.linspace(0,m,N+1)
    rrr = rrr[:-1] + np.diff(rrr)*0.5
    
    table_delta = np.array([fits.Column(name='r', format='E', array=rrr),
                      fits.Column(name='delta', format='E', array=results[0]),
                      fits.Column(name='deltagx', format='E', array=results[1]),
                     ])
    table_cov = np.array([fits.Column(name='cov_delta', format='E', array=results[2].flatten()),
                          fits.Column(name='cov_deltagx', format='E', array=results[3].flatten()),
                     ])

    hdul.append(fits.BinTableHDU.from_columns(table_delta))
    hdul.append(fits.BinTableHDU.from_columns(table_cov))
    
    if args[7]<=0:
        t = 'R'
    elif args[6]>=0:
        t = 'S'
    else:
        t = 'all'
    
    hdul.writeto(f'density_l768_{simu}_Rv{int(args[0])}-{int(args[1])}_{t}_z0{int(10*args[2])}-0{int(10*args[3])}_{sample}.fits',
                 overwrite=True)

def stacking(N, m, 
             lensargs,
             sample,
             L, K, nvoids,
             nk = 100):
    
    print(f"nvoids: {nvoids}")

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
    saveresults(lensargs, nvoids, sample, delta[0], deltagx[0], cov_delta, cov_deltagx)
    
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


### -------- RUN
ncores = 64
args_list = [ ## select samples
    # (6.0,9.622,0.2,0.4,-1.0,-0.8,-1.0,100.0),
    # (6.0,9.622,0.2,0.4,-1.0,-0.8,0.0,100.0),
    # (6.0,9.622,0.2,0.4,-1.0,-0.8,-1.0,0.0),
    # (9.622,50.0,0.2,0.4,-1.0,-0.8,-1.0,100.0),
    # (9.622,50.0,0.2,0.4,-1.0,-0.8,0.0,100.0),
    # (9.622,50.0,0.2,0.4,-1.0,-0.8,-1.0,0.0),
]
sample = 'N25'
for lensargs in args_list:

    stacking(
        N,
        m,
        lensargs,
        sample,
        *lenscat_load(
            *lensargs, 
            flag=2.0, 
            lensname="/mnt/simulations/MICE/voids_MICE.dat",
            split=True, 
            NSPLITS=ncores
        )
    )

print(f'Ended in: {np.round((time.time()-tin)/60.0, 2)} min')
