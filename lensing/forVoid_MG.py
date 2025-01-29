import argparse
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord
from astropy.constants import G,c,M_sun,pc
from astropy.io import fits
import astropy.units as u
from functools import partial
from maria_func import *
from multiprocessing import Pool
import numpy as np
import os
import sys
import time
from tqdm import tqdm

options = {
	'-sample':'pru',
	'-lens_cat':'voids_LCDM_09.dat',
	'-source_cat':'l768_gr_octant_19218.fits',
	'-Rv_min':0.,
	'-Rv_max':50.,
	'-rho1_min':-1.,
	'-rho1_max':1.,
	'-rho2_min':-1.,
	'-rho2_max':100.,
	'-FLAG':2.,
	'-z_min':0.1,
	'-z_max':0.5,
	'-addnoise':False,
	'-RIN':0.05,
	'-ROUT':5.,
	'-ndots':40,
	'-ncores':10,
	'-nk':100,
	'-nslices':1,
}

parser = argparse.ArgumentParser()
for key,val in options.items():
    parser.add_argument(key, action='store',dest=key[1:],default=val,type=type(val))
args = parser.parse_args()

h, Om0, Ode0 = 1.0, 0.3089, 0.6911 #Planck15
cosmo = LambdaCDM(H0=100*h, Om0=Om0, Ode0=Ode0)

#parameters
cvel = c.value;    # Speed of light (m.s-1)
G    = G.value;    # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value    # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

### TODO
## añadir loop para q calcule ambos fR y LCDM

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

def partial_profile(RIN, ROUT, ndots, addnoise,
                    # ra_gal, dec_gal, true_redshift_gal, kappa, gamma1, gamma2,
                    S,
                    RA0, DEC0, Z, Rv):
    
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

    ## solid angle separation in sky from RA0,DEC0
    ## WARNING: memory leak, too many objects to calculate sep
    # sep = coords.separation(SkyCoord(RA0,DEC0,unit='deg')).value
    # mask = (sep < delta)&(true_redshift_gal > (Z+0.1))
    
    ## solid angle sep with maria_func
    ## WARNING: memory leak, too many objects to calculate sep
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
    # catdata_ra = ra_gal[mask]
    # catdata_dec = dec_gal[mask]
    # catdata_z = true_redshift_gal[mask]
    # catdata_kappa = kappa[mask]
    # catdata_gamma1 = gamma1[mask]
    # catdata_gamma2 = gamma2[mask]

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
    et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c/Rv
    #get cross ellipticities
    ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c/Rv
           
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

part_profile_func = partial(
    partial_profile, args.RIN, args.ROUT, args.ndots, args.addnoise, sourcecat_load(args.source_cat),
)
def partial_profile_unpack(minput):
    return part_profile_func(*minput)

## TODO
# para q funcione import, se puede agregar el llamado de los args acá adentro!
# lo unico, hay q repensar la definición de part_profile_func !!
def main(lcat, sample='pru', output_file=None,
         Rv_min=0., Rv_max=50.,
         rho1_min=-1., rho1_max=0.,
         rho2_min=-1., rho2_max=100.,
         z_min = 0.1, z_max = 1.0,
         RIN = .05, ROUT =5.,
         ndots= 40, ncores=10, nk=100,
         addnoise = False, FLAG = 2.):
        
    tini = time.time()
    #reading Lens catalog
    L, K, nvoids = lenscat_load(
        Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
        flag=FLAG, lensname=lcat, split=True, NSPLITS=ncores, nk=nk, octant=False,
    )
    
    # program arguments
    print(' Program arguments '.center(30,"="))
    print('Lens catalog: '.ljust(15,'.'), f' {lcat}'.rjust(15,'.'), sep='')
    # print('Sources catalog: '.ljust(15,'.'), f' {source_cat}'.rjust(15,'.'),sep='')
    print('Output name: '.ljust(15,'.'), f' {sample}'.rjust(15,'.'),sep='')
    print('N of cores: '.ljust(15,'.'), f' {ncores}'.rjust(15,'.'),sep='')
    # print('N of slices: '.ljust(15,'.'), f' {n_runslices}'.rjust(15,'.'),sep='')

    # cosmology
    # print(' Cosmo params '.center(30,"="))
    # print('h: '.ljust(15,'.'), f' {h}'.rjust(15,'.'), sep='')
    # print('Om0: '.ljust(15,'.'), f' {Om0}'.rjust(15,'.'),sep='')
    # print('Ode0: '.ljust(15,'.'), f' {Ode0}'.rjust(15,'.'),sep='')
    
    if rho2_max<=0:
        tipo = 'R'
    elif rho2_min>=0:
        tipo = 'S'
    else:
        tipo = 'all'
    
    # lens arguments
    print(' Void sample '.center(30,"="))
    print('Radii: '.ljust(15,'.'), f' [{Rv_min}, {Rv_max})'.rjust(15,'.'), sep='')
    print('Redshift: '.ljust(15,'.'), f' [{z_min}, {z_max})'.rjust(15,'.'),sep='')
    print('Tipo: '.ljust(15,'.'), f' {tipo}'.rjust(15,'.'),sep='')
    print('Octante: '.ljust(15,'.'), f' {False}'.rjust(15,'.'),sep='')
    
    # profile arguments
    print(' Profile arguments '.center(30,"="))
    print('RMIN: '.ljust(15,'.'), f' {RIN}'.rjust(15,'.'), sep='')
    print('RMAX: '.ljust(15,'.'), f' {ROUT}'.rjust(15,'.'),sep='')
    print('N: '.ljust(15,'.'), f' {ndots}'.rjust(15,'.'),sep='')
    print('N jackknife: '.ljust(15,'.'), f' {nk}'.rjust(15,'.'),sep='')
    print('Shape Noise: '.ljust(15,'.'), f' {addnoise}'.rjust(15,'.'),sep='')

    try:
        os.mkdir('results/')
    except FileExistsError:
        pass
    
    if not output_file:
        output_file = f'results/'
    # Defining radial bins
    bines = np.linspace(RIN,ROUT,num=ndots+1)
    R = (bines[:-1] + np.diff(bines)*0.5)
    # WHERE THE SUMS ARE GOING TO BE SAVED
    
    Ninbin = np.zeros((nk+1,ndots))    
    SIGMAwsum    = np.zeros((nk+1,ndots)) 
    DSIGMAwsum_T = np.zeros((nk+1,ndots)) 
    DSIGMAwsum_X = np.zeros((nk+1,ndots))
                
    print(f'Saved in ../{output_file+sample}.fits')
    LARGO = len(L)
    tslice = np.array([])
    
    for i, Li in enumerate(tqdm(L)):
                
        t1 = time.time()
        num = len(Li)
        
        if num == 1:
            entrada = [Li[1], Li[2],
                       Li[3], Li[0]]
            
            resmap = np.array([partial_profile(entrada)])

        else:
            entrada = np.array([Li.T[1],Li.T[2],Li.T[3],Li.T[0]]).T
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

    # AVERAGE VOID PARAMETERS AND SAVE IT IN HEADER
    zmean    = np.concatenate([L[i][:,3] for i in range(len(L))]).mean()
    rvmean   = np.concatenate([L[i][:,0] for i in range(len(L))]).mean()
    rho2mean = np.concatenate([L[i][:,8] for i in range(len(L))]).mean()
    
    head = fits.Header()
    head.append(('nvoids',int(nvoids)))
    head.append(('cat',lcat))
    head.append(('Rv_min',np.round(Rv_min,2)))
    head.append(('Rv_max',np.round(Rv_max,2)))
    head.append(('Rv_mean',np.round(rvmean,4)))
    head.append(('r1_min',np.round(rho1_min,2)))
    head.append(('r1_max',np.round(rho1_max,2)))
    head.append(('r2_min',np.round(rho2_min,2)))
    head.append(('r2_max',np.round(rho2_max,2)))
    head.append(('r2_mean',np.round(rho2mean,4)))
    head.append(('z_min',np.round(z_min,2)))
    head.append(('z_max',np.round(z_max,2)))
    head.append(('z_mean',np.round(zmean,4)))
    head.append(('SLCS_INFO'))
    head.append(('RMIN',np.round(RIN,4)))
    head.append(('RMAX',np.round(ROUT,4)))
    head.append(('ndots',np.round(ndots,4)))

    table_p = [fits.Column(name='Rp', format='E', array=R),
               fits.Column(name='Sigma',    format='E', array=Sigma.flatten()),
               fits.Column(name='DSigma_T', format='E', array=DSigma_T.flatten()),
               fits.Column(name='DSigma_X', format='E', array=DSigma_X.flatten()),
               fits.Column(name='Ninbin', format='E', array=Ninbin.flatten())]

    tbhdu_p = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))
    
    primary_hdu = fits.PrimaryHDU(header=head)
    
    hdul = fits.HDUList([primary_hdu, tbhdu_p])
    
    hdul.writeto(f'{output_file+sample}.fits',overwrite=True)
    print(f'File saved... {output_file+sample}.fits')
            
    tfin = time.time()
    
    print(f'Partial time: {np.round((tfin-tini)/60. , 3)} mins')
        

def run_in_parts(RIN,ROUT, nslices,
                lcat, sample='pru', output_file=None, Rv_min=0., Rv_max=50., rho1_min=-1., rho1_max=0., 
                rho2_min=-1., rho2_max=100., z_min = 0.1, z_max = 1.0, ndots= 40, ncores=10,
                addnoise=False, FLAG = 2.):

    cuts = np.round(np.linspace(RIN,ROUT,num=nslices+1),2)
    
    try:
        os.mkdir(f'/home/fcaporaso/modified_gravity/lensing/profiles/Rv{round(Rv_min)}-{round(Rv_max)}/')
    except FileExistsError:
        pass
    if not output_file:
        output_file = f'/home/fcaporaso/modified_gravity/lensing/profiles/Rv{round(Rv_min)}-{round(Rv_max)}/'
    
    tslice = np.zeros(nslices)
    #orden inverso: calcula del corte mas externo al mas interno
    #cuts = cuts[::-1]
    for j in np.arange(nslices):
        RIN, ROUT = cuts[j], cuts[j+1]
        #ROUT, RIN = cuts[j], cuts[j+1]
        t1 = time.time()
        print(f'RUN {j+1} out of {nslices} slices')
        #print(f'RUNNING FOR RIN={RIN}, ROUT={ROUT}')
        main(
            lcat, 
            sample+f'rbin_{j}',
            output_file=output_file,
            Rv_min=Rv_min, Rv_max=Rv_max, 
            z_min=z_min, z_max=z_max,
            rho1_min=rho1_min, rho1_max=rho1_max, 
            rho2_min=rho2_min, rho2_max=rho2_max, 
            RIN=RIN, ROUT=ROUT, 
            ndots=ndots//nslices, 
            ncores=ncores, 
            addnoise=addnoise, 
            FLAG=FLAG
        )
        t2 = time.time()
        tslice[j] = (t2-t1)/60.     
        #print('TIME SLICE')
        #print(f'{np.round(tslice[j],2)} min')
        print('Estimated remaining time for run in parts')
        print(f'{np.round(np.mean(tslice[:j+1])*(nslices-(j+1)),2)} min')


if __name__=='__main__':

    # folder = '/home/fcaporaso/cats/L768/'
    # with fits.open(folder+'l768_mg_octant_19219.fits') as f:
    #     g1_mask = np.abs(f[1].data.gamma1) < 10.0
    #     S = f[1].data[g1_mask]
    # sim = ['l768_gr_z04-07_for02-03_19304.fits','l768_mg_z04-07_for02-03_19260.fits']
    # voidcat = ['void_LCDM_09.dat', 'void_fR_09.dat']

    tin = time.time()
    main(
        lcat        = args.lens_cat, 
        sample      = args.sample,
        output_file = None,
        Rv_min      = args.Rv_min, 
        Rv_max      = args.Rv_max,
        z_min       = args.z_min, 
        z_max       = args.z_max,
        rho1_min    = args.rho1_min, 
        rho1_max    = args.rho1_max,
        rho2_min    = args.rho2_min, 
        rho2_max    = args.rho2_max,
        FLAG        = args.FLAG,
        RIN         = args.RIN, 
        ROUT        = args.ROUT,
        ndots       = args.ndots, 
        ncores      = args.ncores, 
        nk          = args.nk,
        addnoise    = False, 
    )
    # run_in_parts(
    #     args.RIN, 
    #     args.ROUT, 
    #     args.nslices,
    #     args.lens_cat, 
    #     args.sample,
    #     Rv_min=args.Rv_min, Rv_max=args.Rv_max, 
    #     rho1_min=args.rho1_min, rho1_max=args.rho1_max, 
    #     rho2_min=args.rho2_min, rho2_max=args.rho2_max, 
    #     z_min=args.z_min, z_max=args.z_max, 
    #     ndots=args.ndots, 
    #     ncores=args.ncores, 
    #     FLAG=args.FLAG
    # )
    tfin = time.time()
    print(f'TOTAL TIME: {np.round((tfin-tin)/60.,2)} min')
