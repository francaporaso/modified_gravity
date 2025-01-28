from argparse import ArgumentParser
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

#parameters
cvel = c.value;    # Speed of light (m.s-1)
G    = G.value;    # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value    # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

class VoidLensing:

    def __init__(self):
        
        # program arguments
        self.sample: str = ''
        self.lens_cat: str = ''
        self.source_cat: str = ''
        self.ncores: int = 2

        # cosmology
        self.h: int = 1.0
        self.Om0: int = 0.3089
        self.OdeO: int = 0.6911
        self.cosmo : LambdaCDM = LambdaCDM(H0=100*self.h, Om0=self.Om0, Ode0=self.Ode0)
        
        # lens arguments
        self.Rv_min: float = 0.0
        self.Rv_max: float = 50.0
        self.z_min: float = 0.0
        self.z_max: float = 10.0
        self.rho1_min: float = -1.0
        self.rho1_max: float = 0.0
        self.rho2_min: float = -1.0
        self.rho2_max: float = 100.0
        self.flag: int = 2
        # self.split: bool = bool(self.ncores-1)
        self.nslices: int = 1
        self.octant: bool = False
        
        # profile arguments
        self.RIN: float = 0.01
        self.ROUT: float = 5.0
        self.ndots: int = 20
        self.nk: int = 100
        self.addnoise: bool = False

        # catalogs
        self.__S: fits.HDUList = sourcecat_load(self.source_cat)

        self.L, self.K, self.nvoids = lenscat_load(
            self.lens_cat,
            self.Rv_min, self.Rv_max, self.z_min, self.z_max, self.rho1_min, self.rho1_max, self.rho2_min, self.rho2_max, self.flag,
            self.split, self.ncores, self.octant, self.nk
        )

    def SigmaCrit(self, zl, zs):
        
        dl  = self.cosmo.angular_diameter_distance(zl).value
        Dl = dl*1.e6*pc #en m
        ds  = self.cosmo.angular_diameter_distance(zs).value           # dist ang diam de la fuente
        dls = self.cosmo.angular_diameter_distance_z1z2(zl, zs).value  # dist ang diam entre fuente y lente
                    
        BETA_array = dls / ds

        return (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)

    def partial_profile(self,
                        minput):

        RA0, DEC0, Z, Rv = minput

        DEGxMPC = self.cosmo.arcsec_per_kpc_proper(Z).to('deg/Mpc').value
        delta = DEGxMPC*(self.ROUT*Rv)

        ## great-circle separation of sources from void centre
        pos_angles = np.arange(0,360,90)*u.deg
        c1 = SkyCoord(RA0, DEC0, unit='deg')
        c2 = c1.directional_offset_by(pos_angles, delta*u.deg)
        mask = (self.__S.true_redshift_gal > (Z+0.1))&(
                self.__S.dec_gal < c2[0].dec.deg)&(self.__S.dec_gal > c2[2].dec.deg)&(
                self.__S.ra_gal < c2[1].ra.deg)&(self.__S.ra_gal > c2[3].ra.deg)

        ## solid angle sep with maria_func
        ## using in case the other mask fails
        if mask.sum() == 0:
            print('Fail for',RA0,DEC0)
            sep = ang_sep(
                    np.deg2rad(RA0), np.deg2rad(DEC0),
                    np.deg2rad(self.__S.ra_gal), np.deg2rad(self.__S.dec_gal)
            )
            mask = (sep < np.deg2rad(delta))&(self.__S.true_redshift_gal >(Z+0.1))
            assert mask.sum() != 0

        catdata = self.__S[mask]

        sigma_c = self.SigmaCrit(Z, catdata.true_redshift_gal)
        
        rads, theta, *_ = eq2p2(
            np.deg2rad(catdata.ra_gal), np.deg2rad(catdata.dec_gal),
            np.deg2rad(RA0), np.deg2rad(DEC0)
        )
                            
        e1 = catdata.gamma1
        e2 = -1.*catdata.gamma2
        # Add shape noise due to intrisic galaxy shapes        
        if self.addnoise:
            es1 = -1.*catdata.defl1
            es2 = catdata.defl2
            e1 += es1
            e2 += es2
        
        #get t,x ellipticities 
        cos2t = np.cos(2*theta)
        sin2t = np.sin(2*theta)
        et = (-e1*cos2t-e2*sin2t)*sigma_c/Rv
        ex = (-e1*sin2t+e2*cos2t)*sigma_c/Rv
            
        #get convergence
        k  = catdata.kappa*sigma_c/Rv
        r = (np.rad2deg(rads)/DEGxMPC)/Rv

        bines = np.linspace(self.RIN,self.ROUT,num=self.ndots+1)
        dig = np.digitize(r,bines)
                
        SIGMAwsum    = np.zeros(self.ndots)
        DSIGMAwsum_T = np.zeros(self.ndots)
        DSIGMAwsum_X = np.zeros(self.ndots)
        N_inbin      = np.zeros(self.ndots)
                                            
        for nbin in range(self.ndots):
            mbin = dig == nbin+1              
            SIGMAwsum[nbin]    = k[mbin].sum()
            DSIGMAwsum_T[nbin] = et[mbin].sum()
            DSIGMAwsum_X[nbin] = ex[mbin].sum()
            N_inbin[nbin]      = np.count_nonzero(mbin) ## hace lo mismo q mbin.sum() pero más rápido
        
        return SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin
    
    def stack(self):
        
        bines = np.linspace(self.RIN,self.ROUT,num=self.ndots+1)
        R = (bines[:-1] + np.diff(bines)*0.5)
        # WHERE THE SUMS ARE GOING TO BE SAVED
        
        Ninbin = np.zeros((self.nk+1,self.ndots))    
        SIGMAwsum    = np.zeros((self.nk+1,self.ndots)) 
        DSIGMAwsum_T = np.zeros((self.nk+1,self.ndots)) 
        DSIGMAwsum_X = np.zeros((self.nk+1,self.ndots))
                    
        print(f'Saved in {self.sample}.fits')
        
        for i, Li in enumerate(tqdm(self.L)):
            num = len(Li)
            
            if num == 1:
                entrada = [
                    Li[1], Li[2], Li[3], Li[0],
                ]                
                resmap = np.array([self.partial_profile(entrada)])

            else:
                entrada = np.array([
                    Li.T[1],Li.T[2],Li.T[3],Li.T[0],
                ]).T
                with Pool(processes=num) as pool:
                    resmap = np.array(pool.map(self.partial_profile,entrada))
                    pool.close()
                    pool.join()
            
            for j, res in enumerate(resmap):
                km = np.tile(self.K[i][j],(self.ndots,1)).T
                                    
                SIGMAwsum    += np.tile(res[0],(self.nk+1,1))*km
                DSIGMAwsum_T += np.tile(res[1],(self.nk+1,1))*km
                DSIGMAwsum_X += np.tile(res[2],(self.nk+1,1))*km
                Ninbin += np.tile(res[3],(self.nk+1,1))*km

        # COMPUTING PROFILE        
        Ninbin[DSIGMAwsum_T == 0] = 1.0
                
        Sigma     = (SIGMAwsum/Ninbin)
        DSigma_T  = (DSIGMAwsum_T/Ninbin)
        DSigma_X  = (DSIGMAwsum_X/Ninbin)

        return R, Sigma, DSigma_T, DSigma_X, Ninbin



def lenscat_load(lens_cat,
                 Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max, flag,
                 ncores:int, octant:bool, nk:int):

    ## 0:Rv, 1:ra, 2:dec, 3:z, 4:xv, 5:yv, 6:zv, 7:rho1, 8:rho2, 9:logp, 10:diff CdM y CdV, 11:flag
    ## CdM: centro de masa
    ## CdV: centro del void
    L = np.loadtxt("/home/fcaporaso/cats/L768/"+lens_cat).T

    if octant:
        # selecciono los void en un octante
        eps = 7.0 # grados de padding entre borde y voids seleccionados
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

    if bool(ncores-1):
        if ncores > nvoids:
            ncores = nvoids
        lbins = int(round(nvoids/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < nvoids)]
        L = np.split(L.T, slices)
        K = np.split(K.T, slices)

    return L, K, nvoids

def sourcecat_load(source_cat):
    folder = '/home/fcaporaso/cats/L768/'
    with fits.open(folder+source_cat) as f:
        mask = np.abs(f[1].data.gamma1) < 10.0
        S = f[1].data[mask]

    return S

def main(lcat, sample='pru', output_file=None,
         Rv_min=0., Rv_max=50.,
         rho1_min=-1., rho1_max=0.,
         rho2_min=-1., rho2_max=100.,
         z_min = 0.1, z_max = 1.0,
         RIN = .05, ROUT =5.,
         ndots= 40, ncores=10, nk=100,
         addnoise = False, FLAG = 2.):
        
    tini = time.time()
    
    print(f'Voids catalog {lcat}')
    print(f'Sample {sample}')
    print(f'RIN : {RIN}')
    print(f'ROUT: {ROUT}')
    print(f'ndots: {ndots}')
    print('Selecting voids with:')
    print(f'{Rv_min}   <=  Rv  < {Rv_max}')
    print(f'{z_min}    <=  Z   < {z_max}')
    print(f'{rho1_min}  <= rho1 < {rho1_max}')
    print(f'{rho2_min}  <= rho2 < {rho2_max}')
        
    if addnoise:
        print('ADDING SHAPE NOISE')
    
    #reading Lens catalog
    L, K, nvoids = lenscat_load(
        Rv_min, Rv_max, z_min, z_max, rho1_min, rho1_max, rho2_min, rho2_max,
        flag=FLAG, lensname=lcat, split=True, NSPLITS=ncores, nk=nk, octant=True,
    )

    print(f'Nvoids {nvoids}')
    print(f'CORRIENDO EN {ncores} CORES')
    print(f'Profile has {ndots} bins')
    print(f'from {RIN} Rv to {ROUT} Rv')
    try:
        os.mkdir('results/')
    except FileExistsError:
        pass
    
    if not output_file:
        output_file = f'results/'
    # Defining radial bins
    
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

    # args = {
    #     '-sample':'pru',
    #     '-lens_cat':'voids_LCDM_09.dat',
    #     '-source_cat':'l768_gr_octant_19218.fits',
    #     '-Rv_min':0.,
    #     '-Rv_max':50.,
    #     '-rho1_min':-1.,
    #     '-rho1_max':1.,
    #     '-rho2_min':-1.,
    #     '-rho2_max':100.,
    #     '-FLAG':2.,
    #     '-z_min':0.1,
    #     '-z_max':0.5,
    #     '-addnoise':False,
    #     '-RIN':0.05,
    #     '-ROUT':5.,
    #     '-ndots':40,
    #     '-ncores':10,
    #     '-nk':100,
    #     '-nslices':1,
    # }

    # parser = ArgumentParser()
    # for key,val in args.items():
    #     parser.add_argument(key, action='store',dest=key[1:],default=val,type=type(val))
    # args = parser.parse_args()

