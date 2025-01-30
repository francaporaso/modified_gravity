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

    def __init__(self,
                lens_cat = '', source_cat = '', sample = '',
                ncores = 2, n_runslices = 1,
                h = 1.0, Om0 = 0.3089, Ode0 = 0.6911,                
                Rv_min = 0.0, Rv_max = 50.0, z_min = 0.0, z_max = 10.0,
                rho1_min = -1.0, rho1_max = 0.0, rho2_min = -1.0, rho2_max = 100.0,
                flag = 2, octant = False,
                RIN = 0.01, ROUT = 5.0, ndots = 20, nk = 100, addnoise = False):
        
        # program arguments
        self.lens_cat: str = lens_cat
        self.source_cat: str = source_cat
        self.sample: str = sample
        self.ncores: int = ncores
        self.n_runslices: int = n_runslices

        # cosmology
        # self.h: float = h
        # self.Om0: float = Om0
        # self.Ode0: float = Ode0
        self.cosmo : LambdaCDM = LambdaCDM(H0=100*h, Om0=Om0, Ode0=Ode0)
        
        # lens arguments
        self.Rv_min: float = 0.0
        self.Rv_max: float = 50.0
        self.z_min: float = 0.0
        self.z_max: float = 10.0
        self.rho1_min: float = -1.0
        self.rho1_max: float = 0.0
        self.rho2_min: float = -1.0
        self.rho2_max: float = 100.0
        self.flag: float = 2.0
        self.octant: bool = False
        
        # profile arguments
        self.RIN: float = 0.01
        self.ROUT: float = 5.0
        self.ndots: int = 20
        self.nk: int = 100
        self.addnoise: bool = False

        # catalogs
        self.S: fits.HDUList|None = None

        self.L = None
        self.K = None
        self.nvoids: int = 0
    
    def load_cats(self):
        self.S = sourcecat_load(self.source_cat)
        self.L, self.K, self.nvoids = lenscat_load(self.lens_cat,
            self.Rv_min, self.Rv_max, self.z_min, self.z_max, self.rho1_min, self.rho1_max, self.rho2_min, self.rho2_max, self.flag,
            self.ncores, self.octant, self.nk)

    def SigmaCrit(self, zl, zs):
        
        dl  = self.cosmo.angular_diameter_distance(zl).value           # observer-lens dist
        Dl = dl*1.e6*pc                                                # in m
        ds  = self.cosmo.angular_diameter_distance(zs).value           # observer-source dist
        dls = self.cosmo.angular_diameter_distance_z1z2(zl, zs).value  # lens-source dist
                    
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
            mbin = (dig == nbin+1)
            SIGMAwsum[nbin]    = k[mbin].sum()
            DSIGMAwsum_T[nbin] = et[mbin].sum()
            DSIGMAwsum_X[nbin] = ex[mbin].sum()
            N_inbin[nbin] = np.count_nonzero(mbin) ## hace lo mismo q mbin.sum() pero más rápido
        
        return SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin
    
    def stack(self):
        print(''.center(14,"="))
        print('RMIN: '.ljust(7,'.'), f' {self.RIN}'.rjust(7,'.'), sep='')
        print('RMAX: '.ljust(7,'.'), f' {self.ROUT}'.rjust(7,'.'),sep='')
        print('N: '.ljust(7,'.'), f' {self.ndots}'.rjust(7,'.'),sep='')
        print(''.center(14,"="))

        bines = np.linspace(self.RIN, self.ROUT, self.ndots+1)
        R = (bines[:-1] + np.diff(bines)*0.5)
        # WHERE THE SUMS ARE GOING TO BE SAVED
        
        Ninbin = np.zeros((self.nk+1,self.ndots))    
        SIGMAwsum    = np.zeros((self.nk+1,self.ndots)) 
        DSIGMAwsum_T = np.zeros((self.nk+1,self.ndots)) 
        DSIGMAwsum_X = np.zeros((self.nk+1,self.ndots))
                            
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


# ============================================================================== FUNCTIONS ===

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

def sourcecat_load(source_cat: str) -> fits.HDUList:
    folder = '/home/fcaporaso/cats/L768/'
    with fits.open(folder+source_cat) as f:
        mask = np.abs(f[1].data.gamma1) < 10.0
        S = f[1].data[mask]

    return S

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

# ==============================================================================  MAIN ===
def main(lens_cat = '',
        source_cat = '',
        sample = '',
        ncores = 2,
        n_runslices= 1,
        h = 1.0, Om0 = 0.3089, Ode0 = 0.6911,
        Rv_min = 15.0, Rv_max = 20.0,
        z_min = 0.2, z_max = 0.3,
        rho1_min = -1.0, rho1_max = 0.0,
        rho2_min = -1.0, rho2_max = 100.0,
        flag = 2.0,
        octant = False,
        RIN = 0.5, ROUT = 5.0,
        ndots= 22,
        nk= 100,
        addnoise = False):

    ## Runing program
    tini = time.time()
    
    v = VoidLensing(
        lens_cat = lens_cat,
        source_cat = source_cat,
        sample = sample,
        ncores = ncores,
        n_runslices= n_runslices,
        h = h, Om0 = Om0, Ode0 = Ode0,
        Rv_min = Rv_min, Rv_max = Rv_max,
        z_min = z_min, z_max = z_max,
        rho1_min = rho1_min, rho1_max = rho1_max,
        rho2_min = rho2_min, rho2_max = rho2_max,
        flag = flag,
        octant = octant,
        RIN = RIN, ROUT = ROUT,
        ndots= ndots,
        nk= nk,
        addnoise = addnoise,
    )

    v.load_cats()

    # program arguments
    print(' Program arguments '.center(30,"="))
    print('Lens catalog: '.ljust(15,'.'), f' {lens_cat}'.rjust(15,'.'), sep='')
    print('Sources catalog: '.ljust(15,'.'), f' {source_cat}'.rjust(15,'.'),sep='')
    print('Output name: '.ljust(15,'.'), f' {sample}'.rjust(15,'.'),sep='')
    print('N of cores: '.ljust(15,'.'), f' {ncores}'.rjust(15,'.'),sep='')
    print('N of slices: '.ljust(15,'.'), f' {n_runslices}'.rjust(15,'.'),sep='')

    # cosmology
    print(' Cosmo params '.center(30,"="))
    print('h: '.ljust(15,'.'), f' {h}'.rjust(15,'.'), sep='')
    print('Om0: '.ljust(15,'.'), f' {Om0}'.rjust(15,'.'),sep='')
    print('Ode0: '.ljust(15,'.'), f' {Ode0}'.rjust(15,'.'),sep='')
    
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
    print('Octante: '.ljust(15,'.'), f' {octant}'.rjust(15,'.'),sep='')
    print('N voids: '.ljust(15,'.'), f' {v.nvoids}'.rjust(15,'.'),sep='')
    
    # profile arguments
    print(' Profile arguments '.center(30,"="))
    print('RMIN: '.ljust(15,'.'), f' {RIN}'.rjust(15,'.'), sep='')
    print('RMAX: '.ljust(15,'.'), f' {ROUT}'.rjust(15,'.'),sep='')
    print('N: '.ljust(15,'.'), f' {ndots}'.rjust(15,'.'),sep='')
    print('N jackknife: '.ljust(15,'.'), f' {nk}'.rjust(15,'.'),sep='')
    print('Shape Noise: '.ljust(15,'.'), f' {addnoise}'.rjust(15,'.'),sep='')
    

    if bool(n_runslices-1):
        R, Sigma, DSigma_T, DSigma_X, Ninbin = v.stack()

        covS = cov_matrix(Sigma[1:,:])
        covDSt = cov_matrix(DSigma_T[1:,:])
        covDSx = cov_matrix(DSigma_X[1:,:])

    else:
        cuts = np.round(np.linspace(RIN, ROUT, n_runslices+1),2)
        R = np.array([])
        Sigma = np.array([])
        DSigma_T = np.array([])
        DSigma_X = np.array([])
        Ninbin = np.array([])

        for j in np.arange(n_runslices):
            print(f'RUN {j+1} out of {n_runslices} slices')
            
            v.RIN, v.ROUT = cuts[j], cuts[j+1]
            v.ndots = ndots//n_runslices
            res_parcial = v.stack()
            R = np.append(R, res_parcial[0])
            Sigma = np.append(Sigma, res_parcial[1])
            DSigma_T = np.append(DSigma_T, res_parcial[2])
            DSigma_X = np.append(DSigma_X, res_parcial[3])
            Ninbin = np.append(Ninbin, res_parcial[4])
        
        Sigma = Sigma.rehsape(nk+1,ndots)
        DSigma_T = DSigma_T.rehsape(nk+1,ndots)
        DSigma_X = DSigma_X.rehsape(nk+1,ndots)
        Ninbin = Ninbin.rehsape(nk+1,ndots)

        covS = cov_matrix(Sigma[1:,:])
        covDSt = cov_matrix(DSigma_T[1:,:])
        covDSx = cov_matrix(DSigma_X[1:,:])

    try:
        os.mkdir('results/')
    except FileExistsError:
        pass
    
    if not output_file:
        output_file = f'results/'
    # Defining radial bins
    
    # AVERAGE VOID PARAMETERS AND SAVE IT IN HEADER
    zmean    = np.concatenate([v.L[i][:,3] for i in range(len(v.L))]).mean()
    rvmean   = np.concatenate([v.L[i][:,0] for i in range(len(v.L))]).mean()
    rho2mean = np.concatenate([v.L[i][:,8] for i in range(len(v.L))]).mean()
    
    head = fits.Header()
    head.append(('nvoids',int(v.nvoids)))
    head.append(('cat',lens_cat))
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
                
    print(f'Partial time: {np.round((time.time()-tini)/60. , 3)} mins')


if __name__=='__main__':

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
    
    tin = time.time()
    
    main(
        lens_cat = args.lens_cat,
        source_cat = args.source_cat,
        sample = args.sample,
        ncores = args.ncores,
        n_runslices= args.n_runslices,
        h = args.h_cosmo, Om0 = args.Om0, Ode0 = args.Ode0,
        Rv_min = args.Rv_min, Rv_max = args.Rv_max,
        z_min = args.z_min, z_max = args.z_max,
        rho1_min = args.rho1_min, rho1_max = args.rho1_max,
        rho2_min = args.rho2_min, rho2_max = args.rho2_max,
        flag = args.flag,
        octant = args.octant,
        RIN = args.RIN, ROUT = args.ROUT,
        ndots= args.ndots,
        nk= args.nk,
        addnoise = args.addnoise,
    )

    tfin = time.time()
    print(f'TOTAL TIME: {np.round((tfin-tin)/60.,2)} min')
