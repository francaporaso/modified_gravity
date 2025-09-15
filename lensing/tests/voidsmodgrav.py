from ctypes import Array
import numpy as np
from astropy.cosmology import LambdaCDM
from astropy.constants import G,c,M_sun,pc
from multiprocessing import Pool

from funcs import eq2p2, lenscat_load

lens_name = 'voids_fR_09.dat',
Rv_min = 8.0,
Rv_max = 12.0,
z_min = 0.2,
z_max = 0.25,
delta_min = -1.0, # void type
delta_max = -0.1, # void type

source_name = 'l768_mg_z04-07_for02-03_19260.fits'

RIN = 0.01,
ROUT = 1.5,
N = 20,
NK = 100,
ncores = 32,


lens_args = dict(
    name = lens_name,
    Rv_min = Rv_min,
    Rv_max = Rv_max,
    z_min = z_min,
    z_max = z_max,
    delta_min = delta_min, # void type
    delta_max = delta_max, # void type
    ncores = ncores,
    nk = NK,
)

source_args = dict(
    name = source_name
)

profile_args = dict(
    RIN = RIN,
    ROUT = ROUT,
    N = N,
    NK = NK,
    ncores = ncores,
)

cosmo_params = dict(
    Om0 = 0.3089,
    Ode0 = 0.6911,
    h = 1.0
)

def sourcecat_load():
    pass

class Lensing:

    def __init__(self, source_args, cosmo_params) -> None:
        
        self.cosmo = LambdaCDM(**cosmo_params)
        self.S = sourcecat_load(**source_args)

        self.cosSra = np.cos(np.deg2rad(self.S.ra))
        self.sinSra = np.sin(np.deg2rad(self.S.ra))
        self.cosSdec = np.cos(np.deg2rad(self.S.dec))
        self.sinSdec = np.sin(np.deg2rad(self.S.dec))


    def run(self, lens_args, profile_args):
        
        self.N : int = profile_args['N']
        self.Nk : int = profile_args['Nk']
        self.ncores : int = profile_args['ncores']
        self.RIN : float = profile_args['RIN']
        self.ROUT : float = profile_args['ROUT']
        
        self.L, self.K, self.nvoids = lenscat_load(**lens_args)
        self.stacking()

    def sigma_crit(self, z_l, z_s):
        d_l = self.cosmo.angular_diameter_distance(z_l).value*pc*1.0e6
        d_s = self.cosmo.angular_diameter_distance(z_s).value
        d_ls = self.cosmo.angular_diameter_distance__z1z2(z_l, z_s).value
        return (((c.value**2.0)/(4.0*np.pi*G.value*d_l))*(d_s/d_ls))*(pc.value**2/M_sun.value)

    def get_masked_data(self, psi, ra0, dec0):
        '''
        usando la interseccion de la esfera con un plano,
        se obtienen los objetos dentro de un spherical cap de 
        radio angular psi
        '''
        ## VER CUENTAS DE ARCHIVO 'sphere_plane_cut.pdf'

        cosdec0 = np.cos(np.deg2rad(dec0))
        sindec0 = np.sin(np.deg2rad(dec0))
        cosra0 = np.cos(np.deg2rad(ra0))
        sinra0 = np.sin(np.deg2rad(ra0))

        mask = cosdec0*cosra0*self.cosSdec*self.cosSra + cosdec0*sinra0*self.cosSdec*self.sinSra + sindec0*self.sinSdec >= np.sqrt(1-np.sin(np.deg2rad(psi))**2)
        return self.S[mask]

    def partial_profile(self, inp):
        ## TODO :: descargar el catalogo de nuevo... no tengo guardados los valores de redshift observado (ie con vel peculiares ie RSD)
        
        SIGMAwsum    = np.zeros(self.N)
        DSIGMAwsum_T = np.zeros(self.N)
        DSIGMAwsum_X = np.zeros(self.N)
        N_inbin      = np.zeros(self.N)
        
        ra0, dec0, z0, Rv0 = inp

        # for ni in range(self.N):
        # adentro del for, mask depende de n... solo quiero las gx en un anillo

        DEGxMPC = self.cosmo.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
        psi = DEGxMPC*self.ROUT*Rv0
        
        catdata = self.get_masked_data(psi, ra0, dec0)
        sigma_c = self.sigma_crit(z0, catdata.observed_redshift_gal)/Rv0

        rads, theta = eq2p2(
            np.deg2rad(catdata.ra_gal), np.deg2rad(catdata.dec_gal),
            np.deg2rad(ra0), np.deg2rad(dec0)
        )

        ## TODO :: al descargar, cambiarle el signo... es más facil y se ahorra calculo
        e1 = -catdata.gamma1
        e2 = -catdata.gamma2

        #get tangential ellipticities 
        cos2t = np.cos(2.0*theta)
        sin2t = np.sin(2.0*theta)
        et = -(e1*cos2t+e2*sin2t)*sigma_c
        ex = (-e1*sin2t+e2*cos2t)*sigma_c
            
        #get convergence
        k  = catdata.kappa*sigma_c

        r = (np.rad2deg(rads)/DEGxMPC)/Rv0
        bines = np.linspace(self.RIN, self.ROUT, num = self.N+1)
        dig = np.digitize(r,bines)

        for nbin in range(self.N):
            mbin = dig == nbin+1              
            SIGMAwsum[nbin]    = k[mbin].sum()
            DSIGMAwsum_T[nbin] = et[mbin].sum()
            DSIGMAwsum_X[nbin] = ex[mbin].sum()
            N_inbin[nbin]      = np.count_nonzero(mbin) ## hace lo mismo q mbin.sum() pero más rápido
        
        return SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin

    def stacking(self):
        
        N_inbin = np.zeros((self.Nk+1, self.N))
        DSigma_t_wsum = np.zeros((self.Nk+1, self.N))
        DSigma_x_wsum = np.zeros((self.Nk+1, self.N))

        with Pool(processes=self.ncores) as pool:
            inp = np.array([self.L.ra, self.L.dec, self.L.redshift, self.L.Rv]).T

            resmap = np.array(pool.imap(self.partial_profile, inp))
            pool.close()
            pool.join()

        for i,r in enumerate(resmap):
            km = np.tile(self.K[i], (self.N,1)).T
            N_inbin += np.tile(r.N_inbin, (self.Nk+1,1))*km
            DSigma_t_wsum += np.tile(r.DSigma_t, (self.Nk+1,1))*km
            DSigma_x_wsum += np.tile(r.DSigma_x, (self.Nk+1,1))*km

        DSigma_t = DSigma_t_wsum/N_inbin
        DSigma_x = DSigma_x_wsum/N_inbin

        return DSigma_t, DSigma_x     

if __name__ == '__main__':
    print('a')
