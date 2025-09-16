from ctypes import Array
import numpy as np
from astropy.cosmology import LambdaCDM
from astropy.constants import G,c,M_sun,pc
from multiprocessing import Pool
from tqdm import tqdm

from funcs import eq2p2, lenscat_load, sourcecat_load


class Lensing:

    def __init__(self, source_args, cosmo_params, profile_args, binning='lin'):
        
        self.N : int      = profile_args['N']
        self.Nk : int     = profile_args['Nk']
        #self.ncores : int = profile_args['ncores']
        self.RIN : float  = profile_args['RIN']
        self.ROUT : float = profile_args['ROUT']

        if binning == 'log':
            self.bines = np.logspace(self.RIN, self.ROUT, self.N+1)
        else:
            self.bines = np.linspace(self.RIN, self.ROUT, self.N+1)

        self.cosmo = LambdaCDM(**cosmo_params)
        self.S = sourcecat_load(**source_args)

        ra_gal_rad  = np.deg2rad(self.S[0])
        dec_gal_rad = np.deg2rad(self.S[1])
        self.cos_ra_gal  = np.cos(ra_gal_rad)
        self.sin_ra_gal  = np.sin(ra_gal_rad)
        self.cos_dec_gal = np.cos(dec_gal_rad)
        self.sin_dec_gal = np.sin(dec_gal_rad)

    def sigma_crit(self, z_l, z_s):
        d_l  = self.cosmo.angular_diameter_distance(z_l).value*pc.value*1.0e6
        d_s  = self.cosmo.angular_diameter_distance(z_s).value
        d_ls = self.cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
        return (((c.value**2.0)/(4.0*np.pi*G.value*d_l))*(d_s/d_ls))*(pc.value**2/M_sun.value)

    def get_masked_data(self, psi, ra0, dec0, z0):
        '''
        usando la interseccion de la esfera con un plano,
        se obtienen los objetos dentro de un spherical cap de 
        radio angular psi
        '''
        ## VER CUENTAS DE ARCHIVO 'sphere_plane_cut.pdf'
        ra0_rad = np.deg2rad(ra0)
        dec0_rad = np.deg2rad(dec0)
        cos_dec0 = np.cos(dec0_rad)

        mask = (cos_dec0*np.cos(ra0_rad)*self.cos_dec_gal*self.cos_ra_gal
                 + cos_dec0*np.sin(ra0_rad)*self.cos_dec_gal*self.sin_ra_gal 
                 + np.sin(dec0_rad)*self.sin_dec_gal >= np.sqrt(1-np.sin(np.deg2rad(psi))**2))
        return self.S[:, mask&(self.S[2]>z0+0.1)]

    ## TODO :: descargar el catalogo de nuevo... no tengo guardados los valores de redshift observado (ie con vel peculiares ie RSD)
    def partial_profile(self, inp):
        
        Sigma_wsum    = np.zeros(self.N)
        DSigma_t_wsum = np.zeros(self.N)
        DSigma_x_wsum = np.zeros(self.N)
        N_inbin       = np.zeros(self.N)
        
        ra0, dec0, z0, Rv0 = inp

        # for ni in range(self.N):
        # adentro del for, mask depende de n... solo quiero las gx en un anillo

        DEGxMPC = self.cosmo.arcsec_per_kpc_proper(z0).to('deg/Mpc').value
        psi = DEGxMPC*self.ROUT*Rv0
        
        catdata = self.get_masked_data(psi, ra0, dec0, z0)
        sigma_c = self.sigma_crit(z0, catdata[2])/Rv0

        rads, theta = eq2p2(
            np.deg2rad(catdata[0]), np.deg2rad(catdata[1]),
            np.deg2rad(ra0), np.deg2rad(dec0)
        )

        ## TODO :: al descargar, cambiarle el signo
        e1 = -catdata[4]
        e2 = -catdata[5]

        #get tangential ellipticities 
        cos2t = np.cos(2.0*theta)
        sin2t = np.sin(2.0*theta)
        et = -(e1*cos2t+e2*sin2t)*sigma_c
        ex = (-e1*sin2t+e2*cos2t)*sigma_c
            
        #get convergence
        k  = catdata[3]*sigma_c

        r = (np.rad2deg(rads)/DEGxMPC)/Rv0
        #bines = self.binspace()
        dig = np.digitize(r, self.bines)

        for nbin in range(self.N):
            mbin = dig == nbin+1              
            Sigma_wsum[nbin]    = k[mbin].sum()
            DSigma_t_wsum[nbin] = et[mbin].sum()
            DSigma_x_wsum[nbin] = ex[mbin].sum()
            N_inbin[nbin]       = np.count_nonzero(mbin) ## idem mbin.sum(), faster
        
        return Sigma_wsum, DSigma_t_wsum, DSigma_x_wsum, N_inbin


def stacking(lens_args,cosmo_params,source_args,profile_args):
    N = profile_args['N']
    Nk = profile_args['Nk']

    N_inbin = np.zeros((Nk+1, N))
    DSigma_t_wsum = np.zeros((Nk+1, N))
    DSigma_x_wsum = np.zeros((Nk+1, N))

    L, K, nvoids = lenscat_load(**lens_args)
    print(f'Nvoids: {nvoids}', flush=True)
    vlen = Lensing(source_args=source_args, cosmo_params=cosmo_params, profile_args=profile_args)

    for i, Li in enumerate(tqdm(L)):
        num = len(Li)
        inp = np.array([Li.T[1], Li.T[2], Li.T[3], Li.T[0]]).T
        with Pool(processes=num) as pool:
            resmap = np.array(pool.map(vlen.partial_profile, inp))
            pool.close()
            pool.join()

        for j,r in enumerate(resmap):
            km = np.tile(K[i][j], (N,1)).T
            N_inbin += np.tile(r[-1], (Nk+1,1))*km
            Sigma_wsum += np.tile(r[0], (Nk+1,1))*km
            DSigma_t_wsum += np.tile(r[1], (Nk+1,1))*km
            DSigma_x_wsum += np.tile(r[2], (Nk+1,1))*km

    Sigma = Sigma_wsum/N_inbin
    DSigma_t = DSigma_t_wsum/N_inbin
    DSigma_x = DSigma_x_wsum/N_inbin

    return vlen.bines, Sigma, DSigma_t, DSigma_x 


if __name__ == '__main__':

    import time

    lens_name = 'voids_LCDM_09.dat'
    Rv_min = 10.0
    Rv_max = 11.0
    z_min = 0.2
    z_max = 0.22
    delta_min = -1.0 # void type
    delta_max = -0.1 # void type

    source_name = 'l768_gr_z04-07_for02-03_19304.fits'

    RIN = 0.1
    ROUT = 1.0
    N = 10
    Nk = 10
    ncores = 8

    lens_args = dict(
        name = lens_name,
        Rv_min = Rv_min,
        Rv_max = Rv_max,
        z_min = z_min,
        z_max = z_max,
        delta_min = delta_min, # void type
        delta_max = delta_max, # void type
        ncores = ncores,
        Nk = Nk,
        fullshape=False,
    )

    source_args = dict(
        name = source_name
    )

    profile_args = dict(
        RIN = RIN,
        ROUT = ROUT,
        N = N,
        Nk = Nk,
        ncores = ncores,
    )

    cosmo_params = dict(
        Om0 = 0.3089,
        Ode0 = 0.6911,
        H0 = 100.0
    )
    print('Start!')
    t1=time.time()
    l = Lensing(source_args=source_args, cosmo_params=cosmo_params, binning='lin')
    
    np.savetxt('test.dat', 
               l.run(lens_args=lens_args, profile_args=profile_args), 
               delimiter=','
               )
    
    print('End!')
    print(f'took {(time.time()-t1)/60.0} s')