from argparse import ArgumentParser
import numpy as np
from astropy.io import fits
from astropy.cosmology import LambdaCDM

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

cosmo = LambdaCDM(H0=100*args.h_cosmo, Om0=args.Om0, Ode0=args.Ode0)

## TODO puede que sea más efficiente simplemente pasando los maximos y minimos
## para z no funcionaría xq tiene q interpolar...
def make_randoms(data,
                 size_random = 100):
    
    print('Making randoms...')
    ra, dec, z = data['ra'], data['dec'], data['z']

    dec = np.deg2rad(dec)
    sindec_rand = np.random.uniform(np.sin(dec.min()), np.sin(dec.max()), size_random)
    dec_rand = np.arcsin(sindec_rand)*(180/np.pi)
    ra_rand  = np.random.uniform(ra.min(), ra.max(), size_random)

    y,xbins  = np.histogram(z, 25)
    x  = xbins[:-1]+0.5*np.diff(xbins)
    n = 3
    poly = np.polyfit(x,y,n)
    zr = np.random.uniform(z.min(),z.max(),1_000_000)
    poly_y = np.poly1d(poly)(zr)
    poly_y[poly_y<0] = 0.
    peso = poly_y/sum(poly_y)
    z_rand = np.random.choice(zr,size_random,replace=True,p=peso)

    randoms = {'ra': ra_rand, 'dec': dec_rand, 'z':z_rand}

    if len(data.keys()) == 4:
        rv = data['rv']
        y,xbins  = np.histogram(rv, 25)
        x  = xbins[:-1] + 0.5*np.diff(xbins)
        n = 3
        poly = np.polyfit(x,y,n)
        rvr = np.random.uniform(rv.min(), rv.max(), 1_000_000)
        poly_y = np.poly1d(poly)(rvr)
        poly_y[poly_y<0] = 0.
        peso = poly_y/sum(poly_y)
        rv_rand = np.random.choice(rvr, size_random, replace=True, p=peso)
    
        randoms['rv'] = rv_rand

    print('Wii randoms!')
    return randoms

def ang2xyz(ra, dec, redshift,
            cosmo=cosmo):

    comdist = cosmo.comoving_distance(redshift).value #Mpc; Mpc/h si h=1
    x = comdist * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    y = comdist * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    z = comdist * np.sin(np.deg2rad(dec))

    return x,y,z

def compute_xi_2d(positions, random_positions,
                  npi = 16, nbins = 12,
                  rmin = 0.1, rmax = 200., pi_max = 60.,
                  NPatches = 16, slop = 0.,
                  cosmo = cosmo, ncores = 4):

    ## Auxiliary functions to compute the covariance
    def func(corrs):
        return corrs[0].weight*0.5
    
    def func2(corrs):
        return corrs[0].weight
    
    """ Compute the galaxy-shape correlation in 3D. """

    # arrays to store the output
    r         = np.zeros(nbins)
    mean_r    = np.zeros(nbins)
    mean_logr = np.zeros(nbins)

    xi    = np.zeros((npi, nbins))
    xi_jk = np.zeros((NPatches, npi, nbins))
    dd_jk = np.zeros_like(xi_jk)
    dr_jk = np.zeros_like(xi_jk)
    rr_jk = np.zeros_like(xi_jk)

    d_p  = cosmo.comoving_distance(positions['z']).value
    d_r  = cosmo.comoving_distance(random_positions['z']).value

    print('Loading catalogs...')
    
    pcat = treecorr.Catalog(ra=positions['ra'], dec=positions['dec'],
                             r=d_p, npatch = NPatches,
                             ra_units='deg', dec_units='deg')

    rcat = treecorr.Catalog(ra=random_positions['ra'], dec=random_positions['dec'],
                             r=d_r, npatch = NPatches,
                             patch_centers = pcat.patch_centers,
                             ra_units='deg', dec_units='deg')

    Nd = pcat.sumw
    Nr = rcat.sumw
    NNpairs = (Nd*(Nd - 1))/2.
    RRpairs = (Nr*(Nr - 1))/2.
    NRpairs = (Nd*Nr)

    f0 = RRpairs/NNpairs
    f1 = RRpairs/NRpairs

    Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
    pibins = zip(Pi[:-1],Pi[1:])

    # now loop over Pi bins, and compute w(r_p | Pi)
    #bar = progressbar.ProgressBar(maxval=npi-1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
    print('Calcualting correlation...')
    for p,(plow,phigh) in enumerate(tqdm(pibins)):

        #bar.update(p)
        dd = treecorr.NNCorrelation(nbins=nbins, min_sep=rmin, max_sep=rmax,
                                    min_rpar=plow, max_rpar=phigh,
                                    bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')
        
        # dr = treecorr.NNCorrelation(nbins=nbins, min_sep=rmin, max_sep=rmax,
        #                             min_rpar=plow, max_rpar=phigh,
        #                             bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')
        
        rr = treecorr.NNCorrelation(nbins=nbins, min_sep=rmin, max_sep=rmax,
                                    min_rpar=plow, max_rpar=phigh,
                                    bin_slop=slop, brute = False, verbose=0, var_method = 'jackknife')

        dd.process(pcat,pcat, metric='Rperp', num_threads = ncores)
        # dr.process(pcat,rcat, metric='Rperp', num_threads = ncores)
        rr.process(rcat,rcat, metric='Rperp', num_threads = ncores)

        r[:] = np.copy(dd.rnom)
        mean_r[:] = np.copy(dd.meanr)
        mean_logr[:] = np.copy(dd.meanlogr)

        # xi[p, :] = (dd.weight*0.5*f0 - (2.*dr.weight)*f1 + rr.weight*0.5) / (rr.weight*0.5)

        xi[p,:], _ = dd.calculateXi(rr=rr)
        
        #Here I compute the variance
        dd_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([dd], 'jackknife', func = func)
        # dr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([dr], 'jackknife', func = func2)
        rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func)

        # dd.finalize()
        # dr.finalize()
        # rr.finalize()

    for i in range(NPatches):

        swd = np.sum(pcat.w[~(pcat.patch == i)])
        swr = np.sum(rcat.w[~(rcat.patch == i)])

        NNpairs_JK = (swd*(swd - 1))/2.
        RRpairs_JK = (swr*(swr - 1))/2.
        NRpairs_JK = (swd*swr)

        xi_jk[i, :, :] = (dd_jk[i, :, :]/NNpairs_JK - (2.*dr_jk[i, :, :])/NRpairs_JK + rr_jk[i, :, :]/RRpairs_JK) / (rr_jk[i, :, :]/RRpairs_JK)

    xi[np.isinf(xi)] = 0. #It sets to 0 the values of xi_gp that are infinite
    xi[np.isnan(xi)] = 0. #It sets to 0 the values of xi_gp that are null

    xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5

    return r, mean_r, mean_logr, xPi, xi, xi_jk


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from plots_vgcf import *
    from scipy.stats import skewnorm

    N = 1_000
    z0,z1 = 0., 0.1

    with fits.open('/home/franco/FAMAF/Lensing/cats/MICE/mice18917.fits') as f:
        z_gal = f[1].data.z_cgal
        
        m_z = (z_gal < z1) & (z_gal >= z0)
        ra  = f[1].data.ra_gal[m_z]
        dec = f[1].data.dec_gal[m_z]
        z_gal = z_gal[m_z]

    if (s := m_z.sum()) < N:
        N = s
    print(N)
    
    ang_pos = {'ra':ra[:N], 'dec':dec[:N], 'z':z_gal[:N]}
    # xyz_pos = ang2xyz(*ang_pos.values())
    # rands_ang = make_randoms(*ang_pos.values(),size_randoms=N)
    # rands_xyz = ang2xyz(*rands_ang.values())
    # mask = (np.abs(rands_box[2]) < 10)
