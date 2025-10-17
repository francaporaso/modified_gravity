import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.table import Table

import sys
sys.path.append('/home/fcaporaso/modified_gravity/')
from lensing.funcs import lenscat_load

def ang2xyz(ra, dec, redshift):
    comdist = comoving_distance(redshift)
    x = comdist * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    y = comdist * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    z = comdist * np.sin(np.deg2rad(dec))
    return x,y,z

def comoving_distance(z):
    return cosmo.comoving_distance(z).value.astype(np.float32) # Mpc/h

def make_randoms(ra, dec, redshift,
                 size_random = 100):

    rng = np.random.default_rng(0)    
    
    sindec_rand = rng.uniform(
        np.sin(np.deg2rad(dec.min())), 
        np.sin(np.deg2rad(dec.max())), 
        size_random
    ).astype(np.float32)
    dec_rand = np.rad2deg(np.arcsin(sindec_rand))
    
    ra_rand  = rng.uniform(
        ra.min(), 
        ra.max(), 
        size_random
    ).astype(np.float32)

    y, x  = np.histogram(redshift, 25)
    x  = 0.5*(x[1:]+x[:-1]) # center of bin
    
    ## segun numpy mejor usar la clase numpy.polynomial.Polynomial instead of np.poly1d
    poly = np.polyfit(x, y, 3)
    zr = rng.uniform(
        redshift.min(), 
        redshift.max(), 
        size_random
    ).astype(np.float32)
    poly_y = np.poly1d(poly)(zr)
    poly_y[poly_y<0] = 0.
    peso = poly_y/sum(poly_y)
    z_rand = rng.choice(zr, size_random, replace=True, p=peso)

    return np.array([ra_rand, dec_rand, z_rand])

class Catalogs:
    def __init__(self, lens_args, source_args, do_rands=True):
        
        ## [0]:rv, [1]:ra, [2]:dec, [3]:redshift
        L, _, self.nvoids = lenscat_load(**lens_args)
        assert self.nvoids != 0, 'No void found with those parameters!'
        self.lenses = Table(L.T, names=['Rv', 'ra', 'dec', 'redshift'])
        self.lenses['dcom'] = comoving_distance(self.lenses['redshift'])
        
        #self.sources = pd.read_parquet(source_args).sample(frac=1.0, random_state=1).to_numpy().T
        try:
            self.sources = Table.read(source_args['name'], format='fits', memmap=True)
        except FileNotFoundError:
            self.sources = Table.read('/home/fcaporaso/cats/L768/'+source_args['name'], format='fits', memmap=True)
        mask = (self.sources['true_redshift_gal'] < lens_args["z_max"]+0.01) & (self.sources['true_redshift_gal'] >= lens_args["z_min"]-0.01)
        self.sources = self.sources[mask]
        self.ngals = len(self.sources)
        assert self.ngals != 0, 'No tracer found with those parameters!'
        
        if do_rands:
            print(' Making randoms '.center(60, '.'), flush=True)

            if 'dcom_gal' not in self.sources.columns:
                self.sources['dcom_gal'] = comoving_distance(self.sources['true_redshift_gal'])

            self.random_lenses = Table(
                make_randoms(
                    self.lenses['ra'],
                    self.lenses['dec'],
                    self.lenses['dcom'],
                    size_random=self.nvoids*10
                ),
                names=['ra','dec','dcom']
            )

            self.random_sources = Table(
                make_randoms(
                    self.sources['ra_gal'],
                    self.sources['dec_gal'],
                    self.sources['dcom_gal'], 
                    size_random=self.ngals*10
                ),
                names=['ra_gal', 'dec_gal', 'dcom_gal']
            )

        else:
            pass
        
        print('N voids: '.ljust(20,'.'), f' {self.nvoids:,}'.rjust(20,'.'),sep='',flush=True)
        print('N sources: '.ljust(20,'.'), f' {self.ngals:,}'.rjust(20,'.'), sep='', flush=True)
        print('N rand voids: '.ljust(20,'.'), f' {self.nvoids*10:,}'.rjust(20,'.'),sep='',flush=True)
        print('N rand sources: '.ljust(20,'.'), f' {self.ngals*10:,}'.rjust(20,'.'),sep='',flush=True)


if __name__ == '__main__':
    print('Making randoms beforehand!')
    print('not implemented...')