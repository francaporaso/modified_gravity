import numpy as np

def make_randoms(ra, dec, redshift,
                 size_random = 100):

    rng = np.random.default_rng(0)    
    
    sindec_rand = rng.uniform(
        np.sin(np.deg2rad(dec.min())), 
        np.sin(np.deg2rad(dec.max())), 
        size_random
    )
    dec_rand = np.rad2deg(np.arcsin(sindec_rand))
    
    ra_rand  = rng.uniform(
        ra.min(), 
        ra.max(), 
        size_random
    )

    y, x  = np.histogram(redshift, 25)
    x  = 0.5*(x[1:]+x[:-1]) # center of bin
    
    ## segun numpy mejor usar la clase numpy.polynomial.Polynomial instead of np.poly1d
    poly = np.polyfit(x, y, 3)
    zr = rng.uniform(redshift.min(), redshift.max(), size_random)
    poly_y = np.poly1d(poly)(zr)
    poly_y[poly_y<0] = 0.
    peso = poly_y/sum(poly_y)
    z_rand = rng.choice(zr, size_random, replace=True, p=peso)

    return np.array([ra_rand, dec_rand, z_rand])
