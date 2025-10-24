import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open('/home/fcaporaso/cats/L768/l768_gr_z02-04_for01-02_19532.fits') as f:
    z_gal = f[1].data['true_redshift_gal']

size_rand = 1000

y, xedge = np.histogram(z_gal, bins=25)
x = 0.5*(xedge[:-1]+xedge[1:])
rng = np.random.default_rng(0)
z_rand = rng.uniform(z_gal.min(), z_gal.max(), size_rand)
p = np.polynomial.Polynomial.fit(x, y, 3)
poly = p(z_rand)
peso = poly/np.sum(poly)
z_rand = rng.choice(z_rand, size_rand, replace=True, p=peso)

plt.stairs(y, xedge, label='True')
plt.hist(z_rand, 25, label='Rand', histtype='step')
plt.legend()
plt.show()