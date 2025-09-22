import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.table import Table
import time

## Cuentas en drive 'IATE/sphere_plane_cut.pdf'
def get_masked_data(psi, ra0, dec0, z0):
    '''
    objects are selected by intersecting the sphere with a plane
    and keeping those inside the spherical cap.
    '''

    ra0_rad = np.deg2rad(ra0)
    dec0_rad = np.deg2rad(dec0)
    cos_dec0 = np.cos(dec0_rad)

    #mask_z = S['true_redshift_gal']>z0+0.1
    mask_field = (cos_dec0*np.cos(ra0_rad)*S['cos_dec_gal']*S['cos_ra_gal']
                + cos_dec0*np.sin(ra0_rad)*S['cos_dec_gal']*S['sin_ra_gal'] 
                + np.sin(dec0_rad)*S['sin_dec_gal'] >= np.sqrt(1-np.sin(np.deg2rad(psi))**2))
    
    return mask_field

#S = Table.read('/home/fcaporaso/cats/L768/l768_gr_z04-07_for02-3_w_trig_19304.fits', format='fits', memmap=True)
rng = np.random.default_rng(1)
N = 10_000_000

S = Table({
    'ra_gal':360.0*rng.random(N),
    'dec_gal':90.0*rng.random(N),
    'true_redshift_gal':np.zeros(N),
})

S['cos_ra_gal'] = np.cos(np.deg2rad(S['ra_gal']))
S['cos_dec_gal'] = np.cos(np.deg2rad(S['dec_gal']))
S['sin_ra_gal'] = np.sin(np.deg2rad(S['ra_gal']))
S['sin_dec_gal'] = np.sin(np.deg2rad(S['dec_gal']))

psi = 6.0 # deg
pad = 0.5 # deg
ra0 = 150.0 # deg
dec0 = 30.0 # deg

# === test 1
# === healpy
NSIDE = 2**6
print(f'{NSIDE=}')
S['pix'] = hp.ang2pix(NSIDE, S['ra_gal'], S['dec_gal'], lonlat=True)

t1 = time.time()

pix_idx = hp.query_disc(nside=NSIDE, vec=hp.ang2vec(ra0, dec0, lonlat=True), radius=np.deg2rad(psi+pad))
mask = np.isin(S['pix'], pix_idx)
t_hp = time.time() - t1

print(f'Healpy took {t_hp} s')

# === test 2
# === sphere+plane intersection

t1 = time.time()
mask2 = get_masked_data(psi, ra0, dec0, 0.0)
t_intrsc = time.time()- t1

print(f'Intersection took {t_intrsc} s')

print(f'Healpy is {t_intrsc/t_hp} times faster than intersect')

# === consistency check
# === are the masks similar?

plt.scatter(ra0, dec0, s=10, c='r')
#plt.scatter(S['ra_gal'], S['dec_gal'], s=1, alpha=0.3, c='dimgray')
plt.scatter(S['ra_gal'][mask2], S['dec_gal'][mask2], s=5, marker='s', alpha=0.5, facecolor='none', c='C1')
plt.scatter(S['ra_gal'][mask], S['dec_gal'][mask], s=2, alpha=0.5, c='C0')
# for p in pix_idx:
#     plt.scatter(S['ra_gal'][S['pix']==p], S['dec_gal'][S['pix']==p], s=5, alpha=0.5, c='C0')
plt.show()