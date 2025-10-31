from astropy.table import Table
import numpy as np
import healpy as hp
from collections import defaultdict
from time import time
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

NSIDE = 64
g = None
d = None

def create_mock_cat(N=100):
    
    global g, d 

    g = Table({
        'idx':np.arange(N, dtype=int),
        'ra':rng.uniform(0.0, 359.9, N).astype(np.float32),
        'dec':rng.uniform(-90.0, 90.0, N).astype(np.float32),
        'z':rng.uniform(0.1, 0.2, N).astype(np.float32),
    })

    g['pix'] = hp.ang2pix(NSIDE, g['ra'], g['dec'], lonlat=True)
    g.sort(['pix'])

    d = defaultdict(list)
    for i in range(N):
        d[g['pix'][i]].append(g['idx'][i])

def get_masked_data(ra0=150.0, dec0=10.0, psi=10.0):

    query_pix_id = hp.query_disc(
        NSIDE, 
        vec=hp.ang2vec(
            ra0, dec0, lonlat=True
        ),
        radius=np.deg2rad(psi)
    )

    query_idx = np.concatenate([d[p] for p in query_pix_id])
    return g[query_idx]

def get_masked_data_lensing(ra0=150.0, dec0=10.0, psi=10.0):

    query_pix_id = hp.query_disc(
        NSIDE, 
        vec=hp.ang2vec(
            ra0, dec0, lonlat=True
        ),
        radius=np.deg2rad(psi)
    )

    mask = np.isin(g['pix'], query_pix_id, assume_unique=True)
    return g[mask]

if __name__ == '__main__':
    N = 10000
    ra0, dec0, psi = 150.0, 10.0, 10.0

    create_mock_cat(N=N)
    masked_g = get_masked_data(ra0=ra0, dec0=dec0, psi=psi)
    #masked_g = get_masked_data_lensing(ra0=ra0, dec0=dec0, psi=psi)
    
    plt.scatter(g['ra'], g['dec'], s=1, alpha=0.3)
    plt.scatter(ra0, dec0, s=3, c='r', marker='s')
    plt.scatter(masked_g['ra'], masked_g['dec'], s=5, alpha=1)

    plt.show()