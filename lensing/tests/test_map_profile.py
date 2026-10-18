"""Compares partial_profile (galaxy catalog) with partial_profile_map (lensing cube)
built from the same synthetic galaxies.

Run with: pytest lensing/tests/test_map_profile.py
"""
import sys
from types import ModuleType, SimpleNamespace

import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

# optional heavy deps not needed here (jackknife, parquet): stub them if missing
for _mod in ('kmeans_radec', 'pyarrow', 'pyarrow.parquet'):
    try:
        __import__(_mod)
    except ImportError:
        sys.modules[_mod] = ModuleType(_mod)
sys.modules['kmeans_radec'].__dict__.setdefault('kmeans_sample', None)
if not hasattr(sys.modules['pyarrow'], 'parquet'):
    sys.modules['pyarrow'].parquet = sys.modules['pyarrow.parquet']

import lensing.stacking as st

rng = np.random.default_rng(0)

NSIDE_CAT = 64
NSIDE_MAP = 256
RA0, DEC0, RV0 = 40.0, 10.0, 15.0
DZ = 0.05
ZEDGES = np.arange(0.1, 1.2, DZ)  # steps, in decreasing z like the real cube
NGAL = 1_500_000


def make_galaxies():
    # galaxies in a cap around the void centre
    vec = hp.ang2vec(RA0, DEC0, lonlat=True)
    pix = hp.query_disc(NSIDE_CAT, vec, np.deg2rad(25.0))
    ra, dec = hp.pix2ang(NSIDE_CAT, rng.choice(pix, NGAL), lonlat=True)
    ra += rng.uniform(-0.5, 0.5, NGAL)
    dec = np.clip(dec + rng.uniform(-0.5, 0.5, NGAL), -90, 90)
    z = rng.uniform(ZEDGES[0], ZEDGES[-1], NGAL)
    # noise-free tangential shear around the void: gamma_t = a(r), so that differences
    # come only from the pixelization (same sign convention as partial_profile)
    rad, theta = st.eq2p2(np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(RA0), np.deg2rad(DEC0))
    a = 0.05*np.exp(-np.rad2deg(rad)/3.0)*(1.0 + z)
    g1 = -a*np.cos(2.0*theta)
    g2 = a*np.sin(2.0*theta)
    k = 0.1*np.exp(-np.rad2deg(rad)/3.0)*(1.0 + z)
    return Table({'ra': ra, 'dec': dec, 'z': z, 'gamma1': g1, 'gamma2': g2, 'kappa': k})


def make_cube(gal):
    step = np.searchsorted(ZEDGES, gal['z'], side='right') - 1
    nz = len(ZEDGES) - 1
    npix = hp.nside2npix(NSIDE_MAP)
    pix = hp.ang2pix(NSIDE_MAP, gal['ra'], gal['dec'], lonlat=True)
    cube = {'nside': NSIDE_MAP, 'nest': False}
    cube['nobj'] = np.zeros((npix, nz), dtype=np.int64)
    np.add.at(cube['nobj'], (pix, step), 1)
    for name in ('gamma1', 'gamma2', 'kappa'):
        cube[name] = np.zeros((npix, nz))
        np.add.at(cube[name], (pix, step), gal[name])
    # steps in decreasing z, as in the real cube
    for key in ('nobj', 'gamma1', 'gamma2', 'kappa'):
        cube[key] = cube[key][:, ::-1]
    cube['zmin'] = ZEDGES[:-1][::-1].copy()
    cube['zmax'] = ZEDGES[1:][::-1].copy()
    cube['zmid'] = 0.5*(cube['zmin'] + cube['zmax'])
    cube['zeff'] = cube['zmid']
    cube['zcut'] = cube['zmin']
    return cube


def setup(gal):
    st.cfg = SimpleNamespace(
        NBINS=10, RIN=0.2, ROUT=2.5, NSIDE=NSIDE_CAT, binning='lin', addnoise=False,
        scols={'ra': 'ra', 'dec': 'dec', 'redshift': 'z', 'gamma1': 'gamma1',
               'gamma2': 'gamma2', 'kappa': 'kappa', 'eps1': 'e1', 'eps2': 'e2', 'pix': 'pix'},
        dz_buffer=0.1, step_criterion='zmin',
    )
    st.cosmo = FlatLambdaCDM(H0=100.0, Om0=0.25, Ob0=0.0)
    st.binspace = np.linspace

    gal['pix'] = hp.ang2pix(NSIDE_CAT, gal['ra'], gal['dec'], lonlat=True)
    gal.sort('pix')
    upix, first = np.unique(gal['pix'], return_index=True)
    last = np.append(first[1:], len(gal))
    st.PIX_TO_IDX = {int(p): (a, b) for p, a, b in zip(upix, first, last)}


def test_map_matches_catalog():
    gal = make_galaxies()
    cube = make_cube(gal)
    setup(gal)

    # z_lens + dz_buffer sits exactly on a step edge, so both cuts select the same galaxies
    z0 = round(ZEDGES[7] - 0.1, 10)
    inp = (RV0, RA0, DEC0, z0)

    # SOURCE holds either the galaxy table or the cube, depending on the mode
    st.SOURCE = gal
    k_c, t_c, x_c, n_c = st.partial_profile(inp)
    st.SOURCE = cube
    k_m, t_m, x_m, n_m = st.partial_profile_map(inp)

    # pixel centres replace galaxy positions, so counts only agree up to the pixelization
    np.testing.assert_allclose(n_m, n_c, rtol=0.15)
    for c, m in ((k_c/n_c, k_m/n_m), (t_c/n_c, t_m/n_m)):
        np.testing.assert_allclose(m, c, atol=0.1*np.abs(c).max())
    # the cross component is zero by construction; the map leaks <1% of the signal
    np.testing.assert_allclose(x_m/n_m, x_c/n_c, atol=0.02*np.abs(t_c/n_c).max())


if __name__ == '__main__':
    test_map_matches_catalog()
    print('ok')
