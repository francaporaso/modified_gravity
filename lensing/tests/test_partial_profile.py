"""The original `partial_profile` was refactored (shared helpers with the map and weighted
functions). This compares it with a frozen copy of the original code (commit 3618b7d),
which doesn't use any helper of the refactor, and tests the empty-disc errors of the
catalog functions.

Run with: python -m pytest lensing/tests/test_partial_profile.py
"""

from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
from astropy.constants import G, c, M_sun, pc
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

import lensing.stacking as st
from lensing.geometry import eq2p2

COSMO = FlatLambdaCDM(H0=100.0, Om0=0.3089)
NSIDE = 64
CAP = 6.0  # deg, radius of the caps with galaxies
CENTRES = [(40.0, 10.0), (200.0, -35.0), (10.0, 84.0)]  # the last one is near a pole
VOIDS = [  # (Rv, ra, dec, z)
    (15.0, 40.0, 10.0, 0.25),
    (8.0, 200.0, -35.0, 0.15),
    (12.0, 10.0, 84.0, 0.30),
    (20.0, 42.0, 8.0, 0.12),
]
SC = (c.value**2.0 / (4.0 * np.pi * G.value)) * (pc.value / M_sun.value) * 1e-6


# --- frozen copy of the original code (the cosmology and the constants written out) -----


def ref_sigma_crit(z_l, z_s):
    d_l = COSMO.angular_diameter_distance(z_l).value
    d_s = COSMO.angular_diameter_distance(z_s).value
    d_ls = COSMO.angular_diameter_distance(z_l, z_s).value
    return SC * (d_s / (d_ls * d_l))


def ref_masked_idx(psi, ra0, dec0, z0):
    queried_pix = hp.query_disc(
        st.cfg.NSIDE,
        vec=hp.ang2vec(ra0, dec0, lonlat=True),
        radius=np.deg2rad(psi * 1.0),
        inclusive=True,
    )
    queried_idranges = np.array(
        [st.PIX_TO_IDX[p] for p in queried_pix if p in st.PIX_TO_IDX]
    )
    idx_arrays = np.concatenate([np.arange(s, e) for s, e in queried_idranges])
    mask_z = st.SOURCE[st.cfg.scols["redshift"]].data[idx_arrays] > (z0 + 0.1)
    return idx_arrays[mask_z]


def ref_partial_profile(inp):
    cfg = st.cfg
    GAMMA_SIGN = (-1.0, -1.0)
    Sigma_wsum = np.zeros(cfg.NBINS)
    DSigma_t_wsum = np.zeros(cfg.NBINS)
    DSigma_x_wsum = np.zeros(cfg.NBINS)
    N_inbin = np.zeros(cfg.NBINS)

    Rv0, ra0, dec0, z0 = inp
    DEGxMPC = COSMO.arcsec_per_kpc_comoving(z0).to("deg/Mpc").value / COSMO.h
    psi = DEGxMPC * cfg.ROUT * Rv0

    idx = ref_masked_idx(psi, ra0, dec0, z0)
    catdata = st.SOURCE[idx]

    sigma_c = ref_sigma_crit(z0, catdata[cfg.scols["redshift"]]) / (Rv0 * COSMO.h)

    rads, theta = eq2p2(
        np.deg2rad(catdata[cfg.scols["ra"]]),
        np.deg2rad(catdata[cfg.scols["dec"]]),
        np.deg2rad(ra0),
        np.deg2rad(dec0),
    )

    e1 = GAMMA_SIGN[0] * catdata[cfg.scols["gamma1"]]
    e2 = GAMMA_SIGN[1] * catdata[cfg.scols["gamma2"]]
    if cfg.addnoise:
        e1 -= GAMMA_SIGN[0] * catdata[cfg.scols["eps1"]]
        e2 -= GAMMA_SIGN[1] * catdata[cfg.scols["eps2"]]

    cos2t = np.cos(2.0 * theta)
    sin2t = np.sin(2.0 * theta)
    et = -(e1 * cos2t + e2 * sin2t) * sigma_c
    ex = (-e1 * sin2t + e2 * cos2t) * sigma_c
    k = catdata[cfg.scols["kappa"]] * sigma_c

    bines = np.linspace(cfg.RIN, cfg.ROUT, cfg.NBINS + 1)
    dig = np.digitize((np.rad2deg(rads) / DEGxMPC) / Rv0, bines)

    for nbin in range(cfg.NBINS):
        mbin = dig == nbin + 1
        Sigma_wsum[nbin] = k[mbin].sum()
        DSigma_t_wsum[nbin] = et[mbin].sum()
        DSigma_x_wsum[nbin] = ex[mbin].sum()
        N_inbin[nbin] = np.count_nonzero(mbin)

    return Sigma_wsum, DSigma_t_wsum, DSigma_x_wsum, N_inbin


# --- synthetic catalog -----------------------------------------------------------------


def make_catalog():
    r = np.random.default_rng(2)
    xyz = r.normal(size=(2_000_000, 3))
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    ra, dec = hp.vec2ang(xyz, lonlat=True)
    near = np.zeros(len(ra), bool)
    for ra0, dec0 in CENTRES:
        near |= np.rad2deg(np.arccos(np.clip(xyz @ hp.ang2vec(ra0, dec0, lonlat=True), -1, 1))) < CAP
    ra, dec = ra[near], dec[near]
    n = len(ra)
    z = r.uniform(0.2, 1.0, n)

    # sources exactly at z0 + 0.1 of the first void, next to its centre: the cut is strict
    m = 300
    ra = np.append(ra, 40.0 + r.uniform(-1, 1, m))
    dec = np.append(dec, 10.0 + r.uniform(-1, 1, m))
    z = np.append(z, np.full(m, VOIDS[0][3] + 0.1))
    n += m

    cat = Table(
        {
            "ra": ra,
            "dec": dec,
            "z": z,
            "gamma1": r.normal(0, 0.05, n),
            "gamma2": r.normal(0, 0.05, n),
            "kappa": r.normal(0.02, 0.03, n),
            "eps1": r.normal(0, 0.2, n),
            "eps2": r.normal(0, 0.2, n),
        }
    )
    cat["pix"] = hp.ang2pix(NSIDE, cat["ra"], cat["dec"], lonlat=True)
    cat.sort("pix")
    upix, first = np.unique(cat["pix"], return_index=True)
    last = np.append(first[1:], len(cat))
    return cat, {int(p): (a, b) for p, a, b in zip(upix, first, last)}


@pytest.fixture(scope="module")
def catalog():
    return make_catalog()


def setup(catalog, addnoise):
    cat, pix_to_idx = catalog
    st.cfg = SimpleNamespace(
        NBINS=10,
        RIN=0.2,
        ROUT=2.5,
        NSIDE=NSIDE,
        binning="lin",
        addnoise=addnoise,
        gamma_sign=(-1.0, -1.0),
        scols={
            "ra": "ra",
            "dec": "dec",
            "redshift": "z",
            "gamma1": "gamma1",
            "gamma2": "gamma2",
            "kappa": "kappa",
            "eps1": "eps1",
            "eps2": "eps2",
        },
    )
    st.cosmo = COSMO
    st.set_binning()
    st.SOURCE = cat
    st.PIX_TO_IDX = pix_to_idx


@pytest.mark.parametrize("addnoise", [False, True])
@pytest.mark.parametrize("void", VOIDS)
def test_partial_profile_is_the_original_one(catalog, void, addnoise):
    setup(catalog, addnoise)
    ref = ref_partial_profile(void)
    new = st.partial_profile(void)

    assert len(new) == 5
    assert ref[3].sum() > 300  # the void has sources
    # the number of sources is exact; the sums differ by the order of the operations
    np.testing.assert_array_equal(new[4], ref[3])
    np.testing.assert_array_equal(new[3], new[4])  # unit weights: W = N
    scale = max(np.abs(ref[0]).max(), np.abs(ref[1]).max())
    for new_i, ref_i in zip(new[:3], ref[:3]):
        np.testing.assert_allclose(new_i, ref_i, rtol=1e-11, atol=1e-13 * scale)
    assert np.abs(new[1]).max() > 0  # not trivially zero


def test_sources_exactly_on_the_z_cut_are_excluded(catalog):
    # z > z0 + 0.1, as the original: with >= the 300 sources at the cut would be counted
    setup(catalog, False)
    Rv0, ra0, dec0, z0 = VOIDS[0]
    deg, psi = st.aperture(Rv0, z0)
    idx = st.query_sources(psi, ra0, dec0, z0 + 0.1, strict=True)
    assert not np.any(st.SOURCE["z"][idx] == z0 + 0.1)
    idx_inclusive = st.query_sources(psi, ra0, dec0, z0 + 0.1, strict=False)
    assert len(idx_inclusive) - len(idx) >= 300


def test_empty_disc_is_an_error(catalog):
    setup(catalog, False)
    # no sources at all around the void (no pixel of the catalog in the disc)
    with pytest.raises(ValueError, match="No sources within"):
        st.partial_profile((15.0, 120.0, -60.0, 0.25))
    st.ZSRC_MIN = 0.6
    with pytest.raises(ValueError, match="No sources within"):
        st.partial_profile_weighted((15.0, 120.0, -60.0, 0.25))
    # sources around, but none behind the void
    with pytest.raises(ValueError, match="z >"):
        st.partial_profile((15.0, 40.0, 10.0, 0.95))
    with pytest.raises(ValueError, match="z >="):
        st.query_sources(3.0, 40.0, 10.0, 1.5, strict=False)
