"""Compares the galaxy-catalog estimator (`partial_profile_weighted`) with the one from
the lensing cube (`partial_profile_map`), both with 1/sigma_crit weights, using the
same synthetic galaxies. Also tests the setup checks of the map mode.

Run with: python -m pytest lensing/tests/test_map_profile.py
"""

import sys
from types import ModuleType, SimpleNamespace

import healpy as hp
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

# optional heavy deps not needed here (jackknife, parquet): stub them if missing
for _mod in ("kmeans_radec", "pyarrow", "pyarrow.parquet"):
    try:
        __import__(_mod)
    except ImportError:
        sys.modules[_mod] = ModuleType(_mod)
sys.modules["kmeans_radec"].__dict__.setdefault("kmeans_sample", None)
if not hasattr(sys.modules["pyarrow"], "parquet"):
    sys.modules["pyarrow"].parquet = sys.modules["pyarrow.parquet"]

import lensing.stacking as st
from shearmap import table2cube as t2c

rng = np.random.default_rng(0)

ORDER = 8  # order of the synthetic cube
NSIDE_MAP = 2**ORDER
NSIDE_CAT = 64
RA0, DEC0, RV0 = 40.0, 10.0, 15.0
Z_LOW, Z_HIGH, Z0 = 0.2, 0.3, 0.25  # void bin and the redshift of the void
NPIX_CAT = 4.0  # radius (deg) of the cap with galaxies
NGAL_PER_PIX = 150
COSMO = FlatLambdaCDM(H0=100.0, Om0=0.3089)  # h = 1


def make_galaxies(on_centres):
    # galaxies in a cap around the void centre, in the pixels of the cube
    vec = hp.ang2vec(RA0, DEC0, lonlat=True)
    pix = hp.query_disc(NSIDE_MAP, vec, np.deg2rad(NPIX_CAT), nest=True)
    pix = np.repeat(pix, NGAL_PER_PIX)
    ra, dec = hp.pix2ang(NSIDE_MAP, pix, nest=True, lonlat=True)
    if not on_centres:
        half = 0.5 * np.rad2deg(hp.nside2resol(NSIDE_MAP))
        ra = ra + rng.uniform(-half, half, len(ra)) / np.cos(np.deg2rad(dec))
        dec = dec + rng.uniform(-half, half, len(dec))
    z = rng.uniform(0.2, 1.0, len(ra))

    # noise-free tangential shear around the void, in the L768 convention: the code
    # rotates (e1, e2) = (-gamma1, -gamma2), so gamma = a * (cos 2theta, sin 2theta)
    # gives a purely tangential, positive signal
    rad, theta = st.eq2p2(
        np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(RA0), np.deg2rad(DEC0)
    )
    a = 0.05 * np.exp(-np.rad2deg(rad) / 3.0) * (1.0 + z)
    k = 0.1 * np.exp(-np.rad2deg(rad) / 3.0) * (1.0 + z)
    return Table(
        {
            "ra": ra,
            "dec": dec,
            "z": z,
            "gamma1": a * np.cos(2.0 * theta),
            "gamma2": a * np.sin(2.0 * theta),
            "kappa": k,
        }
    )


def make_cube(gal):
    """the sums that the CosmoHub query + table2cube produce"""
    nz = len(t2c.ZEDGES) - 1
    npix = hp.nside2npix(NSIDE_MAP)
    zbin = np.searchsorted(t2c.ZEDGES, gal["z"], side="right") - 1
    pix = hp.ang2pix(NSIDE_MAP, gal["ra"], gal["dec"], nest=True, lonlat=True)
    flat = pix * nz + zbin
    chi = COSMO.comoving_distance(gal["z"]).value  # Mpc/h, h = 1
    cube = {
        "order": ORDER,
        "nside": NSIDE_MAP,
        "nest": True,
        "chiref": t2c.CHI_REF,
        "optweight": False,
        "zmin": np.array(t2c.ZEDGES[:-1]),
        "zmax": np.array(t2c.ZEDGES[1:]),
        "nobj": np.bincount(flat, minlength=npix * nz).reshape(npix, nz),
    }
    for name, x in {
        "gamma1": gal["gamma1"],
        "gamma2": gal["gamma2"],
        "kappa": gal["kappa"],
        "invchi": t2c.CHI_REF / chi,
    }.items():
        arr = np.bincount(flat, weights=x, minlength=npix * nz)
        cube[name] = arr.reshape(npix, nz).astype(np.float32)
    return cube


def setup(gal, cube):
    st.cfg = SimpleNamespace(
        NBINS=10,
        RIN=0.2,
        ROUT=2.5,
        NSIDE=NSIDE_CAT,
        binning="lin",
        addnoise=False,
        scols={
            "ra": "ra",
            "dec": "dec",
            "redshift": "z",
            "gamma1": "gamma1",
            "gamma2": "gamma2",
            "kappa": "kappa",
        },
        dz_buffer=0.1,
        weighting="invsc",
        zbins=[(Z_LOW, Z_HIGH)],
    )
    st.cosmo = COSMO
    st.binspace = np.linspace

    # map: sources behind the void bin
    st.SOURCE = cube
    st.collapse_slices(Z_LOW)

    # catalog: pixel index of the galaxies
    gal["pix"] = hp.ang2pix(NSIDE_CAT, gal["ra"], gal["dec"], lonlat=True)
    gal.sort("pix")
    upix, first = np.unique(gal["pix"], return_index=True)
    last = np.append(first[1:], len(gal))
    st.PIX_TO_IDX = {int(p): (a, b) for p, a, b in zip(upix, first, last)}
    return gal


def profiles(gal, cube):
    gal = setup(gal, cube)
    inp = (RV0, RA0, DEC0, Z0)
    map_res = st.partial_profile_map(inp)  # SOURCE is the cube
    st.SOURCE = gal
    cat_res = st.partial_profile_weighted(inp, source_cut="bin")
    return cat_res, map_res


def test_exact_on_pixel_centres():
    # with the galaxies at the pixel centres the two estimators must agree, so this
    # checks the moments, the weights and the rotation, without pixelization
    gal = make_galaxies(on_centres=True)
    (k_c, t_c, x_c, w_c, n_c), (k_m, t_m, x_m, w_m, n_m) = profiles(gal, make_cube(gal))

    assert n_c.sum() > 10_000
    np.testing.assert_array_equal(n_m, n_c)
    np.testing.assert_allclose(w_m, w_c, rtol=1e-4)
    # float32 sums: tolerance relative to the signal (gamma_x is zero by construction)
    scale = max(np.abs(k_c).max(), np.abs(t_c).max())
    for c, m in ((k_c, k_m), (t_c, t_m), (x_c, x_m)):
        np.testing.assert_allclose(m, c, rtol=1e-4, atol=1e-5 * scale)
    # the signal is purely tangential and positive
    assert np.abs(t_c).max() > 100 * np.abs(x_c).max()
    assert np.all(t_c > 0)


def test_gamma_convention_in_all_profile_functions():
    # the original catalog function, the weighted one and the map one must agree on the
    # sign: positive tangential, no cross component, for a tangential field in L768
    # convention. With the MICE convention (e1, e2) = (+gamma1, -gamma2) this fails.
    assert st.GAMMA_SIGN == (-1.0, -1.0)
    gal = make_galaxies(on_centres=False)
    cube = make_cube(gal)
    gal = setup(gal, cube)
    inp = (RV0, RA0, DEC0, Z0)

    map_res = st.partial_profile_map(inp)
    st.SOURCE = gal
    weighted_res = st.partial_profile_weighted(inp, source_cut="bin")
    k, t, x, n = st.partial_profile(inp)  # unit weights, 4-tuple

    for name, (t_, x_) in {
        "map": (map_res[1], map_res[2]),
        "weighted": (weighted_res[1], weighted_res[2]),
        "original": (t, x),
    }.items():
        assert np.all(t_ > 0), name
        assert np.abs(x_).max() < 0.05 * np.abs(t_).max(), name


def test_catalog_selection_reads_every_source_in_the_disc():
    # a small void at high z has a disc of ~1 deg, the size of the NSIDE=64 pixels used
    # to index the catalog: the pixel pre-selection must not drop sources inside it
    gal = make_galaxies(on_centres=False)
    gal = setup(gal, make_cube(gal))
    st.SOURCE = gal
    z_src_min = 0.5
    rv = 6.0
    degxmpc = COSMO.arcsec_per_kpc_proper(z_src_min - 0.1).to("deg/Mpc").value
    psi = degxmpc * st.cfg.ROUT * rv

    # get_masked_idx_fast keeps the sources with z > z0 + 0.1
    idx = st.get_masked_idx_fast(psi, RA0, DEC0, z_src_min - 0.1)
    rad, _ = st.eq2p2(
        np.deg2rad(gal["ra"]), np.deg2rad(gal["dec"]), np.deg2rad(RA0), np.deg2rad(DEC0)
    )
    inside = np.flatnonzero((np.rad2deg(rad) < psi) & (gal["z"] > z_src_min))
    assert len(inside) > 1000
    assert set(inside) <= set(idx.tolist())


def test_lens_columns_check():
    # voids as read_sparkling returns them (no ID column): 12 columns
    n = 50
    voids = rng.normal(size=(12, n))
    voids[0] = rng.uniform(10, 20, n)  # Rv
    voids[1] = rng.uniform(0, 360, n)  # ra
    voids[2] = rng.uniform(-90, 90, n)  # dec
    voids[3] = rng.uniform(0.1, 0.2, n)  # z
    voids[8] = rng.uniform(-1, 100, n)  # delta
    bounds = {"rv": (10.0, 20.0), "z": (0.1, 0.2), "delta": (-1.0, 100.0)}

    good = {"rv": 0, "ra": 1, "dec": 2, "z": 3, "delta": 8}
    st.check_lens_columns(voids, good, bounds)

    # the template used to assume an ID column: every column shifted by one
    shifted = {"rv": 1, "ra": 2, "dec": 3, "z": 4, "delta": 9}
    with pytest.raises(ValueError, match="rv = 1"):
        st.check_lens_columns(voids, shifted, bounds)
    # only z shifted: the check must still find it
    with pytest.raises(ValueError, match="z = 4"):
        st.check_lens_columns(voids, dict(good, z=4), bounds)

    # no voids selected: nothing to check
    st.check_lens_columns(voids[:, :0], shifted, bounds)


def test_pixelization():
    # galaxies spread inside the pixels: only the pixel-centre positions differ
    gal = make_galaxies(on_centres=False)
    (k_c, t_c, x_c, w_c, n_c), (k_m, t_m, x_m, w_m, n_m) = profiles(gal, make_cube(gal))

    np.testing.assert_allclose(n_m, n_c, rtol=0.1)
    for c, m in ((k_c / w_c, k_m / w_m), (t_c / w_c, t_m / w_m)):
        np.testing.assert_allclose(m, c, atol=0.1 * np.abs(c).max())
    # the cross component is zero by construction; the map leaks a few % of the signal
    np.testing.assert_allclose(
        x_m / w_m, x_c / w_c, atol=0.05 * np.abs(t_c / w_c).max()
    )


def test_inverse_sigma_crit_from_moments():
    st.cosmo = COSMO
    z_l = 0.25
    z_s = rng.uniform(0.3, 1.3, 1000)
    chi_s = COSMO.comoving_distance(z_s).value
    chi_l = COSMO.comoving_distance(z_l).value
    via_moments = st.inv_sigma_crit_prefactor(z_l) * (len(z_s) - chi_l * np.sum(1 / chi_s))
    direct = np.sum(1.0 / st.sigma_crit(z_l, z_s))
    assert via_moments == pytest.approx(direct, rel=1e-8)


def test_setup_checks():
    gal = make_galaxies(on_centres=True)
    setup(gal, make_cube(gal))

    # z_low + buffer sits on a slice edge: slice k starts at 0.2 + 0.1 k
    for k, z_low in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
        assert st.collapse_slices(z_low) == k
    assert st.ZSRC_MIN == pytest.approx(0.6)

    # not an edge, or no slice at all
    for z_low in (0.15, 0.6, 0.0):
        with pytest.raises(ValueError, match="slice"):
            st.collapse_slices(z_low)

    st.cfg.zbins = [(0.1, 0.2), (0.4, 0.5)]
    st.check_map_zbins()
    st.cfg.zbins = [(0.1, 0.3)]  # voids up to 0.3 are behind the 0.2-0.3 sources
    with pytest.raises(ValueError, match="wider"):
        st.check_map_zbins()
    st.cfg.zbins = [(0.15, 0.2)]
    with pytest.raises(ValueError, match="slice"):
        st.check_map_zbins()


def test_collapse_sums_selected_slices():
    gal = make_galaxies(on_centres=False)
    cube = make_cube(gal)
    setup(gal, cube)
    st.collapse_slices(0.3)  # slices with zmin >= 0.4
    np.testing.assert_array_equal(
        st.MAPS["nobj"], cube["nobj"][:, 2:].sum(axis=1)
    )
    np.testing.assert_allclose(
        st.MAPS["invchi"], cube["invchi"][:, 2:].sum(axis=1, dtype=np.float64)
    )
    assert st.MAPS["kappa"].dtype == np.float64


def test_load_map_source_and_degrade(tmp_path):
    gal = make_galaxies(on_centres=False)
    cube = make_cube(gal)
    path = tmp_path / "cube.fits"
    t2c.save_cube(cube, path)

    src = st.load_map_source(path)
    assert src["order"] == ORDER and src["nest"]
    assert src["nobj"].shape == cube["nobj"].shape

    low = st.load_map_source(path, order=ORDER - 2)
    assert low["order"] == ORDER - 2 and low["nside"] == NSIDE_MAP // 4
    np.testing.assert_array_equal(
        low["nobj"].sum(axis=0), cube["nobj"].sum(axis=0)
    )
    with pytest.raises(ValueError, match="higher"):
        st.load_map_source(path, order=ORDER + 1)
    src["hdul"].close()


def test_degraded_cube_matches_direct_collapse():
    gal = make_galaxies(on_centres=False)
    cube = make_cube(gal)
    low = t2c.degrade_cube(cube, ORDER - 1)
    # galaxies of pixel p at order 8 belong to pixel p // 4 at order 7
    pix = hp.ang2pix(NSIDE_MAP, gal["ra"], gal["dec"], nest=True, lonlat=True) // 4
    nz = len(t2c.ZEDGES) - 1
    zbin = np.searchsorted(t2c.ZEDGES, gal["z"], side="right") - 1
    expected = np.bincount(pix * nz + zbin, minlength=low["nobj"].size)
    np.testing.assert_array_equal(low["nobj"].ravel(), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
