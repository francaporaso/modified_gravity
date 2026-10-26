"""Compares the galaxy-catalog estimator (`partial_profile_weighted`) with the one from
the lensing cube (`partial_profile_map`), both with 1/sigma_crit weights, using the
same synthetic galaxies. Also tests the setup checks of the map mode.

Run with: python -m pytest lensing/tests/test_map_profile.py
"""

from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

import lensing.stacking as st
from lensing import cosmology, geometry, io
from lensing.geometry import eq2p2
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
    # own generator: the galaxies don't depend on which tests ran before
    rng = np.random.default_rng(1)
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
    rad, theta = eq2p2(
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
    st.set_binning()

    # map: sources behind the void bin (z >= Z_HIGH + 0.1)
    st.SOURCE = cube
    st.collapse_slices(Z_HIGH)

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
    assert geometry.GAMMA_SIGN == (-1.0, -1.0)
    gal = make_galaxies(on_centres=False)
    cube = make_cube(gal)
    gal = setup(gal, cube)
    inp = (RV0, RA0, DEC0, Z0)

    map_res = st.partial_profile_map(inp)
    st.SOURCE = gal
    weighted_res = st.partial_profile_weighted(inp, source_cut="bin")
    k, t, x, w, n = st.partial_profile(inp)  # W = N

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

    idx = st.query_sources(psi, RA0, DEC0, z_src_min, strict=True)
    rad, _ = eq2p2(
        np.deg2rad(gal["ra"]), np.deg2rad(gal["dec"]), np.deg2rad(RA0), np.deg2rad(DEC0)
    )
    inside = np.flatnonzero((np.rad2deg(rad) < psi) & (gal["z"] > z_src_min))
    assert len(inside) > 1000
    assert set(inside) <= set(idx.tolist())


@pytest.mark.parametrize("h", [1.0, 0.7])
def test_angular_scale_is_comoving(h):
    # 1 comoving Mpc/h subtends 1/chi(z) with chi in Mpc/h, and (1+z) less than the proper
    # scale that the original partial_profile uses
    st.cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=0.3089)
    for z in (0.15, 0.4, 0.6):
        chi = cosmology.comoving_chi(st.cosmo, z)  # Mpc/h
        assert cosmology.deg_per_mpch(st.cosmo, z) == pytest.approx(np.rad2deg(1.0 / chi), rel=1e-6)
    proper = st.cosmo.arcsec_per_kpc_proper(0.6).to("deg/Mpc").value / st.cosmo.h
    assert proper / cosmology.deg_per_mpch(st.cosmo, 0.6) == pytest.approx(1.6, rel=1e-6)


def test_ring_at_one_comoving_rv_is_in_the_bin_of_r_over_rv_equal_one():
    # sources on a ring at exactly Rv (comoving) from the void: R/Rv = 1, i.e. bin 3 of
    # NBINS=10 in [0.2, 2.5]. With the proper scale they would land at R/Rv = 1/(1+z0).
    n = 4000
    phi = rng.uniform(0, 2 * np.pi, n)
    theta = np.rad2deg(RV0 / (COSMO.comoving_distance(Z0).value))  # deg
    ra = RA0 + theta * np.cos(phi) / np.cos(np.deg2rad(DEC0))
    dec = DEC0 + theta * np.sin(phi)
    z = rng.uniform(0.4, 1.0, n)  # all behind the void bin
    gal = Table(
        {"ra": ra, "dec": dec, "z": z, "gamma1": np.full(n, 0.01),
         "gamma2": np.full(n, 0.01), "kappa": np.full(n, 0.01)}
    )
    gal = setup(gal, make_cube(gal))
    st.SOURCE = gal
    n_bin = st.partial_profile_weighted((RV0, RA0, DEC0, Z0), source_cut="bin")[4]

    edges = np.linspace(st.cfg.RIN, st.cfg.ROUT, st.cfg.NBINS + 1)
    expected = np.searchsorted(edges, 1.0) - 1
    assert n_bin[expected] == n_bin.sum() == n
    assert 1.0 / (1.0 + Z0) < edges[expected]  # the proper scale would be another bin

    # the original catalog function (unit weights, its own source cut) uses the same scale
    n_orig = st.partial_profile((RV0, RA0, DEC0, Z0))[4]
    assert n_orig.sum() > 0.9 * n and np.argmax(n_orig) == expected
    assert n_orig[expected] == n_orig.sum()


def test_resolution_check_marks(capsys):
    gal = make_galaxies(on_centres=True)
    setup(gal, make_cube(gal))
    st.cfg.rvbins = [(10.0, 20.0)]
    st.cfg.zbins = [(Z_LOW, Z_HIGH)]
    st.SOURCE = dict(st.SOURCE, nside=2048)  # 1.7 arcmin pixels
    st.check_map_resolution()
    out = capsys.readouterr().out
    assert "NSIDE=2048" in out and "✓" in out and "WARNING" not in out
    st.SOURCE = dict(st.SOURCE, nside=64)  # 55 arcmin pixels
    st.check_map_resolution()
    out = capsys.readouterr().out
    assert "✗" in out and "WARNING" in out


@pytest.mark.parametrize("binning", ["lin", "log"])
def test_bin_centres(binning):
    centres = geometry.bin_centres(0.1, 2.5, 4, binning)
    edges = geometry.bin_edges(0.1, 2.5, 4, binning)
    assert len(centres) == 4
    assert np.all((centres > edges[:-1]) & (centres < edges[1:]))
    if binning == "lin":
        np.testing.assert_allclose(centres, [0.4, 1.0, 1.6, 2.2])
    else:
        np.testing.assert_allclose(centres, np.sqrt(edges[:-1] * edges[1:]))
        # equidistant in log: centre ratio is constant
        assert np.ptp(np.diff(np.log(centres))) < 1e-12


def test_lens_summary_median_is_not_the_mean():
    voids = np.zeros((12, 5))
    voids[0] = [10, 11, 12, 13, 20]  # Rv: mean 13.2, median 12
    voids[3] = [0.1, 0.1, 0.1, 0.15, 0.19]  # z
    voids[8] = [-0.9, -0.9, -0.8, -0.1, 50.0]  # delta: mean 9.46, median -0.8
    s = io.lens_summary(voids, {"rv": 0, "ra": 1, "dec": 2, "z": 3, "delta": 8})
    assert s["nvoids"] == 5
    assert s["Rv_mean"] == pytest.approx(13.2) and s["Rv_median"] == 12
    assert s["delta_mean"] == pytest.approx(9.46)
    assert s["delta_median"] == pytest.approx(-0.8)


def test_pixelization():
    # galaxies spread inside the pixels: only the pixel-centre positions differ
    gal = make_galaxies(on_centres=False)
    (k_c, t_c, x_c, w_c, n_c), (k_m, t_m, x_m, w_m, n_m) = profiles(gal, make_cube(gal))

    # the pixels (0.23 deg at order 8) are about as wide as the radial bins here, so the
    # pixel grid moves up to ~10% of the sources between neighbouring bins (the same for
    # any seed) but conserves the total
    np.testing.assert_allclose(n_m.sum(), n_c.sum(), rtol=0.02)
    np.testing.assert_allclose(n_m, n_c, rtol=0.15)
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
    via_moments = cosmology.inv_sigma_crit_prefactor(COSMO, z_l) * (
        len(z_s) - chi_l * np.sum(1 / chi_s)
    )
    direct = np.sum(1.0 / cosmology.sigma_crit(COSMO, z_l, z_s))
    assert via_moments == pytest.approx(direct, rel=1e-8)


def test_setup_checks(capsys):
    gal = make_galaxies(on_centres=True)
    setup(gal, make_cube(gal))

    # z_high + buffer on a slice edge: slice k starts at 0.2 + 0.1 k
    for k, z_high in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
        assert st.collapse_slices(z_high) == k
    assert st.ZSRC_MIN == pytest.approx(0.6)
    # not an edge: the next one, so the sources are never closer than the buffer
    assert st.collapse_slices(0.25) == 2 and st.ZSRC_MIN == pytest.approx(0.4)
    assert st.collapse_slices(0.0) == 0  # below the first edge: all the slices
    # no slice at or above z_high + buffer
    for z_high in (0.55, 0.6, 1.0):
        with pytest.raises(ValueError, match="slice"):
            st.collapse_slices(z_high)

    capsys.readouterr()
    st.cfg.zbins = [(0.1, 0.2), (0.4, 0.5)]
    st.check_map_zbins()
    out = capsys.readouterr().out
    assert "sources z >= 0.30" in out and "sources z >= 0.60" in out and "✗" not in out
    st.cfg.zbins = [(0.1, 0.25)]
    st.check_map_zbins()
    out = capsys.readouterr().out
    assert "sources z >= 0.40" in out and "✗" in out and "WARNING" in out
    st.cfg.zbins = [(0.4, 0.6)]
    with pytest.raises(ValueError, match="slice"):
        st.check_map_zbins()


def test_wide_void_bin_takes_the_sources_behind_its_last_void(capsys):
    # [0.1, 0.3) is wider than dz_buffer: its sources start at 0.3 + 0.1, not 0.1 + 0.1,
    # so the voids at 0.2-0.3 have no sources in front of them
    gal = make_galaxies(on_centres=True)
    cube = make_cube(gal)
    setup(gal, cube)
    st.cfg.zbins = [(0.1, 0.3)]
    st.check_map_zbins()
    assert "sources z >= 0.40" in capsys.readouterr().out
    assert st.collapse_slices(0.3) == 2 and st.ZSRC_MIN == pytest.approx(0.4)
    np.testing.assert_array_equal(st.MAPS["nobj"], cube["nobj"][:, 2:].sum(axis=1))
    # the farthest void of the bin is in front of all the sources by dz_buffer
    assert st.ZSRC_MIN - 0.3 >= st.cfg.dz_buffer - 1e-9


def test_empty_disc_in_the_map_is_an_error():
    # a void in the part of the sky without sources, or with its disc empty
    gal = make_galaxies(on_centres=True)
    setup(gal, make_cube(gal))
    with pytest.raises(ValueError, match="No sources within"):
        st.partial_profile_map((RV0, RA0 + 120.0, -DEC0, Z0))
    st.partial_profile_map((RV0, RA0, DEC0, Z0))  # the void of the other tests is fine


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

    src = io.read_sources_map(path)
    assert src["order"] == ORDER and src["nest"]
    assert src["nobj"].shape == cube["nobj"].shape

    low = io.read_sources_map(path, order=ORDER - 2)
    assert low["order"] == ORDER - 2 and low["nside"] == NSIDE_MAP // 4
    np.testing.assert_array_equal(
        low["nobj"].sum(axis=0), cube["nobj"].sum(axis=0)
    )
    with pytest.raises(ValueError, match="higher"):
        io.read_sources_map(path, order=ORDER + 1)
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
