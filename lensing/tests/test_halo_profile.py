"""Halos: the catalog and map estimators on the comoving scale (no rescaling by Rv), the
reader and checks of the halo table, the config, and a stack written to a file.

Run with: python -m pytest lensing/tests/test_halo_profile.py
"""

import os
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
import toml
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table

import lensing.stacking as st
from lensing import cosmology, io, jackknife
from lensing.geometry import GAMMA_SIGN, bin_centres, bin_edges, eq2p2
from lensing.settings import Config
from shearmap import table2cube as t2c

COSMO = FlatLambdaCDM(H0=100.0, Om0=0.3089)  # h = 1
ORDER = 8  # order of the synthetic cube
NSIDE_MAP = 2**ORDER
NSIDE_CAT = 64
RA0, DEC0 = 40.0, 10.0
Z_L = 0.25  # halos in the bin [0.2, 0.3): their sources start at 0.3 + 0.1 = 0.4
Z_SRC_MIN = 0.4
RIN, ROUT, NBINS = 0.2, 5.0, 6  # comoving Mpc/h, log bins
M_TRUE = 1e13  # Msun
HCOLS = dict(
    ra="ra_gal",
    dec="dec_gal",
    z="true_redshift_gal",
    mag="abs_mag_r01_evolved",
    lm="halo_lm",
    color="color_kind",
    richness="true_richness",
)


def setup_cfg(**kw):
    cfg = SimpleNamespace(
        NBINS=NBINS,
        RIN=RIN,
        ROUT=ROUT,
        NSIDE=NSIDE_CAT,
        binning="log",
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
        gamma_sign=GAMMA_SIGN,
        weighting="invsc",
        zbins=[(0.2, 0.3)],
        lens_type="halos",
        source_mode="catalog",
    )
    cfg.__dict__.update(kw)
    st.cfg = cfg
    st.cosmo = COSMO
    st.set_binning()
    return cfg


def index_catalog(cat):
    """Pixel index of a catalog of sources, as `make_pix2idx_dict` builds it"""
    cat["pix"] = hp.ang2pix(NSIDE_CAT, cat["ra"], cat["dec"], lonlat=True)
    cat.sort("pix")
    upix, first = np.unique(cat["pix"], return_index=True)
    last = np.append(first[1:], len(cat))
    st.PIX_TO_IDX = {int(p): (a, b) for p, a, b in zip(upix, first, last)}
    st.SOURCE = cat


# --- point mass on the comoving bins ---------------------------------------------------


def dsigma_point(R_com, mass=M_TRUE):
    """Point-mass DSigma [Msun/pc^2] at the comoving radius R_com [Mpc/h] of the halo: the
    physical radius is R_com / (1 + z_l)"""
    R_pc = 1e6 * R_com / (1.0 + Z_L)
    return mass / (np.pi * R_pc**2)


# one ring per bin, just inside its lower edge: a scale error of (1 + z) moves them to the
# bin below
RINGS = bin_edges(RIN, ROUT, NBINS, "log")[:-1] * 1.05


def ring_sources(rng, n=300):
    """Sources on rings in the bins, sheared by a point mass (gamma = DSigma / Sigma_crit,
    purely tangential), and foreground sources between the lens and the sources' cut, with
    a large shear that must not be used"""
    chi_l = COSMO.comoving_distance(Z_L).value
    ra, dec, z = [], [], []
    for R in RINGS:
        phi = rng.uniform(0, 2 * np.pi, n)
        theta = R / chi_l  # rad: the comoving radius at the lens distance
        ra.append(RA0 + np.rad2deg(theta * np.cos(phi)) / np.cos(np.deg2rad(DEC0)))
        dec.append(DEC0 + np.rad2deg(theta * np.sin(phi)))
        z.append(rng.uniform(0.45, 1.0, n))
    ra, dec, z = np.concatenate(ra), np.concatenate(dec), np.concatenate(z)

    _, ang = eq2p2(np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(RA0), np.deg2rad(DEC0))
    # each source has the DSigma of its ring
    a = dsigma_point(np.repeat(RINGS, n)) / cosmology.sigma_crit(COSMO, Z_L, z)
    g1, g2 = a * np.cos(2 * ang), a * np.sin(2 * ang)

    nfg = 4 * n
    return Table(
        {
            "ra": np.concatenate([ra, np.full(nfg, RA0 + 0.02)]),
            "dec": np.concatenate([dec, np.full(nfg, DEC0)]),
            "z": np.concatenate([z, np.full(nfg, 0.35)]),
            "gamma1": np.concatenate([g1, np.full(nfg, 1.0)]),
            "gamma2": np.concatenate([g2, np.zeros(nfg)]),
            "kappa": np.zeros(len(ra) + nfg),
        }
    )


def test_point_mass_on_the_comoving_bins():
    setup_cfg()
    index_catalog(ring_sources(np.random.default_rng(3)))
    st.ZSRC_MIN = Z_SRC_MIN

    _, t, x, W, N = st.partial_profile_halo((RA0, DEC0, Z_L))

    # every ring is in its bin; the foreground sources (z = 0.35) are left out
    np.testing.assert_array_equal(N, np.full(NBINS, 300.0))
    np.testing.assert_allclose(t / W, dsigma_point(RINGS), rtol=2e-3)
    assert np.abs(x).max() < 1e-10 * np.abs(t).max()  # no cross component


def test_proper_scale_would_put_the_rings_in_other_bins(monkeypatch):
    # negative control: with the proper scale (1 + z) the rings land in other bins
    setup_cfg()
    index_catalog(ring_sources(np.random.default_rng(3)))
    st.ZSRC_MIN = Z_SRC_MIN
    monkeypatch.setattr(
        st, "deg_per_mpch", lambda cosmo, z: (1.0 + z) * cosmology.deg_per_mpch(cosmo, z)
    )
    _, t, _, W, N = st.partial_profile_halo((RA0, DEC0, Z_L))
    assert not np.array_equal(N, np.full(NBINS, 300.0))


# --- map vs catalog ---------------------------------------------------------------------


def make_galaxies(cap=3.0, ngal=40, seed=1):
    """Galaxies at the pixel centres of the cube around the halo: behind the bin (z >= 0.4)
    and in front of it (0.25 <= z < 0.35), with random shear and convergence"""
    r = np.random.default_rng(seed)
    pix = hp.query_disc(NSIDE_MAP, hp.ang2vec(RA0, DEC0, lonlat=True), np.deg2rad(cap), nest=True)
    ra_p, dec_p = hp.pix2ang(NSIDE_MAP, pix, nest=True, lonlat=True)
    ra = np.concatenate([np.repeat(ra_p, ngal), np.repeat(ra_p, ngal // 4)])
    dec = np.concatenate([np.repeat(dec_p, ngal), np.repeat(dec_p, ngal // 4)])
    z = np.concatenate(
        [r.uniform(0.4, 1.0, len(pix) * ngal), r.uniform(0.25, 0.35, len(pix) * ngal // 4)]
    )
    n = len(ra)
    return Table(
        {
            "ra": ra,
            "dec": dec,
            "z": z,
            "gamma1": r.normal(0, 0.05, n),
            "gamma2": r.normal(0, 0.05, n),
            "kappa": r.normal(0.02, 0.03, n),
        }
    )


def make_cube(gal):
    """The sums that the CosmoHub query + table2cube produce"""
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


def test_halo_map_matches_catalog():
    gal = make_galaxies()
    setup_cfg()
    cube = make_cube(gal)

    st.SOURCE = cube
    st.collapse_slices(0.3)  # the bin [0.2, 0.3): slices from 0.4
    assert st.ZSRC_MIN == pytest.approx(Z_SRC_MIN)
    map_res = st.partial_profile_halo_map((RA0, DEC0, Z_L))

    index_catalog(gal)
    st.ZSRC_MIN = Z_SRC_MIN
    cat_res = st.partial_profile_halo((RA0, DEC0, Z_L))

    # the galaxies at the pixel centres: the two agree
    k_c, t_c, x_c, w_c, n_c = cat_res
    k_m, t_m, x_m, w_m, n_m = map_res
    assert n_c.sum() > 100  # the disc of a halo is ~1 pixel across: few galaxies, some bins empty
    # the sources in front of the cut are left out of both
    deg = cosmology.deg_per_mpch(COSMO, Z_L)
    rad, _ = eq2p2(np.deg2rad(gal["ra"]), np.deg2rad(gal["dec"]), np.deg2rad(RA0), np.deg2rad(DEC0))
    R = np.rad2deg(rad) / deg
    behind = gal["z"] >= Z_SRC_MIN
    assert n_c.sum() == np.count_nonzero(behind & (R >= RIN) & (R < ROUT))

    np.testing.assert_array_equal(n_m, n_c)
    np.testing.assert_allclose(w_m, w_c, rtol=1e-4)
    scale = max(np.abs(k_c).max(), np.abs(t_c).max())
    for c, m in ((k_c, k_m), (t_c, t_m), (x_c, x_m)):
        np.testing.assert_allclose(m, c, rtol=1e-4, atol=1e-5 * scale)


def test_collapse_uses_the_bin_of_the_halos():
    # a halo bin up to 0.5 has its sources from 0.6 (slice edge): the one of [0.4, 0.5)
    setup_cfg(zbins=[(0.4, 0.5)])
    cube = make_cube(make_galaxies())
    st.SOURCE = cube
    assert st.collapse_slices(0.5) == 4  # slices from z = 0.6
    assert st.ZSRC_MIN == pytest.approx(0.6)


def test_halo_zbins_are_checked_against_the_cube(capsys):
    setup_cfg()
    st.SOURCE = make_cube(make_galaxies())
    st.cfg.zbins = [(0.3, 0.5)]  # z_high + 0.1 = 0.6: a slice edge
    st.cfg.dz_buffer = 0.1
    st.check_map_zbins()
    out = capsys.readouterr().out
    assert "sources z >= 0.60" in out and "✗" not in out
    st.cfg.zbins = [(0.5, 0.6)]  # z_high + 0.1 = 0.7: no slice
    with pytest.raises(ValueError, match="slice"):
        st.check_map_zbins()


def test_resolution_check_of_halos(capsys):
    setup_cfg()
    st.SOURCE = dict(make_cube(make_galaxies()), nside=NSIDE_MAP)  # 0.23 deg pixels
    st.cfg.zbins = [(0.2, 0.3)]
    st.check_map_resolution(1.0)  # halos: the bins are in Mpc/h
    out = capsys.readouterr().out
    assert "✗" in out and "WARNING" in out


def test_radial_scale_of_the_lens_type():
    setup_cfg(lens_type="halos")
    assert st.radial_scale() == 1.0
    setup_cfg(lens_type="voids", rvbins=[(10.0, 20.0), (6.0, 8.0)])
    assert st.radial_scale() == 6.0


def test_empty_bins_are_reported(capsys):
    centres = np.array([0.1, 0.2, 0.4])
    N = np.array([[5.0, 0.0, 3.0], [5.0, 0.0, 3.0]])  # full sample and jackknife rows
    st.warn_empty_bins(centres, N, np.ones((2, 3)))
    out = capsys.readouterr().out
    assert "1 radial bin(s)" in out and "0.200" in out and "NaN" in out
    st.warn_empty_bins(centres, np.ones((2, 3)), np.ones((2, 3)))
    assert capsys.readouterr().out == ""


# --- reader and checks of the halo table ----------------------------------------------


def write_halos(path, n=400, seed=5, **changes):
    r = np.random.default_rng(seed)
    cols = {
        "ra_gal": r.uniform(0.0, 360.0, n),
        "dec_gal": np.rad2deg(np.arcsin(r.uniform(-1, 1, n))),
        "true_redshift_gal": r.uniform(0.1, 0.6, n),
        "abs_mag_r01_evolved": r.uniform(-24.0, -20.0, n),
        "halo_lm": r.uniform(12.5, 15.0, n),
        "color_kind": r.integers(0, 3, n),
        "true_richness": r.integers(1, 10, n),
        "kind": np.zeros(n, dtype=int),
    }
    cols.update(changes)
    Table(cols).write(path, format="fits", overwrite=True)
    return path


def test_read_halos_cuts_and_layout(tmp_path):
    path = write_halos(tmp_path / "h.fits")
    t = Table.read(path)
    z, mag = t["true_redshift_gal"], t["abs_mag_r01_evolved"]
    m = (z >= 0.2) & (z < 0.3) & (mag >= -23.0) & (mag < -22.0)

    arr = io.read_halos(path, 0.2, 0.3, -23.0, -22.0, columns=HCOLS)
    assert arr.shape == (len(io.HALO_COLUMNS), m.sum()) and m.sum() > 10
    np.testing.assert_allclose(arr[io.HALO_COLUMNS["z"]], z[m])
    np.testing.assert_allclose(arr[io.HALO_COLUMNS["mag"]], mag[m])
    np.testing.assert_allclose(arr[io.HALO_COLUMNS["lm"]], t["halo_lm"][m])
    np.testing.assert_allclose(arr[io.HALO_COLUMNS["ra"]], t["ra_gal"][m])

    red = io.read_halos(path, 0.2, 0.3, -23.0, -22.0, color=0, columns=HCOLS)
    np.testing.assert_array_equal(red[io.HALO_COLUMNS["color"]], 0)
    assert red.shape[1] == np.count_nonzero(m & (t["color_kind"] == 0))

    mass = (t["halo_lm"] >= 13.0) & (t["halo_lm"] < 14.0)
    lm = io.read_halos(path, 0.2, 0.3, -23.0, -22.0, lm_min=13.0, lm_max=14.0, columns=HCOLS)
    assert lm.shape[1] == np.count_nonzero(m & mass)

    empty = io.read_halos(path, 0.2, 0.3, -30.0, -29.0, columns=HCOLS)
    assert empty.shape == (len(io.HALO_COLUMNS), 0)


def test_halo_table_checks(tmp_path):
    with pytest.raises(ValueError, match="satellites"):
        io.read_halos(
            write_halos(tmp_path / "sat.fits", kind=np.ones(400, dtype=int)),
            0.0, 1.0, -30.0, -10.0, columns=HCOLS,
        )
    with pytest.raises(ValueError, match="not in the halo file"):
        io.read_halos(
            write_halos(tmp_path / "ok.fits"), 0.0, 1.0, -30.0, -10.0,
            columns=dict(HCOLS, mag="M_r"),
        )
    with pytest.raises(ValueError, match="absolute magnitude"):
        io.read_halos(
            write_halos(tmp_path / "mag.fits", abs_mag_r01_evolved=np.full(400, 5.0)),
            0.0, 1.0, -30.0, -10.0, columns=HCOLS,
        )
    with pytest.raises(ValueError, match="coordinates"):
        io.read_halos(
            write_halos(tmp_path / "ra.fits", ra_gal=np.full(400, 400.0)),
            0.0, 1.0, -30.0, -10.0, columns=HCOLS,
        )


# --- config -----------------------------------------------------------------------------


def load_cfg(tmp_path, edit=lambda c: None, gravity="gr"):
    c = toml.load("lensing/config.tmplt.toml")
    edit(c)
    path = tmp_path / "cfg.toml"
    path.write_text(toml.dumps(c))
    return Config(str(path), gravity)


def test_halo_config(tmp_path):
    def halos(c):
        c["run"]["lens_type"] = "halos"

    cfg = load_cfg(tmp_path, halos)
    assert cfg.lens_type == "halos"
    assert cfg.lensname == "~/cats/L768/l768_gr_halos.fits"
    assert cfg.zbins == [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    assert cfg.magbins == [(-30.0, -22.0)] and cfg.lmbins == [None]
    assert cfg.lcols == io.HALO_COLUMNS and cfg.hcols["mag"] == "abs_mag_r01_evolved"
    assert cfg.hcols["richness"] == "true_richness"  # as in the CosmoHub download
    assert (cfg.RIN, cfg.ROUT, cfg.NBINS, cfg.binning) == (0.05, 5.0, 15, "log")
    assert cfg.color == "all"
    assert st.n_stacks(cfg) == 4

    def edit(c):
        halos(c)
        c["halos"]["lm_edges"] = [13.0, 14.0]
        c["halos"]["color"] = 0

    cfg = load_cfg(tmp_path, edit, gravity="mg")
    assert cfg.lensname == "~/cats/L768/l768_mg_halos.fits"
    assert cfg.lmbins == [(13.0, 14.0)] and cfg.color == 0
    assert st.n_stacks(cfg) == 4


def test_voids_keep_the_profile_bins(tmp_path):
    cfg = load_cfg(tmp_path)
    assert cfg.lens_type == "voids"
    assert (cfg.RIN, cfg.ROUT, cfg.binning) == (0.1, 3.0, "lin")
    assert cfg.lcols["rv"] == 0 and cfg.rvbins == [(10.0, 20.0)]


def test_bad_lens_type_and_colour(tmp_path):
    def bad_type(c):
        c["run"]["lens_type"] = "clusters"

    with pytest.raises(ValueError, match="lens_type"):
        load_cfg(tmp_path, bad_type)

    def bad_colour(c):
        c["run"]["lens_type"] = "halos"
        c["halos"]["color"] = 3

    with pytest.raises(ValueError, match="color"):
        load_cfg(tmp_path, bad_colour)


# --- a stack written to a file ------------------------------------------------------------


@pytest.mark.parametrize("source_mode", ["catalog", "map"])
def test_stack_writes_the_halo_profiles(tmp_path, monkeypatch, source_mode):
    if jackknife.kmeans_sample is None:
        pytest.skip("kmeans_radec is not installed")
    monkeypatch.chdir(tmp_path)
    os.makedirs("lensing/results")

    # the cap is wide: the map jackknife samples the pixels of the whole footprint
    gal = make_galaxies(cap=8.0)
    if source_mode == "map":
        setup_cfg(source_mode="map")
        st.SOURCE = make_cube(gal)
    else:
        setup_cfg()
        index_catalog(gal)

    # halos around the centre (within 1.5 deg), in the bin and outside it
    r = np.random.default_rng(7)
    n = 60
    rr, ph = 1.5 * np.sqrt(r.random(n)), 2 * np.pi * r.random(n)
    halos = write_halos(
        tmp_path / "halos.fits",
        n=n,
        seed=7,
        ra_gal=RA0 + rr * np.cos(ph) / np.cos(np.deg2rad(DEC0)),
        dec_gal=DEC0 + rr * np.sin(ph),
        true_redshift_gal=r.uniform(0.2, 0.3, n),
        abs_mag_r01_evolved=r.uniform(-24.0, -21.0, n),
    )
    t = Table.read(halos)
    inbin = (t["abs_mag_r01_evolved"] >= -23.0) & (t["abs_mag_r01_evolved"] < -22.0)
    cfg = st.cfg
    cfg.__dict__.update(
        lensname=str(halos),
        sourcename="sources",
        sample="t",
        NJK=4,
        jk_seed=0,
        rand_factor=2,
        rand_seed=42,
        lcols=io.HALO_COLUMNS,
        hcols=HCOLS,
        color="all",
        magbins=[(-23.0, -22.0)],
        lmbins=[None],
        overwrite=True,
        ncores=1,
    )
    st.REGIONS = jackknife.fit_regions(*st.jackknife_sample(), cfg.NJK, cfg.jk_seed)

    st.stacking_halos((0.2, 0.3), (-23.0, -22.0), None, "GR")

    name = "lensing/results/halos_GR_t_N6_mag-23.0to-22.0_z020-030_colall_binlog"
    name += "_map.fits" if source_mode == "map" else ".fits"
    with fits.open(name) as h:
        head = h[0].header
        nhalos = np.count_nonzero(
            (t["true_redshift_gal"] >= 0.2) & (t["true_redshift_gal"] < 0.3) & inbin
        )
        assert nhalos > 0 and head["NHALOS"] == nhalos
        assert head["RUNITS"] == "comoving" and head["ZSRCMIN"] == pytest.approx(Z_SRC_MIN)
        assert head["NRAND"] == 2 * nhalos
        assert head["SRCMODE"] == source_mode
        prof = h["profiles"].data
        np.testing.assert_allclose(prof["R"], bin_centres(RIN, ROUT, NBINS, "log"))
        # NaN exactly where the sources or the randoms of the bin are missing
        empty = (prof["N"] == 0) | (prof["N_rand"] == 0)
        assert np.all(np.isnan(prof["DSigma_t"]) == empty)
        assert np.all(np.isfinite(prof["DSigma_t"][~empty]))
        assert h["COV_DSIGMA_T"].data.shape == (NBINS, NBINS)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
