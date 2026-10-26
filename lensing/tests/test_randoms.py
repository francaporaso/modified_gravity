"""Tests of the random voids that are subtracted from the profiles: generation, footprint,
the estimator (voids - randoms) and the jackknife covariances of the saved profiles.

Run with: python -m pytest lensing/tests/test_randoms.py
"""

from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
import toml

import lensing.stacking as st
from lensing import io, randoms
from lensing.funcs import cov_matrix
from lensing.jackknife import reduce_profiles
from lensing.settings import Config

LCOLS = {"rv": 0, "ra": 1, "dec": 2, "z": 3, "delta": 8}
everywhere = lambda ra, dec: np.ones(len(ra), dtype=bool)


def gen(voids, factor, rng, footprint):
    return randoms.generate_randoms(voids, factor, rng, footprint, LCOLS["ra"], LCOLS["dec"])


def make_voids(n=300, seed=0):
    # 12 columns as read_sparkling returns them, with a different value in every row
    r = np.random.default_rng(seed)
    voids = r.normal(size=(12, n))
    voids[0] = r.uniform(10, 20, n)  # Rv
    voids[1] = r.uniform(0, 360, n)  # ra
    voids[2] = np.rad2deg(np.arcsin(r.uniform(-1, 1, n)))  # dec
    voids[3] = r.uniform(0.1, 0.2, n)  # z
    voids[8] = r.uniform(-1, 100, n)  # delta
    return voids


@pytest.fixture(autouse=True)
def config():
    st.cfg = SimpleNamespace(lcols=LCOLS, rand_seed=42, source_mode="catalog", NSIDE=8)


def test_randoms_have_the_rv_z_pairs_of_the_voids():
    voids = make_voids()
    # another seed than make_voids: with the same one the two share their uniform draws
    rands = gen(voids, 10, np.random.default_rng(123), everywhere)
    assert rands.shape == (12, 3000)

    # every random is a void with another position: (Rv, z, delta, ...) are those of a void
    keep = [c for c in range(12) if c not in (LCOLS["ra"], LCOLS["dec"])]
    void_rows = {tuple(v) for v in voids[keep].T}
    assert all(tuple(r) in void_rows for r in rands[keep].T)
    # drawn with replacement and uniformly: each void ~10 times
    _, counts = np.unique(rands[LCOLS["rv"]], return_counts=True)
    assert len(counts) > 0.95 * voids.shape[1]
    assert counts.mean() == pytest.approx(10.0, rel=0.1)
    # the positions are new
    assert not set(rands[LCOLS["ra"]]) & set(voids[LCOLS["ra"]])
    # and the input is not modified
    np.testing.assert_array_equal(voids, make_voids())


def test_random_positions_are_uniform_on_the_sphere():
    voids = make_voids(n=500)
    rands = gen(voids, 100, np.random.default_rng(1), everywhere)
    n = rands.shape[1]
    ra, dec = rands[LCOLS["ra"]], rands[LCOLS["dec"]]
    assert ra.min() >= 0 and ra.max() < 360 and np.abs(dec).max() <= 90
    # uniform in ra and in sin(dec): equal-area bands hold the same number of points
    for x, edges in ((ra, np.linspace(0, 360, 11)), (np.sin(np.deg2rad(dec)), np.linspace(-1, 1, 11))):
        counts, _ = np.histogram(x, edges)
        np.testing.assert_allclose(counts, n / 10, rtol=0.06)
    # uniform in dec would give a pile-up at the poles: mean |sin dec| is 1/2
    assert np.abs(np.sin(np.deg2rad(dec))).mean() == pytest.approx(0.5, abs=0.01)


def test_randoms_stay_on_the_footprint():
    voids = make_voids()
    cap = lambda ra, dec: (dec > 30.0) & (ra < 90.0)
    rands = gen(voids, 10, np.random.default_rng(0), cap)
    assert rands.shape[1] == 3000  # redrawn, not dropped
    assert np.all(cap(rands[LCOLS["ra"]], rands[LCOLS["dec"]]))
    # still uniform inside the cap
    sin_dec = np.sin(np.deg2rad(rands[LCOLS["dec"]]))
    assert sin_dec.mean() == pytest.approx(0.5 * (0.5 + 1.0), abs=0.02)

    with pytest.raises(RuntimeError, match="outside the source footprint"):
        gen(voids, 1, np.random.default_rng(0), lambda ra, dec: dec > 89.9999999)
    with pytest.raises(ValueError, match="No voids"):
        gen(voids[:, :0], 10, np.random.default_rng(0), everywhere)


def test_randoms_are_reproducible_and_seed_dependent():
    voids = make_voids()
    args = (z0, z1, r0, r1, d0, d1) = (0.1, 0.2, 10.0, 20.0, -1.0, 100.0)
    a = gen(voids, 5, randoms.stack_rng(42, *args), everywhere)
    b = gen(voids, 5, randoms.stack_rng(42, *args), everywhere)
    np.testing.assert_array_equal(a, b)

    # another void bin or void type: another stream. The type has delta_min = -1 (negative)
    c = gen(voids, 5, randoms.stack_rng(42, z0, z1, r0, r1, -1.0, 0.0), everywhere)
    d = gen(voids, 5, randoms.stack_rng(42, z0, z1, 20.0, 30.0, d0, d1), everywhere)
    assert not np.array_equal(a[LCOLS["ra"]], c[LCOLS["ra"]])
    assert not np.array_equal(a[LCOLS["ra"]], d[LCOLS["ra"]])

    e = gen(voids, 5, randoms.stack_rng(43, *args), everywhere)
    assert not np.array_equal(a[LCOLS["ra"]], e[LCOLS["ra"]])


def test_footprint_of_the_map_and_of_the_catalog():
    nside = 8
    pix = hp.query_disc(nside, hp.ang2vec(40.0, 10.0, lonlat=True), np.deg2rad(15), nest=True)
    nobj = np.zeros(hp.nside2npix(nside), dtype=np.int64)
    nobj[pix] = 3
    inside = (40.0, 10.0)
    outside = (220.0, -10.0)
    ra, dec = np.array([inside[0], outside[0]]), np.array([inside[1], outside[1]])

    st.cfg.source_mode = "map"
    st.SOURCE = {"nside": nside, "nest": True}
    st.MAPS = {"nobj": nobj}
    np.testing.assert_array_equal(st.in_footprint(ra, dec), [True, False])

    # catalog: the pixels of the index (ring ordering, nside of the config)
    st.cfg.source_mode = "catalog"
    st.cfg.NSIDE = nside
    st.PIX_TO_IDX = {int(hp.ang2pix(nside, *inside, lonlat=True)): (0, 5)}
    np.testing.assert_array_equal(st.in_footprint(ra, dec), [True, False])


# --- estimator and covariances ---------------------------------------------------------

NBINS, NJK = 4, 6


def fake_profiles(n, seed, regions, offsets):
    """Per-void sums of a profile function: kappa, gamma_t, gamma_x, W, N. Each void is in
    a region; the kappa and gamma of every void have `offsets[region]` per unit weight,
    which is what a large-scale mode over the region does to voids and randoms alike."""
    r = np.random.default_rng(seed)
    nsrc = r.integers(50, 150, (n, NBINS)).astype(float)
    w = nsrc * r.uniform(0.5, 1.5, (n, NBINS))
    shift = offsets[regions][:, None] * w
    kappa = shift + r.normal(0, 0.5, (n, NBINS)) * np.sqrt(w)
    gt = 0.3 * w + shift + r.normal(0, 0.5, (n, NBINS)) * np.sqrt(w)
    gx = r.normal(0, 0.5, (n, NBINS)) * np.sqrt(w)
    return [(kappa[i], gt[i], gx[i], w[i], nsrc[i]) for i in range(n)]


def setup_fake(nvoids=400, factor=10, offset_scale=0.0):
    r = np.random.default_rng(5)
    kidx = r.integers(0, NJK, nvoids)
    kidx_rand = r.integers(0, NJK, factor * nvoids)
    offsets = r.normal(0, offset_scale, NJK)
    resmap = fake_profiles(nvoids, 1, kidx, offsets)
    randmap = fake_profiles(factor * nvoids, 2, kidx_rand, offsets)
    return resmap, randmap, kidx, kidx_rand


def test_estimator_is_voids_minus_randoms_in_every_jackknife_sample():
    resmap, randmap, kidx, kidx_rand = setup_fake()
    prof, n, n_rand = reduce_profiles(resmap, randmap, kidx, kidx_rand, NJK)

    kappa, gt, gx, w, nn = (np.vstack(x) for x in zip(*resmap))
    kappa_r, gt_r, gx_r, w_r, nn_r = (np.vstack(x) for x in zip(*randmap))
    for j in range(NJK + 1):
        keep = np.ones(len(kidx), bool) if j == 0 else kidx != j - 1
        keep_r = np.ones(len(kidx_rand), bool) if j == 0 else kidx_rand != j - 1
        for name, v, r_ in (("Sigma", kappa, kappa_r), ("DSigma_t", gt, gt_r), ("DSigma_x", gx, gx_r)):
            raw = v[keep].sum(0) / w[keep].sum(0)
            rand = r_[keep_r].sum(0) / w_r[keep_r].sum(0)
            np.testing.assert_allclose(prof[name + "_raw"][j], raw)
            np.testing.assert_allclose(prof[name + "_rand"][j], rand)
            np.testing.assert_allclose(prof[name][j], raw - rand)
        np.testing.assert_allclose(n[j], nn[keep].sum(0))
        np.testing.assert_allclose(n_rand[j], nn_r[keep_r].sum(0))
    assert prof["Sigma"].shape == (NJK + 1, NBINS)


def test_a_constant_kappa_shifts_the_raw_profile_not_the_subtracted_one():
    # a constant kappa0 in every source: it's in the voids and in the randoms alike
    resmap, randmap, kidx, kidx_rand = setup_fake()
    base, _, _ = reduce_profiles(resmap, randmap, kidx, kidx_rand, NJK)
    kappa0 = 0.03

    def shifted(res):
        return [(k + kappa0 * w, gt, gx, w, n) for k, gt, gx, w, n in res]

    prof, _, _ = reduce_profiles(shifted(resmap), shifted(randmap), kidx, kidx_rand, NJK)
    np.testing.assert_allclose(prof["Sigma"], base["Sigma"], atol=1e-12)
    np.testing.assert_allclose(prof["Sigma_raw"], base["Sigma_raw"] + kappa0, atol=1e-12)
    np.testing.assert_allclose(prof["Sigma_rand"], base["Sigma_rand"] + kappa0, atol=1e-12)
    # DSigma_t has the same behaviour for a constant shear, and the other profiles are intact
    np.testing.assert_allclose(prof["DSigma_t"], base["DSigma_t"], atol=1e-12)


def test_region_offsets_common_to_voids_and_randoms_drop_out_of_the_covariance():
    # a large-scale mode per jackknife region (Singh et al. 2017; Davies et al. 2018): it is
    # in the voids and in the randoms of the region, so it only inflates the raw covariance
    resmap, randmap, kidx, kidx_rand = setup_fake(offset_scale=0.3)
    prof, _, _ = reduce_profiles(resmap, randmap, kidx, kidx_rand, NJK)
    err = lambda x: np.sqrt(np.diag(cov_matrix(x[1:])))
    assert np.all(err(prof["Sigma_raw"]) > 5 * err(prof["Sigma"]))
    assert np.all(err(prof["DSigma_t_raw"]) > 5 * err(prof["DSigma_t"]))
    # the cross component has no such mode
    assert np.all(err(prof["DSigma_x_raw"]) < 2 * err(prof["DSigma_x"]))
    # the randoms alone carry the mode, and the subtracted profile is free of it
    assert np.all(err(prof["Sigma_rand"]) > 3 * err(prof["Sigma"]))


def test_each_covariance_is_that_of_its_own_estimator():
    resmap, randmap, kidx, kidx_rand = setup_fake(offset_scale=0.2)
    prof, _, _ = reduce_profiles(resmap, randmap, kidx, kidx_rand, NJK)
    hdus = {h.name: h.data for h in io.profile_hdus(prof)}

    names = [n + s for s in ("", "_raw", "_rand") for n in ("Sigma", "DSigma_t", "DSigma_x")]
    assert set(hdus) == {f"{p}_{n}".upper() for p in ("cov", "jack") for n in names}
    for n in names:
        jack = hdus[f"JACK_{n}".upper()]
        assert jack.shape == (NJK, NBINS)
        np.testing.assert_array_equal(jack, prof[n][1:])
        np.testing.assert_allclose(hdus[f"COV_{n}".upper()], cov_matrix(jack))
    # the subtracted one is not the raw one, and its jackknife is raw - rand sample by sample
    np.testing.assert_allclose(
        hdus["JACK_SIGMA"], hdus["JACK_SIGMA_RAW"] - hdus["JACK_SIGMA_RAND"]
    )
    assert not np.allclose(hdus["COV_SIGMA"], hdus["COV_SIGMA_RAW"])


# --- config ----------------------------------------------------------------------------


def test_config_has_the_randoms_section_and_no_random_files(tmp_path):
    cfg = Config("lensing/config.tmplt.toml", "gr", "09")
    assert cfg.rand_factor == 10 and cfg.rand_seed == 42
    assert not hasattr(cfg, "randsname")

    # an old config with randname and without [randoms] still loads, with the defaults
    old = toml.load("lensing/config.tmplt.toml")
    old["lenses"]["gr"]["randname"] = "~/cats/L768/voids_rand_LCDM_09.dat"
    del old["randoms"]
    (tmp_path / "old.toml").write_text(toml.dumps(old))
    cfg = Config(str(tmp_path / "old.toml"), "gr", "09")
    assert cfg.rand_factor == 10 and cfg.rand_seed == 42

    for bad in ({"factor": 0}, {"factor": 10, "seed": -1}, {"factor": 10, "seed": 1.5}):
        old["randoms"] = bad
        (tmp_path / "bad.toml").write_text(toml.dumps(old))
        with pytest.raises(ValueError, match="randoms"):
            Config(str(tmp_path / "bad.toml"), "gr", "09")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
