"""Readers of the void files and layout of the results file.

Run with: python -m pytest lensing/tests/test_io.py
"""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from lensing import io
from lensing.jackknife import PROFILE_NAMES

LCOLS = {"rv": 0, "ra": 1, "dec": 2, "z": 3, "rho1": 7, "delta": 8, "flag": 11}
SEL = dict(Rv_min=10.0, Rv_max=20.0, z_min=0.1, z_max=0.2, delta_min=-1.0, delta_max=100.0)


def make_table(n=2000, seed=0, has_id=False):
    r = np.random.default_rng(seed)
    L = np.zeros((12, n))
    L[0] = r.uniform(5, 25, n)  # Rv
    L[1] = r.uniform(0, 360, n)  # ra
    L[2] = np.rad2deg(np.arcsin(r.uniform(-1, 1, n)))  # dec
    L[3] = r.uniform(0.05, 0.3, n)  # z
    L[7] = r.uniform(-1.0, 0.5, n)  # rho1
    L[8] = r.uniform(-1, 120, n)  # delta
    L[11] = r.integers(0, 3, n)  # flag
    if has_id:
        L = np.vstack([np.arange(n), L])
    return L


def write(tmp_path, L, name="voids.dat"):
    path = tmp_path / name
    np.savetxt(path, L.T)
    return path


def expected(L, off=0, flag=2):
    m = (
        (L[off + 0] >= 10) & (L[off + 0] < 20) & (L[off + 3] >= 0.1) & (L[off + 3] < 0.2)
        & (L[off + 7] >= -1.0) & (L[off + 7] < 0.0) & (L[off + 8] >= -1) & (L[off + 8] < 100)
        & (L[off + 11] >= flag)
    )
    return m


def test_read_sparkling_cuts_with_the_columns_of_the_config(tmp_path):
    L = make_table()
    path = write(tmp_path, L)
    m = expected(L)
    assert 20 < m.sum() < 200
    voids = io.read_sparkling(path, **SEL, columns=LCOLS)
    assert voids.shape == (12, m.sum())
    np.testing.assert_allclose(voids[0], L[0][m], rtol=1e-6)
    # default columns: the same
    np.testing.assert_array_equal(io.read_sparkling(path, **SEL), voids)

    # the cuts use the rho1 and flag columns of the config: swap them in the config
    swapped = dict(LCOLS, rho1=8, delta=7)
    assert io.read_sparkling(path, **SEL, columns=swapped).shape[1] != voids.shape[1]
    only_flag2 = io.read_sparkling(path, **SEL, flag=2, columns=LCOLS)
    any_flag = io.read_sparkling(path, **SEL, flag=0, columns=LCOLS)
    assert any_flag.shape[1] > only_flag2.shape[1]


def test_file_with_an_id_column(tmp_path):
    L = make_table(has_id=True)
    path = write(tmp_path, L)
    m = expected(L, off=1)
    voids = io.read_sparkling(path, **SEL, has_id=True)
    assert voids.shape == (13, m.sum())
    np.testing.assert_allclose(voids[1], L[1][m], rtol=1e-6)  # Rv, after the ID
    # the flag moves with the others: an old file layout shifted only some of them
    assert np.all(voids[12] >= 2)


def test_wrong_columns_stop_the_run(tmp_path):
    path = write(tmp_path, make_table())
    # the template used to assume an ID column: every column shifted by one
    shifted = {k: v + 1 for k, v in LCOLS.items() if k != "flag"}
    with pytest.raises(ValueError, match="don't look like coordinates"):
        io.read_sparkling(path, **SEL, columns=dict(LCOLS, **shifted))
    # column that doesn't exist
    with pytest.raises(ValueError, match="columns"):
        io.read_sparkling(path, **SEL, columns=dict(LCOLS, flag=40))


# --- results file ----------------------------------------------------------------------

NB, NJK = 5, 4


def fake_profiles():
    r = np.random.default_rng(0)
    profiles = {}
    for suffix in ("", "_raw", "_rand"):
        for name in PROFILE_NAMES:
            profiles[name + suffix] = r.normal(size=(NJK + 1, NB))
    return profiles


def fake_cfg(**kw):
    cfg = SimpleNamespace(
        voidthr="09", sample="t", NBINS=NB, binning="lin", source_mode="map", addnoise=False,
        lensname="v.dat", sourcename="cube.fits", RIN=0.1, ROUT=3.0, NJK=NJK, jk_seed=0,
        rand_factor=10, rand_seed=42, lcols=LCOLS, gamma_sign=(-1.0, -1.0),
    )
    cfg.__dict__.update(kw)
    return cfg


def test_output_path_and_existing_file(tmp_path, capsys):
    cfg = fake_cfg()
    path = io.output_path(cfg, "GR", (10.0, 20.0), (0.1, 0.2), "R")
    assert str(path) == "lensing/results/lensing_GR_thr09_t_N5_Rv10-20_z010-020_typeR_binlin_map.fits"
    assert not str(io.output_path(fake_cfg(source_mode="catalog"), "GR", (10.0, 20.0), (0.1, 0.2), "R")).endswith("_map.fits")

    existing = tmp_path / "x.fits"
    io.check_output_exists(existing)  # nothing there
    existing.write_text("")
    with pytest.raises(OSError, match="already exists"):
        io.check_output_exists(existing)
    io.check_output_exists(existing, overwrite=True)
    assert "WARNING" in capsys.readouterr().out


def test_results_file_layout(tmp_path):
    cfg = fake_cfg()
    voids = make_table(50)
    profiles = fake_profiles()
    head = io.make_header(
        cfg, voids, (10.0, 20.0), (0.1, 0.2), (-1.0, 100.0), "mixed", 500, extra={"nside": 128}
    )
    centres = np.linspace(0.2, 2.8, NB)
    jk_centres = np.column_stack([np.linspace(0, 300, NJK), np.linspace(-60, 60, NJK)])
    N, N_rand = np.full((NJK + 1, NB), 7.0), np.full((NJK + 1, NB), 70.0)
    path = tmp_path / "out.fits"
    io.write_profiles(path, head, centres, profiles, N, N_rand, jk_centres)

    with fits.open(path) as h:
        names = [x.name for x in h]
        assert names[:2] == ["PRIMARY", "PROFILES"] and names[-1] == "JK_CENTRES"
        for n in (p + s for s in ("", "_raw", "_rand") for p in PROFILE_NAMES):
            assert f"COV_{n}".upper() in names and f"JACK_{n}".upper() in names
            assert h[f"JACK_{n}".upper()].data.shape == (NJK, NB)
        assert len(names) == 2 + 18 + 1
        t = h["profiles"].data
        assert list(t.names) == ["R"] + [p + s for s in ("", "_raw", "_rand") for p in PROFILE_NAMES] + ["N", "N_rand"]
        np.testing.assert_allclose(t["Sigma"], profiles["Sigma"][0])
        np.testing.assert_allclose(t["R"], centres)
        np.testing.assert_array_equal(t["N_rand"], N_rand[0])
        hd = h[0].header
        assert hd["NRAND"] == 500 and hd["RANDFAC"] == 10 and hd["RANDSEED"] == 42
        assert hd["RANDSUB"] is True and hd["JKSEED"] == 0 and hd["NSIDE"] == 128
        assert hd["GSIGN1"] == -1.0 and hd["GSIGN2"] == -1.0
        assert hd["NVOIDS"] == 50 and hd["VOIDTYPE"] == "mixed"
        reg = h["jk_centres"].data
        np.testing.assert_allclose(reg["ra"], jk_centres[:, 0])
        np.testing.assert_allclose(reg["dec"], jk_centres[:, 1])
