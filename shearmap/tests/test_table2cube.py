"""Tests of shearmap/table2cube.py with a synthetic (pix, zbin) table.

Run with: python -m pytest shearmap/tests/test_table2cube.py
"""

import numpy as np
import pytest
from astropy.io import fits

from shearmap import table2cube as t2c

ORDER = 3
NPIX = 12 * 4**ORDER
NZ = len(t2c.ZEDGES) - 1
NGAL = 200_000
rng = np.random.default_rng(1)


def make_galaxies():
    gal = {
        "pix": rng.integers(0, NPIX, NGAL),
        "zbin": rng.integers(0, NZ, NGAL),
        "gamma1": rng.normal(0, 0.02, NGAL),
        "gamma2": rng.normal(0, 0.02, NGAL),
        "kappa": rng.normal(0, 0.02, NGAL),
        "invchi": 1.0 / rng.uniform(400.0, 3000.0, NGAL),
    }
    gal["invchi2"] = gal["invchi"] ** 2
    for g in ("gamma1", "gamma2", "kappa"):
        gal[f"{g}_invchi"] = gal[g] * gal["invchi"]
    return gal


def group(gal, drop=()):
    """what the CosmoHub query does, as a FITS table (`drop`: columns to leave out)"""
    flat = gal["pix"] * NZ + gal["zbin"]
    uflat, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    cols = {"pix": uflat // NZ, "zbin": uflat % NZ, "nobj": cnt}
    for name in t2c.SUMS:
        cols[t2c.table_column(name)] = np.bincount(inv, weights=gal[name])
    fcols = [
        fits.Column(name=k, format="K" if v.dtype.kind == "i" else "D", array=v)
        for k, v in cols.items()
        if k not in drop
    ]
    return fits.BinTableHDU.from_columns(fcols), flat


def write_table(path, **kwargs):
    gal = make_galaxies()
    hdu, flat = group(gal, **kwargs)
    hdu.header["COMMENT"] = "SELECT synthetic FROM nowhere"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
    return gal, flat


def dense(gal, name, flat):
    out = np.bincount(flat, weights=gal[name] * t2c.scale(name), minlength=NPIX * NZ)
    return out.reshape(NPIX, NZ)


def test_roundtrip(tmp_path):
    gal, flat = write_table(tmp_path / "table.fits")
    cube = t2c.build_cube(tmp_path / "table.fits", order=ORDER)
    assert cube["optweight"]
    assert cube["nobj"].dtype == np.int16
    np.testing.assert_array_equal(
        cube["nobj"], np.bincount(flat, minlength=NPIX * NZ).reshape(NPIX, NZ)
    )
    for name in t2c.SUMS:
        assert cube[name].dtype == np.float32
        np.testing.assert_allclose(cube[name], dense(gal, name, flat), atol=1e-5, rtol=1e-5)

    t2c.save_cube(cube, tmp_path / "cube.fits")
    with pytest.raises(OSError):
        t2c.save_cube(cube, tmp_path / "cube.fits")
    saved = t2c.load_cube(tmp_path / "cube.fits")
    assert saved["order"] == ORDER and saved["nest"] and saved["optweight"]
    np.testing.assert_array_equal(saved["zmax"], cube["zmax"])
    for name in ("nobj",) + t2c.SUMS:
        np.testing.assert_array_equal(saved[name], cube[name])
        assert not saved[name].flags.owndata  # a view of the memmap, not a copy
    header = saved["hdul"][0].header
    assert "SELECT synthetic FROM nowhere" in str(header["COMMENT"])
    saved["hdul"].close()


def test_missing_required_stops(tmp_path):
    write_table(tmp_path / "table.fits", drop=("sum_invchi", "sum_kappa"))
    with pytest.raises(ValueError, match="missing") as err:
        t2c.build_cube(tmp_path / "table.fits", order=ORDER)
    assert "sum_invchi" in str(err.value) and "sum_kappa" in str(err.value)
    assert not (tmp_path / "cube.fits").exists()


def test_missing_optional_continues(tmp_path):
    drop = tuple(t2c.table_column(n) for n in t2c.OPTIONAL)
    write_table(tmp_path / "table.fits", drop=drop)
    cube = t2c.build_cube(tmp_path / "table.fits", order=ORDER)
    assert not cube["optweight"]
    assert all(n not in cube for n in t2c.OPTIONAL)
    t2c.save_cube(cube, tmp_path / "cube.fits")
    saved = t2c.load_cube(tmp_path / "cube.fits")
    assert not saved["optweight"] and "invchi2" not in saved
    saved["hdul"].close()

    # only some optional columns
    write_table(tmp_path / "table2.fits", drop=("sum_invchi2",))
    cube = t2c.build_cube(tmp_path / "table2.fits", order=ORDER)
    assert not cube["optweight"] and "gamma1_invchi" in cube


def test_bad_table(tmp_path):
    write_table(tmp_path / "table.fits")
    with pytest.raises(ValueError, match="outside"):
        t2c.build_cube(tmp_path / "table.fits", order=ORDER - 1)

    hdu, _ = group(make_galaxies())
    data = hdu.data.copy()
    data["pix"][1], data["zbin"][1] = data["pix"][0], data["zbin"][0]
    bad = fits.BinTableHDU(data)
    fits.HDUList([fits.PrimaryHDU(), bad]).writeto(tmp_path / "dup.fits")
    with pytest.raises(ValueError, match="Duplicated"):
        t2c.build_cube(tmp_path / "dup.fits", order=ORDER)


def test_degrade_matches_direct_build(tmp_path):
    gal, flat = write_table(tmp_path / "table.fits")
    cube = t2c.build_cube(tmp_path / "table.fits", order=ORDER)
    low = t2c.degrade_cube(cube, ORDER - 1)

    gal_low = dict(gal, pix=gal["pix"] // 4)
    flat_low = gal_low["pix"] * NZ + gal_low["zbin"]
    npix_low = NPIX // 4
    assert low["nobj"].shape == (npix_low, NZ) and low["nside"] == 2 ** (ORDER - 1)
    np.testing.assert_array_equal(
        low["nobj"], np.bincount(flat_low, minlength=npix_low * NZ).reshape(npix_low, NZ)
    )
    for name in t2c.SUMS:
        expected = np.bincount(
            flat_low, weights=gal[name] * t2c.scale(name), minlength=npix_low * NZ
        ).reshape(npix_low, NZ)
        np.testing.assert_allclose(low[name], expected, atol=1e-5, rtol=1e-5)
