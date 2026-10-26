"""Helpers of the profile functions: shear rotation and sums per radial bin.

Run with: python -m pytest lensing/tests/test_geometry.py
"""

import numpy as np
import pytest

from lensing import geometry


def test_binsum_sums_the_bins_and_ignores_the_outside():
    edges = np.array([1.0, 2.0, 3.0, 4.0])  # 3 bins
    x = np.array([0.5, 1.5, 1.6, 2.5, 3.5, 3.6, 4.5, 9.0])
    dig = np.digitize(x, edges)  # 0: below, 1..3: bins, 4: above
    np.testing.assert_array_equal(geometry.binsum(dig, None, 3), [2, 1, 2])
    np.testing.assert_allclose(geometry.binsum(dig, 10 * x, 3), [31.0, 25.0, 71.0])
    assert geometry.binsum(dig, None, 3).dtype == np.float64
    assert geometry.binsum(np.array([], dtype=int), None, 3).tolist() == [0, 0, 0]


def test_radial_bin_uses_the_comoving_scale():
    edges = geometry.bin_edges(0.2, 2.5, 10, "lin")
    deg = 0.08  # deg per Mpc/h
    # a source at exactly R = Rv: R/Rv = 1
    rads = np.deg2rad(np.array([15.0 * deg, 30.0 * deg, 0.01 * deg]))
    dig = geometry.radial_bin(rads, deg, 15.0, edges)
    assert dig[0] == np.searchsorted(edges, 1.0)
    assert dig[1] == np.searchsorted(edges, 2.0)
    assert dig[2] == 0  # inside the first edge


def test_rotate_shear_is_tangential_for_the_l768_convention():
    theta = np.linspace(0, 2 * np.pi, 9)
    a = 0.03
    # gamma = a (cos 2theta, sin 2theta) is purely tangential and positive
    et, ex = geometry.rotate_shear(a * np.cos(2 * theta), a * np.sin(2 * theta), theta)
    np.testing.assert_allclose(et, a, atol=1e-15)
    np.testing.assert_allclose(ex, 0.0, atol=1e-15)


def test_gamma_sign_is_the_argument():
    theta = np.array([0.3, 1.1])
    g1, g2 = np.array([0.02, -0.01]), np.array([0.01, 0.03])
    et, ex = geometry.rotate_shear(g1, g2, theta)  # default: the L768 convention
    np.testing.assert_array_equal((et, ex), geometry.rotate_shear(g1, g2, theta, (-1.0, -1.0)))
    et_flipped, ex_flipped = geometry.rotate_shear(g1, g2, theta, (1.0, 1.0))
    np.testing.assert_allclose(et_flipped, -et)
    np.testing.assert_allclose(ex_flipped, -ex)
    # MICE convention: another tangential component, not just a sign
    et_mice, _ = geometry.rotate_shear(g1, g2, theta, (1.0, -1.0))
    assert not np.allclose(np.abs(et_mice), np.abs(et))


@pytest.mark.parametrize("binning", ["lin", "log"])
def test_edges_and_centres_of_the_bins(binning):
    edges = geometry.bin_edges(0.1, 2.5, 6, binning)
    assert len(edges) == 7 and edges[0] == pytest.approx(0.1) and edges[-1] == pytest.approx(2.5)
    centres = geometry.bin_centres(0.1, 2.5, 6, binning)
    assert np.all((centres > edges[:-1]) & (centres < edges[1:]))
