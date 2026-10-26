"""Jackknife regions (reproducible k-means) and covariances.

Run with: python -m pytest lensing/tests/test_jackknife.py
"""

import random

import numpy as np
import pytest

from lensing import jackknife
from lensing.funcs import cov_matrix, normalize_cov

needs_kmeans = pytest.mark.skipif(
    jackknife.kmeans_sample is None, reason="kmeans_radec is not installed"
)


def sky_sample(n=3000, seed=0):
    r = np.random.default_rng(seed)
    return 360.0 * r.random(n), np.rad2deg(np.arcsin(2.0 * r.random(n) - 1.0))


@needs_kmeans
def test_regions_are_reproducible_for_a_seed():
    # kmeans_radec draws its initial centres with Python's `random`: without the seed
    # two fits of the same sample give other regions (and other covariances)
    ra, dec = sky_sample()
    a = jackknife.fit_regions(ra, dec, 12, seed=3)
    b = jackknife.fit_regions(ra, dec, 12, seed=3)
    c = jackknife.fit_regions(ra, dec, 12, seed=4)
    np.testing.assert_array_equal(a.centers, b.centers)
    assert not np.array_equal(a.centers, c.centers)

    ra_v, dec_v = sky_sample(500, seed=1)
    np.testing.assert_array_equal(
        jackknife.assign_regions(a, ra_v, dec_v), jackknife.assign_regions(b, ra_v, dec_v)
    )


@needs_kmeans
def test_fit_regions_leaves_the_random_state_alone():
    ra, dec = sky_sample()
    random.seed(123)
    state = random.getstate()
    jackknife.fit_regions(ra, dec, 6, seed=0)
    assert random.getstate() == state


@needs_kmeans
def test_assign_regions_gives_each_centre_its_own_region():
    ra, dec = sky_sample()
    km = jackknife.fit_regions(ra, dec, 8, seed=0)
    labels = jackknife.assign_regions(km, km.centers[:, 0], km.centers[:, 1])
    np.testing.assert_array_equal(labels, np.arange(8))
    # all the sample is in some region, and every region has points
    assert set(jackknife.assign_regions(km, ra, dec)) == set(range(8))


def test_jackknife_sums_leave_one_region_out():
    r = np.random.default_rng(0)
    x = r.normal(size=(50, 3))
    labels = r.integers(0, 4, 50)
    out = jackknife.jackknife_sums(x, labels, 4)
    np.testing.assert_allclose(out[0], x.sum(axis=0))
    for k in range(4):
        np.testing.assert_allclose(out[k + 1], x[labels != k].sum(axis=0))


def test_covariance_helpers_match_their_loop_definition():
    r = np.random.default_rng(1)
    jk = r.normal(size=(20, 5))

    # the definition: (K-1)/K * sum of the outer products of the deviations
    K = len(jk)
    mean = jk.mean(axis=0)
    expected = sum(np.outer(j - mean, j - mean) for j in jk) * (K - 1) / K
    cov = cov_matrix(jk)
    np.testing.assert_allclose(cov, expected, rtol=1e-12)

    corr = normalize_cov(cov)
    expected_corr = np.array(
        [[cov[i, j] / np.sqrt(cov[i, i] * cov[j, j]) for j in range(5)] for i in range(5)]
    )
    np.testing.assert_allclose(corr, expected_corr, rtol=1e-12)
    np.testing.assert_allclose(np.diag(corr), 1.0)
