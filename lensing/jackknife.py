"""Jackknife regions and the reduction of the per-void profiles to the stacked ones."""

import random

import numpy as np
from kmeans_radec import kmeans_sample

PROFILE_NAMES = ("Sigma", "DSigma_t", "DSigma_x")


def fit_regions(ra, dec, njk, seed):
    """k-means regions of the sky (`njk` of them) from sample points `ra`, `dec` [deg].

    kmeans_radec draws its initial centres with Python's `random`, so it is seeded here
    (and the state of `random` is restored): the same sample and seed give the same regions.
    """
    state = random.getstate()
    random.seed(seed)
    try:
        return kmeans_sample(np.column_stack([ra, dec]), ncen=njk, verbose=0)
    finally:
        random.setstate(state)


def assign_regions(km, ra, dec):
    """Region (0..njk-1) of each position [deg], for the regions `km` of `fit_regions`"""
    return km.find_nearest(np.column_stack([ra, dec]))


def stack_results(res):
    """Stacks the results of a profile function, which return the sums per radial bin
    (kappa, gamma_t, gamma_x, W, N) of each void, into five arrays (nvoids, NBINS)."""
    return tuple(np.vstack(x) for x in zip(*res))


def jackknife_sums(x, labels, njk):
    """Sums of the rows of x over all of them (row 0) and leaving out each region
    (rows 1..njk), for the region `labels` of every row."""
    total = x.sum(axis=0)
    regions = np.vstack([x[labels == k].sum(axis=0) for k in range(njk)])
    return np.vstack([total, total - regions])


def reduce_profiles(resmap, randmap, kidx, kidx_rand, njk):
    """Profiles of the void stack, of the random stack and of their difference, for the
    full sample (row 0) and the njk jackknife samples (rows 1..njk), the voids and randoms
    being left out together by region.

    The estimator (Singh et al. 2017): X = X_voids - X_rand for Sigma, DSigma_t and
    DSigma_x. Returns a dict with those (X), X_raw and X_rand, and the number of sources of
    the voids and of the randoms.
    """
    kappa, gamma_t, gamma_x, w, n = stack_results(resmap)
    kappa_r, gamma_t_r, gamma_x_r, w_r, n_r = stack_results(randmap)

    W = jackknife_sums(w, kidx, njk)
    W_r = jackknife_sums(w_r, kidx_rand, njk)

    profiles = {}
    for name, v, r in zip(
        PROFILE_NAMES, (kappa, gamma_t, gamma_x), (kappa_r, gamma_t_r, gamma_x_r)
    ):
        raw = jackknife_sums(v, kidx, njk) / W
        rand = jackknife_sums(r, kidx_rand, njk) / W_r
        profiles[name] = raw - rand
        profiles[name + "_raw"] = raw
        profiles[name + "_rand"] = rand
    return profiles, jackknife_sums(n, kidx, njk), jackknife_sums(n_r, kidx_rand, njk)
