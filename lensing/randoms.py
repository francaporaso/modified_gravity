"""Random voids of the subtraction (Singh et al. 2017). Generated in each stack."""

import numpy as np


def stack_rng(seed, *bin_edges):
    """Generator of the randoms of a stack: set by `[randoms] seed` and the edges of the
    z, Rv and delta bins, so a rerun gives the same randoms, whatever the gravity."""
    key = [seed] + [int(round(1000 * (x + 10.0))) for x in bin_edges]
    return np.random.default_rng(key)


def generate_randoms(lenses, factor, rng, footprint, ra_col, dec_col):
    """Random voids for the subtraction of the signal that is not due to the voids.

    There are `factor` times as many as voids. Each takes the (Rv, z) pair, and the rest of
    the columns, of a void of the stack drawn with replacement, so they have the same
    distribution as the voids of that stack, jointly. Only (ra, dec), the columns
    `ra_col` and `dec_col`, change: uniform on the sphere, redrawn while they fall outside
    `footprint(ra, dec)`. Voids at the edge of a partial footprint aren't excluded, which
    is only right for full-sky (or nearly) mocks.

    Returns an array with the layout of `lenses`.
    """
    nvoids = lenses.shape[1]
    if nvoids == 0:
        raise ValueError("No voids in this bin: nothing to draw the randoms from.")
    nrand = int(round(factor * nvoids))
    rands = lenses[:, rng.integers(0, nvoids, nrand)]

    def sphere(u):
        return 360.0 * u[:, 0], np.rad2deg(np.arcsin(2.0 * u[:, 1] - 1.0))

    ra, dec = sphere(rng.random((nrand, 2)))
    bad = ~footprint(ra, dec)
    for _ in range(10_000):
        if not bad.any():
            break
        ra[bad], dec[bad] = sphere(rng.random((bad.sum(), 2)))
        bad[bad] = ~footprint(ra[bad], dec[bad])
    else:
        raise RuntimeError(
            f"{bad.sum()} random voids are outside the source footprint after 10000 redraws."
        )
    rands[ra_col] = ra
    rands[dec_col] = dec
    return rands
