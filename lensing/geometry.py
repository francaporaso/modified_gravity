"""Geometry of the stacks: angles around a void, the shear rotation and the radial bins."""

import numpy as np
from astropy.coordinates import angular_separation, position_angle

# Signs applied to (gamma1, gamma2) of the sources to get the (e1, e2) that are rotated to
# the tangential/cross components. L768 convention: both negative, as in the old
# voidsmodgrav.py. The MICE convention (+1, -1) gives no tangential signal in L768.
GAMMA_SIGN: tuple = (-1.0, -1.0)


def eq2p2(ra_gal, dec_gal, RA0, DEC0):
    """
    angular separation and position angle from centre (RA0,DEC0) to gal position (ra_gal, dec_gal)
    returns two angles in radians
    all parameters must be in radians.
    """

    ra_prime = ra_gal - RA0

    rad = angular_separation(ra_prime, dec_gal, 0.0, DEC0)
    theta = position_angle(0.0, DEC0, ra_prime, dec_gal).value

    return rad, theta


def void_offsets(ra, dec, ra0, dec0):
    """`eq2p2` for positions and centre in degrees"""
    return eq2p2(np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(ra0), np.deg2rad(dec0))


def rotate_shear(gamma1, gamma2, theta):
    """Tangential and cross components, from (gamma1, gamma2) and the position angle
    `theta` of the sources around the void. The only place where `GAMMA_SIGN` is applied."""
    e1 = GAMMA_SIGN[0] * gamma1
    e2 = GAMMA_SIGN[1] * gamma2
    cos2t = np.cos(2.0 * theta)
    sin2t = np.sin(2.0 * theta)
    return -(e1 * cos2t + e2 * sin2t), -e1 * sin2t + e2 * cos2t


def bin_edges(rin, rout, nbins, binning):
    """Edges of the radial bins, in units of Rv ('lin' or 'log')"""
    space = np.linspace if binning == "lin" else np.geomspace
    return space(rin, rout, nbins + 1)


def bin_centres(rin, rout, nbins, binning):
    """Centres of the radial bins (in units of Rv): arithmetic midpoints of the edges
    for linear binning, geometric midpoints for log binning."""
    edges = bin_edges(rin, rout, nbins, binning)
    if binning == "log":
        return np.sqrt(edges[:-1] * edges[1:])
    return 0.5 * (edges[:-1] + edges[1:])


def radial_bin(rads, deg_per_mpc, rv, edges):
    """Index of the radial bin (1..nbins; 0 and nbins+1 are outside) of the angular
    distances `rads` [rad] to a void of comoving radius `rv`"""
    return np.digitize((np.rad2deg(rads) / deg_per_mpc) / rv, edges)


def binsum(dig, x, nbins):
    """Sum of `x` in each of the nbins radial bins of `dig`; the number of rows if x is None"""
    s = np.bincount(dig, weights=x, minlength=nbins + 2)[1 : nbins + 1]
    return s.astype(np.float64)
