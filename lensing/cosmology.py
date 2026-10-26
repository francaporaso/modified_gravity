"""Cosmology helpers of the profile functions. They take the cosmology as an argument
(`make_cosmo`), with distances in Mpc/h (h = 1 in the code)."""

import numpy as np
from astropy.constants import G, c, M_sun, pc
from astropy.cosmology import FlatLambdaCDM

SC_CONSTANT: float = (
    (c.value**2.0 / (4.0 * np.pi * G.value)) * (pc.value / M_sun.value) * 1e-6
)


def make_cosmo(h, Om0, Ob0, Ode0=None):
    """Flat LCDM. `Ode0` is only checked: a non-flat model or w0wa needs another class here
    and the 1/sigma_crit moments of the lensing cube (`inv_sigma_crit_prefactor`) are only
    valid for a flat universe."""
    if Ode0 is not None and abs(Om0 + Ode0 - 1.0) > 1e-6:
        raise ValueError(
            f"[cosmology] Om0 + Ode0 = {Om0 + Ode0:.6f}, but only a flat LCDM is implemented."
        )
    return FlatLambdaCDM(H0=100.0 * h, Om0=Om0, Ob0=Ob0)


## WARNING :
## cosmo.distance is in physical units (ie Mpc)
## need to divide out by littleh to get Mpc/h
## important only if h!=1
def sigma_crit(cosmo, z_l, z_s):
    d_l = cosmo.angular_diameter_distance(z_l).value
    d_s = cosmo.angular_diameter_distance(z_s).value
    d_ls = cosmo.angular_diameter_distance(z_l, z_s).value
    return SC_CONSTANT * (d_s / (d_ls * d_l))


def comoving_chi(cosmo, z):
    """Comoving distance in Mpc/h"""
    return cosmo.comoving_distance(z).value * cosmo.h


def deg_per_mpch(cosmo, z):
    """Angle (deg) subtended at redshift z by 1 comoving Mpc/h. The void radius Rv is
    comoving, so R/Rv = 1 sits at Rv/chi(z) from the centre, not at Rv/D_A(z), which
    is (1+z) times larger (that is what the proper scale arcsec_per_kpc_proper gives)."""
    return cosmo.arcsec_per_kpc_comoving(z).to("deg/Mpc").value / cosmo.h


def inv_sigma_crit_prefactor(cosmo, z_l):
    """A(z_l) such that, in a flat universe, 1/sigma_crit(z_l, z_s) = A * (1 - chi_l/chi_s).
    This makes the sum of 1/sigma_crit over sources linear in sum(1/chi_s), which is
    what the lensing cube stores."""
    return cosmo.angular_diameter_distance(z_l).value / SC_CONSTANT
