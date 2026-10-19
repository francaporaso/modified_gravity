from argparse import ArgumentParser

# from astropy.cosmology import WMAP5 as cosmo ## MICE uses WMAP5
from astropy.cosmology import FlatLambdaCDM  ## MICE uses WMAP5
from astropy.constants import G, c, M_sun, pc
from astropy.io import fits
from astropy.table import Table
import healpy as hp
from multiprocessing import get_context
import numpy as np
import os
from pathlib import Path
from time import time, asctime

# import toml
from tqdm import tqdm
from itertools import product

from lensing.funcs import eq2p2, cov_matrix, get_jackknife_kmeans
from lensing.io import read_lens_catalog, read_sources_catalog
from lensing.settings import Config
from shearmap.table2cube import degrade_cube

ctx = get_context("fork")

# --- Fixed globals
cfg: Config
# SOURCE = None
PIX_TO_IDX: dict = {}
SOURCE = None
# map mode: slices of the cube summed behind the current void bin, see `collapse_slices`
MAPS: dict = {}
ZSRC_MIN: float = 0.0
# Signs applied to (gamma1, gamma2) of the sources to get the (e1, e2) that are rotated to
# the tangential/cross components. L768 convention: both negative, as in the old
# voidsmodgrav.py. The MICE convention (+1, -1) gives no tangential signal in L768.
GAMMA_SIGN: tuple = (-1.0, -1.0)
SC_CONSTANT: float = (
    (c.value**2.0 / (4.0 * np.pi * G.value)) * (pc.value / M_sun.value) * 1e-6
)

# "z_cgal" : true-z
# "z_cgal_v" : spec-z
# "z_desdm_mc" : photo-z
# REDSHIFT = "z_cgal_v" # name of the redshift column in the source file
# -----------------


def init_globals():

    global SOURCE
    global binspace
    global cosmo

    # set cosmology
    cosmo = FlatLambdaCDM(H0=100.0 * cfg.h, Om0=cfg.Om0, Ob0=cfg.Ob0)

    # set binning
    binspace = np.linspace if cfg.binning == "lin" else np.geomspace

    # read sources: galaxy catalog or lensing cube (healpix maps)
    if cfg.source_mode == "map":
        if cfg.addnoise:
            raise ValueError(
                "addnoise is not available in map mode (the cube has no eps sums)."
            )
        SOURCE = load_map_source(cfg.cubename, cfg.map_order)
        check_map_zbins()
        check_map_resolution()
    else:
        SOURCE = read_sources_catalog(cfg.sourcename, cat="fits")


def load_map_source(path, order=None):
    """Read the lensing cube. If `order` is lower than the native one, the cube is
    degraded to it by summing the child pixels (exact; loads the cube in memory)."""
    cube = read_sources_catalog(path, cat="map")
    if order is not None and order != cube["order"]:
        if order > cube["order"]:
            raise ValueError(
                f"[sources.map] order={order} is higher than the cube's ({cube['order']})."
            )
        print(f" >> Degrading the cube from order {cube['order']} to {order}", flush=True)
        cube = degrade_cube(cube, order)
    if not cube["nest"]:
        raise ValueError("The lensing cube must be in NESTED ordering.")
    return cube


def source_slice_start(z_low, zmin, buffer):
    """Index of the first cube slice used by a void bin starting at `z_low`: the
    slice with zmin == z_low + buffer. Raises if there is no such slice edge."""
    target = z_low + buffer
    k = int(np.argmin(np.abs(zmin - target)))
    if abs(zmin[k] - target) > 1e-6:
        raise ValueError(
            f"z_low + dz_buffer = {target:.4f} is not the lower edge of a cube slice "
            f"(edges: {np.round(zmin, 4).tolist()})."
        )
    return k


def check_map_zbins():
    """Stops if a void bin of the config can't be served by the cube slices."""
    for z_min, z_max in cfg.zbins:
        source_slice_start(z_min, SOURCE["zmin"], cfg.dz_buffer)
        if z_max - z_min > cfg.dz_buffer + 1e-9:
            raise ValueError(
                f"Void bin [{z_min}, {z_max}) is wider than dz_buffer={cfg.dz_buffer}: "
                "its voids would be behind some of the selected sources."
            )


def collapse_slices(z_low):
    """Sum the cube slices behind the void bin starting at `z_low` into flat maps
    (npix,) in the global `MAPS`, and set `ZSRC_MIN`. Done once per stack, before the
    pool forks, so the workers share the maps."""
    global MAPS, ZSRC_MIN

    k0 = source_slice_start(z_low, SOURCE["zmin"], cfg.dz_buffer)
    MAPS = {"nobj": np.asarray(SOURCE["nobj"][:, k0:]).sum(axis=1, dtype=np.int64)}
    for name in ("gamma1", "gamma2", "kappa", "invchi"):
        MAPS[name] = np.asarray(SOURCE[name][:, k0:]).sum(axis=1, dtype=np.float64)
    ZSRC_MIN = float(SOURCE["zmin"][k0])
    return k0


def check_map_resolution():
    # warns if pixels are too big compared to the narrowest radial bin
    nside = SOURCE["nside"]
    pixsize = np.rad2deg(hp.nside2resol(nside))
    rv_min = min(r[0] for r in cfg.rvbins)
    z_max = max(z[1] for z in cfg.zbins)
    degxmpc = cosmo.arcsec_per_kpc_proper(z_max).to("deg/Mpc").value
    bines = binspace(cfg.RIN, cfg.ROUT, cfg.NBINS + 1)
    binwidth = degxmpc * rv_min * np.diff(bines).min()
    print(
        f" >> Map NSIDE={nside}: pixel {pixsize * 60:.2f} arcmin, narrowest bin {binwidth * 60:.2f} arcmin"
    )
    if pixsize > 0.5 * binwidth:
        print(
            " WARNING: pixel size > half the narrowest radial bin; profiles will be smeared.",
            flush=True,
        )


def make_pix2idx_dict(source):
    global PIX_TO_IDX
    # making a dict of healpix idx for fast query

    # check if presaved dict exists
    path2pix = Path(cfg.pix2idx).expanduser()
    if os.path.exists(path2pix):
        data = np.load(path2pix)
        upix = data["upix"]
        split_idx = data["split_idx"]
        PIX_TO_IDX = dict(zip(upix.tolist(), map(tuple, split_idx.tolist())))

    else:
        print(" >>> Pixel index file couldn't be found. Building from source... ")
        upix, split_idx = np.unique(source[cfg.scols["pix"]].data, return_index=True)
        split_idx = np.append(split_idx, len(source))
        for i, pix in enumerate(upix):
            PIX_TO_IDX[int(pix)] = (split_idx[i], split_idx[i + 1])

        print(f" >>> Saving index to file {cfg.pix2idx}")
        np.savez_compressed(
            path2pix, upix=upix, split_idx=np.array(list(PIX_TO_IDX.values()))
        )


def check_output_exists(output_file, overwrite=False):

    if os.path.exists(output_file):
        if not overwrite:
            raise OSError(
                f"\n{'=' * 60}\n"
                f"Output file already exists: {output_file}\n"
                f"Use --overwrite flag to allow overwriting, or choose a different sample name.\n"
                f"{'=' * 60}"
            )
        else:
            print(f" WARNING: Will overwrite existing file: {output_file}", flush=True)
    return True


## WARNING :
## cosmo.distance is in physical units (ie Mpc)
## need to divide out by littleh to get Mpc/h
## important only if h!=1
def sigma_crit(z_l, z_s):
    d_l = cosmo.angular_diameter_distance(z_l).value
    d_s = cosmo.angular_diameter_distance(z_s).value
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    return SC_CONSTANT * (d_s / (d_ls * d_l))


def comoving_chi(z):
    """Comoving distance in Mpc/h"""
    return cosmo.comoving_distance(z).value * cosmo.h


def inv_sigma_crit_prefactor(z_l):
    """A(z_l) such that, in a flat universe, 1/sigma_crit(z_l, z_s) = A * (1 - chi_l/chi_s).
    This makes the sum of 1/sigma_crit over sources linear in sum(1/chi_s), which is
    what the lensing cube stores."""
    return cosmo.angular_diameter_distance(z_l).value / SC_CONSTANT


def get_masked_idx_zmin(psi, ra0, dec0, zmin):
    """Same as `get_masked_idx_fast` but selecting the sources with z >= zmin"""

    queried_pix = hp.query_disc(
        cfg.NSIDE,
        vec=hp.ang2vec(ra0, dec0, lonlat=True),
        radius=np.deg2rad(psi * 1.1),
        inclusive=True,
    )

    queried_idranges = np.array([PIX_TO_IDX[p] for p in queried_pix if p in PIX_TO_IDX])

    idx_arrays = np.concatenate([np.arange(s, e) for s, e in queried_idranges])

    mask_z = SOURCE[cfg.scols["redshift"]].data[idx_arrays] >= zmin

    return idx_arrays[mask_z]


def get_masked_idx_fast(psi, ra0, dec0, z0):
    """
    objects are selected by pixel on a disc of rad=psi+pad where pad = 0.1*psi
    uses prebuilt _PIX_TO_INDEX dict
    returns the indices of _S where to select
    """

    # inclusive=True: with the default only the pixels whose *centre* is inside the disc
    # are read, so sources inside psi that sit in pixels centred outside it were lost
    # (up to ~10% for the smallest voids at high z, all in the outer radial bins)
    queried_pix = hp.query_disc(
        cfg.NSIDE,
        vec=hp.ang2vec(ra0, dec0, lonlat=True),
        radius=np.deg2rad(psi * 1.1),
        inclusive=True,
    )

    queried_idranges = np.array([PIX_TO_IDX[p] for p in queried_pix if p in PIX_TO_IDX])

    idx_arrays = np.concatenate([np.arange(s, e) for s, e in queried_idranges])

    mask_z = SOURCE[cfg.scols["redshift"]].data[idx_arrays] > (z0 + 0.1)

    return idx_arrays[mask_z]


## distance: is needed a cosmo.h dividing sigma_c when h!=1, else is not needed.
## leaving it for general case...
def partial_profile(inp):

    Sigma_wsum = np.zeros(cfg.NBINS)
    DSigma_t_wsum = np.zeros(cfg.NBINS)
    DSigma_x_wsum = np.zeros(cfg.NBINS)
    N_inbin = np.zeros(cfg.NBINS)

    Rv0, ra0, dec0, z0 = inp
    # for ni in range(N):
    # adentro del for, mask depende de n... solo quiero las gx en un anillo

    DEGxMPC = cosmo.arcsec_per_kpc_proper(z0).to("deg/Mpc").value
    psi = DEGxMPC * cfg.ROUT * Rv0

    idx = get_masked_idx_fast(psi, ra0, dec0, z0)
    catdata = SOURCE[idx]

    # sigma_c = sigma_crit(z0, catdata[REDSHIFT])/Rv0
    ## dividing by cosmo.h gives the correct units! (Rv0 in Mpc/h but sigma_crit in physical Msun*pc^-2)
    ## factor of almost 1.5 difference!
    sigma_c = sigma_crit(z0, catdata[cfg.scols["redshift"]]) / (Rv0 * cosmo.h)

    rads, theta = eq2p2(
        np.deg2rad(catdata[cfg.scols["ra"]]),
        np.deg2rad(catdata[cfg.scols["dec"]]),
        np.deg2rad(ra0),
        np.deg2rad(dec0),
    )

    e1 = GAMMA_SIGN[0] * catdata[cfg.scols["gamma1"]]
    e2 = GAMMA_SIGN[1] * catdata[cfg.scols["gamma2"]]
    if cfg.addnoise:
        e1 -= GAMMA_SIGN[0] * catdata[cfg.scols["eps1"]]
        e2 -= GAMMA_SIGN[1] * catdata[cfg.scols["eps2"]]

    # get tangential ellipticities
    cos2t = np.cos(2.0 * theta)
    sin2t = np.sin(2.0 * theta)
    et = -(e1 * cos2t + e2 * sin2t) * sigma_c
    ex = (-e1 * sin2t + e2 * cos2t) * sigma_c

    # get convergence
    k = catdata[cfg.scols["kappa"]] * sigma_c

    bines = binspace(cfg.RIN, cfg.ROUT, cfg.NBINS + 1)
    dig = np.digitize((np.rad2deg(rads) / DEGxMPC) / Rv0, bines)

    for nbin in range(cfg.NBINS):
        mbin = dig == nbin + 1
        Sigma_wsum[nbin] = k[mbin].sum()
        DSigma_t_wsum[nbin] = et[mbin].sum()
        DSigma_x_wsum[nbin] = ex[mbin].sum()
        N_inbin[nbin] = np.count_nonzero(mbin)  ## idem mbin.sum(), faster

    return Sigma_wsum, DSigma_t_wsum, DSigma_x_wsum, N_inbin


def partial_profile_map(inp):
    """
    Same as `partial_profile_weighted` (Sigma_crit^-1 weights) but the sources are
    the pixels of the lensing cube, summed over the slices behind the void bin
    (`collapse_slices` must have been called for the bin of this void).

    Each pixel holds N, the sums of gamma1, gamma2, kappa and X = sum(CHI_REF/chi_s).
    In a flat universe the sum over its sources of 1/sigma_crit is
    A(z_l) * (N - l*X), with l = chi_l/CHI_REF, so the estimator
        DSigma = sum(gamma) / sum(1/sigma_crit)
    is built from the pixel sums only.

    Returns the sums per radial bin of: kappa, gamma_t and gamma_x (all divided by
    Rv*h), the weights sum(1/sigma_crit) and the number of sources.
    """

    Rv0, ra0, dec0, z0 = inp

    DEGxMPC = cosmo.arcsec_per_kpc_proper(z0).to("deg/Mpc").value
    psi = DEGxMPC * cfg.ROUT * Rv0

    pix = hp.query_disc(
        SOURCE["nside"],
        vec=hp.ang2vec(ra0, dec0, lonlat=True),
        radius=np.deg2rad(psi),
        inclusive=False,  # pixels are assigned to bins by their centre
        nest=SOURCE["nest"],
    )
    if len(pix) == 0:
        return tuple(np.zeros(cfg.NBINS) for _ in range(5))
    pix.sort()

    n = MAPS["nobj"][pix]
    f = 1.0 / (Rv0 * cosmo.h)
    # 1/sigma_crit summed over the sources of each pixel
    w = inv_sigma_crit_prefactor(z0) * (
        n - (comoving_chi(z0) / SOURCE["chiref"]) * MAPS["invchi"][pix]
    )

    g1 = GAMMA_SIGN[0] * MAPS["gamma1"][pix]
    g2 = GAMMA_SIGN[1] * MAPS["gamma2"][pix]
    k = MAPS["kappa"][pix]

    ra_p, dec_p = hp.pix2ang(SOURCE["nside"], pix, nest=SOURCE["nest"], lonlat=True)
    rads, theta = eq2p2(
        np.deg2rad(ra_p), np.deg2rad(dec_p), np.deg2rad(ra0), np.deg2rad(dec0)
    )

    cos2t = np.cos(2.0 * theta)
    sin2t = np.sin(2.0 * theta)
    et = -(g1 * cos2t + g2 * sin2t)
    ex = -g1 * sin2t + g2 * cos2t

    bines = binspace(cfg.RIN, cfg.ROUT, cfg.NBINS + 1)
    dig = np.digitize((np.rad2deg(rads) / DEGxMPC) / Rv0, bines)

    def binsum(x):
        return np.bincount(dig, weights=x, minlength=cfg.NBINS + 2)[1 : cfg.NBINS + 1]

    return binsum(f * k), binsum(f * et), binsum(f * ex), binsum(w), binsum(n)


def partial_profile_weighted(inp, source_cut="bin"):
    """
    Galaxy-catalog counterpart of `partial_profile_map`, to check it. Same estimator
    (sources weighted by 1/sigma_crit) and same return values. `partial_profile` is
    left untouched.

    source_cut: 'bin'  : sources with z >= ZSRC_MIN, like the slices of the cube
                'void' : sources with z >= z_void + dz_buffer
    """

    Rv0, ra0, dec0, z0 = inp

    DEGxMPC = cosmo.arcsec_per_kpc_proper(z0).to("deg/Mpc").value
    psi = DEGxMPC * cfg.ROUT * Rv0

    zmin = ZSRC_MIN if source_cut == "bin" else z0 + cfg.dz_buffer
    idx = get_masked_idx_zmin(psi, ra0, dec0, zmin)
    catdata = SOURCE[idx]

    f = 1.0 / (Rv0 * cosmo.h)
    zs = catdata[cfg.scols["redshift"]]
    w = inv_sigma_crit_prefactor(z0) * (1.0 - comoving_chi(z0) / comoving_chi(zs))

    rads, theta = eq2p2(
        np.deg2rad(catdata[cfg.scols["ra"]]),
        np.deg2rad(catdata[cfg.scols["dec"]]),
        np.deg2rad(ra0),
        np.deg2rad(dec0),
    )

    e1 = GAMMA_SIGN[0] * catdata[cfg.scols["gamma1"]]
    e2 = GAMMA_SIGN[1] * catdata[cfg.scols["gamma2"]]
    if cfg.addnoise:
        e1 = e1 - GAMMA_SIGN[0] * catdata[cfg.scols["eps1"]]
        e2 = e2 - GAMMA_SIGN[1] * catdata[cfg.scols["eps2"]]

    cos2t = np.cos(2.0 * theta)
    sin2t = np.sin(2.0 * theta)
    et = -(e1 * cos2t + e2 * sin2t)
    ex = -e1 * sin2t + e2 * cos2t
    k = catdata[cfg.scols["kappa"]]

    bines = binspace(cfg.RIN, cfg.ROUT, cfg.NBINS + 1)
    dig = np.digitize((np.rad2deg(rads) / DEGxMPC) / Rv0, bines)

    def binsum(x):
        return np.bincount(dig, weights=x, minlength=cfg.NBINS + 2)[1 : cfg.NBINS + 1]

    return binsum(f * k), binsum(f * et), binsum(f * ex), binsum(w), np.bincount(
        dig, minlength=cfg.NBINS + 2
    )[1 : cfg.NBINS + 1].astype(np.float64)


def jackknife_sample(nmax=10_000):
    """ra, dec (deg) of points sampling the source footprint, for the jackknife k-means"""
    if cfg.source_mode == "map":
        npix = hp.nside2npix(SOURCE["nside"])
        pix = np.unique(np.linspace(0, npix - 1, nmax).astype(int))
        pix = pix[np.asarray(SOURCE["nobj"][pix]).sum(axis=1) > 0]
        ra, dec = hp.pix2ang(SOURCE["nside"], pix, nest=SOURCE["nest"], lonlat=True)
        return ra, dec
    jidx = np.arange(1, len(SOURCE) - 1, len(SOURCE) // nmax, dtype=int)
    return SOURCE[cfg.scols["ra"]][jidx], SOURCE[cfg.scols["dec"]][jidx]


def check_lens_columns(lenses, lcols, bounds, name="lenses"):
    """Stops if `[lenses.columns]` doesn't point to the columns that `read_sparkling` used
    to cut the voids. The cut is done with fixed indices (no ID column), so every selected
    void must satisfy the bounds on the configured columns; if not, the columns of the
    config are shifted (e.g. an ID column assumed) and the profiles would be computed
    with the wrong Rv, ra, dec or z.

    bounds: {'rv': (min, max), 'z': (min, max), 'delta': (min, max)}
    """
    if lenses.shape[1] == 0:
        return
    for key, (lo, hi) in bounds.items():
        col = lenses[lcols[key]]
        if not np.all((col >= lo) & (col < hi)):
            raise ValueError(
                f"[lenses.columns] {key} = {lcols[key]} doesn't match the {name} catalog: "
                f"values in [{col.min():.4g}, {col.max():.4g}] but the voids were selected "
                f"with {lo} <= {key} < {hi}. Check the column numbers in the config "
                "(the void files have no ID column: rv=0, ra=1, dec=2, z=3, delta=8)."
            )


def stacking(rv_min, rv_max, z_min, z_max, delta_min, delta_max, gravity):

    lenses = read_lens_catalog(
        filename=cfg.lensname,
        cat="sparkling",
        Rv_min=rv_min,
        Rv_max=rv_max,
        z_min=z_min,
        z_max=z_max,
        delta_min=delta_min,
        delta_max=delta_max,
        flag=cfg.flag,
        has_id=False,
        fullshape=cfg.fullshape,
    )
    # TODO: maybe shuffle the cutted catalogue? the # of voids is not the same
    lensrand = read_lens_catalog(
        filename=cfg.randsname,
        cat="sparkling",
        Rv_min=rv_min,
        Rv_max=rv_max,
        z_min=z_min,
        z_max=z_max,
        delta_min=delta_min,
        delta_max=delta_max,
        flag=cfg.flag,
        has_id=False,
        fullshape=cfg.fullshape,
    )

    nvoids = len(lenses.T)
    nrands = len(lensrand.T)

    bounds = {
        "rv": (rv_min, rv_max),
        "z": (z_min, z_max),
        "delta": (delta_min, delta_max),
    }
    check_lens_columns(lenses, cfg.lcols, bounds, "lenses")
    check_lens_columns(lensrand, cfg.lcols, bounds, "randoms")

    if delta_max <= 0:
        voidtype = "R"
    elif delta_min >= 0:
        voidtype = "S"
    else:
        voidtype = "mixed"

    output_filename = (
        f"lensing/results/lensing_{gravity}_thr{cfg.voidthr}_{cfg.sample}_N{cfg.NBINS}_"
        f"Rv{rv_min:02.0f}-{rv_max:02.0f}_"
        f"z{100 * z_min:03.0f}-{100 * z_max:03.0f}_"
        f"type{voidtype}_bin{cfg.binning}"
    )
    if cfg.source_mode == "map":
        output_filename += "_map"
    if cfg.addnoise:
        output_filename += "w-noise"
    output_filename += ".fits"
    output_filename = Path(output_filename).expanduser()

    assert check_output_exists(output_filename, overwrite=cfg.overwrite)

    # === program arguments
    print(f" {' Settings ':=^60}")
    print(" Lens cat " + f"{': ':.>10}{cfg.lensname}")
    print(" Source cat " + f"{': ':.>8}{cfg.sourcename} ({cfg.source_mode})")
    print(" NCORES " + f"{': ':.>12}{cfg.ncores}\n")

    # === profile arguments
    print(" RMIN " + f"{': ':.>14}{cfg.RIN:.2f}")
    print(" RMAX " + f"{': ':.>14}{cfg.ROUT:.2f}")
    print(" NBINS " + f"{': ':.>13}{cfg.NBINS:<2d}")
    print(" NJK " + f"{': ':.>15}{cfg.NJK:<2d}")
    print(" Binning " + f"{': ':.>11}{cfg.binning}")
    print(" Shape Noise " + f"{': ':.>7}{cfg.addnoise}\n")

    # === lens arguments
    print(f" {' Void sample ':=^60}")
    print(" Radii " + f"{': ':.>13}[{rv_min:.2f}, {rv_max:.2f}) Mpc/h")
    print(" Redshift " + f"{': ':.>10}[{z_min:.2f}, {z_max:.2f})")
    print(" Type " + f"{': ':.>14}[{delta_min},{delta_max}) => {voidtype}")
    print(" # of voids " + f"{': ':.>12}{nvoids}\n", flush=True)

    if cfg.source_mode == "map":
        profile_func = partial_profile_map
        # before the pool forks: the workers share the collapsed maps
        collapse_slices(z_min)
        print(
            f" Sources {': ':.>11}z >= {ZSRC_MIN:.2f} ({cfg.weighting} weights)\n",
            flush=True,
        )
    else:
        profile_func = partial_profile

    N_inbin = np.zeros((cfg.NJK + 1, cfg.NBINS))
    W_inbin = np.zeros((cfg.NJK + 1, cfg.NBINS))
    Sigma_wsum = np.zeros((cfg.NJK + 1, cfg.NBINS))
    DSigma_t_wsum = np.zeros((cfg.NJK + 1, cfg.NBINS))
    DSigma_x_wsum = np.zeros((cfg.NJK + 1, cfg.NBINS))

    Sigma_wsum_rand = np.zeros((cfg.NJK + 1, cfg.NBINS))
    N_inbin_rand = np.zeros((cfg.NJK + 1, cfg.NBINS))
    W_inbin_rand = np.zeros((cfg.NJK + 1, cfg.NBINS))
    # calculating voids
    with ctx.Pool(processes=cfg.ncores) as pool:
        resmap = list(
            tqdm(
                pool.imap(
                    profile_func,
                    lenses[
                        [
                            cfg.lcols["rv"],
                            cfg.lcols["ra"],
                            cfg.lcols["dec"],
                            cfg.lcols["z"],
                        ]
                    ].T,
                ),
                total=nvoids,
            )
        )

        # calculating randoms
        print("\n >> Calculating profiles for random voids...", flush=True)
        randmap = list(
            tqdm(
                pool.imap(
                    profile_func,
                    lensrand[
                        [
                            cfg.lcols["rv"],
                            cfg.lcols["ra"],
                            cfg.lcols["dec"],
                            cfg.lcols["z"],
                        ]
                    ].T,
                ),
                total=nrands,
            )
        )

    print("\n >> Pool ended, stacking...", flush=True)

    # -- reducing...
    # profile functions return (kappa, gamma_t, gamma_x, N) with unit weights, or
    # (kappa, gamma_t, gamma_x, W, N) with the sum of the weights W
    def unpack(res):
        cols = [np.vstack(x) for x in zip(*res)]
        if len(cols) == 4:
            cols.append(cols[3])
        kappa, gamma_t, gamma_x, w, n = cols
        return kappa, gamma_t, gamma_x, w, n

    kappa, gamma_t, gamma_x, wbin, nbin = unpack(resmap)
    kappa_rand, _, _, wbin_rand, nbin_rand = unpack(randmap)

    N_inbin[0] = nbin.sum(axis=0)
    W_inbin[0] = wbin.sum(axis=0)
    Sigma_wsum[0] = kappa.sum(axis=0)
    DSigma_t_wsum[0] = gamma_t.sum(axis=0)
    DSigma_x_wsum[0] = gamma_x.sum(axis=0)

    N_inbin_rand[0] = nbin_rand.sum(axis=0)
    W_inbin_rand[0] = wbin_rand.sum(axis=0)
    Sigma_wsum_rand[0] = kappa_rand.sum(axis=0)

    # calculate jackknife regions and profiles
    ra_sample, dec_sample = jackknife_sample()
    kidx, km = get_jackknife_kmeans(
        ra_sample=ra_sample,
        dec_sample=dec_sample,
        ra_cl=lenses[cfg.lcols["ra"]],
        dec_cl=lenses[cfg.lcols["dec"]],
        nlenses=nvoids,
        NJK=cfg.NJK,
    )
    kidx_rand = km.find_nearest(
        np.column_stack([lensrand[cfg.lcols["ra"]], lensrand[cfg.lcols["dec"]]])
    )

    for j, k in enumerate(range(cfg.NJK)):
        mask = kidx != k
        mask_rand = kidx_rand != k

        N_inbin[j + 1, :] = nbin[mask].sum(axis=0)
        W_inbin[j + 1, :] = wbin[mask].sum(axis=0)
        Sigma_wsum[j + 1, :] = kappa[mask].sum(axis=0)
        DSigma_t_wsum[j + 1, :] = gamma_t[mask].sum(axis=0)
        DSigma_x_wsum[j + 1, :] = gamma_x[mask].sum(axis=0)

        N_inbin_rand[j + 1, :] = nbin_rand[mask_rand].sum(axis=0)
        W_inbin_rand[j + 1, :] = wbin_rand[mask_rand].sum(axis=0)
        Sigma_wsum_rand[j + 1, :] = kappa_rand[mask_rand].sum(axis=0)

    Sigma = Sigma_wsum / W_inbin
    DSigma_t = DSigma_t_wsum / W_inbin
    DSigma_x = DSigma_x_wsum / W_inbin

    Sigma_rand = Sigma_wsum_rand / W_inbin_rand

    extradata = dict(
        nvoids=nvoids,
        z_mean=lenses[cfg.lcols["z"]].mean(),
        z_median=np.median(lenses[cfg.lcols["z"]]),
        Rv_mean=lenses[cfg.lcols["rv"]].mean(),
        Rv_median=np.median(lenses[cfg.lcols["rv"]]),
        delta_mean=lenses[cfg.lcols["delta"]].mean(),
        delta_median=lenses[cfg.lcols["delta"]].mean(),
    )

    # return Sigma, DSigma_t, DSigma_x, extradata
    # =======================

    # ==== Saving
    head = fits.Header()
    head.update(
        {
            "nvoids": extradata["nvoids"],
            "lenscat": cfg.lensname,
            "sourcat": cfg.sourcename,
            "Rv_min": rv_min,
            "Rv_max": rv_max,
            "Rv_mean": extradata["Rv_mean"],
            "Rv_med": extradata["Rv_median"],
            "z_min": z_min,
            "z_max": z_max,
            "z_mean": extradata["z_mean"],
            "z_med": extradata["z_median"],
            "voidtype": voidtype,
            "deltamin": delta_min,
            "deltamax": delta_max,
            "deltamea": extradata["delta_mean"],
            "deltamed": extradata["delta_median"],
            "RIN": cfg.RIN,
            "ROUT": cfg.ROUT,
            "NBINS": cfg.NBINS,
            "NJK": cfg.NJK,
            "binning": cfg.binning,
            "srcmode": cfg.source_mode,
            "HISTORY": f"{asctime()}",
        }
    )
    if cfg.source_mode == "map":
        head.update(
            {
                "nside": SOURCE["nside"],
                "order": SOURCE["order"],
                "weight": cfg.weighting,
                "dzbuffer": cfg.dz_buffer,
                "zsrcmin": ZSRC_MIN,
                "chiref": SOURCE["chiref"],
            }
        )

    table = Table(
        {
            "R": binspace(cfg.RIN, cfg.ROUT, cfg.NBINS),
            "Sigma": Sigma[0],
            "DSigma_t": DSigma_t[0],
            "DSigma_x": DSigma_x[0],
            "Sigma_rand": Sigma_rand[0],
        }
    )
    if cfg.source_mode == "map":
        table["N"] = N_inbin[0]

    # sigma cov_matrix
    covS = cov_matrix(Sigma[1:, :] - Sigma_rand[1:, :])
    cov_hdu = [
        fits.ImageHDU(covS, name="cov_Sigma"),
        fits.ImageHDU(cov_matrix(DSigma_t[1:, :]), name="cov_DSigma_t"),
        fits.ImageHDU(cov_matrix(DSigma_x[1:, :]), name="cov_DSigma_x"),
        fits.ImageHDU(cov_matrix(Sigma_rand[1:, :]), name="cov_Sigma_rand"),
    ]

    jack_hdu = [
        fits.ImageHDU(Sigma[1:, :], name="jack_Sigma"),
        fits.ImageHDU(DSigma_t[1:, :], name="jack_DSigma_t"),
        fits.ImageHDU(DSigma_x[1:, :], name="jack_DSigma_x"),
        fits.ImageHDU(Sigma_rand[1:, :], name="jack_Sigma_rand"),
    ]

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=head),
            fits.BinTableHDU(table, name="profiles"),
            *cov_hdu,
            *jack_hdu,
        ]
    )

    hdul.writeto(output_filename, overwrite=cfg.overwrite)
    print(f" >> File saved in: {output_filename}.\n", flush=True)

    return 0


def main():
    global cfg

    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="lensing/config.toml", action="store"
    )
    parser.add_argument("--ncores", type=int, action="store")
    parser.add_argument("--gravity", type=str, action="store", nargs="+", required=True)
    parser.add_argument(
        "--use_threshold", type=str, action="store", choices=["08", "09"], default="09"
    )
    args = parser.parse_args()

    print(" " + " Start ".center(15, "="))
    tini = time()

    i = 0
    total = 0
    for grav in args.gravity:
        cfg = Config(args.config, grav, args.use_threshold)

        if args.ncores is not None:
            cfg.set_ncores(args.ncores)

        if i == 0:
            total = (
                len(cfg.zbins) * len(cfg.rvbins) * len(cfg.voidtype) * len(args.gravity)
            )
            print(
                f" >> Running {len(cfg.zbins)} redshift bin(s) x {len(cfg.rvbins)} radius bin(s), for {len(cfg.voidtype)} void types."
            )
            print(f" >> Calculating {total} void profiles")

        print(f" >> Running for -{grav.upper()}-")
        init_globals()
        if cfg.source_mode == "catalog":
            make_pix2idx_dict(SOURCE)
        else:
            print(
                f" >> Map mode: cube order {SOURCE['order']} NESTED, "
                f"{len(SOURCE['zmin'])} source slices.",
                flush=True,
            )

        for (z_min, z_max), (rv_min, rv_max) in product(cfg.zbins, cfg.rvbins):
            for void in cfg.voidtype:
                i += 1
                # reset in each loop
                delta_min, delta_max = -1.0, 100.0

                if void == "S":
                    delta_min = 0.0
                elif void == "R":
                    delta_max = 0.0

                print(f"\n [{i}/{total}]")
                check = stacking(
                    rv_min, rv_max, z_min, z_max, delta_min, delta_max, grav.upper()
                )
                assert check == 0, " >> Something went wrong. << "

    print(" End! ".center(15, "="))
    print(f" >> Took {(time() - tini) / 60.0:.3f} min <<\n\n")


if __name__ == "__main__":
    main()
