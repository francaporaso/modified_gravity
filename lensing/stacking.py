from argparse import ArgumentParser

import healpy as hp
from multiprocessing import get_context
import numpy as np
import os
from pathlib import Path
from time import time

from tqdm import tqdm
from itertools import product

from lensing.cosmology import (
    comoving_chi,
    deg_per_mpch,
    inv_sigma_crit_prefactor,
    make_cosmo,
    sigma_crit,
)
from lensing.geometry import (
    bin_centres,
    bin_edges,
    binsum,
    radial_bin,
    rotate_shear,
    void_offsets,
)
from lensing.io import (
    check_output_exists,
    make_header,
    make_header_halos,
    output_path,
    output_path_halos,
    read_lens,
    read_sources,
    write_profiles,
)
from lensing.jackknife import assign_regions, fit_regions, reduce_profiles
from lensing.randoms import generate_randoms, stack_rng
from lensing.settings import VOIDTYPE_DELTA, Config

ctx = get_context("fork")

# --- Fixed globals: set by `init_globals` (and `collapse_slices` in each stack) before the
# pool forks, so the workers share them
cfg: Config
cosmo = None
EDGES = None  # edges of the radial bins [Rv]
PIX_TO_IDX: dict = {}
SOURCE = None
# k-means regions of the jackknife
REGIONS = None
# map mode: slices of the cube summed behind the current void bin, see `collapse_slices`
MAPS: dict = {}
ZSRC_MIN: float = 0.0

# "z_cgal" : true-z
# "z_cgal_v" : spec-z
# "z_desdm_mc" : photo-z
# -----------------


def set_binning():
    global EDGES
    EDGES = bin_edges(cfg.RIN, cfg.ROUT, cfg.NBINS, cfg.binning)


def init_globals():

    global SOURCE
    global cosmo
    global REGIONS

    cosmo = make_cosmo(cfg.h, cfg.Om0, cfg.Ob0, cfg.Ode0)
    set_binning()

    # read sources: galaxy catalog or lensing cube (healpix maps)
    if cfg.source_mode == "map":
        if cfg.addnoise:
            raise ValueError(
                "addnoise is not available in map mode (the cube has no eps sums)."
            )
        SOURCE = read_sources(cfg.cubename, cat="map", order=cfg.map_order)
        check_map_zbins()
        check_map_resolution(radial_scale())
    else:
        SOURCE = read_sources(cfg.sourcename, cat="fits")

    # the same regions for all the stacks of this source
    print(f" >> Fitting {cfg.NJK} jackknife regions (seed {cfg.jk_seed})", flush=True)
    REGIONS = fit_regions(*jackknife_sample(), cfg.NJK, cfg.jk_seed)


def source_slice_start(z_high, zmin, buffer):
    """Index of the first cube slice used by a void bin ending at `z_high`: the first
    slice with zmin >= z_high + buffer, so every void of the bin is at least `buffer` in
    front of its sources, whatever the width of the bin. Raises if there is none."""
    target = z_high + buffer
    k = int(np.searchsorted(zmin, target - 1e-6))
    if k == len(zmin):
        raise ValueError(
            f"z_high + dz_buffer = {target:.4f}: no cube slice starts at or above it "
            f"(edges: {np.round(zmin, 4).tolist()})."
        )
    return k


def check_map_zbins():
    """Stops if a void bin of the config can't be served by the cube slices, and prints
    the sources of each bin (✗ if they start above z_high + dz_buffer, at the next edge)."""
    for z_min, z_max in cfg.zbins:
        k = source_slice_start(z_max, SOURCE["zmin"], cfg.dz_buffer)
        zsrc = SOURCE["zmin"][k]
        on_edge = abs(zsrc - (z_max + cfg.dz_buffer)) < 1e-6
        print(
            f" >> Bin [{z_min:.2f}, {z_max:.2f}): sources z >= {zsrc:.2f} "
            f"(z_high + dz_buffer = {z_max + cfg.dz_buffer:.2f})  {'✓' if on_edge else '✗'}",
            flush=True,
        )
        if not on_edge:
            print(
                " WARNING: z_high + dz_buffer is not a slice edge; the sources start at the next one.",
                flush=True,
            )


def collapse_slices(z_high):
    """Sum the cube slices behind the void bin ending at `z_high` into flat maps
    (npix,) in the global `MAPS`, and set `ZSRC_MIN`. Done once per stack, before the
    pool forks, so the workers share the maps."""
    global MAPS, ZSRC_MIN

    k0 = source_slice_start(z_high, SOURCE["zmin"], cfg.dz_buffer)
    MAPS = {"nobj": np.asarray(SOURCE["nobj"][:, k0:]).sum(axis=1, dtype=np.int64)}
    for name in ("gamma1", "gamma2", "kappa", "invchi"):
        MAPS[name] = np.asarray(SOURCE[name][:, k0:]).sum(axis=1, dtype=np.float64)
    ZSRC_MIN = float(SOURCE["zmin"][k0])
    return k0


def radial_scale():
    """Comoving size [Mpc/h] of the unit of the radial bins: the smallest Rv of the config
    for voids (their bins are in units of Rv), 1 for halos (their bins are in Mpc/h)"""
    if cfg.lens_type == "halos":
        return 1.0
    return min(r[0] for r in cfg.rvbins)


def check_map_resolution(scale):
    # warns if pixels are too big compared to the narrowest radial bin, `scale` being
    # the comoving size of the unit of the bins (`radial_scale`)
    nside = SOURCE["nside"]
    pixsize = np.rad2deg(hp.nside2resol(nside))
    z_max = max(z[1] for z in cfg.zbins)
    binwidth = deg_per_mpch(cosmo, z_max) * scale * np.diff(EDGES).min()
    ok = pixsize <= 0.5 * binwidth
    print(
        f" >> Map NSIDE={nside}: pixel {pixsize * 60:.2f} arcmin, "
        f"narrowest bin {binwidth * 60:.2f} arcmin  {'✓' if ok else '✗'}",
        flush=True,
    )
    if not ok:
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


def in_footprint(ra, dec):
    """True where (ra, dec) [deg] is on a pixel with sources: the pixels of the collapsed
    maps in map mode (`collapse_slices` must have been called), the pixels of the catalog
    index in catalog mode."""
    if cfg.source_mode == "map":
        pix = hp.ang2pix(SOURCE["nside"], ra, dec, nest=SOURCE["nest"], lonlat=True)
        return MAPS["nobj"][pix] > 0
    pix = hp.ang2pix(cfg.NSIDE, ra, dec, lonlat=True)
    keys = np.fromiter(PIX_TO_IDX.keys(), dtype=np.int64, count=len(PIX_TO_IDX))
    return np.isin(pix, keys)


# --- profiles


def aperture(Rv0, z0):
    """Degrees per comoving Mpc/h at the void (Rv is comoving) and angular radius [deg] of
    the disc of sources of the stack"""
    deg = deg_per_mpch(cosmo, z0)
    return deg, deg * cfg.ROUT * Rv0


def empty_disc(what, psi, ra0, dec0):
    return ValueError(
        f"No {what} within {psi:.2f} deg of the void at (ra, dec) = ({ra0:.3f}, {dec0:.3f}): "
        "the void is outside the footprint of the sources, or the sources don't match the voids."
    )


def query_sources(psi, ra0, dec0, zmin, strict=True):
    """Indices of the sources of the catalog (`SOURCE`) on the pixels of a disc of radius
    `psi` [deg] around (ra0, dec0), with z > zmin (z >= zmin if not `strict`). The pixels
    are those of the prebuilt index `PIX_TO_IDX`. Some sources are farther than psi: they
    fall outside the radial bins. Stops if there are none: a void without sources around
    means that the voids and the sources don't cover the same sky."""

    # inclusive=True: with the default only the pixels whose *centre* is inside the disc
    # are read, so sources inside psi that sit in pixels centred outside it were lost
    # (up to ~10% for the smallest voids at high z, all in the outer radial bins)
    queried_pix = hp.query_disc(
        cfg.NSIDE,
        vec=hp.ang2vec(ra0, dec0, lonlat=True),
        radius=np.deg2rad(psi),
        inclusive=True,
    )

    idxranges = [PIX_TO_IDX[p] for p in queried_pix if p in PIX_TO_IDX]
    if not idxranges:
        raise empty_disc("sources", psi, ra0, dec0)
    idx = np.concatenate([np.arange(s, e) for s, e in idxranges])

    z = SOURCE[cfg.scols["redshift"]].data[idx]
    idx = idx[z > zmin] if strict else idx[z >= zmin]
    if len(idx) == 0:
        raise empty_disc(
            f"sources with z {'>' if strict else '>='} {zmin:.3f}", psi, ra0, dec0
        )
    return idx


def source_shear(cat):
    """(gamma1, gamma2) of the sources of the catalog `cat`, minus the intrinsic
    ellipticities (eps) to remove the shape noise if `addnoise`"""
    g1 = cat[cfg.scols["gamma1"]]
    g2 = cat[cfg.scols["gamma2"]]
    if cfg.addnoise:
        g1 = g1 - cat[cfg.scols["eps1"]]
        g2 = g2 - cat[cfg.scols["eps2"]]
    return g1, g2


def radial_sums(ra, dec, ra0, dec0, deg, Rv0, g1, g2, kappa, weight=None, count=None):
    """Sums per radial bin of sources (or pixels of a map) at (ra, dec) [deg] around the void
    at (ra0, dec0) of comoving radius Rv0, with `deg` degrees per Mpc/h.

    Returns kappa, gamma_t and gamma_x (all divided by Rv*h: Rv is in Mpc/h but sigma_crit
    in physical units), the sum of the `weight` (the number of sources if None) and the
    number of sources (`count` of each pixel; one per row if None).
    """
    rads, theta = void_offsets(ra, dec, ra0, dec0)
    et, ex = rotate_shear(g1, g2, theta, cfg.gamma_sign)
    dig = radial_bin(rads, deg, Rv0, EDGES)

    f = 1.0 / (Rv0 * cosmo.h)
    nb = cfg.NBINS
    n = binsum(dig, count, nb)
    w = n if weight is None else binsum(dig, weight, nb)
    return (
        binsum(dig, f * kappa, nb),
        binsum(dig, f * et, nb),
        binsum(dig, f * ex, nb),
        w,
        n,
    )


def partial_profile(inp):
    """
    Original estimator of the galaxy catalogs: the sums of Sigma_crit * (kappa, gamma_t,
    gamma_x) of the sources with z > z_void + 0.1, so the weights are Sigma_crit and
    W = N. Returns the same as `partial_profile_map`.
    """

    Rv0, ra0, dec0, z0 = inp

    deg, psi = aperture(Rv0, z0)

    # the buffer is 0.1 whatever `dz_buffer` is (the map and weighted functions use it)
    idx = query_sources(psi, ra0, dec0, z0 + 0.1, strict=True)
    cat = SOURCE[idx]

    sc = sigma_crit(cosmo, z0, cat[cfg.scols["redshift"]])
    g1, g2 = source_shear(cat)

    return radial_sums(
        cat[cfg.scols["ra"]],
        cat[cfg.scols["dec"]],
        ra0,
        dec0,
        deg,
        Rv0,
        sc * g1,
        sc * g2,
        sc * cat[cfg.scols["kappa"]],
    )


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

    deg, psi = aperture(Rv0, z0)

    pix = hp.query_disc(
        SOURCE["nside"],
        vec=hp.ang2vec(ra0, dec0, lonlat=True),
        radius=np.deg2rad(psi),
        inclusive=False,  # pixels are assigned to bins by their centre
        nest=SOURCE["nest"],
    )
    pix.sort()

    n = MAPS["nobj"][pix]
    if n.sum() == 0:
        raise empty_disc("sources", psi, ra0, dec0)
    # 1/sigma_crit summed over the sources of each pixel
    w = inv_sigma_crit_prefactor(cosmo, z0) * (
        n - (comoving_chi(cosmo, z0) / SOURCE["chiref"]) * MAPS["invchi"][pix]
    )

    ra_p, dec_p = hp.pix2ang(SOURCE["nside"], pix, nest=SOURCE["nest"], lonlat=True)

    return radial_sums(
        ra_p,
        dec_p,
        ra0,
        dec0,
        deg,
        Rv0,
        MAPS["gamma1"][pix],
        MAPS["gamma2"][pix],
        MAPS["kappa"][pix],
        weight=w,
        count=n,
    )


def inv_sigma_crit_weights(z_l, z_s):
    """1/sigma_crit(z_l, z_s) = A(z_l) (1 - chi_l/chi_s) in a flat universe: the weight of
    each source of a lens at z_l"""
    return inv_sigma_crit_prefactor(cosmo, z_l) * (
        1.0 - comoving_chi(cosmo, z_l) / comoving_chi(cosmo, z_s)
    )


def partial_profile_weighted(inp, source_cut="bin"):
    """
    Galaxy-catalog counterpart of `partial_profile_map`, to check it. Same estimator
    (sources weighted by 1/sigma_crit) and same return values. `partial_profile` is
    the original one.

    source_cut: 'bin'  : sources with z >= ZSRC_MIN, like the slices of the cube
                'void' : sources with z >= z_void + dz_buffer
    """

    Rv0, ra0, dec0, z0 = inp

    deg, psi = aperture(Rv0, z0)

    zmin = ZSRC_MIN if source_cut == "bin" else z0 + cfg.dz_buffer
    cat = SOURCE[query_sources(psi, ra0, dec0, zmin, strict=False)]

    w = inv_sigma_crit_weights(z0, cat[cfg.scols["redshift"]])
    g1, g2 = source_shear(cat)

    return radial_sums(
        cat[cfg.scols["ra"]],
        cat[cfg.scols["dec"]],
        ra0,
        dec0,
        deg,
        Rv0,
        g1,
        g2,
        cat[cfg.scols["kappa"]],
        weight=w,
    )


# Halos: the same estimators with a unit radius, so the bins are comoving Mpc/h (no
# rescaling by Rv, the factor of the sums is only 1/h)


def partial_profile_halo(inp):
    """Halo counterpart of `partial_profile_weighted` (catalog, 1/sigma_crit weights, sources
    with z >= ZSRC_MIN). inp = (ra, dec, z) of the halo."""
    ra0, dec0, z0 = inp
    return partial_profile_weighted((1.0, ra0, dec0, z0), source_cut="bin")


def partial_profile_halo_map(inp):
    """Halo counterpart of `partial_profile_map` (cube, 1/sigma_crit weights). inp = (ra,
    dec, z) of the halo; `collapse_slices` must have been called for its bin."""
    ra0, dec0, z0 = inp
    return partial_profile_map((1.0, ra0, dec0, z0))


# --- stacks


def print_stack_settings(rv_bin, z_bin, delta_bin, voidtype, nvoids):
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
    print(" Radii " + f"{': ':.>13}[{rv_bin[0]:.2f}, {rv_bin[1]:.2f}) Mpc/h")
    print(" Redshift " + f"{': ':.>10}[{z_bin[0]:.2f}, {z_bin[1]:.2f})")
    print(" Type " + f"{': ':.>14}[{delta_bin[0]},{delta_bin[1]}) => {voidtype}")
    print(" # of voids " + f"{': ':.>12}{nvoids}\n", flush=True)


def run_pool(profile_func, lenses, lensrand, keys=("rv", "ra", "dec", "z")):
    """Per-void sums (see `partial_profile_map`) of the voids and of the random voids. `keys`:
    the columns of the lenses given to `profile_func`, in its order"""

    def lens_input(voids):
        cols = [cfg.lcols[k] for k in keys]
        return voids[cols].T

    def run(pool, voids):
        chunk = max(1, voids.shape[1] // (cfg.ncores * 20))
        return list(
            tqdm(
                pool.imap(profile_func, lens_input(voids), chunksize=chunk),
                total=voids.shape[1],
            )
        )

    with ctx.Pool(processes=cfg.ncores) as pool:
        resmap = run(pool, lenses)
        print("\n >> Calculating profiles for random voids...", flush=True)
        randmap = run(pool, lensrand)
    return resmap, randmap


def print_halo_settings(z_bin, mag_bin, lm_bin, nhalos):
    print(f" {' Halo sample ':=^60}")
    print(" Lens cat " + f"{': ':.>10}{cfg.lensname}")
    print(" Source cat " + f"{': ':.>8}{cfg.sourcename} ({cfg.source_mode})")
    print(" NCORES " + f"{': ':.>12}{cfg.ncores}\n")

    print(" RMIN " + f"{': ':.>14}{cfg.RIN:.2f} Mpc/h (comoving)")
    print(" RMAX " + f"{': ':.>14}{cfg.ROUT:.2f} Mpc/h (comoving)")
    print(" NBINS " + f"{': ':.>13}{cfg.NBINS:<2d}")
    print(" NJK " + f"{': ':.>15}{cfg.NJK:<2d}")
    print(" Binning " + f"{': ':.>11}{cfg.binning}\n")

    print(f" {' Halos ':=^60}")
    print(" Redshift " + f"{': ':.>10}[{z_bin[0]:.2f}, {z_bin[1]:.2f})")
    print(" M_r " + f"{': ':.>13}[{mag_bin[0]:.2f}, {mag_bin[1]:.2f})")
    if lm_bin is not None:
        print(" log10 M " + f"{': ':.>10}[{lm_bin[0]:.2f}, {lm_bin[1]:.2f})")
    print(" Colour " + f"{': ':.>11}{cfg.color}")
    print(" # of halos " + f"{': ':.>9}{nhalos}\n", flush=True)


def warn_empty_bins(centres, N, N_rand):
    """Warns of the radial bins of the full sample without sources or randoms: their
    profiles are NaN"""
    empty = np.flatnonzero((N[0] == 0) | (N_rand[0] == 0))
    if len(empty):
        R = ", ".join(f"{r:.3f}" for r in centres[empty])
        print(
            f" WARNING: {len(empty)} radial bin(s) without sources at R = {R} Mpc/h "
            "(the pixels are too big for them): their profiles are NaN.",
            flush=True,
        )


def stacking_halos(z_bin, mag_bin, lm_bin, gravity):
    global ZSRC_MIN

    z_min, z_max = z_bin
    mag_min, mag_max = mag_bin
    lm_min, lm_max = lm_bin if lm_bin is not None else (None, None)

    output_filename = output_path_halos(cfg, gravity, z_bin, mag_bin, lm_bin)
    check_output_exists(output_filename, overwrite=cfg.overwrite)

    halos = read_lens(
        filename=cfg.lensname,
        cat="halos",
        z_min=z_min,
        z_max=z_max,
        mag_min=mag_min,
        mag_max=mag_max,
        color=None if cfg.color == "all" else cfg.color,
        lm_min=lm_min,
        lm_max=lm_max,
        columns=cfg.hcols,
    )
    nhalos = halos.shape[1]
    if nhalos == 0:
        raise ValueError(
            f"No halos with z in [{z_min}, {z_max}), M_r in [{mag_min}, {mag_max}) "
            f"and colour {cfg.color}."
        )
    print_halo_settings(z_bin, mag_bin, lm_bin, nhalos)

    if cfg.source_mode == "map":
        profile_func = partial_profile_halo_map
        # before the pool forks: the workers share the collapsed maps
        collapse_slices(z_max)
    else:
        profile_func = partial_profile_halo
        # the sources of the bin: at least dz_buffer behind its farthest halo
        ZSRC_MIN = z_max + cfg.dz_buffer
    print(
        f" Sources {': ':.>11}z >= {ZSRC_MIN:.2f} ({cfg.weighting} weights)\n",
        flush=True,
    )

    # randoms: same (z, M_r, log M) as the halos of this stack, uniform on the footprint.
    # The seed needs positive numbers: the absolute magnitudes
    seed_edges = (z_min, z_max, -mag_max, -mag_min)
    if lm_bin is not None:
        seed_edges += (lm_min, lm_max)
    lensrand = generate_randoms(
        halos,
        cfg.rand_factor,
        stack_rng(cfg.rand_seed, *seed_edges),
        in_footprint,
        cfg.lcols["ra"],
        cfg.lcols["dec"],
    )
    nrands = lensrand.shape[1]
    print(
        f" Randoms {': ':.>11}{nrands} (x{cfg.rand_factor:g}, seed {cfg.rand_seed})\n",
        flush=True,
    )

    resmap, randmap = run_pool(
        profile_func, halos, lensrand, keys=("ra", "dec", "z")
    )

    print("\n >> Pool ended, stacking...", flush=True)

    kidx = assign_regions(REGIONS, halos[cfg.lcols["ra"]], halos[cfg.lcols["dec"]])
    kidx_rand = assign_regions(
        REGIONS, lensrand[cfg.lcols["ra"]], lensrand[cfg.lcols["dec"]]
    )
    # empty bins give 0/0 = NaN, reported by warn_empty_bins
    with np.errstate(divide="ignore", invalid="ignore"):
        profiles, N_inbin, N_inbin_rand = reduce_profiles(
            resmap, randmap, kidx, kidx_rand, cfg.NJK
        )
    centres = bin_centres(cfg.RIN, cfg.ROUT, cfg.NBINS, cfg.binning)
    warn_empty_bins(centres, N_inbin, N_inbin_rand)

    extra = {
        "weight": cfg.weighting,
        "dzbuffer": cfg.dz_buffer,
        "zsrcmin": ZSRC_MIN,
    }
    if cfg.source_mode == "map":
        extra.update(
            nside=SOURCE["nside"], order=SOURCE["order"], chiref=SOURCE["chiref"]
        )
    head = make_header_halos(cfg, halos, z_bin, mag_bin, lm_bin, nrands, extra)

    write_profiles(
        output_filename,
        head,
        centres,
        profiles,
        N_inbin,
        N_inbin_rand,
        REGIONS.centers,
        overwrite=cfg.overwrite,
    )
    print(f" >> File saved in: {output_filename}.\n", flush=True)


def n_stacks(cfg):
    """Number of stacks of one gravity in the config"""
    if cfg.lens_type == "halos":
        return len(cfg.zbins) * len(cfg.magbins) * len(cfg.lmbins)
    return len(cfg.zbins) * len(cfg.rvbins) * len(cfg.voidtype)


def stacking(rv_min, rv_max, z_min, z_max, voidtype, gravity):

    delta_min, delta_max = VOIDTYPE_DELTA[voidtype]
    rv_bin, z_bin, delta_bin = (rv_min, rv_max), (z_min, z_max), (delta_min, delta_max)

    output_filename = output_path(cfg, gravity, rv_bin, z_bin, voidtype)
    check_output_exists(output_filename, overwrite=cfg.overwrite)

    lenses = read_lens(
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
        columns=cfg.lcols,
    )
    nvoids = lenses.shape[1]

    print_stack_settings(rv_bin, z_bin, delta_bin, voidtype, nvoids)

    if cfg.source_mode == "map":
        profile_func = partial_profile_map
        # before the pool forks: the workers share the collapsed maps
        collapse_slices(z_max)
        print(
            f" Sources {': ':.>11}z >= {ZSRC_MIN:.2f} ({cfg.weighting} weights)\n",
            flush=True,
        )
    else:
        profile_func = partial_profile

    # random voids: same (Rv, z) as the voids of this stack, uniform on the footprint
    lensrand = generate_randoms(
        lenses,
        cfg.rand_factor,
        stack_rng(cfg.rand_seed, z_min, z_max, rv_min, rv_max, delta_min, delta_max),
        in_footprint,
        cfg.lcols["ra"],
        cfg.lcols["dec"],
    )
    nrands = lensrand.shape[1]
    print(
        f" Randoms {': ':.>11}{nrands} (x{cfg.rand_factor:g}, seed {cfg.rand_seed})\n",
        flush=True,
    )

    resmap, randmap = run_pool(profile_func, lenses, lensrand)

    print("\n >> Pool ended, stacking...", flush=True)

    # voids and randoms are left out together by region
    kidx = assign_regions(REGIONS, lenses[cfg.lcols["ra"]], lenses[cfg.lcols["dec"]])
    kidx_rand = assign_regions(
        REGIONS, lensrand[cfg.lcols["ra"]], lensrand[cfg.lcols["dec"]]
    )

    profiles, N_inbin, N_inbin_rand = reduce_profiles(
        resmap, randmap, kidx, kidx_rand, cfg.NJK
    )

    extra = None
    if cfg.source_mode == "map":
        extra = {
            "nside": SOURCE["nside"],
            "order": SOURCE["order"],
            "weight": cfg.weighting,
            "dzbuffer": cfg.dz_buffer,
            "zsrcmin": ZSRC_MIN,
            "chiref": SOURCE["chiref"],
        }
    head = make_header(cfg, lenses, rv_bin, z_bin, delta_bin, voidtype, nrands, extra)

    write_profiles(
        output_filename,
        head,
        bin_centres(cfg.RIN, cfg.ROUT, cfg.NBINS, cfg.binning),
        profiles,
        N_inbin,
        N_inbin_rand,
        REGIONS.centers,
        overwrite=cfg.overwrite,
    )
    print(f" >> File saved in: {output_filename}.\n", flush=True)


def main():
    global cfg

    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="lensing/config.toml", action="store"
    )
    parser.add_argument("--ncores", type=int, action="store")
    parser.add_argument("--gravity", type=str, action="store", nargs="+", required=True)
    parser.add_argument(
        "--use_threshold",
        type=str,
        action="store",
        choices=["08", "09"],
        default=None,
        help="void threshold of the file names; by default, the one of the config",
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
            if cfg.source_mode == "map":
                print(
                    f" >> Source mode: MAP (HEALPix lensing cube, {cfg.weighting} weights)"
                )
            else:
                print(" >> Source mode: CATALOG (galaxy catalog)")
            s1, s2 = cfg.gamma_sign
            print(f" >> Shear convention: (e1, e2) = ({s1:+g} gamma1, {s2:+g} gamma2)")
            if cfg.plot:
                print(" WARNING: [run] plot is not implemented yet, ignoring it.")
            if cfg.nback is not None:
                print(
                    " WARNING: [profile] nback is not implemented yet, using all the sources."
                )
            total = n_stacks(cfg) * len(args.gravity)
            if cfg.lens_type == "halos":
                print(
                    f" >> Running {len(cfg.zbins)} redshift bin(s) x {len(cfg.magbins)} magnitude bin(s) x {len(cfg.lmbins)} mass bin(s), halos of colour {cfg.color}."
                )
            else:
                print(
                    f" >> Running {len(cfg.zbins)} redshift bin(s) x {len(cfg.rvbins)} radius bin(s), for {len(cfg.voidtype)} void types."
                )
            print(f" >> Calculating {total} profiles")

        print(
            f" >> Running for -{grav.upper()}-  ({cfg.source_mode}: {cfg.sourcename})"
        )
        init_globals()
        if cfg.source_mode == "catalog":
            make_pix2idx_dict(SOURCE)
        else:
            print(
                f" >> Map mode: cube order {SOURCE['order']} NESTED, "
                f"{len(SOURCE['zmin'])} source slices.",
                flush=True,
            )

        if cfg.lens_type == "halos":
            for z_bin, mag_bin, lm_bin in product(cfg.zbins, cfg.magbins, cfg.lmbins):
                i += 1

                print(f"\n [{i}/{total}]")
                stacking_halos(z_bin, mag_bin, lm_bin, grav.upper())
        else:
            for (z_min, z_max), (rv_min, rv_max) in product(cfg.zbins, cfg.rvbins):
                for voidtype in cfg.voidtype:
                    i += 1

                    print(f"\n [{i}/{total}]")
                    stacking(rv_min, rv_max, z_min, z_max, voidtype, grav.upper())

    print(" End! ".center(15, "="))
    print(f" >> Took {(time() - tini) / 60.0:.3f} min <<\n\n")


if __name__ == "__main__":
    main()
