import os
from astropy.io import fits
from astropy.table import Table
import numpy as np
from pathlib import Path
from time import asctime

from lensing.funcs import cov_matrix
from lensing.jackknife import PROFILE_NAMES


# ======================================================================
# columns of the Sparkling void files without an ID column
SPARKLING_COLUMNS = dict(rv=0, ra=1, dec=2, z=3, rho1=7, delta=8, flag=11)
# 0:Rv, 1:ra, 2:dec, 3:z, 4:xv, 5:yv, 6:zv, 7:rho1, 8:rho2, 9:logp, 10:diff CdM y CdV, 11:flag
# CdM: centro de masa
# CdV: centro del void


def check_void_table(L, columns):
    """Stops if `columns` doesn't point to the right columns of the void table `L` (one row
    per column, as read by `read_sparkling`): with an ID column assumed or missing, the
    cuts would be made on the wrong columns and the profiles centred in the wrong place."""
    if L.shape[0] <= max(columns.values()):
        raise ValueError(
            f"[lenses.columns] points to column {max(columns.values())}, "
            f"but the void file has {L.shape[0]} columns."
        )
    ra, dec = L[columns["ra"]], L[columns["dec"]]
    if ra.min() < 0.0 or ra.max() > 360.0 or dec.min() < -90.0 or dec.max() > 90.0:
        raise ValueError(
            f"[lenses.columns] ra = {columns['ra']}, dec = {columns['dec']} don't look like "
            f"coordinates: ra in [{ra.min():.4g}, {ra.max():.4g}], dec in "
            f"[{dec.min():.4g}, {dec.max():.4g}]. Check the column numbers in the config "
            "(the void files have no ID column: rv=0, ra=1, dec=2, z=3, rho1=7, delta=8, flag=11)."
        )


def read_sparkling(
    filename,
    Rv_min,
    Rv_max,
    z_min,
    z_max,
    delta_min,
    delta_max,
    rho_min=-1.0,
    rho_max=0.0,
    flag=2,
    has_id=False,
    columns=None,
):
    """
    Reads Sparkling voids table into a numpy array

    Args:
        filename (str or Path): Name of the file. Can be a Path object.
        Rv_min (float): Minimum void radius in the table.
        Rv_max (float): Maximum void radius in the table.
        z_min (float): Minimum void redshift in the table.
        z_max (float): Maximum void redshift in the table.
        delta_min (float): Minimum value of the integrated density contrast in
            the table, used for separating void type.
        delta_max (float): Maximum value of the integrated density contrast in
            the table, used for separating void type.
        rho_min (float, optional): Minimum value of the integrated density
            contrast at void centre. Default = -1.0.
        rho_max (float, optional): Maximum value of the integrated density
            contrast at void centre. Default = 0.0.
        flag (int, optional): Flag threshold for proximity to survey wall.
            flag < 1: void centre is not fully inside survey.
            1<= flag < 2: void centre is between 1 and 2 radii of the wall.
            flag==2: void is inside survey up to 2 void radii.
            Default = 2.
        has_id (bool, optional): Set to True if the file contains ID as first
            column, which shifts the default columns by one. Default = False.
        columns (dict, optional): column numbers of 'rv', 'ra', 'dec', 'z', 'rho1',
            'delta' and 'flag' (the `[lenses.columns]` of the config). The ones that are
            missing take the default `SPARKLING_COLUMNS`.

    Returns:
        ndarray: Array of shape (N, V), where N is the number of columns (12 without
        ID) and V is the number of filtered voids.
    """

    cols = {k: v + int(has_id) for k, v in SPARKLING_COLUMNS.items()}
    cols.update(columns or {})

    L = np.loadtxt(filename, dtype="f4").T
    check_void_table(L, cols)

    mask = (
        (L[cols["rv"]] >= Rv_min)
        & (L[cols["rv"]] < Rv_max)
        & (L[cols["z"]] >= z_min)
        & (L[cols["z"]] < z_max)
        & (L[cols["rho1"]] >= rho_min)
        & (L[cols["rho1"]] < rho_max)
        & (L[cols["delta"]] >= delta_min)
        & (L[cols["delta"]] < delta_max)
        & (L[cols["flag"]] >= flag)
    )

    return L[:, mask]


def read_redmapper(filename, **kwargs):
    raise NotImplementedError


LENS_READERS = {
    "sparkling": read_sparkling,
    "redmapper": read_redmapper,
}


# ======================================================================
def read_sources_fits(filename, **kwargs):
    """
    Reads source files with .fits format. Uses `astorpy.table.Table` with memmap
    set to True.

    Args:
        filename (str or Path): Name of the file, can be either str or Path
            object.
        **kwargs: Additional keyword arguments passed to the
            astropy Table object.

    Returns:
        astropy.table.Table : An astropy table object. Columns can be accesed
            using named keys.
    """

    sources = Table.read(filename, memmap=True, format="fits", **kwargs)
    return sources


def read_sources_map(filename):
    """
    Reads a lensing cube built with `shearmap/table2cube.py`. The image
    arrays (shape (NPIX, NZ), NESTED) are memmapped, not loaded in memory, so
    the file is left open on purpose.

    Args:
        filename (str or Path): Name of the cube file.

    Returns:
        dict: see `shearmap.table2cube.load_cube`. `order`, `nside`, `nest`,
            `chiref`, `optweight`, `zmin`/`zmax` (per slice), `nobj`, and the
            sums `gamma1`, `gamma2`, `kappa`, `invchi` (+ optional ones).
    """
    from shearmap.table2cube import load_cube

    return load_cube(Path(filename).expanduser())


SOURCES_READERS = {
    "fits": read_sources_fits,
    "map": read_sources_map,
}


# ======================================================================
def read_lens_catalog(filename, cat="sparkling", **kwargs):
    """
    Read a lens catalog file using the register `LENS_READERS`.

    Args:
        filename (str or Path): Path to the catalog file.
        cat: The reader format key. Must be one of registered readers in
            `LENS_READERS` (e.g., 'sparkling', 'redmapper'). Defaults to 'sparkling'.
        **kwargs: Additional keyword arguments passed directly to the
            underlying reader function specified by `cat`.

    Returns:
        The catalog data object returned by the selected reader (e.g., ndarray).

    Raises:
        ValueError: If `cat` is not found in `LENS_READERS`.

    Example:
        >>> catalog = read_lens_catalog("data/lenses.dat", cat="sparkling", Rv_min=10.0, ...)
    """

    filepath = Path(filename).expanduser()
    if cat not in LENS_READERS:
        raise ValueError(f"Unknown cat {cat}")
    return LENS_READERS.get(cat)(filepath, **kwargs)


def read_sources_catalog(filename, cat="fits", **kwargs):
    """
    Read a sources catalog file using the register `SOURCES_READERS`.

    Args:
        filename (str or Path): Path to the catalog file.
        cat: The reader format key. Must be one of registered readers in
            `SOURCES_READERS` (e.g., 'fits', 'map'). Defaults to 'fits'.
        **kwargs: Additional keyword arguments passed directly to the
            underlying reader function specified by `cat`.

    Returns:
        The catalog data object returned by the selected reader (e.g., Astropy Table).

    Raises:
        ValueError: If `cat` is not found in `SOURCES_READERS`.

    Example:
        >>> catalog = read_sources_catalog("data/sources.fits", cat="fits")
    """
    filepath = Path(filename).expanduser()
    if cat not in SOURCES_READERS:
        raise ValueError(f"Unknown cat {cat}")
    return SOURCES_READERS.get(cat)(filepath, **kwargs)


# ======================================================================
# results of a stack
def output_path(cfg, gravity, rv_bin, z_bin, voidtype):
    """Path of the results file of a stack"""
    rv_min, rv_max = rv_bin
    z_min, z_max = z_bin
    name = (
        f"lensing/results/lensing_{gravity}_thr{cfg.voidthr}_{cfg.sample}_N{cfg.NBINS}_"
        f"Rv{rv_min:02.0f}-{rv_max:02.0f}_"
        f"z{100 * z_min:03.0f}-{100 * z_max:03.0f}_"
        f"type{voidtype}_bin{cfg.binning}"
    )
    if cfg.source_mode == "map":
        name += "_map"
    if cfg.addnoise:
        name += "w-noise"
    return Path(name + ".fits").expanduser()


def check_output_exists(output_file, overwrite=False):
    """Stops if the results file exists and `overwrite` is not set"""
    if os.path.exists(output_file):
        if not overwrite:
            raise OSError(
                f"\n{'=' * 60}\n"
                f"Output file already exists: {output_file}\n"
                f"Use --overwrite flag to allow overwriting, or choose a different sample name.\n"
                f"{'=' * 60}"
            )
        print(f" WARNING: Will overwrite existing file: {output_file}", flush=True)


def lens_summary(lenses, lcols):
    """Mean and median of the properties of the stacked voids"""
    col = lambda key: lenses[lcols[key]]
    return dict(
        nvoids=lenses.shape[1],
        z_mean=col("z").mean(),
        z_median=np.median(col("z")),
        Rv_mean=col("rv").mean(),
        Rv_median=np.median(col("rv")),
        delta_mean=col("delta").mean(),
        delta_median=np.median(col("delta")),
    )


def make_header(cfg, lenses, rv_bin, z_bin, delta_bin, voidtype, nrand, extra=None):
    """Header of the results file; `extra` (dict) is added to it"""
    summary = lens_summary(lenses, cfg.lcols)
    head = fits.Header()
    head.update(
        {
            "nvoids": summary["nvoids"],
            "lenscat": cfg.lensname,
            "sourcat": cfg.sourcename,
            "Rv_min": rv_bin[0],
            "Rv_max": rv_bin[1],
            "Rv_mean": summary["Rv_mean"],
            "Rv_med": summary["Rv_median"],
            "z_min": z_bin[0],
            "z_max": z_bin[1],
            "z_mean": summary["z_mean"],
            "z_med": summary["z_median"],
            "voidtype": voidtype,
            "deltamin": delta_bin[0],
            "deltamax": delta_bin[1],
            "deltamea": summary["delta_mean"],
            "deltamed": summary["delta_median"],
            "RIN": cfg.RIN,
            "ROUT": cfg.ROUT,
            "NBINS": cfg.NBINS,
            "NJK": cfg.NJK,
            "JKSEED": cfg.jk_seed,
            "binning": cfg.binning,
            "srcmode": cfg.source_mode,
            "nrand": nrand,
            "randfac": cfg.rand_factor,
            "randseed": cfg.rand_seed,
            "randsub": True,
            "HISTORY": f"{asctime()}",
        }
    )
    head.update(extra or {})
    return head


def profile_hdus(profiles):
    """Covariance and jackknife samples (HDUs `cov_*` and `jack_*`) of each profile, the
    covariance being that of the same estimator as the profile."""
    names = [n + s for s in ("", "_raw", "_rand") for n in PROFILE_NAMES]
    cov = [fits.ImageHDU(cov_matrix(profiles[n][1:]), name=f"cov_{n}") for n in names]
    jack = [fits.ImageHDU(profiles[n][1:], name=f"jack_{n}") for n in names]
    return cov + jack


def write_profiles(
    path, header, centres, profiles, N, N_rand, jk_centres, overwrite=False
):
    """Writes the results of a stack. `profiles` are the dict of `reduce_profiles`, with the
    full sample in row 0 and the jackknife samples after it.

    HDUs: `profiles` (table), `cov_*` and `jack_*` of each profile, and `jk_centres`, the
    (ra, dec) of the jackknife regions.
    Sigma, DSigma_t, DSigma_x: voids minus randoms; *_raw: voids; *_rand: randoms.
    """
    table = Table({"R": centres})
    for suffix in ("", "_raw", "_rand"):
        for name in PROFILE_NAMES:
            table[name + suffix] = profiles[name + suffix][0]
    table["N"] = N[0]
    table["N_rand"] = N_rand[0]

    regions = Table({"ra": jk_centres[:, 0], "dec": jk_centres[:, 1]})
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.BinTableHDU(table, name="profiles"),
            *profile_hdus(profiles),
            fits.BinTableHDU(regions, name="jk_centres"),
        ]
    )
    hdul.writeto(path, overwrite=overwrite)
