"""Build a (npix, nstep) lensing cube from a long (pix, step) table.

The table holds, for each (HEALPix pixel, simulation step), the number of objects
and the *sums* of gamma1, gamma2 and kappa over the objects in the pixel.
Sums and counts are kept (not means), so that coarse pixels can later be
stacked exactly as a weighted mean by number of objects.
"""
import argparse

import numpy as np
from astropy.io import fits
from astropy.table import Table

LENSING_COLS = ("gamma1", "gamma2", "kappa")


def build_cube(tab, step2z=None, nside=None):
    """Return dict with `nobj`, `gamma1`, `gamma2`, `kappa` of shape (npix, nstep),
    the `steps` array and, if `step2z` is given, `zmin`/`zmax` per step."""
    pix = np.asarray(tab["pix"], dtype=np.int64)
    step = np.asarray(tab["step"], dtype=np.int64)

    if nside is None:
        # the table is complete in pixels, so the largest pixel id fixes nside
        nside = int(round(np.sqrt((pix.max() + 1) / 12)))
    npix = 12 * nside**2
    if pix.min() < 0 or pix.max() >= npix:
        raise ValueError("Found HEALPix pixel outside the expected range.")

    steps = np.unique(step)
    iz = np.searchsorted(steps, step)

    if len(np.unique(pix * len(steps) + iz)) != len(tab):
        raise ValueError("Duplicated (pix, step) rows in the table.")

    cube = {"nside": nside, "steps": steps}
    nobj = np.zeros((npix, len(steps)), dtype=np.int64)
    nobj[pix, iz] = tab["nobj"]
    cube["nobj"] = nobj
    for name in LENSING_COLS:
        arr = np.zeros((npix, len(steps)), dtype=np.float64)
        arr[pix, iz] = tab[name]
        cube[name] = arr

    if step2z is not None:
        z = {int(s): (zmin, zmax) for s, zmin, zmax in
             zip(step2z["step"], step2z["zmin"], step2z["zmax"])}
        cube["zmin"] = np.array([z[int(s)][0] for s in steps], dtype=np.float64)
        cube["zmax"] = np.array([z[int(s)][1] for s in steps], dtype=np.float64)
        if "zmean" in step2z.colnames:
            zm = dict(zip(step2z["step"].tolist(), step2z["zmean"].tolist()))
            cube["zmean"] = np.array([zm[int(s)] for s in steps], dtype=np.float64)
    return cube


def mean_maps(cube):
    """Per-pixel mean of each lensing variable (NaN where there are no objects)."""
    n = cube["nobj"]
    return {
        name: np.divide(cube[name], n, out=np.full(n.shape, np.nan), where=n > 0)
        for name in LENSING_COLS
    }


def save_cube(cube, output_file):
    nside = cube["nside"]
    primary = fits.PrimaryHDU()
    h = primary.header
    h["NSIDE"] = (nside, "HEALPix NSIDE")
    h["ORDERING"] = ("RING", "HEALPix ordering scheme")
    h["NPIX"] = (12 * nside**2, "Number of HEALPix pixels")
    h["NSTEP"] = (len(cube["steps"]), "Number of simulation steps")
    h.add_comment("Data arrays have shape (NPIX, NSTEP); column j is STEPS[j].")
    h.add_comment("SUM_* are sums over objects in the pixel; divide by NOBJ for means.")

    hdus = [primary, fits.ImageHDU(cube["nobj"], name="NOBJ")]
    hdus += [fits.ImageHDU(cube[n], name=f"SUM_{n.upper()}") for n in LENSING_COLS]

    cols = [fits.Column(name="step", format="I", array=cube["steps"])]
    if "zmin" in cube:
        cols += [
            fits.Column(name="zmin", format="D", array=cube["zmin"]),
            fits.Column(name="zmax", format="D", array=cube["zmax"]),
        ]
        if "zmean" in cube:
            cols.append(fits.Column(name="zmean", format="D", array=cube["zmean"]))
    hdus.append(fits.BinTableHDU.from_columns(cols, name="STEPS"))
    fits.HDUList(hdus).writeto(output_file, overwrite=True)


def load_cube(path):
    with fits.open(path, memmap=True) as hdul:
        cube = {"nside": hdul[0].header["NSIDE"], "nobj": hdul["NOBJ"].data.copy()}
        for n in LENSING_COLS:
            cube[n] = hdul[f"SUM_{n.upper()}"].data.copy()
        for c in hdul["STEPS"].columns.names:
            cube[c if c != "step" else "steps"] = np.asarray(hdul["STEPS"].data[c])
    return cube


def load_cube_table(path):
    """Load a saved cube as an astropy Table with one row per pixel.

    Columns `pix`, `nobj`, `sum_gamma1`, `sum_gamma2`, `sum_kappa`; the last four have
    shape (nstep,) per row. `steps`, `zmin`, `zmax` (length nstep) and `nside`
    are stored in `tab.meta`.
    """
    cube = load_cube(path)
    npix = cube["nobj"].shape[0]
    tab = Table()
    tab["pix"] = np.arange(npix)
    tab["nobj"] = cube["nobj"]
    for n in LENSING_COLS:
        tab[f"sum_{n}"] = cube[n]
    for key in ("nside", "steps", "zmin", "zmax"):
        if key in cube:
            tab.meta[key] = cube[key]
    return tab


def degrade_cube(cube, nside_out):
    """Lower the resolution of a RING cube to `nside_out` by summing child pixels.

    Exact, since the cube stores sums and counts.
    """
    import healpy as hp

    nside = cube["nside"]
    if nside_out > nside or nside % nside_out:
        raise ValueError("nside_out must divide nside and be <= nside.")
    fac = (nside // nside_out) ** 2
    out = {k: v for k, v in cube.items() if k not in ("nobj", *LENSING_COLS)}
    out["nside"] = nside_out
    for name in ("nobj", *LENSING_COLS):
        nest = hp.reorder(np.asarray(cube[name]).T, r2n=True).T  # (npix, nstep) in NEST
        nest = nest.reshape(-1, fac, nest.shape[1]).sum(axis=1)
        out[name] = hp.reorder(nest.T, n2r=True).T
    return out


def stack_sources(cube, z_lens, criterion="zmin", dz_buffer=0.1):
    """Stack the steps behind a lens at `z_lens`, weighting by number of objects.

    `cube` is the dict from `load_cube`. A step is a source step if its `criterion`
    redshift (`zmin`, `zmid` or `zmax`) is >= `z_lens + dz_buffer`; `zmin` keeps only
    steps fully behind the lens, `zmax` also keeps the one straddling it.

    Returns a dict with `nobj`, `gamma1`, `gamma2`, `kappa` of shape (npix,), where the
    lensing entries are sum(sums) / sum(nobj) over the selected steps (NaN if nobj == 0),
    plus `nside`, `steps` (selected), `z_lens`, `dz_buffer` and `criterion`.
    """
    z = {"zmin": cube["zmin"], "zmax": cube["zmax"],
         "zmid": 0.5 * (cube["zmin"] + cube["zmax"])}[criterion]
    sel = z >= z_lens + dz_buffer
    if not sel.any():
        raise ValueError(f"No steps with {criterion} >= {z_lens + dz_buffer}.")

    nobj = cube["nobj"][:, sel].sum(axis=1)
    out = {"nobj": nobj}
    for n in LENSING_COLS:
        total = cube[n][:, sel].sum(axis=1)
        out[n] = np.divide(total, nobj, out=np.full(nobj.shape, np.nan), where=nobj > 0)
    out.update(nside=cube["nside"], steps=cube["steps"][sel], z_lens=z_lens,
               dz_buffer=dz_buffer, criterion=criterion)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("step2z", help="FITS table with step, zmin, zmax")
    ap.add_argument("output")
    args = ap.parse_args()

    tab = Table.read(args.table)
    cube = build_cube(tab, Table.read(args.step2z))
    save_cube(cube, args.output)
    print(f"Written {args.output}: nobj shape {cube['nobj'].shape}")
