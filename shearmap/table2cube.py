"""Build a (npix, nz) lensing cube from the long (pix, zbin) table returned by CosmoHub.

Each row of the table holds, for one (HEALPix pixel, source-redshift slice), the number
of galaxies and *sums* of gamma1, gamma2, kappa and of the inverse comoving distance
1/chi of the sources. Sums and counts are kept (not means) so that slices and pixels can
be added exactly afterwards.

Query used to download the table (one per simulation; replace the table name for MG):

    SELECT
      pix,
      zbin,
      COUNT(*)                     AS nobj,
      SUM(gamma1)                  AS sum_gamma1,
      SUM(gamma2)                  AS sum_gamma2,
      SUM(kappa)                   AS sum_kappa,
      SUM(invchi)                  AS sum_invchi,
      SUM(invchi * invchi)         AS sum_invchi2,
      SUM(gamma1 * invchi)         AS sum_gamma1_invchi,
      SUM(gamma2 * invchi)         AS sum_gamma2_invchi,
      SUM(kappa * invchi)          AS sum_kappa_invchi
    FROM (
      SELECT
        shiftright(hpix_29_nest, 36) AS pix,   -- order-11 NESTED pixel
        CASE
          WHEN true_redshift_gal < 0.3 THEN 0
          WHEN true_redshift_gal < 0.4 THEN 1
          WHEN true_redshift_gal < 0.5 THEN 2
          WHEN true_redshift_gal < 0.6 THEN 3
          ELSE 4
        END                          AS zbin,
        gamma1,
        gamma2,
        kappa,
        1.0 / r_gal                  AS invchi  -- r_gal: comoving distance, Mpc/h
      FROM
        l768_gr_v1_1_lensing_c_zphot
      WHERE
        true_redshift_gal >= 0.2
        AND ABS(gamma1) < 10.0
        AND ABS(gamma2) < 10.0
    ) t
    GROUP BY
      pix, zbin

The sources of a void bin starting at z_low are the slices with zmin >= z_low + 0.1,
i.e. slice k serves the void bins starting at 0.1 * (1 ... k + 1). The edges of the slices
are the constant `ZEDGES` below, which must agree with the CASE of the query.

The mean Sigma_crit^-1 of the sources in a pixel follows from the stored sums. For a
flat universe, Sigma_crit^-1(z_l, z_s) = A(z_l) * (1 - chi_l / chi_s), so with
rho = CHI_REF / chi_s and l = chi_l / CHI_REF:

    sum_i Sigma_crit_i^-1        = A * (N - l * SUM_INVCHI)
    sum_i Sigma_crit_i^-2        = A^2 * (N - 2 l SUM_INVCHI + l^2 SUM_INVCHI2)
    sum_i Sigma_crit_i^-1 gamma  = A * (SUM_GAMMA - l * SUM_GAMMA_INVCHI)

All the sums with 1/chi are stored multiplied by CHI_REF (squared for SUM_INVCHI2) so
that float32 holds values of order one.

Layout of the output FITS file (NESTED ordering, axis 0 = pixel, axis 1 = zbin):
    NOBJ                                                   int16 (int32 if needed)
    SUM_GAMMA1, SUM_GAMMA2, SUM_KAPPA, SUM_INVCHI          float32
    SUM_INVCHI2, SUM_GAMMA{1,2}_INVCHI, SUM_KAPPA_INVCHI   float32, optional
    ZBINS                                                  table: zbin, zmin, zmax
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits

ORDER = 11  # HEALPix order of the pixels of the query
ZEDGES = (0.2, 0.3, 0.4, 0.5, 0.6, 1.4)  # edges of the zbin slices (must match the query)
CHI_REF = 1000.0  # Mpc/h, scale of the 1/chi sums

# names inside the cube. Table columns are `nobj` and `sum_<name>`, the HDUs
# `NOBJ` and `SUM_<NAME>`.
REQUIRED = ("gamma1", "gamma2", "kappa", "invchi")
OPTIONAL = ("invchi2", "gamma1_invchi", "gamma2_invchi", "kappa_invchi")
SUMS = REQUIRED + OPTIONAL


def table_column(name):
    return "nobj" if name == "nobj" else f"sum_{name}"


def hdu_name(name):
    return table_column(name).upper()


def scale(name):
    """Factor applied to the sums before storing them in float32"""
    if name == "invchi2":
        return CHI_REF**2
    if name.endswith("invchi"):
        return CHI_REF
    return 1.0


def check_columns(colnames):
    """Check the columns of the table.

    Stops (ValueError) if any required column is missing, listing all of them.
    Missing optional columns are not fatal.

    Returns:
        (list of the optional names found, bool: all optional columns found)
    """
    colnames = {c.lower() for c in colnames}
    required = ["pix", "zbin", "nobj"] + [table_column(n) for n in REQUIRED]
    missing = [c for c in required if c not in colnames]
    if missing:
        raise ValueError(
            f"Required column(s) missing in the table: {', '.join(missing)}. "
            "Nothing was written."
        )

    found = [n for n in OPTIONAL if table_column(n) in colnames]
    absent = [table_column(n) for n in OPTIONAL if n not in found]
    if not found:
        print(
            " NOTE: no optional columns found. Only the Sigma_crit^-1 weighting "
            "will be available."
        )
    elif absent:
        print(
            f" WARNING: optional column(s) missing: {', '.join(absent)}. "
            "The Sigma_crit^-2 weighting will not be available."
        )
    return found, len(absent) == 0


def build_cube(table_path, order=ORDER):
    """Return a dict with the cube built from the CosmoHub table `table_path` (FITS).

    Keys: `order`, `nside`, `nest` (True), `nobj` (npix, nz), one array (npix, nz) per
    name in SUMS found, `zmin`, `zmax`, `optweight`, `chiref`, `comments` (the
    header of the table, with the query) and `source`.

    The table is read column by column. Memory is ~9 GB at order 11 for the whole
    cube, plus the 2 GB of the flat index.
    """
    nside = 2**order
    npix = 12 * nside**2
    nz = len(ZEDGES) - 1

    with fits.open(table_path, memmap=True) as hdul:
        hdu = hdul[1]
        names = {n.lower(): n for n in hdu.columns.names}
        optional, optweight = check_columns(names)
        data = hdu.data
        nrows = len(data)

        def col(name, dtype):
            return np.asarray(data[names[name]], dtype=dtype)

        # ---- indices
        pix = col("pix", np.int64)
        zbin = col("zbin", np.int64)
        if pix.min() < 0 or pix.max() >= npix:
            raise ValueError(
                f"Found HEALPix pixels outside [0, {npix}). Is the table of order {order}?"
            )
        if zbin.min() < 0 or zbin.max() >= nz:
            raise ValueError(
                f"Found zbin outside [0, {nz}). The table does not match ZEDGES."
            )
        flat = pix * nz + zbin
        del pix, zbin

        occupancy = np.bincount(flat, minlength=npix * nz)
        if occupancy.max() > 1:
            raise ValueError("Duplicated (pix, zbin) rows in the table.")
        empty = (occupancy == 0).reshape(npix, nz).mean(axis=0)
        del occupancy
        if np.any(empty == 1.0):
            raise ValueError(f"zbin(s) {np.flatnonzero(empty == 1.0)} have no rows.")

        cube = {
            "order": order,
            "nside": nside,
            "nest": True,
            "zmin": np.array(ZEDGES[:-1], dtype=np.float64),
            "zmax": np.array(ZEDGES[1:], dtype=np.float64),
            "chiref": CHI_REF,
            "optweight": optweight,
            "comments": [str(c) for c in hdu.header.get("COMMENT", [])],
            "source": os.path.basename(str(table_path)),
        }

        # ---- counts
        nobj_t = col("nobj", np.int64)
        if nobj_t.min() < 1:
            raise ValueError("Found rows with nobj < 1.")
        if nobj_t.max() < np.iinfo(np.int16).max:
            dtype = np.int16
        else:
            print(
                f" WARNING: nobj reaches {nobj_t.max()}, storing counts as int32 "
                "(int16 would overflow)."
            )
            dtype = np.int32
        nobj = np.zeros(npix * nz, dtype=dtype)
        nobj[flat] = nobj_t
        if nobj.sum(dtype=np.int64) != nobj_t.sum():
            raise ValueError("Sum of nobj changed while building the cube.")
        cube["nobj"] = nobj.reshape(npix, nz)
        del nobj_t, nobj

        # ---- sums
        for name in REQUIRED + tuple(optional):
            x = col(table_column(name), np.float64) * scale(name)
            if not np.all(np.isfinite(x)):
                raise ValueError(f"Non-finite values in {table_column(name)}.")
            arr = np.zeros(npix * nz, dtype=np.float32)
            arr[flat] = x
            # float32 rounding is ~6e-8 per cell, so the totals must agree well below 1e-6
            diff = abs(arr.sum(dtype=np.float64) - x.sum())
            if diff > 1e-6 * np.abs(x).sum():
                raise ValueError(f"Sum of {table_column(name)} changed on casting.")
            cube[name] = arr.reshape(npix, nz)
            del x, arr

    print(f" Table: {nrows} rows -> cube of shape ({npix}, {nz}), order {order} NESTED")
    for k in range(nz):
        print(
            f"   zbin {k} [{ZEDGES[k]:.1f}, {ZEDGES[k + 1]:.1f}): "
            f"{100 * empty[k]:.2f}% of the pixels have no sources"
        )
    return cube


def save_cube(cube, output_file, overwrite=False):
    """Write the dict of `build_cube` (or `degrade_cube`) to a FITS file."""
    if os.path.exists(output_file) and not overwrite:
        raise OSError(f"{output_file} exists. Use overwrite=True / --overwrite.")

    nz = len(cube["zmin"])
    primary = fits.PrimaryHDU()
    h = primary.header
    h["ORDER"] = (cube["order"], "HEALPix order; NSIDE = 2**ORDER")
    h["NSIDE"] = (cube["nside"], "HEALPix NSIDE")
    h["ORDERING"] = ("NESTED", "HEALPix ordering scheme")
    h["NPIX"] = (12 * cube["nside"] ** 2, "Number of HEALPix pixels")
    h["NZ"] = (nz, "Number of source-redshift slices")
    h["CHIREF"] = (cube["chiref"], "[Mpc/h] scale of the 1/chi sums")
    h["OPTWGT"] = (cube["optweight"], "T if the Sigma_crit^-2 sums are present")
    h.add_comment("Data arrays have shape (NPIX, NZ); column k is slice ZBINS[k].")
    h.add_comment("SUM_* are sums over the objects in the pixel; NOBJ their number.")
    h.add_comment("SUM_INVCHI = CHIREF*sum(1/chi), SUM_INVCHI2 = CHIREF^2*sum(1/chi^2),")
    h.add_comment("SUM_X_INVCHI = CHIREF*sum(X/chi); chi is the comoving distance.")
    h["HISTORY"] = (
        f"Built by table2cube.py from {cube.get('source', '?')} on "
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    )
    if cube.get("comments"):
        h.add_comment("---- header of the input table (CosmoHub query) ----")
        for line in cube["comments"]:
            h.add_comment(line)
    primary.writeto(output_file, overwrite=True)

    # one array at a time: astropy byte-swaps a copy of each when writing
    for name in ("nobj",) + SUMS:
        if name in cube:
            fits.append(output_file, cube[name], fits.Header({"EXTNAME": hdu_name(name)}))

    zbins = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="zbin", format="I", array=np.arange(nz)),
            fits.Column(name="zmin", format="D", array=cube["zmin"]),
            fits.Column(name="zmax", format="D", array=cube["zmax"]),
        ],
        name="ZBINS",
    )
    fits.append(output_file, zbins.data, zbins.header)


def load_cube(path):
    """Open a saved cube. The images are memmapped, not loaded in memory, so the file
    is left open on purpose (`cube["hdul"]`).

    Returns a dict like `build_cube`, without `comments`.
    """
    hdul = fits.open(os.path.expanduser(path), memmap=True)
    head = hdul[0].header
    cube = {
        "hdul": hdul,
        "order": int(head["ORDER"]),
        "nside": int(head["NSIDE"]),
        "nest": head["ORDERING"].strip().upper() == "NESTED",
        "chiref": float(head["CHIREF"]),
        "optweight": bool(head["OPTWGT"]),
        "nobj": hdul["NOBJ"].data,
    }
    for name in SUMS:
        if hdu_name(name) in hdul:
            cube[name] = hdul[hdu_name(name)].data
    zbins = hdul["ZBINS"].data
    cube["zmin"] = np.asarray(zbins["zmin"], dtype=np.float64)
    cube["zmax"] = np.asarray(zbins["zmax"], dtype=np.float64)
    return cube


def degrade_cube(cube, order_out):
    """Lower the resolution of a NESTED cube to `order_out` by summing the child pixels.

    Exact, since the cube stores sums and counts. Useful to test the effect of the
    pixel size without downloading another table.
    """
    order = cube["order"]
    if not cube["nest"]:
        raise ValueError("degrade_cube needs a NESTED cube.")
    if order_out > order:
        raise ValueError(f"order_out must be <= {order}.")
    nchild = 4 ** (order - order_out)

    out = {k: v for k, v in cube.items() if k not in ("hdul", "nobj") + SUMS}
    out["order"] = order_out
    out["nside"] = 2**order_out
    for name in ("nobj",) + SUMS:
        if name not in cube:
            continue
        a = np.asarray(cube[name])
        acc = np.int64 if name == "nobj" else np.float64
        total = a.reshape(-1, nchild, a.shape[1]).sum(axis=1, dtype=acc)
        if name == "nobj":
            dtype = np.int16 if total.max() < np.iinfo(np.int16).max else np.int32
        else:
            dtype = np.float32
        out[name] = total.astype(dtype)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("table", help="FITS table downloaded from CosmoHub")
    ap.add_argument("output", help="output cube (FITS)")
    ap.add_argument("--order", type=int, default=ORDER, help="HEALPix order of the table")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        sys.exit(f"ERROR: {args.output} exists. Use --overwrite.")

    try:
        cube = build_cube(args.table, order=args.order)
    except ValueError as err:
        sys.exit(f"ERROR: {err}")
    save_cube(cube, args.output, overwrite=args.overwrite)
    print(f" Written {args.output}: nobj shape {cube['nobj'].shape}")


if __name__ == "__main__":
    main()
