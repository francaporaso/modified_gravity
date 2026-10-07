from astropy.io import fits
from astropy.table import Table
import numpy as np

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

input_file = "../gr_gamma_map_order5_z01-14.fits"
ORDER = 5
NSIDE = 2**ORDER

ZMIN = 0.10
ZMAX = 1.40
DZ = 0.05

output_file = f"lensing-cube_nside{NSIDE}_z{ZMIN * 10:1.0f}-{ZMAX * 10:1.0f}.fits"


# ----------------------------------------------------------------------
# Read the table produced by CosmoHub
# ----------------------------------------------------------------------

tab = Table.read(input_file)

pix = np.asarray(tab["pix"], dtype=np.int64)
zbin = np.asarray(tab["zbin"], dtype=np.int64)

nobj_table = np.asarray(tab["nobj"], dtype=np.int64)
sum_gamma1_table = np.asarray(tab["sum_gamma1"], dtype=np.float64)
sum_gamma2_table = np.asarray(tab["sum_gamma2"], dtype=np.float64)


# ----------------------------------------------------------------------
# Define the dimensions of the cube
# ----------------------------------------------------------------------

npix = 12 * NSIDE**2

# Number of redshift bins.
# For [ZMIN, ZMAX) with uniform width DZ:
nz = int(round((ZMAX - ZMIN) / DZ))

if len(tab) != npix * nz:
    print(
        f"Warning: table contains {len(tab)} rows, "
        f"while a complete cube would contain {npix * nz} rows."
    )


# ----------------------------------------------------------------------
# Convert the long table into dense (zbin, pixel) arrays
# ----------------------------------------------------------------------

nobj = np.zeros((nz, npix), dtype=np.int64)
sum_gamma1 = np.zeros((nz, npix), dtype=np.float64)
sum_gamma2 = np.zeros((nz, npix), dtype=np.float64)


# The CosmoHub zbin convention in the test file is:
#
#   zbin = floor(z / DZ)
#
# Therefore z = 0.10--0.15 corresponds to zbin = 2.
#
# Convert the original zbin to a zero-based array index.
iz = zbin - int(round(ZMIN / DZ))


# Sanity checks
if np.any(pix < 0) or np.any(pix >= npix):
    raise ValueError("Found HEALPix pixel outside the expected range.")

if np.any(iz < 0) or np.any(iz >= nz):
    raise ValueError("Found redshift bin outside the expected range.")


# Since the SQL GROUP BY produces one row per (pixel, zbin),
# direct assignment is sufficient.
nobj[iz, pix] = nobj_table
sum_gamma1[iz, pix] = sum_gamma1_table
sum_gamma2[iz, pix] = sum_gamma2_table


# ----------------------------------------------------------------------
# Construct FITS header
# ---------------------------------------------------------------------


def construct_cube_fits(
    ORDER, NSIDE, npix, nz, ZMIN, ZMAX, DZ, nobj, sum_gamma1, sum_gamma2, output_file
):
    primary = fits.PrimaryHDU()
    header = primary.header

    header["ORDER"] = (
        ORDER,
        "HEALPix order; NSIDE = 2**ORDER",
    )

    header["NSIDE"] = (
        NSIDE,
        "HEALPix NSIDE",
    )

    header["NPIX"] = (
        npix,
        "Number of HEALPix pixels",
    )

    header["NZ"] = (
        nz,
        "Number of redshift bins",
    )

    header["ZMIN"] = (
        ZMIN,
        "Lower edge of first redshift bin",
    )

    header["ZMAX"] = (
        ZMAX,
        "Upper edge of last redshift bin",
    )

    header["DZ"] = (
        DZ,
        "Redshift-bin width",
    )

    header.add_comment(
        "Data arrays have shape (NZ, NPIX). "
        "Axis 0 is redshift bin and axis 1 is HEALPix pixel."
    )

    header.add_comment("Redshift bin i covers [ZMIN + i*DZ, ZMIN + (i+1)*DZ).")

    header.add_comment("Redshift-bin centre is ZMIN + (i + 0.5)*DZ.")

    header.add_comment(
        "The original CosmoHub zbin convention is retained only "
        "implicitly; the cube uses zero-based redshift-bin indices."
    )

    header.add_comment(
        "NOBJ contains galaxy counts. SUM_GAMMA1 and SUM_GAMMA2 "
        "contain the corresponding sums of shear components."
    )

    # ----------------------------------------------------------------------
    # Create image HDUs
    # ----------------------------------------------------------------------

    hdu_nobj = fits.ImageHDU(
        nobj,
        name="NOBJ",
    )

    hdu_gamma1 = fits.ImageHDU(
        sum_gamma1,
        name="SUM_GAMMA1",
    )

    hdu_gamma2 = fits.ImageHDU(
        sum_gamma2,
        name="SUM_GAMMA2",
    )

    # ----------------------------------------------------------------------
    # Write the cube
    # ----------------------------------------------------------------------

    hdul = fits.HDUList(
        [
            primary,
            hdu_nobj,
            hdu_gamma1,
            hdu_gamma2,
        ]
    )

    hdul.writeto(
        output_file,
        overwrite=True,
    )

    print(f"Written: {output_file}")
    print(f"ORDER = {ORDER}")
    print(f"NSIDE = {NSIDE}")
    print(f"NPIX  = {npix}")
    print(f"NZ    = {nz}")
    print(f"DZ    = {DZ}")
    print(f"Cube shape = ({nz}, {npix})")


construct_cube_fits(
    ORDER=ORDER,
    NSIDE=NSIDE,
    npix=npix,
    nz=nz,
    ZMIN=ZMIN,
    ZMAX=ZMAX,
    DZ=DZ,
    nobj=nobj,
    sum_gamma1=sum_gamma1,
    sum_gamma2=sum_gamma2,
    output_file=output_file,
)


# ----------------------------------------------------------------------
# Example: reconstruct the redshift bins from the FITS file
# ----------------------------------------------------------------------

# with fits.open(output_file, memmap=True) as hdul:
#
#    hdr = hdul[0].header
#
#    nz = hdr["NZ"]
#    zmin = hdr["ZMIN"]
#    zmax = hdr["ZMAX"]
#    dz = hdr["DZ"]
#
#    # Redshift-bin edges
#    z_edges = zmin + np.arange(nz + 1) * dz
#
#    # Redshift-bin centres
#    z_centers = zmin + (np.arange(nz) + 0.5) * dz
#
#    print("\nRedshift bins:")
#    for i in range(nz):
#        print(
#            f"{i:2d}: "
#            f"[{z_edges[i]:.3f}, {z_edges[i+1]:.3f}) "
#            f"center={z_centers[i]:.3f}"
#        )
