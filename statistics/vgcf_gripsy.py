from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import numpy as np
import grispy as gsp
from multiprocessing import Pool

import sys
sys.path.append('../lensing/')
from funcs import lenscat_load

def sphere2box(ra,dec,chi):
    cosdec = np.cos(np.deg2rad(dec))
    x = chi*np.cos(np.deg2rad(ra))*cosdec
    y = chi*np.sin(np.deg2rad(ra))*cosdec
    z = chi*np.sin(np.deg2rad(dec))
    return np.column_stack([x,y,z])

def source_load(name=''):
    t = Table.read(name, memmap=True, format='fits')
    if 'chi_gal' in t.columns:
        return sphere2box(
            t['ra_gal'],
            t['dec_gal'],
            t['chi_gal']
        )
    else:
        return sphere2box(
            t['ra_gal'],
            t['dec_gal'],
            cosmo.comoving_distance(t['redshift_gal']).value
        )

def count_pairs(grid, center, rad, RIN, ROUT, N):

    dist, _ = grid.shell_neighbors(
        sphere2box(*center),
        distance_lower_bound=rad*RIN,
        distance_upper_bound=rad*ROUT,
    )

    pairs, _ = np.histogram(dist, bins=np.linspace(RIN,ROUT,N+1)*rad)
    return pairs

def partial_xi(void):
    center = void[:3]
    rad = void[3]

    n_gal = grid_true.ndata
    n_rand = grid_rand.ndata

    DD = count_pairs(grid_true, center, rad, RIN, ROUT, N)
    RR = count_pairs(grid_true, center, rad, RIN, ROUT, N)

    xi = (DD/RR)*(n_rand/n_gal) - 1.0

    return xi

def void_galaxy_corrfunc(center, Rv):
    """
    Docstring para void_galaxy_corrfunc

    :param data_box: datos (ndarray (ngal,3))
    :param data_voids: voids (ndarray (nvoids,4)=> data_voids[:3]==(ra,dec,redshift); data_voids[3]==rv)
    :param rand_box: randoms (ndarray (nrand, 3))
    :param RIN: float min distance to calculate corrfunc in rv
    :param ROUT:  float max distance to calculate corrfunc in rv
    :param N: int number of bins of corrfunc
    """

    # paralelizar este for?
    with Pool(processes=NCORES) as pool:
        resmap = list(tqdm(pool.imap_unordered(partial_xi, [center, Rv])))

    xi = np.zeros(N)
    for j,r in enumerate(resmap):
        xi[:] += r

    return xi


grid_true = None
grid_rand = None
RIN, ROUT, N = None, None, None

def main(galname='../../cats/L768/l768_gr_z005-070_forcorrfunc.fits',
         voidname='../../cats/L768/voids_LCDM_09.dat'):

    global grid_true, grid_rand
    global RIN, ROUT, N

    RIN = 0.1
    ROUT = 3.0
    N = 20

    S = source_load(galname)
    L,_,nvoids = lenscat_load(voidname, Rv_min=8.0, Rv_max=30.0, z_min=0.2, z_max=0.25, delta_min=-1.0, delta_max=0.0, fullshape=False)

    rng = np.random.default_rng(0)
    n_rand = len(S)*5
    rand_box = sphere2box(
        *np.column_stack([
            rng.uniform(0.0,360.0,n_rand),
            rng.uniform(-90.0,90.0,n_rand),
            cosmo.comoving_distance(rng.uniform(0.2,0.25,n_rand)),value,
        ])
    )

    void_centre = sphere2box(L[1], L[2], cosmo.comoving_distance(L[3]).value)
    Rv = L[0]

    grid_true = gsp.GriSPy(void_centre, N_cells=64)
    grid_rand = gsp.GriSPy(rand_box, N_cells=64)

    xi = void_galaxy_corrfunc(void_centre, Rv)

    redge = np.linspace(RIN, ROUT, N+1)
    r = 0.5*(redge[:-1]+redge[1:])

    for i in range(N):
        print(f"{r[i]:9.3f} | {xi[i]:12.4f}")

def count_pairs_grispy_vectorized(positions, centers, radii, r_norm_edges, periodic=None, metric='euclid'):
    """
    Count pairs using vectorized shell_neighbors with array bounds.
    Most efficient approach for void-normalized bins!

    Parameters
    ----------
    positions : array_like, shape (N_points, dim)
        Point positions
    centers : array_like, shape (N_centers, dim)
        Center positions
    radii : array_like, shape (N_centers,)
        Radius of each center
    r_norm_edges : array_like
        Normalized bin edges (r/R)
    periodic : dict or None
        Periodic boundary conditions
    metric : str
        Distance metric

    Returns
    -------
    counts : array, shape (nbins,)
        Total pairs in each bin
    """

    # Build GriSPy grid from positions
    grid = gsp.GriSPy(positions, periodic=periodic, metric=metric)

    nbins = len(r_norm_edges) - 1
    counts = np.zeros(nbins, dtype=np.int64)

    # Query each shell using vectorized bounds
    for i in range(nbins):
        r_inner_norm = r_norm_edges[i]
        r_outer_norm = r_norm_edges[i + 1]

        # Create array of physical bounds for all centers at once
        lower_bounds = radii * r_inner_norm
        upper_bounds = radii * r_outer_norm

        # Single vectorized call for all centers!
        if r_inner_norm == 0:
            # First bin: use bubble_neighbors
            dist, ind = grid.bubble_neighbors(
                centers,
                distance_upper_bound=upper_bounds
            )
        else:
            # Other bins: use shell_neighbors with array bounds
            dist, ind = grid.shell_neighbors(
                centers,
                distance_lower_bound=lower_bounds,
                distance_upper_bound=upper_bounds
            )

        # Count total pairs across all centers
        counts[i] = sum(len(idx) for idx in ind)

    return counts


def galaxy_void_xcorr_grispy_fast(galaxy_pos, void_centers, void_radii, random_pos,
                                  nbins=20, rmin_factor=0.5, rmax_factor=3.0,
                                  estimator='landy_szalay', periodic=None, metric='euclid'):
    """
    Fast calculation using vectorized shell_neighbors with array bounds.
    This is the RECOMMENDED method - takes full advantage of GriSPy!

    Parameters
    ----------
    galaxy_pos : array_like, shape (N_gal, 3)
        Galaxy positions
    void_centers : array_like, shape (N_voids, 3)
        Void center positions
    void_radii : array_like, shape (N_voids,)
        Void radii
    random_pos : array_like, shape (N_rand, 3)
        Random catalog positions
    nbins : int
        Number of radial bins
    rmin_factor : float
        Minimum radius as fraction of void radius
    rmax_factor : float
        Maximum radius as fraction of void radius
    estimator : str
        'peebles' or 'landy_szalay'
    periodic : dict or None
        Periodic boundaries, e.g., {'x': (0, 1000), 'y': (0, 1000), 'z': (0, 1000)}
    metric : str
        Distance metric for GriSPy

    Returns
    -------
    r_norm : array
        Normalized radial bin centers
    xi : array
        Cross-correlation function
    DD : array
        Data-void pair counts
    RR : array
        Random-void pair counts
    """

    # Normalized bin edges
    r_norm_edges = np.linspace(rmin_factor, rmax_factor, nbins + 1)
    r_norm_centers = 0.5 * (r_norm_edges[:-1] + r_norm_edges[1:])

    print("Counting DD pairs (galaxies around voids) - vectorized...")
    DD = count_pairs_grispy_vectorized(galaxy_pos, void_centers, void_radii,
                                       r_norm_edges, periodic=periodic, metric=metric)

    print("Counting RR pairs (randoms around voids) - vectorized...")
    RR = count_pairs_grispy_vectorized(random_pos, void_centers, void_radii,
                                       r_norm_edges, periodic=periodic, metric=metric)

    # Normalization
    n_gal = len(galaxy_pos)
    n_rand = len(random_pos)

    # Calculate correlation function
    xi = np.zeros(nbins)

    if estimator == 'peebles':
        # Peebles estimator: ξ = (DD/RR) * (N_rand/N_gal) - 1
        mask = RR > 0
        xi[mask] = (DD[mask] / RR[mask]) * (n_rand / n_gal) - 1.0
        xi[~mask] = np.nan

    elif estimator == 'landy_szalay':
        # Landy-Szalay: ξ = (DD*N_rand - RR*N_gal) / (RR*N_gal)
        mask = RR > 0
        xi[mask] = (DD[mask] * n_rand - RR[mask] * n_gal) / (RR[mask] * n_gal)
        xi[~mask] = np.nan
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    return r_norm_centers, xi, DD, RR


def galaxy_void_xcorr_grispy_optimized(galaxy_pos, void_centers, void_radii, random_pos,
                                       nbins=20, rmin_factor=0.5, rmax_factor=3.0,
                                       estimator='landy_szalay', periodic=None, metric='euclid'):
    """
    Optimized version using bubble_neighbors and binning distances manually.
    Faster than multiple shell_neighbors calls.

    Parameters same as galaxy_void_xcorr_grispy.
    """

    # Build grids
    grid_gal = gsp.GriSPy(galaxy_pos, periodic=periodic, metric=metric)
    grid_rand = gsp.GriSPy(random_pos, periodic=periodic, metric=metric)

    n_voids = len(void_radii)
    r_norm_edges = np.linspace(rmin_factor, rmax_factor, nbins + 1)
    r_norm_centers = 0.5 * (r_norm_edges[:-1] + r_norm_edges[1:])

    DD = np.zeros(nbins, dtype=np.int64)
    RR = np.zeros(nbins, dtype=np.int64)

    print("Processing voids...")
    for i in range(n_voids):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_voids} voids")

        center = void_centers[i:i+1]
        radius = void_radii[i]

        # Physical bin edges for this void
        r_edges = r_norm_edges * radius
        max_radius = r_edges[-1]

        # Get all galaxies within max radius
        dist_gal, ind_gal = grid_gal.bubble_neighbors(
            center,
            distance_upper_bound=max_radius
        )

        # Bin the distances
        if len(ind_gal[0]) > 0:
            hist_dd, _ = np.histogram(dist_gal[0], bins=r_edges)
            DD += hist_dd

        # Get all randoms within max radius
        dist_rand, ind_rand = grid_rand.bubble_neighbors(
            center,
            distance_upper_bound=max_radius
        )

        # Bin the distances
        if len(ind_rand[0]) > 0:
            hist_rr, _ = np.histogram(dist_rand[0], bins=r_edges)
            RR += hist_rr

    # Calculate correlation function
    n_gal = len(galaxy_pos)
    n_rand = len(random_pos)

    xi = np.zeros(nbins)

    if estimator == 'peebles':
        mask = RR > 0
        xi[mask] = (DD[mask] / RR[mask]) * (n_rand / n_gal) - 1.0
        xi[~mask] = np.nan

    elif estimator == 'landy_szalay':
        mask = RR > 0
        xi[mask] = (DD[mask] * n_rand - RR[mask] * n_gal) / (RR[mask] * n_gal)
        xi[~mask] = np.nan
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    return r_norm_centers, xi, DD, RR


# Example usage
def example():
    np.random.seed(42)

    # Simulate data in a cubic box
    n_gal = 50000
    n_rand = n_gal*5  # 5x more randoms
    boxsize = 768.0  # Mpc/h

    galaxy_pos = np.random.uniform(0, boxsize, size=(n_gal, 3))
    random_pos = np.random.uniform(0, boxsize, size=(n_rand, 3))

    # Voids
    n_voids = 500
    void_centers = np.random.uniform(50, boxsize - 50, size=(n_voids, 3))
    void_radii = np.random.uniform(10, 50, size=n_voids)  # Mpc/h

    print("="*60)
    print("Galaxy-Void Cross-Correlation with GriSPy")
    print("="*60)
    print(f"N_galaxies: {n_gal}")
    print(f"N_randoms: {n_rand}")
    print(f"N_voids: {n_voids}")
    print(f"Box size: {boxsize} Mpc/h")
    print()

    # Set up periodic boundary conditions (optional)
    periodic = {
        0: (0, boxsize),
        1: (0, boxsize),
        2: (0, boxsize)
    }

    # Calculate with FAST VECTORIZED version (RECOMMENDED!)
    import time
    start = time.time()

    r_norm, xi_ls, DD, RR = galaxy_void_xcorr_grispy_fast(
        galaxy_pos, void_centers, void_radii, random_pos,
        nbins=20, rmin_factor=0.5, rmax_factor=3.0,
        estimator='landy_szalay',
        periodic=periodic,  # Use None for non-periodic
        metric='euclid'
    )

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f} seconds")

    # Also calculate Peebles for comparison
    print("\nCalculating Peebles estimator...")
    _, xi_peebles, _, _ = galaxy_void_xcorr_grispy_fast(
        galaxy_pos, void_centers, void_radii, random_pos,
        nbins=20, rmin_factor=0.5, rmax_factor=3.0,
        estimator='peebles',
        periodic=periodic,
        metric='euclid'
    )

    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"{'r/R_void':>9} | {'ξ (Peebles)':>12} | {'ξ (Landy-Sz)':>13} | {'DD':>7} | {'RR':>7}")
    print("-"*65)

    for i in range(len(r_norm)):
        print(f"{r_norm[i]:9.3f} | {xi_peebles[i]:12.4f} | {xi_ls[i]:13.4f} | "
              f"{DD[i]:7d} | {RR[i]:7d}")

    print("\nExpected behavior:")
    print("  ξ < 0 inside voids (r/R < 1)")
    print("  ξ > 0 at void shells (r/R ≈ 1-1.5)")
    print("  ξ → 0 far from voids (r/R > 2)")

    return r_norm, xi_ls, xi_peebles, DD, RR


if __name__ == "__main__":
    #r_norm, xi_ls, xi_peebles, DD, RR = example()
    main()
