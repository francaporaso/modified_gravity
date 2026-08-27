from astropy.io import fits
from astropy.table import Table
import numpy as np
from pathlib import Path
#import pyarrow as pa
import pyarrow.parquet as pq

# ======================================================================
def read_sparkling(filename,
                   Rv_min, Rv_max, 
                   z_min, z_max, 
                   delta_min, delta_max, 
                   rho_min=-1.0, rho_max=0.0, 
                   flag=2,
                   has_id=False, 
                   fullshape=True):
    '''
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
            column. Default = False.
        fullshape (bool, optional): If True, returns all 12 Sparkling columns. 
            Else, only [Rv, ra, dec, z]. Default = True.
    
    Returns:
        ndarray: Array of shape (N, V), where N is the number of columns (12 if 
        `fullshape` is set to True, else 4) and V is the number of filtered voids.
    '''

    FLG = 11
    if has_id:
        RV,RA,DEC,Z,R1,R2 = 1,2,3,4,8,9
    else:
        RV,RA,DEC,Z,R1,R2 = 0,1,2,3,7,8
    # 0:Rv, 1:ra, 2:dec, 3:z, 4:xv, 5:yv, 6:zv, 7:rho1, 8:rho2, 9:logp, 10:diff CdM y CdV, 11:flag
    # CdM: centro de masa
    # CdV: centro del void
    L = np.loadtxt(filename, dtype='f4').T

    mask = (
        (L[RV] >= Rv_min) & (L[RV] < Rv_max) & 
        (L[Z] >= z_min) & (L[Z] < z_max) & 
        (L[R1] >= rho_min) & (L[R1] < rho_max) & 
        (L[R2] >= delta_min) & (L[R2] < delta_max) & 
        (L[FLG] >= flag)
    )

    if fullshape:
        L = L[:, mask]
    else:
        L = L[[RV,RA,DEC,Z]][:, mask]

    return L

def read_redmapper(filename, **kwargs):
    raise NotImplementedError

LENS_READERS = {
    'sparkling': read_sparkling,
    'redmapper': read_redmapper,
}

# ======================================================================
def read_sources_fits(filename, **kwargs):
    '''
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
    '''

    sources = Table.read(filename, memmap=True, format='fits', **kwargs)
    return sources

def read_sources_parquet(filename, **kwargs):
    '''
    Reads sources files with .parquet format using pyarrow's parquet module.

    Args: 
        filename (str or Path): Name of the file. Can be either a str or a Path
            object.
        **kwargs: Additional keyword arguments passed to the 
            pyarrow.parquet.read_table function.
    '''

    sources = pq.read_table(filename, **kwargs)
    return sources

SOURCES_READERS = {
    'fits': read_sources_fits,
    'parquet': read_sources_parquet,
}

# ======================================================================
def read_lens_catalog(filename, cat='sparkling', **kwargs):
    '''
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
        >>> catalog = read_sources_catalog("data/lenses.dat", cat="sparkling")
    '''
 
    filepath = Path(filename).expanduser()
    if cat not in LENS_READERS:
        raise ValueError(f'Unknown cat {cat}')
    return LENS_READERS.get(cat)(filepath, **kwargs)

def read_sources_catalog(filename, cat='parquet', **kwargs):
    '''
    Read a sources catalog file using the register `SOURCE_READERS`.

    Args:
        filename (str or Path): Path to the catalog file.
        cat: The reader format key. Must be one of registered readers in
            `SOURCES_READERS` (e.g., 'parquet', 'fits'). Defaults to 'parquet'.
        **kwargs: Additional keyword arguments passed directly to the 
            underlying reader function specified by `cat`.

    Returns:
        The catalog data object returned by the selected reader (e.g., DataFrame or Astropy Table).

    Raises:
        ValueError: If `cat` is not found in `SOURCES_READERS`.

    Example:
        >>> catalog = read_sources_catalog("data/sources.fits", cat="fits")
    '''
    filepath = Path(filename).expanduser()
    if cat not in SOURCES_READERS:
        raise ValueError(f'Unknown cat {cat}')
    return SOURCES_READERS.get(cat)(filepath, **kwargs)
