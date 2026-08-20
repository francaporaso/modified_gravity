import numpy as np
from lensing.io import read_sources_catalog

print("reading parquet...")
source = read_sources_catalog(
    "/home/fcaporaso/cats/L768/l768_gr_z020-130_wpix64_23087.parquet",
    cat="parquet",
    use_threads=False,
)
print("done")

print("check for pix order...")
print(np.all(source["pix"][:-1] <= source["pix"][1:]))
print("done")
