from lensing.io import read_sources_catalog

source = read_sources_catalog('/home/fcaporaso/cats/L768/l768_gr_z020-130_wpix64_23087.parquet', cat='parquet')

print(np.all(source['pix'][:-1]<=source['pix'][1:])
