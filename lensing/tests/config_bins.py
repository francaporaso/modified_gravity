import numpy as np

## === voids_LCDM_09.dat ===
# min(Rv), max(Rv)       = 7.14238, 28.66747
# min(z), max(z)         = 0.105433, 0.592559
# min(Delta), max(Delta) = -0.697946, 2.757803

## === voids_LCDM_08.dat ===
# min(Rv), max(Rv)       = 5.31699, 35.41642
# min(z), max(z)         = 0.104411, 0.593738
# min(Delta), max(Delta) = -0.778693, 3.901099

## === voids_fR_09.dat ===
# min(Rv), max(Rv)       = 7.07322, 28.35124
# min(z), max(z)         = 0.105426, 0.592579
# min(Delta), max(Delta) = -0.711576, 2.389899

## === voids_fR_08.dat ===
# min(Rv), max(Rv)       = 5.31251, 35.30436
# min(z), max(z)         = 0.104313, 0.593676
# min(Delta), max(Delta) = -0.751006, 3.618723

z_min, z_max, z_BINS = 0.2, 0.4, 3
z_bins = np.linspace(z_min, z_max, z_BINS+1)
print(f'{z_bins=}')

Rv_min, Rv_max, Rv_BINS = 7.14238, 28.66747, 3
Rv_bins = np.linspace(Rv_min, Rv_max, Rv_BINS+1)
print(f'{Rv_bins=}')

pad_S = 2.757*0.1
pad_R = -0.6979*0.1
deltaS_min, deltaS_max, deltaS_BINS = 0.0+pad_S, 2.757803, 1
deltaR_min, deltaR_max, deltaR_BINS = -0.697946, 0.0+pad_R, 1
deltaS_bins = np.linspace(deltaS_min, deltaS_max, deltaS_BINS+1)
deltaR_bins = np.linspace(deltaR_min, deltaR_max, deltaR_BINS+1)
print(f'{deltaS_bins=}')
print(f'{deltaR_bins=}')

