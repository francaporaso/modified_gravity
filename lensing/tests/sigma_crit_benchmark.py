import numpy as np
import time
from astropy.cosmology import LambdaCDM
from astropy.constants import G, c, M_sun, pc

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

def sigma_crit_orig(cosmo, z_l, z_s):
    d_l  = cosmo.angular_diameter_distance(z_l).value * pc.value * 1.0e6
    d_s  = cosmo.angular_diameter_distance(z_s).value
    d_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).value
    return (((c.value**2.0)/(4.0*np.pi*G.value*d_l)) * (d_s/d_ls)) * (pc.value**2/M_sun.value)

def sigma_crit_vec(cosmo, z_l, z_s):
    z_l = np.atleast_1d(z_l)
    z_s = np.atleast_1d(z_s)
    zl, zs = np.broadcast_arrays(z_l, z_s)
    valid = zs > zl
    sigma_c = np.full(zs.shape, np.inf, dtype=float)
    if np.any(valid):
        d_l  = cosmo.angular_diameter_distance(zl[valid]).value * pc.value * 1.0e6
        d_s  = cosmo.angular_diameter_distance(zs[valid]).value
        d_ls = cosmo.angular_diameter_distance_z1z2(zl[valid], zs[valid]).value
        sigma_c[valid] = ((c.value**2)/(4.0*np.pi*G.value*d_l)) * (d_s/d_ls) * (pc.value**2/M_sun.value)
    if np.isscalar(z_l) and np.isscalar(z_s):
        return sigma_c.item()
    return sigma_c

# Benchmark helper
def timeit(func, *args, nrep=3):
    times = []
    for _ in range(nrep):
        t0 = time.time()
        func(*args)
        times.append(time.time() - t0)
    return np.mean(times)

# Test inputs
z_l = 0.3
z_s = np.random.uniform(0.35, 2.0, size=200_000)  # 200k sources

# Warmup (to avoid initialization overheads)
sigma_crit_orig(cosmo, z_l, z_s[:10])
sigma_crit_vec(cosmo, z_l, z_s[:10])

# Benchmark
t_orig = timeit(sigma_crit_orig, cosmo, z_l, z_s)
t_vec  = timeit(sigma_crit_vec, cosmo, z_l, z_s)

print(f"Original sigma_crit: {t_orig:.3f} s")
print(f"Vectorized sigma_crit: {t_vec:.3f} s")
print(f"Speedup: {t_orig/t_vec:.1f}x")
