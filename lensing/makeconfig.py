import numpy as np
from argparse import ArgumentParser
import toml

parser = ArgumentParser()
parser.add_argument('--dz', action='store', type=float, default=0.05)
parser.add_argument('--z_min', action='store', type=float, default=0.1)
parser.add_argument('--z_max', action='store', type=float, default=0.3)
parser.add_argument('--nRv', action='store', type=int, default=1)
parser.add_argument('--Rv_min', action='store', type=float, default=8.0)
parser.add_argument('--Rv_max', action='store', type=float, default=30.0)
parser.add_argument('--voidtype', action='store', type=str, default=None, choices=['mix', 'S', 'R'])
parser.add_argument('--delta_min', action='store', type=float, default=-1.0)
parser.add_argument('--delta_max', action='store', type=float, default=10.0)
args = parser.parse_args()

rvrange = np.linspace(args.Rv_min, args.Rv_max, args.nRv+1)
radii = np.column_stack([rvrange[:-1], rvrange[1:]])

nz = np.round((args.z_max-args.z_min)/args.dz).astype(int)+1
zrange = np.linspace(args.z_min, args.z_max, nz, endpoint=True)
redshift = np.column_stack([zrange[:-1], zrange[1:]])

if args.voidtype==None:
    delta = [(args.delta_min, args.delta_max)]
elif args.voidtype=='mix':
    delta = [(-1.0,0.0), (0.0,10.0)]
elif args.voidtype=='S':
    delta = [(0.0,10.0)]
elif args.voidtype=='R':
    delta = [(-1.0,0.0)]

print(f'{radii=}')
print(f'{redshift=}')
print(f'{delta=}')

i = 0
for rv in radii:
    for zs in redshift:
        for d in delta:
            config = {
                'NCORES':16,
                'NK':100,
                'BIN':'lin',
                'prof':{
                    'NDOTS':20,
                    'RIN':0.01,
                    'ROUT':2.0
                },
                'void': {
                    'Rv_min':float(rv[0]),
                    'Rv_max':float(rv[1]),
                    'z_min':float(zs[0]),
                    'z_max':float(zs[1]),
                    'delta_min':float(d[0]),
                    'delta_max':float(d[1])
                },
                'sim':{
                    'GR':{
                        'lens':{
                            'void08':'voids_LCDM_08.dat',
                            'void09':'voids_LCDM_09.dat'
                        },
                        'source':{
                            'full':'l768_gr_z020-130_wpix64_23087.fits',
                            'for01-02':'l768_gr_z02-04_for01-02_w-pix64_19532.fits',
                            'for02-03':'l768_gr_z04-07_for02-03_w-pix64_19304.fits',
                            'for05-06':'l768_gr_z060-130_for05-06_w-pix64_23014.fits'
                        }
                    },
                    'fR':{
                        'lens':{
                            'void08':'voids_fR_08.dat',
                            'void09':'voids_fR_09.dat',
                        },
                        'source':{
                            'full':'l768_mg_z020-130_wpix64_23086.fits',
                            'for01-02':'l768_mg_z02-04_for01-02_w-pix64_19531.fits',
                            'for02-03':'l768_mg_z04-07_for02-03_w-pix64_19260.fits',
                            'for05-06':'l768_mg_z060-130_for05-06_w-pix64_23015.fits'
                        }
                    },

                },
            }
            with open(f'config_{i}.toml', 'w') as f:
                toml.dump(config, f)
            i+=1
