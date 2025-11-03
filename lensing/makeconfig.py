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
parser.add_argument('--voidtype', action='store', type=str, default='mix', choices=['mix', 'S', 'R'])
args = parser.parse_args()

radii = [np.linspace(args.Rv_min, args.Rv_max, args.nRv+1)]

zarange = np.arange(args.z_min, args.z_max+args.dz, args.dz)
redshift = np.array([[zarange[i], zarange[i+1]] for i in range(len(zarange)-1)])

if args.voidtype=='mix':
    delta = [(-1.0,0.0), (0.0,10.0)]
elif args.voidtype=='S':
    delta = [(0.0,10.0)]
elif args.voidtype=='R':
    delta = [(-1.0,0.0)]

print(f'{radii=}')
print(f'{redshift=}')
print(f'{delta=}')

i = 0
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
                'Rv_min':radii[0][0],
                'Rv_max':radii[0][1],
                'z_min':zs[0],
                'z_max':zs[1],
                'delta_min':d[0],
                'delta_max':d[1]
            },
            'sim':{
                'GR':{
                    'lens':{
                        'void08':'voids_LCDM_08.dat',
                        'void09':'voids_LCDM_09.dat'
                    },
                    'source':{
                        'for01-02':'l768_gr_z02-04_for01-02_w-pix64_19532.fits',
                        'for02-03':'l768_gr_z04-07_for02-03_w-pix64_19304.fits'
                    }
                },
                'fR':{
                    'lens':{
                        'void08':'voids_fR_08.dat',
                        'void09':'voids_fR_09.dat',
                    },
                    'source':{
                        'for01-02':'l768_mg_z02-04_for01-02_w-pix64_19531.fits',
                        'for02-03':'l768_mg_z04-07_for02-03_w-pix64_19260.fits'
                    }
                },

            },
        }
        with open(f'config_{i}.toml', 'w') as f:
            toml.dump(config, f)
        i+=1
