import argparse
options = {
	'-sample':'pru',
	'-lens_cat':'voids_MICE.dat',
	'-Rv_min':0.,
	'-Rv_max':50.,
	'-rho1_min':-1.,
	'-rho1_max':1.,
	'-rho2_min':-1.,
	'-rho2_max':100.,
	'-FLAG':2.,
	'-z_min':0.1,
	'-z_max':0.5,
	'-addnoise':False,
	'-RIN':0.05,
	'-ROUT':5.,
	'-nbins':40,
	'-ncores':10,
	'-nslices':1.,
}

parser = argparse.ArgumentParser()
for key,val in options.items():
    parser.add_argument(key, action='store',dest=key[1:],default=val,type=type(val))
args = parser.parse_args()
print(args)
print(args.RIN)
print(args.Rv_min)
print(args.addnoise)
