from argparse import ArgumentParser
from astropy.io import fits
import numpy as np
import time
import treecorr
from multiprocessing import Pool

# import sys
# sys.path.append('/home/fcaporaso/modified_gravity/')
# from lensing.funcs import lenscat_load
from stattools import Catalogs, make_randoms


class VoidGalaxyCrossCorrelation:
    
    def __init__(self, config_treecorr):
        print('VGCC init',flush=True)
        self.config : dict = config_treecorr

    def load_treecorrcatalogs(self, cats):
        print('loading cats w treecorr',flush=True)
        if len(cats.lenses) <= self.config['NPatches']:
            print('NPatches < Nvoids..., changing to Nvoids-1',flush=True)
            self.config['NPatches'] = len(cats.lenses)-1

        ## Voids
        self.dvcat = treecorr.Catalog(
            ra=cats.lenses['ra'],
            dec=cats.lenses['dec'],
            w=np.ones(cats.nvoids),
            r=cats.lenses['dcom'],
            npatch=self.config['NPatches'],
            ra_units='deg',
            dec_units='deg',
        )
        print('dvcat done',flush=True)

        ## Tracers (gx)
        self.dgcat = treecorr.Catalog(
            ra=cats.sources['ra_gal'], 
            dec=cats.sources['dec_gal'], 
            w=np.ones(cats.ngals), 
            r=cats.sources['dcom_gal'], 
            patch_centers= self.dvcat.patch_centers,
            ra_units='deg', dec_units='deg'
        )
        print('dgcat done',flush=True)

        ## Random voids
        self.rvcat = treecorr.Catalog(
            ra=cats.random_lenses['ra'], 
            dec=cats.random_lenses['dec'], 
            w=np.ones(cats.nvoids*10), 
            r=cats.random_lenses['dcom'], 
            patch_centers= self.dvcat.patch_centers,
            ra_units='deg', dec_units='deg'
        )
        print('rvcat done',flush=True)

        ## Random tracers (gx)
        self.rgcat = treecorr.Catalog(
            ra=cats.random_sources['ra_gal'], 
            dec=cats.random_sources['dec_gal'], 
            w=np.ones(self.ngals*10), 
            r=cats.random_sources['dcom_gal'], 
            patch_centers= self.dvcat.patch_centers,
            ra_units='deg', dec_units='deg'
        )
        print('rgcat done',flush=True)

    def calculate_corr(self):
        print('calculating corr...',flush=True)
        DvDg = treecorr.NNCorrelation(
            nbins=self.config['ndots'], 
            min_sep=self.config['rmin'], 
            max_sep=self.config['rmax'], 
            bin_slop=self.config['slop'], brute = False, 
            verbose=0, var_method = 'jackknife',
            bin_type=self.config['bin_type']
        )
        print('dvdg done',flush=True)
        DvRg = treecorr.NNCorrelation(
            nbins=self.config['ndots'], 
            min_sep=self.config['rmin'], 
            max_sep=self.config['rmax'], 
            bin_slop=self.config['slop'], brute = False, 
            verbose=0, var_method = 'jackknife',
            bin_type=self.config['bin_type']
        )
        print('dvrg done',flush=True)

        RvDg = treecorr.NNCorrelation(
            nbins=self.config['ndots'], 
            min_sep=self.config['rmin'], 
            max_sep=self.config['rmax'], 
            bin_slop=self.config['slop'], brute = False, 
            verbose=0, var_method = 'jackknife',
            bin_type=self.config['bin_type']
        )
        print('rvdg done',flush=True)

        RvRg = treecorr.NNCorrelation(
            nbins=self.config['ndots'], 
            min_sep=self.config['rmin'], 
            max_sep=self.config['rmax'], 
            bin_slop=self.config['slop'], brute = False, 
            verbose=0, var_method = 'jackknife',
            bin_type=self.config['bin_type']
        )
        print('rvrg done',flush=True)
        
        print('process init',flush=True)
        DvDg.process(self.dvcat, self.dgcat, num_threads=self.config['ncores'])
        print('process DvDg done',flush=True)
        DvRg.process(self.dvcat, self.rgcat, num_threads=self.config['ncores'])
        print('process DvRg done',flush=True)
        RvDg.process(self.rvcat, self.dgcat, num_threads=self.config['ncores'])
        print('process RvDg done',flush=True)
        RvRg.process(self.rvcat, self.rgcat, num_threads=self.config['ncores'])
        print('process RvRg done',flush=True)

        print('calculating xi',flush=True)
        self.r = DvDg.meanr
        self.xi, self.varxi = DvDg.calculateXi(dr=DvRg, rd=RvDg, rr=RvRg)
        self.cov = DvDg.cov
    
    def execute(self, cats):
        self.load_treecorrcatalogs(cats)
        self.calculate_corr()

    def write(self, output_file, lens_args, source_args):
        print('saving init',flush=True)
        if lens_args['rho2_max']<=0:
            tipo = 'R'
        elif lens_args['rho2_min']>=0:
            tipo = 'S'
        else:
            tipo = 'all'

        head = fits.Header()
        head.append(('nvoids',int(self.dvcat.nobj)))
        head.append(('lens',lens_args['name']))
        head.append(('sour',source_args['name'].split('_')[-1][:5],'cosmohub stamp'))
        head.append(('Rv_min',np.round(lens_args['Rv_min'],2)))
        head.append(('Rv_max',np.round(lens_args['Rv_max'],2)))
        # head.append(('Rv_mean',np.round(rvmean,4)))
        head.append(('r1_min',np.round(lens_args['rho1_min'],2)))
        head.append(('r1_max',np.round(lens_args['rho1_max'],2)))
        head.append(('r2_min',np.round(lens_args['rho2_min'],2)))
        head.append(('r2_max',np.round(lens_args['rho2_max'],2)))
        # head.append(('r2_mean',np.round(rho2mean,4)))
        head.append(('z_min',np.round(lens_args['z_min'],2)))
        head.append(('z_max',np.round(lens_args['z_max'],2)))
        # head.append(('z_mean',np.round(zmean,4)))
        # head.append(('SLCS_INFO'))
        head.append(('RMIN',np.round(self.config['rmin'],4)))
        head.append(('RMAX',np.round(self.config['rmax'],4)))
        head.append(('ndots',np.round(self.config['ndots'],4)))
        # head.append(('nk',np.round(args.nk,4),'jackknife slices'))
        head['HISTORY'] = f'{time.asctime()}'

        table_p = [
            fits.Column(name='r', format='E', array=self.r),
            fits.Column(name='Xi', format='E', array=self.xi),
        ]

        table_c = [
            fits.Column(name='cov', format='E', array=self.cov.flatten()),
        ]

        tbhdu_p = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))
        tbhdu_c = fits.BinTableHDU.from_columns(fits.ColDefs(table_c))
        
        primary_hdu = fits.PrimaryHDU(header=head)
        
        hdul = fits.HDUList([primary_hdu, tbhdu_p, tbhdu_c])

        hdul.writeto(output_file,overwrite=True)
        print('saved in', output_file,flush=True)

def main(tree_config, lens_args, source_args, sample):
    
    if lens_args['delta_max']<=0:
        voidtype = 'R'
    elif lens_args['delta_min']>=0:
        voidtype = 'S'
    else:
        voidtype = 'all'


    output_file = (f'results/vgcf_{sample}-{gravity}_L{lens_args["name"].split("_")[-1][:-4]}_'
                   f'Rv{lens_args["Rv_min"]:02.0f}-{lens_args["Rv_max"]:02.0f}_'
                   f'z{100*lens_args["z_min"]:03.0f}-{100*lens_args["z_max"]:03.0f}_'
                   f'type{voidtype}_bin{tree_config["bin_type"]}.fits')

    # === program arguments
    print(f' {" Settings ":=^60}')
    print(' Lens cat '+f'{": ":.>10}{lens_args["name"]}')
    print(' Source cat '+f'{": ":.>8}{source_args["name"]}')
    print(' Output file '+f'{": ":.>7}{output_file}')
    print(' NCORES '+f'{": ":.>12}{tree_config["ncores"]}\n')

    # === profile arguments
    # print(f' {" Profile arguments ":=^60}')
    print(' RMIN '+f'{": ":.>14}{tree_config["rmin"]:.2f}')
    print(' RMAX '+f'{": ":.>14}{tree_config["rmax"]:.2f}')
    print(' N '+f'{": ":.>17}{tree_config["ndots"]:<2d}')
    print(' NK '+f'{": ":.>16}{tree_config["NPatches"]:<2d}')
    print(' Binning '+f'{": ":.>11}{tree_config["bin_type"]}')
    
    # === lens arguments
    print(f' {" Void sample ":=^60}')
    print(' Radii '+f'{": ":.>13}[{lens_args["Rv_min"]:.2f}, {lens_args["Rv_max"]:.2f}) Mpc/h')
    print(' Redshift '+f'{": ":.>10}[{lens_args["z_min"]:.2f}, {lens_args["z_max"]:.2f})')
    print(' Type '+f'{": ":.>14}[{lens_args["delta_min"]},{lens_args["delta_max"]}) => {voidtype}')

    # === executing...
    cats = Catalogs(lens_args, source_args)
    vgcf = VoidGalaxyCrossCorrelation(tree_config)
    vgcf.execute(cats)

    vgcf.write(output_file, lens_args, source_args)

if __name__ == '__main__':    
    print('''
                             __ 
                            / _|
     __   __   __ _    ___  | |_ 
     \ \ / /  / _` |  / __| |  _|
      \ V /  | (_| | | (__  | |  
       \_/    \__, |  \___| |_|  
               __/ |             
              |___/              
    ''',
    flush=True)

    parser = ArgumentParser()
    # parser.add_argument('--lens_cat', type=str, default='voids_LCDM_09.dat', action='store')
    # parser.add_argument('--source_cat', type=str, default='l768_gr_z04-07_for02-03_19304.fits', action='store')
    parser.add_argument('--sample', type=str, default='TEST', action='store')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fROnly', action='store_true')
    group.add_argument('--GROnly', action='store_true')
    parser.add_argument('-c','--ncores', type=int, default=2, action='store')
    # parser.add_argument('-r','--n_runslices', type=int, default=1, action='store')
    # parser.add_argument('--h_cosmo', type=float, default=1.0, action='store')
    # parser.add_argument('--Om0', type=float, default=0.3089, action='store')
    # parser.add_argument('--Ode0', type=float, default=0.6911, action='store')
    parser.add_argument('--Rv_min', type=float, default=1.0, action='store')
    parser.add_argument('--Rv_max', type=float, default=50.0, action='store')
    parser.add_argument('--z_min', type=float, default=0.0, action='store')
    parser.add_argument('--z_max', type=float, default=0.6, action='store')
    parser.add_argument('--delta_min', type=float, default=-1.0, action='store')
    parser.add_argument('--delta_max', type=float, default=100.0, action='store')
    # parser.add_argument('--octant', action='store_true') ## 'store_true' guarda True SOLO cuando se da --octant
    parser.add_argument('--RIN', type=float, default=0.05, action='store')
    parser.add_argument('--ROUT', type=float, default=5.0, action='store')    
    parser.add_argument('-N','--ndots', type=int, default=22, action='store')    
    # parser.add_argument('-K','--nk', type=int, default=100, action='store')    
    # parser.add_argument('--addnoise', action='store_true')
    args = parser.parse_args()

    lens_args = {
        'Rv_min':args.Rv_min,
        'Rv_max':args.Rv_max,
        'z_min':args.z_min,
        'z_max':args.z_max,
        'delta_min':args.delta_min,
        'delta_max':args.delta_max,
        'flag':2.0,
        'octant':False,
        'fullshape':False,
    }

    source_args = {}

    mean_rv = (lens_args['Rv_min']+lens_args['Rv_max'])*0.5 # Eh ... no es tan así

    tree_config = {
        'ndots' : args.ndots, # number of radial bins
        'rmin' : args.RIN*mean_rv, # minimum value for rp (r in case of the quadrupole)
        'rmax' : args.ROUT*mean_rv, # maximum value for rp (r in case of the quadrupole)
        # Related to JK patches
        'NPatches' : 100,
        # Other configuration parameters
        'ncores' : args.ncores, # Number of cores to run in parallel
        'slop' : 0., # Resolution for treecorr
        'box' : False, # Indicates if the data corresponds to a box, otherwise it will assume a lightcone
        'bin_type':'Log',
    } 

    simus = {
        'GR':{
            'lens':'voids_LCDM_09.dat',
            'source':'l768_gr_z02-04_for01-02_19532.fits',
            'rand_lens':None,
            'rand_source':None,
        },
        'fR':{
            'lens':'voids_fR_09.dat',
            'source':'l768_mg_z02-04_for01-02_19531.fits',
            'rand_lens':None,
            'rand_source':None,
        }
    }

    if args.GROnly:
        tin = time.time()
        print(' '+f' EXECUTING -GR- ONLY '.center(60, '$')+' \n')
        lens_args['name']=simus['GR']['lens']
        source_args['name']=simus['GR']['source']
        main(sample=args.sample, tree_config=tree_config, lens_args=lens_args, source_args=source_args)

    elif args.fROnly:
        tin = time.time()
        print(' '+f' EXECUTING -f(R)- ONLY '.center(60, '$')+' \n')
        lens_args['name']=simus['fR']['lens']
        source_args['name']=simus['fR']['source']
        main(sample=args.sample, tree_config=tree_config, lens_args=lens_args, source_args=source_args)

    else:
        tin = time.time()
        for gravity in ['GR','fR']:
            print(' '+f' EXECUTING -{gravity}- '.center(60, '$')+' \n')
            lens_args['name']=simus[gravity]['lens']
            source_args['name']=simus[gravity]['source']
            main(sample=args.sample, tree_config=tree_config, lens_args=lens_args, source_args=source_args)

    print(f'Took {(time.time()-tin)/60.0} min'.center(50,':'), flush=True)
    print('End!', flush=True)
