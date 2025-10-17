from argparse import ArgumentParser
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import treecorr
from multiprocessing import Pool

import sys
sys.path.append('/home/fcaporaso/modified_gravity/')
from lensing.funcs import lenscat_load

def ang2xyz(ra, dec, redshift):

    comdist = cosmo.comoving_distance(redshift).value #Mpc; Mpc/h si h=1
    x = comdist * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    y = comdist * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    z = comdist * np.sin(np.deg2rad(dec))

    return x,y,z

def comoving_distance(z):
    return cosmo.comoving_distance(z).value

def make_randoms(ra, dec, redshift,
                 size_random = 100):

    rng = np.random.default_rng(0)    
    dec = np.deg2rad(dec)
    sindec_rand = rng.uniform(np.sin(dec.min()), np.sin(dec.max()), size_random)
    dec_rand = np.arcsin(sindec_rand)*(180.0/np.pi)
    ra_rand  = rng.uniform(ra.min(), ra.max(), size_random)

    y,xbins  = np.histogram(redshift, 25)
    x  = xbins[:-1]+0.5*np.diff(xbins)
    ## segun numpy mejor usar la clase numpy.polynomial.Polynomial instead of np.poly1d
    poly = np.polyfit(x,y,3)
    # print('polyfit done',flush=True)
    ## poly = np.polynomial.Polynomial.fit(x,y,deg=3)

    zr = rng.uniform(redshift.min(),redshift.max(),size_random)
    poly_y = np.poly1d(poly)(zr)
    # print('poly eval done',flush=True)
    ## poly_y = np.polynomial.polynomial.polyval(zr, poly.coef) ## no da lo mismo....
    poly_y[poly_y<0] = 0.
    peso = poly_y/sum(poly_y)
    z_rand = rng.choice(zr,size_random,replace=True,p=peso)

    # print('Wii randoms!',flush=True)
    #return pd.DataFrame({'ra': ra_rand, 'dec': dec_rand, 'redshift':z_rand})
    return np.array([ra_rand, dec_rand, z_rand])

class Catalogs:
    def __init__(self, lens_args, source_name, do_rands=True):
        
        ## [0]:rv, [1]:ra, [2]:dec, [3]:redshift
        L, _, self.nvoids = lenscat_load(**lens_args)
        assert self.nvoids != 0, 'No void found with those parameters!'
        self.lenses = Table(L.T, names=['Rv', 'ra', 'dec', 'redshift'])
        self.lenses['dcom'] = comoving_distance(self.lenses['redshift'])
        
        #self.sources = pd.read_parquet(source_name).sample(frac=1.0, random_state=1).to_numpy().T
        try:
            self.sources = Table.read(source_name, format='fits', memmap=True)
        except FileNotFoundError:
            self.sources = Table.read('/home/fcaporaso/cats/L768/'+source_name, format='fits', memmap=True)
        mask = (self.sources['true_redshift_gal'] < lens_args["z_max"]+0.01) & (self.sources['true_redshift_gal'] >= lens_args["z_min"]-0.01)
        self.sources = self.sources[mask]
        self.ngals = len(self.sources)
        assert self.ngals != 0, 'No tracer found with those parameters!'
        
        if 'dcom_gal' not in self.sources.columns:
            self.sources['dcom_gal'] = comoving_distance(self.sources['true_redshift_gal'])

        # self.lenses = np.append(self.lenses, [d_com(self.lenses[2])]) ## [4]
        # self.lenses = np.append(self.lenses, [np.ones(self.nvoids)]) ## [5]
        # self.sources = np.append(self.sources, [np.ones(self.ngals)]) ## [5]

        if do_rands:
            print(' Making randoms '.center(60, '.'), flush=True)
            self.random_lenses = Table(
                make_randoms(
                    self.lenses['ra'],
                    self.lenses['dec'],
                    self.lenses['dcom'],
                    size_random=self.nvoids*10
                ),
                names=['ra','dec','dcom']
            )

            self.random_sources = Table(
                make_randoms(
                    self.sources['ra_gal'],
                    self.sources['dec_gal'],
                    self.sources['dcom_gal'], ## le paso la dist comovil y samplea de esos como si fuera redshift, así me ahorro el paso de pasar de z a dist com 
                    size_random=self.ngals*10
                ),
                names=['ra_gal', 'dec_gal', 'dcom_gal']
            )

            ## w=1
            # self.random_lenses = np.append(self.random_lenses, [np.ones(self.nvoids*10)]) ## [3]
            # self.random_sources = np.append(self.random_sources, [np.ones(self.ngals*10)]) ## [3]
            
            # self.random_lenses = np.append(self.random_lenses, [d_com(self.random_lenses[2])]) ## [4]
            #self.random_sources.rename(columns={'redshift':'r_com'}, inplace=True)
        
        else: ## cambiar a numpy...
            self.random_lenses = pd.DataFrame({
                'ra':np.full(len(self.lenses),np.NaN),
                'dec':np.full(len(self.lenses),np.NaN),
                'redshift':np.full(len(self.lenses),np.NaN),
                'r_com':np.full(len(self.lenses),np.NaN),
                'w':np.full(len(self.lenses),np.NaN)
            })
            self.random_sources = pd.DataFrame({
                'ra':np.full(self.ngals,np.NaN),
                'dec':np.full(self.ngals,np.NaN),
                'redshift':np.full(self.ngals,np.NaN),
                'r_com':np.full(self.ngals,np.NaN),
                'w':np.full(self.ngals,np.NaN)
            })
        
        print('N voids: '.ljust(20,'.'), f' {self.nvoids:,}'.rjust(20,'.'),sep='',flush=True)
        print('N sources: '.ljust(20,'.'), f' {self.ngals:,}'.rjust(20,'.'), sep='', flush=True)
        print('N rand voids: '.ljust(20,'.'), f' {self.nvoids*10:,}'.rjust(20,'.'),sep='',flush=True)
        print('N rand sources: '.ljust(20,'.'), f' {self.ngals*10:,}'.rjust(20,'.'),sep='',flush=True)

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

    def write(self, output_file, lens_args, source_name):
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
        head.append(('sour',source_name.split('_')[-1][:5],'cosmohub stamp'))
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

    def plot_corr(self, label):

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_axes([1,1,1,1])
        ax.errorbar(self.r, self.xi, self.varxi, fmt='.-', capsize=2, c='purple', label=label)
        ax.set_xlabel('meanr [Mpc/h]')
        ax.set_ylabel('$\\xi$')
        return ax
        # plt.title('fR')
        #plt.text(40,-0.6, f'$R_v \\in$ ({Rv_min},{Rv_max})')
        #plt.text(40,-0.65, f'$z \\in$ ({z_min},{z_max})')

def main(tree_config, lens_args, source_name, sample):
    
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
    print(' Source cat '+f'{": ":.>8}{source_name}')
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
    cats = Catalogs(lens_args, source_name)
    vgcf = VoidGalaxyCrossCorrelation(tree_config)
    vgcf.execute(cats)

    vgcf.write(output_file, lens_args, source_name)

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
            'source':'l768_gr_z02-04_for01-02_19532.fits'
        },
        'fR':{
            'lens':'voids_fR_09.dat',
            'source':'l768_mg_z02-04_for01-02_19531.fits'
        }
    }

    if args.GROnly:
        tin = time.time()
        print(' '+f' EXECUTING -GR- ONLY '.center(60, '$')+' \n')
        lens_args['name']=simus['GR']['lens']
        main(sample=args.sample, tree_config=tree_config, lens_args=lens_args, source_name=simus['GR']['source'])
    elif args.fROnly:
        tin = time.time()
        print(' '+f' EXECUTING -f(R)- ONLY '.center(60, '$')+' \n')
        lens_args['name']=simus['fR']['lens']
        main(sample=args.sample, tree_config=tree_config, lens_args=lens_args, source_name=simus['fR']['source'])
    else:
        tin = time.time()
        for gravity in ['GR','fR']:
            print(' '+f' EXECUTING -{gravity}- '.center(60, '$')+' \n')
            lens_args['name']=simus[gravity]['lens']
            main(sample=args.sample, tree_config=tree_config, lens_args=lens_args, source_name=simus[gravity]['source'])

    print(f'Took {(time.time()-tin)/60.0} min'.center(50,':'), flush=True)
    print('End!', flush=True)
