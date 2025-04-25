from argparse import ArgumentParser
from astropy.cosmology import LambdaCDM
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import treecorr
from multiprocess import Pool

import sys
sys.path.append('/home/fcaporaso/modified_gravity/')
from lensing.funcs import lenscat_load

cosmo = LambdaCDM(H0=100.0, Om0=0.3089, Ode0=0.6911)

def ang2xyz(ra, dec, redshift,
            cosmo=cosmo):

    comdist = cosmo.comoving_distance(redshift).value #Mpc; Mpc/h si h=1
    x = comdist * np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    y = comdist * np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    z = comdist * np.sin(np.deg2rad(dec))

    return x,y,z

def d_com(z):
    global cosmo
    return cosmo.comoving_distance(z).value

def make_randoms(ra, dec, redshift,
                 size_random = 100):

    rng = np.random.default_rng()    
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
    return ra_rand, dec_rand, z_rand

class Catalogos:
    def __init__(self, cat_config, lens_name, source_name, do_rands=True):
        path = '/home/fcaporaso/cats/L768/'
        
        self.lenses, _, nvoids = lenscat_load(
                lens_cat=path+lens_name,
                Rv_min=cat_config['Rv_min'], 
                Rv_max=cat_config['Rv_max'], 
                z_min=cat_config['z_min'], 
                z_max=cat_config['z_max'], 
                rho1_min=cat_config['rho1_min'], 
                rho1_max=cat_config['rho1_max'], 
                rho2_min=cat_config['rho2_min'], 
                rho2_max=cat_config['rho1_max'], 
                flag=cat_config['flag'],
                octant=cat_config['octant']
            )[0]
        
        assert len(nvoids) != 0, 'No void found with those parameters!'
        print('N voids: '.ljust(20,'.'), f' {len(nvoids):,}'.rjust(20,'.'),sep='',flush=True)
        
        self.sources = pd.read_parquet(path+source_name).sample(frac=1.0, random_state=1).to_numpy()
        ### [0]: kind, [1]:r_gal (com dist), [2]:redshift, [3]:ra, [4]:dec
        #query = f'redshift < {cat_config["z_max"]}+0.1 and redshift >= {cat_config["z_min"]}-0.1'
        mask = (self.sources[2] < cat_config["z_max"]+0.1) & (self.sources[2] >= cat_config["z_min"]-0.1)
        self.sources = self.sources[mask]
        
        assert len(self.sources) != 0, 'No tracer found with those parameters!'
        print('N sources: '.ljust(20,'.'), f' {len(self.sources):,}'.rjust(20,'.'),sep='',flush=True)

        self.lenses['w'] = np.ones(len(self.lenses))
        self.sources['w'] = np.ones(len(self.sources))

        self.lenses['r_com'] = d_com(self.lenses.redshift)

        if do_rands:
            self.random_lenses = make_randoms(
                self.lenses.ra,
                self.lenses.dec,
                self.lenses.redshift,
                size_random=len(self.lenses)*10
            )

            self.random_sources = make_randoms(
                self.sources.ra,
                self.sources.dec,
                self.sources.r_com, 
                size_random=len(self.sources)*10
            )
            self.random_lenses['w'] = np.ones(len(self.random_lenses))
            self.random_sources['w'] = np.ones(len(self.random_sources))
            self.random_lenses['r_com'] = d_com(self.random_lenses.redshift)
            self.random_sources.rename(columns={'redshift':'r_com'}, inplace=True)
            print('N rand voids: '.ljust(20,'.'), f' {len(self.random_lenses):,}'.rjust(20,'.'),sep='',flush=True)
            print('N rand sources: '.ljust(20,'.'), f' {len(self.random_sources):,}'.rjust(20,'.'),sep='',flush=True)
        
        else:
            self.random_lenses = pd.DataFrame({
                'ra':np.full(len(self.lenses),np.NaN),
                'dec':np.full(len(self.lenses),np.NaN),
                'redshift':np.full(len(self.lenses),np.NaN),
                'r_com':np.full(len(self.lenses),np.NaN),
                'w':np.full(len(self.lenses),np.NaN)
            })
            self.random_sources = pd.DataFrame({
                'ra':np.full(len(self.sources),np.NaN),
                'dec':np.full(len(self.sources),np.NaN),
                'redshift':np.full(len(self.sources),np.NaN),
                'r_com':np.full(len(self.sources),np.NaN),
                'w':np.full(len(self.sources),np.NaN)
            })
        
class VoidGalaxyCrossCorrelation:
    
    def __init__(self, config_treecorr):
        print('VGCC init',flush=True)
        self.config : dict = config_treecorr

        print(' Profile arguments '.center(40,"="),flush=True)
        print('RMIN: '.ljust(20,'.'), f' {config_treecorr["rmin"]}'.rjust(20,'.'), sep='',flush=True)
        print('RMAX: '.ljust(20,'.'), f' {config_treecorr["rmax"]}'.rjust(20,'.'),sep='',flush=True)
        print('N: '.ljust(20,'.'), f' {config_treecorr["ndots"]}'.rjust(20,'.'),sep='',flush=True)
        # print('N jackknife: '.ljust(20,'.'), f' {config_treecorr['nk']}'.rjust(20,'.'),sep='')
        # print('Shape Noise: '.ljust(20,'.'), f' {config_treecorr['addnoise}'.rjust(20,'.'),sep='')

    def load_treecorrcatalogs(self, lenses, sources, random_lenses, random_sources):
        print('loading cats w treecorr',flush=True)
        if len(lenses) <= self.config['NPatches']:
            print('NPatches < Nvoids..., changing to Nvoids-1',flush=True)
            self.config['NPatches'] = len(lenses)-1

        ## Voids
        self.dvcat = treecorr.Catalog(
            ra=lenses.ra,
            dec=lenses.dec,
            w=lenses.w,
            r=lenses.r_com,
            npatch=self.config['NPatches'],
            ra_units='deg',
            dec_units='deg',
        )
        print('dvcat done',flush=True)

        ## Tracers (gx)
        self.dgcat = treecorr.Catalog(
            ra=sources.ra, 
            dec=sources.dec, 
            w = sources.w, 
            r=sources.r_com, 
            patch_centers= self.dvcat.patch_centers,
            ra_units='deg', dec_units='deg'
        )
        print('dgcat done',flush=True)

        ## Random voids
        self.rvcat = treecorr.Catalog(
            ra=random_lenses.ra, 
            dec=random_lenses.dec, 
            w = random_lenses.w, 
            r=random_lenses.r_com, 
            patch_centers= self.dvcat.patch_centers,
            ra_units='deg', dec_units='deg'
        )
        print('rvcat done',flush=True)

        ## Random tracers (gx)
        self.rgcat = treecorr.Catalog(
            ra=random_sources.ra, 
            dec=random_sources.dec, 
            w = random_sources.w, 
            r=random_sources.r_com, 
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
    
    def run(self, cats):

        self.load_treecorrcatalogs(
            cats.lenses,
            cats.sources,
            cats.random_lenses,
            cats.random_sources
        )
        self.calculate_corr()

    def write(self, sample, cat_config, lenscat, sourcecat):
        print('saving init',flush=True)
        if cat_config['rho2_max']<=0:
            tipo = 'R'
        elif cat_config['rho2_min']>=0:
            tipo = 'S'
        else:
            tipo = 'all'

        head = fits.Header()
        head.append(('nvoids',int(self.dvcat.nobj)))
        head.append(('lens',lenscat))
        head.append(('sour',sourcecat.split('_')[-1][:5],'cosmohub stamp'))
        head.append(('Rv_min',np.round(cat_config['Rv_min'],2)))
        head.append(('Rv_max',np.round(cat_config['Rv_max'],2)))
        # head.append(('Rv_mean',np.round(rvmean,4)))
        head.append(('r1_min',np.round(cat_config['rho1_min'],2)))
        head.append(('r1_max',np.round(cat_config['rho1_max'],2)))
        head.append(('r2_min',np.round(cat_config['rho2_min'],2)))
        head.append(('r2_max',np.round(cat_config['rho2_max'],2)))
        # head.append(('r2_mean',np.round(rho2mean,4)))
        head.append(('z_min',np.round(cat_config['z_min'],2)))
        head.append(('z_max',np.round(cat_config['z_max'],2)))
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

        output_file = f'results/vgcf_{sample}_{np.ceil(cat_config["Rv_min"]).astype(int)}-{np.ceil(cat_config["Rv_max"]).astype(int)}_z0{int(10.0*cat_config["z_min"])}-0{int(10.0*cat_config["z_max"])}_type{tipo}.fits'

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

def main(tree_config, cat_config, lenscat, sourcecat, sample, ncores):
    
    vgcf = VoidGalaxyCrossCorrelation(tree_config)

    # program arguments
    print(' Catalogs config '.center(40,"="), flush=True)
    print('Lens cat: '.ljust(20,'.'), f' {lenscat}'.rjust(20,'.'), sep='', flush=True)
    print('Sour catalog: '.ljust(20,'.'), f' {sourcecat.split("_")[-1][:5]}'.rjust(20,'.'),sep='', flush=True)
    print('Out: '.ljust(20,'.'), f' {sample}'.rjust(20,'.'),sep='', flush=True)
    print('N cores: '.ljust(20,'.'), f' {ncores}'.rjust(20,'.'),sep='', flush=True)
    
    # lens arguments
    print(' Void sample '.center(40,"="), flush=True)
    print('Radii: '.ljust(20,'.'), f' [{cat_config["Rv_min"]}, {cat_config["Rv_max"]})'.rjust(20,'.'), sep='', flush=True)
    print('Redshift: '.ljust(20,'.'), f' [{cat_config["z_min"]}, {cat_config["z_max"]})'.rjust(20,'.'),sep='', flush=True)
    print('Tipo: '.ljust(20,'.'), f' {tipo}'.rjust(20,'.'),sep='', flush=True)

    cats = Catalogos(cat_config, lenscat, sourcecat)
    vgcf.run(cats)
    vgcf.write(sample+'_'+lenscat.split('_')[1], cat_config, lenscat, sourcecat)

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
    parser.add_argument('--sim', type=str, default='LCDM', action='store', choices=['LCDM','fR','both'])
    parser.add_argument('-c','--ncores', type=int, default=2, action='store')
    # parser.add_argument('-r','--n_runslices', type=int, default=1, action='store')
    # parser.add_argument('--h_cosmo', type=float, default=1.0, action='store')
    # parser.add_argument('--Om0', type=float, default=0.3089, action='store')
    # parser.add_argument('--Ode0', type=float, default=0.6911, action='store')
    parser.add_argument('--Rv_min', type=float, default=1.0, action='store')
    parser.add_argument('--Rv_max', type=float, default=50.0, action='store')
    parser.add_argument('--z_min', type=float, default=0.0, action='store')
    parser.add_argument('--z_max', type=float, default=0.6, action='store')
    parser.add_argument('--rho1_min', type=float, default=-1.0, action='store')
    parser.add_argument('--rho1_max', type=float, default=0.0, action='store')
    parser.add_argument('--rho2_min', type=float, default=-1.0, action='store')
    parser.add_argument('--rho2_max', type=float, default=100.0, action='store')
    parser.add_argument('--flag', type=float, default=2.0, action='store')
    # parser.add_argument('--octant', action='store_true') ## 'store_true' guarda True SOLO cuando se da --octant
    parser.add_argument('--RIN', type=float, default=0.05, action='store')
    parser.add_argument('--ROUT', type=float, default=5.0, action='store')    
    parser.add_argument('-N','--ndots', type=int, default=22, action='store')    
    # parser.add_argument('-K','--nk', type=int, default=100, action='store')    
    # parser.add_argument('--addnoise', action='store_true')
    args = parser.parse_args()

    ## TODO
    ## add octant as flag maybe?
    cat_config = {
        'Rv_min':args.Rv_min,
        'Rv_max':args.Rv_max,
        'z_min':args.z_min,
        'z_max':args.z_max,
        'rho1_min':args.rho1_min,
        'rho1_max':args.rho1_max,
        'rho2_min':args.rho2_min,
        'rho2_max':args.rho2_max,
        'flag':args.flag,
        'octant':True,
    }

    mean_rv = (cat_config['Rv_min']+cat_config['Rv_max'])*0.5
    if cat_config['rho2_max']<=0:
        tipo = 'R'
    elif cat_config['rho2_min']>=0:
        tipo = 'S'
    else:
        tipo = 'all'

    tree_config = {
        'ndots' : args.ndots, # number of radial bins
        'rmin' : args.RIN*mean_rv, # minimum value for rp (r in case of the quadrupole)
        'rmax' : args.ROUT*mean_rv, # maximum value for rp (r in case of the quadrupole)
        # Related to JK patches
        'NPatches' : int(args.ndots**(3./2.)),
        # Other configuration parameters
        'ncores' : args.ncores, # Number of cores to run in parallel
        'slop' : 0., # Resolution for treecorr
        'box' : False, # Indicates if the data corresponds to a box, otherwise it will assume a lightcone
        'bin_type':'Log',
    } 

    cats_name = [
        ('voids_LCDM_09.dat', 'l768_gr_octant_galz00-06_Mr-18_19860.parquet'),
        ('voids_fR_09.dat', 'l768_mg_octant_galz00-06_Mr-18_19861.parquet'),
    ]
    if args.sim == 'both':
    
        tin = time.time()
        for lenscat, sourcecat in cats_name:
            main(tree_config, cat_config, lenscat, sourcecat, args.sample, args.ncores)
    
    elif args.sim == 'LCDM':
        #lenscat = 'voids_LCDM_09.dat' 
        #sourcecat = 'l768_gr_galz00-06_Mr-18_19860.parquet'
        lenscat,sourcecat = cats_name[0]
        tin = time.time()
        main(tree_config, cat_config, lenscat, sourcecat, args.sample, args.ncores)

    else:
        #lenscat = 'voids_fR_09.dat'
        #sourcecat = 'l768_mg_galz00-06_Mr-18_19861.parquet'
        lenscat,sourcecat = cats_name[1]
        tin = time.time()
        main(tree_config, cat_config, lenscat, sourcecat, args.sample, args.ncores)

    print(f'Took {(time.time()-tin)/60.0} min'.center(50,':'), flush=True)
    print('End!', flush=True)
