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
            #w=np.ones(cats.nvoids), # If not given, all ones
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
            #w=np.ones(cats.ngals), 
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
            w=np.ones(cats.ngals*10), 
            r=cats.random_sources['dcom_gal'], 
            patch_centers= self.dvcat.patch_centers,
            ra_units='deg', dec_units='deg'
        )
        print('rgcat done',flush=True)

    def calculate_corr(self):

        if self.config['estimator'] == 'P':
            print('calculating corr w Peebles estimator', flush=True)
            pairs_names = ['DvDg', 'RvRg']
        elif self.config['estimator'] == 'LS':
            print('calculating corr w Landy-Szalay estimator', flush=True)
            pairs_names = ['DvDg', 'DvRg', 'RvDg', 'RvRg']
        else:
            raise ValueError('Estimator should be "P" (Peebles) or "LS" (Landy-Szalay)')

        pairs = {}
        for name in pairs_names:
            pairs[name] = treecorr.NNCorrelation(
                nbins=self.config['ndots'],
                min_sep=self.config['rmin'],
                max_sep=self.config['rmax'],
                bin_slop=self.config['slop'],
                brute=False,
                verbose=0,
                var_method='jackknife',
                bin_type=self.config['bin_type']
            )

        for name, pair in pairs.items():
            cat1 = self.dvcat if name[:2]=='Dv' else self.rvcat
            cat2 = self.dgcat if name[-2:]=='Dg' else self.rgcat    
            pair.process(cat1, cat2, num_threads=self.config['ncores'])

        print('calculating xi',flush=True)
        self.r = pairs['DvDg'].meanr
        if self.config['estimator'] == 'P':
            self.xi, self.varxi = pairs['DvDg'].calculateXi(rr=pairs['RvRg'])
        else:
            self.xi, self.varxi = pairs['DvDg'].calculateXi(
                dr=pairs['DvRg'], 
                rd=pairs['RvDg'], 
                rr=pairs['RvRg']
            )
        self.cov = pairs['DvDg'].cov
    
    def calculate_corr_normdist(self, cats):

        dgcat = treecorr.Catalog(
            ra=cats.sources['ra_gal'], 
            dec=cats.sources['dec_gal'], 
            #w=np.ones(cats.ngals), 
            r=cats.sources['dcom_gal'], 
            npatch=self.config['NPatches'],
            ra_units='deg', dec_units='deg'
        )

        rgcat = treecorr.Catalog(
            ra=cats.random_sources['ra_gal'], 
            dec=cats.random_sources['dec_gal'], 
            #w=np.ones(cats.ngals*10), 
            r=cats.random_sources['dcom_gal'], 
            patch_centers= dgcat.patch_centers,
            ra_units='deg', dec_units='deg'
        )

        rvcat = treecorr.Catalog(
            ra=cats.random_lenses['ra'], 
            dec=cats.random_lenses['dec'], 
            #w=np.ones(cats.nvoids*10), 
            r=cats.random_lenses['dcom'], 
            patch_centers= dgcat.patch_centers,
            ra_units='deg', dec_units='deg'
        )
        
        print('calculating corr w Peebles estimator', flush=True)
        pairs_names = ['DvDg', 'RvRg']

        r = np.linspace(self.config['rmin'], self.config['rmax'], self.config['ndots']+1)
        r = 0.5*(r[:-1]+r[1:])
        xi = np.zeros(self.config['ndots'])
        xi_jk = np.zeros((self.config['NPatches'], self.config['ndots']))

        for void in cats.lenses:
            dvcat = treecorr.Catalog(
                ra=np.array([void['ra']]),
                dec=np.array([void['dec']]),
                #w=np.ones(cats.nvoids), # If not given, all ones
                r=np.array([void['dcom']]),
                patch_centers=dgcat.patch_centers,
                ra_units='deg',
                dec_units='deg',
            )
            pairs = {}
            jk = {}
            for name in pairs_names:
                pairs[name] = treecorr.NNCorrelation(
                    nbins=self.config['ndots'],
                    min_sep=self.config['rmin']*void['Rv'],
                    max_sep=self.config['rmax']*void['Rv'],
                    bin_slop=self.config['slop'],
                    brute=False,
                    verbose=0,
                    var_method='jackknife',
                    bin_type=self.config['bin_type']
                )

            for name, pair in pairs.items():
                if name[0] == 'D':
                    cat1, cat2 = dvcat, dgcat
                else:
                    cat1, cat2 = rvcat, rgcat
                pair.process(cat1, cat2, num_threads=self.config['ncores'])
                jk[name], _ = treecorr.build_multi_cov_design_matrix(
                    [pair],
                    'jackknife',
                    func=lambda c: c[0].weight
                )

            xi += (pairs['DvDg'].weight/pairs['RvRg'].weight)*(pairs['RvRg'].tot/pairs['DvDg'].tot) - 1.0
            
            for i in range(self.config['NPatches']):
                nv_patch = np.sum(dvcat.w[(dvcat!=i)])
                nrv_patch = np.sum(rvcat.w[(rvcat!=i)])
                ng_patch = np.sum(dgcat.w[(dgcat!=i)])
                nrg_patch = np.sum(rgcat.w[(rgcat!=i)])

                ddpairs = nv_patch*ng_patch
                rrpairs = nrv_patch*nrg_patch

                xi_jk[i] += (rrpairs/ddpairs) * (jk['DvDg'][i]/jk['RvRg'][i]) - 1.0
        
        xi /= len(cats.lenses)
        xi_jk /= len(cats.lenses)
        cov = ((self.config['NPatches']-1)/self.config['NPatches'])*np.sum(
            np.einsum(
                'ij,ik->ijk',
                xi_jk-np.mean(xi_jk, axis=0), xi_jk-np.mean(xi_jk, axis=0)
            ),
            axis = 0
        )
        return r, xi, xi_jk, cov

    def execute(self, cats):
        # self.load_treecorrcatalogs(cats)
        # self.calculate_corr()

        # self.r, self.xi = self.calculate_corr_normdist(cats)
        # self.cov = np.full((len(self.r),len(self.r)), np.nan, dtype=np.float32)
        pass

def vgcf_single_simulation(gravity, tree_config, lens_args, source_args, sample):
    
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
    print(' Estimator '+f'{": ":.>9}{tree_config["estimator"]}')
    
    # === lens arguments
    print(f' {" Void sample ":=^60}')
    print(' Radii '+f'{": ":.>13}[{lens_args["Rv_min"]:.2f}, {lens_args["Rv_max"]:.2f}) Mpc/h')
    print(' Redshift '+f'{": ":.>10}[{lens_args["z_min"]:.2f}, {lens_args["z_max"]:.2f})')
    print(' Type '+f'{": ":.>14}[{lens_args["delta_min"]},{lens_args["delta_max"]}) => {voidtype}')

    # === executing...
    cats = Catalogs(lens_args, source_args)
    vgcf = VoidGalaxyCrossCorrelation(tree_config)
    r, xi, xi_jk, cov = vgcf.calculate_corr_normdist(cats)
    
    print('saving init',flush=True)

    head = fits.Header()
    head['nvoids'] = len(cats.lenses)
    head['lenscat'] = lens_args['name']
    head['sourcat'] = source_args['name']
    head['Rv_min'] = f'{cats.lenses["Rv"].min():2.2f}'
    head['Rv_max'] = f'{cats.lenses["Rv"].max():2.2f}'
    head['z_min'] = f'{cats.lenses["redshift"].min():1.3f}'
    head['z_max'] = f'{cats.lenses["redshift"].max():1.3f}'
    head['voidtype'] = voidtype
    head['delta_min'] = lens_args["delta_min"]
    head['delta_max'] = lens_args["delta_max"]
    head['RIN'] = tree_config['rmin']
    head['ROUT'] = tree_config['rmax']
    head['ndots'] = tree_config['ndots']
    head['HISTORY'] = f'{time.asctime()}'

    table_p = [fits.Column(name='r', format='E', array=r),
                fits.Column(name='Xi', format='E', array=xi)]
    xi_jk_hdu = [fits.ImageHDU(xi_jk, name='xi_jackknife')]
    cov_hdu = [fits.ImageHDU(cov, name='cov')]

    primary_hdu = fits.PrimaryHDU(header=head)
    tbhdu_p = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))
    
    hdul = fits.HDUList([primary_hdu, tbhdu_p]+cov_hdu+xi_jk_hdu)
    hdul.writeto(output_file, overwrite=True)
    print('saved in', output_file, flush=True)

def main():

    parser = ArgumentParser()
    parser.add_argument('--sample', type=str, default='TEST', action='store')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fROnly', action='store_true')
    group.add_argument('--GROnly', action='store_true')
    parser.add_argument('-c','--ncores', type=int, default=2, action='store')
    parser.add_argument('--Rv_min', type=float, default=1.0, action='store')
    parser.add_argument('--Rv_max', type=float, default=50.0, action='store')
    parser.add_argument('--z_min', type=float, default=0.0, action='store')
    parser.add_argument('--z_max', type=float, default=0.6, action='store')
    parser.add_argument('--delta_min', type=float, default=-1.0, action='store')
    parser.add_argument('--delta_max', type=float, default=100.0, action='store')
    parser.add_argument('--RIN', type=float, default=0.01, action='store')
    parser.add_argument('--ROUT', type=float, default=2.0, action='store')    
    parser.add_argument('-N','--ndots', type=int, default=22, action='store')    
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

    tree_config = {
        'ndots' : args.ndots, # number of radial bins
        'rmin' : args.RIN, #*mean_rv, # minimum value for rp (r in case of the quadrupole)
        'rmax' : args.ROUT, #*mean_rv, # maximum value for rp (r in case of the quadrupole)
        # Related to JK patches
        'NPatches' : 100,
        # Other configuration parameters
        'ncores' : args.ncores, # Number of cores to run in parallel
        'slop' : 0., # Resolution for treecorr
        'box' : False, # Indicates if the data corresponds to a box, otherwise it will assume a lightcone
        'bin_type':'Linear',
        'estimator':'P' # or 'LS'
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
        print(' '+f' EXECUTING -GR- ONLY '.center(60, '$')+' \n')
        lens_args['name']=simus['GR']['lens']
        source_args['name']=simus['GR']['source']
        vgcf_single_simulation(
            gravity='GR',
            sample=args.sample, 
            tree_config=tree_config, 
            lens_args=lens_args, 
            source_args=source_args
        )

    elif args.fROnly:
        print(' '+f' EXECUTING -f(R)- ONLY '.center(60, '$')+' \n')
        lens_args['name']=simus['fR']['lens']
        source_args['name']=simus['fR']['source']
        vgcf_single_simulation(
            gravity='fR',
            sample=args.sample, 
            tree_config=tree_config, 
            lens_args=lens_args, 
            source_args=source_args
        )

    else:
        for gravity in ['GR','fR']:
            print(' '+f' EXECUTING -{gravity}- '.center(60, '$')+' \n')
            lens_args['name']=simus[gravity]['lens']
            source_args['name']=simus[gravity]['source']
            vgcf_single_simulation(
                gravity=gravity,
                sample=args.sample, 
                tree_config=tree_config, 
                lens_args=lens_args, 
                source_args=source_args
            )

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

    tin = time.time()
    main()
    print('End!', flush=True)
    print(f'Took {(time.time()-tin)/60.0} min'.center(50,':'), flush=True)
