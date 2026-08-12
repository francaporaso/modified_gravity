import numpy as np
import toml

# ==== Input globals
# read from config file
class Config:

    def __init__(self, configfile:str='lensing/config.toml', gravity:str='GR'):

        cfg = toml.load(configfile)

        self.lensname = cfg['lenses'][gravity.lower()]['name']
        self.sourcename = cfg['sources'][gravity.lower()]['name']
        self.randsname = cfg['randoms'][gravity.lower()]['name']

        self.sample = cfg['run']['sample']
        self.ncores = cfg['run']['ncores']
        self.plot = cfg['run']['plot']
        self.overwrite = cfg['run']['overwrite']
        
        self.RIN = cfg['profile']['rin'] #Mpc/h
        self.ROUT = cfg['profile']['rout'] #Mpc/h
        self.NBINS = cfg['profile']['nbins']
        self.NJK = cfg['profile']['njk']
        self.NSIDE = cfg['profile']['nside']
        self.addnoise = cfg['profile']['addnoise']
        self.binning = cfg['profile']['binning']
        #self.nback = cfg['profile']['nback']

        self.zbins = self._edges_to_bins(cfg['lenses']['z_edges'], 'z_edges')
        self.rvbins = self._edges_to_bins(cfg['lenses']['rv_edges'], 'rv_edges')
        #self.deltabins = self._edges_to_bins(cfg['lenses']['delta_edges'], 'delta_edges')
        self.voidtype = cfg['lenses']['voidtype']
        self.flag = cfg['lenses']['flag']
        self.fullshape = cfg['lenses']['fullshape']
        #self.is_MICE = cfg['lenses']['is_mice']
        #self.voidtype = cfg['lenses']['voidtype']

        self.scols = cfg['sources']['columns']
        self.lcols = cfg['lcols']['columns']

        self.h = cfg['cosmology']['h']
        self.Om0 = cfg['cosmology']['Om0']
        self.Ob0 = cfg['cosmology']['Ob0']

    def _edges_to_bins(self, edges, name):
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError(f'[LENSES] {name} must be a list with at least 2 values.')
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo >= hi:
                raise ValueError(f'[LENSES] {name} must be strictly increasing, got {lo} >= {hi}.')
        return list(zip(edges[:-1], edges[1:]))

    def set_ncores(self, new_ncores):
        self.NCORES = new_ncores

