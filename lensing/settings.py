import toml
import re

# void types: bounds [delta_min, delta_max) of the integrated density contrast
VOIDTYPE_DELTA = {'mixed': (-1.0, 100.0), 'S': (0.0, 100.0), 'R': (-1.0, 0.0)}

# threshold of the void file name: '_09' before the extension
THRESHOLD = re.compile(r'_(\d{2})(?=\.[^./]*$)')

class Config:

    def __init__(self, configfile:str='lensing/config.toml', gravity:str='GR', threshold:str=None):

        cfg = toml.load(configfile)

        self.lensname = cfg['lenses'][gravity.lower()]['name']
        self.source_mode = cfg['run'].get('source_mode', 'catalog')
        if self.source_mode not in ('catalog', 'map'):
            raise ValueError(f"[run] source_mode must be 'catalog' or 'map', got {self.source_mode}")
        src = cfg['sources'][gravity.lower()]
        self.sourcename = src.get('name')
        self.pix2idx = src.get('pix2idx')
        self.cubename = src.get('cube')
        if self.source_mode == 'map':
            if self.cubename is None:
                raise ValueError(f"[sources.{gravity.lower()}] needs 'cube' in map mode")
            self.sourcename = self.cubename
        mapcfg = cfg['sources'].get('map', {})
        # sources of a void bin [z_low, z_high) are the cube slices from the first edge >= z_high + dz_buffer
        self.dz_buffer = mapcfg.get('dz_buffer', 0.1)
        # weight of each source in the map estimator. Only 'invsc' (Sigma_crit^-1) for now
        self.weighting = mapcfg.get('weighting', 'invsc')
        if self.weighting != 'invsc':
            raise ValueError(f"[sources.map] weighting must be 'invsc', got {self.weighting}")
        # HEALPix order to degrade the cube to when loading it (None: native order)
        self.map_order = mapcfg.get('order')
        # random voids for the subtraction: generated in each stack, `factor` times the voids
        rands = cfg.get('randoms', {})
        self.rand_factor = rands.get('factor', 10)
        self.rand_seed = rands.get('seed', 42)
        if self.rand_factor <= 0:
            raise ValueError(f"[randoms] factor must be positive, got {self.rand_factor}")
        if not isinstance(self.rand_seed, int) or self.rand_seed < 0:
            raise ValueError(f"[randoms] seed must be a non-negative integer, got {self.rand_seed}")

        self.swap_threshold(threshold)

        self.sample = cfg['run']['sample']
        self.ncores = cfg['run']['ncores']
        self.plot = cfg['run'].get('plot', False) # not implemented yet
        self.overwrite = cfg['run']['overwrite']

        self.RIN = cfg['profile']['rin'] #Mpc/h
        self.ROUT = cfg['profile']['rout'] #Mpc/h
        self.NBINS = cfg['profile']['nbins']
        self.NJK = cfg['profile']['njk']
        # seed of the k-means of the jackknife regions
        self.jk_seed = cfg['profile'].get('jk_seed', 0)
        if not isinstance(self.jk_seed, int) or self.jk_seed < 0:
            raise ValueError(f"[profile] jk_seed must be a non-negative integer, got {self.jk_seed}")
        self.NSIDE = cfg['profile']['nside']
        self.addnoise = cfg['profile']['addnoise']
        self.binning = cfg['profile']['binning']
        # density of sources [arcmin^-2] to degrade the galaxy catalog to (None: all). Not implemented yet
        self.nback = cfg['profile'].get('nback')

        self.zbins = self._edges_to_bins(cfg['lenses']['z_edges'], 'z_edges')
        self.rvbins = self._edges_to_bins(cfg['lenses']['rv_edges'], 'rv_edges')
        self.voidtype = cfg['lenses']['voidtype']
        for t in self.voidtype:
            if t not in VOIDTYPE_DELTA:
                raise ValueError(f"[lenses] voidtype must be in {list(VOIDTYPE_DELTA)}, got {t}")
        self.flag = cfg['lenses']['flag']

        self.scols = cfg['sources']['columns']
        self.lcols = cfg['lenses']['columns']

        self.h = cfg['cosmology']['h']
        self.Om0 = cfg['cosmology']['Om0']
        self.Ob0 = cfg['cosmology']['Ob0']
        # for a non-flat or w0wa cosmology; for now only checked to be flat
        self.Ode0 = cfg['cosmology'].get('Ode0')

    def swap_threshold(self, threshold=None):
        """Sets `voidthr`, the threshold of the void file name ('_09.dat'). If `threshold`
        is given, it replaces the one in the name of the file; if not, the file is used as is."""
        head, sep, name = self.lensname.rpartition('/')
        found = THRESHOLD.search(name)
        if threshold is None:
            self.voidthr = found.group(1) if found else 'NA'
        else:
            if not found:
                raise ValueError(f"Can't set the void threshold {threshold}: "
                                 f"no '_dd' threshold in the void file name {name}")
            name = THRESHOLD.sub('_' + threshold, name, count=1)
            self.lensname = head + sep + name
            self.voidthr = threshold
        print(f' >> Using void thr {self.voidthr}')

    def _edges_to_bins(self, edges, name):
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError(f'[LENSES] {name} must be a list with at least 2 values.')
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo >= hi:
                raise ValueError(f'[LENSES] {name} must be strictly increasing, got {lo} >= {hi}.')
        return list(zip(edges[:-1], edges[1:]))

    def set_ncores(self, new_ncores):
        self.ncores = new_ncores
