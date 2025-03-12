import numpy as np
import treecorr
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.special import lpmn
import math
from scipy import stats
from scipy import spatial

config_setup = dict(col_names = ['ra','dec','r_com','ep1','ep2','w'], # name of the columns of the catalogue related to the coordinate position (ra, dec and comoving distance for the lightcone, x, y and z for a box), projected ellipticity components weights in this order.
                    nbins = 10, # number of radial bins
                    rmin = 0.1, # minimum value for rp (r in case of the quadrupole)
                    rmax = 10., # maximum value for rp (r in case of the quadrupole)
                    pi_max = 60., # maximum value along l.o.s. (Pi) 
                    npi = 5, # number of bins in Pi
                    mubins = 10, # number of bins in mu
                    NPatches = 16,
                    ncores = 30, # Number of cores to run in parallel
                    slop = 0., # Resolution for treecorr
                    box = False, # Indicate if the data corresponds to a box, otherwise it will assume a lightcone
                    grid_resolution = 10, # Controls grid r,mu resolution to compute the quadrupole. 
                    exact_position = True, # Indicates if the coordinates are exactly provided (e.g. simulated data without any error added in the position). Otherwise it will assume that the positions are not  exact and will use ra, dec for matching the catalogues. If this parameter is set as True, it will neglect box and it will set it as False
                    sky_threshold = 1.0 # Threshold for matching the catalogues in arcsecond, used if exact_position
#is set to True.
                    )

def norm_cov(cov):
    cov_norm = np.zeros_like(cov)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            cov_norm[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
    return cov_norm


class duplicate():
    """
        Finds weighted (w) number of duplicate objects in two catalogues. If exact positions (exact_positions = True) are provided, it uses x, y and z coordinates. Otherwise, it will match the objects in sky coordinates (ra,dec) within a given distance threshold in arcseconds given by threshold_arcsec. 

        Arguments:
        -----------
            cat1 (catalogue): Catalogue with object positions and weights.
            cat2 (catalogue): Catalogue with object positions and weights.
            exact_position (bool) : Indicates if the positions provided are exact (within a tolerance of 1.e-6).
            threshold_arcsec (float): threshold in arcseconds to match the catalogues using ra and dec.
        Attributes:
        -----------
            _w1 (ndarray,float): Projected correlation function across the pcor direction.
            _w2 (ndarray,float): Projected correlation function across the pcor direction for each Jackknife resampling.
            _exact_position (bool): Covariance matrix estimated using Jackknife resampling.
            _ind (ndarray,float): Index for cat1 to match cat2. 
            _mdist (ndarray,bool): Mask of the closest objects within the allowed threshold.
            _id_u (ndarray,bool): Mask of the unique neighbors.
            
    """

    def __init__(self,cat1,cat2,exact_position,threshold_arcsec=1.):

        self._exact_position = exact_position
        self._w1 = cat1.w
        self._w2 = cat2.w

        if exact_position:

            tree = spatial.cKDTree(np.array([cat1.x,cat1.y,cat1.z]).T)
            dist,ind=tree.query(np.array([cat2.x,cat2.y,cat2.z]).T)        
            self._ind,self._id_u = np.unique(ind,return_index=True)
            self._mdist = dist < 1.e-6
            
        else:

            c1 = SkyCoord(ra=np.array(cat1.ra)*u.degree, dec=np.array(cat1.dec)*u.degree)
            c2 = SkyCoord(ra=np.array(cat2.ra)*u.degree, dec=np.array(cat2.dec)*u.degree)    
            ind, sep2d, dist = c2.match_to_catalog_sky(c1)
            self._ind,self._id_u = np.unique(ind,return_index=True)
            self._mdist = np.array(sep2d.to(u.arcsec)) < threshold_arcsec

    def Nrep(self,mask1,mask2):
        
        Nw1 = self._w1[self._ind]*(mask1[self._ind]).astype(float)
        Nw2 = self._w2[self._id_u]*(mask2[self._id_u]).astype(float)
        threshold = (self._mdist[self._id_u]).astype(float)
        
        return np.sum(Nw1*Nw2*threshold)


class project_corr():
    
    """
        Computes the projected correlations from 2D correlations.

        Arguments:
        -----------
            xi_s (ndarray): Array of 2D correlation.
            xi_s_jk (ndarray): Array of 2D correlation for each Jackknife resampling.
            rcor (ndarray): Array of projected radial separation bins.
            pcor (ndarray): Array of separation bins in the direction that is going to be projected.
            factor (float): Factor to be included in the integration.
        Attributes:
        -----------
            xip (ndarray): Projected correlation function across the pcor direction.
            xip_jk (ndarray): Projected correlation function across the pcor direction for each Jackknife resampling.
            cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
            std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance. 
    """
    
    def __init__(self,xi_s,xi_s_jk,rcor,pcor,factor = 1):
        NPatches = len(xi_s_jk)
        self.xip = factor*np.trapz(xi_s,pcor,axis=0)
        self.xip_jk = factor*np.trapz(xi_s_jk,pcor,axis=1)
        xip_mean = np.mean(self.xip_jk, axis = 0)
        xi_diff = self.xip_jk - xip_mean
        self.cov_jk = ((NPatches-1)/NPatches)*np.sum(np.einsum('ij,ik->ijk',xi_diff,xi_diff), axis = 0)
        self.std_from_cov = np.sqrt(np.diagonal(self.cov_jk))   

class compute_wgg(project_corr):

    """
        Computes the galaxy-galaxy correlation.
        
        Arguments:
        -----------
            config (dict): Configuration dictionary for the computation.
            dcat (treecorr.Catalog): Catalog of data points.
            rcat (treecorr.Catalog): Catalog of random points.        
        Attributes:
        -----------
            rp (ndarray): Array of projected radial separation bins.
            mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
            mean_rp (ndarray): Mean projected radial separation bins.
            Pi (ndarray): Array of l.o.s. separation bins.
            xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
            xi_jk (ndarray): 2D Correlation function for each Jackknife resampling.
            xip (ndarray): Projected correlation function across the l.o.s.
            xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
            cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
            cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
            std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """
    
    
    def __init__(self,dcat,rcat,config):

        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros((config['npi'], config['nbins']))
        xi_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
        
        dd_jk = np.zeros_like(xi_jk)
        dr_jk = np.zeros_like(xi_jk)
        rr_jk = np.zeros_like(xi_jk)

        
        # Pair normalization fractions
        Nd = dcat.sumw
        Nr = rcat.sumw
        NNpairs = (Nd*(Nd - 1))/2.
        RRpairs = (Nr*(Nr - 1))/2.
        NRpairs = (rcat.sumw*dcat.sumw)
    
        f0 = RRpairs/NNpairs
        f1 = RRpairs/NRpairs

        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])

        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):

            dd = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            dr = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            rr = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            if config['box']:
                factor = 0.5
                dd.process(dcat,dcat, metric='Rperp', num_threads = config['ncores'])
                rr.process(rcat,rcat, metric='Rperp', num_threads = config['ncores'])
            else:
                factor = 1.0
                dd.process(dcat, metric='Rperp', num_threads = config['ncores'])
                rr.process(rcat, metric='Rperp', num_threads = config['ncores'])

            dr.process(dcat,rcat, metric='Rperp', num_threads = config['ncores'])

            r[:] = np.copy(dd.rnom)
            mean_r[:] = np.copy(dd.meanr)
            mean_logr[:] = np.copy(dd.meanlogr)

            xi[p, :] = (dd.weight*factor*f0 - (2.*dr.weight)*f1 + rr.weight*factor) / (rr.weight*factor)


            #Here I compute the variance
            func = lambda corrs: corrs[0].weight*factor
            func2 = lambda corrs: corrs[0].weight
            dd_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([dd], 'jackknife', func = func)
            dr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([dr], 'jackknife', func = func2)
            rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func)

            dd.finalize()
            dr.finalize()
            rr.finalize()

        for i in range(config['NPatches']):
    
            swd = np.sum(dcat.w[~(dcat.patch == i)])
            swr = np.sum(rcat.w[~(rcat.patch == i)])
    
            NNpairs_JK = (swd*(swd - 1))/2.
            RRpairs_JK = (swr*(swr - 1))/2.
            NRpairs_JK = (swd*swr)
            
            f0_jk = RRpairs_JK/NNpairs_JK
            f1_jk = RRpairs_JK/NRpairs_JK
    
            xi_jk[i, :, :] = (dd_jk[i, :, :]*f0_jk*factor - (2.*dr_jk[i, :, :])*f1_jk + rr_jk[i, :, :]*factor) / (rr_jk[i, :, :]*factor)
    
        xi[np.isinf(xi)] = 0. #It sets to 0 the values of xi_gp that are infinite
        xi[np.isnan(xi)] = 0. #It sets to 0 the values of xi_gp that are null
    
        xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5

        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.Pi = xPi
        self.xi = xi
        self.xi_jk = xi_jk

        project_corr.__init__(self,self.xi,self.xi_jk,self.rp,self.Pi)
        self.cov_jk_norm = norm_cov(self.cov_jk)


class compute_wgp(project_corr):
    
    """
    Computes the galaxy-shear correlation.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
    Attributes:
    -----------
        rp (ndarray): Array of projected radial separation bins.
        mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
        mean_rp (ndarray): Mean projected radial separation bins.
        Pi (ndarray): Array of l.o.s. separation bins.
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
        xi_jk (ndarray): 2D Correlation function for each Jackknife resampling.
        xip (ndarray): Projected correlation function across the l.o.s.
        xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """
    
    def __init__(self,pcat,scat,rpcat,rscat,config,dup,dup_random):
        self.sd = []
        self.rr = []
        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros((config['npi'], config['nbins']))
        xi_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
        xi_x = np.zeros((config['npi'], config['nbins']))
        xi_x_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
      
        sr_jk = np.zeros_like(xi_jk)
        sd_jk = np.zeros_like(xi_jk)
        rr_jk = np.zeros_like(xi_jk)
        sr_x_jk = np.zeros_like(xi_jk)
        sd_x_jk = np.zeros_like(xi_jk)

        # get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation   
        Nrep = dup.Nrep(np.ones(len(pcat.w)).astype(bool),np.ones(len(scat.w)).astype(bool))
        Nrep_rand = dup_random.Nrep(np.ones(len(rpcat.w)).astype(bool),np.ones(len(rscat.w)).astype(bool))

        
        NGtot = pcat.sumw*scat.sumw - Nrep
        RRtot = rpcat.sumw*rscat.sumw - Nrep_rand
        RGtot = rpcat.sumw*scat.sumw
        f0 = RRtot / NGtot
        f1 = RRtot / RGtot

        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])

        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            rr = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            sd = treecorr.NGCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            sr = treecorr.NGCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
                
            sd.process(pcat, scat, metric='Rperp', num_threads = config['ncores'])
            sr.process(rpcat, scat, metric='Rperp', num_threads = config['ncores'])
            rr.process(rpcat, rscat, metric='Rperp', num_threads = config['ncores'])

            r[:] = np.copy(rr.rnom)
            mean_r[:] = np.copy(rr.meanr)
            mean_logr[:] = np.copy(rr.meanlogr)
    
            xi[p, :] = (f0 * (sd.xi * sd.weight) - f1 * (sr.xi * sr.weight) ) / rr.weight
            xi_x[p, :] = (f0 * (sd.xi_im * sd.weight) - f1 * (sr.xi_im * sr.weight) ) / rr.weight
        
            #Here I compute the variance
            func_sd = lambda corrs: corrs[0].xi * corrs[0].weight
            func_sd_x = lambda corrs: corrs[0].xi_im * corrs[0].weight
            func_rr = lambda corrs: corrs[0].weight
            sd_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd)
            sr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = func_sd)
            rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func_rr)
            sd_x_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd_x)
            sr_x_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = func_sd_x)

            self.sd += [sd]
            self.rr += [rr]
        
            #sd.finalize()
            #sr.finalize()
            #rr.finalize()

        for i in range(config['NPatches']):
            swd  = np.sum(pcat.w[(pcat.patch != i)])
            sws  = np.sum(scat.w[(scat.patch != i)])
            swr = np.sum(rpcat.w[(rpcat.patch != i)])
            swrs = np.sum(rscat.w[(rscat.patch != i)])

            
            
            NGtot_JK = swd*sws - dup.Nrep(pcat.patch != i,scat.patch != i)
            RRtot_JK = swr*swrs - dup_random.Nrep(rpcat.patch != i,rscat.patch != i)
            RGtot_JK = swr*sws           
            f0_JK = RRtot_JK / NGtot_JK
            f1_JK = RRtot_JK / RGtot_JK
            xi_jk[i, :, :] = (f0_JK * sd_jk[i, :, :] - f1_JK * sr_jk[i, :, :]) / rr_jk[i, :, :]
            xi_x_jk[i, :, :] = (f0_JK * sd_x_jk[i, :, :] - f1_JK * sr_x_jk[i, :, :]) / rr_jk[i, :, :]
    
    
        xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5
        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.Pi = xPi
        self.xi = xi
        self.xi_jk = xi_jk
        self._xi_x = xi_x
        self._xi_x_jk = xi_x_jk

        project_corr.__init__(self,self.xi,self.xi_jk,self.rp,self.Pi)
        self.cov_jk_norm = norm_cov(self.cov_jk)

class compute_delta_sigma():
    
    """
    Computes the galaxy-shear correlation.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
    Attributes:
    -----------
        rp (ndarray): Array of projected radial separation bins.
        mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
        mean_rp (ndarray): Mean projected radial separation bins.
        Pi (ndarray): Array of l.o.s. separation bins.
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
        xi_jk (ndarray): 2D Correlation function for each Jackknife resampling.
        xip (ndarray): Projected correlation function across the l.o.s.
        xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """
    
    def __init__(self,pcat,scat,config):

        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros(config['nbins'])
        xi_jk = np.zeros((config['NPatches'], config['nbins']))
        xi_x = np.zeros(config['nbins'])
        xi_x_jk = np.zeros((config['NPatches'], config['nbins']))
      

        sd_jk = np.zeros_like(xi_jk)


        # get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation   

        # now loop over Pi bins, and compute w(r_p | Pi)
            
        sd = treecorr.NGCorrelation(nbins=config['nbins'], 
                                    min_sep=config['rmin'], 
                                    max_sep=config['rmax'], 
                                    bin_slop=config['slop'], brute = False, 
                                    verbose=0, var_method = 'jackknife')
            
        sd.process(pcat, scat, metric='Rperp', num_threads = config['ncores'])

        r[:] = np.copy(sd.rnom)
        mean_r[:] = np.copy(sd.meanr)
        mean_logr[:] = np.copy(sd.meanlogr)

        
        #Here I compute the variance
        func_sd = lambda corrs: corrs[0].xi
        func_sd_x = lambda corrs: corrs[0].xi_im
        sd_jk, weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd)
        sd_x_jk, weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd_x)


        for i in range(config['NPatches']):
            xi_jk[i, :] = sd_jk[i, :]
            xi_x_jk[i, :] = sd_x_jk[i, :]
    
    
        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.xi = sd.xi
        self.xi_jk = xi_jk
        self._xi_x = sd.xi_im
        self._xi_x_jk = xi_x_jk


class compute_fast_wgp(project_corr):
    
    """
    Allows to compute the projected galaxy-shear correlation (wg+). This version allows the pre-computation of the random number of pairs and then it can be executed to compute the galaxy-shear correlation varying the shapes of the galaxies in the shape catalogue.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
    Methods:
    -----------
        execute(new_shapes=False,shapes=[]): Allows the computation of the projected galaxy-shape correlation. It can be used giving a different shape catalogue as input.
    """
    
    def __init__(self,pcat,scat,rpcat,rscat,config,dup,dup_random):
        
        self.config = config
        self.pcat = pcat
        self.rpcat = rpcat
        self.rscat = rscat
        self.dup = dup
        self.dup_random = dup_random
        self.scat = scat
        self.rr_p = []
        
        # arrays to store the output
         
        self.rr_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])
        
        print('Computing normalisation factors and the number of pairs in the random catalogues...')
        # get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation   
        Nrep = self.dup.Nrep(np.ones(len(self.pcat.w)).astype(bool),np.ones(len(scat.w)).astype(bool))
        Nrep_rand = self.dup_random.Nrep(np.ones(len(self.rpcat.w)).astype(bool),np.ones(len(self.rscat.w)).astype(bool))
        
        NGtot = self.pcat.sumw*scat.sumw - Nrep
        RRtot = self.rpcat.sumw*self.rscat.sumw - Nrep_rand

        self.f0 = RRtot / NGtot
        f0_JK = np.zeros(config['NPatches'])
        
        for i in range(config['NPatches']):
            swd  = np.sum(pcat.w[(pcat.patch != i)])
            sws  = np.sum(scat.w[(scat.patch != i)])
            swr = np.sum(rpcat.w[(self.rpcat.patch != i)])
            swrs = np.sum(rscat.w[(self.rscat.patch != i)])

            NGtot_JK = swd*sws - dup.Nrep(self.pcat.patch != i,scat.patch != i)
            RRtot_JK = swr*swrs - dup_random.Nrep(self.rpcat.patch != i,self.rscat.patch != i)
            f0_JK[i] = RRtot_JK / NGtot_JK

        self.f0_JK = np.repeat(f0_JK,config['nbins']*config['npi']).reshape(config['NPatches'], config['npi'], config['nbins'])
        
        
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            rr = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            rr.process(rpcat, rscat, metric='Rperp', num_threads = config['ncores'])
            self.rr_p += [rr]

            func_rr = lambda corrs: corrs[0].weight
            self.rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func_rr)
        print('Everything ok :) Finished')
        
    def use_new_shapes(self,shapes):
        """
        Define new shape catalogue to compute the galaxy-shear correlation.

        Arguments
        ----------
        shapes : list
            Hosts the catalogue of the shapes (pandas array).
        Attributes:
        ----------
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        """
        
        config = self.config
        
        del(self.scat)
        if config['box']:
            x, y, z, g1, g2, w = config['col_names']
            zshift = 1.e6 #to put the observer far away                   
            scat  = treecorr.Catalog(g1=shapes[g1], 
                                     g2 = shapes[g2],
                                     x = shapes[x], 
                                     y = shapes[z]+zshift,  
                                     z = shapes[y], 
                                     w = shapes[w], 
                                     patch_centers = self.pcat.patch_centers)
        
        else:   
            ra, dec, r, g1, g2, w = config['col_names']
            scat  = treecorr.Catalog(g1 = shapes[g1],
                                          g2 = shapes[g2],
                                          ra=shapes[ra], 
                                          dec=shapes[dec], 
                                          r=shapes[r], 
                                          w = shapes[w], 
                                          patch_centers = self.pcat.patch_centers, 
                                          ra_units='deg', dec_units='deg')
            
        self.scat = scat
    
    def execute(self):
        """
        Compute the galaxy-shear correlation.

        Attributes:
        ----------
        rp (ndarray): Array of projected radial separation bins.
        mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
        mean_rp (ndarray): Mean projected radial separation bins.
        Pi (ndarray): Array of l.o.s. separation bins.
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
        xi_jk (ndarray): 2D Correlation function for each Jackknife resampling.
        xip (ndarray): Projected correlation function across the l.o.s.
        xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
        """
        
        config = self.config
        scat = self.scat

        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros((config['npi'], config['nbins']))
        xi_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
      
        sr_jk = np.zeros_like(xi_jk)
        sd_jk = np.zeros_like(xi_jk)


        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])

        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            rr = self.rr_p[p]
            
            sd = treecorr.NGCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            sr = treecorr.NGCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
                
            sd.process(self.pcat, scat, metric='Rperp', num_threads = config['ncores'])

            r[:] = np.copy(rr.rnom)
            mean_r[:] = np.copy(rr.meanr)
            mean_logr[:] = np.copy(rr.meanlogr)
    
            xi[p, :] = (self.f0 * (sd.xi * sd.weight) ) / rr.weight
        
            #Here I compute the variance
            func_sd = lambda corrs: corrs[0].xi * corrs[0].weight
            func_sd_x = lambda corrs: corrs[0].xi_im * corrs[0].weight
            func_rr = lambda corrs: corrs[0].weight
            sd_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd)
        
            #sd.finalize()
            #sr.finalize()
            #rr.finalize()

        xi_jk = (self.f0_JK * sd_jk) / self.rr_jk
    
        xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5
        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.Pi = xPi
        self.xi = xi
        self.xi_jk = xi_jk

        project_corr.__init__(self,self.xi,self.xi_jk,self.rp,self.Pi)
        self.cov_jk_norm = norm_cov(self.cov_jk)

class compute_wgp2(project_corr):
    """
    Computes the quadrupole component for galaxy-shear correlation. Eq 7 in arXiv 2307.02545.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
    Attributes:
    -----------
        r (ndarray): Array of projected radial separation bins.
        mu (ndarray): Array of mu (cosine of the angle) separation bins.
        xi (ndarray): 2D galaxy-shear plus correlation function in bins of 3D radial coordinate and mu.
        xi_jk (ndarray): 2D galaxy-shear plus correlation function for each Jackknife resampling.
        xip (ndarray): Projected correlation function across mu.
        xip_jk (ndarray): Projected correlation function across mu for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """

    
    def __init__(self,pcat, scat, rpcat, rscat,config,dup,dup_random):

        nbins = config['nbins']*config['grid_resolution']
        pi_max = config['rmax']-0.1
        npi = config['mubins']*config['grid_resolution']
        
        # get pair-normalisation factors

        Nrep = dup.Nrep(np.ones(len(pcat.w)).astype(bool),np.ones(len(scat.w)).astype(bool))
        Nrep_rand = dup_random.Nrep(np.ones(len(rpcat.w)).astype(bool),np.ones(len(rscat.w)).astype(bool))

        
        NGtot = scat.sumw*pcat.sumw - Nrep
        RRtot = rpcat.sumw*rscat.sumw - Nrep_rand
        RGtot = rpcat.sumw*scat.sumw
        f0 = RRtot / NGtot
        f1 = RRtot / RGtot

        # Pi bins
        Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
        r_bins  = np.logspace(np.log10(config['rmin']),np.log10(config['rmax']), nbins+1)

        # Arrays to save results
        gamma_sd = np.array([])
        gamma_sr = np.array([])
        gamma_x_sd = np.array([])
        gamma_x_sr = np.array([])
        
        npairs_sd = np.array([])
        npairs_sr = np.array([])
        npairs_rr = np.array([])
        
        r = np.array([])
        pi = np.array([])
        
        gamma_sd_jk  = np.zeros((config['NPatches'], npi*nbins))
        gamma_sr_jk  = np.zeros((config['NPatches'], npi*nbins))
        gamma_x_sd_jk  = np.zeros((config['NPatches'], npi*nbins))
        gamma_x_sr_jk  = np.zeros((config['NPatches'], npi*nbins))
        
        npairs_sd_jk = np.zeros((config['NPatches'], npi*nbins))
        npairs_sr_jk = np.zeros((config['NPatches'], npi*nbins))
        npairs_rr_jk = np.zeros((config['NPatches'], npi*nbins))
        
        f0_jk = np.zeros(config['NPatches'])
        f1_jk = np.zeros(config['NPatches'])
    
        xPi=(Pi[:-1]+Pi[1:])*0.5
        pibins = zip(Pi[:-1],Pi[1:])
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            
            r0 = max(config['rmin'],np.abs(plow))
            nbins0 = sum(r_bins >= r0)
            
            rr = treecorr.NNCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife')
            sd = treecorr.NGCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife')
            sr = treecorr.NGCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife')  
            
            sd.process(pcat, scat, metric='Euclidean', num_threads = config['ncores'])
            gamma_sd = np.append(gamma_sd,sd.xi)
            gamma_x_sd = np.append(gamma_x_sd,sd.xi_im)
            npairs_sd = np.append(npairs_sd,sd.weight)
            
            sr.process(rpcat, scat, metric='Euclidean', num_threads = config['ncores'])
            gamma_sr = np.append(gamma_sr,sr.xi)
            gamma_x_sr = np.append(gamma_x_sr,sr.xi_im)
            npairs_sr = np.append(npairs_sr,sr.weight)
            
            rr.process(rpcat, rscat, metric='Euclidean', num_threads = config['ncores'])
            npairs_rr = np.append(npairs_rr,rr.weight)
    
    
            #Here I compute the variance
            f_pairs   = lambda corrs: corrs[0].weight
            f_gamma   = lambda corrs: corrs[0].xi
            f_gamma_x = lambda corrs: corrs[0].xi_im
            gamma_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_gamma)
            gamma_sr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = f_gamma)
            gamma_x_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_gamma_x)
            gamma_x_sr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = f_gamma_x)
            
            npairs_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_pairs)
            npairs_sr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = f_pairs)
            npairs_rr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = f_pairs)
    
            r  = np.append(rr.rnom,r)
            pi = np.append(xPi[p]*np.ones(len(rr.rnom)),pi)
        
        for i in range(config['NPatches']):
            ppairs_jk = np.sum(pcat.w[pcat.patch != i])
            spairs_jk = np.sum(scat.w[scat.patch != i])
            rppairs_jk = np.sum(rpcat.w[rpcat.patch != i])        
            rspairs_jk = np.sum(rscat.w[rscat.patch != i])        
            
            NGtot_JK = ppairs_jk*spairs_jk - dup.Nrep(pcat.patch != i,scat.patch != i)
            RRtot_JK = rppairs_jk*rspairs_jk - dup_random.Nrep(rpcat.patch != i,rscat.patch != i) 
            RGtot_JK = spairs_jk*rppairs_jk 
    
            f0_jk[i] = RRtot_JK / NGtot_JK
            f1_jk[i] = RRtot_JK / RGtot_JK
        
        f0_jk = np.repeat(f0_jk,config['nbins']*config['mubins']).reshape(config['NPatches'],config['mubins'],config['nbins'])
        f1_jk = np.repeat(f1_jk,config['nbins']*config['mubins']).reshape(config['NPatches'],config['mubins'],config['nbins'])
        
        gamma_sd_jk  = gamma_sd_jk[:,:len(r)]
        gamma_sr_jk = gamma_sr_jk[:,:len(r)]
        gamma_x_sd_jk = gamma_x_sd_jk[:,:len(r)]
        gamma_x_sr_jk = gamma_x_sr_jk[:,:len(r)]
        
        npairs_sd_jk = npairs_sd_jk[:,:len(r)]
        npairs_sr_jk = npairs_sr_jk[:,:len(r)]
        npairs_rr_jk = npairs_rr_jk[:,:len(r)]

        # Now bin over r and mu
        m = np.abs(pi/r) < 1.
        self._pi = pi
        self._r  = r
        
        mean_gamma = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                               [gamma_sd[m],
                                                gamma_sr[m]], 
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'mean'
                                               )
    
        mean_gamma_x = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                               [gamma_x_sd[m],
                                                gamma_x_sr[m]], 
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'mean'
                                               )
        
        sum_pairs = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                               [npairs_sd[m],
                                                npairs_sr[m], 
                                                npairs_rr[m]], 
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
        
        mean_gamma_sd_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                gamma_sd_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'mean'
                                               )
        
        mean_gamma_sr_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                gamma_sr_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'mean'
                                               )
    
        mean_gamma_x_sd_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]),  
                                                gamma_x_sd_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'mean'
                                               )
        
        mean_gamma_x_sr_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                gamma_x_sr_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'mean'
                                               )
    
        
        sum_pairs_sd_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                npairs_sd_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
        
        sum_pairs_sr_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                npairs_sr_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
        
        sum_pairs_rr_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                npairs_rr_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )

        # Compute the xi and JK
        mu = mean_gamma.x_edge[:-1] + 0.5*np.diff(mean_gamma.x_edge)
        R = 10**(mean_gamma.y_edge[:-1] + 0.5*np.diff(mean_gamma.y_edge))
        sd, sr = mean_gamma.statistic*sum_pairs.statistic[:-1]
        sxd, sxr = mean_gamma_x.statistic*sum_pairs.statistic[:-1]
        rr = sum_pairs.statistic[-1]
        
        sd_jk = mean_gamma_sd_jk.statistic*sum_pairs_sd_jk.statistic
        sr_jk = mean_gamma_sr_jk.statistic*sum_pairs_sr_jk.statistic
        sxd_jk = mean_gamma_x_sd_jk.statistic*sum_pairs_sd_jk.statistic
        sxr_jk = mean_gamma_x_sr_jk.statistic*sum_pairs_sr_jk.statistic
        
        rr_jk = sum_pairs_rr_jk.statistic
    
        self.xi = (f0 * sd - f1 * sr)/ rr
        self.xi_jk = (f0_jk * sd_jk - f1_jk * sr_jk)/ rr_jk
        self._xi_x = (f0 * sxd - f1 * sxr)/ rr
        self._xi_x_jk = (f0_jk * sxd_jk - f1_jk * sxr_jk)/ rr_jk
        self.r  = R
        self.mu = mu

        l = 2
        sab = 2
        self._L_mu = np.zeros(self.xi.shape)
        self._L_mu_jk = np.zeros(self.xi_jk.shape)
        i = 0
        for m in mu:
            self._L_mu[i,:] = lpmn(l, sab, m)[0][-1,-1]
            self._L_mu_jk[:,i,:] = lpmn(l, sab, m)[0][-1,-1]
            i += 1    
            
        self._factor = ((2 * l + 1)/ 2.0)*(math.factorial(l - sab)/math.factorial(l + sab))
        
        project_corr.__init__(self,self._L_mu*self.xi,self._L_mu_jk*self.xi_jk,self.r,self.mu,self._factor)
        self.cov_jk_norm = norm_cov(self.cov_jk)

class compute_fast_wgp2(project_corr):
    
    """
    Allows to compute the quadrupole component for galaxy-shear correlation. Eq 7 in arXiv 2307.02545. This version allows the pre-computation of the random number of pairs and then it can be executed to compute the galaxy-shear correlation varying the shapes of the galaxies in the shape catalogue.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
    Methods:
    -----------
        execute(new_shapes=False,shapes=[]): Allows the computation of the projected galaxy-shape correlation. It can be used giving a different shape catalogue as input.
    """    
    def __init__(self,pcat, scat, rpcat, rscat,config,dup,dup_random):

        self.config = config
        self.pcat = pcat
        self.rpcat = rpcat
        self.rscat = rscat
        self.dup_random = dup_random
        self.dup = dup
        self.scat = scat
        self.rr_p = []
        
        nbins = config['nbins']*config['grid_resolution']
        pi_max = config['rmax']-0.1
        npi = config['mubins']*config['grid_resolution']

        
        # arrays to store the output
        npairs_rr_jk = np.zeros((config['NPatches'], npi*nbins))
        self.npairs_rr = np.array([])

        r = np.array([])
        pi = np.array([])
        
        print('Computing normalisation factors and the number of pairs in the random catalogues...')
        # get pair-normalisation factors
        Nrep = dup.Nrep(np.ones(len(pcat.w)).astype(bool),np.ones(len(scat.w)).astype(bool))
        Nrep_rand = dup_random.Nrep(np.ones(len(rpcat.w)).astype(bool),np.ones(len(rscat.w)).astype(bool))

        NGtot = scat.sumw*self.pcat.sumw - Nrep
        RRtot = self.rpcat.sumw*self.rscat.sumw - Nrep_rand
        self.f0 = RRtot / NGtot
        
        f0_jk = np.zeros(config['NPatches'])
        
        
        # Pi bins
        Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
        r_bins  = np.logspace(np.log10(config['rmin']),np.log10(config['rmax']), nbins+1)
        xPi=(Pi[:-1]+Pi[1:])*0.5
        pibins = zip(Pi[:-1],Pi[1:])
        
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            
            r0 = max(config['rmin'],np.abs(plow))
            nbins0 = sum(r_bins >= r0)

            rr = treecorr.NNCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife') 
            rr.process(rpcat, rscat, metric='Euclidean', num_threads = config['ncores'])

            f_pairs   = lambda corrs: corrs[0].weight
            npairs_rr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = f_pairs)

            r  = np.append(rr.rnom,r)
            pi = np.append(xPi[p]*np.ones(len(rr.rnom)),pi)
            self.npairs_rr = np.append(self.npairs_rr,rr.weight)
            
        for i in range(config['NPatches']):
            ppairs_jk = np.sum(self.pcat.w[self.pcat.patch != i])
            spairs_jk = np.sum(scat.w[scat.patch != i])
            rppairs_jk = np.sum(self.rpcat.w[self.rpcat.patch != i])        
            rspairs_jk = np.sum(self.rscat.w[self.rscat.patch != i])        
            
            NGtot_JK = ppairs_jk*spairs_jk - self.dup.Nrep(self.pcat.patch != i,scat.patch != i)
            RRtot_JK = rppairs_jk*rspairs_jk - self.dup_random.Nrep(self.rpcat.patch != i,self.rscat.patch != i) 
            RGtot_JK = spairs_jk*rppairs_jk 
    
            f0_jk[i] = RRtot_JK / NGtot_JK
        
        self.f0_jk = np.repeat(f0_jk,config['nbins']*config['mubins']).reshape(config['NPatches'],config['mubins'],config['nbins'])


        npairs_rr_jk = npairs_rr_jk[:,:len(r)]
        m = np.abs(pi/r) < 1.
        
        sum_pairs_rr_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                        npairs_rr_jk[:,m],
                                        bins=[config['mubins'],config['nbins']], 
                                        statistic = 'sum'
                                       )
        

        self.rr_jk = sum_pairs_rr_jk.statistic
        print('Everything ok :) Finished')
        
    def use_new_shapes(self,shapes):
        """
        Define new shape catalogue to compute the galaxy-shear correlation.

        Arguments
        ----------
        shapes : list
            Hosts the catalogue of the shapes (pandas array).
        Attributes:
        ----------
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        """
        
        config = self.config
        
        del(self.scat)
        if config['box']:
            x, y, z, g1, g2, w = config['col_names']
            zshift = 1.e6 #to put the observer far away                   
            scat  = treecorr.Catalog(g1=shapes[g1], 
                                     g2 = shapes[g2],
                                     x = shapes[x], 
                                     y = shapes[z]+zshift,  
                                     z = shapes[y], 
                                     w = shapes[w], 
                                     patch_centers = self.pcat.patch_centers)
        
        else:   
            ra, dec, r, g1, g2, w = config['col_names']
            scat  = treecorr.Catalog(g1 = shapes[g1],
                                          g2 = shapes[g2],
                                          ra=shapes[ra], 
                                          dec=shapes[dec], 
                                          r=shapes[r], 
                                          w = shapes[w], 
                                          patch_centers = self.pcat.patch_centers, 
                                          ra_units='deg', dec_units='deg')
            
        self.scat = scat

    def execute(self):
        """
        Compute the galaxy-shear correlation.

        Attributes:
        ----------
            r (ndarray): Array of projected radial separation bins.
            mu (ndarray): Array of mu (cosine of the angle) separation bins.
            xi (ndarray): 2D galaxy-shear plus correlation function in bins of 3D radial coordinate and mu.
            xi_jk (ndarray): 2D galaxy-shear plus correlation function for each Jackknife resampling.
            xip (ndarray): Projected correlation function across mu.
            xip_jk (ndarray): Projected correlation function across mu for each Jackknife resampling.
            cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
            cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
            std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
        """
        
        config = self.config
        scat = self.scat

        nbins = config['nbins']*config['grid_resolution']
        pi_max = config['rmax']-0.1
        npi = config['mubins']*config['grid_resolution']
        
        # Pi bins
        Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
        r_bins  = np.logspace(np.log10(config['rmin']),np.log10(config['rmax']), nbins+1)

        # Arrays to save results
        r = np.array([])
        pi = np.array([])

        gamma_sd = np.array([])        
        npairs_sd = np.array([])
        
        gamma_sd_jk  = np.zeros((config['NPatches'], npi*nbins))        
        npairs_sd_jk = np.zeros((config['NPatches'], npi*nbins))

        xPi=(Pi[:-1]+Pi[1:])*0.5
        pibins = zip(Pi[:-1],Pi[1:])
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            
            r0 = max(config['rmin'],np.abs(plow))
            nbins0 = sum(r_bins >= r0)
            
            sd = treecorr.NGCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife')
            
            sd.process(self.pcat, scat, metric='Euclidean', num_threads = config['ncores'])
            gamma_sd = np.append(gamma_sd,sd.xi)
            npairs_sd = np.append(npairs_sd,sd.weight)
    
    
            #Here I compute the variance
            f_pairs   = lambda corrs: corrs[0].weight
            f_gamma   = lambda corrs: corrs[0].xi
            gamma_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_gamma)        
            npairs_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_pairs)
    
            r  = np.append(sd.rnom,r)
            pi = np.append(xPi[p]*np.ones(len(sd.rnom)),pi)
                
        gamma_sd_jk  = gamma_sd_jk[:,:len(r)]        
        npairs_sd_jk = npairs_sd_jk[:,:len(r)]
        

        # Now bin over r and mu
        m = np.abs(pi/r) < 1.
        self._pi = pi
        self._r  = r
        
        mean_gamma = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                               gamma_sd[m], 
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'mean'
                                               )
    
        
        sum_pairs = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                               [npairs_sd[m],
                                                self.npairs_rr[m]], 
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
        
        mean_gamma_sd_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                gamma_sd_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'mean'
                                               )
        
        sum_pairs_sd_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                npairs_sd_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
                

        # Compute the xi and JK
        mu = mean_gamma.x_edge[:-1] + 0.5*np.diff(mean_gamma.x_edge)
        R = 10**(mean_gamma.y_edge[:-1] + 0.5*np.diff(mean_gamma.y_edge))
        sd = mean_gamma.statistic*sum_pairs.statistic[0]
        rr = sum_pairs.statistic[-1]
        
        sd_jk = mean_gamma_sd_jk.statistic*sum_pairs_sd_jk.statistic        
        
        self.xi = (self.f0 * sd)/ rr
        self.xi_jk = (self.f0_jk * sd_jk)/ self.rr_jk
        self.r  = R
        self.mu = mu

        l = 2
        sab = 2
        self._L_mu = np.zeros(self.xi.shape)
        self._L_mu_jk = np.zeros(self.xi_jk.shape)
        
        i = 0
        for m in mu:
            self._L_mu[i,:] = lpmn(l, sab, m)[0][-1,-1]
            self._L_mu_jk[:,i,:] = lpmn(l, sab, m)[0][-1,-1]
            i += 1    
            
        self._factor = ((2 * l + 1)/ 2.0)*(math.factorial(l - sab)/math.factorial(l + sab))
        
        project_corr.__init__(self,self._L_mu*self.xi,self._L_mu_jk*self.xi_jk,self.r,self.mu,self._factor)
        self.cov_jk_norm = norm_cov(self.cov_jk)
        


class get_wgx(project_corr):
    
    """
    Computes the projected cross component for galaxy-shear correlation. 
    
    Arguments:
    -----------
        xi (ndarray): 2D galaxy-shear cross correlation function in bins of 3D radial coordinate and mu.
        xi_jk (ndarray): 2D galaxy-shear cross correlation function for each Jackknife resampling.
        rcor (ndarray): Array of projected radial separation bins.
        pcor (ndarray): Array of separation bins in the direction that is going to be projected.       
        L_mu_jk (optional, ndarray): Legendre polinomials.
        factor (optional,float): Factor to be included in the integration.

    Attributes:
    -----------
        r (optional, ndarray): Array of 3D radial separation bins.
        mu (optional, ndarray): Array of mu (cosine of the angle) separation bins.
        rp (optional, ndarray): Array of projected radial separation bins.
        pi (optional, ndarray): Array of l.o.s. separation bins.
        xi (ndarray): 2D galaxy-shear cross correlation function.
        xip (ndarray): Projected correlation function across the pcor direction.
        xi_jk (ndarray): 2D galaxy-shear cross correlation function for each Jackknife resampling.
        xip_jk (ndarray): Projected correlation function across pcor for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """

    
    def __init__(self,xi_x,xi_x_jk,rcor,pcor,factor=1,L_mu=1,L_mu_jk=1,q = False):
        
        project_corr.__init__(self,L_mu*xi_x,L_mu_jk*xi_x_jk,rcor,pcor,factor)
        self.xi = xi_x
        self.xi_jk = xi_x_jk
        if q:
            self.r  = rcor
            self.mu = pcor
        else:
            self.rp = rcor
            self.Pi = pcor            
        self.cov_jk_norm = norm_cov(self.cov_jk)

class compute_2p_corr():
    
    """
    Computes correlation estimators useful to measure Intrinsic Alignment.

    This class calculates the projected galaxy-galaxy (wgg), galaxy-shear (wg+, wgx),
    and the quadrupole component of the galaxy-shear (wg+2, wgx2) correlations.
    
    Arguments:
    -----------
        positions (pandas array): Catalogue of galaxy positions.
        shapes (pandas array): Catalogue of galaxy positions and shapes.
        randoms_positions (pandas array): Catalogue of randoms positions.
        randoms_shapes (pandas array): Catalogue of random positions for the shape catalogue.

    Attributes:
    -----------
        config (dict): Configuration dictionary for the computation.
        scat (treecorr.Catalog): Catalog of shapes.
        pcat (treecorr.Catalog): Catalog of positions.
        rpcat (treecorr.Catalog): Catalog of random positions.
        rscat (treecorr.Catalog): Catalog of random positions.
        wgg (object, optional): Projected galaxy-galaxy correlation.
        wgp (object, optional): Projected galaxy-shear plus correlation.
        wgx (object, optional): Projected galaxy-shear cross correlation.
        wgp2 (object, optional): Projected quadrupole galaxy-shear plus correlation.
        wgx2 (object, optional): Projected quadrupole galaxy-shear cross correlation.
        wgp_fast (object, optional): Alternative computation of the projected galaxy-shear plus correlation.
        wgp2_fast (object, optional): Alternative computation of the projected quadrupole galaxy-shear plus correlation.

    Methods:
    -----------
        compute_wgg():
            Computes the projected galaxy-galaxy correlation.
        compute_wgp():
            Computes the projected galaxy-shear plus and cross correlation.
        compute_wgp2():
            Computes the projected quadrupole galaxy-shear plus and cross correlation.
        compute_wgp_fast():
            Computes the projected galaxy-shear plus correlation. Allows the pre-computation of the number of pairs in the random catalogues and computes the correlation for different shapes catalogues.
        compute_wgp2_fast():
            Computes the projected quadrupole galaxy-shear plus correlation. Allows the pre-computation of the number of pairs in the random catalogues and computes the correlation for different shapes catalogues.
            
    """
    def __init__(self,positions,shapes,randoms_positions,randoms_shapes,config):
        print(config)
        for key in config_setup.keys():
            if not config.get(key):
                config[key] = config_setup[key]  
        
        self.config = config
        
        if config['box']:

            
            x, y, z, g1, g2, w = config['col_names']
            zshift = 1.e6 #to put the observer far away
            
            self.pcat  = treecorr.Catalog(x = positions[x], 
                                     y = positions[z]+zshift, 
                                     z = positions[y], 
                                     w = positions[w], 
                                     npatch = config['NPatches'])
                    
            
            self.scat  = treecorr.Catalog(g1=shapes[g1], 
                                     g2 = shapes[g2],
                                     x = shapes[x], 
                                     y = shapes[z]+zshift,  
                                     z = shapes[y], 
                                     w = shapes[w], 
                                     patch_centers = self.pcat.patch_centers)

            
            self.rpcat = treecorr.Catalog(x=randoms_positions[x], 
                                     y=randoms_positions[z]+zshift,  
                                     z=randoms_positions[y], 
                                     npatch = config['NPatches'], 
                                     patch_centers = self.pcat.patch_centers
                                     )
            

            self.rscat = treecorr.Catalog(x=randoms_shapes[x], 
                                     y=randoms_shapes[z]+zshift,  
                                     z=randoms_shapes[y], 
                                     npatch = config['NPatches'], 
                                     patch_centers = self.pcat.patch_centers
                                     )

        
        else:   
            ra, dec, r, g1, g2, w = config['col_names']
                        
            self.pcat  = treecorr.Catalog(ra=positions[ra], 
                                     dec=positions[dec], 
                                     w = positions[w], 
                                     r=positions[r], 
                                     npatch = config['NPatches'], 
                                     ra_units='deg', dec_units='deg')

            self.scat  = treecorr.Catalog(g1 = shapes[g1],
                                          g2 = shapes[g2],
                                          ra=shapes[ra], 
                                          dec=shapes[dec], 
                                          r=shapes[r], 
                                          w = shapes[w], 
                                          patch_centers = self.pcat.patch_centers, 
                                          ra_units='deg', dec_units='deg')
           
            self.rpcat = treecorr.Catalog(ra=randoms_positions[ra],
                                          dec=randoms_positions[dec], 
                                          r=randoms_positions[r], 
                                          npatch = config['NPatches'],
                                          patch_centers = self.pcat.patch_centers, 
                                          ra_units='deg', dec_units='deg')

            self.rscat = treecorr.Catalog(ra=randoms_shapes[ra],
                                      dec=randoms_shapes[dec], 
                                      r=randoms_shapes[r], 
                                      npatch = config['NPatches'],
                                      patch_centers = self.pcat.patch_centers, 
                                      ra_units='deg', dec_units='deg')

        # match catalogues to get number of repeated objects
        self._dup = duplicate(self.pcat,self.scat,config['exact_position'],config['sky_threshold'])
        self._dup_random = duplicate(self.rpcat,self.rscat,config['exact_position'],config['sky_threshold'])



    def compute_wgg(self):       
        self.wgg = compute_wgg(self.pcat,self.rpcat,self.config)

    def compute_delta_sigma(self):       
        self.gs = compute_delta_sigma(self.pcat,self.scat,self.config)
        
    def compute_wgp(self):
        self.wgp = compute_wgp(self.pcat,self.scat,self.rpcat,self.rscat,self.config,self._dup,self._dup_random)
        self.wgx = get_wgx(self.wgp._xi_x,self.wgp._xi_x_jk,self.wgp.rp,self.wgp.Pi)
        
    def compute_wgp2(self):
        self.wgp2 = compute_wgp2(self.pcat,self.scat,self.rpcat,self.rscat,self.config,self._dup,self._dup_random)
        
        self.wgx2 = get_wgx(self.wgp2._xi_x,self.wgp2._xi_x_jk,
                           self.wgp2.r,self.wgp2.mu,
                           self.wgp2._factor,
                           self.wgp2._L_mu_jk[0],
                           self.wgp2._L_mu_jk,
                           True)
        
    def compute_wgp_fast(self):
        self.wgp_fast = compute_fast_wgp(self.pcat,self.scat,self.rpcat,self.rscat,self.config,self._dup,self._dup_random)

    def compute_wgp2_fast(self):
        self.wgp2_fast = compute_fast_wgp2(self.pcat,self.scat,self.rpcat,self.rscat,self.config,self._dup,self._dup_random)


def make_randoms_lightcone(ra, dec, z, size_random, col_names=['ra','dec','z']):

    ra_rand = np.random.uniform(min(ra), max(ra), size_random)
    sindec_rand = np.random.uniform(np.sin(min(dec*np.pi/180)), np.sin(max(dec*np.pi/180)), size_random)
    dec_rand = np.arcsin(sindec_rand)*(180/np.pi)

    y,xbins  = np.histogram(z, 50)
    x  = xbins[:-1]+0.5*np.diff(xbins)
    n = 20
    poly = np.polyfit(x,y,n)
    zr = np.random.uniform(z.min(),z.max(),1000000)
    poly_y = np.poly1d(poly)(zr)
    poly_y[poly_y<0] = 0.
    peso = poly_y/sum(poly_y)
    z_rand = np.random.choice(zr,len(ra_rand),replace=True,p=peso)

    #z_rand = np.random.choice(z,size=len(ra_rand),replace=True)

    d = {col_names[0]: ra_rand, col_names[1]: dec_rand, col_names[2]:z_rand}
    randoms = d
    #randoms = pd.DataFrame(data = d)

    return randoms


def make_randoms_box(x, y, z, size_random, col_names=['x','y','z']):
    
    val_min = x.min()
    val_max = x.max()
    xv, yv, zv = np.random.randint(val_min, val_max+1, size=(3,size_random)).astype('float')
    xv += np.random.uniform(0,1,len(xv))    
    yv += np.random.uniform(0,1,len(yv))    
    zv += np.random.uniform(0,1,len(zv))    
    
    randoms = {col_names[0]: np.clip(xv,val_min,val_max), col_names[1]: np.clip(yv,val_min,val_max), col_names[2]:np.clip(zv,val_min,val_max)}

    return randoms


