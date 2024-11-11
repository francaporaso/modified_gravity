'''for small cuts in radial bins, to be used with a forVoid_paste to unify profiles'''

import sys
import os
sys.path.append('home/fcaporaso/lens_codes_v3.7/')
from maria_func import *
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import LambdaCDM
from astropy.wcs import WCS
# from fit_profiles_curvefit import *
# from astropy.stats import bootstrap
# from astropy.utils import NumpyRNGContext
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from multiprocessing import Pool, Process
import argparse
from astropy.constants import G,c,M_sun,pc
from scipy import stats
# from models_profiles import Gamma
# For map
wcs = WCS(naxis=2)
wcs.wcs.crpix = [0., 0.]
wcs.wcs.cdelt = [1./3600., 1./3600.]
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]    

#parameters
cvel = c.value;    # Speed of light (m.s-1)
G    = G.value;    # Gravitational constant (m3.kg-1.s-2)
pc   = pc.value    # 1 pc (m)
Msun = M_sun.value # Solar mass (kg)

'''
sample     = 'pruDES'
idlist     = None
lcat       = 'voids_MICE.dat'
Rv_min     = 15.
Rv_max     = 25.
rho1_min   = -1.
rho1_max   = 2.
rho2_min   = -1.
rho2_max   = 100.
FLAG       = 2.
z_min      = 0.1
z_max      = 0.3
RIN        = 0.05
ROUT       = 5.0
ndots      = 40
ncores     = 128
hcosmo     = 1.
nback      = 30.
domap      = False
addnoise   = False
'''

def div_area(a, b, num=50):
    '''a(float): radio interno
       b(float): radio externo
       num(int): numero de anillos de igual area
       
       returns
       r(1d-array): radios de los num+1 anillos, con el último elemento igual a b'''
    num = int(num)
    r = np.zeros(num+1)
    r[0] = a
    A = np.pi * (b**2 - a**2)
    
    for k in np.arange(1,num+1):
        r[k] = np.round(np.sqrt(k*A/(num*np.pi) + a**2),2)
        
    if r[-1] != b:
        raise ValueError(f'No se calcularon los radios de forma correcta, el ultimo radio es {r[-1]} != {b}')
    return r


def SigmaCrit(zl, zs, h=1.):
    '''Calcula el Sigma_critico dados los redshifts. 
    Debe ser usada con astropy.cosmology y con astropy.constants
    
    zl:   (float) redshift de la lente (lens)
    zs:   (float) redshift de la fuente (source)
    h :   (float) H0 = 100.*h
    '''

    cosmo = LambdaCDM(H0=100*h, Om0=0.3, Ode0=0.7)

    dl  = cosmo.angular_diameter_distance(zl).value
    Dl = dl*1.e6*pc #en m
    ds  = cosmo.angular_diameter_distance(zs).value              #dist ang diam de la fuente
    dls = cosmo.angular_diameter_distance_z1z2(zl, zs).value      #dist ang diam entre fuente y lente
                
    BETA_array = dls / ds

    return (((cvel**2.0)/(4.0*np.pi*G*Dl))*(1./BETA_array))*(pc**2/Msun)


def partial_map():
        pass

def partial_profile(RA0,DEC0,Z,Rv,
                    RIN,ROUT,ndots,h,
                    addnoise):

        '''
        calcula el perfil de 1 solo void, tomando el centro del void y su redshift
        RA0,DEC0 (float): posicion del centro del void
        Z: redshift del void
        RIN,ROUT: bordes del perfil
        ndots: cantidad de puntos del perfil
        h: cosmologia
        addnoise(bool): agregar ruido (forma intrinseca) a las galaxias de fondo
        devuelve la densidad proyectada (Sigma), el contraste(DSigma), la cant de galaxias por bin (Ninbin) 
        y las totales (Ntot)'''
        
        ndots = int(ndots)

        Rv   = Rv/h *u.Mpc
        cosmo = LambdaCDM(H0=100*h, Om0=0.25, Ode0=0.75)
        
        DEGxMPC = cosmo.arcsec_per_kpc_proper(Z).to('deg/Mpc')
        delta = (DEGxMPC*(ROUT*Rv))

        pos_angles = 0*u.deg, 90*u.deg, 180*u.deg, 270*u.deg
        c1 = SkyCoord(RA0*u.deg, DEC0*u.deg)
        c2 = np.array([c1.directional_offset_by(pos_angle, delta) for pos_angle in pos_angles])

        mask = (S.dec_gal < c2[0].dec.deg)&(S.dec_gal > c2[2].dec.deg)&(S.ra_gal < c2[1].ra.deg)&(
                S.ra_gal > c2[3].ra.deg)&(S.z_cgal > (Z+0.1))
        
        catdata = S[mask]

        del mask, delta

        # sigma_c = SigmaCrit(Z, catdata.z_cgal)
        sigma_c = SigmaCrit(Z, catdata.z_cgal)   # Higuchi et al 2013 (ec 4)
        
        rads, theta, *_ = eq2p2(np.deg2rad(catdata.ra_gal), np.deg2rad(catdata.dec_gal),
                                  np.deg2rad(RA0), np.deg2rad(DEC0))
                               
        
        e1     = catdata.gamma1
        e2     = -1.*catdata.gamma2

        # Add shape noise due to intrisic galaxy shapes        
        if addnoise:
            es1 = -1.*catdata.eps1
            es2 = catdata.eps2
            e1 += es1
            e2 += es2
        
        #get tangential ellipticities 
        et = (-e1*np.cos(2*theta)-e2*np.sin(2*theta))*sigma_c/Rv.value
        #get cross ellipticities
        ex = (-e1*np.sin(2*theta)+e2*np.cos(2*theta))*sigma_c/Rv.value
               
        #get convergence
        k  = catdata.kappa*sigma_c/Rv.value

        r = (np.rad2deg(rads)/DEGxMPC.value)/(Rv.value)
        #r = (np.rad2deg(rads)*3600*KPCSCALE)/(Rv*1000.)
        Ntot = len(catdata)        

        del catdata
        del e1, e2, theta, sigma_c, rads

        bines = np.linspace(RIN,ROUT,num=ndots+1)
        dig = np.digitize(r,bines)
                
        SIGMAwsum    = np.empty(ndots)
        DSIGMAwsum_T = np.empty(ndots)
        DSIGMAwsum_X = np.empty(ndots)
        N_inbin      = np.empty(ndots)
                                             
        for nbin in range(ndots):
                mbin = dig == nbin+1              

                SIGMAwsum[nbin]    = k[mbin].sum()
                DSIGMAwsum_T[nbin] = et[mbin].sum()
                DSIGMAwsum_X[nbin] = ex[mbin].sum()
                N_inbin[nbin]      = np.count_nonzero(mbin)
        
        output = np.array([SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin, Ntot], dtype=object)
        #output = (SIGMAwsum, DSIGMAwsum_T, DSIGMAwsum_X, N_inbin, Ntot)
        
        return output

def partial_profile_unpack(minput):
	return partial_profile(*minput)

        
def main(lcat, sample='pru', output_file=None,
         Rv_min=0., Rv_max=50.,
         rho1_min=-1., rho1_max=0.,
         rho2_min=-1., rho2_max=100.,
         z_min = 0.1, z_max = 1.0,
         domap = False, RIN = .05, ROUT =5.,
         ndots= 40, ncores=10, 
         idlist= None, hcosmo=1.0, 
         addnoise = False, FLAG = 2.):

        '''
        
        INPUT
        ---------------------------------------------------------
        sample         (str) sample name
        Rv_min         (float) lower limit for void radii - >=
        Rv_max         (float) higher limit for void radii - <
        rho1_min       (float) lower limit for inner density - >=
        rho1_max       (float) higher limit for inner density - <
        rho2_min       (float) lower limit for outer density - >=
        rho2_max       (float) higher limit for outer density - <
        FLAG           (float) higher limit for flag - <
        z_min          (float) lower limit for z - >=
        z_max          (float) higher limit for z - <
        domap          (bool) Instead of computing a profile it 
                       will compute a map with 2D bins ndots lsize
        RIN            (float) Inner bin radius of profile
        ROUT           (float) Outer bin radius of profile
        ndots          (int) Number of bins of the profile
        ncores         (int) to run in parallel, number of cores
        h              (float) H0 = 100.*h
        addnoise       (bool) add shape noise
        '''

        cosmo = LambdaCDM(H0=100*hcosmo, Om0=0.25, Ode0=0.75)
        tini = time.time()
        
        print(f'Voids catalog {lcat}')
        print(f'Sample {sample}')
        print(f'RIN : {RIN}')
        print(f'ROUT: {ROUT}')
        print(f'ndots: {ndots}')
        print('Selecting voids with:')
        print(f'{Rv_min}   <=  Rv  < {Rv_max}')
        print(f'{z_min}    <=  Z   < {z_max}')
        print(f'{rho1_min}  <= rho1 < {rho1_max}')
        print(f'{rho2_min}  <= rho2 < {rho2_max}')
        
        if idlist:
                print('From id list '+idlist)
        # else:
                # print(lM_min,' <= log(M) < ',lM_max)
                # print(z_min,' <= z < ',z_max)
                # print(q_min,' <= q < ',q_max)
                # print(rs_min,' <= rs < ',rs_max)
                #print(R5s_min,' <= R5s < ',R5s_max)
                # print('resNFW_S < ',resNFW_max)
                # print('h ',hcosmo)
                # print('misalign '+str(misalign))
        
        if addnoise:
            print('ADDING SHAPE NOISE')
        
        #reading Lens catalog
                
        L = np.loadtxt(folder+lcat).T

        Rv    = L[1]
        ra    = L[2]
        dec   = L[3]
        z     = L[4]
        rho_1 = L[8] #Sobredensidad integrada a un radio de void 
        rho_2 = L[9] #Sobredensidad integrada máxima entre 2 y 3 radios de void 
        flag  = L[11]

        if idlist:
                ides = np.loadtxt(idlist).astype(int)
                mvoids = np.in1d(L[0],ides)
        else:                
                mvoids = ((Rv >= Rv_min)&(Rv < Rv_max))&((z >= z_min)&(z < z_max))&(
                         (rho_1 >= rho1_min)&(rho_1 < rho1_max))&((rho_2 >= rho2_min)&(rho_2 < rho2_max))&(flag >= FLAG)        
        # SELECT RELAXED HALOS
                
        Nvoids = np.count_nonzero(mvoids)

        if Nvoids < ncores:
                ncores = Nvoids
        
        print(f'Nvoids {Nvoids}')
        print(f'CORRIENDO EN {ncores} CORES')
        
        L = L[:,mvoids]
        
        zmean    = np.mean(L[4])
        Rvmean   = np.mean(L[1])
        rho2mean = np.mean(L[9])

        # Define K masks   
        ncen = 100
        
        kmask    = np.zeros((ncen+1,len(ra)))
        kmask[0] = np.ones(len(ra)).astype(bool)
        
        ramin  = np.min(ra)
        cdec   = np.sin(np.deg2rad(dec))
        decmin = np.min(cdec)
        dra    = ((np.max(ra)+1.e-5)  - ramin)/10.
        ddec   = ((np.max(cdec)+1.e-5) - decmin)/10.
        
        c = 1
        
        for a in range(10): 
                for d in range(10): 
                        mra  = (ra  >= ramin + a*dra)&(ra < ramin + (a+1)*dra) 
                        mdec = (cdec >= decmin + d*ddec)&(cdec < decmin + (d+1)*ddec) 
                        # plt.plot(ra[(mra*mdec)],dec[(mra*mdec)],'C'+str(c+1)+',')
                        kmask[c] = ~(mra&mdec)
                        c += 1
        
        ind_rand0 = np.arange(Nvoids)
        np.random.shuffle(ind_rand0)
        
        
        # SPLIT LENSING CAT
        
        lbins = int(round(Nvoids/float(ncores), 0))
        slices = ((np.arange(lbins)+1)*ncores).astype(int)
        slices = slices[(slices < Nvoids)]
        Lsplit = np.split(L.T,slices)
        Ksplit = np.split(kmask.T,slices)
        
        del L

        if domap:
                print('Sin mapa')           

        else:

            print(f'Profile has {ndots} bins')
            print(f'from {RIN} Rv to {ROUT} Rv')
            try:
                os.mkdir('../profiles')
            except FileExistsError:
                pass
            
            if not output_file:
                output_file = f'../profiles/voids/'

            # Defining radial bins
            bines = np.linspace(RIN,ROUT,num=ndots+1)
            R = (bines[:-1] + np.diff(bines)*0.5)

            # WHERE THE SUMS ARE GOING TO BE SAVED
            
            Ninbin = np.zeros((ncen+1,ndots))
            
            SIGMAwsum    = np.zeros((ncen+1,ndots)) 
            DSIGMAwsum_T = np.zeros((ncen+1,ndots)) 
            DSIGMAwsum_X = np.zeros((ncen+1,ndots))
                            
            # FUNCTION TO RUN IN PARALLEL
            partial = partial_profile_unpack
            

        print(f'Saved in ../{output_file+sample}.fits')


        LARGO = len(Lsplit)

        tslice = np.array([])
        
        for l, Lsplit_l in enumerate(Lsplit):
                
                print(f'RUN {l+1} OF {LARGO}')
                
                t1 = time.time()

                num = len(Lsplit_l)
                
                if num == 1:
                        entrada = [Lsplit_l[2], Lsplit_l[3],
                                   Lsplit_l[4],Lsplit_l[1],
                                   RIN,ROUT,ndots,hcosmo,
                                   addnoise]
                        
                        salida = [partial(entrada)]
                else:                
                        rin       = np.full(num, RIN)
                        rout      = np.full(num, ROUT)
                        nd        = np.full(num, ndots, dtype=int)
                        h_array   = np.full(num, hcosmo)
                        addnoise_array = np.full(num, addnoise, dtype=bool)
                        
                        entrada = np.array([Lsplit_l.T[2],Lsplit_l.T[3],
                                            Lsplit_l.T[4],Lsplit_l.T[1],
                                            rin,rout,nd,h_array,
                                            addnoise_array]).T

                        with Pool(processes=num) as pool:
                                salida = np.array(pool.map(partial,entrada))
                                pool.close()
                                pool.join()
                
                for j, profilesums in enumerate(salida):
                        
                        if domap:
                                print('Sin mapa')
                            
                        else:

                            km      = np.tile(Ksplit[l][j],(ndots,1)).T
                            Ninbin += np.tile(profilesums[3],(ncen+1,1))*km
                                                
                            SIGMAwsum    += np.tile(profilesums[0],(ncen+1,1))*km
                            DSIGMAwsum_T += np.tile(profilesums[1],(ncen+1,1))*km
                            DSIGMAwsum_X += np.tile(profilesums[2],(ncen+1,1))*km

                Ntot   = np.array([profilesums[-1] for profilesums in salida])

                t2 = time.time()
                ts = (t2-t1)/60.
                tslice = np.append(tslice, ts)
                print('TIME SLICE')
                print(f'{np.round(ts,4)} min')
                print('Estimated remaining time')
                print(f'{np.round(np.mean(tslice)*(LARGO-(l+1)), 3)} min')

        # AVERAGE VOID PARAMETERS AND SAVE IT IN HEADER

        h = fits.Header()
        h.append(('N_VOIDS',int(Nvoids)))
        h.append(('Lens_cat',lcat))
        #h.append(('MICE version sources 2.0'))
        h.append(('Rv_min',np.round(Rv_min,2)))
        h.append(('Rv_max',np.round(Rv_max,2)))
        h.append(('Rv_mean',np.round(Rvmean,4)))
        h.append(('rho1_min',np.round(rho1_min,2)))
        h.append(('rho1_max',np.round(rho1_max,2)))
        h.append(('rho2_min',np.round(rho2_min,2)))
        h.append(('rho2_max',np.round(rho2_max,2)))
        h.append(('rho2_mean',np.round(rho2mean,4)))
        h.append(('z_min',np.round(z_min,2)))
        h.append(('z_max',np.round(z_max,2)))
        h.append(('z_mean',np.round(zmean,4)))
        h.append(('hcosmo',np.round(hcosmo,4)))
        
        h.append(('---SLICES_INFO---'))
        h.append(('Rp_min',np.round(RIN,4)))
        h.append(('Rp_max',np.round(ROUT,4)))
        h.append(('ndots',np.round(ndots,4)))


        if domap:
                print('Sin mapa')         
        
        else:
                # COMPUTING PROFILE        
                Ninbin[DSIGMAwsum_T == 0] = 1.
                        
                Sigma     = (SIGMAwsum/Ninbin)
                DSigma_T  = (DSIGMAwsum_T/Ninbin)
                DSigma_X  = (DSIGMAwsum_X/Ninbin)
                
            
                table_p = [fits.Column(name='Rp', format='E', array=R),
                           fits.Column(name='Sigma',    format='E', array=Sigma.flatten()),
                           fits.Column(name='DSigma_T', format='E', array=DSigma_T.flatten()),
                           fits.Column(name='DSigma_X', format='E', array=DSigma_X.flatten()),
                           fits.Column(name='Ninbin', format='E', array=Ninbin.flatten())]

        tbhdu_p = fits.BinTableHDU.from_columns(fits.ColDefs(table_p))
        
        primary_hdu = fits.PrimaryHDU(header=h)
        
        hdul = fits.HDUList([primary_hdu, tbhdu_p])
        
        hdul.writeto(f'{output_file+sample}.fits',overwrite=True)

        print(f'File saved... {output_file+sample}.fits')
                
        tfin = time.time()
        
        print(f'Partial time: {np.round((tfin-tini)/60. , 3)} mins')
        

def run_in_parts(RIN,ROUT, nslices,
                lcat, sample='pru',output_file=None, Rv_min=0.,Rv_max=50., rho1_min=-1.,rho1_max=0., 
                rho2_min=-1.,rho2_max=100., z_min = 0.1, z_max = 1.0,domap=False, ndots= 40, ncores=10,
                idlist=None, hcosmo=1.0, addnoise=False, FLAG = 2.):
        '''calcula los RIN, ROUT que toma main para los dif cortes de R y corre el programa
        
        RIN, ROUT: radios interno y externo del profile
        nslices(int): cantidad de cortes
        
        '''
        cuts = np.round(np.linspace(RIN,ROUT,num=nslices+1),2)
        
        try:
                os.mkdir(f'../profiles/voids/Rv_{round(Rv_min)}-{round(Rv_max)}')
        except FileExistsError:
                pass

        if not output_file:
                output_file = f'../profiles/voids/Rv_{round(Rv_min)}-{round(Rv_max)}/'
        
        tslice = np.zeros(nslices)

        #orden inverso: calcula del corte mas externo al mas interno
        #cuts = cuts[::-1]
        for j in np.arange(nslices):
                RIN, ROUT = cuts[j], cuts[j+1]
                #ROUT, RIN = cuts[j], cuts[j+1]
                t1 = time.time()

                print(f'RUN {j+1} out of {nslices} slices')
                #print(f'RUNNING FOR RIN={RIN}, ROUT={ROUT}')

                main(lcat, sample+f'rbin_{j}',output_file=output_file, Rv_min=Rv_min, Rv_max=Rv_max, rho1_min=rho1_min, 
                    rho1_max=rho1_max, rho2_min=rho2_min, rho2_max=rho2_max, z_min=z_min, z_max=z_max, domap=domap,
                    RIN=RIN, ROUT=ROUT, ndots=ndots//nslices, ncores=ncores, idlist=idlist, hcosmo=hcosmo, addnoise=addnoise, FLAG=FLAG)

                t2 = time.time()
                tslice[j] = (t2-t1)/60.     
                #print('TIME SLICE')
                #print(f'{np.round(tslice[j],2)} min')
                print('Estimated remaining time for run in parts')
                print(f'{np.round(np.mean(tslice[:j+1])*(nslices-(j+1)),2)} min')

if __name__=='__main__':
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-sample', action='store', dest='sample',default='pru')
        parser.add_argument('-lens_cat', action='store', dest='lcat',default='voids_MICE.dat')
        parser.add_argument('-Rv_min', action='store', dest='Rv_min', default=0.)
        parser.add_argument('-Rv_max', action='store', dest='Rv_max', default=50.)
        parser.add_argument('-rho1_min', action='store', dest='rho1_min', default=-1.)
        parser.add_argument('-rho1_max', action='store', dest='rho1_max', default=1.)
        parser.add_argument('-rho2_min', action='store', dest='rho2_min', default=-1.)
        parser.add_argument('-rho2_max', action='store', dest='rho2_max', default=100.)
        parser.add_argument('-FLAG', action='store', dest='FLAG', default=2.)
        parser.add_argument('-z_min', action='store', dest='z_min', default=0.1)
        parser.add_argument('-z_max', action='store', dest='z_max', default=0.5)
        parser.add_argument('-domap', action='store', dest='domap', default='False')
        parser.add_argument('-addnoise', action='store', dest='addnoise', default='False')
        parser.add_argument('-RIN', action='store', dest='RIN', default=0.05)
        parser.add_argument('-ROUT', action='store', dest='ROUT', default=5.)
        parser.add_argument('-nbins', action='store', dest='nbins', default=40)
        parser.add_argument('-ncores', action='store', dest='ncores', default=10)
        parser.add_argument('-h_cosmo', action='store', dest='h_cosmo', default=1.)
        parser.add_argument('-ides_list', action='store', dest='idlist', default=None)
        parser.add_argument('-nback', action='store', dest='nback', default=30)
        parser.add_argument('-nslices', action='store', dest='nslices', default=1.)
        args = parser.parse_args()

        sample     = args.sample
        idlist     = args.idlist
        lcat       = args.lcat
        Rv_min     = float(args.Rv_min)
        Rv_max     = float(args.Rv_max) 
        rho1_min   = float(args.rho1_min)
        rho1_max   = float(args.rho1_max) 
        rho2_min   = float(args.rho2_min)
        rho2_max   = float(args.rho2_max) 
        FLAG       = float(args.FLAG) 
        z_min      = float(args.z_min) 
        z_max      = float(args.z_max) 
        RIN        = float(args.RIN)
        ROUT       = float(args.ROUT)
        ndots      = int(args.nbins)
        ncores     = int(args.ncores)
        hcosmo     = float(args.h_cosmo)
        nback      = float(args.nback)
        nslices      = int(args.nslices)

        if args.domap == 'True':
            domap = True
        elif args.domap == 'False':
            domap = False

        if args.addnoise == 'True':
            addnoise = True
        elif args.addnoise == 'False':
            addnoise = False

        folder = '/mnt/simulations/MICE/'
        S      = fits.open(folder+'MICE_sources_HSN_withextra.fits')[1].data
        
        if nback < 30.:
            nselec = int(nback*5157*3600.)
            j      = np.random.choice(np.array(len(S)),nselec)
            S  = S[j]

        #Sgal_coord = SkyCoord(S.ra_gal, S.dec_gal, unit='deg', frame='icrs')

        print('BACKGROUND GALAXY DENSINTY',len(S)/(5157*3600))

        tin = time.time()

        run_in_parts(RIN,ROUT, nslices,
                lcat, sample, Rv_min=Rv_min, Rv_max=Rv_max, rho1_min=rho1_min, rho1_max=rho1_max, rho2_min=rho2_min,
                rho2_max=rho2_max, z_min=z_min, z_max=z_max, ndots=ndots, ncores=ncores, hcosmo=hcosmo, FLAG=FLAG)

        tfin = time.time()

        print(f'TOTAL TIME: {np.round((tfin-tin)/60.,2)} min')

