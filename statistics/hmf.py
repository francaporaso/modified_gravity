import argparse as ag
import numpy as np
from astropy.cosmology import LambdaCDM
from astropy.io import fits

def loadgxcat(catname):
    '''
    Loads gx catalog and select w given masks.
    '''
    with fits.open('home/fcaporaso/cats/L768/'+catname) as f:
        z_halo  = f[1].data.true_redshift_gal
        lm_halo = f[1].data.halo_lm

    return z_halo, lm_halo

def Vol(z):
    '''
    Comoving volume between 2 redshifts z[0],z[1].
    '''
    global cosmo
    chi = cosmo.comoving_distance(z).value
    volcom = 4/3 * np.pi * (chi[1]**3 - chi[0]**3)

    return volcom

def N(logm,z):
    '''
    Count number of halos between logm[0] < N < logm[1] and z[0] < N < z[1].
    '''

    global Tcat
    mask = (Tcat[0] >= logm[0]) & (Tcat[0] < logm[1]) & (Tcat[3] >= z[0]) & (Tcat[3] < z[1])

    return np.sum(mask)

def HMF(logm_min, logm_max, zmin, zmax,
        n_logm=5, n_z=50, sim='fullsky'):
    '''
    Halo Mass Function in the range (rvmin, rvmax) and for the redshifts (zmin,zmax)
    '''
    # global Vcat

    
    logm_bins = np.linspace(logm_min, logm_max, n_logm)
    DlogM = np.diff(logm_bins)[0]

    logM = np.array([[logm_bins[j],logm_bins[j+1]] for j in range(n_logm-1)])
    logm_list = logm_bins[:-1] + np.diff(logm_bins)*0.5

    zbins = np.linspace(zmin,zmax,n_z)
    z = np.array([[zbins[j], zbins[j+1]] for j in range(n_z-1)])

    Nh = np.array([[N(logm_i,z_j) for z_j in z] for logm_i in logM])
    V  = np.array([Vol(z_j) for z_j in z])
    if sim != 'fullsky':
        V /= 8
    hmf = Nh/(V*DlogM)
    e_hmf = np.sqrt(Nh)/(V*DlogM)      ## asumiendo poisson e(x) = sqrt(x)

    return logm_list, hmf, e_hmf

def diff(hmf1, e_hmf1, hmf2, e_hmf2):
    ## hmf1 == GR
    ## hmf2 == MG
    D = hmf1/hmf2 - 1
    eD = np.sqrt( (e_hmf1/hmf2)**2 + (e_hmf2*hmf1/hmf2**2)**2 )

    return D, eD


if __name__ == '__main__':

    options = {
        '--logm_min':5.0, '--logm_max':50.0, '--zmin':0.0, '--zmax':1.0,
        '--flag':2.0,
        '--n_logm':10, '--n_z':3,
        '--filename':'test', '--simuname':'MG',
        '--plot':0
    }

    parser = ag.ArgumentParser()
    for key,value in options.items():
        if key[-4:]=='name':
            parser.add_argument(key, action='store', dest=key[2:], default=value, type=str)
        else:
            parser.add_argument(key, action='store', dest=key[2:], default=value, type=float)
    args = parser.parse_args()
    a = args.__dict__

    a['n_logm'] = int(a['n_logm'])
    a['n_z']  = int(a['n_z'])
    a['plot']   = bool(a['plot'])

    print('====== MG vs GR ======')
    h = 1.0
    Om0, Ode0 = 0.3089, 0.6911
    print('Cosmology: Planck15')
    cosmo = LambdaCDM(H0=100.0*h, Om0=Om0, Ode0=Ode0)

    Tcat = loadgxcat('MG_cosmohub19015.fits')
    logm_mg, hmf_mg, e_hmf_mg = HMF(a['logm_min'], a['logm_max'], a['zmin'], a['zmax'],
                                a['n_logm']+1, a['n_z']+1)

    Tcat = loadgxcat('GR_cosmohub19016.fits')
    logm_gr, hmf_gr, e_hmf_gr = HMF(a['logm_min'], a['logm_max'], a['zmin'], a['zmax'],
                                a['n_logm']+1, a['n_z']+1)
    
    D, eD = diff(hmf_mg, e_hmf_mg, hmf_gr, e_hmf_gr)
    filename = f"z0{int(a['zmin']*10)}-0{int(a['zmax']*10)}_logm{int(a['logm_min'])}-{int(a['logm_max'])}.csv"

    # print(rv_gr == rv_mg)
    # assert False

    if a['plot']:
        print('not implemented, \n saving results')
        pass
    else:   
        np.savetxt('hmf_MG_'+filename, np.column_stack([logm_mg, hmf_mg, e_hmf_mg]), delimiter=',')
        np.savetxt('hmf_GR_'+filename, np.column_stack([logm_gr, hmf_gr, e_hmf_gr]), delimiter=',')
        np.savetxt(f'hmf_diff_GR-MG_'+filename, np.column_stack([D, eD]), delimiter=',')

    print('End!')
