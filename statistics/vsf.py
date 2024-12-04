import argparse as ag
import numpy as np
from astropy.cosmology import LambdaCDM

def loadvoidcat(filename):
    '''
    Loads void catalog and select those w flag == 2.
    '''
    #0: Rv; 1: RA; 2:DEC; 3:z; 4,5,6: xv,yv,zv; 7: rho1; 8: rho2; 11: flag
    Vcat = np.loadtxt(filename).T
    Vcat = Vcat[:,Vcat[11] >= 2.0]
    return Vcat[[0,3],:] ## returns Rv and z

def Vol(chi):
    '''
    Comoving volume between 2 com dist0 chi[0],chi[1].
    '''
    volcom = 4/3 * np.pi * (chi[1]**3 - chi[0]**3)
    return volcom

def N(Rv,z,
      Vcat):
    '''
    Count number of voids between Rv[0] < N < Rv[1] and z[0] < N < z[1].
    '''
    mask = (Vcat[0] >= Rv[0]) & (Vcat[0] < Rv[1]) & (Vcat[1] >= z[0]) & (Vcat[1] < z[1])

    return np.sum(mask)

def VSF(rvmin, rvmax, zmin, zmax,
        n_rv=15, n_z=1, 
        foldername='/home/franco/ICE/modified_gravity/cats/', vcatname='',
        cosmo=LambdaCDM(H0=100, Om0=0.3089, Ode0=0.6911)):
    '''
    Void Size Function in the range (rvmin, rvmax) and for the redshifts (zmin,zmax)
    '''
    Vcat = loadvoidcat(foldername+vcatname)

    logrvmin, logrvmax = np.log10(rvmin), np.log10(rvmax)
    rvbins = np.logspace(logrvmin, logrvmax, n_rv+1)
    DlogRv = (logrvmax-logrvmin)/n_rv

    Rv = np.column_stack((rvbins[:-1],rvbins[1:]))
    rv_list = rvbins[:-1] + np.diff(rvbins)*0.5

    zbins = np.linspace(zmin,zmax,n_z+1)
    z = np.column_stack((zbins[:-1],zbins[1:]))
    chi = cosmo.comoving_distance(z).value

    Nv = np.array([[N(Rv_i,z_j,Vcat) for z_j in z] for Rv_i in Rv])
    V  = np.array([Vol(chi_j) for chi_j in chi])

    vsf = Nv/(V*DlogRv)
    e_vsf = np.sqrt(Nv)/(V*DlogRv) ## asumiendo poisson e(x) = sqrt(x)

    return rv_list, vsf, e_vsf

def diff(vsf1, e_vsf1, vsf2, e_vsf2):
    ## vsf1 == GR
    ## vsf2 == MG
    D = vsf1/vsf2 - 1
    eD = np.sqrt( (e_vsf1/vsf2)**2 + (e_vsf2*vsf1/vsf2**2)**2 )

    return D, eD


if __name__ == '__main__':

    options = {
        '--rvmin':5.0, '--rvmax':50.0, '--zmin':0.0, '--zmax':1.0,
        '--n_rv':15, '--n_z':1,
        '--foldername':'/home/franco/ICE/modified_gravity/cats/', '--vcatname':''
    }
    parser = ag.ArgumentParser()
    for key,value in options.items():
        if key[-4:]=='name':
            parser.add_argument(key, action='store', dest=key[2:], default=value, type=str)
        else:
            parser.add_argument(key, action='store', dest=key[2:], default=value, type=float)
    args = parser.parse_args()
    a = args.__dict__

    a['n_rv'] = int(a['n_rv'])
    a['n_z']  = int(a['n_z'])

    print('======= f(R) vs LCDM =======')
    h, Om0, Ode0 = 1.0, 0.3089, 0.6911
    cosmo = LambdaCDM(H0=100*h, Om0=Om0, Ode0=Ode0)
    print('Cosmology: Planck15')

    if a['vcatname'] != '':
        print('Calculating VSF for: ' + a['vcatname'])
        rv_list, vsf, e_vsf = VSF(a['rvmin'], a['rvmax'], a['zmin'], a['zmax'],
                                  a['n_rv'], a['n_z'],
                                  a['foldername'], a['vcatname'],
                                  cosmo=cosmo)
        
        np.savetxt('/home/franco/ICE/modified_gravity/statistics/results/'+'vsf_'+a['vcatname'][:-4]+f'z_n{a["n_z"]}_{int(10*a["zmin"])}-{int(10*a["zmax"])}'+'.csv',
                   np.column_stack([rv_list, vsf, e_vsf]),
                   delimiter=',')
        
    else:
        voids_name = np.array(['voids_fR_08.dat','voids_fR_09.dat','voids_LCDM_08.dat','voids_LCDM_09.dat'])

        for vname in voids_name:
            rv_list, vsf, e_vsf = VSF(a['rvmin'], a['rvmax'], a['zmin'], a['zmax'],
                                    a['n_rv'], a['n_z'],
                                    a['foldername'], vname,
                                    cosmo=cosmo)
            
            np.savetxt('/home/franco/ICE/modified_gravity/statistics/results/'+'vsf_'+vname[:-4]+f'z_n{a["n_z"]}_{10*int(a["zmin"])}-{10*int(a["zmax"])}'+'.csv',
                    np.column_stack([rv_list, vsf, e_vsf]),
                    delimiter=',')

    print('End!')
