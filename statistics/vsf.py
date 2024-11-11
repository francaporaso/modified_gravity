import argparse as ag
import numpy as np
from astropy.cosmology import LambdaCDM

def loadvoidcat(Mr, Delta, sim):
    #0: Rv; 1: RA; 2:DEC; 3:z; 4,5,6: xv,yv,zv; 7: rho1; 8: rho2; 11: flag
    Vcat = np.loadtxt(f'../cats/voids_{sim}_Mr-{Mr}_Delta-{Delta}.dat').T
    Vcat = Vcat[:,Vcat[11] >= 2.0]
    return Vcat

def Vol(z):
    global cosmo
    chi = cosmo.comoving_distance(z).value
    volcom = 4/3 * np.pi * (chi[1]**3 - chi[0]**3)

    return volcom

def N(Rv,z):
    global Vcat
    mask = (Vcat[0] >= Rv[0]) & (Vcat[0] < Rv[1]) & (Vcat[3] >= z[0]) & (Vcat[3] < z[1])

    return np.sum(mask)

def VSF(rvmin, rvmax, zmin, zmax,
        nbins_Rv=5, nbins_z=50):

    # global Vcat

    logrvmin, logrvmax = np.log10(rvmin), np.log10(rvmax)
    rvbins = np.logspace(logrvmin, logrvmax, nbins_Rv)
    DlogRv = np.diff(np.log10(rvbins))[0]

    Rv = np.array([[rvbins[j],rvbins[j+1]] for j in range(nbins_Rv-1)])
    rv_list = rvbins[:-1] + np.diff(rvbins)*0.5

    zbins = np.linspace(zmin,zmax,nbins_z)
    z = np.array([[zbins[j], zbins[j+1]] for j in range(nbins_z-1)])

    Nv = np.array([[N(Rv_i,z_j) for z_j in z] for Rv_i in Rv])
    V  = np.array([Vol(z_j) for z_j in z])
    vsf = Nv/(V*DlogRv)
    e_vsf = np.sqrt(Nv)/(V*DlogRv)      ## asumiendo poisson e(x) = sqrt(x)

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
        '--flag':2.0,
        '--n_rv':50, '--n_z':5,
        '--Mr':21, '--Delta':0.9,
        '--filename':'test', '--simuname':'MG-GR',
    }
    parser = ag.ArgumentParser()
    for key,value in options.items():
        if key[-4:]=='name':
            parser.add_argument(key, action='store', dest=key[2:], default=value, type=str)
        else:
            parser.add_argument(key, action='store', dest=key[2:], default=value, type=float)
    a = parser.parse_args()

    a.n_rv = int(a.n_rv)
    a.n_z  = int(a.n_z)
    a.Mr   = int(a.Mr)

    h = 1.0

    print('----- MG vs GR -----')
    Om0, Ode0 = 0.3089, 0.6911
    print('Cosmology: Planck15')
    cosmo = LambdaCDM(H0=100.0*h, Om0=Om0, Ode0=Ode0)

    Vcat = loadvoidcat(Mr=a.Mr, Delta=a.Delta, sim='MG')
    if Vcat[0].min() > a.rvmin:
        a.rvmin = Vcat[0].min()
    if Vcat[0].max() < a.rvmax:
        a.rvmax = Vcat[0].max()
    rv_mg, vsf_mg, e_vsf_mg = VSF(a.rvmin, a.rvmax, a.zmin, a.zmax,
                                a.n_rv+1,a.n_z+1)

    Vcat = loadvoidcat(Mr=a.Mr, Delta=a.Delta, sim='GR')
    # if Vcat[0].min() > a.rvmin:
    #     a.rvmin = Vcat[0].min()
    # if Vcat[0].max() < a.rvmax:
    #     a.rvmax = Vcat[0].max()
    rv_gr, vsf_gr, e_vsf_gr = VSF(a.rvmin, a.rvmax, a.zmin, a.zmax,
                                a.n_rv+1,a.n_z+1)
    
    D, eD = diff(vsf_gr, e_vsf_gr, vsf_mg, e_vsf_mg)
    filename = f'z0{int(a.zmin*10)}-0{int(a.zmax*10)}_rv{int(a.rvmin)}-{int(a.rvmax)}_Mr-{a.Mr}_D-{int(a.Delta*10)}.csv'

    np.savetxt('vsf_MG_'+filename, np.column_stack([rv_mg, vsf_mg, e_vsf_mg]), delimiter=',')
    np.savetxt('vsf_GR_'+filename, np.column_stack([rv_gr, vsf_gr, e_vsf_gr]), delimiter=',')
    np.savetxt(f'vsf_diff_GR-MG_'+filename, np.column_stack([D, eD]), delimiter=',')

    if a.simuname == 'MICE':
        print('----- MICE -----')
        Om0, Ode0 = 0.25, 0.75
        print('Cosmology: WMAP7')

        Vcat = np.loadtxt('../../../FAMAF/Lensing/cats/MICE/voids_MICE.dat').T
        Vcat = Vcat[1:,Vcat[11] >= 2]
        if Vcat[0].min() > a.rvmin:
            a.rvmin = Vcat[0].min()
        if Vcat[0].max() < a.rvmax:
            a.rvmax = Vcat[0].max()
        cosmo = LambdaCDM(H0=100.0*h, Om0=Om0, Ode0=Ode0)
        rv_mice, vsf_mice, e_vsf_mice = VSF(a.rvmin, a.rvmax, a.zmin, a.zmax,
                                         a.n_rv+1,a.n_z+1)
        np.savetxt(f'vsf_mice_'+filename, np.column_stack([rv_mice, vsf_mice, e_vsf_mice]), delimiter=',')

        print('Diff con MICE')
        D, eD = diff(vsf_gr, e_vsf_gr, vsf_mice, e_vsf_mice)
        np.savetxt(f'vsf_diff_GR-MICE_'+filename, np.column_stack([D, eD]), delimiter=',')
        D, eD = diff(vsf_mg, e_vsf_mg, vsf_mice, e_vsf_mice)
        np.savetxt(f'vsf_diff_MG-MICE_'+filename, np.column_stack([D, eD]), delimiter=',')

    print('End!')
