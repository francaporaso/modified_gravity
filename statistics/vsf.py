import argparse as ag
import numpy as np
from astropy.cosmology import LambdaCDM

def loadvoidcat(Mr, Delta, sim):
    '''
    Loads void catalog and select those w flag == 2.
    '''
    #0: Rv; 1: RA; 2:DEC; 3:z; 4,5,6: xv,yv,zv; 7: rho1; 8: rho2; 11: flag
    Vcat = np.loadtxt(f'../cats/voids_{sim}_Mr-{Mr}_Delta-{Delta}.dat').T
    Vcat = Vcat[:,Vcat[11] >= 2.0]
    return Vcat

def Vol(z):
    '''
    Comoving volume between 2 redshifts z[0],z[1].
    '''
    global cosmo
    chi = cosmo.comoving_distance(z).value
    volcom = 4/3 * np.pi * (chi[1]**3 - chi[0]**3)

    return volcom

def N(Rv,z):
    '''
    Count number of voids between Rv[0] < N < Rv[1] and z[0] < N < z[1].
    '''
    global Vcat
    mask = (Vcat[0] >= Rv[0]) & (Vcat[0] < Rv[1]) & (Vcat[3] >= z[0]) & (Vcat[3] < z[1])

    return np.sum(mask)

def VSF(rvmin, rvmax, zmin, zmax,
        nbins_Rv=5, nbins_z=50, sim='fullsky'):
    '''
    Void Size Function in the range (rvmin, rvmax) and for the redshifts (zmin,zmax)
    '''
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
    if sim != 'fullsky':
        V /= 8
    vsf = Nv/(V*DlogRv)
    e_vsf = np.sqrt(Nv)/(V*DlogRv)      ## asumiendo poisson e(x) = sqrt(x)

    return rv_list, vsf, e_vsf

def cN(Rv,z):
    '''
    Cumulative version of `N(Rv,z)`
    Count number of voids between N < Rv and z[0] < N < z[1].
    '''

    global Vcat
    mask = (Vcat[0] < Rv[1]) & (Vcat[3] >= z[0]) & (Vcat[3] < z[1])

    return np.sum(mask)

def cVSF(rvmin, rvmax, zmin, zmax,
        nbins_Rv=5, nbins_z=50):
    '''
    Cumulative Void Size Function. Not implemented
    '''

    # global Vcat

    logrvmin, logrvmax = np.log10(rvmin), np.log10(rvmax)
    rvbins = np.logspace(logrvmin, logrvmax, nbins_Rv)
    DlogRv = np.cumsum(np.diff(np.log10(rvbins)))

    Rv = np.array([[rvbins[j],rvbins[j+1]] for j in range(nbins_Rv-1)])
    rv_list = rvbins[:-1] + np.diff(rvbins)*0.5

    zbins = np.linspace(zmin,zmax,nbins_z)
    z = np.array([[zbins[j], zbins[j+1]] for j in range(nbins_z-1)])

    Nv = np.array([[cN(Rv_i,z_j) for z_j in z] for Rv_i in Rv])
    V  = np.array([Vol(z_j) for z_j in z])
    vsf = (Nv.T/DlogRv).T
    vsf /= V
    e_vsf = (np.sqrt(Nv.T)/DlogRv).T      ## asumiendo poisson e(x) = sqrt(x)
    e_vsf /= V 

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
        '--plot':False
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
    a['Mr']   = int(a['Mr'])
    a['plot']   = bool(a['plot'])

    print('----- MG vs GR -----')
    h = 1.0
    Om0, Ode0 = 0.3089, 0.6911
    print('Cosmology: Planck15')
    cosmo = LambdaCDM(H0=100.0*h, Om0=Om0, Ode0=Ode0)

    Vcat = loadvoidcat(Mr=a['Mr'], Delta=a['Delta'], sim='MG')
    # if Vcat[0].min() > a['rvmin']:
    #     a['rvmin'] = Vcat[0].min()
    # if Vcat[0].max() < a['rvmax']:
    #     a['rvmax'] = Vcat[0].max()
    rv_mg, vsf_mg, e_vsf_mg = VSF(a['rvmin'], a['rvmax'], a['zmin'], a['zmax'],
                                a['n_rv']+1,a['n_z']+1)

    Vcat = loadvoidcat(Mr=a['Mr'], Delta=a['Delta'], sim='GR')
    # if Vcat[0].min() > a['rvmin']:
    #     a['rvmin'] = Vcat[0].min()
    # if Vcat[0].max() < a['rvmax']:
    #     a['rvmax'] = Vcat[0].max()
    rv_gr, vsf_gr, e_vsf_gr = VSF(a['rvmin'], a['rvmax'], a['zmin'], a['zmax'],
                                a['n_rv']+1,a['n_z']+1)
    
    D, eD = diff(vsf_mg, e_vsf_mg, vsf_gr, e_vsf_gr)
    filename = f"z0{int(a['zmin']*10)}-0{int(a['zmax']*10)}_rv{int(a['rvmin'])}-{int(a['rvmax'])}_Mr-{a['Mr']}_D-{int(a['Delta']*10)}.csv"

    # print(rv_gr == rv_mg)
    # assert False

    if a['plot']:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        from matplotlib.lines import Line2D
        import scienceplots

        ms = 8

        cmap = plt.cm.binary
        colors = [cmap(i) for i in np.linspace(0.3,1,a['n_z'])]
        custom_lines = [Line2D([0], [0], color=colors[i], lw=1) for i in range(a['n_z'])]
        gr_color = [Line2D([0], [0], color='b', marker='.', markersize=ms, mfc='w', lw=1)]
        mg_color = [Line2D([0], [0], color='r', marker='.', markersize=ms, mfc='w', lw=1)]
        mice_color = [Line2D([0], [0], color='g', marker='.', markersize=ms, mfc='w', lw=1)]

        rvplt = np.logspace(np.log10(a['rvmin']), np.log10(a['rvmax']), a['n_rv']+1)
        zplt = np.linspace(a['zmin'],a['zmax'],a['n_z']+1)
        dz = np.diff(zplt)[0]
        z_str = [f'$z \\in [{zplt[i]:.2f},{zplt[i]+dz:.2f})$' for i in range(a['n_z'])]

        with plt.style.context('science'):
            fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=False,
                                          figsize=(5,10), height_ratios = [1.5,1],
                                          ) # divide as 2x2, plot top left
            colormap = plt.cm.Reds
            colors = [colormap(i) for i in np.linspace(0.3,1,a['n_z'])]
            for i in range(a['n_z']):
                ax1.errorbar(rv_mg,vsf_mg.T[i],e_vsf_mg.T[i],
                             c=colors[i], mfc='w', fmt='.-', ms=ms)

            colormap = plt.cm.Blues
            colors = [colormap(i) for i in np.linspace(0.3,1,a['n_z'])]
            for i in range(a['n_z']):
                ax1.errorbar(rv_gr,vsf_gr.T[i],e_vsf_gr.T[i],
                             c=colors[i], mfc='w', fmt='.-', ms=ms)
            
            ax1.set_ylabel('VSF [$h^3/\\mathrm{Mpc}^3$]')
            ax1.set_title(f"$\\Lambda$CDM vs $f(R)$ ( $M_r = -{a['Mr']}$, $\\Delta = -{a['Delta']}$ )")

            ax1.loglog()
            
            colormap = plt.cm.binary
            colors = [colormap(i) for i in np.linspace(0.3,1.0,a['n_z'])]
            for i in range(a['n_z']):
                ax2.errorbar(rv_mg, D.T[i], eD.T[i],
                             c=colors[i], mfc='w', fmt='.-', ms=ms, label=z_str[i])
            # ax2.fill_between(np.linspace(a['rvmin'],a['rvmax']), -.10, .10, color='gray', alpha=0.2, zorder=1)
            ax2.axhline(0, ls='--', c='gray')
            ax2.set_xlabel('$R_v$ [$\\mathrm{Mpc}$]')
            ax2.set_ylabel('$f(R)/\\Lambda CDM - 1$')
            ax2.set_xticks(rvplt.astype(int))
            ax2.get_xaxis().set_major_formatter(ScalarFormatter())
            ax2.minorticks_off()

            ax1.legend(mg_color+gr_color+custom_lines, ['$f(R)$', '$\\Lambda$CDM']+z_str)
            ax2.legend()

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()
            # plt.savefig(f'vsf_fRvsGR_Mr-{a["Mr"]}_D-{a["Delta"]}.png', dpi=300)

    else:   
        np.savetxt('vsf_MG_'+filename, np.column_stack([rv_mg, vsf_mg, e_vsf_mg]), delimiter=',')
        np.savetxt('vsf_GR_'+filename, np.column_stack([rv_gr, vsf_gr, e_vsf_gr]), delimiter=',')
        np.savetxt(f'vsf_diff_GR-MG_'+filename, np.column_stack([D, eD]), delimiter=',')

    if a['simuname'] == 'MICE':
        
        print('----- MICE vs LCDM -----')
        Om0, Ode0 = 0.25, 0.75
        cosmo = LambdaCDM(H0=100.0*h, Om0=Om0, Ode0=Ode0)
        print('Cosmology: WMAP7')
        
        a['zmin'],a['zmax'] = 0.1, 0.4

        Vcat = np.loadtxt('../../../FAMAF/Lensing/cats/MICE/voids_MICE.dat').T
        Vcat = Vcat[1:,Vcat[11] >= 2]
        rv_mice, vsf_mice, e_vsf_mice = VSF(a['rvmin'], a['rvmax'], a['zmin'], a['zmax'],
                                            a['n_rv']+1,a['n_z']+1, sim='MICE')

        Om0, Ode0 = 0.3089, 0.6911
        cosmo = LambdaCDM(H0=100.0*h, Om0=Om0, Ode0=Ode0)

        Vcat = loadvoidcat(Mr=a['Mr'], Delta=a['Delta'], sim='GR')
        rv_gr, vsf_gr, e_vsf_gr = VSF(a['rvmin'], a['rvmax'], a['zmin'], a['zmax'],
                                      a['n_rv']+1,a['n_z']+1)

        D, eD = diff(vsf_gr, e_vsf_gr, vsf_mice, e_vsf_mice)

        if a['plot']:
            zplt = np.linspace(a['zmin'],a['zmax'],a['n_z']+1)
            dz = np.diff(zplt)[0]
            z_str = [f'$z \\in [{zplt[i]:.2f},{zplt[i]+dz:.2f})$' for i in range(a['n_z'])]
            
            colormap1 = plt.cm.Greens
            colormap2 = plt.cm.Blues
            colors1 = [colormap1(i) for i in np.linspace(0.3,1,a['n_z'])]
            colors2 = [colormap2(i) for i in np.linspace(0.3,1,a['n_z'])]
            with plt.style.context('science'):
                fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, sharey=False,
                                               figsize=(5,10), height_ratios = [1.5,1])
                # fig, ax1 = plt.subplots(1,1, sharex=True, sharey=False,
                #                         figsize=(5,5))
                for i in range(a['n_z']):
                    ax1.errorbar(rv_mice,vsf_mice.T[i],e_vsf_mice.T[i],
                                c=colors1[i], mfc='w', fmt='.-', ms=ms,
                                label=z_str[i])
                    ax1.errorbar(rv_gr,vsf_gr.T[i],e_vsf_gr.T[i],
                                c=colors2[i], mfc='w', fmt='.-', ms=ms)
                    
                colormap = plt.cm.binary
                colors = [colormap(i) for i in np.linspace(0.3,1,a['n_z'])]
                for i in range(a['n_z']):
                    ax2.errorbar(rv_mice, D.T[i], eD.T[i],
                                c=colors[i], mfc='w', fmt='.-', ms=ms, label=z_str[i])
                ax2.axhline(0, ls='--', c='gray')
            
                ax1.loglog()
                ax2.set_xlabel('$R_v$ [$\\mathrm{Mpc}$]')
                ax1.set_ylabel('VSF [$h^3/\\mathrm{Mpc}^3$]')
                fig.suptitle(f"MICE vs $\\Lambda$CDM ( $M_r = -{a['Mr']}$, $\\Delta = -{a['Delta']}$ )")
                ax2.set_ylabel('$\\Lambda\\mathrm{CDM}/\\mathrm{MICE} - 1$')
                ax1.set_xticks(rvplt.astype(int))
                ax1.get_xaxis().set_major_formatter(ScalarFormatter())
                ax1.minorticks_off()
                ax1.legend()
                ax1.legend(mice_color+gr_color+custom_lines, ['MICE', '$\\Lambda$CDM']+z_str)
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(f'vsf_MICEvsGR_Mr-{a["Mr"]}_D-{a["Delta"]}.png',dpi=300)
            # plt.show()
 
        else:
            np.savetxt(f'vsf_mice_'+filename, np.column_stack([rv_mice, vsf_mice, e_vsf_mice]), delimiter=',')

            print('Diff con MICE')
            D, eD = diff(vsf_gr, e_vsf_gr, vsf_mice, e_vsf_mice)
            np.savetxt(f'vsf_diff_GR-MICE_'+filename, np.column_stack([D, eD]), delimiter=',')
            D, eD = diff(vsf_mg, e_vsf_mg, vsf_mice, e_vsf_mice)
            np.savetxt(f'vsf_diff_MG-MICE_'+filename, np.column_stack([D, eD]), delimiter=',')

    print('End!')
