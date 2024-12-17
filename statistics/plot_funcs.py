import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import scienceplots


def diff(vsf1, e_vsf1, vsf2, e_vsf2):
    ## vsf1 == GR
    ## vsf2 == MG
    D = vsf1/vsf2 - 1
    eD = np.sqrt( (e_vsf1/vsf2)**2 + (e_vsf2*vsf1/vsf2**2)**2 )

    return D, eD

def plot_func(a, VSF_MG_dict, VSF_GR_dict):
    rv_mg = VSF_MG_dict['r']
    vsf_mg = VSF_MG_dict['vsf']
    e_vsf_mg = VSF_MG_dict['err']

    rv_gr = VSF_GR_dict['r']
    vsf_gr = VSF_GR_dict['vsf']
    e_vsf_gr = VSF_GR_dict['err']

    D,eD = diff(vsf_mg, e_vsf_mg, vsf_gr, e_vsf_gr)

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
            ax1.errorbar(rv_mg,vsf_mg[i],e_vsf_mg[i],
                            c=colors[i], mfc='w', fmt='.-', ms=ms)

        colormap = plt.cm.Blues
        colors = [colormap(i) for i in np.linspace(0.3,1,a['n_z'])]
        for i in range(a['n_z']):
            ax1.errorbar(rv_gr,vsf_gr[i],e_vsf_gr[i],
                            c=colors[i], mfc='w', fmt='.-', ms=ms)
        
        ax1.set_ylabel('VSF [$h^3/\\mathrm{Mpc}^3$]')
        ax1.set_title(f"$\\Lambda$CDM vs $f(R)$ ($\\Delta = -{a['Delta']}$)")

        ax1.loglog()
        
        colormap = plt.cm.binary
        colors = [colormap(i) for i in np.linspace(0.3,1.0,a['n_z'])]
        for i in range(a['n_z']):
            ax2.errorbar(rv_mg, D[i], eD[i],
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
