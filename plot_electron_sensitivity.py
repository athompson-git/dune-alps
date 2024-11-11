import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rc
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


import sys
sys.path.append("../")
from alplib.couplings import *


def plot_sensitivity(epem_save_file, emgamma_save_file=None, coll_epem_save_file=None, pair_prod_save_file=None):
    dune_dat = np.genfromtxt(epem_save_file)
    dune_ma = dune_dat[:,0]*1e6
    dune_g = dune_dat[:,1]
    dune_ll = 2*abs(max(dune_dat[:,2]) - dune_dat[:,2])
    dune_ll_post_cut = 2*abs(max(dune_dat[:,5]) - dune_dat[:,5])
    dune_ll_bkg_free = dune_dat[:,6]

    unique_masses = np.unique(dune_ma)
    for m in unique_masses:
        dune_ll[dune_ma == m] = 2*(max(dune_dat[:,2][dune_ma == m]) - dune_dat[:,2][dune_ma == m])
        dune_ll_post_cut[dune_ma == m] = 2*(max(dune_dat[:,5][dune_ma == m]) - dune_dat[:,5][dune_ma == m])


    DUNE_MA, DUNE_G = np.meshgrid(np.unique(dune_ma),np.unique(dune_g))
    DUNE_CHI2 = np.reshape(dune_ll, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))
    DUNE_CHI2_POSTCUT = np.reshape(dune_ll_post_cut, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))
    DUNE_CHI2_BKGFREE = np.reshape(dune_ll_bkg_free, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))


    # BEGIN PLOTTING

    fig, ax = plt.subplots(figsize=(8,6))

    DUNE_CL = 4.61
    plt.contour(DUNE_MA, DUNE_G, DUNE_CHI2.transpose(), levels=[DUNE_CL], colors=["crimson"])
    plt.contour(DUNE_MA, DUNE_G, DUNE_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["crimson"], linestyles='dashed')
    plt.contour(DUNE_MA, DUNE_G, DUNE_CHI2_BKGFREE.transpose(), levels=[2.3], colors=["crimson"], linestyles='dotted')

    if emgamma_save_file is not None:
        dune_egamma_dat = np.genfromtxt(emgamma_save_file)
        dune_egamma_ma = dune_egamma_dat[:,0]*1e6
        dune_egamma_g = dune_egamma_dat[:,1]
        dune_egamma_ll = 2*abs(max(dune_egamma_dat[:,2]) - dune_egamma_dat[:,2])
        dune_egamma_ll_post_cut = 2*abs(max(dune_egamma_dat[:,4]) - dune_egamma_dat[:,4])

        unique_masses = np.unique(dune_egamma_ma)
        for m in unique_masses:
            dune_egamma_ll[dune_egamma_ma == m] = 2*(max(dune_egamma_dat[:,2][dune_egamma_ma == m]) - dune_egamma_dat[:,2][dune_egamma_ma == m])
            dune_egamma_ll_post_cut[dune_egamma_ma == m] = 2*(max(dune_egamma_dat[:,4][dune_egamma_ma == m]) - dune_egamma_dat[:,4][dune_egamma_ma == m])

        # plot contours for DUNE sensitivity
        DUNE_EG_MA, DUNE_EG_G = np.meshgrid(np.unique(dune_egamma_ma),np.unique(dune_egamma_g))
        DUNE_EG_CHI2 = np.reshape(dune_egamma_ll, (np.unique(dune_egamma_ma).shape[0],np.unique(dune_egamma_g).shape[0]))
        DUNE_EG_CHI2_POSTCUT = np.reshape(dune_egamma_ll_post_cut, (np.unique(dune_egamma_ma).shape[0],np.unique(dune_egamma_g).shape[0]))

        plt.contour(DUNE_EG_MA, DUNE_EG_G, DUNE_EG_CHI2.transpose(), levels=[DUNE_CL], colors=["gold"])
        plt.contour(DUNE_EG_MA, DUNE_EG_G, DUNE_EG_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["gold"], linestyles='dashed')

    if coll_epem_save_file is not None:
        dune_epemcol_dat = np.genfromtxt(coll_epem_save_file)
        dune_epemcol_ma = dune_epemcol_dat[:,0]*1e6
        dune_epemcol_g = dune_epemcol_dat[:,1]
        dune_epemcol_ll = 2*abs(max(dune_epemcol_dat[:,2]) - dune_epemcol_dat[:,2])
        dune_epemcol_ll_post_cut = 2*abs(max(dune_epemcol_dat[:,4]) - dune_epemcol_dat[:,4])

        # plot contours for DUNE sensitivity
        DUNE_EPEMCOL_MA, DUNE_EPEMCOL_G = np.meshgrid(np.unique(dune_epemcol_ma),np.unique(dune_epemcol_g))
        DUNE_EPEMCOL_CHI2 = np.reshape(dune_epemcol_ll, (np.unique(dune_epemcol_ma).shape[0],np.unique(dune_epemcol_g).shape[0]))
        DUNE_EPEMCOL_CHI2_POSTCUT = np.reshape(dune_epemcol_ll_post_cut, (np.unique(dune_epemcol_ma).shape[0],np.unique(dune_epemcol_g).shape[0]))

        plt.contour(DUNE_EPEMCOL_MA, DUNE_EPEMCOL_G, DUNE_EPEMCOL_CHI2.transpose(), levels=[DUNE_CL], colors=["royalblue"])
        plt.contour(DUNE_EPEMCOL_MA, DUNE_EPEMCOL_G, DUNE_EPEMCOL_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["royalblue"], linestyles='dashed')

    if pair_prod_save_file is not None:
        dune_pairprod_dat = np.genfromtxt(pair_prod_save_file)
        dune_pairprod_ma = dune_pairprod_dat[:,0]*1e6
        dune_pairprod_g = dune_pairprod_dat[:,1]
        dune_pairprod_ll = 2*abs(max(dune_pairprod_dat[:,2]) - dune_pairprod_dat[:,2])
        dune_pairprod_ll_post_cut = dune_pairprod_dat[:,4] # 2*abs(max(dune_pairprod_dat[:,4]) - dune_pairprod_dat[:,4])

        # plot contours for DUNE sensitivity
        DUNE_PAIRPROD_MA, DUNE_PAIRPROD_G = np.meshgrid(np.unique(dune_pairprod_ma),np.unique(dune_pairprod_g))
        DUNE_PAIRPROD_CHI2 = np.reshape(dune_pairprod_ll, (np.unique(dune_pairprod_ma).shape[0],np.unique(dune_pairprod_g).shape[0]))
        DUNE_PAIRPROD_CHI2_POSTCUT = np.reshape(dune_pairprod_ll_post_cut, (np.unique(dune_pairprod_ma).shape[0],np.unique(dune_pairprod_g).shape[0]))

        plt.contour(DUNE_PAIRPROD_MA, DUNE_PAIRPROD_G, DUNE_PAIRPROD_CHI2.transpose(), levels=[DUNE_CL], colors=["green"])
        plt.contour(DUNE_PAIRPROD_MA, DUNE_PAIRPROD_G, DUNE_PAIRPROD_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["green"], linestyles='dashed')


    # import external constraints
    na64_me = np.genfromtxt("../CCMAxion/data/na64_missing_energy.txt", delimiter=",")
    na64 = np.genfromtxt("../CCMAxion/data/na64_electron.txt", delimiter=",")
    e137 = np.genfromtxt("../sharedata/existing_limits/e137_ww_correct_nophoton.txt", delimiter=",")
    e141 = np.genfromtxt("../sharedata/existing_limits/e141_caff.txt", delimiter=",")
    e774 = np.genfromtxt("../sharedata/existing_limits/e774_caff.txt", delimiter=",")
    orsay = np.genfromtxt("../sharedata/existing_limits/orsay_caff.txt", delimiter=",")
    sn1987a = np.genfromtxt("../sharedata/existing_limits/sn1987a_electron_2107-12393.txt", delimiter=",")
    hbstars = np.genfromtxt("../sharedata/existing_limits/hbstars_cont.txt", delimiter=",")
    ccm120 = np.genfromtxt("../sharedata/existing_limits/CCM120_alp_electron.txt")

    caff_factor = (0.3/sqrt(0.22290)) * (0.511e-3 / 80.379) / 2
    beam_dump_color = 'rosybrown'
    qcd_color = 'k'
    na64_color = 'gray'
    astro_color = 'silver'
    plt.fill(na64[:,0], na64[:,1], color=beam_dump_color)
    #plt.fill(na64_me[:,0], na64_me[:,1], color=na64_color, zorder=1)
    plt.fill(e141[:, 0], e141[:, 1]*caff_factor, color=beam_dump_color)
    plt.fill(orsay[:, 0], orsay[:, 1]*caff_factor, color=beam_dump_color)
    plt.fill(e774[:, 0], e774[:, 1]*caff_factor, color=beam_dump_color)
    plt.fill(e137[:, 0], 0.3*e137[:, 1], color=beam_dump_color)
    plt.fill(ccm120[:, 0], ccm120[:, 1], color=beam_dump_color, linewidth=1.0, zorder=1)

    # Astrophysical limits
    plt.fill_between(hbstars[:, 0], sqrt(4*pi*hbstars[:, 1]), y2=np.ones_like(hbstars[:,1]), color=astro_color, alpha=0.5, zorder=0)
    plt.fill(1e6*sn1987a[:,0], sn1987a[:,1], color=astro_color, alpha=0.5, zorder=0)

    # Plot QCD lines
    ma_vals = np.logspace(2, 9, 1000)
    #plt.fill_between(ma_vals, gae_DFSZ(ma_vals, 0.25, "DFSZII"), gae_DFSZ(ma_vals, 120, "DFSZII"),
    #                edgecolor=qcd_color, facecolor='none', hatch="....", linewidth=0.8)
    #plt.plot(ma_vals, gae_DFSZ(ma_vals, 0.25, "DFSZII"), color=qcd_color, ls="dashed", zorder=4, linewidth=1.0)
    #plt.plot(ma_vals, gae_DFSZ(ma_vals, 120, "DFSZII"), color=qcd_color, ls="dashed",  zorder=4, linewidth=1.0)
    #plt.annotate(text='', xy=(1.7e7,3e-4), xytext=(2.3e8,6.3e-7), arrowprops=dict(arrowstyle='<->', color=qcd_color))

    text_fs = 12
    plt.text(2.3e6,3e-5,'Beam Dumps', rotation=0, fontsize=text_fs, color="k", weight="bold")
    #plt.text(9e6,7e-6,'Orsay', rotation=-30, fontsize=text_fs, color="white", weight="bold")
    #plt.text(2e6,3e-4,'E774', rotation=0, fontsize=text_fs, color="k", weight="bold")
    #plt.text(6e6,3.5e-5,'E141', rotation=-30, fontsize=text_fs, color="k", weight="bold")
    #plt.text(5e4,5e-5,'NA64 Missing Energy', fontsize=text_fs, color="k", weight="bold")
    plt.text(6e4,4e-7,'Stellar \nCooling', rotation=0, fontsize=text_fs, color="k", weight="bold")
    #plt.text(6e7,1e-5,'QCD Axion\nModels', rotation=35, fontsize=text_fs, color="k", weight="bold")
    plt.text(2e6,3e-9,'SN1987a', rotation=0, fontsize=text_fs, color="k")
    #plt.text(1.7e4,3.5e-6,'CCM120', rotation=0, fontsize=10, color="k")

    # legends for contours
    from matplotlib.lines import Line2D
    line_dune = Line2D([0], [0], label=r'$a \to e^+ e^-$', color='crimson')
    line_dune_bkgfree = Line2D([0], [0], label=r'$a \to e^+ e^-$ (after cuts)', color='crimson', ls='dashed')
    line_egamma_dune = Line2D([0], [0], label=r'$1\gamma 1e^\pm$', color='gold')
    line_egamma_dune_cuts = Line2D([0], [0], label=r' $1\gamma 1e^\pm$ (after cuts)', color='gold', ls='dashed')
    line_epemcol_dune = Line2D([0], [0], label=r'Collinear $a \to e^+ e^-$', color='royalblue')
    line_epemcol_dune_cuts = Line2D([0], [0], label=r'collinear $a \to e^+ e^-$ (after cuts)', color='royalblue', ls='dashed')
    line_pairprod_dune = Line2D([0], [0], label=r'$a Z \to e^+ e^- Z$', color='green')
    line_pairprod_dune_cuts = Line2D([0], [0], label=r'$a Z \to e^+ e^- Z$ (after cuts)', color='green', ls='dashed')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([line_dune, line_dune_bkgfree])
    if emgamma_save_file is not None:
        handles.extend([line_egamma_dune, line_egamma_dune_cuts])
    if coll_epem_save_file is not None:
        handles.extend([line_epemcol_dune, line_epemcol_dune_cuts])
    if pair_prod_save_file is not None:
        handles.extend([line_pairprod_dune, line_pairprod_dune_cuts])
    plt.legend(handles=handles, loc="lower left", framealpha=1, fontsize=12)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((1e4,4e8))
    plt.ylim((5e-10,6e-4))
    plt.title("DUNE-ND LAr", loc="right", fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('$m_a$ [eV]', fontsize=15)
    plt.ylabel('$g_{ae}$', fontsize=15)
    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.show()


def plot_sensitivity_total(epem_save_file, emgamma_save_file=None, coll_epem_save_file=None, pair_prod_save_file=None):
    dune_dat = np.genfromtxt(epem_save_file)
    dune_ma = dune_dat[:,0]*1e6
    dune_g = dune_dat[:,1]
    dune_ll = 2*abs(max(dune_dat[:,2]) - dune_dat[:,2])
    dune_ll_post_cut = 2*abs(max(dune_dat[:,5]) - dune_dat[:,5])
    dune_ll_bkg_free = dune_dat[:,6]

    unique_masses = np.unique(dune_ma)
    for m in unique_masses:
        dune_ll[dune_ma == m] = 2*(max(dune_dat[:,2][dune_ma == m]) - dune_dat[:,2][dune_ma == m])
        dune_ll_post_cut[dune_ma == m] = 2*(max(dune_dat[:,5][dune_ma == m]) - dune_dat[:,5][dune_ma == m])


    DUNE_MA, DUNE_G = np.meshgrid(np.unique(dune_ma),np.unique(dune_g))
    DUNE_CHI2 = np.reshape(dune_ll, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))
    DUNE_CHI2_POSTCUT = np.reshape(dune_ll_post_cut, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))
    DUNE_CHI2_BKGFREE = np.reshape(dune_ll_bkg_free, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))


    # em - gamma
    dune_egamma_dat = np.genfromtxt(emgamma_save_file)
    dune_egamma_ma = dune_egamma_dat[:,0]*1e6
    dune_egamma_g = dune_egamma_dat[:,1]
    dune_egamma_ll = 2*abs(max(dune_egamma_dat[:,2]) - dune_egamma_dat[:,2])
    dune_egamma_ll_post_cut = 2*abs(max(dune_egamma_dat[:,4]) - dune_egamma_dat[:,4])

    unique_masses = np.unique(dune_egamma_ma)
    for m in unique_masses:
        dune_egamma_ll[dune_egamma_ma == m] = 2*(max(dune_egamma_dat[:,2][dune_egamma_ma == m]) - dune_egamma_dat[:,2][dune_egamma_ma == m])
        dune_egamma_ll_post_cut[dune_egamma_ma == m] = 2*(max(dune_egamma_dat[:,4][dune_egamma_ma == m]) - dune_egamma_dat[:,4][dune_egamma_ma == m])

    DUNE_EG_MA, DUNE_EG_G = np.meshgrid(np.unique(dune_egamma_ma),np.unique(dune_egamma_g))
    DUNE_EG_CHI2 = np.reshape(dune_egamma_ll, (np.unique(dune_egamma_ma).shape[0],np.unique(dune_egamma_g).shape[0]))
    DUNE_EG_CHI2_POSTCUT = np.reshape(dune_egamma_ll_post_cut, (np.unique(dune_egamma_ma).shape[0],np.unique(dune_egamma_g).shape[0]))

    # e+ e- collinear
    dune_epemcol_dat = np.genfromtxt(coll_epem_save_file)
    dune_epemcol_ma = dune_epemcol_dat[:,0]*1e6
    dune_epemcol_g = dune_epemcol_dat[:,1]
    dune_epemcol_ll = 2*abs(max(dune_epemcol_dat[:,2]) - dune_epemcol_dat[:,2])
    dune_epemcol_ll_post_cut = 2*abs(max(dune_epemcol_dat[:,4]) - dune_epemcol_dat[:,4])

    DUNE_EPEMCOL_MA, DUNE_EPEMCOL_G = np.meshgrid(np.unique(dune_epemcol_ma),np.unique(dune_epemcol_g))
    DUNE_EPEMCOL_CHI2 = np.reshape(dune_epemcol_ll, (np.unique(dune_epemcol_ma).shape[0],np.unique(dune_epemcol_g).shape[0]))
    DUNE_EPEMCOL_CHI2_POSTCUT = np.reshape(dune_epemcol_ll_post_cut, (np.unique(dune_epemcol_ma).shape[0],np.unique(dune_epemcol_g).shape[0]))

    # pair production
    dune_pairprod_dat = np.genfromtxt(pair_prod_save_file)
    dune_pairprod_ma = dune_pairprod_dat[:,0]*1e6
    dune_pairprod_g = dune_pairprod_dat[:,1]
    dune_pairprod_ll = 2*abs(max(dune_pairprod_dat[:,2]) - dune_pairprod_dat[:,2])
    dune_pairprod_ll_post_cut = dune_pairprod_dat[:,4] # 2*abs(max(dune_pairprod_dat[:,4]) - dune_pairprod_dat[:,4])

    DUNE_PAIRPROD_MA, DUNE_PAIRPROD_G = np.meshgrid(np.unique(dune_pairprod_ma),np.unique(dune_pairprod_g))
    DUNE_PAIRPROD_CHI2 = np.reshape(dune_pairprod_ll, (np.unique(dune_pairprod_ma).shape[0],np.unique(dune_pairprod_g).shape[0]))
    DUNE_PAIRPROD_CHI2_POSTCUT = np.reshape(dune_pairprod_ll_post_cut, (np.unique(dune_pairprod_ma).shape[0],np.unique(dune_pairprod_g).shape[0]))


    # get total contours

    # BEGIN PLOTTING

    fig, ax = plt.subplots(figsize=(8,6))

    DUNE_CL = 4.61

    c1 = plt.contour(DUNE_MA, DUNE_G, DUNE_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["b"], linestyles='solid')
    c2 = plt.contour(DUNE_PAIRPROD_MA, DUNE_PAIRPROD_G, DUNE_PAIRPROD_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["b"], linestyles='solid')
    c3 = plt.contour(DUNE_EPEMCOL_MA, DUNE_EPEMCOL_G, DUNE_EPEMCOL_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["b"], linestyles='solid')
    c4 = plt.contour(DUNE_EG_MA, DUNE_EG_G, DUNE_EG_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["b"], linestyles='solid')


    # import external constraints
    na64_me = np.genfromtxt("../CCMAxion/data/na64_missing_energy.txt", delimiter=",")
    na64 = np.genfromtxt("../CCMAxion/data/na64_electron.txt", delimiter=",")
    e137 = np.genfromtxt("../sharedata/existing_limits/e137_ww_correct_nophoton.txt", delimiter=",")
    e141 = np.genfromtxt("../sharedata/existing_limits/e141_caff.txt", delimiter=",")
    e774 = np.genfromtxt("../sharedata/existing_limits/e774_caff.txt", delimiter=",")
    orsay = np.genfromtxt("../sharedata/existing_limits/orsay_caff.txt", delimiter=",")
    sn1987a = np.genfromtxt("../sharedata/existing_limits/sn1987a_electron_2107-12393.txt", delimiter=",")
    hbstars = np.genfromtxt("../sharedata/existing_limits/hbstars_cont.txt", delimiter=",")
    ccm120 = np.genfromtxt("../sharedata/existing_limits/CCM120_alp_electron.txt")

    caff_factor = (0.3/sqrt(0.22290)) * (0.511e-3 / 80.379) / 2
    beam_dump_color = 'rosybrown'
    qcd_color = 'k'
    na64_color = 'gray'
    astro_color = 'silver'
    plt.fill(na64[:,0], na64[:,1], color=beam_dump_color)
    #plt.fill(na64_me[:,0], na64_me[:,1], color=na64_color, zorder=1)
    plt.fill(e141[:, 0], e141[:, 1]*caff_factor, color=beam_dump_color)
    plt.fill(orsay[:, 0], orsay[:, 1]*caff_factor, color=beam_dump_color)
    plt.fill(e774[:, 0], e774[:, 1]*caff_factor, color=beam_dump_color)
    plt.fill(e137[:, 0], 0.3*e137[:, 1], color=beam_dump_color)
    plt.fill(ccm120[:, 0], ccm120[:, 1], color=beam_dump_color, linewidth=1.0, zorder=1)

    # Astrophysical limits
    plt.fill_between(hbstars[:, 0], sqrt(4*pi*hbstars[:, 1]), y2=np.ones_like(hbstars[:,1]), color=astro_color, alpha=0.5, zorder=0)
    plt.fill(1e6*sn1987a[:,0], sn1987a[:,1], color=astro_color, alpha=0.5, zorder=0)

    # Plot QCD lines
    ma_vals = np.logspace(2, 9, 1000)
    #plt.fill_between(ma_vals, gae_DFSZ(ma_vals, 0.25, "DFSZII"), gae_DFSZ(ma_vals, 120, "DFSZII"),
    #                edgecolor=qcd_color, facecolor='none', hatch="....", linewidth=0.8)
    #plt.plot(ma_vals, gae_DFSZ(ma_vals, 0.25, "DFSZII"), color=qcd_color, ls="dashed", zorder=4, linewidth=1.0)
    #plt.plot(ma_vals, gae_DFSZ(ma_vals, 120, "DFSZII"), color=qcd_color, ls="dashed",  zorder=4, linewidth=1.0)
    #plt.annotate(text='', xy=(1.7e7,3e-4), xytext=(2.3e8,6.3e-7), arrowprops=dict(arrowstyle='<->', color=qcd_color))

    text_fs = 12
    plt.text(2.3e6,3e-5,'Beam Dumps', rotation=0, fontsize=text_fs, color="k", weight="bold")
    #plt.text(9e6,7e-6,'Orsay', rotation=-30, fontsize=text_fs, color="white", weight="bold")
    #plt.text(2e6,3e-4,'E774', rotation=0, fontsize=text_fs, color="k", weight="bold")
    #plt.text(6e6,3.5e-5,'E141', rotation=-30, fontsize=text_fs, color="k", weight="bold")
    #plt.text(5e4,5e-5,'NA64 Missing Energy', fontsize=text_fs, color="k", weight="bold")
    plt.text(6e4,4e-7,'Stellar \nCooling', rotation=0, fontsize=text_fs, color="k", weight="bold")
    #plt.text(6e7,1e-5,'QCD Axion\nModels', rotation=35, fontsize=text_fs, color="k", weight="bold")
    plt.text(2e6,3e-9,'SN1987a', rotation=0, fontsize=text_fs, color="k")
    #plt.text(1.7e4,3.5e-6,'CCM120', rotation=0, fontsize=10, color="k")

    # legends for contours
    from matplotlib.lines import Line2D
    line_lar = Line2D([0], [0], label=r'DUNE ND-LAr', color='b')
    line_gar = Line2D([0], [0], label=r'DUNE ND-GAr', color='orange')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([line_lar, line_gar])
    plt.legend(handles=handles, loc="lower left", framealpha=1, fontsize=12)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((1e4,4e8))
    plt.ylim((5e-10,6e-4))
    plt.title("DUNE-ND LAr", loc="right", fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('$m_a$ [eV]', fontsize=15)
    plt.ylabel('$g_{ae}$', fontsize=15)
    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.show()




def main():
    #plot_sensitivity(epem_save_file="sensitivities/epem_sensitivity_BKG_cutflow_20240630.txt",
    #                 emgamma_save_file="sensitivities/1g1em_sensitivity_BKG_cutflow_20240323.txt",
    #                 coll_epem_save_file="sensitivities/epem_col_sensitivity_BKG_cutflow_20240324.txt",
    #                 pair_prod_save_file="sensitivities/epem_pp_sensitivity_BKG_cutflow_20240507.txt")


    plot_sensitivity_total(epem_save_file="sensitivities/epem_sensitivity_BKG_cutflow_20240630.txt",
                     emgamma_save_file="sensitivities/1g1em_sensitivity_BKG_cutflow_20240323.txt",
                     coll_epem_save_file="sensitivities/epem_col_sensitivity_BKG_cutflow_20240324.txt",
                     pair_prod_save_file="sensitivities/epem_pp_sensitivity_BKG_cutflow_20240507.txt")



if __name__ == "__main__":
    main()