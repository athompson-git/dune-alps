
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rc
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from scipy.signal import savgol_filter

import sys
sys.path.append("../")
from alplib.constants import *

existing_const_KKL = np.genfromtxt("../sharedata/existing_limits/kelly_kumar_liu_constraints/existing_constraints_KKL.txt")
sn_KKL = np.genfromtxt("../sharedata/existing_limits/kelly_kumar_liu_constraints/SN-Chang-Fiducial.csv", delimiter=",")


# get Dune dump mode
# TODO: for some reason, doesnt match paper!
limits_targetless = np.genfromtxt("../DUNE/limits/dune_target_limits_targetless_3mos_20210301.txt")
limits_targetless_1y = np.genfromtxt("../DUNE/limits/dune_target_limits_targetless_1yr_20210228.txt")
# wpd from paper:
limits_dune_dump = np.genfromtxt("data/other_DUNE_limits/dune_dump_mode_3mo_gagamma_limit.txt")

# get DUNE GAr
limits_gar = np.genfromtxt("sensitivities/photon_sensitivity_GAr_Aug2024.txt")
gar_ma = limits_gar[:,0]*1e6
gar_g = limits_gar[:,1]*1e3
gar_ll = limits_gar[:,2]
GAR_MA, GAR_G = np.meshgrid(np.unique(gar_ma),np.unique(gar_g))
GAR_CHI2 = np.reshape(gar_ll, (np.unique(gar_ma).shape[0],np.unique(gar_g).shape[0]))



def cleanLimitData(masses, lower_limit, upper_limit):
    diff_upper_lower = upper_limit - lower_limit
    upper_limit = np.delete(upper_limit, np.where(diff_upper_lower < 0))
    lower_limit = np.delete(lower_limit, np.where(diff_upper_lower < 0))
    masses = np.delete(masses, np.where(diff_upper_lower < 0))
    joined_limits = np.append(lower_limit, upper_limit[::-1])
    joined_masses = np.append(masses, masses[::-1])
    return joined_masses, joined_limits

# Find where the upper and lower arrays intersect at the tongue and clip
mass_array = limits_targetless[:,0]
upper_limit = limits_targetless[:,2]
lower_limit = limits_targetless[:,1]
diff_upper_lower = upper_limit - lower_limit
upper_limit = np.delete(upper_limit, np.where(diff_upper_lower < 0))
lower_limit = np.delete(lower_limit, np.where(diff_upper_lower < 0))
mass_array = np.delete(mass_array, np.where(diff_upper_lower < 0))
joined_limits_tgtls = np.append(lower_limit, upper_limit[::-1])
joined_masses_tgtls = np.append(mass_array, mass_array[::-1])

joined_masses_tgtls, joined_limits_tgtls = cleanLimitData(limits_targetless[:,0], limits_targetless[:,1], limits_targetless[:,2])
joined_masses_tgtls_1y, joined_limits_tgtls_1y = cleanLimitData(limits_targetless_1y[:,0], limits_targetless_1y[:,1], limits_targetless_1y[:,2])



def plot_sensitivity(save_file_2g, save_file_1g):
    # read in the calculated dune sens for DECAYS
    dune_dat = np.genfromtxt(save_file_2g)
    dune_ma = dune_dat[:,0]*1e6
    dune_g = dune_dat[:,1]*1e3
    dune_ll = 2*abs(max(dune_dat[:,2]) - dune_dat[:,2])*1.18
    dune_ll_post_cut = 2*(max(dune_dat[:,5]) - dune_dat[:,5]*100)
    dune_ll_decay_free = dune_dat[:,-1]*1.6  # compensating for ad-hoc flux increase

    # subtract off the maxima mass by mass
    unique_masses = np.unique(dune_ma)
    for m in unique_masses:
        dune_ll[dune_ma == m] = 2*(max(dune_dat[:,2][dune_ma == m]) - dune_dat[:,2][dune_ma == m])
        dune_ll_post_cut[dune_ma == m] = 2*(max(dune_dat[:,5][dune_ma == m]) - dune_dat[:,5][dune_ma == m])


    # read in the calculated dune sens for SCATTERING
    dune_scatter_dat = np.genfromtxt(save_file_1g)
    #dune_scatter_dat = dune_scatter_dat[dune_scatter_dat[:,0] < max(dune_scatter_dat[:,0])]  # for incomplete scan
    dune_ma_scatter = dune_scatter_dat[:,0]*1e6
    dune_g_scatter = dune_scatter_dat[:,1]*1e3
    dune_ll_scatter = 2*abs(max(dune_scatter_dat[:,2]) - dune_scatter_dat[:,2])
    dune_ll_scatter_acut = 2*abs(max(dune_scatter_dat[:,3]) - dune_scatter_dat[:,3])
    dune_ll_scatter_free = dune_scatter_dat[:,-1]

    unique_masses = np.unique(dune_ma_scatter)
    for m in unique_masses:
        dune_ll_scatter[dune_ma_scatter == m] = 2*(max(dune_scatter_dat[:,2][dune_ma_scatter == m]) - dune_scatter_dat[:,2][dune_ma_scatter == m])
        dune_ll_scatter_acut[dune_ma_scatter == m] = 2*(max(dune_scatter_dat[:,3][dune_ma_scatter == m]) - dune_scatter_dat[:,3][dune_ma_scatter == m])


    # plot contours for DUNE sensitivity
    DUNE_MA, DUNE_G = np.meshgrid(np.unique(dune_ma),np.unique(dune_g))
    DUNE_CHI2 = np.reshape(dune_ll, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))
    DUNE_CHI2_POSTCUT = np.reshape(dune_ll_post_cut, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))
    DUNE_CHI2_POSTCUT_BKGFREE = np.reshape(dune_ll_decay_free, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))

    DUNE_SCATTER_MA, DUNE_SCATTER_G = np.meshgrid(np.unique(dune_ma_scatter),np.unique(dune_g_scatter))
    DUNE_SCATTER_CHI2 = np.reshape(dune_ll_scatter, (np.unique(dune_ma_scatter).shape[0],np.unique(dune_g_scatter).shape[0]))
    DUNE_SCATTER_CHI2_POSTCUT = np.reshape(dune_ll_scatter_acut, (np.unique(dune_ma_scatter).shape[0],np.unique(dune_g_scatter).shape[0]))
    DUNE_SCATTER_CHI2_BKGFREE = np.reshape(dune_ll_scatter_free, (np.unique(dune_ma_scatter).shape[0],np.unique(dune_g_scatter).shape[0]))

    # Plot 2g limit at 90% CL
    DUNE_CL = 4.61

    fig, ax = plt.subplots(figsize=(8,6))    

    plt.contour(DUNE_MA, DUNE_G, DUNE_CHI2.transpose(), levels=[DUNE_CL], colors=["r"])
    plt.contour(DUNE_MA, DUNE_G, 5.0*DUNE_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["r"], linestyles='dashed')
    # 5x to account for better mass cut : checked separately with different scan
    plt.contour(DUNE_MA, DUNE_G, DUNE_CHI2_POSTCUT_BKGFREE.transpose(), levels=[2.3], colors=["r"], linestyles='dotted')

    plt.contour(DUNE_SCATTER_MA, DUNE_SCATTER_G, DUNE_SCATTER_CHI2.transpose(), levels=[DUNE_CL], colors=["royalblue"])
    plt.contour(DUNE_SCATTER_MA, DUNE_SCATTER_G, DUNE_SCATTER_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["royalblue"], linestyles='dashed')
    plt.contour(DUNE_SCATTER_MA, DUNE_SCATTER_G, DUNE_SCATTER_CHI2_BKGFREE.transpose(), levels=[3.0], colors=["royalblue"], linestyles='dotted')

    
    #plt.contour(DUNE_SCATTER_MA, DUNE_SCATTER_G, DUNE_CHI2_POSTCUT + DUNE_SCATTER_CHI2_POSTCUT.transpose(), levels=[DUNE_CL], colors=["k"], linestyles='solid')


    # set legends for contours
    line_1g = Line2D([0], [0], label=r'DUNE $1\gamma$', ls='solid', color='royalblue')
    line_1g_cut = Line2D([0], [0], label=r'DUNE $1\gamma$ (after cuts)', ls='dashed', color='royalblue')
    line_1g_free = Line2D([0], [0], label=r'DUNE $1\gamma$ (Background-Free)', ls='dotted', color='royalblue')
    line_2g = Line2D([0], [0], label=r'DUNE $2\gamma$', ls='solid', color='r')
    line_2g_cut = Line2D([0], [0], label=r'DUNE $2\gamma$ (after cuts)', ls='dashed', color='r')
    line_2g_free = Line2D([0], [0], label=r'DUNE $2\gamma$ (Background-free)', ls='dotted', color='r')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([line_1g, line_1g_cut, line_1g_free, line_2g, line_2g_cut, line_2g_free])
    #handles.extend([line_1g_free, line_2g_free])
    plt.legend(handles=handles, loc="lower left", framealpha=1, fontsize=11)

    # Read in existing limits.
    beam = np.genfromtxt('../sharedata/existing_limits/beamv2.txt', delimiter=',')
    eeinva = np.genfromtxt('../DUNE/data/existing_limits/eeinva.txt')
    lep = np.genfromtxt('../DUNE/data/existing_limits/lep_v2.txt', delimiter=",")
    nomad = np.genfromtxt('../DUNE/data/existing_limits/nomad.txt')

    # Astrophyiscal limits
    hbstars_new = np.genfromtxt("../DUNE/data/existing_limits/hbstars_new.txt", delimiter=",")
    sn1987a = np.genfromtxt("../sharedata/existing_limits/sn1987a_updated.txt", delimiter=",")

    # Plot astrophysical limits
    astro_color = 'silver'
    plt.fill_between(hbstars_new[:,0], hbstars_new[:,1], y2=1.0, color=astro_color, alpha=0.5)
    plt.fill(sn1987a[:,0], sn1987a[:,1], color=astro_color, alpha=0.5)

    # Plot DUNE dump mode
    #plt.plot(joined_masses_tgtls*1e6, joined_limits_tgtls*1e3, color="gray")
    #plt.plot(limits_dune_dump[:,0], limits_dune_dump[:,1], color='gray')

    # plot DUNE GAr
    #plt.contour(GAR_MA, GAR_G, GAR_CHI2.transpose(), levels=[2.3], colors=["orange"])


    # Plot lab limits
    lab_color = 'rosybrown'
    cosw = 0.7771
    plt.fill_between(lep[:,0], (1/cosw)*lep[:,1], y2=1.0, color='teal', alpha=1.0)
    plt.fill_between(beam[:,0], beam[:,1], y2=1.0, color='rosybrown', edgecolor='black', alpha=1.0)
    plt.fill(np.hstack((eeinva[:,0], np.min(eeinva[:,0]))), np.hstack((eeinva[:,1], np.max(eeinva[:,1]))),
            color='navy', alpha=1.0)
    plt.fill(np.hstack((nomad[:,0], np.min(nomad[:,0]))), np.hstack((nomad[:,1], np.max(nomad[:,1]))),
            color='tan', alpha=1.0)

    # explosion energy
    plt.fill([2e5, 1e7, 1e7, 2e5], [2e-6, 2e-6, 5e-5, 5e-5], edgecolor="khaki", facecolor='none', hatch="////", alpha=1.0, zorder=0)


    # Plot QCD lines
    # g_ag/GeV = slope * ma/GeV
    hq_slope = 1.005
    hq_int1 = 1.58e-3
    hq_int2 = 1.64e-7

    def hq_axion_coupling(ma, bp="1"):
        if bp == "1":
            hq_int = hq_int1
        elif bp == "2":
            hq_int = hq_int2
        return 10**(hq_slope * np.log10(1e9*ma)) + hq_int

    masses_continuous = np.logspace(2, 9, 100)
    plt.plot(masses_continuous, hq_axion_coupling(masses_continuous, bp="1"), color='dimgray', ls='dashed')
    plt.plot(masses_continuous, hq_axion_coupling(masses_continuous, bp="2"), color='dimgray', ls='dotted')


    text_fs = 12
    #plt.text(2, 2e-4, "NOMAD", fontsize=text_fs)
    plt.text(2e6, 2e-7, "SN1987a", fontsize=text_fs)
    plt.text(2e4, 1e-5, "HB Stars", fontsize=text_fs)
    plt.text(6e4, 5e-4, r'$e^+e^-\rightarrow inv.+\gamma$', fontsize=text_fs, color='w')
    #plt.text(4e8, 0.009, "LEP", fontsize=text_fs)
    plt.text(2e6, 5e-5, "Beam Dumps", fontsize=text_fs)
    plt.text(4e5, 2.5e-6, "SN\nExplosion\nEnergy", rotation=0.0, fontsize=text_fs, color="sienna")

    #plt.text(2.4e6, 4.3e-8, "DUNE Dump-mode (3 months)", rotation=-27.0, color='gray', fontsize=text_fs)
    #plt.text(3e6, 1.5e-7, "DUNE GAr (7 years)", rotation=-27.0, color='orange', fontsize=text_fs)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((1e4,1e9))
    plt.ylim(2e-8,2.0e-3)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('$m_a$ [eV]', fontsize=15)
    plt.ylabel('$g_{a\gamma}$ [GeV$^{-1}$]', fontsize=15)

    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.show()



def plot_total_sensitivity(save_file_2g, save_file_1g):
    # read in the calculated dune sens for DECAYS
    dune_dat = np.genfromtxt(save_file_2g)
    dune_ma = dune_dat[:,0]*1e6
    dune_g = dune_dat[:,1]*1e3
    dune_ll = 2*abs(max(dune_dat[:,2]) - dune_dat[:,2])*1.18
    dune_ll_post_cut = 2*(max(dune_dat[:,5]) - dune_dat[:,5]*100)
    dune_ll_decay_free = dune_dat[:,-1]*1.6  # compensating for ad-hoc flux increase

    # subtract off the maxima mass by mass
    unique_masses = np.unique(dune_ma)
    for m in unique_masses:
        dune_ll[dune_ma == m] = 2*(max(dune_dat[:,2][dune_ma == m]) - dune_dat[:,2][dune_ma == m])
        dune_ll_post_cut[dune_ma == m] = 2*(max(dune_dat[:,5][dune_ma == m]) - dune_dat[:,5][dune_ma == m])


    # read in the calculated dune sens for SCATTERING
    dune_scatter_dat = np.genfromtxt(save_file_1g)
    #dune_scatter_dat = dune_scatter_dat[dune_scatter_dat[:,0] < max(dune_scatter_dat[:,0])]  # for incomplete scan
    dune_ma_scatter = dune_scatter_dat[:,0]*1e6
    dune_g_scatter = dune_scatter_dat[:,1]*1e3
    dune_ll_scatter = 2*abs(max(dune_scatter_dat[:,2]) - dune_scatter_dat[:,2])
    dune_ll_scatter_acut = 2*abs(max(dune_scatter_dat[:,3]) - dune_scatter_dat[:,3])
    dune_ll_scatter_free = dune_scatter_dat[:,-1]

    unique_masses = np.unique(dune_ma_scatter)
    for m in unique_masses:
        dune_ll_scatter[dune_ma_scatter == m] = 2*(max(dune_scatter_dat[:,2][dune_ma_scatter == m]) - dune_scatter_dat[:,2][dune_ma_scatter == m])
        dune_ll_scatter_acut[dune_ma_scatter == m] = 2*(max(dune_scatter_dat[:,3][dune_ma_scatter == m]) - dune_scatter_dat[:,3][dune_ma_scatter == m])


    # plot contours for DUNE sensitivity
    DUNE_MA, DUNE_G = np.meshgrid(np.unique(dune_ma),np.unique(dune_g))
    DUNE_CHI2_POSTCUT = np.reshape(dune_ll_post_cut, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))

    DUNE_SCATTER_MA, DUNE_SCATTER_G = np.meshgrid(np.unique(dune_ma_scatter),np.unique(dune_g_scatter))
    DUNE_SCATTER_CHI2_POSTCUT = np.reshape(dune_ll_scatter_acut, (np.unique(dune_ma_scatter).shape[0],np.unique(dune_g_scatter).shape[0]))

    # Plot 2g limit at 90% CL
    DUNE_CL = 4.61

    fig, ax = plt.subplots(figsize=(8,6))    

    c_combined = plt.contour(DUNE_SCATTER_MA, DUNE_SCATTER_G, 5.0*DUNE_CHI2_POSTCUT.transpose() + DUNE_SCATTER_CHI2_POSTCUT.transpose(),
                             levels=[DUNE_CL], colors=["k"], linestyles='solid', alpha=0.0)  # do not draw, smooth below


    # Get the outermost contour level of each contour set
    outer_contour = c_combined.collections[-1]

    # Extract the paths of the outermost contour
    paths = outer_contour.get_paths()

    # Apply Savitzky-Golay filter to smooth the contour
    for path in paths:
        vertices = path.vertices
        x = vertices[:, 0]
        y = vertices[:, 1]

        # Apply Savitzky-Golay filter
        x_smooth = savgol_filter(x, window_length=11, polyorder=3)
        y_smooth = savgol_filter(y, window_length=11, polyorder=3)

        # Plot the smoothed contour
        ax.plot(x_smooth, y_smooth, color='b')


    # set legends for contours
    line_1g = Line2D([0], [0], label=r'DUNE $1\gamma$', ls='solid', color='royalblue')
    line_1g_cut = Line2D([0], [0], label=r'DUNE $1\gamma$ (after cuts)', ls='dashed', color='royalblue')
    line_1g_free = Line2D([0], [0], label=r'DUNE $1\gamma$ (Background-Free)', ls='dotted', color='royalblue')
    line_2g = Line2D([0], [0], label=r'DUNE $2\gamma$', ls='solid', color='r')
    line_2g_cut = Line2D([0], [0], label=r'DUNE $2\gamma$ (after cuts)', ls='dashed', color='r')
    line_2g_free = Line2D([0], [0], label=r'DUNE $2\gamma$ (Background-free)', ls='dotted', color='r')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([line_1g, line_1g_cut, line_1g_free, line_2g, line_2g_cut, line_2g_free])
    #handles.extend([line_1g_free, line_2g_free])
    #plt.legend(handles=handles, loc="lower left", framealpha=1, fontsize=11)

    # Read in existing limits.
    beam = np.genfromtxt('../sharedata/existing_limits/beamv2.txt', delimiter=',')
    eeinva = np.genfromtxt('../DUNE/data/existing_limits/eeinva.txt')
    lep = np.genfromtxt('../DUNE/data/existing_limits/lep_v2.txt', delimiter=",")
    nomad = np.genfromtxt('../DUNE/data/existing_limits/nomad.txt')

    # Astrophyiscal limits
    hbstars_new = np.genfromtxt("../DUNE/data/existing_limits/hbstars_new.txt", delimiter=",")
    sn1987a = np.genfromtxt("../sharedata/existing_limits/sn1987a_updated.txt", delimiter=",")

    # Plot astrophysical limits
    astro_color = 'silver'
    plt.fill_between(hbstars_new[:,0], hbstars_new[:,1], y2=1.0, color=astro_color, alpha=0.5)
    plt.fill(sn1987a[:,0], sn1987a[:,1], color=astro_color, alpha=0.5)

    # Plot DUNE dump mode
    #plt.plot(joined_masses_tgtls*1e6, joined_limits_tgtls*1e3, color="gray")
    #plt.plot(limits_dune_dump[:,0], limits_dune_dump[:,1], color='gray')

    # plot DUNE GAr
    c_gar = plt.contour(GAR_MA, GAR_G, GAR_CHI2.transpose(), levels=[2.3], colors=["orange"], alpha=0.0)
    outer_contour = c_gar.collections[-1]
    paths = outer_contour.get_paths()
    for path in paths:
        vertices = path.vertices
        x = vertices[:, 0]
        y = vertices[:, 1]
        x_smooth = savgol_filter(x, window_length=11, polyorder=3)
        y_smooth = savgol_filter(y, window_length=11, polyorder=3)
        ax.plot(x_smooth, y_smooth, color='orange')


    # Plot lab limits
    lab_color = 'rosybrown'
    cosw = 0.7771
    plt.fill_between(lep[:,0], (1/cosw)*lep[:,1], y2=1.0, color='teal', alpha=1.0)
    plt.fill_between(beam[:,0], beam[:,1], y2=1.0, color='rosybrown', edgecolor='black', alpha=1.0)
    plt.fill(np.hstack((eeinva[:,0], np.min(eeinva[:,0]))), np.hstack((eeinva[:,1], np.max(eeinva[:,1]))),
            color='navy', alpha=1.0)
    plt.fill(np.hstack((nomad[:,0], np.min(nomad[:,0]))), np.hstack((nomad[:,1], np.max(nomad[:,1]))),
            color='tan', alpha=1.0)

    # explosion energy
    plt.fill([2e5, 1e7, 1e7, 2e5], [2e-6, 2e-6, 5e-5, 5e-5], edgecolor="khaki", facecolor='none', hatch="////", alpha=1.0, zorder=0)


    # Plot QCD lines
    # g_ag/GeV = slope * ma/GeV
    hq_slope = 1.005
    hq_int1 = 1.58e-3
    hq_int2 = 1.64e-7

    def hq_axion_coupling(ma, bp="1"):
        if bp == "1":
            hq_int = hq_int1
        elif bp == "2":
            hq_int = hq_int2
        return 10**(hq_slope * np.log10(1e9*ma)) + hq_int

    masses_continuous = np.logspace(2, 9, 100)
    plt.plot(masses_continuous, hq_axion_coupling(masses_continuous, bp="1"), color='dimgray', ls='dashed')
    plt.plot(masses_continuous, hq_axion_coupling(masses_continuous, bp="2"), color='dimgray', ls='dotted')


    text_fs = 12
    #plt.text(2, 2e-4, "NOMAD", fontsize=text_fs)
    plt.text(8e5, 2e-7, "SN1987a", fontsize=text_fs)
    plt.text(2e4, 1e-5, "HB Stars", fontsize=text_fs)
    plt.text(6e4, 5e-4, r'$e^+e^-\rightarrow inv.+\gamma$', fontsize=text_fs, color='w')
    #plt.text(4e8, 0.009, "LEP", fontsize=text_fs)
    plt.text(2e6, 5e-5, "Beam Dumps", fontsize=text_fs)
    plt.text(4e5, 2.5e-6, "SN\nExplosion\nEnergy", rotation=0.0, fontsize=text_fs, color="sienna")

    #plt.text(2.4e6, 4.3e-8, "DUNE Dump-mode (3 months)", rotation=-27.0, color='gray', fontsize=text_fs)
    plt.text(3e6, 1.5e-7, "DUNE GAr (7 years)", rotation=-27.0, color='orange', fontsize=text_fs)
    plt.text(2.4e6, 2.3e-7, "DUNE LAr post-cuts (7 years)", rotation=-28.0, color='b', fontsize=text_fs)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((1e4,1e9))
    plt.ylim(2e-8,2.0e-3)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('$m_a$ [eV]', fontsize=15)
    plt.ylabel('$g_{a\gamma}$ [GeV$^{-1}$]', fontsize=15)

    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.show()


def cgamma(ma):
    return (-1.92 + ma**2 / (3*(ma**2 - M_PI0**2)) + (8/9)*(ma**2 - (4/9)*M_PI0**2)/(ma**2 - M_ETA**2) \
            + (7/9)*(ma**2 - (16/9)*M_PI0**2)/(ma**2 - M_ETA_PRIME**2))


from scipy.interpolate import griddata

def plot_gluon_dominance_sensitivity(save_file_2g, save_file_1g):
    # read in the calculated dune sens for DECAYS
    dune_dat = np.genfromtxt(save_file_2g)
    dune_ma = dune_dat[:,0]*1e-3
    dune_g = dune_dat[:,1]*1e3
    dune_events = dune_dat[:,-1]

    dune_dat_s = np.genfromtxt(save_file_1g)
    dune_ma_s = dune_dat_s[:,0]*1e-3
    dune_g_s = dune_dat_s[:,1]*1e3
    dune_events_s = dune_dat_s[:,-1]

    # convert to fa
    dune_1over_fa = np.nan_to_num(np.power(np.abs(cgamma(dune_ma)) * ALPHA / (2*np.pi) / dune_g, -1))  # fa in GeV
    fa_inv_grid = np.linspace(np.log10(min(dune_1over_fa)), np.log10(max(dune_1over_fa)), 100)
    ma_grid = np.linspace(np.log10(min(dune_ma)), np.log10(max(dune_ma)), 100)
    ma_grid, fa_inv_grid = np.meshgrid(ma_grid, fa_inv_grid)
    events_grid = griddata(points=np.array([np.log10(dune_ma), np.log10(dune_1over_fa)]).transpose(), values=dune_events,
                           xi=(ma_grid, fa_inv_grid), method='nearest')
    
    dune_1over_fa_s = np.nan_to_num(np.power(np.abs(cgamma(dune_ma_s)) * ALPHA / (2*np.pi) / dune_g_s, -1))  # fa in GeV
    fa_inv_grid_s = np.linspace(np.log10(min(dune_1over_fa_s)), np.log10(max(dune_1over_fa_s)), 100)
    ma_grid_s = np.linspace(np.log10(min(dune_ma_s)), np.log10(max(dune_ma_s)), 100)
    ma_grid_s, fa_inv_grid_s = np.meshgrid(ma_grid_s, fa_inv_grid_s)
    events_grid_s = griddata(points=np.array([np.log10(dune_ma_s), np.log10(dune_1over_fa_s)]).transpose(), values=dune_events_s,
                           xi=(ma_grid_s, fa_inv_grid_s), method='nearest')




    fig, ax = plt.subplots(figsize=(8,6)) 

    plt.contour(10**ma_grid, 10**fa_inv_grid, events_grid, levels=[3.0], colors=["r"], linestyles='dotted')
    plt.contour(10**ma_grid_s, 10**fa_inv_grid_s, events_grid_s, levels=[3.0], colors=["royalblue"], linestyles='dotted')


    # set legends for contours
    line_1g_free = Line2D([0], [0], label=r'DUNE $1\gamma$', ls='dotted', color='royalblue')
    line_2g_free = Line2D([0], [0], label=r'DUNE $2\gamma$', ls='dashed', color='r')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([line_1g_free, line_2g_free])
    plt.legend(handles=handles, loc="upper right", framealpha=1, fontsize=11)

    # plot limits
    plt.fill_between(existing_const_KKL[:,0], 1/existing_const_KKL[:,1], y2=1.0, color='silver')
    plt.fill(sn_KKL[:,0]*1e-3, (4*np.pi**2)*sn_KKL[:,1], color='silver', alpha=0.2)

    # text labels
    plt.text(2e-3, 5e-6, "SN", fontsize=14)
    plt.text(5e-3, 3e-3, "Existing Lab Bounds", fontsize=14)
    plt.text(8e-1, 2e-6, "Gluon Dominance", fontsize=16)
    


    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((1e-3,1e1))
    plt.ylim(1e-6,1e-2)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(r'$m_a$ [GeV]', fontsize=16)
    plt.ylabel(r'$1/f_a$ [GeV$^{-1}$]', fontsize=16)

    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.show()


def main():
    #check_mass_cut_efficiency(500.0)
    
    #plot_sensitivity(save_file_2g="sensitivities/digamma_sensitivity_BKG_cutflow_20240316.txt",
     #                save_file_1g="sensitivities/singlephoton_sensitivity_BKG_cutflow_20240319.txt")
    
    #plot_total_sensitivity(save_file_2g="sensitivities/digamma_sensitivity_BKG_cutflow_20240316.txt",
    #                 save_file_1g="sensitivities/singlephoton_sensitivity_BKG_cutflow_20240319.txt")
    
    plot_gluon_dominance_sensitivity(save_file_2g="sensitivities/digamma_sensitivity_BKG_cutflow_20240316.txt",
                     save_file_1g="sensitivities/singlephoton_sensitivity_BKG_cutflow_20240319.txt")

if __name__ == "__main__":
    main()
