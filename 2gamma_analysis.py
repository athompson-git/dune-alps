from signal_generators import *

import matplotlib.pyplot as plt
from matplotlib.pylab import rc
from matplotlib.colors import LogNorm
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



##################### SIGNAL TABLES #####################
alp_mass_list = [0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0,500.0,1000.0,2000.0]
alp_coupling_list = [[1e-9,5e-9,1e-8,5e-8,1e-7],[1e-9,5e-9,1e-8,5e-8,1e-7],  # 0.1, 0.2
                     [1e-9,5e-9,1e-8,5e-8,1e-7],[1e-10,5e-10,1e-9,5e-9,1e-8],  # 0.5, 1.0
                     [1e-10,5e-10,1e-9,5e-9,1e-8],[5e-11,1e-10,5e-10,1e-9,5e-9,1e-8],  # 2.0, 5.0
                     [5e-11,1e-10,5e-10,1e-9],  # 10.0, 20.0
                     ]



# GENERATE SIGNALS
# low coupling edge
g1e_10MeV, g2e_10MeV, g1_theta_10MeV, g2_theta_10MeV, \
    m2_10MeV, etotal_10MeV, dtheta_10MeV, wgts_10MeV = generate_alp_events_2gamma(ma=10.0, g=1.8e-9)
g1e_1MeV, g2e_1MeV, g1_theta_1MeV, g2_theta_1MeV, \
    m2_1MeV, etotal_1MeV, dtheta_1MeV, wgts_1MeV = generate_alp_events_2gamma(ma=1.0, g=1.75e-8)
g1e_100MeV, g2e_100MeV, g1_theta_100MeV, g2_theta_100MeV, \
    m2_100MeV, etotal_100MeV, dtheta_100MeV, wgts_100MeV = generate_alp_events_2gamma(ma=100.0, g=4.4e-10)

# high-coupling edge
#g1e_10MeV, g2e_10MeV, g1_theta_10MeV, g2_theta_10MeV, \
#    m2_10MeV, etotal_10MeV, dtheta_10MeV, wgts_10MeV = generate_alp_events_2gamma(ma=50.0, g=5e-8)
#g1e_1MeV, g2e_1MeV, g1_theta_1MeV, g2_theta_1MeV, \
#    m2_1MeV, etotal_1MeV, dtheta_1MeV, wgts_1MeV = generate_alp_events_2gamma(ma=100.0, g=1.1e-8)
#g1e_100MeV, g2e_100MeV, g1_theta_100MeV, g2_theta_100MeV, \
#    m2_100MeV, etotal_100MeV, dtheta_100MeV, wgts_100MeV = generate_alp_events_2gamma(ma=200.0, g=2e-9)



print("No. events = ", np.sum(wgts_1MeV))
print("No. events = ", np.sum(wgts_10MeV))
print("No. events = ", np.sum(wgts_100MeV))

# Output list of 4-vectors to file


##################### SIGNAL ONLY PLOTS #####################
angle_bins = np.logspace(-5, DUNE_SOLID_ANGLE, 50)
energy_bins = np.linspace(30.0e-3, 100, 50)  # GeV
delta_angle_bins = np.linspace(0.0, 1.0, 60)
rad_to_deg = 180.0/np.pi

plt.hist(1e-3*g1e_10MeV, weights=wgts_10MeV, bins=energy_bins, histtype='step')
plt.ylabel("Events")
plt.xlabel(r"$E_{\gamma,1}$ [GeV]")
plt.xlim((energy_bins[0], energy_bins[-1]))
plt.show()
plt.close()


plt.hist(1e-3*g2e_10MeV, weights=wgts_10MeV, bins=energy_bins, histtype='step')
plt.ylabel("Events")
plt.xlabel(r"$E_{\gamma,2}$ [GeV]")
plt.xlim((energy_bins[0], energy_bins[-1]))
plt.show()
plt.close()


plt.hist(1e-3*etotal_10MeV, weights=wgts_10MeV, bins=energy_bins, histtype='step')
plt.ylabel("Events")
plt.xlabel(r"$E_{\gamma}$ [GeV]")
plt.xlim((energy_bins[0], energy_bins[-1]))
plt.show()
plt.close()


plt.hist(g1_theta_10MeV, weights=wgts_10MeV, bins=angle_bins, histtype='step')
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Events")
plt.xlabel(r"$\theta_{\gamma,1}$ [rad]")
plt.show()
plt.close()


plt.hist(g2_theta_10MeV, weights=wgts_10MeV, bins=angle_bins, histtype='step')
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Events")
plt.xlabel(r"$\theta_{\gamma,2}$ [rad]")
plt.show()
plt.close()


plt.hist(m2_10MeV, weights=wgts_10MeV, bins=100, histtype='step')
plt.ylabel("Events")
plt.xlabel(r"$m(\gamma,\gamma)$ [MeV]")
plt.show()
plt.close()


plt.hist2d(1e-3*g2e_10MeV, g1_theta_10MeV, weights=wgts_10MeV, bins=[energy_bins, angle_bins], norm=LogNorm())
plt.colorbar()
plt.ylabel(r"$\theta_\gamma$ [rad]")
plt.xlabel(r"$E_{\gamma}$ [GeV]")
plt.show()
plt.close()















##################### READ IN BKG DATA #####################

from dune_backgrounds import *
#p0_2g_nue_1 etc.

# construct invariant mass
inv_mass_2g_nue = sqrt((p0_2g_nue_1 + p0_2g_nue_2)**2 - (p1_2g_nue_1 + p1_2g_nue_2)**2 - (p2_2g_nue_1 + p2_2g_nue_2)**2 - (p3_2g_nue_1 + p3_2g_nue_2)**2)
inv_mass_2g_numu = sqrt((p0_2g_numu_1 + p0_2g_numu_2)**2 - (p1_2g_numu_1 + p1_2g_numu_2)**2 - (p2_2g_numu_1 + p2_2g_numu_2)**2 - (p3_2g_numu_1 + p3_2g_numu_2)**2)
inv_mass_2g_nuebar = sqrt((p0_2g_nuebar_1 + p0_2g_nuebar_2)**2 - (p1_2g_nuebar_1 + p1_2g_nuebar_2)**2 - (p2_2g_nuebar_1 + p2_2g_nuebar_2)**2 - (p3_2g_nuebar_1 + p3_2g_nuebar_2)**2)
inv_mass_2g_numubar = sqrt((p0_2g_numubar_1 + p0_2g_numubar_2)**2 - (p1_2g_numubar_1 + p1_2g_numubar_2)**2 - (p2_2g_numubar_1 + p2_2g_numubar_2)**2 - (p3_2g_numubar_1 + p3_2g_numubar_2)**2)

# construct dthetas
bkg_2g_nue = Background2Particle(data_file_name="data/2gamma/2gamma_nue_4vectors_DUNE_bkg.txt")
bkg_2g_numu = Background2Particle(data_file_name="data/2gamma/2gamma_numu_4vectors_DUNE_bkg.txt")
bkg_2g_numubar = Background2Particle(data_file_name="data/2gamma/2gamma_numubar_4vectors_DUNE_bkg.txt")
bkg_2g_nuebar = Background2Particle(data_file_name="data/2gamma/2gamma_nuebar_4vectors_DUNE_bkg.txt")


##################### SIGNAL VS BACKGROUND PLOTS #####################


# rescale alp weights to match the plot scale
#wgts_100MeV = 25*wgts_100MeV/np.sum(wgts_100MeV)
#wgts_10MeV = 25*wgts_10MeV/np.sum(wgts_10MeV)
#wgts_1MeV = 25*wgts_1MeV/np.sum(wgts_1MeV)
bkg_energy_bins = np.linspace(30.0e-3, 15, 40)  # GeV

# Total energy
plt.hist([1e-3*(p0_2g_nuebar_1 + p0_2g_nuebar_2), 1e-3*(p0_2g_numubar_1 + p0_2g_numubar_2), 1e-3*(p0_2g_numu_1 + p0_2g_numu_2), 1e-3*(p0_2g_nue_1 + p0_2g_nue_2)],
         weights=[weights1_nuebar_2g, weights1_numubar_2g, weights1_numu_2g, weights1_nue_2g],
         bins=bkg_energy_bins, label=[r"$\bar{\nu_e}(2\gamma)$", r"$\bar{\nu_\mu}(2\gamma)$", r"$\nu_\mu(2\gamma)$", r"$\nu_e(2\gamma)$"],
         stacked=True, histtype='stepfilled', color=[COLOR_NUEBAR, COLOR_NUMUBAR, COLOR_NUMU, COLOR_NUE], alpha=0.5)
plt.hist(1e-3*etotal_1MeV, weights=wgts_1MeV, bins=bkg_energy_bins,
            label="$m_a = 50$ MeV", color='b', histtype='step')
plt.hist(1e-3*etotal_10MeV, weights=wgts_10MeV, bins=bkg_energy_bins,
            label="$m_a = 100$ MeV", color='r', histtype='step')
plt.hist(1e-3*etotal_100MeV, weights=wgts_100MeV, bins=bkg_energy_bins,
            label="$m_a = 200$ MeV", color='forestgreen', histtype='step')
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$E_\gamma$ [GeV]", fontsize=14)
plt.legend(fontsize=12)
plt.yscale('log')
plt.title("FHC", loc="right")
plt.xlim((bkg_energy_bins[0], bkg_energy_bins[-1]))
plt.ylim((1e-4,2e3))
plt.tight_layout()
plt.show()
plt.close()


# mass distribution
mass_bins = np.logspace(-3.2, 2, 50)
plt.hist([inv_mass_2g_nuebar*1e-3, inv_mass_2g_numubar*1e-3, inv_mass_2g_numu*1e-3, inv_mass_2g_nue*1e-3],
         weights=[weights1_nuebar_2g, weights1_numubar_2g, weights1_numu_2g, weights1_nue_2g],
         label=[r"$\bar{\nu_e}(2\gamma)$", r"$\bar{\nu_\mu}(2\gamma)$", r"$\nu_\mu(2\gamma)$", r"$\nu_e(2\gamma)$"],
         bins=mass_bins, stacked=True, histtype='stepfilled', color=[COLOR_NUEBAR, COLOR_NUMUBAR, COLOR_NUMU, COLOR_NUE], alpha=0.5)
plt.hist(1e-3*m2_1MeV, weights=wgts_1MeV, bins=mass_bins,
            label="$m_a = 50$ MeV", color='b', histtype='step')
plt.hist(1e-3*m2_10MeV, weights=wgts_10MeV, bins=mass_bins,
            label="$m_a = 100$ MeV", color='r', histtype='step')
plt.hist(1e-3*m2_100MeV, weights=wgts_100MeV, bins=mass_bins,
            label="$m_a = 200$ MeV", color='forestgreen', histtype='step')
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$m(\gamma\gamma)$ [GeV]", fontsize=14)
plt.legend(fontsize=14, loc="upper left")
plt.yscale('log')
plt.xscale('log')
plt.title("FHC", loc="right")
plt.xlim((mass_bins[0], mass_bins[-1]))
plt.tight_layout()
plt.show()
plt.close()


# ANGULAR distribution
dtheta_bins = np.linspace(1.0, 180.0, 100)
plt.hist([bkg_2g_nuebar.dtheta_deg, bkg_2g_numubar.dtheta_deg, bkg_2g_numu.dtheta_deg, bkg_2g_nue.dtheta_deg],
         weights=[weights1_nuebar_2g, weights1_numubar_2g, weights1_numu_2g, weights1_nue_2g],
         label=[r"$\bar{\nu_e}(2\gamma)$", r"$\bar{\nu_\mu}(2\gamma)$", r"$\nu_\mu(2\gamma)$", r"$\nu_e(2\gamma)$"],
         bins=dtheta_bins, stacked=True, histtype='stepfilled', color=[COLOR_NUEBAR, COLOR_NUMUBAR, COLOR_NUMU, COLOR_NUE], alpha=0.5)
plt.hist(dtheta_1MeV*180.0/pi, weights=wgts_1MeV, bins=dtheta_bins,
            label="$m_a = 50$ MeV", color='b', histtype='step')
plt.hist(dtheta_10MeV*180.0/pi, weights=wgts_10MeV, bins=dtheta_bins,
            label="$m_a = 100$ MeV", color='r', histtype='step')
plt.hist(dtheta_100MeV*180.0/pi, weights=wgts_100MeV, bins=dtheta_bins,
            label="$m_a = 200$ MeV", color='forestgreen', histtype='step')
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$\Delta\theta(\gamma\gamma)$ [deg]", fontsize=14)
plt.legend(fontsize=14, loc="upper right")
plt.yscale('log')
#plt.xscale('log')
plt.title("FHC", loc="right")
plt.xlim((dtheta_bins[0], dtheta_bins[-1]))
plt.ylim((1e-2, 1e3))
plt.tight_layout()
plt.show()
plt.close()




# background 2d dists




##################### MAKE CUTS, REPORT EFFICIENCIES #####################

# ditheta cut
rad_cut = 40.0 * pi/180.0
wgts_1MeV_theta_cut = wgts_1MeV[dtheta_1MeV<rad_cut]
wgts_10MeV_theta_cut = wgts_10MeV[dtheta_10MeV<rad_cut]
wgts_100MeV_theta_cut = wgts_100MeV[dtheta_100MeV<rad_cut]

print("ma = 1 MeV:")
print("--- Counts precut = ", np.sum(wgts_1MeV))
print("--- Counts post-angular cut = ", np.sum(wgts_1MeV_theta_cut))
print("--- Signal Efficiency = ", np.sum(wgts_1MeV_theta_cut)/np.sum(wgts_1MeV))

print("ma = 10 MeV:")
print("--- Counts precut = ", np.sum(wgts_10MeV))
print("--- Counts post-angular cut = ", np.sum(wgts_10MeV_theta_cut))
print("--- Signal Efficiency = ", np.sum(wgts_10MeV_theta_cut)/np.sum(wgts_10MeV))

print("ma = 100 MeV:")
print("--- Counts precut = ", np.sum(wgts_100MeV))
print("--- Counts post-angular cut = ", np.sum(wgts_100MeV_theta_cut))
print("--- Signal Efficiency = ", np.sum(wgts_100MeV_theta_cut)/np.sum(wgts_100MeV))

print("Backgrounds")
print("--- background total precut = ", np.sum(weights1_nue_2g)+np.sum(weights1_nuebar_2g)+np.sum(weights1_numu_2g)+np.sum(weights1_numubar_2g))

print("bkg eff (numubar 2g): {} events, {} accepted ".format(np.sum(weights1_numubar_2g), np.sum(weights1_numubar_2g[bkg_2g_numubar.dtheta_deg<40])/np.sum(weights1_numubar_2g)))
print("bkg eff (nuebar 2g): {} events, {} accepted ".format(np.sum(weights1_nuebar_2g), np.sum(weights1_nuebar_2g[bkg_2g_nuebar.dtheta_deg<40])/np.sum(weights1_nuebar_2g)))
print("bkg eff (numu 2g): {} events, {} accepted ".format(np.sum(weights1_numu_2g), np.sum(weights1_numu_2g[bkg_2g_numu.dtheta_deg<40])/np.sum(weights1_numu_2g)))
print("bkg eff (nue 2g): {} events, {} accepted ".format(np.sum(weights1_nue_2g), np.sum(weights1_nue_2g[bkg_2g_nue.dtheta_deg<40])/np.sum(weights1_nue_2g)))

# mass cut: 100 MeV
nue_wgts_post_mass_cut = weights1_nue_2g*(bkg_2g_nue.dtheta_deg<40)*(bkg_2g_nue.inv_mass<110)*(bkg_2g_nue.inv_mass>90)
numu_wgts_post_mass_cut = weights1_numu_2g*(bkg_2g_numu.dtheta_deg<40)*(bkg_2g_numu.inv_mass<110)*(bkg_2g_numu.inv_mass>90)
nuebar_wgts_post_mass_cut = weights1_nuebar_2g*(bkg_2g_nuebar.dtheta_deg<40)*(bkg_2g_nuebar.inv_mass<110)*(bkg_2g_nuebar.inv_mass>90)
numubar_wgts_post_mass_cut = weights1_numubar_2g*(bkg_2g_numubar.dtheta_deg<40)*(bkg_2g_numubar.inv_mass<110)*(bkg_2g_numubar.inv_mass>90)



print("bkg eff (numubar 2g): {} events, {} accepted, {} pc".format(np.sum(weights1_numubar_2g), np.sum(numubar_wgts_post_mass_cut), np.sum(numubar_wgts_post_mass_cut)/np.sum(weights1_numubar_2g)))
print("bkg eff (nuebar 2g): {} events, {} accepted {} pc".format(np.sum(weights1_nuebar_2g), np.sum(nuebar_wgts_post_mass_cut), np.sum(nuebar_wgts_post_mass_cut)/np.sum(weights1_nuebar_2g)))
print("bkg eff (numu 2g): {} events, {} accepted {} pc".format(np.sum(weights1_numu_2g), np.sum(numu_wgts_post_mass_cut), np.sum(numu_wgts_post_mass_cut)/np.sum(weights1_numu_2g)))
print("bkg eff (nue 2g): {} events, {} accepted {} pc".format(np.sum(weights1_nue_2g), np.sum(nue_wgts_post_mass_cut), np.sum(nue_wgts_post_mass_cut)/np.sum(weights1_nue_2g)))






##################### PLOT AFTER CUTS #####################


# Total energy
plt.hist([1e-3*p0_ep_nue[nue_ep_thetas<2e-2], 1e-3*p0_g_nue[nue_g_thetas<2e-2], 1e-3*p0_em_nue[nue_em_thetas<2e-2]], 
         weights=[E2GAMMA_MISID*weights1_nue_ep[nue_ep_thetas<2e-2], weights1_nue_g[nue_g_thetas<2e-2], E2GAMMA_MISID*weights1_nue_em[nue_em_thetas<2e-2]],
         bins=bkg_energy_bins, label=[r"$\nu(1e^+ 0p) \times 5$\% mis-ID", r"$\nu(1\gamma 0p)$", r"$\nu(1e^- 0p) \times 5$\% mis-ID"],
         stacked=True, histtype='stepfilled', color=['dimgray', 'rosybrown', 'teal'], alpha=0.5)
plt.hist(1e-3*etotal_1MeV_post_cut, weights=wgts_1MeV_theta_cut, bins=bkg_energy_bins,
            label="Collinear ALP Decays ($m_a = 1$ MeV)", color='b', histtype='step')
plt.hist(1e-3*etotal_10MeV_post_cut, weights=wgts_10MeV_theta_cut, bins=bkg_energy_bins,
            label="Collinear ALP Decays ($m_a = 10$ MeV)", color='r', histtype='step')
plt.hist(1e-3*etotal_100MeV_post_cut, weights=wgts_100MeV_theta_cut, bins=bkg_energy_bins,
            label="Collinear ALP Decays ($m_a = 100$ MeV)", color='forestgreen', histtype='step')
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$E_\gamma$ [GeV]", fontsize=14)
plt.legend(fontsize=12)
plt.yscale('log')
plt.title("FHC", loc="right")
plt.xlim((bkg_energy_bins[0], bkg_energy_bins[-1]))
plt.ylim((1e-4,2e3))
plt.tight_layout()
plt.show()
plt.close()

