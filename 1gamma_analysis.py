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



g1e_10MeV, g2e_10MeV, g1_theta_10MeV, g2_theta_10MeV, m2_10MeV, etotal_10MeV, dtheta_10MeV, wgts_10MeV = \
    decay_alp_gen("signal_data/fluxes_photon/FLUX-DECAY_alp_photon_ma-10MeV_gag1e-9MeV-1.txt",
                        ALP_MASS=10.0, GAGAMMA=1e-9, resolved=False)
g1e_1MeV, g2e_1MeV, g1_theta_1MeV, g2_theta_1MeV, m2_1MeV, etotal_1MeV, dtheta_1MeV, wgts_1MeV = \
    decay_alp_gen("signal_data/fluxes_photon/FLUX-DECAY_alp_photon_ma-1MeV_gag1e-8MeV-1.txt",
                        ALP_MASS=1.0, GAGAMMA=1e-8, resolved=False)
g1e_100MeV, g2e_100MeV, g1_theta_100MeV, g2_theta_100MeV, m2_100MeV, etotal_100MeV, dtheta_100MeV, wgts_100MeV = \
    decay_alp_gen("signal_data/fluxes_photon/FLUX-DECAY_alp_photon_ma-100MeV_gag1e-10MeV-1.txt",
                        ALP_MASS=100.0, GAGAMMA=1e-10, resolved=False)



print("No. events = ", np.sum(wgts_1MeV))
print("No. events = ", np.sum(wgts_10MeV))
print("No. events = ", np.sum(wgts_100MeV))



print("Resolved fraction 1 MeV = {}".format(np.sum(dtheta_1MeV * 180.0/np.pi >= 1.0)/dtheta_1MeV.shape[0]))
print("Resolved fraction 10 MeV = {}".format(np.sum(dtheta_10MeV * 180.0/np.pi >= 1.0)/dtheta_10MeV.shape[0]))
print("Resolved fraction 100 MeV = {}".format(np.sum(dtheta_100MeV * 180.0/np.pi >= 1.0)/dtheta_100MeV.shape[0]))


# Output list of 4-vectors to file


##################### SIGNAL ONLY PLOTS #####################
angle_bins = np.logspace(-5, DUNE_SOLID_ANGLE, 50)
energy_bins = np.linspace(30.0e-3, 100, 50)  # GeV
delta_angle_bins = np.linspace(0.0, 5.0, 60)
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


plt.hist2d(1e-3*g2e_1MeV, g1_theta_1MeV, weights=wgts_1MeV, bins=[energy_bins, angle_bins], norm=LogNorm())
plt.colorbar()
plt.yscale('log')
plt.ylabel(r"$\theta_\gamma$ [rad]", fontsize=16)
plt.xlabel(r"$E_{\gamma}$ [GeV]", fontsize=16)
plt.title(r"$m_a = 1$ MeV", loc="right")
plt.show()
plt.close()

plt.hist2d(1e-3*g2e_10MeV, g1_theta_10MeV, weights=wgts_10MeV, bins=[energy_bins, angle_bins], norm=LogNorm())
plt.colorbar()
plt.yscale('log')
plt.ylabel(r"$\theta_\gamma$ [rad]", fontsize=16)
plt.xlabel(r"$E_{\gamma}$ [GeV]", fontsize=16)
plt.title(r"$m_a = 10$ MeV", loc="right")
plt.show()
plt.close()

plt.hist2d(1e-3*g2e_100MeV, g1_theta_100MeV, weights=wgts_100MeV, bins=[energy_bins, angle_bins], norm=LogNorm())
plt.colorbar()
plt.yscale('log')
plt.ylabel(r"$\theta_\gamma$ [rad]", fontsize=16)
plt.xlabel(r"$E_{\gamma}$ [GeV]", fontsize=16)
plt.title(r"$m_a = 100$ MeV", loc="right")
plt.show()
plt.close()


plt.hist(rad_to_deg*dtheta_1MeV, weights=wgts_1MeV, bins=delta_angle_bins, histtype='step', label="1 MeV", density=True)
plt.hist(rad_to_deg*dtheta_10MeV, weights=wgts_10MeV, bins=delta_angle_bins, histtype='step', label="10 MeV", density=True)
plt.hist(rad_to_deg*dtheta_100MeV, weights=wgts_100MeV, bins=delta_angle_bins, histtype='step', label="100 MeV", density=True)
plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r"$\Delta\theta_{\gamma\gamma}$ [rad]", fontsize=16)
plt.ylabel("a.u.", fontsize=16)
plt.legend()
plt.show()














##################### READ IN BKG DATA #####################




# Plot against Backgrounds

# Declare detector and target materials
dune_nd = Material("Ar")
dune_target = Material("C")

# Get neutrino flux per 7 years
numu_fhc_diff_flux = np.genfromtxt("data/numu_flux_DUNE_FHC_per1GeV_m2_POT.txt")
numu_fhc_diff_flux[:,1] *= 2 * EXPOSURE_YEARS*DUNE_POT_PER_YEAR*DUNE_AREA  # convert into per MeV and multiply by 200 MeV bin size

nue_fhc_diff_flux = np.genfromtxt("data/nue_flux_DUNE_FHC_per_1GeV_m2_POT.txt")
nue_fhc_diff_flux[:,1] *= 2 * EXPOSURE_YEARS*DUNE_POT_PER_YEAR*DUNE_AREA  # convert into per MeV

numubar_fhc_diff_flux = np.genfromtxt("data/numubar_flux_DUNE_FHC_per1GeV_m2_POT.txt")
numubar_fhc_diff_flux[:,1] *= 2 * EXPOSURE_YEARS*DUNE_POT_PER_YEAR*DUNE_AREA  # convert into per MeV

nuebar_fhc_diff_flux = np.genfromtxt("data/nuebar_flux_DUNE_FHC_per1GeV_m2_POT.txt")
nuebar_fhc_diff_flux[:,1] *= 2 * EXPOSURE_YEARS*DUNE_POT_PER_YEAR*DUNE_AREA  # convert into per MeV

numu_xs_dat = np.genfromtxt("data/numu_xs_dividedByEnergy_in_cm-2.txt")
numubar_xs_dat = np.genfromtxt("data/numubar_xs_dividedByEnergy_in_cm-2.txt")

def numu_xs(enu):
    return (enu/1000.0) * 10**np.interp(log10(enu/1000.0), log10(numu_xs_dat[:,0]), log10(numu_xs_dat[:,1]))

def numubar_xs(enu):
    return (enu/1000.0) * 10**np.interp(log10(enu/1000.0), log10(numubar_xs_dat[:,0]), log10(numubar_xs_dat[:,1]))

def numu_diff_flux(enu):
    return 10**np.interp(enu, numu_fhc_diff_flux[:,0], np.log10(numu_fhc_diff_flux[:,1]))

def numu_events(enu):
    return numu_diff_flux(enu) * numu_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0*DUNE_LENGTH / 1.0e7

def nue_diff_flux(enu):
    return 10**np.interp(enu, nue_fhc_diff_flux[:,0], np.log10(nue_fhc_diff_flux[:,1]))

def nue_events(enu):
    return nue_diff_flux(enu) * numu_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0*DUNE_LENGTH / 1.0e7

def numubar_diff_flux(enu):
    return 10**np.interp(enu, numubar_fhc_diff_flux[:,0], np.log10(numubar_fhc_diff_flux[:,1]))

def numubar_events(enu):
    return numubar_diff_flux(enu) * numubar_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0*DUNE_LENGTH / 1.0e7

def nuebar_diff_flux(enu):
    return 10**np.interp(enu, nuebar_fhc_diff_flux[:,0], np.log10(nuebar_fhc_diff_flux[:,1]))

def nuebar_events(enu):
    return nuebar_diff_flux(enu) * numubar_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0*DUNE_LENGTH / 1.0e7



# Check total number of nu events:
print("Number of numu = ", 1e6*np.sum(numu_events(numu_fhc_diff_flux[:,0])))
print("Number of nue = ", 1e6*np.sum(nue_events(nue_fhc_diff_flux[:,0])))
print("Number of nuebar = ", 1e6*np.sum(nuebar_events(nuebar_fhc_diff_flux[:,0])))
print("Number of numubar = ", 1e6*np.sum(numubar_events(numubar_fhc_diff_flux[:,0])))
print("Total number = ", 1e6*np.sum(numu_events(numu_fhc_diff_flux[:,0])) + 1e6*np.sum(nue_events(nue_fhc_diff_flux[:,0])) \
      + 1e6*np.sum(nuebar_events(nuebar_fhc_diff_flux[:,0])) + 1e6*np.sum(numubar_events(numubar_fhc_diff_flux[:,0])))



# Import backgrounds
from dune_backgrounds import *
def get_egamma_array(datfile, flavor="numu"):
    bkg = np.genfromtxt(datfile)
    bkg *= 1.0e3
    p0_em = bkg[:,0] + M_E
    p1_em = bkg[:,1]
    p2_em = bkg[:,2]
    p3_em = bkg[:,3]
    if flavor == "numu":
        weights = numu_events(p0_em)
    elif flavor == "nue":
        weights = nue_events(p0_em)
    elif flavor == "numubar":
        weights = numubar_events(p0_em)
    elif flavor == "nuebar":
        weights = nuebar_events(p0_em)
    return p0_em, p1_em, p2_em, p3_em, weights

p0_g_nue, p1_g_nue, p2_g_nue, p3_g_nue, weights1_nue_g = get_egamma_array("data/1g0p/1g0p_nue_4vectors_DUNE_bkg.txt", flavor="nue")
p0_em_nue, p1_em_nue, p2_em_nue, p3_em_nue, weights1_nue_em = get_egamma_array("data/1g0p/1em0p_nue_4vectors_DUNE_bkg.txt", flavor="nue")
p0_ep_nue, p1_ep_nue, p2_ep_nue, p3_ep_nue, weights1_nue_ep = get_egamma_array("data/1g0p/1ep0p_nue_4vectors_DUNE_bkg.txt", flavor="nue")


p0_g_numu, p1_g_numu, p2_g_numu, p3_g_numu, weights1_numu_g = get_1particle_array("data/1g0p/1gamma_numu_4vectors_DUNE_bkg.txt", flavor="numu")
p0_em_numu, p1_em_numu, p2_em_nue, p3_em_numu, weights1_numu_em = get_1particle_array("data/1g0p/1em_numu_4vectors_DUNE_bkg.txt", flavor="numu")
p0_ep_numu, p1_ep_numu, p2_ep_numu, p3_ep_numu, weights1_numu_ep = get_1particle_array("data/1g0p/1ep_numu_4vectors_DUNE_bkg.txt", flavor="numu")

p0_g_nuebar, p1_g_nuebar, p2_g_nuebar, p3_g_nuebar, weights1_nuebar_g = get_1particle_array("data/1g0p/1gamma_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")
p0_em_nuebar, p1_em_nuebar, p2_em_nuebar, p3_em_nuebar, weights1_nuebar_em = get_1particle_array("data/1g0p/1em_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")
p0_ep_nuebar, p1_ep_nuebar, p2_ep_nuebar, p3_ep_nuebar, weights1_nuebar_ep = get_1particle_array("data/1g0p/1ep_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")

p0_g_numubar, p1_g_numubar, p2_g_numubar, p3_g_numubar, weights1_numubar_g = get_1particle_array("data/1g0p/1gamma_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")
p0_em_numubar, p1_em_numubar, p2_em_numubar, p3_em_numubar, weights1_numubar_em = get_1particle_array("data/1g0p/1em_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")
p0_ep_numubar, p1_ep_numubar, p2_ep_numubar, p3_ep_numubar, weights1_numubar_ep = get_1particle_array("data/1g0p/1ep_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")

##################### SIGNAL VS BACKGROUND PLOTS #####################


nue_ep_thetas = arccos(p3_ep_nue/sqrt(p1_ep_nue**2 + p2_ep_nue**2 + p3_ep_nue**2))
nue_em_thetas = arccos(p3_em_nue/sqrt(p1_em_nue**2 + p2_em_nue**2 + p3_em_nue**2))
nue_g_thetas = arccos(p3_g_nue/sqrt(p1_g_nue**2 + p2_g_nue**2 + p3_g_nue**2))

# rescale alp weights to match the plot scale
wgts_100MeV = np.sum(E2GAMMA_MISID*weights1_nue_ep)*wgts_100MeV/np.sum(wgts_100MeV)
wgts_10MeV = np.sum(E2GAMMA_MISID*weights1_nue_ep)*wgts_10MeV/np.sum(wgts_10MeV)
wgts_1MeV = np.sum(E2GAMMA_MISID*weights1_nue_ep)*wgts_1MeV/np.sum(wgts_1MeV)
bkg_energy_bins = np.linspace(30.0e-3, 30, 60)  # GeV

# Total energy
plt.hist([1e-3*p0_ep_nue, 1e-3*p0_g_nue, 1e-3*p0_em_nue], weights=[E2GAMMA_MISID*weights1_nue_ep, weights1_nue_g, E2GAMMA_MISID*weights1_nue_em],
         bins=bkg_energy_bins, label=[r"$\nu(1e^+ 0p) \times 5$\% mis-ID", r"$\nu(1\gamma 0p)$", r"$\nu(1e^- 0p) \times 5$\% mis-ID"],
         stacked=True, histtype='stepfilled', color=['dimgray', 'rosybrown', 'teal'], alpha=0.5)
#plt.hist(1e-3*etotal_1MeV, weights=wgts_1MeV, bins=bkg_energy_bins,
#            label="Collinear ALP Decays ($m_a = 1$ MeV)", color='b', histtype='step')
#plt.hist(1e-3*etotal_10MeV, weights=wgts_10MeV, bins=bkg_energy_bins,
#            label="Collinear ALP Decays ($m_a = 10$ MeV)", color='r', histtype='step')
#plt.hist(1e-3*etotal_100MeV, weights=wgts_100MeV, bins=bkg_energy_bins,
#            label="Collinear ALP Decays ($m_a = 100$ MeV)", color='forestgreen', histtype='step')
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$E_\gamma$ [GeV]", fontsize=14)
plt.legend(fontsize=12)
plt.yscale('log')
#plt.title("FHC", loc="right")
plt.xlim((bkg_energy_bins[0], bkg_energy_bins[-1]))
plt.ylim((1e-4,2e3))
plt.tight_layout()
plt.show()
plt.close()


# Angular distribution wrt beamline
theta_bins = np.logspace(-4, np.log10(np.pi), 100)
plt.hist([nue_ep_thetas, nue_g_thetas, nue_em_thetas], weights=[weights1_nue_ep, weights1_nue_g, weights1_nue_em],
         bins=theta_bins, label=[r"$\nu(1e^+ 0p) \times 5$\% mis-ID", r"$\nu(1\gamma 0p)$", r"$\nu(1e^- 0p) \times 5$\% mis-ID"],
         stacked=True, histtype='stepfilled', color=['dimgray', 'rosybrown', 'teal'], alpha=0.5)
plt.hist(g1_theta_1MeV, weights=wgts_1MeV, bins=theta_bins,
         label="Collinear ALP Decays ($m_a = 1$ MeV)", color='b', histtype='step')
plt.hist(g1_theta_10MeV, weights=wgts_10MeV, bins=theta_bins,
         label="Collinear ALP Decays ($m_a = 10$ MeV)", color='r', histtype='step')
plt.hist(g1_theta_100MeV, weights=wgts_100MeV, bins=theta_bins,
         label="Collinear ALP Decays ($m_a = 100$ MeV)", color='forestgreen', histtype='step')
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$\theta$ [rad]", fontsize=14)
plt.legend(fontsize=12, loc="upper left")
plt.yscale('log')
plt.xscale('log')
plt.title("FHC", loc="right")
plt.xlim((1e-4, np.pi))
plt.tight_layout()
plt.show()
plt.close()




# background 2d dists




##################### MAKE CUTS, REPORT EFFICIENCIES #####################
wgts_1MeV_theta_cut = wgts_1MeV[g1_theta_1MeV<2e-2]
wgts_10MeV_theta_cut = wgts_10MeV[g1_theta_10MeV<2e-2]
wgts_100MeV_theta_cut = wgts_100MeV[g1_theta_100MeV<2e-2]

theta_1MeV_post_cut = g1_theta_1MeV[g1_theta_1MeV<2e-2]
theta_10MeV_post_cut = g1_theta_10MeV[g1_theta_10MeV<2e-2]
theta_100MeV_post_cut = g1_theta_100MeV[g1_theta_100MeV<2e-2]

etotal_1MeV_post_cut = etotal_1MeV[g1_theta_1MeV<2e-2]
etotal_10MeV_post_cut = etotal_10MeV[g1_theta_10MeV<2e-2]
etotal_100MeV_post_cut = etotal_100MeV[g1_theta_100MeV<2e-2]

print("Signal Efficiency 1 MeV = ", np.sum(wgts_1MeV_theta_cut)/np.sum(wgts_1MeV))
print("Signal Efficiency 10 MeV = ", np.sum(wgts_10MeV_theta_cut)/np.sum(wgts_10MeV))

print("bkg eff (nue true gam): {} events, {} accepted ".format(np.sum(weights1_nue_g), np.sum(weights1_nue_g[nue_g_thetas<2e-2])/np.sum(weights1_nue_g)))
print("bkg eff (nue ep): {} events, {} accepted ".format(np.sum(weights1_nue_ep), np.sum(weights1_nue_ep[nue_ep_thetas<2e-2])/np.sum(weights1_nue_ep)))
print("bkg eff (nue em): {} events, {} accepted ".format(np.sum(weights1_nue_em), np.sum(weights1_nue_em[nue_em_thetas<2e-2])/np.sum(weights1_nue_em)))









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

