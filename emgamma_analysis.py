import sys
sys.path.append("../")

from alplib.fluxes import *
from alplib.generators import *


import matplotlib.pyplot as plt
from matplotlib.pylab import rc
from matplotlib.colors import LogNorm
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from scipy.stats import norm


# Flux constants
pot_per_year = 1.1e21
pot_per_sample = 1e6

# detector constants
det_mass = 50000  # kg
det_am = 37.211e3  # mass of target atom in MeV
det_ntargets = det_mass * MEV_PER_KG / det_am
det_z = 18  # atomic number
years = 3.5  # FHC
days = years*365  # days of exposure
det_area = 7.0*5.0  # cross-sectional det area
det_thresh = 1.0  # energy threshold [MeV]
det_length=3.0  # meters
det_dist=574


# Declare detector and target materials
dune_nd = Material("Ar")
dune_target = Material("C")

# Get neutrino flux per 7 years
numu_fhc_diff_flux = np.genfromtxt("data/numu_flux_DUNE_FHC_per1GeV_m2_POT.txt")
numu_fhc_diff_flux[:,1] *= 2 * years*pot_per_year * det_area  # convert into per MeV and multiply by 200 MeV bin size

nue_fhc_diff_flux = np.genfromtxt("data/nue_flux_DUNE_FHC_per_1GeV_m2_POT.txt")
nue_fhc_diff_flux[:,1] *= 2 * years*pot_per_year * det_area  # convert into per MeV

numubar_fhc_diff_flux = np.genfromtxt("data/numubar_flux_DUNE_FHC_per1GeV_m2_POT.txt")
numubar_fhc_diff_flux[:,1] *= 2 * years*pot_per_year * det_area  # convert into per MeV

nuebar_fhc_diff_flux = np.genfromtxt("data/nuebar_flux_DUNE_FHC_per1GeV_m2_POT.txt")
nuebar_fhc_diff_flux[:,1] *= 2 * years*pot_per_year * det_area  # convert into per MeV

numu_xs_dat = np.genfromtxt("data/numu_xs_dividedByEnergy_in_cm-2.txt")
numubar_xs_dat = np.genfromtxt("data/numubar_xs_dividedByEnergy_in_cm-2.txt")

def numu_xs(enu):
    return (enu/1000.0) * 10**np.interp(log10(enu/1000.0), log10(numu_xs_dat[:,0]), log10(numu_xs_dat[:,1]))

def numubar_xs(enu):
    return (enu/1000.0) * 10**np.interp(log10(enu/1000.0), log10(numubar_xs_dat[:,0]), log10(numubar_xs_dat[:,1]))

def numu_diff_flux(enu):
    return 10**np.interp(enu, numu_fhc_diff_flux[:,0], np.log10(numu_fhc_diff_flux[:,1]))

def numu_events(enu):
    return numu_diff_flux(enu) * numu_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0*det_length / 1.0e6

def nue_diff_flux(enu):
    return 10**np.interp(enu, nue_fhc_diff_flux[:,0], np.log10(nue_fhc_diff_flux[:,1]))

def nue_events(enu):
    return nue_diff_flux(enu) * numu_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0*det_length / 1.0e6

def numubar_diff_flux(enu):
    return 10**np.interp(enu, numubar_fhc_diff_flux[:,0], np.log10(numubar_fhc_diff_flux[:,1]))

def numubar_events(enu):
    return numubar_diff_flux(enu) * numubar_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0*det_length / 1.0e6

def nuebar_diff_flux(enu):
    return 10**np.interp(enu, nuebar_fhc_diff_flux[:,0], np.log10(nuebar_fhc_diff_flux[:,1]))

def nuebar_events(enu):
    return nuebar_diff_flux(enu) * numubar_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0*det_length / 1.0e6



# Check total number of nu events:
print("Number of numu = ", 1e6*np.sum(numu_events(numu_fhc_diff_flux[:,0])))
print("Number of nue = ", 1e6*np.sum(nue_events(nue_fhc_diff_flux[:,0])))
print("Number of nuebar = ", 1e6*np.sum(nuebar_events(nuebar_fhc_diff_flux[:,0])))
print("Number of numubar = ", 1e6*np.sum(numubar_events(numubar_fhc_diff_flux[:,0])))
print("Total number = ", 1e6*np.sum(numu_events(numu_fhc_diff_flux[:,0])) + 1e6*np.sum(nue_events(nue_fhc_diff_flux[:,0])) \
      + 1e6*np.sum(nuebar_events(nuebar_fhc_diff_flux[:,0])) + 1e6*np.sum(numubar_events(numubar_fhc_diff_flux[:,0])))



# Import backgrounds
def get_egamma_array(datfile, flavor="numu"):
    bkg = np.genfromtxt(datfile)
    bkg *= 1.0e3
    p0_em = bkg[:,0] + M_E
    p1_em = bkg[:,1]
    p2_em = bkg[:,2]
    p3_em = bkg[:,3]
    p0_gamm = bkg[:,4]
    p1_gamm = bkg[:,5]
    p2_gamm = bkg[:,6]
    p3_gamm = bkg[:,7]
    e_total = p0_em + p0_gamm
    sep_angle = arccos((p1_em*p1_gamm + p2_em*p2_gamm + p3_em*p3_gamm)/(sqrt(p1_em**2 + p2_em**2 + p3_em**2)*sqrt(p1_gamm**2 + p2_gamm**2 + p3_gamm**2)))
    inv_mass = np.sqrt((p0_em + p0_gamm)**2 - (p1_em + p1_gamm)**2 - (p2_em + p2_gamm)**2 - (p3_em + p3_gamm)**2)
    if flavor == "numu":
        weights = numu_events(p0_em + p0_gamm)
    elif flavor == "nue":
        weights = nue_events(p0_em + p0_gamm)
    elif flavor == "numubar":
        weights = numubar_events(p0_em + p0_gamm)
    elif flavor == "nuebar":
        weights = nuebar_events(p0_em + p0_gamm)
    return p0_em, p1_em, p2_em, p3_em, p0_gamm, p1_gamm, p2_gamm, p3_gamm, e_total, sep_angle, inv_mass, weights



p0_em_nue, p1_em_nue, p2_em_nue, p3_em_nue, p0_gamm1_nue, \
    p1_gamm1_nue, p2_gamm1_nue, p3_gamm1_nue, \
        e_total1_nue, sep_angle1_nue, inv_mass1_nue, weights1_nue = get_egamma_array("data/1g1e0p/emgamma_nue_4vectors_DUNE_bkg.txt", flavor="nue")

p0_ep_nue, p1_ep_nue, p2_ep_nue, p3_ep_nue, p0_gamm2_nue, \
    p1_gamm2_nue, p2_gamm2_nue, p3_gamm2_nue, \
        e_total2_nue, sep_angle2_nue, inv_mass2_nue, weights2_nue = get_egamma_array("data/1g1e0p/epgamma_nue_4vectors_DUNE_bkg.txt", flavor="nue")

p0_em_numu, p1_em_numu, p2_em_numu, p3_em_numu, p0_gamm1_numu, \
    p1_gamm1_numu, p2_gamm1_numu, p3_gamm1_numu, \
        e_total1_numu, sep_angle1_numu, inv_mass1_numu, weights1_numu = get_egamma_array("data/1g1e0p/emgamma_numu_4vectors_DUNE_bkg.txt", flavor="numu")

p0_ep_numu, p1_ep_numu, p2_ep_numu, p3_ep_numu, p0_gamm2_numu, \
    p1_gamm2_numu, p2_gamm2_numu, p3_gamm2_numu, \
        e_total2_numu, sep_angle2_numu, inv_mass2_numu, weights2_numu = get_egamma_array("data/1g1e0p/epgamma_numu_4vectors_DUNE_bkg.txt", flavor="numu")

p0_em_nuebar, p1_em_nuebar, p2_em_nuebar, p3_em_nuebar, p0_gamm1_nuebar, \
    p1_gamm1_nuebar, p2_gamm1_nuebar, p3_gamm1_nuebar, \
         e_total1_nuebar, sep_angle1_nuebar, inv_mass1_nuebar, weights1_nuebar = get_egamma_array("data/1g1e0p/emgamma_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")

p0_ep_nuebar, p1_ep_nuebar, p2_ep_nuebar, p3_ep_nuebar, p0_gamm2_nuebar, \
    p1_gamm2_nuebar, p2_gamm2_nuebar, p3_gamm2_nuebar, \
         e_total2_nuebar, sep_angle2_nuebar, inv_mass2_nuebar, weights2_nuebar = get_egamma_array("data/1g1e0p/epgamma_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")

p0_em_numubar, p1_em_numubar, p2_em_numubar, p3_em_numubar, p0_gamm1_numubar, \
    p1_gamm1_numubar, p2_gamm1_numubar, p3_gamm1_numubar, \
         e_total1_numubar, sep_angle1_numubar, inv_mass1_numubar, weights1_numubar = get_egamma_array("data/1g1e0p/emgamma_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")

p0_ep_numubar, p1_ep_numubar, p2_ep_numubar, p3_ep_numubar, p0_gamm2_numubar, \
    p1_gamm2_numubar, p2_gamm2_numubar, p3_gamm2_numubar, \
         e_total2_numubar, sep_angle2_numubar, inv_mass2_numubar, weights2_numubar = get_egamma_array("data/1g1e0p/epgamma_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")



print("Number of SCALED nue (e- gamm)= ", np.sum(weights1_nue))
print("Number of SCALED nue (e- gamm)= ", np.sum(weights2_nue))
print("Number of SCALED numu (e- gamm)= ", np.sum(weights1_numu))
print("Number of SCALED numu (e- gamm)= ", np.sum(weights2_numu))
print("Number of SCALED numubar (e- gamm)= ", np.sum(weights1_numubar))
print("Number of SCALED numubar (e- gamm)= ", np.sum(weights2_numubar))
print("Number of SCALED nue (e- gamm)= ", np.sum(weights1_nuebar))
print("Number of SCALED nue (e- gamm)= ", np.sum(weights2_nuebar))


# Make background plots
energy_bins = np.linspace(1.0, 10000.0, 50)
plt.hist([e_total1_numu, e_total1_numubar, e_total1_nuebar, e_total1_nue],
         weights=[weights1_numu, weights1_numubar, weights1_nuebar, weights1_nue], bins=energy_bins,
         label=[r"$\nu_\mu(e^- \gamma)$", r"$\overline{\nu}_\mu(e^- \gamma)$", r"$\overline{\nu}_e(e^- \gamma)$", r"$\nu_e(e^- \gamma)$"],
         stacked=True, histtype='stepfilled', color=['tan', 'cornflowerblue', 'sienna', 'silver'])
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$E_- + E_\gamma$ [MeV]", fontsize=14)
plt.legend(fontsize=14)
#plt.yscale('log')
plt.title("FHC", loc="right")
plt.xlim((0.0, 10000.0))
plt.tight_layout()
plt.show()
plt.close()

plt.hist([e_total2_numu, e_total2_numubar, e_total2_nue, e_total2_nuebar],
         weights=[weights2_numu, weights2_numubar, weights2_nue, weights2_nuebar], bins=energy_bins,
         label=[r"$\nu_\mu(e^+ \gamma)$", r"$\overline{\nu}_\mu(e^+ \gamma)$", r"$\nu_e(e^+ \gamma)$", r"$\overline{\nu}_e(e^+ \gamma)$"],
         stacked=True, histtype='stepfilled', color=['tan', 'cornflowerblue', 'silver', 'sienna'])
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$E_+ + E_\gamma$ [MeV]", fontsize=14)
plt.legend(fontsize=14)
plt.xlim((0.0, 10000.0))
#plt.yscale('log')
plt.title("FHC", loc="right")
plt.tight_layout()
plt.show()
plt.close()

sep_angle_bins = np.linspace(0.0, 180.0, 51)
rad2deg = 180.0/pi
plt.hist([sep_angle1_numu*rad2deg, sep_angle1_numubar*rad2deg, sep_angle1_nuebar*rad2deg, sep_angle1_nue*rad2deg],
         weights=[weights1_numu, weights1_numubar, weights1_nuebar, weights1_nue], bins=sep_angle_bins,
         label=[r"$\nu_\mu(e^- \gamma)$", r"$\overline{\nu}_\mu(e^- \gamma)$", r"$\overline{\nu}_e(e^- \gamma)$", r"$\nu_e(e^- \gamma)$"],
         stacked=True, histtype='stepfilled', color=['tan', 'cornflowerblue', 'sienna', 'silver'])
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$\Delta \theta(e^-, \gamma)$ [rad]", fontsize=14)
plt.xlim((0.0, 180.0))
plt.legend(fontsize=14)
plt.title("FHC", loc="right")
#plt.yscale('log')
plt.tight_layout()
plt.show()
plt.close()

plt.hist([sep_angle2_numu*rad2deg, sep_angle2_numubar*rad2deg, sep_angle2_nue*rad2deg, sep_angle2_nuebar*rad2deg],
         weights=[weights2_numu, weights2_numubar, weights2_nue, weights2_nuebar], bins=sep_angle_bins,
         label=[r"$\nu_\mu(e^- \gamma)$", r"$\overline{\nu}_\mu(e^- \gamma)$", r"$\nu_e(e^- \gamma)$", r"$\overline{\nu}_e(e^- \gamma)$"],
         stacked=True, histtype='stepfilled', color=['tan', 'cornflowerblue', 'silver', 'sienna'])
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$\Delta \theta(e^+, \gamma)$ [rad]", fontsize=14)
plt.xlim((0.0, 180.0))
plt.legend(fontsize=14)
plt.title("FHC", loc="right")
#plt.yscale('log')
plt.tight_layout()
plt.show()
plt.close()








def compton_gen(energy, ma):
    p1 = LorentzVector(energy, 0.0, 0.0, np.sqrt(energy**2 - ma**2))
    p2 = LorentzVector(M_E, 0.0, 0.0, 0.0)
    mc = Scatter2to2MC(M2InverseCompton(ma, 18), p1, p2, n_samples=100000)
    mc.scatter_sim()
    
    wgts = mc.dsigma_dcos_cm_wgts
    gamma_p4 = mc.p3_lab_4vectors
    electron_p4 = mc.p4_lab_4vectors
    return gamma_p4, electron_p4, wgts