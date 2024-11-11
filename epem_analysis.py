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
years = 3.5
days = years*365  # days of exposure
det_area = 7.0*5.0  # cross-sectional det area
det_thresh = 1.0  # energy threshold [MeV]
det_length=3.0
det_dist=574


# Declare detector and target materials
dune_nd = Material("Ar")
dune_target = Material("C")

# Get neutrino flux per 3.5 years
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
    return numu_diff_flux(enu) * numu_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0 * det_length / 1.0e6

def nue_diff_flux(enu):
    return 10**np.interp(enu, nue_fhc_diff_flux[:,0], np.log10(nue_fhc_diff_flux[:,1]))

def nue_events(enu):
    return nue_diff_flux(enu) * numu_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0 * det_length / 1.0e6

def numubar_diff_flux(enu):
    return 10**np.interp(enu, numubar_fhc_diff_flux[:,0], np.log10(numubar_fhc_diff_flux[:,1]))

def numubar_events(enu):
    return numubar_diff_flux(enu) * numubar_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0 * det_length / 1.0e6

def nuebar_diff_flux(enu):
    return 10**np.interp(enu, nuebar_fhc_diff_flux[:,0], np.log10(nuebar_fhc_diff_flux[:,1]))

def nuebar_events(enu):
    return nuebar_diff_flux(enu) * numubar_xs(enu) * (dune_nd.density * AVOGADRO / (dune_nd.n[0] + dune_nd.z[0])) * 100.0 * det_length / 1.0e6



# Import backgrounds
epem_bkg = np.genfromtxt("data/1em1ep/epem_4vectors_DUNE_bkg.txt")
epem_bkg *= 1.0e3
p0_em = epem_bkg[:,0] + M_E
p1_em = epem_bkg[:,1]
p2_em = epem_bkg[:,2]
p3_em = epem_bkg[:,3]
p0_ep = epem_bkg[:,4] + M_E
p1_ep = epem_bkg[:,5]
p2_ep = epem_bkg[:,6]
p3_ep = epem_bkg[:,7]
inv_mass_bkg = np.sqrt((p0_em + p0_ep)**2 - (p1_em + p1_ep)**2 - (p2_em + p2_ep)**2 - (p3_em + p3_ep)**2)
nue_sep_angle = arccos((p1_em*p1_ep + p2_em*p2_ep + p3_em*p3_ep)/(sqrt(p1_em**2 + p2_em**2 + p3_em**2)*sqrt(p1_ep**2 + p2_ep**2 + p3_ep**2)))
nue_total_energy = p0_em + p0_ep
nue_em_angle =  arccos((p3_em)/(sqrt(p1_em**2 + p2_em**2 + p3_em**2)))
nue_ep_angle =  arccos((p3_ep)/(sqrt(p1_ep**2 + p2_ep**2 + p3_ep**2)))
nue_weights = nue_events(p0_em + p0_ep)
print("# of Nue events = ", np.sum(nue_weights))

epem_bkg_numu = np.genfromtxt("data/1em1ep/epem_numu_4vectors_DUNE_bkg.txt")
epem_bkg_numu *= 1.0e3
p0_em_numu = epem_bkg_numu[:,0] + M_E
p1_em_numu = epem_bkg_numu[:,1]
p2_em_numu = epem_bkg_numu[:,2]
p3_em_numu = epem_bkg_numu[:,3]
p0_ep_numu = epem_bkg_numu[:,4] + M_E
p1_ep_numu = epem_bkg_numu[:,5]
p2_ep_numu = epem_bkg_numu[:,6]
p3_ep_numu = epem_bkg_numu[:,7]
numu_total_energy = p0_em_numu + p0_ep_numu
inv_mass_numu_bkg = np.sqrt((p0_em_numu + p0_ep_numu)**2 - (p1_em_numu + p1_ep_numu)**2 - (p2_em_numu + p2_ep_numu)**2 - (p3_em_numu + p3_ep_numu)**2)
numu_sep_angle = arccos((p1_em_numu*p1_ep_numu + p2_em_numu*p2_ep_numu + p3_em_numu*p3_ep_numu)/(sqrt(p1_em_numu**2 + p2_em_numu**2 + p3_em_numu**2)*sqrt(p1_ep_numu**2 + p2_ep_numu**2 + p3_ep_numu**2)))
numu_em_angle =  arccos((p3_em_numu)/(sqrt(p1_em_numu**2 + p2_em_numu**2 + p3_em_numu**2)))
numu_ep_angle =  arccos((p3_ep_numu)/(sqrt(p1_ep_numu**2 + p2_ep_numu**2 + p3_ep_numu**2)))
numu_weights = numu_events(p0_em_numu + p0_ep_numu)
print("# of Numu events = ", np.sum(numu_weights))

epem_bkg_numubar = np.genfromtxt("data/1em1ep/epem_numubar_4vectors_DUNE_bkg.txt")
epem_bkg_numubar *= 1.0e3
p0_em_numubar = epem_bkg_numubar[:,0] + M_E
p1_em_numubar = epem_bkg_numubar[:,1]
p2_em_numubar = epem_bkg_numubar[:,2]
p3_em_numubar = epem_bkg_numubar[:,3]
p0_ep_numubar = epem_bkg_numubar[:,4] + M_E
p1_ep_numubar = epem_bkg_numubar[:,5]
p2_ep_numubar = epem_bkg_numubar[:,6]
p3_ep_numubar = epem_bkg_numubar[:,7]
numubar_total_energy = p0_em_numubar + p0_ep_numubar
inv_mass_numubar_bkg = np.sqrt((p0_em_numubar + p0_ep_numubar)**2 - (p1_em_numubar + p1_ep_numubar)**2 - (p2_em_numubar + p2_ep_numubar)**2 - (p3_em_numubar + p3_ep_numubar)**2)
numubar_sep_angle = arccos((p1_em_numubar*p1_ep_numubar + p2_em_numubar*p2_ep_numubar + p3_em_numubar*p3_ep_numubar)/(sqrt(p1_em_numubar**2 + p2_em_numubar**2 + p3_em_numubar**2)*sqrt(p1_ep_numubar**2 + p2_ep_numubar**2 + p3_ep_numubar**2)))
numubar_em_angle =  arccos((p3_em_numubar)/(sqrt(p1_em_numubar**2 + p2_em_numubar**2 + p3_em_numubar**2)))
numubar_ep_angle =  arccos((p3_ep_numubar)/(sqrt(p1_ep_numubar**2 + p2_ep_numubar**2 + p3_ep_numubar**2)))
numubar_weights = numubar_events(p0_em_numubar + p0_ep_numubar)
print("# of Numubar events = ", np.sum(numubar_weights))

epem_bkg_nuebar = np.genfromtxt("data/1em1ep/epem_nuebar_4vectors_DUNE_bkg.txt")
epem_bkg_nuebar *= 1.0e3
p0_em_nuebar = epem_bkg_nuebar[:,0] + M_E
p1_em_nuebar = epem_bkg_nuebar[:,1]
p2_em_nuebar = epem_bkg_nuebar[:,2]
p3_em_nuebar = epem_bkg_nuebar[:,3]
p0_ep_nuebar = epem_bkg_nuebar[:,4] + M_E
p1_ep_nuebar = epem_bkg_nuebar[:,5]
p2_ep_nuebar = epem_bkg_nuebar[:,6]
p3_ep_nuebar = epem_bkg_nuebar[:,7]
nuebar_total_energy = p0_em_nuebar + p0_ep_nuebar
nuebar_sep_angle = arccos((p1_em_nuebar*p1_ep_nuebar + p2_em_nuebar*p2_ep_nuebar + p3_em_nuebar*p3_ep_nuebar)/(sqrt(p1_em_nuebar**2 + p2_em_nuebar**2 + p3_em_nuebar**2)*sqrt(p1_ep_nuebar**2 + p2_ep_nuebar**2 + p3_ep_nuebar**2)))
inv_mass_nuebar_bkg = np.sqrt((p0_em_nuebar + p0_ep_nuebar)**2 - (p1_em_nuebar + p1_ep_nuebar)**2 - (p2_em_nuebar + p2_ep_nuebar)**2 - (p3_em_nuebar + p3_ep_nuebar)**2)
nuebar_em_angle =  arccos((p3_em_nuebar)/(sqrt(p1_em_nuebar**2 + p2_em_nuebar**2 + p3_em_nuebar**2)))
nuebar_ep_angle =  arccos((p3_ep_nuebar)/(sqrt(p1_ep_nuebar**2 + p2_ep_nuebar**2 + p3_ep_nuebar**2)))
nuebar_weights = nuebar_events(p0_em_nuebar + p0_ep_nuebar)
print("# of Nuebar events = ", np.sum(nuebar_weights))


# Apply separation angle cuts
nue_weights *= ((180/np.pi) * nue_sep_angle > 1.0)
numu_weights *= ((180/np.pi) * numu_sep_angle > 1.0)
nuebar_weights *= ((180/np.pi) * nuebar_sep_angle > 1.0)
numubar_weights *= ((180/np.pi) * numubar_sep_angle > 1.0)


# Totals
total_p0_em = np.append(np.append(np.append(p0_em_numu, p0_em), p0_em_numubar), p0_em_nuebar)
total_p0_ep = np.append(np.append(np.append(p0_ep_numu, p0_ep), p0_ep_numubar), p0_ep_nuebar)

total_angles_em = np.append(np.append(np.append(numu_em_angle, nue_em_angle), numubar_em_angle), nuebar_em_angle)
total_angles_ep = np.append(np.append(np.append(numu_ep_angle, nue_ep_angle), numubar_ep_angle), nuebar_ep_angle)

total_weights = np.append(np.append(np.append(numu_weights, nue_weights), numubar_weights), nuebar_weights)




print("Numu acceptance = {}/1e6 = {}".format(epem_bkg_numu.shape[0],epem_bkg_numu.shape[0]/1e6))
print("Nue acceptance = {}/1e6 = {}".format(epem_bkg.shape[0],epem_bkg.shape[0]/1e6))
print("Numubar acceptance = {}/1e6 = {}".format(epem_bkg_numubar.shape[0],epem_bkg_numubar.shape[0]/1e6))
print("Nuebar acceptance = {}/1e6 = {}".format(epem_bkg_nuebar.shape[0],epem_bkg_nuebar.shape[0]/1e6))


# BACKGROUNDS

# TOTAL ENERGY
energy_bins = np.linspace(1.0, 10000.0, 50)
plt.hist([numu_total_energy, nue_total_energy, nuebar_total_energy, numubar_total_energy],
         weights=[numu_weights, nue_weights, nuebar_weights, numubar_weights], bins=energy_bins,
         label=[r"$\nu_\mu(e^+ e^-)$", r"$\nu_e(e^+ e^-)$", r"$\overline{\nu}_e(e^+ e^-)$", r"$\overline{\nu}_\mu(e^+ e^-)$"],
         stacked=True, histtype='stepfilled', color=['tan', 'silver', 'sienna', 'cornflowerblue'])
plt.xlim((0.0, 10000.0))
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$E_+ + E_-$ [MeV]", fontsize=14)
plt.title("FHC", loc="right")
plt.legend(fontsize=14)
#plt.yscale('log')
plt.tight_layout()
plt.show()
plt.close()

# SEPARATING ANGLE
sep_angle_bins = np.linspace(0.0, 50.0, 51)
rad2deg = 180.0/pi
plt.hist([numu_sep_angle*rad2deg, nue_sep_angle*rad2deg, nuebar_sep_angle*rad2deg, numubar_sep_angle*rad2deg],
         weights=[numu_weights, nue_weights, nuebar_weights, numubar_weights], bins=sep_angle_bins,
         label=[r"$\nu_\mu(e^+ e^-)$", r"$\nu_e(e^+ e^-)$", r"$\overline{\nu}_e(e^+ e^-)$", r"$\overline{\nu}_\mu(e^+ e^-)$"],
         stacked=True, histtype='stepfilled', color=['tan', 'silver', 'sienna', 'cornflowerblue'])
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.xlabel(r"$\Delta \theta$ [rad]", fontsize=14)
plt.xlim((0.0, 50.0))
plt.title("FHC", loc="right")
plt.legend(fontsize=14)
#plt.yscale('log')
plt.tight_layout()
plt.show()
plt.close()

# INDIVIDUAL ENERGIES 2D PLOT
energy_bins_2d = np.logspace(0.0, 4, 25)
plt.hist2d(total_p0_em, total_p0_ep, weights=total_weights, bins=[energy_bins_2d, energy_bins_2d])
plt.ylim((energy_bins_2d[0], energy_bins_2d[-1]))
plt.xlim((energy_bins_2d[0], energy_bins_2d[-1]))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$E_-$ [MeV]", fontsize=14)
plt.ylabel(r"$E_+$ [MeV]", fontsize=14)
plt.title("FHC", loc="right")
plt.colorbar()
plt.tight_layout()
plt.show()

# INDIVIDUAL ANGLES 2D PLOT
angle_bins_2d = np.linspace(0.0, 50.0, 25)
plt.hist2d(total_angles_em*rad2deg, total_angles_ep*rad2deg, weights=total_weights, bins=[angle_bins_2d, angle_bins_2d])
plt.ylim((angle_bins_2d[0], angle_bins_2d[-1]))
plt.xlim((angle_bins_2d[0], angle_bins_2d[-1]))
plt.xlabel(r"$\theta_-$ [deg]", fontsize=14)
plt.ylabel(r"$\theta_+$ [deg]", fontsize=14)
plt.title("FHC", loc="right")
plt.colorbar()
plt.tight_layout()
plt.show()

# INVARIANT MASS
inv_mass_bins = np.logspace(0, 3, 50)
plt.hist([inv_mass_numu_bkg, inv_mass_bkg, inv_mass_nuebar_bkg, inv_mass_numubar_bkg],
         weights=[numu_weights, nue_weights, nuebar_weights, numubar_weights], histtype='stepfilled',
         color=['tan', 'silver', 'sienna', 'cornflowerblue'], bins=inv_mass_bins,
         label=[r"$\nu_\mu(e^+ e^-)$", r"$\nu_e(e^+ e^-)$", r"$\overline{\nu}_e(e^+ e^-)$", r"$\overline{\nu}_\mu(e^+ e^-)$"],
         stacked=True)
plt.xlim((inv_mass_bins[0], inv_mass_bins[-1]))
#plt.yscale('log')
#plt.ylim((1e-4, 1e2))
plt.xscale('log')
plt.legend(fontsize=14)
plt.title("FHC", loc="right")
plt.xlabel(r"$m(e^+, e^-)$ [MeV]", fontsize=14)
plt.ylabel(r"Events / 3.5 years", fontsize=14)
plt.tight_layout()
plt.show()







# Import signal fluxes
positron_flux = np.genfromtxt("../DUNE/data/epem_flux/positron_DIFF_flux_dPhidE_20210621_TargetSim_QGSP_BIC_AllHP_POT1E6.txt")
positron_flux[:,1] *= pot_per_year/365/S_PER_DAY/pot_per_sample  # s^-1
electron_flux = np.genfromtxt("../DUNE/data/epem_flux/electron_DIFF_flux_dPhidE_20210621_TargetSim_QGSP_BIC_AllHP_POT1E6.txt")
electron_flux[:,1] *= pot_per_year/365/S_PER_DAY/pot_per_sample  # s^-1
photon_flux = np.genfromtxt("../DUNE/data/photon_flux/geant4_flux_DUNE.txt", delimiter=",")
photon_flux[:,1] *= pot_per_year/pot_per_sample/S_PER_DAY/365  # converts to /s




# Generate signal fluxes
TEST_MA = 100.0  # MeV
TEST_GAE = 1e-9

compton_flux = FluxComptonIsotropic(photon_flux=photon_flux, target=dune_target,
                                det_dist=det_dist, det_length=det_length, det_area=det_area,
                                axion_mass=TEST_MA, axion_coupling=TEST_GAE, n_samples=2000)
brem_flux = FluxBremIsotropic(electron_flux, positron_flux, target=dune_target,
                                det_dist=det_dist, det_length=det_length, det_area=det_area,
                                axion_mass=TEST_MA, axion_coupling=TEST_GAE, n_samples=2000, is_isotropic=False)
resonant_flux = FluxResonanceIsotropic(positron_flux, target=dune_target,
                                        det_dist=det_dist, det_length=det_length, det_area=det_area,
                                        axion_mass=TEST_MA, axion_coupling=TEST_GAE, n_samples=10000, is_isotropic=False)
assoc_flux = FluxPairAnnihilationIsotropic(positron_flux, dune_target, det_dist=det_dist,
                                            det_length=det_length, det_area=det_area, axion_mass=TEST_MA, axion_coupling=TEST_GAE,
                                            n_samples=2000, is_isotropic=False)
compton_flux.simulate()
brem_flux.simulate()
resonant_flux.simulate()
assoc_flux.simulate()

compton_flux.propagate()
brem_flux.propagate()
resonant_flux.propagate()
assoc_flux.propagate()

print(np.sum(assoc_flux.decay_axion_weight))


alp_flux_energies = np.concatenate([compton_flux.axion_energy, brem_flux.axion_energy,
                              resonant_flux.axion_energy, assoc_flux.axion_energy]).flatten()
alp_flux_weights = np.concatenate([compton_flux.decay_axion_weight, brem_flux.decay_axion_weight,
                            resonant_flux.decay_axion_weight, assoc_flux.decay_axion_weight]).flatten()
print(alp_flux_energies.shape)

alp_flux_bins = np.linspace(0, 500, 200)
plt.hist(alp_flux_energies, weights=alp_flux_weights, bins=alp_flux_bins, color='k', histtype='step')
plt.hist(compton_flux.axion_energy, weights=compton_flux.decay_axion_weight, bins=alp_flux_bins, color='b', histtype='step')
plt.hist(resonant_flux.axion_energy, weights=resonant_flux.decay_axion_weight, bins=alp_flux_bins, color='r', histtype='step')
plt.hist(brem_flux.axion_energy, weights=brem_flux.decay_axion_weight, bins=alp_flux_bins, color='g', histtype='step')
plt.hist(assoc_flux.axion_energy, weights=assoc_flux.decay_axion_weight, bins=alp_flux_bins, color='pink', histtype='step')

plt.yscale('log')
plt.xscale('log')
plt.show()



# Generate the decays with smearing
def decay_gen(energy, ma, mf=M_E):
    p1 = LorentzVector(energy, 0.0, 0.0, -np.sqrt(energy**2 - ma**2))
    mc = Decay2Body(p1, mf, mf, n_samples=1)
    mc.decay()

    wgts = mc.weights[0]
    fv2 = mc.p2_lab_4vectors[0]
    fv1 = mc.p1_lab_4vectors[0]

    # smear energies
    fv1_res = (0.02+0.15/sqrt(1e3*fv1.energy()))
    fv1_new_energy = norm.rvs(loc=fv1.energy(), scale=fv1_res*fv1.energy())
    fv1_new_p = np.sqrt(fv1_new_energy**2 - M_E**2)

    fv2_res = (0.02+0.15/sqrt(1e3*fv2.energy()))
    fv2_new_energy = norm.rvs(loc=fv2.energy(), scale=fv2_res*fv2.energy())
    fv2_new_p = np.sqrt(fv2_new_energy**2 - M_E**2)

    # Apply cuts: e+/e-/gamma < 30 MeV
    if fv1_new_energy < 30.0:
        wgts *= 0.0
    if fv2_new_energy < 30.0:
        wgts *= 0.0
    
    

    # smear angles: 1 degree angular resolution
    fv1_new_phi = np.sign(fv1.phi()) * abs(norm.rvs(loc=fv1.phi(), scale=pi/180.0))
    fv1_new_theta = np.sign(fv1.theta())*abs(norm.rvs(loc=fv1.theta(), scale=pi/180.0))
    fv2_new_phi = np.sign(fv2.phi())*abs(norm.rvs(loc=fv2.phi(), scale=pi/180.0))
    fv2_new_theta = np.sign(fv2.theta())*abs(norm.rvs(loc=fv2.theta(), scale=pi/180.0))

    fv1_smear = LorentzVector(fv1_new_energy,
                              fv1_new_p*cos(fv1_new_phi)*sin(fv1_new_theta),
                              fv1_new_p*sin(fv1_new_phi)*sin(fv1_new_theta),
                              fv1_new_p*cos(fv1_new_theta))
    fv2_smear = LorentzVector(fv2_new_energy,
                              fv2_new_p*cos(fv2_new_phi)*sin(fv2_new_theta),
                              fv2_new_p*sin(fv2_new_phi)*sin(fv2_new_theta),
                              fv2_new_p*cos(fv2_new_theta))

    v1 = fv1_smear.get_3momentum()
    v2 = fv2_smear.get_3momentum()

    # Apply separation angle cut
    sep_angle = arccos(v1*v2 / abs(v1.mag()*v2.mag()))
    if sep_angle * 180.0/np.pi >= 1.0:
        wgts *= 0.0

    return fv1_smear, fv2_smear, sep_angle, wgts


inv_masses = []
signal_weights = []
fv1_thetas = []
fv2_thetas = []
fv2_phis = []
fv1_phis = []
total_energies = []
sep_angles = []


for i in range(alp_flux_weights.shape[0]):
    fv1, fv2, deltaTheta, wgt = decay_gen(alp_flux_energies[i], TEST_MA)

    pair_fv = fv1 + fv2
    mass = pair_fv.mass()
    inv_masses.append(mass)
    signal_weights.append(alp_flux_weights[i]*wgt)
    fv1_thetas.append(fv1.theta())
    fv1_phis.append(fv1.phi())
    fv2_thetas.append(fv2.theta())
    fv2_phis.append(fv2.phi())
    total_energies.append(pair_fv.energy())
    sep_angles.append(deltaTheta)

signal_weights = np.array(signal_weights) * 1.0 / np.sum(signal_weights)
fv1_phis = np.array(fv1_phis)
fv1_thetas = np.array(fv1_thetas)
fv2_phis = np.array(fv2_phis)
fv2_thetas = np.array(fv2_thetas)
sep_angles = np.array(sep_angles)


# INVARIANT MASS DIST
inv_mass_bins = np.logspace(0, 3, 50)
plt.hist(inv_masses, weights=signal_weights*0.5/np.sum(signal_weights), histtype='step', bins=inv_mass_bins, label=r"$a \to e^+ e^-$, $m_a = 10$ MeV")
plt.hist([inv_mass_numu_bkg, inv_mass_bkg, inv_mass_nuebar_bkg, inv_mass_numubar_bkg],
         weights=[numu_weights, nue_weights, nuebar_weights, numubar_weights], histtype='stepfilled',
         color=['tan', 'silver', 'sienna', 'cornflowerblue'], bins=inv_mass_bins,
         label=[r"$\nu_\mu(e^+ e^-)$", r"$\nu_e(e^+ e^-)$", r"$\overline{\nu}_e(e^+ e^-)$", r"$\overline{\nu}_\mu(e^+ e^-)$"],
         stacked=True)
plt.xlim((inv_mass_bins[0], inv_mass_bins[-1]))
plt.yscale('log')
plt.ylim((1e-4, 1e2))
plt.xscale('log')
plt.legend(fontsize=14)
plt.xlabel(r"$m(e^+, e^-)$ [MeV]", fontsize=14)
plt.ylabel("a.u.", fontsize=14)
plt.show()

# SEPARATING ANGLE
plt.hist(sep_angles, weights=signal_weights*0.5/np.sum(signal_weights), histtype='step', bins=sep_angle_bins, label=r"$a \to e^+ e^-$, $m_a = 10$ MeV")
plt.hist([nue_sep_angle*rad2deg, numu_sep_angle*rad2deg, nuebar_sep_angle*rad2deg, numubar_sep_angle*rad2deg],
         weights=[nue_weights, numu_weights, nuebar_weights, numubar_weights], bins=sep_angle_bins,
         label=[r"$\nu_\mu(e^+ e^-)$", r"$\nu_e(e^+ e^-)$", r"$\overline{\nu}_e(e^+ e^-)$", r"$\overline{\nu}_\mu(e^+ e^-)$"],
         stacked=True, histtype='stepfilled', color=['tan', 'silver', 'sienna', 'cornflowerblue'])
plt.ylabel(r"Events / 7 years", fontsize=14)
plt.xlabel(r"$\Delta \theta$ [deg]", fontsize=14)
plt.legend(fontsize=14)
#plt.yscale('log')
plt.tight_layout()
plt.show()
plt.close()

# TOTAL ENERGY DIST
plt.hist(total_energies, weights=signal_weights*0.5/np.sum(signal_weights), histtype='step', density=True, bins=energy_bins, label=r"$e^+ e^-$ Signal")
#plt.hist([nue_total_energy, numu_total_energy, nuebar_total_energy, numubar_total_energy],
#         weights=[nue_weights, numu_weights, nuebar_weights, numubar_weights], bins=energy_bins,
#         label=[r"$\nu_e(e^+ e^-)$", r"$\nu_\mu(e^+ e^-)$", r"$\overline{\nu}_e(e^+ e^-)$", r"$\overline{\nu}_\mu(e^+ e^-)$"],
#         stacked=True, histtype='stepfilled', color=['tan', 'silver', 'sienna', 'cornflowerblue'])
plt.ylabel(r"Events / 7 years", fontsize=14)
plt.xlabel(r"$E_+ + E_-$ [MeV]", fontsize=14)
plt.legend(fontsize=14)
plt.yscale('log')
plt.tight_layout()
plt.show()
plt.close()