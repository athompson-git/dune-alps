import sys
sys.path.append("../")

from alplib.fluxes import *
from alplib.generators import *
from alplib.cross_section_mc import *


import matplotlib.pyplot as plt

from signal_generators import PrimakoffFromBeam4Vectors


# detector constants
DUNE_MASS = 50000  # kg
DUNE_ATOMIC_MASS = 37.211e3  # mass of target atom in MeV
DUNE_NTARGETS = DUNE_MASS * MEV_PER_KG / DUNE_ATOMIC_MASS
LAR_Z = 18  # atomic number
EXPOSURE_YEARS = 3.5
EXPOSURE_DAYS = EXPOSURE_YEARS*365  # days of exposure
DUNE_AREA = 7.0*5.0  # cross-sectional det area
DUNE_THRESH = 1.0  # energy threshold [MeV]
DUNE_LENGTH=3.0
DUNE_DIST=574
DUNE_SOLID_ANGLE = np.arctan(sqrt(DUNE_AREA / pi) / DUNE_DIST)


# Grab the photon flux below 100 mrad
photon_flux_sub100mrad = np.genfromtxt("../DUNE/data/photon_flux/DUNE_target_photons_2d_sub100mrad_1e6POT.txt", delimiter=",")


pot_per_sample = 1e6
photon_flux_sub100mrad[:,2] *= 1/pot_per_sample  # converts to /s
angle_cut = DUNE_SOLID_ANGLE
forward_photon_flux = np.array([photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,0],
                                photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,1],
                                photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,2]]).transpose()




# Simulate the flux with angles w.r.t. beamline
ALP_MASS = 10  # MeV
ALP_COUPLING = 1e-9  # MeV^-1


# pass flux into axion flux generator
flux_gen = PrimakoffFromBeam4Vectors(forward_photon_flux, target=Material("Ar"), det_dist=DUNE_DIST, det_length=DUNE_LENGTH,
                    det_area=DUNE_AREA, axion_mass=ALP_MASS, axion_coupling=ALP_COUPLING, n_samples=5)

flux_gen.simulate(multicore=False)
flux_gen.propagate(is_isotropic=False)

print("flux sum = ", np.sum(flux_gen.decay_axion_weight))

alp_flux_energies = np.array(flux_gen.axion_energy)[flux_gen.decay_axion_weight>0]
alp_flux_angles = np.array(flux_gen.axion_angle)[flux_gen.decay_axion_weight>0]
alp_flux_wgt = flux_gen.decay_axion_weight[flux_gen.decay_axion_weight>0]
print(alp_flux_wgt)

out_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_wgt]).transpose()
np.savetxt("signal_data/fluxes_photon/FLUX-DECAY_alp_photon_ma-10MeV_gag1e-9MeV-1.txt", out_array)

print("Length of outfile (nsamples): ", out_array.shape[0])


angle_bins = np.logspace(-5, DUNE_SOLID_ANGLE, 50)
energy_bins = np.logspace(np.log10(30.0e-3), 2, 50)  # GeV


plt.hist(alp_flux_angles, weights=alp_flux_wgt, bins=angle_bins, histtype='step')
#plt.vlines(x=DUNE_SOLID_ANGLE, ymin=0.0, ymax=max(alp_flux_wgt), color='k', ls='dotted')
plt.xscale('log')
plt.ylabel("Flux")
plt.xlabel(r"$\theta_z$ [rad]")
plt.xlim((1e-5, DUNE_SOLID_ANGLE))
plt.show()
plt.close()


plt.hist(1e-3*alp_flux_energies, weights=alp_flux_wgt, bins=energy_bins, histtype='step')
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Flux")
plt.xlabel(r"$E_a$ [GeV]")
plt.xlim((30.0e-3, 100.0))
plt.show()
plt.close()

