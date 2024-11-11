# Sensitivity curve finder for the photon coupling (2 GAMMA TOPOLOGY)

from scipy.special import gammaln

# Break up sensitivity contours into 2 pieces: 1 gamma and 2 gamma
print("PHOTON SENSITIVITY GENERATION: BEGIN")
print("loading signal generators...")

import matplotlib.pyplot as plt
from matplotlib.pylab import rc
from matplotlib.colors import LogNorm
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from signal_generators import *
from dune_backgrounds import *

from decimal import Decimal


def run_sens(out_file_name, resume=False):
    # Start with loop over parameters (MeV)
    gagamma_list = np.logspace(-14, 0, 200)
    ma_list = np.logspace(-4, 3, 100)


    # Import flux info
    # Grab the photon flux below 100 mrad
    print("Reading in photon flux...")
    photon_flux_sub100mrad = np.genfromtxt("data/fluxes/dune_target_gamma_1e5POT_2d_sub100mrad_june24.txt")

    pot_per_sample = 1e5
    photon_flux_sub100mrad[:,2] *= DUNE_POT_PER_YEAR/pot_per_sample/365/24/3600  # converts to per s
    angle_cut = DUNE_SOLID_ANGLE
    forward_photon_flux = photon_flux_sub100mrad #np.array([photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,0],
                                    #photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,1],
                                    #photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,2]]).transpose()

    energy_bins_ana = np.logspace(np.log10(30.0), np.log10(15.0e3), 25)
    energy_bins_ana = np.append(energy_bins_ana, 120.0e3)

    if resume:
        previous_dat_file = np.genfromtxt(out_file_name)
        num_ma = np.unique(previous_dat_file[:,0])
        last_ma = max(num_ma)
        last_g = previous_dat_file[-1,1]
        
    else:
        file_out = open(out_file_name, "w")

    resumed = False

    # begin sensitivity loop
    for ma in ma_list:

        if resume:
            if ma < last_ma:
                continue
            if last_g == gagamma_list[-1]:
                resumed = True
                continue

        flux_gen = PrimakoffFromBeam4Vectors(forward_photon_flux, target=Material("C"), det_dist=DUNE_DIST, det_length=DUNE_LENGTH,
                            det_area=DUNE_AREA, axion_mass=ma, axion_coupling=1.0, n_samples=5)

        print("Simulating and propagating ALP flux for ma={}...".format(ma))
        flux_gen.simulate(multicore=False)

        print("Propagating for coupling array...")
        for g in gagamma_list:

            if resume and not resumed:
                if g <= last_g:
                    continue
                else:
                    print("RESUMING SCAN on g={}, ma={}".format(g, ma))
                    resumed = True

            flux_gen.propagate(is_isotropic=False, new_coupling=g)

            # Decay the 4-vectors
            event_gen = PhotonEventGenerator(flux_gen, detector=Material("Ar"))
            n_decays = event_gen.decays(EXPOSURE_DAYS, threshold=30.0)

            file_out = open(out_file_name, "a")
            file_out.write(str(ma) + " " + str(g) + " " + str(n_decays) + '\n')
            file_out.close()





def main():
    
    run_sens(out_file_name="sensitivities/photon_sensitivity_GAr_Aug2024.txt", resume=False)


if __name__ == "__main__":
    main()

