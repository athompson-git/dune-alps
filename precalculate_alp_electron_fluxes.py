# Sensitivity curve finder for the photon coupling (1 GAMMA TOPOLOGY)

from scipy.special import gammaln

# Break up sensitivity contours into 2 pieces: 1 gamma and 2 gamma
print("ELECTRON SENSITIVITY GENERATION: BEGIN")
print("loading signal generators...")

from signal_generators import *


import matplotlib.pyplot as plt




def plot_alp_spectra(ge=1e-6, ma=2.0):

    flux_gen = ElectronALPFromBeam4Vectors(n_samples=50000)

    flux_gen.simulate()

    energy_bins = np.logspace(np.log10(30.0), 4, 100)

    plt.hist(flux_gen.axion_energy, weights=flux_gen.axion_flux, bins=energy_bins, histtype='step', density=True, label="ALP Flux")
    plt.plot(el_diff_flux_dune[:,0],el_diff_flux_dune[:,1]/np.sum(el_diff_flux_dune[:,1]), label="diff flux el")
    plt.plot(pos_diff_flux_dune[:,0],pos_diff_flux_dune[:,1]/np.sum(pos_diff_flux_dune[:,1]), label="diff flux pos")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r"$E_a$ [MeV]")
    plt.legend()
    plt.ylabel("Flux Counts / year")
    plt.show()


def run_sens():
    # Load scan parameters
    ma_list = np.logspace(-2, 4, 20)

    # begin sensitivity loop
    for ma in ma_list:
        flux_gen = ElectronALPFromBeam4Vectors(axion_mass=ma, target=Material("C"), n_samples=1000000)
        flux_gen.simulate()

        print("ON MASS = ", ma)

        alp_flux_energies = np.array(flux_gen.axion_energy)
        alp_flux_angles = np.array(flux_gen.axion_angle)
        alp_flux_wgt = np.array(flux_gen.axion_flux)

        energy_bins = np.logspace(np.log10(30.0), np.log10(120000.0), 100)
        energy_centers = (energy_bins[1:] + energy_bins[:-1])/2
        angle_bins = np.logspace(-6, np.log10(flux_gen.det_sa()), 100)
        angle_centers = (angle_bins[1:] + angle_bins[:-1])/2

        he, ebins = np.histogram(alp_flux_energies, weights=alp_flux_wgt, bins=energy_bins)
        h_stats, ebin_stats = np.histogram(alp_flux_energies, bins=energy_bins)
        pois_error = np.nan_to_num(he*np.sqrt(h_stats)/h_stats)
        print("AVERAGE ENERGY POISSON RELERROR = {}".format(np.mean(pois_error)))
        plt.errorbar(energy_centers, he, yerr=pois_error, marker='o', color='k', mfc='w', linewidth=1.0, label="alp flux")
        plt.hist(el_diff_flux_dune[:,0], weights=max(he)*el_diff_flux_dune[:,1]/max(el_diff_flux_dune[:,1]),
                 bins=energy_bins, histtype='step', label="el flux")
        plt.hist(pos_diff_flux_dune[:,0], weights=max(he)*pos_diff_flux_dune[:,1]/max(pos_diff_flux_dune[:,1]),
                 bins=energy_bins, histtype='step', label="pos flux")
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("energy")
        plt.legend()
        plt.show()
        plt.close()

        ha, abins = np.histogram(alp_flux_angles, weights=alp_flux_wgt, bins=angle_bins)
        ha_stats, abin_stats = np.histogram(alp_flux_angles, bins=angle_bins)
        pois_error = np.nan_to_num(ha*np.sqrt(ha_stats)/ha_stats)
        print("AVERAGE ANGULAR POISSON RELERROR = {}".format(np.mean(pois_error)))
        plt.errorbar(angle_centers, ha, yerr=pois_error, mfc='w', linewidth=1.0, marker='o', color='k')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("angle")
        plt.show()
        plt.close()

            


def main():
    #plot_alp_spectra(ge=1e-7, ma=10.0)

    run_sens()


if __name__ == "__main__":
    main()

