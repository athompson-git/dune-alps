# Sensitivity curve finder for the photon coupling (1 GAMMA TOPOLOGY)

from scipy.special import gammaln

# Break up sensitivity contours into 2 pieces: 1 gamma and 2 gamma
print("ELECTRON SENSITIVITY GENERATION: BEGIN")
print("loading signal generators...")

from signal_generators import *


import matplotlib.pyplot as plt


# Load scan parameters
gae_list = np.logspace(-11, -3, 100)
ma_list = np.logspace(np.log10(2*M_E+0.01), np.log10(500.0), 68)

flat_ma = np.repeat(ma_list, gae_list.shape[0])
flat_gae = np.tile(gae_list, ma_list.shape[0])



# Create background histograms
print("Loading backgrounds...")
from dune_backgrounds import *

bkg_1g_nue = Background1Particle(data_file_name="data/1g0p/1gamma_nue_4vectors_DUNE_bkg.txt", nu_flavor="nue")
bkg_1g_numu = Background1Particle(data_file_name="data/1g0p/1gamma_numu_4vectors_DUNE_bkg.txt", nu_flavor="numu")
bkg_1g_numubar = Background1Particle(data_file_name="data/1g0p/1gamma_numubar_4vectors_DUNE_bkg.txt", nu_flavor="numubar")
bkg_1g_nuebar = Background1Particle(data_file_name="data/1g0p/1gamma_nuebar_4vectors_DUNE_bkg.txt", nu_flavor="nuebar")

energy_bins_ana = np.logspace(np.log10(30.0), np.log10(15.0e3), 25)
energy_bins_ana = np.append(energy_bins_ana, 120.0e3)
binned_bkg_nue, bins_nue = np.histogram(p0_g_nue, weights=weights1_nue_g, bins=energy_bins_ana)
binned_bkg_numu, bins_numu = np.histogram(p0_g_numu, weights=weights1_numu_g, bins=energy_bins_ana)
binned_bkg_nuebar, bins_nuebar = np.histogram(p0_g_nuebar, weights=weights1_nuebar_g, bins=energy_bins_ana)
binned_bkg_numubar, bins_numubar = np.histogram(p0_g_numubar, weights=weights1_numubar_g, bins=energy_bins_ana)

binned_bkg_aggregate = binned_bkg_nuebar + binned_bkg_nue + binned_bkg_numu + binned_bkg_numubar + 1

# with angle cut
nue_1g_acut = bkg_1g_nue.theta_z_deg<1.0
numu_1g_acut = bkg_1g_numu.theta_z_deg<1.0
nuebar_1g_acut = bkg_1g_nuebar.theta_z_deg<1.0
numubar_1g_acut = bkg_1g_numubar.theta_z_deg<1.0

binned_bkg_nue_acut, bins_nue = np.histogram(p0_g_nue, weights=weights1_nue_g*nue_1g_acut, bins=energy_bins_ana)
binned_bkg_numu_acut, bins_numu = np.histogram(p0_g_numu, weights=weights1_numu_g*numu_1g_acut, bins=energy_bins_ana)
binned_bkg_nuebar_acut, bins_nuebar = np.histogram(p0_g_nuebar, weights=weights1_nuebar_g*nuebar_1g_acut, bins=energy_bins_ana)
binned_bkg_numubar_acut, bins_numubar = np.histogram(p0_g_numubar, weights=weights1_numubar_g*numubar_1g_acut, bins=energy_bins_ana)

binned_bkg_acut_aggregate = binned_bkg_nuebar_acut + binned_bkg_nue_acut + binned_bkg_numu_acut + binned_bkg_numubar_acut
binned_bkg_acut_aggregate += 1.0/np.sum(binned_bkg_aggregate)

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


scatter_rescale = DUNE_POT_PER_YEAR/S_PER_DAY/365


def run_sens_pair_prod(out_file_name):
    print("\nLOOP START: writing sensitivity loop to {}".format(out_file_name))

    file_out = open(out_file_name, "w")

    ma_list_scatter = np.logspace(-2, np.log10(2*M_E), 68)

    # begin sensitivity loop
    for ma in ma_list_scatter:

        flux_gen = ElectronALPFromBeam4Vectors(axion_mass=ma, axion_coupling=1.0, target=Material("C"), n_samples=200000)

        print("--------- Simulating and propagating ALP flux for ma={}...".format(ma))
        flux_gen.simulate()

        # assume below decay threshold of 2me
        flux_gen.propagate(is_isotropic=False)

        print("sum of flux weights = {}".format(np.sum(flux_gen.axion_flux)))
        print("sum of propagated weights = {}".format(np.sum(flux_gen.scatter_axion_weight)))
        
        evt_gen = ElectronEventGenerator(flux_gen, detector=Material("Ar"))
        evt_gen.pair_production(1.0, ntargets=DUNE_NTARGETS, days_exposure=scatter_rescale*EXPOSURE_DAYS, threshold=DUNE_THRESH)
        signal_events_unit_norm, signal_bins = np.histogram(evt_gen.axion_energy, weights=evt_gen.pair_weights, bins=energy_bins_ana)

        print("Sum of scatter weights = ", np.sum(signal_events_unit_norm))
        for g in gae_list:

            signal_events = power(g, 4) * signal_events_unit_norm
            
            # calculate the Poisson log-likelihood: null hypothesis is background only
            ll = np.sum(binned_bkg_aggregate * log(signal_events+binned_bkg_aggregate) - signal_events \
                        - binned_bkg_aggregate - gammaln(binned_bkg_aggregate))  # Poisson log-likelihood
            

            # apply angular cut
            signal_events_acut = signal_events
            ll_acut = np.sum(binned_bkg_acut_aggregate * log(signal_events_acut+binned_bkg_acut_aggregate) - signal_events_acut \
                        - binned_bkg_acut_aggregate - gammaln(binned_bkg_acut_aggregate))  # Poisson log-likelihood
            
            file_out = open(out_file_name, "a")
            file_out.write(str(ma) + " " + str(g) + " " + str(ll) + " " + str(ll_acut) + " " + str(np.sum(signal_events)) + '\n')
            file_out.close()



def run_sens_decay(out_file_name, resume=False):
    print("\nLOOP START: writing sensitivity loop to {}".format(out_file_name))

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
            if last_g == gae_list[-1]:
                resumed = True
                continue
        

        flux_gen = ElectronALPFromBeam4Vectors(axion_mass=ma, target=Material("C"), n_samples=200000)

        print("Simulating and propagating ALP flux for ma={}...".format(ma))
        flux_gen.simulate()

        print("Propagating for coupling array...")
        for g in gae_list:
            if resume and not resumed:
                if g <= last_g:
                    continue
                else:
                    print("RESUMING SCAN on g={}, ma={}".format(g, ma))
                    resumed = True
            
            flux_gen.propagate(is_isotropic=False, new_coupling=g)

            print("flux sum = {} at g_ae = {}".format(np.sum(flux_gen.decay_axion_weight), g))

            alp_flux_energies = np.array(flux_gen.axion_energy)
            alp_flux_angles = np.array(flux_gen.axion_angle)
            alp_flux_wgt = flux_gen.decay_axion_weight

            # Decay the 4-vectors
            print("decaying 4-vectors...")
            flux_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_wgt]).transpose()
            ep_energy, em_energy, ep_theta_z, em_theta_z, inv_mass, \
                total_energy, sep_angles, event_weights = \
                    decay_alp_gen(input_flux=flux_array, ALP_MASS=ma, mass_daughters=M_E, resolved=False)
            
            # precuts
            signal_events, signal_bins = np.histogram(total_energy, weights=event_weights, bins=energy_bins_ana)

            # calculate the Poisson log-likelihood: null hypothesis is background only
            ll = np.sum(binned_bkg_aggregate * log(signal_events+binned_bkg_aggregate) - signal_events \
                        - binned_bkg_aggregate - gammaln(binned_bkg_aggregate))  # Poisson log-likelihood
            
            # Apply cuts:
            # (1) angular cut on the e+e- collinear pair must be less than 2e-2 rad
            sig_angle_cut_mask = ep_theta_z<2e-2
            signal_events_angle_cut, signal_bins = np.histogram(total_energy[sig_angle_cut_mask],
                                                                weights=event_weights[sig_angle_cut_mask], bins=energy_bins_ana)

            print("Sig events before after acut = ", np.sum(signal_events_angle_cut), np.sum(signal_events))
            if np.sum(signal_events) > 0:
                sig_acut_eff = np.sum(signal_events_angle_cut)/np.sum(signal_events)
            else:
                sig_acut_eff = 1.0
            print("Background counts afer angular cut = ", np.sum(binned_bkg_acut_aggregate))
            ll_post_acut = np.sum(binned_bkg_acut_aggregate * log(signal_events_angle_cut+binned_bkg_acut_aggregate) - signal_events_angle_cut \
                        - binned_bkg_acut_aggregate - gammaln(binned_bkg_acut_aggregate))  # Poisson log-likelihood

            file_out = open(out_file_name, "a")
            file_out.write(str(ma) + " " + str(g) + " " + str(ll) + " " + str(sig_acut_eff) + " " + str(ll_post_acut) + \
                            " " + str(np.sum(signal_events)) + '\n')
            file_out.close()


def main():
    #plot_alp_spectra(ge=1e-7, ma=10.0)

    run_sens_pair_prod(out_file_name="sensitivities/epem_pp_sensitivity_BKG_cutflow_20241207.txt")


if __name__ == "__main__":
    main()

