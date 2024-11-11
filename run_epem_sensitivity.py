# Sensitivity curve finder for the photon coupling (1 GAMMA TOPOLOGY)

from scipy.special import gammaln

# Break up sensitivity contours into 2 pieces: 1 gamma and 2 gamma
print("ELECTRON SENSITIVITY GENERATION: BEGIN")
print("loading signal generators...")

from signal_generators import *


import matplotlib.pyplot as plt


# Load scan parameters
gae_list = np.logspace(-9, -5, 50)
ma_list = np.logspace(np.log10(2*M_E+0.01), 2.3, 50)

flat_ma = np.repeat(ma_list, gae_list.shape[0])
flat_gae = np.tile(gae_list, ma_list.shape[0])



# Create background histograms
print("Loading backgrounds...")
from dune_backgrounds import *

bkg_epem_nue = Background2Particle(data_file_name="data/1em1ep/epem_nue_4vectors_DUNE_bkg.txt", nu_flavor='nue', mass_particle_1=M_E, mass_particle_2=M_E)
bkg_epem_numu = Background2Particle(data_file_name="data/1em1ep/epem_numu_4vectors_DUNE_bkg.txt", nu_flavor='numu', mass_particle_1=M_E, mass_particle_2=M_E)
bkg_epem_numubar = Background2Particle(data_file_name="data/1em1ep/epem_numubar_4vectors_DUNE_bkg.txt", nu_flavor='numubar', mass_particle_1=M_E, mass_particle_2=M_E)
bkg_epem_nuebar = Background2Particle(data_file_name="data/1em1ep/epem_nuebar_4vectors_DUNE_bkg.txt", nu_flavor='nuebar', mass_particle_1=M_E, mass_particle_2=M_E)

energy_bins_ana = np.logspace(np.log10(30.0), np.log10(15.0e3), 25)
energy_bins_ana = np.append(energy_bins_ana, 120.0e3)
binned_bkg_nue, bins_nue = np.histogram(p0_epem_nue_1 + p0_epem_nue_2, weights=weights1_nue_epem, bins=energy_bins_ana)
binned_bkg_numu, bins_numu = np.histogram(p0_epem_numu_1 + p0_epem_numu_2, weights=weights1_numu_epem, bins=energy_bins_ana)
binned_bkg_nuebar, bins_nuebar = np.histogram(p0_epem_nuebar_1 + p0_epem_nuebar_2, weights=weights1_nuebar_epem, bins=energy_bins_ana)
binned_bkg_numubar, bins_numubar = np.histogram(p0_epem_numubar_1 + p0_epem_numubar_2, weights=weights1_numubar_epem, bins=energy_bins_ana)

binned_bkg_aggregate = binned_bkg_nuebar + binned_bkg_nue + binned_bkg_numu + binned_bkg_numubar

# with angle cut
bkg_nue_angle_cut_mask = bkg_epem_nue.dtheta_deg<20.0
binned_bkg_nue_acut, bins_nue = np.histogram((p0_epem_nue_1 + p0_epem_nue_2),
                                             weights=weights1_nue_epem*bkg_nue_angle_cut_mask, bins=energy_bins_ana)
bkg_numu_angle_cut_mask = bkg_epem_numu.dtheta_deg<20.0
binned_bkg_numu_acut, bins_numu = np.histogram((p0_epem_numu_1 + p0_epem_numu_2),
                                             weights=weights1_numu_epem*bkg_numu_angle_cut_mask, bins=energy_bins_ana)
bkg_nuebar_angle_cut_mask = bkg_epem_nuebar.dtheta_deg<20.0
binned_bkg_nuebar_acut, bins_nuebar = np.histogram((p0_epem_nuebar_1 + p0_epem_nuebar_2),
                                             weights=weights1_nuebar_epem*bkg_nuebar_angle_cut_mask, bins=energy_bins_ana)
bkg_numubar_angle_cut_mask = bkg_epem_numubar.dtheta_deg<20.0
binned_bkg_numubar_acut, bins_numubar = np.histogram((p0_epem_numubar_1 + p0_epem_numubar_2),
                                             weights=weights1_numubar_epem*bkg_numubar_angle_cut_mask, bins=energy_bins_ana)
binned_bkg_post_acut = binned_bkg_nuebar_acut + binned_bkg_nue_acut + binned_bkg_numu_acut + binned_bkg_numubar_acut



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


def run_sens(out_file_name, resume=False):
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
        

        flux_gen = ElectronALPFromBeam4Vectors(axion_mass=ma, target=Material("C"), n_samples=2000, axion_coupling=1.0)

        print("Simulating and propagating ALP flux for ma={}...".format(ma))
        flux_gen.simulate()

        alp_flux_energies = np.array(flux_gen.axion_energy)
        alp_flux_angles = np.array(flux_gen.axion_angle)
        alp_flux_weights = np.array(flux_gen.axion_flux)
        flux_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_weights]).transpose()
        ep_energy, em_energy, ep_theta_z, em_theta_z, inv_mass, \
            total_energy, sep_angles, event_weights = decay_alp_gen(input_flux=flux_array, ALP_MASS=ma, mass_daughters=M_E, resolved=True)

        print("Propagating for coupling array...")
        for g in gae_list:
            
            if resume and not resumed:
                if g <= last_g:
                    continue
                else:
                    print("RESUMING SCAN on g={}, ma={}".format(g, ma))
                    resumed = True
                    
            decayed_weights = propagate_decay(total_energy, event_weights, ma, W_ee(g, ma), rescale_factor=power(g/flux_gen.ge, 2))

            # precuts
            signal_events, signal_bins = np.histogram(total_energy, weights=decayed_weights, bins=energy_bins_ana)

            # calculate the Poisson log-likelihood: null hypothesis is background only
            ll = np.sum(binned_bkg_aggregate * log(signal_events+binned_bkg_aggregate+1) - signal_events \
                        - binned_bkg_aggregate - gammaln(binned_bkg_aggregate+1))  # Poisson log-likelihood
            
            # Apply cuts:
            # (1) angular cut DeltaTheta < 20 deg
            sig_angle_cut_mask = sep_angles<20.0*np.pi/180.0
            signal_events_angle_cut, signal_bins = np.histogram(total_energy[sig_angle_cut_mask],
                                                                weights=decayed_weights[sig_angle_cut_mask], bins=energy_bins_ana)

            print("Sig events before after acut = ", np.sum(signal_events), np.sum(signal_events_angle_cut))
            if np.sum(signal_events) > 0:
                sig_acut_eff = np.sum(signal_events_angle_cut)/np.sum(signal_events)
            else:
                sig_acut_eff = 1.0
            ll_post_acut = np.sum(binned_bkg_post_acut * log(signal_events_angle_cut+binned_bkg_post_acut+1) - signal_events_angle_cut \
                        - binned_bkg_post_acut - gammaln(binned_bkg_post_acut+1))  # Poisson log-likelihood

            # (2) inv_mass cut +/-2.5% of mass
            bkg_nue_mass_cut_mask = abs(bkg_epem_nue.inv_mass - ma) < 0.025*ma
            binned_bkg_nue_mcut, bins_nue = np.histogram((p0_epem_nue_1 + p0_epem_nue_2),
                                                        weights=weights1_nue_epem*bkg_nue_angle_cut_mask*bkg_nue_mass_cut_mask, bins=energy_bins_ana)
            bkg_numu_mass_cut_mask = abs(bkg_epem_numu.inv_mass - ma) < 0.025*ma
            binned_bkg_numu_mcut, bins_numu = np.histogram((p0_epem_numu_1 + p0_epem_numu_2),
                                                        weights=weights1_numu_epem*bkg_numu_angle_cut_mask*bkg_numu_mass_cut_mask, bins=energy_bins_ana)
            bkg_nuebar_mass_cut_mask = abs(bkg_epem_nuebar.inv_mass - ma) < 0.025*ma
            binned_bkg_nuebar_mcut, bins_nuebar = np.histogram((p0_epem_nuebar_1 + p0_epem_nuebar_2),
                                                        weights=weights1_nuebar_epem*bkg_nuebar_angle_cut_mask*bkg_nuebar_mass_cut_mask, bins=energy_bins_ana)
            bkg_numubar_mass_cut_mask = abs(bkg_epem_numubar.inv_mass - ma) < 0.025*ma
            binned_bkg_numubar_mcut, bins_numubar = np.histogram((p0_epem_numubar_1 + p0_epem_numubar_2),
                                                        weights=weights1_numubar_epem*bkg_numubar_angle_cut_mask*bkg_numubar_mass_cut_mask, bins=energy_bins_ana)

            binned_bkg_post_mcut = binned_bkg_nuebar_mcut + binned_bkg_nue_mcut + binned_bkg_numu_mcut + binned_bkg_numubar_mcut
            
            ll_post_mcut = np.sum(binned_bkg_post_mcut * log(signal_events_angle_cut+binned_bkg_post_mcut+1) - signal_events_angle_cut \
                        - binned_bkg_post_mcut - gammaln(binned_bkg_post_mcut+1))  # Poisson log-likelihood
            print("ll post acut: {}, {}, {}".format(np.sum(binned_bkg_post_mcut * log(signal_events_angle_cut+binned_bkg_post_mcut+1)),
                                                    np.sum(signal_events_angle_cut),
                                                    - np.sum(gammaln(binned_bkg_post_mcut+1))))

            file_out = open(out_file_name, "a")
            file_out.write(str(ma) + " " + str(g) + " " + str(ll) + " " + str(sig_acut_eff) + " " + str(ll_post_acut) + \
                           " " + str(ll_post_mcut) + " " + str(np.sum(signal_events)) + '\n')
            file_out.close()


def main():
    #plot_alp_spectra(ge=1e-7, ma=10.0)

    run_sens(out_file_name="sensitivities/epem_sensitivity_BKG_cutflow_20240630.txt", resume=False)


if __name__ == "__main__":
    main()

