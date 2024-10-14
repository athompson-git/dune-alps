# Sensitivity curve finder for the photon coupling (1 GAMMA TOPOLOGY)

from scipy.special import gammaln

# Break up sensitivity contours into 2 pieces: 1 gamma and 2 gamma
print("ELECTRON SENSITIVITY GENERATION: BEGIN")
print("loading signal generators...")

from signal_generators import *


import matplotlib.pyplot as plt


# Load scan parameters
gae_list = np.logspace(-11, -3, 100)
ma_list = np.logspace(-2, np.log10(2*M_E+0.01), 50)

flat_ma = np.repeat(ma_list, gae_list.shape[0])
flat_gae = np.tile(gae_list, ma_list.shape[0])



# Create background histograms
print("Loading backgrounds...")
from dune_backgrounds import *

# particle 1 = gamma, particle 2 = em
bkg_1g1em_nue = Background2Particle(data_file_name="data/1g1e0p/emgamma_nue_4vectors_DUNE_bkg.txt",
                                    mass_particle_1=0.0, mass_particle_2=M_E, nu_flavor="nue")
bkg_1g1em_numu = Background2Particle(data_file_name="data/1g1e0p/emgamma_numu_4vectors_DUNE_bkg.txt",
                                     mass_particle_1=0.0, mass_particle_2=M_E, nu_flavor="numu")
bkg_1g1em_numubar = Background2Particle(data_file_name="data/1g1e0p/emgamma_numubar_4vectors_DUNE_bkg.txt",
                                        mass_particle_1=0.0, mass_particle_2=M_E, nu_flavor="numubar")
bkg_1g1em_nuebar = Background2Particle(data_file_name="data/1g1e0p/emgamma_nuebar_4vectors_DUNE_bkg.txt",
                                       mass_particle_1=0.0, mass_particle_2=M_E, nu_flavor="nuebar")

energy_bins_ana = np.logspace(np.log10(30.0), np.log10(15.0e3), 25)
energy_bins_ana = np.append(energy_bins_ana, 120.0e3)
binned_bkg_nue, bins_nue = np.histogram(p0_1g1em_nue_1 + p0_1g1em_nue_2, weights=weights1_nue_1g1em, bins=energy_bins_ana)
binned_bkg_numu, bins_numu = np.histogram(p0_1g1em_numu_1 + p0_1g1em_numu_2, weights=weights1_numu_1g1em, bins=energy_bins_ana)
binned_bkg_nuebar, bins_nuebar = np.histogram(p0_1g1em_nuebar_1 + p0_1g1em_nuebar_2, weights=weights1_nuebar_1g1em, bins=energy_bins_ana)
binned_bkg_numubar, bins_numubar = np.histogram(p0_1g1em_numubar_1 + p0_1g1em_numubar_2, weights=weights1_numubar_1g1em, bins=energy_bins_ana)

binned_bkg_aggregate = binned_bkg_nuebar + binned_bkg_nue + binned_bkg_numu + binned_bkg_numubar
binned_bkg_aggregate = binned_bkg_aggregate + 1/binned_bkg_aggregate.shape[0]

# with angle cut
bkg_nue_angle_cut_mask = 1.0 #bkg_1g1em_nue.dtheta_deg<20.0
binned_bkg_nue_acut, bins_nue = np.histogram((p0_1g1em_nue_1 + p0_1g1em_nue_2),
                                             weights=weights1_nue_1g1em*bkg_nue_angle_cut_mask, bins=energy_bins_ana)
bkg_numu_angle_cut_mask = bkg_1g1em_numu.dtheta_deg<20.0
binned_bkg_numu_acut, bins_numu = np.histogram((p0_1g1em_numu_1 + p0_1g1em_numu_2),
                                             weights=weights1_numu_1g1em*bkg_numu_angle_cut_mask, bins=energy_bins_ana)
bkg_nuebar_angle_cut_mask = bkg_1g1em_nuebar.dtheta_deg<20.0
binned_bkg_nuebar_acut, bins_nuebar = np.histogram((p0_1g1em_nuebar_1 + p0_1g1em_nuebar_2),
                                             weights=weights1_nuebar_1g1em*bkg_nuebar_angle_cut_mask, bins=energy_bins_ana)
bkg_numubar_angle_cut_mask = bkg_1g1em_numubar.dtheta_deg<20.0
binned_bkg_numubar_acut, bins_numubar = np.histogram((p0_1g1em_numubar_1 + p0_1g1em_numubar_2),
                                             weights=weights1_numubar_1g1em*bkg_numubar_angle_cut_mask, bins=energy_bins_ana)
binned_bkg_post_dthetacut = binned_bkg_nuebar_acut + binned_bkg_nue_acut + binned_bkg_numu_acut + binned_bkg_numubar_acut
binned_bkg_post_dthetacut = binned_bkg_post_dthetacut + 1/binned_bkg_post_dthetacut.shape[0]

# with angle cut and xcut
nue_x = bkg_1g1em_nue.energy_p1 / bkg_1g1em_nue.energy_p2
numu_x = bkg_1g1em_numu.energy_p1 / bkg_1g1em_numu.energy_p2
nuebar_x = bkg_1g1em_nuebar.energy_p1 / bkg_1g1em_nuebar.energy_p2
numubar_x = bkg_1g1em_numubar.energy_p1 / bkg_1g1em_numubar.energy_p2
nue_xcut_mask = (nue_x < 5.0) * (nue_x > 0.1)
numu_xcut_mask = (numu_x < 5.0) * (numu_x > 0.1)
nuebar_xcut_mask = (nuebar_x < 5.0) * (nuebar_x > 0.1)
numubar_xcut_mask = (numubar_x < 5.0) * (numubar_x > 0.1)
binned_bkg_nue_xcut, bins_nue = np.histogram((p0_1g1em_nue_1 + p0_1g1em_nue_2),
                                             weights=weights1_nue_1g1em*bkg_nue_angle_cut_mask*nue_xcut_mask, bins=energy_bins_ana)
binned_bkg_numu_xcut, bins_numu = np.histogram((p0_1g1em_numu_1 + p0_1g1em_numu_2),
                                             weights=weights1_numu_1g1em*bkg_numu_angle_cut_mask*numu_xcut_mask, bins=energy_bins_ana)
binned_bkg_nuebar_xcut, bins_nuebar = np.histogram((p0_1g1em_nuebar_1 + p0_1g1em_nuebar_2),
                                             weights=weights1_nuebar_1g1em*bkg_nuebar_angle_cut_mask*nuebar_xcut_mask, bins=energy_bins_ana)
binned_bkg_numubar_xcut, bins_numubar = np.histogram((p0_1g1em_numubar_1 + p0_1g1em_numubar_2),
                                             weights=weights1_numubar_1g1em*bkg_numubar_angle_cut_mask*numubar_xcut_mask, bins=energy_bins_ana)
binned_bkg_post_xcut = binned_bkg_nuebar_xcut + binned_bkg_nue_xcut + binned_bkg_numu_xcut + binned_bkg_numubar_xcut
binned_bkg_post_xcut = binned_bkg_post_xcut + 1/binned_bkg_post_xcut.shape[0]



def plot_alp_spectra(ge=1e-6, ma=0.5):

    flux_gen = ElectronALPFromBeam4Vectors(axion_mass=ma, target=Material("C"), n_samples=100000)

    flux_gen.simulate()
    flux_gen.propagate(is_isotropic=False, new_coupling=ge)
    
    alp_flux_energies = np.array(flux_gen.axion_energy)
    alp_flux_angles = np.array(flux_gen.axion_angle)
    alp_flux_wgt = flux_gen.scatter_axion_weight

    # Decay the 4-vectors
    print("decaying 4-vectors...")
    flux_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_wgt]).transpose()
    em_energy, gamma_energy, em_theta_z, gamma_theta_z, \
        total_energy, sep_angles, signal_weights = \
            compton_scatter_events(input_flux=flux_array, ALP_MASS=ma, resolved=True)

    dtheta_bins = np.linspace(1.0, 180.0, 100)
    rad2deg = 180.0/np.pi
    print("signal wgt shape = {}".format(signal_weights.shape))

    # rescale signal weights to compare with bkg
    signal_weights = signal_weights * np.sum(bkg_1g1em_nue.weights)/np.sum(signal_weights)

    print(sep_angles*rad2deg)

    plt.hist(sep_angles*rad2deg, weights=signal_weights, bins=dtheta_bins, histtype='step', label=r"ALP Signal ($a e^- \to \gamma e^-$)")
    plt.hist(bkg_1g1em_nue.dtheta_deg, weights=bkg_1g1em_nue.weights, bins=dtheta_bins, label=r"$\nu_e (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_numu.dtheta_deg, weights=bkg_1g1em_numu.weights, bins=dtheta_bins, label=r"$\nu_\mu (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_nuebar.dtheta_deg, weights=bkg_1g1em_nuebar.weights, bins=dtheta_bins, label=r"$\bar{\nu_e} (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_numubar.dtheta_deg, weights=bkg_1g1em_numubar.weights, bins=dtheta_bins, label=r"$\bar{\nu_\mu} (e^- \gamma)$", histtype="step")
    plt.yscale('log')
    plt.xlabel(r"$\Delta\theta$ [deg]", fontsize=16)
    plt.ylabel(r"Counts / 3.5 Years", fontsize=16)
    plt.ylim((1e-1, 1e6))
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close()

    energy_bins = np.logspace(log10(30.0), log10(100000.0), 100)
    plt.hist(total_energy, weights=signal_weights, bins=energy_bins, histtype='step', label=r"ALP Signal ($a e^- \to \gamma e^-$)")
    plt.hist(bkg_1g1em_nue.total_energy, weights=bkg_1g1em_nue.weights, bins=energy_bins, label=r"$\nu_e (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_numu.total_energy, weights=bkg_1g1em_numu.weights, bins=energy_bins, label=r"$\nu_\mu (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_nuebar.total_energy, weights=bkg_1g1em_nuebar.weights, bins=energy_bins, label=r"$\bar{\nu_e} (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_numubar.total_energy, weights=bkg_1g1em_numubar.weights, bins=energy_bins, label=r"$\bar{\nu_\mu} (e^- \gamma)$", histtype="step")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r"$E_{e^-} + E_\gamma$ [MeV]", fontsize=16)
    plt.ylabel(r"Counts / 3.5 Years", fontsize=16)
    plt.ylim((1e-1, 1e6))
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close()

    # look at energy ratios of gamma / e-
    nue_x = bkg_1g1em_nue.energy_p1 / bkg_1g1em_nue.energy_p2
    numu_x = bkg_1g1em_numu.energy_p1 / bkg_1g1em_numu.energy_p2
    nuebar_x = bkg_1g1em_nuebar.energy_p1 / bkg_1g1em_nuebar.energy_p2
    numubar_x = bkg_1g1em_numubar.energy_p1 / bkg_1g1em_numubar.energy_p2
    alp_x = gamma_energy / em_energy

    efraction_bins = np.logspace(-3, 3, 100)
    plt.hist(alp_x, weights=signal_weights, bins=efraction_bins, histtype='step', label=r"ALP Signal ($a e^- \to \gamma e^-$)")
    plt.hist(nue_x, weights=bkg_1g1em_nue.weights, bins=efraction_bins, label=r"$\nu_e (e^- \gamma)$", histtype="step")
    plt.hist(numu_x, weights=bkg_1g1em_numu.weights, bins=efraction_bins, label=r"$\nu_\mu (e^- \gamma)$", histtype="step")
    plt.hist(nuebar_x, weights=bkg_1g1em_nuebar.weights, bins=efraction_bins, label=r"$\bar{\nu_e} (e^- \gamma)$", histtype="step")
    plt.hist(numubar_x, weights=bkg_1g1em_numubar.weights, bins=efraction_bins, label=r"$\bar{\nu_\mu} (e^- \gamma)$", histtype="step")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r"$x = E_\gamma/E_{e^-}$ [MeV]", fontsize=16)
    plt.ylabel(r"Counts / 3.5 Years", fontsize=16)
    plt.ylim((1e-1, 1e6))
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close()

    # perform dtheta cut
    nue_cut_mask = bkg_1g1em_nue.dtheta_deg < 20.0
    numu_cut_mask = bkg_1g1em_numu.dtheta_deg < 20.0
    nuebar_cut_mask = bkg_1g1em_nuebar.dtheta_deg < 20.0
    numubar_cut_mask = bkg_1g1em_numubar.dtheta_deg < 20.0
    signal_dtheta_mask = (sep_angles*rad2deg) < 20.0
    print((signal_weights*signal_dtheta_mask).shape)
    print(signal_weights.shape, signal_dtheta_mask.shape, numubar_cut_mask.shape)
    """
    plt.hist(total_energy, weights=signal_weights*signal_dtheta_mask, bins=energy_bins, histtype='step', label=r"ALP Signal ($a e^- \to \gamma e^-$)")
    plt.hist(bkg_1g1em_nue.total_energy, weights=bkg_1g1em_nue.weights*nue_cut_mask, bins=energy_bins, label=r"$\nu_e (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_numu.total_energy, weights=bkg_1g1em_numu.weights*numu_cut_mask, bins=energy_bins, label=r"$\nu_\mu (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_nuebar.total_energy, weights=bkg_1g1em_nuebar.weights*nuebar_cut_mask, bins=energy_bins, label=r"$\bar{\nu_e} (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_numubar.total_energy, weights=bkg_1g1em_numubar.weights*numubar_cut_mask, bins=energy_bins, label=r"$\bar{\nu_\mu} (e^- \gamma)$", histtype="step")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r"$E_{e^-} + E_\gamma$ [MeV]", fontsize=16)
    plt.ylabel(r"Counts / 3.5 Years", fontsize=16)
    plt.ylim((1e-1, 1e6))
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close()
    """

    # perform dtheta + x cut
    nue_xcut_mask = (nue_x < 5.0) * (nue_x > 0.1)
    numu_xcut_mask = (numu_x < 5.0) * (numu_x > 0.1)
    nuebar_xcut_mask = (nuebar_x < 5.0) * (nuebar_x > 0.1)
    numubar_xcut_mask = (numubar_x < 5.0) * (numubar_x > 0.1)
    signal_xcut_mask = (alp_x < 5.0) * (alp_x > 0.1)
    plt.hist(total_energy, weights=signal_weights*signal_dtheta_mask*signal_xcut_mask, bins=energy_bins, histtype='step', label=r"ALP Signal ($a e^- \to \gamma e^-$)")
    plt.hist(bkg_1g1em_nue.total_energy, weights=bkg_1g1em_nue.weights*nue_cut_mask*nue_xcut_mask,
             bins=energy_bins, label=r"$\nu_e (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_numu.total_energy, weights=bkg_1g1em_numu.weights*numu_cut_mask*numu_xcut_mask,
             bins=energy_bins, label=r"$\nu_\mu (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_nuebar.total_energy, weights=bkg_1g1em_nuebar.weights*nuebar_cut_mask*nuebar_xcut_mask,
             bins=energy_bins, label=r"$\bar{\nu_e} (e^- \gamma)$", histtype="step")
    plt.hist(bkg_1g1em_numubar.total_energy, weights=bkg_1g1em_numubar.weights*numubar_cut_mask*numubar_xcut_mask,
             bins=energy_bins, label=r"$\bar{\nu_\mu} (e^- \gamma)$", histtype="step")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r"$E_{e^-} + E_\gamma$ [MeV]", fontsize=16)
    plt.ylabel(r"Counts / 3.5 Years", fontsize=16)
    plt.ylim((1e-1, 1e6))
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close()



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
        

        flux_gen = ElectronALPFromBeam4Vectors(axion_mass=ma, axion_coupling=1.0, target=Material("C"), n_samples=200000)

        print("Simulating and propagating ALP flux for ma={}...".format(ma))
        flux_gen.simulate()

        flux_gen.propagate(is_isotropic=False, new_coupling=1.0)

        print("flux sum = {} at g_ae = {}".format(np.sum(flux_gen.scatter_axion_weight), 1.0))

        alp_flux_energies = np.array(flux_gen.axion_energy)
        alp_flux_angles = np.array(flux_gen.axion_angle)
        alp_flux_wgt = flux_gen.scatter_axion_weight

        # Decay the 4-vectors
        print("decaying 4-vectors...")
        flux_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_wgt]).transpose()
        em_energy, gamma_energy, em_theta_z, gamma_theta_z, \
            total_energy, sep_angles, event_weights = \
                compton_scatter_events(input_flux=flux_array, ALP_MASS=ma, resolved=True)
        
        alp_x = gamma_energy / em_energy
        signal_xcut_mask = (alp_x < 5.0) * (alp_x > 0.1)
        signal_dtheta_mask = (sep_angles*180.0/pi) < 20.0

        signal_events_g1, signal_bins = np.histogram(total_energy, weights=event_weights, bins=energy_bins_ana)
        signal_events_dthetacut_g1, signal_bins = np.histogram(total_energy, weights=event_weights*signal_dtheta_mask, bins=energy_bins_ana)
        signal_events_xcut_g1, signal_bins = np.histogram(total_energy, weights=event_weights*signal_dtheta_mask*signal_xcut_mask, bins=energy_bins_ana)

        print("Propagating for coupling array...")
        for g in gae_list:
            if resume and not resumed:
                if g <= last_g:
                    continue
                else:
                    print("RESUMING SCAN on g={}, ma={}".format(g, ma))
                    resumed = True
            

            signal_events = signal_events_g1 * power(g,4)
            signal_events_xcut = signal_events_dthetacut_g1*power(g,4)
            signal_events_dthetacut = signal_events_xcut_g1*power(g,4)
        
            # calculate the Poisson log-likelihood: null hypothesis is background only
            ll = np.sum(binned_bkg_aggregate * log(signal_events+binned_bkg_aggregate) - signal_events \
                        - binned_bkg_aggregate - gammaln(binned_bkg_aggregate))
            
            # Apply cuts:
            # (1) dtheta
            ll_post_dthetacut = np.sum(binned_bkg_post_dthetacut * log(signal_events_dthetacut+binned_bkg_post_dthetacut) - signal_events_dthetacut \
                        - binned_bkg_post_dthetacut - gammaln(binned_bkg_post_dthetacut))

            # (2) dtheta cut + xcut
            ll_post_xcut = np.sum(binned_bkg_post_xcut * log(signal_events_xcut+binned_bkg_post_xcut) - signal_events_xcut \
                        - binned_bkg_post_xcut - gammaln(binned_bkg_post_xcut))

            file_out = open(out_file_name, "a")
            file_out.write(str(ma) + " " + str(g) + " " + str(ll) + " "  + str(ll_post_dthetacut) + \
                           " "  + str(ll_post_xcut) + " " + str(np.sum(signal_events)) + '\n')
            file_out.close()


def main():
    plot_alp_spectra(ge=1e-4, ma=0.5)

    #run_sens(out_file_name="sensitivities/1g1em_sensitivity_BKG_cutflow_20240323.txt", resume=False)


if __name__ == "__main__":
    main()

