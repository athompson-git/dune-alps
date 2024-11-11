# Sensitivity curve finder for the photon coupling (1 GAMMA TOPOLOGY)

from scipy.special import gammaln

# Break up sensitivity contours into 2 pieces: 1 gamma and 2 gamma
print("PHOTON SENSITIVITY GENERATION: BEGIN")
print("loading signal generators...")
from signal_generators import *

print("Loading backgrounds...")
from dune_backgrounds import *




def run_sens(out_file_name, resume=True):

    if resume:
        previous_dat_file = np.genfromtxt(out_file_name)
        num_ma = np.unique(previous_dat_file[:,0])
        last_ma = max(num_ma)
        last_g = previous_dat_file[-1,1]
        
    else:
        file_out = open(out_file_name, "w")

    resumed = False

    # Start with loop over parameters (MeV)
    gagamma_list = np.logspace(-14, 0, 100)
    ma_list = np.logspace(-3, 3, 250)


    # Import flux info
    # Grab the photon flux below 100 mrad
    print("Reading in photon flux...")
    photon_flux_sub100mrad = np.genfromtxt("../DUNE/data/photon_flux/DUNE_target_photons_2d_sub100mrad_1e6POT.txt", delimiter=",")


    pot_per_sample = 1e6
    photon_flux_sub100mrad[:,2] *= 1/pot_per_sample  # converts to /s
    angle_cut = DUNE_SOLID_ANGLE
    forward_photon_flux = np.array([photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,0],
                                photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,1],
                                photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,2]]).transpose()


    # Create background histograms

    bkg_1g_nue = Background1Particle(data_file_name="data/1g0p/1gamma_nue_4vectors_DUNE_bkg.txt")
    bkg_1g_numu = Background1Particle(data_file_name="data/1g0p/1gamma_numu_4vectors_DUNE_bkg.txt")
    bkg_1g_numubar = Background1Particle(data_file_name="data/1g0p/1gamma_numubar_4vectors_DUNE_bkg.txt")
    bkg_1g_nuebar = Background1Particle(data_file_name="data/1g0p/1gamma_nuebar_4vectors_DUNE_bkg.txt")

    bkg_1ep_nue = Background1Particle(data_file_name="data/1g0p/1ep_nue_4vectors_DUNE_bkg.txt")
    bkg_1ep_numu = Background1Particle(data_file_name="data/1g0p/1ep_numu_4vectors_DUNE_bkg.txt")
    bkg_1ep_numubar = Background1Particle(data_file_name="data/1g0p/1ep_numubar_4vectors_DUNE_bkg.txt")
    bkg_1ep_nuebar = Background1Particle(data_file_name="data/1g0p/1ep_nuebar_4vectors_DUNE_bkg.txt")

    bkg_1em_nue = Background1Particle(data_file_name="data/1g0p/1em_nue_4vectors_DUNE_bkg.txt")
    bkg_1em_numu = Background1Particle(data_file_name="data/1g0p/1em_numu_4vectors_DUNE_bkg.txt")
    bkg_1em_numubar = Background1Particle(data_file_name="data/1g0p/1em_numubar_4vectors_DUNE_bkg.txt")
    bkg_1em_nuebar = Background1Particle(data_file_name="data/1g0p/1em_nuebar_4vectors_DUNE_bkg.txt")


    energy_bins_ana = np.logspace(np.log10(30.0), np.log10(15.0e3), 25)
    energy_bins_ana = np.append(energy_bins_ana, 120.0e3)
    binned_bkg_nue, bins_nue = np.histogram(p0_g_nue, weights=weights1_nue_g, bins=energy_bins_ana)
    binned_bkg_numu, bins_numu = np.histogram(p0_g_numu, weights=weights1_numu_g, bins=energy_bins_ana)
    binned_bkg_nuebar, bins_nuebar = np.histogram(p0_g_nuebar, weights=weights1_nuebar_g, bins=energy_bins_ana)
    binned_bkg_numubar, bins_numubar = np.histogram(p0_g_numubar, weights=weights1_numubar_g, bins=energy_bins_ana)

    binned_bkg_nue_ep, bins_nue = np.histogram(p0_ep_nue, weights=weights1_nue_ep, bins=energy_bins_ana)
    binned_bkg_numu_ep, bins_numu = np.histogram(p0_ep_numu, weights=weights1_numu_ep, bins=energy_bins_ana)
    binned_bkg_nuebar_ep, bins_nuebar = np.histogram(p0_ep_nuebar, weights=weights1_nuebar_ep, bins=energy_bins_ana)
    binned_bkg_numubar_ep, bins_numubar = np.histogram(p0_ep_numubar, weights=weights1_numubar_ep, bins=energy_bins_ana)

    binned_bkg_nue_em, bins_nue = np.histogram(p0_em_nue, weights=weights1_nue_em, bins=energy_bins_ana)
    binned_bkg_numu_em, bins_numu = np.histogram(p0_em_numu, weights=weights1_numu_em, bins=energy_bins_ana)
    binned_bkg_nuebar_em, bins_nuebar = np.histogram(p0_em_nuebar, weights=weights1_nuebar_em, bins=energy_bins_ana)
    binned_bkg_numubar_em, bins_numubar = np.histogram(p0_em_numubar, weights=weights1_numubar_em, bins=energy_bins_ana)


    binned_bkg_aggregate = binned_bkg_nuebar + binned_bkg_nue + binned_bkg_numu + binned_bkg_numubar \
                        + binned_bkg_nue_ep + binned_bkg_numu_ep + binned_bkg_nuebar_ep + binned_bkg_numubar_ep \
                        + binned_bkg_nue_em + binned_bkg_numu_em + binned_bkg_nuebar_em + binned_bkg_numubar_em
    binned_bkg_aggregate += 1.0/np.sum(binned_bkg_aggregate)  # add a total of one event to ensure each bin is populated


    # get backgrounds with angular cut
    nue_1g_acut = bkg_1g_nue.theta_z_deg<1.0
    numu_1g_acut = bkg_1g_numu.theta_z_deg<1.0
    nuebar_1g_acut = bkg_1g_nuebar.theta_z_deg<1.0
    numubar_1g_acut = bkg_1g_numubar.theta_z_deg<1.0
    nue_1ep_acut = bkg_1ep_nue.theta_z_deg<1.0
    numu_1ep_acut = bkg_1ep_numu.theta_z_deg<1.0
    nuebar_1ep_acut = bkg_1ep_nuebar.theta_z_deg<1.0
    numubar_1ep_acut = bkg_1ep_numubar.theta_z_deg<1.0
    nue_1em_acut = bkg_1em_nue.theta_z_deg<1.0
    numu_1em_acut = bkg_1em_numu.theta_z_deg<1.0
    nuebar_1em_acut = bkg_1em_nuebar.theta_z_deg<1.0
    numubar_1em_acut = bkg_1em_numubar.theta_z_deg<1.0

    binned_bkg_nue_acut, bins_nue = np.histogram(p0_g_nue, weights=weights1_nue_g*nue_1g_acut, bins=energy_bins_ana)
    binned_bkg_numu_acut, bins_numu = np.histogram(p0_g_numu, weights=weights1_numu_g*numu_1g_acut, bins=energy_bins_ana)
    binned_bkg_nuebar_acut, bins_nuebar = np.histogram(p0_g_nuebar, weights=weights1_nuebar_g*nuebar_1g_acut, bins=energy_bins_ana)
    binned_bkg_numubar_acut, bins_numubar = np.histogram(p0_g_numubar, weights=weights1_numubar_g*numubar_1g_acut, bins=energy_bins_ana)

    binned_bkg_nue_ep_acut, bins_nue = np.histogram(p0_ep_nue, weights=weights1_nue_ep*nue_1ep_acut, bins=energy_bins_ana)
    binned_bkg_numu_ep_acut, bins_numu = np.histogram(p0_ep_numu, weights=weights1_numu_ep*numu_1ep_acut, bins=energy_bins_ana)
    binned_bkg_nuebar_ep_acut, bins_nuebar = np.histogram(p0_ep_nuebar, weights=weights1_nuebar_ep*nuebar_1ep_acut, bins=energy_bins_ana)
    binned_bkg_numubar_ep_acut, bins_numubar = np.histogram(p0_ep_numubar, weights=weights1_numubar_ep*numubar_1ep_acut, bins=energy_bins_ana)

    binned_bkg_nue_em_acut, bins_nue = np.histogram(p0_em_nue, weights=weights1_nue_em*nue_1em_acut, bins=energy_bins_ana)
    binned_bkg_numu_em_acut, bins_numu = np.histogram(p0_em_numu, weights=weights1_numu_em*numu_1em_acut, bins=energy_bins_ana)
    binned_bkg_nuebar_em_acut, bins_nuebar = np.histogram(p0_em_nuebar, weights=weights1_nuebar_em*nuebar_1em_acut, bins=energy_bins_ana)
    binned_bkg_numubar_em_acut, bins_numubar = np.histogram(p0_em_numubar, weights=weights1_numubar_em*numubar_1em_acut, bins=energy_bins_ana)

    binned_bkg_acut_aggregate = binned_bkg_nuebar_acut + binned_bkg_nue_acut + binned_bkg_numu_acut + binned_bkg_numubar_acut \
                        + binned_bkg_nue_ep_acut + binned_bkg_numu_ep_acut + binned_bkg_nuebar_ep_acut + binned_bkg_numubar_ep_acut \
                        + binned_bkg_nue_em_acut + binned_bkg_numu_em_acut + binned_bkg_nuebar_em_acut + binned_bkg_numubar_em_acut
    binned_bkg_acut_aggregate += 1.0/np.sum(binned_bkg_aggregate)  # add a total of one event to ensure each bin is populated

    print("\nLOOP START: writing sensitivity loop to {}", out_file_name)

    # rescaling factor to get the scattering events in per second
    scatter_rescale = DUNE_POT_PER_YEAR/S_PER_DAY/365

    # begin sensitivity loop
    for ma in ma_list:
        
        if resume:
            if ma < last_ma:
                continue
            if last_g == gagamma_list[-1]:
                resumed = True
                continue
        
        flux_gen = PrimakoffFromBeam4Vectors(forward_photon_flux, target=Material("C"), det_dist=DUNE_DIST, det_length=DUNE_LENGTH,
                            det_area=DUNE_AREA, axion_mass=ma, axion_coupling=1.0, n_samples=1)

        print("Simulating and propagating ALP flux for ma={}...".format(ma))
        flux_gen.simulate(multicore=False)

        print("Propagating for coupling array...")
        for i, g in enumerate(gagamma_list):

            if resume and not resumed:
                if g <= last_g:
                    continue
                else:
                    print("RESUMING SCAN on g={}, ma={}".format(g, ma))
                    resumed = True

            flux_gen.propagate(is_isotropic=False, new_coupling=g)

            alp_flux_energies = np.array(flux_gen.axion_energy)
            alp_flux_angles = np.array(flux_gen.axion_angle)
            alp_flux_wgt = flux_gen.decay_axion_weight

            # Compute scattering contribution
            # have to rescale the flux to per second for the scattering contribution
            evt_gen = PhotonEventGenerator(flux_gen, detector=Material("Ar"))
            evt_gen.inverse_primakoff(g, ma, ntargets=DUNE_NTARGETS, days_exposure=scatter_rescale*EXPOSURE_DAYS, threshold=DUNE_THRESH)
            scatter_events, signal_bins = np.histogram(evt_gen.axion_energy, weights=evt_gen.scatter_weights, bins=energy_bins_ana)

            # Decay the 4-vectors
            print("decaying 4-vectors...")
            flux_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_wgt]).transpose()
            gamma1_energy, gamma2_energy, gamma1_theta_z, gamma2_theta_z, inv_mass, \
                total_energy, sep_angles, event_weights = decay_alp_gen(input_flux=flux_array, ALP_MASS=ma, resolved=False)
            
            decay_events, signal_bins = np.histogram(total_energy, weights=event_weights, bins=energy_bins_ana)

            signal_events = decay_events + scatter_events

            print("Finished event sim, found {} decay events and {} scatter events".format(np.sum(decay_events), np.sum(scatter_events)))
            # calculate the Poisson log-likelihood: null hypothesis is background only
            ll = np.sum(binned_bkg_aggregate * log(signal_events+binned_bkg_aggregate) - signal_events \
                        - binned_bkg_aggregate - gammaln(binned_bkg_aggregate))  # Poisson log-likelihood
            

            # apply angular cut
            rad2deg = np.pi/180.0
            decay_events_acut, signal_bins_acut = np.histogram(total_energy, weights=event_weights*(gamma1_theta_z < 1.0*rad2deg), bins=energy_bins_ana)
            signal_events_acut = decay_events_acut + scatter_events
            ll_acut = np.sum(binned_bkg_acut_aggregate * log(signal_events_acut+binned_bkg_acut_aggregate) - signal_events_acut \
                        - binned_bkg_acut_aggregate - gammaln(binned_bkg_acut_aggregate))  # Poisson log-likelihood
            
            file_out = open(out_file_name, "a")
            file_out.write(str(ma) + " " + str(g) + " " + str(ll) + " " + str(ll_acut) + " " + str(np.sum(signal_events)) + '\n')
            file_out.close()


def output_alp_4vectors(ma, g, save_file_name="signal_data/2gamma/alp_2gamma_decays_ma-1MeV_gagamma-17e-9.txt"):

    # ma=10.0, g=1.8e-9
    # ma=1.0, g=1.75e-8
    # ma=100.0, g=4.4e-10

    flux_gen = PrimakoffFromBeam4Vectors(forward_photon_flux, target=Material("C"), det_dist=DUNE_DIST, det_length=DUNE_LENGTH,
                            det_area=DUNE_AREA, axion_mass=ma, axion_coupling=g, n_samples=1)

    flux_gen.simulate(multicore=True)
    flux_gen.propagate(is_isotropic=False, new_coupling=g)

    alp_flux_energies = np.array(flux_gen.axion_energy)[flux_gen.decay_axion_weight>0]
    alp_flux_angles = np.array(flux_gen.axion_angle)[flux_gen.decay_axion_weight>0]
    alp_flux_wgt = flux_gen.decay_axion_weight[flux_gen.decay_axion_weight>0]

    # Decay the 4-vectors
    print("decaying 4-vectors...")
    flux_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_wgt]).transpose()
    p0_1, p1_1, p2_1, p3_1, p0_2, p1_2, p2_2, p3_2, wgts = decay_alp_gen(input_flux=flux_array, ALP_MASS=ma, resolved=True, return_4vs=True)
    
    out_array = np.array([p0_1, p1_1, p2_1, p3_1, p0_2, p1_2, p2_2, p3_2, wgts]).transpose()

    np.savetxt(save_file_name, out_array)



def main():
    #check_mass_cut_efficiency(500.0)
    
    #output_alp_4vectors(ma=100.0, g=4.4e-10, save_file_name="signal_data/2gamma/alp_2gamma_decays_ma-100MeV_gagamma-44e-11.txt")
    
    run_sens(out_file_name="sensitivities/singlephoton_sensitivity_BKG_cutflow_20240319.txt", resume=True)


if __name__ == "__main__":
    main()

