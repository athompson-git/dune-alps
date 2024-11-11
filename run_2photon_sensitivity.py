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


def run_sens(out_file_name, resume=False, dump_fluxes=False):
    # Start with loop over parameters (MeV)
    gagamma_list = np.logspace(-14, 0, 200)
    ma_list = np.logspace(0, 3, 100)


    # Import flux info
    # Grab the photon flux below 100 mrad
    print("Reading in photon flux...")
    photon_flux_sub100mrad = np.genfromtxt("data/fluxes/dune_target_gamma_1e5POT_2d_sub100mrad_june24.txt")

    pot_per_sample = 1e5
    photon_flux_sub100mrad[:,2] *= 1/pot_per_sample  # converts to per POT
    angle_cut = DUNE_SOLID_ANGLE
    forward_photon_flux = photon_flux_sub100mrad #np.array([photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,0],
                                    #photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,1],
                                    #photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,2]]).transpose()


    # Create background histograms
    print("Loading backgrounds...")

    bkg_2g_nue = Background2Particle(data_file_name="data/2gamma/2gamma_nue_4vectors_DUNE_bkg.txt", nu_flavor="nue")
    bkg_2g_numu = Background2Particle(data_file_name="data/2gamma/2gamma_numu_4vectors_DUNE_bkg.txt", nu_flavor="numu")
    bkg_2g_numubar = Background2Particle(data_file_name="data/2gamma/2gamma_numubar_4vectors_DUNE_bkg.txt", nu_flavor="numubar")
    bkg_2g_nuebar = Background2Particle(data_file_name="data/2gamma/2gamma_nuebar_4vectors_DUNE_bkg.txt", nu_flavor="nuebar")

    energy_bins_ana = np.logspace(np.log10(30.0), np.log10(15.0e3), 25)
    energy_bins_ana = np.append(energy_bins_ana, 120.0e3)
    binned_bkg_nue, bins_nue = np.histogram(p0_2g_nue_1 + p0_2g_nue_2, weights=weights1_nue_2g, bins=energy_bins_ana)
    binned_bkg_numu, bins_numu = np.histogram(p0_2g_numu_1 + p0_2g_numu_2, weights=weights1_numu_2g, bins=energy_bins_ana)
    binned_bkg_nuebar, bins_nuebar = np.histogram(p0_2g_nuebar_1 + p0_2g_nuebar_2, weights=weights1_nuebar_2g, bins=energy_bins_ana)
    binned_bkg_numubar, bins_numubar = np.histogram(p0_2g_numubar_1 + p0_2g_numubar_2, weights=weights1_numubar_2g, bins=energy_bins_ana)

    binned_bkg_aggregate = binned_bkg_nuebar + binned_bkg_nue + binned_bkg_numu + binned_bkg_numubar

    # with angle cut
    bkg_nue_angle_cut_mask = bkg_2g_nue.dtheta_deg<20.0
    binned_bkg_nue_acut, bins_nue = np.histogram((p0_2g_nue_1 + p0_2g_nue_2),
                                                weights=weights1_nue_2g*bkg_nue_angle_cut_mask, bins=energy_bins_ana)
    bkg_numu_angle_cut_mask = bkg_2g_numu.dtheta_deg<20.0
    binned_bkg_numu_acut, bins_numu = np.histogram((p0_2g_numu_1 + p0_2g_numu_2),
                                                weights=weights1_numu_2g*bkg_numu_angle_cut_mask, bins=energy_bins_ana)
    bkg_nuebar_angle_cut_mask = bkg_2g_nuebar.dtheta_deg<20.0
    binned_bkg_nuebar_acut, bins_nuebar = np.histogram((p0_2g_nuebar_1 + p0_2g_nuebar_2),
                                                weights=weights1_nuebar_2g*bkg_nuebar_angle_cut_mask, bins=energy_bins_ana)
    bkg_numubar_angle_cut_mask = bkg_2g_numubar.dtheta_deg<20.0
    binned_bkg_numubar_acut, bins_numubar = np.histogram((p0_2g_numubar_1 + p0_2g_numubar_2),
                                                weights=weights1_numubar_2g*bkg_numubar_angle_cut_mask, bins=energy_bins_ana)
    binned_bkg_post_acut = binned_bkg_nuebar_acut + binned_bkg_nue_acut + binned_bkg_numu_acut + binned_bkg_numubar_acut
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
            if last_g == gagamma_list[-1]:
                resumed = True
                continue

        flux_gen = PrimakoffFromBeam4Vectors(forward_photon_flux, target=Material("C"), det_dist=DUNE_DIST, det_length=DUNE_LENGTH,
                            det_area=DUNE_AREA, axion_mass=ma, axion_coupling=1.0, n_samples=100)

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

            print("flux sum = ", np.sum(flux_gen.decay_axion_weight))

            alp_flux_energies = np.array(flux_gen.axion_energy)
            alp_flux_angles = np.array(flux_gen.axion_angle)
            alp_flux_wgt = flux_gen.decay_axion_weight

            # Decay the 4-vectors
            print("decaying 4-vectors...")
            flux_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_wgt]).transpose()

            if dump_fluxes:
                if np.sum(alp_flux_wgt) > 0.0:
                    p0_1, p1_1, p2_1, p3_1, p0_2, p1_2, p2_2, p3_2, event_weights \
                        = decay_alp_gen(input_flux=flux_array, ALP_MASS=ma, resolved=True, return_4vs=True)
                    out_array = np.array([p0_1, p1_1, p2_1, p3_1, p0_2, p1_2, p2_2, p3_2, event_weights]).transpose()
                    out_string = "signal_data/2gamma/alp_2gamma_decays_ma-{:.2E}_gagamma-{:.2E}.txt".format(Decimal(ma),Decimal(g))
                    print("SUM OF EVENT WEIGHTS = {} FOR ma-{:.2E}_gagamma-{:.2E} ".format(np.sum(event_weights), Decimal(ma), Decimal(g)))
                    if np.sum(event_weights) > 0.0:
                        np.savetxt(out_string, out_array)
            
            gamma1_energy, gamma2_energy, gamma1_theta_z, gamma2_theta_z, inv_mass, \
                total_energy, sep_angles, event_weights = decay_alp_gen(input_flux=flux_array, ALP_MASS=ma, resolved=True)
            # precuts
            signal_events, signal_bins = np.histogram(total_energy, weights=event_weights, bins=energy_bins_ana)

            # calculate the Poisson log-likelihood: null hypothesis is background only
            ll = np.sum(binned_bkg_aggregate * log(signal_events + binned_bkg_aggregate + 1) - signal_events \
                        - binned_bkg_aggregate - gammaln(binned_bkg_aggregate+1))  # Poisson log-likelihood
            
            # Apply cuts:
            # (1) angular cut DeltaTheta < 40 deg
            sig_angle_cut_mask = sep_angles<20.0*np.pi/180.0
            signal_events_angle_cut, signal_bins = np.histogram(total_energy[sig_angle_cut_mask],
                                                                weights=event_weights[sig_angle_cut_mask], bins=energy_bins_ana)

            print("Sig events before after acut = ", np.sum(signal_events_angle_cut), np.sum(signal_events))
            if np.sum(signal_events) > 0:
                sig_acut_eff = np.sum(signal_events_angle_cut)/np.sum(signal_events)
            else:
                sig_acut_eff = 1.0
            ll_post_acut = np.sum(binned_bkg_post_acut * log(signal_events_angle_cut+binned_bkg_post_acut+1) - signal_events_angle_cut \
                        - binned_bkg_post_acut - gammaln(binned_bkg_post_acut+1))  # Poisson log-likelihood

            # (2) inv_mass cut +/-10% of mass
            sig_mass_cut_mask = (inv_mass < 1.1*ma) * (inv_mass > 0.9*ma)
            #signal_events_mass_cut, signal_bins = np.histogram(total_energy[sig_angle_cut_mask*sig_mass_cut_mask],
             #                                                   weights=event_weights[sig_angle_cut_mask*sig_mass_cut_mask], bins=energy_bins_ana)
            
            bkg_nue_mass_cut_mask = abs(bkg_2g_nue.inv_mass - ma) < 0.025*ma
            binned_bkg_nue_mcut, bins_nue = np.histogram((p0_2g_nue_1 + p0_2g_nue_2),
                                                        weights=weights1_nue_2g*bkg_nue_angle_cut_mask*bkg_nue_mass_cut_mask, bins=energy_bins_ana)
            bkg_numu_mass_cut_mask = abs(bkg_2g_numu.inv_mass - ma) < 0.025*ma
            binned_bkg_numu_mcut, bins_numu = np.histogram((p0_2g_numu_1 + p0_2g_numu_2),
                                                        weights=weights1_numu_2g*bkg_numu_angle_cut_mask*bkg_numu_mass_cut_mask, bins=energy_bins_ana)
            bkg_nuebar_mass_cut_mask = abs(bkg_2g_nuebar.inv_mass - ma) < 0.025*ma
            binned_bkg_nuebar_mcut, bins_nuebar = np.histogram((p0_2g_nuebar_1 + p0_2g_nuebar_2),
                                                        weights=weights1_nuebar_2g*bkg_nuebar_angle_cut_mask*bkg_nuebar_mass_cut_mask, bins=energy_bins_ana)
            bkg_numubar_mass_cut_mask = abs(bkg_2g_numubar.inv_mass - ma) < 0.025*ma
            binned_bkg_numubar_mcut, bins_numubar = np.histogram((p0_2g_numubar_1 + p0_2g_numubar_2),
                                                        weights=weights1_numubar_2g*bkg_numubar_angle_cut_mask*bkg_numubar_mass_cut_mask, bins=energy_bins_ana)

            binned_bkg_post_mcut = binned_bkg_nuebar_mcut + binned_bkg_nue_mcut + binned_bkg_numu_mcut + binned_bkg_numubar_mcut
            
            print("After mass cut, bkg = ", binned_bkg_post_mcut)

            ll_post_mcut = np.sum(binned_bkg_post_mcut * log(signal_events_angle_cut+binned_bkg_post_mcut+1) - signal_events_angle_cut \
                        - binned_bkg_post_mcut - gammaln(binned_bkg_post_mcut+1))  # Poisson log-likelihood
    

            file_out = open(out_file_name, "a")
            file_out.write(str(ma) + " " + str(g) + " " + str(ll) + " " + str(sig_acut_eff) + " " + str(ll_post_acut) + \
                           " " + str(ll_post_mcut) + " " + str(np.sum(signal_events)) + '\n')
            file_out.close()




def plot_sensitivity(save_file):

    # read in the calculated dune sens
    dune_dat = np.genfromtxt(save_file)
    dune_ma = dune_dat[:,0]*1e6
    dune_g = dune_dat[:,1]*1e3
    dune_ll = 2*abs(max(dune_dat[:,2]) - dune_dat[:,2])
    dune_ll_post_cut = 2*abs(max(dune_dat[:,4]) - dune_dat[:,4])

    print(max(dune_ll), min(dune_ll))

    DUNE_MA, DUNE_G = np.meshgrid(np.unique(dune_ma),np.unique(dune_g))
    DUNE_CHI2 = np.reshape(dune_ll, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))
    DUNE_CHI2_POSTCUT = np.reshape(dune_ll_post_cut, (np.unique(dune_ma).shape[0],np.unique(dune_g).shape[0]))
    plt.contour(DUNE_MA, DUNE_G, DUNE_CHI2.transpose(), levels=[6.18], colors=["k"])
    plt.contour(DUNE_MA, DUNE_G, DUNE_CHI2_POSTCUT.transpose(), levels=[6.18], colors=["steelblue", "b", "cornflowerblue"])



    # Read in existing limits.
    beam = np.genfromtxt('../sharedata/existing_limits/beamv2.txt', delimiter=',')
    eeinva = np.genfromtxt('../DUNE/data/existing_limits/eeinva.txt')
    lep = np.genfromtxt('../DUNE/data/existing_limits/lep_v2.txt', delimiter=",")
    nomad = np.genfromtxt('../DUNE/data/existing_limits/nomad.txt')

    # Astrophyiscal limits
    hbstars_new = np.genfromtxt("../DUNE/data/existing_limits/hbstars_new.txt", delimiter=",")
    sn1987a = np.genfromtxt("../sharedata/existing_limits/sn1987a_updated.txt", delimiter=",")

    # Plot astrophysical limits
    astro_color = 'silver'
    plt.fill_between(hbstars_new[:,0], hbstars_new[:,1], y2=1.0, color=astro_color, alpha=0.5)
    plt.fill(sn1987a[:,0], sn1987a[:,1], color=astro_color, alpha=0.5)


    # Plot lab limits
    lab_color = 'rosybrown'
    cosw = 0.7771
    plt.fill_between(lep[:,0], (1/cosw)*lep[:,1], y2=1.0, color='teal', alpha=0.8)
    plt.fill(beam[:,0], beam[:,1], color='rosybrown', edgecolor='black', alpha=0.8)
    plt.fill(np.hstack((eeinva[:,0], np.min(eeinva[:,0]))), np.hstack((eeinva[:,1], np.max(eeinva[:,1]))),
            color='forestgreen', alpha=0.8)
    plt.fill(np.hstack((nomad[:,0], np.min(nomad[:,0]))), np.hstack((nomad[:,1], np.max(nomad[:,1]))),
            color='tan', alpha=0.8)


    # Plot QCD lines
    # g_ag/GeV = slope * ma/GeV
    hq_slope = 1.005
    hq_int1 = 1.58e-3
    hq_int2 = 1.64e-7

    def hq_axion_coupling(ma, bp="1"):
        if bp == "1":
            hq_int = hq_int1
        elif bp == "2":
            hq_int = hq_int2
        return 10**(hq_slope * np.log10(1e9*ma)) + hq_int

    masses_continuous = np.logspace(2, 9, 100)
    plt.plot(masses_continuous, hq_axion_coupling(masses_continuous, bp="1"), color='dimgray', ls='dashed')
    plt.plot(masses_continuous, hq_axion_coupling(masses_continuous, bp="2"), color='dimgray', ls='dotted')


    text_fs = 12
    #plt.text(2, 2e-4, "NOMAD", fontsize=text_fs)
    plt.text(650000, 2e-7, "SN1987a", fontsize=text_fs)
    plt.text(2e4, 2e-5, "HB Stars", fontsize=text_fs)
    plt.text(6e4, 5e-4, r'$e^+e^-\rightarrow inv.+\gamma$', fontsize=text_fs)
    #plt.text(4e8, 0.009, "LEP", fontsize=text_fs)
    plt.text(2e6, 5e-5, "Beam Dumps", fontsize=text_fs)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((1e4,1e9))
    plt.ylim(2e-8,1.0e-3)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('$m_a$ [eV]', fontsize=15)
    plt.ylabel('$g_{a\gamma}$ [GeV$^{-1}$]', fontsize=15)

    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.show()



def main():
    #check_mass_cut_efficiency(500.0)
    
    run_sens(out_file_name="sensitivities/digamma_sensitivity_BKG_cutflow_updated_fluxes.txt", resume=False, dump_fluxes=False)
    #plot_sensitivity("sensitivities/photon_coupling_sensitivity_BKG_cutflow_20240115.txt")


if __name__ == "__main__":
    main()

