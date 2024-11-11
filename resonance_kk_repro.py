import sys
sys.path.append("../")

from alplib.fluxes import *
from alplib.generators import *


import matplotlib.pyplot as plt
from matplotlib.pylab import rc
from matplotlib.colors import LogNorm
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


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



# Import signal fluxes
positron_diff_flux = np.genfromtxt("../DUNE/data/epem_flux/positron_DIFF_flux_dPhidE_20210621_TargetSim_QGSP_BIC_AllHP_POT1E6.txt")
positron_diff_flux[:,1] *= 1/pot_per_sample
electron_diff_flux = np.genfromtxt("../DUNE/data/epem_flux/electron_DIFF_flux_dPhidE_20210621_TargetSim_QGSP_BIC_AllHP_POT1E6.txt")
electron_diff_flux[:,1] *= 1/pot_per_sample
positron_flux = np.genfromtxt("../DUNE/data/epem_flux/positron_INT_flux_dPhidE_20210621_TargetSim_QGSP_BIC_AllHP_POT1E6.txt")
positron_flux[:,1] *= 1/pot_per_sample
electron_flux = np.genfromtxt("../DUNE/data/epem_flux/electron_INT_flux_dPhidE_20210621_TargetSim_QGSP_BIC_AllHP_POT1E6.txt")
electron_flux[:,1] *= 1/pot_per_sample


photon_flux = np.genfromtxt("../DUNE/data/photon_flux/geant4_flux_DUNE.txt", delimiter=",")
photon_flux[:,1] *= 1/pot_per_sample




dune_target = Material("C")


def plot_flux(ma, gae):
    #compton_flux = FluxComptonIsotropic(photon_flux=photon_flux, target=dune_target,
    #                                det_dist=det_dist, det_length=det_length, det_area=det_area,
    #                                axion_mass=TEST_MA, axion_coupling=TEST_GAE, n_samples=10000)
    brem_flux = FluxBremIsotropic(electron_flux, positron_flux, target=dune_target,
                                    det_dist=det_dist, det_length=det_length, det_area=det_area,
                                    axion_mass=ma, axion_coupling=gae, n_samples=2000, is_isotropic=False)
    resonant_flux = FluxResonanceIsotropic(positron_diff_flux, target=dune_target,
                                            det_dist=det_dist, det_length=det_length, det_area=det_area,
                                            axion_mass=ma, axion_coupling=gae, n_samples=10000,
                                            is_isotropic=False, boson_type="vector")

    resonant_flux.simulate()
    brem_flux.simulate()
    #compton_flux.simulate()

    energy_bins = np.logspace(-2, 2, 25)
    plt.hist(1e-3*np.array(resonant_flux.axion_energy), weights=resonant_flux.axion_flux, bins=energy_bins, histtype='step', label="res")
    #plt.hist(1e-3*np.array(compton_flux.axion_energy), weights=compton_flux.axion_flux, bins=energy_bins, histtype='step', label="compton")
    plt.hist(1e-3*np.array(brem_flux.axion_energy), weights=brem_flux.axion_flux, bins=energy_bins, histtype='step', label="brem")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r"ALPs / POT / $g_{ae}^2$", fontsize=14)
    plt.xlabel("$E_a$ [GeV]", fontsize=14)
    plt.title(r"$m_{a} = 10$ MeV", loc="right")
    plt.legend()
    plt.show()



def plot_flux_by_mass():
    petite_res_dune_rates = np.genfromtxt("resonance_kk_repro/petite_resonance_dune_Nv_vs_mV.txt")
    petite_brem_dune_rates = np.genfromtxt("resonance_kk_repro/petite_brem_dune_Nv_vs_mV.txt")
    petite_comp_dune_rates = np.genfromtxt("resonance_kk_repro/petite_compton_dune_Nv_vs_mV.txt")

    mass_list = np.logspace(0, 3, 40)

    res_flux_list = []
    res_vec_flux_list = []
    assoc_flux_list = []
    brem_flux_list = []
    brem_v_flux_list = []
    comp_flux_list = []

    for ma in mass_list:
        print("simulating res for ma = ", ma)

        # pseudoscalar fluxes
        resonant_flux = FluxResonanceIsotropic(positron_diff_flux, target=dune_target,
                                            det_dist=det_dist, det_length=det_length, det_area=det_area,
                                            axion_mass=ma, axion_coupling=1.0, n_samples=100000,
                                            is_isotropic=False, boson_type="pseudoscalar")
        compton_flux = FluxComptonIsotropic(photon_flux=photon_flux, target=dune_target,
                                    det_dist=det_dist, det_length=det_length, det_area=det_area,
                                    axion_mass=ma, axion_coupling=1.0, n_samples=1000, is_isotropic=False)
        brem_flux = FluxBremIsotropic(electron_flux, positron_flux, target=dune_target,
                                    det_dist=det_dist, det_length=det_length, det_area=det_area,
                                    axion_mass=ma, axion_coupling=1.0, n_samples=7000,
                                    is_isotropic=False, boson_type="pseudoscalar")
        assoc_flux = FluxPairAnnihilationIsotropic(positron_flux, dune_target, det_dist=det_dist,
                                            det_length=det_length, det_area=det_area, axion_mass=ma, axion_coupling=0.3,
                                            n_samples=7000, is_isotropic=False)
        
        # vector fluxes
        brem_flux_v = FluxBremIsotropic(electron_flux, positron_flux, target=dune_target,
                                    det_dist=det_dist, det_length=det_length, det_area=det_area,
                                    axion_mass=ma, axion_coupling=0.3, n_samples=500000,
                                    is_isotropic=False, boson_type="vector", max_track_length=5.0)
        resonant_flux_v = FluxResonanceIsotropic(positron_diff_flux, target=dune_target,
                                            det_dist=det_dist, det_length=det_length, det_area=det_area,
                                            axion_mass=ma, axion_coupling=0.3, n_samples=2000000,
                                            is_isotropic=False, boson_type="vector", max_track_length=5.0)

        print("simulating res...")
        resonant_flux.simulate()
        resonant_flux_v.simulate()
        print("simulating compton...")
        compton_flux.simulate()
        print("simulating brem...")
        brem_flux_v.simulate(use_track_length=True)
        brem_flux.simulate(use_track_length=True)
        print("simulating pair annih...")
        assoc_flux.simulate()

        # pot scale: 1.4e22 POT

        res_flux_list.append(np.sum(resonant_flux.axion_flux))
        res_vec_flux_list.append(np.sum(resonant_flux_v.axion_flux)/0.3**2)
        assoc_flux_list.append(np.sum(assoc_flux.axion_flux))
        comp_flux_list.append(np.sum(compton_flux.axion_flux))
        brem_flux_list.append(np.sum(brem_flux.axion_flux))
        brem_v_flux_list.append(np.sum(brem_flux_v.axion_flux)/0.3**2)
    
    # PLOT OUR RATES
    plt.plot(mass_list, res_flux_list, label=r"Resonant $e^+ e^- \to a$ [alplib]", color='r')
    plt.plot(mass_list, res_vec_flux_list, label=r"Resonant $e^+ e^- \to V$ [alplib]", color='mediumpurple')
    plt.plot(mass_list, assoc_flux_list, label=r"Assoc. $e^+ e^- \to a \gamma$ [alplib]", color='orange')
    plt.plot(mass_list, comp_flux_list, label=r"Compton $\gamma e^- \to a e^-$ [alplib]", color='g')
    plt.plot(mass_list, brem_flux_list, label=r"Brem $e^\pm Z \to e^\pm Z a$ [alplib]", color='royalblue')
    plt.plot(mass_list, brem_v_flux_list, label=r"Brem $e^\pm Z \to e^\pm Z V$ [alplib]", color="k")
    
    # Compare with PETITE
    plt.plot(petite_res_dune_rates[:,0], petite_res_dune_rates[:,1], label=r"PETITE $e^+ e^- \to V$", color='mediumpurple', ls='dashed')
    plt.plot(petite_brem_dune_rates[:,0], petite_brem_dune_rates[:,1], label="PETITE Vector brem", color="k", ls='dashed')
    plt.plot(petite_comp_dune_rates[:,0], petite_comp_dune_rates[:,1], label="PETITE Compton", color='g', ls='dashed')


    plt.ylabel(r"$N_V$ / ($1.4 \cdot 10^{22}$ POT) / $g^2$", fontsize=16)
    plt.xlabel(r"$m_V$ [MeV]", fontsize=16)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.ylim((1e16, 1e25))
    #plt.xlim((10,1e4))
    plt.title(r"DUNE ND ($t_{max} = 1$)", loc="right")
    plt.legend()
    plt.show()
    plt.close()

    ######

    # PLOT OUR RATES
    plt.plot(mass_list, res_flux_list, label=r"Resonant Production $e^+ e^- \to a$", color='r')
    plt.plot(mass_list, assoc_flux_list, label=r"Associated Production $e^+ e^- \to a \gamma$", color='orange')
    plt.plot(mass_list, comp_flux_list, label=r"Compton Scattering $\gamma e^- \to a e^-$", color='g')
    plt.plot(mass_list, brem_flux_list, label=r"Bremsstrahlung $e^\pm Z \to e^\pm Z a$", color='royalblue')

    plt.ylabel(r"$N_V$ / POT / $g^2$", fontsize=16)
    plt.xlabel(r"$m_V$ [MeV]", fontsize=16)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.ylim((1e16, 1e25))
    plt.xlim((mass_list[0],mass_list[-1]))
    plt.title(r"DUNE ND ($t_{max} = 1$)", loc="right")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_flux_by_mass_isotropic():
    # Get the fluxes
    pot_per_sample = 1e5
    positron_diff_flux = np.genfromtxt("../CCMAxion/data/ccm_positron_diff_flux_1e5_POT_QGSP_BIC_HP.txt")
    positron_diff_flux[:,1] *= 1/pot_per_sample  # s^-1
    positron_diff_flux[:,0] += M_E

    positron_flux = np.genfromtxt("../CCMAxion/data/ccm_positron_flux_1e5_POT_QGSP_BIC_HP.txt", delimiter=",")
    positron_flux[:,1] *= 1/pot_per_sample  # s^-1
    positron_flux[:,0] += M_E

    electron_diff_flux = np.genfromtxt("../CCMAxion/data/ccm_electron_diff_flux_1e5_POT_QGSP_BIC_HP.txt")
    electron_diff_flux[:,1] *= 1/pot_per_sample  # s^-1
    electron_diff_flux[:,0] += M_E

    electron_flux = np.genfromtxt("../CCMAxion/data/ccm_electron_flux_1e5_POT_QGSP_BIC_HP.txt", delimiter=",")
    electron_flux[:,1] *= 1/pot_per_sample  # s^-1
    electron_flux[:,0] += M_E

    # Get the CCM photon flux
    photon_flux = np.genfromtxt("../CCMAxion/data/ccm_photon_energy_flux_1e5_POT_full.txt")
    photon_flux[:,1] *= 1/pot_per_sample  # converts to /s

    petite_res_dune_rates = np.genfromtxt("resonance_kk_repro/petite_resonance_dune_Nv_vs_mV.txt")
    petite_brem_dune_rates = np.genfromtxt("resonance_kk_repro/petite_brem_dune_Nv_vs_mV.txt")

    mass_list = np.logspace(0.1, 2.5, 30)

    res_flux_list = []
    res_vec_flux_list = []
    assoc_flux_list = []
    brem_flux_list = []
    comp_flux_list = []

    det_dist = 20.0
    det_length = 1.0


    for ma in mass_list:
        print("simulating res for ma = ", ma)
        resonant_flux = FluxResonanceIsotropic(positron_diff_flux, target=dune_target,
                                            det_dist=det_dist, det_length=det_length, det_area=det_area,
                                            axion_mass=ma, axion_coupling=0.3, n_samples=100000,
                                            is_isotropic=True, boson_type="pseudoscalar")
        resonant_flux_v = FluxResonanceIsotropic(positron_diff_flux, target=dune_target,
                                            det_dist=det_dist, det_length=det_length, det_area=det_area,
                                            axion_mass=ma, axion_coupling=0.3, n_samples=100000,
                                            is_isotropic=True, boson_type="vector")
        compton_flux = FluxComptonIsotropic(photon_flux=photon_flux, target=dune_target,
                                    det_dist=det_dist, det_length=det_length, det_area=det_area,
                                    axion_mass=ma, axion_coupling=0.3, n_samples=10, is_isotropic=True)
        brem_flux = FluxBremIsotropic(electron_flux, positron_flux, target=dune_target,
                                    det_dist=det_dist, det_length=det_length, det_area=det_area,
                                    axion_mass=ma, axion_coupling=0.3, n_samples=2000, is_isotropic=True)
        assoc_flux = FluxPairAnnihilationIsotropic(positron_flux, dune_target, det_dist=det_dist,
                                            det_length=det_length, det_area=det_area, axion_mass=ma, axion_coupling=0.3,
                                            n_samples=2000, is_isotropic=True)

        resonant_flux.simulate()
        print("simulating res v...")
        resonant_flux_v.simulate()
        print("simulating compton...")
        compton_flux.simulate()
        print("simulating brem...")
        brem_flux.simulate()
        print("simulating pair annih...")
        assoc_flux.simulate()

        res_flux_list.append(np.sum(resonant_flux.axion_flux))
        res_vec_flux_list.append(np.sum(resonant_flux_v.axion_flux))
        assoc_flux_list.append(np.sum(assoc_flux.axion_flux))
        comp_flux_list.append(np.sum(compton_flux.axion_flux))
        brem_flux_list.append(np.sum(brem_flux.axion_flux))
    

    plt.plot(mass_list, res_flux_list, label=r"Resonant $e^+ e^- \to a$ [alplib]")
    plt.plot(mass_list, res_vec_flux_list, label=r"Resonant $e^+ e^- \to V$ [alplib]")
    plt.plot(mass_list, assoc_flux_list, label=r"Assoc. $e^+ e^- \to a \gamma$ [alplib]")
    plt.plot(mass_list, comp_flux_list, label=r"Compton $\gamma e^- \to a e^-$ [alplib]")
    plt.plot(mass_list, brem_flux_list, label=r"Brem $e^\pm Z \to e^\pm Z a$ [alplib]")
    
    # Compare with PETITE
    #plt.plot(petite_res_dune_rates[:,0], petite_res_dune_rates[:,1], label="PETITE")

    plt.ylabel(r"$N_V$ / POT", fontsize=16)
    plt.xlabel(r"$m_V$ [MeV]", fontsize=16)
    plt.yscale('log')
    plt.xscale('log')
    plt.title("800 MeV on W, Isotropic Source", loc="right")
    plt.xlim((2*M_E, max(mass_list)))
    plt.legend()
    plt.show()





def main():
    plot_flux_by_mass()
    #plot_flux_by_mass_isotropic()

if __name__ == "__main__":
    main()
