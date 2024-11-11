# Simulation for dark bosons coupled to quarks produced from pi0, pi+/- in meson decays
# for solution to the MiniBooNE anomaly


import sys
sys.path.append("../")
from alplib.fluxes import *
from alplib.generators import DarkPrimakoffGenerator
from alplib.fit import *
from alplib.charged_meson_3body import charged_meson_flux_mc
from alplib.couplings import *


import numpy as np
from scipy.stats import chisquare

import matplotlib.pyplot as plt
from matplotlib.pylab import rc
import matplotlib.font_manager
rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
rc('text', usetex=True)



# BNB BEAM DATA for DUNE
DUNE_target_pot_nu = 6.80e20
target_pi0_wgt_nu = DUNE_target_pot_nu / 1e6  # changed 1e-4 --> 1e6 20241003
pip_flux_DUNE = charged_meson_flux_mc("pi_plus", 0.0, 6.5, 0.03, 0.21, n_samples=60000, n_pot=DUNE_target_pot_nu/3)
# dividing by 3 to account for lower microboone acceptance by factor 3
bnb_target_pi0 = np.genfromtxt("data/mb_target_mode/bnb_pi_zero.txt")
bnb_target_pi0[:,:] *= 1e3
bnb_target_pi0[:,3] += M_PI0

DUNE_DIST = 574  # meters
DUNE_MASS = 50000.0  # kg
DUNE_THRESH = 30.0  # MeV
DUNE_AREA = 7.0*5.0  # m^2
DUNE_LENGTH = 3.0  # meters
DUNE_NE = DUNE_LENGTH * 100 * 1.39 * 6.022e23 / 40  # number of electrons per unit area in liquid argon
DUNE_bins = np.logspace(np.log10(30.0), 4)
DUNE_bins = np.append(DUNE_bins, 120000.0)  # add one large bin for high energy bkg-free region
DUNE_det = Material("Ar")

# Define model list
model_dict = {
# Fields: model_name, mediator_type, [mlepton], fixed_mediator_mass
            "SIB1e": ["scalar_ib1", "V", [M_E], None],
            "SIB1mu": ["scalar_ib1", "V", [M_MU], None],
            "PIB1e": ["pseudoscalar_ib1", "V", [M_E], None],
            "PIB1mu": ["pseudoscalar_ib1", "V", [M_MU], None],
            "VIB1e_SMediator": ["vector_ib1", "S", [M_E], None],
            "VIB1mu_SMediator": ["vector_ib1", "S", [M_MU], None],
            "VIB1e_PMediator": ["vector_ib1", "P", [M_E], None],
            "VIB1mu_PMediator": ["vector_ib1", "P", [M_MU], None],
            "VIB1e_Pi0Mediator": ["vector_ib1", "Pi0", [M_E], M_PI0],
            "VIB1mu_Pi0Mediator": ["vector_ib1", "Pi0", [M_MU], M_PI0],
            "VIB2_SMediator": ["vector_ib2", "S", [M_E, M_MU], None],
            "VIB2_PMediator": ["vector_ib2", "P", [M_E, M_MU], None],
            "VIB2_Pi0Mediator": ["vector_ib2", "Pi0", [M_E, M_MU], M_PI0],
            "VIB3_SMediator": ["vector_contact", "S", [M_E, M_MU], None],
            "VIB3_PMediator": ["vector_contact", "P", [M_E, M_MU], None],
            "VIB3_Pi0Mediator": ["vector_contact", "Pi0", [M_E, M_MU], M_PI0],
}



efficiency_data = np.genfromtxt("data/microboone/microboone_1g0p_efficiency.txt", delimiter=",")
efficiency_data[:,0] *= 1e3
efficiency_data[:,1] *= 0.0529 * 1e-2 # data is normalized to 100, so divide by 100 and multiply by 0.05 to get total efficiency = 5%
microboone_eff = Efficiency(efficiency_data)




################ mu B o o N E  Pi+ Events ##################
def simulate_mub_pip_events(m_x, m_y, g_pm, g_pi0, g_N, model_name, alphaBetaRatio=0.0, verbose=False):
    model_params = model_dict[model_name]
    if verbose:
        print("Initializing piplus flux")
    three_body_gen = FluxChargedMeson3BodyDecay(meson_flux=pip_flux_DUNE, boson_mass=m_x, coupling=g_pm, interaction_model=model_params[0],
                                                energy_cut=DUNE_THRESH, det_dist=DUNE_DIST, det_area=DUNE_AREA, det_length=DUNE_LENGTH,
                                                n_samples=1, c0=alphaBetaRatio, lepton_masses=model_params[2])

    three_body_gen.simulate(cut_on_solid_angle=True)
    three_body_gen.propagate()

    scatter_gen = DarkPrimakoffGenerator(three_body_gen, detector=DUNE_det, n_samples=1, mediator=model_params[1])

    lam0 = g_pi0*sqrt(4*pi*ALPHA)/(4*pi*F_PI)
    pip_evis, pip_e_weights, pip_cos, pip_cos_wgts = scatter_gen.get_weights(lam=lam0, gphi=g_N, mphi=m_y, n_e=DUNE_NE)
    evis_signal_shape = np.histogram(pip_evis, weights=pip_e_weights, bins=DUNE_bins)[0]
    return evis_signal_shape



################ mu B o o N E  Pi0 Events ##################
def simulate_mub_target_pi0_events(m_x, m_y, g_pi0, g_N, model_name, verbose=False, n_samples=1):
    model_params = model_dict[model_name]
    if verbose:
        print("Initializeing Pi0 fluxes")
    target_flux = FluxNeutralMeson2BodyDecay(bnb_target_pi0, flux_weight=target_pi0_wgt_nu, boson_mass=m_x,
                                coupling=g_pi0, n_samples=n_samples, apply_angle_cut=False,
                                det_dist=DUNE_DIST, det_area=DUNE_AREA, det_length=DUNE_LENGTH)
    if verbose:
        print("simulating target flux...")
    target_flux.simulate()
    target_flux.propagate()
    
    if verbose:
        print("Nstats = ", len(target_flux.axion_flux))
        print("generating events...")
    target_gen = DarkPrimakoffGenerator(target_flux, detector=DUNE_det, n_samples=1, mediator=model_params[1])

    lam0 = g_pi0*sqrt(4*pi*ALPHA)/(4*pi*F_PI)
    target_evis, target_e_wgts, target_cos, target_cos_wgts = target_gen.get_weights(lam=lam0, gphi=g_N, mphi=m_y, n_e=DUNE_NE)
    nu_evis_signal = np.histogram(target_evis, weights=target_e_wgts, bins=DUNE_bins)[0]
    return nu_evis_signal




# Single-mediator generators

def simulate_mub_pipm_events_2Med(m_x, m_y, g_pm, lam0, g_N, model_name, verbose=False, meson_flux=pip_flux_DUNE):
    model_params = model_dict[model_name]
    if verbose:
        print("Initializing piplus flux")
    three_body_gen = FluxChargedMeson3BodyDecay(meson_flux=meson_flux, boson_mass=m_x, coupling=g_pm, interaction_model=model_params[0],
                                                energy_cut=DUNE_THRESH, det_dist=DUNE_DIST, det_area=DUNE_AREA, det_length=DUNE_LENGTH,
                                                n_samples=5, c0=0.0, lepton_masses=model_params[2])

    three_body_gen.simulate(cut_on_solid_angle=True)
    three_body_gen.propagate()

    scatter_gen = DarkPrimakoffGenerator(three_body_gen, detector=DUNE_det, n_samples=1, mediator=model_params[1])

    pip_evis, pip_e_weights, pip_cos, pip_cos_wgts = scatter_gen.get_weights(lam=lam0, gphi=g_N, mphi=m_y, n_e=DUNE_NE, eff=microboone_eff)
    evis_signal_shape = np.histogram(pip_evis, weights=pip_e_weights, bins=DUNE_bins)[0]
    return evis_signal_shape
