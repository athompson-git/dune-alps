import sys
sys.path.append("../")

from alplib.fluxes import *
from alplib.generators import *
from alplib.cross_section_mc import *

from time import time

from dune_constants import *

pot_per_sample = 1e5

# Grab the photon flux below 100 mrad
photon_flux_sub100mrad = np.genfromtxt("data/fluxes/dune_target_gamma_1e5POT_2d_sub100mrad_june24.txt")

photon_flux_sub100mrad[:,2] *= 1/pot_per_sample  # converts to per POT

# perform angle cut OFF
angle_cut = DUNE_SOLID_ANGLE
#forward_photon_flux = np.array([photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,0],
#                                photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,1],
#                                photon_flux_sub100mrad[photon_flux_sub100mrad[:,1] <= angle_cut][:,2]]).transpose()
forward_photon_flux = photon_flux_sub100mrad

# electron/positron fluxes
pos_diff_flux_dune = np.genfromtxt("../DUNE/data/epem_flux/positron_DIFF_flux_dPhidE_20210621_TargetSim_QGSP_BIC_AllHP_POT1E6.txt")
pos_diff_flux_dune[:,1] *= 1/pot_per_sample  # per POT
el_diff_flux_dune = np.genfromtxt("../DUNE/data/epem_flux/electron_DIFF_flux_dPhidE_20210621_TargetSim_QGSP_BIC_AllHP_POT1E6.txt")
el_diff_flux_dune[:,1] *= 1/pot_per_sample  # per POT



# dump fluxes
gamma_data_dump = np.genfromtxt("data/escaped_proton/gamma_dump_escapedProton_QGSP_BERT_POT1e2.txt")
#electron_data_dump = np.genfromtxt("data/escaped_proton/electron_dump_escapedProton_QGSP_BERT_POT1e2.txt")
#positron_data_dump = np.genfromtxt("data/escaped_proton/positron_dump_escapedProton_QGSP_BERT_POT1e2.txt")

gamma_cosine_dump = gamma_data_dump[:,3] / np.sqrt(gamma_data_dump[:,1]**2 + gamma_data_dump[:,2]**2 + gamma_data_dump[:,3]**2)
#electron_cosine_dump = electron_data_dump[:,3] / np.sqrt(electron_data_dump[:,1]**2 + electron_data_dump[:,2]**2 + electron_data_dump[:,3]**2)
#positron_cosine_dump = positron_data_dump[:,3] / np.sqrt(positron_data_dump[:,1]**2 + positron_data_dump[:,2]**2 + positron_data_dump[:,3]**2)

forward_dump_gamma_energies = 1e3*gamma_data_dump[:,0][np.arccos(gamma_cosine_dump) < DUMP_TO_ND_SOLID_ANGLE]
forward_dump_gamma_angles = np.arccos(gamma_cosine_dump[np.arccos(gamma_cosine_dump) < DUMP_TO_ND_SOLID_ANGLE])
forward_dump_weights = 0.0425*1e-2*np.ones_like(forward_dump_gamma_energies)
forward_dump_gamma_flux = np.array([forward_dump_gamma_energies, forward_dump_gamma_angles, forward_dump_weights]).transpose()




# Energy resolution
def dune_eres_cauchy(E_true, distribution="gaussian"):
    # Takes Etrue in MeV, returns reco energies
    a, b, c = 0.027, 0.024, 0.007

    # Calculate relative energy error, fit inspired by [2503.04432]
    gamma = E_true * (a + b / sqrt(E_true * 1e-3) + c/(E_true * 1e-3))

    # Cauchy distribution Quantile function
    if hasattr(E_true, "__len__"):
        u_rnd = np.random.uniform(0.0, 1.0, E_true.shape[0])
    else:
        u_rnd = np.random.uniform(0.0, 1.0)

    if distribution == "gaussian":
        E_reco = E_true + gamma * sqrt(2) * erfinv(2*u_rnd - 1)
    elif distribution == "cauchy":
        E_reco = E_true + gamma * tan(pi * (u_rnd - 0.5))

    return E_reco




class PrimakoffFromBeam4Vectors(AxionFlux):
    """
    Generator for Primakoff-produced axion flux
    Takes in a flux of photons
    """
    def __init__(self, photon_flux=[0.0,1,1], target=Material("W"), det_dist=4.0, det_length=0.2,
                    det_area=0.04, axion_mass=0.1, axion_coupling=1e-3, n_samples=1000):
        super().__init__(axion_mass, target, det_dist, det_length, det_area)
        self.photon_flux = photon_flux
        self.gagamma = axion_coupling
        self.n_samples = n_samples
        self.target_photon_xs = AbsCrossSection(target)
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        self.support = np.ones(n_samples)

    def det_sa(self):
        return np.arctan(sqrt(self.det_area / pi) / self.det_dist)

    def branching_ratio(self, energy):
        cross_prim = primakoff_sigma(energy, self.target_z, 2*self.target_z,
                                     self.ma, self.gagamma)
        return cross_prim / (cross_prim + (self.target_photon_xs.sigma_cm2(energy) / (100 * METER_BY_MEV) ** 2))
    
    def get_beaming_angle(self, v):
        return np.arcsin(sqrt(1-v**2))
    
    def theta_z(self, theta, cosphi, theta_gamma):
        return abs(arccos(sin(theta_gamma)*cosphi*sin(theta) + cos(theta_gamma)*cos(theta)))

    # Simulate the angular-integrated energy flux.
    def simulate_int(self, photon):
        data_tuple = ([], [], [])

        if photon[0] < self.ma:
            return data_tuple
        rate = photon[2]
        e_gamma = photon[0]
        theta_gamma = abs(photon[1])
        thetas_rnd = exp(np.random.uniform(-12, np.log(pi), self.n_samples))
        phis_rnd = np.random.uniform(0.0,2*pi, self.n_samples)

        thetas_z = arccos(cos(thetas_rnd)*cos(theta_gamma) + cos(phis_rnd)*sin(thetas_rnd)*sin(theta_gamma))

        # Simulate
        def phase_space(theta, phi):
            return self.det_sa() > arccos(cos(theta)*cos(theta_gamma) \
                                   + cos(phi)*sin(theta)*sin(theta_gamma))

        def integrand(theta, phi):
            return phase_space(theta, phi) * \
                   primakoff_dsigma_dtheta(theta, e_gamma, self.target_z, self.ma, g=self.gagamma)
        
        convolution = np.vectorize(integrand)
        dr = 2*pi*(log(pi/exp(-12))/self.n_samples) * convolution(thetas_rnd, phis_rnd) * thetas_rnd

        # Get the branching ratio (numerator already contained in integrand func)
        br = 1/(self.target_photon_xs.sigma_cm2(e_gamma) / (100 * METER_BY_MEV) ** 2)

        # Push back lists and weights
        data_tuple[0].extend(e_gamma*self.support) # elastic limit
        data_tuple[1].extend(thetas_z)
        data_tuple[2].extend(rate * br * dr)  # scatter weights
        return data_tuple
    
    def simulate(self, n_samples=10, multicore=False):  # simulate the ALP flux
        #t1 = time.time()
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        
        if multicore == True:
            print("Running NCPU = ", max(1, multi.cpu_count()-1))
            
            with multi.Pool(max(1, multi.cpu_count()-1)) as pool:
                ntuple = pool.map(self.simulate_int, [f for f in self.photon_flux])
                pool.close()
            
            for tup in ntuple:
                self.axion_energy.extend(tup[0])
                self.axion_angle.extend(tup[1])
                self.axion_flux.extend(tup[2])
        else:
            for f in self.photon_flux:
                tup = self.simulate_int(f)
                self.axion_energy.extend(tup[0])
                self.axion_angle.extend(tup[1])
                self.axion_flux.extend(tup[2])

    def propagate(self, new_coupling=None, is_isotropic=True):
        if new_coupling is not None:
            rescale=power(new_coupling/self.gagamma, 2)
            super().propagate(W_gg(new_coupling, self.ma), rescale)
        else:
            super().propagate(W_gg(self.gagamma, self.ma))
        if is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept


class ElectronALPFromBeam4Vectors(AxionFlux):
    """
    Generator for Primakoff-produced axion flux
    Takes in the differential fluxes of electrons, positrons dN/dE in the target
    which are pointing within detector solid angle
    """
    def __init__(self, electron_flux=el_diff_flux_dune, positron_flux=pos_diff_flux_dune, target=Material("C"),
                 det_dist=DUNE_DIST, det_length=DUNE_LENGTH, det_area=DUNE_AREA, axion_mass=0.1, axion_coupling=1e-3,
                 n_samples=1000, max_track_length=5.0, flux_interpolation="log"):
        super().__init__(axion_mass, target, det_dist, det_length, det_area)
        self.electron_flux = electron_flux
        self.positron_flux = positron_flux
        self.ntarget_area_density = target.rad_length * AVOGADRO / (2*target.z[0])
        self.ge = axion_coupling
        self.n_samples = n_samples
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.decay_axion_weight = []
        self.scatter_axion_weight = []
        self.support = np.ones(n_samples)
        self.max_t = max_track_length
        self.flux_interp = flux_interpolation

        self.res_flux_wgt = []
        self.brem_flux_wgt = []
    
    def electron_flux_dN_dE(self, energy):
        if self.flux_interp == "log":
            return np.exp(np.interp(np.log(energy), np.log(self.electron_flux[:,0]), np.log(self.electron_flux[:,1]))) \
                * np.heaviside(energy - min(self.electron_flux[:,0]), 1.0) * np.heaviside(max(self.electron_flux[:,0])-energy, 1.0)
        
        return np.interp(energy, self.electron_flux[:,0], self.electron_flux[:,1], left=0.0, right=0.0)

    def positron_flux_dN_dE(self, energy):
        if self.flux_interp == "log":
            return np.exp(np.interp(np.log(energy), np.log(self.positron_flux[:,0]), np.log(self.positron_flux[:,1]))) \
                * np.heaviside(energy - min(self.positron_flux[:,0]), 1.0) * np.heaviside(max(self.positron_flux[:,0])-energy, 1.0)
        
        return np.interp(energy, self.positron_flux[:,0], self.positron_flux[:,1], left=0.0, right=0.0)

    def positron_flux_attenuated(self, E0, E1):
        return self.positron_flux_dN_dE(E0) * track_length_integrated_prob(E0, E1, self.max_t)

    def electron_positron_flux_attenuated(self, E0, E1):
        return (self.electron_flux_dN_dE(E0) + self.positron_flux_dN_dE(E0)) * track_length_integrated_prob(E0, E1, self.max_t)
    
    def resonance_peak(self):
        return 2*pi*M_E*power(self.ge / self.ma, 2) / sqrt(1 - power(2*M_E/self.ma, 2))

    def simulate_res(self):
        pass

    def simulate_brem(self, electron, n_samples):
        el_energy = electron[0]
        el_wgt = electron[1]

        ea_max = el_energy * (1 - power(self.ma/el_energy, 2))
        if ea_max <= self.ma:
            return
        
        ea_rnd = power(10, np.random.uniform(np.log10(self.ma), np.log10(ea_max), n_samples))
        theta_rnd = power(10, np.random.uniform(-6, np.log10(self.det_sa()), n_samples))

        mc_vol = np.log(10) * theta_rnd * (np.log10(self.det_sa()) + 6) * \
            np.log(10) * ea_rnd *  (np.log10(ea_max) - np.log10(self.ma)) / n_samples
        diff_br = (self.ntarget_area_density * HBARC**2) * mc_vol \
            * sin(theta_rnd) * brem_dsigma_dea_domega(ea_rnd, theta_rnd, el_energy, self.ge, self.ma, self.target_z)
                
        self.axion_energy.extend(ea_rnd)
        self.axion_angle.extend(theta_rnd)
        self.axion_flux.extend(el_wgt * diff_br)
        self.brem_flux_wgt.extend(el_wgt * diff_br)
    
    def simulate(self):
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []
        self.res_flux_wgt = []
        self.brem_flux_wgt = []
        
        # simulate bremsstrahlung from the electron and positron fluxes!
        ep_min = max(self.ma, M_E)
        if ep_min > max(self.electron_flux[:,0]):
            return

        # setup electron flux grid
        epem_energy_grid = 10**np.random.uniform(log10(ep_min*1.01), log10(max(self.electron_flux[:,0])), int(sqrt(self.n_samples)))
        epem_flux_mc_vol = np.log(10) * (log10(max(self.electron_flux[:,0])) - log10(ep_min*1.01)) / int(sqrt(self.n_samples))

        for el in epem_energy_grid:
            new_energy = np.random.uniform(ep_min, el)
            flux_weight = (el - ep_min) * self.electron_positron_flux_attenuated(el, new_energy)                
            self.simulate_brem([new_energy, flux_weight*epem_flux_mc_vol], n_samples=int(sqrt(self.n_samples)))

        # simulate resonance production and append to arrays
        resonant_energy = -M_E + self.ma**2 / (2 * M_E)
        if resonant_energy + M_E < self.ma:
            return

        if resonant_energy < M_E:
            return

        if resonant_energy > max(self.positron_flux[:,0]):
            return

        e_rnd = np.random.uniform(resonant_energy, max(self.positron_flux[:,0]), self.n_samples)
        mc_vol = (max(self.positron_flux[:,0]) - resonant_energy)

        positron_dist_wgts = np.array([self.positron_flux_attenuated(E1, resonant_energy) for E1 in e_rnd])

        attenuated_flux = mc_vol*np.sum(positron_dist_wgts)/self.n_samples
        wgt = self.target_z * (self.ntarget_area_density * HBARC**2) * self.resonance_peak() * attenuated_flux

        self.axion_energy.append(self.ma**2 / (2 * M_E))
        self.axion_angle.append(self.det_sa()/2)
        self.axion_flux.append(wgt)
        self.res_flux_wgt.append(wgt)

    def propagate(self, new_coupling=None, is_isotropic=True):
        if new_coupling is not None:
            rescale=power(new_coupling/self.ge, 2)
            super().propagate(W_ee(new_coupling, self.ma), rescale)
        else:
            super().propagate(W_ee(self.ge, self.ma))
        if is_isotropic:
            geom_accept = self.det_area / (4*pi*self.det_dist**2)
            self.decay_axion_weight *= geom_accept
            self.scatter_axion_weight *= geom_accept




# Pion production using gluon coupling
class AxionMesonMixingFlux(AxionFlux):
    """
    Axion flux to simulate pion and eta mixing ALP production
    Kelly, Kumar, Liu 2011.05995 inspired
    axion_coupling is f_a in MeV
    """
    def __init__(self, meson_flux=None, meson_mass=M_PI0, target=Material("C"),
                 det_dist=DUNE_DIST, det_length=DUNE_LENGTH, det_area=DUNE_AREA,
                 axion_mass=0.1, f_a=1000.0, meson_type="Pi0", total_pot=DUNE_POT_PER_YEAR,
                 n_samples=1, mesons_per_pot=2.89, c1=1.0, c2=1.0, c3=1.0):
        # set the coupling (photon coupling)
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.f_a = f_a

        super().__init__(axion_mass, target, det_dist, det_length, det_area)
        self.axion_coupling = self.photon_coupling()

        self.meson_flux = meson_flux
        self.n_samples = n_samples
        self.support = np.ones(n_samples)
        self.meson_mass = meson_mass
        self.mesons_per_pot = mesons_per_pot
        self.total_pot = total_pot
        self.meson_norm_wgt = 1.0/meson_flux.shape[0]
        self.meson_type = meson_type
    
    def photon_coupling(self):
        if self.ma > 150.0:
            c_gamma = self.c2 + (5/3)*self.c1
            return c_gamma * ALPHA / (2 * pi * self.f_a)  # in MeV^-1
        
        # ma < QCD scale
        c_gamma = self.c2 + (5/3)*self.c1 + self.c3 * (
            -1.92
            + (1/3) * (self.ma**2 / (self.ma**2 - M_PI0**2))
            + (8/9) * ((self.ma**2 - (4/9)*M_PI0**2) / (self.ma**2 - M_ETA**2))
            + (7/9) * ((self.ma**2 - (16/9)*M_PI0**2) / (self.ma**2 - M_ETA_PRIME**2))
        )

        return c_gamma * ALPHA / (2 * pi * self.f_a)  # in MeV^-1

    def set_new_coupling(self, f_a):
        self.f_a = f_a
        self.axion_coupling = self.photon_coupling()

    def mixing(self):
        if self.meson_type == "Pi0":
            return (1/6) * (F_PI/self.f_a) * (self.ma**2 / (self.ma**2 - M_PI0**2))
        elif self.meson_type == "Eta":
            return (1/sqrt(6)) * (F_PI/self.f_a) * ((self.ma**2 - 4*M_PI0**2 / 9) / (self.ma**2 - M_ETA**2))

    def phase_space_factor(self):
        if self.ma <= self.meson_mass:
            return 1.0
        return np.power(self.ma / self.meson_mass, -1.6)

    def simulate(self):
        self.axion_energy = []
        self.axion_angle = []
        self.axion_flux = []
        self.scatter_axion_weight = []
        self.decay_axion_weight = []

        for m in self.meson_flux:
            if m[0] < self.ma:
                continue

            meson_angle = np.arccos(m[3]/sqrt(m[1]**2 + m[2]**2 + m[3]**2))

            if meson_angle > self.det_sa():
                continue

            self.axion_energy.append(m[0])
            self.axion_angle.append(meson_angle)

            rate = self.total_pot * self.mesons_per_pot * self.meson_norm_wgt \
                * self.phase_space_factor() * (self.mixing())**2
            
            self.axion_flux.append(rate)

    def propagate(self, new_fa=None):
        if new_fa is not None:
            rescale=power(self.f_a/new_fa, 2)
            self.set_new_coupling(new_fa)
            super().propagate(W_gg(self.photon_coupling(), self.ma), rescale)
        else:
            super().propagate(W_gg(self.axion_coupling, self.ma))





def propagate_decay(e_a, wgts, ma, decay_width, rescale_factor=1.0):
    # Get axion Lorentz transformations and kinematics
    p_a = sqrt(e_a**2 - ma**2)
    v_a = p_a / e_a
    boost = e_a / ma
    tau = boost / decay_width if decay_width > 0.0 else np.inf * np.ones_like(boost)

    # Get decay and survival probabilities
    surv_prob = np.exp(-DUNE_DIST / METER_BY_MEV / v_a / tau)
    decay_prob = (1 - np.exp(-DUNE_LENGTH / METER_BY_MEV / v_a / tau))

    return np.asarray(rescale_factor * wgts * surv_prob * decay_prob, dtype=np.float32)



def decay_alp_gen(input_flux_dat_name=None, input_flux=None,
                  ALP_MASS=10.0, resolved=False, mass_daughters=0.0, return_4vs=False):
    
    if input_flux_dat_name is not None:
        input_flux = np.genfromtxt(input_flux_dat_name)

    # define mask
    if resolved:
        mask = (input_flux[:,2] >= 0.0) * (input_flux[:,0] >= 50.0)
    else:
        mask = (input_flux[:,2] >= 0.0) * (input_flux[:,0] >= 30.0)

    alp_flux_energies = input_flux[:,0][mask]
    alp_flux_angles = input_flux[:,1][mask]
    alp_flux_weights = input_flux[:,2][mask]

    # sort by weight and cut away the bottom 1% of the flux
    sorted_wgts = alp_flux_weights[np.argsort(alp_flux_weights)]
    sorted_angles = alp_flux_angles[np.argsort(alp_flux_weights)]
    sorted_energies = alp_flux_energies[np.argsort(alp_flux_weights)]
    wgt_cumsum = np.cumsum(sorted_wgts)/np.sum(sorted_wgts)

    alp_flux_energies = sorted_energies[wgt_cumsum > 0.01]
    alp_flux_angles = sorted_angles[wgt_cumsum > 0.01]
    alp_flux_weights = sorted_wgts[wgt_cumsum > 0.01]


    if (len(alp_flux_weights) == 0) or (np.sum(alp_flux_weights)*FLUX_SCALING < 1e-1):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # pass axion flux into event generator
    # Generate the decays with smearing
    p1 = LorentzVector(alp_flux_energies[0], 0, 0, np.sqrt(alp_flux_energies[0]**2 - ALP_MASS**2))
    mc = Decay2Body(p1, mass_daughters, mass_daughters, n_samples=5)
    def decay_gen(energy, ma, resolved=False):
        p1.set_p4(energy, 0.0, 0.0, np.sqrt(energy**2 - ma**2))
        mc.set_new_decay(p_parent=p1, m1=mass_daughters, m2=mass_daughters)
        mc.decay()

        wgts = mc.weights
        fv2 = mc.p2_lab_4vectors
        fv1 = mc.p1_lab_4vectors

        # Apply energy smearing
        for i in range(mc.weights.shape[0]):
            true_p_1 = fv1[i].momentum()
            reco_p_1 = dune_eres_cauchy(true_p_1)
            reco_p1_1 = reco_p_1 * (fv1[i].p1 / true_p_1)
            reco_p2_1 = reco_p_1 * (fv1[i].p2 / true_p_1)
            reco_p3_1 = reco_p_1 * (fv1[i].p3 / true_p_1)
            reco_p0_1 = sqrt(reco_p1_1**2 + reco_p2_1**2 + reco_p3_1**2 + mass_daughters**2)
            fv1[i].set_p4(reco_p0_1, reco_p1_1, reco_p2_1, reco_p3_1)

            true_p_2 = fv2[i].momentum()
            reco_p_2 = dune_eres_cauchy(true_p_2)
            reco_p1_2 = reco_p_2 * (fv2[i].p1 / true_p_2)
            reco_p2_2 = reco_p_2 * (fv2[i].p2 / true_p_2)
            reco_p3_2 = reco_p_2 * (fv2[i].p3 / true_p_2)
            reco_p0_2 = sqrt(reco_p1_2**2 + reco_p2_2**2 + reco_p3_2**2 + mass_daughters**2)
            fv2[i].set_p4(reco_p0_2, reco_p1_2, reco_p2_2, reco_p3_2)


        # Apply cuts: e+/e-/gamma < 30 MeV
        if resolved:
            fv1_mask = [fv.p0 > 30.0 for fv in fv1]
            fv2_mask = [fv.p0 > 20.0 for fv in fv2]
            wgts *= np.array(fv1_mask)*np.array(fv2_mask)
        else:
            fv_mask = [fv1[i].p0 + fv2[i].p0 > 30.0 for i in range(mc.n_samples)]
            wgts *= np.array(fv_mask)
        
        v1 = [fv.get_3momentum() for fv in fv1]
        v2 = [fv.get_3momentum() for fv in fv2]

        # Apply separation angle cut
        sep_angle = [arccos(v1[i]*v2[i] / abs(v1[i].mag()*v2[i].mag())) for i in range(mc.n_samples)]
        if resolved:
            wgts *= (180.0/np.pi)*np.array(sep_angle) >= 1.0
        elif not resolved:
            wgts *= (180.0/np.pi)*np.array(sep_angle) <= 1.0
        elif resolved == -1:
            wgts *= 1.0

        return fv1, fv2, sep_angle, wgts

    gamma1_p4_list = []
    gamma2_p4_list = []
    sep_angle_list = []
    event_weights = []

    t1 = time()

    for i in range(alp_flux_energies.shape[0]):
        gamma1_p4, gamma2_p4, delta_theta, wgt = decay_gen(alp_flux_energies[i], ALP_MASS, resolved=resolved)

        if np.sum(wgt) <= 0.0:
            continue

        gamma1_p4_list.extend(gamma1_p4)
        gamma2_p4_list.extend(gamma2_p4)
        sep_angle_list.extend(delta_theta)
        event_weights.extend(wgt * alp_flux_weights[i] * FLUX_SCALING)  # flux is in s^-1, already integrated over detector area

    t2 = time()
    print("    ---- Decaying took {} s".format(t2-t1))

    if return_4vs:
        p0_1 = np.array([p4.p0 for p4 in gamma1_p4_list])
        p1_1 = np.array([p4.p1 for p4 in gamma1_p4_list])
        p2_1 = np.array([p4.p2 for p4 in gamma1_p4_list])
        p3_1 = np.array([p4.p3 for p4 in gamma1_p4_list])
        p0_2 = np.array([p4.p0 for p4 in gamma2_p4_list])
        p1_2 = np.array([p4.p1 for p4 in gamma2_p4_list])
        p2_2 = np.array([p4.p2 for p4 in gamma2_p4_list])
        p3_2 = np.array([p4.p3 for p4 in gamma2_p4_list])
        event_weights = np.array(event_weights)
        return p0_1, p1_1, p2_1, p3_1, p0_2, p1_2, p2_2, p3_2, event_weights
    
    # define lists of particle kinematics for signal:
    gamma1_energy = np.array([p4.p0 for p4 in gamma1_p4_list])
    gamma2_energy = np.array([p4.p0 for p4 in gamma2_p4_list])
    gamma1_theta_z = np.array([p4.theta() for p4 in gamma1_p4_list])
    gamma2_theta_z = np.array([p4.theta() for p4 in gamma2_p4_list])
    sep_angles = np.array(sep_angle_list)
    inv_mass = np.array([(gamma1_p4_list[i] + gamma2_p4_list[i]).mass() for i in range(len(gamma1_p4_list))])
    total_energy = np.array(gamma1_energy) + np.array(gamma2_energy)
    event_weights = np.array(event_weights)

    return gamma1_energy, gamma2_energy, gamma1_theta_z, gamma2_theta_z, inv_mass, total_energy, sep_angles, event_weights




def generate_alp_events_2gamma(ma, g, n_flux_samples=1, resolved=True):
    flux_gen = PrimakoffFromBeam4Vectors(forward_photon_flux, target=Material("C"), det_dist=DUNE_DIST, det_length=DUNE_LENGTH,
                            det_area=DUNE_AREA, axion_mass=ma, axion_coupling=g, n_samples=n_flux_samples)

    print("Simulating and propagating ALP flux for ma={}...".format(ma))
    flux_gen.simulate(multicore=True)
    flux_gen.propagate(is_isotropic=False)

    alp_flux_energies = np.array(flux_gen.axion_energy)[flux_gen.decay_axion_weight>0]
    alp_flux_angles = np.array(flux_gen.axion_angle)[flux_gen.decay_axion_weight>0]
    alp_flux_wgt = flux_gen.decay_axion_weight[flux_gen.decay_axion_weight>0]

    # Decay the 4-vectors
    print("decaying 4-vectors...")
    flux_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_wgt]).transpose()
    gamma1_energy, gamma2_energy, gamma1_theta_z, gamma2_theta_z, inv_mass, \
        total_energy, sep_angles, event_weights = decay_alp_gen(input_flux=flux_array, ALP_MASS=ma, resolved=resolved)
    
    return gamma1_energy, gamma2_energy, gamma1_theta_z, gamma2_theta_z, inv_mass, total_energy, sep_angles, event_weights


def generate_alp_events_epem(ma, g, n_flux_samples=1):
    flux_gen = ElectronALPFromBeam4Vectors(axion_mass=ma, axion_coupling=g, target=Material("C"), n_samples=n_flux_samples)

    print("Simulating and propagating ALP flux for ma={}...".format(ma))
    flux_gen.simulate()
    flux_gen.propagate(is_isotropic=False)

    alp_flux_energies = np.array(flux_gen.axion_energy)[flux_gen.decay_axion_weight>0]
    alp_flux_angles = np.array(flux_gen.axion_angle)[flux_gen.decay_axion_weight>0]
    alp_flux_wgt = flux_gen.decay_axion_weight[flux_gen.decay_axion_weight>0]

    # Decay the 4-vectors
    print("decaying 4-vectors...")
    flux_array = np.array([alp_flux_energies, alp_flux_angles, alp_flux_wgt]).transpose()
    gamma1_energy, gamma2_energy, gamma1_theta_z, gamma2_theta_z, inv_mass, \
        total_energy, sep_angles, event_weights = decay_alp_gen(input_flux=flux_array, ALP_MASS=ma, mass_daughters=M_E, resolved=True)
    
    return gamma1_energy, gamma2_energy, gamma1_theta_z, gamma2_theta_z, inv_mass, total_energy, sep_angles, event_weights



# perform the simulation of 4-vectors for a e- > e- gamma
def compton_scatter_events(input_flux_dat_name=None, input_flux=None,
                  ALP_MASS=10.0, resolved=True, return_4vs=False):
    
    if input_flux_dat_name is not None:
        input_flux = np.genfromtxt(input_flux_dat_name)

    alp_flux_energies = input_flux[:,0]
    alp_flux_angles = input_flux[:,1]
    alp_flux_weights = input_flux[:,2]

    p1 = LorentzVector(alp_flux_energies[0], 0, 0, np.sqrt(alp_flux_energies[0]**2 - ALP_MASS**2))
    m2 = M2InverseCompton(ALP_MASS, z=18)  # Argon

    p4_electron = LorentzVector(M_E, 0.0, 0.0, 0.0)

    mc = Scatter2to2MC(m2, p1, p2=LorentzVector(M_E, 0.0, 0.0, 0.0), n_samples=1)

    def scatter_gen(energy, theta_z):
        p = np.sqrt(energy**2 - ALP_MASS**2)
        #phi = np.random.uniform(0.0, 2*pi)
        
        p1.set_p4(energy, 0.0, 0.0, p)
        mc.set_new_scattter(p1, p4_electron)

        mc.scatter_sim()

        wgts = mc.dsigma_dcos_cm_wgts[0] * HBARC**2  # convert to cm^2
        fv_gamma = mc.p3_lab_4vectors[0]
        fv_electron = mc.p4_lab_4vectors[0]

        # Apply cuts: e+/e-/gamma < 30 MeV
        if fv_gamma.p0 < 30.0:
            wgts *= 0.0
        if fv_electron.p0 < 30.0:
            wgts *= 0.0
        
        v_gamma = fv_gamma.get_3momentum()
        v_el = fv_electron.get_3momentum()

        # Apply separation angle cut
        sep_angle = arccos(v_gamma*v_el / abs(v_gamma.mag()*v_el.mag()))
        if resolved:
            wgts *= sep_angle * 180.0/np.pi >= 1.0
        elif not resolved:
            wgts *= sep_angle * 180.0/np.pi <= 1.0
        elif resolved == -1:
            wgts *= 1.0

        return fv_gamma, fv_electron, sep_angle, wgts
    
    gamma_p4_list = []
    electron_p4_list = []
    sep_angle_list = []
    event_weights = []

    for i in range(alp_flux_energies.shape[0]):
        if alp_flux_weights[i] <= 0.0:
            continue
        ea = alp_flux_energies[i]
        theta_z = alp_flux_angles[i]
        flux_weight = alp_flux_weights[i]
        gamma1_p4, gamma2_p4, delta_theta, wgt = scatter_gen(ea, theta_z)

        if wgt <= 0.0:
            continue

        gamma_p4_list.append(gamma1_p4)
        electron_p4_list.append(gamma2_p4)
        sep_angle_list.append(delta_theta)

        # the event weight should be scaled frrom POT^-1 to events / exposure
        # we also need to multiply the weight by density * length since it is a cross section
        event_weights.append(wgt * flux_weight * FLUX_SCALING * DUNE_ATOMS_PER_VOLUME * DUNE_LENGTH)  # flux is in s^-1, already integrated over detector area
    
    # define lists of particle kinematics for signal:
    gamma_energy = np.array([p4.p0 for p4 in gamma_p4_list])
    el_energy = np.array([p4.p0 for p4 in electron_p4_list])
    gamma_theta_z = np.array([p4.theta() for p4 in gamma_p4_list])
    el_theta_z = np.array([p4.theta() for p4 in electron_p4_list])
    sep_angles = np.array(sep_angle_list)
    total_energy = np.array(el_energy) + np.array(gamma_energy)
    event_weights = np.array(event_weights)

    return el_energy, gamma_energy, el_theta_z, gamma_theta_z, total_energy, sep_angles, event_weights
