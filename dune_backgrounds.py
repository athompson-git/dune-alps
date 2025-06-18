import numpy as np
from numpy import sqrt, log, pi
from numpy.random import standard_cauchy

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.pylab import rc
from matplotlib.colors import LogNorm
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import sys
sys.path.append("../")
from signal_generators import *

# import universal constants and exposure numbers
from dune_constants import *


def numu_events(enu):
    return NU_MU_REWGT * np.ones_like(enu)

def nue_events(enu):
    return NU_E_REWGT * np.ones_like(enu)

def numubar_events(enu):
    return NU_MUBAR_REWGT * np.ones_like(enu)

def nuebar_events(enu):
    return NU_EBAR_REWGT * np.ones_like(enu)




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




# Import backgrounds
def get_1particle_array(datfile, flavor="numu", mass_energy=0.0, efficiency=1.0):
    bkg = np.genfromtxt(datfile)
    bkg *= 1.0e3  # convert to MeV
    p0_em = bkg[:,0] + mass_energy
    p1_em = bkg[:,1]
    p2_em = bkg[:,2]
    p3_em = bkg[:,3]
    nu_energy = bkg[:,4]
    if flavor == "numu":
        weights = numu_events(nu_energy)
    elif flavor == "nue":
        weights = nue_events(nu_energy)
    elif flavor == "numubar":
        weights = numubar_events(nu_energy)
    elif flavor == "nuebar":
        weights = nuebar_events(nu_energy)
    return p0_em, p1_em, p2_em, p3_em, efficiency*weights

def get_2particle_array(datfile, flavor="numu", mass_energy=0.0, efficiency=1.0):
    bkg = np.genfromtxt(datfile)
    bkg *= 1.0e3  # convert to MeV
    p0_1 = bkg[:,0] + mass_energy
    p1_1 = bkg[:,1]
    p2_1 = bkg[:,2]
    p3_1 = bkg[:,3]
    p0_2 = bkg[:,4] + mass_energy
    p1_2 = bkg[:,5]
    p2_2 = bkg[:,6]
    p3_2 = bkg[:,7]

    p_mag_1 = p1_1*p1_1 + p2_1*p2_1 + p3_1*p3_1
    p_mag_2 = p1_2*p1_2 + p2_2*p2_2 + p3_2*p3_2
    p1_dot_p2 = p1_1*p1_2 + p2_1*p2_2 + p3_1*p3_2

    dtheta_deg = 180.0 * arccos(p1_dot_p2 / abs(p_mag_1*p_mag_2)) / pi

    if flavor == "numu":
        weights = NU_MU_REWGT * np.heaviside(dtheta_deg - 1.0, 0.0)
    elif flavor == "nue":
        weights = NU_E_REWGT * np.heaviside(dtheta_deg - 1.0, 0.0)
    elif flavor == "numubar":
        weights = NU_MUBAR_REWGT * np.heaviside(dtheta_deg - 1.0, 0.0)
    elif flavor == "nuebar":
        weights = NU_EBAR_REWGT * np.heaviside(dtheta_deg - 1.0, 0.0)
    return p0_1, p1_1, p2_1, p3_1, p0_2, p1_2, p2_2, p3_2, weights*efficiency




class Background2Particle:
    def __init__(self, data_file_name, nu_flavor, mass_particle_1=0.0, mass_particle_2=0.0,
                 verbose=False):
        if nu_flavor not in ["nue", "numu", "nuebar", "numubar"]:
            raise Exception("nu flavor not in {}!".format(["nue", "numu", "nuebar", "numubar"]))
        
        try:
            bkg = np.genfromtxt(data_file_name)
            bkg *= 1.0e3  # convert to MeV
            p0_1 = bkg[:,0] + mass_particle_1
            p1_1 = bkg[:,1]
            p2_1 = bkg[:,2]
            p3_1 = bkg[:,3]
            p0_2 = bkg[:,4] + mass_particle_2
            p1_2 = bkg[:,5]
            p2_2 = bkg[:,6]
            p3_2 = bkg[:,7]

            # Apply energy resolution effects
            p_mag_1 = sqrt(p1_1*p1_1 + p2_1*p2_1 + p3_1*p3_1)
            p_mag_2 = sqrt(p1_2*p1_2 + p2_2*p2_2 + p3_2*p3_2)

            reco_p1 = dune_eres_cauchy(p_mag_1)
            p1_1 = reco_p1 * (p1_1 / p_mag_1)
            p2_1 = reco_p1 * (p2_1 / p_mag_1)
            p3_1 = reco_p1 * (p3_1 / p_mag_1)
            p_mag_1 = sqrt(p1_1*p1_1 + p2_1*p2_1 + p3_1*p3_1)
            p0_1 = np.sqrt(p_mag_1**2 + mass_particle_1**2)

            reco_p2 = dune_eres_cauchy(p_mag_2)
            p1_2 = reco_p2 * (p1_2 / p_mag_2)
            p2_2 = reco_p2 * (p2_2 / p_mag_2)
            p3_2 = reco_p2 * (p3_2 / p_mag_2)
            p_mag_2 = sqrt(p1_2*p1_2 + p2_2*p2_2 + p3_2*p3_2)
            p0_2 = np.sqrt(p_mag_2**2 + mass_particle_2**2)

            if verbose:
                print("Declared bkg arrays")

            p1_dot_p2 = p1_1*p1_2 + p2_1*p2_2 + p3_1*p3_2

            self.dtheta_deg = 180.0 * arccos(p1_dot_p2 / abs(p_mag_1*p_mag_2)) / pi
            self.dtheta_rad = arccos(p1_dot_p2 / abs(p_mag_1*p_mag_2))

            self.inv_mass = np.sqrt((p0_1+p0_2)**2 - (p1_1 + p1_2)**2 - (p2_1 + p2_2)**2 - (p3_1 + p3_2)**2)
            self.total_energy = p0_1 + p0_2
            self.energy_p1 = p0_1
            self.energy_p2 = p0_2

            self.total_p1 = p1_1 + p1_2
            self.total_p2 = p2_1 + p2_2
            self.total_p3 = p3_1 + p3_2
            self.theta_12_deg = arccos(self.total_p3 / sqrt(self.total_p1**2 + self.total_p2**2 + self.total_p3**2)) * 180.0 / pi
            self.theta_12_rad = arccos(self.total_p3 / sqrt(self.total_p1**2 + self.total_p2**2 + self.total_p3**2))
            
            if verbose:
                print("Calculated bkg arrays")
        
            if nu_flavor == "numu":
                weights = NU_MU_REWGT * np.heaviside(self.dtheta_deg - 1.0, 0.0)
            elif nu_flavor == "nue":
                weights = NU_E_REWGT * np.heaviside(self.dtheta_deg - 1.0, 0.0)
            elif nu_flavor == "numubar":
                weights = NU_MUBAR_REWGT * np.heaviside(self.dtheta_deg - 1.0, 0.0)
            elif nu_flavor == "nuebar":
                weights = NU_EBAR_REWGT * np.heaviside(self.dtheta_deg - 1.0, 0.0)
            
            if verbose:
                print("Calculated weights")
            self.weights = weights
        
        except:
            self.total_energy = np.array([-1])
            self.inv_mass = np.array([-1])
            self.energy_p1 = np.array([-1])
            self.energy_p2 = np.array([-1])
            self.dtheta_deg = np.array([-1])
            self.dtheta_rad = np.array([-1])
            self.weights = np.array([0])
    
    def append_other_bkg(self, other_bkg):
        if not isinstance(other_bkg, Background2Particle):
            raise Exception("other_bkg must be an instance of Background2Particle")
        
        self.total_energy = np.concatenate((self.total_energy, other_bkg.total_energy))
        self.inv_mass = np.concatenate((self.inv_mass, other_bkg.inv_mass))
        self.energy_p1 = np.concatenate((self.energy_p1, other_bkg.energy_p1))
        self.energy_p2 = np.concatenate((self.energy_p2, other_bkg.energy_p2))
        self.dtheta_deg = np.concatenate((self.dtheta_deg, other_bkg.dtheta_deg))
        self.dtheta_rad = np.concatenate((self.dtheta_rad, other_bkg.dtheta_rad))
        self.weights = np.concatenate((self.weights, other_bkg.weights))


class Background1Particle:
    def __init__(self, data_file_name, nu_flavor, mass_particle_1=0.0):
        if nu_flavor not in ["nue", "numu", "nuebar", "numubar"]:
            raise Exception("nu flavor not in {}!".format(["nue", "numu", "nuebar", "numubar"]))
        
        try:
            bkg = np.genfromtxt(data_file_name)
            bkg *= 1.0e3  # convert to MeV
            self.p0_1 = bkg[:,0] + mass_particle_1
            self.p1_1 = bkg[:,1]
            self.p2_1 = bkg[:,2]
            self.p3_1 = bkg[:,3]

            p_mag_1 = sqrt(self.p1_1*self.p1_1 + self.p2_1*self.p2_1 + self.p3_1*self.p3_1)

            reco_p = dune_eres_cauchy(p_mag_1)
            self.p1_1 = reco_p * (self.p1_1 / p_mag_1)
            self.p2_1 = reco_p * (self.p2_1 / p_mag_1)
            self.p3_1 = reco_p * (self.p3_1 / p_mag_1)
            p_mag_1 = sqrt(self.p1_1*self.p1_1 + self.p2_1*self.p2_1 + self.p3_1*self.p3_1)
            self.p0_1 = np.sqrt(p_mag_1**2 + mass_particle_1**2)

            self.theta_z_deg = 180.0 * arccos(self.p3_1 / p_mag_1) / pi
            self.theta_z_rad = arccos(self.p3_1 / p_mag_1)

            if nu_flavor == "numu":
                weights = NU_MU_REWGT * np.ones_like(bkg[:,0])
            elif nu_flavor == "nue":
                weights = NU_E_REWGT * np.ones_like(bkg[:,0])
            elif nu_flavor == "numubar":
                weights = NU_MUBAR_REWGT * np.ones_like(bkg[:,0])
            elif nu_flavor == "nuebar":
                weights = NU_EBAR_REWGT * np.ones_like(bkg[:,0])
            
            self.weights = weights
        
        except:
            self.p0_1 = np.array([-1])
            self.p1_1 = np.array([-1])
            self.p2_1 = np.array([-1])
            self.p3_1 = np.array([-1])
            self.theta_z_deg = np.array([-1])
            self.theta_z_rad = np.array([-1])
            self.weights = np.array([0.0])
    
    def append_other_bkg(self, other_bkg):
        if not isinstance(other_bkg, Background1Particle):
            raise Exception("other_bkg must be an instance of Background1Particle")
        
        self.p0_1 = np.concatenate((self.p0_1, other_bkg.p0_1))
        self.p1_1 = np.concatenate((self.p1_1, other_bkg.p1_1))
        self.p2_1 = np.concatenate((self.p2_1, other_bkg.p2_1))
        self.p3_1 = np.concatenate((self.p3_1, other_bkg.p3_1))
        self.theta_z_deg = np.concatenate((self.theta_z_deg, other_bkg.theta_z_deg))
        self.theta_z_rad = np.concatenate((self.theta_z_rad, other_bkg.theta_z_rad))
        self.weights = np.concatenate((self.weights, other_bkg.weights))


"""
# SINGLE GAMMA, SINGLE ELECTRON, SINGLE POSITRON
p0_g_nue, p1_g_nue, p2_g_nue, p3_g_nue, weights1_nue_g = get_1particle_array("data/1g0p/1gamma_nue_4vectors_DUNE_bkg.txt", flavor="nue")
p0_em_nue, p1_em_nue, p2_em_nue, p3_em_nue, weights1_nue_em = get_1particle_array("data/1g0p/1em_nue_4vectors_DUNE_bkg.txt", flavor="nue")
p0_ep_nue, p1_ep_nue, p2_ep_nue, p3_ep_nue, weights1_nue_ep = get_1particle_array("data/1g0p/1ep_nue_4vectors_DUNE_bkg.txt", flavor="nue")

p0_g_numu, p1_g_numu, p2_g_numu, p3_g_numu, weights1_numu_g = get_1particle_array("data/1g0p/1gamma_numu_4vectors_DUNE_bkg.txt", flavor="numu")
p0_em_numu, p1_em_numu, p2_em_nue, p3_em_numu, weights1_numu_em = get_1particle_array("data/1g0p/1em_numu_4vectors_DUNE_bkg.txt", flavor="numu")
p0_ep_numu, p1_ep_numu, p2_ep_numu, p3_ep_numu, weights1_numu_ep = get_1particle_array("data/1g0p/1ep_numu_4vectors_DUNE_bkg.txt", flavor="numu")

p0_g_nuebar, p1_g_nuebar, p2_g_nuebar, p3_g_nuebar, weights1_nuebar_g = get_1particle_array("data/1g0p/1gamma_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")
p0_em_nuebar, p1_em_nuebar, p2_em_nuebar, p3_em_nuebar, weights1_nuebar_em = get_1particle_array("data/1g0p/1em_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")
p0_ep_nuebar, p1_ep_nuebar, p2_ep_nuebar, p3_ep_nuebar, weights1_nuebar_ep = get_1particle_array("data/1g0p/1ep_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")

p0_g_numubar, p1_g_numubar, p2_g_numubar, p3_g_numubar, weights1_numubar_g = get_1particle_array("data/1g0p/1gamma_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")
p0_em_numubar, p1_em_numubar, p2_em_numubar, p3_em_numubar, weights1_numubar_em = get_1particle_array("data/1g0p/1em_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")
p0_ep_numubar, p1_ep_numubar, p2_ep_numubar, p3_ep_numubar, weights1_numubar_ep = get_1particle_array("data/1g0p/1ep_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")


p0_g_nue, p1_g_nue, p2_g_nue, p3_g_nue, weights1_nue_g = get_1particle_array("data/1g0p/1gamma_nue_4vectors_DUNE_bkg.txt", flavor="nue")
p0_em_nue, p1_em_nue, p2_em_nue, p3_em_nue, weights1_nue_em = get_1particle_array("data/1g0p/1em_nue_4vectors_DUNE_bkg.txt", flavor="nue")
p0_ep_nue, p1_ep_nue, p2_ep_nue, p3_ep_nue, weights1_nue_ep = get_1particle_array("data/1g0p/1ep_nue_4vectors_DUNE_bkg.txt", flavor="nue")

p0_g_numu, p1_g_numu, p2_g_numu, p3_g_numu, weights1_numu_g = get_1particle_array("data/1g0p/1gamma_numu_4vectors_DUNE_bkg.txt", flavor="numu")
p0_em_numu, p1_em_numu, p2_em_nue, p3_em_numu, weights1_numu_em = get_1particle_array("data/1g0p/1em_numu_4vectors_DUNE_bkg.txt", flavor="numu")
p0_ep_numu, p1_ep_numu, p2_ep_numu, p3_ep_numu, weights1_numu_ep = get_1particle_array("data/1g0p/1ep_numu_4vectors_DUNE_bkg.txt", flavor="numu")

p0_g_nuebar, p1_g_nuebar, p2_g_nuebar, p3_g_nuebar, weights1_nuebar_g = get_1particle_array("data/1g0p/1gamma_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")
p0_em_nuebar, p1_em_nuebar, p2_em_nuebar, p3_em_nuebar, weights1_nuebar_em = get_1particle_array("data/1g0p/1em_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")
p0_ep_nuebar, p1_ep_nuebar, p2_ep_nuebar, p3_ep_nuebar, weights1_nuebar_ep = get_1particle_array("data/1g0p/1ep_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")

p0_g_numubar, p1_g_numubar, p2_g_numubar, p3_g_numubar, weights1_numubar_g = get_1particle_array("data/1g0p/1gamma_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")
p0_em_numubar, p1_em_numubar, p2_em_numubar, p3_em_numubar, weights1_numubar_em = get_1particle_array("data/1g0p/1em_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")
p0_ep_numubar, p1_ep_numubar, p2_ep_numubar, p3_ep_numubar, weights1_numubar_ep = get_1particle_array("data/1g0p/1ep_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")



# E+ E-
p0_epem_nue_1, p1_epem_nue_1, p2_epem_nue_1, p3_epem_nue_1, \
    p0_epem_nue_2, p1_epem_nue_2, p2_epem_nue_2, p3_epem_nue_2, weights1_nue_epem = get_2particle_array("data/1em1ep/epem_nue_4vectors_DUNE_bkg.txt", flavor="nue", mass_energy=M_E)
p0_epem_numu_1, p1_epem_numu_1, p2_epem_numu_1, p3_epem_numu_1, \
    p0_epem_numu_2, p1_epem_numu_2, p2_epem_numu_2, p3_epem_numu_2, weights1_numu_epem = get_2particle_array("data/1em1ep/epem_numu_4vectors_DUNE_bkg.txt", flavor="numu", mass_energy=M_E)
p0_epem_nuebar_1, p1_epem_nuebar_1, p2_epem_nuebar_1, p3_epem_nuebar_1, \
     p0_epem_nuebar_2, p1_epem_nuebar_2, p2_epem_nuebar_2, p3_epem_nuebar_2, weights1_nuebar_epem = get_2particle_array("data/1em1ep/epem_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar", mass_energy=M_E)
p0_epem_numubar_1, p1_epem_numubar_1, p2_epem_numubar_1, p3_epem_numubar_1, \
     p0_epem_numubar_2, p1_epem_numubar_2, p2_epem_numubar_2, p3_epem_numubar_2, weights1_numubar_epem = get_2particle_array("data/1em1ep/epem_numubar_4vectors_DUNE_bkg.txt", flavor="numubar", mass_energy=M_E)



# 2 GAMMA
p0_2g_nue_1, p1_2g_nue_1, p2_2g_nue_1, p3_2g_nue_1, p0_2g_nue_2, p1_2g_nue_2, p2_2g_nue_2, p3_2g_nue_2, weights1_nue_2g = get_2particle_array("data/2gamma/2gamma_nue_4vectors_DUNE_bkg.txt", flavor="nue")
p0_2g_numu_1, p1_2g_numu_1, p2_2g_numu_1, p3_2g_numu_1, p0_2g_numu_2, p1_2g_numu_2, p2_2g_numu_2, p3_2g_numu_2, weights1_numu_2g = get_2particle_array("data/2gamma/2gamma_numu_4vectors_DUNE_bkg.txt", flavor="numu")
p0_2g_nuebar_1, p1_2g_nuebar_1, p2_2g_nuebar_1, p3_2g_nuebar_1, p0_2g_nuebar_2, p1_2g_nuebar_2, p2_2g_nuebar_2, p3_2g_nuebar_2, weights1_nuebar_2g = get_2particle_array("data/2gamma/2gamma_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")
p0_2g_numubar_1, p1_2g_numubar_1, p2_2g_numubar_1, p3_2g_numubar_1, p0_2g_numubar_2, p1_2g_numubar_2, p2_2g_numubar_2, p3_2g_numubar_2, weights1_numubar_2g = get_2particle_array("data/2gamma/2gamma_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")

inv_mass_2g_nue = sqrt((p0_2g_nue_1 + p0_2g_nue_2)**2 - (p1_2g_nue_1 + p1_2g_nue_2)**2 - (p2_2g_nue_1 + p2_2g_nue_2)**2 - (p3_2g_nue_1 + p3_2g_nue_2)**2)
inv_mass_2g_numu = sqrt((p0_2g_numu_1 + p0_2g_numu_2)**2 - (p1_2g_numu_1 + p1_2g_numu_2)**2 - (p2_2g_numu_1 + p2_2g_numu_2)**2 - (p3_2g_numu_1 + p3_2g_numu_2)**2)
inv_mass_2g_nuebar = sqrt((p0_2g_nuebar_1 + p0_2g_nuebar_2)**2 - (p1_2g_nuebar_1 + p1_2g_nuebar_2)**2 - (p2_2g_nuebar_1 + p2_2g_nuebar_2)**2 - (p3_2g_nuebar_1 + p3_2g_nuebar_2)**2)
inv_mass_2g_numubar = sqrt((p0_2g_numubar_1 + p0_2g_numubar_2)**2 - (p1_2g_numubar_1 + p1_2g_numubar_2)**2 - (p2_2g_numubar_1 + p2_2g_numubar_2)**2 - (p3_2g_numubar_1 + p3_2g_numubar_2)**2)



# E- GAMMA, E+ GAMMA

p0_1g1em_nue_1, p1_1g1em_nue_1, p2_1g1em_nue_1, p3_1g1em_nue_1, \
    p0_1g1em_nue_2, p1_1g1em_nue_2, p2_1g1em_nue_2, p3_1g1em_nue_2, weights1_nue_1g1em = get_2particle_array("data/1g1e0p/emgamma_nue_4vectors_DUNE_bkg.txt", flavor="nue")
p0_1g1em_numu_1, p1_1g1em_numu_1, p2_1g1em_numu_1, p3_1g1em_numu_1, \
    p0_1g1em_numu_2, p1_1g1em_numu_2, p2_1g1em_numu_2, p3_1g1em_numu_2, weights1_numu_1g1em = get_2particle_array("data/1g1e0p/emgamma_numu_4vectors_DUNE_bkg.txt", flavor="numu")
p0_1g1em_nuebar_1, p1_1g1em_nuebar_1, p2_1g1em_nuebar_1, p3_1g1em_nuebar_1, \
     p0_1g1em_nuebar_2, p1_1g1em_nuebar_2, p2_1g1em_nuebar_2, p3_1g1em_nuebar_2, weights1_nuebar_1g1em = get_2particle_array("data/1g1e0p/emgamma_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")
p0_1g1em_numubar_1, p1_1g1em_numubar_1, p2_1g1em_numubar_1, p3_1g1em_numubar_1, \
     p0_1g1em_numubar_2, p1_1g1em_numubar_2, p2_1g1em_numubar_2, p3_1g1em_numubar_2, weights1_numubar_1g1em = get_2particle_array("data/1g1e0p/emgamma_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")


p0_1g1ep_nue_1, p1_1g1ep_nue_1, p2_1g1ep_nue_1, p3_1g1ep_nue_1, \
     p0_1g1ep_nue_2, p1_1g1ep_nue_2, p2_1g1ep_nue_2, p3_1g1ep_nue_2, weights1_nue_1g1ep = get_2particle_array("data/1g1e0p/epgamma_nue_4vectors_DUNE_bkg.txt", flavor="nue")
p0_1g1ep_numu_1, p1_1g1ep_numu_1, p2_1g1ep_numu_1, p3_1g1ep_numu_1, \
     p0_1g1ep_numu_2, p1_1g1ep_numu_2, p2_1g1ep_numu_2, p3_1g1ep_numu_2, weights1_numu_1g1ep = get_2particle_array("data/1g1e0p/epgamma_numu_4vectors_DUNE_bkg.txt", flavor="numu")
p0_1g1ep_nuebar_1, p1_1g1ep_nuebar_1, p2_1g1ep_nuebar_1, p3_1g1ep_nuebar_1, \
     p0_1g1ep_nuebar_2, p1_1g1ep_nuebar_2, p2_1g1ep_nuebar_2, p3_1g1ep_nuebar_2, weights1_nuebar_1g1ep = get_2particle_array("data/1g1e0p/epgamma_nuebar_4vectors_DUNE_bkg.txt", flavor="nuebar")
p0_1g1ep_numubar_1, p1_1g1ep_numubar_1, p2_1g1ep_numubar_1, p3_1g1ep_numubar_1, \
     p0_1g1ep_numubar_2, p1_1g1ep_numubar_2, p2_1g1ep_numubar_2, p3_1g1ep_numubar_2, weights1_numubar_1g1ep = get_2particle_array("data/1g1e0p/epgamma_numubar_4vectors_DUNE_bkg.txt", flavor="numubar")

"""