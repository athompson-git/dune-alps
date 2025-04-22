import numpy as np
from numpy import sqrt, log, pi

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

    nu_energy = bkg[:,8]
    if flavor == "numu":
        weights = numu_events(nu_energy) * np.heaviside(dtheta_deg - 1.0, 0.0)
    elif flavor == "nue":
        weights = nue_events(nu_energy) * np.heaviside(dtheta_deg - 1.0, 0.0)
    elif flavor == "numubar":
        weights = numubar_events(nu_energy) * np.heaviside(dtheta_deg - 1.0, 0.0)
    elif flavor == "nuebar":
        weights = nuebar_events(nu_energy) * np.heaviside(dtheta_deg - 1.0, 0.0)
    return p0_1, p1_1, p2_1, p3_1, p0_2, p1_2, p2_2, p3_2, weights*efficiency



# TODO: finish background class
class Background2Particle:
    def __init__(self, data_file_name, nu_flavor, mass_particle_1=0.0, mass_particle_2=0.0):
        if nu_flavor not in ["nue", "numu", "nuebar", "numubar"]:
            raise Exception("nu flavor not in {}!".format(["nue", "numu", "nuebar", "numubar"]))
        
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

        p_mag_1 = sqrt(p1_1*p1_1 + p2_1*p2_1 + p3_1*p3_1)
        p_mag_2 = sqrt(p1_2*p1_2 + p2_2*p2_2 + p3_2*p3_2)
        p1_dot_p2 = p1_1*p1_2 + p2_1*p2_2 + p3_1*p3_2

        self.dtheta_deg = 180.0 * arccos(p1_dot_p2 / abs(p_mag_1*p_mag_2)) / pi
        self.dtheta_rad = arccos(p1_dot_p2 / abs(p_mag_1*p_mag_2))

        self.inv_mass = np.sqrt((p0_1+p0_2)**2 - (p1_1 + p1_2)**2 - (p2_1 + p2_2)**2 - (p3_1 + p3_2)**2)
        self.total_energy = p0_1 + p0_2
        self.energy_p1 = p0_1
        self.energy_p2 = p0_2
    
        nu_energy = bkg[:,8]
        if nu_flavor == "numu":
            weights = numu_events(nu_energy) * np.heaviside(self.dtheta_deg - 1.0, 0.0)
        elif nu_flavor == "nue":
            weights = nue_events(nu_energy) * np.heaviside(self.dtheta_deg - 1.0, 0.0)
        elif nu_flavor == "numubar":
            weights = numubar_events(nu_energy) * np.heaviside(self.dtheta_deg - 1.0, 0.0)
        elif nu_flavor == "nuebar":
            weights = nuebar_events(nu_energy) * np.heaviside(self.dtheta_deg - 1.0, 0.0)
        
        self.weights = weights


class Background1Particle:
    def __init__(self, data_file_name, nu_flavor, mass_particle_1=0.0):
        if nu_flavor not in ["nue", "numu", "nuebar", "numubar"]:
            raise Exception("nu flavor not in {}!".format(["nue", "numu", "nuebar", "numubar"]))
        
        bkg = np.genfromtxt(data_file_name)
        bkg *= 1.0e3  # convert to MeV
        self.p0_1 = bkg[:,0] + mass_particle_1
        self.p1_1 = bkg[:,1]
        self.p2_1 = bkg[:,2]
        self.p3_1 = bkg[:,3]

        p_mag_1 = sqrt(self.p1_1*self.p1_1 + self.p2_1*self.p2_1 + self.p3_1*self.p3_1)

        self.theta_z_deg = 180.0 * arccos(self.p3_1 / p_mag_1) / pi
        self.theta_z_rad = arccos(self.p3_1 / p_mag_1)

        nu_energy = bkg[:,4]
        if nu_flavor == "numu":
            weights = numu_events(nu_energy)
        elif nu_flavor == "nue":
            weights = nue_events(nu_energy)
        elif nu_flavor == "numubar":
            weights = numubar_events(nu_energy)
        elif nu_flavor == "nuebar":
            weights = nuebar_events(nu_energy)
        
        self.weights = weights



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

