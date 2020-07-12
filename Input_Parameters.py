# %% IMPORT THE REQUIRED LIBRARIES.
import numpy as np

"""
@author: David Poves Ros
"""

"""
This module will be used to introduce the required inputs for the MAIN.py
script. Among these inputs, the liquid characterstics are required, which will
be obtained from Coffman's thesis or by any other web or book, which will be
properly referenced.
Most data can be found at: http://ILThermo.boulder.nist.gov/ILThermo/
"""

# %% GENERAL DATA.


class Liquid_Properties(object):

    def __init__(self, relative_permittivity=10):
        self.vacuum_perm = 1/(4*np.pi*9e9)  # Vacuum Permittivity [Nm^2/C^2]
        self.eps_r = relative_permittivity  # [-]
        self.q_m = 1e6  # [C/kg]. Specific charge. From Coffman's thesis page 58
        self.k_B = 1.38064852e-23  # [m^2 kg s^-2 K^-1]. Boltzmann's constant
        self.h = 6.62607004e-34  # [m^2 kg/s]. Planck's constant

    def EMIBF4(self):
        """
        Complete name: 1-ethyl-3-methylimidazolium Tetrafluoroborate
        """
        # Molecular Weight of the liquid [kg/mol]:
        self.MW = 197.97*1e-3  # [kg/mol]

        # Density of the liquid [kg/m^3] (at 298.15K)
        """
        Obtained from: Great increase of the electrical conductivity of ionic
        liquids in aqueous solutions
        """
        self.rho_0 = 1279  # [kg/m^3]

        # Viscosity of the liquid [Pa-s] (at 298.15K)
        """
        Obtained from: Thermochemistry of ionic liquid heat-transfer fluids
        """
        self.mu_0 = 0.03607  # [Pa-s]

        # Nominal electric conductivity of the liquid [S/m] (at 298.15K)
        """
        Obtained from: Liquid-solid-liquid phase transition hysteresis loops
        in the ionic conductivity of ten imidazolium-based ionic liquids
        """
        self.k_0 = 1.571  # [S/m]

        # Thermal Sensitivity of the liquid [S/m-K]
        """
        Obtained from Coffman's thesis.
        """
        self.k_prime = 0.04  # [S/m-K]

        # Specific heat capacity of the liquid [J/mol-K] (at 298.15K)
        """
        Obtained from: Heat capacities of ionic liquids and their heats of
        solution in molecular liquids
        """
        self.cp_mol = 308.1  # [J/mol-K]
        cp_mass_EMIBF4 = self.cp_mol / self.MW  # [J/kg-K]

        # Nominal Activation Energy (or solvation energy) of the liquid [eV]
        """
        # Obtained from: Ion evaporation from Taylor cones of propylene
        carbonate mixed with ionic liquids, page 453
        """
        self.Solvation_energy= 1.8*1.602176565e-19  # [eV -> J]

        # Thermal conductivity of the fluid [W/m-K] (at 300K)
        """
        Obtained from: Thermochemistry of ionic liquid heat-transfer fluids
        """
        self.k_T = 0.199  # [W/m-K]

        # Intrinsic surface energy of the fluid [N/m].
        """
        Obtained from Coffman's thesis page 58
        """
        self.gamma = 1e-1  # [N/m]