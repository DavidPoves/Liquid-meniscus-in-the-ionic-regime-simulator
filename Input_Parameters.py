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


def General_Parameters():
    vacuum_perm = 1/(4*np.pi*9e9)  # Vacuum Permittivity [Nm^2/C^2]
    eps_r = 10  # [-]. From Coffman's thesis page 58
    q_m = 1e6  # [C/kg]. Specific charge. From Coffman's thesis page 58
    k_B = 1.38064852e-23  # [m^2 kg s^-2 K^-1]. Boltzmann's constant
    h = 6.62607004e-34  # [m^2 kg/s]. Planck's constant

    return vacuum_perm, eps_r, q_m, k_B, h


# %% LIQUID CHARACTERISTICS.
"""
In the next cells, the properties of different Ionic Liquids are defined.
Different sources will be used and properly cited.
"""


def Liquid_Data(liquid):
    """
    Function where the required properties of the different liquids are
    introduced and retrieved to the main script.

    Parameters
    ----------
    liquid : STRING
        String to choose the liquid to be studied:
            - 'EMIBF4': 1-ethyl-3-methylimidazolium Tetrafluoroborate

    Returns
    -------
    liquid_data : NUMPY.ARRAY DATA TYPE.
        Array containing the required parameters for the chosen liquid.

    """

    if liquid == 'EMIBF4':
        # %% EMI-BF4 IL PROPERTIES.
        """
        Complete name: 1-ethyl-3-methylimidazolium Tetrafluoroborate
        """
        # Molecular Weight of the liquid [kg/mol]:
        MW_EMIBF4 = 197.97*1e-3  # [kg/mol]

        # Density of the liquid [kg/m^3] (at 298.15K)
        """
        Obtained from: Great increase of the electrical conductivity of ionic
        liquids in aqueous solutions
        """
        rho_0_EMIBF4 = 1279  # [kg/m^3]

        # Viscosity of the liquid [Pa-s] (at 298.15K)
        """
        Obtained from: Thermochemistry of ionic liquid heat-transfer fluids
        """
        mu_0_EMIBF4 = 0.03607  # [Pa-s]

        # Nominal electric conductivity of the liquid [S/m] (at 298.15K)
        """
        Obtained from: Liquid-solid-liquid phase transition hysteresis loops
        in the ionic conductivity of ten imidazolium-based ionic liquids
        """
        k_0_EMIBF4 = 1.571  # [S/m]

        # Thermal Sensitivity of the liquid [S/m-K]
        """
        Obtained from Coffman's thesis.
        """
        k_prime_EMIBF4 = 0.04  # [S/m-K]

        # Specific heat capacity of the liquid [J/mol-K] (at 298.15K)
        """
        Obtained from: Heat capacities of ionic liquids and their heats of
        solution in molecular liquids
        """
        cp_mol_EMIBF4 = 308.1  # [J/mol-K]
        cp_mass_EMIBF4 = cp_mol_EMIBF4 / MW_EMIBF4  # [J/kg-K]

        # Nominal Activation Energy (or solvation energy) of the liquid [eV]
        """
        # Obtained from: Ion evaporation from Taylor cones of propylene
        carbonate mixed with ionic liquids, page 453
        """
        Solvation_energy_EMIBF4 = 1.8  # [eV]. See Coffman's thesis page

        # Thermal conductivity of the fluid [W/m-K] (at 300K)
        """
        Obtained from: Thermochemistry of ionic liquid heat-transfer fluids
        """
        k_T_EMIBF4 = 0.199  # [W/m-K]

        # Intrinsic surface energy of the fluid [N/m].
        """
        Obtained from Coffman's thesis page 58
        """
        gamma_EMIBF4 = 1e-1  # [N/m]

        # Create the array containing the required liquid properties.
        liquid_data = np.array([MW_EMIBF4, rho_0_EMIBF4, mu_0_EMIBF4,
                                k_0_EMIBF4, k_prime_EMIBF4, cp_mass_EMIBF4,
                                Solvation_energy_EMIBF4, k_T_EMIBF4,
                                gamma_EMIBF4])
        return liquid_data
