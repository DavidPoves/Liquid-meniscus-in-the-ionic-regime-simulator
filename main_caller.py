import fenics as fn

from ElectrospraySimulator.Main_scripts.MAIN import MainWrapper

# Copyright (C) 2020- by David Poves Ros
#
# This file is part of the End of Degree Thesis.
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#

# %% WHAT CONVERGES.
# The following options have been proven to converge to satisfactory results. Notice that the variable parameter depends
# on the selected value of B: variable_parameter = 1/B * np.tan(np.radians(49.3))
"""
All computations have been carried out using 0.01 as a characteristic length from curvature and 1.25 as characteristic
length factor.
-> Taylor cone: - Frontal Delaunay, max size: 0.06, variable_parameter = 24.89523033034176
                - Frontal Delaunay, max.size: 0.05, variable parameter = 24.89523033034176
                - Frontal Delaunay, max size: 0.04, variable_parameter = 24.89523033034176
                - Automatic, max.size: 0.1, variable parameter = 24.89523033034176
                - Automatic, max.size: 0.2, variable parameter = 24.89523033034176
                - Automatic, max.size: 0.3, variable parameter = 24.89523033034176
-> Cosine:      - Frontal Delaunay, max size: 0.06
                - Frontal Delaunay, max size: 0.1
                - Frontal Delaunay, max size: 0.03
                - Automatic, max size: 0.1
                - Automatic, max size 0.03
                - Automatic, max size 0.05
    All cosine computations were done with an initial height of 0.5
-> Parabolic:   - Frontal Delaunay, max size: 0.03
"""


# %% RUN THE SIMULATION


def main_caller(liquid, required_inputs, electrostatics_bcs, Stokes_bcs, **kwargs):
    wrapper = MainWrapper(liquid, required_inputs, electrostatics_bcs, Stokes_bcs, **kwargs)
    return wrapper


def define_inputs():
    """
    In this function, the user must define all the inputs that are required to run a simulation.
    Returns:

    """
    # Define the required inputs.
    req_inputs = {'B': 0.0467,
                  'Lambda': 1.,
                  'E0': 0.99,
                  'C_R': 1e3,
                  'T_h': 1.,
                  'P_r': 0}

    # Define the boundary conditions for the electrostatics.
    """
    Notice the structure of the boundary conditions:

        1. Boundary name as in the .geo file.
        2. The type of boundary condition (Dirichlet or Neumann).
        3. A list where the first value is the value of the bc and the second
            element is the subdomain to which it belongs to.
    """
    z0 = 10  # As defined in the .geo file.
    top_potential = -req_inputs['E0'] * z0  # Potential at top wall (electrode).
    ref_potential = 0  # Reference potential (ground).
    bcs_electrostatics = {'Top_Wall': {'Dirichlet': [top_potential, 'vacuum']},
                          'Inlet': {'Dirichlet': [ref_potential, 'liquid']},
                          'Tube_Wall_R': {'Dirichlet': [ref_potential, 'liquid']},
                          'Bottom_Wall': {'Dirichlet': [ref_potential, 'vacuum']},
                          'Lateral_Wall_R': {'Neumann': 'vacuum'},
                          'Lateral_Wall_L': {'Neumann': 'vacuum'},
                          'Tube_Wall_L': {'Neumann': 'vacuum'}}

    # Define the solver options for the electrostatics solver.
    """
    If the user checks the options given by the code above, one can see all the
    available parameters for a parameter set. These parameter sets are:
        - snes_solver.
        - krylov_solver.
        - lu-solver.
    Each of them contain different parameters which can be modified by the user.
    An example of this is given below.

    It is strongly recommended to use the backtracking (bt) line search option,
    which improves robustness and takes care of NaNs. More info about this
    technique at:
    https://en.wikipedia.org/wiki/Backtracking_line_search
    """
    solver_settings = {"snes_solver": {"linear_solver": "mumps",
                                       "maximum_iterations": 200,
                                       "report": True,
                                       "error_on_nonconvergence": True,
                                       'line_search': 'bt',
                                       'relative_tolerance': 1e-2}}

    # Define the boundary conditions of the Stokes simulation.
    bcs_stokes = {'Tube_Wall_R': {'Dirichlet': ['v', fn.Constant((0., 0.))]}}

    return req_inputs, bcs_electrostatics, solver_settings, bcs_stokes


if __name__ == '__main__':
    # Load the inputs from the function. Edit the inputs in that function.
    must_inputs, bcs_poisson, solver_settings, bcs_fluids = define_inputs()

    # Call the MAIN script.
    main = main_caller(liquid='EMIBF4', required_inputs=must_inputs, electrostatics_bcs=bcs_poisson,
                       Stokes_bcs=bcs_fluids, electrostatics_solver_settings=solver_settings,
                       surface_update_parameter=0.05)

    # Plot the solutions of the different simulations.
    """ To get a full list of all available inputs for this function, at this point the user may call the
    line SciencePlotting.available_methods_options(), or type help(SciencePlotting).
    """
    main.simulation_results.Electrostatics.plot_results(save_images=True, save_mat=True, image_format='eps')
    main.simulation_results.Stokes.plot_results(save_images=True, save_mat=True, image_format='eps')
    main.simulation_results.SurfaceUpdate.plot_results(save_images=True, save_mat=True, image_format='eps', open_folders=True)
