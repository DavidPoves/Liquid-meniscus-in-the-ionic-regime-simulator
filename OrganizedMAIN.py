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

# Import the libraries.
import os

from Solvers.Poisson_solver import Poisson
from Solvers.NS_Solver import Stokes as Stokes_sim

import fenics as fn
import numpy as np
import matplotlib.pyplot as plt

from Input_Parameters import Liquid_Properties
from Tools.PlotPy import PlotPy
from Tools.PostProcessing import PostProcessing
from Tools.MeshConverter import msh2xml
from Tools.SurfaceUpdate import SurfaceUpdate

from MainMenu import run_main_menu


"""
This is the main script of the Bachelor Thesis:
SIMULATION OF THE LIQUID MENISCUS IN THE IONIC REGIME OF CAPILLARY
ELECTROSPRAY THRUSTERS. This will study the equilibrium of the region between
an ionic liquid and vacuum at a meniscus of specified dimensions. In order to
do so, the Laplace and the Navier-Stokes_sim equations for an especific liquid will
be solved using the open source FEM library FEniCS. Another equations will be
solved to finally study the equilibrium at the interface.
This project is heavily based on the the thesis by Chase Coffman:
Electrically-Assisted Evaporation of Charged Fluids: Fundamental Modeling and
Studies on Ionic Liquids.
"""

# %% DEFINE THE MAIN WRAPPER CLASS.


class MainWrapper(object):
    def __init__(self, liquid, required_inputs, electrostatics_bcs, stokes_bcs, **kwargs):
        """
        Initialize the main class. This class acts as a wrapper of all the classes that are required for a proper
        functionality of the simulation process. From the object generated with the call of this __init__method, the
        user will be able to access all the generated data, using the following methods:
            - self.useful_functions: With this method, the user will be able to access the following functions:
                -> self.useful_functions.check_available_plotting_backgrounds(): Check all possible backgrounds that can
                    be used for plotting.
                -> self.useful_functions.check_electrostatics_solver_options(): Check all the possibilities that are
                    available for the user to setup the solver for the electrostatics solver.
                -> self.useful_functions.check_available_liquids(): Check all the liquids that can be selected by the
                    user.
            - self.plotting_settings: Class containing all the plotting options. All plotting options can be changed by
                calling the self.plotting_settings.plotpy.apply_kwargs(kwargs), where kwargs are all the keywords of the
                parameters to be changed.
            - self.liquid_properties: Object containing all the information related with the physical properties of the
                selected liquid. One may check at any time the chosen liquid by calling the
                self.liquid_properties.liquid_used method. To check all the available data, please refer to the
                Input_Parameters.py file, or call the dir(self.liquid_properties) function.
            - self.geometry_info: Object containing all the information regarding the generation of the .geo file. Here,
                one may check the introduced interface function. For example, if the user has chosen a z(r) function,
                we may check that the function was introduced properly by calling the self.geometry_info.interface_fun
                variable. On the other side, if the user has introduced r(s) and z(s) functions, being s the independent
                parameter, one may check these functions by calling the self.geometry_info.interface_fun_r and
                self.geometry_info.interface_fun_z attributes, respectively. To get a full view of all the available
                attributes, refer to the GMSHInterface class from the GMSH_Interface.py class, or call the
                dir(self.geometry_info) function.
            - self.mesh_info: Object containing all the information regarding the mesh. Here, the user may check the
                algorithm used to create the mesh, the minimum and maximum sizes of the cells conforming the cells,
                among many other parameters. To get a full view of all possible attributes, check the Mesh class from
                the Options.py file of the py2gmsh library:
                https://github.com/tridelat/py2gmsh/blob/master/py2gmsh/Options.py
                or call the dir(self.mesh_info) function.
            - self.file_info: Object containing all the information related to file creation. Here, the user may check
                the name of the .geo file, the name and path of the .msh or .xml files, among many other attributes.
                To get a broader of all available possibilities, check the FilePlacement class or check all possible
                attributes by calling the dir(self.file_info) function.
            - self.general_inputs: Object containing all the parameters that are required for all simulations, or
                parameters that are useful at several parts of the simulation process. To check all available
                attributes, check the SimulationGeneralParameters class or call the dir(self.general_inputs) function.
            - self.simulation_results: Object containing all the information generated during the simulations. Available
                attributes are self.simulation_results.Electrostatics, self.simulation_results.Stokes and
                self.simulation_results.SurfaceUpdate. All of these methods contain all the methods and attributes
                generated or defined during the simulations. To check all of them, refer to the method of
                extract_all_info, which can be found at all solvers wrappers. With self.simulation_results the user may
                check which liquid has been used for the simulation with the self.simulation_results.liquid_used
                attribute.
        Args:
            liquid: String containing the liquid to be used. To check all available liquids, initialize the class, call
                the self.useful_functions.check_available_liquids() to get a list with all possible liquids, then, one
                may want to change the chosen liquid, simply re-initialize the class with the preferred liquid option.
            required_inputs: A dictionary containing the following parameters:
                - B: Value of the ratio of the radius at the tip to the radius of the capillary tube.
                - Lambda: Non dimensional sensitivity of the electric conductivity to changes in temperature.
                - E0: External field parameter.
                - C_R: Parameter that includes friction effects from the reservoir along the channel up to the emission
                    region.
                - T_h: Non dimensional temperature.
                - P_r: Non dimensional pressure at reservoir.
            electrostatics_bcs: Dictionary containing the boundary conditions of the electrostatics problem. To know the
                structure of this dictionary, check the define_inputs function from the main_caller.py file.
            stokes_bcs: Same as electrostatics boundary conditions. They both have the same structure.
            **kwargs: Besides all the kwargs that are accepted by the different classes and functions (check classes,
                files for more information, since there is a huge amount of kwargs available for most of the functions),
                one may define the following kwargs:
                    - relative_permittivity: Relative permittivity of the medium wrt vacuum. Default is 10.
                    - checks_folder_name: String containing the name of the folder where the checks files will be saved.
                        Default is CHECKS.
                    - restrictions_folder_name: String containing the name of the folder where the checks files will be
                        saved. Default is RESTRICTIONS.
                    - convection_charge: Required if there is convection charge. Optional, default is 0.
                    - electrostatics_solver_settings: Settings to be used to solve the electrostatics problem. If no
                        parameter is defined, default settings will be loaded. These settings are the ones that have
                        proven to the best to solve this particular problem. Optional, default is None.
                    - run_full_simulation: Boolean indicating if the full simulation process must be executed at once.
                        Optional, default is True.
                    - run_electrostatics_simulation: Boolean indicating if only the electrostatics simulation must be
                        carried out. Only useful if all the other parameters of this class have already been defined.
                        Optional, default is False.
                    - run_stokes_simulation: Boolean indicating if only the Stokes simulation must be carried out. Only
                        useful if all the other parameters of this class have already been defined. Optional, default is
                        False.
                    - initial_potential: Dolfin/FEniCS function containing a function of the potential from another
                        guess. This parameter will be used as an initial guess for the solver of the electrostatics.
                        The initial_surface_charge_density must be also introduced if this kwarg is introduced.
                        Optional, default is None.
                    - initial_surface_charge_density: Dolfin/FEniCS function containing a function of the surface charge
                        density from another guess. This parameter will be used as an initial guess for the solver of
                        the electrostatics. The initial_potential must be also introduced if this kwarg is introduced.
                        Optional, default is None.
        """

        # Load some default values to some of the kwargs.
        kwargs.setdefault('relative_permittivity', 10)
        kwargs.setdefault('checks_folder_name', 'CHECKS')
        kwargs.setdefault('restrictions_folder_name', 'RESTRICTIONS')
        kwargs.setdefault('convection_charge', 0)
        kwargs.setdefault('electrostatics_solver_settings', None)
        kwargs.setdefault('run_full_simulation', True)
        kwargs.setdefault('run_electrostatics_simulation', False)
        kwargs.setdefault('run_stokes_simulation', False)
        kwargs.setdefault('initial_potential', None)
        kwargs.setdefault('initial_surface_charge_density', None)

        # Load useful functions.
        self.useful_functions = UsefulFunctions()

        # Load the plotting options.
        self.plotting_settings = PlottingSettings(**kwargs)

        # Load the selected liquid properties.
        self.liquid_properties = Liquid_Properties(liquid, relative_permittivity=kwargs.get('relative_permittivity'))

        # Create the geometry and the mesh with a GUI (Graphical User Interface).
        self.gui_info = FilePlacement.call_gui()
        self.geometry_info = self.gui_info.geom_data.geo_gen
        self.mesh_info = self.gui_info.geom_data.geo_gen.my_mesh.Options.Mesh

        # Initialize the class containing all the file info and retrieve it.
        self.file_info = FilePlacement()
        self.file_info.get_main_file_info(self.gui_info, **kwargs)
        self.file_info.get_xml_files()

        # Initialize the general inputs class.
        self.general_inputs = SimulationGeneralParameters(required_inputs, self.liquid_properties)

        # Initialize the Simulation class.
        self.simulation_results = RunSimulation(self.general_inputs, self.liquid_properties, self.file_info,
                                                electrostatics_bcs, stokes_bcs, **kwargs)


class UsefulFunctions(object):
    def __init__(self):
        pass

    @staticmethod
    def check_available_plotting_backgrounds():
        print(plt.style.available)

    @staticmethod
    def check_electrostatics_solver_options():
        Poisson.check_solver_options()

    @staticmethod
    def check_available_liquids():
        Liquid_Properties.check_available_liquids()


class FilePlacement(object):
    def __init__(self):
        """
        Initialize the class where all the file-related info will be stored.
        """
        # Preallocate folder-related data.
        self.root_folder = os.getcwd()
        self.mesh_objects_folder = None
        self.restrictions_folder = None
        self.checks_folder = None

        self.geo_filename = None
        self.geo_filepath = None
        self.msh_filename = None
        self.msh_filepath = None
        self.xml_filename = None
        self.xml_filepath = None

    @staticmethod
    def call_gui():
        """
        Call the graphical user interface that will guide the user through the geometry generation.
        Returns:

        """
        return run_main_menu()

    def get_main_file_info(self, gui_info, **kwargs):
        """
        Obtain the main information regarding the folders and files generated by the GUI.
        Args:
            gui_info: Object containing all the information generated by the GUI.
            **kwargs: User kwargs are:
                        - checks_folder_name: String containing the name of the folder where check files will be stored.
                        - restrictions_folder_name: String containing the name of the folder where the restrictions
                        files will be stored.

        Returns:

        """
        # Get .msh-related information.
        self.msh_filepath = gui_info.msh_filename
        self.msh_filename = self.msh_filepath.split('/')[-1]

        # Get .geo-related information
        self.geo_filename = self.msh_filename.split('.')[0] + '.geo'
        self.mesh_objects_folder = self.msh_filepath.split(self.msh_filename)[0][:-1]  # Get mesh objects' folder.
        self.geo_filepath = self.mesh_objects_folder + '/' + self.geo_filename

        # Define the checks and restrictions folder.
        self.checks_folder = self.root_folder + '/' + kwargs.get('checks_folder_name')
        self.restrictions_folder = self.root_folder + '/' + kwargs.get('restrictions_folder_name')

    def get_xml_files(self):
        """
        Obtain .xml files from the .msh file. This method will generate the mesh.xml file and the boundaries and
        subdomains xml files (facet_region.xml and physical_region.xml files, respectively).
        Returns:

        """
        
        self.xml_filename = msh2xml(self.msh_filename, self.root_folder, self.mesh_objects_folder)
        self.xml_filepath = self.mesh_objects_folder + '/' + self.xml_filename


class PlottingSettings():
    def __init__(self, **kwargs):
        """
        Initialize the Plotting class.
        Args:
            **kwargs: Used kwargs are:
                        - use_latex: Boolean indicating whether to use latex interpreter or not.
                        - grid_properties: Dictionary defining the properties the graph grid should have.
                        - font_properties: Dictionary containing the font properties.
                        - save_images: Boolean indicating whether to save the generated plots or not.
                        - save_mat: Boolean indicating whether to export the plot data into .mat files or not.
        """
        # Set default parameters to the kwargs.
        kwargs.setdefault('use_latex', True)
        kwargs.setdefault('grid_properties', None)
        kwargs.setdefault('font_properties', None)
        kwargs.setdefault('save_images', False)
        kwargs.setdefault('save_mat', False)

        # Load the plotting options.
        """
        All the available fonts for matplotlib:
        https://matplotlib.org/1.4.0/users/usetex.html

        To see a full list of background styles check the following code:
        plt.style.available

        Notice there are some background styles where linewidth of the grid cannot be
        controlled. In that case, this parameter will be ignored
        All latex font styles can be found at:
        https://www.overleaf.com/learn/latex/Font_sizes,_families,_and_styles
        """
        if kwargs.get('grid_properties') is None:
            self.grid_properties = {'linestyle': "--", 'color': 'black', 'linewidth': 0.2}
        else:
            self.grid_properties = kwargs.get('grid_properties')

        if kwargs.get('font_properties') is None:
            self.font_properties = {'family': 'serif', 'style': 'Computer Modern Roman',
                                    'fig_title_style': 'bold',
                                    'legend_title_style': 'bold',
                                    'legend_labels_style': 'slanted',
                                    'axis_labels_style': 'slanted'}
        else:
            self.font_properties = kwargs.get('font_properties')

        self.background_style = 'seaborn-paper'
        self.fontsize = 12.
        self.figsize = (12., 7.)
        self.use_latex = kwargs.get('use_latex')
        self.plotpy = PlotPy(latex=self.use_latex, fontsize=self.fontsize, figsize=self.figsize,
                             background_style=self.background_style,
                             font_properties=self.font_properties,
                             grid_properties=self.grid_properties,
                             save_images=kwargs.get('save_images'), save_mat=kwargs.get('save_mat'),
                             extension='jpg')


class RunSimulation(object):
    def __init__(self, general_inputs, liquid_properties, file_info, electrostatics_bcs, stokes_bcs, **kwargs):
        """
        Initialize the class containing all the processes related with the simulations executions.
        Args:
            general_inputs: Object containing the SimulationGeneralParameters class.
            liquid_properties: Object obtained from the Liquid_Parameters class, which is located at Input_Parameters.py
            file_info: Object containing the FilePlacement class.
            electrostatics_bcs: Dictionary containing the electrostatics boundary conditions.
            stokes_bcs: Dictionary containing the Stokes boundary conditions.
            **kwargs: All kwargs accepted by the ElectrostaticsWrapper class.
        """

        # Load some useful parameters for user visualization.
        self.liquid_used = liquid_properties.liquid_used

        # Initialize (or preallocate) the simulation classes.
        self.Electrostatics = ElectrostaticsWrapper(general_inputs, liquid_properties, electrostatics_bcs, **kwargs)

        if kwargs.get('run_full_simulation'):
            # Run electrostatics simulation.
            self.Electrostatics.run_full_simulation(file_info)
            self.Electrostatics.extract_all_info(general_inputs, liquid_properties)

            # Run Stokes simulation.
            self.Stokes = StokesWrapper(general_inputs, liquid_properties, stokes_bcs, self.Electrostatics)
            self.Stokes.run_solver(file_info)
            self.Stokes.extract_all_info(general_inputs)

        elif kwargs.get('run_electrostatics_simulation'):
            self.Electrostatics.run_full_simulation(file_info)

        elif kwargs.get('run_stokes_simulation'):
            self.Stokes.run_solver(file_info)


class SimulationGeneralParameters(object):
    def __init__(self, general_inputs, liquid_properties):
        """
        Load data that is used by all the simulations.
        Args:
            general_inputs: Initial inputs (the ones introduced initially by the user).
            liquid_properties: Object obtained from the Liquid_Parameters class, which is located at Input_Parameters.py
        """
        # Load general parameters.
        self.B = general_inputs['B']  # Ratio of characteristic emission region r* to radius of the fluid channel r0.
        self.T_h = general_inputs['T_h']  # Non dimensional temperature.
        self.E0 = general_inputs['E0']  # Factor determining the upper electrode non-dimensional voltage.
        self.C_R = general_inputs['C_R']  # Non dimensional that includes friction from reservoir to the channel.
        self.P_r = general_inputs['P_r']  # Non dimensional pressure at reservoir.
        self.Lambda = general_inputs['Lambda']  # Non dimensional sensitivity of electric conduct. to changes in temp.
        self.T_0 = 298.15  # Reference temperature [K]

        # Compute terms coming from depending on general and liquid inputs.
        self.Chi = (liquid_properties.h * liquid_properties.k_prime) / (
                    self.Lambda * liquid_properties.k_B * liquid_properties.vacuum_perm * liquid_properties.eps_r)
        self.Phi = liquid_properties.Solvation_energy / (liquid_properties.k_B * self.T_0 * self.T_h)
        self.E_star = (4 * np.pi * liquid_properties.vacuum_perm * liquid_properties.Solvation_energy ** 2) / \
                      1.60218e-19 ** 3  # Critical field of emission.
        self.r_star = 4*liquid_properties.gamma / (liquid_properties.vacuum_perm *
                                                   self.E_star**2)  # Characteristic evaporation region.
        self.r0 = self.r_star / self.B

        # Preallocate required coordinates.
        self.r_nodes, self.z_nodes = None, None
        self.r_mids, self.z_mids = None, None

    def get_nodepoints(self, mesh, boundaries, boundary_id):
        """
        Obtain the node points of a given boundary.
        Args:
            mesh: Dolfin/FEniCS mesh object.
            boundaries: Dolfin/FEniCS MeshFunction object containing the boundaries information.
            boundary_id: Integer identifying the specific boundary. This id comes from the .geo file. More specifically,
            from the id of the physical curve corresponding to the boundary.

        Returns:

        """
        self.r_nodes, self.z_nodes = PostProcessing.get_nodepoints_from_boundary(mesh, boundaries, boundary_id)
        return self.r_nodes, self.z_nodes

    def get_midpoints(self, boundaries, boundary_id):
        """
        Obtain the midpoints of the facets conforming a specific boundary.
        Args:
            boundaries: Dolfin/FEniCS MeshFunction object containing the boundaries information.
            boundary_id: Integer identifying the specific boundary. This id comes from the .geo file. More specifically,
            from the id of the physical curve corresponding to the boundary.

        Returns:

        """
        self.r_mids, self.z_mids = PostProcessing.get_midpoints_from_boundary(boundaries, boundary_id)
        return self.r_mids, self.z_mids


class ElectrostaticsWrapper(PlottingSettings):
    def __init__(self, general_inputs, liquid_properties, electrostatics_bcs, **kwargs):
        """
        Initialize the Electrostatics class. It contains all the available methods to fully run a electrostatics
        simulation and plot all the obtained results.
        Args:
            general_inputs: Object obtained from the SimulationGeneralParameters class.
            liquid_properties: Object obtained from the Liquid_Parameters class, which is located at Input_Parameters.py
            electrostatics_bcs: Dictionary containing the electrostatics boundary conditions.
            **kwargs: Used kwargs are:
                        - init_boundary_conditions_elec: Dictionary containing the boundary conditions for the initial
                        problem. Required if no initial guess is provided.
                        - electrostatics_solver_settings: Dictionary containing the settings to be used by the iterative
                        solver.
                        - convection_charge: If user provides convection charge, it will be loaded into the solver.
                        Otherwise, this term is considered to be 0.
                        - initial_potential: For an iterative process, if there exists a result from a previous
                        iteration, user may load the obtained potential as an initial guess.
                        - initial_surface_charge_density: For an iterative process, if there exists a result from a
                        previous iteration, user may load the obtained surface charge density as an initial guess.

        """
        super().__init__(**kwargs)  # Inherit the Plotting class.

        # Load necessary data.
        self.general_inputs = general_inputs
        self.liquid_properties = liquid_properties

        # Define the boundary conditions.
        self.boundary_conditions = electrostatics_bcs

        # Define default solver settings.
        if kwargs.get('electrostatics_solver_settings') is None:
            self.solver_settings = {"snes_solver": {"linear_solver": "mumps",
                                                    "maximum_iterations": 200,
                                                    "report": True,
                                                    "error_on_nonconvergence": True,
                                                    'line_search': 'bt',
                                                    'relative_tolerance': 1e-4}}
        else:
            self.solver_settings = kwargs.get('electrostatics_solver_settings')

        # Define the inputs for the Electrostatics solver.
        self.inputs = {'Relative_perm': liquid_properties.eps_r,
                       'Non_dimensional_temperature': general_inputs.T_h,
                       'Lambda': general_inputs.Lambda,
                       'Phi': general_inputs.Phi,
                       'B': general_inputs.B,
                       'Chi': general_inputs.Chi,
                       'Convection charge': kwargs.get('convection_charge')}

        # Preallocate all variables.
        self.class_caller = None
        self.mesh = None
        self.boundaries, self.boundaries_ids = None, None
        self.subdomains, self.subdomains_ids = None, None
        self.restrictions_dict = None
        self.potential = None
        self.vacuum_electric_field, self.normal_component_vacuum = None, None
        self.tangential_component = None
        self.surface_charge_density = None
        self.radial_component_vacuum, self.axial_component_vacuum = None, None
        self.normal_component_liquid = None
        self.radial_component_liquid, self.axial_component_liquid = None, None
        self.evaporated_charge, self.conducted_charge = None, None
        self.emitted_current = None
        self.normal_electric_stress = None
        self.coords_nodes, self.coords_mids = None, None

    def run_full_simulation(self, file_info, **kwargs):
        """
        Execute the full process of running an electrostatics simulation. This method will generate the subdomains and
        boundaries data variables; the restrictions files and variables; and the check files. Finally, the solver is
        executed.
        Args:
            file_info: Object containing the FilePlacement class.
            **kwargs: Same kwargs as the __init__ method of this class.

        Returns:

        """
        self.class_caller = Poisson(self.inputs, self.boundary_conditions, file_info.xml_filepath,
                                    file_info.restrictions_folder, file_info.checks_folder)

        # Get the mesh file.
        self.mesh = self.class_caller.get_mesh()

        # Get the boundaries data and its ids.
        self.boundaries, self.boundaries_ids = self.class_caller.get_boundaries()

        # Get the subdomains data and its ids.
        self.subdomains, self.subdomains_ids = self.class_caller.get_subdomains()

        # Write check files for previsualization of the generated data.
        self.write_check_files()

        # Generate multiphenics' restrictions.
        self.generate_restrictions()

        # Export the previous restrictions into readable .xml files.
        self.write_restrictions()

        # Solve the problem.
        self.run_solver(**kwargs)

    def write_check_files(self):
        """
        Write check files at the specified folder. These files are extremely useful to check that all the data has been
        generated as expected. It is very important to check these files prior to proceeding with the rest of the code,
        since any later error related with a wrong subdomain marking, boundaries length or some other types of errors
        may be detected by taking a look at these files. For visualizing these files, Paraview is recommended. To do so,
        just open files with the .xdmf extension.
        Returns:

        """
        self.class_caller.write_check_files()

    def generate_restrictions(self):
        """
        Restrictions are the objects (or files) used by the multiphenics library to recognize a specific
        subdomain/boundary. These are useful to define functions on a specific subdomain or boundary, which is the case
        of the Lagrangian multiplier, which for this problem should be defined only at the interface boundary.
        Returns:

        """
        self.restrictions_dict = self.class_caller.generate_restrictions()

    def write_restrictions(self):
        """
        Export the generated restrictions into .xml files which can be later read by multiphenics. This exporting
        process avoids generating new restrictions if they have been exported.
        it is important to visually check the generated restrictions before continuing with the code. Otherwise, any
        error related with these restrictions will be hard to debug.
        Returns:

        """
        self.class_caller.write_restrictions()

    def run_solver(self, **kwargs):
        """
        Method to run just the Electrostatics solver with the existing parameters of the class.
        Returns:

        """
        self.class_caller.solve(electrostatics_solver_settings=self.solver_settings, **kwargs)

    def extract_all_info(self, general_inputs, liquid_properties):
        """
        Extract all the important data generated from the electrostatics simulation.
        Args:
            general_inputs: Object containing the SimulationGeneralParameters class.
            liquid_properties: Object obtained from the Liquid_Parameters class, which is located at Input_Parameters.py

        Returns:

        """
        self.potential = self.class_caller.phi
        self.vacuum_electric_field = self.class_caller.E_v  # Vacuum electric field.
        self.normal_component_vacuum = self.class_caller.E_v_n  # Normal component of the electric field @interface
        self.tangential_component = self.class_caller.E_t  # @interface
        self.surface_charge_density = self.class_caller.sigma  # Surface charge density at interface.

        # Get coordinates of the nodes and midpoints.
        r_nodes, z_nodes = general_inputs.get_nodepoints(self.mesh, self.boundaries, self.boundaries_ids['Interface'])
        self.coords_nodes = [r_nodes, z_nodes]
        r_mids, z_mids = general_inputs.get_midpoints(self.boundaries, self.boundaries_ids['Interface'])
        self.coords_mids = [r_mids, z_mids]

        # Split the electric field into radial and axial components.
        self.radial_component_vacuum, self.axial_component_vacuum = \
            PostProcessing.extract_from_function(self.vacuum_electric_field, self.coords_nodes)

        # E_v_n_array = PostProcessing.extract_from_function(Electrostatics.normal_component_vacuum, coords_mids)
        E_t_array = PostProcessing.extract_from_function(self.tangential_component, self.coords_mids)

        # Define an auxiliary term for the computations.
        K = 1 + general_inputs.Lambda * (general_inputs.T_h - 1)

        self.normal_component_liquid = (self.normal_component_vacuum - self.surface_charge_density) / \
                                       liquid_properties.eps_r
        E_l_n_array = PostProcessing.extract_from_function(self.normal_component_liquid, self.coords_mids)

        # Get components of the liquid field.
        self.radial_component_liquid, self.axial_component_liquid = \
            Poisson.get_liquid_electric_field(mesh=self.mesh, subdomain_data=self.boundaries,
                                              boundary_id=self.boundaries_ids['Interface'], normal_liquid=E_l_n_array,
                                              tangential_liquid=E_t_array)
        self.radial_component_liquid.append(self.radial_component_liquid[-1])
        self.axial_component_liquid.append(self.axial_component_liquid[-1])

        # Calculate the non-dimensional evaporated charge and current.
        self.evaporated_charge = (self.surface_charge_density * general_inputs.T_h) / (liquid_properties.eps_r *
                                                                                       general_inputs.Chi) \
                                 * fn.exp(-general_inputs.Phi / general_inputs.T_h * (1 - pow(general_inputs.B, 1 / 4) *
                                                                                      fn.sqrt(
                                                                                          self.normal_component_vacuum))
                                          )
        self.conducted_charge = K * self.normal_component_liquid

        # Calculate the emitted current through the interface.
        self.emitted_current = self.class_caller.get_nd_current(self.evaporated_charge)

        # Compute the normal component of the electric stress at the meniscus (electric pressure).
        self.normal_electric_stress = (self.normal_component_vacuum ** 2 - liquid_properties.eps_r *
                                       self.normal_component_liquid ** 2) + \
                                      (liquid_properties.eps_r - 1) * self.tangential_component ** 2

    def plot_results(self, save_images=False, save_mat=False):
        """
        Plot the most important data obtained from the Electrostatics simulation.
        Args:
            save_images: Boolean indicating if images must be saved or not.
            save_mat: Boolean indicating if data must be saved into .mat files.

        Returns:

        """

        # Extract the information from FEniCS functions to numpy arrays.
        E_v_n_array = PostProcessing.extract_from_function(self.normal_component_vacuum, self.coords_mids)
        E_l_n = (self.normal_component_vacuum - self.surface_charge_density) / self.liquid_properties.eps_r
        E_l_n_array = PostProcessing.extract_from_function(E_l_n, self.coords_mids)
        E_t_array = PostProcessing.extract_from_function(self.tangential_component, self.coords_mids)

        sigma_arr = PostProcessing.extract_from_function(self.surface_charge_density, self.coords_mids)
        j_ev_arr = PostProcessing.extract_from_function(self.evaporated_charge, self.coords_mids)
        n_taue_n_arr = PostProcessing.extract_from_function(self.normal_electric_stress, self.coords_mids)

        # Define generic labels.
        x_label = r'$\hat{r}$'
        y_label = r'$\hat{E}$'

        # Introduce user inputs into plotting class.
        self.plotpy.apply_kwargs(save_images=save_images, save_mat=save_mat)

        self.plotpy.lineplot([(self.general_inputs.r_nodes, self.radial_component_vacuum, r'Radial ($\hat{r}$)'),
                              (self.general_inputs.r_nodes, self.axial_component_vacuum, r'Axial ($\hat{z}$)')],
                             xlabel=x_label, ylabel=y_label,
                             fig_title='Radial and axial components of the vacuum electric field',
                             legend_title='Field Components')

        self.plotpy.lineplot([(self.general_inputs.r_nodes, self.radial_component_liquid, r'Radial ($\hat{r}$)'),
                              (self.general_inputs.r_nodes, self.axial_component_liquid, r'Axial ($\hat{z}$)')],
                             xlabel=x_label, ylabel=y_label,
                             fig_title='Radial and axial components of the liquid electric field',
                             legend_title='Field Components')

        self.plotpy.lineplot([(self.general_inputs.r_mids, E_t_array)],
                             xlabel=x_label, ylabel=r'$\hat{E}_t$',
                             fig_title='Tangential component of the electric field at the meniscus')

        self.plotpy.lineplot([(self.general_inputs.r_mids, E_v_n_array, 'Vacuum'),
                              (self.general_inputs.r_mids, E_l_n_array, 'Liquid')],
                             xlabel=x_label, ylabel=y_label,
                             fig_title='Normal components of the electric fields',
                             legend_title='Subdomains')

        self.plotpy.lineplot([(self.general_inputs.r_mids, sigma_arr)],
                             xlabel=x_label, ylabel=r'$\hat{\sigma}$',
                             fig_title='Radial evolution of the surface charge density')

        self.plotpy.lineplot([(self.general_inputs.r_mids, j_ev_arr)],
                             xlabel=x_label, ylabel=r'$\hat{j}_e$',
                             fig_title='Charge evaporation along the meniscus')

        self.plotpy.lineplot([(self.general_inputs.r_mids, n_taue_n_arr)],
                             xlabel=x_label, ylabel=r'$\mathbf{n}\cdot\hat{\bar{\bar{\tau}}}_e \cdot \mathbf{n}$',
                             fig_title='Normal component of the electric stress at the meniscus')


class StokesWrapper(PlottingSettings):
    def __init__(self, general_inputs, liquid_properties, stokes_bcs, electrostatics_simulation):
        """
        Initialize the Stokes class. This class contains all the necessary methods to fully run a Stokes simulation and
        plot all the obtained results.
        Args:
            general_inputs: Object obtained from the SimulationGeneralParameters class.
            liquid_properties: Object obtained from the Liquid_Parameters class, which is located at Input_Parameters.py
            stokes_bcs: Dictionary containing the Stokes boundary conditions.
            electrostatics_simulation: Object containing all the electrostatics simulation results. Obtained from
            the ElectrostaticsWrapper class.
        """
        super().__init__()  # Inherit the Plotting class.

        # Compute required initial quantities.
        E_c = np.sqrt((4 * liquid_properties.gamma) / (general_inputs.r0 * liquid_properties.vacuum_perm))
        self.reference_electric_field = E_c
        E_star = (4 * np.pi * liquid_properties.vacuum_perm * liquid_properties.Solvation_energy ** 2) / 1.60218e-19 ** 3
        self.characteristic_field_of_emission = E_star
        j_star = liquid_properties.k_0 * E_star / liquid_properties.eps_r
        self.characteristic_current_density = j_star
        u_star = j_star / (liquid_properties.rho_0 * liquid_properties.q_m)  # Characteristic velocity.
        self.characteristic_fluid_velocity = u_star
        r_star = general_inputs.B * general_inputs.r0  # Cone tip radius.
        self.tip_radius = r_star
        We = (liquid_properties.rho_0 * u_star ** 2 * r_star) / (2 * liquid_properties.gamma)  # Weber number.
        self.Weber = We
        Ca = liquid_properties.mu_0 * u_star / (2 * liquid_properties.gamma)  # Capillary number.
        self.Capillary = Ca
        Kc = (liquid_properties.vacuum_perm * liquid_properties.eps_r * u_star) / (liquid_properties.k_0 * r_star)
        self.Kc = Kc

        # Define the simulator inputs.
        self.inputs = {'Weber number': self.Weber,
                       'Capillary number': self.Capillary,
                       'Relative perm': liquid_properties.eps_r,
                       'B': general_inputs.B,
                       'Lambda': general_inputs.Lambda,
                       'Non dimensional temperature': general_inputs.T_h,
                       'Sigma': electrostatics_simulation.surface_charge_density,
                       'Phi': general_inputs.Phi,
                       'Chi': general_inputs.Chi,
                       'Potential': electrostatics_simulation.potential,
                       'Kc': self.Kc,
                       'Vacuum electric field': electrostatics_simulation.vacuum_electric_field,
                       'Vacuum normal component': electrostatics_simulation.normal_component_vacuum}

        # Load the boundary conditions.
        self.boundary_conditions = stokes_bcs

        # Load the electrostatics simulation results.
        self.electrostatics_simulation = electrostatics_simulation

        self.class_caller = None

        # Preallocate all data variables.
        self.velocity_field = None
        self.radial_component_velocity, self.axial_component_velocity = None, None
        self.pressure = None
        self.normal_hydraulic_stress = None
        self.r_nodes, self.z_nodes = None, None
        self.r_mids, self.z_mids = None, None
        self.coords_nodes, self.coords_mids = None, None
        self.convection_charge = None
        self.normal_velocity, self.tangential_velocity = None, None

    def run_solver(self, file_info):
        """
        Initialize the Stokes simulator class and solve the problem.
        Args:
            file_info: class containing all the file-related information. Object containing the FilePlacement class.

        Returns:

        """
        self.class_caller = Stokes_sim(self.inputs, self.boundary_conditions,
                                       subdomains=self.electrostatics_simulation.subdomains,
                                       boundaries=self.electrostatics_simulation.boundaries,
                                       mesh=self.electrostatics_simulation.mesh,
                                       boundaries_ids=self.electrostatics_simulation.boundaries_ids,
                                       restrictions_path=file_info.restrictions_folder,
                                       mesh_path=file_info.mesh_objects_folder, filename=file_info.msh_filename)
        # Solve the Stokes system
        self.class_caller.solve()

    def extract_all_info(self, general_inputs):
        """
        Extract all possible information from the Stokes simulation.
        Args:
            general_inputs: Class containing inputs which are general for all simulations. Object containing the
            SimulationGeneralParameters class.

        Returns:

        """
        self.r_nodes, self.z_nodes = general_inputs.r_nodes, general_inputs.z_nodes
        self.coords_nodes = [self.r_nodes, self.z_nodes]
        self.r_mids, self.z_mids = general_inputs.r_mids, general_inputs.z_mids
        self.coords_mids = [self.r_mids, self.z_mids]

        self.velocity_field = self.class_caller.u

        self.radial_component_velocity, self.axial_component_velocity = PostProcessing.extract_from_function(
            self.velocity_field, self.coords_nodes)

        self.normal_velocity = self.class_caller.u_n
        self.tangential_velocity = self.class_caller.u_t

        self.pressure = self.class_caller.p_star - general_inputs.P_r + \
            self.electrostatics_simulation.emitted_current*general_inputs.C_R

        self.normal_hydraulic_stress = \
            self.class_caller.block_project(self.class_caller.theta,self.electrostatics_simulation.mesh,
                                            self.electrostatics_simulation.restrictions_dict['interface_rtc'],
                                            self.electrostatics_simulation.boundaries,
                                            self.electrostatics_simulation.boundaries_ids['Interface'],
                                            space_type='scalar', boundary_type='internal')
        self.convection_charge = self.class_caller.j_conv

    def plot_results(self, save_images=False, save_mat=False):
        """
        Plot the results obtained from the Stokes simulation.
        Args:
            save_images: Boolean indicating if images should be saved or not.
            save_mat: Boolean indicating if .mat files from the plotted data should be saved or not.

        Returns:

        """

        # Obtain the arrays from the FEniCS functions.
        p_arr = PostProcessing.extract_from_function(self.pressure, self.coords_nodes)
        n_tauh_n_arr = PostProcessing.extract_from_function(self.normal_hydraulic_stress, self.coords_mids)
        j_conv_arr = PostProcessing.extract_from_function(self.convection_charge, self.coords_nodes)
        u_n_arr = PostProcessing.extract_from_function(self.normal_velocity, self.coords_mids)
        u_t_arr = PostProcessing.extract_from_function(self.tangential_velocity, self.coords_mids)

        # Apply the user's inputs into the plotting class.
        self.plotpy.apply_kwargs(save_images=save_images, save_mat=save_mat)

        # Plot the results.
        self.plotpy.lineplot([(self.r_nodes, self.radial_component_velocity, r'Radial ($\hat{r}$)'),
                              (self.r_nodes, self.axial_component_velocity, r'Axial ($\hat{z}$)')],
                             xlabel=r'$\hat{r}$', ylabel=r'$\hat{u}$',
                             fig_title='Components of the velocity field',
                             legend_title='Field Components')

        self.plotpy.lineplot([(self.r_nodes, p_arr)], xlabel=r'$\hat{r}$', ylabel=r'$\hat{p}$',
                             fig_title='Pressure along the meniscus')

        self.plotpy.lineplot([(self.r_mids, u_n_arr)],
                             xlabel=r'$\hat{r}$', ylabel=r'$\hat{u}_n$',
                             fig_title='Normal Component of the velocity field.')

        self.plotpy.lineplot([(self.r_mids, u_t_arr)],
                             xlabel=r'$\hat{r}$', ylabel=r'$\hat{u}_t$',
                             fig_title='Tangential Component of the velocity field.')

        self.plotpy.lineplot([(self.r_nodes, j_conv_arr)],
                             xlabel=r'$\hat{r}$', ylabel=r'$\hat{j}_{conv}$',
                             fig_title='Convection charge transport')

        self.plotpy.lineplot([(self.r_mids, n_tauh_n_arr)],
                             xlabel=r'$\hat{r}$', ylabel=r'$\mathbf{n}\cdot\hat{\bar{\bar{\tau}}}_m \cdot \mathbf{n}$',
                             fig_title='Normal component of the hydraulic stress at the meniscus')