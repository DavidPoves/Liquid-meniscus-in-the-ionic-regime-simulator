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
import Input_Parameters as Inp_Parameters
from Solvers.Poisson_solver import Poisson
from Solvers.NS_Solver import Navier_Stokes as NS
import fenics as fn
import numpy as np
from Tools.PlotPy import PlotPy
from Tools.PostProcessing import PostProcessing
import gmsh_api.gmsh as gmsh  # pip install gmsh_api


"""
This is the main script of the Bachelor Thesis:
SIMULATION OF THE LIQUID MENISCUS IN THE IONIC REGIME OF CAPILLARY
ELECTROSPRAY THRUSTERS. This will study the equilibrium of the region between
an ionic liquid and vacuum at a meniscus of specified dimensions. In order to
do so, the Laplace and the Navier-Stokes equations for an especific liquid will
be solved using the open source FEM library FEniCS. Another equations will be
solved to finally study the equilibrium at the interface.
This project is heavily based on the the thesis by Chase Coffman:
Electrically-Assisted Evaporation of Charged Fluids: Fundamental Modeling and
Studies on Ionic Liquids.
"""


def Mesh_generator(mesh_folder_path, name):
    # Initialize the gmsh api to get the elements.
    gmsh.initialize()
    # Save the mesh with v2 to use it with dolfin.
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.)
    # Re-open the file.
    gmsh.open(mesh_folder_path + '/' + name + '.geo')

    # Generate the 2D mesh.
    gmsh.model.mesh.generate(2)  # 2 indicating 2 dimensions.
    filename = name + '.msh'
    gmsh.write(mesh_folder_path + '/' + filename)

    # Finalize the gmsh processes.
    gmsh.finalize()


# Prepare graphing tool.
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
grid_properties = {'linestyle': "--", 'color': 'black', 'linewidth': 0.2}
font_properties = {'family': 'serif', 'style': 'Computer Modern Roman',
                   'fig_title_style': 'bold',
                   'legend_title_style': 'bold',
                   'legend_labels_style': 'slanted',
                   'axis_labels_style': 'slanted'}
background_style = 'seaborn-paper'
x_label = r'$\hat{r}$'
y_label = r'$\hat{E}$'

# Initialize the plot class with the desired parameters.
plotpy = PlotPy(latex=True, fontsize=12., figsize=(12., 7.),
                background_style=background_style,
                font_properties=font_properties,
                grid_properties=grid_properties, save_images=False,
                save_mat=False, extension='jpg')


# %% IMPORT AND DECLARE THE INITIAL DATA.
liquid = 'EMIBF4'  # Choose the liquid to be studied.
vacuum_perm, eps_r, q_m, k_B, h_Planck = Inp_Parameters.General_Parameters()
liquid_data = Inp_Parameters.Liquid_Data(liquid)

MW = liquid_data[0]  # Molecular Weight of the liquid [kg/mol]
rho_0 = liquid_data[1]  # Density of the liquid [kg/m^3] (at 298.15K)
mu_0 = liquid_data[2]  # Viscosity of the liquid [Pa-s] (at 298.15K)
k_0 = liquid_data[3]  # Nominal electric conductivity [S/m] (at 298.15K)
k_prime = liquid_data[4]  # Thermal Sensitivity of the liquid [S/m-K]
cp_mass = liquid_data[5]  # Specific heat capacity [J/mol-K] (at 298.15K)
Solvation_Energy = liquid_data[6]  # Nominal Solvation energy [eV]
delta_G = Solvation_Energy * 1.60218e-19  # Transform eV to Joules.
k_T = liquid_data[7]  # Thermal conductivity of the fluid [W/m-K] (at 300K)
T = 298.15  # Reference temperature [K]
gamma = liquid_data[8]  # Intrinsic surface energy of the fluid [N/m].
ref_potential = 0  # Reference potential [V]
q_m = 1e6  # Specific charge density [C/kg]

# Define values for simulation.
r0_list = [1e-6, 2.5e-6, 4e-6, 4.5e-6, 5e-6]
Lambda_list = [1, 12]
E0_list = [0.99, 0.65]
B_list = [0.0467, 0.00934]
eps_r = 10
C_R = 1e3
P_r_h = 0  # Non dimensional pressure at reservoir.

# %% IMPORT THE MESH.
"""
From version of GMSH 3.0 upwards, the format in which the mesh is saved has
changed, and it is not compatible with the format that dolfin-converter is
expecting to recieve. Therefore, the GMSH 2.0 format is required. In order to
do so, when the mesh has already been defined in GMSH, go to:
File > Export > name_of_file.msh and save the file as ASCII v2. In that way,
dolfin-converter will be able to transform the .msh file into .xml, the
extension that is compatible with FEniCS. This is automatically done in the
code.
"""
# Introduce the number of meniscus points.
N = 800

# Create the folders in which the outputs will be stored.
root_folder = os.getcwd()
mesh_folder_name = 'MESH OBJECTS'
mesh_folder_path = root_folder + '/' + mesh_folder_name

restrictions_folder_name = 'RESTRICTIONS'
restrictions_folder_path = root_folder + '/' + restrictions_folder_name

checks_folder_name = 'CHECKS'
checks_folder_path = root_folder + '/' + checks_folder_name

# Open the desired file, whose name is given by the B parameter.
B_file = B_list[0]
B_id = str(B_file).split('.')
separator = ''
B_id = separator.join(B_id)
name = 'B_' + B_id + '_N_' + str(N)

# Generate the .msh file from the geo file.
Mesh_generator(mesh_folder_path, name)

ofilename = name + '.xml'
ifilename_ = name + '.msh' + ' '
ofilename_ = ofilename + ' '  # Formatting for next command.
input_str = "dolfin-convert " + ifilename_ + ofilename_

"""
Next, we call the dolfin-converter, which will transform the generated
.msh file into a readable .xml file, which is the extension accepted by
FEniCS and multiphenics.
"""
os.chdir(mesh_folder_path)
os.system(input_str)  # Call the dolfin-converter
os.chdir(root_folder)

# %% ELECTROSTATICS.
# Define the values to be used for the simulation.
r0 = r0_list[0]
z0 = 10*r0
T_h = 1
Lambda = Lambda_list[0]
B = B_list[0]
E0 = E0_list[0]
top_potential = -E0*z0/r0
Chi = (h_Planck*k_prime)/(Lambda*k_B*vacuum_perm*eps_r)
Phi = delta_G/(k_B*T)

# Define the constant inputs for the solver.
inputs = {'Relative_perm': eps_r,
          'Contact_line_radius': r0,
          'Non_dimensional_temperature': T_h,
          'Lambda': Lambda,
          'Phi': Phi,
          'B': B,
          'Chi': Chi,
          'Convection charge': 0}

# Define the boundary conditions.
"""
Notice the structure of the boundary conditions:

    1. Boundary name as in the .geo file.
    2. The type of boundary condition (Dirichlet or Neumann).
    3. A list where the first value is the value of the bc and the second
        element is the subdomain to which it belongs to.
"""
boundary_conditions_elec = {'Top_Wall': {'Dirichlet': [top_potential, 'vacuum']},
                            'Inlet': {'Dirichlet': [ref_potential, 'liquid']},
                            'Tube_Wall_R': {'Dirichlet': [ref_potential, 'liquid']},
                            'Bottom_Wall': {'Dirichlet': [ref_potential, 'vacuum']},
                            'Lateral_Wall_R': {'Neumann': 'vacuum'},
                            'Lateral_Wall_L': {'Neumann': 'vacuum'},
                            'Tube_Wall_L': {'Neumann': 'vacuum'}}


# Initialize the Electrostatics class, loading constant parameters.
Electrostatics = Poisson(inputs, boundary_conditions_elec, ofilename_.strip(),
                         mesh_folder_path, restrictions_folder_path,
                         checks_folder_path)

# Get the mesh object.
mesh = Electrostatics.get_mesh()

# Generate a Mesh Function containing all subdomains and boundaries.
interface_name = '"Meniscus"'
boundaries, boundaries_ids = Electrostatics.get_boundaries(interface_name)
subdomains = Electrostatics.get_subdomains()

# Write out visualization files.
"""
At this step of the code, it is very important to check that everything has
been defined as expected. For that purpose, IT IS STRONGLY RECOMMENDED TO
CHECK THE GENERATED FILES AT CHECKS FOLDER BEFORE PROCEEDING ANY FURTHER IN THE
CODE. If not revised, later debug will be much harder if any error arises.
The check will consist on writing out files to visualize that the subdomains
and boundaries have been properly defined. For that purpose, Paraview is
recommended.
"""
Electrostatics.write_check_files()

# CREATE THE RESTRICTIONS.
# Generate the restrictions.
vacuum_rtc, liquid_rtc, domain_rtc, meniscus_rtc = \
    Electrostatics.generate_restrictions()

# Write out for simulation (.xml) and for visualization (.xdmf).
"""
As in the previous step, it is important to visually check the generated
restrictions before continuing with the code. Otherwise, any error related
with these restrictions will be hard to debug.
"""
Electrostatics.write_restrictions()

# DEFINE THE MIDPOINTS OF THE FACETS ON THE INTERFACE.
r_mids, z_mids = PostProcessing.get_midpoints_from_boundary(boundaries, boundaries_ids['Interface'])
coords_mids = [r_mids, z_mids]
r_nodes, z_nodes = PostProcessing.get_nodepoints_from_boundary(mesh,
                                                               boundaries,
                                                               boundaries_ids['Interface'])
coords_nodes = [r_nodes, z_nodes]

# SOLVE THE ELECTROSTATICS.
# Check all the options available for the implemented solver.
# Poisson.check_solver_options()

# Define the solver parameters.
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
snes_solver_parameters = {"snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 50,
                                          "report": True,
                                          "error_on_nonconvergence": True,
                                          'line_search': 'bt',
                                          'relative_tolerance': 1e-4}}
phi, sigma = Electrostatics.solve(solver_parameters=snes_solver_parameters)

# %% RETRIVE ALL THE IMPORTANT INFORMATION.
"""
At this step, from the potential obtained in previous steps we derive the
electric fields at each of the subdomains. Moreover, we will obtain their
normal components at the interface.
"""
E_v = Electrostatics.get_electric_field('vacuum')
E_l = Electrostatics.get_electric_field('liquid')

# Split the electric field into radial and axial components.
E_v_r, E_v_z = Poisson.split_field_components(E_v, coords_nodes)
E_l_r, E_l_z = Poisson.split_field_components(E_l, coords_nodes)

n_v = fn.FacetNormal(mesh)
n = n_v
n_l = fn.as_vector((-n_v[0], -n_v[1]))

E_v_n = Poisson.get_normal_field(n_v, E_v, mesh, boundaries, meniscus_rtc,
                                 boundaries_ids['Interface'], sign="-")
E_v_n_array = PostProcessing.extract_from_function(E_v_n, coords_mids)

E_l_n = Poisson.get_normal_field(n_l, E_l, mesh, boundaries, meniscus_rtc,
                                 boundaries_ids['Interface'], sign="+")
E_l_n_array = PostProcessing.extract_from_function(E_l_n, coords_mids)

sigma_arr = PostProcessing.extract_from_function(sigma, coords_mids)

# Compute the non dimensional evaporated charge and current.
j_ev = (sigma*T_h)/(eps_r*Chi) * fn.exp(-Phi/T_h * (
        1-pow(B, 1/4)*fn.sqrt(E_v_n)))
I_h = Poisson.get_nd_current(boundaries, boundaries_ids, j_ev, r0)

j_ev_arr = PostProcessing.extract_from_function(j_ev, coords_mids)

# %% DATA POSTPROCESSING.
# Check charge conservation.
cc_check = Electrostatics.check_charge_conservation(coords)

# Plot.
plotpy.lineplot([(r_nodes, E_v_r, r'Radial ($\hat{r}$)'),
                 (r_nodes, E_v_z, r'Axial ($\hat{z}$)')],
                xlabel=x_label, ylabel=y_label,
                fig_title='Components of the electric field at vacuum',
                legend_title='Field Components')

plotpy.lineplot([(r_nodes, E_l_r, r'Radial ($\hat{r}$)'),
                 (r_nodes, E_l_z, r'Axial ($\hat{z}$)')],
                xlabel=x_label, ylabel=y_label,
                fig_title='Components of the electric field at liquid',
                legend_title='Field Components')

plotpy.lineplot([(r_mids, E_v_n_array, 'Vacuum'),
                 (r_mids, E_l_n_array, 'Liquid')],
                xlabel=x_label, ylabel=y_label,
                fig_title='Normal components of the electric fields',
                legend_title='Subdomains')

plotpy.lineplot([(r_mids, sigma_arr)],
                xlabel=x_label, ylabel=r'$\hat{\sigma}$',
                fig_title='Radial evolution of the surface charge density')

plotpy.lineplot([(r_mids, cc_check)],
                xlabel=x_label, ylabel=r'$\hat{j}_n^e - \hat{j}_{cond}$',
                yscale='linear',
                fig_title='Kinetic evaporation law minus conduction')

# %% SOLVE STOKES EQUATION.
# Define the required non dimensional parameters.
k = k_0
E_c = np.sqrt((4*gamma)/(r0*vacuum_perm))
E_star = (4*np.pi*vacuum_perm*delta_G**2)/(1.60218e-19)**3
j_star = k*E_star/eps_r
u_star = j_star/(rho_0*q_m)
r_star = B*r0
We = (rho_0*u_star**2*r_star)/(2*gamma)
Ca = mu_0*u_star/(2*gamma)

# Define the boundary conditions.
boundary_conditions_fluids = {'Tube_Wall_R': {'Dirichlet':
                                                ['v', fn.Constant((0., 0.))]}
                              }
inputs_fluids = {'Weber number': We,
                 'Capillary number': Ca,
                 'Relative perm': eps_r,
                 'B': B,
                 'Lambda': Lambda,
                 'Non dimensional temperature': T_h,
                 'Sigma': sigma,
                 'Phi': Phi,
                 'Chi': Chi,
                 'Potential': phi,
                 'Contact line radius': r0}
Stokes = NS(inputs_fluids, boundary_conditions_fluids, subdomains=subdomains,
            boundaries=boundaries, mesh=mesh, boundaries_ids=boundaries_ids,
            restrictions_names=restrictions_names,
            restrictions_path=restrictions_folder_path,
            filename=ofilename_.strip(), mesh_path=mesh_folder_path)
u, p_star, theta = Stokes.solve()
p = p_star - P_r_h + I_h*C_R

# %% RETRIEVE ALL THE IMPORTANT INFORMATION.
u_r, u_z = NS.extract_velocity_components(u, coords)
p_arr = PostProcessing.extract_from_function(p, coords)
u_r, u_z, p_arr = PostProcessing.smooth_data(u_r, u_z, p_arr,
                                             window_length=801, polyorder=3)
check = NS.check_evaporation_condition(mesh, meniscus_rtc, boundaries, u,
                                       j_n_e_h, coords,
                                       boundary_id=boundaries_ids['Interface'])
plotpy.lineplot([(r_mids, u_r, r'Radial ($\hat{r}$)'),
                 (r_mids, u_z, r'Axial ($\hat{z}$)')],
                xlabel=r'$\hat{r}$', ylabel=r'$\hat{u}$',
                fig_title='Components of the velocity field',
                legend_title='Field Components')

plotpy.lineplot([(r_mids, p_arr)],
                xlabel=x_label, ylabel=r'$\hat{p}$',
                fig_title='Pressure along the meniscus')

plotpy.lineplot([(r_mids, check)],
                xlabel=x_label, ylabel=r'$\hat{u}\cdot n - \hat{j}_n^e$',
                fig_title='Check of the charge evaporation.')
