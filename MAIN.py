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
from Solvers.NS_Solver import NavierStokes as NS

import fenics as fn
import numpy as np
import sympy as sym
import bisect

from Input_Parameters import Liquid_Properties
from Tools.PlotPy import PlotPy
from Tools.PostProcessing import PostProcessing
from Tools.MeshConverter import msh2xml
from Tools.GMSH_Interface import GMSHInterface

from MainMenu import run_main_menu



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
LiquidInps = Liquid_Properties(relative_permittivity=10)
LiquidInps.EMIBF4()  # Load EMIBF4 liquid properties.

T_0 = 298.15  # Reference temperature [K]

# Define values for simulation.
Lambda_list = [1, 12]
E0_list = [0.99, 0.65]
B_list = [0.0467, 0.00934]
C_R = 1e3
P_r_h = 0  # Non dimensional pressure at reservoir.

# %% IMPORT THE MESH.
"""
From version of GMSH 3.0 upwards, the format in which the mesh is saved has
changed, and it is not compatible with the format that dolfin-converter is
expecting to receive. Therefore, the GMSH 2.0 format is required. In order to
do so, when the mesh has already been defined in GMSH, go to:
File > Export > name_of_file.msh and save the file as ASCII v2. In that way,
dolfin-converter will be able to transform the .msh file into .xml, the
extension that is compatible with FEniCS. This is automatically done in the
code.
"""

# Call the main menu (main GUI).
app = run_main_menu()

# Get the name of the mesh file.
filepath = app.msh_filename
filename = filepath.split('/')[-1]

# Create the folders in which the outputs will be stored.
mesh_folder_path = '/'.join(filepath.split('/')[:-1])

root_folder = os.getcwd()
restrictions_folder_name = 'RESTRICTIONS'
restrictions_folder_path = root_folder + '/' + restrictions_folder_name

checks_folder_name = 'CHECKS'
checks_folder_path = root_folder + '/' + checks_folder_name

# Call the dolfin-converter if necessary.
if filename.split('.')[-1] != 'xml':
    meshname = msh2xml(filename, root_folder, mesh_folder_path)
    filepath = mesh_folder_path + '/' + meshname

# %% ELECTROSTATICS.
# Define the values to be used for the simulation.
T_h = 1.
Lambda = Lambda_list[0]
B = B_list[0]
E0 = E0_list[0]
Chi = (LiquidInps.h*LiquidInps.k_prime)/(Lambda*LiquidInps.k_B*LiquidInps.vacuum_perm*LiquidInps.eps_r)
Phi = LiquidInps.Solvation_energy/(LiquidInps.k_B*T_0*T_h)

r_star = ((1.602176565e-19)**6*LiquidInps.gamma)/(4*np.pi**2*LiquidInps.vacuum_perm**3*LiquidInps.Solvation_energy**4)
r0 = r_star / B
z0 = 10*r0

# Define the constant inputs for the solver.
inputs = {'Relative_perm': LiquidInps.eps_r,
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

top_potential = -E0*z0/r0
ref_potential = 0
boundary_conditions_elec = {'Top_Wall': {'Dirichlet': [top_potential, 'vacuum']},
                            'Inlet': {'Dirichlet': [ref_potential, 'liquid']},
                            'Tube_Wall_R': {'Dirichlet': [ref_potential, 'liquid']},
                            'Bottom_Wall': {'Dirichlet': [ref_potential, 'vacuum']},
                            'Lateral_Wall_R': {'Neumann': 'vacuum'},
                            'Lateral_Wall_L': {'Neumann': 'vacuum'},
                            'Tube_Wall_L': {'Neumann': 'vacuum'}}

bcs_elec_init = {'Top_Wall': {'Dirichlet': [top_potential, 'vacuum']},
                 'Interface': {'Dirichlet': [ref_potential, 'vacuum']},
                 'Bottom_Wall': {'Dirichlet': [ref_potential, 'vacuum']},
                 }


# Initialize the Electrostatics class, loading constant parameters.
Electrostatics = Poisson(inputs, boundary_conditions_elec, filepath, restrictions_folder_path, checks_folder_path,
                         boundary_conditions_init=bcs_elec_init, liquid_inps=LiquidInps)

# Get the mesh object.
mesh = Electrostatics.get_mesh()

# Generate a Mesh Function containing all subdomains and boundaries.
interface_name = '"Meniscus"'
boundaries, boundaries_ids = Electrostatics.get_boundaries()
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
restrictions_dict = Electrostatics.generate_restrictions()

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
r_nodes, z_nodes = PostProcessing.get_nodepoints_from_boundary(mesh, boundaries, boundaries_ids['Interface'])
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
                                          "maximum_iterations": 200,
                                          "report": True,
                                          "error_on_nonconvergence": True,
                                          'line_search': 'bt',
                                          'relative_tolerance': 1e-2}}
Electrostatics.solve(solver_parameters=snes_solver_parameters)

# %% RETRIEVE ALL THE IMPORTANT INFORMATION.
"""
At this step, from the potential obtained in previous steps we derive the
electric fields at each of the subdomains. Moreover, we will obtain their
normal components at the interface.
"""

# Split the electric field into radial and axial components.
E_v_r, E_v_z = Poisson.split_field_components(Electrostatics.E_v, coords_nodes)

E_v_n_array = PostProcessing.extract_from_function(Electrostatics.E_v_n, coords_mids)
E_t_array = PostProcessing.extract_from_function(Electrostatics.E_t, coords_mids)

# Compute the non dimensional evaporated charge and current.
K = 1+Lambda*(T_h - 1)

E_l_n = (Electrostatics.E_v_n-Electrostatics.sigma)/LiquidInps.eps_r
E_l_n_array = PostProcessing.extract_from_function(E_l_n, coords_mids)
sigma_arr = PostProcessing.extract_from_function(Electrostatics.sigma, coords_mids)

# Get components of the liquid field.
E_l_r, E_l_z = Poisson.get_liquid_electric_field(mesh=mesh, subdomain_data=boundaries,
                                                 boundary_id=boundaries_ids['Interface'], normal_liquid=E_l_n_array,
                                                 tangential_liquid=E_t_array)

j_ev = (Electrostatics.sigma*T_h)/(LiquidInps.eps_r*Chi) * fn.exp(-Phi/T_h * (
        1-pow(B, 1/4)*fn.sqrt(Electrostatics.E_v_n)))
j_cond = K*E_l_n

I_h = Electrostatics.get_nd_current(j_ev)

j_ev_arr = PostProcessing.extract_from_function(j_ev, coords_mids)
j_cond_arr = PostProcessing.extract_from_function(j_cond, coords_mids)

# Compute the normal component of the electric stress at the meniscus (electric pressure).
n_taue_n = (Electrostatics.E_v_n**2-LiquidInps.eps_r*E_l_n**2) + (LiquidInps.eps_r-1)*Electrostatics.E_t**2
n_taue_n_arr = PostProcessing.extract_from_function(n_taue_n, coords_mids)

# %% DATA POSTPROCESSING.
# Check charge conservation.
charge_check = abs(j_ev_arr-j_cond_arr)/j_ev_arr
print(f'Maximum relative difference between evaporated and conducted charge is {max(charge_check)}')

# Plot.
plotpy.lineplot([(r_nodes, E_v_r, r'Radial ($\hat{r}$)'),
                 (r_nodes, E_v_z, r'Axial ($\hat{z}$)')],
                xlabel=x_label, ylabel=y_label,
                fig_title='Components of the electric field at vacuum',
                legend_title='Field Components')

plotpy.lineplot([(r_mids, E_l_r, r'Radial ($\hat{r}$)'),
                 (r_mids, E_l_z, r'Axial ($\hat{z}$)')],
                xlabel=x_label, ylabel=y_label,
                fig_title='Components of the electric field at liquid',
                legend_title='Field Components')

plotpy.lineplot([(r_mids, E_t_array)],
                xlabel=x_label, ylabel=r'$\hat{E}_t$',
                fig_title='Tangential component of the electric field at the meniscus')

plotpy.lineplot([(r_mids, E_v_n_array, 'Vacuum'),
                 (r_mids, E_l_n_array, 'Liquid')],
                xlabel=x_label, ylabel=y_label,
                fig_title='Normal components of the electric fields',
                legend_title='Subdomains')

plotpy.lineplot([(r_mids, sigma_arr)],
                xlabel=x_label, ylabel=r'$\hat{\sigma}$',
                fig_title='Radial evolution of the surface charge density')

plotpy.lineplot([(r_mids, j_ev_arr)],
                xlabel=x_label, ylabel=r'$\hat{j}_e$',
                fig_title='Charge evaporation along the meniscus')

plotpy.lineplot([(r_mids, n_taue_n_arr)],
                xlabel=x_label, ylabel=r'$\mathbf{n}\cdot\hat{\bar{\bar{\tau}}}_e \cdot \mathbf{n}$',
                fig_title='Normal component of the electric stress at the meniscus')

# %% SOLVE STOKES EQUATION.
# Define the required non dimensional parameters.
k = LiquidInps.k_0
E_c = np.sqrt((4*LiquidInps.gamma)/(r0*LiquidInps.vacuum_perm))
E_star = (4*np.pi*LiquidInps.vacuum_perm*LiquidInps.Solvation_energy**2)/(1.60218e-19)**3
j_star = k*E_star/LiquidInps.eps_r
u_star = j_star/(LiquidInps.rho_0*LiquidInps.q_m)
r_star = B*r0
We = (LiquidInps.rho_0*u_star**2*r_star)/(2*LiquidInps.gamma)
Ca = LiquidInps.mu_0*u_star/(2*LiquidInps.gamma)

# Define the boundary conditions.
boundary_conditions_fluids = {'Tube_Wall_R': {'Dirichlet':
                                                ['v', fn.Constant((0., 0.))]}
                              }
inputs_fluids = {'Weber number': We,
                 'Capillary number': Ca,
                 'Relative perm': LiquidInps.eps_r,
                 'B': B,
                 'Lambda': Lambda,
                 'Non dimensional temperature': T_h,
                 'Sigma': Electrostatics.sigma,
                 'Phi': Phi,
                 'Chi': Chi,
                 'Potential': Electrostatics.phi,
                 'Contact line radius': r0,
                 'Vacuum electric field': Electrostatics.E_v,
                 'Vacuum normal component': Electrostatics.E_v_n}

Stokes = NS(inputs_fluids, boundary_conditions_fluids, subdomains=subdomains, boundaries=boundaries, mesh=mesh,
            boundaries_ids=boundaries_ids, restrictions_path=restrictions_folder_path, mesh_path=mesh_folder_path,
            filename=filename)
Stokes.solve()
p = Stokes.p_star - P_r_h + I_h*C_R
theta_fun = NS.block_project(Stokes.theta, mesh, Electrostatics.restrictions_dict['interface_rtc'], boundaries,
                             boundaries_ids['Interface'], space_type='scalar', boundary_type='internal')
theta_fun_arr = PostProcessing.extract_from_function(theta_fun, coords_mids)

# %% RETRIEVE ALL THE IMPORTANT INFORMATION.
u_r, u_z = NS.extract_velocity_components(Stokes.u, coords_nodes)
p_arr = PostProcessing.extract_from_function(p, coords_nodes)
j_conv_arr = PostProcessing.extract_from_function(Stokes.j_conv, coords_nodes)

plotpy.lineplot([(r_nodes, u_r, r'Radial ($\hat{r}$)'),
                 (r_nodes, u_z, r'Axial ($\hat{z}$)')],
                xlabel=r'$\hat{r}$', ylabel=r'$\hat{u}$',
                fig_title='Components of the velocity field',
                legend_title='Field Components')

plotpy.lineplot([(r_nodes, p_arr)],
                xlabel=x_label, ylabel=r'$\hat{p}$',
                fig_title='Pressure along the meniscus')

# Check the normal component of the velocity and the evaporated charge.
check = NS.check_evaporation_condition(Stokes.u_n, j_ev, coords_mids)
u_n_array = PostProcessing.extract_from_function(Stokes.u_n, coords_mids)
u_t_array = PostProcessing.extract_from_function(Stokes.u_t, coords_mids)
plotpy.lineplot([(r_mids, check)],
                xlabel=x_label, ylabel=r'$\hat{u}\cdot n - \hat{j}_n^e$',
                fig_title='Check of the charge evaporation.')

plotpy.lineplot([(r_mids, u_n_array)],
                xlabel=x_label, ylabel=r'$\hat{u}_n$',
                fig_title='Normal Component of the velocity field.')

plotpy.lineplot([(r_mids, u_t_array)],
                xlabel=x_label, ylabel=r'$\hat{u}_t$',
                fig_title='Tangential Component of the velocity field.')

plotpy.lineplot([(r_nodes, j_conv_arr)],
                xlabel=x_label, ylabel=r'$\hat{j}_{conv}$',
                fig_title='Convection charge transport')

plotpy.lineplot([(r_mids, theta_fun_arr)],
                xlabel=x_label, ylabel=r'$\mathbf{n}\cdot\hat{\bar{\bar{\tau}}}_m \cdot \mathbf{n}$',
                fig_title='Normal component of the hydraulic stress at the meniscus')

# %% CHECK CONVECTION CHARGE TRANSPORT.
n = fn.FacetNormal(mesh)
special = (fn.Identity(mesh.topology().dim()) - fn.outer(n, n))*fn.grad(Stokes.sigma)
j_conv = fn.dot(Electrostatics.sigma*n, fn.dot(fn.grad(Stokes.u), n)) - fn.dot(Stokes.u, special)
j_conv = NS.block_project(j_conv, mesh, Electrostatics.restrictions_dict['interface_rtc'], Stokes.boundaries,
                          Stokes.boundaries_ids['Interface'], space_type='scalar', boundary_type='internal', sign='-')
j_conv_arr = PostProcessing.extract_from_function(j_conv, coords_mids)

plotpy.lineplot([(r_mids, j_conv_arr)],
                xlabel=x_label, ylabel=r'$\hat{j}_{conv}$',
                fig_title='Convection charge transport')

# %% SURFACE UPDATE.


def get_derivatives(independent_param, fun):
    """
    Get the derivatives of a given function. Computation of the derivatives have been checked using WolframAlpha.
    Args:
        independent_param: string. The independent parameter with respect which the derivatives will be computed.
        fun: string. The function to be derived. This must contain only a single independent parameter and any angle
        should be introduced in radians.

    Returns:
        Lambda functions of the first and second derivatives, only as a function of the independent parameter.

    """
    sym_ind_param = sym.symbols(independent_param)
    sym_exp = sym.sympify(fun)
    fprime = sym_exp.diff(sym_ind_param)
    fprime2 = fprime.diff(sym_ind_param)
    fprimeLambdified = sym.lambdify(sym_ind_param, fprime, 'numpy')
    fprime2Lambdified = sym.lambdify(sym_ind_param, fprime2, 'numpy')

    return fprimeLambdified, fprime2Lambdified


# Introduce the surface parametrization.
try:
    if app.geom_data.angle_unit == 'degrees':
        z_aux = GMSHInterface.angle_handler(app.geom_data.z_fun.get())
    else:
        z_aux = app.geom_data.z_fun.get()
    ind_var = GMSHInterface.get_independent_var_from_equation(z_aux)
    ind_var_sym = sym.symbols(ind_var)
    z_param = sym.sympify(z_aux)
    zprimeLambdified, zprime2Lambdified = get_derivatives('s', z_param)
    r_param = app.geom_data.r_fun.get()
    r_param = sym.sympify(r_param)
    ind_data = app.geom_data.base_data
except AttributeError:
    if app.geom_data.angle_unit == 'degrees':
        z_aux = GMSHInterface.angle_handler(app.geom_data.z_of_r.get())
    else:
        z_aux = app.geom_data.z_of_r.get()
    z_param = sym.sympify(z_aux)
    zprimeLambdified, zprime2Lambdified = get_derivatives('r', z_param)
    r_data = app.geom_data.base_data

# Compute auxiliary terms.
n_k = np.array([])
del_dot_n = np.array([])
try:  # Case when a z(r) function is defined.
    for num in r_data:
        n_k = np.append(n_k, (1/np.sqrt(1+zprimeLambdified(num)**2))*np.array([-zprimeLambdified(num), 1]))
        del_dot_n = np.append(del_dot_n, ((1+zprimeLambdified(num)**2)*zprimeLambdified(num) + num*zprime2Lambdified(num))/(num*(1+zprimeLambdified(num)**2)**(3/2)))
    del_dot_n = del_dot_n[::-1]
    n_k = n_k.reshape((len(r_data), 2))
    n_k = n_k[::-1]
except NameError:  # when independent functions for r and z were defined.
    sym_eq = sym.sympify(app.geom_data.r_fun.get())
    ind_data = np.array([])
    s = sym.Symbol('s')
    for r in r_nodes:
        ind_data = np.append(ind_data, sym.solvers.solve(sym_eq - r, s)[0])
    for num in ind_data:
        n_k = np.append(n_k, (1/(1+zprimeLambdified(num)**2)**0.5)*np.array([-zprimeLambdified(num), 1]))
        del_dot_n = np.append(del_dot_n, ((1+zprimeLambdified(num)**2)*zprimeLambdified(num) + \
                                          r_param.evalf(subs={ind_var: num})*zprime2Lambdified(num))/(
                r_param.evalf(subs={ind_var: num})*(1+zprimeLambdified(num)**2)**(3/2)))
    del_dot_n = del_dot_n[::-1]
    n_k = n_k.reshape((len(ind_data), 2))
    n_k = n_k[::-1]

# Compute the residuals.
"""
Note: Get coordinates from the nodes to evaluate the fields.
"""
Q = fn.FunctionSpace(mesh, 'DG', 0)
ux = fn.project(Stokes.u.sub(0).dx(0), Q)
uz = fn.project(Stokes.u.sub(1).dx(1), Q)
counter = 0
residuals = np.array([])
for r_coord, z_coord in zip(r_nodes, z_nodes):
    a_diff = E_v_r[counter]**2 - E_v_z[counter]**2 - \
             Stokes.eps_r*(E_l_r[counter]**2 - E_l_z[counter]**2) + \
             Stokes.p_star([r_coord, z_coord]) - I_h*C_R - \
             ((Stokes.eps_r*Ca*np.sqrt(B))/(1+Lambda*(T_h-1)))*(2*ux([r_coord, z_coord]))
    b_diff = 2*E_v_r[counter]*E_v_z[counter] - \
             2*Stokes.eps_r*E_l_r[counter]*E_l_z[counter] - \
             ((Stokes.eps_r*Ca*np.sqrt(B))/(1+Lambda*(T_h-1)))*(ux([r_coord, z_coord]) +
                                                                uz([r_coord, z_coord]))
    c_diff = E_v_z[counter]**2 - E_v_r[counter]**2 - \
             Stokes.eps_r*(E_l_z[counter]**2 - E_l_r[counter]**2) + \
             Stokes.p_star([r_coord, z_coord]) - I_h*C_R - \
             ((Stokes.eps_r*Ca*np.sqrt(B))/(1+Lambda*(T_h-1)))*(2*uz([r_coord, z_coord]))

    # Build the difference tensor.
    diff_tensor = np.array([[a_diff, b_diff],
                            [b_diff, c_diff]])

    # Compute the residual.
    residual = np.dot(np.dot(diff_tensor, n_k[counter, :]), n_k[counter, :]) - 0.5*del_dot_n[counter]
    residuals = np.append(residuals, residual)
    counter += 1
# Compute tau_s for the next iteration.
beta = 0.05
tau_s_next = np.array([])
for loc in np.arange(0, len(residuals)-1):
    tau_s_next = np.append(tau_s_next, 0.5*del_dot_n[loc] + beta*residuals[loc])

# Use equation 3.72 from Ximo's thesis to get the new positions of the nodes.
r = sym.symbols('r')
y = sym.symbols('y', cls=sym.Function)
diffeq = sym.Eq(r*(1+y(r).diff(r)**2)**(3/2)*tau_s_next - 0.5*(1+y(r).diff(r)**2)*y(r).diff(r) - 0.5*r*y(r).diff(r, r),
                0)
