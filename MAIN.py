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
from Solvers.NS_Solver import Stokes

import fenics as fn
import numpy as np
import sympy as sp

from Input_Parameters import Liquid_Properties
from Tools.PlotPy import PlotPy
from Tools.PostProcessing import PostProcessing
from Tools.MeshConverter import msh2xml
from Tools.GMSH_Interface import GMSHInterface
from Tools.BVPInterface import BVPInterface

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

# Define values for simulation (as used in Ximo's thesis).
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

# Get the name of the mesh file and its path.
filepath = app.msh_filename
filename = filepath.split('/')[-1]

# Create the folders in which the outputs will be stored.
mesh_folder_path = '/'.join(filepath.split('/')[:-1])

root_folder = os.getcwd()  # Get the working directory (getcwd = Get Current Working Directory).
restrictions_folder_name = 'RESTRICTIONS'  # Name of the folder where all the defined restrictions will be saved.
restrictions_folder_path = root_folder + '/' + restrictions_folder_name

checks_folder_name = 'CHECKS'  # Name of the folder where all necessary checks will be stored.
checks_folder_path = root_folder + '/' + checks_folder_name

# Call the dolfin-converter if necessary.
if filename.split('.')[-1] != 'xml':  # If the user does not load a DOLFIN-ready file, we need to make conversions.
    meshname = msh2xml(filename, root_folder, mesh_folder_path)
    filepath = mesh_folder_path + '/' + meshname

# %% ELECTROSTATICS.
# Define the values to be used for the simulation.
T_h = 1.  # Non-dimensional temp.
Lambda = Lambda_list[0]
B = B_list[0]
E0 = E0_list[0]

# Define some non-dimensional parameters.
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

top_potential = -E0*z0/r0  # Define the potential at the Top Wall.
ref_potential = 0  # Define ground BC.
boundary_conditions_elec = {'Top_Wall': {'Dirichlet': [top_potential, 'vacuum']},
                            'Inlet': {'Dirichlet': [ref_potential, 'liquid']},
                            'Tube_Wall_R': {'Dirichlet': [ref_potential, 'liquid']},
                            'Bottom_Wall': {'Dirichlet': [ref_potential, 'vacuum']},
                            'Lateral_Wall_R': {'Neumann': 'vacuum'},
                            'Lateral_Wall_L': {'Neumann': 'vacuum'},
                            'Tube_Wall_L': {'Neumann': 'vacuum'}}

# Define the boundary conditions for the initial problem, which is required for the iterative solver (as an init. guess)
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
""" Restrictions are the objects (or files) used by the multiphenics library to recognize a specific subdomain/boundary.
These are useful to define functions on a specific subdomain or boundary, which is the case of the Lagrangian
multiplier, which for this problem should be defined only at the interface boundary.
"""
# Generate the restrictions.
restrictions_dict = Electrostatics.generate_restrictions()

# Write out for simulation (.xml) and for visualization (.xdmf).
"""
As in the previous step, it is important to visually check the generated
restrictions before continuing with the code. Otherwise, any error related
with these restrictions will be hard to debug.
"""
Electrostatics.write_restrictions()

# Get the midpoints and node points that define the interface boundary.
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

# Define an auxiliary term for the computations.
K = 1+Lambda*(T_h - 1)

E_l_n = (Electrostatics.E_v_n-Electrostatics.sigma)/LiquidInps.eps_r
E_l_n_array = PostProcessing.extract_from_function(E_l_n, coords_mids)
sigma_arr = PostProcessing.extract_from_function(Electrostatics.sigma, coords_mids)

# Get components of the liquid field.
E_l_r, E_l_z = Poisson.get_liquid_electric_field(mesh=mesh, subdomain_data=boundaries,
                                                 boundary_id=boundaries_ids['Interface'], normal_liquid=E_l_n_array,
                                                 tangential_liquid=E_t_array)
E_l_r.append(E_l_r[-1])
E_l_z.append(E_l_z[-1])

# Calculate the non-dimensional evaporated charge and current.
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
E_star = (4*np.pi*LiquidInps.vacuum_perm*LiquidInps.Solvation_energy**2)/1.60218e-19**3
j_star = k*E_star/LiquidInps.eps_r
u_star = j_star/(LiquidInps.rho_0*LiquidInps.q_m)  # Characteristic velocity.
r_star = B*r0  # Cone tip radius.
We = (LiquidInps.rho_0*u_star**2*r_star)/(2*LiquidInps.gamma)  # Weber number.
Ca = LiquidInps.mu_0*u_star/(2*LiquidInps.gamma)  # Capillary number.
Kc = (LiquidInps.vacuum_perm * LiquidInps.eps_r * u_star) / (LiquidInps.k_0*r_star)

# Define the boundary conditions.
boundary_conditions_fluids = {'Tube_Wall_R': {'Dirichlet':
                                                ['v', fn.Constant((0., 0.))]}
                              }

# Define the inputs required by the Stokes_sim solver.
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
                 'Kc': Kc,
                 'Vacuum electric field': Electrostatics.E_v,
                 'Vacuum normal component': Electrostatics.E_v_n}

# Initialize the Stokes_sim class and solve.
Stokes_sim = Stokes(inputs_fluids, boundary_conditions_fluids, subdomains=subdomains, boundaries=boundaries, mesh=mesh,
                    boundaries_ids=boundaries_ids, restrictions_path=restrictions_folder_path, mesh_path=mesh_folder_path,
                    filename=filename)
Stokes_sim.solve()

# Obtain useful information from the solution.
p = Stokes_sim.p_star - P_r_h + I_h * C_R
theta_fun = Stokes.block_project(Stokes_sim.theta, mesh, Electrostatics.restrictions_dict['interface_rtc'], boundaries,
                                 boundaries_ids['Interface'], space_type='scalar', boundary_type='internal')
theta_fun_arr = PostProcessing.extract_from_function(theta_fun, coords_mids)

# %% RETRIEVE ALL THE IMPORTANT INFORMATION.
u_r, u_z = Stokes.extract_velocity_components(Stokes_sim.u, coords_nodes)
p_arr = PostProcessing.extract_from_function(p, coords_nodes)
j_conv_arr = PostProcessing.extract_from_function(Stokes_sim.j_conv, coords_nodes)

plotpy.lineplot([(r_nodes, u_r, r'Radial ($\hat{r}$)'),
                 (r_nodes, u_z, r'Axial ($\hat{z}$)')],
                xlabel=r'$\hat{r}$', ylabel=r'$\hat{u}$',
                fig_title='Components of the velocity field',
                legend_title='Field Components')

plotpy.lineplot([(r_nodes, p_arr)],
                xlabel=x_label, ylabel=r'$\hat{p}$',
                fig_title='Pressure along the meniscus')

# Check the normal component of the velocity and the evaporated charge.
check = Stokes_sim.check_evaporation_condition(Stokes_sim.u_n, j_ev, coords_mids)
u_n_array = PostProcessing.extract_from_function(Stokes_sim.u_n, coords_mids)
u_t_array = PostProcessing.extract_from_function(Stokes_sim.u_t, coords_mids)
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

# %% SURFACE UPDATE.


def get_derivatives(independent_param, fun):
    """
    Get the derivatives of a given function. To do, the Sympy library is used. Sympy is quite useful for symbolic
    mathematical operations, and derivatives of a given function (using their own functions) is as easy as calling a
    simple function.

    Computation of the derivatives have been checked using WolframAlpha.
    Args:
        independent_param: string. The independent parameter with respect which the derivatives will be computed.
        fun: string. The function to be derived. This must contain only a single independent parameter and any angle
        should be introduced in radians.

    Returns:
        Lambda functions of the first and second derivatives, only as a function of the independent parameter.

    """
    # Define a Sympy symbol.
    """ Defining a Sympy symbol will be useful to tell the function with respect to which parameter should the function
    compute the derivative.
    """
    sym_ind_param = sp.Symbol(independent_param)

    # Transform the string with the function into a Sympy expression.
    """ When doing this transformation, Sympy will automatically recognize the independent variable as a symbol, in case
    there is only a single parameter. If more than one parameter is in the string, it will recognize several symbols.
    For the latter case, the parse_expr function is recommended. For help on this function, type help(sp.parse_expr).
    """
    sym_exp = sp.sympify(fun)

    # Compute the first and second derivatives of the introduced function.
    fprime = sym_exp.diff(sym_ind_param)
    fprime2 = fprime.diff(sym_ind_param)

    # Substitute possible conflicts between used numerical parser and sympy symbology.
    fprime = fprime.subs({'PI': sp.pi})
    fprime2 = fprime2.subs({'PI': sp.pi})

    # Transform the derivatives into evaluable functions.
    fprimeLambdified = sp.lambdify(sym_ind_param, fprime, 'numpy')
    fprime2Lambdified = sp.lambdify(sym_ind_param, fprime2, 'numpy')

    return fprimeLambdified, fprime2Lambdified


# Introduce the surface parametrization.
if app.geom_data.angle_unit == 'degrees':  # Need to transform degrees into radians.
    if app.geom_data.z_of_r.get() == '':
        fun_surf = GMSHInterface.angle_handler(app.geom_data.z_fun.get())
        r_param = app.geom_data.r_fun.get()
        r_param = sp.sympify(r_param)
        ind_data = app.geom_data.base_data
    else:
        fun_surf = GMSHInterface.angle_handler(app.geom_data.z_of_r.get())
        r_data = app.geom_data.base_data
else:
    if app.geom_data.z_of_r.get() == '':
        fun_surf = app.geom_data.z_fun.get()
        r_param = app.geom_data.r_fun.get()
        r_param = sp.sympify(r_param)
        ind_data = app.geom_data.base_data
    else:
        fun_surf = app.geom_data.z_of_r.get()
        r_data = app.geom_data.base_data

# Get independent variable from equation.
ind_var = GMSHInterface.get_independent_var_from_equation(fun_surf)

# Obtain the derivatives with respect to the independent variable.
zprimeLambdified, zprime2Lambdified = get_derivatives(ind_var, fun_surf)

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
    sym_eq = sp.sympify(app.geom_data.r_fun.get())
    ind_data = np.array([])
    s = sp.Symbol('s')
    for r in r_nodes:
        ind_data = np.append(ind_data, sp.solvers.solve(sym_eq - r, s)[0])
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
Q = fn.FunctionSpace(Electrostatics.mesh, 'DG', 0)
ux = fn.project(Stokes_sim.u.sub(0).dx(0), Q)
uz = fn.project(Stokes_sim.u.sub(1).dx(1), Q)
counter = 0
residuals = np.array([])
for r_coord, z_coord in zip(r_nodes, z_nodes):
    a_diff = E_v_r[counter] ** 2 - E_v_z[counter] ** 2 - \
             Stokes_sim.eps_r * (E_l_r[counter] ** 2 - E_l_z[counter] ** 2) + \
             Stokes_sim.p_star([r_coord, z_coord]) - I_h * C_R - \
             ((Stokes_sim.eps_r * Ca * np.sqrt(B)) / (1 + Lambda * (T_h - 1))) * (2 * ux([r_coord, z_coord]))
    b_diff = 2 * E_v_r[counter] * E_v_z[counter] - \
             2 * Stokes_sim.eps_r * E_l_r[counter] * E_l_z[counter] - \
             ((Stokes_sim.eps_r * Ca * np.sqrt(B)) / (1 + Lambda * (T_h - 1))) * (ux([r_coord, z_coord]) +
                                                                                  uz([r_coord, z_coord]))
    c_diff = E_v_z[counter] ** 2 - E_v_r[counter] ** 2 - \
             Stokes_sim.eps_r * (E_l_z[counter] ** 2 - E_l_r[counter] ** 2) + \
             Stokes_sim.p_star([r_coord, z_coord]) - I_h * C_R - \
             ((Stokes_sim.eps_r * Ca * np.sqrt(B)) / (1 + Lambda * (T_h - 1))) * (2 * uz([r_coord, z_coord]))

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

# Initialize the solver object.
solver = BVPInterface()

# Define the system to be solved.
funs = ['y']
syst = ['y.diff(x, 1)', '2*tau*(1+y.diff(x, 1)**2)**(3/2) - (1/(x+1e-20))*(1+y.diff(x, 1)**2)*y.diff(x, 1)']
solver.define_system(system=syst, functions=funs, vars_dict={'tau': tau_s_next})

# Define the mesh for the solver of the boundary value problem (bvp).
mesh_bvp = np.linspace(0, 1, tau_s_next.size)
solver.load_mesh(mesh_bvp)

# Define the boundary conditions and load them into the solver.
bcs = {'a': {'y.diff(x, 1)': 0}, 'b': {'y': 0}}
solver.load_bcs(bcs)

# Define an initial guess and load it into the solver.
"""
The structure of the initial guess is similar to the one used by the solver, that is:
y_init = [y0, y'0]
"""
init_guess = np.zeros((2, mesh_bvp.size))
init_guess[0, 0] = 0.8
init_guess[0, 1] = 0
solver.load_initial_guess(init_guess)

# Run the solver.
sol = solver.solve()

# Plot the solution.
r_plot = np.linspace(0, 1, 200)
y_plot = sol.sol(r_plot)[0]
plotpy.lineplot([(r_plot, y_plot, 'First iteration'), (r_nodes, z_nodes, 'Initial shape')],
                xlabel=x_label, y_label=r'$\hat{z}$',
                fig_title='Evolution of the meniscus surface after first iteration.',
                legend_title='Surfaces')

# %% WHAT CONVERGES.
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

# %% TESTS.
# Test how to introduce the scipy.sol function into the geometry builder.
# Create a test array of values of r.
""" One could think to simply introduce the nodes created by the bvp solver into the geometry generator. While this
could work in some cases, it has been proven that introducing these nodes (from sol.x) leads to errors in GMSH because
of the small distances between nodes. Thus, it is recommended to create equally spaced nodes with a linspace method. One
can use the length of the sol.x array for a reference on the number of points to use.
"""
r_test = np.linspace(0, 1, len(sol.x))

# Reset required parameters to avoid the class initialization.
app.geom_data.geo_gen.reset_geom_params()

# Generate the geometry and export into .geo file.
app.geom_data.geo_gen.geometry_generator(interface_fun=sol.sol, r=r_test)

# Export the geometry and create the mesh file.
msh_path = app.geom_data.geo_gen.mesh_generation_noGUI(mesh_folder_path+'/Prueba2.geo')
