"""
Code for Daniel's request.
Data:
    - Height: 300um
    - Apex radius: 1um
    - Separation to Top Wall: 250um
    - Top Wall Width: 50um
    - Tube Radius: 200um
"""

# Do the necessary imports.

import numpy as np
import os
import fenics as fn
import multiphenics as mp

from Solvers.Poisson_solver import Poisson
from Tools.GMSH_Interface_Daniel import GMSHInterface
from Tools.PlotGeometry import plot_geo
from Tools.MeshConverter import msh2xml
from Tools.PostProcessing import PostProcessing
from Tools.PlotPy import PlotPy
from Tools.generate_restrictions import Restrictions

# Initialize the plotting class.
grid_properties = {'linestyle': "--", 'color': 'black', 'linewidth': 0.2}
font_properties = {'family': 'serif', 'style': 'Computer Modern Roman',
                   'fig_title_style': 'bold',
                   'legend_title_style': 'bold',
                   'legend_labels_style': 'slanted',
                   'axis_labels_style': 'slanted'}
background_style = 'seaborn-paper'
x_label = r'$r [m]$'
y_label = r'$E [N/C]$'

# Initialize the plot class with the desired parameters.
plotpy = PlotPy(latex=True, fontsize=12., figsize=(12., 7.),
                background_style=background_style,
                font_properties=font_properties,
                grid_properties=grid_properties, save_images=False,
                save_mat=False, extension='jpg')


# %% GEOMETRY GENERATION.

def geo_wrapper(r_array, z_array, separation=250, report=True):
	geo_wrap = GMSHInterface()
	geo_wrap.geometry_generator(r_array, z_array, separation=separation)
	geo_wrap.mesh_generation()
	if report:
		plot_geo(geo_wrap.geo_filename)
	return geo_wrap


# Introduce the data.
r_apex = 1  # Radius of the Taylor cone at apex [um]
r_fluid = 200  # Radius of the fluid channel [um]
px = r_fluid
height = 300  # Height of the cone [um]
separation = 250  # Separation between cone tip and top wall [um]
B = r_apex / r_fluid
w = 1 / B * np.tan(np.radians(49.3))


def r_expr(s):
	return ((1 - 2 * s) * px) / (1 - 2 * s * (1 - s) * (1 - w))


def z_expr(s):
	return (2 * (1 - s) * s * w * (1 / np.tan(np.radians(49.3))) * px) / (1 - 2 * s * (1 - s) * (1 - w))


s_arr = np.linspace(0, 0.5, 200)
r_arr = np.array([r_expr(s) for s in s_arr])
z_arr = np.array([z_expr(s) for s in s_arr])
z_arr *= height / z_arr[-1] * 1e-6
r_arr *= 1e-6

class_call = geo_wrapper(r_arr, z_arr, separation=separation, report=False)

# %% MESH LOADING PROCESS.
class_call.msh_filename = class_call.geo_filename.split('/')[-1].split('.')[0] + '.msh'
root_folder = os.getcwd()
mesh_folder_path = root_folder
meshname = msh2xml(class_call.msh_filename.split('/')[-1], root_folder, mesh_folder_path)
filepath = mesh_folder_path + '/' + meshname

# %% ELECTRIC FIELD COMPUTATION.
# Define the variational problem.
mesh = fn.Mesh(filepath)
bound_name = meshname.split('.')[0] + '_facet_region.xml'
sub_name = meshname.split('.')[0] + '_physical_region.xml'
file_bound = mesh_folder_path + '/' + bound_name
file_sub = mesh_folder_path + '/' + sub_name
subdomains = fn.MeshFunction('size_t', mesh, file_sub)
boundaries = fn.MeshFunction('size_t', mesh, file_bound)

boundaries_ids, _ = GMSHInterface.get_boundaries_ids(class_call.geo_filename)
subdomains_ids = GMSHInterface.get_subdomains_ids(class_call.geo_filename)


# Define material properties.


class Rel_Perm(fn.UserExpression):
	def __init__(self, markers, **kwargs):
		super().__init__(**kwargs)
		self.markers = markers

	def eval_cell(self, values, x, cell):
		if self.markers[cell.index] == subdomains_ids['Vacuum']:
			values[0] = fn.Constant(1.)
		else:
			values[0] = fn.Constant(10.)


rel_perm = Rel_Perm(subdomains, degree=0)

# Check the material properties were properly defined.
# V0 = fn.FunctionSpace(mesh, 'DG', 0)
# Poisson.plot(fn.project(rel_perm, V0))

V = fn.FunctionSpace(mesh, 'Lagrange', 2)
u = fn.TrialFunction(V)
v = fn.TestFunction(V)
dx = fn.Measure('dx')(subdomain_data=subdomains)
dS = fn.Measure('dS')(subdomain_data=boundaries)
ds = fn.Measure('ds')(subdomain_data=boundaries)
a = rel_perm * fn.inner(fn.grad(u), fn.grad(v)) * dx
L = fn.Constant(0.) * v * dx + fn.Constant(1.)*v("-")*dS(boundaries_ids['Interface'])

# Define the boundary conditions.
bc_TOP = fn.DirichletBC(V, fn.Constant(-1e3), boundaries, boundaries_ids['Top_Wall'])
bc_INLET = fn.DirichletBC(V, fn.Constant(0.0), boundaries, boundaries_ids['Inlet'])
bc_BW = fn.DirichletBC(V, fn.Constant(0.0), boundaries, boundaries_ids['Bottom_Wall'])
bcs = [bc_TOP, bc_INLET, bc_BW]

# Solve the problem.
Potential = fn.Function(V)
fn.solve(a == L, Potential, bcs)

vac_rtc = Restrictions.generate_subdomain_restriction(mesh, subdomains, [subdomains_ids['Vacuum']])

# Get the coordinates where the electric field will be evaluated.
r_nodes, z_nodes = PostProcessing.get_nodepoints_from_boundary(mesh, boundaries, boundaries_ids['Interface'])
coords_nodes = [r_nodes, z_nodes]

# Evaluate the field.
Potential_vac = Poisson.block_project(Potential, mesh, vac_rtc, subdomains, subdomains_ids['Vacuum'],
                                      space_type='scalar')
E_vac_tensor = -fn.grad(Potential_vac)

# Do a block projection.
W = mp.BlockFunctionSpace([fn.VectorFunctionSpace(mesh, 'CG', 1)], restrict=[vac_rtc])
trial, = mp.BlockTrialFunction(W)
test, = mp.BlockTestFunction(W)
lhs = [[fn.inner(trial, test)*dx(subdomains_ids['Vacuum'])]]
rhs = [fn.inner(E_vac_tensor, test)*dx(subdomains_ids['Vacuum'])]
LHS = mp.block_assemble(lhs)
RHS = mp.block_assemble(rhs)
bc = mp.DirichletBC(W.sub(0).sub(0), fn.Constant(0.), boundaries, boundaries_ids['Lateral_Wall_L'])
bc = mp.BlockDirichletBC([bc])
bc.apply(LHS)
bc.apply(RHS)
sol = mp.BlockFunction(W)
mp.block_solve(LHS, sol.block_vector(), RHS)
E_vac = sol[0]

E_r, E_z = Poisson.split_field_components(E_vac, coords_nodes)

# Plot the result.
plotpy.lineplot([(r_nodes, E_r, r'Radial ($r$)'),
                 (r_nodes, E_z, r'Axial ($z$)')],
                xlabel=x_label, ylabel=y_label, fig_title=r'Components of the electric field. Cone height: 300 $\mathbf{\mu m}$',
                legend_title='Field Components')
