import os

import fenics as fn
import multiphenics as mp

from MainMenu import run_main_menu
from Tools.GMSH_Interface import GMSHInterface
from Tools.MeshConverter import msh2xml
from Tools.generate_restrictions import Restrictions

# %% GET THE MESH PATH AND NAME.
mesh_app = run_main_menu()
filepath = mesh_app.msh_filename

filename = filepath.split('/')[-1]
mesh_folder_path = '/'.join(filepath.split('/')[:-1])
root_folder = os.getcwd()

if filename.split('.')[-1] != 'xml':
	meshname = msh2xml(filename, root_folder, mesh_folder_path)
else:
	meshname = filename

meshpath = mesh_folder_path + f'/{meshname}'

# %% LOAD THE MESH AND ITS ASSOCIATED FILES INTO FENICS.
mesh = fn.Mesh(meshpath)
boundaries = fn.MeshFunction('size_t', mesh, mesh_folder_path + f"/{meshname.split('.')[0]}_facet_region.xml")
subdomains = fn.MeshFunction('size_t', mesh, mesh_folder_path + f"/{meshname.split('.')[0]}_physical_region.xml")
boundaries_ids, _ = GMSHInterface.get_boundaries_ids(mesh_folder_path + '/' + filename.split('.')[0] + '.geo')
subdomains_ids = GMSHInterface.get_subdomains_ids(mesh_folder_path + '/' + filename.split('.')[0] + '.geo')

# %% CHECK IF SUBDOMAINS 'TALK' BETWEEN EACH OTHER WHEN COMPUTING THE GRADIENT.
dx = fn.Measure('dx')(subdomain_data=subdomains)
dS = fn.Measure('dS')(subdomain_data=subdomains)
dS = dS(boundaries_ids['Interface'])
# Create the restrictions.
vacuum_rtc = Restrictions.generate_subdomain_restriction(mesh, subdomains, [subdomains_ids['Vacuum']])
liquid_rtc = Restrictions.generate_subdomain_restriction(mesh, subdomains, [subdomains_ids['Liquid']])

# Create the function space
V = fn.FunctionSpace(mesh, 'Lagrange', 2)
W = mp.BlockFunctionSpace([V, V], restrict=[vacuum_rtc, liquid_rtc])

# Generate the functions.
test = mp.BlockTestFunction(W)
(v1, v2) = mp.block_split(test)
trial = mp.BlockTrialFunction(W)
(u1, u2) = mp.block_split(trial)

# Create aux. variables.
x = fn.SpatialCoordinate(mesh)
r, z = x[0], x[1]

# Define the variational form.
aa = [[fn.inner(u1, v1)*dx(subdomains_ids['Vacuum']), 0],
	  [0, fn.inner(u2, v2)*dx(subdomains_ids['Liquid'])]]

# Impose a known function on one subdomain and a constant on the other.
bb = [r*z*v1*dx(subdomains_ids['Vacuum']), fn.Constant(0.)*v2*dx(subdomains_ids['Liquid'])]

AA = mp.block_assemble(aa)
BB = mp.block_assemble(bb)

check = mp.BlockFunction(W)
mp.block_solve(AA, check.block_vector(), BB)

# Compute the gradient (1st way).
check_full = fn.project(check[0]+check[1], V)
V_vec = fn.VectorFunctionSpace(mesh, 'Lagrange', 1)
v = fn.TestFunction(V_vec)
u = fn.TrialFunction(V_vec)
a = fn.inner(u, v)*dx(subdomains_ids['Vacuum'])
L = fn.inner(fn.grad(check_full), v)*dx(subdomains_ids['Vacuum'])
check_grad = fn.Function(V_vec)
fn.solve(a == L, check_grad)

# Compute the gradient (2nd way).
"""
Results of this method yield that when doing this projection the gradients for each of the subdomains are computed
without taking into account the information from the other subdomain, which is the desired behaviour.
"""
check_full = fn.project(check[0] + check[1], V)
W = mp.BlockFunctionSpace([V_vec, V_vec], restrict=[vacuum_rtc, liquid_rtc])
test = mp.BlockTestFunction(W)
(v1, v2) = mp.block_split(test)
trial = mp.BlockTrialFunction(W)
(u1, u2) = mp.block_split(trial)

aa = [[fn.inner(u1, v1)*dx(subdomains_ids['Vacuum']), 0],
	  [0, fn.inner(u2, v2)*dx(subdomains_ids['Liquid'])]]
bb = [fn.inner(fn.grad(check[0]), v1)*dx(subdomains_ids['Vacuum']),
	  fn.inner(fn.grad(check[1]), v2)*dx(subdomains_ids['Liquid'])]

AA = mp.block_assemble(aa)
BB = mp.block_assemble(bb)

check_grad_2 = mp.BlockFunction(W)
mp.block_solve(AA, check_grad_2.block_vector(), BB)

# %% TEST
V = fn.FunctionSpace(mesh, 'DG', 2)
check_full = fn.project(check[0] + check[1], V)
V_vec = fn.VectorFunctionSpace(mesh, 'DG', 1)
gradient = fn.project(fn.grad(check_full), V_vec)


