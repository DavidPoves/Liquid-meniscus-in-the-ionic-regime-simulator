import fenics as fn
import multiphenics as mp
import os
import numpy as np

from MainMenu import run_main_menu

from Tools.MeshConverter import msh2xml
from Tools.GMSH_Interface import GMSHInterface
from Tools.generate_restrictions import Restrictions
from Tools.PostProcessing import PostProcessing
import matplotlib.pyplot as plt
import dolfin_dg


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

# %% CHECK DISCONTINUOUS SPACE FUNCTIONS.
dx = fn.Measure('dx')(subdomain_data=subdomains)
dS = fn.Measure('dS')(subdomain_data=boundaries)
dS = dS(boundaries_ids['Interface'])
ds = fn.Measure('ds')(subdomain_data=boundaries)
coords = fn.SpatialCoordinate(mesh)
r, z = coords[0], coords[1]
r_nodes, z_nodes = PostProcessing.get_nodepoints_from_boundary(mesh, boundaries, boundaries_ids['Interface'])

V = fn.FunctionSpace(mesh, 'DG', 2)
u = fn.TrialFunction(V)
v = fn.TestFunction(V)

a = fn.inner(fn.grad(u), fn.grad(v))*dx
L = r*z*v*dx(subdomains_ids['Vacuum']) + fn.Constant(1.)*v*dx(subdomains_ids['Liquid'])
check = fn.Function(V)
# bc = dolfin_dg.DGDirichletBC(ds(boundaries_ids['Lateral_Wall_R']), fn.Constant(10))
bc = fn.DirichletBC(V, 10, boundaries, boundaries_ids['Lateral_Wall_R'], method='geometric')
fn.solve(a == L, check, bc)

V_vec = fn.VectorFunctionSpace(mesh, 'DG', 1)
check_grad = fn.project(fn.grad(check), V_vec)

# %% TEST
V = fn.FunctionSpace(mesh, 'DG', 2)
u = fn.TrialFunction(V)
v = fn.TestFunction(V)
a = fn.inner(fn.grad(u), fn.grad(v))*dx(subdomains_ids['Vacuum']) + \
            fn.inner(u, v)*dx(subdomains_ids['Liquid'])

L = fn.Constant(0.)*v*dx(subdomains_ids['Vacuum']) + \
    fn.Constant(0.)*v*dx(subdomains_ids['Liquid'])

bc = fn.DirichletBC(V, 10, boundaries, boundaries_ids['Top_Wall'], method='geometric')
phi = fn.Function(V)
fn.solve(a == L, phi, bc)
fn.plot(phi)

dolfin_dg.EllipticOperator()