#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:57:27 2020

@author: davidpoves
"""

import fenics as fn
import numpy as np
import matplotlib.pyplot as plt
from Solvers.Poisson_solver import Poisson
import multiphenics as mp

mesh = fn.Mesh('MESH OBJECTS/B_00467_N_800.xml')
boundaries = fn.MeshFunction('size_t', mesh, 'MESH OBJECTS/B_00467_N_800_facet_region.xml')
subdomains = fn.MeshFunction('size_t', mesh, 'MESH OBJECTS/B_00467_N_800_physical_region.xml')

boundaries_ids = {'Bottom_Wall': 1,
                  'Lateral_Wall_R': 2,
                  'Lateral_Wall_L': 4,
                  'Tube_Wall_L': 5,
                  'Inlet': 6,
                  'Top_Wall': 3,
                  'Tube_Wall_R': 7,
                  'Interface': 8}

# %% Solve a simple problem.
V = fn.FunctionSpace(mesh, 'Lagrange', 1)
vacuum_rtc = mp.MeshRestriction(mesh, 'RESTRICTIONS/vacuum_restriction.rtc.xml')
liquid_rtc = mp.MeshRestriction(mesh, 'RESTRICTIONS/liquid_restriction.rtc.xml')
meniscus_rtc = mp.MeshRestriction(mesh, 'RESTRICTIONS/interface_restriction.rtc.xml')

# %%
V = fn.FunctionSpace(mesh, 'Lagrange', 2)
u = fn.TrialFunction(V)
v = fn.TestFunction(V)

dx = fn.Measure('dx')(subdomain_data=subdomains)
dS = fn.Measure('dS')(subdomain_data=boundaries)
dS = dS(boundaries_ids['Interface'])
r = fn.SpatialCoordinate(mesh)[0]

a = r*fn.inner(fn.grad(u), fn.grad(v))*dx(9) + r*10*fn.inner(fn.grad(u), fn.grad(v))*dx(10)
L = -r*fn.Constant(0.)*v("+")*dS

bcs = []
boundary_conditions = {'Top_Wall': {'Dirichlet': [fn.Constant(-10.), 9]},
                       'Inlet': {'Dirichlet': [fn.Constant(0.), 10]},
                       'Tube_Wall_R': {'Dirichlet': [fn.Constant(0.), 10]},
                       'Bottom_Wall': {'Dirichlet': [fn.Constant(0.), 9]},
                       'Lateral_Wall_R': {'Neumann': 9},
                       'Lateral_Wall_L': {'Neumann': 9},
                       'Tube_Wall_L': {'Neumann': 10}}
for i in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[i]:
        sub_id = boundary_conditions[i]['Dirichlet'][1] - 9
        bc_val = boundary_conditions[i]['Dirichlet'][0]
        bc = fn.DirichletBC(V, bc_val, boundaries,
                            boundaries_ids[i])
        bcs.append(bc)
        # Check the created boundary condition.
        assert len(bc.get_boundary_values()) > 0., f'Wrongly defined boundary {i}'

phi = fn.Function(V)
fn.solve(a==L, phi, bcs)

# %% EVALUATE THE GRADIENTS AT THE NODES OF THE INTERFACE.
F = V.dim()
dofmap = V.dofmap()
dofs = dofmap.dofs()
u = fn.Function(V)
bc = fn.DirichletBC(V, fn.Constant(1.0), boundaries, 8)
bc.apply(u.vector())
dofs_bc = list(np.where(u.vector()[:] == 1.0))

dofs_x = V.tabulate_dof_coordinates().reshape(F, mesh.topology().dim())

coords_r = []
coords_z = []

# Get the coordinates of the nodes on the meniscus.
for dof, coord in zip(dofs, dofs_x):
    if dof in dofs_bc[0]:
        coords_r.append(coord[0])
        coords_z.append(coord[1])
r_nodes = np.sort(coords_r)
z_nodes = np.sort(coords_z)[::-1]

# %%
vacuum_mesh = fn.SubMesh(mesh, subdomains, 9)
V_vec = fn.VectorFunctionSpace(mesh, 'Lagrange', 1)
W = mp.BlockFunctionSpace([phi.function_space()], restrict=[vacuum_rtc])
v, = mp.BlockTestFunction(W)
u, = mp.BlockTrialFunction(W)
dx = fn.Measure('dx')(subdomain_data=subdomains)

lhs = [[fn.inner(u, v)*dx(9)]]
rhs = [fn.inner(phi, v)*dx(9)]
RHS = mp.block_assemble(rhs)
LHS = mp.block_assemble(lhs)

phiv = mp.BlockFunction(W)
mp.block_solve(LHS, phiv.block_vector(), RHS)
phiv = phiv[0]

W = mp.BlockFunctionSpace([V_vec], restrict=[vacuum_rtc])
v, = mp.BlockTestFunction(W)
u, = mp.BlockTrialFunction(W)

lhs = [[fn.inner(u, v)*dx(9)]]
rhs = [fn.inner(-fn.nabla_grad(phiv), v)*dx(9)]
RHS = mp.block_assemble(rhs)
LHS = mp.block_assemble(lhs)

E_v = mp.BlockFunction(W)
mp.block_solve(LHS, E_v.block_vector(), RHS)
E_v = E_v[0]

# Evaluate the fields at the nodes.
E_v_eval = []
# E_l_eval = []

for r, z in zip(r_nodes, z_nodes):
    E_v_eval.append(E_v([r, z]))
    # E_l_eval.append(E_l([r, z]))

plt.figure(1)
plt.plot(r_nodes, E_v_eval)
plt.figure(2)
plt.plot(coords_r, E_l_eval)

# %% COMPUTE THE NORMAL PROJECTION OF THE ELECTRIC FIELD.
n = fn.FacetNormal(mesh)
E_v_n = fn.dot(E_v("+"), n("+"))
E_l_n = fn.dot(E_l("-"), n("-"))

dS = fn.Measure('dS')(subdomain_data=boundaries)
dS = dS(7)  # Restrict this to the interface.
h = fn.FacetArea(mesh)

q = fn.TestFunction(V)
E_v_n = fn.assemble(E_v_n*q("+")/h("+")*dS)
E_v_n = fn.Function(V, E_v_n)

q = fn.TestFunction(V)
E_l_n = fn.assemble(E_l_n*q("-")/h("-")*dS)
E_l_n = fn.Function(V, E_l_n)

"""
Normal components of the fields are defined at the midpoints, so we must
evaluate the previous functions on the midpoints of the facets.
"""
r_mids = np.array([])  # Preallocate the coordinates array.
z_mids = np.array([])  # Preallocate the coordinates array.
interface_facets = fn.SubsetIterator(boundaries, 7)
for facet in interface_facets:
    r_mids = np.append(r_mids, facet.midpoint()[0])
    z_mids = np.append(z_mids, facet.midpoint()[1])

# Sort the r coordinates of the midpoints in ascending order.
r_mids = np.sort(r_mids)

# Sort the z coordinates in descending order.
z_mids = np.sort(z_mids)[::-1]

# Evaluate the fields at the midpoints.
E_v_n_eval = []
E_l_n_eval = []

for r, z in zip(r_mids, z_mids):
    E_v_n_eval.append(E_v_n([r, z+fn.DOLFIN_EPS]))
    E_l_n_eval.append(E_l_n([r, z]))

plt.figure(3)
plt.plot(r_mids, E_v_n_eval)
plt.figure(4)
plt.plot(z_mids, E_l_n_eval)