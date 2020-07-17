#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:11:54 2020

@author: davidpoves
"""

import fenics as fn
import numpy as np
import matplotlib.pyplot as plt

mesh = fn.Mesh('B_00467_N_800.xml')
D = mesh.topology().dim()
boundaries = fn.MeshFunction('size_t', mesh, 'B_00467_N_800_facet_region.xml')
mesh.init(D-1, D)  # build connection between facets and cells.

# %% Extract the nodes coordinates.
V = fn.FunctionSpace(mesh, 'Lagrange', 1)
F = V.dim()
dofmap = V.dofmap()
dofs = dofmap.dofs()
u = fn.Function(V)
bc = fn.DirichletBC(V, fn.Constant(1.0), boundaries, 8)
bc.apply(u.vector())
dofs_bc = list(np.where(u.vector()[:] == 1.0))

dofs_x = V.tabulate_dof_coordinates().reshape(F, D)

coords_r = []
coords_z = []

# Get the coordinates of the nodes on the meniscus.
for dof, coord in zip(dofs, dofs_x):
    if dof in dofs_bc[0]:
        coords_r.append(coord[0])
        coords_z.append(coord[1])
coords_r = np.sort(coords_r)
coords_z = np.sort(coords_z)[::-1]

plt.plot(coords_r, coords_z)

# Get the midpoints of the meniscus' facets.
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

# %% Evaluate the gradient of a known function.
class FEniCSExpression(fn.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = x[0]*x[1]

    def value_shape(self):
        return ()

expression = FEniCSExpression()
function = fn.interpolate(expression, V)
"""
Interpolation means to evaluate the expression at all dofs of the function
space.
"""
function_grad = fn.grad(function)  # Compute the gradient.
V_vec = fn.VectorFunctionSpace(mesh, 'CG', 1)
function_grad_evaluable = fn.project(function_grad, V_vec)

E_eval = []
for r, z in zip(coords_r, coords_z):
    E_eval.append(function_grad_evaluable([r, z]))

error_r = []
error_z = []
# Compute the error of the gradient.
for ix in range(0, len(E_eval)-1):
    error_r.append(abs(E_eval[ix][1]-coords_r[ix])/E_eval[ix][1])
    error_z.append(abs(E_eval[ix][0]-coords_z[ix])/E_eval[ix][0])

# From previous part, we can conclude that the gradient is computed at nodes.

# Now, we will check the dot product.
n = fn.FacetNormal(mesh)

fun_dot_n = fn.dot(function_grad("+"), n("+"))
dS = fn.Measure('dS')(subdomain_data=boundaries)
dS = dS(7)
q = fn.TestFunction(V)
h = fn.FacetArea(mesh)
fun_dot_n = fn.assemble(fun_dot_n*q("+")/h("+")*dS)
fun_dot_n = fn.Function(V, fun_dot_n)

fun_dot_n_eval = []
for r, z in zip(r_mids, z_mids):
    fun_dot_n_eval.append(fun_dot_n([r, z]))

# Compare the values in fun_dot_n_eval with the ones from the next line.
# np.dot((E_eval[1]+E_eval[0])/2, [u_dot_n_r_eval[0], u_dot_n_z_eval[0]])

# %% Check the normal vectors.
# Create a velocity vector.
V_element = fn.FiniteElement('Lagrange', 'triangle', 1)
dS = fn.Measure('dS')(subdomain_data=boundaries)
dS = dS(8)
u = fn.Expression(("1.", "0."), degree=V_element._degree)
u_fun = fn.interpolate(u, V_vec)
n = fn.FacetNormal(mesh)
u_n = fn.dot(u_fun("+"), n("+"))
h = fn.FacetArea(mesh)
q = fn.TestFunction(V)
u_dot_n = fn.assemble(u_n*q("+")/h("+")*dS)
u_dot_n = fn.Function(V, u_dot_n)

r_mids = np.array([])  # Preallocate the coordinates array.
z_mids = np.array([])  # Preallocate the coordinates array.

# Get the midpoints of the meniscus' facets.
interface_facets = fn.SubsetIterator(boundaries, 8)
for facet in interface_facets:
    r_mids = np.append(r_mids, facet.midpoint()[0])
    z_mids = np.append(z_mids, facet.midpoint()[1])

# Sort the r coordinates of the midpoints in ascending order.
r_mids = np.sort(r_mids)

# Sort the z coordinates in descending order.
z_mids = np.sort(z_mids)[::-1]

# Evaluate the dot product at the midpoints.
u_dot_n_r_eval = []
for r, z in zip(r_mids, z_mids):
    u_dot_n_r_eval.append(u_dot_n([r, z]))

u = fn.Expression(("0.", "1."), degree=V_element._degree)
u_fun = fn.interpolate(u, V_vec)
n = fn.FacetNormal(mesh)
u_n = fn.dot(u_fun("+"), n("+"))
h = fn.FacetArea(mesh)
q = fn.TestFunction(V)
u_dot_n = fn.assemble(u_n*q("+")/h("+")*dS)
u_dot_n = fn.Function(V, u_dot_n)

# Evaluate the dot product at the midpoints.
u_dot_n_z_eval = []
for r, z in zip(r_mids, z_mids):
    u_dot_n_z_eval.append(u_dot_n([r, z]))

n_pack = zip([[r, z] for r,z in zip(r_mids, z_mids)], [[nr, nz] for nr,nz in zip(u_dot_n_r_eval, u_dot_n_z_eval)])  # Package containing the origin coords and the vector.
for ori, n_ in n_pack:
    soa = np.array([[ori[0], ori[1], n_[0], n_[1]]])
    X, Y, U, V = zip(*soa)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    plt.draw()

ax.plot(r_mids, z_mids)

# %% COMPUTE THE NORMAL VECTORS FROM THE NODES DATA.
n_nodes = []
for ix in range(1, len(coords_r)):
    dr = coords_r[ix] - coords_r[ix-1]
    dz = coords_z[ix] - coords_z[ix-1]
    n_nodes.append([-dz, dr])
n_pack = zip([[r, z] for r,z in zip(r_mids, z_mids)], n_nodes)  # Package containing the origin coords and the vector.
for ori, n_ in n_pack:
    soa = np.array([[ori[0], ori[1], n_[0], n_[1]]])
    X, Y, U, V = zip(*soa)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    plt.draw()

ax.plot(r_mids, z_mids)