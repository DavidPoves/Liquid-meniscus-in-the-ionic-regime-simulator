import fenics as fn
import numpy as np
import matplotlib.pyplot as plt
from gmsh_handle import gmsh_handle
import sys

b0 = 1e-5
h = b0
tol = 1e-14

# %% Check boundaries.
# boundary_file = fn.File('boundaries.pvd')
# boundary_file << boundaries

# %%
mesh = fn.Mesh('Tank.xml')
D = mesh.topology().dim()
boundaries = fn.MeshFunction('size_t', mesh, 'Tank_facet_region.xml')
mesh.init(D-1, D)  # build connection between facets and cells.
physical_curves = gmsh_handle.get_physical_curves('Tank.geo')

def z(r, h, b0):
    A = h
    B = np.pi/(2*b0)
    return A*np.cos(B*r)


map_fun = lambda r: z(r, h, b0)

r_coords = np.linspace(0, b0, 800)
z_coords = list(map(map_fun, r_coords))

# boundaries.set_all(0)
# Meniscus().mark(boundaries, 1)

n = []
origin = []
for i in range(mesh.num_facets()):
    Facet = fn.Facet(mesh, i)
    if boundaries[Facet] == physical_curves['"Meniscus"']:
        mdp = Facet.midpoint()
        origin.append(mdp.array())
        n.append([Facet.normal().x(), Facet.normal().y(), Facet.normal().z()])

n_pack = zip(origin, n)  # Package containing the origin coords and the vector.

plt.figure()
for ori, n_ in n_pack:
    soa = np.array([[ori[0], ori[1], n_[0], n_[1]]])
    X, Y, U, V = zip(*soa)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    plt.draw()

try:
    ax.plot(r_coords, z_coords)
except NameError:
    sys.exit('No normal vector was found. Stopping the execution')
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
plt.show()

# %% PRUEBA SIMPLE


class Lateral(fn.SubDomain):
    def inside(self, x, on_boundary):
        return fn.near(x[1], 1, tol)


mesh = fn.UnitSquareMesh(2, 2)
D = mesh.topology().dim()
boundaries = fn.MeshFunction('size_t', mesh, D-1)
boundaries.set_all(0)
Lateral().mark(boundaries, 1)
coords = mesh.coordinates()
mesh.init(D-1, D)  # build connection between facets and cells.

n = []
origin = []
for i in range(mesh.num_facets()):
    Facet = fn.Facet(mesh, i)
    if boundaries[Facet] == 1:
        mdp = Facet.midpoint()
        origin.append(mdp.array())
        n.append([Facet.normal().x(), Facet.normal().y(), Facet.normal().z()])

n_pack = zip(origin, n)  # Package containing the origin coords and the vector.
plt.figure()
for ori, n_ in n_pack:
    soa = np.array([[ori[0], ori[1], n_[0], n_[1]]])
    X, Y, U, V = zip(*soa)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
    plt.draw()
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
fn.plot(mesh)
plt.show()
