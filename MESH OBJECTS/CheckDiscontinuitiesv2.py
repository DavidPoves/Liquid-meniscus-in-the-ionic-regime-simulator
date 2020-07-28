import fenics as fn
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------


def get_nodepoints_from_boundary(boundaries_data, boundary_id):
	# Define an auxiliary Function Space.
	V = fn.FunctionSpace(mesh, 'Lagrange', 1)

	# Get the dimension of the auxiliary Function Space.
	F = V.dim()

	# Generate a map of the degrees of freedom (=nodes for this case).
	dofmap = V.dofmap()
	dofs = dofmap.dofs()

	# Apply a Dirichlet BC to a function to get nodes where the bc is applied.
	u = fn.Function(V)
	bc = fn.DirichletBC(V, fn.Constant(1.0), boundaries_data, boundary_id)
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
	coords_r = np.sort(coords_r)
	coords_z = np.sort(coords_z)[::-1]

	return coords_r, coords_z

# ---------------------------------------------------------------------------------------------------------------------
meshname = 'Prueba.xml'
mesh = fn.Mesh(meshname)
boundaries = fn.MeshFunction('size_t', mesh, f"{meshname.split('.')[0]}_facet_region.xml")
subdomains = fn.MeshFunction('size_t', mesh, f"{meshname.split('.')[0]}_physical_region.xml")
boundaries_ids = {'Inlet': 1, 'Tube_Wall_R': 2, 'Tube_Wall_L': 3, 'Bottom_Wall': 4, 'Lateral_Wall_R': 5, 'Top_Wall': 6,
                  'Lateral_Wall_L': 7, 'Interface': 8}
subdomains_ids = {'Vacuum': 9, 'Liquid': 10}

# Test the discontinuity.
V_disc = fn.FunctionSpace(mesh, 'DG', 2)
u = fn.TrialFunction(V_disc)
v = fn.TestFunction(V_disc)

dx = fn.Measure('dx')(subdomain_data=subdomains)
coords = fn.SpatialCoordinate(mesh)
r, z = coords[0], coords[1]

a = fn.inner(u, v)*dx(subdomains_ids['Vacuum']) + fn.inner(u, v)*dx(subdomains_ids['Liquid'])
L = r*z*v*dx(subdomains_ids['Vacuum']) + fn.Constant(1.)*v*dx(subdomains_ids['Liquid'])

check = fn.Function(V_disc)
fn.solve(a==L, check)

# Get coordinates from boundary.
coords_r, coords_z = get_nodepoints_from_boundary(boundaries, boundaries_ids['Interface'])

# %% Get the solution at one side of the interface (given by the sign)
V_cont = fn.FunctionSpace(mesh, 'Lagrange', 2)
dS = fn.Measure('dS')(subdomain_data=boundaries)
dS = dS(boundaries_ids['Interface'])
v = fn.TestFunction(V_cont)
u = fn.TrialFunction(V_cont)

sign = "-"  # "+" for liquid, "-" for vacuum
a = fn.inner(u(sign), v(sign))*dS + fn.inner(u, v)*dx(subdomains_ids['Vacuum']) + fn.inner(u, v)*dx(subdomains_ids['Liquid'])
L = check(sign)*v(sign)*dS + fn.Constant(0.)*v*dx(subdomains_ids['Vacuum']) + fn.Constant(0.)*v*dx(subdomains_ids['Liquid'])

check_2 = fn.Function(V_cont)
fn.solve(a==L, check_2)

print(check_2(coords_r[10], coords_z[10]))
