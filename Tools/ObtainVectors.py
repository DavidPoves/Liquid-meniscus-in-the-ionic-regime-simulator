import fenics as fn
import numpy as np

class ObtainVectors(object):
	def __init__(self):
		pass

	@staticmethod
	def get_normal_vectors_boundary(mesh, subdomain_data, boundary_id):
		n_r = []
		n_z = []
		for i in range(mesh.num_facets()):
			facet = fn.Facet(mesh, i)
			if subdomain_data[i] == boundary_id:
				n_r.append(facet.normal().x())
				n_z.append(facet.normal().y())

		# Stack both vectors together into a multidimensional array.
		n = np.column_stack((n_r, n_z))

		# Sort the array.
		n = ObtainVectors.sort_by_column(n, 1)

		return n

	@staticmethod
	def get_tangential_vectors_boundary(n=None, mesh=None, subdomain_data=None, boundary_id=None):
		if n is not None:
			return np.column_stack((n[:, 1], [-1*nr for nr in n[:, 0]]))
		else:
			n = ObtainVectors.get_normal_vectors_boundary(mesh, subdomain_data, boundary_id)
			return np.column_stack((n[:, 1], [-1*nr for nr in n[:, 0]]))

	@staticmethod
	def sort_by_column(array, column):
		return array[array[:, column].argsort()]
