import matplotlib.pyplot as plt
import numpy as np

from Tools.GMSH_Interface import GMSHInterface


def plot_geo(filepath):
	# Extract the points and curves from the .geo file.
	points_dict, curves_dict = GMSHInterface.extract_points_from_geo(filepath)
	boundaries_ids, physical_dict = GMSHInterface.get_boundaries_ids(filepath)

	plot_dict = dict()

	# Now, we have all the information to plot unify curves by their physical group.
	for name, curves in physical_dict.items():
		aux_arr = np.array(())
		for curve in curves.split(','):
			points = curves_dict[curve.strip()]
			for point_str in points.strip().split(','):
				point_coord = points_dict[point_str.strip()]
				aux_arr = np.append(aux_arr, (point_coord[0], point_coord[1]))
		plot_dict[name.strip()] = aux_arr

	plt.figure()
	for tag, curve in plot_dict.items():
		r = curve[::2].T
		r = r.reshape(len(r), 1)
		z = curve[1::2].T
		z = z.reshape(len(z), 1)
		arr = np.concatenate((r, z), axis=1)
		arr = arr[arr[:, 0].argsort()]
		plt.plot(arr[:, 0], arr[:, 1], label=tag)
	plt.show()
	plt.legend()


if __name__ == '__main__':
	path = "/Users/davidpoves/TFG/CODE/TFG-V2/MESH OBJECTS/Prueba.geo"
	plot_geo(path)
