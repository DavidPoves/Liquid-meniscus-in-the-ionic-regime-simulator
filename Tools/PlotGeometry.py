import matplotlib.pyplot as plt
import numpy as np
from mpldatacursor import datacursor

from Tools.GMSH_Interface import GMSHInterface


def on_pick(event):
	artist = event.artist
	xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
	x, y = artist.get_xdata(), artist.get_ydata()
	ind = event.ind
	print('Artist picked:', event.artist)
	print('{} vertices picked'.format(len(ind)))
	print('Pick between vertices {} and {}'.format(min(ind), max(ind) + 1))
	print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))
	print('Data point:', x[ind[0]], y[ind[0]])


def plot_geo(filepath):
	"""
	Plot a .geo file and interact with the graph. User may check the names of the physical curves (boundaries) by
	hovering the cursor over the desired curve/boundary.
	Args:
		filepath: Path of the .geo file.

	Returns:

	"""
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
		r = curve[::2].T  # Extract r coordinates
		r = r.reshape(len(r), 1)  # Reshape for proper dimensions
		z = curve[1::2].T
		z = z.reshape(len(z), 1)

		arr = np.concatenate((r, z), axis=1)  # Join the two previous arrays to create an unique array.
		arr = arr[arr[:, 0].argsort()]  # Sort the values for proper plotting.

		plt.plot(arr[:, 0], arr[:, 1], label=tag)
	Cursor = datacursor(formatter='{label}'.format)
	plt.legend()
	plt.show(block=False)

	return Cursor


if __name__ == '__main__':
	path = "/Users/davidpoves/TFG/CODE/TFG-V2/MESH OBJECTS/Prueba.geo"
	interactive_data = plot_geo(path)
