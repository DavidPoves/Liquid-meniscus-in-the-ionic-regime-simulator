import scipy.io as sio
import os
import _collections_abc
import numpy as np

""" Create a function that will be used bu several plotting functions to export plotted data to .mat files, which will
ready to be used by Matlab for post processing purposes.
"""


def export2mat(data, filename, mat_folder):
	"""
	Export data to .mat files. Data should be introduced in the form [x_data, y_data].
	Dictionaries should also use this same order: {'x_label': x_data, 'y_label': y_data}.
	:param data: Information to be exported. It can be a dictionary or an array like object.
	:param filename: String containing the name of the .mat file.
	:param mat_folder: Folder where the file should be saved.
	:return:
	"""
	filepath = os.path.join(mat_folder, filename + '.mat')
	# Previously check the type of data introduced and raise error if non compatibilities arise.
	if not isinstance(data, (_collections_abc.Sequence, np.ndarray, dict)):
		raise TypeError(f'Data type {type(data)} is not valid to create a .mat file.')
	if isinstance(data, dict):
		data_temp = list()
		for data_arr in list(data.values()):
			data_temp.append(data_arr)
		data = data_temp
	# Create the mat from a dictionary
	data_mat = {'x': data[0], 'y': data[1]}
	sio.savemat(filepath, data_mat)
