# Copyright (C) 2020- by David Poves Ros
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SciencePlotting is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#

# Import the libraries.
import numpy as np
import os
import re
import warnings
import _collections_abc
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
from ElectrospraySimulator.Tools.fplot import FPlot
from ElectrospraySimulator.Tools.export2mat import export2mat


class SciencePlotting(object):

	def __init__(self, **kwargs):
		"""
		Initialize the plotting class.
		:param kwargs: To get a full list of all available inputs when initializing this class, check the static method
		check_init_inputs(), or type help(SciencePlotting).
		"""
		# Set defaults for kwargs.
		kwargs.setdefault('use_latex', True)
		kwargs.setdefault('figsize', (12, 7))
		kwargs.setdefault('fontsize', 16)
		kwargs.setdefault('images_folder', os.path.join(os.getcwd(), 'IMAGES'))
		kwargs.setdefault('mat_folder', os.path.join(os.getcwd(), 'MAT FILES'))
		kwargs.setdefault('style', 'science')

		# Perform the appropriate checks on the introduced kwargs.

		# Set the kwargs.
		self.use_latex = kwargs.get('use_latex')
		self.figsize = kwargs.get('figsize')
		self.fontsize = kwargs.get('fontsize')
		self.images_folder = kwargs.get('images_folder')
		self.mat_folder = kwargs.get('mat_folder')

		# Process the styles input.
		if not SciencePlotting._check_style('science'):
			raise ImportError('The science style is missing. Install it from https://pypi.org/project/SciencePlots/ or '
			                  'run: pip install SciencePlots')
		self.style = np.array(['science'])
		input_styles = kwargs.get('style')
		if isinstance(input_styles, str):
			if SciencePlotting._check_style(input_styles):
				self.style = np.append(self.style, input_styles)
			else:
				warnings.warn(f'The introduced style {input_styles} is not installed. Check your available styles with '
				              'plt.style.available. Ignoring the introduced style')

		elif isinstance(input_styles, (_collections_abc.Sequence, np.ndarray)) and not isinstance(input_styles, str):
			for st in input_styles:
				if st != 'science':  # To avoid overriding the science style, which is introduced by default.
					self.style = np.append(self.style, st)
				else:
					pass

		# If the user does not want to use latex, add the corresponding style.
		if not self.use_latex:
			self.style = np.append(self.style, 'no-latex')

		# Set the introduced parameters.
		plt.style.use(self.style)
		plt.rcParams.update({'font.size': self.fontsize})
		plt.rcParams.update({'figure.figsize': self.figsize})

		# Initialize other data.
		self.image_format = None
		self.fig, self.ax = None, None
		self.save_mat, self.save_fig = None, None
		self.fig_title, self.open_folders = None, None

	@staticmethod
	def _clean_name(name):
		"""
		Use regex pattern to eliminate non-alphanumeric characters.
		:param name: String to be cleaned.
		:return: String without non-alphanumeric characters.
		"""
		return re.sub("[^0-9a-zA-Z]+", '', name)

	@staticmethod
	def _open_directory(directory):
		"""
		Open a given directory for the user
		:param directory: String containing the full path to be opened.
		:return:
		"""
		webbrowser.open('file:///' + directory)

	@staticmethod
	def _check_style(style):
		"""
		Check if a given style is defined within Matplotlib.
		:param style: String containing the style.
		:return: Boolean indicating if the style is defined or not.
		"""
		return style in plt.style.available

	@staticmethod
	def create_pandas_dataframe(data, columns=None):
		"""
		Create a Pandas dataframe from given data. This can be useful to export data to other formats, like text files.
		:param data: Data to be used for the dataframe. It can be an array like or a dictionary object.
		:param columns: Array like containing the names of the files. If it is introduced and the data type is a dict,
		the keys of the dict will be ignored. Optional, default is None
		:return: The Panda's dataframe.
		"""
		data_processed = dict()
		if isinstance(data, dict):
			if columns is None:
				for column, value in list(data.items()):
					data_processed[SciencePlotting._clean_name(column)] = value
			else:
				assert isinstance(columns,
				                  (_collections_abc.Sequence, np.ndarray)), 'Columns input must be an array like object.'
				for column, value in zip(columns, list(data.values())):
					data_processed[column] = value
			df = pd.DataFrame(data_processed, columns=list(data_processed.keys()))
		elif isinstance(data, (_collections_abc.Sequence, np.ndarray)) and not isinstance(data, str):
			if columns is None:  # The user has not introduced a dictionary nor columns. Default headers are created.
				columns = [f'Column {i}' for i in np.arange(1, len(data) + 1)]
			else:
				assert isinstance(columns,
				                  (_collections_abc.Sequence, np.ndarray)), 'Columns input must be an array like object.'
				assert len(columns) == len(data), f'Shape mismatch between columns and data: {len(columns)}, {len(data)}'
			data_dict = dict()
			for column, data_column in zip(columns, data):
				data_dict[column] = data_column
			df = pd.DataFrame(data_dict, columns=list(data_dict.keys()))
		else:  # The introduced object cannot be converted into a dataframe
			raise TypeError(f'The introduced data type ({type(data)}) is not valid. It must be an iterable object ('
			                f'except strings) or a dictionary')

		pd.set_option('colheader_justify', 'center')  # Justify the column headers to the center.

		return df

	@staticmethod
	def _create_directory(path):
		"""
		Create a directory on a specified path if it does not exist.
		:param path: Path where the folder is supposed to be. If it does not exist, it will be created.
		:return:
		"""
		if not os.path.isdir(path):  # Check if the folder where images will be saved exists.
			os.makedirs(path)

	def _create_mat_file(self, data, filename, open_folder=False):
		"""
		Create a .mat file from given data and save it into the specified folder.
		:param data: Data to be exported. Can be a dictionary or an array like object.
		:param filename: String containing the name of the .mat file to be created.
		:param open_folder: Boolean indicating if the program should open the folder where the file was stored or not.
		Optional, default is False
		:return:
		"""
		SciencePlotting._create_directory(self.mat_folder)
		export2mat(data, filename, self.mat_folder)

		# Open the folder where saved if required.
		if open_folder:
			SciencePlotting._open_directory(self.mat_folder)
		else:
			pass

	@staticmethod
	def check_init_inputs():
		"""
		Show the available inputs when initializing the SciencePlotting class.
		- use_latex: Boolean indicating if latex should be used or not. Optional, default is True.
		- figsize: Tuple with the size of the figure to be shown. Optional, default is (12, 7).
		- fontsize: Integer indicating the size of the letters on the plot. Optional, default is 16.
		- images_folder: String containing the full path of the folder where images should be saved.
		Optional, and by default a new folder will be created (if not existing) in the current working
		directory.
		- mat_folder: Same as images_folder. Here .mat files will be stored.
		- style: String or array like object containing the styles to be used for plotting. See
		available styles using matplotlib.get_configdir(). By default, the science style will always be
		used, and other user inputs will be combined to the science style.

		Returns:
		"""
		print(['use_latex', 'figsize', 'fontsize', 'images_folder', 'mat_folder', 'style'])

	@staticmethod
	def available_methods_inputs():
		"""
		The available inputs for the lineplot and fplot methods.
		- fig_title: String containing the title of the figure. Will be used if the user needs this title to be
		written on the plot or if the image needs to be saved. In the latter, the name of the file will be this
		kwarg. Optional, default is Figure.
		- write_title: Boolean indicating if the title should be written on the plot. Optional, default is False.
		- legend_title: String containing the title of the legend. It will be used if more than one curve is
		plotted on the same graph. Optional, default is a blank string.
		- xscale: String containing the scale of the x axis. See Matplotlib documentation for a list of all
		possible options. Optional, default is linear.
		- yscale: Same as xscale, but for the y axis.
		- xlabel: Label of the x axis. It will be 'x' if when needed, the user does not introduce any label.
		Optional, default is None.
		- ylabel: Same as xlabel, but for y axis. Optional, default is y.
		- save_fig: Boolean indicating if the image should be saved or not. Optional, default is False.
		- save_mat: Boolean indicating if the plotted data should be exported to a .mat file, which can be
		imported using Matlab. Optional, default is False.
		- image_format: String containing the format of the image. This will be used only when the image is
		saved. The extension can be introduced with or without the '.' at the beginning. Optional, default is
		.png.
		- open_folders: Boolean indicating if the folders in which files have been saved should be opened for
		the user. This has been implemented in such a way that it is multiplatform compatible. Optional, default
		is False.
		"""
		lst = ['fig_title', 'write_title', 'legend_title', 'xscale', 'yscale', 'xlabel', 'ylabel', 'save_fig',
		       'save_mat', 'image_format', 'open_folders']
		print(lst)

	@staticmethod
	def _set_kwargs_defaults(**kwargs):
		"""
		Set default values for the kwargs. To get a list of all the available kwargs, call the static method
		available_inputs or type help(SciencePlotting)
		:return: Default kwargs.
		"""
		kwargs.setdefault('fig_title', 'Figure')  # Figure title.
		kwargs.setdefault('write_title', False)  # Decides whether the title should be written within the plot or not.
		kwargs.setdefault('legend_title', '')  # Write a legend title.
		kwargs.setdefault('xscale', 'linear')  # Set a scale for the x axis.
		kwargs.setdefault('yscale', 'linear')  # Set a scale for the y axis.
		kwargs.setdefault('xlabel', None)  # Set a default label for the x axis.
		kwargs.setdefault('ylabel', 'y')  # Set a default label for the y axis.
		kwargs.setdefault('save_fig', False)  # Decides if the figure will be saved or not.
		kwargs.setdefault('save_mat', False)  # Decides if the data is exported to a .mat file or not.
		kwargs.setdefault('image_format', '.png')  # Set an extension for the image. This will be used if saving image.
		kwargs.setdefault('open_folders',
		                  False)  # Indicate if the program should open the folder where files are stored.
		return kwargs

	def _save_figure(self):
		"""
		Save the figure created.
		:param kwargs: Kwargs defined in the method _set_kwargs_defaults.
		:return:
		"""
		SciencePlotting._create_directory(self.images_folder)
		self.fig.savefig(os.path.join(self.images_folder, self.fig_title + self.image_format))
		if self.open_folders:
			SciencePlotting._open_directory(self.images_folder)
		else:
			pass

	def _write_title(self, **kwargs):
		"""
		Write the title of the figure on the plot.
		:param kwargs: The only kwarg to be used is the 'fig_title' one.
		:return:
		"""
		if kwargs.get('write_title'):
			self.fig.suptitle(self.fig_title)
		else:
			pass

	def lineplot(self, data, **kwargs):
		"""
		Create a plot given x and y data. This method has been implemented to offer the user high flexibility on how
		data is introduced. Two main options are available:
			1. In case only one curve is plotted, the data must be introduced as a dictionary, where the first key will
			be the label of the x axis, and its value will be the array containing the x data. The second key will be the
			y label and its value must be the array containing the data on the y axis.
			2. In case more than curve is plotted on the same graph, the user has freedom on how data is introduced.
			In case the user decides to introduce data using iterables (not a string), the user may introduce
			first x data, next y data and finally a string containing the label of the curve. Otherwise, an error will
			raise. In this last case, the user should also introduce x and y labels. Otherwise, default values will be
			used. In case the user introduces a dictionary, the label of the curve will be selected by the key of the
			y data.
		:param data: Data to be plotted. Can be a dictionary or any iterable (except strings).
		:param kwargs: Kwargs accepted on _set_kwargs_defaults method.
		:return:
		"""

		fig, ax = plt.subplots()  # Create axes and figure objects at once.
		self.fig = fig
		self.ax = ax

		kwargs = SciencePlotting._set_kwargs_defaults(**kwargs)
		self.save_mat = kwargs.get('save_mat')
		self.save_fig = kwargs.get('save_fig')
		self.fig_title = kwargs.get('fig_title')
		self.open_folders = kwargs.get('open_folders')

		# Set default values to other variables.
		xlabel = kwargs.get('xlabel')
		ylabel = kwargs.get('ylabel')

		# Process necessary data before proceeding.
		if self.save_fig:  # Process some data if the image will be saved.
			self.image_format = '.' + kwargs.get('image_format').split('.')[0]  # Process image format.
		else:
			pass

		# Check if the title should be written.
		self._write_title(**kwargs)

		# Check how data was introduced by the user.
		if isinstance(data, (_collections_abc.Sequence, np.ndarray, dict)) and not isinstance(data, str):
			# If this condition is fulfilled, the introduced object is an iterable object or a dictionary.
			if isinstance(data, dict):  # Only one curve will be plotted using the dictionary data.
				plot_data = np.array(list(data.values()), dtype=object)

				# Get a list of all the keys of the dictionary.
				labels = np.array(list(data.keys()))

				# Plot according to user inputs.
				if data.get('label') is not None:
					self.ax.plot(plot_data[0], plot_data[1], label=data.get('label'))
					self.ax.legend(title=kwargs.get('legend_title'))
				else:
					self.ax.plot(plot_data[0], plot_data[1])

				# Set the xlabel according to the user input.
				if kwargs.get('xlabel') is None:  # Use a x label from the dictionary.
					self.ax.set(xlabel=labels[0])
					xlabel = labels[0]
				else:  # Use the x label introduced by the user.
					self.ax.set(xlabel=kwargs.get('xlabel'))
				self.ax.set(ylabel=labels[1])
				ylabel = labels[1]

				if self.save_mat:
					self._create_mat_file(data, self.fig_title, open_folder=self.open_folders)
				else:
					pass

			elif isinstance(data, (_collections_abc.Sequence, np.ndarray)):
				for data_iter in data:
					if not isinstance(data_iter, dict):  # Another iterable has been introduced.
						assert len(
							data_iter) == 3, 'When drawing more than 1 curve on the same plot, a label must be given'
						self.ax.plot(data_iter[0], data_iter[1], label=data_iter[-1])
						if self.save_mat:
							self._create_mat_file(data_iter, self.fig_title +
							                      f'_{SciencePlotting._clean_name(data_iter[-1])}',
							                      open_folder=self.open_folders)
					else:  # The data is introduced using dictionaries.
						plot_data = np.array(list(data_iter.values()))
						labels = np.array(list(data_iter.keys()))
						self.ax.plot(plot_data[0], plot_data[1], label=labels[-1])
						if self.save_mat:
							self._create_mat_file(data_iter, self.fig_title +
							                      f'_{SciencePlotting._clean_name(labels[-1])}',
							                      open_folder=self.open_folders)

				# Set x and y labels.
				self.ax.set(ylabel=kwargs.get('ylabel'))
				if kwargs.get('xlabel') is not None:
					self.ax.set(xlabel=kwargs.get('xlabel'))
				else:
					try:
						self.ax.set(xlabel=labels[0])
					except UnboundLocalError:  # In case a non dictionary iterable was introduced.
						self.ax.set(xlabel='x')

				# Set legend title.x
				self.ax.legend(title=kwargs.get('legend_title'))  # Set the legend title.

			self.ax.autoscale(tight=True, axis='x')

			# Set scales for the axis.
			self.ax.set(xscale=kwargs.get('xscale'))
			self.ax.set(yscale=kwargs.get('yscale'))

			# Save the figure if required.
			if self.save_fig:
				self._save_figure()
		else:
			plt.close(self.fig)  # Close the figure to avoid pop up of a white plot.
			raise TypeError(f'Data type {type(data)} is not a valid data type to be plotted.')

	def fplot(self, function, xlimits, **kwargs):
		"""
		Function used to plot given functions given the x limits, based on the built-in Matlab function fplot.
		Similarly to the lineplot method, here the user has also freedom on how the function may be introduced.
			1. In the case where only one function is plotted, the user must introduce a callable function which only
			depends on the x data.
			2. In case the user plots several functions on one plot, the user may introduce data using any iterable
			(except for a string), where the function and its label are introduced together. An example of this would be
			function = [[function1(x), 'label1'], [function2(x), 'label2']], where instead of lists we could also use
			any iterable object, like numpy arrays, tuples, etc.
		:param function: The function/list of functions to be introduced, following the procedures explained above.
		:param xlimits: Array like containing the lower and upper limits on the x axis.
		:param kwargs: All the kwargs accepted by the FPlot class and the ones coming from the SciencePlotting. To get a
		list of all available options for this method, call the static method available_method_inputs from the
		SciencePlotting class and the static method available_inputs() from the FPlot class, or type help(SciencePlotting)
		and help(FPlot).
		:return:
		"""
		fig, ax = plt.subplots()  # Create axes and figure objects at once.
		self.fig = fig
		self.ax = ax
		kwargs = SciencePlotting._set_kwargs_defaults(**kwargs)
		kwargs.setdefault('mat_folder', self.mat_folder)
		self.fig_title = kwargs['fig_title']
		self.open_folders = kwargs.get('open_folders')
		self.save_fig = kwargs.get('save_fig')

		# Do previous checks.
		if not hasattr(function, '__call__') and not isinstance(function, (_collections_abc.Sequence, np.ndarray, str)):
			plt.close(self.fig)
			raise TypeError(
				f'The functions input must be a callable or an iterable containing a callable, not a {type(function)}')
		if not isinstance(xlimits, (_collections_abc.Sequence, np.ndarray)):
			if isinstance(xlimits, str):
				plt.close(self.fig)
				raise TypeError('The xlimits input must be an iterable (and not a STRING).')
			else:
				plt.close(self.fig)
				raise TypeError(f'{type(xlimits)} is not a valid input for xlimits. It must be an iterable object.')

		# Write the title of the plot if necessary
		self._write_title(**kwargs)

		# Set the labels for the axis.
		if kwargs.get('xlabel') is not None:
			self.ax.set(xlabel=kwargs.get('xlabel'))
		else:
			self.ax.set(xlabel='x')
		self.ax.set(ylabel=kwargs.get('ylabel'))

		# Set scales for the axis.
		self.ax.set(xscale=kwargs.get('xscale'))
		self.ax.set(yscale=kwargs.get('yscale'))

		self.ax.autoscale(tight=True, axis='x')

		# Call the plotting class.
		FPlot(self.ax, function, xlimits, **kwargs)

		self._write_title(**kwargs)

		# Save the figure if required.
		if self.save_fig:
			self.image_format = '.' + kwargs.get('image_format').split('.')[0]
			self._save_figure()

		if self.save_mat:
			if self.open_folders:
				self._open_directory(self.mat_folder)
