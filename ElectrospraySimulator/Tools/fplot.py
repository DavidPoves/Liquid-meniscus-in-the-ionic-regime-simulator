"""
1D Callable function plotting based on the work done in Matlab with their built in function fplot. This function will
adaptively evaluate the function on the points where critical evaluations are obtained. In this way, functions that are
badly represented with linearly spaced points might be well represented using this function.
Function based on the fplot proposed by Damon McDougall in https://github.com/matplotlib/matplotlib/pull/1143/files.
Updated and modified by David Poves Ros to include:
    - Array of callables compatibility: Now the user may introduce an iterable (not a string) containing lists of the
    functions and their corresponding labels.
    - Implemented the ability to export plot data to .mat files. This capability can be used even when several functions
    are introduced.
    - Arguments handling process has been rewritten to add more flexibility on how the user introduces them. Before, the
    order in which the arguments were introduced mattered. Now, an algorithm has been implemented to smartly select the
    callables based on how the user introduces the data.
    - Update to introduce compatibility with matplotlib 3.3.2.
    - Update the way in which the limits of the y axis were defined. Now, since several curves may be introduced, the
    minimum and maximum limits are chosen so that it fits the curve with the absolute minimum and the one with the
    absolute maximum.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import _collections_abc
from ElectrospraySimulator.Tools.export2mat import export2mat


class FPlot(object):
    def __init__(self, axes, *args, **kwargs):
        """
        Initialize the Fplot class to achieve adaptive plotting capabilities. This is similar to Matlab's fplot.
        :param axes: Axes object of the plot.
        :param fig: Figure object of the plot.
        :param args: These are the callable function and the x-limits of the plot.
        :param kwargs: Accepted kwargs can be checked from the available_inputs method of this class, or by calling
        help(FPlot)
        """
        # Preallocate data.
        self.functions = list()
        self.labels = list()
        self.counter = 0
        self.mn = 1e100  # Set arbitrary minimum to the y limit.
        self.mx = 1e-100  # Set arbitrary maximum to the y limit.

        # Handle kwargs.
        kwargs.setdefault('mat_folder', os.path.join(os.getcwd(), 'MAT FILES'))
        kwargs.setdefault('MeshResolution', 1000)
        kwargs.setdefault('fig_title', 'FPlot-Figure')
        self.n = kwargs.get('MeshResolution')
        self.mat_folder = kwargs.get('mat_folder')
        self.save_mat = kwargs.get('save_mat')
        self.fig_title = kwargs.get('fig_title')

        # Create the mat folder if necessary.
        if self.save_mat:
            FPlot._create_directory(self.mat_folder)

        # Process the arguments introduced by the user.
        self._process_args(*args)

        self.axes = axes
        self.axes.set_autoscale_on(False)

        self.x = np.linspace(self.limits[0], self.limits[1], self.n)

        if self.functions is not None and self.labels is not None:  # Introduce array of callables compatibility.
            for function in self.functions:
                self.f = function
                self._create_plot()
                self.counter += 1
        else:
            self._create_plot()

        # Set the limits based on the absolute minima and maxima.
        self.axes.set_ylim([self.mn, self.mx])

    @staticmethod
    def available_inputs():
        """
        Show all the available options when calling this class.
        - MeshResolution: Set the number of points to be used to plot the function. The adaptive algorithm will use this
        number of points to get the best possible set of points to evaluate the function. Optional, default is 1000.
        - mat_folder: String containing the path where the created .mat files will be saved. Optional, by default a
        folder called MAT FILES will be created on the current working directory.
        - save_mat: Boolean indicating if .mat files containing the data from the plot should be created. Optional,
        default is None.
        - fig_title: String containing the tile of the created figure. This string will be used to define the name of the
        .mat file in case it is generated. Otherwise, this input has no effect. Optional, default is FPlot-Figure

        Returns:
        """
        print(['MeshResolution', 'mat_folder', 'save_mat', 'fig_title'])

    @staticmethod
    def _create_directory(path):
        """Create a directory on a specified path if it does not exist.
        :param path: Path where the folder is supposed to be. If it does not exist, it will be created.
        :return:
        """
        if not os.path.isdir(path):
            os.makedirs(path)

    def _create_plot(self):
        self.f_values = np.asarray([self.f(xi) for xi in self.x])

        if self.labels is not None:  # If labels were introduced, plot them. Valid when multiple functions were used.
            self.fline, = self.axes.plot(self.x, self.f_values, label=self.labels[self.counter])
            self.axes.legend(frameon=False)  # Show the legend.
        else:  # No labels were created
            self.fline, = self.axes.plot(self.x, self.f_values)
        self._process_singularities()
        self.axes.set_xlim([self.x[0], self.x[-1]])
        mn, mx = np.nanmin(self.f_values), np.nanmax(self.f_values)
        self.axes.set_ylim([mn, mx])

        # Check if the obtained values of y limits are suitable for the whole plot.
        self._check_y_limits(mn, mx)

        self.axes.callbacks.connect('xlim_changed', self._update)
        self.axes.callbacks.connect('ylim_changed', self._update)

        # At this step, all the data has been processed. Hence, it is save to export data to .mat files if needed.
        if self.save_mat:
            if self.labels is not None:
                self._save_mat({'x': self.x, 'y': self.f_values}, self.fig_title + f'_{self.labels[self.counter]}')
            else:
                self._save_mat({'x': self.x, 'y': self.f_values}, self.fig_title)

    def _check_y_limits(self, mn, mx):
        # Check if the minimum of a given function is the absolute minimum of all the functions.
        if mn < self.mn:
            self.mn = mn

        # Check if the maximum of a given function is the absolute maximum of all the functions.
        if mx > self.mx:
            self.mx = mx

    def _process_functions_array(self):
        for item in self.f:
            try:
                for item_label in item:  # If a label was introduced.
                    if hasattr(item_label, '__call__'):
                        self.functions.append(item_label)
                    elif isinstance(item_label, str):
                        self.labels.append(item_label)
                    else:
                        raise TypeError(f'Non valid input type {type(item_label)}')
                assert len(self.labels) > 0, 'When introducing several functions, their labels are required.'
            except TypeError:  # No label was introduced.
                if hasattr(item, '__call__'):
                    self.functions.append(item)
                else:
                    raise TypeError()  # To make sure that the object is not callable (maybe limits were introduced).

    def _process_args(self, *args):
        self.f = args[0]
        self.limits = args[1]
        try:  # Support array of functions.
            self._process_functions_array()
        except TypeError:  # Check if the function is callable.
            try:
                assert hasattr(self.f, '__call__'), 'Non valid type of function introduced. It must be callable.'
                assert isinstance(self.limits, (_collections_abc.Sequence, np.ndarray)) and not hasattr(
                    self.limits, '__call__'), 'The limits must be introduced with an iterable object.'
                self.functions = None  # No array of functions was introduced.
                self.labels = None
            except AssertionError:
                self.f = args[1]
                self.limits = args[0]
                try:  # Support array of functions.
                    self._process_functions_array()
                except TypeError:  # Check if the function is callable.
                    assert hasattr(self.f,
                                   '__call__'), 'Non valid type of function introduced. It must be callable.'
                    assert isinstance(self.limits, (_collections_abc.Sequence, np.ndarray)) and not hasattr(
                        self.limits, '__call__'), 'The limits must be introduced with an iterable object.'
                    self.functions = None  # No array of functions was introduced.
                    self.labels = None

    def _update(self, axes):
        # bounds is (l, b, w, h)
        bounds = axes.viewLim.bounds
        self.x = np.linspace(bounds[0], bounds[0] + bounds[2], self.n)
        self.f_values = [self.f(xi) for xi in self.x]
        self._process_singularities()
        self.fline.set_data(self.x, self.f_values)
        self.axes.figure.canvas.draw_idle()

    def _process_singularities(self):
        # Note:  d[i] == f_values[i+1] - f_values[i]
        d = np.diff(self.f_values)

        # 80% is arbitrary.  Perhaps more control could be offered here?
        badness = np.where(d > 0.80 * self.axes.viewLim.bounds[3])[0]

        # We don't draw the singularities
        for b in badness:
            self.f_values[b] = np.nan
            self.f_values[b + 1] = np.nan

    def _save_mat(self, data, filename):
        export2mat(data, filename, self.mat_folder)


def fplot(ax, *args, **kwargs):
    """
    Plot a callable function using adaptive plotting capabilities. This function will detect anomalies, singularities
    and bad plots due to linearly spaced points, which may incur into bad quality graphs.
    :param ax: The figure axis' object.
    :param args: Valid arguments are a callable function and the x limits of the plot.
    :param kwargs: Accepted kwargs are the ones from SciencePlotting. See its docs for a full reference.
    :return: FPlot object containing all the information of the generated plot.
    """
    return FPlot(ax, *args, **kwargs)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    plt.style.use('science')
    functions = [[lambda beta: (1-beta*(1-np.sqrt(0.05)))**2/(1-beta*(1-0.05)), '0.05'],
                 [lambda beta: (1-beta*(1-np.sqrt(0.08)))**2/(1-beta*(1-0.08)), '0.08']]
    fplot(ax, [0, 1], functions, save_mat=True, fig_title='Propulsive Efficiency')
