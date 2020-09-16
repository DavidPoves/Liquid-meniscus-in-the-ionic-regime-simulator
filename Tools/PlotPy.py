# Copyright (C) 2020- by David Poves Ros
#
# This file is part of the End of Degree Thesis.
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import subprocess
import platform
import shutil
import distutils.spawn as spwn
import scipy.io as sio


class PlotPy(object):
    """
    This class will contain all the necessary methods to make the plots as
    customizable as possible using the seaborn and matplotlib libraries.
    """

    def __init__(self, **kwargs):
        """
        Initialize the PlotPy class object.
        Args:
            **kwargs: Accepted kwargs are:
                - latex: Boolean to let the class know if latex may be used or not.
                - fontsize: Value of the fontsize of the plot (integer or float).
                - font_properties: Dictionary containing properties of the fonts. This is specially useful when using
                latex. Available font properties are: family, style, fig_title_style, legend_title_style,
                legend_labels_style and axis_labels_style. Family and style properties' options are the ones available
                for Matplotlib. With respect to the rest of the options, the reader may refer to:
                https://www.overleaf.com/learn/latex/font_typefaces and
                https://www.overleaf.com/learn/latex/Font_sizes,_families,_and_styles \n
                - grid_properties: Dictionary whose keys are the options Matplotlib has for the creation of grids. All
                available options can be found at Matplotlib's docs.\n
                - figsize: Tuple of the figure size.
                - background_style: All available background styles can be found at Matplotlib' docs.
                - save_images: Boolean to indicate wether images might be saved or not.
                - images_folder: String of the path of the folder where images may be saved, in case save_images is True
                - extension: String containing the extension of the image to be saved, like '.jpg', or '.png'.
                - save_mat: Boolean indicating if the data should be stored in .mat files.
                - mat_folder: String of the path of the folder where the .mat files may be stored if save_mat is True.

        """
        kwargs.setdefault('latex', False)
        kwargs.setdefault('fontsize', 12)
        kwargs.setdefault('font_properties',
                          {'family': 'serif', 'style': 'Computer Modern Roman'})
        kwargs.setdefault('grid_properties',
                          {'linestyle': "--", 'color': 'black',
                           'linewidth': 0.2})
        kwargs.setdefault('figsize', (12, 7))
        kwargs.setdefault('background_style', 'seaborn-paper')
        kwargs.setdefault('save_images', False)
        kwargs.setdefault('images_folder', os.getcwd() + '/IMAGES')
        kwargs.setdefault('extension', '.jpg')
        kwargs.setdefault('save_mat', False)
        kwargs.setdefault('mat_folder', os.getcwd() + '/MAT_FILES')

        # Handle possible errors.
        self.apply_kwargs(**kwargs)

        inp_keys = list(kwargs.keys())
        for key in inp_keys:
            try:
                _ = kwargs[key]
            except KeyError:
                raise NameError(f"Unrecognized keyword {key}")

        # Load the kwargs into the plot object.
        self.fontproperties = kwargs.get('font_properties')
        self.gridproperties = kwargs.get('grid_properties')

        # Modify Matplotlib settings accordingly.
        plt.rc('font', **{'family': self.fontproperties.get('family'),
                          self.fontproperties.get('family'): [
                              self.fontproperties.get('style')]})
        plt.rc('grid', linestyle=self.gridproperties.get('linestyle'),
               color=self.gridproperties.get('color'),
               linewidth=self.gridproperties.get('linewidth'))
        plt.rc('xtick', labelsize='medium')
        plt.rc('ytick', labelsize='medium')
        plt.rcParams.update({'figure.autolayout': True})
        if self.use_latex:
            # Check if latex is on the system.
            # loc = PlotPy.find_prog('latex')
            # if loc is None:
            #     raise FileNotFoundError('Latex was not found on your system. Check if added to PATH.')
            os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
            plt.rc('text', usetex=self.use_latex)

        # Define other variables.
        self.x_ = list()
        self.y_ = list()
        self.labels_raw_ = list()
        self.labels_ = list()
        self.legend = False
        self.xlabel_ = ''
        self.ylabel_ = ''

    def apply_kwargs(self, **kwargs):
        """
        Change a specific parameter without re-initializing the whole plotting
        object.
        """
        # Handle the change in the use of latex.
        if kwargs.get('latex') is not None and isinstance(kwargs.get('latex'), bool):
            self.use_latex = kwargs.get('latex')
            plt.rc('text', usetex=kwargs.get('latex'))
        elif not isinstance(kwargs.get('latex'), bool) and kwargs.get('latex') is not None:
            raise TypeError(f"The latex indicator should be a bool, not a {type(kwargs['latex'])}")

        # Handle the change of the fontsize.
        if kwargs.get('fontsize') is not None and (isinstance(kwargs.get('fontsize'), int) or isinstance(kwargs.get('fontsize'), float)):
            self.fontsize = kwargs.get('fontsize')
        elif not isinstance(kwargs.get('fontsize'), int) and not isinstance(kwargs.get('fontsize'), float) and kwargs.get('fontsize') is not None:
            raise TypeError(f"The fontsize type should be an integer or float, not a {type(kwargs['fontsize'])}")

        # Handle the change of figure size.
        if kwargs.get('figsize') is not None and (isinstance(kwargs.get('figsize'), tuple) or isinstance(kwargs.get('figsize'), list)):
            # Check the objects inside the tuple:
                for obj in kwargs.get('figsize'):
                    if not isinstance(obj, int) and not isinstance(obj, float):
                        raise TypeError(f"{obj} is not a valid figure size indicator.")
                if isinstance(kwargs.get('figsize'), tuple):
                    self.figsize = kwargs.get('figsize')
                else:
                    self.figsize = tuple(kwargs.get('figsize'))
        elif kwargs.get('figsize') is not None and not (isinstance(kwargs.get('figsize'), tuple) or isinstance(kwargs.get('figsize'), list)):
            raise TypeError(f"The figsize type should be a tuple or a list, not a {type(kwargs['figsize'])}")

        # Handle the save image flag.
        if kwargs.get('save_images') is not None and isinstance(kwargs.get('save_images'), bool):
            self.saveimages = kwargs.get('save_images')
        elif kwargs.get('save_images') is not None and not isinstance(kwargs.get('save_images'), bool):
            raise TypeError(f"The save_images indicator should be a bool, not a {type(kwargs['save_images'])}")

        # Handle the background style of the plot.
        if kwargs.get('background_style') is not None and isinstance(kwargs.get('background_style'), str):
            # We need to check if the specified style is available for the user
            if kwargs.get('background_style') in plt.style.available:
                self.backgroundstyle = kwargs.get('background_style')
                plt.style.use(self.backgroundstyle)
            else:
                raise NotImplementedError(f"The background style {kwargs.get('background_style')} is not available")
        elif kwargs.get('background_style') is not None and not isinstance(kwargs.get('background_style'), str):
            raise TypeError(f"The background style should be a string not a {type(kwargs.get('background_style'))}")

        # Handle the change of the images folder.
        if kwargs.get('images_folder') is not None and isinstance(kwargs.get('images_folder'), str):
            self.images_folder = kwargs.get('images_folder')
        elif kwargs.get('images_folder') is not None and not isinstance(kwargs.get('images_folder'), str):
            raise TypeError(f"The folder path must be a string not a {type(kwargs.get('images_folder'))}")

        # Handle the change of the image's extension.
        if kwargs.get('extension') is not None and isinstance(kwargs.get('extension'), str):
            extensions_avail = ['eps', 'pgf', 'pdf', 'png', 'ps', 'raw', 'rgba', 'svg',
                                'svgz', 'jpeg', 'jpg', 'tif', 'tiff']
            if kwargs.get('extension').split('.')[-1] in extensions_avail:
                self.extension = '.' + kwargs.get('extension').split('.')[-1]
            else:
                raise NotImplementedError(f"The extension {kwargs.get('extension')} is not available.")
        elif kwargs.get('extension') is not None and not isinstance(kwargs.get('extension'), str):
            raise TypeError(f"The extension must be a string not a {type(kwargs.get('extension'))}")

        # Handle the save_mat indicator.
        if kwargs.get('save_mat') is not None and isinstance(kwargs.get('save_mat'), bool):
            self.savemat = kwargs.get('save_mat')
        elif kwargs.get('save_mat') is not None and not isinstance(kwargs.get('save_mat'), bool):
            raise TypeError(f"The save_mat indicator must be a bool not a {type(kwargs.get('images_folder'))}")

        # Handle the mat_folder path.
        if kwargs.get('mat_folder') is not None and isinstance(kwargs.get('mat_folder'), str):
            self.mat_folder = kwargs.get('mat_folder')
        elif kwargs.get('mat_folder') is not None and not isinstance(kwargs.get('mat_folder'), str):
            raise TypeError(f"The folder path must be a string not a {type(kwargs.get('mat_folder'))}")

    def change_init_settings(self, **kwargs):
        self.apply_kwargs(**kwargs)

    @staticmethod
    def find_prog(prog):
        if spwn.find_executable(prog):
            loc = spwn.find_executable(prog)
            return loc
        else:
            return None

    @staticmethod
    def easy_plot(x, y, **kwargs):
        """
        Easily plot the data using the Matplotlib backend.

        Parameters
        ----------
        x: Any Matplotlib compatible format.
            Data on the x axis.
        y: Any Matplotlib compatible format.
            Data on the y axis.
        **kwargs:
            Any kwargs accepted by Matplotlib.

        Returns
        -------
        None.

        """
        plt.plot(x, y, **kwargs)

    def create_pandas_df(self, x, y):
        d = {self.xlabel_: x, self.ylabel_: y}
        return pd.DataFrame(d)

    def save_mat_file(self, x, y, **kwargs):
        """
        Save data as a .mat file, which can be opened Matlab. This file will be saved in the folder which may be
        introduced to the class when initializing the class under the kwarg mat_folder. Otherwise, it will be saved in
        a default folder, called MAT_FILES.
        Args:
            x: x-data array.
            y: y-data array.
            **kwargs: The accepted kwarg is fig_title, which will name the file. It must be a string.

        Returns:

        """
        mat_file_name = ''
        for word in kwargs.get('fig_title').split(' '):
            if word != kwargs.get('fig_title').split(' ')[-1]:
                mat_file_name += word + '_'
            else:
                if self.legend:
                    mat_file_name += word + '_' + self.labels_raw_[self.i].replace(" ", "") + '.mat'
                else:
                    mat_file_name += word + '.mat'
        data_dict = {'x': x, 'y': y}
        sio.savemat(mat_file_name, data_dict)

        # Move the mat file to the corresponding folder.
        files = [i for i in os.listdir(os.getcwd()) if i.endswith(".mat") and os.path.isfile(os.path.join(os.getcwd(), i))]
        if not os.path.isdir(self.mat_folder):
            os.makedirs(self.mat_folder)
        for f in files:
            shutil.move(os.path.join(os.getcwd(), f), self.mat_folder)

    def lineplot(self, data, **kwargs):
        """
        Create a lineplot for given data. To introduce data, one may use arrays, in which tuples will be introduced.
        These tuples will contain the x data, y data and a string with the label of the plot, if necessary. Notice that
        this last element is optional. In case a label is detected, a legend will be automatically displayed on the
        graph. A simple example could be (once the PlotPy class is initialized):
            PlotPy.lineplot([(x_data, y_data)]), when only a plot is necessary.
            PlotPy.lineplot([(x_data_1, y_data_1, label_1), (x_data_2, y_data_2, label_2)]), when several plots on the
            same figure are necessary.
        Other options are explained on the kwargs section of Args.
        Args:
            data: Array containing tuples with the corresponding x and y data and their label for the legend. This last
            element is optional.
            **kwargs: Available kwargs are:
                        - xlabel: String containing the label of the x axis. It should be a raw string in case latex
                        commands are used.
                        - ylabel: Same as xlabel, but for the y axis.
                        - fig_title: String containing the title of the figure. Use raw strings in case latex commands
                        are used.
                        - legend_title: String of the title of the legend of the figure. Use raw strings if latex
                        commands are used.
                        - grid: Boolean indicating if a grid on the figure is to be drawn.
                        - xscale: String containing the scale of the x axis. Available options are the ones availale for
                        Matplotlib.
                        - yscale: Same as xscale, but for the y axis.
                        - scientific: Use scientific notation on the axis info.


        Returns:

        """

        # Assign default values to the kwargs.
        kwargs.setdefault('xlabel', 'x')
        kwargs.setdefault('ylabel', 'y')
        kwargs.setdefault('fig_title', '')
        kwargs.setdefault('legend_title', '')
        kwargs.setdefault('grid', True)
        kwargs.setdefault('xscale', 'linear')
        kwargs.setdefault('yscale', 'linear')
        kwargs.setdefault('scientific', True)

        # Assign to each of the available font styles their respective latex code.
        avail_styles = {'medium': 'textmd', 'bold': 'textbf',
                        'upright': 'textup', 'italic': 'textit',
                        'slanted': 'textsl', 'small_caps': 'textsc'}

        # Extract the information from the data array.
        self.x_ = [tup[0] for tup in data]
        self.y_ = [tup[1] for tup in data]
        try:  # Try finding a label within the tuple.
            labels_raw = [tup[2] for tup in data]
            self.labels_raw_ = labels_raw
            self.labels_ = labels_raw
            try:
                legend_labels_style = self.fontproperties['legend_labels_style']
                if legend_labels_style in avail_styles and self.use_latex:
                    style_code = avail_styles.get(legend_labels_style)
                    self.labels_ = []
                    for label in labels_raw:
                        label_str = '\\' + style_code + '{' + f"{label}" + '}'  # Assign to the label the latex command
                        self.labels_.append(label_str)
                elif legend_labels_style not in avail_styles and self.use_latex:
                    print(f"Unrecognized legend labels style {legend_labels_style}. Using the default one...", flush=True)
            except KeyError:
                pass

            zip_data = zip(self.x_, self.y_, self.labels_)
            self.legend = True  # Automatically display the legend of the figure if a label is found.
        except IndexError:  # In case a label is not found, an IndexError will be caught, disabling the legend.
            zip_data = zip(self.x_, self.y_)
            self.legend = False

        self.xlabel_ = kwargs.get('xlabel')
        self.ylabel_ = kwargs.get('ylabel')
        try:
            axis_labels_style = self.fontproperties['axis_labels_style']
            if axis_labels_style in avail_styles and self.use_latex:
                style_code = avail_styles.get(axis_labels_style)
                self.xlabel_ = '\\' + style_code + '{' + f"{kwargs.get('xlabel')}" + '}'
                self.ylabel_ = '\\' + style_code + '{' + f"{kwargs.get('ylabel')}" + '}'
            elif axis_labels_style not in avail_styles and self.use_latex:
                print(f"Unrecognized axis labels style {axis_labels_style}. Using the default one...", flush=True)
        except KeyError:
            pass

        fig = plt.figure(figsize=self.figsize)
        if kwargs.get('scientific'):
            plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax = plt.gca()
        if self.legend:
            self.i = 0
            for x, y, label in zip_data:
                plt.plot(x, y, label=label)
                if self.savemat:
                    self.save_mat_file(x, y, **kwargs)
                    self.i += 1
            if kwargs.get('legend_title') != '':
                legend_title = kwargs.get('legend_title')
                try:
                    legend_title_style = self.fontproperties['legend_title_style']
                    if legend_title_style in avail_styles and self.use_latex:
                        style_code = avail_styles.get(legend_title_style)
                        legend_title = '\\' + style_code + '{' + f"{kwargs.get('legend_title')}" + '}'
                    elif legend_title_style not in avail_styles and self.use_latex:
                        print(f"Unrecognized legend title style {legend_title_style}. Using the default one...", flush=True)
                        legend_title = kwargs.get('legend_title')
                except KeyError:
                    pass

                l = plt.legend(title=legend_title, fontsize=self.fontsize-2,
                               loc='center left',
                               bbox_to_anchor=(0, 1.02, 1, 0.2),mode='expand',
                               ncol=len(self.labels_))
            else:
                l = plt.legend(fontsize=self.fontsize-2, loc='center left',
                               bbox_to_anchor=(0, 1.02, 1, 0.2),mode='expand',
                               ncol=len(self.labels_))
            l.get_frame().set_linewidth(0.)
            plt.tight_layout()
        else:
            for x, y in zip_data:
                plt.plot(x, y)
                if self.savemat:
                    self.save_mat_file(x, y, **kwargs)
        fig_title = kwargs.get('fig_title')
        if fig_title != '':
            try:
                fig_title_style = self.fontproperties['fig_title_style']
                if fig_title_style in avail_styles and self.use_latex:
                    style_code = avail_styles.get(fig_title_style)
                    fig_title = '\\' + style_code + '{' + f"{kwargs.get('fig_title')}" + '}'
                elif fig_title_style not in avail_styles and self.use_latex:
                    print(f"Unrecognized figure title style {fig_title_style}. Using the default one...", flush=True)
            except KeyError:
                pass
        plt.title(fig_title, fontsize=self.fontsize+2)
        plt.xlabel(self.xlabel_, fontsize=self.fontsize)
        plt.ylabel(self.ylabel_, fontsize=self.fontsize)
        plt.xscale(kwargs.get('xscale'))
        plt.yscale(kwargs.get('yscale'))
        if kwargs.get('grid'):
            plt.grid()
        else:
            pass
        # Show the plot.
        plt.show()
        os.environ["PATH"] += ":/usr/local/bin:/usr/local/bin/gs"

        # Save the image if specified.
        if self.saveimages:
            folder = self.images_folder
            if fig_title != '':
                title_words = kwargs.get('fig_title').split(' ')
                image_title = ''
                for word in title_words:
                    if word == title_words[-1]:
                        image_title += word
                    else:
                        image_title += word + '_'
            else:
                image_title = 'Figure'
            image_title += self.extension
            try:
                plt.savefig(folder + '/' + image_title,
                            format=self.extension.split('.')[-1], dpi=1000)
            except FileNotFoundError:
                os.mkdir(folder)
                plt.savefig(folder + '/' + image_title,
                            format=self.extension.split('.')[-1], dpi=1000)
