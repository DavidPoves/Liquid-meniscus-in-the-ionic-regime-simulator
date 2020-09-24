import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

import os
import numpy as np

from Tools.GMSH_Interface import GMSHInterface
from Tools.CreateMesh import str_2_num, write_mesh
from Tools.ToolTip_creator import CreateToolTip
import Tools.PredefinedFuns as PreFuns


class MainMenu(tk.Frame):
	def __init__(self, master=None):
		"""
		Initialize the MainMenu class. This will launch the GUI.
		Args:
			master: No user input required. Default is None.
		"""
		tk.Frame.__init__(self, master)

		# Create the options buttons.
		load_button = tk.Button(master, text='Load Geometry/Mesh', command=lambda: self.load_geometry_mesh(master))
		load_button.grid(row=1, column=0, padx=10, pady=10)
		load_button.configure(foreground='BLACK', activeforeground='BLACK')
		CreateToolTip(load_button, 'Load a compatible file.')

		create_button = tk.Button(master, text='Create a new geometry',
		                          command=lambda: self.create_geometry(master))
		create_button.grid(row=1, column=2, padx=10, pady=10)
		create_button.configure(foreground='BLACK', activeforeground='BLACK')
		CreateToolTip(create_button, 'Create a new geometry from scratch.')
		self.geom_data = None
		self.msh_filename = ''

	def load_geometry_mesh(self, master):
		"""
		In case the user decides to load a file, this method will load the selected file. It will launch a dialog where
		the user may choose the file. Accepted extensions are .geo, .msh and .xml.
		Args:
			master: Master window of the GUI.

		Returns:

		"""
		ftypes = [('Dolfin Mesh File', '*.xml'), ('GMSH Geometry File', '*geo'), ('GMSH Mesh File', '*.msh'),
		          ('All files', '*')]
		filename = filedialog.askopenfilename(initialdir=os.getcwd(), filetypes=ftypes)
		assert isinstance(filename, str), 'Select a proper file.'
		if filename == '':
			raise NameError('No file was selected. Stopping execution')
		if filename.split('.')[-1] == 'geo':
			self.msh_filename = write_mesh(filename)
		elif filename.split('.')[-1] == 'msh' or filename.split('.')[-1] == 'xml':
			self.msh_filename = filename
		label = tk.Label(master, text='File was properly loaded. You can now close this window.', justify='center')
		label.grid(row=2, column=1)

	def create_geometry(self, master):
		self.geom_data = GeometryGeneration(master, self)


class GeometryGeneration(tk.Frame):

	def __init__(self, master1, main):
		"""
		Initialize the GeometryGeneration class, which will contain all the methods and attributes required to generate
		a geometry. When initialized, a GUI will pop up, where the user will be able to personalize all the geometry
		options to generate the desired .geo file.
		Args:
			master1: master window.
			main: main menu object
		"""
		master2 = tk.Tk()
		master2.title('Create a new Geometry')
		tk.Frame.__init__(self, master2)
		self.master2 = master2
		self.finish = False
		self.msh_filename = ''

		# Create the Labels of the inputs.
		tk.Label(master2, text='Select an option for the interface expression').grid(row=0, column=0)
		tk.Label(master2, text='Expression z=f(r)').grid(row=1, column=0)
		tk.Label(master2, text='Expression for r').grid(row=2, column=0)
		tk.Label(master2, text='Expression for z').grid(row=3, column=0)
		tk.Label(master2, text='Number of points:').grid(row=4, column=0)
		tk.Label(master2, text='Initial independent parameter coordinate:').grid(row=5, column=0)
		tk.Label(master2, text='Final independent parameter coordinate:').grid(row=6, column=0)

		# Create the string variables that will store the inputs of the user.
		self.user_choice = tk.StringVar(master2)
		self.user_choice.set('Select an option...')
		self.default_choice = self.user_choice.get()
		self.z_of_r = tk.StringVar(master2)
		self.r_fun = tk.StringVar(master2)
		self.z_fun = tk.StringVar(master2)
		self.number_points = tk.StringVar(master2)
		self.number_points.set('200')
		self.initial_ind_coord = tk.StringVar(master2)
		self.initial_ind_coord.set('0')
		self.final_ind_coord = tk.StringVar(master2)
		self.final_ind_coord.set('1')
		self.degrees_var = tk.BooleanVar(master2)
		self.degrees = False
		self.angle_unit = 'radians'
		self.base_data = None

		# Create the option menu.
		self.options_dict = {'z = f(r)': 'z = f(r)', 'Expressions for r and z': 'Expressions for r and z',
		                     'Predefined function': 'Predefined function'}
		self.options_list = list(self.options_dict.values())
		option_menu = tk.OptionMenu(master2, self.user_choice, *self.options_dict, command=self.option_fun)
		option_menu.grid(row=0, column=1, padx=10, pady=10)
		option_menu.configure(foreground='BLACK', activeforeground='BLACK')

		# Create an option to introduce the number of points.
		tk.Entry(master=self.master2, textvariable=self.number_points, justify='center').grid(row=4, column=1,
		                                                                                      padx=10, pady=10)

		# Create a button to close the create geometry menu.
		close_but = tk.Button(self.master, text='Save and close.', command=lambda: self.close_fun(master1, main))
		close_but.grid(row=7, column=2, padx=10, pady=10)
		close_but.configure(foreground='BLACK', activeforeground='BLACK')

	def option_fun(self, value):
		self.user_choice.set(value)

		# Call the function controlling the input boxes.
		self.control_boxes()

	def control_boxes(self):
		tk.Entry(master=self.master2, textvariable=self.initial_ind_coord, state=tk.DISABLED, justify='center').grid(
			row=5, column=1, padx=10, pady=10)
		tk.Entry(master=self.master2, textvariable=self.final_ind_coord, justify='center').grid(row=6, column=1,
		                                                                                        padx=10, pady=10)
		degrees_check = tk.Checkbutton(master=self.master2, variable=self.degrees_var, text='Degrees?',
		                               command=self.check_angle_units)
		degrees_check.grid(row=7, column=1, padx=10, pady=10)
		CreateToolTip(degrees_check, 'If any angle is introduced, check this option to set\n'
		                             'degrees as the unit to be used. Otherwise, radians\n'
		                             'will be used.\n'
		                             'Note: Ignore this option if no angles are introduced.')
		if self.user_choice.get() == self.options_list[0]:  # This means that the user want a f(z) option.
			self.z_fun.set('')
			self.r_fun.set('')
			tk.Entry(master=self.master2, textvariable=self.z_of_r, justify='center').grid(row=1, column=1,
			                                                                               padx=10, pady=10)
			tk.Entry(master=self.master2, textvariable=self.r_fun, state=tk.DISABLED, justify='center').grid(
				row=2, column=1, padx=10, pady=10)
			tk.Entry(master=self.master2, textvariable=self.z_fun, state=tk.DISABLED, justify='center').grid(
				row=3, column=1, padx=10, pady=10)
		elif self.user_choice.get() == self.options_list[-1]:
			fun_data = PredefinedFunctions(self)
			if fun_data.user_fun.get() == 'Half Taylor Cone':
				tk.Entry(master=self.master2, textvariable=self.z_of_r, state=tk.DISABLED, justify='center').grid(
					row=1, column=1, padx=10, pady=10)
				self.r_fun.set('((1-2*s)*1)/(1-2*s*(1-s)*(1-20))')
				self.z_fun.set('(2*(1-s)*s*20*(1/tan(49.3))*1)/(1-2*s*(1-s)*(1-20))')
				tk.Entry(master=self.master2, textvariable=self.r_fun, justify='center').grid(row=2, column=1,
				                                                                              padx=10, pady=10)
				tk.Entry(master=self.master2, textvariable=self.z_fun, justify='center').grid(row=3, column=1,
				                                                                              padx=10, pady=10)
				self.degrees_var.set(True)
		else:
			tk.Entry(master=self.master2, textvariable=self.z_of_r, state=tk.DISABLED, justify='center').grid(
				row=1, column=1, padx=10, pady=10)
			tk.Entry(master=self.master2, textvariable=self.r_fun, justify='center').grid(row=2, column=1,
			                                                                              padx=10, pady=10)
			tk.Entry(master=self.master2, textvariable=self.z_fun, justify='center').grid(row=3, column=1,
			                                                                              padx=10, pady=10)

	def check_angle_units(self):
		"""
		This function checks the option chosen by the user on the Degrees? checkbox from the GUI.
		Returns:

		"""
		if self.degrees_var.get():
			self.degrees = True
			self.angle_unit = 'degrees'
		else:
			self.degrees = False
			self.angle_unit = 'radians'

	def close_fun(self, master, main):
		# Check that inputs are correct.
		if self.user_choice.get() == self.options_list[0] and self.z_of_r.get() == '':
			messagebox.showwarning(title='Error', message='No function was introduced, and it cannot be left blank.\n'
			                                              'Introduce a valid function.')
			return
		elif self.user_choice.get() == self.options_list[1]:
			if self.r_fun.get() == '' or self.z_fun.get() == '':
				messagebox.showwarning(title='Error', message='One of the functions was not introduced.\n'
				                                              'Introduce a valid function.')
				return
		elif self.user_choice.get() == self.default_choice:
			messagebox.showwarning(title='Error', message='Please, select an option before proceeding.')
			return

		self.master2.destroy()
		label = tk.Label(master, text='File was properly loaded. You can now close this window.', justify='center')
		label.grid(row=2, column=1)
		master.destroy()

		# Generate the .geo file from the given data.
		self.geo_gen = GMSHInterface()
		num_points = int(self.number_points.get())
		initial_ind_coord = str_2_num(self.initial_ind_coord.get())
		final_ind_coord = str_2_num(self.final_ind_coord.get())
		self.base_data = np.linspace(initial_ind_coord, final_ind_coord, num_points)
		if self.z_of_r.get() != '':
			self.geo_gen.geometry_generator(interface_fun=self.z_of_r.get(), r=self.base_data)
		elif self.z_fun.get() is not None and self.r_fun.get() is not None:
			self.geo_gen.geometry_generator(interface_fun_r=self.r_fun.get(), interface_fun_z=self.z_fun.get(),
			                                independent_param=self.base_data, angle_unit=self.angle_unit)
		self.msh_filename = self.geo_gen.mesh_generation_GUI()
		main.msh_filename = self.msh_filename


class PredefinedFunctions(tk.Frame):
	def __init__(self, input_data):
		masterPlot = tk.Tk()
		masterPlot.title('Browse predefined functions.')
		tk.Frame.__init__(self, masterPlot)
		self.masterPlot = masterPlot
		self.geo_input = input_data
		self.fig_pos = 0
		self.fig = None
		self.canvas = None
		self.toolbar = None

		tk.Label(master=masterPlot, text='Select a predefined function.').pack()

		# Create an option Menu.
		predef_funs_show = {'Half Taylor Cone': 'Taylor Cone', 'Cosine Function': 'Cosine Function',
		                    'Parabolic Function': 'Parabolic Function', 'Straight Line': 'Straight Line'}
		self.predef_funs = {'Half Taylor Cone': PreFuns.TaylorCone,
		                    'Cosine Function': PreFuns.CosineFunction,
		                    'Parabolic Function': PreFuns.ParabolicFunction,
		                    'Straight Line': PreFuns.StraightLine
		                    }
		self.user_fun = tk.StringVar()
		self.user_fun.set('Select a function.')
		self.default_user_fun = self.user_fun.get()

		opts_menu = tk.OptionMenu(self.masterPlot, self.user_fun, *predef_funs_show, command=self.plot_option)
		opts_menu.pack()
		opts_menu.configure(foreground='BLACK', activeforeground='BLACK')

		close_but = tk.Button(self.masterPlot, text='Save choice', command=self.save)
		close_but.pack()
		close_but.configure(foreground='BLACK', activeforeground='BLACK')

	def plot_option(self, value):
		if self.user_fun.get() == self.default_user_fun:
			messagebox.showwarning(title='Select a function', text='Please, select a function before proceeding.')
		else:
			if self.fig_pos == 0:
				self.fig = Figure(figsize=(5, 4), dpi=100)
			else:
				# Eliminate the previous figure to avoid overlapping.
				self.fig.clf()
				self.canvas.get_tk_widget().destroy()
				self.toolbar.destroy()
			self.plot()

	def plot(self):
		if self.user_fun.get() == 'Half Taylor Cone':
			var = np.linspace(str_2_num(self.geo_input.initial_ind_coord.get()),
			                  str_2_num(self.geo_input.final_ind_coord.get())/2,
			                  int(str_2_num(self.geo_input.number_points.get())))
		else:
			var = np.linspace(str_2_num(self.geo_input.initial_ind_coord.get()),
			                  str_2_num(self.geo_input.final_ind_coord.get()),
			                  int(str_2_num(self.geo_input.number_points.get())))
		r, z = self.predef_funs.get(self.user_fun.get())(var)
		self.fig_pos += 1
		ax = self.fig.add_subplot()
		ax.plot(r, z)
		self.fig.suptitle(self.user_fun.get())

		self.canvas = FigureCanvasTkAgg(self.fig, master=self.masterPlot)  # A tk.DrawingArea.
		self.canvas.draw()
		self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

		self.toolbar = NavigationToolbar2Tk(self.canvas, self.masterPlot)
		self.toolbar.update()
		self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

	def update_degrees_units(self):
		self.geo_input.check_angle_units()
		degrees_check = tk.Checkbutton(master=self.geo_input.master2, variable=self.geo_input.degrees_var,
		                               text='Degrees?', state=tk.DISABLED,
		                               command=self.geo_input.check_angle_units)
		degrees_check.grid(row=7, column=1, padx=10, pady=10)
		CreateToolTip(degrees_check, 'If any angle is introduced, check this option to set\n'
		                             'degrees as the unit to be used. Otherwise, radians\n'
		                             'will be used.\n'
		                             'Note: Ignore this option if no angles are introduced.')

	def save(self):
		# Load the chosen function into the Geometry menu.
		if self.user_fun.get() == 'Half Taylor Cone':
			self.geo_input.r_fun.set('((1-2*s)*1)/(1-2*s*(1-s)*(1-20))')
			self.geo_input.z_fun.set('(2*(1-s)*s*20*(1/tan(49.3))*1)/(1-2*s*(1-s)*(1-20))')
			self.geo_input.final_ind_coord.set('0.5')
			self.geo_input.degrees_var.set(True)
			tk.Entry(master=self.geo_input.master2, textvariable=self.geo_input.r_fun, justify='center').grid(
				row=2, column=1, padx=10, pady=10)
			tk.Entry(master=self.geo_input.master2, textvariable=self.geo_input.z_fun, justify='center').grid(
				row=3, column=1, padx=10, pady=10)
			tk.Entry(master=self.geo_input.master2, textvariable=self.geo_input.z_of_r, state=tk.DISABLED,
			         justify='center').grid(row=1, column=1, padx=10, pady=10)
			self.geo_input.degrees_var.set(True)
			self.update_degrees_units()
		elif self.user_fun.get() == 'Cosine Function':
			self.geo_input.z_of_r.set('0.5*cos(PI/2 * r)')
			tk.Entry(master=self.geo_input.master2, textvariable=self.geo_input.r_fun, state=tk.DISABLED,
			         justify='center').grid(row=2, column=1, padx=10, pady=10)
			tk.Entry(master=self.geo_input.master2, textvariable=self.geo_input.z_fun, state=tk.DISABLED,
			         justify='center').grid(row=3, column=1, padx=10, pady=10)
			tk.Entry(master=self.geo_input.master2, textvariable=self.geo_input.z_of_r, state=tk.NORMAL,
			         justify='center').grid(row=1, column=1, padx=10, pady=10)
			self.geo_input.degrees_var.set(False)
			self.update_degrees_units()
		elif self.user_fun.get() == 'Parabolic Function' or self.user_fun.get() == 'Straight Line':
			tk.Entry(master=self.geo_input.master2, textvariable=self.geo_input.r_fun, state=tk.DISABLED,
			         justify='center').grid(row=2, column=1, padx=10, pady=10)
			tk.Entry(master=self.geo_input.master2, textvariable=self.geo_input.z_fun, state=tk.DISABLED,
			         justify='center').grid(row=3, column=1, padx=10, pady=10)
			if self.user_fun.get() == 'Parabolic Function':
				a = -0.5 / (1 - 0) ** 2
				self.geo_input.z_of_r.set(f'{str(a)}*(r-0)^2 + 0.5')
			if self.user_fun.get() == 'Straight Line':
				self.geo_input.z_of_r.set('0.5*(1-r)')
			tk.Entry(master=self.geo_input.master2, textvariable=self.geo_input.z_of_r, state=tk.NORMAL,
			         justify='center').grid(row=1, column=1, padx=10, pady=10)

		self.masterPlot.destroy()


def run_main_menu():
	root = tk.Tk()
	root.title('Main Menu: Selection of the geometry.')
	app = MainMenu(root)
	root.mainloop()
	return app


if __name__ == '__main__':
	app = run_main_menu()
