import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import os
import numpy as np
import re

from Tools.GMSH_Interface import GMSHInterface
from Tools.CreateMesh import str_2_num
from Tools.ToolTip_creator import CreateToolTip


class MainMenu(tk.Frame):
	def __init__(self, master=None):
		tk.Frame.__init__(self, master)

		# Create the options buttons.
		load_button = tk.Button(master, text='Load Geometry/Mesh', command=lambda: MainMenu.load_geometry_mesh(master))
		load_button.grid(row=1, column=0, padx=10, pady=10)
		load_button.configure(foreground='BLACK', activeforeground='BLACK')
		CreateToolTip(load_button, 'Load a compatible file.')

		create_button = tk.Button(master, text='Create a new geometry',
		                          command=lambda: self.create_geometry(master))
		create_button.grid(row=1, column=2, padx=10, pady=10)
		create_button.configure(foreground='BLACK', activeforeground='BLACK')
		CreateToolTip(create_button, 'Create a new geometry from scratch.')
		self.geom_data = None

	@staticmethod
	def load_geometry_mesh(master):
		ftypes = [('Dolfin Mesh File', '*.xml'), ('GMSH Geometry File', '*geo'), ('GMSH Mesh File', '*.msh'),
		          ('All files', '*')]
		filename = filedialog.askopenfilename(initialdir=os.getcwd(), filetypes=ftypes)
		assert isinstance(filename, str), 'Select a proper file.'
		if filename == '':
			raise NameError('No file was selected. Stopping execution')
		label = tk.Label(master, text='File was properly loaded. You can now close this window.', justify='center')
		label.grid(row=2, column=1)

	def create_geometry(self, master):
		self.geom_data = GeometryGeneration(master)


class GeometryGeneration(tk.Frame):

	def __init__(self, master1):
		master2 = tk.Tk()
		master2.title('Create a new Geometry')
		tk.Frame.__init__(self, master2)
		self.master2 = master2
		self.finish = False

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

		# Create the option menu.
		self.options_dict = {'z = f(r)': 'z = f(r)', 'Expressions for r and z': 'Expressions for r and z'}
		self.options_list = list(self.options_dict.values())
		option_menu = tk.OptionMenu(master2, self.user_choice, *self.options_dict, command=self.option_fun)
		option_menu.grid(row=0, column=1, padx=10, pady=10)
		option_menu.configure(foreground='BLACK', activeforeground='BLACK')

		# Create an option to introduce the number of points.
		tk.Entry(master=self.master2, textvariable=self.number_points, justify='center').grid(row=4, column=1,
		                                                                                      padx=10, pady=10)

		# Create a button to close the create geometry menu.
		close_but = tk.Button(self.master, text='Save and close.', command=lambda: self.close_fun(master1))
		close_but.grid(row=7, column=2, padx=10, pady=10)
		close_but.configure(foreground='BLACK', activeforeground='BLACK')

	def option_fun(self, value):
		self.user_choice.set(value)

		# Call the function controlling the input boxes.
		self.control_boxes()

	def control_boxes(self):
		tk.Entry(master=self.master2, textvariable=self.initial_ind_coord, state=tk.DISABLED,justify='center').grid(
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
				row=3, column=1,  padx=10, pady=10)
		else:
			tk.Entry(master=self.master2, textvariable=self.z_of_r, state=tk.DISABLED, justify='center').grid(
				row=1, column=1, padx=10, pady=10)
			tk.Entry(master=self.master2, textvariable=self.r_fun, justify='center').grid(row=2, column=1,
			                                                                              padx=10, pady=10)
			tk.Entry(master=self.master2, textvariable=self.z_fun, justify='center').grid(row=3, column=1,
			                                                                              padx=10, pady=10)

	def check_angle_units(self):
		if self.degrees_var.get():
			self.degrees = True
			self.angle_unit = 'degrees'
		else:
			self.degrees = False
			self.angle_unit = 'radians'

	def close_fun(self, master):
		# Check that inputs are correct.
		if self.user_choice.get() == self.options_list[0] and self.z_of_r.get() == '':
			messagebox.showwarning(title='Error', message='No function was introduced, and it cannot be left blank.\n'
			                                              'Introduce a valid function.')
			return
		elif self.user_choice.get() == self.options_list[1]:
			if self.r_fun.get() == '' or self.z_fun.get() == '':
				messagebox.showwarning(title='Error', message='One of the functions was not introduced.\n'
				                                              'Introduce a valid function.')
		elif self.user_choice.get() == self.default_choice:
			messagebox.showwarning(title='Error', message='Please, select an option before proceeding.')
			return

		self.master2.destroy()
		label = tk.Label(master, text='File was properly loaded. You can now close this window.', justify='center')
		label.grid(row=2, column=1)
		master.destroy()

		# Generate the .geo file from the given data.
		geo_gen = GMSHInterface()
		num_points = int(self.number_points.get())
		initial_ind_coord = str_2_num(self.initial_ind_coord.get())
		final_ind_coord = str_2_num(self.final_ind_coord.get())
		base_data = np.linspace(initial_ind_coord, final_ind_coord, num_points)
		if self.z_of_r.get() != '':
			geo_gen.geometry_generator(interface_fun=self.z_of_r.get(), r=base_data)
		elif self.z_fun.get() is not None and self.r_fun.get() is not None:
			geo_gen.geometry_generator(interface_fun_r=self.r_fun.get(), interface_fun_z=self.z_fun.get(),
			                           independent_param=base_data, angle_unit=self.angle_unit)
		geo_gen.mesh_generation()


def run_main_menu():
	root = tk.Tk()
	root.title('Main Menu: Selection of the geometry.')
	app = MainMenu(root)
	root.mainloop()
	return app


if __name__ == '__main__':
	app = run_main_menu()
