from setuptools import setup

setup(
	name='Liquid meniscus in the ionic regime ElectrospraySimulator',
	version='2.0',
	packages=['ElectrospraySimulator', 'ElectrospraySimulator.Tools', 'ElectrospraySimulator.Solvers',
	          'ElectrospraySimulator.Main_scripts', 'ElectrospraySimulator.GUI_scripts',
	          'ElectrospraySimulator.Geometry_scripts'],
	url='https://github.com/DavidPoves/TFG-V2',
	license='MIT',
	author='DAVID POVES ROS',
	author_email='daser1998@gmail.com',
	description='Simulation of the liquid meniscus in the ionic regime of capillary electrospray thrusters.'
)
