import setuptools


setuptools.setup(
	name="Electrospray Simulator", # Replace with your own username
	version="0.1",
	author="David Poves Ros",
	author_email="daser1998@gmail.com",
	description="Simulator of electrospray thrusters",
	long_description="Simulation of the liquid meniscus in the ionic regime of capillary electrospray thrusters.",
	long_description_content_type="text/markdown",
	url="https://github.com/DavidPoves/Liquid-meniscus-in-the-ionic-regime-simulator",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: MacOS :: MacOS X",
		"Operating System :: POSIX :: Linux"
	],
	python_requires='>=3.8',
	)
