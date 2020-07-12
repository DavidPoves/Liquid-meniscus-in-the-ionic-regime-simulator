# Import the modules.
import numpy as np
from scipy.signal import savgol_filter
from Tools.fitter_v2 import Fitter
import fenics as fn
import numpy as np


class PostProcessing(object):
    def __init__(self, x_data, y_data):
        return None

    @staticmethod
    def smooth_data(*args, window_length=801, polyorder=3):
        """
        Method to smooth the data using the scipy's function savgol_filter.
        This filter uses the Savitzky-Golay filter technique to smooth the
        data.

        Parameters
        ----------
        data : list/array
            List/array containing the data to be smoothed.
        window_length : int, optional
            The length of the filter window (i.e the number of coefficients).
            This number must be a positive odd integer. The default is 801.
        polyorder : int, optional
            The order of the polynomial used to fit the samples.
            It must be less than window_length. The default is 3.

        Returns
        -------
        numpy.ndarray
            Numpy array containing the smoothed data.

        """
        data = [savgol_filter(data, window_length, polyorder) for data in args]
        size = np.size(data)
        if np.shape(data) == (1, size):
            data = np.transpose(data)
        return data

    @staticmethod
    def load_lmfit_models():
        avail_models = ['GaussianModel', 'LorentzianModel',
                        'SplitLorentzianModel', 'VoigtModel',
                        'PseudoVoigtModel', 'MoffatModel', 'Pearson7Model',
                        'StudentsTModel', 'BreitWignerModel', 'LognormalModel',
                        'DampedOscillatorModel',
                        'DampedHarmonicOscillatorModel',
                        'ExponentialGaussianModel', 'SkewedGaussianModel',
                        'SkewedVoigtModel', 'ThermalDistributionModel',
                        'DoniachModel', 'ConstantModel', 'LinearModel',
                        'QuadraticModel', 'PolynomialModel', 'StepModel',
                        'RectangleModel', 'ExponentialModel', 'PowerLawModel']
        return avail_models

    @staticmethod
    def load_lmfit_model(model):
        """
        Load a fitting model from lmfit library (pip install lmfit).

        Parameters
        ----------
        model : str
            Name of the lmfit model, according to the library docs. A list of
            available models can be checked from the
            PostProcessing.load_lmfit_models() method.

        Returns
        -------
        klass : class
            Class object of the model introduced.

        """
        # Check if the introduced model is available.
        avail_models = PostProcessing.load_lmfit_models()
        assert model in avail_models, f'Model {model} is not available. Check available methods using PostProcessing.load_lmfit_models()'

        # Import the model.
        mod = __import__('lmfit.models', fromlist=[model])
        klass = getattr(mod, model)

        return klass

    @staticmethod
    def determine_best_fit_model_Fitter(x_data, y_data, **kwargs):
        """
        Determine the best statistical model to fit the given data. These
        models are extracted from scipy.stats module. Fitter is a wrapper of
        these models. See Fitter doc for more information about this module.

        Parameters
        ----------
        x_data : array like
            Array of the x data.
        y_data : array like
            Array of the y data (data to be fitted).
        **kwargs :
            All kwargs accepted by the Fitter module. See documentation for
            more information.

        Returns
        -------
        best : dict
            Dictionary containing the model and the parameters to fit the given
            data.

        """
        f = Fitter(x_data, y_data, **kwargs)
        f.fit()
        best = f.get_best()
        f.summary()
        return best

    @staticmethod
    def determine_best_fit_model_lmfit(x_data, y_data, combinations=2,
                                       report=True, **kwargs):
        """
        Determine the best model to fit the given data from the models availa-
        ble from the lmfit library.

        Parameters
        ----------
        x_data : array like
            Array of the x array.
        y_data : array like
            Array of the y data (data to be fitted).
        combinations : int, optional
            Combination of models to be checked. At the moment, only one model
            is checked. The default is 2.
        report : bool, optional
            Show results of the fitting process. The default is True.
        **kwargs :
            Available kwargs are:
                - max_pol_degree: This kwarg is used when checking the polyno-
                    mial model. This kwarg will limit the maximum degree of the
                    polynomial to be checked. Optional, default is 5.
                - show_top: This kwarg is only useful when report=True. This
                    kwarg will limit the top models to be shown. For example,
                    if show_top=5, the report will show the 5 models that
                    better fit the given data. Optional, default is 5.

        Returns
        -------
        None.

        """
        avail_models = PostProcessing.load_lmfit_models()

        # Handle kwargs.
        kwargs.setdefault('max_pol_degree', 5)
        kwargs.setdefault('show_top', 5)
        if report:
            assert isinstance(kwargs.get('show_top'), int), f'show_type must be an integer.'
            show_top = kwargs.get('show_top')

        # First, we need to check the model that better fits the given data.
        results_dict = {}
        for model in avail_models:
            model_class = PostProcessing.load_lmfit_model(model)
            if model == 'PolynomialModel':
                polynomial_comparison = {}
                for i in range(1, kwargs.get('max_pol_degree')+1):
                    key = 'PolynomialModel_Degree' + str(i)
                    trial_model = model_class(degree=i)
                    params = trial_model.make_params()
                    try:
                        fit = trial_model.fit(y_data, params, x=x_data)
                        polynomial_comparison[key] = fit.redchi
                    except ValueError:
                        polynomial_comparison[key] = np.inf
                # Get the degree that better fits the data.
                min_key = min(polynomial_comparison,
                              key=polynomial_comparison.get)
                min_redchi = polynomial_comparison[min_key]
                results_dict[min_key] = min_redchi
            else:
                trial_model = model_class()
                params = trial_model.make_params()
                try:
                    fit = trial_model.fit(y_data, params, x=x_data)
                    results_dict[model] = fit.redchi
                except ValueError:
                    results_dict[model] = np.inf

        # Once all the models have been checked, sort the dictionary by redchi.
        results_dict = {k: v for k, v in sorted(results_dict.items(),
                                                key=lambda item: abs(1-abs(item[1])))}

        # Print out the results in case it is requested.
        if report:
            print("\n-----------------------------------------------------------------------------------\n")
            print(f"Showing the top {show_top} models that better fit the given data (the closer to one the better):\n")
            counter = 1
            for key, value in results_dict.items():
                if counter <= show_top:
                    print(f'Model: {key}  --> reduced chi-square = {value}')
                    counter += 1
            print("\n-----------------------------------------------------------------------------------\n")

    @staticmethod
    def fit_data(x_data, y_data, models, report=True, **kwargs):
        """
        Fit the data for a given model or combination of models.

        Parameters
        ----------
        x_data : array like
            Array of the x data.
        y_data : array like
            Array of the y data (data to be fitted).
        models : array like.
            Array containing the name of the model(s) to be used during the
            fitting process. They must be introduced as strings.
        report : bool, optional
            Bool to indicate if a report of the results should be shown.
            The default is True.
        **kwargs :
            Available kwargs are:
                - degree: In case of using a polynomial model, a degree must be
                    introduced.

        Returns
        -------
        function
            Function of the model(s) that fits the data. It can be used by
            simply calling fit_fun(x_data). Fitting parameters are already
            introduced.

        """
        # First, we need to get the model selected by the user.
        avail_models = PostProcessing.load_lmfit_models()
        models_list = []
        for model in models:
            assert isinstance(model, str), f'Models must be introduced as strings, not as {type(model)}.'
            assert model in avail_models, f'Model {model} is not available. Check available methods using PostProcessing.load_lmfit_models()'
            mod = __import__('lmfit.models', fromlist=[model])
            klass = getattr(mod, model)
            if model == 'PolynomialModel':
                assert kwargs.get('degree') is not None, f'If you use a PolynomialModel, you must introduce a degree.'
                models_list.append(klass(degree=kwargs.get('degree')))
            else:
                prefix = 'model' + str(len(models_list)+1)  # To avoid overlapping of model parameters.
                models_list.append(klass(prefix=prefix))
        for i in range(0, len(models_list)):
            if i == 0:
                user_model = models_list[i]
            else:
                user_model += models_list[i]
        params = user_model.make_params()
        result = user_model.fit(y_data, params, x=x_data)

        if report:
            print("\n------------------------------------------------------\n")
            print("                Fitting results                  ")
            print(result.fit_report())
            print("\n------------------------------------------------------\n")
            result.plot()
        """
        We need to retrieve the function corresponding to the created model,
        independently from the number of input models.
        """
        # Redefine the parameters, so we can obtain the ones corresponding to
        # the best fit.
        params = result.params

        # Define the function to be returned to the user.
        def fit_fun(x):
            return user_model.eval(params=params, x=x)

        return fit_fun

    @staticmethod
    def extract_from_function(fun, coords):
        """
        Extract the values of a given function at the desired coordinates.

        Parameters
        ----------
        fun : dolfin.function.function.Function
            Dolfin/FEniCS function containing the information to be evaluated.
        coords : array like
            List/array contaning the coodinates (in this case, of the points
            forming the interface). This list/array must have the following
            form: coords = [r_coords, z_coords], where r_coords and z_coords
            are the lists/arrays containing the radial and axial coordinates,
            respectively.

        Returns
        -------
        fun_arr : numpy.ndarray
            Numpy array containing the desired data.

        """
        # Check if the coordinates have the proper type.
        assert isinstance(coords, (list, np.ndarray)), f'Coordinates must be in an array like object, not a {type(coords)}.'

        # Extract the r and z coordinates.
        r_coords = coords[0]
        z_coords = coords[1]
        zip_coords = zip(r_coords, z_coords)

        fun_arr = []
        for r, z in zip_coords:
            fun_arr.append(fun([r, z]))
        return np.array(fun_arr)

    @staticmethod
    def get_midpoints_from_boundary(boundaries, boundary_id):
        """
        Get the midpoints from the facets defining the given boundary.

        Parameters
        ----------
        boundaries : dolfin.cpp.mesh.MeshFunctionSizet
            Dolfin/FEniCS object containing the information regarding the
            boundaries of the mesh.
        boundary_id : int
            Boundary identification number.

        Returns
        -------
        r_mids : numpy.ndarray
            Numpy array containing the r coordinates of the middle points.
        z_mids : numpy.ndarray
            Numpy array containing the z coordinates of the middle points.

        """
        r_mids = np.array([])  # Preallocate the coordinates array.
        z_mids = np.array([])  # Preallocate the coordinates array.

        # Get the midpoints of the meniscus' facets.
        interface_facets = fn.SubsetIterator(boundaries, boundary_id)
        for facet in interface_facets:
            r_mids = np.append(r_mids, facet.midpoint()[0])
            z_mids = np.append(z_mids, facet.midpoint()[1])

        # Sort the r coordinates of the midpoints in ascending order.
        r_mids = np.sort(r_mids)

        # Sort the z coordinates in descending order.
        z_mids = np.sort(z_mids)[::-1]

        return r_mids, z_mids

    @staticmethod
    def get_nodepoints_from_boundary(mesh, boundaries, boundary_id):

        # Define an auxiliary Function Space.
        V = fn.FunctionSpace(mesh, 'Lagrange', 1)

        # Get the dimension of the auxiliary Function Space.
        F = V.dim()

        # Generate a map of the degrees of freedom (=nodes for this case).
        dofmap = V.dofmap()
        dofs = dofmap.dofs()

        # Apply a Dirichlet BC to a function to get nodes where the bc is applied.
        u = fn.Function(V)
        bc = fn.DirichletBC(V, fn.Constant(1.0), boundaries, 8)
        bc.apply(u.vector())
        dofs_bc = list(np.where(u.vector()[:] == 1.0))

        dofs_x = V.tabulate_dof_coordinates().reshape(F, mesh.topology().dim())

        coords_r = []
        coords_z = []

        # Get the coordinates of the nodes on the meniscus.
        for dof, coord in zip(dofs, dofs_x):
            if dof in dofs_bc[0]:
                coords_r.append(coord[0])
                coords_z.append(coord[1])
        coords_r = np.sort(coords_r)
        coords_z = np.sort(coords_z)[::-1]

        return coords_r, coords_z

    @staticmethod
    def create_FEniCSExpression_from_function(fun):
        """
        Create a FEniCS Expression from an user-defined function.

        Parameters
        ----------
        fun : function(x)
            Function to be defined as an Expression. This function should only
            be a function of x. If more args are required, the code of this
            method should be modified by changing fun(x[0]) by
            fun(x[0], args...)

        Returns
        -------
        dolfin.function.expression.UserExpression
            FEniCS User Expression object containing the introduced function.

        """
        class FEniCSExpression(fn.UserExpression):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def eval(self, values, x):
                values[0] = fun(x[0])

            def value_shape(self):
                return ()

        expression = FEniCSExpression()
        return expression
