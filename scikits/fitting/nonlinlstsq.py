###############################################################################
# Copyright (c) 2007-2018, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

"""Non-linear least-squares fitter.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

from builtins import range
import copy

import numpy as np
import scipy.optimize

from .generic import ScatterFit
from .utils import squash

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  NonLinearLeastSquaresFit
# ----------------------------------------------------------------------------------------------------------------------


class NonLinearLeastSquaresFit(ScatterFit):
    """Fit generic function to data based on non-linear least squares optimisation.

    This fits a function of the form ``y = f(p, x)`` to x-y data, by finding a
    a least-squares estimate for the parameter vector ``p``. The function takes
    an ``x`` array (or scalar) of shape ``D_x = (dx_1, dx_2, ...)`` as input,
    and produces a ``y`` array (or scalar) of shape ``D_y = (dy_1, dy_2, ...)``
    as output. The underlying non-linear least-squares optimiser is typically
    iterative, starting from an initial guess of the parameter vector and
    incrementally reducing the squared fitting error until it converges on a
    *local* minimum of the cost function.

    The function ``f(p, x)`` should be able to operate on sequences of ``x``
    arrays (i.e. should be vectorised). That is, it should accept ``x`` arrays
    of shape (D_x, N) to produce ``y`` arrays of shape (D_y, N). This is the
    way in which x-y data is presented to the :meth:`fit` method. Note that the
    array sequence is concatenated along the *last* dimension (i.e. as
    columns). If it cannot, use the helper function :func:`vectorize_fit_func`
    to wrap the function before passing it to this class.

    If available, the Jacobian of the function, ``J = g(p, x)``, should return
    an array ``J`` of shape (D_y, P), where ``P = len(p)`` is the number of
    function parameters. Each element of this array indicates the derivative of
    the ``i``'th output value with respect to the ``j``'th parameter, evaluated
    at the given ``p`` and ``x``. This function should also be vectorised,
    similar to ``f``, so that an input ``x`` array of shape (D_x, N) produces
    an output ``J`` array of shape (D_y, P, N).

    Parameters
    ----------
    func : function, signature ``y = f(p, x)``
        Generic function to be fit to x-y data (should be vectorised)
    initial_params : array-like, shape (P,)
        Initial guess of function parameter vector *p*
    enabled_params: sequence of int or sequence of bool, shape (P,), optional
        Subset of parameters that will be optimised, identified either by their
        integer indices or by the True values in a boolean mask. The deselected
        parameters are fixed at their initial values. All parameters are
        optimised by default.
    func_jacobian : function, signature ``J = g(p, x)``, optional
        Jacobian of function f, if available, where J has the shape (D_y, P),
        with D_y the normal ``y`` shape returned by f (should be vectorised)
    kwargs : dict, optional
        Additional keyword arguments are passed to underlying SciPy optimiser

    Attributes
    ----------
    params : array of float, shape (P,)
        Final optimal value for parameter vector (starts off as initial value)
    cov_params : array of float, shape (P, P)
        Standard covariance matrix of parameters, only set after :func:`fit`

    Notes
    -----
    This uses the SciPy :func:`scipy.optimize.leastsq` routine to find the
    optimal parameter vector using modified Levenberg-Marquardt optimisation.

    """
    def __init__(self, func, initial_params, enabled_params=None,
                 func_jacobian=None, **kwargs):
        ScatterFit.__init__(self)
        self.func = func
        # Preserve this for repeatability of fits
        # (also ensure it is floating-point to please residuals() function)
        self.initial_params = np.asarray(initial_params).astype(float)
        self.func_jacobian = func_jacobian
        # Extra keyword arguments to optimiser
        self._extra_args = kwargs
        self.params = self.initial_params
        self.enabled_params = np.asarray(
            enabled_params if enabled_params is not None else
            [True] * len(self.params))
        self.cov_params = None

    def fit(self, x, y, std_y=1.0):
        """Fit function to data, using non-linear least-squares optimisation.

        This determines the optimal parameter vector ``p*`` so that the
        function ``y = f(p, x)`` best fits the observed x-y data, in a
        least-squares sense. The x-y data is a sequence of N ``x`` arrays of
        shape D_x and a sequence of N corresponding ``y`` arrays of shape D_y.
        These sequences are concatenated along the *last* dimension (i.e. as
        columns) to form the *x* and *y* arrays.

        Parameters
        ----------
        x : array-like, shape (D_x, N)
            Sequence of input values as columns of a numpy array
        y : array-like, shape (D_y, N)
            Sequence of output values as columns of a numpy array
        std_y : float or array-like, shape (D_y, N), optional
            Measurement error or uncertainty of `y` values, expressed as
            standard deviation in units of `y`

        Returns
        -------
        self : :class:`NonLinearLeastSquaresFit` object
            Reference to self, to allow chaining of method calls

        """
        x, y = np.asarray(x), np.asarray(y)
        # Initialise full set of parameters
        # (subset to be optimised will be inserted into this array before use)
        params = self.initial_params[:]

        # Calculate R = prod(D_y) * N weighted residuals
        # (leastsq will minimise sum(residuals ** 2))
        def residuals(p):
            params[self.enabled_params] = p
            r = (y - self.func(params, x)) / std_y
            return r.ravel()
        # Register Jacobian function if applicable
        if self.func_jacobian is not None:
            # Jacobian (R, P) matrix of function at given p and x values
            # (derivatives along rows)
            def jacobian(p):
                params[self.enabled_params] = p
                # Produce Jacobian of residual - array with shape (D_y, P, N)
                residual_jac = - self.func_jacobian(params, x) / std_y
                # Squash every axis except second-last parameter axis together,
                # to get (R, P) shape
                flatten_axes = list(range(len(residual_jac.shape) - 2)) + [
                    len(residual_jac.shape) - 1]
                ravel_jac = squash(residual_jac, flatten_axes,
                                   move_to_start=True)
                # Jacobian of residuals has shape (R, P)
                return ravel_jac[:, self.enabled_params]
            self._extra_args['Dfun'] = jacobian
        # Optimise, starting from copy of same initial parameter vector
        # for each call of fit (x0 used to be clobbered)
        p, cov_p, infodict, mesg, ier = scipy.optimize.leastsq(
            residuals, self.initial_params[self.enabled_params], full_output=1,
            **self._extra_args)
        # Try to salvage a singular precision matrix by using
        # the pseudo-inverse in this case
        if cov_p is None:
            # The calculation of cov_p is lifted from scipy.optimize.leastsq
            ipvt, fjac = infodict['ipvt'], infodict['fjac']
            perm = np.take(np.eye(len(ipvt)), ipvt - 1, 0)
            R = np.dot(np.triu(fjac.T[:len(ipvt), :]), perm)
            precision_mat, rcond = np.dot(R.T, R), 1e-15
            try:
                cov_p = np.linalg.pinv(precision_mat, rcond)
            except np.linalg.LinAlgError:
                # The standard SVD in NumPy is based on Lapack DGESDD, which is
                # fast but occasionally struggles on pathological matrices,
                # resulting in a LinAlgError (see NumPy ticket #990) - then
                # all bets are off
                cov_p = np.zeros(precision_mat.shape)
            max_var = np.diag(cov_p).max()
            bad_variances = np.diag(cov_p) <= rcond * max_var
            bad_var = max_var / rcond if max_var > 0 else 1e100
            cov_p[bad_variances, bad_variances] = bad_var
        params[self.enabled_params] = p
        self.params = params
        self.cov_params = np.zeros((len(params), len(params)))
        subset = np.ix_(self.enabled_params, self.enabled_params)
        self.cov_params[subset] = cov_p
        return self

    def __call__(self, x):
        """Evaluate fitted function on new data.

        Evaluates the fitted function ``y = f(p*, x)`` on new *x* data.

        Parameters
        ----------
        x : array, shape D_x or (D_x, M)
            Sequence of input values as columns of a numpy array

        Returns
        -------
        y : array, shape D_y or (D_y, M)
            Sequence of output values as columns of a numpy array

        """
        return self.func(self.params, x)

    def __copy__(self):
        """Shallow copy operation."""
        return NonLinearLeastSquaresFit(
            self.func, self.params, self.enabled_params, self.func_jacobian,
            **self._extra_args)

    def __deepcopy__(self, memo):
        """Deep copy operation.

        Don't deepcopy stored functions, as this is not supported in Python 2.4
        (Python 2.5 supports it...).

        Parameters
        ----------
        memo : dict
            Dictionary that caches objects that are already copied

        """
        return NonLinearLeastSquaresFit(
            self.func, copy.deepcopy(self.params, memo),
            copy.deepcopy(self.enabled_params, memo), self.func_jacobian,
            **(copy.deepcopy(self._extra_args, memo)))
