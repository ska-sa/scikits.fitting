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

"""Polynomial fitters.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

from builtins import range
import numpy as np
import scipy
import scipy.interpolate

from .generic import ScatterFit, NotFittedError
from .linlstsq import LinearLeastSquaresFit
from .utils import offset_scale_mat

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  Polynomial1DFit
# ----------------------------------------------------------------------------------------------------------------------


class Polynomial1DFit(ScatterFit):
    """Fit polynomial to 1-D data.

    This is built on top of :class:`LinearLeastSquaresFit`. It improves on the
    standard NumPy :func:`numpy.polyfit` routine by automatically centring the
    data, handling measurement uncertainty and calculating the resulting
    parameter covariance matrix.

    Parameters
    ----------
    max_degree : int, non-negative
        Maximum polynomial degree to use (automatically reduced if there are
        not enough data points)
    rcond : float, optional
        Relative condition number of fit (smallest singular value that will be
        used to fit polynomial, has sensible default)

    Attributes
    ----------
    poly : array of float, shape (P,)
        Polynomial coefficients (highest order first), set after :func:`fit`
    cov_poly : array of float, shape (P, P)
        Covariance matrix of coefficients, only set after :func:`fit`

    """
    def __init__(self, max_degree, rcond=None):
        ScatterFit.__init__(self)
        self.max_degree = max_degree
        self._lstsq = LinearLeastSquaresFit(rcond)
        # The following attributes are only set after :func:`fit`
        self.poly = None
        self.cov_poly = None
        self._mean = None

    def _regressor(self, x):
        """Form normalised regressor / design matrix from input vector.

        The design matrix is Vandermonde for polynomial regression.

        Parameters
        ----------
        x : array of float, shape (N,)
            Input to function as a numpy array

        Returns
        -------
        X : array of float, shape (P, N)
            Regressor / design matrix to be used in least-squares fit

        """
        return np.vander(x - self._mean, len(self.poly)).T

    def fit(self, x, y, std_y=1.0):
        """Fit polynomial to data.

        Parameters
        ----------
        x : array-like, shape (N,)
            Known input values as a 1-D numpy array or sequence
        y : array-like, shape (N,)
            Known output values as a 1-D numpy array, or sequence
        std_y : float or array-like, shape (N,), optional
            Measurement error or uncertainty of `y` values, expressed as
            standard deviation in units of `y`

        Returns
        -------
        self : :class:`Polynomial1DFit` object
            Reference to self, to allow chaining of method calls

        """
        # Upcast x and y to doubles, to ensure a high enough precision
        # for the polynomial coefficients
        x = np.atleast_1d(np.asarray(x, dtype='double'))
        y = np.atleast_1d(np.asarray(y, dtype='double'))
        # Polynomial fits perform better if input data is centred
        # around origin [see numpy.polyfit help]
        self._mean = x.mean()
        # Reduce polynomial degree if there are not enough points to fit
        # (degree should be < len(x))
        degree = min(self.max_degree, len(x) - 1)
        # Initialise parameter vector, as its length is used
        # to create design matrix of right shape in _regressor
        self.poly = np.zeros(degree + 1)
        # Solve least-squares regression problem
        self._lstsq.fit(self._regressor(x), y, std_y)
        # Convert polynomial (and cov matrix) so that it applies
        # to original unnormalised data
        tfm = offset_scale_mat(len(self.poly), self._mean)
        self.poly = np.dot(tfm, self._lstsq.params)
        self.cov_poly = np.dot(tfm, np.dot(self._lstsq.cov_params, tfm.T))
        return self

    def __call__(self, x, full_output=False):
        """Evaluate polynomial on new data.

        Parameters
        ----------
        x : array-like of float, shape (M,)
            Input to function as a 1-D numpy array, or sequence
        full_output : {False, True}, optional
            True if output uncertainty should also be returned

        Returns
        -------
        y : array of float, shape (M,)
            Output of function as a 1-D numpy array
        std_y : array of float, shape (M,), optional
            Uncertainty of function output, expressed as standard deviation

        """
        x = np.atleast_1d(np.asarray(x))
        if (self.poly is None) or (self._mean is None):
            raise NotFittedError("Polynomial not fitted to data yet - "
                                 "first call .fit method")
        return self._lstsq(self._regressor(x), full_output)

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  Polynomial2DFit
# ----------------------------------------------------------------------------------------------------------------------


class Polynomial2DFit(ScatterFit):
    """Fit polynomial to 2-D data.

    This models the one-dimensional (scalar) `y` data as a polynomial function
    of the two-dimensional (vector) `x` data. The 2-D polynomial has
    P = (degrees[0] + 1) * (degrees[1] + 1) coefficients. This fitter is built
    on top of :class:`LinearLeastSquaresFit`.

    Parameters
    ----------
    degrees : list of 2 ints
        Non-negative polynomial degree to use for each dimension of *x*
    rcond : float, optional
        Relative condition number of fit (smallest singular value that will be
        used to fit polynomial, has sensible default)

    Attributes
    ----------
    poly : array of float, shape (P,)
        Polynomial coefficients (highest order first), set after :func:`fit`
    cov_poly : array of float, shape (P, P)
        Covariance matrix of coefficients, only set after :func:`fit`

    """
    def __init__(self, degrees, rcond=None):
        ScatterFit.__init__(self)
        self.degrees = degrees
        # Underlying least-squares fitter
        self._lstsq = LinearLeastSquaresFit(rcond)
        # The following attributes are only set after :func:`fit`
        self.poly = None
        self.cov_poly = None
        self._mean = None
        self._scale = None

    def _regressor(self, x):
        """Form normalised regressor / design matrix from set of input vectors.

        Parameters
        ----------
        x : array of float, shape (2, N)
            Input to function as a 2-D numpy array

        Returns
        -------
        X : array of float, shape (P, N)
            Regressor / design matrix to be used in least-squares fit

        Notes
        -----
        This normalises the 2-D input vectors by centering and scaling them.
        It then forms a regressor matrix with a column per input vector. Each
        column is given by the outer product of the monomials of the first
        dimension with the monomials of the second dimension of the input
        vector, in decreasing polynomial order. For example, if *degrees* is
        (1, 2) and the normalised elements of each input vector in *x* are
        *x_0* and *x_1*, respectively, the column takes the form::

            outer([x_0, 1], [x1 ^ 2, x1, 1])
            = [x_0 * x_1 ^ 2, x_0 * x_1, x_0 * 1, 1 * x_1 ^ 2, 1 * x_1, 1 * 1]
            = [x_0 * x_1 ^ 2, x_0 * x_1, x_0, x_1 ^ 2, x_1, 1]

        This is closely related to the Vandermonde matrix of *x*.

        """
        x_norm = (x - self._mean[:, np.newaxis]) / self._scale[:, np.newaxis]
        v1 = np.vander(x_norm[0], self.degrees[0] + 1)
        v2 = np.vander(x_norm[1], self.degrees[1] + 1).T
        return np.vstack([v1[:, n][np.newaxis, :] * v2
                          for n in range(v1.shape[1])])

    def fit(self, x, y, std_y=1.0):
        """Fit polynomial to data.

        This fits a polynomial defined on 2-D data to the provided (x, y)
        pairs. The 2-D *x* coordinates do not have to lie on a regular grid,
        and can be in any order.

        Parameters
        ----------
        x : array-like, shape (2, N)
            Known input values as a 2-D numpy array, or sequence
        y : array-like, shape (N,)
            Known output values as a 1-D numpy array, or sequence
        std_y : float or array-like, shape (N,), optional
            Measurement error or uncertainty of `y` values, expressed as
            standard deviation in units of `y`

        Returns
        -------
        self : :class:`Polynomial2DFit` object
            Reference to self, to allow chaining of method calls

        """
        # Upcast x and y to doubles, to ensure a high enough precision
        # for the polynomial coefficients
        x = np.atleast_2d(np.array(x, dtype='double'))
        y = np.atleast_1d(np.array(y, dtype='double'))
        # Polynomial fits perform better if input data is centred
        # around origin and scaled [see numpy.polyfit help]
        self._mean = x.mean(axis=1)
        self._scale = np.abs(x - self._mean[:, np.newaxis]).max(axis=1)
        self._scale[self._scale == 0.0] = 1.0
        # Solve least squares regression problem
        self._lstsq.fit(self._regressor(x), y, std_y)
        # Convert polynomial (and cov matrix) so that it applies
        # to original unnormalised data
        tfm0 = offset_scale_mat(self.degrees[0] + 1, self._mean[0],
                                self._scale[0])
        tfm1 = offset_scale_mat(self.degrees[1] + 1, self._mean[1],
                                self._scale[1])
        tfm = np.kron(tfm0, tfm1)
        self.poly = np.dot(tfm, self._lstsq.params)
        self.cov_poly = np.dot(tfm, np.dot(self._lstsq.cov_params, tfm.T))
        return self

    def __call__(self, x, full_output=False):
        """Evaluate polynomial on new data.

        Parameters
        ----------
        x : array-like, shape (2, M)
            Input to function as a 2-D numpy array, or sequence
        full_output : {False, True}, optional
            True if output uncertainty should also be returned

        Returns
        -------
        y : array, shape (M,)
            Output of function as a 1-D numpy array
        std_y : array of float, shape (M,), optional
            Uncertainty of function output, expressed as standard deviation

        """
        x = np.atleast_2d(np.asarray(x))
        if ((self.poly is None) or (self._mean is None) or
           (self._scale is None)):
            raise NotFittedError("Polynomial not fitted to data yet - "
                                 "first call .fit method")
        return self._lstsq(self._regressor(x), full_output)

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  PiecewisePolynomial1DFit
# ----------------------------------------------------------------------------------------------------------------------


def _stepwise_interp(xi, yi, x):
    """Step-wise interpolate (or extrapolate) (xi, yi) values to x positions.

    Given a set of N ``(x, y)`` points, provided in the *xi* and *yi* arrays,
    this will calculate ``y``-coordinate values for a set of M
    ``x``-coordinates provided in the *x* array, using step-wise (zeroth-order)
    interpolation and extrapolation.

    The input *x* coordinates are compared to the fixed *xi* values, and the
    largest *xi* value smaller than or approximately equal to each *x* value is
    selected. The corresponding *yi* value is then returned. For *x* values
    below the entire set of *xi* values, the smallest *xi* value is selected.
    The steps of the interpolation therefore start at each *xi* value and
    extends to the right (above it) until the next bigger *xi*, except for the
    first step, which extends to the left (below it) as well, and the last
    step, which extends until positive infinity.

    Parameters
    ----------
    xi : array, shape (N,)
        Array of fixed x-coordinates, sorted in ascending order and with no
        duplicate values
    yi : array, shape (N,)
        Corresponding array of fixed y-coordinates
    x : float or array, shape (M,)
        Array of x-coordinates at which to do interpolation of y-values

    Returns
    -------
    y : float or array, shape (M,)
        Array of interpolated y-values

    Notes
    -----
    The equality check of *x* values is approximate on purpose, to handle some
    degree of numerical imprecision in floating-point values. This is important
    for step-wise interpolation, as there are potentially large discontinuities
    in *y* at the *xi* values, which makes it sensitive to small mismatches in
    *x*. For continuous interpolation (linear and up) this is unnecessary.

    """
    # Find lowest xi value >= x (end of segment containing x)
    end = np.atleast_1d(xi.searchsorted(x))
    # Associate any x smaller than smallest xi with closest segment (first one)
    # This linearly extrapolates the first segment to -inf on the left
    end[end == 0] += 1
    start = end - 1
    # *After* setting segment starts, associate any x bigger than biggest xi
    # with the last segment (order important, otherwise last segment
    # will be ignored)
    end[end == len(xi)] -= 1

    # First get largest "equality" difference tolerated for x and xi
    # (set to zero for integer types)
    try:
        # pylint: disable-msg=E1101
        xi_smallest_diff = 20 * np.finfo(xi.dtype).resolution
    except ValueError:
        xi_smallest_diff = 0
    try:
        # pylint: disable-msg=E1101
        x_smallest_diff = 20 * np.finfo(x.dtype).resolution
    except ValueError:
        x_smallest_diff = 0
    smallest_diff = max(x_smallest_diff, xi_smallest_diff)
    # Find x that are exactly equal to some xi or slightly below it,
    # which will assign it to the wrong segment
    equal_or_just_below = xi[end] - x < smallest_diff
    # Move these segments one higher (except for the last one, which stays put)
    start[equal_or_just_below] = end[equal_or_just_below]
    # Ensure that output y has same shape as input x
    # (especially, let scalar input result in scalar output)
    start = np.reshape(start, np.shape(x))
    return yi[start]


def _linear_interp(xi, yi, x):
    """Linearly interpolate (or extrapolate) (xi, yi) values to x positions.

    Given a set of N ``(x, y)`` points, provided in the *xi* and *yi* arrays,
    this will calculate ``y``-coordinate values for a set of M
    ``x``-coordinates provided in the *x* array, using linear interpolation
    and extrapolation.

    Parameters
    ----------
    xi : array, shape (N,)
        Array of fixed x-coordinates, sorted in ascending order and with no
        duplicate values
    yi : array, shape (N,)
        Corresponding array of fixed y-coordinates
    x : float or array, shape (M,)
        Array of x-coordinates at which to do interpolation of y-values

    Returns
    -------
    y : float or array, shape (M,)
        Array of interpolated y-values

    """
    # Find lowest xi value >= x (end of segment containing x)
    end = np.atleast_1d(xi.searchsorted(x))
    # Associate any x found outside xi range with closest segment (first or
    # last one). This linearly extrapolates the first and last segment
    # to -inf and +inf, respectively.
    end[end == 0] += 1
    end[end == len(xi)] -= 1
    start = end - 1
    # Ensure that output y has same shape as input x
    # (especially, let scalar input result in scalar output)
    start, end = np.reshape(start, np.shape(x)), np.reshape(end, np.shape(x))
    # Set up weight such that xi[start] => 0 and xi[end] => 1
    end_weight = (x - xi[start]) / (xi[end] - xi[start])
    return (1.0 - end_weight) * yi[start] + end_weight * yi[end]


class PiecewisePolynomial1DFit(ScatterFit):
    """Fit piecewise polynomial to 1-D data.

    This fits a series of polynomials between adjacent points in a
    one-dimensional data set. The resulting piecewise polynomial curve passes
    exactly through the given data points and may also match the local gradient
    at each point if the maximum polynomial degree *max_degree* is at least 3.

    If *max_degree* is 0, step-wise interpolation is done between the points in
    the data set. Each input *x* value is assigned the *y* value of the largest
    *x* value in the data set that is smaller than or equal to the input *x*.
    If the input *x* is smaller than all the *x* values in the data set, the
    *y* value of the smallest data set *x* value is chosen instead.

    If *max_degree* is 1, linear interpolation is done. The resulting curve is
    continuous but has sharp corners at the data points. If *max_degree* is 3,
    cubic interpolation is used and the resulting is curve is smooth (up to the
    first derivative).

    This should primarily be used for interpolation between points and not for
    extrapolation outside the data range, which could lead to wildly inaccurate
    results (especially if *max_degree* is high).

    Parameters
    ----------
    max_degree : int
        Maximum polynomial degree (>= 0) to use in each segment between data
        points (automatically reduced if there are not enough data points or
        where derivatives are not available, such as in the first and last
        segment)

    Notes
    -----
    This is based on :class:`scipy.interpolate.PiecewisePolynomial`.

    """
    def __init__(self, max_degree=3):
        ScatterFit.__init__(self)
        self.max_degree = max_degree
        self._poly = None

    def fit(self, x, y):
        """Fit piecewise polynomial to data.

        Parameters
        ----------
        x : array-like, shape (N,)
            Known input values as a 1-D numpy array or sequence
        y : array-like, shape (N,)
            Known output values as a 1-D numpy array, or sequence

        Returns
        -------
        self : :class:`PiecewisePolynomial1DFit` object
            Reference to self, to allow chaining of method calls

        Raises
        ------
        ValueError
            If *x* contains duplicate values, which leads to infinite gradients

        """
        # Upcast x and y to doubles, to ensure a high enough precision
        # for the polynomial coefficients
        x = np.atleast_1d(np.array(x, dtype='double'))
        # Only upcast y if numerical interpolation will actually happen -
        # since stepwise interpolation simply copies y values, this allows
        # interpolation of non-numeric types (e.g. strings)
        if (len(x) == 1) or (self.max_degree == 0):
            y = np.atleast_1d(y)
        else:
            y = np.atleast_1d(np.array(y, dtype='double'))
        # Sort x in ascending order, as PiecewisePolynomial expects sorted data
        x_ind = np.argsort(x)
        x, y = x[x_ind], y[x_ind]
        # This list will contain y values and corresponding derivatives
        y_list = np.atleast_2d(y).transpose().tolist()
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("Two consecutive points have same x-coordinate - "
                             "infinite gradient not allowed")
        # Maximum derivative order warranted by polynomial degree
        # and number of data points
        max_deriv = min((self.max_degree - 1) // 2, len(x) - 2) + 1
        if max_deriv > 1:
            # Length of x interval straddling each data point
            # (from previous to next point)
            x_interval = np.convolve(np.diff(x), [1.0, 1.0], 'valid')
            y_deriv = y
        # Recursively calculate the n'th derivative of y, up to maximum order
        for n in range(1, max_deriv):
            # The difference between (n-1)'th derivative of y at previous
            # and next point, divided by interval
            y_deriv = np.convolve(np.diff(y_deriv),
                                  [1.0, 1.0], 'valid') / x_interval
            x_interval = x_interval[1:-1]
            for m in range(len(y_deriv)):
                y_list[m + n].append(y_deriv[m])
        if len(x) == 1:
            # Constant interpolation to all new x values
            self._poly = lambda new_x: np.tile(y[0], np.asarray(new_x).shape)
        elif self.max_degree == 0:
            # SciPy PiecewisePolynomial does not support degree 0 -
            # use home-brewed interpolator instead
            self._poly = lambda new_x: _stepwise_interp(x, y,
                                                        np.asarray(new_x))
        elif self.max_degree == 1:
            # Home-brewed linear interpolator is *way* faster than
            # SciPy 0.7.0 PiecewisePolynomial
            self._poly = lambda new_x: _linear_interp(x, y, np.asarray(new_x))
        else:
            try:
                # New-style piecewise polynomials available
                # since scipy 0.14.0, enforced since 0.18.0
                self._poly = scipy.interpolate.BPoly.from_derivatives(
                    x, y_list, orders=None)
            except AttributeError:
                # Old-style piecewise polynomials available
                # in scipy 0.7.0 - 0.17.1
                self._poly = scipy.interpolate.PiecewisePolynomial(
                    x, y_list, orders=None, direction=1)
        return self

    def __call__(self, x):
        """Evaluate piecewise polynomial on new data.

        Parameters
        ----------
        x : float or array-like, shape (M,)
            Input to function as a scalar, 1-D numpy array or sequence

        Returns
        -------
        y : float or array, shape (M,)
            Output of function as a scalar o 1-D numpy array

        """
        if self._poly is None:
            raise NotFittedError("Piecewise polynomial not fitted to data "
                                 "yet - first call .fit method")
        return self._poly(x)
