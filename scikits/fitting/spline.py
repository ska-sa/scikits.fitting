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

"""Spline fitters.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

from builtins import zip
from builtins import range
import numpy as np
import scipy.interpolate

from .generic import ScatterFit, GridFit, NotFittedError
from .utils import sort_grid, desort_grid

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  Spline1DFit
# ----------------------------------------------------------------------------------------------------------------------


class Spline1DFit(ScatterFit):
    """Fit a B-spline to 1-D data.

    This wraps :class:`scipy.interpolate.UnivariateSpline`, which is based on
    Paul Dierckx's DIERCKX (or FITPACK) routines (specifically ``curfit`` for
    fitting and ``splev`` for evaluation).

    Parameters
    ----------
    degree : int, optional
        Degree of spline, in range 1-5 [default=3, i.e. cubic B-spline]
    min_size : float, optional
        Size of smallest features to fit in the data, expressed in units of
        *x*. This determines the smoothness of fitted spline. Roughly stated,
        any oscillation in the fitted curve will have a period bigger than
        *min_size*. Works best if *x* is uniformly spaced.
    kwargs : dict, optional
        Additional keyword arguments are passed to underlying spline class

    """
    def __init__(self, degree=3, min_size=0.0, **kwargs):
        ScatterFit.__init__(self)
        self.degree = degree
        # Size of smallest features to fit
        self._min_size = min_size
        # Extra keyword arguments to spline class
        self._extra_args = kwargs
        # Interpolator function, only set after :func:`fit`
        self._interp = None

    def fit(self, x, y, std_y=1.0):
        """Fit spline to 1-D data.

        The minimum number of data points is N = degree + 1.

        Parameters
        ----------
        x : array-like, shape (N,)
            Known input values as a 1-D numpy array or sequence
        y : array-like, shape (N,)
            Known output values as a 1-D numpy array, or sequence
        std_y : float or array-like, shape (N,), optional
            Measurement error or uncertainty of `y` values, expressed as
            standard deviation in units of `y` (overrides min_size setting)

        Returns
        -------
        self : :class:`Spline1DFit` object
            Reference to self, to allow chaining of method calls

        """
        # Check dimensions of known data
        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if y.size < self.degree + 1:
            raise ValueError("Not enough data points for spline fit: "
                             "requires at least %d, only got %d" %
                             (self.degree + 1, y.size))
        # Ensure that x is in strictly ascending order
        if np.any(np.diff(x) < 0):
            sort_ind = x.argsort()
            x = x[sort_ind]
            y = y[sort_ind]
        # Deduce standard deviation of y if not given, based on specified
        # size of smallest features
        if self._min_size > 0.0 and std_y == 1.0:
            # Number of samples, and sample period
            # (assuming samples are uniformly spaced in x)
            N, xstep = len(x), np.abs(np.mean(np.diff(x)))
            # Convert feature size to digital frequency (based on
            # k / N = Ts / T using FFT notation). The frequency index k is
            # clipped so that k > 0, to avoid including DC power in stdev calc
            # (i.e. slowest oscillation is N samples), and k <= N / 2,
            # which represents a 2-sample oscillation.
            min_freq_ind = np.clip(int(np.round(N * xstep / self._min_size)),
                                   1, N // 2)
            # Find power in signal above the minimum cutoff frequency using
            # periodogram. Reduce spectral leakage resulting from edge effects
            # by removing DC and windowing the signal.
            window = np.hamming(N)
            periodo = np.abs(np.fft.fft((y - y.mean()) * window)) ** 2
            periodo /= (window ** 2).sum()
            periodo[1:(N // 2)] *= 2.0
            std_y = np.sqrt(np.sum(periodo[min_freq_ind:(N // 2 + 1)]) / N)
        # Convert uncertainty into array of shape (N,)
        if np.isscalar(std_y):
            std_y = np.tile(std_y, y.shape)
        std_y = np.atleast_1d(np.asarray(std_y))
        # Lower bound on uncertainty is determined by floating-point
        # resolution (no upper bound)
        np.clip(std_y, max(np.mean(np.abs(y)), 1e-20) * np.finfo(y.dtype).eps,
                np.inf, out=std_y)
        self._interp = scipy.interpolate.UnivariateSpline(
            x, y, w=1. / std_y, k=self.degree, **self._extra_args)
        return self

    def __call__(self, x):
        """Evaluate spline on new data.

        Parameters
        ----------
        x : array-like, shape (M,)
            Input to function as a 1-D numpy array, or sequence

        Return
        ------
        y : array, shape (M,)
            Output of function as a 1-D numpy array

        """
        x = np.atleast_1d(np.asarray(x))
        if self._interp is None:
            raise NotFittedError("Spline not fitted to data yet - "
                                 "first call .fit method")
        return self._interp(x)

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  Spline2DScatterFit
# ----------------------------------------------------------------------------------------------------------------------


class Spline2DScatterFit(ScatterFit):
    """Fit a B-spline to scattered 2-D data.

    This wraps :class:`scipy.interpolate.SmoothBivariateSpline`, which is based
    on Paul Dierckx's DIERCKX (or FITPACK) routines (specifically ``surfit``
    for fitting and ``bispev`` for evaluation). The 2-D ``x`` coordinates do
    not have to lie on a regular grid, and can be in any order.

    Parameters
    ----------
    degree : sequence of 2 ints, optional
        Degree (1-5) of spline in x and y directions
    kwargs : dict, optional
        Additional keyword arguments are passed to underlying spline class

    """
    def __init__(self, degree=(3, 3), **kwargs):
        ScatterFit.__init__(self)
        self.degree = degree
        # Extra keyword arguments to spline class
        self._extra_args = kwargs
        # Interpolator function, only set after :func:`fit`
        self._interp = None

    def fit(self, x, y, std_y=1.0):
        """Fit spline to 2-D scattered data in unstructured form.

        The minimum number of data points is
        ``N = (degree[0]+1)*(degree[1]+1)``. The 2-D *x* coordinates do not
        have to lie on a regular grid, and can be in any order.

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
        self : :class:`Spline2DScatterFit` object
            Reference to self, to allow chaining of method calls

        """
        # Check dimensions of known data
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if (len(x.shape) != 2) or (x.shape[0] != 2) or (
          len(y.shape) != 1) or (y.shape[0] != x.shape[1]):
            raise ValueError("Spline interpolator requires input data with "
                             "shape (2, N) and output data with shape (N,), "
                             "got %s and %s instead" % (x.shape, y.shape))
        if y.size < (self.degree[0] + 1) * (self.degree[1] + 1):
            raise ValueError("Not enough data points for spline fit: requires "
                             "at least %d, only got %d" %
                             ((self.degree[0] + 1) * (self.degree[1] + 1),
                              y.size))
        # Convert uncertainty into array of shape (N,)
        if np.isscalar(std_y):
            std_y = np.tile(std_y, y.shape)
        std_y = np.atleast_1d(np.asarray(std_y))
        # Lower bound on uncertainty is determined by floating-point resolution
        # (no upper bound)
        np.clip(std_y, max(np.mean(np.abs(y)), 1e-20) * np.finfo(y.dtype).eps,
                np.inf, out=std_y)
        self._interp = scipy.interpolate.SmoothBivariateSpline(
            x[0], x[1], y, w=1. / std_y, kx=self.degree[0], ky=self.degree[1],
            **self._extra_args)
        return self

    def __call__(self, x):
        """Evaluate spline on new scattered data.

        Parameters
        ----------
        x : array-like, shape (2, M)
            Input to function as a 2-D numpy array, or sequence

        Returns
        -------
        y : array, shape (M,)
            Output of function as a 1-D numpy array

        """
        # Check dimensions
        x = np.atleast_2d(np.asarray(x))
        if (len(x.shape) != 2) or (x.shape[0] != 2):
            raise ValueError("Spline interpolator requires input data "
                             "with shape (2, M), got %s instead" % (x.shape,))
        if self._interp is None:
            raise NotFittedError("Spline not fitted to data yet - "
                                 "first call .fit method")
        # Loop over individual data points, as underlying bispev routine
        # expects regular grid in x
        return np.array([self._interp(x[0, n], x[1, n])
                         for n in range(x.shape[1])]).squeeze()

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  Spline2DGridFit
# ----------------------------------------------------------------------------------------------------------------------


class Spline2DGridFit(GridFit):
    """Fit a B-spline to 2-D data on a rectangular grid.

    This wraps :mod:`scipy.interpolate.RectBivariateSpline`, which is based on
    Paul Dierckx's DIERCKX (or FITPACK) routines (specifically ``regrid`` for
    fitting and ``bispev`` for evaluation). The 2-D ``x`` coordinates define a
    rectangular grid. They do not have to be in ascending order, as both the
    fitting and evaluation routines sort them for you.

    Parameters
    ----------
    degree : sequence of 2 ints, optional
        Degree (1-5) of spline in x and y directions
    kwargs : dict, optional
        Additional keyword arguments are passed to underlying spline class

    """
    def __init__(self, degree=(3, 3), **kwargs):
        GridFit.__init__(self)
        self.degree = degree
        # Extra keyword arguments to spline class
        self._extra_args = kwargs
        # Interpolator function, only set after :func:`fit`
        self._interp = None

    def fit(self, x, y, std_y=None):
        """Fit spline to 2-D data on a rectangular grid.

        This fits a scalar function defined on 2-D data to the provided grid.
        The first sequence in *x* defines the M 'x' axis ticks (in any order),
        while the second sequence in *x* defines the N 'y' axis ticks (also in
        any order). The provided function output *y* contains the corresponding
        'z' values on the grid, in an array of shape (M, N). The minimum number
        of data points is ``(degree[0]+1)*(degree[1]+1)``.

        Parameters
        ----------
        x : sequence of 2 sequences, of lengths M and N
            Known input grid specified by sequence of 2 sequences of axis ticks
        y : array-like, shape (M, N)
            Known output values as a 2-D numpy array
        std_y : None or float or array-like, shape (M, N), optional
            Measurement error or uncertainty of `y` values, expressed as
            standard deviation in units of `y`. If None, uncertainty
            propagation is disabled (typically to save time as this can be
            costly to calculate when M*N is large).

        Returns
        -------
        self : :class:`Spline2DGridFit` object
            Reference to self, to allow chaining of method calls

        Notes
        -----
        This propagates uncertainty through the spline fit based on the main
        idea of [1]_, as expressed in Eq. (13) in the paper. Take note that
        this equation contains an error -- the square brackets on the
        right-hand side should enclose the entire sum over i and not just the
        summand.

        .. [1] Enting, I. G., Trudinger, C. M., and Etheridge, D. M.,
           "Propagating data uncertainty through smoothing spline fits,"
           Tellus, vol. 58B, pp. 305-309, 2006.

        """
        # Check dimensions of known data
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        y = np.atleast_2d(np.asarray(y))
        if ((len(x) != 2) or (len(x[0].shape) != 1) or
            (len(x[1].shape) != 1) or (len(y.shape) != 2) or
                (y.shape[0] != len(x[0])) or (y.shape[1] != len(x[1]))):
            raise ValueError("Spline interpolator requires input data with "
                             "shape [(M,), (N,)] and output data "
                             "with shape (M, N), got %s and %s instead" %
                             ([ax.shape for ax in x], y.shape))
        if y.size < (self.degree[0] + 1) * (self.degree[1] + 1):
            raise ValueError("Not enough data points for spline fit: "
                             "requires at least %d, only got %d" %
                             ((self.degree[0] + 1) * (self.degree[1] + 1),
                              y.size))
        # Ensure that 'x' and 'y' coordinates are both in ascending order
        # (requirement of underlying regrid)
        xs, ys, zs = sort_grid(x[0], x[1], y)
        self._interp = scipy.interpolate.RectBivariateSpline(
            xs, ys, zs, kx=self.degree[0], ky=self.degree[1],
            **self._extra_args)
        # Disable uncertainty propagation if no std_y is given
        if std_y is None:
            self._std_fitted_y = None
        else:
            # Uncertainty should have same shape as y
            # (or get tiled to that shape if it is scalar)
            std_y = np.atleast_2d(np.asarray(std_y))
            self._std_fitted_y = (np.tile(std_y, y.shape) if
                                  std_y.shape == (1, 1) else std_y)
            if self._std_fitted_y.shape != y.shape:
                raise ValueError("Spline interpolator requires uncertainty "
                                 "to be scalar or to have shape "
                                 "%s (same as data), got %s instead" %
                                 (y.shape, self._std_fitted_y.shape))
            # Create list of interpolators, one per value in y,
            # by setting each y value to 1 in turn (and the rest 0)
            self._std_interps = []
            testz = np.zeros(zs.size)
            for m in range(zs.size):
                testz[:] = 0.0
                testz[m] = 1.0
                interp = scipy.interpolate.RectBivariateSpline(
                    xs, ys, testz.reshape(zs.shape), kx=self.degree[0],
                    ky=self.degree[1], **self._extra_args)
                self._std_interps.append(interp)
        return self

    def __call__(self, x, full_output=False):
        """Evaluate spline on a new rectangular grid.

        Evaluates the fitted scalar function on 2-D grid provided in *x*. The
        first sequence in *x* defines the K 'x' axis ticks (in any order),
        while the second sequence in *x* defines the L 'y' axis ticks (also in
        any order). The function returns the corresponding 'z' values on the
        grid, in an array of shape (K, L).

        Parameters
        ----------
        x : sequence of 2 sequences, of lengths K and L
            2-D input grid specified by sequence of 2 sequences of axis ticks
        full_output : {False, True}, optional
            True if output uncertainty should also be returned

        Returns
        -------
        y : float array, shape (K, L)
            Output of function as a 2-D numpy array
        std_y : None or float array, shape (K, L), optional
            Uncertainty of function output, expressed as standard deviation
            (or None if no 'y' uncertainty was supplied during fitting)

        """
        # Check dimensions
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        if (len(x) != 2) or (len(x[0].shape) != 1) or (len(x[1].shape) != 1):
            raise ValueError("Spline interpolator requires input data with "
                             "shape [(K,), (L,)], got %s instead" %
                             ([ax.shape for ax in x],))
        if self._interp is None:
            raise NotFittedError("Spline not fitted to data yet - "
                                 "first call .fit method")
        # The standard DIERCKX 2-D spline evaluation function (bispev) expects
        # a rectangular grid in ascending order. Therefore, sort coordinates,
        # evaluate on the sorted grid, and return the desorted result
        x0s, x1s = sorted(x[0]), sorted(x[1])
        y = desort_grid(x[0], x[1], self._interp(x0s, x1s))
        if not full_output:
            return y
        if self._std_fitted_y is None:
            return y, None
        # The output y variance is a weighted sum of the variances of the
        # fitted y values, according to Enting's method
        var_ys = np.zeros(y.shape)
        for std_fitted_y, std_interp in zip(self._std_fitted_y.ravel(),
                                            self._std_interps):
            var_ys += (std_fitted_y * std_interp(x0s, x1s)) ** 2
        return y, desort_grid(x[0], x[1], np.sqrt(var_ys))
