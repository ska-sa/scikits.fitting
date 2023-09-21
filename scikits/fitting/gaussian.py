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

"""Gaussian fitter.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

import numpy as np

from .generic import ScatterFit
from .nonlinlstsq import NonLinearLeastSquaresFit

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  GaussianFit
# ----------------------------------------------------------------------------------------------------------------------


class GaussianFit(ScatterFit):
    """Fit Gaussian curve to multi-dimensional data.

    This fits a D-dimensional Gaussian function (with diagonal covariance
    matrix or single scalar variance) to x-y data. Don't confuse this with
    fitting a Gaussian probability density function (pdf) to random data!

    Parameters
    ----------
    mean : array-like of float, shape (D,)
        Initial guess of D-dimensional mean vector
    std : array-like of float, shape (D,), or float
        Initial guess of D-dimensional vector of standard deviations, or a
        single standard deviation for all dimensions
    height : float
        Initial guess of height of Gaussian curve

    Attributes
    ----------
    mean : array of float, shape (D,)
        D-dimensional mean vector, either initial guess or final optimal value
    std : array of float, shape (D,), or float
        D-dimensional standard deviation vector or scalar, either initial guess
        or final optimal value
    height : float
        Height of Gaussian curve, either initial guess or final optimal value
    std_mean : array of float, shape (D,)
        Standard error of mean vector, only set after :func:`fit`
    std_std : array of float, shape (D,), or float
        Standard error of standard deviation, only set after :func:`fit`
    std_height : float
        Standard error of height, only set after :func:`fit`

    Raises
    ------
    ValueError
        If dimensions of mean and/or std are incompatible

    Notes
    -----
    This is built on top of :class:`NonLinearLeastSquaresFit`. One option that
    was considered is fitting the Gaussian internally to the log of the data.
    This is more robust in some scenarios, but cannot handle negative data,
    which frequently occur in noisy problems. With log data, the optimisation
    criterion is not quite least-squares in the original x-y domain as well.

    """
    # pylint: disable-msg=W0612
    def __init__(self, mean, std, height):
        ScatterFit.__init__(self)
        self.mean = np.atleast_1d(mean)
        self.std = np.atleast_1d(std)
        self.height = height
        if ((self.mean.ndim != 1) or (self.std.ndim != 1) or
           (self.std.shape not in [self.mean.shape, (1,)])):
            raise ValueError("Dimensions of mean and/or "
                             "standard deviation incorrect")
        # Make sure a single standard devation is a plain scalar,
        # and create parameter vector for optimisation
        if self.std.shape == (1,):
            self.std = self.std[0]
        params = np.r_[self.mean, self.height, self.std]

        def gauss_diagcov(p, x):
            """Evaluate D-dimensional Gaussian with diagonal covariance matrix.

            Parameters
            ----------
            p : array-like, shape (D + 2,) or (2*D + 1,)
                Parameters, consisting of D-dim mean vector + scalar height
                + D-dim standard deviation vector (or scalar)
            x : array-like, shape (D, N) or (D,)
                Sequence of D-dimensional input values as columns of array

            Returns
            -------
            y : array, shape (N,), or float
                Sequence of 1-D output values as a NumPy array

            """
            p, x = np.atleast_1d(p), np.atleast_1d(x)
            D = x.shape[0]
            x_min_mu = x - p[:D, np.newaxis] if x.ndim > 1 else x - p[:D]
            var = (np.tile(p[D + 1] ** 2, D)
                   if len(p) == D + 2 else p[D + 1:] ** 2)
            return p[D] * np.exp(-0.5 * np.dot(1 / var, x_min_mu * x_min_mu))

        def jac_gauss_diagcov(p, x):
            """Evaluate Jacobian of D-dimensional diagonal Gaussian.

            The parameter vector has dimension P = 2*D + 1 for a vector of
            standard deviations, or P = D + 2 for a scalar standard devation.

            Parameters
            ----------
            p : array-like, shape (P,)
                Parameters, consisting of D-dim mean vector + scalar height
                + D-dim standard deviation vector (or scalar)
            x : array-like, shape (D, N)
                Sequence of D-dimensional input values as columns of array

            Returns
            -------
            J : array, shape (P, N)
                Jacobian matrix / vector J(p, x) of partial derivatives

            """
            p, x = np.atleast_1d(p), np.atleast_2d(x)
            D = x.shape[0]
            mu = p[:D, np.newaxis]
            # Ensure std is a D-dimensional vector
            sigma = (np.tile(p[D + 1], (D, 1))
                     if len(p) == D + 2 else p[D + 1:, np.newaxis])
            norm_x = (x - mu) / sigma
            dy_dheight = np.exp(-0.5 * (norm_x * norm_x).sum(axis=0))
            y = p[D] * dy_dheight
            dy_dmean = y * norm_x / sigma
            dy_dstd = dy_dmean * norm_x
            dy_dstd = dy_dstd.sum(axis=0) if len(p) == D + 2 else dy_dstd
            return np.vstack((dy_dmean, dy_dheight, dy_dstd))

        # Internal non-linear least squares fitter
        self._interp = NonLinearLeastSquaresFit(
            gauss_diagcov, params, func_jacobian=jac_gauss_diagcov)
        self.std_mean = self.std_std = self.std_height = None

    def fit(self, x, y, std_y=1.0):
        """Fit a Gaussian curve to data.

        The mean, standard deviation and height can be obtained from the
        corresponding member variables after this is run.

        Parameters
        ----------
        x : array-like, shape (D, N)
            Sequence of D-dimensional input values as columns of a numpy array
        y : array-like, shape (N,)
            Sequence of 1-D output values as a numpy array
        std_y : float or array-like, shape (N,), optional
            Measurement error or uncertainty of `y` values, expressed as
            standard deviation in units of `y`

        Returns
        -------
        self : :class:`GaussianFit` object
            Reference to self, to allow chaining of method calls

        """
        self._interp.fit(x, y, std_y)
        # Recreate Gaussian parameters
        D = len(self.mean)
        self.mean = self._interp.params[:D]
        self.height = self._interp.params[D]
        self.std = (self._interp.params[D + 1]
                    if len(self._interp.params) == D + 2
                    else self._interp.params[D + 1:])
        # Since standard deviations only appear in squared form
        # in cost function, they have a sign ambiguity
        self.std = np.abs(self.std)
        std_params = np.sqrt(np.diag(self._interp.cov_params))
        self.std_mean = std_params[:D]
        self.std_height = std_params[D]
        self.std_std = (std_params[D + 1] if len(self._interp.params) == D + 2
                        else std_params[D + 1:])
        return self

    def __call__(self, x):
        """Evaluate function ``y = f(x)`` on new data.

        Parameters
        ----------
        x : array-like, shape (D, M) or (D,)
            Sequence of D-dimensional input values as columns of numpy array

        Returns
        -------
        y : array, shape (M,), or float
            Sequence of 1-D output values as a numpy array

        """
        return self._interp(x)
