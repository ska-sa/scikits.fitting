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

"""Linear least-squares fitter.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

import warnings

import numpy as np

from .generic import ScatterFit, NotFittedError

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  LinearLeastSquaresFit
# ----------------------------------------------------------------------------------------------------------------------


class LinearLeastSquaresFit(ScatterFit):
    r"""Fit linear regression model to data using the SVD.

    This fits a linear function of the form :math:`y = p^T x` to a sequence of
    N P-dimensional input vectors :math:`x` and a corresponding sequence of N
    output measurements :math:`y`. The input to the fitter is presented as an
    input *design matrix* :math:`X` of shape (P, N) and an N-dimensional output
    *measurement vector* :math:`y`. The P-dimensional *parameter vector*
    :math:`p` is determined by the fitting procedure. The fitter can use
    uncertainties on the `y` measurements and also produces a covariance matrix
    for the parameters. The number of parameters, P, is determined by the shape
    of :math:`X` when :meth:`fit` is called.

    Parameters
    ----------
    rcond : float or None, optional
        Relative condition number of the fit. Singular values smaller than this
        relative to the largest singular value will be ignored. The default
        value is N * eps, where eps is the relative precision of the float
        type, about 2e-16 in most cases, and N is length of output vector `y`.

    Attributes
    ----------
    params : array of float, shape (P,)
        Fitted parameter vector
    cov_params : array of float, shape (P, P)
        Standard covariance matrix of parameters

    Notes
    -----
    The :meth:`fit` method finds the optimal parameter vector :math:`p` that
    minimises the sum of squared weighted residuals, given by

    .. math:: \chi^2 = \sum_{i=1}^N \left[\frac{y_i - \sum_{j=1}^P p_j x_{ji}}{\sigma_i}\right]^2

    where :math:`x_{ji}` are the elements of the design matrix :math:`X` and
    :math:`\sigma_i` is the uncertainty associated with measurement
    :math:`y_i`. The problem is solved using the singular-value decomposition
    (SVD) of the design matrix, based on the description in Section 15.4 of
    [1]_. This gives the same parameter solution as the NumPy function
    :func:`numpy.linalg.lstsq`, but also provides the covariance matrix of the
    parameters.

    .. [1] Press, Teukolsky, Vetterling, Flannery, "Numerical Recipes in C,"
       Second Edition, 1992.

    """
    def __init__(self, rcond=None):
        ScatterFit.__init__(self)
        self.rcond = rcond
        self.params = None
        self.cov_params = None

    def fit(self, x, y, std_y=1.0):
        """Fit linear regression model to x-y data using the SVD.

        Parameters
        ----------
        x : array-like, shape (P, N)
            Known input values as design matrix (one row per desired parameter)
        y : array-like, shape (N,)
            Known output measurements as sequence or numpy array
        std_y : float or array-like, shape (N,), optional
            Measurement error or uncertainty of `y` values, expressed as
            standard deviation in units of `y`

        Returns
        -------
        self : :class:`LinearLeastSquaresFit` object
            Reference to self, to allow chaining of method calls

        """
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        # Convert uncertainty into array of shape (N,)
        if np.isscalar(std_y):
            std_y = np.tile(std_y, y.shape)
        std_y = np.atleast_1d(np.asarray(std_y))
        # Lower bound on uncertainty is determined by floating-point
        # resolution (no upper bound)
        np.clip(std_y, max(np.mean(np.abs(y)), 1e-20) * np.finfo(y.dtype).eps,
                np.inf, out=std_y)
        # Normalise uncertainty to avoid numerical blow-up
        # (only relative uncertainty matters for parameter solution)
        max_std_y = std_y.max()
        std_y /= max_std_y
        # Weight design matrix columns and output vector by `y` uncertainty
        A = x / std_y[np.newaxis, :]
        b = y / std_y
        # Perform SVD on A, which is transpose of usual design matrix -
        # let A^T = Ur S V^T to correspond with NRinC
        # Shapes: A ~ PxN, b ~ N, V ~ PxP, s ~ P, S = diag(s) ~ PxP,
        # "reduced U" Ur ~ NxP and Urt = Ur^T ~ PxN
        V, s, Urt = np.linalg.svd(A, full_matrices=False)
        # Set all "small" singular values below this relative cutoff equal to 0
        s_cutoff = (len(x) * np.finfo(x.dtype).eps * s[0]
                    if self.rcond is None else self.rcond * s[0])
        # Warn if the effective rank < P
        # (i.e. some singular values are considered to be zero)
        if np.any(s < s_cutoff):
            warnings.warn('Least-squares fit may be poorly conditioned')
        # Invert zero singular values to infinity, as we are actually
        # interested in reciprocal of s, and zero singular values should be
        # replaced by zero reciprocal values a la pseudo-inverse
        s[s < s_cutoff] = np.inf
        # Solve linear least-squares problem using SVD
        # (see NRinC, 2nd ed, Eq. 15.4.17)
        # In matrix form: p = V S^(-1) Ur^T b = Vs Ur^T b, where Vs = V S^(-1)
        Vs = V / s[np.newaxis, :]
        self.params = np.dot(Vs, np.dot(Urt, b))
        # Also obtain covariance matrix of parameters
        # (see NRinC, 2nd ed, Eq. 15.4.20)
        # In matrix form: Cp = V S^(-2) V^T = Vs Vs^T
        # (also rescaling with max std_y)
        self.cov_params = np.dot(Vs, Vs.T) * (max_std_y ** 2)
        return self

    def __call__(self, x, full_output=False):
        """Evaluate linear regression model on new x data.

        Parameters
        ----------
        x : array-like, shape (P, M)
            New input values as design matrix (one row per fitted parameter)
        full_output : {False, True}, optional
            True if output uncertainty should also be returned

        Returns
        -------
        y : array, shape (M,)
            Corresponding output of function as a numpy array
        std_y : array, shape (M,), optional
            Uncertainty of function output, expressed as standard deviation

        """
        if (self.params is None) or (self.cov_params is None):
            raise NotFittedError("Linear regression model not fitted to data "
                                 "yet - first call .fit method")
        A = np.atleast_2d(np.asarray(x))
        y = np.dot(self.params, A)
        if full_output:
            return y, np.sqrt(np.sum(A * np.dot(self.cov_params, A), axis=0))
        else:
            return y
