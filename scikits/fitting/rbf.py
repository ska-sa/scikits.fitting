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

"""RBF fitter.

:author: Ludwig Schwardt
:license: Modified BSD

"""

import numpy as np
import scipy.interpolate

from .generic import ScatterFit, NotFittedError

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  RbfScatterFit
# ----------------------------------------------------------------------------------------------------------------------


class RbfScatterFit(ScatterFit):
    """Do radial basis function (RBF) interpolation of scattered multi-dimensional data.

    This uses the :class:`scipy.interpolate.Rbf` class. The D-dimensional ``x``
    coordinates do not have to lie on a regular grid, and can be in any order.

    Parameters
    ----------
    kwargs : dict, optional
        Additional keyword arguments are passed to underlying Rbf class

    """
    def __init__(self, **kwargs):
        ScatterFit.__init__(self)
        # Extra keyword arguments to Rbf class
        self._extra_args = kwargs
        # Interpolator function, only set after :func:`fit`
        self._interp = None

    def fit(self, x, y):
        """Fit RBF to D-dimensional scattered data in unstructured form.

        The D-dimensional *x* coordinates do not have to lie on a regular grid,
        and can be in any order.

        Parameters
        ----------
        x : array-like, shape (D, N)
            Known input values as a numpy array or sequence
        y : array-like, shape (N,)
            Known output values as a 1-D numpy array or sequence

        Returns
        -------
        self : :class:`RbfScatterFit` object
            Reference to self, to allow chaining of method calls

        """
        # Check dimensions of known data
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if ((len(x.shape) != 2) or
           (len(y.shape) != 1) or (y.shape[0] != x.shape[1])):
            raise ValueError("RBF interpolator requires input data with "
                             "shape (D, N) and output data with shape (N,), "
                             "got %s and %s instead" % (x.shape, y.shape))
        # RBF interpolator available since scipy 0.7.0
        self._interp = scipy.interpolate.Rbf(
            *np.vstack((x, y)), **self._extra_args)
        return self

    def __call__(self, x):
        """Evaluate RBF on new scattered data.

        Parameters
        ----------
        x : array-like, shape (D, M)
            Input to function as a numpy array or sequence

        Returns
        -------
        y : array, shape (M,)
            Output of function as a 1-D numpy array

        """
        # Check dimensions
        x = np.atleast_2d(np.asarray(x))
        if (len(x.shape) != 2):
            raise ValueError("RBF interpolator requires input data with "
                             "shape (D, M), got %s instead" % (x.shape,))
        if self._interp is None:
            raise NotFittedError("RBF not fitted to data yet - "
                                 "first call .fit method")
        return self._interp(*x)
