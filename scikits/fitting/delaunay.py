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

"""Delaunay fitters.

These interpolators are the ones underlying :mod:`scipy.interpolate.griddata`,
unpacked to split the fitting and evaluation stages.

:author: Ludwig Schwardt
:license: Modified BSD

"""
import numpy as np
# Interpolators based on triangulation available since scipy 0.9.0
from scipy.interpolate import (CloughTocher2DInterpolator,
                               LinearNDInterpolator, NearestNDInterpolator)

from .generic import ScatterFit, NotFittedError

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  Delaunay2DScatterFit
# ----------------------------------------------------------------------------------------------------------------------


class Delaunay2DScatterFit(ScatterFit):
    """Interpolate scalar function of 2-D data, based on Delaunay triangulation.

    The *x* data for this object should have two rows, containing the 'x' and
    'y' coordinates of points in a plane. The 2-D points are therefore stored
    as column vectors in *x*. The *y* data for this object is a 1-D array,
    which represents the scalar 'z' value of the function defined on the plane
    (the symbols in quotation marks are the typical mathematical names for
    these variables). The 2-D *x* coordinates do not have to lie on a regular
    grid, and can be in any order.

    Parameters
    ----------
    interp_type : {'cubic', 'linear', 'nearest'}, optional
        String indicating type of interpolation
    default_val : float, optional
        Default value used when trying to extrapolate beyond convex hull of
        known data (ignored for 'nearest')
    jitter : bool, optional
        True to add small amount of jitter to *x* to make degenerate
        triangulation unlikely (generally not needed)

    """
    def __init__(self, interp_type='cubic', default_val=np.nan, jitter=False):
        ScatterFit.__init__(self)
        interps = ('cubic', 'linear', 'nearest')
        if interp_type not in interps:
            raise ValueError("Interpolator has to be one of %s, not %r" %
                             (interps, interp_type))
        self.interp_type = interp_type
        self.default_val = default_val
        self.jitter = jitter
        # Interpolator function, only set after :func:`fit`
        self._interp = None

    def fit(self, x, y):
        """Fit function ``y = f(x)`` to data.

        This fits a scalar function defined on 2-D data to the provided x-y
        pairs. The 2-D *x* coordinates do not have to lie on a regular grid,
        and can be in any order.

        Parameters
        ----------
        x : array-like, shape (2, N)
            Known input values as a 2-D numpy array, or sequence
        y : array-like, shape (N,)
            Known output values as a 1-D numpy array, or sequence

        Returns
        -------
        self : :class:`Delaunay2DScatterFit` object
            Reference to self, to allow chaining of method calls

        """
        # Check dimensions of known data
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if (len(x.shape) != 2) or (x.shape[0] != 2) or \
           (len(y.shape) != 1) or (y.shape[0] != x.shape[1]):
            raise ValueError("Delaunay interpolator requires input data with "
                             "shape (2, N) and output data with shape (N,), "
                             "got %s and %s instead" % (x.shape, y.shape))
        if self.jitter:
            x = x + 0.00001 * x.std(axis=1)[:, np.newaxis] * \
                np.random.standard_normal(x.shape)
        if self.interp_type == 'cubic':
            self._interp = CloughTocher2DInterpolator(
                x.T, y, fill_value=self.default_val)
        elif self.interp_type == 'linear':
            self._interp = LinearNDInterpolator(
                x.T, y, fill_value=self.default_val)
        else:
            self._interp = NearestNDInterpolator(x.T, y)
        return self

    def __call__(self, x):
        """Evaluate function ``y = f(x)`` on new data.

        Evaluates the fitted scalar function on 2-D data provided in *x*.

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
            raise ValueError("Delaunay interpolator requires input data with "
                             "shape (2, M), got %s instead" % (x.shape,))
        if self._interp is None:
            raise NotFittedError("Interpolator function not fitted to data "
                                 "yet - first call .fit method")
        return self._interp(x.T)
