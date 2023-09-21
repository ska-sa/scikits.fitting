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

"""Generic fitters and base classes.

:author: Ludwig Schwardt
:license: Modified BSD

"""

from builtins import range
from builtins import object
import copy

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# --- EXCEPTIONS
# ----------------------------------------------------------------------------------------------------------------------


class NotFittedError(Exception):
    """Fitter was called with new data before being fit to existing data."""

# ----------------------------------------------------------------------------------------------------------------------
# --- INTERFACE :  ScatterFit
# ----------------------------------------------------------------------------------------------------------------------


class ScatterFit(object):
    """Interface for interpolators that operate on scattered data (not on grid).

    This defines the interface for interpolator functions that operate on
    unstructured scattered input data (i.e. not on a grid). The input data
    consists of a sequence of ``x`` coordinates and a sequence of corresponding
    ``y`` data, where the order of the ``x`` coordinates does not matter and
    their location can be arbitrary. The ``x`` coordinates can have an
    arbritrary dimension (although most classes are specialised for 1-D or 2-D
    data), in which case they are given as column vectors in the input array.

    The initialiser should be used to specify parameters of the interpolator
    function, such as polynomial degree.

    """
    def __init__(self):
        pass

    def fit(self, x, y):
        """Fit function ``y = f(x)`` to data.

        This function should reset any state associated with previous
        ``(x, y)`` data fits, and preserve all state that was set by the
        initialiser.

        Parameters
        ----------
        x : array-like, shape (N,) for 1-D data, or (D, N) otherwise
            Known input values as sequence or ndarray (order does not matter)
        y : array-like, shape (N,)
            Known output values as sequence or ndarray

        Returns
        -------
        self : :class:`ScatterFit` object
            Reference to self, to allow chaining of method calls

        """
        raise NotImplementedError

    def __call__(self, x):
        """Evaluate function ``y = f(x)`` on new data.

        Parameters
        ----------
        x : array-like, shape (M,) for 1-D data, or (D, M) otherwise
            Input to function as sequence or ndarray (order does not matter)

        Returns
        -------
        y : array, shape (M,)
            Output of function as a numpy array

        """
        raise NotImplementedError

    def eval(self, x):
        """Evaluate function on new data. See __call__ docstring for help."""
        return self.__call__(x)

# ----------------------------------------------------------------------------------------------------------------------
# --- INTERFACE :  GridFit
# ----------------------------------------------------------------------------------------------------------------------


class GridFit(object):
    """Interface for interpolators that operate on data on a grid.

    This defines the interface for interpolator functions that operate on input
    data that lie on a grid. The input data consists of a sequence of x-axis
    tick sequences and the corresponding array of y data. The shape of this
    array matches the corresponding lengths of the axis tick sequences.
    The axis tick sequences are assumed to be in ascending order. The ``x``
    sequence can contain an arbitrary number of axes of different lengths
    (although most classes are specialised for 1-D or 2-D data).

    The initialiser should be used to specify parameters of the interpolator
    function, such as polynomial degree.

    """
    def __init__(self):
        pass

    def fit(self, x, y):
        """Fit function ``y = f(x)`` to data.

        This function should reset any state associated with previous
        ``(x, y)`` data fits, and preserve all state that was set by the
        initialiser.

        Parameters
        ----------
        x : sequence of array-likes, length D
            Known axis tick values as a sequence of numpy arrays (each in
            ascending order) with corresponding lengths n_1, n_2, ..., n_D
        y : array-like, shape (n_1, n_2, ..., n_D)
            Known output values as a D-dimensional numpy array

        Returns
        -------
        self : :class:`GridFit` object
            Reference to self, to allow chaining of method calls

        """
        raise NotImplementedError

    def __call__(self, x):
        """Evaluate function ``y = f(x)`` on new data.

        Parameters
        ----------
        x : sequence of array-likes, length D
            Input to function as a sequence of numpy arrays (each in ascending
            order) with corresponding lengths m_1, m_2, ..., m_D

        Returns
        -------
        y : array, shape (m_1, m_2, ..., m_D)
            Output of function as a D-dimensional numpy array

        """
        raise NotImplementedError

    def eval(self, x):
        """Evaluate function on new data. See __call__ docstring for help."""
        return self.__call__(x)

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  Independent1DFit
# ----------------------------------------------------------------------------------------------------------------------


class Independent1DFit(ScatterFit):
    """Interpolate a D-dimensional matrix along a given axis, using a set of
    independent 1-D interpolators.

    This simplifies the simultaneous interpolation of a set of one-dimensional
    ``x-y`` relationships. It assumes that ``x`` is 1-D, while ``y`` is
    D-dimensional and to be independently interpolated along ``D-1`` of its
    dimensions.

    Parameters
    ----------
    interp : object
        ScatterFit object to be cloned into an array of interpolators
    axis : int
        Axis of ``y`` matrix which will vary with independent ``x`` variable

    """
    def __init__(self, interp, axis):
        ScatterFit.__init__(self)
        self._interp = interp
        self._axis = axis
        # Array of interpolators, only set after ``fit``
        self._interps = None

    def fit(self, x, y):
        """Fit a set of stored interpolators to one axis of *y* matrix.

        Parameters
        ----------
        x : array-like, shape (N,)
            Known input values as a 1-D numpy array or sequence
        y : array-like, shape (d_1, d_2, ..., N, ..., d_D)
            Known output values as a D-dimensional numpy array

        Returns
        -------
        self : :class:`Independent1DFit` object
            Reference to self, to allow chaining of method calls

        """
        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if self._axis >= len(y.shape):
            raise ValueError("Provided y-array does not have the "
                             "specified axis %d" % (self._axis,))
        if y.shape[self._axis] != len(x):
            raise ValueError("Number of elements in x and along "
                             "specified axis of y differ")
        # Shape of array of interpolators
        # (same shape as y, but without 'independent' specified axis)
        interp_shape = list(y.shape)
        interp_shape.pop(self._axis)
        # Create blank array of interpolators
        self._interps = np.ndarray(interp_shape, dtype=type(self._interp))
        num_interps = np.array(interp_shape).prod()
        # Move specified axis to the end of list
        new_axis_order = list(range(len(y.shape)))
        new_axis_order.pop(self._axis)
        new_axis_order.append(self._axis)
        # Rearrange to form 2-D array of data and 1-D array of interpolators
        flat_y = y.transpose(new_axis_order).reshape(num_interps, len(x))
        flat_interps = self._interps.ravel()
        # Clone basic interpolator and fit x and each row
        # of the flattened y matrix independently
        for n in range(num_interps):
            flat_interps[n] = copy.deepcopy(self._interp)
            flat_interps[n].fit(x, flat_y[n])
        return self

    def __call__(self, x):
        """Evaluate set of interpolator functions on new data.

        Parameters
        ----------
        x : array-like, shape (M,)
            Input to function as a 1-D numpy array or sequence

        Returns
        -------
        y : array, shape (d_1, d_2, ..., M, ..., d_D)
            Output of function as a D-dimensional numpy array

        """
        if self._interps is None:
            raise NotFittedError("Interpolator functions not fitted to data "
                                 "yet - first call .fit method")
        x = np.atleast_1d(np.asarray(x))
        # Create blank output array with specified axis appended at shape end
        out_shape = list(self._interps.shape)
        out_shape.append(len(x))
        y = np.ndarray(out_shape)
        num_interps = np.array(self._interps.shape).prod()
        # Rearrange to form 2-D array of data and 1-D array of interpolators
        flat_y = y.reshape(num_interps, len(x))
        assert flat_y.base is y, ("Reshaping array resulted in a copy instead "
                                  "of a view - bad news for this code...")
        flat_interps = self._interps.ravel()
        # Apply each interpolator to x and store in appropriate row of y
        for n in range(num_interps):
            flat_y[n] = flat_interps[n](x)
        # Create list of indices that will move specified axis
        # from last place to correct location
        new_axis_order = list(range(len(out_shape)))
        new_axis_order.insert(self._axis, new_axis_order.pop())
        return y.transpose(new_axis_order)
