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

"""Utility functions.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

from builtins import range
import copy

import numpy as np
import scipy

# ----------------------------------------------------------------------------------------------------------------------
# --- FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------


def squash(x, flatten_axes, move_to_start=True):
    """Flatten array, but not necessarily all the way to a 1-D array.

    This helper function is useful for broadcasting functions of arbitrary
    dimensionality along a given array. The array x is transposed and reshaped,
    so that the axes with indices listed in *flatten_axes* are collected either
    at the start or end of the array (based on the *move_to_start* flag). These
    axes are also flattened to a single axis, while preserving the total number
    of elements in the array. The reshaping and transposition usually results
    in a view of the original array, although a copy may result e.g. if
    discontiguous *flatten_axes* are chosen. The two extreme cases are
    ``flatten_axes = []`` or None, which results in the original array with no
    flattening, and ``flatten_axes = range(len(x.shape))``, which is equivalent
    to ``x.ravel()`` and therefore full flattening.

    Parameters
    ----------
    x : array-like
        N-dimensional array to squash
    flatten_axes : list of ints
        List of axes along which *x* should be flattened
    move_to_start : bool, optional
        Flag indicating whether flattened axis is moved to start / end of array

    Returns
    -------
    y : array
        Semi-flattened version of *x*, as numpy array

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros((2, 4, 10))
    >>> # no flattening, x returned unchanged:
    >>> squash(x, [], True).shape
    (2, 4, 10)
    >>> squash(x, (1), True).shape
    (4, 2, 10)
    >>> squash(x, (1), False).shape
    (2, 10, 4)
    >>> squash(x, (0, 2), True).shape
    (20, 4)
    >>> squash(x, (0, 2), False).shape
    (4, 20)
    >>> # same as x.ravel():
    >>> squash(x, (0, 1, 2), True).shape
    (80,)

    """
    x = np.asarray(x)
    x_shape = np.atleast_1d(np.asarray(x.shape))
    # Split list of axes into those that will be flattened and the rest,
    # which are considered the main axes
    flatten_axes = np.atleast_1d(np.asarray(flatten_axes)).tolist()
    if flatten_axes == [None]:
        flatten_axes = []
    main_axes = list(set(range(len(x_shape))) - set(flatten_axes))
    # After flattening, the array will contain `flatten_shape` number of
    # `main_shape`-shaped subarrays
    flatten_shape = [x_shape[flatten_axes].prod()]
    # Don't add any singleton dimensions during flattening - rather leave
    # the matrix as is
    if flatten_shape == [1]:
        flatten_shape = []
    main_shape = x_shape[main_axes].tolist()
    # Move specified axes to the beginning (or end) of list of axes,
    # and transpose and reshape array accordingly
    if move_to_start:
        return x.transpose(
            flatten_axes + main_axes).reshape(flatten_shape + main_shape)
    else:
        return x.transpose(
            main_axes + flatten_axes).reshape(main_shape + flatten_shape)


def unsquash(x, flatten_axes, original_shape, move_from_start=True):
    """Restore an array that was reshaped by :func:`squash`.

    Parameters
    ----------
    x : array-like
        N-dimensional array to unsquash
    flatten_axes : List of ints
        List of (original) axes along which *x* was flattened
    original_shape : List of ints
        Original shape of *x*, before flattening
    move_from_start : bool, optional
        Flag indicating whether flattened axes were moved to array start / end

    Returns
    -------
    y : array, shape *original_shape*
        Restored version of *x*, as numpy array

    """
    x = np.asarray(x)
    original_shape = np.atleast_1d(np.asarray(original_shape))
    # Split list of axes into those that will be flattened and the rest,
    # which are considered the main axes
    flatten_axes = np.atleast_1d(np.asarray(flatten_axes)).tolist()
    if flatten_axes == [None]:
        flatten_axes = []
    main_axes = list(set(range(len(original_shape))) - set(flatten_axes))
    # After unflattening, the flatten_axes will be reconstructed with
    # the correct dimensionality
    unflatten_shape = original_shape[flatten_axes].tolist()
    # Don't add any singleton dimensions during flattening - rather
    # leave the matrix as is
    if unflatten_shape == [1]:
        unflatten_shape = []
    main_shape = original_shape[main_axes].tolist()
    # Move specified axes from the beginning (or end) of list of axes,
    # and transpose and reshape array accordingly
    if move_from_start:
        return x.reshape(unflatten_shape + main_shape).transpose(
            np.array(flatten_axes + main_axes).argsort())
    else:
        return x.reshape(main_shape + unflatten_shape).transpose(
            np.array(main_axes + flatten_axes).argsort())


def scalar(x):
    """Ensure that a variable contains a scalar.

    If `x` is a scalar, it is returned unchanged. If `x` is an array with a
    single element, that element is extracted. If `x` contains more than one
    element, an exception is raised.

    Parameters
    ----------
    x : object or array of shape () or shape (1, 1, ...)
        Scalar or array equivalent to a scalar

    Return
    ------
    scalar_x : object
        Original x or single element extracted from array

    Raises
    ------
    AssertionError
        If `x` contains more than one element

    """
    squeezed_x = np.squeeze(x)
    assert np.shape(squeezed_x) == (), "Expected array %s to be scalar" % (x,)
    return np.atleast_1d(squeezed_x)[0]


def sort_grid(x, y, z):
    """Ensure that the coordinates of a rectangular grid are in ascending order.

    Parameters
    ----------
    x : array, shape (M,)
        1-D array of x coordinates, in any order
    y : array, shape (N,)
        1-D array of y coordinates, in any order
    z : array, shape (M, N)
        2-D array of values which correspond to the coordinates in *x* and *y*

    Returns
    -------
    xx : array, shape (M,)
        1-D array of x coordinates, in ascending order
    yy : array, shape (N,)
        1-D array of y coordinates, in ascending order
    zz : array, shape (M, N)
        2-D array of values which correspond to coordinates in *xx* and *yy*

    """
    x_ind = np.argsort(x)
    y_ind = np.argsort(y)
    return x[x_ind], y[y_ind], z[x_ind, :][:, y_ind]


def desort_grid(x, y, z):
    """Undo the effect of :func:`sort_grid`.

    This shuffles a rectangular grid of values (based on ascending coordinates)
    to correspond to the original order.

    Parameters
    ----------
    x : array, shape (M,)
        1-D array of x coordinates, in the original (possibly unsorted) order
    y : array, shape (N,)
        1-D array of y coordinates, in the original (possibly unsorted) order
    z : array, shape (M, N)
        2-D array of values which correspond to sorted (x, y) coordinates

    Returns
    -------
    zz : array, shape (M, N)
        2-D array of values which correspond to original coordinates in x and y

    """
    x_ind = np.argsort(np.argsort(x))
    y_ind = np.argsort(np.argsort(y))
    return z[x_ind, :][:, y_ind]


def vectorize_fit_func(func):
    """Factory that creates vectorised version of function to be fitted to data.

    This takes functions of the form ``y = f(p, x)`` which cannot handle
    sequences of input arrays for ``x``, and wraps it in a loop which calls
    ``f`` with the column vectors of ``x``, and returns the corresponding
    outputs as an array of column vectors.

    Parameters
    ----------
    func : function, signature ``y = f(p, x)``
        Function ``f(p, x)`` to be vectorised along last dimension of ``x``

    Returns
    -------
    vec_func : function
        Vectorised version of function

    """
    def vec_func(p, x):
        # Move last dimension to front (becomes sequence of column arrays)
        column_seq_x = np.rollaxis(np.asarray(x), -1)
        # Get corresponding sequence of output column arrays
        column_seq_y = np.array([func(p, xx) for xx in column_seq_x])
        # Move column dimension back to the end
        return np.rollaxis(column_seq_y, 0, len(column_seq_y.shape))
    return vec_func


def randomise(interp, x, y, method='shuffle'):
    """Randomise fitted function parameters by resampling residuals.

    This allows estimation of the sampling distribution of the parameters of a
    fitted function, by repeatedly running this method and collecting the
    statistics (e.g. variance) of the parameters of the resulting interpolator
    object. Alternatively, it can form part of a bigger Monte Carlo run.

    The method assumes that the interpolator has already been fit to data. It
    obtains the residuals ``r = y - f(x)``, and resamples them to form ``r*``
    according to the specified method. The final step re-fits the interpolator
    to the pseudo-data ``(x, f(x) + r*)``, which yields a slightly different
    estimate of the function parameters every time the method is called.
    The method is therefore non-deterministic. Three resampling methods are
    supported:

    - 'shuffle': permute the residuals (sample from the residuals without
      replacement)
    - 'normal': replace the residuals with zero-mean Gaussian noise with the
      same variance
    - 'bootstrap': sample from the existing residuals, with replacement

    Parameters
    ----------
    interp : object
        The interpolator object to randomise (not clobbered by this method)
    x : array-like
        Known input values as a numpy array (typically the data to which the
        function was originally fitted)
    y : array-like
        Known output values as a numpy array (typically the data to which the
        function was originally fitted)
    method : {'shuffle', 'normal', 'bootstrap'}, optional
        Resampling technique used to resample residuals

    Returns
    -------
    random_interp : object
        Randomised interpolator object

    """
    # Make copy to prevent destruction of original interpolator
    random_interp = copy.deepcopy(interp)
    true_y = np.asarray(y)
    fitted_y = random_interp(x)
    residuals = true_y - fitted_y
    # Resample residuals
    if method == 'shuffle':
        np.random.shuffle(residuals.ravel())
    elif method == 'normal':
        residuals = residuals.std() * np.random.standard_normal(
            residuals.shape)
    elif method == 'bootstrap':
        sample = np.random.randint(residuals.size, size=residuals.size)
        residuals = residuals.ravel()[sample].reshape(residuals.shape)
    # Refit function on pseudo-data
    random_interp.fit(x, fitted_y + residuals)
    return random_interp


def pascal(n):
    """Create n-by-n upper triangular Pascal matrix.

    This square matrix contains Pascal's triangle as its upper triangle. For
    example, for n=5 the output will be::

      1 1 1 1 1
      0 1 2 3 4
      0 0 1 3 6
      0 0 0 1 4
      0 0 0 0 1

    Parameters
    ----------
    n : integer
        Positive integer indicating size of desired matrix

    Returns
    -------
    u : array of float, shape (n, n)
        Upper triangular Pascal matrix

    Notes
    -----
    For more details on the Pascal matrix, see the Wikipedia entry [1]_. The
    matrix is calculated using matrix exponentiation of a superdiagonal matrix.
    Although it theoretically consists of integers only, the matrix entries
    grow factorially with *n* and typically overflow the integer representation
    for n > 100. A less exact floating-point representation is therefore used
    instead (similar to setting exact=0 in :func:`scipy.factorial`).

    .. [1] http://en.wikipedia.org/wiki/Pascal_matrix

    Examples
    --------
    >>> pascal(4)
    array([[ 1.,  1.,  1.,  1.],
           [ 0.,  1.,  2.,  3.],
           [ 0.,  0.,  1.,  3.],
           [ 0.,  0.,  0.,  1.]])

    """
    # Create special superdiagonal matrix X
    x = np.diag(np.arange(1., n), 1)
    # Evaluate matrix exponential Un = exp(X) via direct series expansion,
    # since X is nilpotent
    # That is, Un = I + X + X^2 / 2! + X^3 / 3! + ... + X^(n-1) / (n-1)!
    term = x[:]
    # The first two terms [I + X] are trivial
    u = np.eye(n) + term
    # Accumulate the series terms
    for k in range(2, n - 1):
        term = np.dot(term, x) / k
        u += term
    # The last term [X^(n-1) / (n-1)!] is also trivial - a zero matrix
    # with a single one in the top right corner
    u[0, -1] = 1.
    return u


def offset_scale_mat(n, offset=0., scale=1.):
    r"""Matrix that transforms polynomial coefficients to account for offset/scale.

    This matrix can be used to transform a vector of polynomial coefficients
    that operate on scaled and shifted data to a vector of coefficients that
    perform the same action on the unscaled and unshifted data. The offset and
    scale factor is thereby incorporated into the polynomial coefficients.

    Given two *n*-dimensional vectors of coefficients (highest order first),
    :math:`p` and :math:`q`, related by

    .. math:: \sum_{i=0}^{n-1} p_i \left(\frac{x - m}{s}\right)^{n-1-i} = \sum_{k=0}^{n-1} q_k x^{n-1-k},

    with offset :math:`m` and scale :math:`s`, this calculates the matrix
    :math:`M` so that :math:`q = M p`.

    Parameters
    ----------
    n : integer
        Number of polynomial coefficients, equal to (degree + 1)
    offset : float, optional
        Offset that is subtracted from data
    scale : float, optional
        Data is divided by this scale

    Returns
    -------
    M : array of float, shape (n, n)
        Resulting transformation matrix

    Examples
    --------
    >>> offset_scale_mat(4, 3, 2)
    array([[ 0.125,  0.   ,  0.   ,  0.   ],
           [-1.125,  0.25 ,  0.   ,  0.   ],
           [ 3.375, -1.5  ,  0.5  ,  0.   ],
           [-3.375,  2.25 , -1.5  ,  1.   ]])

    """
    poly_offset = np.fliplr(np.vander([-offset], n))
    offset_mat = scipy.linalg.toeplitz(poly_offset, np.r_[1., np.zeros(n - 1)])
    poly_scale = np.vander([scale], n)
    return np.fliplr(np.flipud(pascal(n))) * offset_mat / poly_scale
