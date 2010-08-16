"""Unified interface to SciPy function fitting routines.

This module provides a unified interface to function fitting. All interpolation
routines conform to the following simple method interface:

- __init__(p) : set parameters of interpolation function, e.g. polynomial degree
- fit(x, y) : fit given input-output data
- __call__(x) : evaluate function on new input data

Each interpolation routine falls in one of two categories: scatter fitting or
grid fitting. They share the same interface, only differing in the definition
of input data x.

Scatter-fitters operate on unstructured scattered input data (i.e. not on a
grid). The input data consists of a sequence of ``x`` coordinates and a sequence
of corresponding ``y`` data, where the order of the ``x`` coordinates does not
matter and their location can be arbitrary. The ``x`` coordinates can have an
arbritrary dimension (although most classes are specialised for 1-D or 2-D
data). If the dimension is bigger than 1, the coordinates are provided as an
array of column vectors. These fitters have ScatterFit as base class.

Grid-fitters operate on input data that lie on a grid. The input data consists
of a sequence of x-axis tick sequences and the corresponding array of y data.
These fitters have GridFit as base class.

The module is organised as follows:

Scatter fitters
---------------

- :class:`ScatterFit` : Abstract base class for scatter fitters
- :class:`Polynomial1DFit` : Fit polynomial to 1-D data
- :class:`Polynomial2DFit` : Fit polynomial to 2-D data
- :class:`PiecewisePolynomial1DFit` : Fit piecewise polynomial to 1-D data
- :class:`ReciprocalFit` : Interpolate the reciprocal of data
- :class:`Independent1DFit` : Interpolate N-dimensional matrix along given axis
- :class:`Delaunay2DScatterFit` : Interpolate scalar function of 2-D data, based on
                                  Delaunay triangulation (scattered data version)
- :class:`NonLinearLeastSquaresFit` : Fit a generic function to data, based on
                                      non-linear least squares optimisation.
- :class:`GaussianFit` : Fit Gaussian curve to multi-dimensional data
- :class:`Spline1DFit` : Fit a B-spline to 1-D data
- :class:`Spline2DScatterFit` : Fit a B-spline to scattered 2-D data
- :class:`RbfScatterFit` : Do radial basis function (RBF) interpolation

Grid fitters
------------

- :class:`GridFit` : Abstract base class for grid fitters
- :class:`Delaunay2DGridFit` : Interpolate scalar function defined on 2-D grid,
                               based on Delaunay triangulation
- :class:`Spline2DGridFit` : Fit a B-spline to 2-D data on a rectangular grid

Helper functions
----------------

- :func:`squash` : Flatten array, but not necessarily all the way to a 1-D array
- :func:`unsquash' : Restore an array that was reshaped by :func:`squash`
- :func:`sort_grid` : Ensure that the coordinates of a rectangular grid are in
                      ascending order
- :func:`desort_grid` : Undo the effect of :func:`sort_grid`
- :func:`vectorize_fit_func` : Factory that creates vectorised version of
                               function to be fitted to data
- :func:`randomise` : Randomise fitted function parameters by resampling
                      residuals

"""

import copy
import logging

import numpy as np
import scipy.optimize       # NonLinearLeastSquaresFit
import scipy.interpolate    # Spline1DFit, Spline2DScatterFit, Spline2DGridFit,
                            # RbfScatterFit, PiecewisePolynomial1DFit
# Since scipy 0.7.0 the delaunay module lives in scikits
try:
    # pylint: disable-msg=E0611
    import scikits.delaunay as delaunay   # Delaunay2DScatterFit, Delaunay2DGridFit
    delaunay_found = True
except ImportError:
    # In scipy 0.6.0 and before, the delaunay module is in the sandbox
    try:
        # pylint: disable-msg=E0611,F0401
        import scipy.sandbox.delaunay as delaunay
        delaunay_found = True
    except ImportError:
        # Matplotlib also has delaunay module these days - use as last resort (more convenient than scikits)
        try:
            # pylint: disable-msg=E0611,F0401
            import matplotlib.delaunay as delaunay
            delaunay_found = True
        except ImportError:
            delaunay_found = False

logger = logging.getLogger("scape.fitting")

#----------------------------------------------------------------------------------------------------------------------
#--- EXCEPTIONS
#----------------------------------------------------------------------------------------------------------------------

class NotFittedError(Exception):
    """Error: Fitter was called with new data before being fit to existing data."""
    pass

#----------------------------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------

def squash(x, flatten_axes, move_to_start=True):
    """Flatten array, but not necessarily all the way to a 1-D array.

    This helper function is useful for broadcasting functions of arbitrary
    dimensionality along a given array. The array x is transposed and reshaped,
    so that the axes with indices listed in *flatten_axes* are collected either
    at the start or end of the array (based on the *move_to_start* flag). These
    axes are also flattened to a single axis, while preserving the total number
    of elements in the array. The reshaping and transposition usually results in
    a view of the original array, although a copy may result e.g. if
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
        Flag indicating whether flattened axis is moved to start or end of array

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
    # Split list of axes into those that will be flattened and the rest, which are considered the main axes
    flatten_axes = np.atleast_1d(np.asarray(flatten_axes)).tolist()
    if flatten_axes == [None]:
        flatten_axes = []
    main_axes = list(set(range(len(x_shape))) - set(flatten_axes))
    # After flattening, the array will contain `flatten_shape` number of `main_shape`-shaped subarrays
    flatten_shape = [x_shape[flatten_axes].prod()]
    # Don't add any singleton dimensions during flattening - rather leave the matrix as is
    if flatten_shape == [1]:
        flatten_shape = []
    main_shape = x_shape[main_axes].tolist()
    # Move specified axes to the beginning (or end) of list of axes, and transpose and reshape array accordingly
    if move_to_start:
        return x.transpose(flatten_axes + main_axes).reshape(flatten_shape + main_shape)
    else:
        return x.transpose(main_axes + flatten_axes).reshape(main_shape + flatten_shape)

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
        Flag indicating whether flattened axes were moved to start or end of array

    Returns
    -------
    y : array, shape *original_shape*
        Restored version of *x*, as numpy array

    """
    x = np.asarray(x)
    original_shape = np.atleast_1d(np.asarray(original_shape))
    # Split list of axes into those that will be flattened and the rest, which are considered the main axes
    flatten_axes = np.atleast_1d(np.asarray(flatten_axes)).tolist()
    if flatten_axes == [None]:
        flatten_axes = []
    main_axes = list(set(range(len(original_shape))) - set(flatten_axes))
    # After unflattening, the flatten_axes will be reconstructed with the correct dimensionality
    unflatten_shape = original_shape[flatten_axes].tolist()
    # Don't add any singleton dimensions during flattening - rather leave the matrix as is
    if unflatten_shape == [1]:
        unflatten_shape = []
    main_shape = original_shape[main_axes].tolist()
    # Move specified axes from the beginning (or end) of list of axes, and transpose and reshape array accordingly
    if move_from_start:
        return x.reshape(unflatten_shape + main_shape).transpose(np.array(flatten_axes + main_axes).argsort())
    else:
        return x.reshape(main_shape + unflatten_shape).transpose(np.array(main_axes + flatten_axes).argsort())


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
        2-D array of values which correspond to the coordinates in *xx* and *yy*

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
        Function ``f(p, x)`` to be vectorised along last dimension of input ``x``

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
        residuals = residuals.std() * np.random.standard_normal(residuals.shape)
    elif method == 'bootstrap':
        residuals = residuals.ravel()[np.random.randint(residuals.size, size=residuals.size)].reshape(residuals.shape)
    # Refit function on pseudo-data
    random_interp.fit(x, fitted_y + residuals)
    return random_interp

#----------------------------------------------------------------------------------------------------------------------
#--- INTERFACE :  ScatterFit
#----------------------------------------------------------------------------------------------------------------------

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

        This function should reset any state associated with previous ``(x, y)``
        data fits, and preserve all state that was set by the initialiser.

        Parameters
        ----------
        x : array-like, shape (N,) for 1-D data, or (D, N) otherwise
            Known input values as sequence or numpy array (order does not matter)
        y : array-like, shape (N,)
            Known output values as sequence or numpy array

        """
        raise NotImplementedError

    def __call__(self, x):
        """Evaluate function ``y = f(x)`` on new data.

        Parameters
        ----------
        x : array-like, shape (M,) for 1-D data, or (D, M) otherwise
            Input to function as sequence or numpy array (order does not matter)

        Returns
        -------
        y : array, shape (M,)
            Output of function as a numpy array

        """
        raise NotImplementedError

#----------------------------------------------------------------------------------------------------------------------
#--- INTERFACE :  GridFit
#----------------------------------------------------------------------------------------------------------------------

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

        This function should reset any state associated with previous ``(x, y)``
        data fits, and preserve all state that was set by the initialiser.

        Parameters
        ----------
        x : sequence of array-likes, length D
            Known axis tick values as a sequence of numpy arrays (each in
            ascending order) with corresponding lengths n_1, n_2, ..., n_D
        y : array-like, shape (n_1, n_2, ..., n_D)
            Known output values as a D-dimensional numpy array

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

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Polynomial1DFit
#----------------------------------------------------------------------------------------------------------------------

class Polynomial1DFit(ScatterFit):
    """Fit polynomial to 1-D data.

    This uses numpy's polyfit and polyval.

    Parameters
    ----------
    max_degree : int, non-negative
        Maximum polynomial degree to use (automatically reduced if there are not
        enough data points)
    rcond : float, optional
        Relative condition number of fit (smallest singular value that will be
        used to fit polynomial, has sensible default)

    Arguments
    ---------
    poly : real array
        Polynomial coefficients (highest order first), only set after :func:`fit`

    """
    def __init__(self, max_degree, rcond=None):
        ScatterFit.__init__(self)
        self.max_degree = max_degree
        self._rcond = rcond
        # Mean of input data, only set after :func:`fit`
        self._mean = None
        self.poly = None

    def fit(self, x, y):
        """Fit polynomial to data.

        Parameters
        ----------
        x : array-like, shape (N,)
            Known input values as a 1-D numpy array or sequence
        y : array-like, shape (N,)
            Known output values as a 1-D numpy array, or sequence

        """
        # Upcast x and y to doubles, to ensure a high enough precision for the polynomial coefficients
        x = np.atleast_1d(np.array(x, dtype='double'))
        y = np.atleast_1d(np.array(y, dtype='double'))
        # Polynomial fits perform better if input data is centred around origin [see numpy.polyfit help]
        self._mean = x.mean()
        # Reduce polynomial degree if there are not enough points to fit (degree should be < len(x))
        self.poly = np.polyfit(x - self._mean, y, min((self.max_degree, len(x) - 1)), rcond = self._rcond)

    def __call__(self, x):
        """Evaluate polynomial on new data.

        Parameters
        ----------
        x : array-like, shape (M,)
            Input to function as a 1-D numpy array, or sequence

        Returns
        -------
        y : array, shape (M,)
            Output of function as a 1-D numpy array

        """
        if (self.poly is None) or (self._mean is None):
            raise NotFittedError("Polynomial not fitted to data yet - first call .fit method")
        x = np.atleast_1d(np.asarray(x))
        return np.polyval(self.poly, x - self._mean)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Polynomial2DFit
#----------------------------------------------------------------------------------------------------------------------

class Polynomial2DFit(ScatterFit):
    """Fit polynomial to 2-D data.

    This extends :func:`numpy.polyfit` and :func:`numpy.polyval` to 2 dimensions.
    The 2-D polynomial has (degrees[0] + 1) * (degrees[1] + 1) coefficients.

    Parameters
    ----------
    degrees : list of 2 ints
        Non-negative polynomial degree to use for each dimension of *x*

    Arguments
    ---------
    poly : real array, shape ((degrees[0] + 1) * (degrees[1] + 1),)
        Polynomial coefficients (highest order first), only set after :func:`fit`

    """
    def __init__(self, degrees):
        ScatterFit.__init__(self)
        self.degrees = degrees
        # Mean and scale of input data, only set after :func:`fit`
        self._mean = None
        self._scale = None
        self.poly = None

    def _regressor(self, x):
        """Form normalised regressor matrix from set of input vectors.

        Parameters
        ----------
        x : array, shape (2, N)
            Input to function as a 2-D numpy array

        Returns
        -------
        X : array, shape (N, (degrees[0] + 1) * (degrees[1] + 1))
            Regressor matrix

        Notes
        -----
        This normalises the 2-D input vectors by centering and scaling them.
        It then forms a regressor matrix with a row per input vector. Each row
        is given by the outer product of the monomials of the first dimension
        with the monomials of the second dimension of the input vector, in
        decreasing polynomial order. For example, if *degrees* is (1, 2) and
        the elements of each input vector in *x* are *x_0* and *x_1*,
        respectively, the row takes the form::

            outer([x_0, 1], [x1 ^ 2, x1, 1])
            = [x_0 * x_1 ^ 2, x_0 * x_1, x_0 * 1, 1 * x_1 ^ 2, 1 * x_1, 1 * 1]
            = [x_0 * x_1 ^ 2, x_0 * x_1, x_0, x_1 ^ 2, x_1, 1]

        This is closely related to the Vandermonde matrix of *x*.

        """
        x_norm = (x - self._mean[:, np.newaxis]) / self._scale[:, np.newaxis]
        v1 = np.vander(x_norm[0], self.degrees[0] + 1)
        v2 = np.vander(x_norm[1], self.degrees[1] + 1)
        return np.hstack([v1[:, n][:, np.newaxis] * v2 for n in xrange(v1.shape[1])])

    def fit(self, x, y):
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

        """
        # Upcast x and y to doubles, to ensure a high enough precision for the polynomial coefficients
        x = np.atleast_2d(np.array(x, dtype='double'))
        y = np.atleast_1d(np.array(y, dtype='double'))
        # Polynomial fits perform better if input data is centred around origin and scaled [see numpy.polyfit help]
        self._mean = x.mean(axis=1)
        self._scale = np.abs(x - self._mean[:, np.newaxis]).max(axis=1)
        self._scale[self._scale == 0.0] = 1.0
        # Solve least squares regression problem
        results = np.linalg.lstsq(self._regressor(x), y)
        poly, rank = results[0], results[2]
        if rank != len(poly):
            logger.warning('Polynomial fit may be poorly conditioned')
        self.poly = poly

    def __call__(self, x):
        """Evaluate polynomial on new data.

        Parameters
        ----------
        x : array-like, shape (2, M)
            Input to function as a 2-D numpy array, or sequence

        Returns
        -------
        y : array, shape (M,)
            Output of function as a 1-D numpy array

        """
        if (self.poly is None) or (self._mean is None):
            raise NotFittedError("Polynomial not fitted to data yet - first call .fit method")
        x = np.atleast_2d(np.asarray(x))
        return np.dot(self._regressor(x), self.poly)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  PiecewisePolynomial1DFit
#----------------------------------------------------------------------------------------------------------------------

def _stepwise_interp(xi, yi, x):
    """Step-wise interpolate (or extrapolate) (xi, yi) values to x positions.

    Given a set of N ``(x, y)`` points, provided in the *xi* and *yi* arrays,
    this will calculate ``y``-coordinate values for a set of M ``x``-coordinates
    provided in the *x* array, using step-wise (zeroth-order) interpolation and
    extrapolation.

    The input *x* coordinates are compared to the fixed *xi* values, and the
    largest *xi* value smaller than or approximately equal to each *x* value is
    selected. The corresponding *yi* value is then returned. For *x* values
    below the entire set of *xi* values, the smallest *xi* value is selected.
    The steps of the interpolation therefore start at each *xi* value and extends
    to the right (above it) until the next bigger *xi*, except for the first
    step, which extends to the left (below it) as well, and the last step, which
    extends until positive infinity.

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
    # with the last segment (order important, otherwise last segment will be ignored)
    end[end == len(xi)] -= 1

    # First get largest "equality" difference tolerated for x and xi (set to zero for integer types)
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
    # Find x that are exactly equal to some xi or slightly below it, which will assign it to the wrong segment
    equal_or_just_below = xi[end] - x < smallest_diff
    # Move these segments one higher (except for the last one, which stays put)
    start[equal_or_just_below] = end[equal_or_just_below]
    # Ensure that output y has same shape as input x (especially, let scalar input result in scalar output)
    start = np.reshape(start, np.shape(x))
    return yi[start]

def _linear_interp(xi, yi, x):
    """Linearly interpolate (or extrapolate) (xi, yi) values to x positions.

    Given a set of N ``(x, y)`` points, provided in the *xi* and *yi* arrays,
    this will calculate ``y``-coordinate values for a set of M ``x``-coordinates
    provided in the *x* array, using linear interpolation and extrapolation.

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
    # Associate any x found outside xi range with closest segment (first or last one)
    # This linearly extrapolates the first and last segment to -inf and +inf, respectively
    end[end == 0] += 1
    end[end == len(xi)] -= 1
    start = end - 1
    # Ensure that output y has same shape as input x (especially, let scalar input result in scalar output)
    start, end = np.reshape(start, np.shape(x)), np.reshape(end, np.shape(x))
    # Set up weight such that xi[start] => 0 and xi[end] => 1
    end_weight = (x - xi[start]) / (xi[end] - xi[start])
    return (1.0 - end_weight) * yi[start] + end_weight * yi[end]

class PiecewisePolynomial1DFit(ScatterFit):
    """Fit piecewise polynomial to 1-D data.

    This fits a series of polynomials between adjacent points in a
    one-dimensional data set. The resulting piecewise polynomial curve passes
    exactly through the given data points and may also match the local gradient
    at each point if the maximum polynomial degree, *max_degree*, is at least 3.

    If *max_degree* is 0, step-wise interpolation is done between the points in
    the data set. Each input *x* value is assigned the *y* value of the largest
    *x* value in the data set that is smaller than or equal to the input *x*. If
    the input *x* is smaller than all the *x* values in the data set, the *y*
    value of the smallest data set *x* value is chosen instead.

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

        Raises
        ------
        ValueError
            If *x* contains duplicate values, which leads to infinite gradients

        """
        # Upcast x and y to doubles, to ensure a high enough precision for the polynomial coefficients
        x = np.atleast_1d(np.array(x, dtype='double'))
        # Only upcast y if numerical interpolation will actually happen - since stepwise interpolation
        # simply copies y values, this allows interpolation of non-numeric types (e.g. strings)
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
            raise ValueError("Two consecutive points have same x-coordinate - infinite gradient not allowed")
        # Maximum derivative order warranted by polynomial degree and number of data points
        max_deriv = min((self.max_degree - 1) // 2, len(x) - 2) + 1
        if max_deriv > 1:
            # Length of x interval straddling each data point (from previous to next point)
            x_interval = np.convolve(np.diff(x), [1.0, 1.0], 'valid')
            y_deriv = y
        # Recursively calculate the n'th derivative of y, up to maximum order
        for n in xrange(1, max_deriv):
            # The difference between (n-1)'th derivative of y at previous and next point, divided by interval
            y_deriv = np.convolve(np.diff(y_deriv), [1.0, 1.0], 'valid') / x_interval
            x_interval = x_interval[1:-1]
            for m in xrange(len(y_deriv)):
                y_list[m + n].append(y_deriv[m])
        if len(x) == 1:
            # Constant interpolation to all new x values
            self._poly = lambda new_x: np.tile(y[0], np.asarray(new_x).shape)
        elif self.max_degree == 0:
            # SciPy PiecewisePolynomial does not support degree 0 - use home-brewed interpolator instead
            self._poly = lambda new_x: _stepwise_interp(x, y, np.asarray(new_x))
        elif self.max_degree == 1:
            # Home-brewed linear interpolator is *way* faster than SciPy 0.7.0 PiecewisePolynomial
            self._poly = lambda new_x: _linear_interp(x, y, np.asarray(new_x))
        else:
            try:
                self._poly = scipy.interpolate.PiecewisePolynomial(x, y_list, orders=None, direction=1)
            except AttributeError:
                raise ImportError("SciPy 0.7.0 or newer needed for higher-order piecewise polynomials")

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
            raise NotFittedError("Piecewise polynomial not fitted to data yet - first call .fit method")
        return self._poly(x)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  ReciprocalFit
#----------------------------------------------------------------------------------------------------------------------

class ReciprocalFit(ScatterFit):
    """Interpolate the reciprocal of data.

    This allows any ScatterFit object to fit the reciprocal of a data set,
    without having to invert the data and the results explicitly.

    Parameters
    ----------
    interp : object
        ScatterFit object to use on the reciprocal of the data

    """
    def __init__(self, interp):
        ScatterFit.__init__(self)
        self._interp = copy.deepcopy(interp)

    def fit(self, x, y):
        """Fit stored interpolator to reciprocal of data, i.e. fit function ``1/y = f(x)``.

        Parameters
        ----------
        x : array-like
            Known input values as a numpy array
        y : array-like
            Known output values as a numpy array

        """
        y = np.asarray(y)
        self._interp.fit(x, 1.0 / y)

    def __call__(self, x):
        """Evaluate function ``1/f(x)`` on new data, where f is interpolated from previous data.

        Parameters
        ----------
        x : array-like
            Input to function as a numpy array

        Returns
        -------
        y : array
            Output of function as a numpy array

        """
        return 1.0 / self._interp(x)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Independent1DFit
#----------------------------------------------------------------------------------------------------------------------

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
        Axis of ``y`` matrix which will vary with the independent ``x`` variable

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

        """
        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if self._axis >= len(y.shape):
            raise ValueError("Provided y-array does not have the specified axis %d" % (self._axis,))
        if y.shape[self._axis] != len(x):
            raise ValueError("Number of elements in x and along specified axis of y differ")
        # Shape of array of interpolators (same shape as y, but without 'independent' specified axis)
        interp_shape = list(y.shape)
        interp_shape.pop(self._axis)
        # Create blank array of interpolators
        self._interps = np.ndarray(interp_shape, dtype=type(self._interp))
        num_interps = np.array(interp_shape).prod()
        # Move specified axis to the end of list
        new_axis_order = range(len(y.shape))
        new_axis_order.pop(self._axis)
        new_axis_order.append(self._axis)
        # Rearrange to form 2-D array of data and 1-D array of interpolators
        flat_y = y.transpose(new_axis_order).reshape(num_interps, len(x))
        flat_interps = self._interps.ravel()
        # Clone basic interpolator and fit x and each row of the flattened y matrix independently
        for n in range(num_interps):
            flat_interps[n] = copy.deepcopy(self._interp)
            flat_interps[n].fit(x, flat_y[n])


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
            raise NotFittedError("Interpolator functions not fitted to data yet - first call .fit method")
        x = np.atleast_1d(np.asarray(x))
        # Create blank output array with specified axis appended at the end of shape
        out_shape = list(self._interps.shape)
        out_shape.append(len(x))
        y = np.ndarray(out_shape)
        num_interps = np.array(self._interps.shape).prod()
        # Rearrange to form 2-D array of data and 1-D array of interpolators
        flat_y = y.reshape(num_interps, len(x))
        assert flat_y.base is y, "Reshaping array resulted in a copy instead of a view - bad news for this code..."
        flat_interps = self._interps.ravel()
        # Apply each interpolator to x and store in appropriate row of y
        for n in range(num_interps):
            flat_y[n] = flat_interps[n](x)
        # Create list of indices that will move specified axis from last place to correct location
        new_axis_order = range(len(out_shape))
        new_axis_order.insert(self._axis, new_axis_order.pop())
        return y.transpose(new_axis_order)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Delaunay2DScatterFit
#----------------------------------------------------------------------------------------------------------------------

class Delaunay2DScatterFit(ScatterFit):
    """Interpolate scalar function of 2-D data, based on Delaunay triangulation
    (scattered data version).

    The *x* data for this object should have two rows, containing the 'x' and
    'y' coordinates of points in a plane. The 2-D points are therefore stored as
    column vectors in *x*. The *y* data for this object is a 1-D array, which
    represents the scalar 'z' value of the function defined on the plane
    (the symbols in quotation marks are the names for these variables used in
    the ``delaunay`` documentation). The 2-D *x* coordinates do not have to
    lie on a regular grid, and can be in any order. Jittering a regular grid
    seems to be troublesome, though...

    Parameters
    ----------
    interp_type : {'nn'}, optional
        String indicating type of interpolation (only 'nn' currently supported)
    default_val : float, optional
        Default value used when trying to extrapolate beyond convex hull of
        known data
    jitter : bool, optional
        True to add small amount of jitter to *x* to make degenerate
        triangulation unlikely

    """
    def __init__(self, interp_type='nn', default_val=np.nan, jitter=False):
        if not delaunay_found:
            raise ImportError("Delaunay module not found - install it from " +
                              "scikits (or recompile SciPy <= 0.6.0 with sandbox enabled)")
        ScatterFit.__init__(self)
        if interp_type != 'nn':
            raise ValueError("Only 'nn' interpolator currently supports unstructured data not on a regular grid...")
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

        """
        # Check dimensions of known data
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if (len(x.shape) != 2) or (x.shape[0] != 2) or (len(y.shape) != 1) or (y.shape[0] != x.shape[1]):
            raise ValueError("Delaunay interpolator requires input data with shape (2, N) and " +
                             "output data with shape (N,), got %s and %s instead" % (x.shape, y.shape))
        if self.jitter:
            x = x + 0.00001 * x.std(axis=1)[:, np.newaxis] * np.random.standard_normal(x.shape)
        try:
            tri = delaunay.Triangulation(x[0], x[1])
        # This triangulation package is not very robust - in case of error, try once more, with fresh jitter
        except KeyError:
            x = x + 0.00001 * x.std(axis=1)[:, np.newaxis] * np.random.standard_normal(x.shape)
            tri = delaunay.Triangulation(x[0], x[1])
        if self.interp_type == 'nn':
            self._interp = tri.nn_interpolator(y, default_value=self.default_val)

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
            raise ValueError("Delaunay interpolator requires input data with shape (2, M), got %s instead" % x.shape)
        if self._interp is None:
            raise NotFittedError("Interpolator function not fitted to data yet - first call .fit method")
        return self._interp(x[0], x[1])

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Delaunay2DGridFit
#----------------------------------------------------------------------------------------------------------------------

class Delaunay2DGridFit(GridFit):
    """Interpolate a scalar function defined on a 2-D grid, based on Delaunay
    triangulation.

    The *x* data sequence for this object should have two items: the 'x' and
    'y' axis ticks (both in ascending order) defining a grid of points in a
    plane. The *y* data for this object is a 2-D array of shape
    ``(len(x[0]), len(x[1]))``, which represents the scalar 'z' value of the
    function defined on the grid (the symbols in quotation marks are the names
    for these variables used in the delaunay documentation).

    It is assumed that the 'x' and 'y' axis ticks are uniformly spaced during
    evaluation, as this is a requirement of the underlying library. Even more
    restricting is the requirement that the first and last tick should coincide
    on both axes during fitting... Any points lying outside the intersection of
    the 'x' and 'y' axis tick sets will be given default values during
    evaluation. The 'x' and 'y' axes may have a different number of ticks
    (although it is not recommended).

    Parameters
    ----------
    interp_type : {'linear', 'nn'}, optional
        String indicating type of interpolation
    default_val : float, optional
        Default value used when trying to extrapolate beyond known grid

    """
    def __init__(self, interp_type='nn', default_val=np.nan):
        if not delaunay_found:
            raise ImportError("Delaunay module not found - install it from" +
                              " scikits (or recompile SciPy <= 0.6.0 with sandbox enabled)")
        GridFit.__init__(self)
        self.interp_type = interp_type
        self.default_val = default_val
        # Interpolator function, only set after :func:`fit`
        self._interp = None

    def fit(self, x, y):
        """Fit function ``y = f(x)`` to data.

        This fits a scalar function defined on 2-D data to the provided grid.
        The first sequence in *x* defines the M 'x' axis ticks, while the second
        sequence in *x* defines the N 'y' axis ticks (all in ascending order).
        The provided function output *y* contains the corresponding 'z' values
        on the grid, in an array of shape (M, N). The first and last values of
        x[0] and x[1] should match up, to minimise any unexpected results.

        Parameters
        ----------
        x : sequence of 2 sequences, of lengths M and N
            Known input grid specified by sequence of 2 sequences of axis ticks
        y : array-like, shape (M, N)
            Known output values as a 2-D numpy array

        """
        # Check dimensions of known data
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        y = np.atleast_2d(np.asarray(y))
        if (len(x) != 2) or (len(x[0].shape) != 1) or (len(x[1].shape) != 1) or (len(y.shape) != 2) or \
           (y.shape[0] != len(x[0])) or (y.shape[1] != len(x[1])):
            raise ValueError("Delaunay interpolator requires input data with shape [(M,), (N,)] and output data " +
                             "with shape (M, N), got %s and %s instead" % ([ax.shape for ax in x], y.shape))
        if (x[0][0] != x[1][0]) or (x[0][-1] != x[1][-1]):
            logger.warning("The first and last values of x[0] and x[1] do not match up, " +
                           "which may lead to unexpected results...")
        # Create rectangular mesh, and triangulate
        x1, x0 = np.meshgrid(x[1], x[0])
        tri = delaunay.Triangulation(x0.ravel(), x1.ravel())
        if self.interp_type == 'nn':
            self._interp = tri.nn_interpolator(y.ravel(), default_value=self.default_val)
        elif self.interp_type == 'linear':
            self._interp = tri.linear_interpolator(y.ravel(), default_value=self.default_val)

    def __call__(self, x):
        """Evaluate function ``y = f(x)`` on new data.

        Evaluates the fitted scalar function on 2-D grid provided in *x*. The
        first sequence in *x* defines the K 'x' axis ticks (in ascending order),
        while the second sequence in *x* defines the L 'y' axis ticks.
        The function returns the corresponding 'z' values on the grid, in an
        array of shape (K, L). It is assumed that the 'x' and 'y' axis ticks are
        uniformly spaced, as this is a requirement of the underlying library.
        Only the first and last ticks, and the number of ticks, are therefore
        used to construct the grid, while the rest of the values are ignored...

        Parameters
        ----------
        x : sequence of 2 sequences, of lengths K and L
            2-D input grid specified by sequence of 2 sequences of axis ticks

        Returns
        -------
        y : array, shape (K, L)
            Output of function as a 2-D numpy array

        """
        # Check dimensions
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        if (len(x) != 2) or (len(x[0].shape) != 1) or (len(x[1].shape) != 1):
            raise ValueError("Delaunay interpolator requires input data with shape [(K,), (L,)], " +
                             "got %s instead" % ([ax.shape for ax in x],))
        if self._interp is None:
            raise NotFittedError("Interpolator function not fitted to data yet - first call .fit method")
        return self._interp[x[0][0]:x[0][-1]:len(x[0])*1j, x[1][0]:x[1][-1]:len(x[1])*1j]

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  NonLinearLeastSquaresFit
#----------------------------------------------------------------------------------------------------------------------

class NonLinearLeastSquaresFit(ScatterFit):
    """Fit a generic function to data, based on non-linear least squares optimisation.

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
    of shape (D_x, N) to produce ``y`` arrays of shape (D_y, N). This is the way
    in which x-y data is presented to the :meth:`fit` method. Note that the
    array sequence is concatenated along the *last* dimension (i.e. as columns).
    If it cannot, use the helper function :func:`vectorize_fit_func` to wrap the
    function before passing it to this class.

    If available, the Jacobian of the function, ``J = g(p, x)``, should return
    an array ``J`` of shape (D_y, P), where ``P = len(p)`` is the number of
    function parameters. Each element of this array indicates the derivative of
    the ``i``'th output value with respect to the ``j``'th parameter, evaluated
    at the given ``p`` and ``x``. This function should also be vectorised,
    similar to ``f``, so that an input ``x`` array of shape (D_x, N) produces an
    output ``J`` array of shape (D_y, P, N).

    Parameters
    ----------
    func : function, signature ``y = f(p, x)``
        Generic function to be fit to x-y data (should be vectorised)
    initial_params : sequence, length P
        Initial guess of function parameter vector *p*
    func_jacobian : function, signature ``J = g(p, x)``, optional
        Jacobian of function f, if available, where J has the shape (D_y, P),
        with D_y the normal ``y`` shape returned by the function
    method : string, optional
        Optimisation method (name of corresponding func:`scipy.optimize`
        function)
    kwargs : dict, optional
        Additional keyword arguments are passed to underlying optimiser

    Arguments
    ---------
    params : sequence, length P
        Final optimal value for function parameter vector
        (starts off as initial value)

    """
    def __init__(self, func, initial_params, func_jacobian=None, method='leastsq', **kwargs):
        ScatterFit.__init__(self)
        self.func = func
        # Preserve this for repeatability of fits
        self.initial_params = initial_params
        self.func_jacobian = func_jacobian
        try:
            self._optimizer = scipy.optimize.__dict__[method]
        except KeyError:
            raise ValueError('Optimisation method "' + method + '" unknown - should be one of: ' +
                             'anneal fmin fmin_bfgs fmin_cg fmin_l_bfgs_b fmin_powell fmin_tnc leastsq')
        # Extra keyword arguments to optimiser
        self._extra_args = kwargs
        if not method in ('fmin_l_bfgs_b', 'fmin_tnc'):
            self._extra_args['full_output'] = 1
        self.params = initial_params

    def fit(self, x, y):
        """Fit function to data, by performing non-linear least squares optimisation.

        This determines the optimal parameter vector ``p*`` so that the function
        ``y = f(p, x)`` best fits the observed x-y data, in a least-squares
        sense. The x-y data is a sequence of N ``x`` arrays of shape D_x and
        a sequence of N corresponding ``y`` arrays of shape D_y. These sequences
        are concatenated along the *last* dimension (i.e. as columns) to form
        the *x* and *y* arrays.

        Parameters
        ----------
        x : array-like, shape (D_x, N)
            Sequence of input values as columns of a numpy array
        y : array-like, shape (D_y, N)
            Sequence of output values as columns of a numpy array

        """
        x = np.asarray(x)
        y = np.asarray(y)
        # Sum-of-squares cost function to be minimised (or R = prod(D_y) * N residuals for leastsq)
        def cost(p):
            residuals = y - self.func(p, x)
            if self._optimizer.__name__ == 'leastsq':
                return residuals.ravel()
            else:
                return (residuals**2).sum()
        # Register Jacobian function if applicable
        if not self.func_jacobian is None:
            # Jacobian (R, P) matrix of function at given p and x values (derivatives along rows)
            def jacobian(p):
                # Produce Jacobian of residual - array with shape (D_y, P, N)
                residual_jac = - self.func_jacobian(p, x)
                # Squash every axis except second-last parameter axis together, to get (R, P) shape
                flatten_axes = range(len(residual_jac.shape) - 2) + [len(residual_jac.shape) - 1]
                ravel_jac = squash(residual_jac, flatten_axes, move_to_start=True)
                if self._optimizer.__name__ == 'leastsq':
                    # Jacobian of residuals has shape (R, P)
                    return ravel_jac
                else:
                    # Jacobian of cost function (sum of squared residuals) has shape (P,) instead
                    residuals = y - self.func(p, x)
                    return np.dot(ravel_jac.transpose(), 2.0 * residuals.ravel())
            if self._optimizer.__name__ == 'leastsq':
                self._extra_args['Dfun'] = jacobian
            else:
                self._extra_args['fprime'] = jacobian
        # Do optimisation (copy initial parameters, as the optimiser clobbers them with final values)
        self.params = self._optimizer(cost, copy.deepcopy(self.initial_params), **self._extra_args)[0]

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
        return NonLinearLeastSquaresFit(self.func, self.params, self.func_jacobian,
                                        self._optimizer.__name__, **self._extra_args)

    def __deepcopy__(self, memo):
        """Deep copy operation.

        Don't deepcopy stored functions, as this is not supported in Python 2.4
        (Python 2.5 supports it...).

        Parameters
        ----------
        memo : dict
            Dictionary that caches objects that are already copied

        """
        return NonLinearLeastSquaresFit(self.func, copy.deepcopy(self.params, memo), self.func_jacobian,
                                        copy.deepcopy(self._optimizer.__name__, memo),
                                        **(copy.deepcopy(self._extra_args, memo)))

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  GaussianFit
#----------------------------------------------------------------------------------------------------------------------

class GaussianFit(ScatterFit):
    """Fit Gaussian curve to multi-dimensional data.

    This fits a D-dimensional Gaussian curve (with diagonal covariance matrix
    or single scalar variance) to x-y data. Don't confuse this with fitting a
    Gaussian probability density function (pdf) to random data! The underlying
    optimiser is a modified Levenberg-Marquardt algorithm
    (:func:`scipy.optimize.leastsq`).

    Parameters
    ----------
    mean : array-like, shape (D,)
        Initial guess of D-dimensional mean vector
    var : array-like, shape (D,), or float
        Initial guess of D-dimensional vector of variances, or a single variance
        for all dimensions
    height : float
        Initial guess of height of Gaussian curve

    Arguments
    ---------
    mean : array, shape (D,)
        D-dimensional mean vector, either initial guess or final optimal value
    var : array, shape (D,), or float
        D-dimensional variance vector or single variance, either initial guess
        or final optimal value
    height : float
        Height of Gaussian curve, either initial guess or final optimal value

    Raises
    ------
    ValueError
        If dimensions of mean and/or var are incompatible

    Notes
    -----
    One option that was considered is fitting the Gaussian internally to the log
    of the data. This is more robust in some scenarios, but cannot handle
    negative data, which frequently occur in noisy problems. With log data,
    the optimisation criterion is not quite least-squares in the original x-y
    domain as well.

    """
    # pylint: disable-msg=W0612
    def __init__(self, mean, var, height):
        ScatterFit.__init__(self)
        self.mean = np.atleast_1d(mean)
        self.var = np.atleast_1d(var)
        if (len(self.mean.shape) != 1) or (len(self.var.shape) != 1) or \
           (self.var.shape not in [self.mean.shape, (1,)]):
            raise ValueError("Dimensions of mean and/or variance incorrect")
        self.height = height
        # Make sure a single variance is a plain scalar, and create parameter vector for optimisation
        if self.var.shape == (1,):
            self.var = self.var[0]
            params = np.concatenate((self.mean, [self.height, 1.0 / self.var]))
        else:
            params = np.concatenate((self.mean, [self.height], 1.0 / self.var))
        def gauss_diagcov(p, x):
            """Evaluate D-dimensional Gaussian with diagonal covariance matrix.

            Parameters
            ----------
            p : array-like, shape (D + 2,) or (2*D + 1,)
                Parameters, consisting of D-dim mean vector + scalar height
                + scalar (or D-dim) inverse variance
            x : array-like, shape (D, N) or (D,)
                Sequence of D-dimensional input values as columns of numpy array

            Returns
            -------
            y : array, shape (N,), or float
                Sequence of 1-D output values as a numpy array

            """
            p, x = np.atleast_1d(p), np.atleast_1d(x)
            D = x.shape[0]
            x_min_mu = x - p[:D, np.newaxis] if x.ndim > 1 else x - p[:D]
            if len(p) == D + 2:
                ivar = np.tile(p[D + 1], D)
            else:
                ivar = p[D + 1:]
            return p[D] * np.exp(-0.5 * np.dot(ivar, x_min_mu * x_min_mu))
        # Internal non-linear least squares fitter
        self._interp = NonLinearLeastSquaresFit(gauss_diagcov, params, method='leastsq')

    def fit(self, x, y):
        """Fit a Gaussian curve to data.

        The mean, variance and height can be obtained from the corresponding
        member variables after this is run.

        Parameters
        ----------
        x : array-like, shape (D, N)
            Sequence of D-dimensional input values as columns of a numpy array
        y : array-like, shape (N,)
            Sequence of 1-D output values as a numpy array

        """
        self._interp.fit(x, y)
        # Recreate Gaussian parameters
        D = len(self.mean)
        self.mean = self._interp.params[:D]
        self.height = self._interp.params[D]
        if len(self._interp.params) == D + 2:
            self.var = 1.0 / self._interp.params[D + 1]
        else:
            self.var = 1.0 / self._interp.params[D + 1:]

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

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Spline1DFit
#----------------------------------------------------------------------------------------------------------------------

class Spline1DFit(ScatterFit):
    """Fit a B-spline to 1-D data.

    This wraps :class:`scipy.interpolate.UnivariateSpline`, which is based on
    Paul Dierckx's DIERCKX (or FITPACK) routines (specifically ``curfit`` for
    fitting and ``splev`` for evaluation).

    Parameters
    ----------
    degree : int, optional
        Degree of spline, in range 1-5 [default=3, i.e. cubic B-spline]
    std_y : function, signature ``s = f(x, y)``, optional
        Function evaluating the standard deviation of *y*. This is evaluated
        during fit() with the (x, y) data to fit as parameters, and should
        return an array of shape (N,) representing the standard deviation of
        each sample in *y*.
    min_size : float, optional
        Size of smallest features to fit in the data, expressed in units of *x*.
        This determines the smoothness of fitted spline. Roughly stated, any
        oscillation in the fitted curve will have a period bigger than *min_size*.
        Works best if *x* is uniformly spaced. If set, this overrides *std_y*.
    kwargs : dict, optional
        Additional keyword arguments are passed to underlying spline class

    """
    def __init__(self, degree=3, std_y=lambda x, y: np.tile(1.0, len(y)), min_size=0.0, **kwargs):
        ScatterFit.__init__(self)
        self.degree = degree
        # Standard deviation of y
        self._std_y = std_y
        # Size of smallest features to fit
        self._min_size = min_size
        # Extra keyword arguments to spline class
        self._extra_args = kwargs
        # Interpolator function, only set after :func:`fit`
        self._interp = None

    def fit(self, x, y):
        """Fit spline to 1-D data.

        The minimum number of data points is N = degree + 1.

        Parameters
        ----------
        x : array-like, shape (N,)
            Known input values as a 1-D numpy array or sequence
        y : array-like, shape (N,)
            Known output values as a 1-D numpy array, or sequence

        """
        # Check dimensions of known data
        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if y.size < self.degree + 1:
            raise ValueError("Not enough data points for spline fit: requires at least " +
                             str(self.degree + 1) + ", only got " + str(y.size))
        # Ensure that x is in strictly ascending order
        if np.any(np.diff(x) < 0):
            sort_ind = x.argsort()
            x = x[sort_ind]
            y = y[sort_ind]
        # Deduce standard deviation of y, based on specified size of smallest features
        if self._min_size > 0.0:
            # Number of samples, and sample period (assuming samples are uniformly spaced in x)
            N, xstep = len(x), np.abs(np.mean(np.diff(x)))
            # Convert feature size to digital frequency (based on k / N = Ts / T using FFT notation).
            # The frequency index k is clipped so that k > 0, to avoid including DC power in stdev calc
            # (i.e. slowest oscillation is N samples), and k <= N / 2, which represents a 2-sample oscillation.
            min_freq_ind = np.clip(int(np.round(N * xstep / self._min_size)), 1, N // 2)
            # Find power in signal above the minimum cutoff frequency using periodogram
            # Reduce spectral leakage resulting from edge effects by removing DC and windowing the signal
            window = np.hamming(N)
            periodo = (np.abs(np.fft.fft((y - y.mean()) * window)) ** 2) / (window ** 2).sum()
            periodo[1:(N // 2)] *= 2.0
            high_freq_std_y = np.sqrt(np.sum(periodo[min_freq_ind:(N // 2 + 1)]) / N)
            self._std_y = lambda x, y: np.tile(high_freq_std_y, N)
        self._interp = scipy.interpolate.UnivariateSpline(x, y, w=1.0 / self._std_y(x, y), k=self.degree,
                                                          **self._extra_args)

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
            raise NotFittedError("Spline not fitted to data yet - first call .fit method")
        return self._interp(x)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Spline2DScatterFit
#----------------------------------------------------------------------------------------------------------------------

class Spline2DScatterFit(ScatterFit):
    """Fit a B-spline to scattered 2-D data.

    This wraps :class:`scipy.interpolate.SmoothBivariateSpline`, which is based
    on Paul Dierckx's DIERCKX (or FITPACK) routines (specifically ``surfit`` for
    fitting and ``bispev`` for evaluation). The 2-D ``x`` coordinates do not
    have to lie on a regular grid, and can be in any order.

    Parameters
    ----------
    degree : sequence of 2 ints, optional
        Degree (1-5) of spline in x and y directions
    std_y : function, signature ``s = f(x, y)``, optional
        Function evaluating the standard deviation of *y*. This is evaluated
        during fit() with the (x, y) data to fit as parameters, and should
        return an array of shape (N,) representing the standard deviation of
        each sample in *y*.
    kwargs : dict, optional
        Additional keyword arguments are passed to underlying spline class

    """
    def __init__(self, degree=(3, 3), std_y=lambda x, y: np.tile(1.0, len(y)), **kwargs):
        ScatterFit.__init__(self)
        self.degree = degree
        # Extra keyword arguments to spline class
        self._extra_args = kwargs
        # Standard deviation of y
        self._std_y = std_y
        # Interpolator function, only set after :func:`fit`
        self._interp = None

    def fit(self, x, y):
        """Fit spline to 2-D scattered data in unstructured form.

        The minimum number of data points is ``N = (degree[0]+1)*(degree[1]+1)``.
        The 2-D *x* coordinates do not have to lie on a regular grid, and can be
        in any order.

        Parameters
        ----------
        x : array-like, shape (2, N)
            Known input values as a 2-D numpy array, or sequence
        y : array-like, shape (N,)
            Known output values as a 1-D numpy array, or sequence

        """
        # Check dimensions of known data
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if (len(x.shape) != 2) or (x.shape[0] != 2) or (len(y.shape) != 1) or (y.shape[0] != x.shape[1]):
            raise ValueError("Spline interpolator requires input data with shape (2, N) and " +
                             "output data with shape (N,), got %s and %s instead" % (x.shape, y.shape))
        if y.size < (self.degree[0] + 1) * (self.degree[1] + 1):
            raise ValueError("Not enough data points for spline fit: requires at least %d, only got %d" %
                             ((self.degree[0] + 1) * (self.degree[1] + 1), y.size))
        self._interp = scipy.interpolate.SmoothBivariateSpline(x[0], x[1], y, w=1.0 / self._std_y(x, y),
                                                              kx=self.degree[0], ky=self.degree[1], **self._extra_args)

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
            raise ValueError("Spline interpolator requires input data with shape (2, M), got %s instead" % (x.shape,))
        if self._interp is None:
            raise NotFittedError("Spline not fitted to data yet - first call .fit method")
        # Loop over individual data points, as underlying bispev routine expects regular grid in x
        return np.array([self._interp(x[0, n], x[1, n]) for n in xrange(x.shape[1])]).squeeze()

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Spline2DGridFit
#----------------------------------------------------------------------------------------------------------------------

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

    def fit(self, x, y):
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

        """
        # Check dimensions of known data
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        y = np.atleast_2d(np.asarray(y))
        if (len(x) != 2) or (len(x[0].shape) != 1) or (len(x[1].shape) != 1) or (len(y.shape) != 2) or \
           (y.shape[0] != len(x[0])) or (y.shape[1] != len(x[1])):
            raise ValueError("Spline interpolator requires input data with shape [(M,), (N,)] and output data " +
                             "with shape (M, N), got %s and %s instead" % ([ax.shape for ax in x], y.shape))
        if y.size < (self.degree[0] + 1) * (self.degree[1] + 1):
            raise ValueError("Not enough data points for spline fit: requires at least %d, only got %d" %
                             ((self.degree[0] + 1) * (self.degree[1] + 1), y.size))
        # Ensure that 'x' and 'y' coordinates are both in ascending order (requirement of underlying regrid)
        xs, ys, zs = sort_grid(x[0], x[1], y)
        self._interp = scipy.interpolate.RectBivariateSpline(xs, ys, zs, kx=self.degree[0], ky=self.degree[1],
                                                             **self._extra_args)

    def __call__(self, x):
        """Evaluate spline on a new rectangular grid.

        Evaluates the fitted scalar function on 2-D grid provided in *x*. The
        first sequence in *x* defines the K 'x' axis ticks (in any order), while
        the second sequence in *x* defines the L 'y' axis ticks (also in any
        order). The function returns the corresponding 'z' values on the grid,
        in an array of shape (K, L).

        Parameters
        ----------
        x : sequence of 2 sequences, of lengths K and L
            2-D input grid specified by sequence of 2 sequences of axis ticks

        Returns
        -------
        y : array-like, shape (K, L)
            Output of function as a 2-D numpy array

        """
        # Check dimensions
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        if (len(x) != 2) or (len(x[0].shape) != 1) or (len(x[1].shape) != 1):
            raise ValueError("Spline interpolator requires input data with shape [(K,), (L,)], " +
                             "got %s instead" % ([ax.shape for ax in x],))
        if self._interp is None:
            raise NotFittedError("Spline not fitted to data yet - first call .fit method")
        # The standard DIERCKX 2-D spline evaluation function (bispev) expects a rectangular grid in ascending order
        # Therefore, sort coordinates, evaluate on the sorted grid, and return the desorted result
        return desort_grid(x[0], x[1], self._interp(sorted(x[0]), sorted(x[1])))

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  RbfScatterFit
#----------------------------------------------------------------------------------------------------------------------

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
        try:
            scipy.interpolate.Rbf
        except AttributeError:
            raise ImportError("scipy.interpolate.Rbf class not found - you need SciPy 0.7.0 or newer")
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

        """
        # Check dimensions of known data
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if (len(x.shape) != 2) or (len(y.shape) != 1) or (y.shape[0] != x.shape[1]):
            raise ValueError("RBF interpolator requires input data with shape (D, N) " +
                             "and output data with shape (N,), got %s and %s instead" % (x.shape, y.shape))
        self._interp = scipy.interpolate.Rbf(*np.vstack((x, y)), **self._extra_args)

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
            raise ValueError("RBF interpolator requires input data with shape (D, M), got %s instead" % (x.shape,))
        if self._interp is None:
            raise NotFittedError("RBF not fitted to data yet - first call .fit method")
        return self._interp(*x)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  SampledTemplateFit
#----------------------------------------------------------------------------------------------------------------------

#class SampledTemplateFit(ScatterFit):
