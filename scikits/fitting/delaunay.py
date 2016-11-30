"""Delaunay fitters.

:author: Ludwig Schwardt
:license: Modified BSD

"""
import logging

import numpy as np
# Since scipy 0.7.0 the delaunay module lives in scikits
try:
    # pylint: disable-msg=E0611
    import scikits.delaunay as delaunay
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

from .generic import ScatterFit, GridFit, NotFittedError


logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  Delaunay2DScatterFit
# ----------------------------------------------------------------------------------------------------------------------


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

        Returns
        -------
        self : :class:`Delaunay2DScatterFit` object
            Reference to self, to allow chaining of method calls

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
            raise ValueError("Delaunay interpolator requires input data with shape (2, M), got %s instead" % x.shape)
        if self._interp is None:
            raise NotFittedError("Interpolator function not fitted to data yet - first call .fit method")
        return self._interp(x[0], x[1])
