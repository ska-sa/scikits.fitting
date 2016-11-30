"""Delaunay fitters.

There are currently (2016) two viable Delaunay options:

  - scipy.spatial.Delaunay (since scipy 0.9.0)
  - matplotlib.tri.Triangulation (since matplotlib 0.98.3)

Since matplotlib 1.4.0, both are based on Qhull and therefore quite robust.
In addition matplotlib has linear and cubic interpolators on top of the
triangulation, which is what scikits.fitting wants. Therefore we use the
latter for now, until scipy gains interpolation too.

:author: Ludwig Schwardt
:license: Modified BSD

"""
import logging

import numpy as np
import matplotlib.tri as mtri

from .generic import ScatterFit, NotFittedError


logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# --- CLASS :  Delaunay2DScatterFit
# ----------------------------------------------------------------------------------------------------------------------


class Delaunay2DScatterFit(ScatterFit):
    """Interpolate scalar function of 2-D data, based on Delaunay triangulation.

    The *x* data for this object should have two rows, containing the 'x' and
    'y' coordinates of points in a plane. The 2-D points are therefore stored as
    column vectors in *x*. The *y* data for this object is a 1-D array, which
    represents the scalar 'z' value of the function defined on the plane
    (the symbols in quotation marks are the names for these variables used in
    the ``matplotlib.tri`` documentation). The 2-D *x* coordinates do not have
    to lie on a regular grid, and can be in any order.

    Parameters
    ----------
    interp_type : {'cubic', 'cubic_fast', 'linear'}, optional
        String indicating type of interpolation
    default_val : float, optional
        Default value used when trying to extrapolate beyond convex hull of
        known data
    jitter : bool, optional
        True to add small amount of jitter to *x* to make degenerate
        triangulation unlikely (generally not needed with Qhull back-end)

    """
    def __init__(self, interp_type='cubic', default_val=np.nan, jitter=False):
        ScatterFit.__init__(self)
        interps = ('cubic', 'cubic_fast', 'linear')
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
        tri = mtri.Triangulation(x[0], x[1])
        if self.interp_type == 'cubic':
            self._interp = mtri.CubicTriInterpolator(tri, y)
        elif self.interp_type == 'cubic_fast':
            self._interp = mtri.CubicTriInterpolator(tri, y, kind='geom')
        else:
            self._interp = mtri.LinearTriInterpolator(tri, y)
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
        return self._interp(x[0], x[1]).filled(self.default_val)
