Fitting SciKit
==============

A framework for fitting functions to data with SciPy which unifies the various
available interpolation methods and provides a common interface to them based
on the following simple methods:

- ``Fitter.__init__(p)``: set parameters of interpolation function, e.g. polynomial degree
- ``Fitter.fit(x, y)``: fit given input-output data
- ``Fitter.__call__(x)`` or ``Fitter.eval(x)``: evaluate function on new input data

Each interpolation routine falls in one of two categories: scatter fitting or
grid fitting. They share the same interface, only differing in the definition
of input data ``x``.

Scatter-fitters operate on unstructured scattered input data (i.e. not on a
grid). The input data consists of a sequence of ``x`` coordinates and a sequence
of corresponding ``y`` data, where the order of the ``x`` coordinates does not
matter and their location can be arbitrary. The ``x`` coordinates can have an
arbritrary dimension (although most classes are specialised for 1-D or 2-D
data). If the dimension is bigger than 1, the coordinates are provided as an
array of column vectors. These fitters have ``ScatterFit`` as base class.

Grid-fitters operate on input data that lie on a grid. The input data consists
of a sequence of x-axis tick sequences and the corresponding array of ``y``
data. These fitters have ``GridFit`` as base class.

The module is organised as follows:

Scatter fitters
---------------

- ``ScatterFit``: Abstract base class for scatter fitters
- ``LinearLeastSquaresFit``: Fit linear regression model to data using SVD
- ``Polynomial1DFit``: Fit polynomial to 1-D data
- ``Polynomial2DFit``: Fit polynomial to 2-D data
- ``PiecewisePolynomial1DFit``: Fit piecewise polynomial to 1-D data
- ``Independent1DFit``: Interpolate N-dimensional matrix along given axis
- ``Delaunay2DScatterFit``: Interpolate scalar function of 2-D data, based on
  Delaunay triangulation and cubic / linear interpolation
- ``NonLinearLeastSquaresFit``: Fit a generic function to data, based on
  non-linear least squares optimisation
- ``GaussianFit``: Fit Gaussian curve to multi-dimensional data
- ``Spline1DFit``: Fit a B-spline to 1-D data
- ``Spline2DScatterFit``: Fit a B-spline to scattered 2-D data
- ``RbfScatterFit``: Do radial basis function (RBF) interpolation

Grid fitters
------------

- ``GridFit``: Abstract base class for grid fitters
- ``Spline2DGridFit``: Fit a B-spline to 2-D data on a rectangular grid

Helper functions
----------------

- ``squash``: Flatten array, but not necessarily all the way to a 1-D array
- ``unsquash``: Restore an array that was reshaped by ``squash``
- ``sort_grid``: Ensure that the coordinates of a rectangular grid are in
  ascending order
- ``desort_grid``: Undo the effect of ``sort_grid``
- ``vectorize_fit_func``: Factory that creates vectorised version of
  function to be fitted to data
- ``randomise``: Randomise fitted function parameters by resampling residuals

Source
------
https://github.com/ska-sa/scikits.fitting

Contact
-------
Ludwig Schwardt <ludwig at ska.ac.za>
