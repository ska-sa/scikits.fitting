## @file fitting.py
#
# Classes for encapsulating interpolator functions.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>
# @date 2007-08-28

# pylint: disable-msg=C0103,C0302,R0903

import scipy.optimize as optimize           # NonLinearLeastSquaresFit
import scipy.sandbox.delaunay as delaunay   # Delaunay2DScatterFit, Delaunay2DGridFit
import scipy.interpolate as dierckx         # Spline1DFit, Spline2DScatterFit, Spline2DGridFit
import numpy as np
import numpy.random as random               # randomise()
import copy
import logging

logger = logging.getLogger("xdmsbe.xdmsbelib.fitting")

#----------------------------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------

## Flatten array, but not necessarily all the way to a 1-D array.
# This helper function is useful for broadcasting functions of arbitrary dimensionality along a given array.
# The array x is transposed and reshaped, so that the axes with indices listed in flattenAxes are collected
# either at the start or end of the array (based on the moveToStart flag). These axes are also flattened to
# a single axis, while preserving the total number of elements in the array. The reshaping and transposition
# usually results in a view of the original array, although a copy may result e.g. if discontiguous flattenAxes
# are chosen. The two extreme cases are flattenAxes = [] or None, which results in the original array with no
# flattening, and flattenAxes = range(len(x.shape)), which is equivalent to x.ravel() and therefore full flattening.
#
# Examples:
# x.shape => (2,4,10)
# squash(x, [], True).shape => (2,4,10) [no flattening, x returned unchanged]
# squash(x, (1), True).shape => (4,2,10)
# squash(x, (1), False).shape => (2,10,4)
# squash(x, (0,2), True).shape => (20,4)
# squash(x, (0,2), False).shape => (4,20)
# squash(x, (0,1,2), True).shape => (80,) [same as x.ravel()]
#
# @param x           Numpy array, or sequence
# @param flattenAxes List of axes along which x should be flattened
# @param moveToStart Flag indicating whether flattened axis is moved to start or end of array [default=True]
# @return            Semi-flattened version of x, as numpy array
def squash(x, flattenAxes, moveToStart=True):
    x = np.asarray(x)
    xShape = np.atleast_1d(np.asarray(x.shape))
    # Split list of axes into those that will be flattened and the rest, which are considered the main axes
    flattenAxes = np.atleast_1d(np.asarray(flattenAxes)).tolist()
    if flattenAxes == [None]:
        flattenAxes = []
    mainAxes = list(set(range(len(xShape))) - set(flattenAxes))
    # After flattening, the array will contain flattenShape number of mainShape-shaped subarrays
    flattenShape = [xShape[flattenAxes].prod()]
    # Don't add any singleton dimensions during flattening - rather leave the matrix as is
    if flattenShape == [1]:
        flattenShape = []
    mainShape = xShape[mainAxes].tolist()
    # Move specified axes to the beginning (or end) of list of axes, and transpose and reshape array accordingly
    if moveToStart:
        return x.transpose(flattenAxes + mainAxes).reshape(flattenShape + mainShape)
    else:
        return x.transpose(mainAxes + flattenAxes).reshape(mainShape + flattenShape)

## Restore an array that was reshaped by squash().
# @param x             Numpy array, or sequence
# @param flattenAxes   List of (original) axes along which x was flattened
# @param originalShape Original shape of x, before flattening
# @param moveFromStart Flag indicating whether flattened axes were moved to start or end of array [default=True]
# @return              Restored version of x, as numpy array
def unsquash(x, flattenAxes, originalShape, moveFromStart=True):
    x = np.asarray(x)
    originalShape = np.atleast_1d(np.asarray(originalShape))
    # Split list of axes into those that will be flattened and the rest, which are considered the main axes
    flattenAxes = np.atleast_1d(np.asarray(flattenAxes)).tolist()
    if flattenAxes == [None]:
        flattenAxes = []
    mainAxes = list(set(range(len(originalShape))) - set(flattenAxes))
    # After unflattening, the flattenAxes will be reconstructed with the correct dimensionality
    unflattenShape = originalShape[flattenAxes].tolist()
    # Don't add any singleton dimensions during flattening - rather leave the matrix as is
    if unflattenShape == [1]:
        unflattenShape = []
    mainShape = originalShape[mainAxes].tolist()
    # Move specified axes from the beginning (or end) of list of axes, and transpose and reshape array accordingly
    if moveFromStart:
        return x.reshape(unflattenShape + mainShape).transpose(np.array(flattenAxes + mainAxes).argsort())
    else:
        return x.reshape(mainShape + unflattenShape).transpose(np.array(mainAxes + flattenAxes).argsort())

## Ensure that the coordinates of a rectangular grid are all in ascending order.
# @param  x 1-D array of x coordinates, of shape (M), in any order
# @param  y 1-D array of y coordinates, of shape (N), in any order
# @param  z 2-D array of values which correspond to the coordinates in x and y, of shape (M, N)
# @return x 1-D array of x coordinates, of shape (M), in ascending order
# @return y 1-D array of y coordinates, of shape (N), in ascending order
# @return z 2-D array of values which correspond to the coordinates in x and y, of shape (M, N)
def sort_grid(x, y, z):
    xInd = np.argsort(x)
    yInd = np.argsort(y)
    return x[xInd], y[yInd], z[xInd, :][:, yInd]

## Shuffle a rectangular grid of values (based on ascending coordinates) to correspond to the original order.
# This undoes the effect of sort_grid.
# @param  x 1-D array of x coordinates, of shape (M), in the original (possibly unsorted) order
# @param  y 1-D array of y coordinates, of shape (N), in the original (possibly unsorted) order
# @param  z 2-D array of values which correspond to sorted (x,y) coordinates, of shape (M, N)
# @return z 2-D array of values which correspond to the original coordinates in x and y, of shape (M, N)
def desort_grid(x, y, z):
    xInd = np.argsort(np.argsort(x))
    yInd = np.argsort(np.argsort(y))
    return z[xInd, :][:, yInd]

## Factory that creates a vectorised version of a function to be fitted to data.
# This takes functions of the form 'y = f(p,x)' which cannot handle sequences of input arrays for x, and wraps
# it in a loop which calls f with the elements of the sequence of x values, and returns the corresponding sequence.
# @param func Function f(p,x) to be vectorised along input x
# @return Vectorised version of function
def vectorizeFitFunc(func):
    def vecFunc(p, x):
        return np.array([func(p, xx) for xx in x])
    return vecFunc

## Randomise fitted function parameters by resampling residuals.
# This allows estimation of the sampling distribution of the parameters of a fitted function, by repeatedly running
# this method and collecting the statistics (e.g. variance) of the parameters of the resulting interpolator object.
# Alternatively, it can form part of a bigger Monte Carlo run.
# The method assumes that the interpolator has already been fit to data. It obtains the residuals 'r = y - f(x)',
# and resamples them to form r* according to the specified method. The final step re-fits the interpolator to the
# pseudo-data (x, f(x) + r*), which yields a slightly different estimate of the function parameters every time the
# method is called. The method is therefore non-deterministic.
# Three resampling methods are supported:
# - 'shuffle': permute the residuals (sample from the residuals without replacement)
# - 'normal': replace the residuals with zero-mean Gaussian noise with the same variance
# - 'bootstrap': sample from the existing residuals, with replacement
# @param interp The interpolator object to randomise (not clobbered by this method)
# @param x      Known input values as a numpy array (typically the data to which the function was originally fitted)
# @param y      Known output values as a numpy array (typically the data to which the function was originally fitted)
# @param method Resampling technique used to resample residuals ('shuffle', 'normal', or 'bootstrap')
# @return       Randomised interpolator object
def randomise(interp, x, y, method='shuffle'):
    # Make copy to prevent destruction of original interpolator
    randomInterp = copy.deepcopy(interp)
    trueY = np.asarray(y)
    fittedY = randomInterp(x)
    residuals = trueY - fittedY
    # Resample residuals
    if method == 'shuffle':
        random.shuffle(residuals.ravel())
    elif method == 'normal':
        residuals = residuals.std() * random.standard_normal(residuals.shape)
    elif method == 'bootstrap':
        residuals = residuals.ravel()[random.randint(residuals.size, size=residuals.size)].reshape(residuals.shape)
    # Refit function on pseudo-data
    randomInterp.fit(x, fittedY + residuals)
    return randomInterp

#----------------------------------------------------------------------------------------------------------------------
#--- INTERFACE :  ScatterFit
#----------------------------------------------------------------------------------------------------------------------

## Interface object for interpolator functions that operate on scattered data (not on a grid).
# This defines the interface for interpolator functions that operate on unstructured scattered input data (i.e. not 
# on a grid). The input data consists of a sequence of x coordinates and a sequence of corresponding y data, 
# where the order of the x coordinates does not matter and their location can be arbitrary. The x coordinates can have 
# an arbritrary dimension (although most classes are specialised for 1-D or 2-D data).
class ScatterFit(object):
    
    ## Initialiser.
    # The initialiser should be used to specify parameters of the interpolator function, 
    # such as polynomial degree.
    # @param self The current object
    def __init__(self):
        pass
    
    ## Fit function 'y = f(x)' to data.
    # This function should reset any state associated with previous (x,y) data fits, and preserve
    # all state that was set by the initialiser.
    # @param self The current object
    # @param x    Known input values as a numpy array (order does not matter)
    # @param y    Known output values as a numpy array
    def fit(self, x, y):
        raise NotImplementedError
    
    ## Evaluate function 'y = f(x)' on new data.
    # @param self The current object
    # @param x    Input to function as a numpy array (order does not matter)
    # @return     Output of function as a numpy array
    def __call__(self, x):
        raise NotImplementedError

#----------------------------------------------------------------------------------------------------------------------
#--- INTERFACE :  GridFit
#----------------------------------------------------------------------------------------------------------------------

## Interface object for interpolator functions that operate on data on a grid.
# This defines the interface for interpolator functions that operate on input data that lie on a grid. The input data 
# consists of a sequence of x axis tick sequences and the corresponding array of y data. The shape of this array matches
# the corresponding lengths of the axis tick sequences. The axis tick sequences are assumed to be in ascending order.
# The x sequence can contain an arbitrary number of axes (although most classes are specialised for 1-D or 2-D data).
class GridFit(object):

    ## Initialiser.
    # The initialiser should be used to specify parameters of the interpolator function, 
    # such as polynomial degree.
    # @param self The current object
    def __init__(self):
        pass

    ## Fit function 'y = f(x)' to data.
    # This function should reset any state associated with previous (x,y) data fits, and preserve
    # all state that was set by the initialiser.
    # @param self The current object
    # @param x    Known axis tick values as a sequence of numpy arrays (each in ascending order)
    # @param y    Known output values as a numpy array
    def fit(self, x, y):
        raise NotImplementedError

    ## Evaluate function 'y = f(x)' on new data.
    # @param self The current object
    # @param x    Input to function as a sequence of numpy arrays (each in ascending order)
    # @return     Output of function as a numpy array
    def __call__(self, x):
        raise NotImplementedError

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Polynomial1DFit
#----------------------------------------------------------------------------------------------------------------------

## Fits polynomial to 1-D data.
# This uses numpy's polyfit and polyval.
class Polynomial1DFit(ScatterFit):
    ## Initialiser.
    # @param self      The current object
    # @param maxDegree Maximum polynomial degree to use (reduced if there are not enough data points)
    # @param rcond     Relative condition number of fit
    #                  (smallest singular value that will be used to fit polynomial, has sensible default)
    def __init__(self, maxDegree, rcond=None):
        ScatterFit.__init__(self)
        ## @var maxDegree
        # Polynomial degree
        self.maxDegree = maxDegree
        ## @var _rcond
        # Relative condition number of fit
        self._rcond = rcond
        ## @var _mean
        # Mean of input data, only set after fit()
        self._mean = None
        ## @var poly
        # Polynomial coefficients, only set after fit()
        self.poly = None
    
    ## Fit polynomial to data.
    # @param self The current object
    # @param x    Known input values as a 1-D numpy array or sequence
    # @param y    Known output values as a 1-D numpy array, or sequence
    def fit(self, x, y):
        # Upcast x and y to doubles, to ensure a high enough precision for the polynomial coefficients
        x = np.atleast_1d(np.array(x, dtype='double'))
        y = np.atleast_1d(np.array(y, dtype='double'))
        # Polynomial fits perform better if input data is centred around origin [see numpy.polyfit help]
        self._mean = x.mean()
        # Reduce polynomial degree if there is not enough points to fit (degree should be < len(x))
        self.poly = np.polyfit(x - self._mean, y, min((self.maxDegree, len(x)-1)), rcond = self._rcond)
    
    ## Evaluate polynomial on new data.
    # @param self The current object
    # @param x    Input to function as a 1-D numpy array, or sequence
    # @return     Output of function as a 1-D numpy array
    def __call__(self, x):
        if (self.poly == None) or (self._mean == None):
            raise AttributeError, "Polynomial not fitted to data yet - first call 'fit'."
        x = np.atleast_1d(np.asarray(x))
        return np.polyval(self.poly, x - self._mean)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  ReciprocalFit
#----------------------------------------------------------------------------------------------------------------------

## Interpolates the reciprocal of data.
# This allows any ScatterFit object to fit the reciprocal of a data set, without having to invert the data
# and the results explicitly.
class ReciprocalFit(ScatterFit):
    ## Initialiser
    # @param self The current object
    # @param interp ScatterFit object to use on the reciprocal of the data
    def __init__(self, interp):
        ScatterFit.__init__(self)
        ## @var _interp
        # Internal interpolator object
        self._interp = copy.deepcopy(interp)
    
    # Fit stored interpolator to reciprocal of data, i.e. fit function '1/y = f(x)'.
    # @param self The current object
    # @param x    Known input values as a numpy array
    # @param y    Known output values as a numpy array
    def fit(self, x, y):
        y = np.asarray(y)
        self._interp.fit(x, 1.0 / y)
    
    ## Evaluate function '1/f(x)' on new data, where f is interpolated from previous data.
    # @param self The current object
    # @param x    Input to function as a numpy array
    # @return     Output of function as a numpy array
    def __call__(self, x):
        return 1.0 / self._interp(x)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Independent1DFit
#----------------------------------------------------------------------------------------------------------------------

## Interpolates an N-dimensional matrix along a given axis, using a set of independent 1-D interpolators.
# This simplifies the simultaneous interpolation of a set of one-dimensional x-y relationships.
# It assumes that x is 1-D, while y is N-D and to be independently interpolated along N-1 of its dimensions.
class Independent1DFit(ScatterFit):
    ## Initialiser
    # @param self The current object
    # @param interp ScatterFit object to use on each 1-D segment
    # @param axis Axis of 'y' matrix which will vary with the independent 'x' variable
    def __init__(self, interp, axis):
        ScatterFit.__init__(self)
        ## @var _interp
        # Internal interpolator object to be cloned into an array of interpolators
        self._interp = interp
        ## @var _axis
        # Axis of 'y' matrix which will vary with the independent 'x' variable
        self._axis = axis
        ## @var _interps
        # Array of interpolators, only set after fit()
        self._interps = None
    
    ## Fit a set of stored interpolators to one axis of 'y' matrix.
    # @param self The current object
    # @param x    Known input values as a 1-D numpy array or sequence
    # @param y    Known output values as an N-D numpy array
    def fit(self, x, y):
        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if self._axis >= len(y.shape):
            raise ValueError, "Provided y-array does not have the specified axis " + str(self._axis) + "."
        if y.shape[self._axis] != len(x):
            raise ValueError, "Number of elements in x and along specified axis of y differ."
        # Shape of array of interpolators (same shape as y, but without 'independent' specified axis)
        interpShape = list(y.shape)
        interpShape.pop(self._axis)
        # Create blank array of interpolators
        self._interps = np.ndarray(interpShape, dtype=type(self._interp))
        numInterps = np.array(interpShape).prod()
        # Move specified axis to the end of list
        newAxisOrder = range(len(y.shape))
        newAxisOrder.pop(self._axis)
        newAxisOrder.append(self._axis)
        # Rearrange to form 2-D array of data and 1-D array of interpolators
        flatY = y.transpose(newAxisOrder).reshape(numInterps, len(x))
        flatInterps = self._interps.ravel()
        # Clone basic interpolator and fit x and each row of the flattened y matrix independently
        for n in range(numInterps):
            flatInterps[n] = copy.deepcopy(self._interp)
            flatInterps[n].fit(x, flatY[n])
    
    ## Evaluate set of interpolator functions on new data.
    # @param self The current object
    # @param x    Input to function as a 1-D numpy array, or sequence
    # @return     Output of function as an N-D numpy array
    def __call__(self, x):
        if self._interps == None:
            raise AttributeError, "Interpolator functions not fitted to data yet - first call 'fit'."
        x = np.atleast_1d(np.asarray(x))
        # Create blank output array with specified axis appended at the end of shape
        outShape = list(self._interps.shape)
        outShape.append(len(x))
        y = np.ndarray(outShape)
        numInterps = np.array(self._interps.shape).prod()
        # Rearrange to form 2-D array of data and 1-D array of interpolators
        flatY = y.reshape(numInterps, len(x))
        assert flatY.base is y, "Reshaping array resulted in a copy instead of a view - bad news for this code..."
        flatInterps = self._interps.ravel()
        # Apply each interpolator to x and store in appropriate row of y
        for n in range(numInterps):
            flatY[n] = flatInterps[n](x)
        # Create list of indices that will move specified axis from last place to correct location
        newAxisOrder = range(len(outShape))
        newAxisOrder.insert(self._axis, newAxisOrder.pop())
        return y.transpose(newAxisOrder)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Delaunay2DScatterFit
#----------------------------------------------------------------------------------------------------------------------

## Interpolates a scalar function of 2-D data, based on Delaunay triangulation (scattered data version).
# The x data for this object should have two rows, containing the 'x' and 'y' coordinates of points in a plane.
# The 2-D points are therefore stored as column vectors in x. The y data for this object is a 1-D array, which
# represents the scalar 'z' value of the function defined on the plane (the symbols in quotation marks are the
# names for these variables used in the delaunay documentation). The 2-D x coordinates do not have to lie on a 
# regular grid, and can be in any order. Jittering a regular grid seems to be troublesome, though...
class Delaunay2DScatterFit(ScatterFit):
    ## Initialiser
    # @param self       The current object
    # @param interpType String indicating type of interpolation (only 'nn' currently supported) ['nn']
    # @param defaultVal Default value used when trying to extrapolate beyond convex hull of known data [default=NaN]
    # @param jitter     True to add small amount of jitter to x to make degenerate triangulation unlikely [False]
    def __init__(self, interpType='nn', defaultVal=np.nan, jitter=False):
        ScatterFit.__init__(self)
        if interpType != 'nn':
            raise TypeError, "Only 'nn' interpolator currently supports unstructured data not on a regular grid..."
        ## @var interpType
        # String indicating type of interpolation ('linear' or 'nn')
        self.interpType = interpType
        ## @var defaultVal
        # Default value used when trying to extrapolate beyond convex hull of known data
        self.defaultVal = defaultVal
        ## @var jitter
        # True if small amount of jitter is added to x to make degenerate triangles unlikely during triangulation
        self.jitter = jitter
        ## @var _interp
        # Interpolator function, only set after fit()
        self._interp = None
    
    ## Fit function 'y = f(x)' to data.
    # This fits a scalar function defined on 2-D data to the provided x-y pairs.
    # The 2-D x coordinates do not have to lie on a regular grid, and can be in any order.
    # @param self The current object
    # @param x    Known input values as a 2-D numpy array, or sequence (of shape (2,N))
    # @param y    Known output values as a 1-D numpy array, or sequence (of shape (N))
    def fit(self, x, y):
        # Check dimensions of known data
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if (len(x.shape) != 2) or (x.shape[0] != 2) or (len(y.shape) != 1) or (y.shape[0] != x.shape[1]):
            raise ValueError, "Delaunay interpolator requires input data with shape (2,N) and output data with " \
                              " shape (N), got " + str(x.shape) + " and " + str(y.shape) + " instead."
        if self.jitter:
            x = x + 0.00001 * x.std(axis=1)[:, np.newaxis] * np.random.standard_normal(x.shape)
        tri = delaunay.Triangulation(x[0], x[1])
        if self.interpType == 'nn':
            self._interp = tri.nn_interpolator(y, default_value=self.defaultVal)
    
    ## Evaluate function 'y = f(x)' on new data.
    # Evaluates the fitted scalar function on 2-D data provided in x.
    # @param self The current object
    # @param x    Input to function as a 2-D numpy array, or sequence (of shape (2,N))
    # @return     Output of function as a 1-D numpy array (of shape (N))
    def __call__(self, x):
        # Check dimensions
        x = np.atleast_2d(np.asarray(x))
        if (len(x.shape) != 2) or (x.shape[0] != 2):
            raise ValueError, "Delaunay interpolator requires input data with shape (2,N), got " + \
                              str(x.shape) + " instead."
        if self._interp == None:
            raise AttributeError, "Interpolator function not fitted to data yet - first call 'fit'."
        return self._interp(x[0], x[1])

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Delaunay2DGridFit
#----------------------------------------------------------------------------------------------------------------------

## Interpolates a scalar function defined on a 2-D grid, based on Delaunay triangulation.
# The x data sequence for this object should have two items: the 'x' and 'y' axis ticks (both in ascending order) 
# defining a grid of points in a plane. The y data for this object is a 2-D array of shape (len(x[0]), len(x[1])), 
# which represents the scalar 'z' value of the function defined on the grid (the symbols in quotation marks are the 
# names for these variables used in the delaunay documentation). It is assumed that the 'x' and 'y' axis ticks are
# uniformly spaced during evaluation, as this is a requirement of the underlying library. Even more restricting is 
# the requirement that the first and last tick should coincide on both axes during fitting... Any points lying outside
# the intersection of the 'x' and 'y' axis tick sets will be given default values during evaluation. The 'x' and 'y' 
# axes may have a different number of ticks (although it is not recommended).
class Delaunay2DGridFit(GridFit):
    ## Initialiser
    # @param self       The current object
    # @param interpType String indicating type of interpolation ('linear' or 'nn') ['nn']
    # @param defaultVal Default value used when trying to extrapolate beyond known grid [default=NaN]
    def __init__(self, interpType='nn', defaultVal=np.nan):
        GridFit.__init__(self)
        ## @var interpType
        # String indicating type of interpolation ('linear' or 'nn')
        self.interpType = interpType
        ## @var defaultVal
        # Default value used when trying to extrapolate beyond known data grid
        self.defaultVal = defaultVal
        ## @var _interp
        # Interpolator function, only set after fit()
        self._interp = None
    
    ## Fit function 'y = f(x)' to data.
    # This fits a scalar function defined on 2-D data to the provided grid. The first sequence in x defines
    # the M 'x' axis ticks (in ascending order), while the second sequence in x defines the N 'y' axis ticks.
    # The provided function output y contains the corresponding 'z' values on the grid, in an array of shape (M, N).
    # The first and last values of x[0] and x[1] should match up, to minimise any unexpected results.
    # @param self The current object
    # @param x    Known input grid specified by sequence of 2 sequences of axis ticks (of lengths M and N)
    # @param y    Known output values as a 2-D numpy array of shape (M, N)
    def fit(self, x, y):
        # Check dimensions of known data
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        y = np.atleast_2d(np.asarray(y))
        if (len(x) != 2) or (len(x[0].shape) != 1) or (len(x[1].shape) != 1) or (len(y.shape) != 2) or \
           (y.shape[0] != len(x[0])) or (y.shape[1] != len(x[1])):
            raise ValueError, "Delaunay interpolator requires input data with shape [(M,), (N,)] " \
                              " and output data with shape (M, N), got " + str([ax.shape for ax in x]) + \
                              " and " + str(y.shape) + " instead."
        if (x[0][0] != x[1][0]) or (x[0][-1] != x[1][-1]):
            print "WARNING: The first and last values of x[0] and x[1] do not match up, " + \
                  "which may lead to unexpected results..."
        # Create rectangular mesh, and triangulate
        x1, x0 = np.meshgrid(x[1], x[0])
        tri = delaunay.Triangulation(x0.ravel(), x1.ravel())
        if self.interpType == 'nn':
            self._interp = tri.nn_interpolator(y.ravel(), default_value=self.defaultVal)
        elif self.interpType == 'linear':
            self._interp = tri.linear_interpolator(y.ravel(), default_value=self.defaultVal)
    
    ## Evaluate function 'y = f(x)' on new data.
    # Evaluates the fitted scalar function on 2-D grid provided in x. The first sequence in x defines
    # the M 'x' axis ticks (in ascending order), while the second sequence in x defines the N 'y' axis ticks.
    # The function returns the corresponding 'z' values on the grid, in an array of shape (M, N).
    # It is assumed that the 'x' and 'y' axis ticks are uniformly spaced, as this is a requirement of the 
    # underlying library. Only the first and last ticks, and the number of ticks, are therefore used
    # to construct the grid, while the rest of the values are ignored...
    # @param self The current object
    # @param x    2-D input grid specified by sequence of 2 sequences of axis ticks (of lengths M and N)
    # @return     Output of function as a 2-D numpy array of shape (M, N)
    def __call__(self, x):
        # Check dimensions
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        if (len(x) != 2) or (len(x[0].shape) != 1) or (len(x[1].shape) != 1):
            raise ValueError, "Delaunay interpolator requires input data with shape [(M,), (N,)], got " + \
                              str([ax.shape for ax in x]) + " instead."
        if self._interp == None:
            raise AttributeError, "Interpolator function not fitted to data yet - first call 'fit'."
        return self._interp[x[0][0]:x[0][-1]:len(x[0])*1j, x[1][0]:x[1][-1]:len(x[1])*1j]

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  NonLinearLeastSquaresFit
#----------------------------------------------------------------------------------------------------------------------

## Fit a generic function to data, based on non-linear least squares optimisation.
# This fits a function of the form 'y = f(p,x)' to x-y data, where the parameter vector p is optimised via
# least squares. It is assumed that the data presented to fit() consists of a sequence of x and y arrays,
# where each element in the sequence is of the right shape to serve as input or output to f(). The helper
# functions squash() and unsquash() are useful to get the x and y arrays in this form.
#
# The function f(p,x) should be able to operate on sequences of x arrays (i.e. should be vectorised). If it
# cannot, use the helper function vectorizeFitFunc() to wrap the function before passing it to this class.
#
# The Jacobian of the function (if available) should return an array of shape (normal y shape, N), where
# N = len(p) is the number of function parameters. Each element of this array indicates the derivative of the
# i'th output value with respect to the j'th parameter, evaluated at the given p and x. This function should
# also be vectorised, similar to f.
class NonLinearLeastSquaresFit(ScatterFit):
    ## Initialiser.
    # @param self          The current object
    # @param func          Generic function to be fit to x-y data, of the form 'y = f(p,x)' (should be vectorised)
    # @param initialParams Initial guess of function parameter vector p
    # @param funcJacobian  Jacobian of function f, if available, with signature 'J = f(p,x)', where J has the
    #                      shape (y shape produced by f(p,x), len(p))
    # @param method        Optimisation method (name of corresponding scipy.optimize function) [default='leastsq']
    # @param kwargs        Additional keyword arguments are passed to underlying optimiser
    # pylint: disable-msg=R0913
    def __init__(self, func, initialParams, funcJacobian=None, method='leastsq', **kwargs):
        ScatterFit.__init__(self)
        ## @var func
        # Generic function object to be fit to data
        self.func = func
        ## @var initialParams
        # Initial guess for function parameter vector (preserve it for repeatability of fits)
        self.initialParams = initialParams
        ## @var funcJacobian
        # Jacobian of function, if available
        self.funcJacobian = funcJacobian
        try:
            ## @var _optimizer
            # Optimiser method from scipy.optimize to use
            self._optimizer = optimize.__dict__[method]
        except KeyError:
            raise KeyError, 'Optimisation method "' + method + '" unknown - should be one of: ' \
                            'anneal fmin fmin_bfgs fmin_cg fmin_l_bfgs_b fmin_powell fmin_tnc leastsq'
        ## @var _extraArgs
        # Extra keyword arguments to optimiser
        self._extraArgs = kwargs
        self._extraArgs['full_output'] = 1
        ## @var params
        # Final optimal value for function parameter vector (starts off as initial value)
        self.params = initialParams
    
    ## Fit function to data, by performing non-linear least squares optimisation.
    # This determines the optimal parameter vector p* so that the function 'y = f(p,x)' best fits the
    # observed x-y data, in a least-squares sense.
    # @param self The current object
    # @param x    Sequence of input values as a numpy array, of shape (K, normal x shape)
    # @param y    Sequence of output values as a numpy array, of shape (K, normal y shape)
    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        # Sum-of-squares cost function to be minimised (or M residuals for leastsq)
        def cost(p):
            residuals = y - self.func(p, x)
            if self._optimizer.__name__ == 'leastsq':
                return residuals.ravel()
            else:
                return (residuals**2).sum()
        # Register Jacobian function if applicable
        if self.funcJacobian != None:
            # Jacobian (M,N) matrix of function at given p and x values (derivatives along rows)
            def jacobian(p):
                # Produce Jacobian of residual - array with shape (K, normal y shape, N)
                residualJac = - self.funcJacobian(p, x)
                # Squash every axis except last one together, to get (M,N) shape
                ravelJac = squash(residualJac, range(len(residualJac.shape)-1), moveToStart=True)
                if self._optimizer.__name__ == 'leastsq':
                    # Jacobian of residuals has shape (M,N)
                    return ravelJac
                else:
                    # Jacobian of cost function (sum of squared residuals) has shape (N) instead
                    residuals = y - self.func(p, x)
                    return np.dot(ravelJac.transpose(), 2.0 * residuals.ravel())
            if self._optimizer.__name__ == 'leastsq':
                self._extraArgs['Dfun'] = jacobian
            else:
                self._extraArgs['fprime'] = jacobian
        # Do optimisation (copy initial parameters, as the optimiser clobbers them with final values)
        # pylint: disable-msg=W0142
        self.params = self._optimizer(cost, copy.deepcopy(self.initialParams), **self._extraArgs)[0]
    
    ## Evaluate fitted function on new data.
    # Evaluates the fitted function 'y = f(p*,x)' on new x data.
    # @param self The current object
    # @param x    Input to function as a numpy array
    # @return     Output of function as a numpy array
    def __call__(self, x):
        return self.func(self.params, x)
    
    ## Shallow copy operation.
    # @param self The current object
    # pylint: disable-msg=W0142
    def __copy__(self):
        return NonLinearLeastSquaresFit(self.func, self.params, self.funcJacobian, \
                                        self._optimizer.__name__, **self._extraArgs)
    
    ## Deep copy operation.
    # Don't deepcopy stored functions, as this is not supported in Python 2.4 (Python 2.5 supports it...).
    # @param self The current object
    # @param memo Dictionary that caches objects that are already copied
    # pylint: disable-msg=W0142
    def __deepcopy__(self, memo):
        return NonLinearLeastSquaresFit(self.func, copy.deepcopy(self.params, memo), self.funcJacobian, \
                                        copy.deepcopy(self._optimizer.__name__, memo), \
                                        **(copy.deepcopy(self._extraArgs, memo)))

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  GaussianFit
#----------------------------------------------------------------------------------------------------------------------

## Fit Gaussian curve to multi-dimensional data.
# This fits a D-dimensional Gaussian curve (with diagonal covariance matrix) to x-y data. Don't confuse this
# with fitting a Gaussian pdf to random data! The underlying optimiser is a modified Levenberg-Marquardt algorithm
# (scipy.optimize.leastsq).
#
# One option that was considered is fitting the Gaussian internally to the log of the data. This is more robust
# in some scenarios, but cannot handle negative data, which frequently occur in noisy problems. With log data,
# the optimisation criterion is not quite least-squares in the original x-y domain as well.
class GaussianFit(ScatterFit):
    ## Initialiser
    # @param self   The current object
    # @param mean   Initial guess of D-dimensional mean vector
    # @param var    Initial guess of D-dimensional vector of variances
    # @param height Initial guess of height of Gaussian curve
    # pylint: disable-msg=W0612
    def __init__(self, mean, var, height):
        # D-dimensional Gaussian curve with diagonal covariance matrix, in vectorised form
        def gauss_diagcov(p, x):
            dim = (len(p) - 1) // 2
            xminmu = x - p[np.newaxis, 0:dim]
            return p[2*dim] * np.exp(-0.5 * np.dot(xminmu * xminmu, p[dim:2*dim]))
        # Jacobian of D-dimensional log Gaussian with diagonal covariance matrix, in vectorised form
        def lngauss_diagcov_jac(p, x):
            dim = (len(p) - 1) // 2
            K = x.shape[0]
            xminmu = x - p[np.newaxis, 0:dim]
            dFdMu = xminmu * p[np.newaxis, dim:2*dim]
            dFdSigma = -0.5 * xminmu * xminmu
            dFdHeight = np.ones((K, 1))
            return np.hstack((dFdMu, dFdSigma, dFdHeight))
        ScatterFit.__init__(self)
        ## @var mean
        # D-dimensional mean vector, either initial guess or final optimal value
        self.mean = np.atleast_1d(np.asarray(mean))
        ## @var var
        # D-dimensional variance vector, either initial guess or final optimal value
        self.var = np.atleast_1d(np.asarray(var))
        if (len(self.mean.shape) != 1) or (len(self.var.shape) != 1) or (self.mean.shape != self.var.shape):
            raise ValueError, "Dimensions of mean and/or variance incorrect (should be rank-1 and the same)."
        ## @var height
        # Height of Gaussian curve, either initial guess or final optimal value
        self.height = height
        # Create parameter vector for optimisation
        params = np.concatenate((self.mean, 1.0 / self.var, [self.height]))
        # Jacobian not working yet...
#        self._interp = NonLinearLeastSquaresFit(lngauss_diagcov, params, lngauss_diagcov_jac, method='leastsq')
        ## @var _interp
        # Internal non-linear least squares fitter
        self._interp = NonLinearLeastSquaresFit(gauss_diagcov, params, method='leastsq')
    
    ## Fit a Gaussian curve to data.
    # The mean, variance and height can be obtained from the corresponding member variables after this is run.
    # @param self The current object
    # @param x    Sequence of D-dimensional input values as a numpy array, of shape (K, D)
    # @param y    Sequence of 1-D output values as a numpy array, of shape (K,)
    def fit(self, x, y):
        self._interp.fit(x, y)
        # Recreate Gaussian parameters
        dim = len(self.mean)
        self.mean = self._interp.params[0:dim]
        self.var = 1.0 / self._interp.params[dim:2*dim]
        self.height = self._interp.params[2*dim]
    
    ## Evaluate function 'y = f(x)' on new data.
    # @param self The current object
    # @param x    Input to function as a numpy array, of shape (K, D)
    # @return     Output of function as a numpy array, of shape (K,)
    def __call__(self, x):
        return self._interp(x)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Spline1DFit
#----------------------------------------------------------------------------------------------------------------------

## Fits a B-spline to 1-D data.
# This uses scipy.interpolate, which is based on Paul Dierckx's DIERCKX (or FITPACK) routines (specifically
# 'curfit' for fitting and 'splev' for evaluation).
class Spline1DFit(ScatterFit):
    ## Initialiser.
    # @param self   The current object
    # @param degree Degree of spline (in range 1-5) [3, i.e. cubic B-spline]
    # @param method Spline class (name of corresponding scipy.interpolate class) ['UnivariateSpline']
    # @param kwargs Additional keyword arguments are passed to underlying spline class
    def __init__(self, degree = 3, method = 'UnivariateSpline', **kwargs):
        ScatterFit.__init__(self)
        ## @var degree
        # Degree of spline
        self.degree = degree
        try:
            ## @var _splineClass
            # Class in scipy.interpolate to use for spline fitting
            self._splineClass = dierckx.__dict__[method]
        except KeyError:
            raise KeyError, 'Spline class "' + method + '" unknown - should be one of: ' \
                  + ' '.join([name for name in dierckx.__dict__.iterkeys() if name.find('UnivariateSpline') >= 0])
        ## @var _extraArgs
        # Extra keyword arguments to spline class
        self._extraArgs = kwargs
        ## @var _interp
        # Interpolator function, only set after fit()
        self._interp = None
    
    ## Fit spline to 1-D data.
    # The minimum number of data points is N = degree + 1.
    # @param self The current object
    # @param x    Known input values as a 1-D numpy array or sequence
    # @param y    Known output values as a 1-D numpy array, or sequence
    # pylint: disable-msg=W0142
    def fit(self, x, y):
        # Check dimensions of known data
        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if y.size < self.degree + 1:
            raise ValueError, "Not enough data points for spline fit: requires at least " + \
                              str(self.degree + 1) + ", only got " + str(y.size)
        # Ensure that x is in strictly ascending order
        if np.any(np.diff(x) < 0):
            sortInd = x.argsort()
            x = x[sortInd]
            y = y[sortInd]
        self._interp = self._splineClass(x, y, k = self.degree, **self._extraArgs)
    
    ## Evaluate spline on new data.
    # @param self The current object
    # @param x    Input to function as a 1-D numpy array, or sequence
    # @return     Output of function as a 1-D numpy array
    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x))
        if self._interp == None:
            raise AttributeError, "Spline not fitted to data yet - first call 'fit'."
        return self._interp(x)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Spline2DScatterFit
#----------------------------------------------------------------------------------------------------------------------

## Fits a B-spline to scattered 2-D data.
# This uses scipy.interpolate, which is based on Paul Dierckx's DIERCKX (or FITPACK) routines (specifically
# 'surfit' for fitting and 'bispev' for evaluation). The 2-D x coordinates do not have to lie on a regular grid,
# and can be in any order.
class Spline2DScatterFit(ScatterFit):
    ## Initialiser.
    # @param self   The current object
    # @param degree Degree (1-5) of spline in x and y directions [(3, 3), i.e. bicubic B-spline]
    # @param method Spline class (name of corresponding scipy.interpolate class) ['SmoothBivariateSpline']
    # @param kwargs Additional keyword arguments are passed to underlying spline class
    def __init__(self, degree = (3, 3), method = 'SmoothBivariateSpline', **kwargs):
        ScatterFit.__init__(self)
        ## @var degree
        # Degree of spline as a sequence of 2 elements, one for x and one for y direction
        self.degree = degree
        try:
            ## @var _splineClass
            # Class in scipy.interpolate to use for spline fitting
            self._splineClass = dierckx.__dict__[method]
        except KeyError:
            raise KeyError, 'Spline class "' + method + r'" unknown - should be one of: ' \
                  + ' '.join([name for name in dierckx.__dict__.iterkeys() if name.find('BivariateSpline') >= 0])
        ## @var _extraArgs
        # Extra keyword arguments to spline class
        self._extraArgs = kwargs
        ## @var _interp
        # Interpolator function, only set after fit()
        self._interp = None
    
    ## Fit spline to 2-D scattered data in unstructured form.
    # The minimum number of data points is N = (degree[0]+1)*(degree[1]+1).
    # The 2-D x coordinates do not have to lie on a regular grid, and can be in any order.
    # @param self The current object
    # @param x    Known input values as a 2-D numpy array, or sequence (of shape (2,N))
    # @param y    Known output values as a 1-D numpy array, or sequence (of shape (N))
    # pylint: disable-msg=W0142
    def fit(self, x, y):
        # Check dimensions of known data
        x = np.atleast_2d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
        if (len(x.shape) != 2) or (x.shape[0] != 2) or (len(y.shape) != 1) or (y.shape[0] != x.shape[1]):
            raise ValueError, "Spline interpolator requires input data with shape (2,N) and output data with " \
                              " shape (N), got " + str(x.shape) + " and " + str(y.shape) + " instead."
        if y.size < (self.degree[0] + 1) * (self.degree[1] + 1):
            raise ValueError, "Not enough data points for spline fit: requires at least " + \
                              str((self.degree[0] + 1) * (self.degree[1] + 1)) + ", only got " + str(y.size)
        self._interp = self._splineClass(x[0], x[1], y, kx = self.degree[0], ky = self.degree[1], **self._extraArgs)
    
    ## Evaluate spline on new scattered data.
    # @param self The current object
    # @param x    Input to function as a 2-D numpy array, or sequence (of shape (2,N))
    # @return     Output of function as a 1-D numpy array (of shape (N))
    def __call__(self, x):
        # Check dimensions
        x = np.atleast_2d(np.asarray(x))
        if (len(x.shape) != 2) or (x.shape[0] != 2):
            raise ValueError, "Spline interpolator requires input data with shape (2,N), got " + \
                              str(x.shape) + " instead."
        if self._interp == None:
            raise AttributeError, "Spline not fitted to data yet - first call 'fit'."
        # Loop over individual data points, as underlying bispev routine expects regular grid in x
        return np.array([self._interp(x[0, n], x[1, n]) for n in xrange(x.shape[1])]).squeeze()

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Spline2DGridFit
#----------------------------------------------------------------------------------------------------------------------

## Fits a B-spline to 2-D data on a rectangular grid.
# This uses scipy.interpolate, which is based on Paul Dierckx's DIERCKX (or FITPACK) routines (specifically
# 'regrid' for fitting and 'bispev' for evaluation). The 2-D x coordinates define a rectangular grid.
# They do not have to be in ascending order, as both the fitting and evaluation routines sort them for you.
class Spline2DGridFit(GridFit):
    ## Initialiser.
    # @param self   The current object
    # @param degree Degree (1-5) of spline in x and y directions [(3, 3), i.e. bicubic B-spline]
    # @param method Spline class (name of corresponding scipy.interpolate class) ['RectBivariateSpline']
    # @param kwargs Additional keyword arguments are passed to underlying spline class
    def __init__(self, degree = (3, 3), method = 'RectBivariateSpline', **kwargs):
        GridFit.__init__(self)
        ## @var degree
        # Degree of spline as a sequence of 2 elements, one for x and one for y direction
        self.degree = degree
        try:
            ## @var _splineClass
            # Class in scipy.interpolate to use for spline fitting
            self._splineClass = dierckx.__dict__[method]
        except KeyError:
            raise KeyError, 'Spline class "' + method + r'" unknown - should be one of: ' \
                  + ' '.join([name for name in dierckx.__dict__.iterkeys() if name.find('BivariateSpline') >= 0])
        ## @var _extraArgs
        # Extra keyword arguments to spline class
        self._extraArgs = kwargs
        ## @var _interp
        # Interpolator function, only set after fit()
        self._interp = None

    ## Fit spline to 2-D data on a rectangular grid.
    # This fits a scalar function defined on 2-D data to the provided grid. The first sequence in x defines
    # the M 'x' axis ticks (in any order), while the second sequence in x defines the N 'y' axis ticks
    # (also in any order). The provided function output y contains the corresponding 'z' values on the 
    # grid, in an array of shape (M, N). The minimum number of data points is (degree[0]+1)*(degree[1]+1).
    # @param self The current object
    # @param x    Known input grid specified by sequence of 2 sequences of axis ticks (of lengths M and N)
    # @param y    Known output values as a 2-D numpy array of shape (M, N)
    # pylint: disable-msg=W0142
    def fit(self, x, y):
        # Check dimensions of known data
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        y = np.atleast_2d(np.asarray(y))
        if (len(x) != 2) or (len(x[0].shape) != 1) or (len(x[1].shape) != 1) or (len(y.shape) != 2) or \
           (y.shape[0] != len(x[0])) or (y.shape[1] != len(x[1])):
            raise ValueError, "Spline interpolator requires input data with shape [(M,), (N,)] " \
                              " and output data with shape (M, N), got " + str([ax.shape for ax in x]) + \
                              " and " + str(y.shape) + " instead."
        if y.size < (self.degree[0] + 1) * (self.degree[1] + 1):
            raise ValueError, "Not enough data points for spline fit: requires at least " + \
                              str((self.degree[0] + 1) * (self.degree[1] + 1)) + ", only got " + str(y.size)
        # Ensure that 'x' and 'y' coordinates are both in ascending order (requirement of underlying regrid)
        xs, ys, zs = sort_grid(x[0], x[1], y)
        self._interp = self._splineClass(xs, ys, zs, kx = self.degree[0], ky = self.degree[1], **self._extraArgs)

    ## Evaluate spline on a new rectangular grid.
    # Evaluates the fitted scalar function on 2-D grid provided in x. The first sequence in x defines
    # the M 'x' axis ticks (in any order), while the second sequence in x defines the N 'y' axis ticks (also in
    # any order). The function returns the corresponding 'z' values on the grid, in an array of shape (M, N).
    # @param self The current object
    # @param x    2-D input grid specified by sequence of 2 sequences of axis ticks (of lengths M and N)
    # @return     Output of function as a 2-D numpy array of shape (M, N)
    def __call__(self, x):
        # Check dimensions
        x = [np.atleast_1d(np.asarray(ax)) for ax in x]
        if (len(x) != 2) or (len(x[0].shape) != 1) or (len(x[1].shape) != 1):
            raise ValueError, "Spline interpolator requires input data with shape [(M,), (N,)], got " + \
                              str([ax.shape for ax in x]) + " instead."
        if self._interp == None:
            raise AttributeError, "Spline not fitted to data yet - first call 'fit'."
        # The standard DIERCKX 2-D spline evaluation function (bispev) expects a rectangular grid in ascending order
        # Therefore, sort coordinates, evaluate on the sorted grid, and return the desorted result
        return desort_grid(x[0], x[1], self._interp(sorted(x[0]), sorted(x[1])))

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Baseline1DFit
#----------------------------------------------------------------------------------------------------------------------

## Fits a baseline to 1-D data.
# This fits a specified interpolation function (typically a polynomial or spline) to
# a 1-D x-y data sequence, where this function serves as the "baseline" of the data.
# The baseline is defined as a low-order function that closely approximates the 
# lowest parts of the y data across the entire sequence. That is, the data sequence
# is assumed to be the sum of a smooth global baseline component and a compact positive
# component, and optionally some noise component. The fitting process figures out the
# location of the compact part, and uses the remaining data to estimate the baseline
# component. The compact part has to be wide enough to prevent latching onto a random
# peak. The parts of the data used to estimate the baseline is stored in the object,
# as well as residual statistics. Future versions might use multiple compact components.
class Baseline1DFit(ScatterFit):
    
    ## Initialiser.
    # @param self The current object
    # @param interp ScatterFit object that will be fit to baseline segments
    # @param minWidth Minimum width for a gap in baseline (in units of x)
    # @param maxProb Maximum probability tolerated for observing a gap in baseline
    #                if it is assumed the gap occurred by chance
    # @param widthFunc Optional function that transforms x to another domain
    #                  for performing width calculations
    # pylint: disable-msg=E0602,R0914
    def __init__(self, interp, minWidth, maxProb, widthFunc=(lambda x: x)):
        ScatterFit.__init__(self)
        ## @var _baseline
        # Internal interpolator object that represents the actual baseline
        self._baseline = interp
        ## @var _minWidth
        # Minimum width for a gap in baseline, which is typically associated with
        # beamwidth, especially when widthFunc is also specified
        self._minWidth = minWidth
        ## @var _maxProb
        # If a contiguous gap in the baseline has a higher probability to occur
        # by chance than maxProb, it is ignored and considered part of the baseline
        self._maxProb = maxProb
        ## @var _widthFunc
        # Function that transforms x to another domain for performing width
        # calculations, useful when the unit of x is not physically meaningful
        self._widthFunc = widthFunc
        ## @var partOfBaseline
        # Vector of bools that indicates which parts of x were used to estimate
        # the baseline
        self.partOfBaseline = None
        ## @var residualStdev
        # Standard deviation of residuals of the last fit
        self.residualStdev = None
        ## @var durbinWatson
        # The Durbin-Watson statistic of the last fit, which is an indication of
        # the correlation between successive residual values. It ranges from 0 to 4.
        # Ideally it should be 2, which means uncorrelated residuals. A value of 0
        # means maximum positive correlation (indicating data that slowly drifts 
        # around the fitted function), while a value of 4 means maximum negative
        # correlation (indicating data that oscillates too fast around the fitted
        # function). A value of above 1 is considered "random enough".
        self.durbinWatson = None
        
    ## Fit baseline function to data.
    # @param self The current object
    # @param x    Known input values as a 1-D numpy array or sequence
    # @param y    Known output values as a 1-D numpy array, or sequence
    def fit(self, x, y):
        N = len(x)
        # Initially the entire data sequence is considered part of the baseline
        self.partOfBaseline = np.array(N * [True])
        newPartOfBaseline = np.array(N * [True])
        # As a safety measure, cap the number of iterations for pathological cases
        # that oscillate instead of converging (usually converges in a few iterations)
        # pylint: disable-msg=W0612
        for iteration in xrange(20):
            # Fit baseline function to parts of data designated as baseline
            baselineX, baselineY = x[self.partOfBaseline], y[self.partOfBaseline]
            self._baseline.fit(baselineX, baselineY)
            # Divide data into points above and below the fitted baseline
            abovebelow = y > self._baseline(x)
            # Calculate residual stats
            r = baselineY - self._baseline(baselineX)
            self.residualStdev = r.std()
            dr = np.diff(r)
            self.durbinWatson = np.dot(dr, dr) / np.dot(r, r)
            # Find runs of points above and below the fitted baseline, stored as
            # the first element in each run (and ending on the one-past-end index)
            runBorders = np.arange(N)[np.diff(abovebelow)] + 1
            runBorders = np.array([0] + runBorders.tolist() + [N])
            # Calculate length of each run, and whether it is above or below the line
            runLengths = np.diff(runBorders)
            numRuns = len(runLengths)
            runAbove = (np.arange(numRuns) % 2) != abovebelow[0]
            # Sort runs above the baseline according to decreasing length
            # sortedRunsAbove = np.flipud(np.arange(numRuns)[runAbove][runLengths[runAbove].argsort()])
            # Find index of longest contiguous run above the baseline, as candidate gap
            gap = np.arange(numRuns)[runAbove][runLengths[runAbove].argmax()]
            gapStart, gapEnd = runBorders[gap], runBorders[gap+1]
            # Check that "physical" width of gap exceeds minimum
            if np.abs(self._widthFunc(x[gapEnd]) - self._widthFunc(x[gapStart])) < self._minWidth:
                break
            # The probability of observing at least 1 length-k subsequence of the same type
            # (heads/tails or True/False) in a sequence of total length N, assuming the
            # heads/tails are equiprobable, is approximately P = 1 - (1 - 0.5^k) ^ (N-k)
            segmentProb = 1.0 - (1.0 - 0.5 ** runLengths[gap]) ** (N - runLengths[gap])
            # Check that "statistical" width of gap exceeds some minimum
            if segmentProb > self._maxProb:
                break
            # Let baseline segments be everything except the largest gap
            newPartOfBaseline[:] = True
            newPartOfBaseline[gapStart:gapEnd] = False
            # If the baseline segments did not change from the previous iteration, we are done
            if np.equal(newPartOfBaseline, self.partOfBaseline).all():
                break
            self.partOfBaseline[:] = newPartOfBaseline
    
    ## Evaluate function 'y = f(x)' on new data.
    # @param self The current object
    # @param x    Input to function as a 1-D numpy array, or sequence
    # @return     Output of function as a 1-D numpy array
    def __call__(self, x):
        return self._baseline(x)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  SampledTemplateFit
#----------------------------------------------------------------------------------------------------------------------

#class SampledTemplateFit(ScatterFit):
