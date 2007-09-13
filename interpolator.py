## @file interpolator.py
#
# Classes for encapsulating interpolator functions.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>
# @date 2007-08-28

# pylint: disable-msg=C0103,R0903

import scipy.optimize as optimize
import scipy.sandbox.delaunay as delaunay
import numpy as np
import copy
import logging

logger = logging.getLogger("xdmsbe.xdmsbelib.interpolator")

#----------------------------------------------------------------------------------------------------------------------
#--- FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------

## Flatten array, but not necessarily all the way to a 1-D array.
# This function is useful for broadcasting functions of arbitrary dimensionality along a given array. The array x is
# transposed and reshaped, so that the axes with indices listed in flattenAxes are collected either at the start or
# end of the array (based on the moveToStart flag). These axes are also flattened to a single axis, while preserving 
# the total number of elements in the array. The reshaping and transposition usually results in a view of the 
# original array, although a copy may result e.g. if discontiguous flattenAxes are chosen. The two extreme cases are
# flattenAxes = [] or None, which results in the original array with no flattening, and flattenAxes = 
# range(len(x.shape)), which is equivalent to x.ravel() and therefore full flattening.
# 
# Examples:
# x.shape => (2,4,10)
# semi_flatten(x, [], True).shape => (2,4,10) [no flattening, x returned unchanged]
# semi_flatten(x, (1), True).shape => (4,2,10)
# semi_flatten(x, (1), False).shape => (2,10,4)
# semi_flatten(x, (0,2), True).shape => (20,4)
# semi_flatten(x, (0,2), False).shape => (4,20)
# semi_flatten(x, (0,1,2), True).shape => (80,) [same as x.ravel()]
#
# @param x           Numpy array, or sequence
# @param flattenAxes List of axes along which x should be flattened
# @param moveToStart Flag indicating whether flattened axis is moved to start or end of array [default=True]
# @return            Semi-flattened version of x, as numpy array
def semi_flatten(x, flattenAxes, moveToStart=True):
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

## Restore an array that was reshaped by semi_flatten().
# @param x             Numpy array, or sequence
# @param flattenAxes   List of (original) axes along which x was flattened
# @param originalShape Original shape of x, before flattening
# @param moveFromStart Flag indicating whether flattened axes were moved to start or end of array [default=True]
# @return              Restored version of x, as numpy array
def semi_unflatten(x, flattenAxes, originalShape, moveFromStart=True):
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


#----------------------------------------------------------------------------------------------------------------------
#--- INTERFACE :  Interpolator
#----------------------------------------------------------------------------------------------------------------------

## Interface object for interpolator functions.
# This defines the interface for interpolator functions, which are derived from this class.
class Interpolator(object):
    
    ## Initialiser
    # The initialiser should be used to specify parameters of the interpolator function,
    # such as polynomial degree.
    # @param self The current object
    def __init__(self):
        pass
    
    ## Fit function 'y = f(x)' to data.
    # This function should reset any state associated with previous (x,y) data fits, and preserve
    # all state that was set by the initialiser.
    # @param self The current object
    # @param x    Known input values as a numpy array
    # @param y    Known output values as a numpy array
    # 
    def fit(self, x, y):
        raise NotImplementedError
    
    ## Evaluate function 'y = f(x)' on new data.
    # @param self The current object
    # @param x    Input to function as a numpy array
    # @return     Output of function as a numpy array
    def __call__(self, x):
        raise NotImplementedError

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  PolynomialFit
#----------------------------------------------------------------------------------------------------------------------

## Fits polynomial to 1-D data.
# This uses numpy's polyfit and polyval.
class PolynomialFit(Interpolator):
    ## Initialiser.
    # @param self      The current object
    # @param maxDegree Maximum polynomial degree to use (reduced if there are not enough data points)
    # @param rcond     Relative condition number of fit
    #                  (smallest singular value that will be used to fit polynomial, has sensible default)
    def __init__(self, maxDegree, rcond=None):
        Interpolator.__init__(self)
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
        x = np.atleast_1d(np.asarray(x))
        y = np.atleast_1d(np.asarray(y))
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
# This allows any Interpolator to fit the reciprocal of a data set, without having to invert the data
# and the results explicitly.
class ReciprocalFit(Interpolator):
    ## Initialiser
    # @param self The current object
    # @param interp Interpolator object to use on the reciprocal of the data
    def __init__(self, interp):
        Interpolator.__init__(self)
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
class Independent1DFit(Interpolator):
    ## Initialiser
    # @param self The current object
    # @param interp Interpolator object to use on each 1-D segment
    # @param axis Axis of 'y' matrix which will vary with the independent 'x' variable
    def __init__(self, interp, axis):
        Interpolator.__init__(self)
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
#--- CLASS :  Delaunay2DFit
#----------------------------------------------------------------------------------------------------------------------

## Interpolates a scalar function of 2-D data, based on Delaunay triangulation.
# The x data for this object should have two rows, containing the 'x' and 'y' coordinates of points in a plane.
# The 2-D points are therefore stored as column vectors in x. The y data for this object is a 1-D array, which
# represents the scalar 'z' value of the function defined on the plane (the symbols in quotation marks are the
# names for these variables used in the delaunay documentation.)
class Delaunay2DFit(Interpolator):
    ## Initialiser
    # @param self       The current object
    # @param interpType String indicating type of interpolation ('linear' or 'nn': only 'nn' currently supported)
    # @param defaultVal Default value used when trying to extrapolate beyond convex hull of known data [default=NaN]
    def __init__(self, interpType='nn', defaultVal=np.nan):
        Interpolator.__init__(self)
        ## @var interpType
        # String indicating type of interpolation ('linear' or 'nn')
        self.interpType = interpType
        ## @var defaultVal
        # Default value used when trying to extrapolate beyond convex hull of known data
        self.defaultVal = defaultVal
        ## @var _interp
        # Interpolator function, only set after fit()
        self._interp = None
    
    ## Fit function 'y = f(x)' to data.
    # This fits a scalar function defined on 2-D data to the provided x-y pairs.
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
        tri = delaunay.Triangulation(x[0], x[1])
        if self.interpType == 'linear':
            # Use 'nn' throughout until linear interpolation works in scipy
            self._interp = tri.nn_interpolator(y, default_value=self.defaultVal)
        else:
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
#--- CLASS :  NonLinearLeastSquaresFit
#----------------------------------------------------------------------------------------------------------------------

## Fits a generic function to data, based on non-linear least squares optimisation of a parameter vector.
# This fits a function of the form 'y = f(p,x)' to x-y data, where the parameter vector p is optimised via
# least squares. It is assumed that the data presented to fit() consist of stacks of x and y arrays, where
# each element in the x stack is of the right shape to serve as input to f(), and each element of the y stack
# is compatible with the output of f(). The (list of) axes along which the x and y data are stacked, can be 
# specified independently.
# @todo Upgrade single axis stacking to multi-dimensional stacking, with helper functions to perform
#       "partial unravelling" of arrays (and the reverse process). This is also useful for Independent1DFit.
class NonLinearLeastSquaresFit(Interpolator):
    ## Initialiser.
    # @param self    The current object
    # @param func    Generic function to be fit to x-y data, of the form 'y = f(p,x)'
    # @param params0 Initial guess of function parameter vector p
    # @param xAxes   List of axes along which x data are stacked [default=None]
    # @param yAxes   List of axes along which y data are stacked [default=None]
    # @param method  Optimisation method (name of corresponding scipy.optimize function) [default='fmin_bfgs']
    # @param kwargs  Additional keyword arguments are passed to underlying optimiser
    # pylint: disable-msg=R0913
    def __init__(self, func, params0, xAxes=None, yAxes=None, method='fmin_bfgs', **kwargs):
        Interpolator.__init__(self)
        ## @var func
        # Generic function object to be fit to data
        self.func = func
        ## @var params
        # Function parameter vector, either initial guess or final optimal value
        self.params = params0
        ## @var _xAxes
        # Axes along which x data are stacked
        self._xAxes = xAxes
        ## @var _yAxes
        # Axes along which y data are stacked
        self._yAxes = yAxes
        try:
            ## @var _optimizer
            # Optimiser method from scipy.optimize to use
            self._optimizer = optimize.__dict__[method]
        except KeyError:
            raise KeyError, 'Optimisation method "' + method + '" unknown - should be one of:\n' \
                            + str(optimize.__dict__.iterkeys())
        ## @var _optimArgs
        # Extra keyword arguments to optimiser
        self._optimArgs = kwargs
        self._optimArgs.update({'full_output': 1})
        ## @var _yShape
        # Saved shape of output data, only set after fit()
        self._yShape = None
        
    ## Fit function to data, by performing non-linear least squares optimisation.
    # This determines the optimal parameter vector p* so that the function 'y = f(p,x)' best fits the
    # observed x-y data, in a least-squares sense.
    # @param self The current object
    # @param x    Known input values as a numpy array
    # @param y    Known output values as a numpy array
    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        self._yShape = y.shape
        # Sum-of-squares cost function to be minimised
        def cost(p):
            if not self._xAxes:
                yfunc = self.func(p, x)
            else:
                yfunc = np.array([self.func(p, xx) for xx in semi_flatten(x, self._xAxes)])
            if self._optimizer.__name__ == 'leastsq':
                return (semi_flatten(y, self._yAxes) - yfunc).ravel()
            else:
                return ((semi_flatten(y, self._yAxes) - yfunc)**2).sum()
        # Do optimisation
        # pylint: disable-msg=W0142
        self.params = self._optimizer(cost, self.params, **self._optimArgs)[0]
    
    ## Evaluate fitted function on new data.
    # Evaluates the fitted function 'y = f(p*,x)' on new x data.
    # @param self The current object
    # @param x    Input to function as a numpy array
    # @return     Output of function as a numpy array
    def __call__(self, x):
        x = np.asarray(x)
        if not self._xAxes:
            y = self.func(self.params, x)
        else:
            y = np.array([self.func(self.params, xx) for xx in semi_flatten(x, self._xAxes)])
        return semi_unflatten(y, self._yAxes, self._yShape)

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  GaussianFit
#----------------------------------------------------------------------------------------------------------------------

##
#class GaussianFit(Interpolator):
    ## Initialiser
    # @param self The current object
#    def __init__(self, dim, interp):
#        Interpolator.__init__(self)
#        self.dim = dim
    
    
    ## Fit a Gaussian to data.
    # @param self The current object
    # @param x    Known input values as a numpy array
    # @param y    Known output values as a numpy array
#    def fit(self, x, y):
#        pass
    
    ## Evaluate function 'y = f(x)' on new data.
    # Evaluates the fitted scalar function on 2-D data provided in x.
    # @param self The current object
    # @param x    Input to function as a 2-D numpy array, or sequence (of shape (2,N))
    # @return     Output of function as a 1-D numpy array (of shape (N))
#    def __call__(self, x):
#        pass
    
#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  SampledTemplateFit
#----------------------------------------------------------------------------------------------------------------------

#class SampledTemplateFit(Interpolator):

