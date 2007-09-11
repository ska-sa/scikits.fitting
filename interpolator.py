## @file interpolator.py
#
# Classes for encapsulating interpolator functions.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>
# @date 2007-08-28

# pylint: disable-msg=C0103,R0903

import scipy.sandbox.delaunay as delaunay
import numpy as np
import copy

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
        x = np.asarray(x)
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
        y = np.asarray(y)
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
# represents the scalar 'z' value of the function defined on the plane (all symbols in quotation marks refer to
# the versions found in the delaunay documentation.)
class Delaunay2DFit(Interpolator):
    ## Initialiser
    # @param self       The current object
    # @param interpType String indicating type of interpolation ('linear' or 'nn': only 'nn' currently supported)
    # @param defaultVal Default value used when trying to extrapolate beyond convex hull of known data [default=NaN]
    def __init__(self, interpType, defaultVal=np.nan):
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
#--- CLASS :  TemplateFunctionFit
#----------------------------------------------------------------------------------------------------------------------

#
#class TemplateFunctionFit(Interpolator):
#    ## Initialiser
#    # @param self The current object
#    # @param templateFunc Template function to be fit to x-y data
#    def __init__(self, templateFunc):
#        Interpolator.__init__(self)
#        self._templateFunc = templateFunc
    
#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  GaussianFit
#----------------------------------------------------------------------------------------------------------------------

#class GaussianFit(Interpolator):

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  SampledTemplateFit
#----------------------------------------------------------------------------------------------------------------------

#class SampledTemplateFit(Interpolator):

