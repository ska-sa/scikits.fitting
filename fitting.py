## @file fitting.py
#
# Classes for encapsulating interpolator functions.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>
# @date 2007-08-28

# pylint: disable-msg=C0103,R0903

import scipy.optimize as optimize           # NonLinearLeastSquaresFit
import scipy.sandbox.delaunay as delaunay   # Delaunay2DFit
import scipy.interpolate as dierckx         # Spline1DFit, Spline2DFit
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
#--- INTERFACE :  GenericFit
#----------------------------------------------------------------------------------------------------------------------

## Interface object for interpolator functions.
# This defines the interface for interpolator functions, which are derived from this class.
class GenericFit(object):
    
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
    def fit(self, x, y):
        raise NotImplementedError
    
    ## Evaluate function 'y = f(x)' on new data.
    # @param self The current object
    # @param x    Input to function as a numpy array
    # @return     Output of function as a numpy array
    def __call__(self, x):
        raise NotImplementedError

#----------------------------------------------------------------------------------------------------------------------
#--- CLASS :  Polynomial1DFit
#----------------------------------------------------------------------------------------------------------------------

## Fits polynomial to 1-D data.
# This uses numpy's polyfit and polyval.
class Polynomial1DFit(GenericFit):
    ## Initialiser.
    # @param self      The current object
    # @param maxDegree Maximum polynomial degree to use (reduced if there are not enough data points)
    # @param rcond     Relative condition number of fit
    #                  (smallest singular value that will be used to fit polynomial, has sensible default)
    def __init__(self, maxDegree, rcond=None):
        GenericFit.__init__(self)
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
# This allows any GenericFit object to fit the reciprocal of a data set, without having to invert the data
# and the results explicitly.
class ReciprocalFit(GenericFit):
    ## Initialiser
    # @param self The current object
    # @param interp GenericFit object to use on the reciprocal of the data
    def __init__(self, interp):
        GenericFit.__init__(self)
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
class Independent1DFit(GenericFit):
    ## Initialiser
    # @param self The current object
    # @param interp GenericFit object to use on each 1-D segment
    # @param axis Axis of 'y' matrix which will vary with the independent 'x' variable
    def __init__(self, interp, axis):
        GenericFit.__init__(self)
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
# names for these variables used in the delaunay documentation.) The 2-D x coordinates do not have to lie on a 
# regular grid, and can be in any order. Jittering a regular grid seems to be troublesome, though...
class Delaunay2DFit(GenericFit):
    ## Initialiser
    # @param self       The current object
    # @param interpType String indicating type of interpolation ('linear' or 'nn': only 'nn' currently supported)
    # @param defaultVal Default value used when trying to extrapolate beyond convex hull of known data [default=NaN]
    # @param jitter     True to add small amount of jitter to x to make degenerate triangulation unlikely [False]
    def __init__(self, interpType='nn', defaultVal=np.nan, jitter=False):
        GenericFit.__init__(self)
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
class NonLinearLeastSquaresFit(GenericFit):
    ## Initialiser.
    # @param self         The current object
    # @param func         Generic function to be fit to x-y data, of the form 'y = f(p,x)' (should be vectorised)
    # @param params0      Initial guess of function parameter vector p
    # @param funcJacobian Jacobian of function f, if available, with signature 'J = f(p,x)', where J has the
    #                     shape (y shape produced by f(p,x), len(p))
    # @param method       Optimisation method (name of corresponding scipy.optimize function) [default='leastsq']
    # @param kwargs       Additional keyword arguments are passed to underlying optimiser
    # pylint: disable-msg=R0913
    def __init__(self, func, params0, funcJacobian=None, method='leastsq', **kwargs):
        GenericFit.__init__(self)
        ## @var func
        # Generic function object to be fit to data
        self.func = func
        ## @var params
        # Function parameter vector, either initial guess or final optimal value
        self.params = params0
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
                return squash(residualJac, range(len(residualJac.shape)-1), moveToStart=True)
            if self._optimizer.__name__ == 'leastsq':
                self._extraArgs['Dfun'] = jacobian
            else:
                self._extraArgs['fprime'] = jacobian
        # Do optimisation
        # pylint: disable-msg=W0142
        self.params = self._optimizer(cost, self.params, **self._extraArgs)[0]
    
    ## Evaluate fitted function on new data.
    # Evaluates the fitted function 'y = f(p*,x)' on new x data.
    # @param self The current object
    # @param x    Input to function as a numpy array
    # @return     Output of function as a numpy array
    def __call__(self, x):
        return self.func(self.params, x)

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
class GaussianFit(GenericFit):
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
        GenericFit.__init__(self)
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
# This uses scipy.interpolate, which is based on Paul Dierckx's DIERCKX (or FITPACK) routines.
class Spline1DFit(GenericFit):
    ## Initialiser.
    # @param self   The current object
    # @param degree Degree of spline (in range 1-5) [3, i.e. cubic B-spline]
    # @param method Spline class (name of corresponding scipy.interpolate class) ['UnivariateSpline']
    # @param kwargs Additional keyword arguments are passed to underlying spline class
    def __init__(self, degree = 3, method = 'UnivariateSpline', **kwargs):
        GenericFit.__init__(self)
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
#--- CLASS :  Spline2DFit
#----------------------------------------------------------------------------------------------------------------------

## Fits a B-spline to 2-D data.
# This uses scipy.interpolate, which is based on Paul Dierckx's DIERCKX (or FITPACK) routines.
# The 2-D x coordinates do not have to lie on a regular grid, and can be in any order.
class Spline2DFit(GenericFit):
    ## Initialiser.
    # @param self   The current object
    # @param degree Degree (1-5) of spline in x and y directions [(3, 3), i.e. bicubic B-spline]
    # @param method Spline class (name of corresponding scipy.interpolate class) ['SmoothBivariateSpline']
    # @param kwargs Additional keyword arguments are passed to underlying spline class
    def __init__(self, degree = (3, 3), method = 'SmoothBivariateSpline', **kwargs):
        GenericFit.__init__(self)
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
    
    ## Fit spline to 2-D data.
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
    
    ## Evaluate spline on new data.
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
#--- CLASS :  SampledTemplateFit
#----------------------------------------------------------------------------------------------------------------------

#class SampledTemplateFit(GenericFit):
