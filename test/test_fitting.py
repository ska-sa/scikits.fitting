## @file test_fitting.py
#
# Unit tests for fitting functions.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>
# @date 2007-09-04

import unittest
import xdmsbe.xdmsbelib.fitting as fitting
import numpy as np
import numpy.random as random

class Polynomial1DFitTestCases(unittest.TestCase):
    
    def setUp(self):
        self.poly = np.array([1.0, -2.0, 1.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = np.polyval(self.poly, self.x)
    
    def test_fit_eval(self):
        interp = fitting.Polynomial1DFit(2)
        self.assertRaises(AttributeError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertAlmostEqual(interp._mean, 0.0, places=10)
        np.testing.assert_almost_equal(interp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
    
    def test_reduce_degree(self):
        interp = fitting.Polynomial1DFit(2)
        interp.fit([1.0],[1.0])
        np.testing.assert_almost_equal(interp.poly, [1.0], decimal=10)

class ReciprocalFitTestCases(unittest.TestCase):
    
    def setUp(self):
        self.poly = np.array([1.0, 2.0, 10.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = 1.0 / np.polyval(self.poly, self.x)
    
    def test_fit_eval(self):
        interp = fitting.ReciprocalFit(fitting.Polynomial1DFit(2))
        self.assertRaises(AttributeError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertAlmostEqual(interp._interp._mean, 0.0, places=10)
        np.testing.assert_almost_equal(interp._interp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)

class Independent1DFitTestCases(unittest.TestCase):
    
    def setUp(self):
        self.poly1 = np.array([1.0, -2.0, 20.0])
        self.poly2 = np.array([1.0, 2.0, 10.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = np.ndarray(shape=(2,7,3))
        self.yTooLowDim = np.zeros(shape=(3))
        self.yWrongSize = np.zeros(shape=(2,5,3))
        self.axis = 1
        self.y[0,:,0] = np.polyval(self.poly1, self.x)
        self.y[0,:,1] = np.polyval(self.poly2, self.x)
        self.y[0,:,2] = np.polyval(self.poly1, self.x)
        self.y[1,:,0] = np.polyval(self.poly2, self.x)
        self.y[1,:,1] = np.polyval(self.poly1, self.x)
        self.y[1,:,2] = np.polyval(self.poly2, self.x)
    
    def test_fit_eval(self):
        interp = fitting.Independent1DFit(fitting.Polynomial1DFit(2), self.axis)
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.x, self.yTooLowDim)
        self.assertRaises(ValueError, interp.fit, self.x, self.yWrongSize)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertEqual(interp._axis, self.axis)
        self.assertEqual(interp._interps.shape, (2,3))
        np.testing.assert_almost_equal(interp._interps[0,0].poly, self.poly1, decimal=10)
        np.testing.assert_almost_equal(interp._interps[0,1].poly, self.poly2, decimal=10)
        np.testing.assert_almost_equal(interp._interps[0,2].poly, self.poly1, decimal=10)
        np.testing.assert_almost_equal(interp._interps[1,0].poly, self.poly2, decimal=10)
        np.testing.assert_almost_equal(interp._interps[1,1].poly, self.poly1, decimal=10)
        np.testing.assert_almost_equal(interp._interps[1,2].poly, self.poly2, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)

class Delaunay2DFitTestCases(unittest.TestCase):
    
    def setUp(self):
        # Square diamond shape
        self.x = np.array([[-1,0,0,0,1], [0,-1,0,1,0]])
        self.y = np.array([1,1,1,1,1])
        self.testx = np.array([[-0.5,0,0.5,0], [0,-0.5,0.5,0]])
        self.testy = np.array([1,1,1,1])
        self.defaultVal = 100
        self.outsidex = np.array([[10],[10]])
        self.outsidey = np.array([self.defaultVal])
    
    def test_fit_eval_linear(self):
        interp = fitting.Delaunay2DFit('linear', defaultVal=self.defaultVal)
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=10)
        np.testing.assert_almost_equal(outsidey, self.outsidey, decimal=10)
    
    def test_fit_eval_nn(self):
        interp = fitting.Delaunay2DFit('nn', defaultVal=self.defaultVal)
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=10)
        np.testing.assert_almost_equal(outsidey, self.outsidey, decimal=10)

class NonLinearLeastSquaresFitTestCases(unittest.TestCase):
    
    def setUp(self):
        # Quadratic function centred at p
        func = lambda p, x: ((x - p)**2).sum()
        self.vFunc = fitting.vectorizeFitFunc(func)
        self.trueParams = np.array([1, -4])
        self.initParams = np.array([0, 0])
        self.x = 4.0*np.random.randn(20, 2)
        self.y = self.vFunc(self.trueParams, self.x)
        # 2-D log Gaussian function
        def lngauss_diagcov(p, x):
            xminmu = x - p[np.newaxis, 0:2]
            return p[4] - 0.5 * np.dot(xminmu * xminmu, p[2:4])
        self.func2 = lngauss_diagcov
        self.trueParams2 = np.array([3, -2, 10, 10, 4])
        self.initParams2 = np.array([0, 0, 1, 1, 0])
        self.x2 = np.random.randn(80, 2)
        self.y2 = lngauss_diagcov(self.trueParams2, self.x2)
    
    def test_fit_eval_func1(self):
        self.assertRaises(KeyError, fitting.NonLinearLeastSquaresFit, \
                          self.vFunc, self.initParams, method='bollie')
        interp = fitting.NonLinearLeastSquaresFit(self.vFunc, self.initParams, method='fmin_bfgs', disp=0)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.params, self.trueParams, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=5)
    
    def test_fit_eval_gauss(self):
        interp2 = fitting.NonLinearLeastSquaresFit(self.func2, self.initParams2, method='leastsq')
        interp2.fit(self.x2, self.y2)
        y2 = interp2(self.x2)
        np.testing.assert_almost_equal(interp2.params, self.trueParams2, decimal=10)
        np.testing.assert_almost_equal(y2, self.y2, decimal=10)

class GaussianFitTestCases(unittest.TestCase):
    
    def setUp(self):
        # For a more challenging fit, move the true mean away from the origin, i.e. away from the region
        # being randomly sampled in self.x. Fitting a Gaussian to a segment that does not contain a clear peak
        # works fine if the fit is done to the log of the data, but fails in the linear domain.
        self.trueMean, self.trueVar, self.trueHeight = [0, 0], [10, 20], 4
        trueGauss = fitting.GaussianFit(self.trueMean, self.trueVar, self.trueHeight)
        self.x = 7*np.random.randn(80, 2)
        self.y = trueGauss(self.x)
        self.initMean, self.initVar, self.initHeight = [3, -2], [1, 1], 1
    
    def test_fit_eval_diagcov(self):
        interp = fitting.GaussianFit(self.initMean, self.initVar, self.initHeight)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.mean, self.trueMean, decimal=7)
        np.testing.assert_almost_equal(interp.var, self.trueVar, decimal=7)
        np.testing.assert_almost_equal(interp.height, self.trueHeight, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=7)

class Spline1DFitTestCases(unittest.TestCase):
    
    def setUp(self):
        # Training data is randomly sampled parabola
        self.poly = np.array([1.0, -2.0, 1.0])
        self.x = np.random.randn(40)
        self.y = np.polyval(self.poly, self.x)
        # Test data is random samples of same parabola, but ensure that samples do not fall outside training set
        self.testx = 0.2*np.random.randn(40)
        self.testy = np.polyval(self.poly, self.testx)
    
    def test_fit_eval(self):
        interp = fitting.Spline1DFit(3)
        self.assertRaises(AttributeError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=10)

class Spline2DFitTestCases(unittest.TestCase):
    
    def setUp(self):
        # Training data is randomly sampled parabola
        poly = np.array([1.0, 2.0, 1.0])
        self.x = np.random.randn(2, 100)
        self.y = poly[0]*self.x[0]**2 + poly[1]*self.x[0]*self.x[1] + poly[2]*self.x[1]**2
        # Test data is random samples of same parabola, but ensure that samples do not fall outside training set
        self.testx = 0.2*np.random.randn(2, 100)
        self.testy = poly[0]*self.testx[0]**2 + poly[1]*self.testx[0]*self.testx[1] + poly[2]*self.testx[1]**2
    
    def test_fit_eval(self):
        interp = fitting.Spline2DFit((3, 3))
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=10)

class RandomisedFitTestCases(unittest.TestCase):
    
    def setUp(self):
        self.poly = np.array([1.0, -2.0, 1.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = np.polyval(self.poly, self.x)
        self.numRuns = 100
        self.yNoisy = self.y + 0.01*random.randn(self.numRuns, len(self.y))
    
    def test_randomised_polyfit(self):
        interp = fitting.Polynomial1DFit(2)
        # Perfect fit (no noise)
        interp.fit(self.x, self.y)
        randomInterp = fitting.randomise(interp, self.x, self.y, 'unknown')
        y = randomInterp(self.x)
        np.testing.assert_almost_equal(randomInterp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        randomInterp = fitting.randomise(interp, self.x, self.y, 'shuffle')
        y = randomInterp(self.x)
        np.testing.assert_almost_equal(randomInterp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        # Fit polynomials to a set of noisy samples
        noisyPoly = []
        for noisyY in self.yNoisy:
            interp.fit(self.x, noisyY)
            noisyPoly.append(interp.poly)
        noisyPoly = np.array(noisyPoly)
        # Randomise polynomial fit to first noisy sample in various ways
        shufflePoly = np.array([fitting.randomise(interp, self.x, self.yNoisy[0], 'shuffle').poly \
                                for n in range(self.numRuns)])
        np.testing.assert_almost_equal(shufflePoly.mean(axis=0), noisyPoly[0], decimal=2)
        np.testing.assert_almost_equal(shufflePoly.std(axis=0), noisyPoly.std(axis=0), decimal=2)
        normalPoly = np.array([fitting.randomise(interp, self.x, self.yNoisy[0], 'normal').poly \
                               for n in range(self.numRuns)])
        np.testing.assert_almost_equal(normalPoly.mean(axis=0), noisyPoly[0], decimal=2)
        np.testing.assert_almost_equal(normalPoly.std(axis=0), noisyPoly.std(axis=0), decimal=2)
        bootPoly = np.array([fitting.randomise(interp, self.x, self.yNoisy[0], 'bootstrap').poly \
                             for n in range(self.numRuns)])
        np.testing.assert_almost_equal(bootPoly.mean(axis=0), noisyPoly[0], decimal=2)
        np.testing.assert_almost_equal(bootPoly.std(axis=0), noisyPoly.std(axis=0), decimal=2)
