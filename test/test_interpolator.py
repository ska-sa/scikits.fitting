## @file test_interpolator.py
#
# Unit tests for interpolator functions.
#
# copyright (c) 2007 SKA/KAT. All rights reserved.
# @author Ludwig Schwardt <ludwig@ska.ac.za>
# @date 2007-09-04

import unittest
import xdmsbe.xdmsbelib.interpolator as interpolator
import numpy as np

class Polynomial1DFitTestCases(unittest.TestCase):
    
    def setUp(self):
        self.poly = np.array([1.0, -2.0, 1.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = np.polyval(self.poly, self.x)
        
    def test_fit_eval(self):
        interp = interpolator.Polynomial1DFit(2)
        self.assertRaises(AttributeError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertAlmostEqual(interp._mean, 0.0, places=7)
        np.testing.assert_almost_equal(interp.poly, self.poly, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=7)
    
    def test_reduce_degree(self):
        interp = interpolator.Polynomial1DFit(2)
        interp.fit([1.0],[1.0])
        np.testing.assert_almost_equal(interp.poly, [1.0], decimal=7)
        
class ReciprocalFitTestCases(unittest.TestCase):
    
    def setUp(self):
        self.poly = np.array([1.0, 2.0, 10.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = 1.0 / np.polyval(self.poly, self.x)
    
    def test_fit_eval(self):
        interp = interpolator.ReciprocalFit(interpolator.Polynomial1DFit(2))
        self.assertRaises(AttributeError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertAlmostEqual(interp._interp._mean, 0.0, places=7)
        np.testing.assert_almost_equal(interp._interp.poly, self.poly, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=7)

class Independent1DFit(unittest.TestCase):
    
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
        interp = interpolator.Independent1DFit(interpolator.Polynomial1DFit(2), self.axis)
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.x, self.yTooLowDim)
        self.assertRaises(ValueError, interp.fit, self.x, self.yWrongSize)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertEqual(interp._axis, self.axis)
        self.assertEqual(interp._interps.shape, (2,3))
        np.testing.assert_almost_equal(interp._interps[0,0].poly, self.poly1, decimal=7)
        np.testing.assert_almost_equal(interp._interps[0,1].poly, self.poly2, decimal=7)
        np.testing.assert_almost_equal(interp._interps[0,2].poly, self.poly1, decimal=7)
        np.testing.assert_almost_equal(interp._interps[1,0].poly, self.poly2, decimal=7)
        np.testing.assert_almost_equal(interp._interps[1,1].poly, self.poly1, decimal=7)
        np.testing.assert_almost_equal(interp._interps[1,2].poly, self.poly2, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=7)

class Delaunay2DFit(unittest.TestCase):

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
        interp = interpolator.Delaunay2DFit('linear', defaultVal=self.defaultVal)
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        np.testing.assert_almost_equal(y, self.y, decimal=7)
        np.testing.assert_almost_equal(testy, self.testy, decimal=7)
        np.testing.assert_almost_equal(outsidey, self.outsidey, decimal=7)
        
    def test_fit_eval_nn(self):
        interp = interpolator.Delaunay2DFit('nn', defaultVal=self.defaultVal)
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        np.testing.assert_almost_equal(y, self.y, decimal=7)
        np.testing.assert_almost_equal(testy, self.testy, decimal=7)
        np.testing.assert_almost_equal(outsidey, self.outsidey, decimal=7)

class NonLinearLeastSquaresFit(unittest.TestCase):

    def setUp(self):
        # Quadratic function centred at p
        func = lambda p, x: ((x - p)**2).sum()
        self.vFunc = interpolator.vectorizeFitFunc(func)
        self.trueParams = np.array([1, -4])
        self.initParams = np.array([0, 0])
        self.x = 4.0*np.random.randn(20, 2)
        self.y = self.vFunc(self.trueParams, self.x)
        # 2-D log Gaussian function
        def lngauss_diagcov(p, x):
            xminmu = x - np.repeat(p[np.newaxis, 0:2], x.shape[0], axis=0)
            return p[4] - 0.5 * np.dot(xminmu * xminmu, p[2:4])
        self.func2 = lngauss_diagcov
        self.trueParams2 = np.array([3, -2, 10, 10, 4])
        self.initParams2 = np.array([0, 0, 1, 1, 0])
        self.x2 = np.random.randn(80, 2)
        self.y2 = lngauss_diagcov(self.trueParams2, self.x2)
    
    def test_fit_eval_func1(self):
        self.assertRaises(KeyError, interpolator.NonLinearLeastSquaresFit, \
                          self.vFunc, self.initParams, method='bollie')
        interp = interpolator.NonLinearLeastSquaresFit(self.vFunc, self.initParams, method='fmin_bfgs', disp=0)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.params, self.trueParams, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=5)
    
    def test_fit_eval_gauss(self):
        interp2 = interpolator.NonLinearLeastSquaresFit(self.func2, self.initParams2, method='leastsq')
        interp2.fit(self.x2, self.y2)
        y2 = interp2(self.x2)
        np.testing.assert_almost_equal(interp2.params, self.trueParams2, decimal=7)
        np.testing.assert_almost_equal(y2, self.y2, decimal=7)

class GaussianFit(unittest.TestCase):

    def setUp(self):
        self.trueMean, self.trueVar, self.trueHeight = [3, -2], [10, 20], 4
        trueGauss = interpolator.GaussianFit(self.trueMean, self.trueVar, self.trueHeight)
        self.x = np.random.randn(80, 2)
        self.y = trueGauss(self.x)
        self.initMean, self.initVar, self.initHeight = [0, 0], [1, 1], 1

    def test_fit_eval_diagcov(self):
        interp = interpolator.GaussianFit(self.initMean, self.initVar, self.initHeight)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.mean, self.trueMean, decimal=7)
        np.testing.assert_almost_equal(interp.var, self.trueVar, decimal=7)
        np.testing.assert_almost_equal(interp.height, self.trueHeight, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=7)
