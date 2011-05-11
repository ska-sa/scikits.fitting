"""Unit tests for the fitting module."""
# pylint: disable-msg=C0103,W0212

import unittest
import numpy as np
import scipy.interpolate
from scape import fitting

class UtilityFunctionsTestCases(unittest.TestCase):
    """Exercise utility functions."""

    def setUp(self):
        self.x = np.random.rand(2, 4, 10)

    def test_squash(self):
        """UtilityFunctions: Test squash and unsquash."""
        y1 = fitting.squash(self.x, [], True)
        y1a = fitting.squash(self.x, None, True)
        y2 = fitting.squash(self.x, (1), False)
        y3 = fitting.squash(self.x, (0, 2), True)
        y4 = fitting.squash(self.x, (0, 2), False)
        y5 = fitting.squash(self.x, (0, 1, 2), True)
        self.assertEqual(y1.shape, (2, 4, 10))
        self.assertEqual(y1a.shape, (2, 4, 10))
        self.assertEqual(y2.shape, (2, 10, 4))
        self.assertEqual(y3.shape, (20, 4))
        self.assertEqual(y4.shape, (4, 20))
        self.assertEqual(y5.shape, (80,))
        np.testing.assert_array_equal(fitting.unsquash(y1, [], (2, 4, 10), True), self.x)
        np.testing.assert_array_equal(fitting.unsquash(y1a, None, (2, 4, 10), True), self.x)
        np.testing.assert_array_equal(fitting.unsquash(y2, (1), (2, 4, 10), False), self.x)
        np.testing.assert_array_equal(fitting.unsquash(y3, (0, 2), (2, 4, 10), True), self.x)
        np.testing.assert_array_equal(fitting.unsquash(y4, (0, 2), (2, 4, 10), False), self.x)
        np.testing.assert_array_equal(fitting.unsquash(y5, (0, 1, 2), (2, 4, 10), True), self.x)

class LinearLeastSquaresFitTestCases(unittest.TestCase):
    """Fit linear regression model to data from a known model, and compare."""

    def setUp(self):
        self.params = np.array([0.1, -0.2, 0.0, 0.5, 0.5])
        self.N = 1000
        self.x = np.random.randn(len(self.params), self.N)
        self.y = np.dot(self.params, self.x)
        t = np.arange(0., 10., 10. / self.N)
        self.poly_x = np.vander(t, 5).T
        self.poly_y = np.dot(self.params, self.poly_x)

    def test_fit_eval(self):
        """LinearLeastSquaresFit: Basic function fitting and evaluation using data from a known function."""
        interp = fitting.LinearLeastSquaresFit()
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.params, self.params, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)

    def test_cov_params(self):
        """LinearLeastSquaresFit: Obtain sample statistics of parameters and compare to calculated covariance matrix."""
        interp = fitting.LinearLeastSquaresFit()
        std_y = 1.0
        M = 200
        param_set = np.zeros((len(self.params), M))
        for n in range(M):
            yn = self.poly_y + std_y * np.random.randn(len(self.poly_y))
            interp.fit(self.poly_x, yn, std_y)
            param_set[:, n] = interp.params
        mean_params = param_set.mean(axis=1)
        norm_params = param_set - mean_params[:, np.newaxis]
        cov_params = np.dot(norm_params, norm_params.T) / M
        std_params = np.sqrt(np.diag(interp.cov_params))
        self.assertTrue((np.abs(mean_params - self.params) / std_params < 0.25).all(),
                        "Sample mean parameter vector differs too much from true value")
        self.assertTrue((np.abs(cov_params - interp.cov_params) / np.abs(interp.cov_params) < 0.5).all(),
                        "Sample parameter covariance matrix differs too much from expected one")

    def test_vs_numpy(self):
        """LinearLeastSquaresFit: Compare fitter to np.linalg.lstsq."""
        interp = fitting.LinearLeastSquaresFit()
        interp.fit(self.x, self.y)
        params = np.linalg.lstsq(self.x.T, self.y)[0]
        np.testing.assert_almost_equal(interp.params, params, decimal=10)
        rcond = 1e-3
        interp = fitting.LinearLeastSquaresFit(rcond)
        interp.fit(self.poly_x, self.poly_y)
        params = np.linalg.lstsq(self.poly_x.T, self.poly_y, rcond)[0]
        np.testing.assert_almost_equal(interp.params, params, decimal=10)

class Polynomial1DFitTestCases(unittest.TestCase):
    """Fit a 1-D polynomial to data from a known polynomial, and compare."""

    def setUp(self):
        self.poly = np.array([1.0, -2.0, 1.0])
        # Zero mean case
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = np.polyval(self.poly, self.x)
        # Non-zero mean case
        self.x2 = np.arange(0., 10.0, 1.0)
        self.y2 = np.polyval(self.poly, self.x2)
        self.randx = np.random.randn(100)
        self.randp = np.random.randn(4)

    def test_fit_eval(self):
        """Polynomial1DFit: Basic function fitting and evaluation on known zero-mean data."""
        interp = fitting.Polynomial1DFit(2)
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertAlmostEqual(interp._mean, 0.0, places=10)
        np.testing.assert_almost_equal(interp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)

    def test_fit_eval2(self):
        """Polynomial1DFit: Basic function fitting and evaluation on known non-zero-mean data."""
        interp = fitting.Polynomial1DFit(2)
        interp.fit(self.x2, self.y2)
        y2 = interp(self.x2)
        np.testing.assert_almost_equal(interp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y2, self.y2, decimal=10)

    def test_cov_params(self):
        """Polynomial1DFit: Obtain sample statistics of parameters and compare to calculated covariance matrix."""
        interp = fitting.Polynomial1DFit(2)
        std_y = 1.3
        M = 200
        poly_set = np.zeros((len(self.poly), M))
        for n in range(M):
            yn = self.y2 + std_y * np.random.randn(len(self.y2))
            interp.fit(self.x2, yn, std_y)
            poly_set[:, n] = interp.poly
        mean_poly = poly_set.mean(axis=1)
        norm_poly = poly_set - mean_poly[:, np.newaxis]
        cov_poly = np.dot(norm_poly, norm_poly.T) / M
        std_poly = np.sqrt(np.diag(interp.cov_poly))
        self.assertTrue((np.abs(mean_poly - self.poly) / std_poly < 0.25).all(),
                        "Sample mean coefficient vector differs too much from true value")
        self.assertTrue((np.abs(cov_poly - interp.cov_poly) / np.abs(interp.cov_poly) < 0.5).all(),
                        "Sample coefficient covariance matrix differs too much from expected one")

    def test_vs_numpy(self):
        """Polynomial1DFit: Compare fitter to np.polyfit and np.polyval."""
        x, p = self.randx, self.randp
        y = p[0] * (x ** 3) + p[1] * (x ** 2) + p[2] * x + p[3]
        interp = fitting.Polynomial1DFit(3)
        interp.fit(x, y)
        interp_y = interp(x)
        np_poly = np.polyfit(x, y, 3)
        np_y = np.polyval(np_poly, x)
        self.assertAlmostEqual(interp._mean, self.randx.mean(), places=10)
        np.testing.assert_almost_equal(interp.poly, np_poly, decimal=10)
        np.testing.assert_almost_equal(interp_y, np_y, decimal=10)

    # pylint: disable-msg=R0201
    def test_reduce_degree(self):
        """Polynomial1DFit: Check that polynomial degree is reduced with too few data points."""
        interp = fitting.Polynomial1DFit(2)
        interp.fit([1.0], [1.0])
        np.testing.assert_almost_equal(interp.poly, [1.0], decimal=10)

class Polynomial2DFitTestCases(unittest.TestCase):
    """Fit a 2-D polynomial to data from a known polynomial, and compare."""

    def setUp(self):
        self.poly = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        self.degrees = (1, 2)
        # Zero mean case
        x1 = np.arange(-1., 1.1, 0.1)
        x2 = np.arange(-1., 1.2, 0.2)
        xx1, xx2 = np.meshgrid(x1, x2)
        self.x = X = np.vstack((xx1.ravel(), xx2.ravel()))
        A = np.c_[X[0] * X[1]**2, X[0] * X[1], X[0], X[1]**2, X[1], np.ones(X.shape[1])].T
        self.y = np.dot(self.poly, A)
        # Non-zero mean (and uneven scale) case
        x1 = np.arange(0., 10.)
        x2 = np.arange(0., 5.)
        xx1, xx2 = np.meshgrid(x1, x2)
        self.x2 = X = np.vstack((xx1.ravel(), xx2.ravel()))
        A = np.c_[X[0] * X[1]**2, X[0] * X[1], X[0], X[1]**2, X[1], np.ones(X.shape[1])].T
        self.y2 = np.dot(self.poly, A)

    def test_fit_eval(self):
        """Polynomial2DFit: Basic function fitting and evaluation on known zero-mean data."""
        interp = fitting.Polynomial2DFit(self.degrees)
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp._mean, [0.0, 0.0], decimal=10)
        np.testing.assert_almost_equal(interp._scale, [1.0, 1.0], decimal=10)
        np.testing.assert_almost_equal(interp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)

    def test_fit_eval2(self):
        """Polynomial2DFit: Basic function fitting and evaluation on known non-zero-mean data."""
        interp = fitting.Polynomial2DFit(self.degrees)
        interp.fit(self.x2, self.y2)
        y2 = interp(self.x2)
        np.testing.assert_almost_equal(interp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y2, self.y2, decimal=10)

    def test_cov_params(self):
        """Polynomial2DFit: Obtain sample statistics of parameters and compare to calculated covariance matrix."""
        interp = fitting.Polynomial2DFit(self.degrees)
        std_y = 1.7
        M = 200
        poly_set = np.zeros((len(self.poly), M))
        for n in range(M):
            yn = self.y2 + std_y * np.random.randn(len(self.y2))
            interp.fit(self.x2, yn, std_y)
            poly_set[:, n] = interp.poly
        mean_poly = poly_set.mean(axis=1)
        norm_poly = poly_set - mean_poly[:, np.newaxis]
        cov_poly = np.dot(norm_poly, norm_poly.T) / M
        std_poly = np.sqrt(np.diag(interp.cov_poly))
        self.assertTrue((np.abs(mean_poly - self.poly) / std_poly < 0.25).all(),
                        "Sample mean coefficient vector differs too much from true value")
        self.assertTrue((np.abs(cov_poly - interp.cov_poly) / np.abs(interp.cov_poly) < 0.5).all(),
                        "Sample coefficient covariance matrix differs too much from expected one")

class PiecewisePolynomial1DFitTestCases(unittest.TestCase):
    """Fit a 1-D piecewise polynomial to data from a known polynomial, and compare."""

    def setUp(self):
        self.poly = np.array([1.0, 2.0, 3.0, 4.0])
        self.x = np.linspace(-3.0, 2.0, 100)
        self.y = np.polyval(self.poly, self.x)

    def test_fit_eval(self):
        """PiecewisePolynomial1DFit: Basic function fitting and evaluation using data from a known function."""
        # Ignore test if SciPy version is below 0.7.0
        try:
            scipy.interpolate.PiecewisePolynomial
        except AttributeError:
            return
        interp = fitting.PiecewisePolynomial1DFit(max_degree=3)
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, [0, 0], [1, 2])
        interp.fit(self.x[::2], self.y[::2])
        y = interp(self.x)
        np.testing.assert_almost_equal(y[5:-5], self.y[5:-5], decimal=10)
        # Fit a single data point
        interp.fit(self.x[0], self.y[0])
        y = interp(self.x)
        np.testing.assert_equal(y, np.tile(self.y[0], self.x.shape))

    def test_stepwise_interp(self):
        """PiecewisePolynomial1DFit: Test underlying zeroth-order interpolator."""
        x = np.sort(np.random.rand(100)) * 4. - 2.5
        y = np.random.randn(100)
        interp = fitting.PiecewisePolynomial1DFit(max_degree=0)
        interp.fit(x, y)
        np.testing.assert_almost_equal(interp(x), y, decimal=10)
        np.testing.assert_almost_equal(interp(x + 1e-15), y, decimal=10)
        np.testing.assert_almost_equal(interp(x - 1e-15), y, decimal=10)
        np.testing.assert_almost_equal(fitting._stepwise_interp(x, y, x), y, decimal=10)
        np.testing.assert_almost_equal(interp(self.x), fitting._stepwise_interp(x, y, self.x), decimal=10)

    def test_linear_interp(self):
        """PiecewisePolynomial1DFit: Test underlying first-order interpolator."""
        x = np.sort(np.random.rand(100)) * 4. - 2.5
        y = np.random.randn(100)
        interp = fitting.PiecewisePolynomial1DFit(max_degree=1)
        interp.fit(x, y)
        np.testing.assert_almost_equal(interp(x), y, decimal=10)
        np.testing.assert_almost_equal(fitting._linear_interp(x, y, x), y, decimal=10)
        np.testing.assert_almost_equal(interp(self.x), fitting._linear_interp(x, y, self.x), decimal=10)

class ReciprocalFitTestCases(unittest.TestCase):
    """Check the ReciprocalFit class."""

    def setUp(self):
        self.poly = np.array([1.0, 2.0, 10.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = 1.0 / np.polyval(self.poly, self.x)

    def test_fit_eval(self):
        """ReciprocalFit: Basic function fitting and evaluation using data from a known function."""
        interp = fitting.ReciprocalFit(fitting.Polynomial1DFit(2))
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertAlmostEqual(interp._interp._mean, 0.0, places=10)
        np.testing.assert_almost_equal(interp._interp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)

class Independent1DFitTestCases(unittest.TestCase):
    """Check the Independent1DFit class."""

    def setUp(self):
        self.poly1 = np.array([1.0, -2.0, 20.0])
        self.poly2 = np.array([1.0, 2.0, 10.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = np.ndarray(shape=(2, 7, 3))
        self.y_too_low_dim = np.zeros(shape=(3))
        self.y_wrong_size = np.zeros(shape=(2, 5, 3))
        self.axis = 1
        self.y[0, :, 0] = np.polyval(self.poly1, self.x)
        self.y[0, :, 1] = np.polyval(self.poly2, self.x)
        self.y[0, :, 2] = np.polyval(self.poly1, self.x)
        self.y[1, :, 0] = np.polyval(self.poly2, self.x)
        self.y[1, :, 1] = np.polyval(self.poly1, self.x)
        self.y[1, :, 2] = np.polyval(self.poly2, self.x)

    def test_fit_eval(self):
        """Independent1DFit: Basic function fitting and evaluation using data from a known function."""
        interp = fitting.Independent1DFit(fitting.Polynomial1DFit(2), self.axis)
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.x, self.y_too_low_dim)
        self.assertRaises(ValueError, interp.fit, self.x, self.y_wrong_size)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertEqual(interp._axis, self.axis)
        self.assertEqual(interp._interps.shape, (2, 3))
        np.testing.assert_almost_equal(interp._interps[0, 0].poly, self.poly1, decimal=10)
        np.testing.assert_almost_equal(interp._interps[0, 1].poly, self.poly2, decimal=10)
        np.testing.assert_almost_equal(interp._interps[0, 2].poly, self.poly1, decimal=10)
        np.testing.assert_almost_equal(interp._interps[1, 0].poly, self.poly2, decimal=10)
        np.testing.assert_almost_equal(interp._interps[1, 1].poly, self.poly1, decimal=10)
        np.testing.assert_almost_equal(interp._interps[1, 2].poly, self.poly2, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)

class Delaunay2DScatterFitTestCases(unittest.TestCase):
    """Check the Delaunay2DScatterFit class."""

    def setUp(self):
        # Square diamond shape
        self.x = np.array([[-1, 0, 0, 0, 1], [0, -1, 0, 1, 0]])
        self.y = np.array([1, 1, 1, 1, 1])
        self.testx = np.array([[-0.5, 0, 0.5, 0], [0, -0.5, 0.5, 0]])
        self.testy = np.array([1, 1, 1, 1])
        self.default_val = -100
        self.outsidex = np.array([[10], [10]])
        self.outsidey = np.array([self.default_val])

    def test_fit_eval_nn(self):
        """Delaunay2DScatterFit: Basic function fitting and evaluation using data from a known function."""
        interp = fitting.Delaunay2DScatterFit(default_val=self.default_val)
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=10)
        np.testing.assert_almost_equal(outsidey, self.outsidey, decimal=10)

class Delaunay2DGridFitTestCases(unittest.TestCase):
    """Check the Delaunay2DGridFit class."""

    def setUp(self):
        # Training data is uniformly sampled parabola (make sure x and y ranges coincide)
        poly = np.array([1.0, 2.0, 1.0])
        self.x = [np.linspace(-3, 3, 30), np.linspace(-3, 3, 30)]
        xx1, xx0 = np.meshgrid(self.x[1], self.x[0])
        self.y = poly[0]*xx0*xx0 + poly[1]*xx0*xx1 + poly[2]*xx1*xx1
        # Test data is uniform samples of same parabola, but ensure that samples do not fall outside training set
        self.testx = [np.linspace(-1, 1, 8), np.linspace(-1, 1, 12)]
        testx1, testx0 = np.meshgrid(self.testx[1], self.testx[0])
        self.testy = poly[0]*testx0**2 + poly[1]*testx0*testx1 + poly[2]*testx1**2
        self.default_val = -100.0
        # For some reason doesn't work for a single point - requires at least a 2x2 grid
        self.outsidex = [np.array([100, 200]), np.array([100, 200])]
        self.outsidey = np.tile(self.default_val, (len(self.outsidex[0]), len(self.outsidex[1])))

    def test_fit_eval_nn(self):
        """Delaunay2DGridFit: Basic function fitting and evaluation using data from a known function, using 'nn' gridder."""
        interp = fitting.Delaunay2DGridFit('nn', default_val=self.default_val)
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        np.testing.assert_almost_equal(y[5:-5, 5:-5], self.y[5:-5, 5:-5], decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=1)
        np.testing.assert_almost_equal(outsidey, self.outsidey, decimal=10)

    def test_fit_eval_linear(self):
        """Delaunay2DGridFit: Basic function fitting and evaluation using data from a known function, using 'linear' gridder."""
        interp = fitting.Delaunay2DGridFit('linear', default_val=self.default_val)
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        np.testing.assert_almost_equal(y[5:-5, 5:-5], self.y[5:-5, 5:-5], decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=1)
        np.testing.assert_almost_equal(outsidey, self.outsidey, decimal=10)

class NonLinearLeastSquaresFitTestCases(unittest.TestCase):
    """Check the NonLinearLeastSquaresFit class."""

    def setUp(self):
        # Quadratic function centred at p
        func = lambda p, x: ((x - p) ** 2).sum()
        self.vFunc = fitting.vectorize_fit_func(func)
        self.true_params = np.array([1, -4])
        self.init_params = np.array([0, 0])
        self.x = 4.0 * np.random.randn(2, 20)
        self.y = self.vFunc(self.true_params, self.x)
        # 2-D log Gaussian function
        def lngauss_diagcov(p, x):
            xminmu = x - p[:2, np.newaxis]
            return p[4] - 0.5 * np.dot(p[2:4], xminmu * xminmu)
        self.func2 = lngauss_diagcov
        self.true_params2 = np.array([3, -2, 10, 10, 4])
        self.init_params2 = np.array([0, 0, 1, 1, 0])
        self.x2 = np.random.randn(2, 80)
        self.y2 = lngauss_diagcov(self.true_params2, self.x2)
        # Linear function
        self.func3 = lambda p, x: np.dot(p, x)
        self.jac3 = lambda p, x: x
        self.true_params3 = np.array([-0.1, 0.2, -0.3, 0.0, 0.5])
        self.init_params3 = np.zeros(5)
        t = np.arange(0, 10., 10. / 100)
        self.x3 = np.vander(t, 5).T
        self.y3 = self.func3(self.true_params3, self.x3)

    def test_fit_eval_func1(self):
        """NonLinearLeastSquaresFit: Basic function fitting and evaluation using data from a known function."""
        interp = fitting.NonLinearLeastSquaresFit(self.vFunc, self.init_params)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.params, self.true_params, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=5)

    def test_fit_eval_gauss(self):
        """NonLinearLeastSquaresFit: Check fit on a 2-D log Gaussian function."""
        interp2 = fitting.NonLinearLeastSquaresFit(self.func2, self.init_params2)
        interp2.fit(self.x2, self.y2)
        y2 = interp2(self.x2)
        np.testing.assert_almost_equal(interp2.params, self.true_params2, decimal=10)
        np.testing.assert_almost_equal(y2, self.y2, decimal=10)

    def test_fit_eval_linear(self):
        """NonLinearLeastSquaresFit: Compare to LinearLeastSquaresFit on a linear problem (and check use of Jacobian)."""
        lin = fitting.LinearLeastSquaresFit()
        lin.fit(self.x3, self.y3, std_y=2.0)
        nonlin = fitting.NonLinearLeastSquaresFit(self.func3, self.init_params3, self.jac3)
        nonlin.fit(self.x3, self.y3, std_y=2.0)
        # A correct Jacobian helps a lot...
        np.testing.assert_almost_equal(nonlin.params, self.true_params3, decimal=11)
        np.testing.assert_almost_equal(nonlin.cov_params, lin.cov_params, decimal=11)
        nonlin_nojac = fitting.NonLinearLeastSquaresFit(self.func3, self.init_params3)
        nonlin_nojac.fit(self.x3, self.y3, std_y=0.1)
        np.testing.assert_almost_equal(nonlin_nojac.params, self.true_params3, decimal=6)
        # Covariance matrix is way smaller than linear one...

class GaussianFit2VarTestCases(unittest.TestCase):
    """Check the GaussianFit class with different variances on each dimension."""

    def setUp(self):
        # For a more challenging fit, move the true mean away from the origin, i.e. away from the region
        # being randomly sampled in self.x. Fitting a Gaussian to a segment that does not contain a clear peak
        # works fine if the fit is done to the log of the data, but fails in the linear domain.
        self.true_mean, self.true_std, self.true_height = [0., 0.], [3., 5.], 4.
        true_gauss = fitting.GaussianFit(self.true_mean, self.true_std, self.true_height)
        self.x = 7. * np.random.randn(2, 300)
        self.y = true_gauss(self.x)
        self.init_mean, self.init_std, self.init_height = [3., -2.], [1., 1.], 1.

    def test_fit_eval_diagcov(self):
        """GaussianFit (independent stdevs): Basic function fitting and evaluation using data from a known function."""
        interp = fitting.GaussianFit(self.init_mean, self.init_std, self.init_height)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.mean, self.true_mean, decimal=7)
        np.testing.assert_almost_equal(interp.std, self.true_std, decimal=7)
        np.testing.assert_almost_equal(interp.height, self.true_height, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=7)

    def test_cov_params(self):
        """GaussianFit (independent stdevs): Obtain sample statistics of parameters and compare to calculated covariance matrix."""
        interp = fitting.GaussianFit(self.init_mean, self.init_std, self.init_height)
        true_params = np.r_[self.true_mean, self.true_height, self.true_std]
        std_y = 0.1
        M = 200
        param_set = np.zeros((len(true_params), M))
        for n in range(M):
            interp.fit(self.x, self.y + std_y * np.random.randn(len(self.y)), std_y)
            param_set[:, n] = np.r_[interp.mean, interp.height, interp.std]
        mean_params = param_set.mean(axis=1)
        norm_params = param_set - mean_params[:, np.newaxis]
        cov_params = np.dot(norm_params, norm_params.T) / M
        estm_std_params = np.sqrt(np.diag(cov_params))
        std_params = np.r_[interp.std_mean, interp.std_height, interp.std_std]
        self.assertTrue((np.abs(mean_params - true_params) / std_params < 0.2).all(),
                        "Sample mean parameter vector differs too much from true value")
        # Only check diagonal of cov matrix - the rest is probably affected by linearisation
        self.assertTrue((np.abs(estm_std_params - std_params) / std_params < 0.2).all(),
                        "Sample parameter standard deviation differs too much from expected one")

class GaussianFit1VarTestCases(unittest.TestCase):
    """Check the GaussianFit class with a single variance on all dimensions."""

    def setUp(self):
        self.true_mean, self.true_std, self.true_height = [0, 0], np.sqrt(10), 4
        true_gauss = fitting.GaussianFit(self.true_mean, self.true_std, self.true_height)
        self.x = 7 * np.random.randn(2, 80)
        self.y = true_gauss(self.x)
        self.init_mean, self.init_std, self.init_height = [3, -2], 1, 1

    def test_fit_eval_diagcov(self):
        """GaussianFit (shared stdev): Basic function fitting and evaluation using data from a known function."""
        interp = fitting.GaussianFit(self.init_mean, self.init_std, self.init_height)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.mean, self.true_mean, decimal=7)
        np.testing.assert_almost_equal(interp.std, self.true_std, decimal=7)
        np.testing.assert_almost_equal(interp.height, self.true_height, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=7)

class Spline1DFitTestCases(unittest.TestCase):
    """Check the Spline1DFit class."""

    def setUp(self):
        # Training data is randomly sampled parabola
        self.poly = np.array([1.0, -2.0, 1.0])
        self.x = np.random.randn(40)
        self.y = np.polyval(self.poly, self.x)
        # Test data is random samples of same parabola, but ensure that samples do not fall outside training set
        self.testx = 0.2*np.random.randn(40)
        self.testy = np.polyval(self.poly, self.testx)

    def test_fit_eval(self):
        """Spline1DFit: Basic function fitting and evaluation using data from a known function."""
        interp = fitting.Spline1DFit(3)
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=10)

class Spline2DScatterFitTestCases(unittest.TestCase):
    """Check the Spline2DScatterFit class."""

    def setUp(self):
        # Training data is randomly sampled parabola
        poly = np.array([1.0, 2.0, 1.0])
        self.x = np.random.randn(2, 100)
        self.y = poly[0]*self.x[0]**2 + poly[1]*self.x[0]*self.x[1] + poly[2]*self.x[1]**2
        # Test data is random samples of same parabola, but ensure that samples do not fall outside training set
        self.testx = 0.2*np.random.randn(2, 100)
        self.testy = poly[0]*self.testx[0]**2 + poly[1]*self.testx[0]*self.testx[1] + poly[2]*self.testx[1]**2

    def test_fit_eval(self):
        """Spline2DScatterFit: Basic function fitting and evaluation using data from a known function."""
        interp = fitting.Spline2DScatterFit((3, 3))
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=10)

class Spline2DGridFitTestCases(unittest.TestCase):
    """Check the Spline2DGridFit class."""

    def setUp(self):
        self.fit_dims = (10, 20)
        self.eval_dims = (8, 12)
        # Training data is randomly sampled parabola (also does not have to be in ascending order)
        poly = np.array([1.0, 2.0, 1.0])
        self.x = [np.random.randn(self.fit_dims[0]), np.random.randn(self.fit_dims[1])]
        xx1, xx0 = np.meshgrid(self.x[1], self.x[0])
        self.y = poly[0]*xx0*xx0 + poly[1]*xx0*xx1 + poly[2]*xx1*xx1
        # Test data is random samples of same parabola, but ensure that samples do not fall outside training set
        self.testx = [0.1*np.random.randn(self.eval_dims[0]), 0.1*np.random.randn(self.eval_dims[1])]
        testx1, testx0 = np.meshgrid(self.testx[1], self.testx[0])
        self.testy = poly[0]*testx0**2 + poly[1]*testx0*testx1 + poly[2]*testx1**2

    def test_fit_eval(self):
        """Spline2DGridFit: Basic function fitting and evaluation using data from a known function."""
        interp = fitting.Spline2DGridFit((3, 3))
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        np.testing.assert_almost_equal(y, self.y, decimal=9)
        np.testing.assert_almost_equal(testy, self.testy, decimal=8)

    def test_uncertainty_propagation(self):
        """Spline2DGridFit: Test uncertainty propagation."""
        # Calculate output data uncertainty on test data
        interp = fitting.Spline2DGridFit((3, 3))
        self.assertRaises(ValueError, interp.fit, self.x, self.y, self.x[0])
        interp.fit(self.x, self.y, std_y=0.1)
        testy, std_testy = interp(self.testx, full_output=True)
        # Estimate data uncertainty using Monte Carlo
        y_ensemble = []
        for m in xrange(3000):
            interp = fitting.Spline2DGridFit((3, 3))
            interp.fit(self.x, self.y + 0.1*np.random.randn(*self.y.shape))
            y_ensemble.append(interp(self.testx))
        std_y_mc = np.dstack(y_ensemble).std(axis=2)
        # This is only accurate to a few percent, because of the relatively small number of Monte Carlo samples
        np.testing.assert_almost_equal(std_testy, std_y_mc, decimal=1)

class RbfScatterFitTestCases(unittest.TestCase):
    """Check the RbfScatterFit class (only if Rbf is installed in SciPy)."""

    def setUp(self):
        # Square diamond shape
        self.x = np.array([[-1, 0, 0, 0, 1], [0, -1, 0, 1, 0]])
        self.y = np.array([1, 1, 1, 1, 1])
        self.testx = np.array([[-0.5, 0, 0.5, 0], [0, -0.5, 0.5, 0]])
        self.testy = np.array([1, 1, 1, 1])

    def test_fit_eval(self):
        """RbfScatterFit: Basic function fitting and evaluation using data from a known function."""
        try:
            interp = fitting.RbfScatterFit()
        except ImportError:
            return
        self.assertRaises(fitting.NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=2)

class RandomisedFitTestCases(unittest.TestCase):
    """Check the randomisation of existing fits via RandomisedFit."""

    def setUp(self):
        self.poly = np.array([1.0, -2.0, 1.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = np.polyval(self.poly, self.x)
        self.num_runs = 100
        self.yNoisy = self.y + 0.001 * np.random.randn(self.num_runs, len(self.y))

    def test_randomised_polyfit(self):
        """RandomisedFit: Randomise the fit of a polynomial fitter."""
        interp = fitting.Polynomial1DFit(2)
        # Perfect fit (no noise)
        interp.fit(self.x, self.y)
        random_interp = fitting.randomise(interp, self.x, self.y, 'unknown')
        y = random_interp(self.x)
        np.testing.assert_almost_equal(random_interp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        random_interp = fitting.randomise(interp, self.x, self.y, 'shuffle')
        y = random_interp(self.x)
        np.testing.assert_almost_equal(random_interp.poly, self.poly, decimal=10)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        # Fit polynomials to a set of noisy samples
        noisy_poly = []
        for noisy_y in self.yNoisy:
            interp.fit(self.x, noisy_y)
            noisy_poly.append(interp.poly)
        noisy_poly = np.array(noisy_poly)
        # Randomise polynomial fit to first noisy sample in various ways
        # pylint: disable-msg=W0612
        shuffle_poly = np.array([fitting.randomise(interp, self.x, self.yNoisy[0], 'shuffle').poly
                                 for n in range(self.num_runs)])
        np.testing.assert_almost_equal(shuffle_poly.mean(axis=0), noisy_poly[0], decimal=2)
        np.testing.assert_almost_equal(shuffle_poly.std(axis=0), noisy_poly.std(axis=0), decimal=2)
        normal_poly = np.array([fitting.randomise(interp, self.x, self.yNoisy[0], 'normal').poly
                                for n in range(self.num_runs)])
        np.testing.assert_almost_equal(normal_poly.mean(axis=0), noisy_poly[0], decimal=2)
        np.testing.assert_almost_equal(normal_poly.std(axis=0), noisy_poly.std(axis=0), decimal=2)
        boot_poly = np.array([fitting.randomise(interp, self.x, self.yNoisy[0], 'bootstrap').poly
                              for n in range(self.num_runs)])
        np.testing.assert_almost_equal(boot_poly.mean(axis=0), noisy_poly[0], decimal=2)
        np.testing.assert_almost_equal(boot_poly.std(axis=0), noisy_poly.std(axis=0), decimal=2)
