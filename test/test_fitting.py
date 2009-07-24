"""Unit tests for the fitting module."""
# pylint: disable-msg=C0103,W0212

import unittest
import numpy as np
from scape import fitting

class Polynomial1DFitTestCases(unittest.TestCase):
    """Fit a 1-D polynomial to data from a known polynomial, and compare."""

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

    # pylint: disable-msg=R0201
    def test_reduce_degree(self):
        interp = fitting.Polynomial1DFit(2)
        interp.fit([1.0], [1.0])
        np.testing.assert_almost_equal(interp.poly, [1.0], decimal=10)

class ReciprocalFitTestCases(unittest.TestCase):
    """Check the ReciprocalFit class."""

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
        interp = fitting.Independent1DFit(fitting.Polynomial1DFit(2), self.axis)
        self.assertRaises(AttributeError, interp, self.x)
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
        interp = fitting.Delaunay2DScatterFit(default_val=self.default_val)
        self.assertRaises(AttributeError, interp, self.x)
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
        interp = fitting.Delaunay2DGridFit('nn', default_val=self.default_val)
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        np.testing.assert_almost_equal(y[5:-5, 5:-5], self.y[5:-5, 5:-5], decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=1)
        np.testing.assert_almost_equal(outsidey, self.outsidey, decimal=10)

    def test_fit_eval_linear(self):
        interp = fitting.Delaunay2DGridFit('linear', default_val=self.default_val)
        self.assertRaises(AttributeError, interp, self.x)
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
        func = lambda p, x: ((x - p)**2).sum()
        self.vFunc = fitting.vectorize_fit_func(func)
        self.true_params = np.array([1, -4])
        self.init_params = np.array([0, 0])
        self.x = 4.0*np.random.randn(20, 2)
        self.y = self.vFunc(self.true_params, self.x)
        # 2-D log Gaussian function
        def lngauss_diagcov(p, x):
            xminmu = x - p[np.newaxis, 0:2]
            return p[4] - 0.5 * np.dot(xminmu * xminmu, p[2:4])
        self.func2 = lngauss_diagcov
        self.true_params2 = np.array([3, -2, 10, 10, 4])
        self.init_params2 = np.array([0, 0, 1, 1, 0])
        self.x2 = np.random.randn(80, 2)
        self.y2 = lngauss_diagcov(self.true_params2, self.x2)

    def test_fit_eval_func1(self):
        self.assertRaises(KeyError, fitting.NonLinearLeastSquaresFit, \
                          self.vFunc, self.init_params, method='bollie')
        interp = fitting.NonLinearLeastSquaresFit(self.vFunc, self.init_params, method='fmin_bfgs', disp=0)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.params, self.true_params, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=5)

    def test_fit_eval_gauss(self):
        interp2 = fitting.NonLinearLeastSquaresFit(self.func2, self.init_params2, method='leastsq')
        interp2.fit(self.x2, self.y2)
        y2 = interp2(self.x2)
        np.testing.assert_almost_equal(interp2.params, self.true_params2, decimal=10)
        np.testing.assert_almost_equal(y2, self.y2, decimal=10)

class GaussianFit2VarTestCases(unittest.TestCase):
    """Check the GaussianFit class with different variances on each dimension."""

    def setUp(self):
        # For a more challenging fit, move the true mean away from the origin, i.e. away from the region
        # being randomly sampled in self.x. Fitting a Gaussian to a segment that does not contain a clear peak
        # works fine if the fit is done to the log of the data, but fails in the linear domain.
        self.true_mean, self.true_var, self.true_height = [0, 0], [10, 20], 4
        true_gauss = fitting.GaussianFit(self.true_mean, self.true_var, self.true_height)
        self.x = 7*np.random.randn(80, 2)
        self.y = true_gauss(self.x)
        self.init_mean, self.init_var, self.init_height = [3, -2], [1, 1], 1

    def test_fit_eval_diagcov(self):
        interp = fitting.GaussianFit(self.init_mean, self.init_var, self.init_height)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.mean, self.true_mean, decimal=7)
        np.testing.assert_almost_equal(interp.var, self.true_var, decimal=6)
        np.testing.assert_almost_equal(interp.height, self.true_height, decimal=7)
        np.testing.assert_almost_equal(y, self.y, decimal=7)

class GaussianFit1VarTestCases(unittest.TestCase):
    """Check the GaussianFit class with a single variance on all dimensions."""

    def setUp(self):
        self.true_mean, self.true_var, self.true_height = [0, 0], 10, 4
        true_gauss = fitting.GaussianFit(self.true_mean, self.true_var, self.true_height)
        self.x = 7*np.random.randn(80, 2)
        self.y = true_gauss(self.x)
        self.init_mean, self.init_var, self.init_height = [3, -2], 1, 1

    def test_fit_eval_diagcov(self):
        interp = fitting.GaussianFit(self.init_mean, self.init_var, self.init_height)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        np.testing.assert_almost_equal(interp.mean, self.true_mean, decimal=7)
        np.testing.assert_almost_equal(interp.var, self.true_var, decimal=6)
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
        interp = fitting.Spline1DFit(3)
        self.assertRaises(AttributeError, interp, self.x)
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
        interp = fitting.Spline2DScatterFit((3, 3))
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=10)

class Spline2DGridFitTestCases(unittest.TestCase):
    """Check the Spline2DGridFit class."""

    def setUp(self):
        # Training data is randomly sampled parabola (but remember to keep data in ascending order)
        poly = np.array([1.0, 2.0, 1.0])
        self.x = [sorted(np.random.randn(10)), sorted(np.random.randn(20))]
        xx1, xx0 = np.meshgrid(self.x[1], self.x[0])
        self.y = poly[0]*xx0*xx0 + poly[1]*xx0*xx1 + poly[2]*xx1*xx1
        # Test data is random samples of same parabola, but ensure that samples do not fall outside training set
        self.testx = [sorted(0.2*np.random.randn(8)), sorted(0.2*np.random.randn(12))]
        testx1, testx0 = np.meshgrid(self.testx[1], self.testx[0])
        self.testy = poly[0]*testx0**2 + poly[1]*testx0*testx1 + poly[2]*testx1**2

    def test_fit_eval(self):
        interp = fitting.Spline2DGridFit((3, 3))
        self.assertRaises(AttributeError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        np.testing.assert_almost_equal(y, self.y, decimal=10)
        np.testing.assert_almost_equal(testy, self.testy, decimal=10)

class RbfScatterFitTestCases(unittest.TestCase):
    """Check the RbfScatterFit class (only if Rbf is installed in SciPy)."""

    def setUp(self):
        # Square diamond shape
        self.x = np.array([[-1, 0, 0, 0, 1], [0, -1, 0, 1, 0]])
        self.y = np.array([1, 1, 1, 1, 1])
        self.testx = np.array([[-0.5, 0, 0.5, 0], [0, -0.5, 0.5, 0]])
        self.testy = np.array([1, 1, 1, 1])

    def test_fit_eval(self):
        try:
            interp = fitting.RbfScatterFit()
        except ImportError:
            return
        self.assertRaises(AttributeError, interp, self.x)
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
        self.yNoisy = self.y + 0.001*np.random.randn(self.num_runs, len(self.y))

    def test_randomised_polyfit(self):
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
