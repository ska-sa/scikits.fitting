#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for non-linear least-squares fitter.

:author: Ludwig Schwardt
:license: Modified BSD

"""

import numpy as np
from numpy.testing import *

from scikits.fitting import NonLinearLeastSquaresFit, LinearLeastSquaresFit, vectorize_fit_func

class TestNonLinearLeastSquaresFit(TestCase):
    """Check the NonLinearLeastSquaresFit class."""

    def setUp(self):
        # Quadratic function centred at p
        func = lambda p, x: ((x - p) ** 2).sum()
        self.vFunc = vectorize_fit_func(func)
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
        self.enabled_params_int = [0, 1, 2, 4]
        self.enabled_params_bool = [True, True, True, False, True]
        t = np.arange(0, 10., 10. / 100)
        self.x3 = np.vander(t, 5).T
        self.y3 = self.func3(self.true_params3, self.x3)

    def test_fit_eval_func1(self):
        """NonLinearLeastSquaresFit: Basic function fitting and evaluation using data from a known function."""
        interp = NonLinearLeastSquaresFit(self.vFunc, self.init_params)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        assert_almost_equal(interp.params, self.true_params, decimal=7)
        assert_almost_equal(y, self.y, decimal=5)

    def test_fit_eval_gauss(self):
        """NonLinearLeastSquaresFit: Check fit on a 2-D log Gaussian function."""
        interp2 = NonLinearLeastSquaresFit(self.func2, self.init_params2)
        interp2.fit(self.x2, self.y2)
        y2 = interp2(self.x2)
        assert_almost_equal(interp2.params, self.true_params2, decimal=10)
        assert_almost_equal(y2, self.y2, decimal=10)

    def test_fit_eval_linear(self):
        """NonLinearLeastSquaresFit: Compare to LinearLeastSquaresFit on a linear problem (and check use of Jacobian)."""
        lin = LinearLeastSquaresFit()
        lin.fit(self.x3, self.y3, std_y=2.0)
        nonlin = NonLinearLeastSquaresFit(self.func3, self.init_params3, func_jacobian=self.jac3)
        nonlin.fit(self.x3, self.y3, std_y=2.0)
        # A correct Jacobian helps a lot...
        assert_almost_equal(nonlin.params, self.true_params3, decimal=11)
        assert_almost_equal(nonlin.cov_params, lin.cov_params, decimal=11)
        nonlin_nojac = NonLinearLeastSquaresFit(self.func3, self.init_params3)
        nonlin_nojac.fit(self.x3, self.y3, std_y=0.1)
        assert_almost_equal(nonlin_nojac.params, self.true_params3, decimal=6)
        # Covariance matrix is way smaller than linear one...
        
    def test_enabled_params(self):
        """NonLinearLeastSquaresFit: Check whether subset of parameters can be optimised."""
        lin = LinearLeastSquaresFit()
        lin.fit(self.x3[self.enabled_params_int, :], self.y3, std_y=2.0)
        lin_cov_params = np.zeros((len(self.true_params3), len(self.true_params3)))
        lin_cov_params[np.ix_(self.enabled_params_int, self.enabled_params_int)] = lin.cov_params
        nonlin = NonLinearLeastSquaresFit(self.func3, self.init_params3, self.enabled_params_int, self.jac3)
        nonlin.fit(self.x3, self.y3, std_y=2.0)
        assert_almost_equal(nonlin.params, self.true_params3, decimal=11)
        assert_almost_equal(nonlin.cov_params, lin_cov_params, decimal=11)
        nonlin = NonLinearLeastSquaresFit(self.func3, self.init_params3, self.enabled_params_bool, self.jac3)
        nonlin.fit(self.x3, self.y3, std_y=2.0)
        assert_almost_equal(nonlin.params, self.true_params3, decimal=11)
        assert_almost_equal(nonlin.cov_params, lin_cov_params, decimal=11)

if __name__ == "__main__":
    run_module_suite()
