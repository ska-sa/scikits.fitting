###############################################################################
# Copyright (c) 2007-2018, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

"""Tests for polynomial fitters.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

from builtins import range
import numpy as np
from numpy.testing import TestCase, assert_equal, assert_almost_equal

from scikits.fitting import (Polynomial1DFit, Polynomial2DFit,
                             PiecewisePolynomial1DFit, NotFittedError)
from scikits.fitting.poly import _stepwise_interp, _linear_interp


class TestPolynomial1DFit(TestCase):
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
        """Polynomial1DFit: Basic function fitting + evaluation (zero-mean)."""
        interp = Polynomial1DFit(2)
        self.assertRaises(NotFittedError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertAlmostEqual(interp._mean, 0.0, places=10)
        assert_almost_equal(interp.poly, self.poly, decimal=10)
        assert_almost_equal(y, self.y, decimal=10)

    def test_fit_eval2(self):
        """Polynomial1DFit: Basic fitting and evaluation (non-zero-mean)."""
        interp = Polynomial1DFit(2)
        interp.fit(self.x2, self.y2)
        y2 = interp(self.x2)
        assert_almost_equal(interp.poly, self.poly, decimal=10)
        assert_almost_equal(y2, self.y2, decimal=10)

    def test_cov_params(self):
        """Polynomial1DFit: Compare parameter stats to covariance matrix."""
        interp = Polynomial1DFit(2)
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
        self.assertTrue(
            (np.abs(mean_poly - self.poly) / std_poly < 0.5).all(),
            "Sample mean coefficient vector differs too much from true value")
        self.assertTrue(
            (np.abs(cov_poly - interp.cov_poly) /
             np.abs(interp.cov_poly) < 0.5).all(),
            "Sample coefficient covariance matrix differs too much")

    def test_vs_numpy(self):
        """Polynomial1DFit: Compare fitter to np.polyfit and np.polyval."""
        x, p = self.randx, self.randp
        y = p[0] * (x ** 3) + p[1] * (x ** 2) + p[2] * x + p[3]
        interp = Polynomial1DFit(3)
        interp.fit(x, y)
        interp_y = interp(x)
        np_poly = np.polyfit(x, y, 3)
        np_y = np.polyval(np_poly, x)
        self.assertAlmostEqual(interp._mean, self.randx.mean(), places=10)
        assert_almost_equal(interp.poly, np_poly, decimal=10)
        assert_almost_equal(interp_y, np_y, decimal=10)

    # pylint: disable-msg=R0201
    def test_reduce_degree(self):
        """Polynomial1DFit: Reduce polynomial degree if too few data points."""
        interp = Polynomial1DFit(2)
        interp.fit([1.0], [1.0])
        assert_almost_equal(interp.poly, [1.0], decimal=10)


class TestPolynomial2DFit(TestCase):
    """Fit a 2-D polynomial to data from a known polynomial, and compare."""

    def setUp(self):
        self.poly = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        self.degrees = (1, 2)
        # Zero mean case
        x1 = np.arange(-1., 1.1, 0.1)
        x2 = np.arange(-1., 1.2, 0.2)
        xx1, xx2 = np.meshgrid(x1, x2)
        self.x = X = np.vstack((xx1.ravel(), xx2.ravel()))
        A = np.c_[X[0] * X[1]**2,
                  X[0] * X[1],
                  X[0],
                  X[1]**2,
                  X[1],
                  np.ones(X.shape[1])].T
        self.y = np.dot(self.poly, A)
        # Non-zero mean (and uneven scale) case
        x1 = np.arange(0., 10.)
        x2 = np.arange(0., 5.)
        xx1, xx2 = np.meshgrid(x1, x2)
        self.x2 = X = np.vstack((xx1.ravel(), xx2.ravel()))
        A = np.c_[X[0] * X[1]**2,
                  X[0] * X[1],
                  X[0],
                  X[1]**2,
                  X[1],
                  np.ones(X.shape[1])].T
        self.y2 = np.dot(self.poly, A)

    def test_fit_eval(self):
        """Polynomial2DFit: Basic function fitting + evaluation (zero-mean)."""
        interp = Polynomial2DFit(self.degrees)
        self.assertRaises(NotFittedError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        assert_almost_equal(interp._mean, [0.0, 0.0], decimal=10)
        assert_almost_equal(interp._scale, [1.0, 1.0], decimal=10)
        assert_almost_equal(interp.poly, self.poly, decimal=10)
        assert_almost_equal(y, self.y, decimal=10)

    def test_fit_eval2(self):
        """Polynomial2DFit: Basic fitting and evaluation (non-zero-mean)."""
        interp = Polynomial2DFit(self.degrees)
        interp.fit(self.x2, self.y2)
        y2 = interp(self.x2)
        assert_almost_equal(interp.poly, self.poly, decimal=10)
        assert_almost_equal(y2, self.y2, decimal=10)

    def test_cov_params(self):
        """Polynomial2DFit: Compare parameter stats to covariance matrix."""
        interp = Polynomial2DFit(self.degrees)
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
        self.assertTrue(
            (np.abs(mean_poly - self.poly) / std_poly < 0.5).all(),
            "Sample mean coefficient vector differs too much from true value")
        self.assertTrue(
            (np.abs(cov_poly - interp.cov_poly) /
             np.abs(interp.cov_poly) < 1.0).all(),
            "Sample coefficient covariance matrix differs too much")


class TestPiecewisePolynomial1DFit(TestCase):
    """Fit a 1-D piecewise polynomial to data from a known polynomial."""

    def setUp(self):
        self.poly = np.array([1.0, 2.0, 3.0, 4.0])
        self.x = np.linspace(-3.0, 2.0, 100)
        self.y = np.polyval(self.poly, self.x)

    def test_fit_eval(self):
        """PiecewisePolynomial1DFit: Basic function fitting and evaluation."""
        interp = PiecewisePolynomial1DFit(max_degree=3)
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, [0, 0], [1, 2])
        interp.fit(self.x[::2], self.y[::2])
        y = interp(self.x)
        assert_almost_equal(y[5:-5], self.y[5:-5], decimal=10)
        # Fit a single data point
        interp.fit(self.x[0], self.y[0])
        y = interp(self.x)
        assert_equal(y, np.tile(self.y[0], self.x.shape))

    def test_stepwise_interp(self):
        """PiecewisePolynomial1DFit: Test underlying 0th-order interpolator."""
        x = np.sort(np.random.rand(100)) * 4. - 2.5
        y = np.random.randn(100)
        interp = PiecewisePolynomial1DFit(max_degree=0)
        interp.fit(x, y)
        assert_almost_equal(interp(x), y, decimal=10)
        assert_almost_equal(interp(x + 1e-15), y, decimal=10)
        assert_almost_equal(interp(x - 1e-15), y, decimal=10)
        assert_almost_equal(_stepwise_interp(x, y, x), y, decimal=10)
        assert_almost_equal(interp(self.x), _stepwise_interp(x, y, self.x),
                            decimal=10)

    def test_linear_interp(self):
        """PiecewisePolynomial1DFit: Test underlying 1st-order interpolator."""
        x = np.sort(np.random.rand(100)) * 4. - 2.5
        y = np.random.randn(100)
        interp = PiecewisePolynomial1DFit(max_degree=1)
        interp.fit(x, y)
        assert_almost_equal(interp(x), y, decimal=10)
        assert_almost_equal(_linear_interp(x, y, x), y, decimal=10)
        assert_almost_equal(interp(self.x), _linear_interp(x, y, self.x),
                            decimal=10)
