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

"""Tests for spline fitters.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

from builtins import range
import numpy as np
from numpy.testing import TestCase, assert_almost_equal

from scikits.fitting import (Spline1DFit, Spline2DScatterFit, Spline2DGridFit,
                             NotFittedError)


class TestSpline1DFit(TestCase):
    """Check the Spline1DFit class."""

    def setUp(self):
        # Training data is randomly sampled parabola
        self.poly = np.array([1.0, -2.0, 1.0])
        self.x = np.random.randn(40)
        self.y = np.polyval(self.poly, self.x)
        # Test data is random samples of same parabola, but ensure that
        # samples do not fall outside training set
        self.testx = 0.2 * np.random.randn(40)
        self.testy = np.polyval(self.poly, self.testx)

    def test_fit_eval(self):
        """Spline1DFit: Basic function fitting and evaluation."""
        interp = Spline1DFit(degree=3, min_size=0.5)
        self.assertRaises(NotFittedError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        assert_almost_equal(y, self.y, decimal=10)
        assert_almost_equal(testy, self.testy, decimal=10)


class TestSpline2DScatterFit(TestCase):
    """Check the Spline2DScatterFit class."""

    def setUp(self):
        # Training data is randomly sampled parabola
        poly = np.array([1.0, 2.0, 1.0])
        self.x = np.random.randn(2, 100)
        self.y = (poly[0] * self.x[0] ** 2 +
                  poly[1] * self.x[0] * self.x[1] +
                  poly[2] * self.x[1] ** 2)
        # Test data is random samples of same parabola, but ensure that
        # samples do not fall outside training set
        self.testx = 0.2 * np.random.randn(2, 100)
        self.testy = (poly[0] * self.testx[0] ** 2 +
                      poly[1] * self.testx[0] * self.testx[1] +
                      poly[2] * self.testx[1] ** 2)

    def test_fit_eval(self):
        """Spline2DScatterFit: Basic function fitting and evaluation."""
        interp = Spline2DScatterFit((3, 3))
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        assert_almost_equal(y, self.y, decimal=10)
        assert_almost_equal(testy, self.testy, decimal=10)


class TestSpline2DGridFit(TestCase):
    """Check the Spline2DGridFit class."""

    def setUp(self):
        self.fit_dims = (10, 20)
        self.eval_dims = (8, 12)
        # Training data is uniformly sampled parabola
        # (does not have to be in ascending order)
        poly = np.array([1.0, 2.0, 1.0])
        x0 = np.linspace(0., 1., self.fit_dims[0])
        x1 = np.linspace(0., 1., self.fit_dims[1])
        np.random.shuffle(x0)
        np.random.shuffle(x1)
        self.x = [x0, x1]
        xx1, xx0 = np.meshgrid(self.x[1], self.x[0])
        self.y = (poly[0] * xx0 * xx0 +
                  poly[1] * xx0 * xx1 +
                  poly[2] * xx1 * xx1)
        # Test data is random samples of same parabola, but ensure that
        # samples do not fall outside training set
        self.testx = [0.25 + 0.5 * np.random.rand(self.eval_dims[0]),
                      0.25 + 0.5 * np.random.rand(self.eval_dims[1])]
        testx1, testx0 = np.meshgrid(self.testx[1], self.testx[0])
        self.testy = (poly[0] * testx0 ** 2 +
                      poly[1] * testx0 * testx1 +
                      poly[2] * testx1 ** 2)

    def test_fit_eval(self):
        """Spline2DGridFit: Basic function fitting and evaluation."""
        interp = Spline2DGridFit((3, 3))
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        assert_almost_equal(y, self.y, decimal=9)
        assert_almost_equal(testy, self.testy, decimal=8)

    def test_uncertainty_propagation(self):
        """Spline2DGridFit: Test uncertainty propagation."""
        # Calculate output data uncertainty on test data
        interp = Spline2DGridFit((3, 3))
        self.assertRaises(ValueError, interp.fit, self.x, self.y, self.x[0])
        interp.fit(self.x, self.y, std_y=0.1)
        testy, std_testy = interp(self.testx, full_output=True)
        # Estimate data uncertainty using Monte Carlo
        y_ensemble = []
        for m in range(3000):
            interp = Spline2DGridFit((3, 3))
            interp.fit(self.x, self.y + 0.1 * np.random.randn(*self.y.shape))
            y_ensemble.append(interp(self.testx))
        std_y_mc = np.dstack(y_ensemble).std(axis=2)
        # This is only accurate to a few percent, because of
        # the relatively small number of Monte Carlo samples
        rel_std_diff = np.abs(std_y_mc - std_testy) / np.abs(std_testy)
        rel_std_diff_p90 = sorted(rel_std_diff.ravel())[
            int(0.90 * rel_std_diff.size)]
        self.assertTrue(rel_std_diff_p90 < 0.1)
