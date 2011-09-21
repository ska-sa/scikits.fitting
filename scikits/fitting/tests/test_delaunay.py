#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Delaunay fitters.

:author: Ludwig Schwardt
:license: Modified BSD

"""

import numpy as np
from numpy.testing import *

from scikits.fitting import Delaunay2DScatterFit, Delaunay2DGridFit, NotFittedError

class TestDelaunay2DScatterFit(TestCase):
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
        # At least exercise the jitter code path
        interp = Delaunay2DScatterFit(default_val=self.default_val, jitter=True)
        interp.fit(self.x, self.y)
        interp = Delaunay2DScatterFit(default_val=self.default_val, jitter=False)
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        assert_almost_equal(y, self.y, decimal=10)
        assert_almost_equal(testy, self.testy, decimal=10)
        assert_almost_equal(outsidey, self.outsidey, decimal=10)

class TestDelaunay2DGridFit(TestCase):
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
        interp = Delaunay2DGridFit('nn', default_val=self.default_val)
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        assert_almost_equal(y[5:-5, 5:-5], self.y[5:-5, 5:-5], decimal=10)
        assert_almost_equal(testy, self.testy, decimal=1)
        assert_almost_equal(outsidey, self.outsidey, decimal=10)

    def test_fit_eval_linear(self):
        """Delaunay2DGridFit: Basic function fitting and evaluation using data from a known function, using 'linear' gridder."""
        interp = Delaunay2DGridFit('linear', default_val=self.default_val)
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        assert_almost_equal(y[5:-5, 5:-5], self.y[5:-5, 5:-5], decimal=10)
        assert_almost_equal(testy, self.testy, decimal=1)
        assert_almost_equal(outsidey, self.outsidey, decimal=10)

if __name__ == "__main__":
    run_module_suite()
