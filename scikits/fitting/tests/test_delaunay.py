#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for Delaunay fitters.

:author: Ludwig Schwardt
:license: Modified BSD

"""

import numpy as np
from numpy.testing import TestCase, assert_almost_equal, run_module_suite

from scikits.fitting import Delaunay2DScatterFit, NotFittedError


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


if __name__ == "__main__":
    run_module_suite()
