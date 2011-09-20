#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for RBF fitter.

:author: Ludwig Schwardt
:license: Modified BSD

"""

import numpy as np
from numpy.testing import *

try:
    from scikits.fitting import RbfScatterFit, NotFittedError
    rbf_found = True
except ImportError:
    rbf_found = False

class TestRbfScatterFit(TestCase):
    """Check the RbfScatterFit class (only if Rbf is installed in SciPy)."""

    def setUp(self):
        # Square diamond shape
        self.x = np.array([[-1, 0, 0, 0, 1], [0, -1, 0, 1, 0]])
        self.y = np.array([1, 1, 1, 1, 1])
        self.testx = np.array([[-0.5, 0, 0.5, 0], [0, -0.5, 0.5, 0]])
        self.testy = np.array([1, 1, 1, 1])

    def test_fit_eval(self):
        """RbfScatterFit: Basic function fitting and evaluation using data from a known function."""
        if rbf_found:
            interp = RbfScatterFit()
        else:
            return
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        assert_almost_equal(y, self.y, decimal=10)
        assert_almost_equal(testy, self.testy, decimal=2)

if __name__ == "__main__":
    run_module_suite()
