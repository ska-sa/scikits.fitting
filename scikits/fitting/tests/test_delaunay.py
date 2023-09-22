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

"""Tests for Delaunay fitters.

:author: Ludwig Schwardt
:license: Modified BSD

"""

import numpy as np
from numpy.testing import TestCase, assert_almost_equal

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

    def test_fit_eval(self):
        """Delaunay2DScatterFit: Basic function fitting and evaluation."""
        # At least exercise the jitter code path
        interp = Delaunay2DScatterFit(default_val=self.default_val,
                                      jitter=True)
        interp.fit(self.x, self.y)
        # Test cubic interpolation
        interp = Delaunay2DScatterFit(default_val=self.default_val,
                                      jitter=False)
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y)
        interp.fit(self.x, self.y)
        self.assertRaises(ValueError, interp, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        assert_almost_equal(y, self.y, decimal=10)
        assert_almost_equal(testy, self.testy, decimal=10)
        assert_almost_equal(outsidey, self.outsidey, decimal=10)
        # Test linear interpolation
        interp = Delaunay2DScatterFit(default_val=self.default_val,
                                      interp_type='linear')
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        assert_almost_equal(y, self.y, decimal=10)
        assert_almost_equal(testy, self.testy, decimal=10)
        assert_almost_equal(outsidey, self.outsidey, decimal=10)
        # Test nearest-neighbour interpolation
        interp = Delaunay2DScatterFit(default_val=self.default_val,
                                      interp_type='nearest')
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        outsidey = interp(self.outsidex)
        assert_almost_equal(y, self.y, decimal=10)
        assert_almost_equal(testy, self.testy, decimal=10)
        # Nearest-neighbour interpolation has no outside value
        # assert_almost_equal(outsidey, self.outsidey, decimal=10)
