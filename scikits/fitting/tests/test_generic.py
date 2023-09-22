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

"""Tests for generic fitters.

:author: Ludwig Schwardt
:license: Modified BSD

"""

import numpy as np
from numpy.testing import TestCase, assert_almost_equal

from scikits.fitting import Independent1DFit, Polynomial1DFit, NotFittedError


class TestIndependent1DFit(TestCase):
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
        """Independent1DFit: Basic function fitting and evaluation."""
        interp = Independent1DFit(Polynomial1DFit(2), self.axis)
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.x, self.y_too_low_dim)
        self.assertRaises(ValueError, interp.fit, self.x, self.y_wrong_size)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        self.assertEqual(interp._axis, self.axis)
        self.assertEqual(interp._interps.shape, (2, 3))
        assert_almost_equal(interp._interps[0, 0].poly, self.poly1, decimal=10)
        assert_almost_equal(interp._interps[0, 1].poly, self.poly2, decimal=10)
        assert_almost_equal(interp._interps[0, 2].poly, self.poly1, decimal=10)
        assert_almost_equal(interp._interps[1, 0].poly, self.poly2, decimal=10)
        assert_almost_equal(interp._interps[1, 1].poly, self.poly1, decimal=10)
        assert_almost_equal(interp._interps[1, 2].poly, self.poly2, decimal=10)
        assert_almost_equal(y, self.y, decimal=10)
