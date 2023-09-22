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

"""Tests for RBF fitter.

:author: Ludwig Schwardt
:license: Modified BSD

"""

import numpy as np
from numpy.testing import TestCase, assert_almost_equal

from scikits.fitting import RbfScatterFit, NotFittedError


class TestRbfScatterFit(TestCase):
    """Check the RbfScatterFit class (only if Rbf is installed in SciPy)."""

    def setUp(self):
        # Square diamond shape
        self.x = np.array([[-1, 0, 0, 0, 1], [0, -1, 0, 1, 0]])
        self.y = np.array([1, 1, 1, 1, 1])
        self.testx = np.array([[-0.5, 0, 0.5, 0], [0, -0.5, 0.5, 0]])
        self.testy = np.array([1, 1, 1, 1])

    def test_fit_eval(self):
        """RbfScatterFit: Basic function fitting and evaluation."""
        interp = RbfScatterFit()
        self.assertRaises(NotFittedError, interp, self.x)
        self.assertRaises(ValueError, interp.fit, self.y, self.y[:-1])
        interp.fit(self.x, self.y)
        y = interp(self.x)
        testy = interp(self.testx)
        assert_almost_equal(y, self.y, decimal=10)
        assert_almost_equal(testy, self.testy, decimal=2)
