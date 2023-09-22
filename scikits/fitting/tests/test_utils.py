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

"""Tests for utility functions.

:author: Ludwig Schwardt
:license: Modified BSD

"""

from builtins import range
import numpy as np
from numpy.testing import TestCase, assert_array_equal, assert_almost_equal

from scikits.fitting import squash, unsquash, randomise, Polynomial1DFit


class TestUtils(TestCase):
    """Exercise utility functions."""

    def setUp(self):
        self.x = np.random.rand(2, 4, 10)

    def test_squash(self):
        """Utils: Test squash and unsquash."""
        y1 = squash(self.x, [], True)
        y1a = squash(self.x, None, True)
        y2 = squash(self.x, (1), False)
        y3 = squash(self.x, (0, 2), True)
        y4 = squash(self.x, (0, 2), False)
        y5 = squash(self.x, (0, 1, 2), True)
        self.assertEqual(y1.shape, (2, 4, 10))
        self.assertEqual(y1a.shape, (2, 4, 10))
        self.assertEqual(y2.shape, (2, 10, 4))
        self.assertEqual(y3.shape, (20, 4))
        self.assertEqual(y4.shape, (4, 20))
        self.assertEqual(y5.shape, (80,))
        assert_array_equal(unsquash(y1, [], (2, 4, 10), True), self.x)
        assert_array_equal(unsquash(y1a, None, (2, 4, 10), True), self.x)
        assert_array_equal(unsquash(y2, (1), (2, 4, 10), False), self.x)
        assert_array_equal(unsquash(y3, (0, 2), (2, 4, 10), True), self.x)
        assert_array_equal(unsquash(y4, (0, 2), (2, 4, 10), False), self.x)
        assert_array_equal(unsquash(y5, (0, 1, 2), (2, 4, 10), True), self.x)


class TestRandomise(TestCase):
    """Check the randomisation of existing fits via randomise function."""

    def setUp(self):
        self.poly = np.array([1.0, -2.0, 1.0])
        self.x = np.arange(-3.0, 4.0, 1.0)
        self.y = np.polyval(self.poly, self.x)
        self.num_runs = 100
        self.y_noisy = self.y + 0.001 * np.random.randn(self.num_runs,
                                                        len(self.y))

    def test_randomised_polyfit(self):
        """Randomise: Randomise the fit of a polynomial fitter."""
        interp = Polynomial1DFit(2)
        # Perfect fit (no noise)
        interp.fit(self.x, self.y)
        random_interp = randomise(interp, self.x, self.y, 'unknown')
        y = random_interp(self.x)
        assert_almost_equal(random_interp.poly, self.poly, decimal=10)
        assert_almost_equal(y, self.y, decimal=10)
        random_interp = randomise(interp, self.x, self.y, 'shuffle')
        y = random_interp(self.x)
        assert_almost_equal(random_interp.poly, self.poly, decimal=10)
        assert_almost_equal(y, self.y, decimal=10)
        # Fit polynomials to a set of noisy samples
        noisy_poly = []
        for yn in self.y_noisy:
            interp.fit(self.x, yn)
            noisy_poly.append(interp.poly)
        noisy_poly = np.array(noisy_poly)
        # Randomise polynomial fit to first noisy sample in various ways
        # pylint: disable-msg=W0612
        shuffle_poly = np.array([randomise(interp, self.x, self.y_noisy[0],
                                           'shuffle').poly
                                 for n in range(self.num_runs)])
        assert_almost_equal(shuffle_poly.mean(axis=0), noisy_poly[0],
                            decimal=2)
        assert_almost_equal(shuffle_poly.std(axis=0), noisy_poly.std(axis=0),
                            decimal=2)
        normal_poly = np.array([randomise(interp, self.x, self.y_noisy[0],
                                          'normal').poly
                                for n in range(self.num_runs)])
        assert_almost_equal(normal_poly.mean(axis=0), noisy_poly[0], decimal=2)
        assert_almost_equal(normal_poly.std(axis=0), noisy_poly.std(axis=0),
                            decimal=2)
        boot_poly = np.array([randomise(interp, self.x, self.y_noisy[0],
                                        'bootstrap').poly
                              for n in range(self.num_runs)])
        assert_almost_equal(boot_poly.mean(axis=0), noisy_poly[0], decimal=2)
        assert_almost_equal(boot_poly.std(axis=0), noisy_poly.std(axis=0),
                            decimal=2)
