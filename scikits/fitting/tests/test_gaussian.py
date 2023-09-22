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

"""Tests for Gaussian fitter.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

from builtins import range
import numpy as np
from numpy.testing import TestCase, assert_almost_equal

from scikits.fitting import GaussianFit


class TestGaussianFitDiag(TestCase):
    """Check GaussianFit class with different variances on each dimension."""

    def setUp(self):
        # For a more challenging fit, move the true mean away from the origin,
        # i.e. away from the region being randomly sampled in self.x. Fitting
        # a Gaussian to a segment that does not contain a clear peak works fine
        # if the fit is done to the log of the data, but fails in the linear
        # domain.
        self.true_mean = [0., 0.]
        self.true_std = [3., 5.]
        self.true_height = 4.
        true_gauss = GaussianFit(self.true_mean, self.true_std,
                                 self.true_height)
        self.x = 7. * np.random.randn(2, 300)
        self.y = true_gauss(self.x)
        self.init_mean = [3., -2.]
        self.init_std = [1., 1.]
        self.init_height = 1.

    def test_fit_eval_diagcov(self):
        """GaussianFit (independent stdevs): Basic fitting and evaluation."""
        interp = GaussianFit(self.init_mean, self.init_std, self.init_height)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        assert_almost_equal(interp.mean, self.true_mean, decimal=7)
        assert_almost_equal(interp.std, self.true_std, decimal=7)
        assert_almost_equal(interp.height, self.true_height, decimal=7)
        assert_almost_equal(y, self.y, decimal=7)

    def test_cov_params(self):
        """GaussianFit (independent stdevs): Compare stats to cov matrix."""
        interp = GaussianFit(self.init_mean, self.init_std, self.init_height)
        true_params = np.r_[self.true_mean, self.true_height, self.true_std]
        std_y = 0.1
        M = 200
        param_set = np.zeros((len(true_params), M))
        for n in range(M):
            interp.fit(self.x,
                       self.y + std_y * np.random.randn(len(self.y)), std_y)
            param_set[:, n] = np.r_[interp.mean, interp.height, interp.std]
        mean_params = param_set.mean(axis=1)
        norm_params = param_set - mean_params[:, np.newaxis]
        cov_params = np.dot(norm_params, norm_params.T) / M
        estm_std_params = np.sqrt(np.diag(cov_params))
        std_params = np.r_[interp.std_mean, interp.std_height, interp.std_std]
        self.assertTrue(
            (np.abs(mean_params - true_params) / std_params < 0.5).all(),
            "Sample mean parameter vector differs too much from true value")
        # Only check diagonal of cov matrix -
        # the rest is probably affected by linearisation
        self.assertTrue(
            (np.abs(estm_std_params - std_params) / std_params < 0.2).all(),
            "Sample parameter standard deviation differs too much")


class TestGaussianFitCircular(TestCase):
    """Check the GaussianFit class with a single variance on all dimensions."""

    def setUp(self):
        self.true_mean = [0, 0]
        self.true_std = np.sqrt(10)
        self.true_height = 4
        true_gauss = GaussianFit(self.true_mean, self.true_std,
                                 self.true_height)
        self.x = 7 * np.random.randn(2, 80)
        self.y = true_gauss(self.x)
        self.init_mean = [3, -2]
        self.init_std = 1
        self.init_height = 1

    def test_fit_eval_diagcov(self):
        """GaussianFit (shared stdev): Basic function fitting + evaluation."""
        interp = GaussianFit(self.init_mean, self.init_std, self.init_height)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        assert_almost_equal(interp.mean, self.true_mean, decimal=7)
        assert_almost_equal(interp.std, self.true_std, decimal=7)
        assert_almost_equal(interp.height, self.true_height, decimal=7)
        assert_almost_equal(y, self.y, decimal=7)


class TestGaussianFitDegenerate(TestCase):
    """Check the GaussianFit class with a singular parameter cov matrix."""

    def setUp(self):
        self.true_mean, self.true_std, self.true_height = [0, 0], 1., 0.
        true_gauss = GaussianFit(self.true_mean, self.true_std,
                                 self.true_height)
        self.x = 7 * np.random.randn(2, 80)
        self.y = true_gauss(self.x)
        self.init_mean, self.init_std, self.init_height = [0, 0], 1, 0

    def test_fit_eval_diagcov(self):
        """GaussianFit (shared stdev): Basic function fitting + evaluation."""
        interp = GaussianFit(self.init_mean, self.init_std, self.init_height)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        assert_almost_equal(interp.mean, self.true_mean, decimal=7)
        assert_almost_equal(interp.std, self.true_std, decimal=7)
        assert_almost_equal(interp.height, self.true_height, decimal=7)
        assert_almost_equal(y, self.y, decimal=7)
