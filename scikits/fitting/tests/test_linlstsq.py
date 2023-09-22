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

"""Tests for linear least-squares fitter.

:author: Ludwig Schwardt
:license: Modified BSD

"""
from __future__ import division

import warnings

from builtins import range
import numpy as np
from numpy.testing import TestCase, assert_equal, assert_almost_equal

from scikits.fitting import LinearLeastSquaresFit, NotFittedError


class TestLinearLeastSquaresFit(TestCase):
    """Fit linear regression model to data from a known model, and compare."""

    def setUp(self):
        self.params = np.array([0.1, -0.2, 0.0, 0.5, 0.5])
        self.N = 1000
        self.x = np.random.randn(len(self.params), self.N)
        self.y = np.dot(self.params, self.x)
        t = np.arange(0., 10., 10. / self.N)
        self.poly_x = np.vander(t, 5).T
        self.poly_y = np.dot(self.params, self.poly_x)

    def test_fit_eval(self):
        """LinearLeastSquaresFit: Basic function fitting and evaluation."""
        interp = LinearLeastSquaresFit()
        self.assertRaises(NotFittedError, interp, self.x)
        interp.fit(self.x, self.y)
        y = interp(self.x)
        assert_almost_equal(interp.params, self.params, decimal=10)
        assert_almost_equal(y, self.y, decimal=10)

    def test_cov_params(self):
        """LinearLeastSquaresFit: Compare param stats to covariance matrix."""
        interp = LinearLeastSquaresFit()
        std_y = 1.0
        M = 200
        param_set = np.zeros((len(self.params), M))
        for n in range(M):
            yn = self.poly_y + std_y * np.random.randn(len(self.poly_y))
            interp.fit(self.poly_x, yn, std_y)
            param_set[:, n] = interp.params
        mean_params = param_set.mean(axis=1)
        norm_params = param_set - mean_params[:, np.newaxis]
        cov_params = np.dot(norm_params, norm_params.T) / M
        std_params = np.sqrt(np.diag(interp.cov_params))
        self.assertTrue(
            (np.abs(mean_params - self.params) / std_params < 0.25).all(),
            "Sample mean parameter vector differs too much from true value")
        self.assertTrue(
            (np.abs(cov_params - interp.cov_params) /
             np.abs(interp.cov_params) < 1.0).all(),
            "Sample parameter covariance matrix differs too much")

    def test_vs_numpy(self):
        """LinearLeastSquaresFit: Compare fitter to np.linalg.lstsq."""
        interp = LinearLeastSquaresFit()
        interp.fit(self.x, self.y)
        params = np.linalg.lstsq(self.x.T, self.y, rcond=-1)[0]
        assert_almost_equal(interp.params, params, decimal=10)
        rcond = 1e-3
        interp = LinearLeastSquaresFit(rcond)
        with warnings.catch_warnings(record=True) as warning:
            interp.fit(self.poly_x, self.poly_y)
            assert_equal(str(warning[0].message),
                         "Least-squares fit may be poorly conditioned")
        params = np.linalg.lstsq(self.poly_x.T, self.poly_y, rcond)[0]
        assert_almost_equal(interp.params, params, decimal=10)
