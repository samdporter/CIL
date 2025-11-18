#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 University College London
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import unittest

import numpy as np

from .utils import initialise_tests
from .testclass import CCPiTestClass
from cil.framework import ImageGeometry
from cil.optimisation.operators.KernelOperator import KernelOperator, NUMBA_AVAIL


initialise_tests()


def _center_index(shape):
    return tuple(dim // 2 for dim in shape)


def _gradient_image(geometry):
    image = geometry.allocate(0.0)
    arr = image.as_array()
    grid = np.indices(arr.shape).sum(axis=0)
    image.fill(grid.astype(np.float64))
    return image


@unittest.skipUnless(NUMBA_AVAIL, "Numba backend required for KernelOperator tests.")
class TestKernelOperator(CCPiTestClass):
    def setUp(self):
        self.geometry = ImageGeometry(5, 5, 5)

    def test_uniform_kernel_behaves_like_identity(self):
        operator = KernelOperator(
            self.geometry,
            num_neighbours=3,
            sigma_anat=1.0,
            sigma_dist=1.0,
            normalize_kernel=True,
            use_mask=False,
        )
        operator.set_anatomical_image(self.geometry.allocate(1.0))
        emission = self.geometry.allocate(5.0)

        result = operator.direct(emission)
        self.assertNumpyArrayAlmostEqual(result.as_array(), emission.as_array(), decimal=6)

    def test_dot_product_between_direct_and_adjoint(self):
        operator = KernelOperator(
            self.geometry,
            num_neighbours=3,
            sigma_anat=0.5,
            sigma_dist=1.0,
            normalize_kernel=False,
            use_mask=False,
            hybrid=False,
        )
        operator.set_anatomical_image(_gradient_image(self.geometry))

        x = self.geometry.allocate("random", seed=7)
        y = self.geometry.allocate("random", seed=11)

        forward = operator.direct(x).as_array()
        adjoint = operator.adjoint(y).as_array()

        dot_forward = float(np.sum(forward * y.as_array()))
        dot_adjoint = float(np.sum(x.as_array() * adjoint))
        self.assertAlmostEqual(dot_forward, dot_adjoint, places=6)

    def test_mask_shape_and_index_bounds(self):
        mask_k = 5
        operator = KernelOperator(
            self.geometry,
            num_neighbours=3,
            sigma_anat=0.2,
            sigma_dist=0.5,
            normalize_kernel=False,
            use_mask=True,
            mask_k=mask_k,
        )
        operator.set_anatomical_image(_gradient_image(self.geometry))

        operator.direct(self.geometry.allocate(1.0))
        mask = operator.mask
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape[-1], mask_k)

        n_cubed = operator.num_neighbours**3
        self.assertTrue(np.all(mask >= 0))
        self.assertTrue(np.all(mask < n_cubed))

    def test_normalize_features_scales_anatomical_image(self):
        operator = KernelOperator(
            self.geometry,
            normalize_features=True,
            normalize_kernel=False,
        )
        anat = _gradient_image(self.geometry)
        operator.set_anatomical_image(anat)

        stored = operator.anatomical_image.as_array()
        raw = anat.as_array()
        std = raw.std()
        expected = raw / std if std > 0 else raw
        self.assertTrue(np.allclose(stored, expected, rtol=1e-6, atol=1e-6))

    def test_distance_weighting_changes_kernel_response(self):
        center = _center_index(self.geometry.shape)

        spike = self.geometry.allocate(0.0)
        spike_arr = spike.as_array()
        spike_arr.fill(0.0)
        spike_arr[center] = 1.0
        spike.fill(spike_arr)

        operator_no_weight = KernelOperator(
            self.geometry,
            num_neighbours=5,
            sigma_anat=0.5,
            sigma_dist=1.0,
            normalize_kernel=True,
            distance_weighting=False,
            use_mask=False,
        )
        operator_no_weight.set_anatomical_image(self.geometry.allocate(1.0))
        res_no = operator_no_weight.direct(spike).as_array()

        operator_weighted = KernelOperator(
            self.geometry,
            num_neighbours=5,
            sigma_anat=0.5,
            sigma_dist=0.5,
            normalize_kernel=True,
            distance_weighting=True,
            use_mask=False,
        )
        operator_weighted.set_anatomical_image(self.geometry.allocate(1.0))
        res_dist = operator_weighted.direct(spike).as_array()

        self.assertGreater(res_dist[center], res_no[center])

    def test_sigma_emission_affects_hybrid_kernel(self):
        operator = KernelOperator(
            self.geometry,
            num_neighbours=3,
            sigma_anat=0.2,
            sigma_dist=1.0,
            sigma_emission=0.2,
            normalize_kernel=False,
            use_mask=False,
            hybrid=True,
        )
        operator.set_anatomical_image(_gradient_image(self.geometry))

        emission = self.geometry.allocate("random", seed=21)
        tight = operator.direct(emission).as_array()
        operator.set_sigma_emission(5.0)
        wide = operator.direct(emission).as_array()

        self.assertFalse(np.allclose(tight, wide, atol=1e-6, rtol=1e-5))

    def test_anatomical_weight_cache_invalidation(self):
        operator = KernelOperator(
            self.geometry,
            num_neighbours=3,
            sigma_anat=0.5,
            use_mask=False,
        )
        operator.set_anatomical_image(self.geometry.allocate(1.0))
        emission = self.geometry.allocate(1.0)

        operator.direct(emission)
        self.assertIsNotNone(operator._anatomical_weights)

        operator.set_anatomical_image(self.geometry.allocate(2.0))
        self.assertIsNone(operator._anatomical_weights)

        operator.direct(emission)
        self.assertIsNotNone(operator._anatomical_weights)

        operator.set_sigma_anat(0.8)
        self.assertIsNone(operator._anatomical_weights)


if __name__ == "__main__":
    unittest.main()
