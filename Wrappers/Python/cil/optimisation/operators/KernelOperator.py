# Copyright 2025 United Kingdom Research and Innovation
# Copyright 2025 University College London
# Copyright 2025 Sam Porter
# Author(s): Sam Porter, sam.porter.18@ucl.ac.uk
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Mapping
from typing import Any

import numpy as np
from cil.optimisation.operators import LinearOperator

try:
    import numba

    NUMBA_AVAIL = True
except (ImportError, OSError):  # pragma: no cover - optional dependency
    NUMBA_AVAIL = False
    numba = None  # type: ignore


DEFAULT_PARAMETERS = {
    "num_neighbours": 5,
    "sigma_anat": 0.1,
    "sigma_dist": 5,
    "sigma_emission": 0.1,
    "normalize_features": True,
    "normalize_kernel": True,
    "use_mask": True,
    "mask_k": 20,
    "recalc_mask": False,
    "distance_weighting": False,
    "hybrid": False,
}

PARAMETER_NAMES = tuple(DEFAULT_PARAMETERS)


class BaseKernelOperator(LinearOperator):
    r"""Common functionality for anatomical kernel operators.

    Parameters
    ----------
    domain_geometry : ImageGeometry
        Geometry describing the domain/range of the operator.
    num_neighbours : int, optional
        Odd kernel size (same along each axis) used to gather neighbourhood samples.
    sigma_anat : float, optional
        Standard deviation controlling the anatomical Gaussian weights.
    sigma_dist : float, optional
        Standard deviation used by distance weighting when enabled.
    sigma_emission : float, optional
        Standard deviation for the emission term used by hybrid kernels.
    normalize_features : bool, optional
        If ``True`` the anatomical image is normalized to unit variance.
    normalize_kernel : bool, optional
        If ``True`` the kernel response is normalised by the accumulated weights.
    use_mask : bool, optional
        Enables sparse kernels that keep only the ``mask_k`` closest neighbours.
    mask_k : int or None, optional
        Number of neighbours to retain when masking is active. ``None`` keeps all.
    recalc_mask : bool, optional
        Forces the mask and the pre-computed weights to be recomputed every call.
    distance_weighting : bool, optional
        Applies radial distance weights to the anatomical kernel.
    hybrid : bool, optional
        Enables emission weighting that depends on the current emission estimate.
    """
    def __init__(
        self,
        domain_geometry,
        *,
        num_neighbours: int = DEFAULT_PARAMETERS["num_neighbours"],
        sigma_anat: float = DEFAULT_PARAMETERS["sigma_anat"],
        sigma_dist: float = DEFAULT_PARAMETERS["sigma_dist"],
        sigma_emission: float = DEFAULT_PARAMETERS["sigma_emission"],
        normalize_features: bool = DEFAULT_PARAMETERS["normalize_features"],
        normalize_kernel: bool = DEFAULT_PARAMETERS["normalize_kernel"],
        use_mask: bool = DEFAULT_PARAMETERS["use_mask"],
        mask_k: int | None = DEFAULT_PARAMETERS["mask_k"],
        recalc_mask: bool = DEFAULT_PARAMETERS["recalc_mask"],
        distance_weighting: bool = DEFAULT_PARAMETERS["distance_weighting"],
        hybrid: bool = DEFAULT_PARAMETERS["hybrid"],
    ) -> None:
        super().__init__(domain_geometry=domain_geometry, range_geometry=domain_geometry)
        self.anatomical_image = None
        self.mask = None
        self.backend = "numba"
        self.freeze_emission_kernel = False
        self.frozen_emission_kernel = None
        self._normalisation_map = None
        self._anatomical_weights: np.ndarray | None = None
        self._parameters: dict[str, Any] = {}
        self.set_parameters(
            {
                "num_neighbours": num_neighbours,
                "sigma_anat": sigma_anat,
                "sigma_dist": sigma_dist,
                "sigma_emission": sigma_emission,
                "normalize_features": normalize_features,
                "normalize_kernel": normalize_kernel,
                "use_mask": use_mask,
                "mask_k": mask_k,
                "recalc_mask": recalc_mask,
                "distance_weighting": distance_weighting,
                "hybrid": hybrid,
            }
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(
        self,
        parameters: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        """Set one or more operator parameters.

        Parameters
        ----------
        parameters : Mapping of str to Any, optional
            Mapping whose keys correspond to supported parameter names.
        **kwargs : Any
            Additional ``name=value`` pairs overriding entries in ``parameters``.

        Raises
        ------
        KeyError
            If an unknown parameter name is supplied.
        TypeError
            If ``parameters`` is not a mapping.
        """
        updates: dict[str, Any] = {}
        if parameters is not None:
            if not isinstance(parameters, Mapping):
                raise TypeError("parameters must be a mapping of parameter values.")
            updates.update(parameters)
        updates.update(kwargs)
        if not updates:
            return
        for key, value in updates.items():
            if key not in PARAMETER_NAMES:
                raise KeyError(f"Unknown parameter '{key}'.")
            setattr(self, key, value)

    def set_num_neighbours(self, value: int) -> None:
        self.num_neighbours = value

    def set_sigma_anat(self, value: float) -> None:
        self.sigma_anat = value

    def set_sigma_dist(self, value: float) -> None:
        self.sigma_dist = value

    def set_sigma_emission(self, value: float) -> None:
        self.sigma_emission = value

    def set_normalize_features(self, value: bool) -> None:
        self.normalize_features = value

    def set_normalize_kernel(self, value: bool) -> None:
        self.normalize_kernel = value

    def set_use_mask(self, value: bool) -> None:
        self.use_mask = value

    def set_mask_k(self, value: int | None) -> None:
        self.mask_k = value

    def set_recalc_mask(self, value: bool) -> None:
        self.recalc_mask = value

    def set_distance_weighting(self, value: bool) -> None:
        self.distance_weighting = value

    def set_hybrid(self, value: bool) -> None:
        self.hybrid = value

    @property
    def num_neighbours(self) -> int:
        return self._parameters["num_neighbours"]

    @num_neighbours.setter
    def num_neighbours(self, value: int) -> None:
        int_value = int(value)
        if int_value < 1 or int_value % 2 == 0:
            raise ValueError("num_neighbours must be an odd positive integer.")
        self._parameters["num_neighbours"] = int_value
        self._invalidate_mask()
        self._invalidate_anatomical_weights()

    @property
    def sigma_anat(self) -> float:
        return self._parameters["sigma_anat"]

    @sigma_anat.setter
    def sigma_anat(self, value: float) -> None:
        if value <= 0:
            raise ValueError("sigma_anat must be > 0.")
        self._parameters["sigma_anat"] = value
        self._invalidate_anatomical_weights()

    @property
    def sigma_dist(self) -> float:
        return self._parameters["sigma_dist"]

    @sigma_dist.setter
    def sigma_dist(self, value: float) -> None:
        if value <= 0:
            raise ValueError("sigma_dist must be > 0.")
        self._parameters["sigma_dist"] = value
        self._invalidate_anatomical_weights()

    @property
    def sigma_emission(self) -> float:
        return self._parameters["sigma_emission"]

    @sigma_emission.setter
    def sigma_emission(self, value: float) -> None:
        if value <= 0:
            raise ValueError("sigma_emission must be > 0.")
        self._parameters["sigma_emission"] = value

    @property
    def normalize_features(self) -> bool:
        return self._parameters["normalize_features"]

    @normalize_features.setter
    def normalize_features(self, value: bool) -> None:
        self._parameters["normalize_features"] = value

    @property
    def normalize_kernel(self) -> bool:
        return self._parameters["normalize_kernel"]

    @normalize_kernel.setter
    def normalize_kernel(self, value: bool) -> None:
        self._parameters["normalize_kernel"] = value
        self._normalisation_map = None

    @property
    def use_mask(self) -> bool:
        return self._parameters["use_mask"]

    @use_mask.setter
    def use_mask(self, value: bool) -> None:
        self._parameters["use_mask"] = value
        self._invalidate_mask()
        self._invalidate_anatomical_weights()

    @property
    def mask_k(self) -> int | None:
        return self._parameters["mask_k"]

    @mask_k.setter
    def mask_k(self, value: int | None) -> None:
        if value is None:
            self._parameters["mask_k"] = None
        else:
            int_value = int(value)
            if int_value < 1:
                raise ValueError("mask_k must be >= 1 when provided.")
            self._parameters["mask_k"] = int_value
        self._invalidate_mask()
        self._invalidate_anatomical_weights()

    @property
    def recalc_mask(self) -> bool:
        return self._parameters["recalc_mask"]

    @recalc_mask.setter
    def recalc_mask(self, value: bool) -> None:
        self._parameters["recalc_mask"] = value
        if value:
            self._invalidate_mask()
            self._invalidate_anatomical_weights()

    @property
    def distance_weighting(self) -> bool:
        return self._parameters["distance_weighting"]

    @distance_weighting.setter
    def distance_weighting(self, value: bool) -> None:
        self._parameters["distance_weighting"] = value
        self._invalidate_anatomical_weights()

    @property
    def hybrid(self) -> bool:
        return self._parameters["hybrid"]

    @hybrid.setter
    def hybrid(self, value: bool) -> None:
        self._parameters["hybrid"] = value
        if not value:
            self.freeze_emission_kernel = False
        self.frozen_emission_kernel = None

    def _invalidate_mask(self) -> None:
        self.mask = None

    def _invalidate_anatomical_weights(self) -> None:
        self._anatomical_weights = None

    def set_anatomical_image(self, image):
        """Attach the anatomical guiding image.

        Parameters
        ----------
        image : DataContainer
            Image whose voxel intensities drive the anatomical weights.
        """
        if self.normalize_features:
            arr = image.as_array()
            std = arr.std()
            norm = arr / std if std > 1e-12 else arr
            tmp = image.clone()
            tmp.fill(norm)
            self.anatomical_image = tmp
        else:
            self.anatomical_image = image
        self._invalidate_mask()
        self._invalidate_anatomical_weights()

    def precompute_mask(self):
        """Return the k-nearest neighbour mask for the anatomical image."""
        if not NUMBA_AVAIL:
            raise RuntimeError("Numba backend required for mask precomputation.")
        if self.anatomical_image is None:
            raise RuntimeError(
                "An anatomical image must be set before precomputing a mask."
            )
        n = self.num_neighbours
        total = n**3
        mask_k = self.mask_k
        k = mask_k if mask_k is not None else total
        k = max(1, min(int(k), total))
        arr = np.ascontiguousarray(self.anatomical_image.as_array(), dtype=np.float64)
        return _nb_precompute_mask(arr, n, k)

    def precompute_anatomical_weights(self):
        """Pre-compute anatomical (and optional distance) weights.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n0, n1, n2, n**3)`` for dense kernels or
            ``(n0, n1, n2, mask_k)`` when sparse masking is enabled.
        """
        if not NUMBA_AVAIL:
            raise RuntimeError("Numba backend required for weight precomputation.")
        if self.anatomical_image is None:
            raise RuntimeError(
                "An anatomical image must be set before precomputing weights."
            )

        n = self.num_neighbours
        arr = np.ascontiguousarray(self.anatomical_image.as_array(), dtype=np.float64)

        if self.use_mask:
            if self.recalc_mask or self.mask is None:
                self.mask = self.precompute_mask()
            if self.mask is None:
                raise RuntimeError("Mask precomputation failed.")
            return _nb_precompute_anatomical_weights_mask(
                arr, self.mask, n, self.sigma_anat, self.sigma_dist, self.distance_weighting
            )
        return _nb_precompute_anatomical_weights(
            arr, n, self.sigma_anat, self.sigma_dist, self.distance_weighting
        )

    def _update_hybrid_reference(self, emission_array: np.ndarray) -> np.ndarray:
        """Store or refresh the reference emission volume used by hybrid kernels."""
        if not self.hybrid:
            return emission_array
        if self.freeze_emission_kernel:
            if self.frozen_emission_kernel is None:
                self.frozen_emission_kernel = np.array(emission_array, copy=True)
            return self.frozen_emission_kernel
        if emission_array is None:
            raise ValueError("Hybrid emission reference requires an emission array.")
        self.frozen_emission_kernel = np.array(emission_array, copy=True)
        return self.frozen_emission_kernel

    def _get_hybrid_reference(self) -> np.ndarray | None:
        """Return the emission volume required by the hybrid adjoint."""
        if not self.hybrid:
            return None
        if self.frozen_emission_kernel is None:
            raise RuntimeError(
                "Hybrid emission reference has not been initialised. "
                "Call direct() (or explicitly freeze a reference) before adjoint()."
            )
        return self.frozen_emission_kernel

    def apply(self, x):
        """Apply the kernel to ``x``."""
        if self.anatomical_image is None:
            raise RuntimeError("Set an anatomical image before calling apply().")
        return self.neighbourhood_kernel(
            x,
            self.anatomical_image,
            self.num_neighbours,
            self.sigma_anat,
            self.sigma_dist,
            self.sigma_emission,
            self.normalize_kernel,
            self.use_mask,
            self.hybrid,
        )

    def direct(self, x, out=None):
        """Alias of :meth:`apply` to satisfy the :class:`LinearOperator` API."""
        res = self.apply(x)
        if out is None:
            return res
        out.fill(res.as_array())
        return out

    def adjoint(self, x, out=None):
        """Adjoint of the operator (identical to :meth:`direct`)."""
        res = self.direct(x)
        if out is None:
            return res
        out.fill(res.as_array())
        return out


if NUMBA_AVAIL:
    # Numba kernels using pre-computed anatomical weights for dense/sparse cases

    class KernelOperator(BaseKernelOperator):
        """Numba-accelerated anatomical kernel operator."""
        def __init__(
            self,
            domain_geometry,
            *,
            num_neighbours: int = DEFAULT_PARAMETERS["num_neighbours"],
            sigma_anat: float = DEFAULT_PARAMETERS["sigma_anat"],
            sigma_dist: float = DEFAULT_PARAMETERS["sigma_dist"],
            sigma_emission: float = DEFAULT_PARAMETERS["sigma_emission"],
            normalize_features: bool = DEFAULT_PARAMETERS["normalize_features"],
            normalize_kernel: bool = DEFAULT_PARAMETERS["normalize_kernel"],
            use_mask: bool = DEFAULT_PARAMETERS["use_mask"],
            mask_k: int | None = DEFAULT_PARAMETERS["mask_k"],
            recalc_mask: bool = DEFAULT_PARAMETERS["recalc_mask"],
            distance_weighting: bool = DEFAULT_PARAMETERS["distance_weighting"],
            hybrid: bool = DEFAULT_PARAMETERS["hybrid"],
        ):
            super().__init__(
                domain_geometry,
                num_neighbours=num_neighbours,
                sigma_anat=sigma_anat,
                sigma_dist=sigma_dist,
                sigma_emission=sigma_emission,
                normalize_features=normalize_features,
                normalize_kernel=normalize_kernel,
                use_mask=use_mask,
                mask_k=mask_k,
                recalc_mask=recalc_mask,
                distance_weighting=distance_weighting,
                hybrid=hybrid,
            )
            self.backend = "numba"

        def neighbourhood_kernel(
            self,
            x,
            image,
            num_neighbours,
            sigma_anat,
            sigma_dist,
            sigma_emission,
            normalize_kernel,
            use_mask,
            hybrid,
        ):
            """Run the forward kernel on ``x`` using (optionally pre-computed) weights."""
            arr = image.as_array()
            x_arr = x.as_array()
            ref_arr = self._update_hybrid_reference(x_arr) if hybrid else x_arr
            norm_arr = (
                np.zeros_like(arr, dtype=np.float64)
                if normalize_kernel
                else np.zeros((1, 1, 1), dtype=np.float64)
            )
            n = num_neighbours

            # Pre-compute or retrieve cached anatomical weights
            if self._anatomical_weights is None or self.recalc_mask:
                self._anatomical_weights = self.precompute_anatomical_weights()

            # Use sparse or dense pre-computed kernel based on masking
            if use_mask:
                if self.mask is None:
                    raise RuntimeError("Mask was not initialised before kernel evaluation.")
                res = _nb_kernel_precomputed_sparse(
                    x_arr,
                    ref_arr,
                    self._anatomical_weights,
                    self.mask,
                    norm_arr,
                    n,
                    sigma_emission,
                    normalize_kernel,
                    hybrid,
                )
            else:
                res = _nb_kernel_precomputed(
                    x_arr,
                    ref_arr,
                    self._anatomical_weights,
                    norm_arr,
                    n,
                    sigma_emission,
                    normalize_kernel,
                    hybrid,
                )

            out = image.clone()
            out.fill(res)
            self._normalisation_map = norm_arr if normalize_kernel else None
            return out

        def adjoint(self, x, out=None):
            """Numba implementation of the adjoint operator."""
            x_arr = x.as_array()
            norm_arr = (
                self._normalisation_map
                if self.normalize_kernel
                else np.zeros((1, 1, 1), dtype=np.float64)
            )
            if self.normalize_kernel and norm_arr is None:
                raise RuntimeError(
                    "Normalization map has not been initialised. "
                    "Call direct() before adjoint() when using normalize_kernel=True."
                )

            n = self.num_neighbours

            # Pre-compute or retrieve cached anatomical weights
            if self._anatomical_weights is None or self.recalc_mask:
                self._anatomical_weights = self.precompute_anatomical_weights()

            ref_arr = self._get_hybrid_reference() if self.hybrid else x_arr

            # Use sparse or dense pre-computed adjoint kernel based on masking
            if self.use_mask:
                if self.mask is None:
                    raise RuntimeError("Mask was not initialised before adjoint evaluation.")
                res = _nb_adjoint_precomputed_sparse(
                    x_arr,
                    ref_arr,
                    self._anatomical_weights,
                    self.mask,
                    norm_arr,
                    n,
                    self.sigma_emission,
                    self.hybrid,
                )
            else:
                res = _nb_adjoint_precomputed(
                    x_arr,
                    ref_arr,
                    self._anatomical_weights,
                    norm_arr,
                    n,
                    self.sigma_emission,
                    self.hybrid,
                )

            img = x.clone()
            img.fill(res)
            if out is None:
                return img
            out.fill(res)
            return out


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_precompute_mask(anat_arr, n, k_keep):
        """
        Precompute sparse mask as integer indices of the k_keep most similar neighbors.
        Returns shape (s0, s1, s2, k_keep) with integer indices in [0, n³).
        """
        s0, s1, s2 = anat_arr.shape
        total = n ** 3
        half = n // 2
        # Return sparse indices instead of boolean mask
        mask_indices = np.zeros((s0, s1, s2, k_keep), dtype=np.int32)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    diffs = np.empty(total, dtype=np.float64)
                    center = anat_arr[i, j, k]
                    idx = 0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                diffs[idx] = abs(anat_arr[ii, jj, kk] - center)
                                idx += 1

                    # Find indices of k_keep smallest differences
                    # Using argsort (partial sort would be better but numba doesn't support it well)
                    sorted_indices = np.argsort(diffs)
                    mask_indices[i, j, k, :] = sorted_indices[:k_keep]

        return mask_indices


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_precompute_anatomical_weights(anat_arr, n, sigma_anat, sigma_dist, distance_weighting):
        """
        Pre-compute anatomical weights for all voxels and all n³ neighbors.
        Returns shape: (s0, s1, s2, n³)
        """
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        total = n ** 3
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist

        # Pre-compute distance weights
        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        weights = np.zeros((s0, s1, s2, total), dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    ca = anat_arr[i, j, k]
                    idx = 0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                diff_an = anat_arr[ii, jj, kk] - ca
                                wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                                weights[i, j, k, idx] = wi_an * wd_an[di + half, dj + half, dk + half]
                                idx += 1

        return weights


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_precompute_anatomical_weights_mask(anat_arr, mask_indices, n, sigma_anat, sigma_dist, distance_weighting):
        """
        Pre-compute anatomical weights for all voxels using sparse mask indices.
        mask_indices: shape (s0, s1, s2, k) with integer indices in [0, n³)
        Returns: shape (s0, s1, s2, k) with weights for valid neighbors only.
        """
        s0, s1, s2 = anat_arr.shape
        half = n // 2
        k = mask_indices.shape[3]  # Number of kept neighbors
        sig2_an = 2.0 * sigma_anat * sigma_anat
        dist2_an = 2.0 * sigma_dist * sigma_dist

        # Pre-compute distance weights for all possible offsets
        wd_an = np.ones((n, n, n), dtype=np.float64)
        if distance_weighting:
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    for dk in range(-half, half + 1):
                        d2 = di * di + dj * dj + dk * dk
                        wd_an[di + half, dj + half, dk + half] = np.exp(-d2 / dist2_an)

        # Sparse weights - only store k neighbors per voxel
        weights = np.zeros((s0, s1, s2, k), dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k_vox in range(s2):
                    ca = anat_arr[i, j, k_vox]

                    # Iterate only over the k valid neighbors
                    for k_idx in range(k):
                        flat_idx = mask_indices[i, j, k_vox, k_idx]

                        # Convert flat index back to (di, dj, dk) offset
                        dk = (flat_idx % n) - half
                        dj = ((flat_idx // n) % n) - half
                        di = (flat_idx // (n * n)) - half

                        # Apply boundary conditions
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        kk = k_vox + dk
                        if kk < 0:
                            kk = -kk - 1
                        elif kk >= s2:
                            kk = 2 * s2 - kk - 1

                        # Compute anatomical weight
                        diff_an = anat_arr[ii, jj, kk] - ca
                        wi_an = np.exp(-(diff_an * diff_an) / sig2_an)
                        weights[i, j, k_vox, k_idx] = wi_an * wd_an[di + half, dj + half, dk + half]

        return weights


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_kernel_precomputed(
        x_arr,
        ref_arr,
        anat_weights,
        norm_arr,
        n,
        sigma_emission,
        normalize,
        hybrid,
    ):
        """
        Forward kernel using pre-computed anatomical weights (dense version, no mask).
        Only calculates emission weights (if hybrid) and applies to data.
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        total = n ** 3
        sig2_em = 2.0 * sigma_emission * sigma_emission

        out = np.empty_like(x_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    c_ref = ref_arr[i, j, k]
                    sumv = 0.0
                    wsum = 0.0
                    idx = 0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                # Get pre-computed anatomical weight
                                w = anat_weights[i, j, k, idx]

                                # Apply emission weight if hybrid
                                if hybrid and w > 0.0:
                                    diff_em = ref_arr[ii, jj, kk] - c_ref
                                    wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                    w *= wi_em

                                sumv += x_arr[ii, jj, kk] * w
                                wsum += w
                                idx += 1

                    if normalize:
                        if wsum > 1e-12:
                            sumv /= wsum
                            norm_arr[i, j, k] = wsum
                        else:
                            norm_arr[i, j, k] = 1.0
                    out[i, j, k] = sumv

        return out


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_kernel_precomputed_sparse(
        x_arr,
        ref_arr,
        anat_weights,
        mask_indices,
        norm_arr,
        n,
        sigma_emission,
        normalize,
        hybrid,
    ):
        """
        Forward kernel using pre-computed anatomical weights (sparse version with mask).
        Only iterates over k masked neighbors per voxel.
        anat_weights: shape (s0, s1, s2, k)
        mask_indices: shape (s0, s1, s2, k) with integer indices in [0, n³)
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        k = anat_weights.shape[3]
        sig2_em = 2.0 * sigma_emission * sigma_emission

        out = np.empty_like(x_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k_vox in range(s2):
                    c_ref = ref_arr[i, j, k_vox]
                    sumv = 0.0
                    wsum = 0.0

                    # Iterate only over k valid neighbors
                    for k_idx in range(k):
                        flat_idx = mask_indices[i, j, k_vox, k_idx]

                        # Convert flat index to (di, dj, dk) offset
                        dk = (flat_idx % n) - half
                        dj = ((flat_idx // n) % n) - half
                        di = (flat_idx // (n * n)) - half

                        # Apply boundary conditions
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        kk = k_vox + dk
                        if kk < 0:
                            kk = -kk - 1
                        elif kk >= s2:
                            kk = 2 * s2 - kk - 1

                        # Get pre-computed anatomical weight
                        w = anat_weights[i, j, k_vox, k_idx]

                        # Apply emission weight if hybrid
                        if hybrid and w > 0.0:
                            diff_em = ref_arr[ii, jj, kk] - c_ref
                            wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                            w *= wi_em

                        sumv += x_arr[ii, jj, kk] * w
                        wsum += w

                    if normalize:
                        if wsum > 1e-12:
                            sumv /= wsum
                            norm_arr[i, j, k_vox] = wsum
                        else:
                            norm_arr[i, j, k_vox] = 1.0
                    out[i, j, k_vox] = sumv

        return out


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_adjoint_precomputed(
        x_arr,
        ref_arr,
        anat_weights,
        norm_arr,
        n,
        sigma_emission,
        hybrid,
    ):
        """
        Adjoint kernel using pre-computed anatomical weights (dense version, no mask).
        Only calculates emission weights (if hybrid) and applies to data.
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        sig2_em = 2.0 * sigma_emission * sigma_emission

        out = np.zeros_like(x_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    val = x_arr[i, j, k]
                    if norm_arr.shape[0] > 1:
                        norm = norm_arr[i, j, k]
                        val = val / norm if norm > 1e-12 else 0.0
                    c_ref = ref_arr[i, j, k]
                    idx = 0

                    for di in range(-half, half + 1):
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        for dj in range(-half, half + 1):
                            jj = j + dj
                            if jj < 0:
                                jj = -jj - 1
                            elif jj >= s1:
                                jj = 2 * s1 - jj - 1
                            for dk in range(-half, half + 1):
                                kk = k + dk
                                if kk < 0:
                                    kk = -kk - 1
                                elif kk >= s2:
                                    kk = 2 * s2 - kk - 1

                                # Get pre-computed anatomical weight
                                w = anat_weights[i, j, k, idx]

                                # Apply emission weight if hybrid
                                if hybrid and w > 0.0:
                                    diff_em = ref_arr[ii, jj, kk] - c_ref
                                    wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                                    w *= wi_em

                                out[ii, jj, kk] += val * w
                                idx += 1

        return out


    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _nb_adjoint_precomputed_sparse(
        x_arr,
        ref_arr,
        anat_weights,
        mask_indices,
        norm_arr,
        n,
        sigma_emission,
        hybrid,
    ):
        """
        Adjoint kernel using pre-computed anatomical weights (sparse version with mask).
        Only iterates over k masked neighbors per voxel.
        anat_weights: shape (s0, s1, s2, k)
        mask_indices: shape (s0, s1, s2, k) with integer indices in [0, n³)
        """
        s0, s1, s2 = x_arr.shape
        half = n // 2
        k = anat_weights.shape[3]
        sig2_em = 2.0 * sigma_emission * sigma_emission

        out = np.zeros_like(x_arr, dtype=np.float64)

        for i in numba.prange(s0):
            for j in range(s1):
                for k_vox in range(s2):
                    val = x_arr[i, j, k_vox]
                    if norm_arr.shape[0] > 1:
                        norm = norm_arr[i, j, k_vox]
                        val = val / norm if norm > 1e-12 else 0.0
                    c_ref = ref_arr[i, j, k_vox]

                    # Iterate only over k valid neighbors
                    for k_idx in range(k):
                        flat_idx = mask_indices[i, j, k_vox, k_idx]

                        # Convert flat index to (di, dj, dk) offset
                        dk = (flat_idx % n) - half
                        dj = ((flat_idx // n) % n) - half
                        di = (flat_idx // (n * n)) - half

                        # Apply boundary conditions
                        ii = i + di
                        if ii < 0:
                            ii = -ii - 1
                        elif ii >= s0:
                            ii = 2 * s0 - ii - 1
                        jj = j + dj
                        if jj < 0:
                            jj = -jj - 1
                        elif jj >= s1:
                            jj = 2 * s1 - jj - 1
                        kk = k_vox + dk
                        if kk < 0:
                            kk = -kk - 1
                        elif kk >= s2:
                            kk = 2 * s2 - kk - 1

                        # Get pre-computed anatomical weight
                        w = anat_weights[i, j, k_vox, k_idx]

                        # Apply emission weight if hybrid
                        if hybrid and w > 0.0:
                            diff_em = ref_arr[ii, jj, kk] - c_ref
                            wi_em = np.exp(-(diff_em * diff_em) / sig2_em)
                            w *= wi_em

                        out[ii, jj, kk] += val * w

        return out


else:

    class KernelOperator(BaseKernelOperator):  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Numba backend not available. Please install numba to use KernelOperator."
            )
