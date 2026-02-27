# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Numba-accelerated sparse matrix operations for Polymerase.

Provides CpuOptimizedCsrMatrix, a subclass of csr_matrix_plus with
Numba JIT-compiled kernels for hot-path operations. Falls back to
the base class methods when Numba is unavailable.
"""

import numpy as np

from .matrix import csr_matrix_plus

try:
    import numba  # noqa: F401
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# --- Numba JIT kernels (defined at module level for compilation caching) ---

if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _binmax_kernel(data, indices, indptr, n_rows):
        """Find max per row and set matching values to 1, others to 0.

        Returns a new data array of int8.
        """
        out = np.zeros(len(data), dtype=np.int8)
        for i in prange(n_rows):
            start = indptr[i]
            end = indptr[i + 1]
            if start == end:
                continue
            # Find row max
            row_max = data[start]
            for j in range(start + 1, end):
                if data[j] > row_max:
                    row_max = data[j]
            # Set matching values to 1
            for j in range(start, end):
                if data[j] == row_max:
                    out[j] = 1
        return out

    @njit(cache=True)
    def _choose_random_kernel(data, indptr, n_rows, seed):
        """Randomly select one nonzero per row. Sequential (PRNG not parallel-safe)."""
        np.random.seed(seed)
        out = data.copy()
        for i in range(n_rows):
            start = indptr[i]
            end = indptr[i + 1]
            if end - start > 1:
                chosen = start + np.random.randint(0, end - start)
                for j in range(start, end):
                    if j != chosen:
                        out[j] = 0
        return out

    @njit(parallel=True, cache=True)
    def _apply_threshold_kernel(data, threshold):
        """Set values below threshold to 0."""
        out = np.empty_like(data)
        for i in prange(len(data)):
            out[i] = data[i] if data[i] >= threshold else 0
        return out

    @njit(parallel=True, cache=True)
    def _apply_indicator_kernel(data):
        """Set positive values to 1, others to 0."""
        out = np.empty(len(data), dtype=np.uint8)
        for i in prange(len(data)):
            out[i] = 1 if data[i] > 0 else 0
        return out


class CpuOptimizedCsrMatrix(csr_matrix_plus):
    """Numba-accelerated CSR matrix extending csr_matrix_plus.

    Each override checks HAS_NUMBA and falls back to super() if unavailable.
    Returns type(self)(...) to preserve class through operations.
    """

    def binmax(self, axis=None):
        if not HAS_NUMBA or axis != 1:
            return super().binmax(axis)

        _data = _binmax_kernel(self.data.astype(np.float64), self.indices, self.indptr, self.shape[0])
        ret = type(self)(
            (_data, self.indices.copy(), self.indptr.copy()),
            shape=self.shape,
        )
        ret.eliminate_zeros()
        return ret

    def choose_random(self, axis=None):
        if not HAS_NUMBA or axis != 1:
            return super().choose_random(axis)

        # Use numpy's current random state to generate a seed for numba
        seed = np.random.randint(0, 2**31 - 1)
        _data = _choose_random_kernel(
            self.data.copy(),
            self.indptr,
            self.shape[0],
            seed,
        )
        ret = type(self)(
            (_data, self.indices.copy(), self.indptr.copy()),
            shape=self.shape,
        )
        ret.eliminate_zeros()
        return ret

    def threshold_filter(self, threshold):
        if not HAS_NUMBA:
            return super().threshold_filter(threshold)

        _data = _apply_threshold_kernel(self.data, threshold)
        ret = type(self)(
            (_data, self.indices.copy(), self.indptr.copy()),
            shape=self.shape,
        )
        ret.eliminate_zeros()
        return ret

    def indicator(self):
        if not HAS_NUMBA:
            return super().indicator()

        _data = _apply_indicator_kernel(self.data)
        ret = type(self)(
            (_data, self.indices.copy(), self.indptr.copy()),
            shape=self.shape,
        )
        ret.eliminate_zeros()
        return ret
