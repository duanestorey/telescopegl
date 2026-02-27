# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""GPU backend using CuPy + cuSPARSE."""

import scipy.sparse

from .ops import ComputeOps


class GpuOps(ComputeOps):
    """GPU backend using CuPy + cuSPARSE."""

    def __init__(self):
        import cupy as cp
        import cupyx.scipy.sparse as cpsparse

        self._cp = cp
        self._cpsparse = cpsparse

    # --- Backend info ---

    @property
    def device(self) -> str:
        return 'gpu'

    @property
    def name(self) -> str:
        return 'GPU (CuPy/cuSPARSE)'

    # --- Internal helpers ---

    def _to_gpu(self, matrix):
        """Transfer scipy sparse or numpy array to GPU."""
        if scipy.sparse.issparse(matrix):
            return self._cpsparse.csr_matrix(matrix)
        return self._cp.asarray(matrix)

    def _to_cpu(self, matrix):
        """Transfer back from GPU."""
        if self._cpsparse.issparse(matrix):
            return matrix.get()
        if scipy.sparse.issparse(matrix):
            return matrix
        if hasattr(matrix, 'get'):  # CuPy array
            return matrix.get()
        return matrix

    # --- Sparse matrix operations ---

    def dot(self, A, B):
        A_gpu = self._to_gpu(A)
        B_gpu = self._to_gpu(B)
        result = A_gpu.dot(B_gpu)
        return self._to_cpu(result)

    def norm(self, matrix, axis):
        # CuPy sparse doesn't have our custom norm, fall back to CPU
        from ..sparse.matrix import csr_matrix_plus

        if not isinstance(matrix, csr_matrix_plus):
            matrix = csr_matrix_plus(matrix)
        return matrix.norm(axis)

    def scale(self, matrix, axis):
        from ..sparse.matrix import csr_matrix_plus

        if not isinstance(matrix, csr_matrix_plus):
            matrix = csr_matrix_plus(matrix)
        return matrix.scale(axis)

    def multiply(self, A, B):
        A_gpu = self._to_gpu(A)
        B_gpu = self._to_gpu(B)
        result = A_gpu.multiply(B_gpu)
        return self._to_cpu(result)

    def sum(self, matrix, axis):
        A_gpu = self._to_gpu(matrix)
        result = A_gpu.sum(axis=axis)
        if hasattr(result, 'get'):
            return result.get().A1 if hasattr(result, 'A1') else result.get().flatten()
        return result

    def binmax(self, matrix, axis):
        from ..sparse.matrix import csr_matrix_plus

        if not isinstance(matrix, csr_matrix_plus):
            matrix = csr_matrix_plus(matrix)
        return matrix.binmax(axis)

    def threshold(self, matrix, thresh):
        from ..sparse.matrix import csr_matrix_plus

        if not isinstance(matrix, csr_matrix_plus):
            matrix = csr_matrix_plus(matrix)
        return matrix.threshold_filter(thresh)

    def choose_random(self, matrix, axis):
        from ..sparse.matrix import csr_matrix_plus

        if not isinstance(matrix, csr_matrix_plus):
            matrix = csr_matrix_plus(matrix)
        return matrix.choose_random(axis)

    def indicator(self, matrix):
        from ..sparse.matrix import csr_matrix_plus

        if not isinstance(matrix, csr_matrix_plus):
            matrix = csr_matrix_plus(matrix)
        return matrix.indicator()

    # --- Dense linear algebra ---

    def svd(self, matrix, k):
        A_gpu = self._to_gpu(matrix)
        if self._cpsparse.issparse(A_gpu):
            A_gpu = A_gpu.toarray()
        U, s, Vt = self._cp.linalg.svd(A_gpu, full_matrices=False)
        return U[:, :k].get(), s[:k].get(), Vt[:k, :].get()

    def pca(self, matrix, n_components):
        U, s, Vt = self.svd(matrix, n_components)
        return U * s

    def nnls(self, A, b):
        # CuPy doesn't have NNLS natively — fall back to CPU
        import numpy as np
        from scipy.optimize import nnls

        A_cpu = A.get() if hasattr(A, 'get') else np.asarray(A)
        b_cpu = b.get() if hasattr(b, 'get') else np.asarray(b)
        return nnls(A_cpu, b_cpu)

    def nmf(self, matrix, n_components):
        # NMF not available in CuPy — fall back to CPU
        import numpy as np
        from sklearn.decomposition import NMF

        m = matrix.get() if hasattr(matrix, 'get') else np.asarray(matrix)
        model = NMF(n_components=n_components, init='nndsvd')
        W = model.fit_transform(m)
        H = model.components_
        return W, H

    def solve(self, A, b):
        A_gpu = self._cp.asarray(A) if not hasattr(A, 'device') else A
        b_gpu = self._cp.asarray(b) if not hasattr(b, 'device') else b
        result = self._cp.linalg.solve(A_gpu, b_gpu)
        return result.get()

    # --- Array utilities ---

    def to_dense(self, sparse_matrix):
        if scipy.sparse.issparse(sparse_matrix):
            return sparse_matrix.toarray()
        gpu = self._to_gpu(sparse_matrix)
        if self._cpsparse.issparse(gpu):
            return gpu.toarray().get()
        return gpu.get() if hasattr(gpu, 'get') else gpu

    def to_sparse(self, dense_matrix):
        return self._cpsparse.csr_matrix(self._cp.asarray(dense_matrix))

    def zeros(self, shape, dtype=float):
        return self._cp.zeros(shape, dtype=dtype)

    def array(self, data, dtype=None):
        return self._cp.asarray(data, dtype=dtype)

    def arange(self, *args):
        return self._cp.arange(*args)
