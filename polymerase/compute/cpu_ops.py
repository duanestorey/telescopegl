# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""CPU backend using scipy.sparse + numpy."""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .ops import ComputeOps


class CpuOps(ComputeOps):
    """Stock CPU backend using scipy.sparse + numpy."""

    # --- Backend info ---

    @property
    def device(self) -> str:
        return 'cpu'

    @property
    def name(self) -> str:
        return 'CPU (scipy/numpy)'

    # --- Sparse matrix operations ---

    def dot(self, A, B):
        return A.dot(B)

    def norm(self, matrix, axis):
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
        return A.multiply(B)

    def sum(self, matrix, axis):
        result = matrix.sum(axis=axis)
        return np.asarray(result).flatten()

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
        return scipy.sparse.linalg.svds(matrix, k=k)

    def pca(self, matrix, n_components):
        U, s, Vt = scipy.sparse.linalg.svds(matrix, k=n_components)
        return U * s

    def nnls(self, A, b):
        from scipy.optimize import nnls

        return nnls(A, b)

    def nmf(self, matrix, n_components):
        from sklearn.decomposition import NMF

        model = NMF(n_components=n_components, init='nndsvd')
        W = model.fit_transform(matrix)
        H = model.components_
        return W, H

    def solve(self, A, b):
        return np.linalg.solve(A, b)

    # --- Array utilities ---

    def to_dense(self, sparse_matrix):
        if scipy.sparse.issparse(sparse_matrix):
            return sparse_matrix.toarray()
        return np.asarray(sparse_matrix)

    def to_sparse(self, dense_matrix):
        return scipy.sparse.csr_matrix(dense_matrix)

    def zeros(self, shape, dtype=float):
        return np.zeros(shape, dtype=dtype)

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype)

    def arange(self, *args):
        return np.arange(*args)
