# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Abstract interface for backend-agnostic compute operations.

Primer authors use these methods instead of calling scipy/cupy directly.
The active backend handles dispatch transparently.

Example usage in a primer::

    class MyPrimer(Primer):
        def configure(self, opts, compute):
            self.ops = compute

        def commit(self, output_dir, exp_tag):
            normalised = self.ops.norm(self._matrix, axis=1)
            result = self.ops.dot(normalised, self._weights)
"""

from abc import ABC, abstractmethod


class ComputeOps(ABC):
    """Backend-agnostic compute operations for primer authors."""

    # --- Sparse matrix operations ---

    @abstractmethod
    def dot(self, A, B):
        """Sparse or dense matrix multiply. ``A @ B``."""

    @abstractmethod
    def norm(self, matrix, axis):
        """Row/column normalise a sparse matrix."""

    @abstractmethod
    def scale(self, matrix, axis):
        """Scale values to [0, 1] along *axis*."""

    @abstractmethod
    def multiply(self, A, B):
        """Element-wise multiply (Hadamard product)."""

    @abstractmethod
    def sum(self, matrix, axis):
        """Sum along *axis*, returns dense array."""

    @abstractmethod
    def binmax(self, matrix, axis):
        """Binary indicator of maximum per row/column."""

    @abstractmethod
    def threshold(self, matrix, thresh):
        """Zero out values below *thresh*."""

    @abstractmethod
    def choose_random(self, matrix, axis):
        """Randomly select one nonzero per row/column."""

    @abstractmethod
    def indicator(self, matrix):
        """Binary indicator: 1 where nonzero, 0 elsewhere."""

    # --- Dense linear algebra ---

    @abstractmethod
    def svd(self, matrix, k):
        """Truncated SVD.  Returns ``(U, s, Vt)``."""

    @abstractmethod
    def pca(self, matrix, n_components):
        """PCA via truncated SVD.  Returns transformed matrix."""

    @abstractmethod
    def nnls(self, A, b):
        """Non-negative least squares: min ||Ax - b|| s.t. x >= 0."""

    @abstractmethod
    def nmf(self, matrix, n_components):
        """Non-negative matrix factorisation.  Returns ``(W, H)``."""

    @abstractmethod
    def solve(self, A, b):
        """Solve linear system ``Ax = b``."""

    # --- Array utilities ---

    @abstractmethod
    def to_dense(self, sparse_matrix):
        """Convert sparse to dense array."""

    @abstractmethod
    def to_sparse(self, dense_matrix):
        """Convert dense to CSR sparse."""

    @abstractmethod
    def zeros(self, shape, dtype=float):
        """Create zero array on active device."""

    @abstractmethod
    def array(self, data, dtype=None):
        """Create array on active device from Python data."""

    @abstractmethod
    def arange(self, *args):
        """Range array on active device."""

    # --- Backend info ---

    @property
    @abstractmethod
    def device(self) -> str:
        """``'cpu'``, ``'cpu_numba'``, or ``'gpu'``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
