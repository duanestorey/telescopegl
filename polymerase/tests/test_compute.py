# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Tests for polymerase.compute module."""

import numpy as np
import pytest
import scipy.sparse

from polymerase.compute import get_ops
from polymerase.compute.cpu_ops import CpuOps
from polymerase.compute.ops import ComputeOps


@pytest.fixture
def cpu_ops():
    return CpuOps()


@pytest.fixture
def sparse_matrix():
    """3x4 sparse matrix for testing."""
    row = np.array([0, 0, 1, 1, 2, 2, 2])
    col = np.array([0, 2, 1, 3, 0, 1, 2])
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    return scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 4))


@pytest.fixture
def dense_matrix():
    return np.array([[1.0, 0, 2.0, 0], [0, 3.0, 0, 4.0], [5.0, 6.0, 7.0, 0]])


# --- Backend info ---


class TestBackendInfo:
    def test_get_ops_returns_compute_ops(self):
        ops = get_ops()
        assert isinstance(ops, ComputeOps)

    def test_cpu_ops_device(self, cpu_ops):
        assert cpu_ops.device == 'cpu'

    def test_cpu_ops_name(self, cpu_ops):
        assert 'CPU' in cpu_ops.name


# --- Sparse operations ---


class TestSparseOps:
    def test_dot(self, cpu_ops, sparse_matrix):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        result = cpu_ops.dot(sparse_matrix, v)
        expected = sparse_matrix.dot(v)
        np.testing.assert_array_almost_equal(result, expected)

    def test_norm(self, cpu_ops, sparse_matrix):
        result = cpu_ops.norm(sparse_matrix, 1)
        # Each row should sum to 1
        row_sums = np.asarray(result.sum(axis=1)).flatten()
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0, 1.0])

    def test_scale(self, cpu_ops, sparse_matrix):
        result = cpu_ops.scale(sparse_matrix, 1)
        # Max per row should be 1
        row_maxes = np.asarray(result.max(axis=1).toarray()).flatten()
        np.testing.assert_array_almost_equal(row_maxes, [1.0, 1.0, 1.0])

    def test_multiply(self, cpu_ops, sparse_matrix):
        scalar = 2.0
        result = cpu_ops.multiply(sparse_matrix, scalar)
        expected = sparse_matrix.multiply(scalar)
        np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())

    def test_sum_axis0(self, cpu_ops, sparse_matrix):
        result = cpu_ops.sum(sparse_matrix, 0)
        expected = np.asarray(sparse_matrix.sum(axis=0)).flatten()
        np.testing.assert_array_almost_equal(result, expected)

    def test_sum_axis1(self, cpu_ops, sparse_matrix):
        result = cpu_ops.sum(sparse_matrix, 1)
        expected = np.asarray(sparse_matrix.sum(axis=1)).flatten()
        np.testing.assert_array_almost_equal(result, expected)

    def test_threshold(self, cpu_ops, sparse_matrix):
        result = cpu_ops.threshold(sparse_matrix, 3.0)
        # Values below 3 should be zeroed
        dense = result.toarray()
        assert dense[0, 0] == 0  # 1.0 < 3.0
        assert dense[0, 2] == 0  # 2.0 < 3.0
        assert dense[1, 0] == 0  # no entry
        assert dense[1, 1] == 3.0
        assert dense[2, 2] == 7.0


# --- Dense linear algebra ---


class TestDenseOps:
    def test_solve(self, cpu_ops):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([9.0, 8.0])
        x = cpu_ops.solve(A, b)
        np.testing.assert_array_almost_equal(A.dot(x), b)

    def test_nnls(self, cpu_ops):
        A = np.array([[1.0, 0], [0, 1.0], [1.0, 1.0]])
        b = np.array([1.0, 2.0, 2.5])
        x, residual = cpu_ops.nnls(A, b)
        assert x[0] >= 0
        assert x[1] >= 0

    def test_svd(self, cpu_ops, dense_matrix):
        U, s, Vt = cpu_ops.svd(scipy.sparse.csr_matrix(dense_matrix), k=2)
        assert U.shape == (3, 2)
        assert s.shape == (2,)
        assert Vt.shape == (2, 4)

    def test_pca(self, cpu_ops, dense_matrix):
        result = cpu_ops.pca(scipy.sparse.csr_matrix(dense_matrix), n_components=2)
        assert result.shape == (3, 2)


# --- Array utilities ---


class TestArrayUtils:
    def test_to_dense(self, cpu_ops, sparse_matrix):
        result = cpu_ops.to_dense(sparse_matrix)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, sparse_matrix.toarray())

    def test_to_sparse(self, cpu_ops, dense_matrix):
        result = cpu_ops.to_sparse(dense_matrix)
        assert scipy.sparse.issparse(result)
        np.testing.assert_array_equal(result.toarray(), dense_matrix)

    def test_zeros(self, cpu_ops):
        result = cpu_ops.zeros((3, 4))
        assert result.shape == (3, 4)
        assert np.all(result == 0)

    def test_array(self, cpu_ops):
        result = cpu_ops.array([1, 2, 3], dtype=np.float64)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_arange(self, cpu_ops):
        result = cpu_ops.arange(5)
        np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])
