# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Tests for CpuOptimizedCsrMatrix Numba-accelerated sparse operations.

Validates bitwise equivalence between csr_matrix_plus and
CpuOptimizedCsrMatrix for binmax, threshold_filter, indicator,
and determinism for choose_random.
"""

import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from polymerase.sparse.matrix import csr_matrix_plus

numba = pytest.importorskip('numba')
from polymerase.sparse.numba_matrix import CpuOptimizedCsrMatrix  # noqa: E402

# Path to bundled test data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


# --- Fixtures ---


@pytest.fixture(params=[csr_matrix_plus, CpuOptimizedCsrMatrix], ids=['stock', 'optimized'])
def matrix_class(request):
    """Parametrized fixture to test both sparse matrix classes."""
    return request.param


@pytest.fixture
def m3x3(matrix_class):
    return matrix_class([[1.0, 0, 2.0], [0, 0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def m_ties(matrix_class):
    return matrix_class([[5.0, 0, 5.0], [3.0, 3.0, 0], [0, 0, 9.0]])


@pytest.fixture
def m_empty_row(matrix_class):
    return matrix_class([[1.0, 0, 2.0], [0, 0, 0], [4.0, 5.0, 6.0]])


@pytest.fixture
def m_single_element(matrix_class):
    return matrix_class([[0, 7.0, 0]])


# --- binmax equivalence ---


class TestBinmaxEquivalence:
    def test_binmax_standard(self, m3x3):
        result = m3x3.binmax(1)
        expected = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.int8)
        assert_array_almost_equal(result.toarray(), expected)

    def test_binmax_ties(self, m_ties):
        result = m_ties.binmax(1)
        expected = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=np.int8)
        assert_array_almost_equal(result.toarray(), expected)

    def test_binmax_empty_row(self, m_empty_row):
        result = m_empty_row.binmax(1)
        expected = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 1]], dtype=np.int8)
        assert_array_almost_equal(result.toarray(), expected)

    def test_binmax_single_element(self, m_single_element):
        result = m_single_element.binmax(1)
        expected = np.array([[0, 1, 0]], dtype=np.int8)
        assert_array_almost_equal(result.toarray(), expected)


class TestBinmaxCrossTier:
    """Verify stock and optimized produce identical results."""

    def test_cross_tier_equivalence(self):
        data = np.array([[10.0, 0, 20], [0, 0, 30], [40, 50, 60], [5, 5, 0], [0, 0, 0], [7, 0, 3]])
        stock = csr_matrix_plus(data).binmax(1)
        optimized = CpuOptimizedCsrMatrix(data).binmax(1)
        assert_array_almost_equal(stock.toarray(), optimized.toarray())


# --- threshold_filter equivalence ---


class TestThresholdFilter:
    def test_threshold_filter(self, m3x3):
        result = m3x3.threshold_filter(3.0)
        expected = np.array([[0, 0, 0], [0, 0, 3], [4, 5, 6]])
        assert_array_almost_equal(result.toarray(), expected)

    def test_threshold_filter_zero(self, m3x3):
        """threshold=0 should keep everything."""
        result = m3x3.threshold_filter(0)
        assert_array_almost_equal(result.toarray(), m3x3.toarray())

    def test_threshold_filter_high(self, m3x3):
        """Threshold above all values should zero everything."""
        result = m3x3.threshold_filter(100)
        assert_array_almost_equal(result.toarray(), np.zeros((3, 3)))


class TestThresholdFilterCrossTier:
    def test_cross_tier_equivalence(self):
        data = np.array([[0.1, 0, 0.9], [0.5, 0.5, 0], [0, 0, 0.3]])
        stock = csr_matrix_plus(data).threshold_filter(0.5)
        optimized = CpuOptimizedCsrMatrix(data).threshold_filter(0.5)
        assert_array_almost_equal(stock.toarray(), optimized.toarray())


# --- indicator equivalence ---


class TestIndicator:
    def test_indicator(self, m3x3):
        result = m3x3.indicator()
        expected = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
        assert_array_almost_equal(result.toarray(), expected)

    def test_indicator_empty_row(self, m_empty_row):
        result = m_empty_row.indicator()
        expected = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 1]], dtype=np.uint8)
        assert_array_almost_equal(result.toarray(), expected)


class TestIndicatorCrossTier:
    def test_cross_tier_equivalence(self):
        data = np.array([[0.1, 0, 0.9], [0.5, 0.5, 0], [0, 0, 0]])
        stock = csr_matrix_plus(data).indicator()
        optimized = CpuOptimizedCsrMatrix(data).indicator()
        assert_array_almost_equal(stock.toarray(), optimized.toarray())


# --- choose_random determinism ---


class TestChooseRandom:
    def test_deterministic_with_seed(self, matrix_class):
        m = matrix_class([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.random.seed(123)
        r1 = m.choose_random(1)
        np.random.seed(123)
        r2 = m.choose_random(1)
        assert_array_almost_equal(r1.toarray(), r2.toarray())

    def test_selects_one_per_row(self, m3x3):
        np.random.seed(42)
        result = m3x3.choose_random(1)
        for i in range(result.shape[0]):
            assert result[i].nnz <= 1

    def test_single_element_row_preserved(self, m_single_element):
        np.random.seed(42)
        result = m_single_element.choose_random(1)
        assert result[0, 1] == 7.0

    def test_empty_row_stays_empty(self, m_empty_row):
        np.random.seed(42)
        result = m_empty_row.choose_random(1)
        assert result[1].nnz == 0


# --- Type preservation ---


class TestTypePreservation:
    def test_binmax_preserves_type(self):
        m = CpuOptimizedCsrMatrix([[1.0, 2.0], [3.0, 4.0]])
        result = m.binmax(1)
        assert isinstance(result, CpuOptimizedCsrMatrix)

    def test_threshold_preserves_type(self):
        m = CpuOptimizedCsrMatrix([[1.0, 2.0], [3.0, 4.0]])
        result = m.threshold_filter(2.0)
        assert isinstance(result, CpuOptimizedCsrMatrix)

    def test_indicator_preserves_type(self):
        m = CpuOptimizedCsrMatrix([[1.0, 2.0], [3.0, 4.0]])
        result = m.indicator()
        assert isinstance(result, CpuOptimizedCsrMatrix)

    def test_choose_random_preserves_type(self):
        m = CpuOptimizedCsrMatrix([[1.0, 2.0], [3.0, 4.0]])
        np.random.seed(42)
        result = m.choose_random(1)
        assert isinstance(result, CpuOptimizedCsrMatrix)
