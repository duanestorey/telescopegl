# -*- coding: utf-8 -*-
"""Comprehensive tests for csr_matrix_plus sparse matrix class.

Tests all methods: norm, scale, binmax, choose_random, apply_func,
count, check_equal, save/load, ceil, log1p, expm1, and arithmetic.
"""

from tempfile import TemporaryFile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from telescope.utils.sparse_plus import csr_matrix_plus


def sparse_equal(m1, m2):
    if m1.shape != m2.shape:
        return False
    return (m1 != m2).nnz == 0


# --- Fixtures ---

@pytest.fixture
def m3x3():
    """Standard 3x3 test matrix with zeros."""
    return csr_matrix_plus([[1, 0, 2], [0, 0, 3], [4, 5, 6]])


@pytest.fixture
def m3x3_float():
    """3x3 float matrix."""
    return csr_matrix_plus([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def m_with_zero_row():
    """Matrix with an all-zero row."""
    return csr_matrix_plus([[1, 0, 2], [0, 0, 0], [4, 5, 6]])


@pytest.fixture
def m_single_row():
    """Single row matrix."""
    return csr_matrix_plus([[3, 0, 7]])


@pytest.fixture
def m_ties():
    """Matrix with tied max values per row."""
    return csr_matrix_plus([[5, 0, 5], [3, 3, 0], [0, 0, 9]])


# --- Identity / element access ---

def test_identity(m3x3):
    assert m3x3[0, 0] == 1
    assert m3x3[0, 1] == 0
    assert m3x3[0, 2] == 2
    assert m3x3[1, 2] == 3
    assert m3x3[2, 0] == 4
    assert m3x3[2, 1] == 5
    assert m3x3[2, 2] == 6


# --- norm ---

def test_norm_all(m3x3):
    total = 1 + 2 + 3 + 4 + 5 + 6  # 21
    result = m3x3.norm()
    expected = csr_matrix_plus([
        [1./total, 0, 2./total],
        [0, 0, 3./total],
        [4./total, 5./total, 6./total]
    ])
    assert sparse_equal(result, expected)


def test_norm_row(m3x3):
    result = m3x3.norm(1)
    expected = csr_matrix_plus([
        [1./3, 0, 2./3],
        [0, 0, 1.],
        [4./15, 5./15, 6./15]
    ])
    assert sparse_equal(result, expected)


def test_norm_row_with_zero_row(m_with_zero_row):
    result = m_with_zero_row.norm(1)
    expected = csr_matrix_plus([
        [1./3, 0, 2./3],
        [0, 0, 0],
        [4./15, 5./15, 6./15]
    ])
    assert sparse_equal(result, expected)


def test_norm_preserves_type(m3x3):
    result = m3x3.norm(1)
    assert isinstance(result, csr_matrix_plus)


def test_norm_single_row(m_single_row):
    result = m_single_row.norm(1)
    assert_array_almost_equal(result.toarray(), [[3./10, 0, 7./10]])


# --- scale ---

def test_scale_global(m3x3):
    result = m3x3.scale()
    # max value is 6, so everything divides by 6
    expected = np.array([
        [1./6, 0, 2./6],
        [0, 0, 3./6],
        [4./6, 5./6, 1.]
    ])
    assert_array_almost_equal(result.toarray(), expected)


def test_scale_row(m3x3):
    result = m3x3.scale(1)
    expected = np.array([
        [1./2, 0, 1.],
        [0, 0, 1.],
        [4./6, 5./6, 1.]
    ])
    assert_array_almost_equal(result.toarray(), expected)


def test_scale_preserves_type(m3x3):
    result = m3x3.scale()
    assert isinstance(result, csr_matrix_plus)


def test_scale_single_row(m_single_row):
    result = m_single_row.scale(1)
    assert_array_almost_equal(result.toarray(), [[3./7, 0, 1.]])


# --- binmax ---

def test_binmax_row(m3x3):
    result = m3x3.binmax(1)
    expected = csr_matrix_plus([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ])
    assert sparse_equal(result, expected)


def test_binmax_with_ties(m_ties):
    result = m_ties.binmax(1)
    # Both 5s should be 1, both 3s should be 1
    expected = csr_matrix_plus([
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 1]
    ])
    assert sparse_equal(result, expected)


def test_binmax_preserves_type(m3x3):
    result = m3x3.binmax(1)
    assert isinstance(result, csr_matrix_plus)


def test_binmax_single_row(m_single_row):
    result = m_single_row.binmax(1)
    expected = csr_matrix_plus([[0, 0, 1]])
    assert sparse_equal(result, expected)


# --- choose_random ---

def test_choose_random_selects_one_per_row(m3x3):
    np.random.seed(42)
    result = m3x3.choose_random(1)
    # Each row should have at most 1 nonzero
    for i in range(result.shape[0]):
        row_nnz = result[i].nnz
        assert row_nnz <= 1, f"Row {i} has {row_nnz} nonzeros, expected <= 1"


def test_choose_random_preserves_type(m3x3):
    np.random.seed(42)
    result = m3x3.choose_random(1)
    assert isinstance(result, csr_matrix_plus)


def test_choose_random_single_nonzero_row():
    """A row with only one nonzero should keep it."""
    m = csr_matrix_plus([[0, 0, 5], [3, 0, 0], [1, 2, 3]])
    np.random.seed(42)
    result = m.choose_random(1)
    assert result[0, 2] == 5
    assert result[1, 0] == 3


def test_choose_random_deterministic_with_seed():
    m = csr_matrix_plus([[1, 2, 3], [4, 5, 6]])
    np.random.seed(123)
    r1 = m.choose_random(1)
    np.random.seed(123)
    r2 = m.choose_random(1)
    assert sparse_equal(r1, r2)


# --- apply_func ---

def test_apply_func_threshold(m3x3_float):
    result = m3x3_float.apply_func(lambda x: x if x >= 3.0 else 0)
    # apply_func doesn't eliminate zeros, so check dense output
    expected = np.array([
        [0, 0, 0],
        [0, 0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    assert_array_almost_equal(result.toarray(), expected)


def test_apply_func_indicator(m3x3_float):
    result = m3x3_float.apply_func(lambda x: 1 if x > 0 else 0)
    # All nonzero values become 1
    assert all(v == 1 for v in result.data)
    assert result.nnz == m3x3_float.nnz


def test_apply_func_identity(m3x3_float):
    result = m3x3_float.apply_func(lambda x: x)
    assert sparse_equal(result, m3x3_float)


def test_apply_func_preserves_type(m3x3_float):
    result = m3x3_float.apply_func(lambda x: x)
    assert isinstance(result, csr_matrix_plus)


# --- count ---

def test_count_row(m3x3):
    result = m3x3.count(1)
    expected = np.array([[2], [1], [3]])
    assert_array_almost_equal(result, expected)


def test_count_with_zero_row(m_with_zero_row):
    result = m_with_zero_row.count(1)
    expected = np.array([[2], [0], [3]])
    assert_array_almost_equal(result, expected)


def test_count_single_row(m_single_row):
    result = m_single_row.count(1)
    expected = np.array([[2]])
    assert_array_almost_equal(result, expected)


# --- check_equal ---

def test_check_equal_same(m3x3):
    assert m3x3.check_equal(m3x3.copy())


def test_check_equal_different():
    m1 = csr_matrix_plus([[1, 0], [0, 1]])
    m2 = csr_matrix_plus([[1, 0], [0, 2]])
    assert not m1.check_equal(m2)


def test_check_equal_different_shape():
    m1 = csr_matrix_plus([[1, 0]])
    m2 = csr_matrix_plus([[1], [0]])
    assert not m1.check_equal(m2)


# --- save / load ---

def test_save_load(m3x3):
    outfile = TemporaryFile()
    m3x3.save(outfile)
    outfile.seek(0)
    loaded = csr_matrix_plus.load(outfile)
    assert sparse_equal(m3x3, loaded)


def test_save_load_float(m3x3_float):
    outfile = TemporaryFile()
    m3x3_float.save(outfile)
    outfile.seek(0)
    loaded = csr_matrix_plus.load(outfile)
    assert sparse_equal(m3x3_float, loaded)


# --- ceil ---

def test_ceil():
    m = csr_matrix_plus([[0.3, 0, 1.7], [0, 0, 2.1]])
    result = m.ceil()
    expected = np.array([[1, 0, 2], [0, 0, 3]])
    assert_array_almost_equal(result.toarray(), expected)


# --- expm1 / log1p ---

def test_expm1(m3x3_float):
    result = m3x3_float.expm1()
    # expm1(x) = e^x - 1
    expected_vals = np.expm1(m3x3_float.toarray())
    # Only compare nonzero positions
    assert_array_almost_equal(result.toarray(), expected_vals)


def test_log1p():
    m = csr_matrix_plus([[1.0, 0, 2.0], [0, 0, 3.0]])
    result = m.log1p()
    expected_vals = np.log1p(m.toarray())
    assert_array_almost_equal(result.toarray(), expected_vals)


def test_expm1_log1p_roundtrip(m3x3_float):
    """expm1 and log1p should be approximate inverses for sparse data."""
    result = m3x3_float.expm1().log1p()
    assert_array_almost_equal(result.toarray(), m3x3_float.toarray(), decimal=10)


# --- multiply (inherited from scipy but important for EM) ---

def test_multiply_scalar(m3x3_float):
    result = m3x3_float.multiply(2.0)
    expected = csr_matrix_plus([[2, 0, 4], [0, 0, 6], [8, 10, 12]])
    assert_array_almost_equal(result.toarray(), expected.toarray())


def test_multiply_vector(m3x3_float):
    """Row-wise multiply by column vector (used in EM for weighting)."""
    vec = np.array([[1], [2], [3]])
    result = m3x3_float.multiply(vec)
    expected = np.array([[1, 0, 2], [0, 0, 6], [12, 15, 18]])
    assert_array_almost_equal(result.toarray(), expected)


def test_multiply_row_vector(m3x3_float):
    """Column-wise multiply by row vector (used in EM for pi * theta)."""
    vec = np.array([2, 3, 4])
    result = m3x3_float.multiply(vec)
    expected = np.array([[2, 0, 8], [0, 0, 12], [8, 15, 24]])
    assert_array_almost_equal(result.toarray(), expected)


# --- astype ---

def test_astype(m3x3_float):
    result = m3x3_float.astype(np.uint8)
    assert result.dtype == np.uint8
    assert result[0, 0] == 1
    assert result[2, 2] == 6


# --- sum (inherited but critical for EM) ---

def test_sum_all(m3x3):
    assert m3x3.sum() == 21


def test_sum_rows(m3x3):
    result = m3x3.sum(1)
    expected = np.array([[3], [3], [15]])
    assert_array_almost_equal(result, expected)


def test_sum_cols(m3x3):
    result = m3x3.sum(0)
    expected = np.array([[5, 5, 11]])
    assert_array_almost_equal(result, expected)


# --- max (inherited but used in scale/EM) ---

def test_max_all(m3x3):
    assert m3x3.max() == 6


def test_max_rows(m3x3):
    result = m3x3.max(1).toarray()
    expected = np.array([[2], [3], [6]])
    assert_array_almost_equal(result, expected)


# --- Integration: chained operations (like EM init) ---

def test_em_init_chain():
    """Simulate the EM initialization: raw_scores.scale().multiply(100).expm1()"""
    raw = csr_matrix_plus([[10, 0, 20], [0, 0, 30], [40, 50, 60]])
    Q = raw.scale().multiply(100.0).expm1()
    # Should produce a valid sparse matrix with finite values
    assert Q.shape == (3, 3)
    assert np.all(np.isfinite(Q.data))
    assert Q.nnz == raw.nnz
    assert isinstance(Q, csr_matrix_plus)


def test_norm_then_binmax():
    """Simulate reassignment: z.binmax(1) after norm."""
    m = csr_matrix_plus([[1.0, 0, 3.0], [2.0, 2.0, 0], [0, 0, 5.0]])
    normed = m.norm(1)
    bm = normed.binmax(1)
    # Row 0: max is 3/4, col 2
    assert bm[0, 2] == 1
    assert bm[0, 0] == 0
    # Row 1: tied at 0.5, both should be 1
    assert bm[1, 0] == 1
    assert bm[1, 1] == 1
    # Row 2: only col 2
    assert bm[2, 2] == 1
