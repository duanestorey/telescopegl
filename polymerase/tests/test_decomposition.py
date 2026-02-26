# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Tests for connected component block decomposition.

Tests cover:
- Synthetic 2-block matrix decomposition
- Single-block matrix (n_components == 1)
- Round-trip: split_matrix -> merge_results
- Bundled test data confirms 1 component (all ERVK)
"""
import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import scipy.sparse
from polymerase.sparse.matrix import csr_matrix_plus
from polymerase.sparse.decompose import find_blocks, split_matrix, merge_results

# Path to bundled test data
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
)


# --- Fixtures ---

@pytest.fixture
def two_block_matrix():
    """Synthetic matrix with 2 independent blocks.

    Features 0-1 share reads (rows 0-2), features 2-3 share reads (rows 3-5).
    No cross-mapping between blocks.
    """
    # 6 fragments x 4 features
    return csr_matrix_plus([
        [10,  5,  0,  0],   # row 0: maps to feat 0,1
        [ 8,  0,  0,  0],   # row 1: maps to feat 0 only
        [ 0,  7,  0,  0],   # row 2: maps to feat 1 only
        [ 0,  0, 12,  3],   # row 3: maps to feat 2,3
        [ 0,  0,  0,  9],   # row 4: maps to feat 3 only
        [ 0,  0,  6,  0],   # row 5: maps to feat 2 only
    ], dtype=np.float64)


@pytest.fixture
def single_block_matrix():
    """All features connected through shared reads."""
    return csr_matrix_plus([
        [10,  5,  0],
        [ 0,  7,  3],
        [ 4,  0,  6],
    ], dtype=np.float64)


# --- find_blocks ---

class TestFindBlocks:

    def test_two_blocks(self, two_block_matrix):
        n_comp, labels = find_blocks(two_block_matrix)
        assert n_comp == 2
        # Features 0,1 should be in same block; features 2,3 in another
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_single_block(self, single_block_matrix):
        n_comp, labels = find_blocks(single_block_matrix)
        assert n_comp == 1
        assert np.all(labels == labels[0])

    def test_diagonal_matrix_all_separate(self):
        """Identity-like matrix: each feature is its own block."""
        m = csr_matrix_plus(np.eye(4))
        n_comp, labels = find_blocks(m)
        assert n_comp == 4

    def test_fully_connected(self):
        """Single row mapping to all features → 1 block."""
        m = csr_matrix_plus([[1, 2, 3, 4]])
        n_comp, labels = find_blocks(m)
        assert n_comp == 1


# --- split_matrix ---

class TestSplitMatrix:

    def test_two_blocks_split(self, two_block_matrix):
        n_comp, labels = find_blocks(two_block_matrix)
        blocks = split_matrix(two_block_matrix, labels, n_comp)
        assert len(blocks) == 2

        # Each block should have the right dimensions
        total_rows = sum(b[0].shape[0] for b in blocks)
        total_cols = sum(b[0].shape[1] for b in blocks)
        assert total_rows == 6  # All rows accounted for
        assert total_cols == 4  # All features accounted for

    def test_single_block_split(self, single_block_matrix):
        n_comp, labels = find_blocks(single_block_matrix)
        blocks = split_matrix(single_block_matrix, labels, n_comp)
        assert len(blocks) == 1
        assert blocks[0][0].shape == single_block_matrix.shape

    def test_split_preserves_data(self, two_block_matrix):
        """Data values should be preserved after splitting."""
        n_comp, labels = find_blocks(two_block_matrix)
        blocks = split_matrix(two_block_matrix, labels, n_comp)

        for sub_matrix, feat_indices, row_indices in blocks:
            original_sub = two_block_matrix[row_indices][:, feat_indices]
            assert_array_almost_equal(
                sub_matrix.toarray(),
                original_sub.toarray()
            )


# --- merge_results round-trip ---

class TestMergeResults:

    def test_round_trip_pi_theta(self, two_block_matrix):
        """Merge should reconstruct pi and theta arrays at correct indices."""
        n_comp, labels = find_blocks(two_block_matrix)
        blocks = split_matrix(two_block_matrix, labels, n_comp)
        K = two_block_matrix.shape[1]

        # Fake block results with known values
        block_results = []
        for sub_matrix, feat_indices, row_indices in blocks:
            k = len(feat_indices)
            pi_b = np.ones(k) * 0.5
            theta_b = np.ones(k) * 0.25
            z_b = scipy.sparse.csr_matrix(np.ones((sub_matrix.shape[0], k)) * 0.1)
            lnl_b = 100.0
            block_results.append((pi_b, theta_b, z_b, lnl_b))

        pi, theta, z_rows, z_cols, z_data, lnl = merge_results(
            block_results, blocks, K
        )

        assert pi.shape == (K,)
        assert theta.shape == (K,)
        assert lnl == 200.0  # Sum of block lnls
        # After normalization, pi sums to 1.0 globally
        # 4 features × 0.5 each → sum=2.0 → normalized to 0.25 each
        np.testing.assert_almost_equal(pi.sum(), 1.0)
        assert np.all(pi == 0.25)
        assert np.all(theta == 0.25)

    def test_round_trip_z_shape(self, two_block_matrix):
        """Z coordinates should map back to original matrix dimensions."""
        n_comp, labels = find_blocks(two_block_matrix)
        blocks = split_matrix(two_block_matrix, labels, n_comp)
        N, K = two_block_matrix.shape

        block_results = []
        for sub_matrix, feat_indices, row_indices in blocks:
            k = len(feat_indices)
            n = sub_matrix.shape[0]
            pi_b = np.ones(k) / k
            theta_b = np.ones(k) / k
            z_b = scipy.sparse.random(n, k, density=0.5, format='csr')
            lnl_b = 50.0
            block_results.append((pi_b, theta_b, z_b, lnl_b))

        pi, theta, z_rows, z_cols, z_data, lnl = merge_results(
            block_results, blocks, K
        )

        # All row indices should be valid
        assert np.all(z_rows < N)
        assert np.all(z_cols < K)

        # Reconstruct z as CSR
        z = scipy.sparse.csr_matrix(
            (z_data, (z_rows, z_cols)), shape=(N, K)
        )
        assert z.shape == (N, K)


# --- Bundled test data ---

class TestBundledData:

    def test_bundled_data_single_block(self):
        """Bundled test data (ERVK family) should be 1 connected component."""
        from polymerase.core.model import Polymerase
        from polymerase.annotation import get_annotation_class

        samfile = os.path.join(DATA_DIR, 'alignment.bam')
        gtffile = os.path.join(DATA_DIR, 'annotation.gtf')

        if not os.path.exists(samfile) or not os.path.exists(gtffile):
            pytest.skip("Bundled test data not found")

        import tempfile

        class MockOpts:
            def __init__(self):
                self.samfile = samfile
                self.gtffile = gtffile
                self.no_feature_key = '__no_feature'
                self.overlap_mode = 'threshold'
                self.overlap_threshold = 0.2
                self.annotation_class = 'intervaltree'
                self.stranded_mode = 'None'
                self.attribute = 'locus'
                self.ncpu = 1
                self.updated_sam = False
                self.tempdir = None
                self.outdir = tempfile.mkdtemp()
                self.exp_tag = 'test'
                self.version = 'test'

            def outfile_path(self, suffix):
                return os.path.join(self.outdir, f'{self.exp_tag}-{suffix}')

        opts = MockOpts()
        Annotation = get_annotation_class('intervaltree')
        annot = Annotation(gtffile, 'locus', opts.stranded_mode)
        ts = Polymerase(opts)
        ts.load_alignment(annot)

        n_comp, labels = find_blocks(ts.raw_scores)
        assert n_comp == 1, f"Expected 1 block for ERVK data, got {n_comp}"
