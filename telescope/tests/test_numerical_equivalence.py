# -*- coding: utf-8 -*-
"""Numerical equivalence tests across tiers and modes.

Tests cover:
- Stock vs CPU-optimized: pi, theta, lnl within rtol=1e-10
- Standard EM vs block-parallel EM on single block: exact match
- All 6 reassignment modes produce identical count vectors
"""
import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

from telescope.utils.sparse_plus import csr_matrix_plus
from telescope.utils.model import Telescope, TelescopeLikelihood
from telescope.utils.annotation import get_annotation_class
from telescope.utils import backend

__author__ = 'Duane Storey'

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
)


class MockOpts:
    def __init__(self):
        self.em_epsilon = 1e-7
        self.max_iter = 100
        self.pi_prior = 0
        self.theta_prior = 200000
        self.samfile = os.path.join(DATA_DIR, 'alignment.bam')
        self.gtffile = os.path.join(DATA_DIR, 'annotation.gtf')
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
        self.reassign_mode = 'exclude'
        self.conf_prob = 0.9
        self.version = 'test'
        self.use_likelihood = True
        self.skip_em = False

    def outfile_path(self, suffix):
        return os.path.join(self.outdir, f'{self.exp_tag}-{suffix}')


@pytest.fixture(scope='module')
def pipeline_data():
    """Load bundled test data once for the module."""
    samfile = os.path.join(DATA_DIR, 'alignment.bam')
    gtffile = os.path.join(DATA_DIR, 'annotation.gtf')

    if not os.path.exists(samfile) or not os.path.exists(gtffile):
        pytest.skip("Bundled test data not found")

    opts = MockOpts()
    Annotation = get_annotation_class('intervaltree')
    annot = Annotation(gtffile, 'locus', opts.stranded_mode)
    ts = Telescope(opts)
    ts.load_alignment(annot)
    return ts, opts


# --- Stock vs CPU-optimized equivalence ---

class TestStockVsOptimized:
    """Verify stock and optimized backends produce identical results."""

    def _run_em_with_backend(self, raw_scores, opts, backend_name):
        """Run EM with a specific backend."""
        if backend_name == 'stock':
            backend._active_backend = backend.BackendInfo(name='cpu_stock')
            scores = csr_matrix_plus(raw_scores)
        else:
            backend.configure()
            from telescope.utils.backend import get_sparse_class
            cls = get_sparse_class()
            scores = cls(raw_scores)

        seed = 42
        np.random.seed(seed)
        tl = TelescopeLikelihood(scores, opts)
        tl.em(use_likelihood=True)
        return tl

    def test_pi_equivalence(self, pipeline_data):
        ts, opts = pipeline_data
        tl_stock = self._run_em_with_backend(ts.raw_scores, opts, 'stock')
        tl_opt = self._run_em_with_backend(ts.raw_scores, opts, 'optimized')
        assert_allclose(tl_stock.pi, tl_opt.pi, rtol=1e-10)

    def test_theta_equivalence(self, pipeline_data):
        ts, opts = pipeline_data
        tl_stock = self._run_em_with_backend(ts.raw_scores, opts, 'stock')
        tl_opt = self._run_em_with_backend(ts.raw_scores, opts, 'optimized')
        assert_allclose(tl_stock.theta, tl_opt.theta, rtol=1e-10)

    def test_lnl_equivalence(self, pipeline_data):
        ts, opts = pipeline_data
        tl_stock = self._run_em_with_backend(ts.raw_scores, opts, 'stock')
        tl_opt = self._run_em_with_backend(ts.raw_scores, opts, 'optimized')
        assert_allclose(tl_stock.lnl, tl_opt.lnl, rtol=1e-10)


# --- Standard EM vs block-parallel EM ---

class TestEMParallel:
    """Block-parallel on single-block data should match standard EM."""

    def test_single_block_fallback(self, pipeline_data):
        """em_parallel on single-block data falls back to em() → identical."""
        ts, opts = pipeline_data
        seed = ts.get_random_seed()

        np.random.seed(seed)
        tl_standard = TelescopeLikelihood(ts.raw_scores, opts)
        tl_standard.em(use_likelihood=True)

        np.random.seed(seed)
        tl_parallel = TelescopeLikelihood(ts.raw_scores, opts)
        tl_parallel.em_parallel(use_likelihood=True)

        assert_allclose(tl_standard.pi, tl_parallel.pi, rtol=1e-10)
        assert_allclose(tl_standard.theta, tl_parallel.theta, rtol=1e-10)
        assert_allclose(tl_standard.lnl, tl_parallel.lnl, rtol=1e-10)

    def test_multi_block_synthetic(self):
        """Synthetic 2-block matrix: parallel should converge correctly."""
        scores = csr_matrix_plus([
            [10,  5,  0,  0],
            [ 8,  3,  0,  0],
            [ 0,  7,  0,  0],
            [ 0,  0, 12,  3],
            [ 0,  0,  0,  9],
            [ 0,  0,  6,  4],
        ], dtype=np.float64)

        class Opts:
            em_epsilon = 1e-7
            max_iter = 100
            pi_prior = 0
            theta_prior = 200000

        opts = Opts()

        np.random.seed(42)
        tl = TelescopeLikelihood(scores, opts)
        tl.em_parallel(use_likelihood=True)

        assert np.isfinite(tl.lnl)
        assert tl.z is not None
        assert tl.z.shape == scores.shape
        # z rows should sum to 1
        row_sums = tl.z.sum(1).A1
        assert_allclose(row_sums, np.ones(6), atol=1e-10)


# --- All 6 reassignment modes produce identical count vectors ---

class TestReassignmentEquivalence:
    """All reassignment modes should produce identical results across tiers."""

    def test_all_modes_stock_vs_optimized(self, pipeline_data):
        ts, opts = pipeline_data
        modes = ['exclude', 'choose', 'average', 'conf', 'unique', 'all']

        # Stock
        backend._active_backend = backend.BackendInfo(name='cpu_stock')
        np.random.seed(42)
        tl_stock = TelescopeLikelihood(csr_matrix_plus(ts.raw_scores), opts)
        tl_stock.em(use_likelihood=True)

        # Optimized
        backend.configure()
        from telescope.utils.backend import get_sparse_class
        cls = get_sparse_class()
        np.random.seed(42)
        tl_opt = TelescopeLikelihood(cls(ts.raw_scores), opts)
        tl_opt.em(use_likelihood=True)

        for mode in modes:
            np.random.seed(123)
            counts_stock = tl_stock.reassign(mode, 0.9).sum(0).A1
            np.random.seed(123)
            counts_opt = tl_opt.reassign(mode, 0.9).sum(0).A1
            assert_allclose(counts_stock, counts_opt, rtol=1e-10,
                            err_msg=f"Mode '{mode}' counts differ")
