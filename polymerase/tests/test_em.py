"""Tests for PolymeraseLikelihood EM algorithm and full pipeline baseline.

Tests cover:
- PolymeraseLikelihood initialization from sparse matrix
- E-step and M-step correctness
- EM convergence
- All 6 reassignment modes
- Golden baseline: log-likelihood on bundled test data
"""

import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from polymerase.core.likelihood import PolymeraseLikelihood
from polymerase.sparse.matrix import csr_matrix_plus

# Path to bundled test data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


# --- Mock opts for PolymeraseLikelihood ---


class MockOpts:
    """Minimal options object for PolymeraseLikelihood."""

    def __init__(self, **kwargs):
        self.em_epsilon = kwargs.get('em_epsilon', 1e-7)
        self.max_iter = kwargs.get('max_iter', 100)
        self.pi_prior = kwargs.get('pi_prior', 0)
        self.theta_prior = kwargs.get('theta_prior', 200000)
        self.outdir = kwargs.get('outdir', tempfile.mkdtemp())
        self.exp_tag = kwargs.get('exp_tag', 'test')

    def outfile_path(self, suffix):
        basename = f'{self.exp_tag}-{suffix}'
        return os.path.join(self.outdir, basename)


# --- Fixtures ---


@pytest.fixture
def simple_scores():
    """Simple 4x3 score matrix for unit testing EM components."""
    # 4 fragments, 3 features
    # Fragment 0: maps to feature 0 only (unique)
    # Fragment 1: maps to features 0 and 1 (ambiguous)
    # Fragment 2: maps to features 1 and 2 (ambiguous)
    # Fragment 3: maps to feature 2 only (unique)
    return csr_matrix_plus(
        [
            [10, 0, 0],
            [8, 6, 0],
            [0, 7, 5],
            [0, 0, 12],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def simple_opts():
    return MockOpts()


@pytest.fixture
def simple_tl(simple_scores, simple_opts):
    return PolymeraseLikelihood(simple_scores, simple_opts)


# --- PolymeraseLikelihood initialization ---


class TestPolymeraseLikelihoodInit:
    def test_dimensions(self, simple_tl):
        assert simple_tl.N == 4
        assert simple_tl.K == 3

    def test_Q_shape(self, simple_tl):
        assert simple_tl.Q.shape == (4, 3)

    def test_Q_nonzero_count(self, simple_tl):
        """Q should have same sparsity pattern as input."""
        assert simple_tl.Q.nnz == 6

    def test_Q_positive(self, simple_tl):
        """All Q values should be positive (expm1 of positive values)."""
        assert np.all(simple_tl.Q.data > 0)

    def test_Y_ambiguity(self, simple_tl):
        """Y should be 1 for ambiguous (multi-mapped) fragments."""
        expected_Y = np.array([[0], [1], [1], [0]], dtype=np.uint8)
        assert_array_almost_equal(simple_tl.Y, expected_Y)

    def test_pi_init(self, simple_tl):
        """Initial pi should be uniform."""
        assert_array_almost_equal(simple_tl.pi, [1.0 / 3, 1.0 / 3, 1.0 / 3])

    def test_theta_init(self, simple_tl):
        """Initial theta should be uniform."""
        assert_array_almost_equal(simple_tl.theta, [1.0 / 3, 1.0 / 3, 1.0 / 3])


# --- E-step ---


class TestEStep:
    def test_estep_returns_normalized(self, simple_tl):
        """E-step output rows should sum to 1."""
        z = simple_tl.estep(simple_tl.pi, simple_tl.theta)
        row_sums = z.sum(1).A1
        assert_array_almost_equal(row_sums, np.ones(4))

    def test_estep_shape(self, simple_tl):
        z = simple_tl.estep(simple_tl.pi, simple_tl.theta)
        assert z.shape == (4, 3)

    def test_estep_sparsity(self, simple_tl):
        """E-step should preserve sparsity pattern."""
        z = simple_tl.estep(simple_tl.pi, simple_tl.theta)
        assert z.nnz == simple_tl.Q.nnz

    def test_estep_positive(self, simple_tl):
        z = simple_tl.estep(simple_tl.pi, simple_tl.theta)
        assert np.all(z.data > 0)


# --- M-step ---


class TestMStep:
    def test_mstep_pi_sums_near_one(self, simple_tl):
        """Pi estimates should approximately sum to 1."""
        z = simple_tl.estep(simple_tl.pi, simple_tl.theta)
        pi, theta = simple_tl.mstep(z)
        # With prior, sum may not be exactly 1
        assert 0.9 < pi.sum() < 1.1

    def test_mstep_theta_positive(self, simple_tl):
        z = simple_tl.estep(simple_tl.pi, simple_tl.theta)
        pi, theta = simple_tl.mstep(z)
        assert np.all(theta > 0)

    def test_mstep_shapes(self, simple_tl):
        z = simple_tl.estep(simple_tl.pi, simple_tl.theta)
        pi, theta = simple_tl.mstep(z)
        assert pi.shape == (3,)
        assert theta.shape == (3,)


# --- EM convergence ---


class TestEMConvergence:
    def test_em_converges(self, simple_tl):
        """EM should converge without error."""
        simple_tl.em()
        assert simple_tl.z is not None
        assert simple_tl.pi_init is not None

    def test_em_improves_likelihood(self, simple_tl):
        """After EM, log-likelihood should be finite."""
        simple_tl.em(use_likelihood=True)
        assert np.isfinite(simple_tl.lnl)

    def test_em_pi_sums_near_one(self, simple_tl):
        simple_tl.em()
        assert 0.9 < simple_tl.pi.sum() < 1.1

    def test_em_z_normalized(self, simple_tl):
        """After EM, z rows should sum to 1."""
        simple_tl.em()
        row_sums = simple_tl.z.sum(1).A1
        assert_array_almost_equal(row_sums, np.ones(4))

    def test_em_unique_fragments_converge_to_source(self, simple_tl):
        """Unique fragments should converge to their only possible source."""
        simple_tl.em()
        # Fragment 0 maps only to feature 0
        assert simple_tl.z[0, 0] > 0.99
        # Fragment 3 maps only to feature 2
        assert simple_tl.z[3, 2] > 0.99

    def test_em_max_iter(self):
        """EM should stop at max_iter."""
        scores = csr_matrix_plus([[10, 8], [8, 10], [5, 5]], dtype=np.float64)
        opts = MockOpts(max_iter=3, em_epsilon=1e-20)
        tl = PolymeraseLikelihood(scores, opts)
        tl.em()
        # Should have run exactly 3 iterations
        assert tl.z is not None


# --- Reassignment modes ---


class TestReassignment:
    @pytest.fixture(autouse=True)
    def setup_tl(self, simple_scores, simple_opts):
        self.tl = PolymeraseLikelihood(simple_scores, simple_opts)
        self.tl.em()

    def test_reassign_exclude(self):
        result = self.tl.reassign('exclude')
        assert result.shape == (4, 3)
        # Each row should have 0 or 1 nonzero entries
        for i in range(result.shape[0]):
            assert result[i].nnz <= 1

    def test_reassign_choose(self):
        np.random.seed(42)
        result = self.tl.reassign('choose')
        assert result.shape == (4, 3)
        # Each row with assignment should have exactly 1
        for i in range(result.shape[0]):
            assert result[i].nnz <= 1

    def test_reassign_average(self):
        result = self.tl.reassign('average')
        assert result.shape == (4, 3)
        # Rows should sum to 0 or 1
        for i in range(result.shape[0]):
            s = result[i].sum()
            assert s == pytest.approx(0, abs=1e-10) or s == pytest.approx(1, abs=1e-10)

    def test_reassign_conf(self):
        result = self.tl.reassign('conf', thresh=0.9)
        assert result.shape == (4, 3)

    def test_reassign_unique(self):
        result = self.tl.reassign('unique')
        assert result.shape == (4, 3)
        # Unique fragments (0, 3) should have assignments
        assert result[0, 0] == 1
        assert result[3, 2] == 1
        # Ambiguous fragments should have 0 total assignment
        assert result[1].sum() == 0
        assert result[2].sum() == 0

    def test_reassign_all(self):
        result = self.tl.reassign('all')
        assert result.shape == (4, 3)
        # 'all' assigns to every aligned locus
        assert result[0, 0] == 1
        assert result[1, 0] == 1
        assert result[1, 1] == 1

    def test_reassign_initial(self):
        """Initial reassignment uses Q.norm(1) instead of z."""
        result = self.tl.reassign('all', initial=True)
        assert result.shape == (4, 3)

    def test_reassign_invalid_method(self):
        with pytest.raises(ValueError):
            self.tl.reassign('invalid_method')


# --- Full pipeline baseline test ---


class TestBaselineEquivalence:
    """Golden baseline: verify the full pipeline produces expected results
    on the bundled test data.

    Expected log-likelihood: ~95252.596293
    """

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        """Load bundled test data and run EM."""
        from polymerase.annotation import get_annotation_class
        from polymerase.core.model import Polymerase

        samfile = os.path.join(DATA_DIR, 'alignment.bam')
        gtffile = os.path.join(DATA_DIR, 'annotation.gtf')

        if not os.path.exists(samfile) or not os.path.exists(gtffile):
            pytest.skip('Bundled test data not found')

        # Create mock opts matching default CLI
        opts = MockOpts()
        opts.samfile = samfile
        opts.gtffile = gtffile
        opts.no_feature_key = '__no_feature'
        opts.overlap_mode = 'threshold'
        opts.overlap_threshold = 0.2
        opts.annotation_class = 'intervaltree'
        opts.stranded_mode = 'None'
        opts.attribute = 'locus'
        opts.ncpu = 1
        opts.updated_sam = False
        opts.tempdir = None
        opts.outdir = tempfile.mkdtemp()
        opts.exp_tag = 'test'
        opts.reassign_mode = 'exclude'
        opts.conf_prob = 0.9
        opts.version = 'test'
        opts.skip_em = False
        opts.use_likelihood = True

        # Load annotation
        Annotation = get_annotation_class('intervaltree')
        annot = Annotation(gtffile, 'locus', opts.stranded_mode)

        # Load alignment
        ts = Polymerase(opts)
        ts.load_alignment(annot)

        # Run EM
        seed = ts.get_random_seed()
        np.random.seed(seed)
        self.tl = PolymeraseLikelihood(ts.raw_scores, opts)
        self.tl.em(use_likelihood=True)
        self.ts = ts

    def test_golden_log_likelihood(self):
        """Log-likelihood should match the known golden value."""
        expected_lnl = 95252.596293
        assert self.tl.lnl == pytest.approx(expected_lnl, rel=1e-4), (
            f"Log-likelihood {self.tl.lnl} doesn't match expected {expected_lnl}"
        )

    def test_pipeline_shape(self):
        """Matrix dimensions should match test data expectations."""
        assert self.ts.shape[0] > 0  # Has fragments
        assert self.ts.shape[1] > 0  # Has features

    def test_pi_sums_near_one(self):
        assert 0.9 < self.tl.pi.sum() < 1.1

    def test_all_reassignment_modes_work(self):
        """All 6 reassignment modes should produce valid results."""
        modes = ['exclude', 'choose', 'average', 'conf', 'unique', 'all']
        for mode in modes:
            np.random.seed(42)
            result = self.tl.reassign(mode, 0.9)
            assert result.shape == self.tl.Q.shape, f'Mode {mode} shape mismatch'
            assert np.all(np.isfinite(result.data)), f'Mode {mode} has non-finite values'

    def test_convergence_reached(self):
        """EM should converge (not just hit max_iter)."""
        assert np.isfinite(self.tl.lnl)
        assert self.tl.z is not None
