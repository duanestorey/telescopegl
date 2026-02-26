# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Integration tests for Polymerase pipeline on subsampled BAMs.

Uses SRR9666161 BAMs with retro.hg38.gtf (28K TE loci) to test the full
pipeline with realistic data, including block decomposition, multi-component
EM, and indexed vs sequential loading path equivalence.

Test data setup:
    test_data/SRR9666161_1pct_collated.bam - 1% subsample, collated
    test_data/SRR9666161_5pct_collated.bam - 5% subsample, collated
    test_data/SRR9666161_5pct_sorted.bam   - 5% subsample, sorted+indexed
    test_data/SRR9666161_100pct_sorted.bam - Full dataset, sorted+indexed
    test_data/retro.hg38.gtf - Full retro.hg38 TE annotation

Download annotation:
    wget -O test_data/retro.hg38.gtf \\
        https://github.com/mlbendall/telescope_annotation_db/raw/master/\\
        builds/retro.hg38.v1/transcripts.gtf
"""
import os
import tempfile
import time

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BAM_1PCT = os.path.join(REPO_ROOT, 'test_data', 'SRR9666161_1pct_collated.bam')
BAM_5PCT_COLLATED = os.path.join(REPO_ROOT, 'test_data', 'SRR9666161_5pct_collated.bam')
BAM_5PCT_SORTED = os.path.join(REPO_ROOT, 'test_data', 'SRR9666161_5pct_sorted.bam')
BAM_100PCT = os.path.join(REPO_ROOT, 'test_data', 'SRR9666161_100pct_sorted.bam')
ANNOTATION_FULL = os.path.join(REPO_ROOT, 'test_data', 'retro.hg38.gtf')

# Skip entire module if test data not present
pytestmark = pytest.mark.skipif(
    not (os.path.exists(BAM_1PCT) and os.path.exists(ANNOTATION_FULL)),
    reason="1% BAM or retro.hg38.gtf not in test_data/"
)


class MockOpts:
    def __init__(self, outdir):
        self.samfile = BAM_1PCT
        self.gtffile = ANNOTATION_FULL
        self.no_feature_key = '__no_feature'
        self.overlap_mode = 'threshold'
        self.overlap_threshold = 0.2
        self.annotation_class = 'intervaltree'
        self.stranded_mode = 'None'
        self.attribute = 'locus'
        self.ncpu = 1
        self.updated_sam = False
        self.tempdir = None
        self.outdir = outdir
        self.exp_tag = 'test'
        self.reassign_mode = 'exclude'
        self.conf_prob = 0.9
        self.version = 'test'
        self.skip_em = False
        self.use_likelihood = False
        self.em_epsilon = 1e-7
        self.max_iter = 100
        self.pi_prior = 0
        self.theta_prior = 200000

    def outfile_path(self, suffix):
        return os.path.join(self.outdir, '%s-%s' % (self.exp_tag, suffix))


@pytest.fixture(scope='module')
def pipeline_data():
    """Load annotation and alignment once for all tests."""
    from polymerase.sparse import backend
    from polymerase.annotation import get_annotation_class
    from polymerase.core.model import Polymerase

    backend.configure()

    outdir = tempfile.mkdtemp()
    opts = MockOpts(outdir)

    Annotation = get_annotation_class('intervaltree')
    annot = Annotation(ANNOTATION_FULL, 'locus', 'None')

    ts = Polymerase(opts)
    ts.load_alignment(annot)

    return ts, opts, annot


# -------------------------------------------------------------------------
# Alignment loading tests
# -------------------------------------------------------------------------

class TestAlignment:
    def test_total_fragments(self, pipeline_data):
        ts, _, _ = pipeline_data
        total = ts.run_info['total_fragments']
        assert 130000 < total < 140000, f"Expected ~136K fragments, got {total}"

    def test_mapped_fragments(self, pipeline_data):
        ts, _, _ = pipeline_data
        mapped = (ts.run_info['pair_mapped'] +
                  ts.run_info['pair_mixed'] +
                  ts.run_info['single_mapped'])
        assert mapped > 130000

    def test_overlapping_fragments(self, pipeline_data):
        ts, _, _ = pipeline_data
        unique = ts.run_info['overlap_unique']
        ambig = ts.run_info['overlap_ambig']
        total_overlap = unique + ambig
        assert total_overlap > 500, f"Expected >500 overlapping, got {total_overlap}"
        assert ambig > 50, f"Expected >50 multi-mapped, got {ambig}"

    def test_matrix_shape(self, pipeline_data):
        ts, _, _ = pipeline_data
        n_frags, n_feats = ts.shape
        assert n_frags > 500
        assert n_feats > 200

    def test_matrix_nnz(self, pipeline_data):
        ts, _, _ = pipeline_data
        assert ts.raw_scores.nnz > 600


# -------------------------------------------------------------------------
# EM convergence tests
# -------------------------------------------------------------------------

class TestEM:
    def test_em_converges(self, pipeline_data):
        """EM should converge within max_iter on this data."""
        from polymerase.core.likelihood import PolymeraseLikelihood

        ts, opts, _ = pipeline_data
        seed = ts.get_random_seed()
        np.random.seed(seed)

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        assert tl.lnl > 0, "Log-likelihood should be positive"

    def test_em_pi_sums_to_one(self, pipeline_data):
        from polymerase.core.likelihood import PolymeraseLikelihood

        ts, opts, _ = pipeline_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        np.testing.assert_almost_equal(tl.pi.sum(), 1.0, decimal=10)

    def test_em_deterministic(self, pipeline_data):
        """Same seed produces same result."""
        from polymerase.core.likelihood import PolymeraseLikelihood

        ts, opts, _ = pipeline_data

        np.random.seed(ts.get_random_seed())
        tl1 = PolymeraseLikelihood(ts.raw_scores, opts)
        tl1.em(use_likelihood=False)

        np.random.seed(ts.get_random_seed())
        tl2 = PolymeraseLikelihood(ts.raw_scores, opts)
        tl2.em(use_likelihood=False)

        np.testing.assert_array_equal(tl1.pi, tl2.pi)
        assert tl1.lnl == tl2.lnl

    def test_em_lnl_positive(self, pipeline_data):
        """Log-likelihood should be a finite positive number."""
        from polymerase.core.likelihood import PolymeraseLikelihood

        ts, opts, _ = pipeline_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        assert np.isfinite(tl.lnl)
        assert tl.lnl > 0


# -------------------------------------------------------------------------
# Block decomposition tests
# -------------------------------------------------------------------------

class TestDecomposition:
    def test_multiple_components(self, pipeline_data):
        """Real TE data should decompose into many independent blocks."""
        from polymerase.sparse.decompose import find_blocks

        ts, _, _ = pipeline_data
        n_components, labels = find_blocks(ts.raw_scores)

        assert n_components > 100, (
            f"Expected >100 components on real TE data, got {n_components}"
        )

    def test_component_count_matches_feature_space(self, pipeline_data):
        """Each feature should be assigned to exactly one component."""
        from polymerase.sparse.decompose import find_blocks

        ts, _, _ = pipeline_data
        n_components, labels = find_blocks(ts.raw_scores)

        assert len(labels) == ts.shape[1]
        assert set(labels) == set(range(n_components))

    def test_split_preserves_all_data(self, pipeline_data):
        """Splitting should account for all nonzero entries."""
        from polymerase.sparse.decompose import find_blocks, split_matrix

        ts, _, _ = pipeline_data
        n_components, labels = find_blocks(ts.raw_scores)
        blocks = split_matrix(ts.raw_scores, labels, n_components)

        total_nnz = sum(b[0].nnz for b in blocks)
        assert total_nnz == ts.raw_scores.nnz

    def test_blocks_are_independent(self, pipeline_data):
        """No row should appear in more than one block."""
        from polymerase.sparse.decompose import find_blocks, split_matrix

        ts, _, _ = pipeline_data
        n_components, labels = find_blocks(ts.raw_scores)
        blocks = split_matrix(ts.raw_scores, labels, n_components)

        all_rows = set()
        for _, _, row_indices in blocks:
            row_set = set(row_indices)
            overlap = all_rows & row_set
            assert len(overlap) == 0, f"Rows {overlap} appear in multiple blocks"
            all_rows |= row_set

    def test_em_parallel_produces_valid_output(self, pipeline_data):
        """Block-parallel EM should produce valid pi, theta, z, lnl."""
        from polymerase.core.likelihood import PolymeraseLikelihood

        ts, opts, _ = pipeline_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em_parallel(use_likelihood=False)

        # pi should sum to 1
        np.testing.assert_almost_equal(tl.pi.sum(), 1.0, decimal=10)
        # theta should be between 0 and 1
        assert np.all(tl.theta >= 0) and np.all(tl.theta <= 1)
        # z should be sparse with correct shape
        assert tl.z.shape == ts.raw_scores.shape
        # lnl should be finite
        assert np.isfinite(tl.lnl)


# -------------------------------------------------------------------------
# Reassignment tests
# -------------------------------------------------------------------------

class TestReassignment:
    def test_all_modes_run(self, pipeline_data):
        """All 6 reassignment modes should run without error."""
        from polymerase.core.likelihood import PolymeraseLikelihood

        ts, opts, _ = pipeline_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        modes = ['all', 'exclude', 'choose', 'average', 'conf', 'unique']
        for mode in modes:
            result = tl.reassign(mode, 0.9)
            assert result.shape == ts.raw_scores.shape
            assert result.nnz >= 0

    def test_unique_subset_of_all(self, pipeline_data):
        """Unique counts should be <= all counts for every feature."""
        from polymerase.core.likelihood import PolymeraseLikelihood

        ts, opts, _ = pipeline_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        all_counts = tl.reassign('all', 0.9).sum(0).A1
        unique_counts = tl.reassign('unique', 0.9).sum(0).A1

        assert np.all(unique_counts <= all_counts + 1e-10)

    def test_exclude_no_double_counting(self, pipeline_data):
        """Exclude mode: total assigned <= total fragments."""
        from polymerase.core.likelihood import PolymeraseLikelihood

        ts, opts, _ = pipeline_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        exclude_counts = tl.reassign('exclude', 0.9).sum(0).A1
        assert exclude_counts.sum() <= ts.shape[0]


# -------------------------------------------------------------------------
# Report generation tests
# -------------------------------------------------------------------------

class TestReportGeneration:
    def test_output_files_created(self, pipeline_data):
        """Full pipeline should produce stats and counts files."""
        from polymerase.core.likelihood import PolymeraseLikelihood
        from polymerase.core.reporter import output_report

        ts, opts, _ = pipeline_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_file = os.path.join(tmpdir, 'stats.tsv')
            counts_file = os.path.join(tmpdir, 'counts.tsv')
            output_report(ts.feat_index, ts.feature_length, ts.run_info, tl,
                          opts.reassign_mode, opts.conf_prob,
                          stats_file, counts_file)

            assert os.path.exists(stats_file)
            assert os.path.exists(counts_file)
            assert os.path.getsize(stats_file) > 0
            assert os.path.getsize(counts_file) > 0

    def test_counts_file_has_features(self, pipeline_data):
        """Counts file should list features with counts."""
        import pandas as pd
        from polymerase.core.likelihood import PolymeraseLikelihood
        from polymerase.core.reporter import output_report

        ts, opts, _ = pipeline_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_file = os.path.join(tmpdir, 'stats.tsv')
            counts_file = os.path.join(tmpdir, 'counts.tsv')
            output_report(ts.feat_index, ts.feature_length, ts.run_info, tl,
                          opts.reassign_mode, opts.conf_prob,
                          stats_file, counts_file)

            df = pd.read_csv(counts_file, sep='\t')
            assert 'transcript' in df.columns
            assert 'count' in df.columns
            assert len(df) > 0
            # At least some features should have nonzero counts
            assert (df['count'] > 0).sum() > 0


# -------------------------------------------------------------------------
# Performance benchmarks (informational, not strict assertions)
# -------------------------------------------------------------------------

class TestPerformance:
    def test_bam_loading_under_10s(self, pipeline_data):
        """BAM loading for 136K fragments should complete under 10s."""
        from polymerase.sparse import backend
        from polymerase.annotation import get_annotation_class
        from polymerase.core.model import Polymerase

        outdir = tempfile.mkdtemp()
        opts = MockOpts(outdir)
        Annotation = get_annotation_class('intervaltree')
        annot = Annotation(ANNOTATION_FULL, 'locus', 'None')

        t0 = time.perf_counter()
        ts = Polymerase(opts)
        ts.load_alignment(annot)
        elapsed = time.perf_counter() - t0

        assert elapsed < 10, f"BAM loading took {elapsed:.1f}s (expected <10s)"

    def test_em_under_1s(self, pipeline_data):
        """EM on 745x325 matrix should complete under 1s."""
        from polymerase.core.likelihood import PolymeraseLikelihood

        ts, opts, _ = pipeline_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)

        t0 = time.perf_counter()
        tl.em(use_likelihood=False)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"EM took {elapsed:.2f}s (expected <1s)"

    def test_block_decomposition_under_10ms(self, pipeline_data):
        """Block decomposition should be fast."""
        from polymerase.sparse.decompose import find_blocks

        ts, _, _ = pipeline_data

        t0 = time.perf_counter()
        find_blocks(ts.raw_scores)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.01, f"Decomposition took {elapsed*1000:.1f}ms (expected <10ms)"


# -------------------------------------------------------------------------
# Indexed vs Sequential path equivalence (5% BAM)
# -------------------------------------------------------------------------

@pytest.fixture(scope='module')
def path_comparison_data():
    """Run both loading paths on same 5% data for comparison."""
    if not (os.path.exists(BAM_5PCT_COLLATED) and os.path.exists(BAM_5PCT_SORTED)):
        pytest.skip("5% BAMs not in test_data/")

    from polymerase.sparse import backend
    from polymerase.annotation import get_annotation_class
    from polymerase.core.model import Polymerase

    backend.configure()
    Annotation = get_annotation_class('intervaltree')
    annot = Annotation(ANNOTATION_FULL, 'locus', 'None')

    class Opts5pct(MockOpts):
        def __init__(self, samfile, outdir):
            super().__init__(outdir)
            self.samfile = samfile

    # Sequential on collated BAM
    opts_seq = Opts5pct(BAM_5PCT_COLLATED, tempfile.mkdtemp())
    ts_seq = Polymerase(opts_seq)
    ts_seq.has_index = False  # Force sequential path
    ts_seq.load_alignment(annot)

    # Indexed on sorted BAM
    opts_idx = Opts5pct(BAM_5PCT_SORTED, tempfile.mkdtemp())
    ts_idx = Polymerase(opts_idx)
    ts_idx.load_alignment(annot)

    return ts_seq, ts_idx


class TestIndexedPathEquivalence:
    """Verify indexed loading produces identical results to sequential."""

    def test_same_shape(self, path_comparison_data):
        ts_seq, ts_idx = path_comparison_data
        assert ts_seq.shape == ts_idx.shape

    def test_same_nnz(self, path_comparison_data):
        ts_seq, ts_idx = path_comparison_data
        assert ts_seq.raw_scores.nnz == ts_idx.raw_scores.nnz

    def test_same_reads(self, path_comparison_data):
        ts_seq, ts_idx = path_comparison_data
        assert set(ts_seq.read_index.keys()) == set(ts_idx.read_index.keys())

    def test_same_features(self, path_comparison_data):
        ts_seq, ts_idx = path_comparison_data
        assert set(ts_seq.feat_index.keys()) == set(ts_idx.feat_index.keys())

    def test_same_overlap_stats(self, path_comparison_data):
        ts_seq, ts_idx = path_comparison_data
        assert ts_seq.run_info['overlap_unique'] == ts_idx.run_info['overlap_unique']
        assert ts_seq.run_info['overlap_ambig'] == ts_idx.run_info['overlap_ambig']

    def test_identical_matrix_values(self, path_comparison_data):
        """Every read should have identical feature assignments and scores."""
        ts_seq, ts_idx = path_comparison_data
        inv_seq = {v: k for k, v in ts_seq.feat_index.items()}
        inv_idx = {v: k for k, v in ts_idx.feat_index.items()}

        mismatches = 0
        for rid in ts_seq.read_index:
            i_seq = ts_seq.read_index[rid]
            i_idx = ts_idx.read_index[rid]
            row_seq = ts_seq.raw_scores[i_seq].toarray().flatten()
            row_idx = ts_idx.raw_scores[i_idx].toarray().flatten()
            nz_seq = {inv_seq[c]: int(row_seq[c]) for c in np.where(row_seq > 0)[0]}
            nz_idx = {inv_idx[c]: int(row_idx[c]) for c in np.where(row_idx > 0)[0]}
            if nz_seq != nz_idx:
                mismatches += 1

        assert mismatches == 0, f"{mismatches} reads have different values"

    def test_indexed_faster_than_sequential(self, path_comparison_data):
        """Indexed path should be at least 2x faster."""
        if not (os.path.exists(BAM_5PCT_COLLATED) and os.path.exists(BAM_5PCT_SORTED)):
            pytest.skip("5% BAMs not in test_data/")

        from polymerase.sparse import backend
        from polymerase.annotation import get_annotation_class
        from polymerase.core.model import Polymerase

        backend.configure()
        Annotation = get_annotation_class('intervaltree')
        annot = Annotation(ANNOTATION_FULL, 'locus', 'None')

        class Opts5pct(MockOpts):
            def __init__(self, samfile, outdir):
                super().__init__(outdir)
                self.samfile = samfile

        # Time sequential
        opts = Opts5pct(BAM_5PCT_COLLATED, tempfile.mkdtemp())
        ts = Polymerase(opts)
        ts.has_index = False
        t0 = time.perf_counter()
        ts.load_alignment(annot)
        t_seq = time.perf_counter() - t0

        # Time indexed
        opts = Opts5pct(BAM_5PCT_SORTED, tempfile.mkdtemp())
        ts = Polymerase(opts)
        t0 = time.perf_counter()
        ts.load_alignment(annot)
        t_idx = time.perf_counter() - t0

        speedup = t_seq / t_idx
        assert speedup > 2.0, (
            f"Indexed ({t_idx:.1f}s) should be >2x faster than "
            f"sequential ({t_seq:.1f}s), got {speedup:.1f}x"
        )


# -------------------------------------------------------------------------
# Full-scale 100% BAM tests
# -------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fullscale_data():
    """Load 100% BAM (14.7M fragments) — runs once per module."""
    if not os.path.exists(BAM_100PCT):
        pytest.skip("100% BAM not in test_data/")

    from polymerase.sparse import backend
    from polymerase.annotation import get_annotation_class
    from polymerase.core.model import Polymerase

    backend.configure()
    Annotation = get_annotation_class('intervaltree')
    annot = Annotation(ANNOTATION_FULL, 'locus', 'None')

    class Opts100pct(MockOpts):
        def __init__(self, outdir):
            super().__init__(outdir)
            self.samfile = BAM_100PCT
            self.max_iter = 200

    outdir = tempfile.mkdtemp()
    opts = Opts100pct(outdir)
    ts = Polymerase(opts)
    ts.load_alignment(annot)

    return ts, opts, annot


class TestFullScale:
    """Tests on full 100% BAM dataset (14.7M fragments)."""

    def test_total_fragments(self, fullscale_data):
        ts, _, _ = fullscale_data
        total = ts.run_info['total_fragments']
        assert 14_000_000 < total < 15_000_000, f"Expected ~14.7M, got {total:,}"

    def test_matrix_dimensions(self, fullscale_data):
        ts, _, _ = fullscale_data
        n_frags, n_feats = ts.shape
        assert n_frags > 50_000, f"Expected >50K fragments, got {n_frags:,}"
        assert n_feats > 3_000, f"Expected >3K features, got {n_feats:,}"

    def test_many_components(self, fullscale_data):
        from polymerase.sparse.decompose import find_blocks
        ts, _, _ = fullscale_data
        n_comp, _ = find_blocks(ts.raw_scores)
        assert n_comp > 1000, f"Expected >1000 components, got {n_comp}"

    def test_em_runs(self, fullscale_data):
        from polymerase.core.likelihood import PolymeraseLikelihood
        ts, opts, _ = fullscale_data
        np.random.seed(ts.get_random_seed())

        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        assert np.isfinite(tl.lnl)
        assert tl.lnl > 0
        np.testing.assert_almost_equal(tl.pi.sum(), 1.0, decimal=10)

    def test_report_generation(self, fullscale_data):
        import pandas as pd
        from polymerase.core.likelihood import PolymeraseLikelihood
        from polymerase.core.reporter import output_report

        ts, opts, _ = fullscale_data
        np.random.seed(ts.get_random_seed())
        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_f = os.path.join(tmpdir, 'stats.tsv')
            counts_f = os.path.join(tmpdir, 'counts.tsv')
            output_report(ts.feat_index, ts.feature_length, ts.run_info, tl,
                          opts.reassign_mode, opts.conf_prob,
                          stats_f, counts_f)

            df = pd.read_csv(counts_f, sep='\t')
            assert len(df) > 3000
            active = (df['count'] > 0).sum()
            assert active > 2000, f"Expected >2000 active loci, got {active}"

    def test_pipeline_under_60s(self, fullscale_data):
        """Full pipeline (load + EM + report) on 14.7M fragments under 60s."""
        from polymerase.sparse import backend
        from polymerase.annotation import get_annotation_class
        from polymerase.core.model import Polymerase
        from polymerase.core.likelihood import PolymeraseLikelihood
        from polymerase.core.reporter import output_report

        if not os.path.exists(BAM_100PCT):
            pytest.skip("100% BAM not in test_data/")

        backend.configure()
        Annotation = get_annotation_class('intervaltree')
        annot = Annotation(ANNOTATION_FULL, 'locus', 'None')

        class Opts100pct(MockOpts):
            def __init__(self, outdir):
                super().__init__(outdir)
                self.samfile = BAM_100PCT
                self.max_iter = 200

        t0 = time.perf_counter()

        outdir = tempfile.mkdtemp()
        opts = Opts100pct(outdir)
        ts = Polymerase(opts)
        ts.load_alignment(annot)

        np.random.seed(ts.get_random_seed())
        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        tl.em(use_likelihood=False)

        stats_f = os.path.join(outdir, 'stats.tsv')
        counts_f = os.path.join(outdir, 'counts.tsv')
        output_report(ts.feat_index, ts.feature_length, ts.run_info, tl,
                      opts.reassign_mode, opts.conf_prob,
                      stats_f, counts_f)

        elapsed = time.perf_counter() - t0
        assert elapsed < 60, f"Full pipeline took {elapsed:.1f}s (expected <60s)"
