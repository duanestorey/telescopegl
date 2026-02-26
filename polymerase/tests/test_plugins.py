# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Tests for polymerase.plugins module — ABCs, snapshots, registry, builtins."""

import os
import tempfile

import pytest
import numpy as np
import scipy.sparse

from polymerase.plugins.abc import Primer, Cofactor
from polymerase.plugins.snapshots import AnnotationSnapshot, AlignmentSnapshot
from polymerase.plugins.registry import PluginRegistry


# =========================================================================
# ABC enforcement
# =========================================================================

class TestPrimerABC:
    def test_cannot_instantiate_bare(self):
        """Primer is abstract — must implement name and commit."""
        with pytest.raises(TypeError):
            Primer()

    def test_concrete_primer_ok(self):
        class MyPrimer(Primer):
            @property
            def name(self):
                return "test"
            def commit(self, output_dir, exp_tag, console=None):
                pass
        p = MyPrimer()
        assert p.name == "test"
        assert p.version == "0.0.0"
        assert p.cli_options() is None

    def test_hooks_are_optional(self):
        class MinimalPrimer(Primer):
            @property
            def name(self):
                return "minimal"
            def commit(self, output_dir, exp_tag, console=None):
                pass
        p = MinimalPrimer()
        # Should not raise
        p.on_annotation_loaded(None)
        p.on_matrix_built(None)


class TestCofactorABC:
    def test_cannot_instantiate_bare(self):
        with pytest.raises(TypeError):
            Cofactor()

    def test_concrete_cofactor_ok(self):
        class MyCofactor(Cofactor):
            @property
            def name(self):
                return "test-cof"
            @property
            def primer_name(self):
                return "test"
            def transform(self, primer_output_dir, output_dir, exp_tag, console=None):
                pass
        c = MyCofactor()
        assert c.name == "test-cof"
        assert c.primer_name == "test"


# =========================================================================
# Snapshot immutability
# =========================================================================

class TestSnapshots:
    def test_annotation_snapshot_frozen(self):
        snap = AnnotationSnapshot(
            loci={'A': []}, feature_lengths={'A': 100},
            num_features=1, gtf_path='/tmp/test.gtf', attribute_name='locus',
        )
        with pytest.raises(AttributeError):
            snap.num_features = 42

    def test_alignment_snapshot_frozen(self):
        m = scipy.sparse.csr_matrix((3, 4))
        snap = AlignmentSnapshot(
            bam_path='/tmp/test.bam', run_info={'total_fragments': 100},
            read_index={'r1': 0}, feat_index={'f1': 0},
            shape=(3, 4), raw_scores=m, feature_lengths={'f1': 500},
        )
        with pytest.raises(AttributeError):
            snap.shape = (10, 20)

    def test_annotation_snapshot_fields(self):
        snap = AnnotationSnapshot(
            loci={'A': [1, 2]}, feature_lengths={'A': 100},
            num_features=1, gtf_path='/tmp/test.gtf', attribute_name='locus',
        )
        assert snap.num_features == 1
        assert snap.gtf_path == '/tmp/test.gtf'
        assert snap.attribute_name == 'locus'


# =========================================================================
# Registry
# =========================================================================

class TestRegistry:
    def test_discover_finds_assign(self):
        reg = PluginRegistry()
        reg.discover()
        names = [p.name for p in reg.primers]
        assert 'assign' in names

    def test_discover_with_filter(self):
        reg = PluginRegistry()
        reg.discover(active_primers=['assign'])
        assert len(reg.primers) == 1
        assert reg.primers[0].name == 'assign'

    def test_discover_unknown_primer_skipped(self):
        reg = PluginRegistry()
        reg.discover(active_primers=['nonexistent'])
        assert len(reg.primers) == 0

    def test_cofactors_loaded_for_active_primers(self):
        reg = PluginRegistry()
        reg.discover(active_primers=['assign'])
        cofactor_names = [c.name for c in reg.cofactors]
        assert 'family-agg' in cofactor_names
        assert 'normalize' in cofactor_names

    def test_cofactors_not_loaded_when_primer_inactive(self):
        reg = PluginRegistry()
        reg.discover(active_primers=['nonexistent'])
        assert len(reg.cofactors) == 0

    def test_list_available(self):
        reg = PluginRegistry()
        reg.discover()
        primers, cofactors = reg.list_available()
        assert 'assign' in primers
        assert primers['assign']['builtin'] is True

    def test_notify_calls_hooks(self):
        """Registry notify should call the named method on primers."""
        received = []

        class SpyPrimer(Primer):
            @property
            def name(self):
                return "spy"
            def on_matrix_built(self, snapshot):
                received.append(snapshot)
            def commit(self, output_dir, exp_tag, console=None):
                pass

        reg = PluginRegistry()
        reg._primers = [SpyPrimer()]
        reg.notify('on_matrix_built', 'test_data')
        assert received == ['test_data']

    def test_configure_all_passes_opts(self):
        """configure_all should call configure on each plugin."""
        class ReceiverPrimer(Primer):
            @property
            def name(self):
                return "receiver"
            def configure(self, opts, compute):
                self.received_opts = opts
                self.received_compute = compute
            def commit(self, output_dir, exp_tag, console=None):
                pass

        p = ReceiverPrimer()
        reg = PluginRegistry()
        reg._primers = [p]
        reg.configure_all('my_opts', 'my_compute')
        assert p.received_opts == 'my_opts'
        assert p.received_compute == 'my_compute'

    def test_commit_all_handles_failure_gracefully(self):
        """A failing primer should log a warning, not crash."""
        class FailPrimer(Primer):
            @property
            def name(self):
                return "fail"
            def commit(self, output_dir, exp_tag, console=None):
                raise RuntimeError("intentional failure")

        reg = PluginRegistry()
        reg._primers = [FailPrimer()]
        # Should not raise
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.commit_all(tmpdir, 'test')


# =========================================================================
# Built-in assign primer
# =========================================================================

class TestAssignPrimer:
    def test_assign_primer_name(self):
        from polymerase.plugins.builtin.assign import AssignPrimer
        p = AssignPrimer()
        assert p.name == "assign"
        assert p.version == "2.0.0"

    def test_assign_primer_produces_output(self):
        """Run assign primer on test data, verify output files."""
        from polymerase.plugins.builtin.assign import AssignPrimer
        from polymerase.plugins.snapshots import AlignmentSnapshot
        from polymerase.compute.cpu_ops import CpuOps
        from polymerase.sparse.backend import get_sparse_class, configure
        from polymerase.core.model import Polymerase

        configure()

        # Load test data
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bam_path = os.path.join(_base, 'data', 'alignment.bam')
        gtf_path = os.path.join(_base, 'data', 'annotation.gtf')

        with tempfile.TemporaryDirectory() as tmpdir:
            class MockOpts:
                samfile = bam_path
                gtffile = gtf_path
                attribute = 'locus'
                no_feature_key = '__no_feature'
                ncpu = 1
                overlap_mode = 'threshold'
                overlap_threshold = 0.2
                annotation_class = 'intervaltree'
                stranded_mode = 'None'
                updated_sam = False
                skip_em = False
                reassign_mode = 'exclude'
                conf_prob = 0.9
                pi_prior = 0
                theta_prior = 200000
                em_epsilon = 1e-7
                max_iter = 100
                use_likelihood = False
                parallel_blocks = False
                outdir = tmpdir
                exp_tag = 'test'
                quiet = True
                debug = False
                logfile = None
                version = '2.0.0'
                tempdir = None

                def outfile_path(self, suffix):
                    return os.path.join(tmpdir, 'test-' + suffix)

            opts = MockOpts()
            from polymerase.annotation import get_annotation_class
            Annotation = get_annotation_class('intervaltree')
            annot = Annotation(gtf_path, 'locus', 'None')

            ts = Polymerase(opts)
            ts.load_alignment(annot)

            snap = AlignmentSnapshot(
                bam_path=bam_path,
                run_info=dict(ts.run_info),
                read_index=dict(ts.read_index),
                feat_index=dict(ts.feat_index),
                shape=ts.shape,
                raw_scores=ts.raw_scores,
                feature_lengths=dict(ts.feature_length),
            )

            primer = AssignPrimer()
            primer.configure(opts, CpuOps())
            primer.on_matrix_built(snap)
            primer.commit(tmpdir, 'test')

            stats_file = os.path.join(tmpdir, 'test-run_stats.tsv')
            counts_file = os.path.join(tmpdir, 'test-TE_counts.tsv')
            assert os.path.exists(stats_file)
            assert os.path.exists(counts_file)
            assert os.path.getsize(stats_file) > 0
            assert os.path.getsize(counts_file) > 0


# =========================================================================
# Built-in cofactors
# =========================================================================

class TestFamilyAggCofactor:
    def test_name_and_primer(self):
        from polymerase.plugins.builtin.family_agg import FamilyAggCofactor
        c = FamilyAggCofactor()
        assert c.name == "family-agg"
        assert c.primer_name == "assign"

    def test_transform_produces_output(self):
        """Family-agg should produce family and class count files."""
        from polymerase.plugins.builtin.family_agg import FamilyAggCofactor

        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gtf_path = os.path.join(_base, 'data', 'annotation.gtf')

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock counts file
            counts_file = os.path.join(tmpdir, 'test-TE_counts.tsv')
            with open(counts_file, 'w') as fh:
                fh.write("transcript\tcount\n")
                fh.write("HML2_1p36.21a\t50\n")
                fh.write("HML2_1q22\t100\n")
                fh.write("__no_feature\t0\n")

            out_dir = os.path.join(tmpdir, 'family-agg')
            os.makedirs(out_dir)

            class MockOpts:
                gtffile = gtf_path
                attribute = 'locus'

            c = FamilyAggCofactor()
            c.configure(MockOpts(), None)
            c.transform(tmpdir, out_dir, 'test')

            family_f = os.path.join(out_dir, 'test-family_counts.tsv')
            class_f = os.path.join(out_dir, 'test-class_counts.tsv')
            assert os.path.exists(family_f)
            assert os.path.exists(class_f)


class TestNormalizeCofactor:
    def test_name_and_primer(self):
        from polymerase.plugins.builtin.normalize import NormalizeCofactor
        c = NormalizeCofactor()
        assert c.name == "normalize"
        assert c.primer_name == "assign"
