# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

""" Polymerase assign

"""
import sys
import os
from time import time
import logging as lg
import gc
import tempfile
import atexit
import shutil

import numpy as np

from . import SubcommandOptions, configure_logging
from ..utils.helpers import format_minutes as fmtmins
from ..core.model import Polymerase, scPolymerase

# Default bundled GTF annotation (retro.hg38 from telescope_annotation_db)
_DEFAULT_GTF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'retro.hg38.gtf',
)
from ..annotation import get_annotation_class
from ..sparse import backend


class IDOptions(SubcommandOptions):

    def __init__(self, args, sc):
        super().__init__(args)
        self.sc = sc
        if self.logfile is None:
            self.logfile = sys.stderr

        # Default to bundled retro.hg38 annotation if no GTF specified
        if getattr(self, 'gtffile', None) is None:
            self.gtffile = _DEFAULT_GTF
            lg.info('Using bundled annotation: %s', os.path.basename(self.gtffile))

        if hasattr(self, 'tempdir') and self.tempdir is None:
            if hasattr(self, 'ncpu') and self.ncpu > 1:
                self.tempdir = tempfile.mkdtemp()
                atexit.register(shutil.rmtree, self.tempdir)

    def outfile_path(self, suffix):
        basename = '%s-%s' % (self.exp_tag, suffix)
        return os.path.join(self.outdir, basename)

class BulkIDOptions(IDOptions):

    OPTS = """
    - Input Options:
        - samfile:
            positional: True
            help: Path to alignment file. Alignment file can be in SAM or BAM
                  format. File must be collated so that all alignments for a
                  read pair appear sequentially in the file.
        - gtffile:
            positional: True
            nargs: "?"
            help: Path to annotation file (GTF format). If omitted, uses the
                  bundled retro.hg38 TE annotation.
        - attribute:
            default: locus
            help: GTF attribute that defines a transposable element locus. GTF
                  features that share the same value for --attribute will be
                  considered as part of the same locus.
        - no_feature_key:
            default: __no_feature
            help: Used internally to represent alignments. Must be different
                  from all other feature names.
        - ncpu:
            default: 1
            type: int
            help: Number of cores to use. (Multiple cores not supported yet).
        - tempdir:
            help: Path to temporary directory. Temporary files will be stored
                  here. Default uses python tempfile package to create the
                  temporary directory.
    - Reporting Options:
        - quiet:
            action: store_true
            help: Silence (most) output.
        - verbose:
            action: store_true
            help: Show detailed progress including EM iterations and timing.
        - debug:
            action: store_true
            help: Print debug messages.
        - logfile:
            type: argparse.FileType('r')
            help: Log output to this file.
        - outdir:
            default: .
            help: Output directory.
        - exp_tag:
            default: polymerase
            help: Experiment tag
        - updated_sam:
            action: store_true
            help: Generate an updated alignment file.
    - Run Modes:
        - reassign_mode:
            default: exclude
            choices:
                - exclude
                - choose
                - average
                - conf
                - unique
            help: >
                  Reassignment mode. After EM is complete, each fragment is
                  reassigned according to the expected value of its membership
                  weights. The reassignment method is the method for resolving
                  the "best" reassignment for fragments that have multiple
                  possible reassignments.
                  Available modes are: "exclude" - fragments with multiple best
                  assignments are excluded from the final counts; "choose" -
                  the best assignment is randomly chosen from among the set of
                  best assignments; "average" - the fragment is divided evenly
                  among the best assignments; "conf" - only assignments that
                  exceed a certain threshold (see --conf_prob) are accepted;
                  "unique" - only uniquely aligned reads are included.
                  NOTE: Results using all assignment modes are included in the
                  statistics report by default. This argument determines what
                  mode will be used for the outputted counts file.
        - conf_prob:
            type: float
            default: 0.9
            help: Minimum probability for high confidence assignment.
        - overlap_mode:
            default: threshold
            choices:
                - threshold
                - intersection-strict
                - union
            help: Overlap mode. The method used to determine whether a fragment
                  overlaps feature.
        - overlap_threshold:
            type: float
            default: 0.2
            help: Fraction of fragment that must be contained within a feature
                  to be assigned to that locus. Ignored if --overlap_method is
                  not "threshold".
        - annotation_class:
            default: intervaltree
            choices:
                - intervaltree
                - htseq
            help: Annotation class to use for finding overlaps. Both htseq and
                  intervaltree appear to yield identical results. Performance
                  differences are TBD.
        - stranded_mode:
            type: str
            default: None
            choices:
                - None
                - RF
                - R
                - FR
                - F
            help: Options for considering feature strand when assigning reads. 
                  If None, for each feature in the annotation, returns counts for the positive strand and negative strand. 
                  If not None, specifies the orientation of paired end reads (RF - read 1 reverse strand, read 2 forward strand) and
                  single end reads (F - forward strand). 
    - Model Parameters:
        - pi_prior:
            type: int
            default: 0
            help: Prior on π. Equivalent to adding n unique reads.
        - theta_prior:
            type: int
            default: 200000
            help: >
                  Prior on θ. Equivalent to adding n non-unique reads. NOTE: It
                  is recommended to set this prior to a large value. This
                  increases the penalty for non-unique reads and improves
                  accuracy.
        - em_epsilon:
            type: float
            default: 1e-7
            help: EM Algorithm Epsilon cutoff
        - max_iter:
            type: int
            default: 100
            help: EM Algorithm maximum iterations
        - use_likelihood:
            action: store_true
            help: Use difference in log-likelihood as convergence criteria.
        - skip_em:
            action: store_true
            help: Exits after loading alignment and saving checkpoint file.
    - Performance Options:
        - parallel_blocks:
            action: store_true
            help: Decompose into independent blocks and solve in parallel.
    """

    old_opts = """
    - bootstrap:
        hide: True
        type: int
        help: Set to an integer > 0 to turn on bootstrapping. Number of
              bootstrap replicates to perform.
    - bootstrap_ci:
        hide: True
        type: float
        default: 0.95
        help: Size of bootstrap confidence interval
    - out_matrix:
        action: store_true
        help: Output alignment matrix
    """

class scIDOptions(IDOptions):

    OPTS = """
    - Input Options:
        - samfile:
            positional: True
            help: Path to alignment file. Alignment file can be in SAM or BAM
                  format. File must be collated so that all alignments for a
                  read pair appear sequentially in the file.
        - gtffile:
            positional: True
            nargs: "?"
            help: Path to annotation file (GTF format). If omitted, uses the
                  bundled retro.hg38 TE annotation.
        - barcode_tag:
            type: str
            default: CB
            help: Name of the field in the BAM/SAM file containing the barcode for each read.
        - attribute:
            default: locus
            help: GTF attribute that defines a transposable element locus. GTF
                  features that share the same value for --attribute will be
                  considered as part of the same locus.
        - no_feature_key:
            default: __no_feature
            help: Used internally to represent alignments. Must be different
                  from all other feature names.
        - ncpu:
            default: 1
            type: int
            help: Number of cores to use. (Multiple cores not supported yet).
        - tempdir:
            help: Path to temporary directory. Temporary files will be stored
                  here. Default uses python tempfile package to create the
                  temporary directory.
    - Reporting Options:
        - quiet:
            action: store_true
            help: Silence (most) output.
        - verbose:
            action: store_true
            help: Show detailed progress including EM iterations and timing.
        - debug:
            action: store_true
            help: Print debug messages.
        - logfile:
            type: argparse.FileType('r')
            help: Log output to this file.
        - outdir:
            default: .
            help: Output directory.
        - exp_tag:
            default: polymerase
            help: Experiment tag
        - updated_sam:
            action: store_true
            help: Generate an updated alignment file.
    - Run Modes:
        - reassign_mode:
            default: exclude
            choices:
                - all
                - exclude
                - choose
                - average
                - conf
                - unique
            help: >
                  Reassignment mode. After EM is complete, each fragment is
                  reassigned according to the expected value of its membership
                  weights. The reassignment method is the method for resolving
                  the "best" reassignment for fragments that have multiple
                  possible reassignments.
                  Available modes are: "exclude" - fragments with multiple best
                  assignments are excluded from the final counts; "choose" -
                  the best assignment is randomly chosen from among the set of
                  best assignments; "average" - the fragment is divided evenly
                  among the best assignments; "conf" - only assignments that
                  exceed a certain threshold (see --conf_prob) are accepted;
                  "unique" - only uniquely aligned reads are included.
                  NOTE: Results using all assignment modes are included in the
                  statistics report by default. This argument determines what
                  mode will be used for the outputted counts file.
        - use_every_reassign_mode:
            action: store_true
            help: Whether to output count matrices generated using every reassign mode. 
                  If specified, six output count matrices will be generated, 
                  corresponding to the six possible reassignment methods (all, exclude, 
                  choose, average, conf, unique). 
        - conf_prob:
            type: float
            default: 0.9
            help: Minimum probability for high confidence assignment.
        - overlap_mode:
            default: threshold
            choices:
                - threshold
                - intersection-strict
                - union
            help: Overlap mode. The method used to determine whether a fragment
                  overlaps feature.
        - overlap_threshold:
            type: float
            default: 0.2
            help: Fraction of fragment that must be contained within a feature
                  to be assigned to that locus. Ignored if --overlap_method is
                  not "threshold".
        - annotation_class:
            default: intervaltree
            choices:
                - intervaltree
                - htseq
            help: Annotation class to use for finding overlaps. Both htseq and
                  intervaltree appear to yield identical results. Performance
                  differences are TBD.
        - stranded_mode:
            type: str
            default: None
            choices:
                - None
                - RF
                - R
                - FR
                - F
            help: Options for considering feature strand when assigning reads. 
                  If None, for each feature in the annotation, returns counts for the positive strand and negative strand. 
                  If not None, specifies the orientation of paired end reads (RF - read 1 reverse strand, read 2 forward strand) and
                  single end reads (F - forward strand). 
    - Model Parameters:
        - pi_prior:
            type: int
            default: 0
            help: Prior on π. Equivalent to adding n unique reads.
        - theta_prior:
            type: int
            default: 200000
            help: >
                  Prior on θ. Equivalent to adding n non-unique reads. NOTE: It
                  is recommended to set this prior to a large value. This
                  increases the penalty for non-unique reads and improves
                  accuracy.
        - em_epsilon:
            type: float
            default: 1e-7
            help: EM Algorithm Epsilon cutoff
        - max_iter:
            type: int
            default: 100
            help: EM Algorithm maximum iterations
        - use_likelihood:
            action: store_true
            help: Use difference in log-likelihood as convergence criteria.
        - skip_em:
            action: store_true
            help: Exits after loading alignment and saving checkpoint file.
    - Performance Options:
        - parallel_blocks:
            action: store_true
            help: Decompose into independent blocks and solve in parallel.
    """

    old_opts = """
    - bootstrap:
        hide: True
        type: int
        help: Set to an integer > 0 to turn on bootstrapping. Number of
              bootstrap replicates to perform.
    - bootstrap_ci:
        hide: True
        type: float
        default: 0.95
        help: Size of bootstrap confidence interval
    - out_matrix:
        action: store_true
        help: Output alignment matrix
    """


def _parse_primers_arg(opts):
    """Parse --primers CLI arg into a list of primer names, or None for all."""
    raw = getattr(opts, 'primers', None)
    if raw is None or raw == 'all':
        return None
    return [p.strip() for p in raw.split(',') if p.strip()]


def _backend_display_name(be):
    """Human-readable backend name for console output."""
    names = {
        'gpu': 'GPU (CuPy)',
        'cpu_optimized': 'CPU-Optimized (Numba)',
        'cpu_stock': 'CPU (scipy)',
    }
    return names.get(be.name, be.name)


def run(args, sc=True):
    """Platform orchestrator: loads data, builds snapshots, delegates to primers.

    Args:
        args: Parsed argparse namespace.
        sc: True for single-cell mode.
    """
    from ..plugins.snapshots import AnnotationSnapshot, AlignmentSnapshot
    from ..plugins.registry import PluginRegistry
    from ..compute import get_ops

    option_class = scIDOptions if sc else BulkIDOptions
    opts = option_class(args, sc=sc)
    console = configure_logging(opts)
    backend.configure()
    lg.info('\n{}\n'.format(opts))
    total_time = time()

    # Banner
    console.banner(opts.version)

    # Input section
    _be = backend.get_backend()
    _bam_name = os.path.basename(opts.samfile)
    _gtf_name = os.path.basename(opts.gtffile)
    _using_bundled = (os.path.abspath(opts.gtffile) == os.path.abspath(_DEFAULT_GTF))

    console.section('Input')
    console.item('BAM', _bam_name)
    if _using_bundled:
        console.item('Annotation', '{} [bundled]'.format(_gtf_name))
    else:
        console.item('Annotation', _gtf_name)
    console.item('Backend', _backend_display_name(_be))
    console.blank()

    # --- Platform: Load Data ---
    ts = scPolymerase(opts) if sc else Polymerase(opts)

    Annotation = get_annotation_class(opts.annotation_class)
    lg.info('Loading annotation...')
    stime = time()
    annot = Annotation(opts.gtffile, opts.attribute, opts.stranded_mode)
    _annot_elapsed = time() - stime
    lg.info("Loaded annotation in {}".format(fmtmins(_annot_elapsed)))
    lg.info('Loaded {} features.'.format(len(annot.loci)))
    console.verbose('Loaded annotation: {:,} features ({:.1f}s)'.format(
        len(annot.loci), _annot_elapsed))

    lg.info('Loading alignments...')
    stime = time()
    ts.load_alignment(annot)
    _aln_elapsed = time() - stime
    lg.info("Loaded alignment in {}".format(fmtmins(_aln_elapsed)))

    ts.print_summary(lg.INFO)

    _ri = ts.run_info
    _total = int(_ri.get('total_fragments', 0))
    _unique = int(_ri.get('overlap_unique', 0))
    _ambig = int(_ri.get('overlap_ambig', 0))
    _overlap = _unique + _ambig

    console.status('Loading alignment... done ({:.1f}s)'.format(_aln_elapsed))
    console.detail('{:,} fragments \u2014 {:,} overlap annotation ({:,} unique, {:,} ambiguous)'.format(
        _total, _overlap, _unique, _ambig))
    console.blank()

    if _overlap == 0:
        console.status('No alignments overlapping annotation.')
        lg.info("polymerase assign complete (%s)" % fmtmins(time() - total_time))
        return

    # --- Platform: Build Snapshots ---
    annot_snapshot = AnnotationSnapshot(
        loci=dict(annot.loci),
        feature_lengths=annot.feature_length(),
        num_features=len(annot.loci),
        gtf_path=opts.gtffile,
        attribute_name=opts.attribute,
    )

    annot = None  # free memory
    lg.debug('garbage: {:d}'.format(gc.collect()))

    ts.save(opts.outfile_path('checkpoint'))

    alignment_snapshot = AlignmentSnapshot(
        bam_path=opts.samfile,
        run_info=dict(ts.run_info),
        read_index=dict(ts.read_index),
        feat_index=dict(ts.feat_index),
        shape=ts.shape,
        raw_scores=ts.raw_scores,
        feature_lengths=dict(ts.feature_length),
    )

    # Stash tmp_bam path on opts so the assign primer can find it for updated_sam
    if hasattr(ts, 'tmp_bam'):
        opts._tmp_bam_path = ts.tmp_bam

    # --- Platform: Discover & Run Primers ---
    registry = PluginRegistry()
    registry.discover(active_primers=_parse_primers_arg(opts))
    registry.configure_all(opts, get_ops())

    # Show loaded plugins
    console.section('Primers')
    for p in registry.primers:
        console.plugin(p.name, p.description, p.version)
    console.blank()

    if registry.cofactors:
        console.section('Cofactors')
        for c in registry.cofactors:
            console.plugin(c.name, c.description)
        console.blank()

    registry.notify('on_annotation_loaded', annot_snapshot)
    registry.notify('on_matrix_built', alignment_snapshot)
    registry.commit_all(opts.outdir, opts.exp_tag, console=console)

    # Completion
    console.blank()
    console.status('Completed in {:.1f}s'.format(time() - total_time))
    console.blank()
    lg.info("polymerase assign complete (%s)" % fmtmins(time() - total_time))
