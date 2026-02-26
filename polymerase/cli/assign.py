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
from ..annotation import get_annotation_class
from ..sparse import backend


class IDOptions(SubcommandOptions):

    def __init__(self, args, sc):
        super().__init__(args)
        self.sc = sc
        if self.logfile is None:
            self.logfile = sys.stderr

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
            help: Path to annotation file (GTF format)
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
            help: Path to annotation file (GTF format)
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
    configure_logging(opts)
    backend.configure()
    lg.info('\n{}\n'.format(opts))
    total_time = time()

    # --- Platform: Load Data ---
    ts = scPolymerase(opts) if sc else Polymerase(opts)

    Annotation = get_annotation_class(opts.annotation_class)
    lg.info('Loading annotation...')
    stime = time()
    annot = Annotation(opts.gtffile, opts.attribute, opts.stranded_mode)
    lg.info("Loaded annotation in {}".format(fmtmins(time() - stime)))
    lg.info('Loaded {} features.'.format(len(annot.loci)))

    lg.info('Loading alignments...')
    stime = time()
    ts.load_alignment(annot)
    lg.info("Loaded alignment in {}".format(fmtmins(time() - stime)))

    ts.print_summary(lg.INFO)

    if ts.run_info['overlap_unique'] + ts.run_info['overlap_ambig'] == 0:
        lg.info("No alignments overlapping annotation")
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

    registry.notify('on_annotation_loaded', annot_snapshot)
    registry.notify('on_matrix_built', alignment_snapshot)
    registry.commit_all(opts.outdir, opts.exp_tag)

    lg.info("polymerase assign complete (%s)" % fmtmins(time() - total_time))
