# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

""" Polymerase resume

"""
import sys
import os
from time import time
import logging as lg
import gc

import numpy as np

from . import configure_logging
from ..utils.helpers import format_minutes as fmtmins

from ..core.model import Polymerase, scPolymerase
from .assign import IDOptions
from ..sparse import backend

class BulkResumeOptions(IDOptions):
    OPTS = """
    - Input Options:
        - checkpoint:
            positional: True
            help: Path to checkpoint file.
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
                  Polymerase report by default. This argument determines what
                  mode will be used for the "final counts" column.
        - conf_prob:
            type: float
            default: 0.9
            help: Minimum probability for high confidence assignment.
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
    - Performance Options:
        - parallel_blocks:
            action: store_true
            help: Decompose into independent blocks and solve in parallel.
    """

class scResumeOptions(IDOptions):

    OPTS = """
    - Input Options:
        - checkpoint:
            positional: True
            help: Path to checkpoint file.
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
                  Polymerase report by default. This argument determines what
                  mode will be used for the "final counts" column.
        - conf_prob:
            type: float
            default: 0.9
            help: Minimum probability for high confidence assignment.
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
    - Performance Options:
        - parallel_blocks:
            action: store_true
            help: Decompose into independent blocks and solve in parallel.
    """

def run(args, sc=True):
    """Resume from checkpoint: load saved matrix and run EM via primers.

    Args:
        args: Parsed argparse namespace.
        sc: True for single-cell mode.
    """
    from ..plugins.snapshots import AlignmentSnapshot
    from ..plugins.registry import PluginRegistry
    from ..compute import get_ops

    option_class = scResumeOptions if sc else BulkResumeOptions
    opts = option_class(args, sc=sc)
    configure_logging(opts)
    backend.configure()
    lg.info('\n{}\n'.format(opts))
    total_time = time()

    # Load checkpoint
    lg.info('Loading Polymerase object from file...')
    Polymerase_class = scPolymerase if sc else Polymerase
    ts = Polymerase_class.load(opts.checkpoint)
    ts.opts = opts

    ts.print_summary(lg.INFO)

    # Build alignment snapshot from checkpoint
    alignment_snapshot = AlignmentSnapshot(
        bam_path='',  # not available from checkpoint
        run_info=dict(ts.run_info),
        read_index=dict(ts.read_index),
        feat_index=dict(ts.feat_index),
        shape=ts.shape,
        raw_scores=ts.raw_scores,
        feature_lengths=dict(ts.feature_length),
    )

    # Discover & run primers (assign only for resume)
    registry = PluginRegistry()
    registry.discover(active_primers=['assign'])
    registry.configure_all(opts, get_ops())

    registry.notify('on_matrix_built', alignment_snapshot)
    registry.commit_all(opts.outdir, opts.exp_tag)

    lg.info("polymerase resume complete (%s)" % fmtmins(time() - total_time))
