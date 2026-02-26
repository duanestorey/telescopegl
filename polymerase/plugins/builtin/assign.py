# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""The assign primer — TE quantification via EM algorithm.

This is the original Telescope/Polymerase pipeline refactored as a primer.
It receives annotation and alignment snapshots from the platform, runs the
EM algorithm, and writes stats + counts reports.
"""
import os
import logging as lg
from time import time

import numpy as np

from ..abc import Primer
from ...utils.helpers import format_minutes as fmtmins


class AssignPrimer(Primer):
    """Quantify TE expression using Expectation-Maximization."""

    @property
    def name(self) -> str:
        return "assign"

    @property
    def description(self) -> str:
        return "Quantify TE expression using Expectation-Maximization"

    @property
    def version(self) -> str:
        return "2.0.0"

    def configure(self, opts, compute) -> None:
        self._opts = opts
        self._compute = compute
        self._annotation = None
        self._alignment = None

    def on_annotation_loaded(self, snapshot) -> None:
        self._annotation = snapshot

    def on_matrix_built(self, snapshot) -> None:
        self._alignment = snapshot

    def commit(self, output_dir, exp_tag, console=None) -> None:
        from ...core.likelihood import PolymeraseLikelihood
        from ...core.reporter import output_report, update_sam

        if self._alignment is None:
            lg.warning("assign primer: no alignment data, skipping")
            return

        opts = self._opts

        if getattr(opts, 'skip_em', False):
            lg.info("Skipping EM...")
            if console:
                console.detail('Skipping EM (--skip_em)')
            return

        # Seed RNG (identical to original pipeline)
        shape = self._alignment.shape
        total_frags = self._alignment.run_info['total_fragments']
        seed = (total_frags % (shape[0] * shape[1])) % 4294967295
        lg.debug("Random seed: {}".format(seed))
        np.random.seed(seed)

        # Create likelihood model
        ts_model = PolymeraseLikelihood(self._alignment.raw_scores, opts)

        # Run EM
        lg.info('Running Expectation-Maximization...')
        stime = time()
        if getattr(opts, 'parallel_blocks', False):
            ts_model.em_parallel(use_likelihood=opts.use_likelihood, loglev=lg.INFO)
        else:
            ts_model.em(use_likelihood=opts.use_likelihood, loglev=lg.INFO)
        _em_elapsed = time() - stime
        lg.info("EM completed in %s" % fmtmins(_em_elapsed))

        if console:
            _con = 'converged' if ts_model.num_iterations < opts.max_iter else 'terminated'
            console.detail('EM {} after {} iterations ({:.2f}s)'.format(
                _con, ts_model.num_iterations, _em_elapsed))
            console.detail('Final log-likelihood: {:,.2f}'.format(ts_model.lnl))

        # Generate report
        lg.info("Generating Report...")
        if console:
            console.detail('Generating report... done')
        stats_file = os.path.join(output_dir, '%s-run_stats.tsv' % exp_tag)
        counts_file = os.path.join(output_dir, '%s-TE_counts.tsv' % exp_tag)

        output_report(
            self._alignment.feat_index,
            self._alignment.feature_lengths,
            self._alignment.run_info,
            ts_model,
            opts.reassign_mode,
            opts.conf_prob,
            stats_file,
            counts_file,
        )

        # Updated SAM (if requested and tmp_bam is available)
        if getattr(opts, 'updated_sam', False):
            tmp_bam = getattr(opts, '_tmp_bam_path', None)
            if tmp_bam and os.path.exists(tmp_bam):
                lg.info("Creating updated SAM file...")
                updated_file = os.path.join(output_dir, '%s-updated.bam' % exp_tag)
                update_sam(
                    tmp_bam,
                    self._alignment.read_index,
                    self._alignment.feat_index,
                    self._alignment.run_info,
                    opts,
                    ts_model,
                    updated_file,
                )
