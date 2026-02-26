#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark Telescope pipeline on 1% subsampled BAM with full TE annotation.

Uses SRR9666161_1pct_collated.bam (136K fragments, 30MB) with retro.hg38.gtf
(28K features) for realistic benchmarking of all pipeline stages including
BAM loading, EM convergence, block decomposition, and reassignment.

Run:
    python -m telescope.tests.benchmark_1pct [--repeat N] [--backend stock|auto]
"""

import argparse
import json
import os
import sys
import tempfile
import time
import tracemalloc
from collections import OrderedDict

import numpy as np

__author__ = 'Duane Storey'

# Paths relative to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DATA = os.path.join(REPO_ROOT, 'test_data')
BUNDLED_DATA = os.path.join(REPO_ROOT, 'telescope', 'data')

BAM_1PCT = os.path.join(TEST_DATA, 'SRR9666161_1pct_collated.bam')
ANNOTATION_FULL = os.path.join(TEST_DATA, 'retro.hg38.gtf')
ANNOTATION_BUNDLED = os.path.join(BUNDLED_DATA, 'annotation.gtf')


class BenchOpts:
    def __init__(self, samfile, gtffile, outdir):
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
        self.outdir = outdir
        self.exp_tag = 'bench'
        self.reassign_mode = 'exclude'
        self.conf_prob = 0.9
        self.version = 'benchmark'
        self.skip_em = False
        self.use_likelihood = True
        self.em_epsilon = 1e-7
        self.max_iter = 100
        self.pi_prior = 0
        self.theta_prior = 200000

    def outfile_path(self, suffix):
        return os.path.join(self.outdir, '%s-%s' % (self.exp_tag, suffix))


def print_table(results, title='Benchmark Results'):
    """Print benchmark results table."""
    border = '+' + '-' * 35 + '+' + '-' * 18 + '+'
    header = '| {:33s} | {:16s} |'.format('Stage', 'Time')

    print()
    print('=' * 57)
    print(f'  {title}')
    print(f'  Backend: {results["backend"]}')
    print('=' * 57)
    print(border)
    print(header)
    print(border)

    rows = [
        ('Annotation load ({} loci)'.format(results.get('n_features', '?')),
         '{:.1f}ms'.format(results['annotation_s'] * 1000)),
        ('BAM load ({} frags)'.format(results.get('n_fragments', '?')),
         '{:.0f}ms'.format(results['bam_load_s'] * 1000)),
        ('Matrix build ({}x{}, {} nnz)'.format(
            *results.get('matrix_shape', ['?', '?']),
            results.get('nnz', '?')),
         '(included in BAM load)'),
        ('EM init',
         '{:.1f}ms'.format(results['em_init_s'] * 1000)),
        ('EM convergence ({} iters)'.format(results.get('em_iters', '?')),
         '{:.1f}ms'.format(results['em_converge_s'] * 1000)),
        ('  E-step',
         '{:.2f}ms/iter'.format(results['estep_ms'])),
        ('  M-step',
         '{:.2f}ms/iter'.format(results['mstep_ms'])),
        ('  Log-likelihood calc',
         '{:.2f}ms/iter'.format(results['lnl_calc_ms'])),
        ('Reassignment (7 modes)',
         '{:.1f}ms'.format(results['reassign_s'] * 1000)),
        ('Memory peak',
         '{:.1f} MB'.format(results['memory_peak_mb'])),
    ]

    if 'decompose_s' in results:
        rows.insert(4, (
            'Block decomposition ({} blocks)'.format(
                results.get('n_blocks', '?')),
            '{:.1f}ms'.format(results['decompose_s'] * 1000)))
        rows.insert(5, (
            'EM parallel ({} blocks)'.format(
                results.get('n_blocks', '?')),
            '{:.1f}ms'.format(results['em_parallel_s'] * 1000)))

    for stage, timing in rows:
        print('| {:33s} | {:16s} |'.format(stage, timing))
        print(border)

    total = results.get('total_s', 0)
    print()
    print(f'  Total pipeline: {total:.2f}s')
    print(f'  Log-likelihood: {results["log_likelihood"]:.6f}')
    if 'log_likelihood_parallel' in results:
        print(f'  Log-likelihood (parallel): {results["log_likelihood_parallel"]:.6f}')
    print()


def run_benchmark(backend_mode='auto', repeat=3):
    """Run full pipeline benchmark on 1% BAM data."""
    from telescope.utils import backend

    if backend_mode == 'stock':
        backend._active_backend = backend.BackendInfo()
    else:
        backend.configure()

    active = backend.get_backend()

    if not os.path.exists(BAM_1PCT):
        print(f"ERROR: BAM not found: {BAM_1PCT}")
        print("Place SRR9666161_1pct_collated.bam in test_data/")
        sys.exit(1)
    if not os.path.exists(ANNOTATION_FULL):
        print(f"ERROR: Annotation not found: {ANNOTATION_FULL}")
        print("Download: wget -O test_data/retro.hg38.gtf "
              "https://github.com/mlbendall/telescope_annotation_db/raw/master/"
              "builds/retro.hg38.v1/transcripts.gtf")
        sys.exit(1)

    outdir = tempfile.mkdtemp()
    opts = BenchOpts(BAM_1PCT, ANNOTATION_FULL, outdir)

    from telescope.utils.annotation import get_annotation_class
    from telescope.utils.model import Telescope, TelescopeLikelihood

    results = OrderedDict()
    results['backend'] = active.name

    print(f'Backend: {active.name} (numba={active.has_numba}, '
          f'mkl={active.has_mkl}, threadsafe={active.threadsafe_parallelism})')
    print(f'BAM: {BAM_1PCT}')
    print(f'Annotation: {ANNOTATION_FULL}')
    print(f'Repeat: {repeat}x')
    print()

    total_start = time.perf_counter()

    # --- 1. Annotation ---
    Annotation = get_annotation_class('intervaltree')
    annot_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        annot = Annotation(ANNOTATION_FULL, 'locus', 'None')
        annot_times.append(time.perf_counter() - t0)
    results['annotation_s'] = np.median(annot_times)
    results['n_features'] = len(annot.loci)

    # --- 2. BAM load ---
    bam_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        ts = Telescope(opts)
        ts.load_alignment(annot)
        bam_times.append(time.perf_counter() - t0)
    results['bam_load_s'] = np.median(bam_times)
    results['matrix_shape'] = list(ts.shape)
    results['nnz'] = int(ts.raw_scores.nnz)
    results['n_fragments'] = ts.run_info.get('total_fragments', '?')
    results['n_unique'] = ts.run_info.get('overlap_unique', 0)
    results['n_ambig'] = ts.run_info.get('overlap_ambig', 0)

    if results['n_unique'] + results['n_ambig'] == 0:
        print("WARNING: No fragments overlap annotation. Check BAM/GTF compatibility.")
        sys.exit(1)

    # --- 3. EM init + convergence ---
    init_times = []
    em_times = []
    lnl_values = []

    for _ in range(repeat):
        seed = ts.get_random_seed()
        np.random.seed(seed)

        t0 = time.perf_counter()
        tl = TelescopeLikelihood(ts.raw_scores, opts)
        init_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tl.em(use_likelihood=True)
        em_times.append(time.perf_counter() - t0)
        lnl_values.append(float(tl.lnl))

    results['em_init_s'] = np.median(init_times)
    results['em_converge_s'] = np.median(em_times)
    results['log_likelihood'] = lnl_values[-1]

    # Count iterations
    seed = ts.get_random_seed()
    np.random.seed(seed)
    tl_iter = TelescopeLikelihood(ts.raw_scores, opts)
    inum = 0
    pi, theta = tl_iter.pi.copy(), tl_iter.theta.copy()
    converged = False
    while not converged and inum < opts.max_iter:
        z = tl_iter.estep(pi, theta)
        pi_new, theta_new = tl_iter.mstep(z)
        inum += 1
        if opts.use_likelihood:
            lnl_new = tl_iter.calculate_lnl(z, pi_new, theta_new)
            converged = inum > 1 and abs(lnl_new - tl_iter.lnl) < opts.em_epsilon
            tl_iter.lnl = lnl_new
        else:
            converged = abs(pi_new - pi).sum() < opts.em_epsilon
        pi, theta = pi_new, theta_new
    results['em_iters'] = inum

    # --- 4. E-step / M-step detail ---
    seed = ts.get_random_seed()
    np.random.seed(seed)
    tl_detail = TelescopeLikelihood(ts.raw_scores, opts)
    n_detail = min(10, results['em_iters'])
    estep_times = []
    mstep_times = []
    lnl_times = []
    pi, theta = tl_detail.pi.copy(), tl_detail.theta.copy()
    for _ in range(n_detail):
        t0 = time.perf_counter()
        z = tl_detail.estep(pi, theta)
        estep_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pi, theta = tl_detail.mstep(z)
        mstep_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tl_detail.calculate_lnl(z, pi, theta)
        lnl_times.append(time.perf_counter() - t0)

    results['estep_ms'] = np.median(estep_times) * 1000
    results['mstep_ms'] = np.median(mstep_times) * 1000
    results['lnl_calc_ms'] = np.median(lnl_times) * 1000

    # --- 5. Block decomposition + parallel EM ---
    from telescope.utils.decompose import find_blocks, split_matrix

    decomp_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        n_components, labels = find_blocks(ts.raw_scores)
        decomp_times.append(time.perf_counter() - t0)
    results['decompose_s'] = np.median(decomp_times)
    results['n_blocks'] = n_components

    # Parallel EM (sequential since not threadsafe, but exercises the path)
    parallel_times = []
    parallel_lnls = []
    for _ in range(repeat):
        seed = ts.get_random_seed()
        np.random.seed(seed)
        tl_par = TelescopeLikelihood(ts.raw_scores, opts)
        t0 = time.perf_counter()
        tl_par.em_parallel(use_likelihood=True)
        parallel_times.append(time.perf_counter() - t0)
        parallel_lnls.append(float(tl_par.lnl))
    results['em_parallel_s'] = np.median(parallel_times)
    results['log_likelihood_parallel'] = parallel_lnls[-1]

    # --- 6. Reassignment ---
    seed = ts.get_random_seed()
    np.random.seed(seed)
    tl_reassign = TelescopeLikelihood(ts.raw_scores, opts)
    tl_reassign.em(use_likelihood=True)

    reassign_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for mode in ('all', 'exclude', 'choose', 'average', 'conf', 'unique'):
            tl_reassign.reassign(mode, 0.9)
        reassign_times.append(time.perf_counter() - t0)
    # Add the final_conf with initial=True
    t0 = time.perf_counter()
    tl_reassign.reassign('conf', 0.9, True)
    reassign_times[-1] += time.perf_counter() - t0
    results['reassign_s'] = np.median(reassign_times)

    # --- 7. Memory ---
    tracemalloc.start()
    seed = ts.get_random_seed()
    np.random.seed(seed)
    ts_mem = Telescope(opts)
    ts_mem.load_alignment(annot)
    tl_mem = TelescopeLikelihood(ts_mem.raw_scores, opts)
    tl_mem.em(use_likelihood=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['memory_peak_mb'] = peak / (1024 * 1024)

    results['total_s'] = time.perf_counter() - total_start

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Telescope benchmark on 1%% subsampled BAM')
    parser.add_argument('--repeat', type=int, default=3,
                        help='Repetitions per stage (default: 3)')
    parser.add_argument('--backend', choices=['stock', 'auto'], default='auto',
                        help='Backend mode (default: auto)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    results = run_benchmark(backend_mode=args.backend, repeat=args.repeat)
    print_table(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
