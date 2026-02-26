#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of Polymerase.
# Original Telescope code by Matthew L. Bendall (https://github.com/mlbendall/telescope)
#
# New code and modifications by Duane Storey (https://github.com/duanestorey) and Claude (Anthropic).
# Licensed under MIT License.

"""Multi-tier benchmark for Polymerase pipeline.

Runs the full pipeline with the active backend and prints a reference table
comparable to the stock baseline. Validates the golden log-likelihood.

Run:
    python -m polymerase.tests.benchmark_em [--repeat N] [--output FILE] [--backend stock|auto]
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

# Reference values from stock baseline
REFERENCE = {
    'annotation_load_s':  0.0038,
    'bam_load_s':         0.419,
    'em_init_s':          0.0012,
    'em_converge_s':      0.0397,
    'estep_ms':           0.71,
    'mstep_ms':           0.20,
    'lnl_calc_ms':        0.67,
    'memory_peak_mb':     7.65,
    'golden_lnl':         95252.596614,
}


# ---------------------------------------------------------------------------
# Mock options matching default CLI
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _fmt_speedup(current, reference):
    """Format speedup ratio."""
    if reference == 0 or current == 0:
        return ''
    ratio = reference / current
    if ratio >= 1.0:
        return f'{ratio:.2f}x faster'
    else:
        return f'{1/ratio:.2f}x slower'


def print_table(results, reference=REFERENCE):
    """Print a reference comparison table."""
    border = '+' + '-' * 29 + '+' + '-' * 15 + '+' + '-' * 15 + '+' + '-' * 18 + '+'
    header = '| {:27s} | {:13s} | {:13s} | {:16s} |'.format(
        'Stage', 'Current', 'Reference', 'Speedup')

    print()
    print(border)
    print(header)
    print(border)

    rows = [
        ('Annotation load',
         f"{results['annotation_s']*1000:.1f}ms",
         f"{reference['annotation_load_s']*1000:.1f}ms",
         _fmt_speedup(results['annotation_s'], reference['annotation_load_s'])),

        ('BAM load + matrix',
         f"{results['bam_load_s']*1000:.0f}ms",
         f"{reference['bam_load_s']*1000:.0f}ms",
         _fmt_speedup(results['bam_load_s'], reference['bam_load_s'])),

        ('EM init',
         f"{results['em_init_s']*1000:.1f}ms",
         f"{reference['em_init_s']*1000:.1f}ms",
         _fmt_speedup(results['em_init_s'], reference['em_init_s'])),

        (f"EM convergence ({results['em_iters']} iters)",
         f"{results['em_converge_s']*1000:.1f}ms",
         f"{reference['em_converge_s']*1000:.1f}ms",
         _fmt_speedup(results['em_converge_s'], reference['em_converge_s'])),

        ('E-step',
         f"{results['estep_ms']:.2f}ms/iter",
         f"{reference['estep_ms']:.2f}ms/iter",
         _fmt_speedup(results['estep_ms'], reference['estep_ms'])),

        ('M-step',
         f"{results['mstep_ms']:.2f}ms/iter",
         f"{reference['mstep_ms']:.2f}ms/iter",
         _fmt_speedup(results['mstep_ms'], reference['mstep_ms'])),

        ('Log-likelihood',
         f"{results['lnl_calc_ms']:.2f}ms/iter",
         f"{reference['lnl_calc_ms']:.2f}ms/iter",
         _fmt_speedup(results['lnl_calc_ms'], reference['lnl_calc_ms'])),

        ('Memory peak',
         f"{results['memory_peak_mb']:.2f} MB",
         f"{reference['memory_peak_mb']:.2f} MB",
         ''),
    ]

    for stage, current, ref, speedup in rows:
        print('| {:27s} | {:13s} | {:13s} | {:16s} |'.format(
            stage, current, ref, speedup))
        print(border)

    print()

    # Golden lnl check
    lnl = results['log_likelihood']
    expected = reference['golden_lnl']
    passed = abs(lnl - expected) / abs(expected) < 1e-4
    status = 'PASS' if passed else 'FAIL'
    print(f'Golden log-likelihood: {lnl:.6f}  (expected {expected:.6f})  [{status}]')
    print()
    return passed


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------

def run_benchmark(backend_mode='auto', repeat=3):
    """Run full pipeline benchmark and collect timing data."""
    from polymerase.sparse import backend

    if backend_mode == 'stock':
        # Force stock backend
        backend._active_backend = backend.BackendInfo()
    else:
        backend.configure()

    active = backend.get_backend()

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
    )
    samfile = os.path.join(data_dir, 'alignment.bam')
    gtffile = os.path.join(data_dir, 'annotation.gtf')

    if not os.path.exists(samfile) or not os.path.exists(gtffile):
        print("ERROR: Test data not found in", data_dir)
        sys.exit(1)

    outdir = tempfile.mkdtemp()
    opts = BenchOpts(samfile, gtffile, outdir)

    from polymerase.annotation import get_annotation_class
    from polymerase.core.model import Polymerase
    from polymerase.core.likelihood import PolymeraseLikelihood

    results = OrderedDict()
    results['backend'] = active.name

    import scipy
    results['versions'] = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'numba': 'n/a',
    }
    if active.has_numba:
        import numba
        results['versions']['numba'] = numba.__version__

    print('=' * 80)
    print(f'Polymerase Benchmark — Backend: {active.name}')
    print(f'  numba={active.has_numba}  mkl={active.has_mkl}')
    print(f'  Data: {samfile}')
    print(f'  Repeat: {repeat}x per stage')
    print('=' * 80)

    # --- 1. Annotation ---
    Annotation = get_annotation_class('intervaltree')
    annot_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        annot = Annotation(gtffile, 'locus', 'None')
        annot_times.append(time.perf_counter() - t0)
    results['annotation_s'] = np.median(annot_times)

    # --- 2. BAM load ---
    bam_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        ts = Polymerase(opts)
        ts.load_alignment(annot)
        bam_times.append(time.perf_counter() - t0)
    results['bam_load_s'] = np.median(bam_times)
    results['matrix_shape'] = list(ts.shape)
    results['nnz'] = int(ts.raw_scores.nnz)

    # --- 3. EM init + convergence ---
    init_times = []
    em_times = []
    em_iters_list = []
    lnl_values = []

    for _ in range(repeat):
        seed = ts.get_random_seed()
        np.random.seed(seed)

        t0 = time.perf_counter()
        tl = PolymeraseLikelihood(ts.raw_scores, opts)
        init_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tl.em(use_likelihood=True)
        em_times.append(time.perf_counter() - t0)

        lnl_values.append(float(tl.lnl))

    results['em_init_s'] = np.median(init_times)
    results['em_converge_s'] = np.median(em_times)
    results['log_likelihood'] = lnl_values[-1]

    # Count iterations by running once more
    seed = ts.get_random_seed()
    np.random.seed(seed)
    tl2 = PolymeraseLikelihood(ts.raw_scores, opts)
    inum = 0
    pi, theta = tl2.pi.copy(), tl2.theta.copy()
    converged = False
    while not converged and inum < opts.max_iter:
        z = tl2.estep(pi, theta)
        pi_new, theta_new = tl2.mstep(z)
        inum += 1
        if opts.use_likelihood:
            lnl_new = tl2.calculate_lnl(z, pi_new, theta_new)
            converged = inum > 1 and abs(lnl_new - tl2.lnl) < opts.em_epsilon
            tl2.lnl = lnl_new
        else:
            converged = abs(pi_new - pi).sum() < opts.em_epsilon
        pi, theta = pi_new, theta_new
    results['em_iters'] = inum

    # --- 4. E-step / M-step detail ---
    seed = ts.get_random_seed()
    np.random.seed(seed)
    tl3 = PolymeraseLikelihood(ts.raw_scores, opts)
    n_detail = 10
    estep_times = []
    mstep_times = []
    lnl_times = []
    pi, theta = tl3.pi.copy(), tl3.theta.copy()
    for _ in range(n_detail):
        t0 = time.perf_counter()
        z = tl3.estep(pi, theta)
        estep_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pi, theta = tl3.mstep(z)
        mstep_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tl3.calculate_lnl(z, pi, theta)
        lnl_times.append(time.perf_counter() - t0)

    results['estep_ms'] = np.median(estep_times) * 1000
    results['mstep_ms'] = np.median(mstep_times) * 1000
    results['lnl_calc_ms'] = np.median(lnl_times) * 1000

    # --- 5. Memory ---
    tracemalloc.start()
    seed = ts.get_random_seed()
    np.random.seed(seed)
    ts_mem = Polymerase(opts)
    ts_mem.load_alignment(annot)
    tl_mem = PolymeraseLikelihood(ts_mem.raw_scores, opts)
    tl_mem.em(use_likelihood=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['memory_peak_mb'] = peak / (1024 * 1024)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Polymerase multi-tier benchmark')
    parser.add_argument('--repeat', type=int, default=3,
                        help='Repetitions per stage (default: 3)')
    parser.add_argument('--backend', choices=['stock', 'auto'], default='auto',
                        help='Backend: stock (scipy only) or auto-detect (default: auto)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    results = run_benchmark(backend_mode=args.backend, repeat=args.repeat)
    passed = print_table(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'Results saved to {args.output}')

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
