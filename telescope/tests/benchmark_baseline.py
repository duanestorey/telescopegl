#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline benchmark for Telescope pipeline (stock CPU / scipy).

Measures wall-clock time and peak memory for each pipeline stage:
  1. Annotation loading (GTF → IntervalTree)
  2. BAM loading + overlap assignment → sparse matrix
  3. EM algorithm (E-step, M-step, convergence)
  4. Reassignment (all 6 modes)

Run:
    python -m telescope.tests.benchmark_baseline [--repeat N] [--output FILE]

Results are printed as a table and optionally saved to JSON.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import tracemalloc
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np

# ---------------------------------------------------------------------------
# Timing / memory helpers
# ---------------------------------------------------------------------------

@contextmanager
def timed_block(label, results):
    """Context manager that records wall-clock seconds into results[label]."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    results[label] = elapsed


def measure_peak_memory(func, *args, **kwargs):
    """Run func and return (result, peak_memory_MB)."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 * 1024)


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
        # EM parameters
        self.em_epsilon = 1e-7
        self.max_iter = 100
        self.pi_prior = 0
        self.theta_prior = 200000

    def outfile_path(self, suffix):
        return os.path.join(self.outdir, '%s-%s' % (self.exp_tag, suffix))


# ---------------------------------------------------------------------------
# Benchmark stages
# ---------------------------------------------------------------------------

def benchmark_annotation(gtffile, repeat=1):
    """Benchmark GTF annotation loading."""
    from telescope.utils.annotation import get_annotation_class
    Annotation = get_annotation_class('intervaltree')

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        annot = Annotation(gtffile, 'locus', 'None')
        times.append(time.perf_counter() - start)

    n_loci = len(annot.loci)
    n_chroms = len(annot.itree)
    n_intervals = sum(len(t) for t in annot.itree.values())

    return {
        'stage': 'annotation_load',
        'times': times,
        'mean_s': np.mean(times),
        'std_s': np.std(times),
        'n_loci': n_loci,
        'n_chroms': n_chroms,
        'n_intervals': n_intervals,
    }, annot


def benchmark_alignment_loading(opts, annot, repeat=1):
    """Benchmark BAM loading + sparse matrix construction."""
    from telescope.utils.model import Telescope

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        ts = Telescope(opts)
        ts.load_alignment(annot)
        times.append(time.perf_counter() - start)

    return {
        'stage': 'bam_load_and_matrix',
        'times': times,
        'mean_s': np.mean(times),
        'std_s': np.std(times),
        'matrix_shape': list(ts.shape),
        'nnz': int(ts.raw_scores.nnz),
        'density': float(ts.raw_scores.nnz / (ts.shape[0] * ts.shape[1])),
    }, ts


def benchmark_em(ts, opts, repeat=1):
    """Benchmark EM algorithm (init + convergence)."""
    from telescope.utils.model import TelescopeLikelihood

    times_init = []
    times_em = []
    lnl_values = []
    iteration_counts = []

    for _ in range(repeat):
        seed = ts.get_random_seed()
        np.random.seed(seed)

        # Init
        start = time.perf_counter()
        tl = TelescopeLikelihood(ts.raw_scores, opts)
        times_init.append(time.perf_counter() - start)

        # EM
        start = time.perf_counter()
        tl.em(use_likelihood=True)
        times_em.append(time.perf_counter() - start)

        lnl_values.append(float(tl.lnl))

    return {
        'stage': 'em_algorithm',
        'times_init': times_init,
        'times_em': times_em,
        'mean_init_s': np.mean(times_init),
        'mean_em_s': np.mean(times_em),
        'std_init_s': np.std(times_init),
        'std_em_s': np.std(times_em),
        'log_likelihood': lnl_values[-1],
        'N': tl.N,
        'K': tl.K,
    }, tl


def benchmark_estep_mstep(ts, opts, n_iterations=10):
    """Benchmark individual E-step and M-step times."""
    from telescope.utils.model import TelescopeLikelihood

    seed = ts.get_random_seed()
    np.random.seed(seed)
    tl = TelescopeLikelihood(ts.raw_scores, opts)

    estep_times = []
    mstep_times = []
    lnl_times = []

    pi, theta = tl.pi.copy(), tl.theta.copy()
    for _ in range(n_iterations):
        start = time.perf_counter()
        z = tl.estep(pi, theta)
        estep_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        pi, theta = tl.mstep(z)
        mstep_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        tl.calculate_lnl(z, pi, theta)
        lnl_times.append(time.perf_counter() - start)

    return {
        'stage': 'estep_mstep_detail',
        'n_iterations': n_iterations,
        'estep_mean_ms': np.mean(estep_times) * 1000,
        'estep_std_ms': np.std(estep_times) * 1000,
        'mstep_mean_ms': np.mean(mstep_times) * 1000,
        'mstep_std_ms': np.std(mstep_times) * 1000,
        'lnl_mean_ms': np.mean(lnl_times) * 1000,
        'lnl_std_ms': np.std(lnl_times) * 1000,
    }


def benchmark_reassignment(tl, repeat=1):
    """Benchmark all 6 reassignment modes."""
    modes = ['exclude', 'choose', 'average', 'conf', 'unique', 'all']
    results = {'stage': 'reassignment'}

    for mode in modes:
        times = []
        for _ in range(repeat):
            np.random.seed(42)
            start = time.perf_counter()
            tl.reassign(mode, thresh=0.9)
            times.append(time.perf_counter() - start)
        results[mode + '_mean_ms'] = np.mean(times) * 1000
        results[mode + '_std_ms'] = np.std(times) * 1000

    return results


def benchmark_sparse_ops(ts):
    """Benchmark key sparse_plus operations used in the pipeline."""
    from telescope.utils.sparse_plus import csr_matrix_plus

    mat = csr_matrix_plus(ts.raw_scores)
    results = {'stage': 'sparse_ops'}
    ops = OrderedDict()

    # scale
    start = time.perf_counter()
    for _ in range(100):
        mat.scale()
    ops['scale_mean_us'] = (time.perf_counter() - start) / 100 * 1e6

    # norm(1)
    start = time.perf_counter()
    for _ in range(100):
        mat.norm(1)
    ops['norm1_mean_us'] = (time.perf_counter() - start) / 100 * 1e6

    # expm1
    start = time.perf_counter()
    for _ in range(100):
        mat.expm1()
    ops['expm1_mean_us'] = (time.perf_counter() - start) / 100 * 1e6

    # multiply by scalar
    start = time.perf_counter()
    for _ in range(100):
        mat.multiply(100.0)
    ops['multiply_scalar_mean_us'] = (time.perf_counter() - start) / 100 * 1e6

    # multiply by vector
    vec = np.random.rand(mat.shape[1])
    start = time.perf_counter()
    for _ in range(100):
        mat.multiply(vec)
    ops['multiply_vector_mean_us'] = (time.perf_counter() - start) / 100 * 1e6

    # binmax
    start = time.perf_counter()
    for _ in range(100):
        mat.binmax(1)
    ops['binmax_mean_us'] = (time.perf_counter() - start) / 100 * 1e6

    results.update(ops)
    return results


# ---------------------------------------------------------------------------
# Full-pipeline memory measurement
# ---------------------------------------------------------------------------

def benchmark_memory(opts, annot):
    """Measure peak memory for the full pipeline."""
    from telescope.utils.model import Telescope, TelescopeLikelihood

    tracemalloc.start()

    ts = Telescope(opts)
    ts.load_alignment(annot)
    _, after_load = tracemalloc.get_traced_memory()

    seed = ts.get_random_seed()
    np.random.seed(seed)
    tl = TelescopeLikelihood(ts.raw_scores, opts)
    _, after_init = tracemalloc.get_traced_memory()

    tl.em(use_likelihood=True)
    _, after_em = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    return {
        'stage': 'memory_mb',
        'after_bam_load': after_load / (1024 * 1024),
        'after_em_init': after_init / (1024 * 1024),
        'after_em_converge': after_em / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmarks(repeat=3, output_file=None):
    """Run all benchmarks and print results."""
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

    all_results = OrderedDict()
    all_results['backend'] = 'stock_cpu_scipy'
    all_results['repeat'] = repeat

    # Python/library versions
    import scipy
    all_results['versions'] = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'scipy': scipy.__version__,
    }

    print("=" * 65)
    print("Telescope Baseline Benchmark (Stock CPU / scipy)")
    print("=" * 65)
    print(f"  Repeat: {repeat}x per stage")
    print(f"  Data:   {samfile}")
    print()

    # 1. Annotation
    print("[1/6] Annotation loading...")
    annot_results, annot = benchmark_annotation(gtffile, repeat)
    all_results['annotation'] = annot_results
    print(f"       {annot_results['mean_s']:.4f}s (±{annot_results['std_s']:.4f}s)")
    print(f"       {annot_results['n_loci']} loci, {annot_results['n_intervals']} intervals")
    print()

    # 2. BAM loading
    print("[2/6] BAM loading + matrix construction...")
    bam_results, ts = benchmark_alignment_loading(opts, annot, repeat)
    all_results['bam_load'] = bam_results
    print(f"       {bam_results['mean_s']:.4f}s (±{bam_results['std_s']:.4f}s)")
    print(f"       Matrix: {bam_results['matrix_shape']} nnz={bam_results['nnz']} density={bam_results['density']:.4f}")
    print()

    # 3. EM algorithm
    print("[3/6] EM algorithm (init + convergence)...")
    em_results, tl = benchmark_em(ts, opts, repeat)
    all_results['em'] = em_results
    print(f"       Init:  {em_results['mean_init_s']:.4f}s (±{em_results['std_init_s']:.4f}s)")
    print(f"       EM:    {em_results['mean_em_s']:.4f}s (±{em_results['std_em_s']:.4f}s)")
    print(f"       LnL:   {em_results['log_likelihood']:.6f}")
    print()

    # 4. E-step / M-step detail
    print("[4/6] E-step / M-step detail (10 iterations)...")
    step_results = benchmark_estep_mstep(ts, opts, n_iterations=10)
    all_results['estep_mstep'] = step_results
    print(f"       E-step: {step_results['estep_mean_ms']:.3f}ms (±{step_results['estep_std_ms']:.3f}ms)")
    print(f"       M-step: {step_results['mstep_mean_ms']:.3f}ms (±{step_results['mstep_std_ms']:.3f}ms)")
    print(f"       LnL:    {step_results['lnl_mean_ms']:.3f}ms (±{step_results['lnl_std_ms']:.3f}ms)")
    print()

    # 5. Reassignment
    print("[5/6] Reassignment modes...")
    reassign_results = benchmark_reassignment(tl, repeat)
    all_results['reassignment'] = reassign_results
    for mode in ['exclude', 'choose', 'average', 'conf', 'unique', 'all']:
        mean = reassign_results[mode + '_mean_ms']
        std = reassign_results[mode + '_std_ms']
        print(f"       {mode:10s}: {mean:.3f}ms (±{std:.3f}ms)")
    print()

    # 6. Sparse ops
    print("[6/6] Sparse operations (100 iterations each)...")
    sparse_results = benchmark_sparse_ops(ts)
    all_results['sparse_ops'] = sparse_results
    for key, val in sparse_results.items():
        if key == 'stage':
            continue
        name = key.replace('_mean_us', '')
        print(f"       {name:20s}: {val:.1f}µs")
    print()

    # Memory
    print("[+] Memory profiling...")
    mem_results = benchmark_memory(opts, annot)
    all_results['memory'] = mem_results
    print(f"       After BAM load:    {mem_results['after_bam_load']:.2f} MB")
    print(f"       After EM init:     {mem_results['after_em_init']:.2f} MB")
    print(f"       After EM converge: {mem_results['after_em_converge']:.2f} MB")
    print()

    print("=" * 65)
    print("Benchmark complete.")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Telescope baseline benchmark')
    parser.add_argument('--repeat', type=int, default=3,
                        help='Number of repetitions per stage (default: 3)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()
    run_benchmarks(repeat=args.repeat, output_file=args.output)


if __name__ == '__main__':
    main()
