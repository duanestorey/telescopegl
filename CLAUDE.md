# Polymerase - Claude Code Project Context

## What is Polymerase?

Polymerase resolves multi-mapped RNA-seq reads to transposable element (TE) loci using an Expectation-Maximization (EM) algorithm on sparse matrices (N fragments x K features).

## Pipeline Overview

```
Load BAM -> Build sparse matrix -> Run EM -> Reassign reads -> Generate reports
```

Two BAM loading paths:
- **Sequential** (collated BAM): Streams all reads, pairs by name
- **Indexed** (coordinate-sorted + .bai): Two-pass approach -- fetches TE regions first, then buffers relevant fragments. 4-8x faster, produces identical matrix.

## Key Architecture

### Core Files

| File | Purpose |
|------|---------|
| `polymerase/core/model.py` | `Polymerase` class: BAM loading (`_load_sequential`, `_load_indexed`, `_load_parallel`), matrix construction, `Assigner` |
| `polymerase/core/likelihood.py` | `PolymeraseLikelihood` class: EM algorithm (estep, mstep, lnl, em, em_parallel, reassign) |
| `polymerase/core/reporter.py` | `output_report()` + `update_sam()` -- I/O separated from algorithm |
| `polymerase/sparse/matrix.py` | `csr_matrix_plus` -- extends scipy CSR with `norm`, `scale`, `binmax`, `choose_random`, `apply_func`, `count`, `threshold_filter`, `indicator` |
| `polymerase/cli/assign.py` | `assign` subcommand: CLI options (`BulkIDOptions`, `scIDOptions`), `run()` orchestrator |
| `polymerase/cli/resume.py` | `resume` subcommand: loads checkpoint, re-runs EM with different params |
| `polymerase/__main__.py` | Entry point, argparse subcommand routing |
| `polymerase/cli/__init__.py` | `SubcommandOptions` base class (YAML-based CLI), `configure_logging()`, `BIG_INT` |

### Optimization Files

| File | Purpose |
|------|---------|
| `polymerase/sparse/backend.py` | Two-tier backend: CPU-Optimized/Numba+MKL -> Stock CPU/scipy. Auto-detection via `configure()` / `get_backend()` / `get_sparse_class()` |
| `polymerase/sparse/numba_matrix.py` | `CpuOptimizedCsrMatrix(csr_matrix_plus)` -- Numba JIT kernels for `binmax`, `choose_random`, `threshold_filter`, `indicator` |
| `polymerase/sparse/decompose.py` | Connected component block decomposition: `find_blocks()`, `split_matrix()`, `merge_results()` |

### Annotation Files
| File | Purpose |
|------|---------|
| `polymerase/annotation/intervaltree.py` | IntervalTree-based annotation (primary) |
| `polymerase/annotation/bisect.py` | Bisect-based annotation (alternative) |
| `polymerase/annotation/__init__.py` | Annotation factory |

### Alignment Files
| File | Purpose |
|------|---------|
| `polymerase/alignment/fragments.py` | Fragment fetching, read pairing, `_find_primary()` for supplementary alignment handling |
| `polymerase/alignment/calignment.pyx` | Cython `AlignedPair` class |

## EM Algorithm (PolymeraseLikelihood)

**Mathematical model:**
- `Q[i,j]` = scaled alignment score (likelihood of fragment i from transcript j)
- `Y[i]` = 1 if fragment i is ambiguous (multi-mapped), 0 if unique
- `z[i,j]` = posterior probability of assignment
- `pi[j]` = proportion of fragments from transcript j
- `theta[j]` = proportion of ambiguous fragments assigned to transcript j

**E-step:** `z[i,j] = (pi[j] * theta[j]^Y[i] * Q[i,j]) / sum_k(...)`
**M-step:** MAP estimates with priors: `pi_prior` (default 0), `theta_prior` (default 200000)

**Convergence:** `abs(pi_new - pi_old).sum() < epsilon` (default 1e-7) or max 100 iterations.

**Six reassignment modes:** `exclude`, `choose`, `average`, `conf`, `unique`, `all`

**Block-parallel EM:** `em_parallel()` decomposes the score matrix into independent blocks via connected components, runs EM on each block (ThreadPoolExecutor for Numba backends), and merges results.

## Indexed BAM Loading

`_load_indexed()` in `model.py` uses a two-pass approach:
1. **Phase 1**: `pysam.fetch()` on TE regions (with 500bp padding) to identify relevant fragment names
2. **Phase 2**: Stream full BAM, buffering all alignments for relevant fragments (mates + non-TE mappings for `__no_feature`)
3. **Phase 3**: Process buffered fragments with identical logic to sequential path

Auto-detected when BAM has a `.bai` index and `--updated_sam` is not set.

**`__no_feature`**: Column 0 in the matrix. Acts as a noise sink -- fragments mapping to both a TE and a non-TE region have probability split between the TE and `__no_feature`. Absorbs ~7% of pi mass on typical data. Essential for preventing phantom TE expression.

## CLI Options Pattern

Options are defined as YAML strings in `BulkIDOptions.OPTS` / `scIDOptions.OPTS`, parsed by `SubcommandOptions._parse_yaml_opts()`, and converted to argparse arguments. Access via `opts.attribute_name`.

## Test Data

### Bundled (in repo)
- `polymerase/data/alignment.bam` -- small test BAM (1000 fragments, 59 features)
- `polymerase/data/annotation.gtf` -- small test GTF
- `test_data/SRR9666161_1pct_collated.bam` -- 1% subsample, collated (30MB, ~136K fragments)
- `test_data/retro.hg38.gtf` -- full retro.hg38 TE annotation (28,513 features)

### External (not in repo, tests skip gracefully)
- `test_data/SRR9666161_5pct_collated.bam` -- 5% subsample, collated
- `test_data/SRR9666161_5pct_sorted.bam` + `.bai` -- 5% subsample, sorted+indexed
- `test_data/SRR9666161_100pct_sorted.bam` + `.bai` -- full dataset (14.7M fragments)

### Golden values
- Log-likelihood on bundled test data: ~ 95252.596293
- Test data has 1 connected component (all ERVK family)
- 100% data: 75,121 fragments x 4,372 features, 2,794 components

## Tests

170 tests across 7 test files:
- `test_annotation_parsers.py` -- 8 tests
- `test_sparse_plus.py` -- 45 tests
- `test_em.py` -- 33 tests (EM + baseline equivalence)
- `test_cpu_optimized_sparse.py` -- 33 tests (Numba vs stock)
- `test_decomposition.py` -- 10 tests (block decomposition)
- `test_numerical_equivalence.py` -- 6 tests (cross-backend)
- `test_1pct_pipeline.py` -- 35 tests (1%/5%/100% BAM pipeline, indexed path equivalence)

## Key Conventions

- Sparse matrices use `csr_matrix_plus` from `matrix.py` (or `CpuOptimizedCsrMatrix` when Numba available)
- `get_sparse_class()` from `backend.py` returns the appropriate sparse class
- pysam opened with `check_sq=False` to skip sequence dictionary validation
- pysam uses `threads=N` for multi-threaded BAM decompression
- Fragment counts tracked via `collections.Counter` (`alninfo`)
- `__no_feature` sentinel for reads not overlapping annotation (always column 0)
- All float64 precision
- Random seed derived from `total_fragments % shape[0] * shape[1]`
- `_find_primary()` in `fragments.py` ensures primary alignment is used for fragment classification (not supplementary)

## Dependencies

### Required
- `scipy.sparse` (CSR matrices)
- `numpy` (arrays, linear algebra)
- `pysam` (BAM reading/writing)
- `intervaltree` (genomic interval queries)
- `pandas` (report generation)
- `cython` (AlignedPair performance)

### Optional (CPU-optimized tier)
- `numba` (JIT compilation for sparse kernels)
- `sparse_dot_mkl` (MKL-accelerated sparse ops)
- `threadpoolctl` (BLAS thread control)

## Optimizations Implemented

- **Backend abstraction** (`backend.py`): Auto-detects Numba+MKL, falls back to stock scipy
- **Numba JIT sparse kernels** (`numba_matrix.py`): 5-50x faster `binmax`, `choose_random`, `threshold_filter`, `indicator`
- **Connected component block decomposition** (`decompose.py`): Splits matrix into independent sub-problems
- **Block-parallel EM** (`likelihood.py` `em_parallel`): ThreadPoolExecutor across blocks (Numba releases GIL)
- **Batch COO matrix construction** (`model.py`): Replaced DOK loop with bulk COO + duplicate resolution
- **pysam multi-threaded decompression**: All BAM reads use `threads=N`
- **Indexed BAM loading** (`model.py` `_load_indexed`): Region pre-filtering skips ~99% of reads
- **Supplementary alignment fix** (`fragments.py` `_find_primary`): Correct PM/PX classification regardless of BAM sort order

### Deferred (future sprint)
- Parallel reassignment (ThreadPoolExecutor for 6 modes) — see TODO in `polymerase/core/likelihood.py`
- In-memory parallel BAM loading (replace temp files) — see TODO in `polymerase/core/model.py`
