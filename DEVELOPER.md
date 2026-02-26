# Developer Notes

## Installation

### 1. Clone repository

```bash
git clone git@github.com:duanestorey/polymerase.git
```

### 2. Create conda environment

An environment file is included in this repo.

```bash
mamba env create -n polymerase_dev -f environment.yml
conda activate polymerase_dev
```

### 3. Install in development mode

```bash
pip install -e .
```

### 4. Optional: Install CPU-optimized dependencies

```bash
pip install numba sparse-dot-mkl threadpoolctl
```

These are auto-detected at runtime. When available, Numba JIT kernels replace
stock scipy operations for `binmax`, `choose_random`, `threshold_filter`, and
`indicator` — typically 5-50x faster on those operations.

## Testing

### Quick test (bundled data)

```bash
eval $(polymerase test) 2>&1 | grep 'Final log-likelihood' | grep -q '95252.56293' && echo "Test OK" || echo "Test FAIL"
```

### Full test suite

```bash
pytest polymerase/tests/ -v
```

170 tests covering:
- Sparse matrix operations (`test_sparse_plus.py`)
- EM algorithm and baseline equivalence (`test_em.py`)
- Block decomposition (`test_decomposition.py`)
- CPU-optimized vs stock equivalence (`test_cpu_optimized_sparse.py`)
- Numerical equivalence across backends (`test_numerical_equivalence.py`)
- Pipeline tests on 1% BAM data (`test_1pct_pipeline.py`)

### Test data

**Bundled in repo** (used by default):
- `polymerase/data/alignment.bam` — small test BAM (1000 fragments)
- `polymerase/data/annotation.gtf` — small test GTF (59 features)
- `test_data/SRR9666161_1pct_collated.bam` — 1% subsample (30MB, ~136K fragments)
- `test_data/retro.hg38.gtf` — full retro.hg38 TE annotation (28,513 features)

**Optional large files** (tests skip gracefully if absent):
- `test_data/SRR9666161_5pct_collated.bam` — 5% subsample, collated
- `test_data/SRR9666161_5pct_sorted.bam` + `.bai` — 5% subsample, sorted+indexed
- `test_data/SRR9666161_100pct_sorted.bam` + `.bai` — full dataset (~14.7M fragments)

### Benchmarking

```bash
python -m pytest polymerase/tests/test_1pct_pipeline.py -v -k "TestFullScale" -s
```

The full-scale tests (require 100% BAM) report per-stage timing and verify the
complete pipeline runs under 60 seconds.

## Architecture

```
polymerase/
├── __init__.py                          # Package init
├── __main__.py                          # Entry point, argparse routing
├── cli/
│   ├── __init__.py                      # SubcommandOptions, configure_logging()
│   ├── assign.py                        # assign subcommand
│   └── resume.py                        # resume subcommand
├── core/
│   ├── model.py                         # Polymerase orchestrator: BAM loading, matrix construction, Assigner
│   ├── likelihood.py                    # PolymeraseLikelihood: EM algorithm (estep, mstep, em, em_parallel, reassign)
│   └── reporter.py                      # output_report() + update_sam()
├── sparse/
│   ├── matrix.py                        # csr_matrix_plus: custom scipy CSR
│   ├── numba_matrix.py                  # CpuOptimizedCsrMatrix: Numba JIT subclass
│   ├── backend.py                       # Backend auto-detection and dispatch
│   └── decompose.py                     # Connected component block decomposition
├── annotation/
│   ├── __init__.py                      # Annotation factory
│   ├── intervaltree.py                  # IntervalTree-based annotation
│   └── bisect.py                        # Bisect-based annotation
├── alignment/
│   ├── fragments.py                     # Fragment fetching, read pairing, _find_primary()
│   ├── _helpers.py                      # Alignment helpers
│   ├── calignment.pyx                   # Cython AlignedPair
│   └── calignment.pxd                   # Cython declarations
├── utils/
│   ├── helpers.py                       # Utilities (phred, format_minutes, merge_overlapping_regions)
│   └── colors.py                        # Color utilities
├── tests/
│   ├── test_annotation_parsers.py       # 8 tests
│   ├── test_sparse_plus.py              # 45 tests
│   ├── test_em.py                       # 33 tests
│   ├── test_cpu_optimized_sparse.py     # 33 tests
│   ├── test_decomposition.py            # 10 tests
│   ├── test_numerical_equivalence.py    # 6 tests
│   ├── test_1pct_pipeline.py            # 35 tests
│   └── benchmark_1pct.py               # Benchmarking script
└── data/
    ├── alignment.bam
    ├── annotation.gtf
    └── report.tsv
```

## BAM Loading Paths

1. **Indexed** (`_load_indexed`): For coordinate-sorted BAMs with `.bai` index.
   Two-pass: fetches TE regions -> buffers relevant fragments -> processes identically
   to sequential. 4-8x faster.

2. **Sequential** (`_load_sequential`): For collated/name-sorted BAMs. Streams all
   reads, groups by name, pairs and assigns.

3. **Parallel** (`_load_parallel`): Multi-process variant of sequential for sorted BAMs.
   Uses `multiprocessing` with temp files.

Auto-detection in `load_alignment()`: indexed BAM + no `--updated_sam` -> indexed path.
