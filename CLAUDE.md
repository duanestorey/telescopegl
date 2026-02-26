# Telescope - Claude Code Project Context

## What is Telescope?

Telescope resolves multi-mapped RNA-seq reads to transposable element (TE) loci using an Expectation-Maximization (EM) algorithm on sparse matrices (N fragments x K features).

## Pipeline Overview

```
Load BAM → Build sparse matrix → Run EM → Reassign reads → Generate reports
```

## Key Architecture

### Core Files

| File | Purpose |
|------|---------|
| `telescope/utils/model.py` | Core classes: `Telescope` (BAM loading, matrix building), `TelescopeLikelihood` (EM algorithm), `Assigner` (overlap assignment) |
| `telescope/utils/sparse_plus.py` | `csr_matrix_plus` — extends scipy CSR with `norm`, `scale`, `binmax`, `choose_random`, `apply_func`, `count` |
| `telescope/telescope_assign.py` | `assign` subcommand: CLI options (`BulkIDOptions`, `scIDOptions`), `run()` orchestrator |
| `telescope/telescope_resume.py` | `resume` subcommand: loads checkpoint, re-runs EM with different params |
| `telescope/__main__.py` | Entry point, argparse subcommand routing |
| `telescope/utils/__init__.py` | `SubcommandOptions` base class (YAML-based CLI), `configure_logging()`, `BIG_INT` |

### Annotation Files
| File | Purpose |
|------|---------|
| `telescope/utils/_annotation_intervaltree.py` | IntervalTree-based annotation (primary) |
| `telescope/utils/_annotation_bisect.py` | Bisect-based annotation (alternative) |
| `telescope/utils/annotation.py` | Annotation factory |

### Alignment Files
| File | Purpose |
|------|---------|
| `telescope/utils/alignment.py` | Fragment fetching, read pairing |
| `telescope/utils/alignment_parsers.py` | BAM parsing helpers |
| `telescope/utils/calignment.pyx` | Cython `AlignedPair` class |

## EM Algorithm (TelescopeLikelihood)

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

## CLI Options Pattern

Options are defined as YAML strings in `BulkIDOptions.OPTS` / `scIDOptions.OPTS`, parsed by `SubcommandOptions._parse_yaml_opts()`, and converted to argparse arguments. Access via `opts.attribute_name`.

## Test Data

- `telescope/data/alignment.bam` — test BAM file
- `telescope/data/annotation.gtf` — test GTF annotation
- Expected log-likelihood on test data: ≈ 95252.596293
- Test data has 1 connected component (all ERVK family)

## Key Conventions

- Sparse matrices use `csr_matrix_plus` from `sparse_plus.py` (aliased as `csr_matrix` in model.py)
- pysam opened with `check_sq=False` to skip sequence dictionary validation
- Fragment counts tracked via `collections.Counter` (`alninfo`)
- `__no_feature` sentinel for reads not overlapping annotation
- All float64 precision
- Random seed derived from `total_fragments % shape[0] * shape[1]`

## Dependencies

- `scipy.sparse` (CSR matrices)
- `numpy` (arrays, linear algebra)
- `pysam` (BAM reading/writing)
- `intervaltree` (genomic interval queries)
- `pandas` (report generation)
- `cython` (AlignedPair performance)

## Acceleration Plan

See `PLAN.md` for the GPU + parallel acceleration plan covering:
1. Three-tier backend (GPU/CuPy → CPU-Optimized/Numba+MKL → Stock CPU/scipy)
2. Block decomposition via connected components
3. Region pre-filtering for I/O optimization
4. pysam multi-threaded decompression
5. Parallel reassignment
6. Batch COO matrix construction
