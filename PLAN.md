# Polymerase Optimization Plan

## Context

Polymerase uses an EM algorithm on sparse matrices (N fragments x K features) to resolve multi-mapped RNA-seq reads. The pipeline is I/O-dominated: BAM loading accounts for 90%+ of runtime on real data. Optimizations target both I/O and computation.

---

## Optimization Dimensions

```
Dimension 1: ELIMINATE WASTED I/O      -> Region pre-filtering skips ~99% of reads
Dimension 2: SPEED UP I/O              -> pysam multi-threaded BAM decompression
Dimension 3: DECOMPOSE the problem     -> Connected components split matrix into independent blocks
Dimension 4: ACCELERATE the math       -> Numba JIT kernels for sparse operations
Dimension 5: PARALLELIZE pipeline tail -> Concurrent reassignment, batch matrix construction
```

### Pipeline Architecture

```
Stage 0: Smart BAM Access
    +- If BAM is coordinate-sorted + indexed:
    |     -> Two-pass: fetch TE regions -> buffer relevant fragments
    |     -> SKIPS ~99% of reads entirely
    |     -> Multi-threaded decompression (pysam threads=N)
    +- If BAM is name-collated:
          -> Sequential loading with pysam threads=N

Stage 1: Batch Matrix Construction (COO arrays -> CSR)

Stage 2: Block Decomposition (scipy connected_components)
    +- Block A  (N_a x K_a) -- e.g., HERVK family
    +- Block B  (N_b x K_b) -- e.g., LINE1 family
    +- Block C  (N_c x K_c) -- e.g., Alu family
              | parallel (ThreadPoolExecutor, Numba releases GIL)

Stage 3: EM on each block
    +- Block A -> Thread 1 -> pi_a, theta_a
    +- Block B -> Thread 2 -> pi_b, theta_b
    +- Block C -> Thread 3 -> pi_c, theta_c
              | merge results

Stage 4: Reassignment + Reports
```

---

## Measured Results

### 5% BAM (~677K fragments, 940 features)

| Path | Load Time | Total | Speedup |
|------|-----------|-------|---------|
| Sequential (collated) | 8.2s | 8.5s | 1x |
| Indexed (sorted+indexed) | 1.9s | 2.1s | **4.4x** |

Both paths produce identical matrices (0 mismatches).

### 100% BAM (~14.7M fragments, 4,372 features, 2,794 components)

| Stage | Time | % of Total |
|-------|------|------------|
| BAM loading | 23.2s | 92.8% |
| EM (block-parallel) | 0.7s | 2.8% |
| Annotation load | 1.0s | 4.0% |
| Report generation | 0.1s | 0.4% |
| **Total** | **25.0s** | |

3,188 active TE loci. Top locus: L1FLnI_11p15.4m with 11,164 reads.

### Key insight

I/O dominates at 93% of runtime. The EM is already fast enough with block decomposition (0.7s for 75K x 4.4K matrix with 2,794 blocks). Further CPU math optimization has negligible impact on end-to-end time.

---

## Implementation Status

### Phase 0: Test Foundation + Code Re-org

- Migrated nose -> pytest
- 86 tests (sparse_plus, EM, baseline equivalence, annotation)
- Module split: model.py -> model.py + likelihood.py + reporter.py
- Removed Python 2 compatibility, dead code, unused files
- Baseline benchmarking

### Phase 1: CPU Optimizations

- **`backend.py`**: Two-tier backend (CPU-Optimized / Stock CPU), auto-detection
- **`numba_matrix.py`**: Numba JIT kernels for `binmax`, `choose_random`, `threshold_filter`, `indicator`
- **`decompose.py`**: Connected component block decomposition (`find_blocks`, `split_matrix`, `merge_results`)
- **Tests**: 33 CPU-optimized sparse tests, 10 decomposition tests, 6 numerical equivalence tests

### Phase 2: Pipeline Integration

- `threshold_filter()` and `indicator()` on `csr_matrix_plus`
- Backend-aware `likelihood.py` (uses `get_sparse_class()`)
- Backend-aware `model.py` (batch COO, pysam threads)
- `em_parallel()` in `PolymeraseLikelihood`

### Phase 3: Indexed BAM Loading

- **`_load_indexed()`**: Two-pass approach (TE region fetch -> full buffer -> process)
- **`_get_te_regions()`**: Extract TE coordinates with 500bp padding, merge overlapping
- **`_find_primary()`**: Fix supplementary alignment classification bug
- **Auto-detection**: `has_index and not updated_sam` -> indexed path
- 1% test data (30MB BAM + 18MB GTF) checked into repo
- 35 pipeline tests including indexed/sequential equivalence

### Deferred (future sprint)

- **Parallel reassignment**: ThreadPoolExecutor for 6 reassignment modes concurrently. Low impact (~0.1s currently). See TODO in `polymerase/core/likelihood.py`.
- **In-memory parallel loading**: Replace temp file I/O in `_load_parallel()` with in-memory return values. See TODO in `polymerase/core/model.py`.

---

## Key Technical Notes

- **Block decomposition is mathematically exact**: Connected components yield fully independent sub-problems. No approximation.
- **`__no_feature` is a noise sink**: Column 0, absorbs ~7% of pi mass. Essential for preventing phantom TE expression. Removing it inflates TE counts; fragments mapping to both TE and non-TE regions lose their "maybe not a TE" signal.
- **Supplementary alignment bug**: Stock Telescope misclassifies fragments when supplementary alignments precede primary alignments in collated BAMs. Fixed by `_find_primary()`.
- **Test data has 1 component** (all ERVK): Block decomposition shows effect on real data (2,794 blocks on 100% data).
- **All float64 precision**: Results match within machine epsilon across backends.

---

## File Summary

### Created
| File | Purpose |
|------|---------|
| `polymerase/sparse/backend.py` | Backend auto-detection and dispatch |
| `polymerase/sparse/numba_matrix.py` | Numba JIT sparse operations |
| `polymerase/sparse/decompose.py` | Block decomposition |
| `polymerase/core/likelihood.py` | EM algorithm (extracted from model.py) |
| `polymerase/core/reporter.py` | Report I/O (extracted from model.py) |
| `polymerase/tests/test_sparse_plus.py` | 45 sparse matrix tests |
| `polymerase/tests/test_em.py` | 33 EM + baseline tests |
| `polymerase/tests/test_cpu_optimized_sparse.py` | 33 Numba equivalence tests |
| `polymerase/tests/test_decomposition.py` | 10 block decomposition tests |
| `polymerase/tests/test_numerical_equivalence.py` | 6 cross-backend tests |
| `polymerase/tests/test_1pct_pipeline.py` | 35 pipeline tests |
| `polymerase/tests/benchmark_1pct.py` | Benchmarking script |
| `test_data/SRR9666161_1pct_collated.bam` | 1% test BAM (30MB) |
| `test_data/retro.hg38.gtf` | Full TE annotation (18MB) |

### Modified
| File | Changes |
|------|---------|
| `polymerase/core/model.py` | `_load_indexed()`, `_get_te_regions()`, batch COO, pysam threads, backend-aware sparse |
| `polymerase/alignment/fragments.py` | `_find_primary()`, supplementary alignment fix |
| `polymerase/sparse/matrix.py` | `threshold_filter()`, `indicator()` |
| `polymerase/cli/assign.py` | `backend.configure()` call |
| `polymerase/cli/resume.py` | `backend.configure()` call |

### Deleted
| File | Reason |
|------|--------|
| `telescope/utils/alignment_parsers.py` (deleted pre-rename) | Entirely unused |

---

## Verification

```bash
# Full test suite (170 tests)
pytest polymerase/tests/ -v

# Golden log-likelihood on bundled test data
eval $(polymerase test) 2>&1 | grep 'Final log-likelihood'
# Expected: 95252.596293

# End-to-end comparison (requires sorted+indexed BAM)
polymerase assign sorted.bam annotation.gtf --outdir /tmp/indexed --exp_tag indexed
polymerase assign collated.bam annotation.gtf --outdir /tmp/sequential --exp_tag sequential
diff /tmp/indexed/indexed-TE_counts.tsv /tmp/sequential/sequential-TE_counts.tsv
```
