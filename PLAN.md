# Telescope GPU + Parallel Acceleration Plan

## Context

Telescope uses an EM algorithm on sparse matrices (N fragments x K features) to resolve multi-mapped RNA-seq reads. The current pipeline is entirely serial: load BAM → build matrix → run EM → reassign → report. We can attack this from **five dimensions** simultaneously, not just GPU-accelerating the math.

---

## The Five Dimensions of Speedup

```
Dimension 1: ELIMINATE WASTED I/O      → Region pre-filtering skips ~90% of reads that don't overlap TEs
Dimension 2: SPEED UP I/O              → pysam multi-threaded BAM decompression + parallel region loading
Dimension 3: DECOMPOSE the problem     → Connected components split matrix into independent blocks
Dimension 4: GPU-accelerate the math   → CuPy sparse ops for EM steps within each block
Dimension 5: PARALLELIZE pipeline tail → Concurrent reassignment, batch matrix construction
```

### Combined Architecture

```
Stage 0: Smart BAM Access Strategy
    ├─ If BAM is coordinate-sorted + indexed:
    │     → Pre-filter: only fetch regions overlapping TE annotation
    │     → Parallel region loading (multiprocessing, N workers)
    │     → Multi-threaded decompression (pysam threads=N)
    │     → SKIPS ~90% of reads entirely
    └─ If BAM is name-collated:
          → Sequential loading with pysam threads=N
          → Original path, just faster decompression

Stage 1: Parallel BAM Loading (only annotated regions)
    ┌─ TE Region 1 → Worker 1 → mappings_1  (pysam threads=2)
    ├─ TE Region 2 → Worker 2 → mappings_2  (pysam threads=2)
    └─ TE Region R → Worker N → mappings_R  (pysam threads=2)
              ↓ merge (in-memory, not files)

Stage 2: Batch Matrix Construction (COO arrays → CSR, replacing DOK loop)
              ↓

Stage 3: Block Decomposition (scipy connected_components on feature graph)
    ┌─ Block A  (N_a × K_a) — e.g., HERVK family reads
    ├─ Block B  (N_b × K_b) — e.g., LINE1 family reads
    └─ Block C  (N_c × K_c) — e.g., Alu family reads
              ↓ parallel

Stage 4: EM on each block (GPU per block, or CPU cores for small blocks)
    ┌─ Block A → GPU stream 0 → π_a, θ_a
    ├─ Block B → GPU stream 1 → π_b, θ_b
    └─ Block C → CPU thread   → π_c, θ_c
              ↓ merge results

Stage 5: Parallel Reassignment (ThreadPool, 6 modes concurrently)
    ┌─ 'conf'    → Thread 1
    ├─ 'exclude' → Thread 2
    ├─ 'choose'  → Thread 3
    ├─ 'average' → Thread 4
    ├─ 'unique'  → Thread 5
    └─ 'all'     → Thread 6
              ↓ merge → reports
```

---

## Concrete Speedup Estimates

### Baseline Time Breakdown (30M fragments, 5K features, single-threaded CPU)

| Phase | Operation | Time Est. | % of Total |
|-------|-----------|-----------|------------|
| 1a | BAM decompression + read | ~40s | 25% |
| 1b | Fragment pairing (Cython) | ~15s | 9% |
| 1c | IntervalTree overlap queries (30M frags) | ~25s | 15% |
| 1d | Process overlap results | ~10s | 6% |
| 2 | DOK matrix construction | ~15s | 9% |
| 3 | EM init (scale, expm1, etc.) | ~3s | 2% |
| 4 | EM (30 iters × ~2s each) | ~60s | 37% |
| 5 | Reassignment (6 modes) | ~10s | 6% |
| 6 | Report generation | ~2s | 1% |
| **Total** | | **~160s** | |

**Loading is 55% of total time. EM is 37%. Everything else is 8%.**

### Per-Optimization Impact

| Optimization | Affects | Factor | Saves |
|---|---|---|---|
| **Region pre-filter** (biggest single win) | Phases 1a-1d (90s) | Skip 90-97% of reads | **~80-87s** |
| **pysam threads=4** | BAM I/O (40s) | 2-3x decompression | **~20s** |
| **Parallel region loading** (4 workers) | Phases 1a-1d | 2-3x | **~35s** |
| Batch COO matrix | Phase 2 (15s) | 2x | ~7s |
| **GPU EM** (CuPy) | Phase 4 (60s) | 5-20x | **~50s** |
| **Block decomposition** | Phase 4 (60s) | 2-5x (parallel) | ~35s |
| GPU + Blocks combined | Phase 4 (60s) | 10-50x | ~58s |
| Parallel reassignment | Phase 5 (10s) | 4x | ~7s |

### Combined Scenarios

| Scenario | Loading | EM | Other | Total | vs Baseline |
|---|---|---|---|---|---|
| **Baseline (stock CPU)** | ~90s | ~60s | ~25s | **~160s** | 1x |
| CPU-Optimized (Numba+MKL, no I/O changes) | ~90s | ~20s | ~15s | **~125s** | 1.3x |
| CPU-Optimized + pysam threads | ~55s | ~20s | ~15s | **~90s** | 1.8x |
| CPU-Optimized + pre-filter + threads | ~5s | ~20s | ~10s | **~35s** | **4.6x** |
| CPU-Optimized + pre-filter + blocks | ~5s | ~8s | ~8s | **~21s** | **7.6x** |
| GPU only (EM phase, no I/O changes) | ~90s | ~5s | ~15s | **~110s** | 1.5x |
| GPU + pre-filter + pysam threads | ~5s | ~5s | ~8s | **~18s** | **8.9x** |
| **GPU + pre-filter + blocks + all opts** | ~5s | ~2s | ~5s | **~12s** | **~13x** |
| **All optimizations (large data, 100M frags)** | ~10s | ~5s | ~10s | **~25s** vs ~500s | **~20x** |

**Key insights:**
1. I/O optimization (pre-filtering + pysam threads) is the biggest single win because loading dominates runtime
2. CPU-Optimized tier (Numba + MKL) gives 3x EM speedup with NO GPU required
3. Block decomposition helps at every tier (CPU and GPU)
4. Combined, all dimensions multiply: **8-11x on realistic data**

### Tier Comparison for EM Phase Only (removing I/O from equation)

| Tier | EM per iteration | 30 iterations | Notes |
|---|---|---|---|
| Stock CPU (scipy) | ~2.0s | ~60s | Current baseline |
| CPU-Optimized (MKL+Numba) | ~0.7s | ~20s | 3x faster, no GPU needed |
| GPU (CuPy) | ~0.2s | ~5s | 12x faster, needs CUDA |
| GPU + blocks (5 blocks) | ~0.07s | ~2s | 30x faster, parallel GPU streams |

---

## Libraries: Three-Tier Acceleration

The design auto-selects the best available backend: **GPU → CPU-Optimized → CPU-Stock**.
All tiers benefit from the I/O and pipeline optimizations (pysam threading, region pre-filtering, parallel reassignment, batch COO, block decomposition).

### Tier 1: Stock CPU (baseline, always available)
Uses current scipy.sparse — no new dependencies. Still benefits from all algorithmic improvements (block decomposition, region pre-filtering, parallel reassignment).

### Tier 2: CPU-Optimized (when MKL/Numba detected)

| Library | Purpose | Speedup | Install |
|---------|---------|---------|---------|
| **sparse_dot_mkl** | MKL-accelerated sparse matrix multiply. Uses Intel AVX-512/AVX-2 SIMD + multi-threading. Drop-in for `A.multiply(B)` / `A @ B` in E-step/M-step. | 2-5x on sparse ops | `pip install sparse-dot-mkl` |
| **Numba** | JIT-compile Python loops in sparse_plus.py (binmax, choose_random, apply_func) to native LLVM code with SIMD auto-vectorization. | 5-50x on loop ops | `pip install numba` |
| **threadpoolctl** | Runtime control of BLAS thread count (MKL/OpenBLAS). Ensures underlying scipy BLAS uses all cores. | 2-4x on BLAS ops | `pip install threadpoolctl` |

**How sparse_dot_mkl helps the EM core:**
The E-step does `Q.multiply(Y).multiply(pi * theta)` — element-wise sparse multiplies. With MKL, these use hardware-vectorized SIMD instructions and multi-threading internally. On Intel CPUs with AVX-512, this is 2-5x faster than stock scipy.

**How Numba helps sparse_plus.py:**
```python
# Before: Python loop in binmax (~50ms for 1M rows)
for d_start, d_end, d_max in rit:
    _data[d_start:d_end] = (self.data[d_start:d_end] == d_max)

# After: Numba JIT (~1ms for 1M rows)
@numba.njit(parallel=True)
def _binmax_jit(data, indptr, row_max, out):
    for row in numba.prange(len(indptr) - 1):
        for i in range(indptr[row], indptr[row+1]):
            out[i] = 1 if data[i] == row_max[row] else 0
```

### Tier 3: GPU (when CuPy + CUDA detected)

| Library | Purpose | Speedup | Install |
|---------|---------|---------|---------|
| **CuPy** | GPU sparse matrix ops via cuSPARSE. Drop-in replacement for scipy.sparse CSR. | 5-20x on sparse ops | `pip install cupy-cuda12x` |
| **CuPy RawKernels** | Custom CUDA kernels for binmax/choose_random. | 10-100x on loop ops | (included in CuPy) |

### Auto-detection in backend.py:
```python
def configure(use_gpu=False):
    if use_gpu:
        try: import cupy; return GPUBackend()
        except: lg.warning('CuPy unavailable, trying CPU-optimized')

    try:
        import sparse_dot_mkl, numba
        return CPUOptimizedBackend()
    except:
        return CPUStockBackend()  # Always available
```

### Shared across all tiers (stdlib, no new deps)

| Library | Purpose |
|---------|---------|
| **concurrent.futures** | Thread/process pools for reassignment & block EM |
| **multiprocessing** | Parallel BAM loading (already in codebase) |
| **scipy.sparse.csgraph** | Connected components for block decomposition (already a dep) |

---

## Implementation Steps

### Phase 0: Test Foundation + Code Re-org ✅ COMPLETED

**Branch:** `feature/phase-0` | **Tests:** 86/86 passing | **Status:** All steps done

#### Step 0a: Fix Assigner constructor bug ✅
Fixed missing `opts` parameter in `alignment.py` parallel loading path.

#### Step 0b: Migrate from nose to pytest ✅
- Converted `test_annotation_parsers.py` from nose decorators to pytest fixtures/assertions
- Added `pytest.ini` with configuration
- Removed nose dependency

#### Step 0c: Complete sparse_plus test coverage ✅
Added 45 tests in `test_sparse_plus.py` covering all `csr_matrix_plus` methods:
`scale`, `binmax`, `choose_random`, `apply_func`, `count`, `ceil`, `log1p`, `expm1`, `norm`, `save`/`load`, matrix arithmetic, edge cases.

#### Step 0d: Add EM algorithm and baseline equivalence tests ✅
Added 33 tests in `test_em.py`:
- Unit tests for `TelescopeLikelihood`: init, E-step, M-step, convergence
- All 6 reassignment modes: `exclude`, `choose`, `average`, `conf`, `unique`, `all`
- Golden baseline: full pipeline on test data → log-likelihood ≈ 95252.596614
- `TestBaselineEquivalence` class: end-to-end pipeline correctness

#### Step 0e: Remove dead code ✅
- Removed Python 2 compatibility imports from 13 files (`from __future__`, `from past.utils import old_div`, `from builtins import *`)
- Replaced `old_div()` with `//` operator in `helpers.py`
- Removed ~55 lines of commented-out dead code from `model.py` (deprecated `load_mappings`, `_mapping_to_matrix`)
- Removed unused `gc`, `fmtmins` imports
- **Deleted `telescope/utils/alignment_parsers.py`** entirely (unused `TelescopeAlignment`, `TelescopeRead`, `iterread`)
- Fixed missing `import re` and regex escape warning in `helpers.py`

#### Step 0f: Split model.py into focused modules ✅
Split the 833-line god file into 3 focused modules:

| New Module | Lines | Responsibility |
|---|---|---|
| `telescope/utils/likelihood.py` | 197 | `TelescopeLikelihood` class (EM algorithm: init, estep, mstep, lnl, em, reassign) |
| `telescope/utils/reporter.py` | 123 | `output_report()` + `update_sam()` functions (I/O separated from algorithm) |
| `telescope/utils/model.py` | 522 | `Telescope` orchestrator + `Assigner` + BAM loading + matrix construction |

Backward-compatible re-exports maintained in `model.py`. All 86 tests pass with the split.

**Note:** `loader.py` and `matrix_builder.py` from the original plan were NOT created — BAM loading and matrix construction remain in `model.py` since they're tightly coupled to the `Telescope` class. The EM algorithm (`likelihood.py`) and I/O (`reporter.py`) were the cleanest separation points.

#### Step 0g: Extract shared EM+report pipeline — DEFERRED
Deferred to Phase 2 when backend integration will motivate this refactor naturally.

#### Step 0h: Install CPU-optimized dependencies ✅
```bash
pip install numba sparse-dot-mkl threadpoolctl pytest
```
All installed and verified on macOS. Numba JIT compiles successfully. MKL sparse ops functional.

#### Step 0-extra: Baseline benchmarking ✅
Created `telescope/tests/benchmark_baseline.py` — comprehensive benchmarking script measuring all pipeline stages.
Results saved to `telescope/tests/benchmark_baseline_results.json`.

**Baseline Results (test data: 1000 fragments × 59 features, macOS):**

| Stage | Metric | Value |
|-------|--------|-------|
| Annotation load | mean | 3.8ms |
| BAM load + matrix | mean | 419ms |
| EM init | mean | 1.2ms |
| EM convergence (25 iters) | mean | 39.7ms |
| E-step | per iteration | 0.71ms |
| M-step | per iteration | 0.20ms |
| Log-likelihood calc | per iteration | 0.67ms |
| Memory peak | after EM | 7.65 MB |
| Log-likelihood | final | 95252.596614 |

**Sparse ops (100 iterations, microseconds):**

| Operation | Mean µs |
|-----------|---------|
| scale | 22.8 |
| norm(1) | 146.1 |
| expm1 | 43.4 |
| multiply (scalar) | 16.8 |
| multiply (vector) | 52.0 |
| binmax | 707.1 |

**Reassignment modes (milliseconds):**

| Mode | Mean ms |
|------|---------|
| exclude | 0.85 |
| choose | 0.83 |
| average | 0.82 |
| conf | 1.42 |
| unique | 1.62 |
| all | 1.17 |

#### Cython compatibility fixes (incidental)
- Fixed `calignment.pyx` and `.pxd` for Cython 3.0 + pysam 0.23.3 compatibility
- Added pysam include dirs to `setup.py`
- Downgraded Cython 3.2.4 → 3.0.12 for pysam compatibility
- Fixed `'rU'` mode → `'r'` in 3 files (Python 3.13)
- Fixed `stranded_mode` None vs 'None' comparison
- Fixed `subregion()` missing `run_stranded` attribute

---

### Current Architecture (after Phase 0)

```
telescope/
├── __main__.py              # Entry point, argparse routing
├── __init__.py
├── telescope_assign.py      # assign subcommand: BulkIDOptions, scIDOptions, run()
├── telescope_resume.py      # resume subcommand: loads checkpoint, re-runs EM
├── utils/
│   ├── model.py             # Telescope orchestrator (522 lines): BAM loading, matrix construction, Assigner
│   ├── likelihood.py        # TelescopeLikelihood (197 lines): EM algorithm core
│   ├── reporter.py          # output_report() + update_sam() (123 lines): I/O separated from algorithm
│   ├── sparse_plus.py       # csr_matrix_plus: custom scipy CSR with norm/scale/binmax/etc.
│   ├── alignment.py         # Fragment fetching, read pairing
│   ├── _alignment.py        # Pure Python AlignedPair fallback
│   ├── calignment.pyx/pxd   # Cython AlignedPair (performance)
│   ├── annotation.py        # Annotation factory
│   ├── _annotation_intervaltree.py  # IntervalTree annotation (primary)
│   ├── _annotation_bisect.py       # Bisect annotation (alternative)
│   ├── helpers.py           # phred(), format_minutes(), region parsing
│   ├── colors.py            # BAM tag colors
│   └── __init__.py          # SubcommandOptions base, configure_logging()
├── tests/
│   ├── test_annotation_parsers.py   # 8 tests (pytest)
│   ├── test_sparse_plus.py          # 45 tests
│   ├── test_em.py                   # 33 tests (EM + baseline equivalence)
│   ├── benchmark_baseline.py        # Benchmarking script
│   └── benchmark_baseline_results.json  # Baseline results (stock CPU)
└── data/
    ├── alignment.bam         # Test BAM (1000 fragments)
    └── annotation.gtf        # Test GTF (99 loci, 59 features after filtering)
```

**Key target files for Tier 1/2 optimization:**
- `likelihood.py` — EM algorithm: estep, mstep, reassign (Numba JIT, MKL sparse ops)
- `sparse_plus.py` — sparse matrix operations: binmax, choose_random, apply_func (Numba JIT)
- `model.py` — BAM loading: pysam threading, region pre-filtering, batch COO construction
- New: `backend.py` — tier auto-detection and dispatch
- New: `cpu_optimized_sparse.py` — Numba-accelerated csr_matrix_plus subclass
- New: `decompose.py` — connected component block decomposition

---

### Phase 1: Foundation (new files, no behavior changes)

#### Step 1: Create `telescope/utils/backend.py`
Backend abstraction for CPU/GPU selection.
- `configure(use_gpu, device_id)` — called once at startup from CLI
- `get_backend()` — returns CPU or GPU module
- `is_gpu()` — quick check
- `to_host(arr)` / `to_device(arr)` — transfer helpers
- Auto-fallback to CPU if CuPy not installed

#### Step 2: Create `telescope/utils/gpu_sparse.py`
GPU sparse matrix class implementing the same interface as `csr_matrix_plus` (`sparse_plus.py`):

Must implement: `multiply`, `sum`, `max`, `norm`, `scale`, `expm1`, `log1p`, `count`, `binmax`, `choose_random`, `apply_func`, `ceil`, `astype`, `to_cpu`, `copy`, and arithmetic operators.

Key details:
- `binmax` and `choose_random` need CuPy `RawKernel` CUDA kernels (one thread per row iterating over CSR segment)
- `apply_func` replaced with vectorized specializations: `threshold(val)` and `indicator()`
- Constructor accepts `scipy.sparse.csr_matrix` (auto-transfers to GPU)

#### Step 2b: Create `telescope/utils/cpu_optimized_sparse.py`
CPU-optimized sparse matrix class using Numba JIT and optionally sparse_dot_mkl:

- Subclass of `csr_matrix_plus` (inherits all stock scipy methods)
- Overrides performance-critical methods with Numba-compiled versions:
  - `binmax` → `@numba.njit(parallel=True)` with `prange` over rows
  - `choose_random` → `@numba.njit` with Numba's random
  - `apply_func` → `@numba.njit` vectorized over `.data` array
  - `norm` → optionally use sparse_dot_mkl for the underlying multiply
- Falls back to parent class methods if Numba not available
- No CUDA dependency — pure CPU with SIMD vectorization

#### Step 3: Create `telescope/utils/decompose.py`
Block-diagonal decomposition module.
- `find_blocks(score_matrix)` — find connected components in the feature-feature adjacency graph
- `split_matrix(score_matrix, block_labels)` — split N×K matrix into independent sub-matrices
- `merge_results(block_results, block_labels, K)` — reassemble π, θ, z from independent EM results

Algorithm:
```python
def find_blocks(score_matrix):
    # Build feature-feature graph: features connected if any read maps to both
    ftf = (score_matrix > 0).T @ (score_matrix > 0)  # K×K
    n_components, labels = scipy.sparse.csgraph.connected_components(ftf)
    return n_components, labels  # labels[j] = block ID for feature j
```

### Phase 2: Wire GPU into EM (modify existing code)

#### Step 4: Modify `telescope/utils/likelihood.py` — TelescopeLikelihood

**`__init__`**: If GPU active, wrap matrices in `GpuCsrMatrix`:
- `self.Q`, `self.Y`, `self._weights` → GPU sparse
- `self.pi`, `self.theta`, `self._pisum0` → CuPy arrays
- Scalars stay as Python floats

**`estep`**: Replace `csr_matrix(...)` with `type(self.Q)(...)` so it works with either backend. The `.multiply()`, `.norm()` calls work identically on both.

**`mstep`**: Add `.A1` compatibility helper for CuPy matrices that may not have `.A1`.

**`calculate_lnl`**: Same pattern as estep.

**`reassign`**: Works automatically since `binmax`, `choose_random`, `apply_func`, `norm` dispatch to GPU versions when `self.z` is a `GpuCsrMatrix`.

**`em`**: Convergence check `abs(_pi - self.pi).sum()` works on CuPy natively. No structural changes.

### Phase 3: Block decomposition + parallel EM

#### Step 5: Modify `telescope/utils/likelihood.py` — Add block-parallel EM

New method `TelescopeLikelihood.em_parallel(...)`:
```python
def em_parallel(self, ...):
    # 1. Find connected components
    n_blocks, block_labels = find_blocks(self.raw_scores)

    if n_blocks == 1:
        # No decomposition benefit, run standard EM
        return self.em(...)

    # 2. Split into sub-problems
    sub_problems = split_matrix(self.Q, self.Y, block_labels)

    # 3. Run EM on each block (parallel)
    #    Each block has its OWN convergence check (epsilon, max_iter).
    #    Small blocks with clear signal converge fast (5 iters),
    #    large ambiguous blocks may need 30+. This is free performance.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_blocks) as pool:
        futures = []
        for block in sub_problems:
            sub_tl = TelescopeLikelihood._from_block(block, self.opts)
            futures.append(pool.submit(sub_tl.em, ...))
        results = [f.result() for f in futures]

    # 4. Merge results back
    self.pi, self.theta, self.z = merge_results(results, block_labels, self.K)
```

**GIL considerations for block parallelism:**
- GPU tier: CuPy releases the GIL — ThreadPoolExecutor works perfectly
- CPU-Optimized tier: Numba `@njit` releases the GIL — ThreadPoolExecutor works
- Stock CPU tier: Python loops in sparse_plus.py DO NOT release the GIL, so ThreadPoolExecutor will serialize those parts. For stock CPU, blocks still help with memory (smaller matrices) and early convergence, but true parallelism requires ProcessPoolExecutor (with pickle overhead). **Recommendation:** Use ThreadPoolExecutor for GPU/CPU-Optimized tiers; fall back to sequential block processing on stock CPU tier (still benefits from per-block early convergence).

For GPU: each block can use a different CUDA stream for concurrent GPU execution.

#### Step 6: Modify `telescope/telescope_assign.py` — Integrate block EM into `run()`

After `TelescopeLikelihood` is created (line 434), choose EM path:
```python
if opts.gpu or opts.parallel_blocks:
    ts_model.em_parallel(...)
else:
    ts_model.em(...)  # Original path, unchanged
```

### Phase 4: I/O Optimization (the biggest single win)

#### Step 7: Add pysam multi-threaded decompression

**Easiest change, zero algorithmic risk.** BAM files are BGZF-compressed. htslib (which pysam wraps) supports multi-threaded decompression via the `threads` parameter. Currently NO pysam call uses it.

**Files:** `model.py` lines 97, 224, 486; `alignment.py` line 205

Change every `pysam.AlignmentFile(...)` call to include `threads=N`:
```python
# Before:
pysam.AlignmentFile(self.opts.samfile, check_sq=False)
# After:
pysam.AlignmentFile(self.opts.samfile, check_sq=False, threads=opts.ncpu)
```

**Expected impact:** 2-3x faster BAM reading with zero algorithmic changes.

#### Step 8: Region pre-filtering (skip ~90% of reads)

**Biggest single speedup.** For a typical RNA-seq dataset, ~90% of reads map to non-TE regions and are discarded after processing. Currently ALL reads are parsed, paired, overlap-checked, and thrown away.

**Approach:** If the BAM is coordinate-sorted + indexed, build a list of genomic regions from the annotation (TE locus coordinates with padding), then fetch ONLY reads from those regions:

```python
def _get_te_regions(self, annotation):
    """Build regions of interest from TE annotation."""
    regions = []
    for chrom, itree in annotation.itree.items():
        for iv in itree:
            # Add padding for read length
            regions.append((chrom, max(0, iv.begin - 500), iv.end + 500))
    return merge_overlapping_regions(regions)

def _load_prefiltered(self, annotation):
    """Only read BAM regions that overlap annotated TEs."""
    te_regions = self._get_te_regions(annotation)
    # Process each region (can parallelize across workers)
    for region in te_regions:
        with pysam.AlignmentFile(self.opts.samfile, threads=2) as sf:
            for ci, aln in fetch_pairs_sorted(sf.fetch(*region), region):
                # Only TE-overlapping reads reach here
                ...
```

**Auto-detection:** Check `sf.has_index()` (already done at line 98). If indexed → use pre-filtered path. If not → fall back to sequential.

**Expected impact:** 5-20x faster loading (skip 90-97% of reads — TEs occupy ~45% of genome by sequence but a much smaller fraction by unique mappable loci). For whole blood RNA-seq, pre-filtering could drop loading from ~90s → ~3-10s.

**Preserving total_fragments count:** Pre-filtering skips non-TE reads, but the total count is needed for normalization context. Solution: use `pysam`'s index stats (already accessed at `model.py` line 100: `sf.mapped` and `sf.unmapped`). This gives total mapped/unmapped counts from the BAI index in sub-second time — one line of code, preserves backward compatibility:
```python
self.run_info['total_fragments'] = sf.mapped + sf.unmapped  # from BAI index
```

#### Step 9: Improve parallel BAM loading in `model.py`

Fix `_load_parallel` (lines 175-206):
- Replace file-based I/O with in-memory return values (return lists of tuples instead of temp files)
- Use TE-region pre-filtering per worker (combine with Step 8)
- Each worker gets a subset of TE regions
- Use `concurrent.futures.ProcessPoolExecutor` for cleaner error handling
- Add pysam `threads=2` per worker (threads × processes = total parallelism)

### Phase 5: Pipeline tail parallelization

#### Step 10: Batch matrix construction in `model.py`

Replace the sequential DOK loop (lines 296-309):
```python
# Current: slow DOK — dict lookups + sparse access per mapping
_m1 = scipy.sparse.dok_matrix(dim, dtype=np.uint16)
for code, rid, fid, ascr, alen in miter:
    _m1[i, j] = max(_m1[i, j], (rescale[ascr] + alen))
```

With batch COO construction:
```python
# New: collect arrays, build COO, resolve duplicates via groupby-max
rows, cols, data = [], [], []
for code, rid, fid, ascr, alen in miter:
    rows.append(i); cols.append(j); data.append(rescale[ascr] + alen)
_m1 = scipy.sparse.coo_matrix((data, (rows, cols)), shape=...)
_m1 = _resolve_duplicates_max(_m1).tocsr()
```

Note: The index-building (`setdefault`) loop must remain sequential, but the sparse matrix construction becomes a single vectorized bulk operation.

#### Step 11: Parallel reassignment in `reporter.py`

In `output_report()` (in `telescope/utils/reporter.py`), run the 6 `tl.reassign()` calls concurrently:
```python
from concurrent.futures import ThreadPoolExecutor

modes = [
    ('final_conf', 'conf', _rprob, False),
    ('init_aligned', 'all', None, True),
    ('unique_count', 'unique', None, False),
    ('init_best', 'exclude', None, True),
    ('init_best_random', 'choose', None, True),
    ('init_best_avg', 'average', None, True),
]
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = {name: pool.submit(tl.reassign, mode, prob, initial=init)
               for name, mode, prob, init in modes}
    results = {name: f.result().sum(0).A1 for name, f in futures.items()}
```

### Phase 6: CLI and packaging

#### Step 12: Add CLI flags
In `BulkIDOptions.OPTS` and `scIDOptions.OPTS` (`telescope_assign.py`):
```yaml
- Performance Options:
    - gpu:
        action: store_true
        help: Use GPU acceleration via CuPy (requires CuPy + CUDA GPU).
    - gpu_device:
        type: int
        default: 0
        help: GPU device ID to use.
    - parallel_blocks:
        action: store_true
        help: Decompose matrix into independent blocks and solve in parallel.
```

In `run()` (line 372): Call `backend.configure(use_gpu=opts.gpu)` before model creation.

Same changes in `telescope_resume.py`.

#### Step 13: Update `setup.py`
```python
extras_require={
    'gpu': ['cupy-cuda12x>=13.0'],
    'optimized': ['sparse-dot-mkl', 'numba', 'threadpoolctl'],
}
```

### Phase 7: Testing and benchmarking

#### Step 14: Create `telescope/tests/test_gpu_sparse.py`
Unit tests for every `GpuCsrMatrix` method, parametrized for CPU/GPU backends. Auto-skip GPU tests if CuPy unavailable.

#### Step 15: Create `telescope/tests/test_cpu_optimized_sparse.py`
Unit tests for CPU-optimized sparse matrix class. Verify Numba JIT methods produce identical results to stock scipy.

#### Step 16: Create `telescope/tests/test_decomposition.py`
Tests for `find_blocks`, `split_matrix`, `merge_results`:
- Test with a known block-diagonal matrix (2 disconnected blocks)
- Verify that decomposed EM produces identical results to full EM
- Test edge case: single block (no decomposition benefit)
- **Generate synthetic test data** with reads mapping to two distinct TE families (e.g., HERVK and L1HS with no shared multi-mappers) to exercise the split/merge path in CI. The bundled test data has only 1 component (all ERVK), so without synthetic data the decomposition path is never tested in CI.

#### Step 17: Create `telescope/tests/test_numerical_equivalence.py`
Full-pipeline equivalence test on bundled test data:
- CPU standard EM vs GPU EM: `np.allclose(pi, rtol=1e-5)`
- Standard EM vs block-decomposed EM: exact match (mathematically equivalent)
- Log-likelihood must match: ≈ 95252.596614
- All 6 reassignment mode counts must match

#### Step 18: Create `telescope/tests/benchmark_em.py`
Standalone benchmark:
- Times each pipeline stage separately
- Reports speedup per dimension (GPU, blocks, parallel loading, etc.)
- Tests on bundled data and optionally on larger synthetic data

---

## Files Summary

### Created in Phase 0 ✅:
| File | Purpose |
|------|---------|
| `telescope/utils/likelihood.py` | `TelescopeLikelihood` class — EM algorithm core (estep, mstep, lnl, em, reassign) |
| `telescope/utils/reporter.py` | `output_report()` + `update_sam()` — I/O separated from algorithm |
| `telescope/tests/test_sparse_plus.py` | 45 tests covering all `csr_matrix_plus` methods |
| `telescope/tests/test_em.py` | 33 tests: EM unit tests + baseline equivalence |
| `telescope/tests/benchmark_baseline.py` | Comprehensive baseline benchmarking script |
| `telescope/tests/benchmark_baseline_results.json` | Baseline benchmark results (stock CPU / scipy) |
| `pytest.ini` | pytest configuration |
| `CLAUDE.md` | Project context for Claude Code |

### Deleted in Phase 0 ✅:
| File | Reason |
|------|--------|
| `telescope/utils/alignment_parsers.py` | Entirely unused (`TelescopeAlignment`, `TelescopeRead`, `iterread`) |

### To create (Phase 1+):
| File | Purpose |
|------|---------|
| `telescope/utils/backend.py` | Three-tier backend abstraction (GPU / CPU-optimized / CPU-stock) |
| `telescope/utils/gpu_sparse.py` | GPU sparse matrix class (CuPy + RawKernels) |
| `telescope/utils/cpu_optimized_sparse.py` | CPU-optimized sparse class (Numba JIT + MKL) |
| `telescope/utils/decompose.py` | Connected component block decomposition |
| `telescope/tests/test_gpu_sparse.py` | GPU sparse unit tests |
| `telescope/tests/test_cpu_optimized_sparse.py` | CPU-optimized sparse unit tests |
| `telescope/tests/test_decomposition.py` | Block decomposition tests |
| `telescope/tests/test_numerical_equivalence.py` | Full pipeline all-tiers comparison |
| `telescope/tests/benchmark_em.py` | Performance benchmarking (all tiers) |

### Modified in Phase 0 ✅:
| File | Changes |
|------|---------|
| `telescope/utils/model.py` | Reduced from 833→522 lines. Removed dead code, Python 2 imports. Extracted `TelescopeLikelihood` to `likelihood.py`, I/O to `reporter.py`. Backward-compat re-exports maintained. |
| `telescope/utils/sparse_plus.py` | Removed Python 2/future imports |
| `telescope/utils/helpers.py` | Removed Python 2 imports, replaced `old_div`, added `import re`, fixed regex escape |
| `telescope/utils/alignment.py` | Removed Python 2 imports, fixed Assigner constructor |
| `telescope/utils/calignment.pyx` + `.pxd` | Cython 3.0 + pysam 0.23.3 compatibility |
| `telescope/utils/__init__.py` | Removed future import |
| `telescope/utils/_alignment.py` | Removed Python 2 imports |
| `telescope/utils/_annotation_*.py` | Removed Python 2 imports |
| `telescope/utils/annotation.py` | Removed Python 2 imports |
| `telescope/telescope_assign.py` | Removed Python 2 imports |
| `telescope/telescope_resume.py` | Removed Python 2 imports |
| `telescope/__init__.py` | Removed future import |
| `setup.py` | Added pysam include dirs for Cython |

### To modify (Phase 1+):
| File | Planned Changes |
|------|---------|
| `telescope/utils/likelihood.py` | Backend-agnostic constructors in estep/mstep/lnl, `em_parallel()` |
| `telescope/utils/model.py` | Batch COO matrix construction, pysam threading, region pre-filter |
| `telescope/telescope_assign.py` | `--gpu`, `--gpu_device`, `--parallel_blocks` CLI flags, `backend.configure()` call |
| `telescope/telescope_resume.py` | Same CLI and backend changes |
| `telescope/utils/reporter.py` | Parallel reassignment |
| `setup.py` | `extras_require={'gpu': ['cupy-cuda12x'], 'optimized': ['sparse-dot-mkl', 'numba', 'threadpoolctl']}` |

### Untouched (CPU path preserved):
| File | Reason |
|------|--------|
| `telescope/utils/sparse_plus.py` | Stock CPU sparse class stays as-is (Tier 1 baseline) |
| `telescope/__main__.py` | Subparsers handle options |
| All annotation files | Not in hot path |

---

## Verification Plan

### 1. Unit tests
```bash
pytest telescope/tests/test_gpu_sparse.py -v
pytest telescope/tests/test_cpu_optimized_sparse.py -v
pytest telescope/tests/test_decomposition.py -v
```

### 2. Numerical equivalence (most important)
```bash
pytest telescope/tests/test_numerical_equivalence.py -v
```
Verifies: CPU vs GPU, standard vs block-decomposed, all reassignment modes match.

### 3. End-to-end on bundled test data
```bash
# CPU baseline (original path)
telescope assign telescope/data/alignment.bam telescope/data/annotation.gtf \
    --outdir /tmp/cpu --exp_tag baseline

# GPU + block decomposition
telescope assign telescope/data/alignment.bam telescope/data/annotation.gtf \
    --outdir /tmp/gpu --exp_tag accelerated --gpu --parallel_blocks

# Compare
diff /tmp/cpu/baseline-TE_counts.tsv /tmp/gpu/accelerated-TE_counts.tsv
```

### 4. Benchmark
```bash
python -m telescope.tests.benchmark_em           # CPU baseline
python -m telescope.tests.benchmark_em --gpu     # GPU only
python -m telescope.tests.benchmark_em --gpu --parallel_blocks  # Full acceleration
```

### 5. Log-likelihood verification
Both paths must produce log-likelihood ≈ 95252.596614 on test data (confirmed by tests and benchmarks).

---

## Key Technical Notes

- **Block decomposition is mathematically exact**: Connected components in a bipartite graph yield fully independent sub-problems. Solving each block separately produces identical results to solving the full matrix. No approximation.
- **Test data has 1 component** (all ERVK family): Block decomposition won't show speedup on test data, but will on real data with diverse TE families.
- **GPU memory**: Peak ≈ 4x sparse matrix size. For N=5M, K=5K, ~0.1% density: ~2GB GPU RAM. Fits on 8GB+ GPUs. Blocks make this even better (smaller matrices per block).
- **Precision**: All float64. CuPy supports float64. Results match within ~1e-5 rtol.
- **Fallback**: `--gpu` auto-falls back to CPU if CuPy not installed. Block decomposition works with or without GPU.
